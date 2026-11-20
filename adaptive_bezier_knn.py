"""
Adaptive KNN + local quadratic tensor-product Bezier reconstruction
===================================================================

Run:
    pip install numpy scipy matplotlib plotly pandas
    python adaptive_bezier_knn.py --obj <file.obj>
    python adaptive_bezier_knn.py --obj <file.obj> --k-min <kmin> --k-max <kmax> --delta-k <deltak> --variogram-k <kvario> --z-tolerance <z> --convexity <none|convex|concave> --grid-size <gsize>

The algorithm runs a local reconstruction at every input point:
  1. KNN neighborhoods with increasing K.
  2. Local PCA tangent frame.
  3. Bezier 2nd degree (9 control heights).
  4. A local empirical variogram of detrended normal heights.
  5. Neighborhood radius <= practical variogram range.
  6. Reconstruction RMSE <= adaptive normal-displacement tolerance.

"""

from __future__ import annotations

import webbrowser 
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize
from scipy.special import comb
import plotly.graph_objects as go



import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 1. OBJ loading
# ---------------------------------------------------------------------------

def load_obj(path: str | Path):
    """Read OBJ vertices and polygon faces. Faces are triangulated as fans."""
    vertices = []
    triangles = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                fields = line.split()
                if len(fields) >= 4:
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])

            elif line.startswith("f "):
                fields = line.split()[1:]
                face = []
                for token in fields:
                    # Handles v, v/vt, v//vn and v/vt/vn.
                    raw_index = int(token.split("/")[0])
                    # OBJ indices are 1-based; negative indices are relative
                    # to the number of vertices already read.
                    idx = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    face.append(idx)
                # Fan triangulation for polygons with > 3 vertices.
                for j in range(1, len(face) - 1):
                    triangles.append([face[0], face[j], face[j + 1]])

    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int)

    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("The OBJ file contains no valid 3D vertices.")
    points = points[np.all(np.isfinite(points), axis=1)]

    if len(faces):
        valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
        faces = faces[valid]

    return points, faces


# ---------------------------------------------------------------------------
# 2. Local PCA frame and coordinates
# ---------------------------------------------------------------------------

def local_pca(neighborhood: np.ndarray):
    """Return centroid and orthonormal local frame (e1, e2, normal)."""
    center = neighborhood.mean(axis=0)
    X = neighborhood - center
    cov = (X.T @ X) / max(len(X), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)

    normal = eigvecs[:, order[0]]
    e1 = eigvecs[:, order[2]]
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2) + 1e-15
    e1 = np.cross(e2, normal)
    e1 /= np.linalg.norm(e1) + 1e-15

    # Deterministic normal orientation: positive dot with global +Z where possible.
    if np.dot(normal, np.array([0.0, 0.0, 1.0])) < 0:
        normal = -normal
        e2 = -e2

    return center, e1, e2, normal, eigvals[order]


def to_local(points: np.ndarray, center, e1, e2, normal):
    X = points - center
    return np.column_stack((X @ e1, X @ e2, X @ normal))


def to_global(u, v, w, center, e1, e2, normal):
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    w = np.asarray(w).reshape(-1, 1)
    return center + u * e1 + v * e2 + w * normal


# ---------------------------------------------------------------------------
# 3. Empirical variogram and adaptive tolerance
# ---------------------------------------------------------------------------

@dataclass
class VariogramResult:
    practical_range: float
    sill: float
    epsilon: float
    lag_centers: np.ndarray
    semivariance: np.ndarray


def empirical_variogram(u, v, residuals, n_bins=10, z_tolerance=1.96):
    """
    Empirical isotropic semivariogram of detrended normal-height residuals.

    gamma(h) = 1/(2 |N(h)|) sum_(pairs in bin h) (r_i-r_j)^2

    Practical range: first lag whose semivariance reaches 95% of the
    estimated sill. Sill: median of the last third of populated bins.
    """
    coords = np.column_stack((u, v))
    r = np.asarray(residuals, dtype=float)
    n = len(r)

    if n < 8:
        raise ValueError("At least 8 points are needed for a local variogram.")

    # Pairwise distances and semivariances.
    ii, jj = np.triu_indices(n, k=1)
    h = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    gamma_pairs = 0.5 * (r[ii] - r[jj]) ** 2

    positive = h > 1e-12
    h, gamma_pairs = h[positive], gamma_pairs[positive]
    if len(h) < 5:
        raise ValueError("Not enough distinct pairwise distances for variogram.")

    h_max = float(np.max(h))
    edges = np.linspace(0.0, h_max * (1.0 + 1e-10), n_bins + 1)
    bin_id = np.digitize(h, edges) - 1

    centers, values = [], []
    for b in range(n_bins):
        mask = bin_id == b
        if np.any(mask):
            centers.append(float(np.mean(h[mask])))
            values.append(float(np.mean(gamma_pairs[mask])))

    centers = np.asarray(centers)
    values = np.asarray(values)

    if len(values) < 3:
        sill = float(np.var(r))
        practical_range = h_max
    else:
        tail_start = max(0, int(np.floor(2 * len(values) / 3)))
        tail = values[tail_start:]
        sill = float(np.median(tail)) if len(tail) else float(np.var(r))
        sill = max(sill, 1e-12)
        reached = np.flatnonzero(values >= 0.95 * sill)
        practical_range = float(centers[reached[0]]) if len(reached) else h_max

    # A normal-displacement tolerance in the same units as w.
    epsilon = float(z_tolerance * np.sqrt(max(sill, 1e-12)))

    return VariogramResult(
        practical_range=max(practical_range, 1e-12),
        sill=sill,
        epsilon=epsilon,
        lag_centers=centers,
        semivariance=values,
    )


def estimate_local_variogram(local_xyz, n_bins=10, z_tolerance=1.96):
    """
    Detrend local heights with a plane, then estimate the variogram of
    residual normal displacements. The local plane is only a variogram
    detrending step; the final reconstruction is Bezier.
    """
    u, v, w = local_xyz.T
    A = np.column_stack((np.ones(len(u)), u, v))
    beta, *_ = np.linalg.lstsq(A, w, rcond=None)
    detrended = w - A @ beta
    return empirical_variogram(
        u, v, detrended, n_bins=n_bins, z_tolerance=z_tolerance
    )


# ---------------------------------------------------------------------------
# 4. Quadratic  Bezier patch
# ---------------------------------------------------------------------------

def bernstein(degree, i, x):
    return comb(degree, i) * np.asarray(x) ** i * (1.0 - np.asarray(x)) ** (degree - i)


def bezier_design_matrix(s, t, degree=2):
    s = np.asarray(s).ravel()
    t = np.asarray(t).ravel()
    columns = []
    for i in range(degree + 1):
        for j in range(degree + 1):
            columns.append(bernstein(degree, i, s) * bernstein(degree, j, t))
    return np.column_stack(columns)


def _bezier_hessian_design(s, t, bounds, degree=2, step=1e-4):
    """Finite-difference Hessian of each Bernstein basis function."""
    s = np.asarray(s)
    t = np.asarray(t)
    ds = max(step, 1e-5)
    dt = max(step, 1e-5)

    A0 = bezier_design_matrix(s, t, degree)
    Assp = bezier_design_matrix(s + ds, t, degree)
    Assm = bezier_design_matrix(s - ds, t, degree)
    Attp = bezier_design_matrix(s, t + dt, degree)
    Attm = bezier_design_matrix(s, t - dt, degree)
    Asp_tp = bezier_design_matrix(s + ds, t + dt, degree)
    Asp_tm = bezier_design_matrix(s + ds, t - dt, degree)
    Asm_tp = bezier_design_matrix(s - ds, t + dt, degree)
    Asm_tm = bezier_design_matrix(s - ds, t - dt, degree)

    Hss = (Assp - 2 * A0 + Assm) / ds**2
    Htt = (Attp - 2 * A0 + Attm) / dt**2
    Hst = (Asp_tp - Asp_tm - Asm_tp + Asm_tm) / (4 * ds * dt)

    # Convert derivatives w.r.t. s,t to derivatives w.r.t. u,v.
    umin, umax, vmin, vmax = bounds
    Huu = Hss / max((umax - umin) ** 2, 1e-15)
    Hvv = Htt / max((vmax - vmin) ** 2, 1e-15)
    Huv = Hst / max((umax - umin) * (vmax - vmin), 1e-15)
    return Huu, Huv, Hvv


def fit_bezier_patch(u, v, w, degree=2, convexity="none", convexity_grid=5):
    """
    Fit a degree-(2,2) Bezier height patch to raw observations.
    """
    u, v, w = map(lambda x: np.asarray(x, dtype=float).ravel(), (u, v, w))
    if len(u) < (degree + 1) ** 2:
        raise ValueError("Too few points for the selected Bezier degree.")

    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    if umax - umin < 1e-12 or vmax - vmin < 1e-12:
        raise ValueError("Degenerate local tangent-plane coordinates.")

    bounds = (umin, umax, vmin, vmax)
    s = (u - umin) / (umax - umin)
    t = (v - vmin) / (vmax - vmin)
    A = bezier_design_matrix(s, t, degree)
    coeff0, *_ = np.linalg.lstsq(A, w, rcond=None)

    if convexity in ("convex", "concave"):
        g = np.linspace(0.03, 0.97, convexity_grid)
        S, T = np.meshgrid(g, g)
        Huu, Huv, Hvv = _bezier_hessian_design(
            S.ravel(), T.ravel(), bounds, degree
        )
        sign = 1.0 if convexity == "convex" else -1.0

        def objective(c):
            residual = A @ c - w
            return 0.5 * np.dot(residual, residual)

        def jacobian(c):
            return A.T @ (A @ c - w)

        def psd_constraints(c):
            a = sign * (Huu @ c)
            b = sign * (Huv @ c)
            d = sign * (Hvv @ c)
            # PSD for a 2x2 symmetric matrix: a >= 0, d >= 0, ad-b^2 >= 0.
            return np.concatenate((a, d, a * d - b * b))

        result = minimize(
            objective,
            coeff0,
            jac=jacobian,
            constraints=[{"type": "ineq", "fun": psd_constraints}],
            method="SLSQP",
            options={"maxiter": 250, "ftol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"Convexity-constrained fit failed: {result.message}")
        coeff = result.x
    elif convexity == "none":
        coeff = coeff0
    else:
        raise ValueError("convexity must be 'none', 'convex', or 'concave'.")

    fitted = A @ coeff
    ctrl_h = coeff.reshape(degree + 1, degree + 1)
    return {
        "control_heights": ctrl_h,
        "bounds": bounds,
        "degree": degree,
        "coefficients": coeff,
        "fitted": fitted,
        "residuals": w - fitted,
        "convexity": convexity,
    }


def evaluate_bezier(patch, u_query, v_query):
    umin, umax, vmin, vmax = patch["bounds"]
    s = (np.asarray(u_query) - umin) / (umax - umin)
    t = (np.asarray(v_query) - vmin) / (vmax - vmin)
    A = bezier_design_matrix(s, t, patch["degree"])
    return A @ patch["coefficients"]
'''
from scipy.spatial import ConvexHull
from matplotlib.path import Path
def trim_bezier_patch(
    control_points,
    bounds,
    u_points,
    v_points,
    degree=2,
    resolution=50
):
    """
    Evaluate the fitted Bezier patch on a dense grid,
    then trim it to the convex hull of the projected points.
    """

    # 1. Convex hull of the neighborhood in local (u,v) coordinates
    points_2d = np.column_stack((u_points, v_points))
    hull = ConvexHull(points_2d)

    polygon = points_2d[hull.vertices]
    path = Path(polygon)

    # 2. Create a dense grid over the fitted patch domain
    umin, umax, vmin, vmax = bounds

    U, V = np.meshgrid(
        np.linspace(umin, umax, resolution),
        np.linspace(vmin, vmax, resolution)
    )

    # 3. Evaluate the already-fitted Bezier patch
    W = evaluate_bezier(
        control_points,
        bounds,
        U.ravel(),
        V.ravel(),
        degree=degree
    ).reshape(U.shape)

    # 4. Determine which surface points are inside the hull
    grid_points = np.column_stack((U.ravel(), V.ravel()))
    mask = path.contains_points(grid_points).reshape(U.shape)

    # 5. Remove the surface outside the hull
    W_trimmed = np.where(mask, W, np.nan)

    return U, V, W_trimmed, mask
'''

def make_patch_grid(patch, frame, grid_size=7):
    center, e1, e2, normal = frame
    umin, umax, vmin, vmax = patch["bounds"]
    uu = np.linspace(umin, umax, grid_size)
    vv = np.linspace(vmin, vmax, grid_size)
    U, V = np.meshgrid(uu, vv)
    W = evaluate_bezier(patch, U.ravel(), V.ravel()).reshape(U.shape)
    xyz = to_global(U.ravel(), V.ravel(), W.ravel(), center, e1, e2, normal)
    return xyz.reshape(grid_size, grid_size, 3)


# ---------------------------------------------------------------------------
# 5. Adaptive K selection for one query point
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_index: int
    selected_k: int
    radius: float
    rmse: float
    mae: float
    epsilon: float
    practical_range: float
    sill: float
    patch: dict
    frame: tuple
    neighborhood_indices: np.ndarray


def select_adaptive_patch(
    query_index,
    points,
    knn_indices,
    knn_distances,
    k_values,
    variogram_k,
    degree=2,
    z_tolerance=1.96,
    convexity="none",
):
    """
    Estimate one local variogram from a pilot neighborhood, then test each K.
    Select the largest K satisfying:
        radius(K) <= local practical range
        RMSE(K) <= local epsilon
    """
    pilot_k = min(variogram_k, knn_indices.shape[1])
    pilot_idx = knn_indices[query_index, :pilot_k]
    pilot_xyz = points[pilot_idx]

    center, e1, e2, normal, _ = local_pca(pilot_xyz)
    pilot_local = to_local(pilot_xyz, center, e1, e2, normal)

    try:
        vg = estimate_local_variogram(
            pilot_local, z_tolerance=z_tolerance
        )
    except (ValueError, np.linalg.LinAlgError):
        return None

    valid_candidates = []
    for k in k_values:
        if k > knn_indices.shape[1] or k < (degree + 1) ** 2 + 3:
            continue

        radius = float(knn_distances[query_index, k - 1])
        if radius > vg.practical_range:
            # KNN radii are nondecreasing, so larger K will also fail.
            break

        idx = knn_indices[query_index, :k]
        xyz = points[idx]

        # Re-estimate frame at the candidate K.
        c, a, b, n, _ = local_pca(xyz)
        local = to_local(xyz, c, a, b, n)
        u, v, w = local.T

        try:
            patch = fit_bezier_patch(
                u, v, w, degree=degree, convexity=convexity
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

        residuals = patch["residuals"]
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        # A candidate is accepted only if its normal residual scale is
        # within the variogram-derived local tolerance.
        if rmse <= vg.epsilon:
            valid_candidates.append(
                QueryResult(
                    query_index=query_index,
                    selected_k=int(k),
                    radius=radius,
                    rmse=rmse,
                    mae=mae,
                    epsilon=vg.epsilon,
                    practical_range=vg.practical_range,
                    sill=vg.sill,
                    patch=patch,
                    frame=(c, a, b, n),
                    neighborhood_indices=idx.copy(),
                )
            )

    if not valid_candidates:
        return None

    # Largest admissible neighborhood, as in the proposed adaptive-K rule.
    return max(valid_candidates, key=lambda result: result.selected_k)


# ---------------------------------------------------------------------------
# 6. Run over the entire point set
# ---------------------------------------------------------------------------

def run_adaptive_reconstruction(
    points,
    k_min=15,
    k_max=60,
    delta_k=5,
    variogram_k=40,
    degree=2,
    z_tolerance=1.96,
    convexity="none",
    verbose=True,
):
    """
    Every point is used as a query point. Its own index is removed from KNN.
    """
    n_points = len(points)
    if n_points < k_min + 1:
        raise ValueError("Point cloud is too small for the requested K_min.")

    k_max = min(k_max, n_points - 1)
    k_values = np.arange(k_min, k_max + 1, delta_k, dtype=int)
    if len(k_values) == 0:
        raise ValueError("No candidate K values. Check k_min/k_max/delta_k.")

    tree = cKDTree(points)
    distances_all, indices_all = tree.query(points, k=k_max + 1)

    # Remove each query point itself (first neighbor at distance zero).
    distances = distances_all[:, 1:]
    indices = indices_all[:, 1:]

    results = [None] * n_points
    for i in range(n_points):
        results[i] = select_adaptive_patch(
            i, points, indices, distances, k_values,
            variogram_k=variogram_k,
            degree=degree,
            z_tolerance=z_tolerance,
            convexity=convexity,
        )
        if verbose and (i + 1) % max(1, n_points // 20) == 0:
            print(f"Processed {i + 1}/{n_points} query points")

    return results, k_values


# ---------------------------------------------------------------------------
# 7. Export numerical results
# ---------------------------------------------------------------------------

def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    patch_records = []
    for result in results:
        if result is None:
            continue
        rows.append({
            "query_index": result.query_index,
            "selected_K": result.selected_k,
            "neighborhood_radius": result.radius,
            "reconstruction_RMSE": result.rmse,
            "reconstruction_MAE": result.mae,
            "adaptive_epsilon": result.epsilon,
            "variogram_practical_range": result.practical_range,
            "variogram_sill": result.sill,
        })
        patch_records.append({
            "query_index": result.query_index,
            "selected_K": result.selected_k,
            "bounds": list(result.patch["bounds"]),
            "control_heights": result.patch["control_heights"].tolist(),
            "frame_center": result.frame[0].tolist(),
            "frame_e1": result.frame[1].tolist(),
            "frame_e2": result.frame[2].tolist(),
            "frame_normal": result.frame[3].tolist(),
        })

    csv_path = output_dir / "adaptive_knn_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "query_index", "selected_K", "neighborhood_radius",
            "reconstruction_RMSE", "reconstruction_MAE",
            "adaptive_epsilon", "variogram_practical_range", "variogram_sill"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "bezier_patches.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(patch_records, f, indent=2)

    return csv_path, json_path

# ---------------------------------------------------------------------------
# 8. Plot all selected Bezier patches & diagnostics (OBJ-aware)
# ---------------------------------------------------------------------------

def plot_all_patches(
    points,
    results,
    obj_name="model",
    output_html="adaptive_bezier_patches.html",
    grid_size=5,
    point_size=2.0,
    max_patches=800,
):
    valid_results = [r for r in results if r is not None]
    print(f"[{obj_name}] Rendering {min(len(valid_results), max_patches)} of {len(valid_results)} patches...")

    if len(valid_results) > max_patches:
        step = len(valid_results) // max_patches
        display_results = valid_results[::step]
    else:
        display_results = valid_results

    vertices = []
    faces = []
    intensities = []
    g = grid_size

    for result in display_results:
        patch_grid = make_patch_grid(result.patch, result.frame, grid_size=g)
        offset = len(vertices)
        pts_flat = patch_grid.reshape(-1, 3)
        vertices.extend(pts_flat.tolist())
        intensities.extend([result.selected_k] * len(pts_flat))

        for r in range(g - 1):
            for c in range(g - 1):
                a = offset + r * g + c
                b = a + 1
                d = offset + (r + 1) * g + c
                e = d + 1
                faces.append([a, b, e])
                faces.append([a, e, d])

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=point_size, color="#444444", opacity=0.45),
        name=f"{obj_name} points ({len(points):,})",
    ))

    if vertices and faces:
        V = np.asarray(vertices)
        F = np.asarray(faces)
        fig.add_trace(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            intensity=intensities,
            colorscale="Viridis",
            colorbar=dict(title="Selected K*", len=0.6),
            opacity=0.75,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
            name=f"Bezier patches ({len(display_results):,} shown)",
        ))

    fig.update_layout(
        title=(
            f"Adaptive Bezier Reconstruction: <b>{obj_name}.obj</b> "
            f"({len(valid_results):,}/{len(results):,} points fitted)"
        ),
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    output_path = Path(output_html).resolve()
    # include_plotlyjs=True makes the HTML self-contained (works offline)
    fig.write_html(str(output_path), include_plotlyjs=True, auto_open=False)

    # Explicitly open using a proper file:// URI
    webbrowser.open(output_path.as_uri())
    return fig



def plot_diagnostics(results, obj_name="model", output_path="diagnostics.png"):
    """Save diagnostic plots without blocking execution."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend prevents window freeze
    import matplotlib.pyplot as plt

    valid = [r for r in results if r is not None]
    if not valid:
        print("No successful local patches; skipping diagnostic plot.")
        return

    idx = np.array([r.query_index for r in valid])
    K = np.array([r.selected_k for r in valid])
    rmse = np.array([r.rmse for r in valid])
    eps = np.array([r.epsilon for r in valid])
    radius = np.array([r.radius for r in valid])
    prange = np.array([r.practical_range for r in valid])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Adaptive KNN + Bezier Diagnostics — {obj_name}.obj", fontsize=13, fontweight="bold")

    axes[0, 0].scatter(idx, K, s=6, alpha=0.7, color="#2b5c8f")
    axes[0, 0].set(title="Selected K* by query point", xlabel="Query index", ylabel="K*")

    axes[0, 1].scatter(idx, rmse, s=6, alpha=0.7, label="Reconstruction RMSE")
    axes[0, 1].scatter(idx, eps, s=6, alpha=0.7, label="Adaptive tolerance (ε)")
    axes[0, 1].set(title="Error vs. Variogram Tolerance", xlabel="Query index", ylabel="Normal displacement")
    axes[0, 1].legend()

    axes[1, 0].scatter(idx, radius, s=6, alpha=0.7, label="Neighborhood radius")
    axes[1, 0].scatter(idx, prange, s=6, alpha=0.7, label="Practical variogram range")
    axes[1, 0].set(title="Spatial Locality Constraint", xlabel="Query index", ylabel="Distance")
    axes[1, 0].legend()

    bins = np.arange(K.min(), K.max() + 2) - 0.5
    axes[1, 1].hist(K, bins=bins, color="#2e8b57", edgecolor="white")
    axes[1, 1].set(title="Distribution of Selected K*", xlabel="K*", ylabel="Point count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True, help="Path to input OBJ file")
    parser.add_argument("--output", default=None, help="Optional custom output directory")
    parser.add_argument("--k-min", type=int, default=15)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument("--delta-k", type=int, default=5)
    parser.add_argument("--variogram-k", type=int, default=40)
    parser.add_argument("--z-tolerance", type=float, default=1.96)
    parser.add_argument("--convexity", choices=["none", "convex", "concave"], default="none")
    parser.add_argument("--grid-size", type=int, default=6)
    args = parser.parse_args()

    obj_path = Path(args.obj)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    obj_name = obj_path.stem
    output_dir = Path(args.output) if args.output else Path(f"output_{obj_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    points, faces = load_obj(obj_path)
    print(f"[{obj_name}] Loaded {len(points):,} vertices and {len(faces):,} triangles.")

    effective_k_max = min(args.k_max, len(points) - 1)
    effective_k_min = min(args.k_min, max(12, effective_k_max - 5))

    t_recon_start = time.perf_counter()
    results, k_values = run_adaptive_reconstruction(
        points,
        k_min=effective_k_min,
        k_max=effective_k_max,
        delta_k=args.delta_k,
        variogram_k=min(args.variogram_k, len(points) - 1),
        degree=2,
        z_tolerance=args.z_tolerance,
        convexity=args.convexity,
    )
    t_recon = time.perf_counter() - t_recon_start

    n_success = sum(r is not None for r in results)
    pts_per_sec = len(points) / max(t_recon, 1e-9)
    print(f"[{obj_name}] Successful local patches: {n_success}/{len(points)}")
    print(f"[{obj_name}] Reconstruction time: {t_recon:.2f}s ({pts_per_sec:,.1f} pts/sec)")

    csv_path, json_path = save_results(results, output_dir)
    diag_path = output_dir / f"{obj_name}_diagnostics.png"
    html_path = output_dir / f"{obj_name}_bezier_patches.html"

    plot_diagnostics(results, obj_name=obj_name, output_path=diag_path)
    plot_all_patches(
        points,
        results,
        obj_name=obj_name,
        output_html=html_path,
        grid_size=args.grid_size,
    )

    t_total = time.perf_counter() - t_start
    print(f"[{obj_name}] Saved outputs to: {output_dir.resolve()}")
    print(f"  - 3D Interactive Plot: {html_path.name}")
    print(f"  - Diagnostics Image:   {diag_path.name}")
    print(f"  - Summary CSV:         {csv_path.name}")
    print(f"[{obj_name}] Total runtime: {t_total:.2f}s")



if __name__ == "__main__":
    main()
