"""
Local point-cloud estimation pipeline
======================================

Real point cloud neighborhood
             │
             ▼
Robust scatter matrix Σ
             │
             ▼
Local frame (e₁, e₂, n)
             │
             ▼
Tangent plane spanned by e₁ and e₂
             │
             ▼
Real point xⱼ → local coordinates (uⱼ, vⱼ, wⱼ)
             │
             ▼
(uⱼ, vⱼ) = Bézier parameter location
wⱼ = observed height
             │
             ▼
Fit Bézier height function h(u,v)
             │
             ▼
Local Bézier patch:
(u, v, h(u,v))
             │
             ▼
Map back:
X̂ = c + u e₁ + v e₂ + h(u,v)n
             │
             ▼
Reconstructed local surface in ℝ³
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ----------------------------------------------------------------------
# 0. Point cloud
# ----------------------------------------------------------------------
def make_point_cloud(n=4000, a=0.6, b=-0.3, noise=0.015, outlier_frac=0.03, seed=0):
    """Synthetic fallback used only when no real scan is given (demo/testing)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    z = a * x**2 + b * y**2 + noise * rng.standard_normal(n)
    pts = np.column_stack([x, y, z])

    n_out = int(outlier_frac * n)
    idx = rng.choice(n, n_out, replace=False)
    pts[idx] += rng.normal(scale=0.15, size=(n_out, 3))
    return pts


def _read_ascii_ply(path):
    """Minimal ASCII PLY reader (x,y,z only), used if open3d isn't installed."""
    with open(path, 'r') as f:
        lines = f.readlines()
    if lines[0].strip() != 'ply':
        raise ValueError("Not a PLY file")
    if 'format ascii' not in lines[1]:
        raise ValueError("Only ASCII PLY is supported without open3d installed "
                          "(pip install open3d for binary PLY/PCD support)")
    n_verts = None
    props = []
    i = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])
        if line.startswith('property'):
            props.append(line.split()[-1])
        if line.strip() == 'end_header':
            break
    xi, yi, zi = props.index('x'), props.index('y'), props.index('z')
    data = np.array([list(map(float, l.split())) for l in lines[i + 1:i + 1 + n_verts]])
    return data[:, [xi, yi, zi]]


def load_point_cloud(path):
    """
    Load a real point cloud from disk. Supports:
      - .xyz / .txt / .csv  : plain text, first 3 columns are x,y,z
      - .ply                : via open3d if installed, else a minimal ASCII-only reader
      - .pcd                : via open3d (required)
    Returns an (N, 3) numpy array.
    """
    ext = path.lower().rsplit('.', 1)[-1]

    if ext in ('xyz', 'txt', 'csv'):
        try:
            data = np.loadtxt(path, delimiter=',')
        except ValueError:
            data = np.loadtxt(path)
        return np.asarray(data[:, :3], dtype=float)

    if ext == 'ply':
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points)
        except ImportError:
            return _read_ascii_ply(path)

    if ext == 'pcd':
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError("Reading .pcd requires open3d: pip install open3d "
                               "--break-system-packages") from e
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)

    raise ValueError(f"Unsupported extension: .{ext} "
                      f"(supported: .xyz, .txt, .csv, .ply, .pcd)")


# ----------------------------------------------------------------------
# 1. kNN
# ----------------------------------------------------------------------
def knn(tree, p, k=30):
    _, idx = tree.query(p, k=k)
    return idx


# ----------------------------------------------------------------------
# 2. Robust scatter matrix (Tyler's M-estimator)
# ----------------------------------------------------------------------
def tyler_m_estimator(X, tol=1e-6, max_iter=100):
    p = X.shape[1]
    mu = X.mean(axis=0)
    for _ in range(50):  # spatial median (Weiszfeld) for a robust center
        d = np.linalg.norm(X - mu, axis=1)
        d[d < 1e-12] = 1e-12
        w = 1.0 / d
        mu_new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(mu_new - mu) < 1e-9:
            mu = mu_new
            break
        mu = mu_new

    Xc = X - mu
    Sigma = np.eye(p)
    for _ in range(max_iter):
        Sinv = np.linalg.inv(Sigma)
        quad = np.einsum('ij,jk,ik->i', Xc, Sinv, Xc)
        quad[quad < 1e-12] = 1e-12
        Sigma_new = (p / len(Xc)) * (Xc.T @ (Xc / quad[:, None]))
        Sigma_new /= np.trace(Sigma_new) / p  # Tyler is scale-free, fix trace
        if np.linalg.norm(Sigma_new - Sigma) < tol:
            Sigma = Sigma_new
            break
        Sigma = Sigma_new
    return Sigma, mu


# ----------------------------------------------------------------------
# 3. Ellipsoid = eigendecomposition of Sigma (semi-axes + orientation)
# ----------------------------------------------------------------------
def ellipsoid_from_sigma(Sigma):
    eigval, eigvec = np.linalg.eigh(Sigma)      # ascending: l1 <= l2 <= l3
    semi_axes = np.sqrt(np.clip(eigval, 1e-12, None))
    return eigval, eigvec, semi_axes


# ----------------------------------------------------------------------
# 4. Local frame (v1, v2, v3): v3 = normal, v1/v2 = tangent basis
# ----------------------------------------------------------------------
def local_frame(eigvec, ref_up=np.array([0, 0, 1.0])):
    v3 = eigvec[:, 0]              # normal = smallest-eigenvalue direction
    v1 = eigvec[:, 2]              # tangent, largest eigenvalue
    v2 = eigvec[:, 1]              # tangent, middle eigenvalue
    if v3 @ ref_up < 0:            # consistent orientation
        v3 = -v3
    return v1, v2, v3


# ----------------------------------------------------------------------
# 5. Project neighbors into (u, v, w):  3D -> 2D tangent coords + height
# ----------------------------------------------------------------------
def project_to_local(p0, v1, v2, v3, neighbors):
    rel = neighbors - p0
    u = rel @ v1
    v = rel @ v2
    w = rel @ v3
    return u, v, w


# ----------------------------------------------------------------------
# 6. Probe along local field: a regular (u, v) sampling grid over the
#    neighborhood's tangent-plane extent
# ----------------------------------------------------------------------
def probe_grid(u, v, res=9, pad=1.15):
    umax = np.abs(u).max() * pad + 1e-9
    vmax = np.abs(v).max() * pad + 1e-9
    gu, gv = np.meshgrid(np.linspace(-umax, umax, res),
                          np.linspace(-vmax, vmax, res))
    return gu, gv


# ----------------------------------------------------------------------
# 7. Obtain height h(u, v): interpolate scattered w samples onto the grid
# ----------------------------------------------------------------------
def obtain_height(u, v, w, gu, gv):
    h = griddata(np.column_stack([u, v]), w, (gu, gv), method='linear')
    nan_mask = np.isnan(h)
    if nan_mask.any():
        h_nn = griddata(np.column_stack([u, v]), w, (gu, gv), method='nearest')
        h[nan_mask] = h_nn[nan_mask]
    return h


# ----------------------------------------------------------------------
# 8. Fit Bezier patch: tensor-product Bernstein-basis least squares
# ----------------------------------------------------------------------
def bernstein(n, i, t):
    return comb(n, i) * t**i * (1 - t) ** (n - i)


def bezier_design_matrix(s, t, degree):
    """s, t in [0,1], flattened. Returns (N, (degree+1)^2) Bernstein design matrix."""
    n = degree
    cols = []
    for i in range(n + 1):
        Bi = bernstein(n, i, s)
        for j in range(n + 1):
            Bj = bernstein(n, j, t)
            cols.append(Bi * Bj)
    return np.column_stack(cols)


def fit_bezier_patch(gu, gv, gh, degree=2):
    """
    Fit a degree x degree tensor-product Bezier patch to a height grid.
    Returns control_points with shape (degree+1, degree+1, 3) in (u, v, h),
    plus the (u,v) normalization bounds needed to evaluate the patch later.
    """
    umin, umax = gu.min(), gu.max()
    vmin, vmax = gv.min(), gv.max()
    s = (gu.ravel() - umin) / (umax - umin + 1e-12)   # normalize to [0,1]
    t = (gv.ravel() - vmin) / (vmax - vmin + 1e-12)

    A = bezier_design_matrix(s, t, degree)
    coeffs, *_ = np.linalg.lstsq(A, gh.ravel(), rcond=None)  # heights of control pts

    n = degree
    ctrl_h = coeffs.reshape(n + 1, n + 1)
    ctrl_u = umin + (umax - umin) * np.linspace(0, 1, n + 1)
    ctrl_v = vmin + (vmax - vmin) * np.linspace(0, 1, n + 1)
    CU, CV = np.meshgrid(ctrl_u, ctrl_v, indexing='ij')
    control_points = np.stack([CU, CV, ctrl_h], axis=-1)
    return control_points, (umin, umax, vmin, vmax)


def evaluate_bezier(control_points, bounds, u_query, v_query, degree=2):
    umin, umax, vmin, vmax = bounds
    s = (u_query - umin) / (umax - umin + 1e-12)
    t = (v_query - vmin) / (vmax - vmin + 1e-12)
    A = bezier_design_matrix(np.atleast_1d(s), np.atleast_1d(t), degree)
    ctrl_h = control_points[:, :, 2].ravel()
    return A @ ctrl_h


def reconstruct_global(mu, v1, v2, v3, u, v, h):
    """(u, v, h) in the local frame -> back to a 3D point in world coordinates."""
    u = np.atleast_1d(u)[:, None]
    v = np.atleast_1d(v)[:, None]
    h = np.atleast_1d(h)[:, None]
    return mu + u * v1 + v * v2 + h * v3


# ----------------------------------------------------------------------
# 8b. NLLS refinement: Point Distance Minimization (PDM)
#
# The linear fit above minimizes HEIGHT residual: distance measured only
# along the normal, at each point's own orthogonal-projection (u_i, v_i).
# On a curved patch the true closest surface point is generally elsewhere.
# PDM instead minimizes true Euclidean point-to-surface distance:
#
#     min_{control net, footpoints (s_i,t_i)}  sum_i || p_i - S(s_i,t_i) ||^2
#
# S(s,t) is nonlinear in (s,t) -> genuinely nonlinear least squares.
# Solved here by Gauss-Newton alternation (the standard CAGD approach):
#   (a) fixed control net  -> re-find each point's closest-point footpoint
#   (b) fixed footpoints   -> re-solve control net (linear, all 3 coords)
# and the control points are now FREE (u, v, h) triples, not a fixed grid.
# ----------------------------------------------------------------------
from scipy.optimize import minimize_scalar, minimize  # noqa: E402


def _eval_surface_free(control_points, s, t, degree):
    """Evaluate a free-form (u,v,h all free per control point) Bezier patch."""
    A = bezier_design_matrix(np.atleast_1d(s), np.atleast_1d(t), degree)
    ctrl_flat = control_points.reshape(-1, 3)   # ((degree+1)^2, 3)
    return A @ ctrl_flat                          # (N, 3) points in (u,v,h)


def _find_footpoint(control_points, p, degree, s0, t0):
    """Closest-point (s,t) on the current surface to data point p, via local search."""
    def obj(st):
        s, t = np.clip(st, 0.0, 1.0)
        q = _eval_surface_free(control_points, s, t, degree)[0]
        return np.sum((q - p) ** 2)
    res = minimize(obj, x0=[s0, t0], method='L-BFGS-B',
                    bounds=[(0.0, 1.0), (0.0, 1.0)])
    return res.x


def fit_bezier_patch_nlls(u, v, w, degree=2, n_outer_iter=4, verbose=False):
    """
    NLLS (Point Distance Minimization) fit of a free-form Bezier patch to
    scattered local-frame data (u, v, w). Returns control_points shaped
    (degree+1, degree+1, 3) with entries (u, v, h) -- no longer a fixed
    grid in (u, v) like the linear fit, and footpoints (s_i, t_i).
    """
    pts = np.column_stack([u, v, w])
    n = degree

    # --- initialize from the cheap linear (grid) fit ---
    gu, gv = probe_grid(u, v, res=9)
    gh = obtain_height(u, v, w, gu, gv)
    control_points, bounds = fit_bezier_patch(gu, gv, gh, degree=degree)
    umin, umax, vmin, vmax = bounds

    s = np.clip((u - umin) / (umax - umin + 1e-12), 0, 1)
    t = np.clip((v - vmin) / (vmax - vmin + 1e-12), 0, 1)

    for outer in range(n_outer_iter):
        # (a) footpoint step: fixed control net, re-find closest (s_i, t_i)
        for i in range(len(pts)):
            s[i], t[i] = _find_footpoint(control_points, pts[i], degree, s[i], t[i])

        # (b) linear step: fixed footpoints, re-solve control net (u,v,h jointly)
        A = bezier_design_matrix(s, t, degree)
        new_ctrl = np.zeros_like(control_points.reshape(-1, 3))
        for coord in range(3):
            coeffs, *_ = np.linalg.lstsq(A, pts[:, coord], rcond=None)
            new_ctrl[:, coord] = coeffs
        control_points = new_ctrl.reshape(n + 1, n + 1, 3)

        if verbose:
            resid = _eval_surface_free(control_points, s, t, degree) - pts
            print(f"  NLLS iter {outer}: RMS point-to-surface dist = "
                  f"{np.sqrt(np.mean(np.sum(resid**2, axis=1))):.5f}")

    return control_points, (s, t)


def point_to_surface_rms(control_points, s, t, u, v, w, degree=2):
    pts = np.column_stack([u, v, w])
    pred = _eval_surface_free(control_points, s, t, degree)
    return np.sqrt(np.mean(np.sum((pred - pts) ** 2, axis=1)))


# ----------------------------------------------------------------------
# Curvature straight from the fitted patch's control net (no ground truth
# needed: this is what you actually query on a real, unknown scan).
# For a Bezier patch of degree 2, the second derivatives at the patch
# center are a fixed linear combination of the 3x3 control-point heights.
# ----------------------------------------------------------------------
def curvature_at_center(control_points):
    h = control_points[:, :, 2]  # 3x3 grid of control-point heights
    # central second differences on the control net (degree-2 Bezier
    # control net is evenly spaced in (u,v), so this is exact for the
    # quadratic patch, not just an approximation)
    h_uu = h[2, 1] - 2 * h[1, 1] + h[0, 1]
    h_vv = h[1, 2] - 2 * h[1, 1] + h[1, 0]
    h_uv = (h[2, 2] - h[2, 0] - h[0, 2] + h[0, 0]) / 4
    Hess = np.array([[h_uu, h_uv], [h_uv, h_vv]])
    K_gauss = np.linalg.det(Hess)          # Gaussian curvature (grad ~ 0 at center)
    H_mean = np.trace(Hess) / 2            # mean curvature
    k1, k2 = np.linalg.eigvalsh(Hess)      # principal curvatures
    return K_gauss, H_mean, (k1, k2)


# ----------------------------------------------------------------------
# Full per-point pipeline
# ----------------------------------------------------------------------
def run_pipeline_on_point(pts, tree, p0, k=30, bezier_degree=2, grid_res=9):
    idx = knn(tree, p0, k=k)
    neighbors = pts[idx]

    Sigma, mu = tyler_m_estimator(neighbors)
    eigval, eigvec, semi_axes = ellipsoid_from_sigma(Sigma)
    v1, v2, v3 = local_frame(eigvec)

    u, v, w = project_to_local(mu, v1, v2, v3, neighbors)
    gu, gv = probe_grid(u, v, res=grid_res)
    gh = obtain_height(u, v, w, gu, gv)
    control_points, bounds = fit_bezier_patch(gu, gv, gh, degree=bezier_degree)

    return {
        "center": mu, "frame": (v1, v2, v3), "eigval": eigval, "semi_axes": semi_axes,
        "uvw": (u, v, w), "grid": (gu, gv, gh),
        "control_points": control_points, "bounds": bounds,
    }


# ----------------------------------------------------------------------
# Demo: sample 150 random query points, run the pipeline on each
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(1)
    pts = make_point_cloud(n=4000)
    tree = cKDTree(pts)

    query_idx = rng.choice(len(pts), size=150, replace=False)
    results = []
    for qi in query_idx:
        p0 = pts[qi]
        res = run_pipeline_on_point(pts, tree, p0, k=30, bezier_degree=2, grid_res=9)
        results.append(res)

    print(f"Processed {len(results)} query points through the full pipeline.")

    # quick sanity metric: median residual between fitted patch and raw w's
    residuals = []
    for res in results:
        u, v, w = res["uvw"]
        pred = evaluate_bezier(res["control_points"], res["bounds"], u, v, degree=2)
        residuals.append(np.median(np.abs(pred - w)))
    print(f"Median |fit - actual height| across all 150 patches: {np.median(residuals):.5f}")

    # ------------------------------------------------------------------
    # Ground-truth comparison: this is the part raw points can't give you.
    # The synthetic cloud has a known noise-free surface z = a*x^2 + b*y^2.
    # For each query point, evaluate the Bezier patch AT ITS OWN CENTER
    # (u=0, v=0) and reconstruct the corresponding (x, y, z), then compare:
    #   (a) the query point's own raw noisy z          -> "before"
    #   (b) the Bezier-smoothed z at the same location  -> "after"
    # against the true noise-free z(x, y).
    # ------------------------------------------------------------------
    raw_err, bezier_err = [], []
    for res, qi in zip(results, query_idx):
        p0 = pts[qi]
        mu = res["center"]
        v1, v2, v3 = res["frame"]

        h0 = evaluate_bezier(res["control_points"], res["bounds"], 0.0, 0.0, degree=2)[0]
        recon = reconstruct_global(mu, v1, v2, v3, 0.0, 0.0, h0)[0]

        z_true_raw = ground_truth_z(p0[0], p0[1])
        z_true_recon = ground_truth_z(recon[0], recon[1])

        raw_err.append(abs(p0[2] - z_true_raw))
        bezier_err.append(abs(recon[2] - z_true_recon))

    raw_err, bezier_err = np.array(raw_err), np.array(bezier_err)
    print(f"\nGround-truth comparison over {len(results)} query points:")
    print(f"  median |raw point z - true z|      : {np.median(raw_err):.5f}")
    print(f"  median |Bezier-recon z - true z|    : {np.median(bezier_err):.5f}")
    print(f"  improvement factor (median)         : {np.median(raw_err) / np.median(bezier_err):.2f}x")

    # visualize one example patch
    example = results[0]
    u, v, w = example["uvw"]
    gu, gv, gh = example["grid"]
    ctrl = example["control_points"]

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(u, v, w, c='k', s=15, label='neighbors (u,v,w)')
    ax1.scatter(ctrl[:, :, 0], ctrl[:, :, 1], ctrl[:, :, 2],
                c='red', s=40, marker='^', label='Bezier control points')
    su, sv = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    ub = gu.min() + (gu.max() - gu.min()) * su
    vb = gv.min() + (gv.max() - gv.min()) * sv
    hb = evaluate_bezier(ctrl, example["bounds"], ub.ravel(), vb.ravel(), degree=2).reshape(ub.shape)
    ax1.plot_surface(ub, vb, hb, alpha=0.3, color='teal')
    ax1.set_title("Local frame: neighbors + fitted Bezier patch")
    ax1.legend(loc='upper left', fontsize=7)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pts_show = pts[rng.choice(len(pts), 1200, replace=False)]
    ax2.scatter(pts_show[:, 0], pts_show[:, 1], pts_show[:, 2], c='lightgray', s=3)
    centers = np.array([r["center"] for r in results])
    ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='crimson', s=8)
    ax2.set_title("150 randomly sampled query points on the cloud")

    plt.tight_layout()
    out_path = "/home/claude/local_bezier_pipeline_demo.png"
    plt.savefig(out_path, dpi=140)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
