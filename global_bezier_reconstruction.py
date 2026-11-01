"""
global_bezier_reconstruction.py

Local robust Bézier reconstruction of a point cloud.

Pipeline
point cloud
    -> kNN neighborhoods
    -> Tyler scatter matrix
    -> local tangent frame (e1, e2, n)
    -> local coordinates (u, v, w)
    -> local Bézier height fit
    -> evaluate each patch
    -> map patches back to R^3
    -> merge overlapping reconstructed patches

This builds a GLOBAL RECONSTRUCTION FROM MANY LOCAL BÉZIER PATCHES.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from math import comb


# 0. ROBUST LOCAL SCATTER: TYLER M-ESTIMATOR


def tyler_m_estimator(X, tol=1e-6, max_iter=200, ridge=1e-8):
    """
    X: shape (k, 3), rows are points and columns are x,y,z.

    Returns
    -------
    Sigma : (3, 3) trace-normalized robust scatter matrix
    mu    : (3,) robust spatial center
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Tyler expects X with shape (k, 3), got {X.shape}")
    n_samples, p = X.shape
    if n_samples <= p:
        raise ValueError(f"Need more than p={p} points for Tyler; got {n_samples}")

    # Robust spatial median using Weiszfeld iterations
    mu = X.mean(axis=0)

    for _ in range(100):
        d = np.linalg.norm(X - mu, axis=1)
        d = np.maximum(d, 1e-12)

        weights = 1.0 / d

        mu_new = (
            weights[:, None] * X
        ).sum(axis=0) / weights.sum()

        if np.linalg.norm(mu_new - mu) < 1e-10:
            mu = mu_new
            break

        mu = mu_new

    Xc = X - mu

    # If the neighborhood is degenerate, stop early
    if np.linalg.matrix_rank(Xc) < 2:
        raise ValueError(
            "Neighborhood is nearly rank-deficient."
        )

    Sigma = np.eye(p)

    for _ in range(max_iter):

        Sigma_reg = Sigma + ridge * np.eye(p)

        try:
            Sinv = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(Sigma_reg)

        quad = np.einsum(
            "ij,jk,ik->i",
            Xc,
            Sinv,
            Xc
        )

        quad = np.maximum(quad, 1e-12)

        Sigma_new = (
            (p / n_samples)
            * (Xc.T @ (Xc / quad[:, None]))
        )

        tr = np.trace(Sigma_new)

        if not np.isfinite(tr) or tr <= 1e-14:
            raise FloatingPointError(
                "Tyler update became invalid."
            )

        # Tyler is scale-free: fix trace to p
        Sigma_new *= p / tr

        if not np.all(np.isfinite(Sigma_new)):
            raise FloatingPointError(
                "Tyler produced NaN or Inf."
            )

        if np.linalg.norm(Sigma_new - Sigma, ord="fro") < tol:
            Sigma = Sigma_new
            break

        Sigma = Sigma_new

    return Sigma, mu


# 1. LOCAL FRAME


def local_frame_from_scatter(Sigma):
    """
    e1, e2, n : each shape (3,)
    eigvals   : descending order
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    e1 = eigvecs[:, 0]
    e2 = eigvecs[:, 1]
    n = eigvecs[:, 2]

    # Ensure a right-handed orthonormal frame
    n = np.cross(e1, e2)
    n /= max(np.linalg.norm(n), 1e-12)

    return e1, e2, n, eigvals


# 2. LOCAL COORDINATES

def project_to_local_frame(X, center, e1, e2, n):
    D = X - center
    u = D @ e1
    v = D @ e2
    w = D @ n
    return u, v, w


# 3. BERNSTEIN / BÉZIER MODEL

def bernstein_basis(x, degree):
    x = np.asarray(x, dtype=float)

    B = np.zeros((len(x), degree + 1))

    for a in range(degree + 1):
        B[:, a] = (
            comb(degree, a)
            * x**a
            * (1.0 - x)**(degree - a)
        )

    return B


def bezier_design_matrix(s, t, degree):
    """
    Tensor-product Bézier design matrix.

    h(s,t) = sum_ab P_ab B_a(s) B_b(t)
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    Bs = bernstein_basis(s, degree)
    Bt = bernstein_basis(t, degree)

    n = len(s)
    n_control = (degree + 1)**2

    A = np.zeros((n, n_control))

    col = 0

    for a in range(degree + 1):
        for b in range(degree + 1):
            A[:, col] = Bs[:, a] * Bt[:, b]
            col += 1

    return A


def fit_local_bezier_patch(
    X,
    center,
    e1,
    e2,
    n,
    degree=2,
    weighted=True,
    regularization=1e-8
):
    """
    Fits one local Bézier height patch.

    The patch is fitted in local coordinates:
        w = h(u,v)

    Returns a dictionary containing the frame, normalization,
    control heights, fit diagnostics, and local data.
    """
    u, v, w = project_to_local_frame(
        X, center, e1, e2, n
    )

    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()

    u_range = max(u_max - u_min, 1e-12)
    v_range = max(v_max - v_min, 1e-12)

    # Bézier parameters in [0,1]^2
    s = (u - u_min) / u_range
    t = (v - v_min) / v_range

    A = bezier_design_matrix(s, t, degree)

    n_control = A.shape[1]

    if len(w) < n_control:
        raise ValueError(
            f"Degree {degree} needs {n_control} control values, "
            f"but the neighborhood has only {len(w)} points."
        )

    # Optional center-weighting in the tangent plane
    if weighted:
        r = np.sqrt(u**2 + v**2)
        sigma_r = max(np.median(r), 1e-12)

        weights = np.exp(
            -0.5 * (r / sigma_r)**2
        )
    else:
        weights = np.ones_like(w)

    sqrt_weights = np.sqrt(weights)

    Aw = sqrt_weights[:, None] * A
    bw = sqrt_weights * w

    # Small ridge regularization for numerical stability
    A_aug = np.vstack([
        Aw,
        np.sqrt(regularization) * np.eye(n_control)
    ])

    b_aug = np.concatenate([
        bw,
        np.zeros(n_control)
    ])

    control_vector, _, rank, singular_values = (
        np.linalg.lstsq(
            A_aug,
            b_aug,
            rcond=None
        )
    )

    w_fitted = A @ control_vector

    residuals = w - w_fitted

    rmse = np.sqrt(
        np.mean(residuals**2)
    )

    nrmse = rmse / max(
        np.std(w),
        1e-12
    )

    return {
        "center": center,
        "e1": e1,
        "e2": e2,
        "n": n,
        "degree": degree,
        "control_vector": control_vector,
        "control_heights": control_vector.reshape(
            degree + 1,
            degree + 1
        ),
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "u_range": u_range,
        "v_range": v_range,
        "u": u,
        "v": v,
        "w": w,
        "s": s,
        "t": t,
        "w_fitted": w_fitted,
        "residuals": residuals,
        "rmse": rmse,
        "nrmse": nrmse,
        "rank": rank,
        "condition_number": np.linalg.cond(A),
        "weights": weights
    }


# 4. EVALUATE ONE PATCH AND MAP BACK TO GLOBAL 3D

def evaluate_local_bezier_patch(
    patch,
    grid_size=20,
    trim_to_local_disk=True
):
    """
    Evaluates one fitted local Bézier patch on a dense parameter grid
    and maps it back to global R^3.

    Returns
    -------
    surface_points : (m, 3)
    S, T, U, V, W  : grid arrays for plotting
    """
    degree = patch["degree"]

    s_axis = np.linspace(0.0, 1.0, grid_size)
    t_axis = np.linspace(0.0, 1.0, grid_size)

    S, T = np.meshgrid(
        s_axis,
        t_axis,
        indexing="xy"
    )

    s_flat = S.ravel()
    t_flat = T.ravel()

    A_grid = bezier_design_matrix(
        s_flat,
        t_flat,
        degree
    )

    W = (
        A_grid @ patch["control_vector"]
    ).reshape(
        grid_size,
        grid_size
    )

    U = (
        patch["u_min"]
        + S * patch["u_range"]
    )

    V = (
        patch["v_min"]
        + T * patch["v_range"]
    )

    center = patch["center"]
    e1 = patch["e1"]
    e2 = patch["e2"]
    n = patch["n"]

    global_grid = (
        center[None, None, :]
        + U[:, :, None] * e1[None, None, :]
        + V[:, :, None] * e2[None, None, :]
        + W[:, :, None] * n[None, None, :]
    )

    surface_points = global_grid.reshape(-1, 3)




# 5. BUILD THE GLOBAL COLLECTION OF LOCAL PATCHES

def build_global_local_bezier_surface(
    points,
    query_indices=None,
    n_query=100,
    k=30,
    degree=2,
    grid_size=15,
    weighted=True,
    trim_to_local_disk=True,
    random_state=0,
    verbose=True
):
    """
    Returns
    -------
    result dictionary containing:
        patches
        reconstructed_points
        query_indices
        neighbor_indices
        diagnostics
    """
    points = np.asarray(
        points,
        dtype=float
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {points.shape}"
        )

    N = len(points)

    if query_indices is None:

        rng = np.random.default_rng(
            random_state
        )

        n_query = min(
            n_query,
            N
        )

        query_indices = rng.choice(
            N,
            size=n_query,
            replace=False
        )

    query_indices = np.asarray(
        query_indices,
        dtype=int
    )

    query_points = (
        points[query_indices]
    )

    tree = cKDTree(points)

    # Query all centers at once.
    # Shape is (number_of_patches, k), with no extra singleton axis.
    _, neighbor_indices = tree.query(
        query_points,
        k=k
    )

    patches = []
    reconstructed_parts = []
    diagnostics = []

    previous_normal = None

    for i, idx in enumerate(neighbor_indices):

        neighborhood = points[idx]

        try:
            Sigma, center = (
                tyler_m_estimator(
                    neighborhood
                )
            )

            e1, e2, n, eigvals = (
                local_frame_from_scatter(
                    Sigma
                )
            )

            # Orient neighboring normals consistently.
            # This does not change the fitted geometric surface if all
            # local heights are computed with the same chosen normal.
            if previous_normal is not None:
                if np.dot(
                    n,
                    previous_normal
                ) < 0:
                    n = -n

            patch = fit_local_bezier_patch(
                neighborhood,
                center,
                e1,
                e2,
                n,
                degree=degree,
                weighted=weighted
            )

            patch["Sigma"] = Sigma
            patch["eigvals"] = eigvals
            patch["query_index"] = query_indices[i]
            patch["neighbor_indices"] = idx

            surface_points, S, T, U, V, W = (
                evaluate_local_bezier_patch(
                    patch,
                    grid_size=grid_size,
                    trim_to_local_disk=trim_to_local_disk
                )
            )

            patch["S"] = S
            patch["T"] = T
            patch["U"] = U
            patch["V"] = V
            patch["W"] = W

            patches.append(patch)
            reconstructed_parts.append(
                surface_points
            )

            diagnostics.append({
                "patch": i,
                "query_index": query_indices[i],
                "rmse": patch["rmse"],
                "nrmse": patch["nrmse"],
                "rank": patch["rank"],
                "condition_number": patch[
                    "condition_number"
                ],
                "status": "ok"
            })

            previous_normal = n

            if verbose:
                print(
                    f"Patch {i:03d} | "
                    f"RMSE={patch['rmse']:.4e} | "
                    f"NRMSE={patch['nrmse']:.3f} | "
                    f"cond={patch['condition_number']:.2e}"
                )

        except Exception as error:

            diagnostics.append({
                "patch": i,
                "query_index": query_indices[i],
                "status": "failed",
                "error": str(error)
            })

            if verbose:
                print(
                    f"Patch {i:03d} FAILED: {error}"
                )

    if len(reconstructed_parts) == 0:
        raise RuntimeError(
            "No local Bézier patch was successfully fitted."
        )

    reconstructed_points = np.vstack(
        reconstructed_parts
    )

    return {
        "patches": patches,
        "reconstructed_points": reconstructed_points,
        "query_indices": query_indices,
        "neighbor_indices": neighbor_indices,
        "diagnostics": diagnostics
    }


# ============================================================
# 6. VISUALIZATION
# ============================================================

def plot_global_reconstruction(
    original_points,
    reconstructed_points,
    max_original=5000,
    max_reconstructed=20000
):
    """
    Plot original point cloud and merged local Bézier reconstruction.
    """
    original_points = np.asarray(
        original_points
    )

    reconstructed_points = np.asarray(
        reconstructed_points
    )

    rng = np.random.default_rng(1)

    if len(original_points) > max_original:
        idx = rng.choice(
            len(original_points),
            max_original,
            replace=False
        )
        original_plot = original_points[idx]
    else:
        original_plot = original_points

    if len(reconstructed_points) > max_reconstructed:
        idx = rng.choice(
            len(reconstructed_points),
            max_reconstructed,
            replace=False
        )
        reconstructed_plot = (
            reconstructed_points[idx]
        )
    else:
        reconstructed_plot = (
            reconstructed_points
        )

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        original_plot[:, 0],
        original_plot[:, 1],
        original_plot[:, 2],
        s=2,
        alpha=0.25,
        label="Original point cloud"
    )

    ax.scatter(
        reconstructed_plot[:, 0],
        reconstructed_plot[:, 1],
        reconstructed_plot[:, 2],
        s=2,
        alpha=0.65,
        label="Merged local Bézier reconstruction"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        "Global Reconstruction from Overlapping Local Bézier Patches"
    )

    ax.legend()

    plt.show()


def plot_one_patch(
    points,
    patch
):
    """
    Plot one original neighborhood and its fitted local Bézier patch.
    """
    idx = patch["neighbor_indices"]

    neighborhood = points[idx]

    center = patch["center"]

    e1 = patch["e1"]
    e2 = patch["e2"]
    n = patch["n"]

    U = patch["U"]
    V = patch["V"]
    W = patch["W"]

    global_grid = (
        center[None, None, :]
        + U[:, :, None] * e1[None, None, :]
        + V[:, :, None] * e2[None, None, :]
        + W[:, :, None] * n[None, None, :]
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        neighborhood[:, 0],
        neighborhood[:, 1],
        neighborhood[:, 2],
        s=35,
        label="Neighborhood"
    )

    ax.plot_surface(
        global_grid[:, :, 0],
        global_grid[:, :, 1],
        global_grid[:, :, 2],
        alpha=0.55
    )

    ax.scatter(
        center[0],
        center[1],
        center[2],
        s=100,
        marker="x",
        label="Local center"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        "One Local Bézier Patch"
    )

    ax.legend()

    plt.show()


# ============================================================
# 7. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # IMPORTANT:
    # Replace this line with your own point cloud array.
    #
    # If you already have:
    #     points_array = np.asarray(pcd.points)
    #
    # then use:
    #     points = points_array
    # --------------------------------------------------------

    raise RuntimeError(
        "Load your point cloud and call "
        "build_global_local_bezier_surface(). "
        "See the example below in the comments."
    )


# Example notebook usage:
#
# points_array = np.asarray(pcd.points)
#
# result = build_global_local_bezier_surface(
#     points=points_array,
#     n_query=150,       # number of local patch centers
#     k=30,              # points per neighborhood
#     degree=2,          # quadratic Bézier patch
#     grid_size=15,      # samples per patch direction
#     weighted=True,
#     trim_to_local_disk=True,
#     random_state=42,
#     verbose=True
# )
#
# reconstructed_points = result["reconstructed_points"]
#
# print(
#     "Reconstructed point shape:",
#     reconstructed_points.shape
# )
#
# plot_global_reconstruction(
#     points_array,
#     reconstructed_points
# )
#
# # Inspect one patch:
# plot_one_patch(
#     points_array,
#     result["patches"][0]
# )
