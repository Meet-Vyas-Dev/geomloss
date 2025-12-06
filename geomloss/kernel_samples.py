"""Implements kernel ("gaussian", "laplacian", "energy") norms between sampled measures.

.. math::
    \\text{Loss}(\\alpha,\\beta) 
        ~&=~ \\text{Loss}\\big( \sum_{i=1}^N \\alpha_i \,\delta_{x_i} \,,\, \sum_{j=1}^M \\beta_j \,\delta_{y_j} \\big) 
        ~=~ \\tfrac{1}{2} \|\\alpha-\\beta\|_k^2 \\\\
        &=~ \\tfrac{1}{2} \langle \\alpha-\\beta \,,\, k\star (\\alpha - \\beta) \\rangle \\\\
        &=~ \\tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N  \\alpha_i \\alpha_j \cdot k(x_i,x_j) 
          + \\tfrac{1}{2} \sum_{i=1}^M \sum_{j=1}^M  \\beta_i \\beta_j \cdot k(y_i,y_j) \\\\
        &-~\sum_{i=1}^N \sum_{j=1}^M  \\alpha_i \\beta_j \cdot k(x_i,y_j)

where k(x,y) is a similarity kernel. For distance metrics D(x,y),
this module implements the 'Laplacian' kernel k(x,y) = exp(-D(x,y) / blur).
"""

import numpy as np
import torch

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_sum
    from pykeops.torch.cluster import (
        grid_cluster,
        cluster_ranges_centroids,
        sort_clusters,
        from_matrix,
        swap_axes,
    )
    from pykeops.torch import LazyTensor

    keops_available = True
except:
    keops_available = False

from .utils import scal, squared_distances, distances
from .distance_metrics import DISTANCE_METRICS, get_distance_metric

# --- NEW: Define metric types ---
_METRICS_AS_DISTANCE = {
    # Lp Family
    "minkowski", "manhattan", "cityblock", "l1", "taxicab", "euclidean", "l2",
    "chebyshev", "linf", "supremum", "max", "weighted_minkowski",
    "weighted_cityblock", "weighted_euclidean", "weighted_chebyshev",
    # L1 Family
    "sorensen", "dice", "czekanowski", "gower", "soergel", "kulczynski_d1",
    "canberra", "lorentzian",
    # Intersection Family (Distances)
    "intersection", "wave_hedges", "tanimoto", "jaccard_distance",
    # Squared-chord Family
    "fidelity", "bhattacharyya", "hellinger", "matusita", "squared_chord",
    # Squared L2 (χ²) Family
    "pearson_chi2", "neyman_chi2", "squared_l2", "squared_euclidean",
    "probabilistic_symmetric_chi2", "divergence", "clark",
    "additive_symmetric_chi2",
    # Shannon's Entropy Family
    "kl", "kullback_leibler", "jeffreys", "j_divergence", "k_divergence",
    "topsoe", "js", "jensen_shannon", "jensen_difference",
    # Combination Family
    "taneja", "kumar_johnson", "avg_l1_linf",
}

_METRICS_AS_SIMILARITY = {
    # Intersection Family (Similarities)
    "czekanowski_similarity", "motyka", "kulczynski_s1", "ruzicka",
    # Inner Product Family
    "inner_product", "harmonic_mean", "cosine", "kumar_hassebrook",
    "pce", "jaccard", "dice_coefficient",
}
# --- End of new lists ---


# ==============================================================================
#                   KWARG SANITIZER FOR KERNEL FUNCTIONS
# ==============================================================================

# Kernels that accept only (x, y, blur, use_keops, ranges) or subsets thereof.
# We list some common kernel names and the kwargs they accept.
_ALLOWED_KERNEL_KWARGS = {
    "gaussian": {"blur", "use_keops", "ranges"},
    "laplacian": {"blur", "use_keops", "ranges"},
    "energy": {"use_keops", "ranges"},  # energy_kernel doesn't use blur in practice
    # distance metric kernels will be sanitized separately
}

# Distance metrics accept only (use_keops, ranges) + their own internal parameters.
_ALLOWED_DISTANCE_METRIC_KWARGS = {name: {"use_keops", "ranges"} for name in DISTANCE_METRICS.keys()}


def sanitize_kernel_kwargs(name, kwargs):
    """
    Remove unsupported keyword arguments before calling any kernel.
    Prevents errors like: energy_kernel() got unexpected keyword 'p'.
    """
    clean = {}

    if name in _ALLOWED_KERNEL_KWARGS:
        allowed = _ALLOWED_KERNEL_KWARGS[name]
        for k, v in kwargs.items():
            if k in allowed:
                clean[k] = v
        return clean

    # Otherwise it's likely a distance-metric kernel; do NOT pass 'blur' or other SamplesLoss kwargs
    if name in _ALLOWED_DISTANCE_METRIC_KWARGS:
        allowed = _ALLOWED_DISTANCE_METRIC_KWARGS[name]
        for k, v in kwargs.items():
            if k in allowed:
                clean[k] = v
        return clean

    # Fallback: pass only commonly safe kwargs if present
    for k in ("use_keops", "ranges"):
        if k in kwargs:
            clean[k] = kwargs[k]
    return clean


class DoubleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output


def double_grad(x):
    return DoubleGrad.apply(x)


# ============================================================================== 
#                             All backends
# ==============================================================================


def gaussian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    C2 = squared_distances(x / blur, y / blur, use_keops=use_keops)
    K = (-C2 / 2).exp()

    if use_keops and ranges is not None:
        K.ranges = ranges
    return K


def laplacian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    C = distances(x / blur, y / blur, use_keops=use_keops)
    K = (-C).exp()

    if use_keops and ranges is not None:
        K.ranges = ranges
    return K


def energy_kernel(x, y, blur=None, use_keops=False, ranges=None):
    # N.B.: We never truncate the energy distance kernel
    # Energy kernel is -D, not exp(-D/blur).
    return -distances(x, y, use_keops=use_keops)


# --- REPLACED function ---
def distance_metric_kernel(x, y, metric_name, blur=0.05, use_keops=False, ranges=None, **kwargs):
    """
    Generic kernel for distance metrics from distance_metrics module.
    Applies the exp(-D/blur) conversion for distance metrics.
    Returns raw similarity scores for similarity metrics.
    """
    metric_func = get_distance_metric(metric_name)

    # We pass sanitized kwargs to the metric func; metrics expect (use_keops, ranges) and their own params.
    metric_safe_kwargs = sanitize_kernel_kwargs(metric_name, kwargs)
    raw_metric = metric_func(x, y, use_keops=use_keops, ranges=ranges, **metric_safe_kwargs)

    if metric_name in _METRICS_AS_DISTANCE:
        # We got a distance D. Apply the Laplacian-style kernel conversion.
        # This matches the 'laplacian_kernel' convention: exp(-D/blur)
        if blur is None:
            blur = 1.0  # Use blur=1 as a default if not provided

        K = (-raw_metric / blur).exp()

    elif metric_name in _METRICS_AS_SIMILARITY:
        # We got a similarity S. Use it directly as the kernel.
        # Note: 'blur' is ignored for pure similarity metrics.
        K = raw_metric

    else:
        # Fallback/Error: This should not happen if lists are correct
        raise ValueError(f"Metric '{metric_name}' not categorized as distance or similarity.")

    if use_keops and ranges is not None:
        K.ranges = ranges
    return K
# --- End of replaced function ---


kernel_routines = {
    "gaussian": gaussian_kernel,
    "laplacian": laplacian_kernel,
    "energy": energy_kernel,
}

# Add all distance metrics from distance_metrics module to kernel_routines
for metric_name in DISTANCE_METRICS.keys():
    # Create a closure to capture metric_name properly and sanitize kwargs before calling
    def _make_closure(_name):
        return lambda x, y, blur=0.05, use_keops=False, ranges=None, **kwargs: \
            distance_metric_kernel(
                x, y, _name,
                blur=blur, use_keops=use_keops, ranges=ranges,
                **sanitize_kernel_kwargs(_name, kwargs)
            )
    kernel_routines[metric_name] = _make_closure(metric_name)


# --- MODIFIED function ---
def kernel_loss(
    α,
    x,
    β,
    y,
    blur=0.05,
    kernel=None,
    name=None,
    potentials=False,
    use_keops=False,
    ranges_xx=None,
    ranges_yy=None,
    ranges_xy=None,
    **kwargs
):
    # --- NEW: Add the KeOps Guard ---
    if use_keops and not keops_available:
        raise ImportError(
            "The 'online' backend for kernels requires pykeops. "
            "Install with: pip install pykeops"
        )
    # --- End of guard ---

    if kernel is None:
        if name is None:
            raise ValueError("Either 'kernel' or 'name' must be provided.")
        kernel = kernel_routines[name]

    # Determine kernel-name for sanitization (fallback to kernel.__name__ if name is None)
    kernel_name_for_sanitize = name if name is not None else getattr(kernel, "__name__", "")

    # sanitize kwargs for calls (this removes unsupported SamplesLoss kwargs)
    safe_kwargs = sanitize_kernel_kwargs(kernel_name_for_sanitize, kwargs)

    # ... (rest of function is unchanged)

    # (B,N,N) tensor
    K_xx = kernel(
        double_grad(x), x.detach(),
        blur=blur, use_keops=use_keops, ranges=ranges_xx, **safe_kwargs
    )
    # (B,M,M) tensor
    K_yy = kernel(
        double_grad(y), y.detach(),
        blur=blur, use_keops=use_keops, ranges=ranges_yy, **safe_kwargs
    )
    # (B,N,M) tensor
    K_xy = kernel(
        x, y,
        blur=blur, use_keops=use_keops, ranges=ranges_xy, **safe_kwargs
    )

    # (B,N,N) @ (B,N) = (B,N)
    a_x = (K_xx @ α.detach().unsqueeze(-1)).squeeze(-1)
    # (B,M,M) @ (B,M) = (B,M)
    b_y = (K_yy @ β.detach().unsqueeze(-1)).squeeze(-1)
    # (B,N,M) @ (B,M) = (B,N)
    b_x = (K_xy @ β.unsqueeze(-1)).squeeze(-1)

    if potentials:
        # (B,M,N) @ (B,N) = (B,M)
        Kt = K_xy.t() if use_keops else K_xy.transpose(1, 2)
        a_y = (Kt @ α.unsqueeze(-1)).squeeze(-1)
        return a_x - b_x, b_y - a_y

    else:  # Return the Kernel norm. N.B.: we assume that 'kernel' is symmetric:
        batch = x.dim() > 2
        return (
            0.5 * scal(double_grad(α), a_x, batch=batch)
            + 0.5 * scal(double_grad(β), b_y, batch=batch)
            - scal(α, b_x, batch=batch)
        )
# --- End of modified function ---


# ==============================================================================
#                       backend == "tensorized"
# ==============================================================================

from functools import partial

kernel_tensorized = partial(kernel_loss, use_keops=False)


# ==============================================================================
#                       backend == "online"
# ==============================================================================

kernel_online = partial(kernel_loss, use_keops=True)


# ==============================================================================
#                       backend == "multiscale"
# ==============================================================================


def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter


def kernel_multiscale(
    α,
    x,
    β,
    y,
    blur=0.05,
    kernel=None,
    name=None,
    truncate=5,
    diameter=None,
    cluster_scale=None,
    potentials=False,
    verbose=False,
    **kwargs
):
    if truncate is None or name == "energy":
        return kernel_online(
            α.unsqueeze(0),
            x.unsqueeze(0),
            β.unsqueeze(0),
            y.unsqueeze(0),
            blur=blur,
            kernel=kernel,
            truncate=truncate,
            name=name,
            potentials=potentials,
            **kwargs
        )

    # Renormalize our point cloud so that blur = 1:
    # Center the point clouds just in case, to prevent numeric overflows:
    center = (x.mean(-2, keepdim=True) + y.mean(-2, keepdim=True)) / 2
    x, y = x - center, y - center
    x_ = x / blur
    y_ = y / blur

    # Don't forget to normalize the diameter too!
    if cluster_scale is None:
        D = x.shape[-1]
        if diameter is None:
            diameter = max_diameter(x_.view(-1, D), y_.view(-1, D))
        else:
            diameter = diameter / blur
        cluster_scale = diameter / (np.sqrt(D) * 2000 ** (1 / D))

    # Put our points in cubic clusters:
    cell_diameter = cluster_scale * np.sqrt(x_.shape[-1])
    x_lab = grid_cluster(x_, cluster_scale)
    y_lab = grid_cluster(y_, cluster_scale)

    # Compute the ranges and centroids of each cluster:
    ranges_x, x_c, α_c = cluster_ranges_centroids(x_, x_lab, weights=α)
    ranges_y, y_c, β_c = cluster_ranges_centroids(y_, y_lab, weights=β)

    if verbose:
        print(
            "{}x{} clusters, computed at scale = {:2.3f}".format(
                len(x_c), len(y_c), cluster_scale
            )
        )

    # Sort the clusters, making them contiguous in memory:
    (α, x), x_lab = sort_clusters((α, x), x_lab)
    (β, y), y_lab = sort_clusters((β, y), y_lab)

    with torch.no_grad():  # Compute our block-sparse reduction ranges:
        # Compute pairwise distances between clusters:
        C_xx = squared_distances(x_c, x_c)
        C_yy = squared_distances(y_c, y_c)
        C_xy = squared_distances(x_c, y_c)

        # Compute the boolean masks:
        keep_xx = C_xx <= (truncate + cell_diameter) ** 2
        keep_yy = C_yy <= (truncate + cell_diameter) ** 2
        keep_xy = C_xy <= (truncate + cell_diameter) ** 2

        # Compute the KeOps reduction ranges:
        ranges_xx = from_matrix(ranges_x, ranges_x, keep_xx)
        ranges_yy = from_matrix(ranges_y, ranges_y, keep_yy)
        ranges_xy = from_matrix(ranges_x, ranges_y, keep_xy)

    return kernel_loss(
        α,
        x,
        β,
        y,
        blur=blur,
        kernel=kernel,
        name=name,
        potentials=potentials,
        use_keops=True,
        ranges_xx=ranges_xx,
        ranges_yy=ranges_yy,
        ranges_xy=ranges_xy,
        **kwargs
    )
