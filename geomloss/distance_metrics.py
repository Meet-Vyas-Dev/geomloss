"""
Implements a comprehensive collection of distance metrics between point clouds.

Note: All distance-style functions return the positive distance D. 
Similarity-style functions return a positive similarity S.
The conversion to a kernel K = exp(-D/blur) is handled by the kernel_samples module.
"""

import numpy as np
import torch

try:
    from pykeops.torch import LazyTensor
    keops_available = True
except:
    keops_available = False


# ============================================================================== 
#                 Helper Functions for Distance Computation
# ==============================================================================

def _squared_distances(x, y, use_keops=False):
    """Compute squared Euclidean distances between points."""
    if use_keops and keops_available:
        if x.dim() == 2:
            x_i = LazyTensor(x[:, None, :])
            y_j = LazyTensor(y[None, :, :])
        elif x.dim() == 3:
            x_i = LazyTensor(x[:, :, None, :])
            y_j = LazyTensor(y[:, None, :, :])
        else:
            raise ValueError("Incorrect number of dimensions")
        return ((x_i - y_j) ** 2).sum(-1)
    else:
        if x.dim() == 2:
            D_xx = (x * x).sum(-1).unsqueeze(1)
            D_xy = torch.matmul(x, y.permute(1, 0))
            D_yy = (y * y).sum(-1).unsqueeze(0)
        elif x.dim() == 3:
            D_xx = (x * x).sum(-1).unsqueeze(2)
            D_xy = torch.matmul(x, y.permute(0, 2, 1))
            D_yy = (y * y).sum(-1).unsqueeze(1)
        else:
            raise ValueError("Incorrect number of dimensions")
        return D_xx - 2 * D_xy + D_yy


def _distances(x, y, use_keops=False):
    """Compute Euclidean distances between points."""
    if use_keops:
        return _squared_distances(x, y, use_keops=use_keops).sqrt()
    else:
        return torch.sqrt(torch.clamp_min(_squared_distances(x, y), 1e-8))


def _lazy_tensor(x, y, use_keops=False):
    """Create LazyTensor objects for x and y if using KeOps."""
    if not use_keops or not keops_available:
        return x, y

    if x.dim() == 2:
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
    elif x.dim() == 3:
        x_i = LazyTensor(x[:, :, None, :])
        y_j = LazyTensor(y[:, None, :, :])
    else:
        raise ValueError("Incorrect number of dimensions")
    return x_i, y_j


def _safe_div(numerator, denominator, eps=1e-8):
    """Safe division avoiding division by zero."""
    # Check if denominator is a LazyTensor
    if keops_available and hasattr(denominator, 'relu') and not isinstance(denominator, torch.Tensor):
        # LazyTensor case - use element-wise operations
        return numerator / (denominator + eps)
    else:
        # Regular tensor case
        return numerator / (denominator.clamp_min(eps))


def _safe_clamp_min(x, min_val=1e-8):
    """Safe clamping that works with both Tensors and LazyTensors."""
    # Check if it's a LazyTensor
    if keops_available and hasattr(x, 'relu') and not isinstance(x, torch.Tensor):
        # LazyTensor case - use element-wise addition for clamping
        return x + min_val
    else:
        # Regular tensor case
        return torch.clamp_min(x, min_val)



def _safe_log(x, eps=1e-8):
    """Safe logarithm avoiding log(0)."""
    # Check if it's a LazyTensor (has log method)
    if keops_available and hasattr(x, 'log') and not isinstance(x, torch.Tensor):
        # LazyTensor case - use element-wise operations
        return (x + eps).log()
    else:
        # Regular tensor case
        return torch.log(torch.clamp_min(x, eps))


# ==============================================================================
#                     Positivity decorator for required metrics
# ==============================================================================

# list of metric keys (in DISTANCE_METRICS registry) that require non-negative inputs
_POSITIVE_REQUIRED_SET = {
    # L1 / intersection / set-similarity family
    "sorensen", "dice", "czekanowski", "soergel", "kulczynski_d1", "canberra",
    "gower", "intersection", "wave_hedges", "tanimoto", "jaccard_distance",
    "ruzicka", "czekanowski_similarity", "motyka", "kulczynski_s1",

    # squared-chord + fidelity family
    "fidelity", "bhattacharyya", "hellinger", "squared_chord", "matusita",

    # chi^2 family
    "pearson_chi2", "neyman_chi2", "probabilistic_symmetric_chi2",
    "divergence", "clark", "additive_symmetric_chi2",

    # Shannon / information divergences
    "kl", "kullback_leibler", "jeffreys", "j_divergence",
    "k_divergence", "topsoe", "js", "jensen_shannon", "jensen_difference",

    # combination family which uses geometric mean etc.
    "taneja", "kumar_johnson"
}


def _requires_positive(fn):
    """Decorator that clamps x,y to eps where the metric requires positivity."""
    def wrapper(x, y, *args, **kwargs):
        # try to discover metric name from kwargs if provided (not required)
        metric_name = kwargs.get("metric_name", None)
        # fallback: if wrapper was applied manually we simply enforce positivity unconditionally
        # BUT to adhere to Option 1 we only clamp if this function is in the required set
        # we detect this by checking the original function's __name__ against the set
        name = metric_name or fn.__name__
        # map some function names to registry keys if necessary:
        name_key = name.lower()
        if name_key.startswith("kullback") or name_key.startswith("kl"):
            name_key = "kl"
        if name_key in _POSITIVE_REQUIRED_SET:
            eps = 1e-8
            x = torch.clamp(x, min=eps)
            y = torch.clamp(y, min=eps)
        return fn(x, y, *args, **kwargs)
    # preserve metadata
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ============================================================================== 
#                     Lp (Minkowski) Distance Family
# ==============================================================================

def minkowski_distance(x, y, p=2, blur=None, use_keops=False, ranges=None, **kwargs):
    """Lp (Minkowski) distance: ||x-y||_p = (sum_i |x_i - y_i|^p)^(1/p)

    Args:
        p: Order of the norm (default=2 for Euclidean)
    """
    if p == 2:
        return euclidean_distance(x, y, use_keops=use_keops, ranges=ranges)
    elif p == 1:
        return manhattan_distance(x, y, use_keops=use_keops, ranges=ranges)
    elif np.isinf(p):
        return chebyshev_distance(x, y, use_keops=use_keops, ranges=ranges)

    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = ((x_i - y_j).abs() ** p).sum(-1) ** (1.0 / p)
    else:
        D = torch.norm(x.unsqueeze(-2) - y.unsqueeze(-3), p=p, dim=-1)

    return D


def manhattan_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """L1 distance (Manhattan/City Block/Taxicab): sum_i |x_i - y_i|"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = (x_i - y_j).abs().sum(-1)
        if ranges is not None:
            D.ranges = ranges
    else:
        D = torch.sum(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1)

    return D


def euclidean_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """L2 distance (Euclidean): sqrt(sum_i (x_i - y_i)^2)"""
    if use_keops and keops_available:
        D = _squared_distances(x, y, use_keops=True).sqrt()
        if ranges is not None:
            D.ranges = ranges
    else:
        D = _distances(x, y, use_keops=False)

    return D


def chebyshev_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """L∞ distance (Chebyshev/Supremum/Max): max_i |x_i - y_i|"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        # KeOps max returns (value, index) when using reduction, so pick [0]
        D = (x_i - y_j).abs().max(dim=-1)[0]
    else:
        D = torch.max(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1)[0]

    return D


def weighted_minkowski_distance(x, y, weights=None, p=2, blur=None, use_keops=False, ranges=None, **kwargs):
    """Weighted Lp distance: (sum_i w_i * |x_i - y_i|^p)^(1/p)"""
    if weights is None:
        return minkowski_distance(x, y, p=p, use_keops=use_keops, ranges=ranges)

    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        w_i = LazyTensor(weights.view(1, 1, -1))
        D = (w_i * ((x_i - y_j).abs() ** p)).sum(-1) ** (1.0 / p)
    else:
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        D = (weights * (torch.abs(diff) ** p)).sum(-1) ** (1.0 / p)

    return D


# ============================================================================== 
#                                 L1 Family
# ==============================================================================

@_requires_positive
def sorensen_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Sørensen distance (Dice/Czekanowski): sum_i |x_i - y_i| / sum_i (x_i + y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = ((x_i - y_j).abs()).sum(-1)
        denominator = _safe_clamp_min((x_i + y_j).sum(-1), 1e-8)
        D = numerator / denominator
    else:
        numerator = torch.sum(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(x.unsqueeze(-2) + y.unsqueeze(-3), dim=-1).clamp_min(1e-8)
        D = numerator / denominator

    return D


def gower_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Gower distance: (1/d) * sum_i |x_i - y_i|"""
    d = x.shape[-1]
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = (x_i - y_j).abs().sum(-1) / d
    else:
        D = torch.sum(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1) / d

    return D


@_requires_positive
def soergel_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Soergel distance: sum_i |x_i - y_i| / sum_i max(x_i, y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = ((x_i - y_j).abs()).sum(-1)
        denominator = _safe_clamp_min((x_i.max(y_j)).sum(-1), 1e-8)
        D = numerator / denominator
    else:
        numerator = torch.sum(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(torch.max(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1).clamp_min(1e-8)
        D = numerator / denominator

    return D


@_requires_positive
def kulczynski_d1_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Kulczynski d1 distance: sum_i |x_i - y_i| / sum_i min(x_i, y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = ((x_i - y_j).abs()).sum(-1)
        denominator = _safe_clamp_min((x_i.min(y_j)).sum(-1), 1e-8)
        D = numerator / denominator
    else:
        numerator = torch.sum(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(torch.min(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1).clamp_min(1e-8)
        D = numerator / denominator

    return D


@_requires_positive
def canberra_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Canberra distance: sum_i |x_i - y_i| / (|x_i| + |y_i|)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i - y_j).abs()
        denominator = x_i.abs() + y_j.abs()
        D = (numerator / denominator.clamp_min(1e-8)).sum(-1)
    else:
        numerator = torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3))
        denominator = torch.abs(x.unsqueeze(-2)) + torch.abs(y.unsqueeze(-3))
        D = torch.sum(_safe_div(numerator, denominator), dim=-1)

    return D


def lorentzian_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Lorentzian distance: sum_i log(1 + |x_i - y_i|)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = ((x_i - y_j).abs() + 1).log().sum(-1)
    else:
        D = torch.sum(_safe_log(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)) + 1), dim=-1)

    return D


# ============================================================================== 
#                             Intersection Family
# ==============================================================================

@_requires_positive
def intersection_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Intersection distance: 1 - sum_i min(x_i, y_i) / sum_i max(x_i, y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i.min(y_j)).sum(-1)
        denominator = _safe_clamp_min((x_i.max(y_j)).sum(-1), 1e-8)
        D = 1 - numerator / denominator
    else:
        numerator = torch.sum(torch.min(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(torch.max(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1).clamp_min(1e-8)
        D = 1 - numerator / denominator

    return D


@_requires_positive
def wave_hedges_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Wave Hedges distance: sum_i (1 - min(x_i, y_i) / max(x_i, y_i))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        min_val = x_i.min(y_j)
        max_val = x_i.max(y_j)
        D = (1 - min_val / max_val.clamp_min(1e-8)).sum(-1)
    else:
        min_val = torch.min(x.unsqueeze(-2), y.unsqueeze(-3))
        max_val = torch.max(x.unsqueeze(-2), y.unsqueeze(-3))
        D = torch.sum(1 - _safe_div(min_val, max_val), dim=-1)

    return D


@_requires_positive
def czekanowski_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Czekanowski similarity: 2 * sum_i min(x_i, y_i) / sum_i (x_i + y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = 2 * (x_i.min(y_j)).sum(-1)
        denominator = _safe_clamp_min((x_i + y_j).sum(-1), 1e-8)
        S = numerator / denominator
    else:
        numerator = 2 * torch.sum(torch.min(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(x.unsqueeze(-2) + y.unsqueeze(-3), dim=-1).clamp_min(1e-8)
        S = numerator / denominator

    return S  # Returns similarity S


@_requires_positive
def motyka_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Motyka similarity: sum_i min(x_i, y_i) / sum_i (x_i + y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i.min(y_j)).sum(-1)
        denominator = _safe_clamp_min((x_i + y_j).sum(-1), 1e-8)
        S = numerator / denominator
    else:
        numerator = torch.sum(torch.min(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(x.unsqueeze(-2) + y.unsqueeze(-3), dim=-1).clamp_min(1e-8)
        S = numerator / denominator

    return S


@_requires_positive
def kulczynski_s1_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Kulczynski s1 similarity: sum_i min(x_i, y_i) / sum_i |x_i - y_i|"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i.min(y_j)).sum(-1)
        denominator = _safe_clamp_min((x_i - y_j).abs().sum(-1), 1e-8)
        S = numerator / denominator
    else:
        numerator = torch.sum(torch.min(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3)), dim=-1).clamp_min(1e-8)
        S = numerator / denominator

    return S


def tanimoto_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Tanimoto distance (Jaccard): 1 - sum_i min(x_i, y_i) / sum_i max(x_i, y_i)"""
    return intersection_distance(x, y, use_keops=use_keops, ranges=ranges, **kwargs)


@_requires_positive
def ruzicka_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Ruzicka similarity: sum_i min(x_i, y_i) / sum_i max(x_i, y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i.min(y_j)).sum(-1)
        denominator = _safe_clamp_min((x_i.max(y_j)).sum(-1), 1e-8)
        S = numerator / denominator
    else:
        numerator = torch.sum(torch.min(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1)
        denominator = torch.sum(torch.max(x.unsqueeze(-2), y.unsqueeze(-3)), dim=-1).clamp_min(1e-8)
        S = numerator / denominator

    return S


# ============================================================================== 
#                             Inner Product Family
# ==============================================================================

def inner_product_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Inner product similarity: sum_i x_i * y_i"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        S = (x_i * y_j).sum(-1)
    else:
        S = torch.sum(x.unsqueeze(-2) * y.unsqueeze(-3), dim=-1)

    return S


def harmonic_mean_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Harmonic mean similarity: 2 * sum_i (x_i * y_i) / sum_i (x_i + y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = 2 * (x_i * y_j).sum(-1)
        denominator = _safe_clamp_min((x_i + y_j).sum(-1), 1e-8)
        S = numerator / denominator
    else:
        numerator = 2 * torch.sum(x.unsqueeze(-2) * y.unsqueeze(-3), dim=-1)
        denominator = torch.sum(x.unsqueeze(-2) + y.unsqueeze(-3), dim=-1).clamp_min(1e-8)
        S = numerator / denominator

    return S


def cosine_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Cosine similarity: sum_i (x_i * y_i) / (||x|| * ||y||)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i * y_j).sum(-1)
        # PyKeOps compatible clamping using element-wise max
        x_norm = ((x_i ** 2).sum(-1).sqrt() + 1e-8).relu() + 1e-8
        y_norm = ((y_j ** 2).sum(-1).sqrt() + 1e-8).relu() + 1e-8
        S = numerator / (x_norm * y_norm)
    else:
        numerator = torch.sum(x.unsqueeze(-2) * y.unsqueeze(-3), dim=-1)
        x_norm = torch.sqrt(torch.sum(x.unsqueeze(-2) ** 2, dim=-1)).clamp_min(1e-8)
        y_norm = torch.sqrt(torch.sum(y.unsqueeze(-3) ** 2, dim=-1)).clamp_min(1e-8)
        S = numerator / (x_norm * y_norm)

    return S


def kumar_hassebrook_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Kumar-Hassebrook (PCE) similarity: sum_i (x_i * y_i) / (sum_i x_i^2 + sum_i y_i^2 - sum_i (x_i * y_i))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = (x_i * y_j).sum(-1)
        x_sq = (x_i ** 2).sum(-1)
        y_sq = (y_j ** 2).sum(-1)
        denominator = (x_sq + y_sq - numerator).clamp_min(1e-8)
        S = numerator / denominator
    else:
        numerator = torch.sum(x.unsqueeze(-2) * y.unsqueeze(-3), dim=-1)
        x_sq = torch.sum(x.unsqueeze(-2) ** 2, dim=-1)
        y_sq = torch.sum(y.unsqueeze(-3) ** 2, dim=-1)
        denominator = (x_sq + y_sq - numerator).clamp_min(1e-8)
        S = numerator / denominator

    return S


def jaccard_similarity(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Jaccard similarity (distinct from Tanimoto): sum_i (x_i * y_i) / (sum_i x_i^2 + sum_i y_i^2 - sum_i (x_i * y_i))"""
    return kumar_hassebrook_similarity(x, y, use_keops=use_keops, ranges=ranges)


def dice_coefficient(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Dice coefficient (distinct from Sørensen): 2 * sum_i (x_i * y_i) / (sum_i x_i^2 + sum_i y_i^2)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = 2 * (x_i * y_j).sum(-1)
        x_sq = (x_i ** 2).sum(-1)
        y_sq = (y_j ** 2).sum(-1)
        denominator = (x_sq + y_sq).clamp_min(1e-8)
        S = numerator / denominator
    else:
        numerator = 2 * torch.sum(x.unsqueeze(-2) * y.unsqueeze(-3), dim=-1)
        x_sq = torch.sum(x.unsqueeze(-2) ** 2, dim=-1)
        y_sq = torch.sum(y.unsqueeze(-3) ** 2, dim=-1)
        denominator = (x_sq + y_sq).clamp_min(1e-8)
        S = numerator / denominator

    return S


# ============================================================================== 
#                             Squared-chord Family
# ==============================================================================

@_requires_positive
def fidelity_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Fidelity distance: 1 - sum_i sqrt(x_i * y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        S = ((x_i * y_j).sqrt()).sum(-1)
    else:
        S = torch.sum(torch.sqrt(torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 0)), dim=-1)

    return 1 - S


@_requires_positive
def bhattacharyya_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Bhattacharyya distance: -log(sum_i sqrt(x_i * y_i))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        S = ((x_i * y_j).sqrt()).sum(-1)
        # Use LazyTensor-compatible log
        D = -(S + 1e-8).log()
    else:
        S = torch.sum(torch.sqrt(torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 0)), dim=-1)
        D = -_safe_log(S)

    return D


@_requires_positive
def hellinger_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Hellinger distance (Matusita): sqrt(2 * (1 - sum_i sqrt(x_i * y_i)))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        S = ((x_i * y_j).sqrt()).sum(-1)
        # PyKeOps returns standard tensor after reduction, so we can use torch operations
        D = ((2 * (1 - S)).relu()).sqrt()
    else:
        S = torch.sum(torch.sqrt(torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 0)), dim=-1)
        D = torch.sqrt(torch.clamp_min(2 * (1 - S), 0))

    return D


@_requires_positive
def squared_chord_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Squared-chord distance: sum_i (sqrt(x_i) - sqrt(y_i))^2"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = ((x_i.sqrt() - y_j.sqrt()) ** 2).sum(-1)
    else:
        sqrt_x = torch.sqrt(torch.clamp_min(x.unsqueeze(-2), 0))
        sqrt_y = torch.sqrt(torch.clamp_min(y.unsqueeze(-3), 0))
        D = torch.sum((sqrt_x - sqrt_y) ** 2, dim=-1)

    return D


# ============================================================================== 
#                           Squared L2 (χ²) Family
# ==============================================================================

@_requires_positive
def pearson_chi2_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Pearson χ² distance: sum_i (x_i - y_i)^2 / y_i"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = (((x_i - y_j) ** 2) / y_j.clamp_min(1e-8)).sum(-1)
    else:
        numerator = (x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2
        D = torch.sum(_safe_div(numerator, y.unsqueeze(-3)), dim=-1)

    return D


@_requires_positive
def neyman_chi2_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Neyman χ² distance: sum_i (x_i - y_i)^2 / x_i"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = (((x_i - y_j) ** 2) / x_i.clamp_min(1e-8)).sum(-1)
    else:
        numerator = (x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2
        D = torch.sum(_safe_div(numerator, x.unsqueeze(-2)), dim=-1)

    return D


def squared_l2_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Squared L2 distance (Squared Euclidean): sum_i (x_i - y_i)^2"""
    D = _squared_distances(x, y, use_keops=use_keops)
    return D


@_requires_positive
def probabilistic_symmetric_chi2_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Probabilistic Symmetric χ² distance: 2 * sum_i (x_i - y_i)^2 / (x_i + y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = 2 * ((x_i - y_j) ** 2)
        denominator = (x_i + y_j).clamp_min(1e-8)
        D = (numerator / denominator).sum(-1)
    else:
        numerator = 2 * (x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2
        denominator = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8)
        D = torch.sum(_safe_div(numerator, denominator), dim=-1)

    return D


@_requires_positive
def divergence_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Divergence distance: 2 * sum_i (x_i - y_i)^2 / (x_i + y_i)^2"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = 2 * ((x_i - y_j) ** 2)
        denominator = (x_i + y_j).clamp_min(1e-8) ** 2
        D = (numerator / denominator).sum(-1)
    else:
        numerator = 2 * (x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2
        denominator = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8) ** 2
        D = torch.sum(_safe_div(numerator, denominator), dim=-1)

    return D


@_requires_positive
def clark_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Clark distance: sqrt(sum_i ((x_i - y_i) / (x_i + y_i))^2)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        diff = x_i - y_j
        sum_val = (x_i + y_j).clamp_min(1e-8)
        D = ((diff / sum_val) ** 2).sum(-1).sqrt()
    else:
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        sum_val = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8)
        D = torch.sqrt(torch.sum((_safe_div(diff, sum_val)) ** 2, dim=-1))

    return D


@_requires_positive
def additive_symmetric_chi2_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Additive Symmetric χ² distance: sum_i ((x_i - y_i)^2 * (x_i + y_i)) / (x_i * y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = ((x_i - y_j) ** 2) * (x_i + y_j)
        denominator = (x_i * y_j).clamp_min(1e-8)
        D = (numerator / denominator).sum(-1)
    else:
        numerator = ((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2) * (x.unsqueeze(-2) + y.unsqueeze(-3))
        denominator = (x.unsqueeze(-2) * y.unsqueeze(-3)).clamp_min(1e-8)
        D = torch.sum(_safe_div(numerator, denominator), dim=-1)

    return D


# ============================================================================== 
#                          Shannon's Entropy Family
# ==============================================================================

@_requires_positive
def kullback_leibler_divergence(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Kullback-Leibler (KL) Divergence: sum_i x_i * log(x_i / y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = (x_i * ((x_i + 1e-8) / (y_j + 1e-8)).log()).sum(-1)
    else:
        ratio = _safe_div(x.unsqueeze(-2), y.unsqueeze(-3))
        D = torch.sum(x.unsqueeze(-2) * _safe_log(ratio), dim=-1)

    return D


@_requires_positive
def jeffreys_divergence(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Jeffreys (J) Divergence: sum_i (x_i - y_i) * log(x_i / y_i)"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = ((x_i - y_j) * ((x_i + 1e-8) / (y_j + 1e-8)).log()).sum(-1)
    else:
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        ratio = _safe_div(x.unsqueeze(-2), y.unsqueeze(-3))
        D = torch.sum(diff * _safe_log(ratio), dim=-1)

    return D


@_requires_positive
def k_divergence(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """K-divergence: sum_i x_i * log(2*x_i / (x_i + y_i))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        D = (x_i * ((2 * x_i + 1e-8) / (x_i + y_j + 1e-8)).log()).sum(-1)
    else:
        numerator = 2 * x.unsqueeze(-2)
        denominator = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8)
        D = torch.sum(x.unsqueeze(-2) * _safe_log(_safe_div(numerator, denominator)), dim=-1)

    return D


@_requires_positive
def topsoe_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Topsoe distance: sum_i (x_i * log(2*x_i / (x_i + y_i)) + y_i * log(2*y_i / (x_i + y_i)))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        # PyKeOps compatible clamping using element-wise operations
        sum_xy = (x_i + y_j) + 1e-8
        D = (x_i * ((2 * x_i + 1e-8) / sum_xy).log() +
             y_j * ((2 * y_j + 1e-8) / sum_xy).log()).sum(-1)
    else:
        sum_xy = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8)
        term1 = x.unsqueeze(-2) * _safe_log(_safe_div(2 * x.unsqueeze(-2), sum_xy))
        term2 = y.unsqueeze(-3) * _safe_log(_safe_div(2 * y.unsqueeze(-3), sum_xy))
        D = torch.sum(term1 + term2, dim=-1)

    return D


def jensen_shannon_divergence(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Jensen-Shannon (JS) Divergence: 0.5 * (sum_i x_i * log(2*x_i / (x_i + y_i)) + sum_i y_i * log(2*y_i / (x_i + y_i)))"""
    # topsoe_distance already clamps inside; reuse it with 0.5 factor
    D = topsoe_distance(x, y, use_keops=use_keops, ranges=ranges, **kwargs)
    return 0.5 * D


@_requires_positive
def jensen_difference(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Jensen difference: sum_i ((x_i * log(x_i) + y_i * log(y_i)) / 2 - ((x_i + y_i) / 2) * log((x_i + y_i) / 2))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        sum_xy = x_i + y_j
        avg_xy = sum_xy / 2
        term1 = (x_i * (x_i + 1e-8).log() + y_j * (y_j + 1e-8).log()) / 2
        term2 = avg_xy * (avg_xy + 1e-8).log()
        D = (term1 - term2).sum(-1)
    else:
        sum_xy = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8)
        avg_xy = sum_xy / 2
        term1 = (x.unsqueeze(-2) * _safe_log(x.unsqueeze(-2)) +
                 y.unsqueeze(-3) * _safe_log(y.unsqueeze(-3))) / 2
        term2 = avg_xy * _safe_log(avg_xy)
        D = torch.sum(term1 - term2, dim=-1)

    return D


# ============================================================================== 
#                             Combination Family
# ==============================================================================

@_requires_positive
def taneja_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Taneja distance: sum_i ((x_i + y_i) / 2) * log((x_i + y_i) / (2 * sqrt(x_i * y_i)))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        sum_xy = (x_i + y_j).clamp_min(1e-8)
        avg_xy = sum_xy / 2
        geom_mean = (x_i * y_j).sqrt().clamp_min(1e-8)
        D = (avg_xy * ((sum_xy) / (2 * geom_mean)).log()).sum(-1)
    else:
        sum_xy = (x.unsqueeze(-2) + y.unsqueeze(-3)).clamp_min(1e-8)
        avg_xy = sum_xy / 2
        geom_mean = torch.sqrt(torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 1e-8))
        D = torch.sum(avg_xy * _safe_log(_safe_div(sum_xy, 2 * geom_mean)), dim=-1)

    return D


@_requires_positive
def kumar_johnson_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Kumar-Johnson distance: sum_i (x_i^2 - y_i^2)^2 / (2 * (x_i * y_i)^(3/2))"""
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        numerator = ((x_i ** 2) - (y_j ** 2)) ** 2
        denominator = (2 * (x_i * y_j).clamp_min(1e-8) ** 1.5)
        D = (numerator / denominator).sum(-1)
    else:
        numerator = (x.unsqueeze(-2) ** 2 - y.unsqueeze(-3) ** 2) ** 2
        denominator = 2 * (torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 1e-8) ** 1.5)
        D = torch.sum(_safe_div(numerator, denominator), dim=-1)

    return D


def avg_l1_linf_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Avg (L1, L∞) distance: (L1 + L∞) / 2"""
    # Note: These functions now return positive distances
    l1 = manhattan_distance(x, y, use_keops=use_keops, ranges=ranges)
    linf = chebyshev_distance(x, y, use_keops=use_keops, ranges=ranges)
    D = (l1 + linf) / 2

    return D


# ============================================================================== 
#                       Distance Metrics Registry
# ==============================================================================

DISTANCE_METRICS = {
    # Lp (Minkowski) Distance Family
    "minkowski": minkowski_distance,
    "manhattan": manhattan_distance,
    "cityblock": manhattan_distance,
    "l1": manhattan_distance,
    "taxicab": manhattan_distance,
    "euclidean": euclidean_distance,
    "l2": euclidean_distance,
    "chebyshev": chebyshev_distance,
    "linf": chebyshev_distance,
    "supremum": chebyshev_distance,
    "max": chebyshev_distance,
    "weighted_minkowski": weighted_minkowski_distance,
    "weighted_cityblock": weighted_minkowski_distance,
    "weighted_euclidean": lambda x, y, **kw: weighted_minkowski_distance(x, y, p=2, **kw),
    "weighted_chebyshev": lambda x, y, **kw: weighted_minkowski_distance(x, y, p=np.inf, **kw),

    # L1 Family
    "sorensen": sorensen_distance,
    "dice": sorensen_distance,
    "czekanowski": sorensen_distance,
    "gower": gower_distance,
    "soergel": soergel_distance,
    "kulczynski_d1": kulczynski_d1_distance,
    "canberra": canberra_distance,
    "lorentzian": lorentzian_distance,

    # Intersection Family
    "intersection": intersection_distance,
    "wave_hedges": wave_hedges_distance,
    "czekanowski_similarity": czekanowski_similarity,
    "motyka": motyka_similarity,
    "kulczynski_s1": kulczynski_s1_similarity,
    "tanimoto": tanimoto_distance,
    "jaccard_distance": tanimoto_distance,
    "ruzicka": ruzicka_similarity,

    # Inner Product Family
    "inner_product": inner_product_similarity,
    "harmonic_mean": harmonic_mean_similarity,
    "cosine": cosine_similarity,
    "kumar_hassebrook": kumar_hassebrook_similarity,
    "pce": kumar_hassebrook_similarity,
    "jaccard": jaccard_similarity,
    "dice_coefficient": dice_coefficient,

    # Squared-chord Family
    "fidelity": fidelity_distance,
    "bhattacharyya": bhattacharyya_distance,
    "hellinger": hellinger_distance,
    "matusita": hellinger_distance,
    "squared_chord": squared_chord_distance,

    # Squared L2 (χ²) Family
    "pearson_chi2": pearson_chi2_distance,
    "neyman_chi2": neyman_chi2_distance,
    "squared_l2": squared_l2_distance,
    "squared_euclidean": squared_l2_distance,
    "probabilistic_symmetric_chi2": probabilistic_symmetric_chi2_distance,
    "divergence": divergence_distance,
    "clark": clark_distance,
    "additive_symmetric_chi2": additive_symmetric_chi2_distance,

    # Shannon's Entropy Family
    "kl": kullback_leibler_divergence,
    "kullback_leibler": kullback_leibler_divergence,
    "jeffreys": jeffreys_divergence,
    "j_divergence": jeffreys_divergence,
    "k_divergence": k_divergence,
    "topsoe": topsoe_distance,
    "js": jensen_shannon_divergence,
    "jensen_shannon": jensen_shannon_divergence,
    "jensen_difference": jensen_difference,

    # Combination Family
    "taneja": taneja_distance,
    "kumar_johnson": kumar_johnson_distance,
    "avg_l1_linf": avg_l1_linf_distance,
}


def get_distance_metric(name):
    """Retrieve a distance metric function by name.

    Args:
        name (str): Name of the distance metric

    Returns:
        callable: Distance metric function

    Raises:
        ValueError: If metric name is not found
    """
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name not in DISTANCE_METRICS:
        available = sorted(DISTANCE_METRICS.keys())
        raise ValueError(
            f"Unknown distance metric: '{name}'. "
            f"Available metrics: {', '.join(available[:10])}... "
            f"(and {len(available) - 10} more)"
        )
    return DISTANCE_METRICS[name]
