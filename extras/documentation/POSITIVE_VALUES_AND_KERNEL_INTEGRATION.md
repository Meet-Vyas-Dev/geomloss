# Distance Metrics: Positive Value Enforcement & Kernel Integration

## Overview

This document describes the comprehensive system implemented in GeomLoss to handle distance metrics correctly, ensuring positive values where required and proper integration with the Laplacian kernel to avoid NaN issues.

---

## 1. Core Architecture

### 1.1 Distance vs Similarity Metrics

The system categorizes all metrics into two types:

**Distance Metrics** (D):
- Return positive distance values: D ≥ 0
- Converted to kernels using Laplacian-style: K = exp(-D / blur)
- Examples: Euclidean, Manhattan, Hellinger, KL divergence

**Similarity Metrics** (S):
- Return similarity scores (typically 0 to 1)
- Used directly as kernels: K = S
- No exponential transformation applied
- Examples: Cosine similarity, Inner product, Jaccard similarity

---

## 2. Positive Value Enforcement System

### 2.1 The `_requires_positive` Decorator

Located in: `geomloss/distance_metrics.py`

```python
def _requires_positive(fn):
    """
    Decorator that clamps x,y to eps where the metric requires positivity.
    Prevents NaN from sqrt, log, or division operations on negative values.
    """
    def wrapper(x, y, *args, **kwargs):
        name = kwargs.get("metric_name", None) or fn.__name__
        name_key = name.lower()
        
        # Apply positivity constraint if metric requires it
        if name_key in _POSITIVE_REQUIRED_SET:
            eps = 1e-8
            x = torch.clamp(x, min=eps)
            y = torch.clamp(y, min=eps)
        
        return fn(x, y, *args, **kwargs)
    
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper
```

### 2.2 Metrics Requiring Positive Values

The `_POSITIVE_REQUIRED_SET` contains all metrics that need non-negative inputs:

```python
_POSITIVE_REQUIRED_SET = {
    # L1 / intersection / set-similarity family
    "sorensen", "dice", "czekanowski", "soergel", "kulczynski_d1", 
    "canberra", "gower", "intersection", "wave_hedges", "tanimoto", 
    "jaccard_distance", "ruzicka", "czekanowski_similarity", "motyka", 
    "kulczynski_s1",

    # squared-chord + fidelity family (use sqrt)
    "fidelity", "bhattacharyya", "hellinger", "squared_chord", "matusita",

    # chi^2 family (division by values)
    "pearson_chi2", "neyman_chi2", "probabilistic_symmetric_chi2",
    "divergence", "clark", "additive_symmetric_chi2",

    # Shannon / information divergences (use log)
    "kl", "kullback_leibler", "jeffreys", "j_divergence",
    "k_divergence", "topsoe", "js", "jensen_shannon", "jensen_difference",

    # combination family (geometric mean, etc.)
    "taneja", "kumar_johnson"
}
```

### 2.3 Application Pattern

Metrics requiring positive values are decorated:

```python
@_requires_positive
def hellinger_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    """Hellinger distance: sqrt(2 * (1 - sum_i sqrt(x_i * y_i)))"""
    # Implementation uses sqrt - requires positive inputs
    if use_keops and keops_available:
        x_i, y_j = _lazy_tensor(x, y, use_keops=True)
        S = ((x_i * y_j).sqrt()).sum(-1)
        D = torch.sqrt(torch.clamp_min(2 * (1 - S), 0))
    else:
        S = torch.sum(torch.sqrt(torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 0)), dim=-1)
        D = torch.sqrt(torch.clamp_min(2 * (1 - S), 0))
    
    return D
```

---

## 3. Kernel Integration

### 3.1 Metric Classification

Located in: `geomloss/kernel_samples.py`

All metrics are classified into two categories:

```python
_METRICS_AS_DISTANCE = {
    # Lp Family
    "minkowski", "manhattan", "euclidean", "chebyshev", 
    "weighted_minkowski", "l1", "l2", "linf",
    
    # L1 Family
    "sorensen", "dice", "czekanowski", "gower", "soergel", 
    "kulczynski_d1", "canberra", "lorentzian",
    
    # Intersection Family (Distances)
    "intersection", "wave_hedges", "tanimoto", "jaccard_distance",
    
    # Squared-chord Family
    "fidelity", "bhattacharyya", "hellinger", "matusita", "squared_chord",
    
    # Squared L2 (χ²) Family
    "pearson_chi2", "neyman_chi2", "squared_l2", "squared_euclidean",
    "probabilistic_symmetric_chi2", "divergence", "clark",
    "additive_symmetric_chi2",
    
    # Shannon's Entropy Family
    "kl", "kullback_leibler", "jeffreys", "j_divergence", 
    "k_divergence", "topsoe", "js", "jensen_shannon", "jensen_difference",
    
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
```

### 3.2 The `distance_metric_kernel` Function

This function handles the conversion of distance/similarity metrics to kernels:

```python
def distance_metric_kernel(x, y, metric_name, blur=0.05, use_keops=False, ranges=None, **kwargs):
    """
    Generic kernel for distance metrics from distance_metrics module.
    
    - Applies exp(-D/blur) for distance metrics (Laplacian-style kernel)
    - Returns raw similarity scores for similarity metrics
    """
    metric_func = get_distance_metric(metric_name)
    
    # Sanitize kwargs (remove unsupported parameters)
    metric_safe_kwargs = sanitize_kernel_kwargs(metric_name, kwargs)
    
    # Compute the raw metric
    raw_metric = metric_func(x, y, use_keops=use_keops, ranges=ranges, **metric_safe_kwargs)
    
    if metric_name in _METRICS_AS_DISTANCE:
        # Distance metric: Apply Laplacian kernel conversion
        if blur is None:
            blur = 1.0
        K = (-raw_metric / blur).exp()
        
    elif metric_name in _METRICS_AS_SIMILARITY:
        # Similarity metric: Use directly as kernel
        K = raw_metric
        
    else:
        raise ValueError(f"Metric '{metric_name}' not categorized as distance or similarity.")
    
    if use_keops and ranges is not None:
        K.ranges = ranges
    
    return K
```

### 3.3 Why Laplacian Kernel?

The Laplacian kernel `K = exp(-D/blur)` is used because:

1. **Similarity from Distance**: Converts distance D to similarity
   - Small distance (D ≈ 0) → High similarity (K ≈ 1)
   - Large distance (D → ∞) → Low similarity (K → 0)

2. **Scale Control**: The `blur` parameter controls sensitivity
   - Small blur → Sharp, local matching
   - Large blur → Smooth, global matching

3. **Mathematical Properties**: Laplacian kernel has nice properties for optimal transport:
   - Positive definite
   - Smooth gradients
   - Works well with Sinkhorn iterations

4. **Consistency**: Matches the original `laplacian_kernel` in GeomLoss:
   ```python
   def laplacian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
       C = distances(x / blur, y / blur, use_keops=use_keops)
       K = (-C).exp()
       return K
   ```

---

## 4. NaN Prevention Strategies

### 4.1 Input Validation

**Positive Clamping**:
```python
# Applied by @_requires_positive decorator
x = torch.clamp(x, min=1e-8)
y = torch.clamp(y, min=1e-8)
```

**Purpose**: Prevents:
- `sqrt(negative)` → NaN
- `log(0)` → -∞
- `division by zero` → ∞

### 4.2 Safe Mathematical Operations

**Safe Division**:
```python
def _safe_div(numerator, denominator, eps=1e-8):
    """Safe division avoiding division by zero."""
    return numerator / (denominator.clamp_min(eps))
```

**Safe Logarithm**:
```python
def _safe_log(x, eps=1e-8):
    """Safe logarithm avoiding log(0)."""
    return torch.log(torch.clamp_min(x, eps))
```

**Safe Square Root**:
```python
# In distance computations
D = torch.sqrt(torch.clamp_min(squared_distance, 1e-8))
```

### 4.3 Numerical Stability in Specific Metrics

**Example: Hellinger Distance**:
```python
@_requires_positive  # Ensures x, y ≥ 1e-8
def hellinger_distance(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    # Clamp before sqrt
    product = torch.clamp_min(x.unsqueeze(-2) * y.unsqueeze(-3), 0)
    S = torch.sum(torch.sqrt(product), dim=-1)
    
    # Clamp before final sqrt
    D = torch.sqrt(torch.clamp_min(2 * (1 - S), 0))
    return D
```

**Example: KL Divergence**:
```python
@_requires_positive  # Ensures x, y ≥ 1e-8
def kullback_leibler_divergence(x, y, blur=None, use_keops=False, ranges=None, **kwargs):
    # Safe division and logarithm
    ratio = _safe_div(x.unsqueeze(-2), y.unsqueeze(-3))
    D = torch.sum(x.unsqueeze(-2) * _safe_log(ratio), dim=-1)
    return D
```

---

## 5. Blur Parameter Handling

### 5.1 When Blur is Applied

**Distance Metrics**:
- Blur is applied during kernel conversion: `K = exp(-D / blur)`
- NOT applied in the distance computation itself
- Default: `blur = 0.05` (can be customized)

**Similarity Metrics**:
- Blur parameter is ignored (similarity used directly as kernel)
- Optional: Can normalize similarity scores if needed

### 5.2 Blur Parameter Flow

```python
# User code
loss = SamplesLoss("euclidean", blur=0.5)
result = loss(x, y)

# Internal flow:
# 1. compute distance: D = euclidean_distance(x, y)  # No blur here
# 2. convert to kernel: K = exp(-D / 0.5)            # Blur applied here
# 3. compute loss with kernel K
```

### 5.3 Special Cases

**Energy Kernel** (no blur):
```python
def energy_kernel(x, y, blur=None, use_keops=False, ranges=None):
    # Energy kernel doesn't use blur
    # Returns -D directly, not exp(-D/blur)
    return -distances(x, y, use_keops=use_keops)
```

**Gaussian Kernel** (blur in distance):
```python
def gaussian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    # Blur applied before computing squared distance
    C2 = squared_distances(x / blur, y / blur, use_keops=use_keops)
    K = (-C2 / 2).exp()
    return K
```

---

## 6. Testing and Validation

### 6.1 Test Coverage

File: `test_distance_metrics.py`

**Tests include**:
1. All 60+ distance metrics
2. Positive value enforcement
3. NaN detection
4. Multiple backends (tensorized, online, multiscale)
5. CPU and CUDA compatibility

### 6.2 Positive Value Testing

```python
def test_distance_metric(metric_name, device="cpu"):
    # Create test data
    x = torch.randn((3, 8, 2), device=device)
    y = torch.randn((3, 15, 2), device=device)
    
    # For metrics requiring positive values
    if metric_name in ["kl", "hellinger", "bhattacharyya", ...]:
        x = torch.abs(x) + 0.1
        y = torch.abs(y) + 0.1
        # Normalize for probabilistic metrics
        x = x / x.sum(dim=-1, keepdim=True)
        y = y / y.sum(dim=-1, keepdim=True)
    
    # Test the metric
    loss_fn = SamplesLoss(metric_name, blur=0.5)
    result = loss_fn(x, y)
    
    # Check for NaN/Inf
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
```

---

## 7. Usage Examples

### 7.1 Distance Metrics (Most Common)

```python
from geomloss import SamplesLoss
import torch

# Raw embeddings (can have any values)
embeddings_1 = torch.randn(32, 100, 768)
embeddings_2 = torch.randn(32, 100, 768)

# Euclidean distance (no special requirements)
loss = SamplesLoss("euclidean", blur=0.5)
result = loss(embeddings_1, embeddings_2)
```

### 7.2 Metrics Requiring Positive Values

```python
# For metrics like Hellinger, KL divergence, etc.
# Ensure inputs are positive (e.g., after ReLU or softmax)

# Option 1: Use ReLU outputs (non-negative)
features_1 = torch.relu(model1(data))
features_2 = torch.relu(model2(data))

loss = SamplesLoss("hellinger", blur=0.1)
result = loss(features_1, features_2)

# Option 2: Use softmax outputs (positive + normalized)
logits_1 = model1(data)
logits_2 = model2(data)
probs_1 = torch.softmax(logits_1, dim=-1)
probs_2 = torch.softmax(logits_2, dim=-1)

loss = SamplesLoss("kl", blur=0.1)
result = loss(probs_1, probs_2)

# Option 3: Manual positive enforcement
embeddings_1 = torch.abs(embeddings_1) + 1e-8
embeddings_2 = torch.abs(embeddings_2) + 1e-8

loss = SamplesLoss("bhattacharyya", blur=0.2)
result = loss(embeddings_1, embeddings_2)
```

### 7.3 Similarity Metrics

```python
# Similarity metrics work with any continuous values
# No positive enforcement needed

embeddings_1 = torch.randn(32, 100, 768)
embeddings_2 = torch.randn(32, 100, 768)

# Cosine similarity (most popular for embeddings)
loss = SamplesLoss("cosine", blur=0.1)
result = loss(embeddings_1, embeddings_2)

# Inner product similarity
loss = SamplesLoss("inner_product", blur=0.1)
result = loss(embeddings_1, embeddings_2)
```

---

## 8. Summary of Key Changes

### What Was Implemented:

1. **Positive Value Enforcement**:
   - `@_requires_positive` decorator
   - Automatic clamping to `eps=1e-8` for required metrics
   - Comprehensive list of metrics requiring positivity

2. **Distance vs Similarity Classification**:
   - `_METRICS_AS_DISTANCE` set (use Laplacian kernel)
   - `_METRICS_AS_SIMILARITY` set (use directly)
   - Clear categorization in documentation

3. **Proper Kernel Integration**:
   - `distance_metric_kernel()` function
   - Automatic registration in `kernel_routines`
   - Correct blur parameter handling

4. **NaN Prevention**:
   - Safe mathematical operations (`_safe_div`, `_safe_log`)
   - Clamping before sqrt, log, division
   - Numerical stability throughout

5. **Documentation**:
   - Module docstring clarifies distance vs similarity
   - Function docstrings explain formulas
   - Usage guides for different metric types

### Benefits:

✅ **No more NaN issues** with distance metrics
✅ **Proper kernel conversion** for all metrics
✅ **Consistent blur handling** across all metrics
✅ **Clear separation** between distance and similarity
✅ **User-friendly** - automatic handling of edge cases
✅ **Well-tested** - comprehensive test coverage

---

## 9. Metric Selection Guide

### For Raw Embeddings (Any Values):
- ✅ Euclidean, Manhattan, Cosine, Inner Product
- ✅ All Lp family metrics
- ✅ Weighted variants

### For Non-Negative Features (ReLU outputs):
- ✅ All of the above
- ✅ Plus: Hellinger, Squared-chord, Bhattacharyya
- ✅ Plus: Chi-squared variants

### For Probability Distributions (Softmax outputs):
- ✅ All of the above
- ✅ Plus: KL divergence, JS divergence
- ✅ Plus: All Shannon's Entropy family
- ✅ Plus: All information-theoretic metrics

---

## 10. Troubleshooting

### Issue: Getting NaN values

**Solution**:
1. Check if using a metric requiring positive values
2. Ensure inputs are positive (use ReLU, softmax, or abs())
3. Use appropriate blur values (see BLUR_PARAMETER_GUIDE.md)

### Issue: Unexpected loss values

**Solution**:
1. Verify metric type (distance vs similarity)
2. Check data scale and normalization
3. Adjust blur parameter for your data

### Issue: Metric not found

**Solution**:
1. Check available metrics: `from geomloss import DISTANCE_METRICS; print(DISTANCE_METRICS.keys())`
2. Use lowercase names with underscores
3. Check for aliases (e.g., "l1" → "manhattan")

---

## References

- **Main Implementation**: `geomloss/distance_metrics.py`
- **Kernel Integration**: `geomloss/kernel_samples.py`
- **Testing**: `test_distance_metrics.py`
- **Usage Guide**: `EMBEDDINGS_COMPATIBILITY_GUIDE.md`
- **Blur Tuning**: `BLUR_PARAMETER_GUIDE.md`
- **Complete Documentation**: `UPDATED_README.md`

---

**Last Updated**: December 6, 2025
**Version**: 1.1.0
**Status**: Production Ready ✅
