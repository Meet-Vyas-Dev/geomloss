# Extended Distance Metrics for GeomLoss

This document provides comprehensive documentation for all the newly implemented distance metrics in the GeomLoss library.

## Overview

The library has been extended with **60+ distance metrics** organized into 8 families:

1. **Lp (Minkowski) Distance Family** - 8 metrics
2. **L1 Family** - 6 metrics  
3. **Intersection Family** - 7 metrics
4. **Inner Product Family** - 6 metrics
5. **Squared-chord Family** - 4 metrics
6. **Squared L2 (χ²) Family** - 7 metrics
7. **Shannon's Entropy Family** - 6 metrics
8. **Combination Family** - 3 metrics

## Quick Start

```python
import torch
from geomloss import SamplesLoss

# Create sample data
x = torch.randn((3, 100, 2))  # 3 batches, 100 points, 2D
y = torch.randn((3, 150, 2))  # 3 batches, 150 points, 2D

# Use any distance metric
loss = SamplesLoss("cosine", blur=0.5)
result = loss(x, y)
```

## Distance Metric Families

### 1. Lp (Minkowski) Distance Family

The Minkowski distance generalizes many common distance metrics.

**Formula:** $L_p(x,y) = \left(\sum_i |x_i - y_i|^p\right)^{1/p}$

**Available Metrics:**

- `"minkowski"` - General Lp distance (specify p parameter)
- `"manhattan"`, `"cityblock"`, `"l1"`, `"taxicab"` - L1 distance ($p=1$)
- `"euclidean"`, `"l2"` - L2 distance ($p=2$)
- `"chebyshev"`, `"linf"`, `"supremum"`, `"max"` - L∞ distance ($p=\infty$)
- `"weighted_minkowski"` - Weighted Lp distance
- `"weighted_cityblock"` - Weighted L1
- `"weighted_euclidean"` - Weighted L2
- `"weighted_chebyshev"` - Weighted L∞

**Example:**
```python
# Manhattan distance (L1)
L1 = SamplesLoss("manhattan", blur=0.5, backend="tensorized")
result = L1(x, y)

# Euclidean distance (L2)
L2 = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
result = L2(x, y)

# Chebyshev distance (L∞)
Linf = SamplesLoss("chebyshev", blur=0.5, backend="tensorized")
result = Linf(x, y)

# Weighted Minkowski with custom weights
weights = torch.tensor([1.0, 2.0])  # Weight features differently
Lw = SamplesLoss("weighted_minkowski", blur=0.5, backend="tensorized")
result = Lw(x, y, weights=weights, p=2)
```

### 2. L1 Family

Variations and extensions of the L1 distance.

**Available Metrics:**

- `"sorensen"`, `"dice"`, `"czekanowski"` - Sørensen distance: $\frac{\sum_i |x_i - y_i|}{\sum_i (x_i + y_i)}$
- `"gower"` - Gower distance: $\frac{1}{d}\sum_i |x_i - y_i|$
- `"soergel"` - Soergel distance: $\frac{\sum_i |x_i - y_i|}{\sum_i \max(x_i, y_i)}$
- `"kulczynski_d1"` - Kulczynski d1: $\frac{\sum_i |x_i - y_i|}{\sum_i \min(x_i, y_i)}$
- `"canberra"` - Canberra distance: $\sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}$
- `"lorentzian"` - Lorentzian distance: $\sum_i \log(1 + |x_i - y_i|)$

**Example:**
```python
# Canberra distance (useful for data near zero)
canberra = SamplesLoss("canberra", blur=0.5)
result = canberra(x, y)

# Sørensen distance (normalized by sum)
sorensen = SamplesLoss("sorensen", blur=0.5)
result = sorensen(x, y)
```

### 3. Intersection Family

Metrics based on set intersection concepts.

**Available Metrics:**

- `"intersection"` - Intersection distance: $1 - \frac{\sum_i \min(x_i, y_i)}{\sum_i \max(x_i, y_i)}$
- `"wave_hedges"` - Wave Hedges: $\sum_i \left(1 - \frac{\min(x_i, y_i)}{\max(x_i, y_i)}\right)$
- `"czekanowski_similarity"` - Czekanowski similarity: $\frac{2\sum_i \min(x_i, y_i)}{\sum_i (x_i + y_i)}$
- `"motyka"` - Motyka similarity: $\frac{\sum_i \min(x_i, y_i)}{\sum_i (x_i + y_i)}$
- `"kulczynski_s1"` - Kulczynski s1: $\frac{\sum_i \min(x_i, y_i)}{\sum_i |x_i - y_i|}$
- `"tanimoto"`, `"jaccard_distance"` - Tanimoto/Jaccard distance
- `"ruzicka"` - Ruzicka similarity: $\frac{\sum_i \min(x_i, y_i)}{\sum_i \max(x_i, y_i)}$

**Example:**
```python
# Jaccard distance (for similarity comparison)
jaccard = SamplesLoss("tanimoto", blur=0.5)
result = jaccard(x, y)

# Ruzicka similarity
ruzicka = SamplesLoss("ruzicka", blur=0.5)
result = ruzicka(x, y)
```

### 4. Inner Product Family

Metrics based on inner products and angles.

**Available Metrics:**

- `"inner_product"` - Inner product similarity: $\sum_i x_i \cdot y_i$
- `"harmonic_mean"` - Harmonic mean similarity: $\frac{2\sum_i x_i y_i}{\sum_i (x_i + y_i)}$
- `"cosine"` - Cosine similarity: $\frac{\sum_i x_i y_i}{\|x\| \|y\|}$
- `"kumar_hassebrook"`, `"pce"` - Kumar-Hassebrook (PCE) similarity
- `"jaccard"` - Jaccard similarity (distinct from Tanimoto)
- `"dice_coefficient"` - Dice coefficient: $\frac{2\sum_i x_i y_i}{\sum_i x_i^2 + \sum_i y_i^2}$

**Example:**
```python
# Cosine similarity (angular distance)
cosine = SamplesLoss("cosine", blur=0.5)
result = cosine(x, y)

# Dice coefficient (overlap measure)
dice = SamplesLoss("dice_coefficient", blur=0.5)
result = dice(x, y)
```

### 5. Squared-chord Family

Metrics involving square roots, useful for probability distributions.

**Available Metrics:**

- `"fidelity"` - Fidelity distance: $1 - \sum_i \sqrt{x_i y_i}$
- `"bhattacharyya"` - Bhattacharyya distance: $-\log(\sum_i \sqrt{x_i y_i})$
- `"hellinger"`, `"matusita"` - Hellinger distance: $\sqrt{2(1 - \sum_i \sqrt{x_i y_i})}$
- `"squared_chord"` - Squared-chord: $\sum_i (\sqrt{x_i} - \sqrt{y_i})^2$

**Example:**
```python
# Hellinger distance (for probability distributions)
# Normalize data to be probability distributions
x_prob = torch.abs(x) + 0.1
x_prob = x_prob / x_prob.sum(dim=-1, keepdim=True)
y_prob = torch.abs(y) + 0.1
y_prob = y_prob / y_prob.sum(dim=-1, keepdim=True)

hellinger = SamplesLoss("hellinger", blur=0.5)
result = hellinger(x_prob, y_prob)
```

### 6. Squared L2 (χ²) Family

Chi-squared type metrics, commonly used in statistics.

**Available Metrics:**

- `"pearson_chi2"` - Pearson χ²: $\sum_i \frac{(x_i - y_i)^2}{y_i}$
- `"neyman_chi2"` - Neyman χ²: $\sum_i \frac{(x_i - y_i)^2}{x_i}$
- `"squared_l2"`, `"squared_euclidean"` - Squared Euclidean: $\sum_i (x_i - y_i)^2$
- `"probabilistic_symmetric_chi2"` - Probabilistic Symmetric χ²
- `"divergence"` - Divergence distance: $\frac{2\sum_i (x_i - y_i)^2}{(x_i + y_i)^2}$
- `"clark"` - Clark distance: $\sqrt{\sum_i \left(\frac{x_i - y_i}{x_i + y_i}\right)^2}$
- `"additive_symmetric_chi2"` - Additive Symmetric χ²

**Example:**
```python
# Squared Euclidean (faster than taking sqrt)
squared_l2 = SamplesLoss("squared_l2", blur=0.5)
result = squared_l2(x, y)

# Pearson chi-squared (statistical test)
pearson = SamplesLoss("pearson_chi2", blur=0.5)
result = pearson(x, y)
```

### 7. Shannon's Entropy Family

Information-theoretic divergences.

**Available Metrics:**

- `"kl"`, `"kullback_leibler"` - KL Divergence: $\sum_i x_i \log\frac{x_i}{y_i}$
- `"jeffreys"`, `"j_divergence"` - Jeffreys Divergence: $\sum_i (x_i - y_i)\log\frac{x_i}{y_i}$
- `"k_divergence"` - K-divergence: $\sum_i x_i \log\frac{2x_i}{x_i + y_i}$
- `"topsoe"` - Topsøe distance
- `"js"`, `"jensen_shannon"` - Jensen-Shannon Divergence
- `"jensen_difference"` - Jensen difference

**Example:**
```python
# KL Divergence (requires probability distributions)
x_prob = torch.abs(x) + 0.1
x_prob = x_prob / x_prob.sum(dim=-1, keepdim=True)
y_prob = torch.abs(y) + 0.1
y_prob = y_prob / y_prob.sum(dim=-1, keepdim=True)

kl = SamplesLoss("kl", blur=0.5)
result = kl(x_prob, y_prob)

# Jensen-Shannon (symmetric version of KL)
js = SamplesLoss("js", blur=0.5)
result = js(x_prob, y_prob)
```

### 8. Combination Family

Hybrid metrics combining multiple distance concepts.

**Available Metrics:**

- `"taneja"` - Taneja distance
- `"kumar_johnson"` - Kumar-Johnson distance
- `"avg_l1_linf"` - Average of L1 and L∞: $\frac{L_1 + L_\infty}{2}$

**Example:**
```python
# Average of L1 and L∞
avg_dist = SamplesLoss("avg_l1_linf", blur=0.5)
result = avg_dist(x, y)
```

## Backend Options

All metrics support three backends:

- `"tensorized"` - Full matrix computation (best for small problems)
- `"online"` - KeOps lazy evaluation (memory efficient)
- `"multiscale"` - Coarse-to-fine with clustering (best for large problems)

**Example:**
```python
# Tensorized backend (stores full distance matrix)
L_tensor = SamplesLoss("cosine", blur=0.5, backend="tensorized")

# Online backend (lazy evaluation, saves memory)
L_online = SamplesLoss("cosine", blur=0.5, backend="online")

# Multiscale backend (for very large point clouds)
L_multi = SamplesLoss("cosine", blur=0.5, backend="multiscale")
```

## Advanced Usage

### Custom Parameters

Some metrics accept additional parameters:

```python
# Minkowski with custom p
L_minkowski = SamplesLoss("minkowski", blur=0.5)
result = L_minkowski(x, y, p=3)  # p=3 Minkowski distance

# Weighted metrics
weights = torch.tensor([1.0, 2.0, 0.5])
L_weighted = SamplesLoss("weighted_euclidean", blur=0.5)
result = L_weighted(x, y, weights=weights)
```

### Batch Processing

All metrics support batched computation:

```python
# Batched data: (batch_size, num_points, dimension)
x = torch.randn((10, 100, 3))  # 10 batches
y = torch.randn((10, 150, 3))

L = SamplesLoss("euclidean", blur=0.5)
result = L(x, y)  # Returns (10,) tensor with one distance per batch
```

### GPU Acceleration

All metrics support CUDA:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn((3, 100, 2), device=device)
y = torch.randn((3, 150, 2), device=device)

L = SamplesLoss("cosine", blur=0.5)
result = L(x, y)  # Computed on GPU
```

## Choosing the Right Metric

### For General Use
- **Euclidean** (`"euclidean"`, `"l2"`) - Most common, good default
- **Manhattan** (`"manhattan"`, `"l1"`) - Less sensitive to outliers
- **Cosine** (`"cosine"`) - When direction matters more than magnitude

### For Probability Distributions
- **Hellinger** (`"hellinger"`) - Symmetric, bounded
- **KL Divergence** (`"kl"`) - Information theory applications
- **Jensen-Shannon** (`"js"`) - Symmetric version of KL

### For Sparse Data
- **Canberra** (`"canberra"`) - Good for data near zero
- **Jaccard** (`"tanimoto"`) - Set similarity

### For Statistical Analysis
- **Pearson χ²** (`"pearson_chi2"`) - Hypothesis testing
- **Bhattacharyya** (`"bhattacharyya"`) - Classification

## Performance Tips

1. **Use appropriate backend:**
   - Small datasets (<1000 points): `"tensorized"`
   - Medium datasets: `"online"`  
   - Large datasets (>10000 points): `"multiscale"`

2. **Choose metrics wisely:**
   - Simple metrics (L1, L2) are faster
   - Information-theoretic metrics (KL, JS) are slower

3. **Normalize data when needed:**
   - Probabilistic metrics require normalized inputs
   - Some metrics are scale-sensitive

## Complete Example

```python
import torch
from geomloss import SamplesLoss

# Generate sample point clouds
torch.manual_seed(42)
x = torch.randn((5, 200, 3))  # 5 batches, 200 points, 3D
y = torch.randn((5, 300, 3))  # 5 batches, 300 points, 3D

# Test different metrics
metrics = ["euclidean", "cosine", "manhattan", "canberra"]

for metric_name in metrics:
    loss_fn = SamplesLoss(metric_name, blur=0.5, backend="tensorized")
    result = loss_fn(x, y)
    print(f"{metric_name:15s}: {result.mean().item():.6f}")

# Use probability-based metrics
x_prob = torch.abs(x) + 0.01
x_prob = x_prob / x_prob.sum(dim=-1, keepdim=True)
y_prob = torch.abs(y) + 0.01
y_prob = y_prob / y_prob.sum(dim=-1, keepdim=True)

prob_metrics = ["hellinger", "js", "kl"]
for metric_name in prob_metrics:
    loss_fn = SamplesLoss(metric_name, blur=0.5, backend="tensorized")
    result = loss_fn(x_prob, y_prob)
    print(f"{metric_name:15s}: {result.mean().item():.6f}")
```

## Reference

For the complete list of available metrics, use:

```python
from geomloss import DISTANCE_METRICS
print(f"Available metrics: {len(DISTANCE_METRICS)}")
print(sorted(DISTANCE_METRICS.keys()))
```

To get a specific metric function:

```python
from geomloss import get_distance_metric
metric_func = get_distance_metric("euclidean")
```

## Citation

If you use these extended distance metrics, please cite the original GeomLoss library and acknowledge the extensions made by Meet J. Vyas.
