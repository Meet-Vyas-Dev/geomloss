# Understanding the Blur Parameter in GeomLoss

## Quick Answer

**No, `blur=0.5` is NOT universal!** The optimal blur value depends on:
1. Your data scale and normalization
2. The distance metric you're using
3. Your specific application (matching, divergence, etc.)
4. The typical distances in your data

---

## What is the Blur Parameter?

The `blur` parameter controls the **scale/bandwidth** of the kernel smoothing in optimal transport computations. It determines how "spread out" the probability mass is around each point.

### Mathematical Role:

For most metrics, the blur parameter appears in the kernel as:
```
K(x, y) = exp(-d(x, y) / blur)
```

Or for some metrics:
```
K(x, y) = exp(-d(x, y) / blur²)
```

Where `d(x, y)` is the distance between points.

**Effect:**
- **Smaller blur** → Sharper/more local matching → Emphasizes nearby points
- **Larger blur** → Smoother/more global matching → Allows distant points to interact

---

## Blur Values by Distance Metric

### 1. **Cosine Distance** (Your Question!)

**Typical Range:** `blur = 0.01` to `blur = 2.0`

**Key Insight:** Cosine distance outputs are in the range `[0, 2]`:
- 0 = identical directions
- 1 = orthogonal vectors  
- 2 = opposite directions

**Recommended Values:**

```python
# For normalized embeddings (unit vectors):

# STRICT matching (only very similar vectors match):
blur = 0.05  # Very sharp, local matching

# MODERATE matching (similar vectors match):
blur = 0.1 to 0.3  # Good default range

# LOOSE matching (broader similarity):
blur = 0.5 to 1.0  # Allows more distant points to interact

# VERY LOOSE matching:
blur = 2.0  # Almost global averaging
```

**Rule of Thumb for Cosine:**
- Start with `blur = 0.1` for normalized embeddings
- Increase if you want smoother matching
- Decrease if you want stricter/sharper matching

**Example:**
```python
from geomloss import SamplesLoss
import torch

# L2-normalized embeddings (unit vectors)
embeddings_1 = torch.nn.functional.normalize(torch.randn(100, 768), dim=-1)
embeddings_2 = torch.nn.functional.normalize(torch.randn(100, 768), dim=-1)

# Conservative (strict matching)
loss_strict = SamplesLoss("cosine", blur=0.05)

# Balanced (moderate matching) - RECOMMENDED
loss_balanced = SamplesLoss("cosine", blur=0.1)

# Relaxed (loose matching)
loss_relaxed = SamplesLoss("cosine", blur=0.5)

print(f"Strict:   {loss_strict(embeddings_1, embeddings_2).item():.6f}")
print(f"Balanced: {loss_balanced(embeddings_1, embeddings_2).item():.6f}")
print(f"Relaxed:  {loss_relaxed(embeddings_1, embeddings_2).item():.6f}")
```

---

### 2. **Euclidean Distance**

**Typical Range:** Depends heavily on your data scale!

**Key Insight:** Euclidean distance can range from 0 to very large values depending on:
- Dimensionality of your embeddings
- Whether they're normalized
- The scale of your data

**Recommended Approach:**

```python
import torch

# Step 1: Measure typical distances in your data
embeddings = torch.randn(1000, 768)  # Your embeddings

# Compute pairwise distances (sample)
sample_1 = embeddings[:100]
sample_2 = embeddings[100:200]
distances = torch.cdist(sample_1, sample_2, p=2)

median_distance = distances.median().item()
mean_distance = distances.mean().item()
std_distance = distances.std().item()

print(f"Median distance: {median_distance:.4f}")
print(f"Mean distance: {mean_distance:.4f}")
print(f"Std distance: {std_distance:.4f}")

# Step 2: Set blur based on typical distance
# Rule: blur ≈ 0.1 to 0.5 × median_distance
blur_conservative = 0.1 * median_distance
blur_moderate = 0.3 * median_distance
blur_relaxed = 0.5 * median_distance

print(f"\nRecommended blur values:")
print(f"  Conservative: {blur_conservative:.4f}")
print(f"  Moderate: {blur_moderate:.4f}")
print(f"  Relaxed: {blur_relaxed:.4f}")
```

**Common Cases:**

```python
# Case 1: Normalized embeddings (unit vectors)
# Typical Euclidean distance: ~1.4 (since sqrt(2) for orthogonal unit vectors)
blur = 0.1 to 0.5  # Good range

# Case 2: Unnormalized BERT embeddings (large scale)
# Typical distance: ~10 to 100
blur = 1.0 to 10.0  # Scale accordingly

# Case 3: Small-scale features (0-1 range)
# Typical distance: ~0.5 to 2
blur = 0.05 to 0.5
```

---

### 3. **Manhattan (L1) Distance**

**Typical Range:** Usually larger than Euclidean

**Key Insight:** L1 distance ≈ sqrt(d) × L2 distance (roughly)

```python
# For normalized embeddings:
blur = 0.5 to 2.0  # Larger than Euclidean

# For high-dimensional data (d > 100):
blur = 1.0 to 5.0  # Even larger
```

---

### 4. **Squared Euclidean Distance**

**Typical Range:** Much larger values (distances are squared!)

```python
# If Euclidean blur would be 0.5:
blur_squared = 0.5 ** 2 = 0.25

# If Euclidean blur would be 1.0:
blur_squared = 1.0 ** 2 = 1.0

# Rule: blur_squared ≈ (blur_euclidean)²
```

---

## How to Choose Optimal Blur

### Method 1: Data-Driven (Recommended)

```python
import torch
from geomloss import SamplesLoss

def find_optimal_blur(embeddings_1, embeddings_2, metric="cosine", blur_range=None):
    """
    Find optimal blur by measuring typical distances.
    """
    # Default blur ranges by metric
    if blur_range is None:
        blur_ranges = {
            "cosine": (0.01, 2.0),
            "euclidean": (0.01, 10.0),
            "manhattan": (0.1, 10.0),
        }
        blur_range = blur_ranges.get(metric, (0.01, 10.0))
    
    # Sample some points to measure distances
    n_samples = min(200, len(embeddings_1), len(embeddings_2))
    sample_1 = embeddings_1[:n_samples]
    sample_2 = embeddings_2[:n_samples]
    
    # Compute distances with the specified metric
    if metric == "cosine":
        # Normalize and compute cosine distance
        s1_norm = torch.nn.functional.normalize(sample_1, dim=-1)
        s2_norm = torch.nn.functional.normalize(sample_2, dim=-1)
        # Cosine distance = 1 - cosine_similarity
        similarity = torch.matmul(s1_norm, s2_norm.T)
        distances = 1 - similarity
    elif metric == "euclidean":
        distances = torch.cdist(sample_1, sample_2, p=2)
    elif metric == "manhattan":
        distances = torch.cdist(sample_1, sample_2, p=1)
    else:
        # Fallback to Euclidean
        distances = torch.cdist(sample_1, sample_2, p=2)
    
    # Statistics
    median_dist = distances.median().item()
    mean_dist = distances.mean().item()
    percentile_25 = distances.quantile(0.25).item()
    percentile_75 = distances.quantile(0.75).item()
    
    # Recommended blur values
    blur_conservative = max(0.1 * median_dist, blur_range[0])
    blur_moderate = max(0.3 * median_dist, blur_range[0])
    blur_relaxed = min(0.5 * median_dist, blur_range[1])
    
    print(f"Distance Statistics for '{metric}':")
    print(f"  25th percentile: {percentile_25:.4f}")
    print(f"  Median:          {median_dist:.4f}")
    print(f"  Mean:            {mean_dist:.4f}")
    print(f"  75th percentile: {percentile_75:.4f}")
    print(f"\nRecommended blur values:")
    print(f"  Conservative (sharp matching):  {blur_conservative:.4f}")
    print(f"  Moderate (balanced):            {blur_moderate:.4f}")
    print(f"  Relaxed (smooth matching):      {blur_relaxed:.4f}")
    
    return blur_moderate

# Usage:
embeddings_1 = torch.randn(500, 768)
embeddings_2 = torch.randn(500, 768)

optimal_blur = find_optimal_blur(embeddings_1, embeddings_2, metric="cosine")

# Use the recommended blur
loss = SamplesLoss("cosine", blur=optimal_blur)
result = loss(embeddings_1, embeddings_2)
```

---

### Method 2: Cross-Validation

```python
import torch
from geomloss import SamplesLoss

def grid_search_blur(x_train, y_train, x_val, y_val, metric="cosine"):
    """
    Find best blur via validation set.
    """
    # Define blur candidates based on metric
    if metric == "cosine":
        blur_candidates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    elif metric == "euclidean":
        # Adjust based on your data scale
        blur_candidates = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    else:
        blur_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    
    print(f"Testing {len(blur_candidates)} blur values for {metric}:")
    for blur in blur_candidates:
        loss_fn = SamplesLoss(metric, blur=blur)
        
        # Compute loss on validation set
        val_loss = loss_fn(x_val, y_val).mean().item()
        results.append((blur, val_loss))
        
        print(f"  blur={blur:.4f}: val_loss={val_loss:.6f}")
    
    # Find minimum
    best_blur, best_loss = min(results, key=lambda x: x[1])
    print(f"\nBest blur: {best_blur:.4f} (val_loss={best_loss:.6f})")
    
    return best_blur

# Usage:
x_train = torch.randn(100, 768)
y_train = torch.randn(100, 768)
x_val = torch.randn(50, 768)
y_val = torch.randn(50, 768)

best_blur = grid_search_blur(x_train, y_train, x_val, y_val, metric="cosine")
```

---

### Method 3: Rule of Thumb by Application

```python
# Application: Contrastive Learning (pull similar, push dissimilar)
# Goal: Strong discrimination between positive and negative pairs
blur_contrastive = 0.05 to 0.1  # Sharp, local

# Application: Distribution Matching (align two sets)
# Goal: Smooth alignment of distributions
blur_matching = 0.3 to 0.7  # Moderate to relaxed

# Application: Clustering/Grouping
# Goal: Find natural clusters
blur_clustering = 0.1 to 0.3  # Moderate

# Application: Generative Model Training
# Goal: Realistic sample generation
blur_generative = 0.5 to 1.0  # Relaxed, smooth
```

---

## Specific Answer: Optimal Blur for Cosine Distance

### For Cosine Distance Specifically:

**Default Recommendation: `blur = 0.1`**

This works well for most normalized embeddings because:
- Cosine distances are typically in range [0, 2]
- Median cosine distance between random unit vectors ≈ 1.0
- `blur = 0.1` gives moderate smoothing

**Adjustment Guidelines:**

```python
from geomloss import SamplesLoss
import torch

# Normalized embeddings (BERT, ResNet, etc.)
embeddings = torch.nn.functional.normalize(torch.randn(100, 768), dim=-1)

# Conservative (strict): Only very similar vectors match
loss_strict = SamplesLoss("cosine", blur=0.05)
# Use when: You want precise matching, contrastive learning

# Balanced (default): Moderate similarity matching  
loss_balanced = SamplesLoss("cosine", blur=0.1)  # ← START HERE
# Use when: General-purpose embedding comparison

# Moderate: Broader matching
loss_moderate = SamplesLoss("cosine", blur=0.3)
# Use when: You want smoother gradients, distribution matching

# Relaxed: Very smooth matching
loss_relaxed = SamplesLoss("cosine", blur=0.7)
# Use when: Generative models, very smooth alignment
```

### Quick Test to Find Your Optimal Cosine Blur:

```python
import torch
from geomloss import SamplesLoss

# Your actual embeddings
embeddings_1 = your_embeddings_1  # e.g., BERT outputs
embeddings_2 = your_embeddings_2

# Normalize (important for cosine!)
e1_norm = torch.nn.functional.normalize(embeddings_1, dim=-1)
e2_norm = torch.nn.functional.normalize(embeddings_2, dim=-1)

# Test different blur values
blur_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

print("Cosine Distance Loss with Different Blur Values:")
for blur in blur_values:
    loss_fn = SamplesLoss("cosine", blur=blur)
    result = loss_fn(e1_norm, e2_norm).mean().item()
    print(f"blur={blur:.2f}: loss={result:.6f}")

# Choose the blur that gives reasonable gradients and matches your goal
```

---

## Common Blur Values by Use Case

| Use Case | Metric | Recommended Blur | Rationale |
|----------|--------|------------------|-----------|
| **Sentence Embeddings (BERT)** | Cosine | 0.1 - 0.3 | Normalized, moderate matching |
| **Image Features (ResNet)** | Cosine | 0.1 - 0.5 | Normalized, smoother for images |
| **Image Features (ResNet)** | Euclidean | 0.5 - 2.0 | Depends on feature scale |
| **Word2Vec Embeddings** | Cosine | 0.05 - 0.2 | Often normalized, precise matching |
| **Contrastive Learning** | Cosine | 0.05 - 0.1 | Sharp discrimination needed |
| **Point Cloud Registration** | Euclidean | 0.1 - 1.0 | Scale of point cloud |
| **Distribution Matching** | Any | 0.3 - 1.0 | Smoother alignment |
| **Generative Models** | Any | 0.5 - 2.0 | Very smooth gradients |

---

## Effects of Blur on Training

### Small Blur (e.g., 0.05 for Cosine):
- ✅ **Sharp, local matching**
- ✅ Strong discrimination
- ✅ Faster convergence for easy tasks
- ⚠️ Can be too strict, unstable gradients
- ⚠️ May miss broader patterns

### Medium Blur (e.g., 0.1-0.3 for Cosine):
- ✅ **Balanced matching**
- ✅ Stable gradients
- ✅ Good for most applications
- ✅ Captures local and some global structure

### Large Blur (e.g., 0.7-1.0 for Cosine):
- ✅ **Smooth, global matching**
- ✅ Very stable gradients
- ✅ Good for difficult optimization
- ⚠️ May be too permissive
- ⚠️ Slower convergence

---

## Practical Workflow

### Step 1: Start with Data-Driven Estimate

```python
# Measure typical distances in your data
def estimate_blur(embeddings_1, embeddings_2, metric="cosine"):
    if metric == "cosine":
        # Normalize
        e1 = torch.nn.functional.normalize(embeddings_1, dim=-1)
        e2 = torch.nn.functional.normalize(embeddings_2, dim=-1)
        # Cosine distance
        similarity = torch.matmul(e1[:100], e2[:100].T)
        distances = 1 - similarity
    else:
        distances = torch.cdist(embeddings_1[:100], embeddings_2[:100], p=2)
    
    median = distances.median().item()
    # Start with 30% of median distance
    return 0.3 * median

blur_initial = estimate_blur(your_embeddings_1, your_embeddings_2, "cosine")
print(f"Initial blur estimate: {blur_initial:.4f}")
```

### Step 2: Experiment with Range

```python
# Try ±50% around initial estimate
blur_low = blur_initial * 0.5
blur_high = blur_initial * 1.5

print(f"Try blur range: [{blur_low:.4f}, {blur_high:.4f}]")
```

### Step 3: Monitor During Training

```python
# During training, log losses with different blur values
for epoch in range(num_epochs):
    # Your training loop
    ...
    
    # Log with current blur
    loss = SamplesLoss("cosine", blur=current_blur)
    result = loss(embeddings_1, embeddings_2)
    
    # Optionally: Adapt blur during training
    # if gradients_unstable:
    #     current_blur *= 1.1  # Increase (smoother)
    # elif convergence_slow:
    #     current_blur *= 0.9  # Decrease (sharper)
```

---

## TL;DR - Quick Recommendations

### For Cosine Distance:
```python
# QUICK START (works for most cases):
blur = 0.1  # Default for normalized embeddings

# CONTRASTIVE LEARNING:
blur = 0.05  # Sharp discrimination

# GENERAL EMBEDDING COMPARISON:
blur = 0.1 to 0.3  # Balanced

# DISTRIBUTION MATCHING:
blur = 0.3 to 0.7  # Smooth alignment
```

### For Euclidean Distance:
```python
# Measure your typical distances first!
typical_distance = torch.cdist(sample_1, sample_2).median().item()

# Then:
blur = 0.3 * typical_distance  # Good starting point
```

### General Rule:
**`blur ≈ 0.1 to 0.5 × typical_distance_in_your_data`**

---

## Summary

**No, `blur=0.5` is not universal!**

- **For Cosine Distance**: Start with `blur=0.1` (for normalized embeddings)
- **For Euclidean Distance**: Measure your data first, then use `blur ≈ 0.3 × median_distance`
- **Smaller blur** = sharper matching, stricter
- **Larger blur** = smoother matching, more permissive
- **Always adapt** to your specific data scale and application

**Best Practice**: 
1. Measure typical distances in your data
2. Start with `blur = 0.3 × median_distance`
3. Experiment with ±50% range
4. Choose based on validation performance
