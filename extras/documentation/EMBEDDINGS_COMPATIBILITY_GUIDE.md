# Distance Metrics Compatibility Guide: Raw Embeddings vs Probability Distributions

## Summary

**YES! Most metrics (45+) work perfectly with raw feature embeddings (continuous vectors).**

Only a subset (~18 metrics) specifically require or assume probability distributions. This guide clarifies which metrics to use for your latent representations.

---

## ✅ Metrics for Raw Feature Embeddings (45+ metrics)

These metrics work with **any continuous-valued vectors** (latent representations, embeddings, feature vectors, etc.):

### 1. **Lp and L1 Family (7 metrics)** - ALL WORK WITH RAW EMBEDDINGS ✅

Perfect for continuous feature vectors:

- **`euclidean_distance`** ✅ - Most common choice for embeddings
  - Formula: `sqrt(sum((x - y)^2))`
  - Use case: General-purpose distance for neural network embeddings
  
- **`manhattan_distance`** ✅ - L1 distance
  - Formula: `sum(|x - y|)`
  - Use case: Robust to outliers, good for sparse embeddings
  
- **`chebyshev_distance`** ✅ - Maximum difference
  - Formula: `max(|x - y|)`
  - Use case: When you care about worst-case feature difference
  
- **`minkowski_distance`** ✅ - Generalized Lp
  - Formula: `(sum(|x - y|^p))^(1/p)`
  - Use case: Flexible distance with adjustable p parameter
  
- **`canberra_distance`** ✅ - Weighted L1
  - Formula: `sum(|x - y| / (|x| + |y|))`
  - Use case: Emphasizes small differences when values are small
  
- **`bray_curtis_distance`** ✅
  - Formula: `sum(|x - y|) / sum(|x + y|)`
  - Use case: Normalized L1 distance
  
- **`soergel_distance`** ✅
  - Formula: `sum(|x - y|) / sum(max(x, y))`
  - Use case: Ratio-based L1 variant

**Recommendation for embeddings**: Start with **`euclidean_distance`** or **`cosine_distance`** (see below).

---

### 2. **Inner Product Family (10 metrics)** - ALL WORK WITH RAW EMBEDDINGS ✅

Perfect for normalized or unit-length embeddings:

- **`cosine_distance`** ✅ - **HIGHLY RECOMMENDED for embeddings**
  - Formula: `1 - (x·y) / (||x|| ||y||)`
  - Use case: Direction-based similarity, standard for word embeddings, sentence embeddings, image features
  - **Very popular in deep learning!**
  
- **`inner_product_similarity`** ✅
  - Formula: `sum(x * y)`
  - Use case: Direct dot product similarity
  
- **`jaccard_distance`** ✅
  - Formula: Based on min/max operations
  - Use case: Set-like similarity for continuous features
  
- **`kumar_hassebrook_distance`** ✅
  - Use case: Alternative inner product-based metric
  
- **`czekanowski_distance`** ✅ (Dice variant)
  - Use case: Normalized inner product
  
- **`motyka_distance`** ✅
  - Use case: Min/sum ratio metric
  
- **`ruzicka_distance`** ✅
  - Use case: Min/max ratio metric
  
- **`tanimoto_distance`** ✅
  - Use case: Extended Jaccard for continuous values
  
- **`harmonic_mean_similarity`** ✅
  - Use case: Harmonic mean of features
  
- **`fidelity_similarity`** ✅
  - Formula: `sum(sqrt(x * y))`
  - Use case: Works with non-negative features (like ReLU outputs)

**Recommendation for embeddings**: **`cosine_distance`** is the gold standard for neural network embeddings.

---

### 3. **Intersection Family (12 metrics)** - MOST WORK WITH RAW EMBEDDINGS ✅

These work with any continuous values:

- **`intersection_distance`** ✅
  - Formula: Based on min/max operations
  - Use case: Continuous set-like similarity
  
- **`gower_distance`** ✅
  - Use case: Mixed-type distance (handles different feature types)
  
- **`kulczynski_distance`** ✅
  - Use case: Average of two ratios
  
- **`tanimoto_extended_distance`** ✅
  - Use case: Extended Jaccard for general vectors
  
- **`inner_product_distance`** ✅
  - Use case: Negative inner product as distance
  
- **`harmonic_mean_distance`** ✅
  - Use case: Harmonic mean-based metric
  
- **`kumar_johnson_distance`** ✅
  - Use case: Squared difference ratio
  
- **`avg_l1_linf_distance`** ✅
  - Formula: `(L1 + L∞) / 2`
  - Use case: Average of Manhattan and Chebyshev
  
- **`divergence_distance`** ✅
  - Formula: `2 * sum((x - y)^2 / (x + y)^2)`
  - Use case: Normalized squared difference (works with any positive features)
  
- **`dice_distance`** ✅
  - Use case: Dice coefficient for continuous values
  
- ⚠️ **`pearson_chi2_distance`** - Works but assumes non-negative values
- ⚠️ **`neyman_chi2_distance`** - Works but assumes non-negative values

---

### 4. **Squared-chord Family (6 metrics)** - Work with NON-NEGATIVE embeddings ⚠️

These assume **non-negative values** (like ReLU outputs, softmax features, etc.):

- **`squared_chord_distance`** ⚠️ (needs x, y ≥ 0)
  - Formula: `sum((sqrt(x) - sqrt(y))^2)`
  - Use case: Non-negative embeddings (e.g., after ReLU)
  
- **`hellinger_distance`** ⚠️ (needs x, y ≥ 0)
  - Formula: `sqrt(sum((sqrt(x) - sqrt(y))^2))`
  - Use case: Non-negative features
  
- **`matusita_distance`** ⚠️ (needs x, y ≥ 0)
  - Use case: Scaled Hellinger distance
  
- **`squared_chi2_distance`** ⚠️ (needs x, y ≥ 0)
- **`pearson_chi2_squared_distance`** ⚠️ (needs x, y ≥ 0)
- **`additive_symmetric_chi2_distance`** ⚠️ (needs x, y ≥ 0)

**Note**: These work if your embeddings are non-negative (e.g., output of ReLU, softmax, absolute values).

---

### 5. **Squared L2 Family (7 metrics)** - MOST WORK WITH RAW EMBEDDINGS ✅

- **`squared_euclidean_distance`** ✅ - **VERY COMMON for embeddings**
  - Formula: `sum((x - y)^2)`
  - Use case: Faster than Euclidean (no sqrt), very popular in deep learning
  
- **`clark_distance`** ✅
  - Formula: `sqrt(sum(((x - y) / (x + y))^2))`
  - Use case: Normalized squared difference
  
- **`sorensen_distance`** ✅
  - Use case: Normalized L1-like metric
  
- ⚠️ **`kl_divergence`** - Designed for probabilities but can work with normalized features
- ⚠️ **`jeffreys_divergence`** - Same as above
- ⚠️ **`k_divergence`** - Same as above
- ⚠️ **`topsoe_distance`** - Same as above

**Recommendation**: **`squared_euclidean_distance`** is extremely common for embeddings (faster than Euclidean).

---

### 6. **Combination Family (7 metrics)** - MIXED ⚠️

- **`avg_l1_linf_distance`** ✅ - Works with any embeddings
  - Formula: Average of L1 and L∞
  
- ⚠️ Others assume non-negative or probability-like values

---

## ❌ Metrics Specifically for Probability Distributions (18 metrics)

These metrics are **designed for probability distributions** and may give unexpected results with general embeddings:

### Shannon's Entropy Family (13 metrics) - PROBABILITY ONLY ❌

These require normalized probability distributions:

- `kullback_leibler_divergence` (KL) ❌
- `jensen_shannon_divergence` (JS) ❌
- `jensen_difference` ❌
- `bhattacharyya_distance` ❌
- `hellinger_entropy_distance` ❌
- `matusita_entropy_distance` ❌
- `squared_chord_entropy_distance` ❌
- `harmonic_mean_divergence` ❌
- `ag_mean_divergence` ❌
- `symmetric_kl_divergence` ❌
- `resistor_average_distance` ❌
- `probabilistic_symmetric_chi2` ❌
- `triangular_discrimination` ❌

### Why? 
These involve `log(x/y)`, entropy terms, and assume:
- Values sum to 1 (probability constraint)
- All values are non-negative
- Statistical properties of distributions

**When to use these**: Comparing softmax outputs, attention weights, categorical distributions, generative model outputs.

---

## 🎯 Recommended Metrics for Raw Feature Embeddings

Based on popularity in deep learning research:

### **Top 5 for General Embeddings:**

1. **`cosine_distance`** ✅
   - **Most popular** for embeddings (word2vec, BERT, ResNet features, etc.)
   - Scale-invariant, focuses on direction
   - Example: `SamplesLoss("cosine", blur=0.5)`

2. **`euclidean_distance`** ✅
   - **Classic choice**, considers both direction and magnitude
   - Standard Euclidean space geometry
   - Example: `SamplesLoss("euclidean", blur=0.5)`

3. **`squared_euclidean_distance`** ✅
   - **Faster** than Euclidean (no sqrt)
   - Common in loss functions (e.g., MSE)
   - Example: `SamplesLoss("squared_l2_distance", blur=0.5)`

4. **`manhattan_distance`** ✅
   - **Robust to outliers**
   - Good for sparse embeddings
   - Example: `SamplesLoss("manhattan", blur=0.5)`

5. **`inner_product_similarity`** ✅
   - Direct similarity measure
   - Fast computation
   - Example: `SamplesLoss("inner_product_similarity", blur=0.5)`

### **For Specific Use Cases:**

**Normalized/Unit-length embeddings** (e.g., L2-normalized features):
- `cosine_distance` ✅ (becomes equivalent to Euclidean on unit sphere)
- `inner_product_similarity` ✅

**Non-negative embeddings** (e.g., ReLU outputs):
- All the above ✅
- Plus: `hellinger_distance`, `squared_chord_distance`, `fidelity_similarity`

**High-dimensional sparse embeddings**:
- `manhattan_distance` ✅
- `canberra_distance` ✅

**When you want rotation invariance**:
- `euclidean_distance` ✅
- `squared_euclidean_distance` ✅

**When you want scale invariance**:
- `cosine_distance` ✅

---

## 💡 Practical Examples

### Example 1: BERT Embeddings (continuous, normalized)

```python
from geomloss import SamplesLoss
import torch

# BERT outputs (batch_size, seq_len, hidden_dim=768)
embeddings_1 = torch.randn(32, 100, 768)  # e.g., sentence embeddings
embeddings_2 = torch.randn(32, 100, 768)

# Best choices:
loss_cosine = SamplesLoss("cosine", blur=0.5)
loss_euclidean = SamplesLoss("euclidean", blur=0.5)
loss_squared = SamplesLoss("squared_l2_distance", blur=0.5)

result_cosine = loss_cosine(embeddings_1, embeddings_2)
result_euclidean = loss_euclidean(embeddings_1, embeddings_2)
result_squared = loss_squared(embeddings_1, embeddings_2)
```

### Example 2: ResNet Features (continuous, mixed sign)

```python
# ResNet50 features (batch_size, num_patches, feature_dim=2048)
features_1 = torch.randn(16, 49, 2048)  # Can have negative values
features_2 = torch.randn(16, 49, 2048)

# Good choices:
loss_cosine = SamplesLoss("cosine", blur=0.1)
loss_euclidean = SamplesLoss("euclidean", blur=0.1)
loss_manhattan = SamplesLoss("manhattan", blur=0.1)

result = loss_cosine(features_1, features_2)
```

### Example 3: After ReLU (non-negative)

```python
# Features after ReLU activation (all non-negative)
relu_features_1 = torch.relu(torch.randn(32, 100, 512))
relu_features_2 = torch.relu(torch.randn(32, 100, 512))

# More options available:
loss_cosine = SamplesLoss("cosine", blur=0.5)
loss_euclidean = SamplesLoss("euclidean", blur=0.5)
loss_hellinger = SamplesLoss("hellinger_distance", blur=0.5)  # Now works!
loss_fidelity = SamplesLoss("fidelity_similarity", blur=0.5)  # Now works!

result_cosine = loss_cosine(relu_features_1, relu_features_2)
result_hellinger = loss_hellinger(relu_features_1, relu_features_2)
```

### Example 4: Softmax Outputs (probabilities)

```python
# Softmax outputs (proper probability distributions)
logits_1 = torch.randn(32, 100, 10)
logits_2 = torch.randn(32, 100, 10)
probs_1 = torch.softmax(logits_1, dim=-1)  # Sum to 1
probs_2 = torch.softmax(logits_2, dim=-1)

# Now you can use probability metrics:
loss_kl = SamplesLoss("kl_divergence", blur=0.1)
loss_js = SamplesLoss("js_divergence", blur=0.1)
loss_hellinger = SamplesLoss("hellinger_distance", blur=0.1)
loss_bhattacharyya = SamplesLoss("bhattacharyya_distance", blur=0.1)

# Plus all the general metrics still work:
loss_cosine = SamplesLoss("cosine", blur=0.1)
loss_euclidean = SamplesLoss("euclidean", blur=0.1)
```

---

## 📊 Quick Reference Table

| Metric Family | Works with Raw Embeddings? | Requirements | Best For |
|---------------|---------------------------|--------------|----------|
| **Lp/L1** (7) | ✅ YES | Any continuous values | General embeddings |
| **Inner Product** (10) | ✅ YES | Any continuous values | Normalized embeddings |
| **Intersection** (12) | ✅ MOSTLY | Preferably non-negative | Set-like features |
| **Squared-chord** (6) | ⚠️ LIMITED | Non-negative values | ReLU outputs |
| **Squared L2** (partial) | ✅ SOME | Depends on specific metric | Squared distances |
| **Shannon's Entropy** (13) | ❌ NO | Probability distributions | Softmax outputs |
| **Combination** (7) | ⚠️ MIXED | Depends on specific metric | Specialized use |

---

## ✅ Final Recommendations

### For Your Latent Representations (Continuous Vectors):

**Start with these 3 metrics** (most widely used in deep learning):

```python
# Option 1: Cosine distance (most popular for embeddings)
loss = SamplesLoss("cosine", blur=0.5, backend="tensorized")

# Option 2: Euclidean distance (classic choice)
loss = SamplesLoss("euclidean", blur=0.5, backend="tensorized")

# Option 3: Squared Euclidean (faster, no sqrt)
loss = SamplesLoss("squared_l2_distance", blur=0.5, backend="tensorized")
```

**All 45+ general metrics work with your embeddings!** You have lots of options to experiment with.

**Avoid only the 13 Shannon's Entropy metrics** unless you're comparing probability distributions (like softmax outputs).

---

## 🔍 How to Know if a Metric Works with Embeddings

**✅ Safe for embeddings if the metric:**
- Uses basic operations: difference, absolute value, squared difference
- Uses dot products, norms, min/max operations
- Doesn't assume values sum to 1
- Examples: Euclidean, Cosine, Manhattan, Inner Product

**❌ Designed for probabilities if the metric:**
- Contains "divergence" or "entropy" in the name
- Uses logarithms of ratios (log(x/y))
- Assumes statistical properties
- Examples: KL divergence, JS divergence, Entropy-based metrics

**⚠️ Needs non-negative values if the metric:**
- Uses square roots of values (sqrt(x))
- Contains "chi-squared" in the name
- Examples: Hellinger, Squared-chord, Chi-squared variants

---

## Questions?

**Q: Can I use KL divergence on raw embeddings?**  
A: Not recommended. KL divergence assumes probability distributions (sum to 1, non-negative). For embeddings, use Cosine or Euclidean instead.

**Q: What if my embeddings have negative values?**  
A: Use: Euclidean, Cosine, Manhattan, Inner Product, or any Lp metric. Avoid: Hellinger, Squared-chord (they use sqrt which needs non-negative).

**Q: What's the best metric for BERT/transformer embeddings?**  
A: **Cosine distance** is the most popular choice. Euclidean is also commonly used.

**Q: Can I use multiple metrics?**  
A: Absolutely! Try different metrics and see which works best for your task.

**Q: What about the original GeomLoss metrics (Gaussian, Laplacian, Energy)?**  
A: These are **kernel-based** and work with any continuous embeddings. They're different from distance metrics but also suitable for embeddings.

---

**TL;DR**: **45+ metrics work with raw embeddings!** Use `cosine`, `euclidean`, or `squared_euclidean` as your first choices. Avoid only the 13 Shannon's Entropy metrics (those need probability distributions).
