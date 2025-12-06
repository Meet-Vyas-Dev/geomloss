"""
Quick Reference: All Available Distance Metrics in GeomLoss
============================================================

This script prints a categorized list of all available distance metrics.
"""

from geomloss import DISTANCE_METRICS

print("=" * 80)
print("GeomLoss Distance Metrics - Quick Reference")
print("=" * 80)
print(f"\nTotal metrics available: {len(DISTANCE_METRICS)}")
print("=" * 80)

# Organize metrics by family
families = {
    "📏 Lp (Minkowski) Family": [
        "minkowski", "manhattan", "cityblock", "l1", "taxicab",
        "euclidean", "l2", "chebyshev", "linf", "supremum", "max",
        "weighted_minkowski", "weighted_cityblock", "weighted_euclidean", "weighted_chebyshev"
    ],
    
    "1️⃣ L1 Family": [
        "sorensen", "dice", "czekanowski", "gower", "soergel",
        "kulczynski_d1", "canberra", "lorentzian"
    ],
    
    "🤝 Intersection Family": [
        "intersection", "wave_hedges", "czekanowski_similarity",
        "motyka", "kulczynski_s1", "tanimoto", "jaccard_distance", "ruzicka"
    ],
    
    "🧭 Inner Product Family": [
        "inner_product", "harmonic_mean", "cosine",
        "kumar_hassebrook", "pce", "jaccard", "dice_coefficient"
    ],
    
    "🎻 Squared-chord Family": [
        "fidelity", "bhattacharyya", "hellinger", "matusita", "squared_chord"
    ],
    
    "📊 Squared L2 (χ²) Family": [
        "pearson_chi2", "neyman_chi2", "squared_l2", "squared_euclidean",
        "probabilistic_symmetric_chi2", "divergence", "clark", "additive_symmetric_chi2"
    ],
    
    "⚛️ Shannon's Entropy Family": [
        "kl", "kullback_leibler", "jeffreys", "j_divergence",
        "k_divergence", "topsoe", "js", "jensen_shannon", "jensen_difference"
    ],
    
    "🧩 Combination Family": [
        "taneja", "kumar_johnson", "avg_l1_linf"
    ]
}

# Print each family
for family_name, metrics in families.items():
    print(f"\n{family_name}")
    print("-" * 80)
    available = [m for m in metrics if m in DISTANCE_METRICS]
    for i, metric in enumerate(available, 1):
        print(f"  {i:2d}. {metric}")

print("\n" + "=" * 80)
print("Classic GeomLoss Metrics (already implemented)")
print("=" * 80)
classic = ["sinkhorn", "hausdorff", "energy", "gaussian", "laplacian"]
for i, metric in enumerate(classic, 1):
    print(f"  {i}. {metric}")

print("\n" + "=" * 80)
print("Usage Examples")
print("=" * 80)

print("""
# Basic usage:
from geomloss import SamplesLoss
import torch

x = torch.randn((3, 100, 2))
y = torch.randn((3, 150, 2))

# Use any metric:
loss = SamplesLoss("cosine", blur=0.5)
result = loss(x, y)

# Commonly used metrics:
L_euclidean = SamplesLoss("euclidean", blur=0.5)      # Most common
L_manhattan = SamplesLoss("manhattan", blur=0.5)       # Less sensitive to outliers
L_cosine = SamplesLoss("cosine", blur=0.5)            # Angular similarity
L_hellinger = SamplesLoss("hellinger", blur=0.5)      # For probability distributions
L_kl = SamplesLoss("kl", blur=0.5)                    # Information theory

# All metrics support:
# - GPU: Just pass device="cuda" tensors
# - Batching: (batch_size, num_points, dimension)
# - Multiple backends: backend="tensorized"/"online"/"multiscale"
""")

print("=" * 80)
print("For detailed documentation, see: DISTANCE_METRICS.md")
print("=" * 80)
