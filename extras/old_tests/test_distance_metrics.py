"""
Test script for all distance metrics in the geomloss library.
This script validates that all newly implemented distance metrics work correctly.
"""

import torch
from geomloss import SamplesLoss, DISTANCE_METRICS
import warnings

def test_distance_metric(metric_name, device="cpu", dtype=torch.float32):
    """Test a single distance metric."""
    try:
        # Create sample data
        x = torch.randn((3, 8, 2), dtype=dtype, device=device)
        y = torch.randn((3, 15, 2), dtype=dtype, device=device)
        
        # Normalize to positive values for metrics that require it (e.g., KL divergence)
        if metric_name in ["kl", "kullback_leibler", "jeffreys", "j_divergence", 
                           "k_divergence", "topsoe", "js", "jensen_shannon", 
                           "jensen_difference", "bhattacharyya", "hellinger", 
                           "fidelity", "squared_chord"]:
            x = torch.abs(x) + 0.1
            y = torch.abs(y) + 0.1
            # Normalize to sum to 1 along feature dimension for probabilistic metrics
            x = x / x.sum(dim=-1, keepdim=True)
            y = y / y.sum(dim=-1, keepdim=True)
        
        # Test tensorized backend
        loss_fn = SamplesLoss(
            loss=metric_name,
            blur=0.5,
            backend="tensorized"
        )
        result = loss_fn(x, y)
        
        # Check if result is valid
        if torch.isnan(result).any() or torch.isinf(result).any():
            return f"❌ {metric_name}: NaN or Inf detected"
        
        return f"✓ {metric_name}: {result.mean().item():.6f}"
    
    except Exception as e:
        return f"❌ {metric_name}: {str(e)[:50]}"


def main():
    print("=" * 80)
    print("Testing GeomLoss Distance Metrics")
    print("=" * 80)
    
    # Test on CPU
    device = "cpu"
    dtype = torch.float32
    
    print(f"\nDevice: {device}, Dtype: {dtype}")
    print("-" * 80)
    
    # Group metrics by family
    families = {
        "Lp (Minkowski) Family": [
            "minkowski", "manhattan", "euclidean", "chebyshev",
            "weighted_minkowski", "l1", "l2", "linf"
        ],
        "L1 Family": [
            "sorensen", "gower", "soergel", "kulczynski_d1", 
            "canberra", "lorentzian"
        ],
        "Intersection Family": [
            "intersection", "wave_hedges", "czekanowski_similarity",
            "motyka", "kulczynski_s1", "tanimoto", "ruzicka"
        ],
        "Inner Product Family": [
            "inner_product", "harmonic_mean", "cosine",
            "kumar_hassebrook", "jaccard", "dice_coefficient"
        ],
        "Squared-chord Family": [
            "fidelity", "bhattacharyya", "hellinger", "squared_chord"
        ],
        "Squared L2 (χ²) Family": [
            "pearson_chi2", "neyman_chi2", "squared_l2",
            "probabilistic_symmetric_chi2", "divergence", 
            "clark", "additive_symmetric_chi2"
        ],
        "Shannon's Entropy Family": [
            "kl", "jeffreys", "k_divergence", "topsoe",
            "js", "jensen_difference"
        ],
        "Combination Family": [
            "taneja", "kumar_johnson", "avg_l1_linf"
        ]
    }
    
    # Test classic metrics first
    print("\n📊 Classic Metrics:")
    classic_metrics = ["energy", "gaussian", "laplacian"]
    for metric in classic_metrics:
        result = test_distance_metric(metric, device=device, dtype=dtype)
        print(f"  {result}")
    
    # Test all new metrics by family
    total_tested = 0
    total_passed = 0
    
    for family_name, metrics in families.items():
        print(f"\n📊 {family_name}:")
        for metric in metrics:
            if metric in DISTANCE_METRICS:
                result = test_distance_metric(metric, device=device, dtype=dtype)
                print(f"  {result}")
                total_tested += 1
                if "✓" in result:
                    total_passed += 1
    
    print("\n" + "=" * 80)
    print(f"Summary: {total_passed}/{total_tested} metrics passed")
    print("=" * 80)
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Testing on CUDA")
        print("=" * 80)
        
        device = "cuda"
        print(f"\nDevice: {device}")
        print("-" * 80)
        
        # Test a few representative metrics on CUDA
        test_metrics = ["euclidean", "cosine", "kl", "hellinger", "sorensen"]
        for metric in test_metrics:
            result = test_distance_metric(metric, device=device, dtype=dtype)
            print(f"  {result}")


def test_comparison_with_existing():
    """Compare new euclidean/manhattan implementations with existing energy kernel."""
    print("\n" + "=" * 80)
    print("Comparison Test: New vs Existing Implementations")
    print("=" * 80)
    
    device = "cpu"
    x = torch.randn((3, 8, 2), dtype=torch.float, device=device)
    y = torch.randn((3, 15, 2), dtype=torch.float, device=device)
    
    # Test L2 (euclidean) - should be similar to energy distance
    L_euclidean = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
    L_energy = SamplesLoss("energy", blur=0.5, backend="tensorized")
    
    result_euclidean = L_euclidean(x, y)
    result_energy = L_energy(x, y)
    
    print(f"\nEuclidean result: {result_euclidean.mean().item():.6f}")
    print(f"Energy result: {result_energy.mean().item():.6f}")
    print(f"Difference: {(result_euclidean - result_energy).abs().mean().item():.8f}")
    
    # Test that results are reasonable (both should be negative distances)
    if result_euclidean.mean() < 0 and result_energy.mean() < 0:
        print("✓ Both metrics return negative distances (as expected for kernel convention)")
    else:
        print("⚠ Warning: Expected negative distances")


def test_metric_properties():
    """Test mathematical properties of distance metrics."""
    print("\n" + "=" * 80)
    print("Property Tests")
    print("=" * 80)
    
    device = "cpu"
    torch.manual_seed(42)
    
    # Test symmetry for symmetric metrics
    x = torch.randn((2, 5, 3), dtype=torch.float, device=device)
    y = torch.randn((2, 5, 3), dtype=torch.float, device=device)
    
    symmetric_metrics = ["euclidean", "manhattan", "cosine", "chebyshev"]
    
    print("\n🔄 Testing Symmetry:")
    for metric in symmetric_metrics:
        L = SamplesLoss(metric, blur=0.5, backend="tensorized")
        result_xy = L(x, y)
        result_yx = L(y, x)
        diff = (result_xy - result_yx).abs().mean().item()
        
        if diff < 1e-5:
            print(f"  ✓ {metric}: symmetric (diff={diff:.2e})")
        else:
            print(f"  ⚠ {metric}: not symmetric (diff={diff:.2e})")
    
    # Test identity (distance from point to itself should be zero for true distances)
    print("\n🎯 Testing Identity (x, x):")
    identity_metrics = ["euclidean", "manhattan"]
    for metric in identity_metrics:
        L = SamplesLoss(metric, blur=0.5, backend="tensorized")
        result = L(x, x)
        
        if result.abs().mean().item() < 1e-4:
            print(f"  ✓ {metric}: d(x,x) ≈ 0 (value={result.mean().item():.2e})")
        else:
            print(f"  ⚠ {metric}: d(x,x) ≠ 0 (value={result.mean().item():.6f})")


if __name__ == "__main__":
    print("\n" + "🧪" * 40)
    print("GeomLoss Extended Distance Metrics Test Suite")
    print("🧪" * 40)
    
    main()
    test_comparison_with_existing()
    test_metric_properties()
    
    print("\n" + "🎉" * 40)
    print("Testing Complete!")
    print("🎉" * 40)
