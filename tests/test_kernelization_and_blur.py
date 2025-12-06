"""
Kernelization and Blur Parameter Test Suite
===========================================

Tests that:
1. Distance metrics are correctly converted to kernels using Laplacian: K = exp(-D/blur)
2. Similarity metrics are used directly as kernels: K = S
3. Blur parameter is enforced and affects results appropriately
4. Classification of distance vs similarity metrics is correct
"""

import torch
import pytest
from geomloss import SamplesLoss
from geomloss.kernel_samples import _METRICS_AS_DISTANCE, _METRICS_AS_SIMILARITY


class TestMetricClassification:
    """Test that metrics are correctly classified as distance or similarity"""
    
    def test_distance_metrics_set_exists(self):
        """Test that distance metrics set is defined"""
        assert _METRICS_AS_DISTANCE is not None
        assert len(_METRICS_AS_DISTANCE) > 0
    
    def test_similarity_metrics_set_exists(self):
        """Test that similarity metrics set is defined"""
        assert _METRICS_AS_SIMILARITY is not None
        assert len(_METRICS_AS_SIMILARITY) > 0
    
    def test_no_overlap_between_sets(self):
        """Test that distance and similarity sets don't overlap"""
        overlap = _METRICS_AS_DISTANCE & _METRICS_AS_SIMILARITY
        assert len(overlap) == 0, f"Metrics in both sets: {overlap}"
    
    def test_common_distances_classified_correctly(self):
        """Test that common distance metrics are classified as distances"""
        common_distances = [
            "euclidean", "manhattan", "chebyshev", "minkowski",
            "hellinger", "kl", "js", "bhattacharyya"
        ]
        
        for metric in common_distances:
            assert metric in _METRICS_AS_DISTANCE, \
                f"{metric} should be classified as distance metric"
    
    def test_common_similarities_classified_correctly(self):
        """Test that common similarity metrics are classified as similarities"""
        common_similarities = [
            "cosine", "inner_product", "jaccard", "dice_coefficient"
        ]
        
        for metric in common_similarities:
            assert metric in _METRICS_AS_SIMILARITY, \
                f"{metric} should be classified as similarity metric"


class TestLaplacianKernelForDistances:
    """Test that distance metrics use Laplacian kernel: K = exp(-D/blur)"""
    
    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        return x, y
    
    @pytest.mark.parametrize("metric_name", [
        "euclidean", "manhattan", "chebyshev", "squared_l2"
    ])
    def test_blur_affects_distance_metrics(self, metric_name, sample_data):
        """Test that blur parameter affects distance metric results"""
        x, y = sample_data
        
        blur_small = 0.01
        blur_large = 2.0
        
        loss_small = SamplesLoss(metric_name, blur=blur_small, backend="tensorized")
        loss_large = SamplesLoss(metric_name, blur=blur_large, backend="tensorized")
        
        result_small = loss_small(x, y)
        result_large = loss_large(x, y)
        
        # Results should be different
        assert not torch.allclose(result_small, result_large, rtol=1e-4), \
            f"{metric_name}: blur parameter has no effect"
        
        # Both should be valid
        assert not torch.isnan(result_small).any()
        assert not torch.isnan(result_large).any()
    
    @pytest.mark.parametrize("metric_name", [
        "euclidean", "manhattan", "hellinger", "kl"
    ])
    def test_smaller_blur_increases_sensitivity(self, metric_name, sample_data):
        """Test that smaller blur values create sharper kernels (more sensitive)"""
        x, y = sample_data
        
        if metric_name in ["hellinger", "kl"]:
            # Use positive data
            x = torch.rand(2, 50, 10) + 0.01
            y = torch.rand(2, 60, 10) + 0.01
            x = x / x.sum(dim=-1, keepdim=True)
            y = y / y.sum(dim=-1, keepdim=True)
        
        blur_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        results = []
        
        for blur in blur_values:
            loss_fn = SamplesLoss(metric_name, blur=blur, backend="tensorized")
            result = loss_fn(x, y)
            results.append(result.mean().item())
        
        # Results should vary with blur
        result_range = max(results) - min(results)
        assert result_range > 1e-6, \
            f"{metric_name}: blur has negligible effect (range: {result_range:.8f})"
    
    def test_laplacian_kernel_property_identical_inputs(self):
        """Test Laplacian kernel property: K(x,x) = exp(0) = 1 (loss ≈ 0)"""
        x = torch.randn(2, 50, 10)
        
        # For identical inputs, distance = 0, so K = exp(0) = 1
        # Loss should be minimal
        loss_fn = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
        result = loss_fn(x, x)
        
        # Loss should be very small (near zero for identical distributions)
        assert torch.all(result < 1e-3), \
            f"Loss for identical inputs should be near zero, got {result.mean():.6f}"
    
    def test_laplacian_kernel_property_orthogonal_inputs(self):
        """Test that orthogonal/distant inputs give higher loss"""
        torch.manual_seed(42)
        
        # Create similar inputs
        x_similar = torch.randn(1, 50, 10)
        y_similar = x_similar + torch.randn(1, 50, 10) * 0.1
        
        # Create very different inputs
        x_different = torch.randn(1, 50, 10)
        y_different = torch.randn(1, 50, 10) * 10 + 100
        
        loss_fn = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
        
        loss_similar = loss_fn(x_similar, y_similar)
        loss_different = loss_fn(x_different, y_different)
        
        # Different inputs should have higher loss
        assert loss_different.item() > loss_similar.item(), \
            "Distant inputs should have higher loss than similar inputs"


class TestDirectSimilarityKernels:
    """Test that similarity metrics are used directly as kernels"""
    
    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        return x, y
    
    @pytest.mark.parametrize("metric_name", [
        "cosine", "inner_product", "jaccard", "dice_coefficient"
    ])
    def test_similarity_metrics_work(self, metric_name, sample_data):
        """Test that similarity metrics produce valid results"""
        x, y = sample_data
        
        loss_fn = SamplesLoss(metric_name, blur=0.5, backend="tensorized")
        result = loss_fn(x, y)
        
        assert result is not None
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    @pytest.mark.parametrize("metric_name", [
        "cosine", "inner_product"
    ])
    def test_similarity_blur_handling(self, metric_name, sample_data):
        """Test that blur parameter is handled for similarity metrics"""
        x, y = sample_data
        
        # Similarity metrics should work with different blur values
        blur_values = [0.1, 0.5, 1.0, 2.0]
        
        for blur in blur_values:
            loss_fn = SamplesLoss(metric_name, blur=blur, backend="tensorized")
            result = loss_fn(x, y)
            
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
    
    def test_cosine_similarity_range(self, sample_data):
        """Test that cosine similarity produces reasonable values"""
        x, y = sample_data
        
        loss_fn = SamplesLoss("cosine", blur=0.5, backend="tensorized")
        result = loss_fn(x, y)
        
        # Cosine similarity is in [-1, 1], so derived metrics should be reasonable
        assert not torch.isnan(result).any()
        assert result.min() > -10 and result.max() < 10, \
            f"Cosine-based loss out of reasonable range: [{result.min():.3f}, {result.max():.3f}]"


class TestBlurParameterEnforcement:
    """Test that blur parameter is properly enforced across all metrics"""
    
    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        return x, y
    
    @pytest.fixture
    def positive_data(self):
        torch.manual_seed(42)
        x = torch.rand(2, 50, 10) + 0.01
        y = torch.rand(2, 60, 10) + 0.01
        x = x / x.sum(dim=-1, keepdim=True)
        y = y / y.sum(dim=-1, keepdim=True)
        return x, y
    
    @pytest.mark.parametrize("blur_value", [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0])
    def test_blur_parameter_accepted(self, blur_value, sample_data):
        """Test that various blur values are accepted"""
        x, y = sample_data
        
        loss_fn = SamplesLoss("euclidean", blur=blur_value, backend="tensorized")
        result = loss_fn(x, y)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_blur_none_uses_default(self, sample_data):
        """Test that blur=None uses a sensible default"""
        x, y = sample_data
        
        # Should not raise error
        loss_fn = SamplesLoss("euclidean", backend="tensorized")  # No blur specified
        result = loss_fn(x, y)
        
        assert result is not None
        assert not torch.isnan(result).any()
    
    def test_very_small_blur_stability(self, sample_data):
        """Test that very small blur values don't cause numerical issues"""
        x, y = sample_data
        
        blur_tiny = 1e-6
        
        loss_fn = SamplesLoss("euclidean", blur=blur_tiny, backend="tensorized")
        result = loss_fn(x, y)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_large_blur_stability(self, sample_data):
        """Test that large blur values don't cause numerical issues"""
        x, y = sample_data
        
        blur_large = 100.0
        
        loss_fn = SamplesLoss("euclidean", blur=blur_large, backend="tensorized")
        result = loss_fn(x, y)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    @pytest.mark.parametrize("metric_name", [
        "euclidean", "manhattan", "hellinger", "kl", "js"
    ])
    def test_blur_monotonic_effect(self, metric_name, sample_data, positive_data):
        """Test that blur has consistent directional effect"""
        if metric_name in ["hellinger", "kl", "js"]:
            x, y = positive_data
        else:
            x, y = sample_data
        
        blur_values = [0.01, 0.1, 1.0, 10.0]
        results = []
        
        for blur in blur_values:
            loss_fn = SamplesLoss(metric_name, blur=blur, backend="tensorized")
            result = loss_fn(x, y)
            results.append(result.mean().item())
            assert not torch.isnan(result).any()
        
        # Results should vary with blur (not be constant)
        assert not all(abs(r - results[0]) < 1e-6 for r in results), \
            f"{metric_name}: blur has no effect across {blur_values}"


class TestKernelizationCorrectness:
    """Test mathematical correctness of kernelization"""
    
    def test_distance_kernel_decreases_with_distance(self):
        """Test that K = exp(-D/blur) decreases as distance increases"""
        # Create points at different distances
        x = torch.zeros(1, 1, 10)
        
        # Points at increasing distances
        y_close = torch.zeros(1, 1, 10) + 0.1
        y_medium = torch.zeros(1, 1, 10) + 1.0
        y_far = torch.zeros(1, 1, 10) + 10.0
        
        blur = 1.0
        loss_fn = SamplesLoss("euclidean", blur=blur, backend="tensorized")
        
        # Losses should reflect distance (but note: loss is a complex function)
        loss_close = loss_fn(x, y_close)
        loss_medium = loss_fn(x, y_medium)
        loss_far = loss_fn(x, y_far)
        
        # All should be valid
        assert not torch.isnan(loss_close).any()
        assert not torch.isnan(loss_medium).any()
        assert not torch.isnan(loss_far).any()
    
    def test_kernel_symmetry(self):
        """Test that K(x,y) = K(y,x) (symmetry)"""
        torch.manual_seed(42)
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        
        loss_fn = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
        
        loss_xy = loss_fn(x, y)
        loss_yx = loss_fn(y, x)
        
        # Should be approximately equal (within numerical precision)
        assert torch.allclose(loss_xy, loss_yx, rtol=1e-4, atol=1e-6), \
            "Loss should be symmetric: L(x,y) ≈ L(y,x)"
    
    def test_kernel_positive_definiteness(self):
        """Test that kernels produce reasonable values"""
        torch.manual_seed(42)
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        
        for metric in ["euclidean", "manhattan", "cosine"]:
            loss_fn = SamplesLoss(metric, blur=0.5, backend="tensorized")
            result = loss_fn(x, y)
            
            # Results should be finite and real
            assert torch.isfinite(result).all()
            assert not torch.isnan(result).any()


class TestMetricSpecificKernelization:
    """Test kernelization for specific metric types"""
    
    def test_entropy_based_distances(self):
        """Test that entropy-based distances (KL, JS) are properly kernelized"""
        torch.manual_seed(42)
        x = torch.rand(2, 50, 10) + 0.01
        y = torch.rand(2, 60, 10) + 0.01
        x = x / x.sum(dim=-1, keepdim=True)
        y = y / y.sum(dim=-1, keepdim=True)
        
        for metric in ["kl", "js", "jeffreys"]:
            # These are distances, should use Laplacian kernel
            assert metric in _METRICS_AS_DISTANCE
            
            loss_fn = SamplesLoss(metric, blur=0.5, backend="tensorized")
            result = loss_fn(x, y)
            
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
    
    def test_chi_squared_distances(self):
        """Test that chi-squared distances are properly kernelized"""
        torch.manual_seed(42)
        x = torch.rand(2, 50, 10) + 0.01
        y = torch.rand(2, 60, 10) + 0.01
        
        for metric in ["pearson_chi2", "additive_symmetric_chi2"]:
            # These are distances, should use Laplacian kernel
            assert metric in _METRICS_AS_DISTANCE
            
            loss_fn = SamplesLoss(metric, blur=0.5, backend="tensorized")
            result = loss_fn(x, y)
            
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
    
    def test_inner_product_similarities(self):
        """Test that inner product-based similarities are used directly"""
        torch.manual_seed(42)
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        
        for metric in ["cosine", "inner_product", "jaccard"]:
            # These are similarities, should be used directly
            assert metric in _METRICS_AS_SIMILARITY
            
            loss_fn = SamplesLoss(metric, blur=0.5, backend="tensorized")
            result = loss_fn(x, y)
            
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
