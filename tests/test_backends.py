"""
Backend-Specific Test Suite
============================

Comprehensive tests for all three backends:
- Tensorized (pure PyTorch)
- Online (PyKeOps lazy evaluation)
- Multiscale (hierarchical computation)

Tests backend-specific behavior and performance characteristics.
"""

import torch
import pytest
import warnings
from geomloss import SamplesLoss

# Suppress PyKeOps warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='pykeops')


class TestTensorizedBackend:
    """Tests specific to tensorized backend"""
    
    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        x = torch.randn(3, 100, 10)
        y = torch.randn(3, 120, 10)
        return x, y
    
    @pytest.mark.parametrize("metric_name", [
        "euclidean", "manhattan", "cosine", "chebyshev",
        "inner_product", "squared_l2", "minkowski"
    ])
    def test_tensorized_all_metrics(self, metric_name, sample_data):
        """Test that all distance metrics work with tensorized backend"""
        x, y = sample_data
        
        loss_fn = SamplesLoss(metric_name, blur=0.5, backend="tensorized")
        result = loss_fn(x, y)
        
        assert result is not None
        assert result.shape == (3,)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    @pytest.mark.parametrize("metric_name", [
        "hellinger", "kl", "js", "bhattacharyya", "sorensen"
    ])
    def test_tensorized_positive_metrics(self, metric_name):
        """Test positive-value metrics with tensorized backend"""
        torch.manual_seed(42)
        x = torch.rand(3, 100, 10) + 0.01
        y = torch.rand(3, 120, 10) + 0.01
        x = x / x.sum(dim=-1, keepdim=True)
        y = y / y.sum(dim=-1, keepdim=True)
        
        loss_fn = SamplesLoss(metric_name, blur=0.5, backend="tensorized")
        result = loss_fn(x, y)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_tensorized_cuda_support(self, sample_data):
        """Test tensorized backend with CUDA if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        x, y = sample_data
        x_cuda = x.cuda()
        y_cuda = y.cuda()
        
        loss_fn = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
        result = loss_fn(x_cuda, y_cuda)
        
        assert result.is_cuda
        assert not torch.isnan(result).any()


class TestOnlineBackend:
    """Tests specific to online (PyKeOps) backend"""
    
    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        x = torch.randn(2, 150, 8)
        y = torch.randn(2, 200, 8)
        return x, y
    
    @pytest.mark.parametrize("metric_name", [
        "euclidean", "manhattan", "cosine", "energy", "gaussian", "laplacian"
    ])
    def test_online_basic_metrics(self, metric_name, sample_data):
        """Test basic metrics with online backend"""
        x, y = sample_data
        
        try:
            loss_fn = SamplesLoss(metric_name, blur=0.5, backend="online")
            result = loss_fn(x, y)
            
            assert result is not None
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
        except Exception as e:
            # PyKeOps may not be available or configured
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip(f"PyKeOps not available: {str(e)}")
            else:
                raise
    
    def test_online_vs_tensorized_consistency(self, sample_data):
        """Test that online backend gives similar results to tensorized"""
        x, y = sample_data
        
        try:
            # Tensorized (reference)
            loss_tensorized = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
            result_tensorized = loss_tensorized(x, y)
            
            # Online
            loss_online = SamplesLoss("euclidean", blur=0.5, backend="online")
            result_online = loss_online(x, y)
            
            # Check consistency
            relative_error = torch.abs(result_online - result_tensorized) / (torch.abs(result_tensorized) + 1e-8)
            assert torch.max(relative_error) < 0.01, \
                f"Online backend differs from tensorized by {torch.max(relative_error):.6f}"
        
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip(f"PyKeOps not available: {str(e)}")
            else:
                raise
    
    def test_online_large_scale(self):
        """Test online backend with large point clouds (where it should excel)"""
        torch.manual_seed(42)
        x = torch.randn(2, 1000, 10)
        y = torch.randn(2, 1200, 10)
        
        try:
            loss_fn = SamplesLoss("euclidean", blur=0.5, backend="online")
            result = loss_fn(x, y)
            
            assert result is not None
            assert not torch.isnan(result).any()
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip(f"PyKeOps not available: {str(e)}")
            else:
                raise
    
    @pytest.mark.parametrize("metric_name", ["hellinger", "kl", "js"])
    def test_online_positive_metrics(self, metric_name):
        """Test positive-value metrics with online backend"""
        torch.manual_seed(42)
        x = torch.rand(2, 150, 8) + 0.01
        y = torch.rand(2, 200, 8) + 0.01
        x = x / x.sum(dim=-1, keepdim=True)
        y = y / y.sum(dim=-1, keepdim=True)
        
        try:
            loss_fn = SamplesLoss(metric_name, blur=0.5, backend="online")
            result = loss_fn(x, y)
            
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip(f"PyKeOps not available: {str(e)}")
            else:
                raise


class TestMultiscaleBackend:
    """Tests specific to multiscale backend"""
    
    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        # Multiscale needs more points to be effective
        x = torch.randn(2, 500, 10)
        y = torch.randn(2, 600, 10)
        return x, y
    
    @pytest.mark.parametrize("metric_name", [
        "euclidean", "manhattan", "cosine", "energy", "gaussian", "laplacian"
    ])
    def test_multiscale_basic_metrics(self, metric_name, sample_data):
        """Test basic metrics with multiscale backend"""
        x, y = sample_data
        
        try:
            loss_fn = SamplesLoss(metric_name, blur=0.5, backend="multiscale")
            result = loss_fn(x, y)
            
            assert result is not None
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
        except Exception as e:
            # PyKeOps may not be available
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip(f"PyKeOps not available: {str(e)}")
            else:
                raise
    
    def test_multiscale_vs_tensorized_consistency(self, sample_data):
        """Test that multiscale backend gives similar results to tensorized"""
        x, y = sample_data
        
        try:
            # Tensorized (reference)
            loss_tensorized = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
            result_tensorized = loss_tensorized(x, y)
            
            # Multiscale
            loss_multiscale = SamplesLoss("euclidean", blur=0.5, backend="multiscale")
            result_multiscale = loss_multiscale(x, y)
            
            # Check consistency (multiscale is approximate, so allow larger tolerance)
            relative_error = torch.abs(result_multiscale - result_tensorized) / (torch.abs(result_tensorized) + 1e-8)
            assert torch.max(relative_error) < 0.05, \
                f"Multiscale backend differs from tensorized by {torch.max(relative_error):.6f}"
        
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip(f"PyKeOps not available: {str(e)}")
            else:
                raise
    
    def test_multiscale_hierarchical_nature(self):
        """Test that multiscale works with hierarchical computation"""
        torch.manual_seed(42)
        # Large dataset where multiscale should provide efficiency
        # Note: multiscale doesn't support batch size > 1, so use batch=1
        x = torch.randn(1, 2000, 10)
        y = torch.randn(1, 2500, 10)
        
        try:
            loss_fn = SamplesLoss("euclidean", blur=0.5, backend="multiscale")
            result = loss_fn(x, y)
            
            assert result is not None
            assert not torch.isnan(result).any()
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower() or "grid_cluster" in str(e):
                pytest.skip(f"PyKeOps not available or not configured: {str(e)}")
            else:
                raise


class TestBackendSelection:
    """Test backend selection and fallback behavior"""
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend name raises error"""
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        
        with pytest.raises((ValueError, KeyError, TypeError)):
            loss_fn = SamplesLoss("euclidean", blur=0.5, backend="invalid_backend")
            loss_fn(x, y)
    
    def test_backend_auto_detection(self):
        """Test that backend auto-detection works"""
        x = torch.randn(2, 50, 10)
        y = torch.randn(2, 60, 10)
        
        # Default backend should work
        loss_fn = SamplesLoss("euclidean", blur=0.5)  # No backend specified
        result = loss_fn(x, y)
        
        assert result is not None
        assert not torch.isnan(result).any()
    
    @pytest.mark.parametrize("backend", ["tensorized", "online", "multiscale"])
    def test_all_backends_available_metrics(self, backend):
        """Test that common metrics work across all backends"""
        x = torch.randn(2, 100, 10)
        y = torch.randn(2, 120, 10)
        
        common_metrics = ["euclidean", "energy", "gaussian"]
        
        for metric in common_metrics:
            try:
                loss_fn = SamplesLoss(metric, blur=0.5, backend=backend)
                result = loss_fn(x, y)
                
                assert not torch.isnan(result).any()
            except Exception as e:
                if backend in ["online", "multiscale"] and \
                   ("pykeops" in str(e).lower() or "keops" in str(e).lower()):
                    pytest.skip(f"PyKeOps not available for {backend}")
                else:
                    raise


class TestBackendPerformance:
    """Test backend performance characteristics (not strict timing)"""
    
    def test_tensorized_small_data(self):
        """Test that tensorized backend works well with small data"""
        x = torch.randn(5, 20, 10)
        y = torch.randn(5, 25, 10)
        
        loss_fn = SamplesLoss("euclidean", blur=0.5, backend="tensorized")
        result = loss_fn(x, y)
        
        assert result is not None
        assert not torch.isnan(result).any()
    
    def test_online_medium_data(self):
        """Test that online backend works with medium data"""
        x = torch.randn(3, 200, 10)
        y = torch.randn(3, 250, 10)
        
        try:
            loss_fn = SamplesLoss("euclidean", blur=0.5, backend="online")
            result = loss_fn(x, y)
            
            assert result is not None
            assert not torch.isnan(result).any()
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip("PyKeOps not available")
            else:
                raise
    
    def test_multiscale_large_data(self):
        """Test that multiscale backend works with large data"""
        x = torch.randn(2, 1000, 10)
        y = torch.randn(2, 1200, 10)
        
        try:
            loss_fn = SamplesLoss("euclidean", blur=0.5, backend="multiscale")
            result = loss_fn(x, y)
            
            assert result is not None
            assert not torch.isnan(result).any()
        except Exception as e:
            if "pykeops" in str(e).lower() or "keops" in str(e).lower():
                pytest.skip("PyKeOps not available")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
