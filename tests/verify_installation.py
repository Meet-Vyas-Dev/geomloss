"""
Quick Verification Script
=========================

Verifies that the GeomLoss extended implementation is working correctly.
Run this before submitting a pull request.
"""

import sys
import torch

print("=" * 80)
print("GeomLoss Extended Distance Metrics - Quick Verification")
print("=" * 80)

# Test 1: Import
print("\n[1/5] Testing imports...")
try:
    from geomloss import SamplesLoss, DISTANCE_METRICS
    print(f"✅ Imports successful")
    print(f"   Total metrics available: {len(DISTANCE_METRICS)}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create sample data
print("\n[2/5] Creating sample data...")
try:
    x = torch.randn(2, 50, 10)
    y = torch.randn(2, 60, 10)
    print(f"✅ Sample data created: x{x.shape}, y{y.shape}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 3: Test common metrics
print("\n[3/5] Testing common distance metrics...")
test_metrics = ["euclidean", "manhattan", "cosine", "hellinger", "kl", "js"]
passed = 0
failed = 0

for metric in test_metrics:
    try:
        if metric in ["hellinger", "kl", "js"]:
            # Use positive data
            x_pos = torch.rand(2, 50, 10) + 0.01
            y_pos = torch.rand(2, 60, 10) + 0.01
            x_pos = x_pos / x_pos.sum(dim=-1, keepdim=True)
            y_pos = y_pos / y_pos.sum(dim=-1, keepdim=True)
            loss_fn = SamplesLoss(metric, blur=0.5)
            result = loss_fn(x_pos, y_pos)
        else:
            loss_fn = SamplesLoss(metric, blur=0.5)
            result = loss_fn(x, y)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        print(f"   ✅ {metric:15s}: {result.mean().item():.6f}")
        passed += 1
    except Exception as e:
        print(f"   ❌ {metric:15s}: {e}")
        failed += 1

if failed > 0:
    print(f"\n❌ {failed}/{len(test_metrics)} metrics failed")
    sys.exit(1)
else:
    print(f"\n✅ All {passed} metrics passed")

# Test 4: Test backends
print("\n[4/5] Testing backends...")
backends_passed = 0
backends_failed = 0

for backend in ["tensorized", "online", "multiscale"]:
    try:
        loss_fn = SamplesLoss("euclidean", blur=0.5, backend=backend)
        result = loss_fn(x, y)
        assert not torch.isnan(result).any()
        print(f"   ✅ {backend:12s}: working")
        backends_passed += 1
    except Exception as e:
        if "pykeops" in str(e).lower() or "keops" in str(e).lower() or "grid_cluster" in str(e):
            print(f"   ⏭️  {backend:12s}: skipped (PyKeOps not available)")
        else:
            print(f"   ❌ {backend:12s}: {e}")
            backends_failed += 1

if backends_failed > 0:
    print(f"\n❌ {backends_failed} backends failed")
    sys.exit(1)

# Test 5: Test gradient flow
print("\n[5/5] Testing gradient flow...")
try:
    x_grad = torch.randn(2, 30, 10, requires_grad=True)
    y_grad = torch.randn(2, 40, 10, requires_grad=True)
    
    loss_fn = SamplesLoss("euclidean", blur=0.5)
    result = loss_fn(x_grad, y_grad)
    loss = result.sum()
    loss.backward()
    
    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    assert not torch.isinf(x_grad.grad).any()
    
    print("   ✅ Gradients flow correctly")
    print(f"   Gradient norm: {x_grad.grad.norm().item():.6f}")
except Exception as e:
    print(f"   ❌ Gradient flow failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ ALL VERIFICATIONS PASSED")
print("=" * 80)
print("\nThe implementation is working correctly!")
print("You can now:")
print("  1. Run full test suite: python run_tests.py")
print("  2. Review PULL_REQUEST_README.md for PR description")
print("  3. Check PR_FILE_GUIDE.md for what to include in PR")
print("=" * 80)
