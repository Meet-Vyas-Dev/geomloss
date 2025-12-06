"""
Test Runner for GeomLoss Test Suite
====================================

Runs all tests and generates a comprehensive report.
"""

import subprocess
import sys


def run_tests():
    """Run all test suites"""
    
    print("=" * 80)
    print("GeomLoss Extended Distance Metrics - Test Suite")
    print("=" * 80)
    
    test_suites = [
        ("Basic Distance Metrics", "tests/test_distance_metrics_comprehensive.py"),
        ("Backend Compatibility", "tests/test_backends.py"),
        ("Kernelization and Blur", "tests/test_kernelization_and_blur.py"),
    ]
    
    results = {}
    
    for suite_name, test_file in test_suites:
        print(f"\n{'=' * 80}")
        print(f"Running: {suite_name}")
        print(f"{'=' * 80}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-x"],
                capture_output=False,
                text=True,
                cwd="."
            )
            
            results[suite_name] = result.returncode == 0
            
        except Exception as e:
            print(f"Error running {suite_name}: {e}")
            results[suite_name] = False
    
    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}\n")
    
    for suite_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {suite_name}")
    
    all_passed = all(results.values())
    
    print(f"\n{'=' * 80}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("The library is ready for pull request.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the output above.")
    print(f"{'=' * 80}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_tests())
