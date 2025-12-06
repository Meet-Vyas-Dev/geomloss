Following Files need to be updated to fix the issues:

File 1- `test_distance_metrics.py` - 
To fix - 1 Distance Metric test is not working properly 

Files 2 & 3 - `test_pykeops_backends.py`, `test_backend_summary.py` -

To fix - All online tests are failing intermittently, need to investigate and fix the root cause

File 4 - `Updated Readme.md` 

To fix  - 
1. The readme file has incorrect information about the number of distance metrics implemented. 
2. The funtions and usage of the `test_distance_metrics.py` file are not correct as the original file has only 200 lines of code but the readme mentions 500+ lines of code. And the readme also mentions that the file includes performance benchmarks which is not true. 
- test_with_samples_loss()   # Full pipeline testing
- test_gradient_flow()       # Backward pass validation
- test_batch_processing()    # Multi-batch handling
