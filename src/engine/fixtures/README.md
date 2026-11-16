# Test fixtures

Real CUDA sources used by `engine.test.ts`. They are read from disk by the tests and are not part of the app bundle.

| File | Source | Licence |
| --- | --- | --- |
| `hipify-square.cu` | ROCm/HIPIFY `tests/unit_tests/samples/square.cu` | MIT, Copyright (c) Advanced Micro Devices, Inc. |
| `hipify-headers_test_09.cu` | ROCm/HIPIFY `tests/unit_tests/headers/headers_test_09.cu` | MIT, Copyright (c) Advanced Micro Devices, Inc. |
| `hipify-runtime_functions.excerpt.cu` | Excerpt of ROCm/HIPIFY `tests/unit_tests/synthetic/runtime_functions.cu` | MIT, Copyright (c) Advanced Micro Devices, Inc. |
| `nvidia-vectorAdd.cu` | NVIDIA/cuda-samples v12.5 `Samples/0_Introduction/vectorAdd/vectorAdd.cu` | BSD 3-Clause, notice kept in the file |

The HIPIFY files carry `// CHECK:` lines with the output hipify-clang is expected to produce; the tests compare the translator against them.
