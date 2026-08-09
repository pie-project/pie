#pragma once
// Stub csrc/src/cuda_check.hpp: no device, nothing to check.
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(x) do { (void)(x); } while (0)
#define CUDA_CHECK_LAST() do { } while (0)
