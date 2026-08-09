#pragma once
// Stub for cuda_check.hpp. Every call recorded by cuda_runtime.h returns
// cudaSuccess, so the macro only needs to evaluate its argument exactly once
// -- the argument is the recorded call, and dropping it would erase the
// transcript.
#include "cuda_runtime.h"
#define CUDA_CHECK(x) do { (void)(x); } while (0)
