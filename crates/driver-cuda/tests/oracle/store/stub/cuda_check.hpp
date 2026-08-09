#pragma once
#include <stdexcept>
#include <cuda_runtime.h>
#define CUDA_CHECK(expr) do { if ((expr) != cudaSuccess) throw std::runtime_error("cuda"); } while (0)
