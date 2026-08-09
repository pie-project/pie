// `qwen3.hpp` includes this only for `WeightView`, which actually lives in
// kernels-cuda. Nothing else in gemm.hpp is reachable from the bind path.
#pragma once
#include "weight_view.hpp"
