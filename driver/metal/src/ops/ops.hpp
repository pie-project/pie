#pragma once

// Umbrella include for the Metal (MLX) driver op surface. Model graphs
// (`pie_metal_driver::model`) include this single header to reach every op.

#include "ops/activation.hpp"
#include "ops/attention.hpp"
#include "ops/elementwise.hpp"
#include "ops/embedding.hpp"
#include "ops/gated_delta.hpp"
#include "ops/gemm.hpp"
#include "ops/moe.hpp"
#include "ops/norm.hpp"
#include "ops/rope.hpp"
#include "ops/tensor.hpp"
