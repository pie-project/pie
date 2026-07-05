#pragma once

// Embedding lookup + row gather/scatter helpers.

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// Embedding lookup. `table`:[vocab, hidden] row-major; `ids`:[n_tokens] int.
// Returns [n_tokens, hidden] (token-major, matching the driver's activation
// layout) so it feeds straight into the first norm/linear.
Tensor embedding(const Tensor& table, const Tensor& ids);

// Gather rows (tokens) from a token-major `[n, features]` tensor by
// `indices`:[m] -> `[m, features]`. Used for logit-row gathering and
// speculative-decode token selection.
Tensor gather_rows(const Tensor& x, const Tensor& indices);

}  // namespace pie_metal_driver::ops
