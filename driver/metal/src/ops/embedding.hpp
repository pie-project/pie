#pragma once

// Embedding lookup + row gather/scatter helpers.

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// Embedding lookup. `table`:[vocab, hidden] row-major; `ids`:[n_tokens] int.
// Returns [hidden, n_tokens] (feature-major, matching the driver's
// activation layout) so it feeds straight into the first norm/linear.
Tensor embedding(const Tensor& table, const Tensor& ids);

// Gather columns (tokens) from a feature-major `[features, n]` tensor by
// `indices`:[m] -> `[features, m]`. Used for logit-row gathering and
// speculative-decode token selection.
Tensor gather_cols(const Tensor& x, const Tensor& indices);

// Gather rows from a `[n, features]` row-major tensor by `indices`:[m] ->
// `[m, features]`.
Tensor gather_rows(const Tensor& x, const Tensor& indices);

}  // namespace pie_metal_driver::ops
