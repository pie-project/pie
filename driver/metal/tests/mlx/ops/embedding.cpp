#include "ops/embedding.hpp"

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

Tensor embedding(const Tensor& table, const Tensor& ids) {
    // table:[vocab, hidden], ids:[n] -> take rows along axis 0 -> [n, hidden].
    return mx::take(table, ids, /*axis=*/0);
}

Tensor gather_rows(const Tensor& x, const Tensor& indices) {
    // x:[n, features], indices:[m] -> [m, features].
    return mx::take(x, indices, /*axis=*/0);
}

}  // namespace pie_metal_driver::ops
