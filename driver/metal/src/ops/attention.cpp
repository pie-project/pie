#include "ops/attention.hpp"

#include <cmath>
#include <vector>

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

namespace {

// Read a small index tensor to a host int32 vector (used for the per-request
// CSR offsets that drive the paged gather). These arrays are tiny
// (n_req + 1), so the eval + copy cost is negligible.
std::vector<int> to_host_i32(const Tensor& t) {
    Tensor i = mx::astype(t, mx::int32);
    i.eval();
    const int* p = i.data<int>();
    return std::vector<int>(p, p + i.size());
}

float resolve_scale(const AttnParams& params) {
    if (params.scale > 0.0f) return params.scale;
    const int hd = params.head_dim > 0 ? params.head_dim : 1;
    return 1.0f / std::sqrt(static_cast<float>(hd));
}

// SDPA over contiguous, single-sequence 3D operands:
//   q:[nq, H, d], k:[nkv, Hkv, d], v:[nkv, Hkv, d] -> [nq, H, d].
// Reshapes to MLX's [B=1, heads, L, d] convention and back. GQA is handled
// natively by MLX when Hkv < H.
Tensor sdpa_contiguous(const Tensor& q, const Tensor& k, const Tensor& v,
                       float scale, const std::string& mask_mode,
                       const std::optional<Tensor>& mask) {
    // [L, heads, d] -> [1, heads, L, d]
    Tensor q4 = mx::expand_dims(mx::transpose(q, {1, 0, 2}), 0);
    Tensor k4 = mx::expand_dims(mx::transpose(k, {1, 0, 2}), 0);
    Tensor v4 = mx::expand_dims(mx::transpose(v, {1, 0, 2}), 0);

    Tensor o4 = mx::fast::scaled_dot_product_attention(
        q4, k4, v4, scale, mask_mode, mask);

    // [1, heads, L, d] -> [L, heads, d]
    return mx::transpose(mx::squeeze(o4, 0), {1, 0, 2});
}

}  // namespace

Tensor sdpa(const Tensor& q, const Tensor& k, const Tensor& v,
            const AttnParams& params, const std::optional<Tensor>& mask) {
    const std::string mode = mask.has_value() ? "" : "causal";
    return sdpa_contiguous(q, k, v, resolve_scale(params), mode, mask);
}

Tensor paged_attention(const Tensor& q,
                       const Tensor& k_cache,
                       const Tensor& v_cache,
                       const Tensor& page_table,
                       const Tensor& qo_indptr,
                       const Tensor& kv_page_indptr,
                       const Tensor& last_page_lens,
                       int page_size,
                       const AttnParams& params) {
    // Reference implementation: per-request page-gather + dense SDPA. Correct
    // and end-to-end usable; the optimized fused paged-attention Metal kernel
    // (src/kernels/paged_attention.metal) replaces this hot path once delta's
    // PagedKV layout is published. k_cache/v_cache assumed
    // [n_pages, page_size, n_kv_heads, head_dim].
    const std::vector<int> qo   = to_host_i32(qo_indptr);
    const std::vector<int> kvp  = to_host_i32(kv_page_indptr);
    const std::vector<int> lpl  = to_host_i32(last_page_lens);

    const float scale = resolve_scale(params);
    const int n_req = static_cast<int>(qo.size()) - 1;
    const int kv_heads = k_cache.shape(2);
    const int head_dim = k_cache.shape(3);

    std::vector<Tensor> outputs;
    outputs.reserve(n_req);

    for (int r = 0; r < n_req; ++r) {
        const int q_start = qo[r];
        const int q_end   = qo[r + 1];
        const int nq      = q_end - q_start;
        if (nq <= 0) continue;

        const int pg_start = kvp[r];
        const int pg_end   = kvp[r + 1];
        const int n_pages  = pg_end - pg_start;
        if (n_pages <= 0) continue;
        const int n_kv = (n_pages - 1) * page_size + lpl[r];

        // Gather this request's physical pages (sliced on-device from the
        // page table), then flatten + clip to n_kv.
        Tensor pages = mx::astype(
            mx::slice(page_table, {pg_start}, {pg_end}), mx::int32);
        Tensor k_pg = mx::take(k_cache, pages, 0);  // [n_pages, page_size, Hkv, d]
        Tensor v_pg = mx::take(v_cache, pages, 0);
        Tensor k_flat = mx::reshape(k_pg, {n_pages * page_size, kv_heads, head_dim});
        Tensor v_flat = mx::reshape(v_pg, {n_pages * page_size, kv_heads, head_dim});
        Tensor k_r = mx::slice(k_flat, {0, 0, 0}, {n_kv, kv_heads, head_dim});
        Tensor v_r = mx::slice(v_flat, {0, 0, 0}, {n_kv, kv_heads, head_dim});

        Tensor q_r = mx::slice(q, {q_start, 0, 0},
                               {q_end, q.shape(1), q.shape(2)});

        // Causal alignment: MLX aligns the nq queries to the end of the
        // n_kv keys, so each new token attends its full history.
        outputs.push_back(
            sdpa_contiguous(q_r, k_r, v_r, scale, "causal", std::nullopt));
    }

    if (outputs.empty()) {
        return mx::zeros({0, q.shape(1), q.shape(2)}, q.dtype());
    }
    return mx::concatenate(outputs, 0);
}

}  // namespace pie_metal_driver::ops
