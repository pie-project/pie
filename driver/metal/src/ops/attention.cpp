#include "ops/attention.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <mlx/mlx.h>

#include "kv_cache.hpp"

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

// Manual masked attention for the cases MLX's fused causal SDPA can't express:
// attention-logit softcapping (Gemma2/3) and sliding-window attention (Gemma
// local layers). Operates on contiguous single-sequence 3D operands
//   q:[nq, H, d], k:[nkv, Hkv, d], v:[nkv, Hkv, d] -> [nq, H, d]
// in float32 for reference-grade numerics, then casts back to q's dtype. GQA is
// handled by repeating the kv heads to H. Causal alignment matches MLX: the nq
// queries align to the END of the nkv keys (absolute pos = (nkv - nq) + j).
Tensor manual_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                        float scale, float softcap, int sliding_window) {
    const int nq = q.shape(0);
    const int H = q.shape(1);
    const int d = q.shape(2);
    const int nkv = k.shape(0);
    const int Hkv = k.shape(1);
    const int n_rep = (Hkv > 0) ? H / Hkv : 1;

    // [L, heads, d] -> [heads, L, d], GQA-expanded, float32.
    Tensor kk = (n_rep > 1) ? mx::repeat(k, n_rep, 1) : k;
    Tensor vv = (n_rep > 1) ? mx::repeat(v, n_rep, 1) : v;
    Tensor qh = mx::astype(mx::transpose(q, {1, 0, 2}), mx::float32);
    Tensor kh = mx::astype(mx::transpose(kk, {1, 0, 2}), mx::float32);
    Tensor vh = mx::astype(mx::transpose(vv, {1, 0, 2}), mx::float32);

    // scores: [H, nq, nkv] = (qh @ kh^T) * scale.
    Tensor scores =
        mx::multiply(mx::matmul(qh, mx::swapaxes(kh, -1, -2)), mx::array(scale));
    if (softcap > 0.0f) {
        scores = mx::multiply(mx::tanh(scores * (1.0f / softcap)),
                              mx::array(softcap));
    }

    // Additive mask [nq, nkv]: causal (k_idx <= q_abs) and, for SWA, within the
    // window (k_idx > q_abs - sliding_window).
    const int offset = nkv - nq;
    Tensor q_idx = mx::expand_dims(mx::arange(offset, offset + nq, mx::int32), 1);
    Tensor k_idx = mx::expand_dims(mx::arange(0, nkv, mx::int32), 0);
    Tensor allowed = mx::less_equal(k_idx, q_idx);  // causal
    if (sliding_window > 0) {
        Tensor lo = mx::subtract(q_idx, mx::array(sliding_window));
        allowed = mx::logical_and(allowed, mx::greater(k_idx, lo));
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor mask_add = mx::where(allowed, mx::array(0.0f), mx::array(neg_inf));
    scores = mx::add(scores, mx::expand_dims(mask_add, 0));  // broadcast over H

    Tensor probs = mx::softmax(scores, -1, /*precise=*/true);
    Tensor out = mx::matmul(probs, vh);  // [H, nq, d]
    return mx::astype(mx::transpose(out, {1, 0, 2}), q.dtype());
}

}  // namespace

Tensor sdpa(const Tensor& q, const Tensor& k, const Tensor& v,
            const AttnParams& params, const std::optional<Tensor>& mask) {
    const float scale = resolve_scale(params);
    // Softcap / sliding-window aren't expressible via MLX's fused causal SDPA;
    // fall back to the manual masked path (only meaningful without an explicit
    // mask — Gemma drives these through the causal path).
    if (!mask.has_value() &&
        (params.softcap > 0.0f || params.sliding_window > 0)) {
        return manual_attention(q, k, v, scale, params.softcap,
                                params.sliding_window);
    }
    const std::string mode = mask.has_value() ? "" : "causal";
    return sdpa_contiguous(q, k, v, scale, mode, mask);
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
    // Production path: per-request page-gather + MLX fused SDPA. MLX's
    // fast::scaled_dot_product_attention is a heavily-tuned flash-attention
    // kernel; benchmarking showed a naive one-thread-per-(token,head) custom
    // Metal kernel is 5-15x SLOWER than this, so we keep MLX sdpa. A custom
    // win requires a proper split-KV flash-decoding kernel (cross-threadgroup
    // reduction) -- tracked as future perf work. Pure MLX => also runs CPU-only.
    // k_cache/v_cache assumed [n_pages, page_size, n_kv_heads, head_dim].
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
        // n_kv keys, so each new token attends its full history. Softcap /
        // sliding-window (Gemma) need the manual masked path.
        if (params.softcap > 0.0f || params.sliding_window > 0) {
            outputs.push_back(manual_attention(q_r, k_r, v_r, scale,
                                               params.softcap,
                                               params.sliding_window));
        } else {
            outputs.push_back(
                sdpa_contiguous(q_r, k_r, v_r, scale, "causal", std::nullopt));
        }
    }

    if (outputs.empty()) {
        return mx::zeros({0, q.shape(1), q.shape(2)}, q.dtype());
    }
    return mx::concatenate(outputs, 0);
}

Tensor paged_attention(const Tensor& q, const PagedKV& kv,
                       const AttnParams& params) {
    return paged_attention(q, kv.k_pages, kv.v_pages, kv.page_table,
                           kv.qo_indptr, kv.kv_page_indptr, kv.last_page_lens,
                           kv.page_size, params);
}

}  // namespace pie_metal_driver::ops
