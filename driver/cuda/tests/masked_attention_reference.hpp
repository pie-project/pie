#pragma once

// CPU golden reference for the PTIR masked-attention parity test
// (overview §5.2/§6.4 fork-sharing enabler).
//
// The production path (`ops::launch_attention_flashinfer_prefill_custom`,
// FlashInfer `MaskMode::kCustom`) presents FROZEN fork pages as FULL — they
// count fully in `kv_len` — and excludes the invalid residual key positions via
// a per-(query,key) attention mask the inferlet feeds (the same BRLE custom-mask
// machinery tree-speculation verification uses). Masked positions get `-inf`
// before softmax, which is exactly prefix truncation (W6/W11): NO attention
// kernel changes; fork validity is pure mask lowering.
//
// This header is the single host-side source of truth for that math. It takes
// the SAME logical inputs the driver attends — dense Q, the logical (gathered)
// K/V the page table resolves to, and the DECODED `qo_len × kv_len` boolean mask
// (bit=1 ⇒ attend, matching `brle.hpp`) — and returns the reference attention
// output. The parity test builds fork geometries (prompt-page aliasing,
// mid-chain frozen pages, designated-child tails, within-page forks), runs the
// driver kernel, and compares within fp32-accumulation tolerance.
//
// Pure C++ (no CUDA headers) so the math is unit-testable host-only.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace pie_attn_ref {

// Row-major dense tensors, described by explicit strides so the caller can feed
// whatever head layout the test uses. All values are fp32 (the caller rounds
// bf16 inputs once and shares them with the device, as the sampler harness does).

// Attention problem for ONE request (lane): `qo_len` query rows over `kv_len`
// key/value positions, `num_qo_heads` query heads sharing `num_kv_heads` KV
// heads (GQA group = num_qo_heads / num_kv_heads), each head `head_dim` wide.
struct Problem {
    int qo_len = 0;
    int kv_len = 0;
    int num_qo_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    float scale = 0.0f;  // usually 1/sqrt(head_dim)

    // q: [qo_len, num_qo_heads, head_dim], row-major.
    const float* q = nullptr;
    // k, v: [kv_len, num_kv_heads, head_dim], row-major (logical/gathered order,
    // the order the page table resolves KV into).
    const float* k = nullptr;
    const float* v = nullptr;
    // mask: [qo_len * kv_len] row-major bytes, bit semantics of `brle.hpp`
    // (mask[q*kv_len + j] != 0 ⇒ query q attends key j). A null mask means
    // "all-attend" (dense), useful for the no-regression cross-check.
    const std::uint8_t* mask = nullptr;
};

// Reference attention output: [qo_len, num_qo_heads, head_dim], row-major.
// Numerically stable softmax (subtract row max). A query row with NO unmasked
// key (all `-inf`) yields all-zeros for that row — the caller must not construct
// such rows for a real fork geometry (every live lane attends >= its own tail).
inline std::vector<float> attention(const Problem& p) {
    const int Q = p.qo_len;
    const int KV = p.kv_len;
    const int HQ = p.num_qo_heads;
    const int HKV = p.num_kv_heads;
    const int D = p.head_dim;
    const int group = (HKV > 0) ? (HQ / HKV) : 1;

    std::vector<float> out(static_cast<std::size_t>(Q) * HQ * D, 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(KV), 0.0f);

    const float kNegInf = -std::numeric_limits<float>::infinity();

    for (int q = 0; q < Q; ++q) {
        for (int h = 0; h < HQ; ++h) {
            const int kvh = (group > 0) ? (h / group) : 0;
            const float* qvec = p.q + ((static_cast<std::size_t>(q) * HQ + h) * D);

            // 1. scores[j] = scale * <q, k_j> + (mask ? 0 : -inf).
            float row_max = kNegInf;
            for (int j = 0; j < KV; ++j) {
                const bool attend =
                    (p.mask == nullptr) ||
                    (p.mask[static_cast<std::size_t>(q) * KV + j] != 0);
                if (!attend) {
                    scores[j] = kNegInf;
                    continue;
                }
                const float* kvec =
                    p.k + ((static_cast<std::size_t>(j) * HKV + kvh) * D);
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qvec[d] * kvec[d];
                const float s = dot * p.scale;
                scores[j] = s;
                if (s > row_max) row_max = s;
            }

            // 2. softmax (stable) over the unmasked keys.
            float denom = 0.0f;
            if (row_max != kNegInf) {
                for (int j = 0; j < KV; ++j) {
                    if (scores[j] == kNegInf) {
                        scores[j] = 0.0f;
                        continue;
                    }
                    const float w = std::exp(scores[j] - row_max);
                    scores[j] = w;
                    denom += w;
                }
            }

            // 3. out[q,h] = sum_j softmax_j * v_j (all-zeros if the row is fully
            //    masked, denom == 0).
            float* ovec = out.data() + ((static_cast<std::size_t>(q) * HQ + h) * D);
            if (denom > 0.0f) {
                const float inv = 1.0f / denom;
                for (int j = 0; j < KV; ++j) {
                    const float w = scores[j] * inv;
                    if (w == 0.0f) continue;
                    const float* vvec =
                        p.v + ((static_cast<std::size_t>(j) * HKV + kvh) * D);
                    for (int d = 0; d < D; ++d) ovec[d] += w * vvec[d];
                }
            }
        }
    }
    return out;
}

// Max absolute elementwise difference between two equal-length buffers — the
// parity metric (compare against an fp32-accumulation tolerance).
inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

}  // namespace pie_attn_ref
