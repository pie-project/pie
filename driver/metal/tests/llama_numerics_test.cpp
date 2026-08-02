// Does the llama family compute the right numbers?
//
// Everything else this family has is structural. `llama_decode_step_test` says
// the DAG resolves, the widths are legal and the pool is big enough;
// `llama_bind_test` says every slot a kernel declares is bound. Both would pass
// a model that rotates with the wrong rope base, scales attention by 1.0, norms
// qwen3's heads across the whole projection instead of within one, or hands
// `down` the first expert's activation four times. Those are the four places
// this family was reasoned about rather than checked, and every one of them
// produces a model that RUNS.
//
// So this runs the real decoder and compares it against an independent fp32
// reference written from the .metal files -- not against the driver's own C++,
// which would agree with itself.
//
// Three choices make it a localiser rather than a pass/fail:
//
//   * It colours with `no_recycle`, so every activation value gets a buffer of
//     its own and the output of EVERY dispatch can be read back. A failure
//     names the first dispatch that diverges, which is the kernel at fault --
//     compared at the logits alone, a wrong rope and a wrong SwiGLU look the
//     same.
//   * It steps four tokens. At position 0 the rotation is the identity, so a
//     rope with the wrong base passes; the KV cache is also one entry deep, so
//     attention is a no-op that returns v. Neither is true at position 3.
//   * There is no checkpoint. The weights are synthesised in the MLX affine-U4
//     layout the kernels read, so the reference knows the exact dequantized
//     value of every element and the comparison is against arithmetic rather
//     than against a second implementation's rounding.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "batch/decode_abi.hpp"
#include "kernels/decode_psos.hpp"
#include "loader/heap_bind.hpp"
#include "model/llama/bind.hpp"
#include "model/llama/decode_consts.hpp"
#include "model/llama/decode_step.hpp"
#include "model/llama/encode.hpp"
#include "device_tuning.hpp"
#include "model/llama/geometry.hpp"
#include "model/llama/kernels.hpp"
#include "model/llama/scratch.hpp"
#include "mtl4_context.hpp"

using pie::metal::DecodeGeometry;
using pie::metal::DecodeStepPsos;
using pie::metal::IoSlot;
using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;
using pie::metal::StepEncoder;
using pie::metal::WeightBind;
using pie::metal::kIoSlotCount;
using pie::metal::load_decode_psos;
using pie::metal::weight_binds;
namespace model = pie::metal::model;
namespace llama = pie::metal::llama;
using llama::BoundLlama;
using llama::Dispatch;
using llama::Kind;
using llama::LlamaGeometry;
using llama::LlamaPsos;
using llama::ScratchPlan;
using llama::Use;
using llama::bind_llama_consts;
using llama::bind_llama_dag;
using llama::build_llama_dag;
using llama::build_llama_psos;
using llama::build_llama_scratch;
using llama::color_llama_scratch;
using llama::encode_llama_step;
using pie::metal::MultiBatchPsos;
using pie::metal::load_multibatch_psos;
using llama::llama_kv_bytes_per_layer;
using llama::llama_pool_elems;
using llama::shared_kind;

namespace {

int g_pass = 0, g_fail = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    ok ? ++g_pass : ++g_fail;
}

// ── bfloat16, the activation type every kernel in this family uses ──────────

float from_bf16(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// Round-to-nearest-even, which is what the GPU's `static_cast<bfloat>` does.
/// Truncating instead biases every activation low, and over a two-layer
/// residual stream that drift is larger than the tolerance below.
std::uint16_t to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return std::uint16_t((bits + 0x7fffu + lsb) >> 16);
}

float rbf(float f) { return from_bf16(to_bf16(f)); }

/// A deterministic value in [-1, 1). Not `rand`: the reference and the staged
/// weights must agree bit for bit, and across runs, or a failure is unreadable.
float hashf(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return float(double(x >> 40) / double(1 << 23)) - 1.0f;
}

// ── MLX affine-U4, synthesised ──────────────────────────────────────────────

/// One quantized tensor: the bytes the kernel reads, and what they mean.
///
/// `deq` is not a second encoding of the same thing -- it is READ BACK from the
/// nibbles and the bf16 scale/bias that were actually stored, so the reference
/// cannot be right about a weight the GPU sees differently. Quantizing a float
/// matrix and comparing against the pre-quantization floats would fail by the
/// quantization error, which is not what is being tested.
struct QuantTensor {
    int experts = 1, n = 0, k = 0;
    std::vector<std::uint32_t> packed;   // [E, N, K/8]
    std::vector<std::uint16_t> scales;   // [E, N, K/64]
    std::vector<std::uint16_t> biases;   // [E, N, K/64]
    std::vector<float> deq;              // [E, N, K]

    float at(int e, int row, int col) const {
        return deq[((std::size_t(e) * std::size_t(n) + std::size_t(row)) * std::size_t(k)) +
                   std::size_t(col)];
    }
};

constexpr int kGroup = 64;

QuantTensor make_quant(int experts, int n, int k, std::uint64_t seed) {
    QuantTensor q;
    q.experts = experts;
    q.n = n;
    q.k = k;
    const std::size_t groups = std::size_t(k / kGroup);
    q.packed.assign(std::size_t(experts) * std::size_t(n) * std::size_t(k / 8), 0);
    q.scales.assign(std::size_t(experts) * std::size_t(n) * groups, 0);
    q.biases.assign(std::size_t(experts) * std::size_t(n) * groups, 0);
    q.deq.assign(std::size_t(experts) * std::size_t(n) * std::size_t(k), 0.0f);

    for (int e = 0; e < experts; ++e) {
        for (int row = 0; row < n; ++row) {
            const std::uint64_t rk = seed * 1000003ULL + std::uint64_t(e) * 7919ULL +
                                     std::uint64_t(row);
            for (std::size_t gi = 0; gi < groups; ++gi) {
                // Centred on zero: `bias = -7.5 * scale` puts nibble 0..15 on
                // [-7.5s, 7.5s], so a projection of a zero-mean activation is
                // zero-mean too. A one-sided weight matrix makes every logit a
                // large common-mode number, and a relative tolerance against
                // that hides real error.
                const float s = 0.02f * (1.0f + 0.5f * hashf(rk * 131ULL + gi));
                const std::uint16_t sb = to_bf16(s);
                const std::uint16_t bb = to_bf16(-7.5f * s);
                const std::size_t gidx =
                    (std::size_t(e) * std::size_t(n) + std::size_t(row)) * groups + gi;
                q.scales[gidx] = sb;
                q.biases[gidx] = bb;
                const float sf = from_bf16(sb), bf = from_bf16(bb);
                for (int j = 0; j < kGroup; ++j) {
                    const int col = int(gi) * kGroup + j;
                    const std::uint64_t h = rk * 1000003ULL + std::uint64_t(col);
                    const std::uint32_t nib = std::uint32_t((h >> 7) & 0xfu);
                    const std::size_t pidx =
                        ((std::size_t(e) * std::size_t(n) + std::size_t(row)) *
                         std::size_t(k / 8)) +
                        std::size_t(col / 8);
                    // Nibble j of a uint32 is element `base + j`, low first --
                    // the packing `embed_gather_4bit` reads with
                    // `(pack >> ((k % 8) * 4)) & 0xf` and `qdot` reads as four
                    // masked fields of a uint16.
                    q.packed[pidx] |= nib << ((col % 8) * 4);
                    q.deq[((std::size_t(e) * std::size_t(n) + std::size_t(row)) *
                           std::size_t(k)) +
                          std::size_t(col)] = sf * float(nib) + bf;
                }
            }
        }
    }
    return q;
}

/// A norm's learned gain, stored bf16 and read back so the reference uses the
/// stored value rather than the one it meant to store.
std::vector<float> make_norm(int width, std::uint64_t seed, std::vector<std::uint16_t>& raw) {
    raw.resize(std::size_t(width));
    std::vector<float> out(std::size_t(width), 0.0f);
    for (int i = 0; i < width; ++i) {
        raw[std::size_t(i)] = to_bf16(1.0f + 0.25f * hashf(seed * 7919ULL + std::uint64_t(i)));
        out[std::size_t(i)] = from_bf16(raw[std::size_t(i)]);
    }
    return out;
}

// ── the reference decoder ───────────────────────────────────────────────────

using Vec = std::vector<float>;

/// Every value the reference produces, keyed by the DAG position that produced
/// it. Compared one by one against the pool.
using Trace = std::unordered_map<int, Vec>;

void round_bf16(Vec& v) {
    for (float& f : v) f = rbf(f);
}

Vec rms_norm(const Vec& x, const Vec& w, int axis, float eps) {
    const int rows = int(x.size()) / axis;
    Vec out(x.size());
    for (int r = 0; r < rows; ++r) {
        float acc = 0;
        for (int i = 0; i < axis; ++i) {
            const float xi = x[std::size_t(r * axis + i)];
            acc += xi * xi;
        }
        const float inv = 1.0f / std::sqrt(acc / float(axis) + eps);
        for (int i = 0; i < axis; ++i) {
            // The kernel rounds `x * inv_mean` to bf16 BEFORE multiplying by
            // the gain -- `wv * static_cast<T>(x[i] * local_inv_mean[0])`.
            out[std::size_t(r * axis + i)] =
                w[std::size_t(i)] * rbf(x[std::size_t(r * axis + i)] * inv);
        }
    }
    return out;
}

Vec matvec(const QuantTensor& q, const Vec& x, int expert = 0) {
    Vec y(std::size_t(q.n), 0.0f);
    for (int r = 0; r < q.n; ++r) {
        float acc = 0;
        for (int c = 0; c < q.k; ++c) acc += q.at(expert, r, c) * x[std::size_t(c)];
        y[std::size_t(r)] = acc;
    }
    return y;
}

void rope_inplace(Vec& x, int heads, int head_dim, int position, float theta) {
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < half; ++i) {
            // `exp2(-d * log2(theta))` -- the kernel's `base` is log2(theta),
            // and `d` divides by HALF the head because llama rotates the whole
            // head (`rotary_dims() == head_dim`).
            const float d = float(i) / float(half);
            const float inv = std::exp2(-d * std::log2(theta));
            const float t = float(position) * inv;
            const int i1 = h * head_dim + i, i2 = i1 + half;
            const float x1 = x[std::size_t(i1)], x2 = x[std::size_t(i2)];
            x[std::size_t(i1)] = rbf(x1 * std::cos(t) - x2 * std::sin(t));
            x[std::size_t(i2)] = rbf(x1 * std::sin(t) + x2 * std::cos(t));
        }
    }
}

float silu_mul_one(float g, float u) {
    // The kernel rounds three times -- sigmoid, then the product with gate,
    // then the product with up -- because MLX evaluates it as three ops.
    const float y = 1.0f / (1.0f + std::exp(-std::fabs(g)));
    const float sg = rbf(g < 0.0f ? 1.0f - y : y);
    const float sil = rbf(g * sg);
    return rbf(sil * u);
}

/// One layer's KV, [n_kv_heads][max_ctx][head_dim], the contiguous layout
/// `k_head_stride = max_ctx * head_dim` describes.
struct RefKv {
    Vec k, v;
};

struct Reference {
    const LlamaGeometry& g;
    const QuantTensor& embed;
    const QuantTensor& head;
    const std::vector<QuantTensor>& wq;
    const std::vector<QuantTensor>& wk;
    const std::vector<QuantTensor>& wv;
    const std::vector<QuantTensor>& wo;
    const std::vector<QuantTensor>& wgate;
    const std::vector<QuantTensor>& wup;
    const std::vector<QuantTensor>& wdown;
    const std::vector<QuantTensor>& wrouter;
    const std::vector<Vec>& n_attn;
    const std::vector<Vec>& n_ffn;
    const std::vector<Vec>& n_q;
    const std::vector<Vec>& n_k;
    const Vec& n_final;
    std::vector<RefKv>& kv;
    std::vector<int> last_ids;
    Vec last_w;
    /// How far the LAST selected expert's logit sat above the first rejected
    /// one. When this is inside the router matvec's own error the selection is
    /// genuinely ambiguous: fp32 and bf16 sort the two differently and the row
    /// runs different experts, which is a property of the weights rather than
    /// a fault in the driver.
    float last_margin = 0.0f;
    /// Every margin the current step decided on, against the DAG position of
    /// the selection that decided it. Kept in full rather than reduced,
    /// because the reduction threw away the one thing the reader needs: a tie
    /// makes the dispatches AFTER it incomparable and leaves the ones before
    /// it as sound as any other row's. Minimising over the stack lost that
    /// boundary, and worse, produced a number from one layer that was then
    /// weighed against a threshold measured at another.
    std::vector<std::pair<int, float>> step_margins;
    /// The most recent layer's router logits, as the reference computed them.
    /// Compared against the device's to MEASURE how far apart the two routers
    /// are, instead of assuming a number for it.
    Vec last_router_logits;

    /// Run one token, recording every dispatch's output by DAG position.
    Trace step(const std::vector<Dispatch>& dag, int token, int position) {
        Trace t;
        step_margins.clear();
        const int hd = g.head_dim, nq = g.n_q_heads, nkv = g.n_kv_heads;
        Vec resid, normed, q, k, v, attn, gate, up, act, block, logits;
        Vec expert_w;
        std::vector<int> expert_ids;

        for (std::size_t i = 0; i < dag.size(); ++i) {
            const Dispatch& d = dag[i];
            const int L = d.layer;
            const int at = int(i);
            switch (d.kind) {
                case Kind::EmbedGather:
                    resid.assign(std::size_t(g.hidden), 0.0f);
                    for (int c = 0; c < g.hidden; ++c) {
                        resid[std::size_t(c)] = rbf(embed.at(0, token, c));
                    }
                    t[at] = resid;
                    break;
                case Kind::AttnNorm:
                    normed = rms_norm(resid, n_attn[std::size_t(L)], g.hidden, g.eps);
                    round_bf16(normed);
                    t[at] = normed;
                    break;
                case Kind::QmvQ:
                    q = matvec(wq[std::size_t(L)], normed);
                    round_bf16(q);
                    t[at] = q;
                    break;
                case Kind::QmvK:
                    k = matvec(wk[std::size_t(L)], normed);
                    round_bf16(k);
                    t[at] = k;
                    break;
                case Kind::QmvV:
                    v = matvec(wv[std::size_t(L)], normed);
                    round_bf16(v);
                    t[at] = v;
                    break;
                case Kind::QNorm:
                    // Per HEAD, over head_dim. Normalising across the whole
                    // projection instead mixes the heads.
                    q = rms_norm(q, n_q[std::size_t(L)], hd, g.eps);
                    round_bf16(q);
                    t[at] = q;
                    break;
                case Kind::KNorm:
                    k = rms_norm(k, n_k[std::size_t(L)], hd, g.eps);
                    round_bf16(k);
                    t[at] = k;
                    break;
                case Kind::RopeQ:
                    rope_inplace(q, nq, hd, position, g.rope_theta);
                    t[at] = q;
                    break;
                case Kind::RopeK:
                    rope_inplace(k, nkv, hd, position, g.rope_theta);
                    t[at] = k;
                    break;
                case Kind::KvAppend: {
                    RefKv& c = kv[std::size_t(L)];
                    for (int h = 0; h < nkv; ++h) {
                        for (int e = 0; e < hd; ++e) {
                            const std::size_t dst =
                                std::size_t(h) * std::size_t(g.kv_max_ctx) * std::size_t(hd) +
                                std::size_t(position) * std::size_t(hd) + std::size_t(e);
                            c.k[dst] = k[std::size_t(h * hd + e)];
                            c.v[dst] = v[std::size_t(h * hd + e)];
                        }
                    }
                    break;  // writes the cache, not the pool
                }
                case Kind::Sdpa: {
                    const RefKv& c = kv[std::size_t(L)];
                    const int gqa = nq / nkv;
                    const float scale = 1.0f / std::sqrt(float(hd));
                    const int n = position + 1;
                    attn.assign(std::size_t(nq) * std::size_t(hd), 0.0f);
                    for (int h = 0; h < nq; ++h) {
                        const int kh = h / gqa;
                        Vec s(std::size_t(n), 0.0f);
                        float mx = -3.0e38f;
                        for (int j = 0; j < n; ++j) {
                            float acc = 0;
                            for (int e = 0; e < hd; ++e) {
                                const std::size_t src =
                                    std::size_t(kh) * std::size_t(g.kv_max_ctx) *
                                        std::size_t(hd) +
                                    std::size_t(j) * std::size_t(hd) + std::size_t(e);
                                acc += (scale * q[std::size_t(h * hd + e)]) * c.k[src];
                            }
                            s[std::size_t(j)] = acc;
                            mx = std::max(mx, acc);
                        }
                        float sum = 0;
                        for (int j = 0; j < n; ++j) {
                            s[std::size_t(j)] = std::exp(s[std::size_t(j)] - mx);
                            sum += s[std::size_t(j)];
                        }
                        for (int e = 0; e < hd; ++e) {
                            float acc = 0;
                            for (int j = 0; j < n; ++j) {
                                const std::size_t src =
                                    std::size_t(kh) * std::size_t(g.kv_max_ctx) *
                                        std::size_t(hd) +
                                    std::size_t(j) * std::size_t(hd) + std::size_t(e);
                                acc += s[std::size_t(j)] * c.v[src];
                            }
                            attn[std::size_t(h * hd + e)] = acc / sum;
                        }
                    }
                    round_bf16(attn);
                    t[at] = attn;
                    break;
                }
                case Kind::QmvO:
                    block = matvec(wo[std::size_t(L)], attn);
                    round_bf16(block);
                    t[at] = block;
                    break;
                case Kind::AttnResidual:
                case Kind::FfnResidual:
                    for (int c = 0; c < g.hidden; ++c) {
                        resid[std::size_t(c)] =
                            rbf(block[std::size_t(c)] + resid[std::size_t(c)]);
                    }
                    t[at] = resid;
                    break;
                case Kind::FfnNorm:
                    normed = rms_norm(resid, n_ffn[std::size_t(L)], g.hidden, g.eps);
                    round_bf16(normed);
                    t[at] = normed;
                    break;
                case Kind::QmvGate:
                    gate = matvec(wgate[std::size_t(L)], normed);
                    round_bf16(gate);
                    t[at] = gate;
                    break;
                case Kind::QmvUp:
                    up = matvec(wup[std::size_t(L)], normed);
                    round_bf16(up);
                    t[at] = up;
                    break;
                case Kind::SiluMul:
                    act.assign(std::size_t(g.intermediate), 0.0f);
                    for (int c = 0; c < g.intermediate; ++c) {
                        act[std::size_t(c)] =
                            silu_mul_one(gate[std::size_t(c)], up[std::size_t(c)]);
                    }
                    t[at] = act;
                    break;
                case Kind::QmvDown:
                    block = matvec(wdown[std::size_t(L)], act);
                    round_bf16(block);
                    t[at] = block;
                    break;

                // ── routed ──
                case Kind::Router:
                    logits = matvec(wrouter[std::size_t(L)], normed);
                    round_bf16(logits);
                    t[at] = logits;
                    break;
                case Kind::RouterTopK: {
                    const int kk = g.experts_per_token;
                    expert_ids.assign(std::size_t(kk), 0);
                    expert_w.assign(std::size_t(kk), 0.0f);
                    Vec pool = logits;
                    Vec chosen(std::size_t(kk), 0.0f);
                    for (int r = 0; r < kk; ++r) {
                        int best = 0;
                        // `>` and not `>=`: the kernel resolves a tie toward the
                        // LOWER expert index, at both levels of its reduction.
                        for (int e = 1; e < g.n_experts; ++e) {
                            if (pool[std::size_t(e)] > pool[std::size_t(best)]) best = e;
                        }
                        expert_ids[std::size_t(r)] = best;
                        chosen[std::size_t(r)] = pool[std::size_t(best)];
                        pool[std::size_t(best)] = -3.0e38f;
                    }
                    // Softmax over the k SELECTED logits, not over all n.
                    float mx = -3.0e38f, sum = 0;
                    for (int r = 0; r < kk; ++r) mx = std::max(mx, chosen[std::size_t(r)]);
                    for (int r = 0; r < kk; ++r) {
                        chosen[std::size_t(r)] = std::exp(chosen[std::size_t(r)] - mx);
                        sum += chosen[std::size_t(r)];
                    }
                    for (int r = 0; r < kk; ++r) {
                        expert_w[std::size_t(r)] = rbf(chosen[std::size_t(r)] / sum);
                    }
                    // `pool` now holds the rejected logits; the selected ones
                    // were knocked out as they were taken.
                    float best_rejected = -3.0e38f;
                    for (int e = 0; e < g.n_experts; ++e) {
                        best_rejected = std::max(best_rejected, pool[std::size_t(e)]);
                    }
                    float worst_selected = 3.0e38f;
                    for (int r = 0; r < kk; ++r) {
                        worst_selected = std::min(worst_selected, logits[std::size_t(expert_ids[std::size_t(r)])]);
                    }
                    last_router_logits = logits;
                    last_margin = worst_selected - best_rejected;
                    step_margins.push_back({at, last_margin});
                    last_ids = expert_ids;
                    last_w = expert_w;
                    break;  // two outputs of different types; checked via the combine
                }
                case Kind::ExpertGate:
                case Kind::ExpertUp: {
                    const int kk = g.experts_per_token;
                    const QuantTensor& w =
                        d.kind == Kind::ExpertGate ? wgate[std::size_t(L)] : wup[std::size_t(L)];
                    Vec out(std::size_t(kk) * std::size_t(g.moe_intermediate), 0.0f);
                    for (int s = 0; s < kk; ++s) {
                        // Slot stride 0: every expert reads the ONE shared norm
                        // output. This is half of the asymmetry `down` completes.
                        const Vec y = matvec(w, normed, expert_ids[std::size_t(s)]);
                        for (int c = 0; c < g.moe_intermediate; ++c) {
                            out[std::size_t(s * g.moe_intermediate + c)] =
                                rbf(y[std::size_t(c)]);
                        }
                    }
                    (d.kind == Kind::ExpertGate ? gate : up) = out;
                    t[at] = out;
                    break;
                }
                case Kind::ExpertSiluMul: {
                    const std::size_t n =
                        std::size_t(g.experts_per_token) * std::size_t(g.moe_intermediate);
                    act.assign(n, 0.0f);
                    for (std::size_t c = 0; c < n; ++c) act[c] = silu_mul_one(gate[c], up[c]);
                    t[at] = act;
                    break;
                }
                case Kind::ExpertDown: {
                    const int kk = g.experts_per_token;
                    Vec out(std::size_t(kk) * std::size_t(g.hidden), 0.0f);
                    for (int s = 0; s < kk; ++s) {
                        // Slot stride `moe_intermediate`: each expert reads ITS
                        // OWN slot of the SwiGLU stack. Reading slot 0 for all
                        // of them is a plausible wrong token, not a crash.
                        Vec x(std::size_t(g.moe_intermediate), 0.0f);
                        for (int c = 0; c < g.moe_intermediate; ++c) {
                            x[std::size_t(c)] = act[std::size_t(s * g.moe_intermediate + c)];
                        }
                        const Vec y = matvec(wdown[std::size_t(L)], x,
                                             expert_ids[std::size_t(s)]);
                        for (int c = 0; c < g.hidden; ++c) {
                            out[std::size_t(s * g.hidden + c)] = rbf(y[std::size_t(c)]);
                        }
                    }
                    t[at] = out;
                    act = out;
                    break;
                }
                case Kind::ExpertCombine: {
                    block.assign(std::size_t(g.hidden), 0.0f);
                    for (int c = 0; c < g.hidden; ++c) {
                        float acc = 0;
                        for (int s = 0; s < g.experts_per_token; ++s) {
                            acc += expert_w[std::size_t(s)] *
                                   act[std::size_t(s * g.hidden + c)];
                        }
                        block[std::size_t(c)] = rbf(acc);
                    }
                    t[at] = block;
                    break;
                }

                case Kind::RowGather:
                    t[at] = resid;  // one row, gathered from index 0
                    break;
                case Kind::FinalRms:
                    normed = rms_norm(resid, n_final, g.hidden, g.eps);
                    round_bf16(normed);
                    t[at] = normed;
                    break;
                case Kind::LmHead: {
                    Vec y = matvec(head, normed);
                    round_bf16(y);
                    t[at] = y;
                    break;
                }
                case Kind::Argmax:
                    break;
            }
        }
        return t;
    }
};

// ── the device side ─────────────────────────────────────────────────────────

void write_u32(SlotHandle s, std::uint32_t v) {
    if (s.contents() != nullptr) std::memcpy(s.contents(), &v, 4);
}

void write_u32s(SlotHandle s, const std::vector<std::uint32_t>& v) {
    if (s.contents() != nullptr && !v.empty()) std::memcpy(s.contents(), v.data(), v.size() * 4);
}

/// Stage one quantized tensor under the three names `push_quant` asked for.
void stage_quant(RawMetalContext& ctx, BoundLlama& b, const std::vector<WeightBind>& binds,
                 const QuantTensor& q) {
    for (const WeightBind& wb : binds) {
        const void* src = nullptr;
        std::size_t bytes = 0;
        switch (wb.bind_index) {
            case 0:
                src = q.packed.data();
                bytes = q.packed.size() * 4;
                break;
            case 1:
                src = q.scales.data();
                bytes = q.scales.size() * 2;
                break;
            case 2:
                src = q.biases.data();
                bytes = q.biases.size() * 2;
                break;
            default:
                continue;
        }
        SlotHandle h = ctx.heap_alloc(bytes);
        if (h.contents() != nullptr) std::memcpy(h.contents(), src, bytes);
        b.weights[wb.tensor] = h;
    }
}

void stage_norm(RawMetalContext& ctx, BoundLlama& b, const std::vector<WeightBind>& binds,
                const std::vector<std::uint16_t>& w) {
    for (const WeightBind& wb : binds) {
        SlotHandle h = ctx.heap_alloc(w.size() * 2);
        if (h.contents() != nullptr) std::memcpy(h.contents(), w.data(), w.size() * 2);
        b.weights[wb.tensor] = h;
    }
}

/// How far one dispatch's output is from the reference, as ||got - want|| over
/// ||want||.
///
/// Not a per-element bound. These tensors have a wide dynamic range, so the
/// largest RELATIVE error is always on the smallest element and says nothing --
/// a correct kernel and a broken one both have elements near zero. The L2 ratio
/// is what separates them, and it separates them by two orders of magnitude:
/// bf16 rounding accumulated over two layers lands around 1%, while a rope that
/// does not rotate, an attention that reads the wrong KV head, or an expert fed
/// the wrong slot are all O(1).
// The two halves `rel_l2` divides, kept apart.
//
// A row is judged against the scale of the DISPATCH that produced it, not
// against its own -- see `judge` -- and that needs the error and the magnitude
// separately.
struct L2 {
    float err;    // ||got - want||
    float scale;  // ||want||
};
L2 l2_parts(const Vec& got, const Vec& want) {
    if (got.size() != want.size() || want.empty()) return L2{1e30f, 1.0f};
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i]) - double(want[i]);
        num += d * d;
        den += double(want[i]) * double(want[i]);
    }
    return L2{float(std::sqrt(num)), float(std::sqrt(den))};
}

float rel_l2(const Vec& got, const Vec& want) {
    if (got.size() != want.size() || want.empty()) return 1e30f;
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i]) - double(want[i]);
        num += d * d;
        den += double(want[i]) * double(want[i]);
    }
    if (den <= 0.0) return num > 0.0 ? 1e30f : 0.0f;
    return float(std::sqrt(num / den));
}

struct Model {
    QuantTensor embed, head;
    std::vector<QuantTensor> wq, wk, wv, wo, wgate, wup, wdown, wrouter;
    std::vector<Vec> n_attn, n_ffn, n_q, n_k;
    Vec n_final;
    std::vector<std::vector<std::uint16_t>> r_attn, r_ffn, r_q, r_k;
    std::vector<std::uint16_t> r_final;
};

void build_model(const LlamaGeometry& g, Model& m) {
    const int ffn_out = g.is_moe() ? g.moe_intermediate : g.intermediate;
    const int experts = g.is_moe() ? g.n_experts : 1;
    m.embed = make_quant(1, g.vocab, g.hidden, 11);
    m.head = g.tied_embeddings ? m.embed : make_quant(1, g.vocab, g.hidden, 12);
    m.n_final = make_norm(g.hidden, 13, m.r_final);
    for (int L = 0; L < g.n_layers; ++L) {
        const std::uint64_t s = 100 + std::uint64_t(L) * 17;
        m.wq.push_back(make_quant(1, g.q_width(), g.hidden, s + 1));
        m.wk.push_back(make_quant(1, g.kv_width(), g.hidden, s + 2));
        m.wv.push_back(make_quant(1, g.kv_width(), g.hidden, s + 3));
        m.wo.push_back(make_quant(1, g.hidden, g.q_width(), s + 4));
        m.wgate.push_back(make_quant(experts, ffn_out, g.hidden, s + 5));
        m.wup.push_back(make_quant(experts, ffn_out, g.hidden, s + 6));
        m.wdown.push_back(make_quant(experts, g.hidden, ffn_out, s + 7));
        m.wrouter.push_back(g.is_moe() ? make_quant(1, g.n_experts, g.hidden, s + 8)
                                       : QuantTensor{});
        m.r_attn.emplace_back();
        m.r_ffn.emplace_back();
        m.r_q.emplace_back();
        m.r_k.emplace_back();
        m.n_attn.push_back(make_norm(g.hidden, s + 9, m.r_attn.back()));
        m.n_ffn.push_back(make_norm(g.hidden, s + 10, m.r_ffn.back()));
        m.n_q.push_back(make_norm(g.head_dim, s + 11, m.r_q.back()));
        m.n_k.push_back(make_norm(g.head_dim, s + 12, m.r_k.back()));
    }
}

/// `rows` > 1 runs the whole prompt as ONE fire instead of a token at a time,
/// and `paged` swaps the attention ABI. Both compare against the SAME
/// sequential fp32 reference: a batched fire's row r must compute what the
/// r-th decode computes, because causal attention over the rows means row r
/// sees keys 0..r and nothing else. That is the entire claim the M>1 path
/// makes, and it is checked here per DISPATCH rather than at the logits.
void run_case(const char* who, LlamaGeometry g, RawMetalContext& ctx,
              const std::string& kernels_dir, float tol, int rows = 1, bool paged = false,
              int steps = 0, int requests = 1) {
    std::printf("\n-- %s --\n", who);
    const int R = rows < 1 ? 1 : rows;
    // `steps` is only meaningful at one row: it makes the sequential path run a
    // sequence as long as a batched case's, which is what separates "the batch
    // is wrong" from "a longer sequence simply accumulates more rounding".
    const int S = steps > 0 ? steps : 0;
    if (paged) {
        g.paged_kv_enabled = true;
        g.kv_page_size = 32;
        g.kv_max_ctx = ((g.kv_max_ctx + 31) / 32) * 32;
        g.total_pages = g.kv_max_ctx / 32;
    }
    Model m;
    build_model(g, m);

    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);
    const ScratchPlan plan = build_llama_scratch(dag, g);
    // Every value gets its own buffer, so every dispatch's output is readable.
    const model::ScratchColoring col =
        color_llama_scratch(dag, plan, /*no_recycle=*/true);
    if (!col.hazard_free) {
        expect(false, std::string(who) + ": the colouring is hazard-free");
        return;
    }

    BoundLlama b;
    for (const Dispatch& d : dag) {
        const auto binds = weight_binds(shared_kind(d.kind, g), d.layer, DecodeGeometry{}, false);
        if (binds.empty()) continue;
        if (b.weights.count(binds.front().tensor) != 0) continue;
        const int L = d.layer;
        switch (d.kind) {
            case Kind::EmbedGather: stage_quant(ctx, b, binds, m.embed); break;
            case Kind::LmHead:      stage_quant(ctx, b, binds, m.head); break;
            case Kind::QmvQ:        stage_quant(ctx, b, binds, m.wq[std::size_t(L)]); break;
            case Kind::QmvK:        stage_quant(ctx, b, binds, m.wk[std::size_t(L)]); break;
            case Kind::QmvV:        stage_quant(ctx, b, binds, m.wv[std::size_t(L)]); break;
            case Kind::QmvO:        stage_quant(ctx, b, binds, m.wo[std::size_t(L)]); break;
            case Kind::QmvGate:
            case Kind::ExpertGate:  stage_quant(ctx, b, binds, m.wgate[std::size_t(L)]); break;
            case Kind::QmvUp:
            case Kind::ExpertUp:    stage_quant(ctx, b, binds, m.wup[std::size_t(L)]); break;
            case Kind::QmvDown:
            case Kind::ExpertDown:  stage_quant(ctx, b, binds, m.wdown[std::size_t(L)]); break;
            case Kind::Router:      stage_quant(ctx, b, binds, m.wrouter[std::size_t(L)]); break;
            case Kind::AttnNorm:    stage_norm(ctx, b, binds, m.r_attn[std::size_t(L)]); break;
            case Kind::FfnNorm:     stage_norm(ctx, b, binds, m.r_ffn[std::size_t(L)]); break;
            case Kind::QNorm:       stage_norm(ctx, b, binds, m.r_q[std::size_t(L)]); break;
            case Kind::KNorm:       stage_norm(ctx, b, binds, m.r_k[std::size_t(L)]); break;
            case Kind::FinalRms:    stage_norm(ctx, b, binds, m.r_final); break;
            default: break;
        }
    }

    b.pool.resize(std::size_t(col.colors_used));
    // Padded, like the engine's: a dense projection's GEMM rounds its batch up
    // to a whole tile and writes the padding rows for real.
    const int pool_rows = llama::llama_qmm_pool_rows(R);
    const std::vector<std::size_t> elems =
        llama_pool_elems(dag, plan, col, g, pool_rows, pool_rows);
    for (int c = 0; c < col.colors_used; ++c) {
        b.pool[std::size_t(c)] = ctx.heap_alloc(elems[std::size_t(c)] * 2);
    }
    if (g.is_moe()) {
        b.zero_bias = ctx.heap_alloc(std::size_t(std::max(g.hidden, g.moe_intermediate)) * 2);
        if (b.zero_bias.contents() != nullptr) {
            std::memset(b.zero_bias.contents(), 0, b.zero_bias.size);
        }
    }
    b.io.resize(kIoSlotCount);
    const std::size_t io_bytes =
        std::max<std::size_t>(4096, std::size_t(g.total_pages + 8) * 4);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[std::size_t(i)] = ctx.heap_alloc(io_bytes);
    b.kv.resize(std::size_t(g.n_layers));
    std::vector<RefKv> ref_kv(std::size_t(g.n_layers));
    for (int L = 0; L < g.n_layers; ++L) {
        const std::size_t bytes = llama_kv_bytes_per_layer(g, g.kv_max_ctx, 2);
        b.kv[std::size_t(L)].k = ctx.heap_alloc(bytes);
        b.kv[std::size_t(L)].v = ctx.heap_alloc(bytes);
        if (b.kv[std::size_t(L)].k.contents() != nullptr) {
            std::memset(b.kv[std::size_t(L)].k.contents(), 0, bytes);
            std::memset(b.kv[std::size_t(L)].v.contents(), 0, bytes);
        }
        const std::size_t n =
            std::size_t(g.n_kv_heads) * std::size_t(g.kv_max_ctx) * std::size_t(g.head_dim);
        ref_kv[std::size_t(L)].k.assign(n, 0.0f);
        ref_kv[std::size_t(L)].v.assign(n, 0.0f);
    }

    LlamaPsos ll;
    DecodeStepPsos base;
    MultiBatchPsos mb;
    std::string err;
    if (!build_llama_psos(ctx, kernels_dir, g, ll, &err) ||
        !load_decode_psos(ctx, kernels_dir, base, g.quant, &err)) {
        expect(false, std::string(who) + ": pipelines compiled (" + err + ")");
        return;
    }
    if (paged && !load_multibatch_psos(
                     ctx, kernels_dir, mb, g.quant, &err,
                     pie::metal::MultiBatchPsoFeatures{
                         .routed = g.is_moe(),
                         .splitk = true,
                         .fp16_precast = !g.is_moe() &&
                             g.quant.bits == 4 && g.quant.group == 64})) {
        expect(false, std::string(who) + ": multi-batch pipelines compiled (" + err + ")");
        return;
    }
    const MultiBatchPsos* mbp = paged ? &mb : nullptr;
    bind_llama_consts(ctx, dag, g, R, paged);
    // A split projection is two dispatches sharing one argument table, and the
    // second one reads the partials the first wrote. Skipping this bind is not
    // an unbound-slot crash: the reduce sums whatever the partials buffer holds
    // and writes it over the projection's output.
    SlotHandle splitk_partial = ctx.heap_alloc(
        sizeof(float) * llama::llama_splitk_partial_elems(g, R));
    std::vector<SlotHandle> splitk_keep;
    llama::bind_llama_splitk(ctx, dag, g, R, splitk_partial, splitk_keep);
    SlotHandle fp16_input = ctx.heap_alloc(
        sizeof(std::uint16_t) * std::size_t(llama::llama_qmm_pool_rows(R)) *
        std::size_t(std::max(g.hidden, g.intermediate)));
    std::vector<SlotHandle> fp16_keep;
    llama::bind_llama_fp16_qmm(ctx, dag, g, R, R, R, fp16_input, fp16_keep);
    try {
        bind_llama_dag(ctx, b, dag, g, col, /*ordinal_base=*/0, paged);
    } catch (const std::exception& e) {
        expect(false, std::string(who) + ": bound (" + e.what() + ")");
        return;
    }
    ctx.make_resident();

    // Which value each dispatch WROTE, so its output can be found in the pool.
    //
    // The pool is read once, after the step -- so only the LAST writer of a
    // value is still visible in it. Rope and the qk-norms write in place, so
    // `QmvQ -> QNorm -> RopeQ` are three dispatches sharing one value and only
    // the rope's result survives. Comparing the other two against the reference
    // would fail on a correct model, which is a worse failure than missing
    // them: the rope's own check subsumes both, because it reads what they
    // wrote.
    std::vector<int> wrote(dag.size(), -1);
    std::unordered_map<int, int> last_writer;
    for (const Use& u : plan.uses) {
        if (!u.is_write) continue;
        wrote[std::size_t(u.index)] = u.value;
        auto it = last_writer.find(u.value);
        if (it == last_writer.end() || u.index > it->second) last_writer[u.value] = u.index;
    }
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const int v = wrote[i];
        if (v >= 0 && last_writer[v] != int(i)) wrote[i] = -1;
    }

    Reference ref{g,        m.embed,  m.head,   m.wq,    m.wk,     m.wv,
                  m.wo,     m.wgate,  m.wup,    m.wdown, m.wrouter, m.n_attn,
                  m.n_ffn,  m.n_q,    m.n_k,    m.n_final, ref_kv};

    std::vector<int> tokens = {7, 130, 42, 901};
    // Enough rows to fill a GEMM tile when the case asks for it. Arbitrary but
    // fixed, and inside the vocabulary.
    for (int i = 0; int(tokens.size()) < std::max(R, S); ++i) {
        tokens.push_back((i * 137 + 11) % g.vocab);
    }
    int first_bad = -1;
    std::string first_bad_name;
    float worst = 0.0f;
    int compared = 0;

    // Every comparison, judged only once they are all in.
    //
    // The obvious metric -- ||got-want|| / ||want||, per row -- asks a
    // different question of every row, because it uses each row's own
    // magnitude as the yardstick. A row whose output nearly cancels is then
    // held to a far tighter ABSOLUTE standard than its neighbours, and fails
    // for being small rather than for being wrong. That is not hypothetical:
    // a dense fire of 36 rows over two requests put a single row of the layer-0
    // down projection at 0.2373 while every other row sat under 0.05, and that
    // row's output had a norm of 5.2 where the others ran from 16 to 190 --
    // an absolute error SEVEN TIMES SMALLER than a row that passed.
    //
    // So the row's own magnitude is used as a yardstick only while it is at
    // least what the dispatch typically produces; below that, the dispatch's
    // scale takes over. No row is judged more strictly than before and a row
    // is never asked to be more accurate for having come out small. The scale
    // is measured from the run rather than written down, so it cannot be the
    // number that makes today's pass.
    struct Cmp {
        int disp;
        float err;
        float scale;
        std::string where;
    };
    std::vector<Cmp> cmps;

    // Fills the paged IO for a fire of `n` rows at positions `p0 .. p0+n-1`,
    // all of one request. The page list is the identity, so the page table is
    // exercised as a real indirection while staying easy to reason about.
    // One row of a fire: which token, at which position OF ITS OWN REQUEST,
    // owned by which request, written to which physical KV page. Everything the
    // paged ABI needs per row, and the reason there is one writer rather than
    // two -- the single-request case is the plan where every row says req 0.
    struct RowPlan {
        int token = 0;
        int pos = 0;
        int req = 0;
        int page = 0;
    };

    auto write_paged_io_rows = [&](const std::vector<RowPlan>& rows,
                                   const std::vector<std::uint32_t>& page_indices,
                                   const std::vector<std::uint32_t>& page_indptr,
                                   const std::vector<std::uint32_t>& qo_indptr) {
        std::vector<std::uint32_t> ids, pos, req, wpage, woff, sample;
        int max_pos = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            ids.push_back(std::uint32_t(rows[r].token));
            pos.push_back(std::uint32_t(rows[r].pos));
            req.push_back(std::uint32_t(rows[r].req));
            wpage.push_back(std::uint32_t(rows[r].page));
            woff.push_back(std::uint32_t(rows[r].pos % g.kv_page_size));
            sample.push_back(std::uint32_t(r));
            max_pos = std::max(max_pos, rows[r].pos);
        }
        write_u32s(b.io[std::size_t(IoSlot::TokenId)], ids);
        write_u32s(b.io[std::size_t(IoSlot::Position)], pos);
        write_u32s(b.io[std::size_t(IoSlot::ReqOfToken)], req);
        write_u32s(b.io[std::size_t(IoSlot::WPage)], wpage);
        write_u32s(b.io[std::size_t(IoSlot::WOff)], woff);
        write_u32s(b.io[std::size_t(IoSlot::KvPageIndices)], page_indices);
        write_u32s(b.io[std::size_t(IoSlot::KvPageIndptr)], page_indptr);
        write_u32s(b.io[std::size_t(IoSlot::QoIndptr)], qo_indptr);
        write_u32s(b.io[std::size_t(IoSlot::SampleRows)], sample);
        write_u32(b.io[std::size_t(IoSlot::SeqLen)], std::uint32_t(max_pos + 1));
        write_u32s(b.io[std::size_t(IoSlot::AttnMaskStride)], {0u});
        if (b.io[std::size_t(IoSlot::AttnMaskEnabled)].contents() != nullptr) {
            std::memset(b.io[std::size_t(IoSlot::AttnMaskEnabled)].contents(), 0, rows.size());
        }
    };

    // The single-request plan: contiguous positions, the identity page list.
    auto write_paged_io = [&](int p0, int n, const std::vector<int>& toks) {
        std::vector<RowPlan> rows;
        for (int r = 0; r < n; ++r) {
            rows.push_back(RowPlan{toks[std::size_t(p0 + r)], p0 + r, 0,
                                   (p0 + r) / g.kv_page_size});
        }
        std::vector<std::uint32_t> pages;
        for (int i = 0; i < g.total_pages; ++i) pages.push_back(std::uint32_t(i));
        write_paged_io_rows(rows, pages, {0u, std::uint32_t(g.total_pages)},
                            {0u, std::uint32_t(n)});
    };

    // Compares one dispatch's output, at row `row` of a [rows, width] tensor.
    // The row pitch is the M=1 trace's own length, which already carries the
    // expert-slot axis for the routed values -- a routed value is
    // [rows, k, width] and its M=1 trace is [k, width].
    float worst_pristine = 0.0f;
    // The four values stacked per expert SLOT. When the router picks the same
    // experts in a different slot order these hold the same numbers permuted,
    // which is not a disagreement about anything.
    const auto is_slot_stacked = [](Kind k) {
        return k == Kind::ExpertGate || k == Kind::ExpertUp || k == Kind::ExpertSiluMul ||
               k == Kind::ExpertDown;
    };

    // The permutation the GPU actually produced, per layer.
    //
    // Read rather than predicted, and that is not a weakening: the sort's
    // scatter ranks each pair with an atomic, so the order WITHIN one expert's
    // run is whatever the threads raced to, and a host that recomputed it would
    // be asserting an order the kernel never promised. What the kernel does
    // promise -- that every pair gets exactly one position and that the row it
    // lands on reads its own expert -- is checked below, and once the mapping
    // is known the projections' arithmetic is compared element for element as
    // before.
    std::vector<const std::int32_t*> perm_of(std::size_t(g.n_layers), nullptr);
    if (g.is_moe()) {
        for (std::size_t i = 0; i < dag.size(); ++i) {
            if (dag[i].kind != Kind::ExpertSort) continue;
            // The sort's FIRST write is `perm`; see `build_llama_scratch`.
            int pv = -1;
            for (const Use& u : plan.uses) {
                if (u.is_write && u.index == int(i) && (pv < 0 || u.value < pv)) pv = u.value;
            }
            if (pv < 0) continue;
            const SlotHandle& slot = b.pool[std::size_t(col.color_of_value[std::size_t(pv)])];
            if (slot.contents() == nullptr) continue;
            perm_of[std::size_t(dag[i].layer)] = static_cast<const std::int32_t*>(slot.contents());
        }
    }

    // Where token `row`'s k slots landed, as (buffer row, reference slot)
    // pairs. Empty for a dense model or a layer whose sort was not readable,
    // which is what puts the caller back on the flat layout.
    const auto sorted_pairs = [&](int layer, int row) {
        std::vector<std::pair<int, int>> out;
        if (layer < 0 || !g.is_moe()) return out;
        const std::int32_t* perm = perm_of[std::size_t(layer)];
        if (perm == nullptr) return out;
        const int kk = g.experts_per_token;
        const int padded = llama_moe_sorted_rows(g, rows);
        for (int pp = 0; pp < padded; ++pp) {
            const std::int32_t sel = perm[pp];
            if (sel >= 0 && sel / kk == row) out.push_back({pp, int(sel % kk)});
        }
        return out;
    };

    auto compare_all = [&](const Trace& want, int row, int label, bool skip_slots = false,
                           int cut = std::numeric_limits<int>::max()) {
        for (std::size_t i = 0; i < dag.size(); ++i) {
            if (int(i) >= cut) break;
            const auto it = want.find(int(i));
            if (it == want.end()) continue;
            if (skip_slots && is_slot_stacked(dag[i].kind)) continue;
            const int v = wrote[i];
            if (v < 0) continue;
            const int c = col.color_of_value[std::size_t(v)];
            const SlotHandle& slot = b.pool[std::size_t(c)];
            if (slot.contents() == nullptr) continue;
            // Between the gather and the scatter the rows are in EXPERT order,
            // padded, so `row * width` is not where this row's slots landed.
            // `perm` says where they did.
            if (is_expert_sorted(dag[i].kind)) {
                const std::size_t width =
                    it->second.size() / std::size_t(g.experts_per_token);
                for (const auto& pr : sorted_pairs(dag[i].layer, row)) {
                    const std::size_t off = std::size_t(pr.first) * width;
                    if ((off + width) * 2 > slot.size) continue;
                    const auto* raw =
                        static_cast<const std::uint16_t*>(slot.contents()) + off;
                    Vec got(width);
                    for (std::size_t e = 0; e < width; ++e) got[e] = from_bf16(raw[e]);
                    const Vec ref_slot(it->second.begin() + std::size_t(pr.second) * width,
                                       it->second.begin() + (std::size_t(pr.second) + 1) * width);
                    ++compared;
                    const L2 p2 = l2_parts(got, ref_slot);
                    char buf[176];
                    std::snprintf(buf, sizeof buf,
                                  "row %d slot %d at sorted %d, dispatch %d (kind %d, layer %d)",
                                  label, pr.second, pr.first, int(i), int(dag[i].kind),
                                  dag[i].layer);
                    cmps.push_back(Cmp{int(i), p2.err, p2.scale, buf});
                }
                continue;
            }
            const std::size_t w = it->second.size();
            const std::size_t off = std::size_t(row) * w;
            if ((off + w) * 2 > slot.size) continue;
            const auto* raw = static_cast<const std::uint16_t*>(slot.contents()) + off;
            Vec got(w);
            for (std::size_t e = 0; e < w; ++e) got[e] = from_bf16(raw[e]);
            ++compared;
            const L2 p2 = l2_parts(got, it->second);
            {
                char buf[160];
                std::snprintf(buf, sizeof buf, "row %d, dispatch %d (kind %d, layer %d)", label,
                              int(i), int(dag[i].kind), dag[i].layer);
                cmps.push_back(Cmp{int(i), p2.err, p2.scale, buf});
            }
            const float w1 = rel_l2(got, it->second);
            // Layer 0's Q/K/V read the token embedding, which is exact. They
            // are therefore the ONLY dispatches whose error is their own rather
            // than something they inherited, and at sixteen rows they are
            // `affine_qmm_t`. Everything downstream of them sits behind
            // attention, which is a softmax: it turns a small difference in the
            // scores into a visibly different mixture of values, and that
            // amplification is a property of the arithmetic, not evidence about
            // the kernel. So the kernel under test gets its own tight bound.
            if (dag[i].layer == 0 &&
                (dag[i].kind == Kind::QmvQ || dag[i].kind == Kind::QmvK ||
                 dag[i].kind == Kind::QmvV)) {
                if (w1 > worst_pristine) worst_pristine = w1;
            }
            if (w1 <= tol) continue;
            if (std::getenv("PIE_NUM_DEBUG") != nullptr) {
                double ng = 0, nw = 0;
                for (std::size_t e = 0; e < w; ++e) { ng += double(got[e]) * got[e]; nw += double(it->second[e]) * it->second[e]; }
                std::printf("    [dbg] row %d disp %zu kind %d layer %d rel_l2 %.4f |g|/|w| %.4f got/want:",
                            label, i, int(dag[i].kind), dag[i].layer, double(w1),
                            nw > 0 ? std::sqrt(ng / nw) : 0.0);
                for (std::size_t e = 0; e < w && e < 4; ++e) {
                    std::printf(" %.4f/%.4f", double(got[e]), double(it->second[e]));
                }
                std::printf("\n");
            }
        }
    };

    // The verdict, once every row of every dispatch has been seen.
    //
    // A dispatch's scale is the root-mean-square of its rows' magnitudes, which
    // is the size of the thing it computes. A row fails when its error exceeds
    // `tol` times THAT -- so a row that is small because it cancelled is held
    // to the same absolute standard as the row beside it, and a row that is
    // wrong is still wrong however large it is.
    const auto judge = [&]() {
        std::map<int, std::pair<double, int>> sq;  // dispatch -> (sum of scale^2, n)
        for (const auto& c : cmps) {
            auto& e = sq[c.disp];
            e.first += double(c.scale) * c.scale;
            e.second += 1;
        }
        for (const auto& c : cmps) {
            const auto& e = sq[c.disp];
            const double rms = e.second > 0 ? std::sqrt(e.first / e.second) : 0.0;
            // A FLOOR, not a replacement. A row larger than its dispatch's
            // typical output keeps its own yardstick and the bar on it does
            // not move; only a row well under that scale is rescued, and what
            // rescues it is being held to the same ABSOLUTE error as its
            // neighbours instead of a proportionally tiny one. Dividing by the
            // rms outright would do the opposite as well, tightening the bar
            // on every row above the mean -- which is most of a routed
            // mixture, whose experts differ in magnitude by design.
            const double denom = std::max(double(c.scale), rms);
            const float rel = denom > 0.0 ? float(c.err / denom)
                                          : (c.err > 0.0f ? 1e30f : 0.0f);
            if (rel > worst) worst = rel;
            if (rel > tol && first_bad < 0) {
                first_bad = c.disp;
                first_bad_name = c.where;
            }
        }
    };

    // Which rows the router sent somewhere the reference did not, and whether
    // that difference is decidable. Two expert logits close enough that fp32
    // and bf16 sort them differently is a property of the weights, not of the
    // driver, and every dispatch below such a row is computing a different
    // expert's arithmetic -- so the row is excluded, LOUDLY. A routing
    // difference at a clear margin is a real fault and is left to fail.
    //
    // One copy, used by both the single-request and the two-request arms: the
    // rule for what counts as a tie must not be able to differ between them.
    //
    // The threshold is MEASURED, not chosen. The device's router logits and the
    // reference's differ by some amount d -- bf16 against fp32, a different
    // reduction order -- and a gap narrower than that is a gap neither
    // implementation can be said to have resolved. Writing a constant here
    // instead would be picking the number that makes today's run pass.
    //
    // What comes back is not a yes/no but a CUT: the DAG position from which
    // this row stops being comparable. The two facts available are of
    // different kinds and are used for different things. The device's ids say
    // WHETHER a selection came apart -- but only the last layer's, since every
    // layer writes that one value and the pool is read once. The reference's
    // margins say WHERE it could have: a layer that decided at a margin wider
    // than the routers' own disagreement cannot have flipped, so the earliest
    // flip is at the first layer that decided inside it. Everything before
    // that point ran the same arithmetic as the reference and is held to the
    // same tolerance as any other row; only from there on is the row set
    // aside. Discarding the whole row -- which is what a boolean did -- threw
    // away every layer beneath the tie for a tie that happened above them.
    auto route_skips = [&](const std::vector<std::vector<int>>& want_ids,
                           const std::vector<std::vector<std::pair<int, float>>>& want_margins,
                           const std::vector<Vec>& want_logits, int n_rows, int& ambiguous,
                           std::vector<bool>& permuted) {
        const int kAll = int(dag.size());
        std::vector<int> cut(std::size_t(n_rows), kAll);
        permuted.assign(std::size_t(n_rows), false);
        ambiguous = 0;
        if (!g.is_moe() || plan.expert_ids_by_layer.empty()) return cut;
        const auto& ids_slot =
            b.pool[std::size_t(col.color_of_value[std::size_t(plan.expert_ids_by_layer.back())])];
        const auto* ids = static_cast<const std::int32_t*>(ids_slot.contents());
        if (ids == nullptr) return cut;
        // The device's own router logits, for the last layer -- the same layer
        // whose selection `ids` holds.
        int router_disp = -1;
        for (std::size_t i = 0; i < dag.size(); ++i) {
            if (dag[i].kind == Kind::Router && wrote[i] >= 0) router_disp = int(i);
        }
        for (int r = 0; r < n_rows; ++r) {
            bool apart = false;
            for (int e = 0; e < g.experts_per_token; ++e) {
                if (ids[r * g.experts_per_token + e] !=
                    want_ids[std::size_t(r)][std::size_t(e)]) {
                    apart = true;
                }
            }
            if (!apart) continue;
            // Same experts, different slot order? Then nothing was routed
            // wrongly: the per-slot tensors are a permutation of each other and
            // the combine -- which is what the rest of the model reads -- is
            // unaffected. Only those four values are set aside.
            std::vector<int> dev_set, ref_set;
            for (int e = 0; e < g.experts_per_token; ++e) {
                dev_set.push_back(ids[r * g.experts_per_token + e]);
                ref_set.push_back(want_ids[std::size_t(r)][std::size_t(e)]);
            }
            std::sort(dev_set.begin(), dev_set.end());
            std::sort(ref_set.begin(), ref_set.end());
            if (dev_set == ref_set) {
                permuted[std::size_t(r)] = true;
                std::printf("    note: row %d selected the same experts in a different slot "
                            "order; its per-slot tensors are a permutation, the combine is "
                            "compared as usual\n", r);
                continue;
            }
            // How far apart the two routers actually are on this row.
            float d = 0.0f;
            if (router_disp >= 0 && !want_logits[std::size_t(r)].empty()) {
                const auto& rl = b.pool[std::size_t(
                    col.color_of_value[std::size_t(wrote[std::size_t(router_disp)])])];
                const auto* raw = static_cast<const std::uint16_t*>(rl.contents());
                const std::size_t w = want_logits[std::size_t(r)].size();
                if (raw != nullptr && (std::size_t(r + 1) * w) * 2 <= rl.size) {
                    for (std::size_t e = 0; e < w; ++e) {
                        d = std::max(d, std::fabs(from_bf16(raw[std::size_t(r) * w + e]) -
                                                  want_logits[std::size_t(r)][e]));
                    }
                }
            }
            if (std::getenv("PIE_NUM_DEBUG") != nullptr && router_disp >= 0 &&
                !want_logits[std::size_t(r)].empty()) {
                const auto& rl = b.pool[std::size_t(
                    col.color_of_value[std::size_t(wrote[std::size_t(router_disp)])])];
                const auto* raw = static_cast<const std::uint16_t*>(rl.contents());
                const std::size_t w = want_logits[std::size_t(r)].size();
                std::printf("    [dbg] row %d ids dev", r);
                for (int e = 0; e < g.experts_per_token; ++e) {
                    std::printf(" %d", ids[r * g.experts_per_token + e]);
                }
                std::printf("  ref");
                for (int e = 0; e < g.experts_per_token; ++e) {
                    std::printf(" %d", want_ids[std::size_t(r)][std::size_t(e)]);
                }
                std::printf("\n    [dbg]   logits dev:");
                for (std::size_t e = 0; e < w; ++e) {
                    std::printf(" %.4f", double(from_bf16(raw[std::size_t(r) * w + e])));
                }
                std::printf("\n    [dbg]   logits ref:");
                for (std::size_t e = 0; e < w; ++e) {
                    std::printf(" %.4f", double(want_logits[std::size_t(r)][e]));
                }
                std::printf("\n");
            }
            // Two logits within 2d of each other are a coin flip: shifting
            // either by the error already observed reverses the order.
            //
            // `d` is measured at the only router the pool still holds, and is
            // then applied to every layer's margin. That is an approximation
            // and is stated as one: the layers' routers are different matrices
            // and need not disagree by the same amount. It errs toward
            // cutting, never toward excusing, which is the safe direction --
            // an under-cut row fails loudly, an over-cut one only loses
            // coverage that the boolean was losing outright.
            const float kAmbiguous = 2.0f * d;
            int first_tie = kAll;
            float tie_margin = 0.0f;
            for (const auto& lm : want_margins[std::size_t(r)]) {
                if (lm.second < kAmbiguous) {
                    first_tie = lm.first;
                    tie_margin = lm.second;
                    break;
                }
            }
            if (first_tie < kAll) {
                cut[std::size_t(r)] = first_tie;
                ++ambiguous;
                std::printf("    note: row %d routed differently; the earliest selection it "
                            "could have flipped on decided at a margin of %.4f, inside the "
                            "routers' own disagreement of %.4f -- row compared up to dispatch "
                            "%d and set aside after it\n",
                            r, double(tie_margin), double(d), first_tie);
            } else {
                std::printf("    row %d routed differently and no selection in the step was "
                            "close enough to explain it: the routers' disagreement of %.4f "
                            "covers none of its margins\n", r, double(d));
            }
        }
        return cut;
    };

    // ── two requests in ONE fire ──
    //
    // Everything above this fires a single sequence with an identity page list,
    // which is the one arrangement where ignoring the page table still gives
    // the right answer. Two requests is where paging becomes load-bearing: each
    // owns its own pages, each counts positions from ITS OWN zero, and row r
    // must attend only its own request's keys.
    //
    // The pages are deliberately SWAPPED -- request 0 lives on page 1 and
    // request 1 on page 0 -- so an implementation that derives a page from the
    // position, or that reads the page list in order and ignores the per-request
    // slice of it, computes the other sequence's attention and fails.
    if (requests == 2) {
        const int a = R / 2;
        const int bcount = R - a;
        if (a < 1 || bcount < 1 || a > g.kv_page_size || bcount > g.kv_page_size) {
            expect(false, std::string(who) + ": each request fits in one page");
            return;
        }
        std::vector<RefKv> kv_b = ref_kv;  // its own cache, starting empty
        Reference ref_b{g,       m.embed, m.head,  m.wq,      m.wk,    m.wv,
                        m.wo,    m.wgate, m.wup,   m.wdown,   m.wrouter, m.n_attn,
                        m.n_ffn, m.n_q,   m.n_k,   m.n_final, kv_b};

        // Two different token streams, so a row that attends the wrong request
        // gets visibly wrong numbers rather than coincidentally right ones.
        std::vector<int> tok_a, tok_b;
        for (int i = 0; i < a; ++i) tok_a.push_back(tokens[std::size_t(i)]);
        for (int i = 0; i < bcount; ++i) {
            tok_b.push_back(tokens[std::size_t((i * 31 + 5) % int(tokens.size()))]);
        }

        std::vector<Trace> wants;
        std::vector<std::vector<int>> want_ids;
        std::vector<std::vector<std::pair<int, float>>> want_margins;
        std::vector<Vec> want_logits;
        const auto take = [&](Reference& rf, int tok, int pos) {
            wants.push_back(rf.step(dag, tok, pos));
            want_ids.push_back(rf.last_ids);
            want_margins.push_back(rf.step_margins);
            want_logits.push_back(rf.last_router_logits);
        };
        for (int i = 0; i < a; ++i) take(ref, tok_a[std::size_t(i)], i);
        for (int i = 0; i < bcount; ++i) take(ref_b, tok_b[std::size_t(i)], i);

        std::vector<RowPlan> rows;
        for (int i = 0; i < a; ++i) rows.push_back(RowPlan{tok_a[std::size_t(i)], i, 0, 1});
        for (int i = 0; i < bcount; ++i) {
            rows.push_back(RowPlan{tok_b[std::size_t(i)], i, 1, 0});
        }
        // Request 0's slice of the page list is {1}; request 1's is {0}.
        write_paged_io_rows(rows, {1u, 0u}, {0u, 1u, 2u},
                            {0u, std::uint32_t(a), std::uint32_t(R)});
        ctx.run_step([&](StepEncoder& se) {
            encode_llama_step(se, dag, g, base, ll, /*ordinal_base=*/0, mbp, R, R);
        });
        int ambiguous = 0;
        std::vector<bool> permuted;
        const std::vector<int> cut =
            route_skips(want_ids, want_margins, want_logits, R, ambiguous, permuted);
        // What a row set aside for a routing tie still contributed. The cut
        // exists so that the layers BENEATH the tie are held to the tolerance
        // like anyone else's; if this comes back zero the cut has collapsed
        // into the boolean it replaced and the coverage is gone silently.
        int salvaged = 0;
        for (int r = 0; r < R; ++r) {
            const int before = compared;
            compare_all(wants[std::size_t(r)], r, r, permuted[std::size_t(r)],
                        cut[std::size_t(r)]);
            if (cut[std::size_t(r)] < int(dag.size())) salvaged += compared - before;
        }
        if (ambiguous > 0) {
            expect(salvaged > 0, std::string(who) +
                   ": a row set aside for a routing tie is still compared up to it");
        }
        expect(ambiguous * 4 <= R,
               std::string(who) + ": most rows routed the same way as the reference");

        judge();
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "%s: %d dispatch outputs match, both requests (worst rel_l2 %.4f)", who,
                      compared, double(worst));
        expect(first_bad < 0, msg);
        if (first_bad >= 0) std::printf("    first divergence: %s\n", first_bad_name.c_str());
        expect(compared > 0, std::string(who) + ": something was actually compared");
        return;
    }

    // The batched case: ONE fire over the whole prompt, then every row checked
    // against the decode the reference would have run at that position.
    if (R > 1) {
        std::vector<Trace> wants;
        std::vector<std::vector<int>> want_ids;
        std::vector<std::vector<std::pair<int, float>>> want_margins;
        std::vector<Vec> want_logits;
        for (int step = 0; step < R; ++step) {
            wants.push_back(ref.step(dag, tokens[std::size_t(step)], step));
            want_ids.push_back(ref.last_ids);
            want_margins.push_back(ref.step_margins);
            want_logits.push_back(ref.last_router_logits);
        }
        write_paged_io(0, R, tokens);
        ctx.run_step([&](StepEncoder& se) {
            encode_llama_step(se, dag, g, base, ll, /*ordinal_base=*/0, mbp, R, R);
        });
        int ambiguous = 0;
        std::vector<bool> permuted;
        const std::vector<int> cut =
            route_skips(want_ids, want_margins, want_logits, R, ambiguous, permuted);
        // What a row set aside for a routing tie still contributed. The cut
        // exists so that the layers BENEATH the tie are held to the tolerance
        // like anyone else's; if this comes back zero the cut has collapsed
        // into the boolean it replaced and the coverage is gone silently.
        int salvaged = 0;
        for (int r = 0; r < R; ++r) {
            const int before = compared;
            compare_all(wants[std::size_t(r)], r, r, permuted[std::size_t(r)],
                        cut[std::size_t(r)]);
            if (cut[std::size_t(r)] < int(dag.size())) salvaged += compared - before;
        }
        if (ambiguous > 0) {
            expect(salvaged > 0, std::string(who) +
                   ": a row set aside for a routing tie is still compared up to it");
        }
        expect(ambiguous * 4 <= R, std::string(who) +
               ": most rows routed the same way as the reference");

        judge();
        char msg3[256];
        std::snprintf(msg3, sizeof msg3,
                      "%s: the projections reading the exact embedding are within 2%% "
                      "(worst rel_l2 %.4f)", who, double(worst_pristine));
        expect(worst_pristine <= 0.02f, msg3);

        char msg2[256];
        std::snprintf(msg2, sizeof msg2,
                      "%s: %d dispatch outputs match the reference (worst rel_l2 %.4f)", who,
                      compared, double(worst));
        expect(first_bad < 0, msg2);
        if (first_bad >= 0) std::printf("    first divergence: %s\n", first_bad_name.c_str());
        expect(compared > 0, std::string(who) + ": something was actually compared");
        return;
    }

    const int n_steps = S > 0 ? S : int(tokens.size());
    for (int step = 0; step < n_steps; ++step) {
        if (paged) {
            write_paged_io(step, 1, tokens);
        } else {
            write_u32(b.io[std::size_t(IoSlot::TokenId)],
                      std::uint32_t(tokens[std::size_t(step)]));
            write_u32(b.io[std::size_t(IoSlot::Position)], std::uint32_t(step));
            write_u32(b.io[std::size_t(IoSlot::SeqLen)], std::uint32_t(step) + 1u);
            write_u32(b.io[std::size_t(IoSlot::SampleRows)], 0u);
        }
        ctx.run_step([&](StepEncoder& se) {
            encode_llama_step(se, dag, g, base, ll, /*ordinal_base=*/0, mbp, 1, 1);
        });

        const Trace want = ref.step(dag, tokens[std::size_t(step)], step);
        if (g.is_moe() && std::getenv("PIE_NUM_DEBUG") != nullptr && !plan.expert_ids_by_layer.empty()) {
            const auto& ids_slot =
                b.pool[std::size_t(col.color_of_value[std::size_t(plan.expert_ids_by_layer.back())])];
            const auto& w_slot =
                b.pool[std::size_t(col.color_of_value[std::size_t(plan.expert_weights_by_layer.back())])];
            const auto* ids = static_cast<const std::int32_t*>(ids_slot.contents());
            const auto* wt = static_cast<const std::uint16_t*>(w_slot.contents());
            std::printf("    [dbg] router ids(dev)");
            for (int e = 0; e < g.experts_per_token; ++e) std::printf(" %d", ids[e]);
            std::printf("  w(dev)");
            for (int e = 0; e < g.experts_per_token; ++e) std::printf(" %.4f", double(from_bf16(wt[e])));
            std::printf("  ids(ref)");
            for (int e = 0; e < g.experts_per_token; ++e) std::printf(" %d", ref.last_ids[std::size_t(e)]);
            std::printf("  w(ref)");
            for (int e = 0; e < g.experts_per_token; ++e) std::printf(" %.4f", double(ref.last_w[std::size_t(e)]));
            std::printf("\n");
        }
        for (std::size_t i = 0; i < dag.size(); ++i) {
            const auto it = want.find(int(i));
            if (it == want.end()) continue;
            const int v = wrote[i];
            if (v < 0) continue;
            const int c = col.color_of_value[std::size_t(v)];
            const SlotHandle& slot = b.pool[std::size_t(c)];
            if (slot.contents() == nullptr) continue;
            const auto* raw = static_cast<const std::uint16_t*>(slot.contents());
            // The expert-sorted values hold the same numbers in the order the
            // router grouped them, so they are compared slot by slot through
            // the permutation rather than as one flat vector.
            if (is_expert_sorted(dag[i].kind)) {
                const std::size_t width =
                    it->second.size() / std::size_t(g.experts_per_token);
                for (const auto& pr : sorted_pairs(dag[i].layer, 0)) {
                    Vec got(width);
                    for (std::size_t e = 0; e < width; ++e) {
                        got[e] = from_bf16(raw[std::size_t(pr.first) * width + e]);
                    }
                    const Vec ref_slot(it->second.begin() + std::size_t(pr.second) * width,
                                       it->second.begin() + (std::size_t(pr.second) + 1) * width);
                    ++compared;
                    const float ws = rel_l2(got, ref_slot);
                    if (ws > worst) worst = ws;
                    if (ws > tol && first_bad < 0) {
                        first_bad = int(i);
                        char buf[176];
                        std::snprintf(buf, sizeof buf,
                                      "token %d slot %d at sorted %d, dispatch %d (kind %d, layer %d)",
                                      step, pr.second, pr.first, int(i), int(dag[i].kind),
                                      dag[i].layer);
                        first_bad_name = buf;
                    }
                }
                continue;
            }
            Vec got(it->second.size());
            for (std::size_t e = 0; e < got.size(); ++e) got[e] = from_bf16(raw[e]);
            ++compared;
            const float w1 = rel_l2(got, it->second);
            const bool ok = w1 <= tol;
            if (w1 > worst) worst = w1;
            if (!ok && std::getenv("PIE_NUM_DEBUG") != nullptr) {
                std::printf("    [dbg] tok %d disp %zu kind %d layer %d rel_l2 %.4f  got/want:",
                            step, i, int(dag[i].kind), dag[i].layer, double(w1));
                const std::size_t half = got.size() / 2;
                for (std::size_t e = 0; e < got.size() && e < 4; ++e) {
                    std::printf(" %.4f/%.4f", double(got[e]), double(it->second[e]));
                }
                std::printf("  |mid|");
                for (std::size_t e = half; e < got.size() && e < half + 4; ++e) {
                    std::printf(" %.4f/%.4f", double(got[e]), double(it->second[e]));
                }
                std::printf("\n");
            }
            if (!ok && first_bad < 0) {
                first_bad = int(i);
                char buf[160];
                std::snprintf(buf, sizeof buf, "token %d, dispatch %d (kind %d, layer %d)",
                              step, int(i), int(dag[i].kind), dag[i].layer);
                first_bad_name = buf;
            }
        }
    }

    judge();
    char msg[256];
    std::snprintf(msg, sizeof msg, "%s: %d dispatch outputs match the reference (worst rel_l2 %.4f)",
                  who, compared, double(worst));
    expect(first_bad < 0, msg);
    if (first_bad >= 0) std::printf("    first divergence: %s\n", first_bad_name.c_str());
    expect(compared > 0, std::string(who) + ": something was actually compared");
}

LlamaGeometry base_geometry() {
    LlamaGeometry g;
    // `affine_qmv_fast` needs K % 512 == 0 and N % 8 == 0, and the family
    // compiles SDPA only at d_128. Every width here is chosen for that.
    g.hidden = 512;
    g.n_layers = 2;
    g.vocab = 1024;
    g.n_q_heads = 4;
    g.n_kv_heads = 2;
    g.head_dim = 128;
    g.intermediate = 512;
    g.kv_max_ctx = 64;
    g.rope_theta = 500000.0f;
    return g;
}

}  // namespace

int main() {
    std::printf("llama_numerics_test — the decoder against an independent fp32 reference\n");

    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_FOR_TEST
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
#endif
    if (kernels_dir.empty()) {
        std::printf("  SKIP  no kernels directory\n");
        return 0;
    }
    auto ctx = RawMetalContext::create(std::size_t(3) << 30);
    if (!ctx) {
        std::printf("  SKIP  no Metal device\n");
        return 0;
    }

    // 2% relative L2. The GPU reduces K in a different order from the
    // reference and rounds to bf16 at every dispatch, so exact equality is not
    // available -- two layers of that lands near 1%. Everything this test is
    // for is O(1): a rope that does not rotate, attention on the wrong KV head,
    // three quarters of a residual left unwritten. There is no error mode
    // between the two, which is why the threshold does not need to be tuned.
    run_case("llama-3 (dense, tied)", base_geometry(), *ctx, kernels_dir, 0.06f);

    LlamaGeometry untied = base_geometry();
    untied.tied_embeddings = false;
    run_case("untied head", untied, *ctx, kernels_dir, 0.06f);

    LlamaGeometry qwen3 = base_geometry();
    qwen3.qk_norm = true;
    run_case("qwen3 (dense, qk-norm)", qwen3, *ctx, kernels_dir, 0.06f);

    LlamaGeometry moe = base_geometry();
    moe.qk_norm = true;
    moe.n_experts = 8;
    moe.experts_per_token = 2;
    moe.moe_intermediate = 512;
    run_case("qwen3-moe (routed)", moe, *ctx, kernels_dir, 0.06f);

    // More experts than a simdgroup has lanes.
    //
    // The sort's prefix over the experts is a TWO-LEVEL scan -- a simd scan
    // within each simdgroup, then a residual scan over the simdgroup totals --
    // and with eight experts the second level is one entry whose offset is
    // always zero. Every routed case above ran the scan with its second half
    // inert, so dropping the simdgroup offset entirely broke nothing and
    // nothing said so.
    //
    // Forty, not sixty-four: it puts eight live lanes in the second simdgroup
    // and leaves the rest dead, so the partial group and the last-lane write
    // are exercised too rather than only the aligned case.
    LlamaGeometry moe_wide = moe;
    moe_wide.n_experts = 40;
    moe_wide.experts_per_token = 4;
    run_case("qwen3-moe (40 experts: two simdgroups of prefix)", moe_wide, *ctx, kernels_dir,
             0.06f, /*rows=*/4, /*paged=*/true);

    // ── the paged ABI ──
    //
    // The same arithmetic through a page table. Run at one row first, so that
    // a failure here means the paged attention and NOT the batching: the two
    // changed together and are the only two things that could have.
    run_case("llama-3 (dense, paged)", base_geometry(), *ctx, kernels_dir, 0.06f,
             /*rows=*/1, /*paged=*/true);
    run_case("qwen3-moe (routed, paged)", moe, *ctx, kernels_dir, 0.06f, /*rows=*/1,
             /*paged=*/true);

    // ── the batched path ──
    //
    // Four tokens as ONE fire, checked row by row against the four decodes the
    // reference runs. This is the whole claim M>1 makes: row r must compute
    // what the r-th decode computes, because causal attention gives row r keys
    // 0..r and nothing more. Checked per dispatch, so a divergence names the
    // kernel rather than showing up as different text.
    run_case("llama-3 (dense, 4 rows in one fire)", base_geometry(), *ctx, kernels_dir, 0.06f,
             /*rows=*/4, /*paged=*/true);
    run_case("qwen3-moe (routed, 4 rows in one fire)", moe, *ctx, kernels_dir, 0.06f,
             /*rows=*/4, /*paged=*/true);

    // ── the GEMM ──
    //
    // Four rows is below `kQmmMinBatch`, so every projection above is still a
    // matvec and the batched cases prove nothing about `affine_qmm_t`. Sixteen
    // rows fills a tile, which switches the dense projections onto a different
    // KERNEL -- not a wider launch of the same one -- and is the only thing
    // here that exercises it.
    // The control. Same sixteen tokens, same weights, but decoded ONE AT A TIME
    // on the matvec path that the four-token cases already proved. Whatever
    // rel_l2 this reports is the cost of the sequence alone -- attention at
    // position 15 sums sixteen bf16 terms where position 3 summed four -- and
    // is the floor the batched cases below have to be read against. Without it
    // a tolerance wide enough for sixteen tokens looks like a tolerance widened
    // to hide the GEMM.
    run_case("llama-3 (dense, 16 sequential decodes: the control)", base_geometry(), *ctx,
             kernels_dir, 0.12f, /*rows=*/1, /*paged=*/true, /*steps=*/16);
    run_case("llama-3 (dense, 16 rows: the GEMM)", base_geometry(), *ctx, kernels_dir, 0.12f,
             /*rows=*/16, /*paged=*/true);
    run_case("qwen3-moe (routed, 16 rows: GEMM for the dense projections)", moe, *ctx,
             kernels_dir, 0.12f, /*rows=*/16, /*paged=*/true);

    // The batched mixture itself. `moe_should_batch` wants an expert's run to
    // hold `moe_batch_min_per_expert` rows, which for eight experts at two
    // slots a token is `8 * min_per / 2` rows -- sixteen at the measured four.
    // At forty-eight the sort pads, the three routed projections become
    // `affine_qmm_t_routed`, and this is the ONLY case that runs them.
    //
    // Stated as an expectation rather than assumed, because the threshold is a
    // TUNING answer and a change to either side of it would silently move this
    // case onto the path the other five already cover -- which is what happened
    // when `min_per` went from eight to four: sixteen rows crossed over, and
    // this line is why that was visible rather than discovered later.
    //
    // What is pinned is that both paths are still reached, not where the line
    // between them sits: the row counts here are read off the current
    // threshold rather than being constants of their own.
    const int min_per = pie::metal::moe_batch_min_per_expert();
    const int batches_at = moe.n_experts * min_per / moe.experts_per_token;
    expect(llama_moe_tile_rows(moe, batches_at) > 1,
           "a run that reaches the threshold takes the batched path");
    expect(llama_moe_tile_rows(moe, batches_at - 1) == 1, "one row short does not");
    run_case("qwen3-moe (routed, 48 rows: the batched mixture)", moe, *ctx, kernels_dir, 0.12f,
             /*rows=*/48, /*paged=*/true);

    // The same eight rows as the two-request case below, but as ONE request.
    // It is what separates "eight rows is wrong" from "two requests is wrong",
    // and neither answer is guessable from the other.
    run_case("qwen3-moe (routed, 8 rows, one request)", moe, *ctx, kernels_dir, 0.06f,
             /*rows=*/8, /*paged=*/true);

    // ── two requests, one fire ──
    //
    // The last untested dimension: everything above serves ONE sequence, so
    // nothing had ever made two of them share a fire while attending different
    // pages from their own positions.
    run_case("llama-3 (dense, 2 requests x 4 rows)", base_geometry(), *ctx, kernels_dir, 0.06f,
             /*rows=*/8, /*paged=*/true, /*steps=*/0, /*requests=*/2);
    run_case("qwen3-moe (routed, 2 requests x 4 rows)", moe, *ctx, kernels_dir, 0.06f,
             /*rows=*/8, /*paged=*/true, /*steps=*/0, /*requests=*/2);
    run_case("llama-3 (dense, 2 requests x 8 rows: the GEMM)", base_geometry(), *ctx,
             kernels_dir, 0.12f, /*rows=*/16, /*paged=*/true, /*steps=*/0, /*requests=*/2);

    // ── the tiled attention ──
    //
    // At 32 rows or more the paged attention stops giving a query row a whole
    // threadgroup and gives it a simdgroup, so that thirty-two rows can share
    // one staged block of K/V. Everything above runs under 32 rows and so had
    // never launched it at all.
    //
    // Forty rows over two requests, deliberately: the request boundary falls
    // INSIDE the first tile, which is the case the kernel handles by walking
    // the runs of equal `req_of_token` rather than assuming a tile is one
    // request. Assuming it fails here and nowhere else.
    //
    // The tile's OTHER edge -- the phantom rows of the rounded-up grid, which
    // the kernel retires by reading N -- is not checked here and cannot be.
    // Deleting that guard makes those rows write past the attention output's
    // extent, and a write past an extent is not a number this test can read;
    // every row count tried passes with the guard removed. What pins it is a
    // shape assertion in llama_decode_step_test: the tiled grid is taller than
    // N, so the kernel must be told N.
    //
    // The routed model comes along, because two requests and a mixture is the
    // one combination the batched sort had never been fired at.
    run_case("llama-3 (dense, 40 rows over 2 requests: the tiled attention)", base_geometry(),
             *ctx, kernels_dir, 0.12f, /*rows=*/40, /*paged=*/true, /*steps=*/0,
             /*requests=*/2);
    run_case("qwen3-moe (routed, 40 rows over 2 requests: the tiled attention)", moe, *ctx,
             kernels_dir, 0.12f, /*rows=*/40, /*paged=*/true, /*steps=*/0, /*requests=*/2);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
