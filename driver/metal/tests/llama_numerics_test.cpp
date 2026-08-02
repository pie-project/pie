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

    /// Run one token, recording every dispatch's output by DAG position.
    Trace step(const std::vector<Dispatch>& dag, int token, int position) {
        Trace t;
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

void run_case(const char* who, LlamaGeometry g, RawMetalContext& ctx,
              const std::string& kernels_dir, float tol) {
    std::printf("\n-- %s --\n", who);
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
    const std::vector<std::size_t> elems = llama_pool_elems(dag, plan, col, g);
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
    for (int i = 0; i < kIoSlotCount; ++i) b.io[std::size_t(i)] = ctx.heap_alloc(4096);
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
    std::string err;
    if (!build_llama_psos(ctx, kernels_dir, g, ll, &err) ||
        !load_decode_psos(ctx, kernels_dir, base, /*with_argmax=*/false, &err)) {
        expect(false, std::string(who) + ": pipelines compiled (" + err + ")");
        return;
    }
    bind_llama_consts(ctx, dag, g, /*rows=*/1, /*paged=*/false);
    try {
        bind_llama_dag(ctx, b, dag, g, col);
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

    const std::vector<int> tokens = {7, 130, 42, 901};
    int first_bad = -1;
    std::string first_bad_name;
    float worst = 0.0f;
    int compared = 0;

    for (int step = 0; step < int(tokens.size()); ++step) {
        write_u32(b.io[std::size_t(IoSlot::TokenId)], std::uint32_t(tokens[std::size_t(step)]));
        write_u32(b.io[std::size_t(IoSlot::Position)], std::uint32_t(step));
        write_u32(b.io[std::size_t(IoSlot::SeqLen)], std::uint32_t(step) + 1u);
        write_u32(b.io[std::size_t(IoSlot::SampleRows)], 0u);
        ctx.run_step([&](StepEncoder& se) { encode_llama_step(se, dag, g, base, ll); });

        const Trace want = ref.step(dag, tokens[std::size_t(step)], step);
        if (g.is_moe() && std::getenv("PIE_NUM_DEBUG") != nullptr && plan.expert_ids_value >= 0) {
            const auto& ids_slot =
                b.pool[std::size_t(col.color_of_value[std::size_t(plan.expert_ids_value)])];
            const auto& w_slot =
                b.pool[std::size_t(col.color_of_value[std::size_t(plan.expert_weights_value)])];
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

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
