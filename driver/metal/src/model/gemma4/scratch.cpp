// Gemma 4's activation dataflow.
//
// The same job `batch/scratch.cpp` does for qwen3.5: walk the DAG in order and
// say, for every dispatch, which activation value each of its buffers reads or
// writes. Colouring those live ranges onto a pool is the shared half and is
// reused; naming the values is the family's, and gemma4's differ in three ways:
//
//  * **A norm sandwich.** Attention and the FFN are each wrapped: normalise the
//    input, run the block, normalise the OUTPUT, then add the residual. So a
//    block's result is normalised before it rejoins the stream, which qwen3.5
//    never does.
//  * **Per-Layer Embeddings.** A second stream, computed once before the stack
//    and consumed a slice at a time, which no other family here has.
//  * **KV-shared layers read pages they did not write.** Those layers emit no
//    k/v projection at all, so the dataflow simply has nothing to thread.

#include "scratch.hpp"

#include "decode_step.hpp"

namespace pie::metal::gemma4 {

namespace bi {
// The bind indices this dataflow touches, named so the walk below reads as
// dataflow rather than as arithmetic.
constexpr std::uint8_t RmsX = 0, RmsOut = 2;
constexpr std::uint8_t QmvX = 3, QmvOut = 4;
constexpr std::uint8_t EmbedOut = 4;
constexpr std::uint8_t RopeX = 0;
constexpr std::uint8_t SdpaQ = 0, SdpaOut = 3;
constexpr std::uint8_t KvAppendK = 0, KvAppendV = 1;
constexpr std::uint8_t GegluGate = 0, GegluUp = 1, GegluOut = 2;
constexpr std::uint8_t ResidX = 0, ResidR = 1, ResidOut = 2;
// The fused norm+residual keeps bind::Rms's prefix and appends the residual.
constexpr std::uint8_t RRX = 0, RROut = 2, RRResid = 4;
constexpr std::uint8_t VNormX = 0, VNormOut = 1;
constexpr std::uint8_t ScalarX = 0, ScalarOut = 2;
constexpr std::uint8_t CombineProj = 0, CombineToken = 1, CombineOut = 2;
constexpr std::uint8_t SoftcapIn = 0, SoftcapOut = 1;
}  // namespace bi

ScratchPlan build_gemma4_scratch(const std::vector<Dispatch>& dag, const Gemma4Geometry& g) {
    ScratchPlan plan;
    int next_value = 0;
    auto fresh = [&] { return next_value++; };
    auto rd = [&](int ord, std::uint8_t b, int v) { plan.uses.push_back({ord, b, v, false}); };
    auto wr = [&](int ord, std::uint8_t b, int v) { plan.uses.push_back({ord, b, v, true}); };

    // The residual stream, and the temporaries that hang off it.
    int resid = -1;    // the stream itself
    int normed = -1;   // the norm output currently feeding projections
    int q = -1, kk = -1, vv = -1, attn = -1;
    int gp = -1, up = -1, act = -1;
    int block = -1;    // a block's output, between its post-norm and its residual add
    // PLE: one table for the whole stack, plus this layer's slice.
    int ple_tok = -1, ple_proj = -1, ple = -1;
    int ple_gate = -1, ple_act = -1, ple_back = -1;

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        // Position, not `d.ordinal`: the prefill DAG shifts its ordinals clear
        // of the decode path's, and this is a time axis.
        const int o = static_cast<int>(di);
        switch (d.kind) {
            case Kind::EmbedGather:
                resid = fresh();
                wr(o, bi::EmbedOut, resid);
                break;

            // ── PLE precompute, once ──
            case Kind::PleTokenGather:
                ple_tok = fresh();
                wr(o, bi::EmbedOut, ple_tok);
                break;
            case Kind::PleProjGemv:
                ple_proj = fresh();
                rd(o, bi::QmvX, resid);
                wr(o, bi::QmvOut, ple_proj);
                break;
            case Kind::PleProjNorm:
                rd(o, bi::RmsX, ple_proj);
                wr(o, bi::RmsOut, ple_proj);
                break;
            case Kind::PleCombine:
                ple = fresh();
                rd(o, bi::CombineProj, ple_proj);
                rd(o, bi::CombineToken, ple_tok);
                wr(o, bi::CombineOut, ple);
                break;

            // ── attention ──
            case Kind::AttnNorm:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::QmvQ:
                q = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, q);
                break;
            case Kind::QmvK:
                kk = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, kk);
                break;
            case Kind::QmvV:
                vv = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, vv);
                break;
            case Kind::QNorm:
                rd(o, bi::RmsX, q);
                wr(o, bi::RmsOut, q);
                break;
            case Kind::KNorm:
                rd(o, bi::RmsX, kk);
                wr(o, bi::RmsOut, kk);
                break;
            case Kind::VNorm:
                // Weightless, and in place: V is normalised on its way to the cache.
                rd(o, bi::VNormX, vv);
                wr(o, bi::VNormOut, vv);
                break;
            case Kind::RopeQ:
                rd(o, bi::RopeX, q);
                wr(o, bi::RopeX, q);
                break;
            case Kind::RopeK:
                rd(o, bi::RopeX, kk);
                wr(o, bi::RopeX, kk);
                break;
            case Kind::KvAppend:
                rd(o, bi::KvAppendK, kk);
                rd(o, bi::KvAppendV, vv);
                break;
            case Kind::Sdpa:
                attn = fresh();
                rd(o, bi::SdpaQ, q);
                wr(o, bi::SdpaOut, attn);
                break;
            case Kind::QmvO:
                block = fresh();
                rd(o, bi::QmvX, attn);
                wr(o, bi::QmvOut, block);
                break;
            case Kind::PostAttnResidual: {
                // The sandwich's second half and the add it always precedes:
                // normalise the BLOCK's output, then rejoin the stream.
                const int next = fresh();
                rd(o, bi::RRX, block);
                rd(o, bi::RRResid, resid);
                wr(o, bi::RROut, next);
                resid = next;
                break;
            }

            // ── FFN ──
            case Kind::FfnNorm:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::QmvGate:
                gp = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, gp);
                break;
            case Kind::QmvUp:
                up = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, up);
                break;
            case Kind::GegluTanh:
                act = fresh();
                rd(o, bi::GegluGate, gp);
                rd(o, bi::GegluUp, up);
                wr(o, bi::GegluOut, act);
                break;
            case Kind::QmvDown:
                block = fresh();
                rd(o, bi::QmvX, act);
                wr(o, bi::QmvOut, block);
                break;
            case Kind::PostFfnResidual: {
                const int next = fresh();
                rd(o, bi::RRX, block);
                rd(o, bi::RRResid, resid);
                wr(o, bi::RROut, next);
                resid = next;
                break;
            }

            // ── per-layer embedding residual ──
            case Kind::PleGateGemv:
                ple_gate = fresh();
                rd(o, bi::QmvX, resid);
                wr(o, bi::QmvOut, ple_gate);
                break;
            case Kind::PleGeglu:
                // Gated by the stream, valued by this layer's slice of the table.
                ple_act = fresh();
                rd(o, bi::GegluGate, ple_gate);
                rd(o, bi::GegluUp, ple);
                wr(o, bi::GegluOut, ple_act);
                break;
            case Kind::PleProjLayerGemv:
                ple_back = fresh();
                rd(o, bi::QmvX, ple_act);
                wr(o, bi::QmvOut, ple_back);
                break;
            case Kind::PleResidualScaled: {
                const int next = fresh();
                rd(o, bi::RRX, ple_back);
                rd(o, bi::RRResid, resid);
                wr(o, bi::RROut, next);
                resid = next;
                break;
            }
            case Kind::LayerScalar: {
                const int next = fresh();
                rd(o, bi::ScalarX, resid);
                wr(o, bi::ScalarOut, next);
                resid = next;
                break;
            }

            // ── tail ──
            case Kind::FinalRms:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::LmHead: {
                const int logits = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, logits);
                plan.logits_value = logits;
                break;
            }
            case Kind::FinalSoftcap:
                rd(o, bi::SoftcapIn, plan.logits_value);
                wr(o, bi::SoftcapOut, plan.logits_value);
                break;
            case Kind::Argmax:
                rd(o, 0, plan.logits_value);
                break;
        }
    }
    plan.value_count = next_value;
    (void)g;
    return plan;
}

}  // namespace pie::metal::gemma4
