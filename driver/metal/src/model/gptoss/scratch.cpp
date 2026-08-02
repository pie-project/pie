// GPT-OSS's activation dataflow.
//
// The same job `batch/scratch.cpp` does for qwen3.5: walk the DAG in order and
// say, for every dispatch, which activation value each of its buffers reads or
// writes. Colouring those live ranges onto a pool is the shared half and is
// reused; naming the values is the family's, and gpt-oss's differ in one way
// that matters:
//
//   **The MoE's operands are a k-wide stack.** `ExpertGate`/`ExpertUp`/
//   `ExpertDown` each write `experts_per_token` rows -- one per selected expert
//   -- into a single value, and `ExpertCombine` reduces the stack to one row.
//   So the widths in the pool are not all `hidden`: the expert stack is
//   `k * intermediate`, which is what sizes the pool.
//
// Everything else is the plainest of the three families: one norm before
// attention, one before the MLP, a residual add after each, and no per-layer
// side stream.

#include "scratch.hpp"

#include "decode_step.hpp"

namespace pie::metal::gptoss {

namespace bi {
// The bind indices this dataflow touches, named so the walk reads as dataflow
// rather than as arithmetic.
constexpr std::uint8_t RmsX = 0, RmsOut = 2;
constexpr std::uint8_t QmvX = 3, QmvOut = 4;
constexpr std::uint8_t EmbedOut = 4;
constexpr std::uint8_t RopeX = 0;
constexpr std::uint8_t SdpaQ = 0, SdpaOut = 3;
constexpr std::uint8_t KvAppendK = 0, KvAppendV = 1;
constexpr std::uint8_t ResidX = 0, ResidR = 1, ResidOut = 2;
constexpr std::uint8_t TopkLogits = 0, TopkIds = 1, TopkWeights = 2;
constexpr std::uint8_t SwiGate = 0, SwiUp = 1, SwiOut = 2;
constexpr std::uint8_t CombineY = 0, CombineW = 1, CombineOut = 2;
constexpr std::uint8_t QmvExpertIds = 8;
}  // namespace bi

ScratchPlan build_gptoss_scratch(const std::vector<Dispatch>& dag, const GptOssGeometry& g) {
    ScratchPlan plan;
    int next_value = 0;
    auto fresh = [&] { return next_value++; };
    auto rd = [&](int i, std::uint8_t b, int v) { plan.uses.push_back({i, b, v, false}); };
    auto wr = [&](int i, std::uint8_t b, int v) { plan.uses.push_back({i, b, v, true}); };

    int resid = -1;   // the residual stream
    int normed = -1;  // the norm output currently feeding projections
    int q = -1, kk = -1, vv = -1, attn = -1, block = -1;
    // The routing decision, and the k-wide expert stack it drives.
    int router_logits = -1, expert_ids = -1, expert_w = -1;
    int gate = -1, up = -1, act = -1, expert_out = -1;

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int o = static_cast<int>(di);
        switch (d.kind) {
            case Kind::EmbedGather:
                resid = fresh();
                wr(o, bi::EmbedOut, resid);
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
            case Kind::SdpaSink:
                attn = fresh();
                rd(o, bi::SdpaQ, q);
                wr(o, bi::SdpaOut, attn);
                break;
            case Kind::QmvO:
                block = fresh();
                rd(o, bi::QmvX, attn);
                wr(o, bi::QmvOut, block);
                break;
            case Kind::AttnResidual: {
                const int next = fresh();
                rd(o, bi::ResidX, block);
                rd(o, bi::ResidR, resid);
                wr(o, bi::ResidOut, next);
                resid = next;
                break;
            }

            // ── the sparse MoE ──
            case Kind::FfnNorm:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::RouterGemv:
                router_logits = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, router_logits);
                break;
            case Kind::RouterTopK:
                // Two outputs, and both are read by something else: the ids by
                // all three routed matvecs, the weights by the combine.
                expert_ids = fresh();
                plan.expert_ids_by_layer.push_back(expert_ids);
                expert_w = fresh();
                rd(o, bi::TopkLogits, router_logits);
                wr(o, bi::TopkIds, expert_ids);
                wr(o, bi::TopkWeights, expert_w);
                break;
            case Kind::ExpertGate:
                gate = fresh();
                rd(o, bi::QmvX, normed);
                rd(o, bi::QmvExpertIds, expert_ids);
                wr(o, bi::QmvOut, gate);
                break;
            case Kind::ExpertUp:
                up = fresh();
                rd(o, bi::QmvX, normed);
                rd(o, bi::QmvExpertIds, expert_ids);
                wr(o, bi::QmvOut, up);
                break;
            case Kind::ExpertSwiGlu:
                act = fresh();
                rd(o, bi::SwiGate, gate);
                rd(o, bi::SwiUp, up);
                wr(o, bi::SwiOut, act);
                break;
            case Kind::ExpertDown:
                expert_out = fresh();
                rd(o, bi::QmvX, act);
                rd(o, bi::QmvExpertIds, expert_ids);
                wr(o, bi::QmvOut, expert_out);
                break;
            case Kind::ExpertCombine:
                block = fresh();
                rd(o, bi::CombineY, expert_out);
                rd(o, bi::CombineW, expert_w);
                wr(o, bi::CombineOut, block);
                break;
            case Kind::FfnResidual: {
                const int next = fresh();
                rd(o, bi::ResidX, block);
                rd(o, bi::ResidR, resid);
                wr(o, bi::ResidOut, next);
                resid = next;
                break;
            }

            // ── tail ──
            // The sampled rows, compacted: everything after this is [S, *]. It
            // writes a value of its own rather than in place, because its input
            // is [N, hidden] and its output is [S, hidden] -- aliasing them
            // would make the pool's slot sizing a lie.
            case Kind::RowGather: {
                const int gathered = fresh();
                rd(o, 0, resid);
                wr(o, 1, gathered);
                resid = gathered;
                break;
            }
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
            case Kind::Argmax:
                rd(o, 0, plan.logits_value);
                break;
        }
    }
    plan.value_count = next_value;
    (void)g;
    return plan;
}

}  // namespace pie::metal::gptoss
