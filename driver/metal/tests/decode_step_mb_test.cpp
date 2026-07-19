#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_set>

#include "batch_schedule.hpp"
#include "decode_consts.hpp"
#include "decode_step_mb.hpp"
#include "heap_bind.hpp"
#include "scratch.hpp"

using namespace pie::metal;

namespace {
int pass = 0, fail = 0;
void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    ok ? ++pass : ++fail;
}

int required_slots(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather: return 6;
        case Kernel::Rms:
        case Kernel::FfnRms:
        case Kernel::QNorm:
        case Kernel::KNorm:
        case Kernel::FinalRms: return 4;
        case Kernel::QmvIn: case Kernel::QmvInZ: case Kernel::QmvOut:
        case Kernel::QmvQ: case Kernel::QmvK: case Kernel::QmvV: case Kernel::QmvO:
        case Kernel::QmvGate: case Kernel::QmvUp: case Kernel::QmvDown:
        case Kernel::QmvLmHead: return 7;
        case Kernel::GdnInA: case Kernel::GdnInB: return 5;
        case Kernel::GdnPrepSlotted: return 14;
        case Kernel::GdnCoreSlotted: return 12;
        case Kernel::GatedRms: return 5;
        case Kernel::Residual:
        case Kernel::LayerOut:
        case Kernel::SiluMul:
        case Kernel::QSplit: return 4;
        case Kernel::Rope:
        case Kernel::RopeK: return 5;
        case Kernel::KvAppendPaged: return 15;
        case Kernel::SdpaPaged: return 12;
        case Kernel::AttnGate: return 3;
        default: return 0;
    }
}
}  // namespace

int main() {
    std::printf("[decode_step_mb]\n");
    DecodeGeometry g;
    g.paged_kv_enabled = true;
    g.max_tokens = 4;
    g.max_requests = 4;
    g.max_slots = 4;
    g.total_pages = 8;
    g.kv_page_size = 32;
    expect(
        paged_attention_mask_pitch_bytes(g) == 256 &&
            paged_pool_size_supported(g, 3) &&
            paged_pool_size_supported(g, 8) &&
            !paged_pool_size_supported(g, 9),
        "masked prefill keeps the fixed allocated 8-page pitch and rejects "
        "unsupported growth");

    const auto m1 = build_decode_dag(g, false, false, true);
    const auto mb1 = build_decode_dag_mb(g, 1);
    const auto mb4 = build_decode_dag_mb(g, 4);
    const auto prefill = build_decode_prefill_dags(g, 4);
    expect(m1.size() < size_t(kPrefillOrdinalStride),
           "prefill argument-table ordinal ranges cannot overlap");
    expect(m1.size() == mb1.size() && m1.size() == mb4.size(),
           "M=1 and N=4 paged DAGs preserve every canonical dispatch");
    bool mapped = true;
    for (size_t i = 0; i < m1.size(); ++i) {
        if (mb1[i].ordinal != int(i) + kMultiBatchOrdinalBase ||
            mb1[i].layer != m1[i].layer) mapped = false;
    }
    expect(mapped, "paged ordinals are disjoint from sealed M=1 argument tables");
    bool saw_mb_grid = false, saw_paged = false, saw_slotted = false;
    for (const Dispatch& d : mb4) {
        if (d.kind == Kernel::EmbedGather && d.grid.y == 4) saw_mb_grid = true;
        if (d.kind == Kernel::KvAppendPaged && d.grid.z == 4) saw_paged = true;
        if (d.kind == Kernel::GdnCoreSlotted && d.grid.z == 4u * g.gdn_v_heads)
            saw_slotted = true;
    }
    expect(saw_mb_grid && saw_paged && saw_slotted,
           "row/page/state launch grids scale with N while M=1 remains structurally equivalent");
    bool stream_is_full_and_ordered = prefill.size() == 4;
    for (size_t t = 0; t < prefill.size(); ++t) {
        stream_is_full_and_ordered &= prefill[t].size() == m1.size();
        stream_is_full_and_ordered &=
            prefill[t].front().ordinal == kPrefillOrdinalBase + int(t) * kPrefillOrdinalStride;
        stream_is_full_and_ordered &= prefill[t].front().grid.y == 1;
    }
    expect(stream_is_full_and_ordered,
           "N=4 prefill owns four complete N=1 DAGs in stable causal command-stream order");

    const uint32_t toks[] = {1, 2, 3, 4, 5};
    const uint32_t qo[] = {0, 3, 5};
    const uint32_t pi[] = {0, 1, 2};
    const uint32_t pages[] = {0, 1};
    const uint32_t last[] = {3, 2};
    const uint32_t slots[] = {0, 1};
    const uint8_t flags[] = {1, 1};
    const BatchSchedule prompt_schedule =
        build_batch_schedule(toks, 5, qo, pi, last, slots, flags, 3, 32);
    expect(!prompt_schedule.is_pure_decode &&
               prompt_schedule.req_of_token == std::vector<uint32_t>({0, 0, 0, 1, 1}) &&
               prompt_schedule.slot_of_token == std::vector<uint32_t>({0, 0, 0, 1, 1}),
           "two prompt requests (3,2) preserve request-major/per-slot token order");

    auto ctx = RawMetalContext::create(32u << 20);
    expect(ctx != nullptr, "RawMetalContext created for binding coverage");
    if (!ctx) return 1;
    BoundDecode b;
    b.gdn.resize(g.n_layers);
    std::unordered_set<std::string> weight_names;
    for (const auto& dispatch : mb4) {
        for (const auto& binding :
             weight_binds(dispatch.kind, dispatch.layer, g, true)) {
            weight_names.insert(binding.tensor);
        }
    }
    for (const std::string& name : weight_names) {
        b.weights.emplace(name, ctx->heap_alloc(256));
    }
    for (int i = 0; i < kIoSlotCount; ++i) b.io[i] = ctx->heap_alloc(4096);
    for (int l = 0; l < g.n_layers; ++l) {
        if (!DecodeGeometry::is_full_attn(l)) {
            b.gdn[l].conv_state = ctx->heap_alloc(4096);
            b.gdn[l].conv_state_out = ctx->heap_alloc(4096);
            b.gdn[l].recurrent_state = ctx->heap_alloc(4096);
            b.gdn[l].conv_bias_zero = ctx->heap_alloc(4096);
        }
    }
    std::vector<SlotHandle> kp(size_t(g.n_layers)), vp(size_t(g.n_layers));
    for (int l = 0; l < g.n_layers; ++l) if (DecodeGeometry::is_full_attn(l)) {
        kp[l] = ctx->heap_alloc(4096); vp[l] = ctx->heap_alloc(4096);
    }
    bind_decode_dag_mb(*ctx, b, mb4, g, kp, vp, true);
    const ScratchSchedule sched = build_scratch_schedule(mb4, g);
    const auto prefill_sched = build_scratch_schedule(prefill.front(), g);
    std::vector<SlotHandle> scratch(size_t(std::max(sched.colors_used, prefill_sched.colors_used)));
    for (auto& s : scratch) s = ctx->heap_alloc(4096);
    bind_scratch(*ctx, mb4, sched, scratch.data(), int(scratch.size()));
    bind_decode_consts(*ctx, mb4, g, 4096, true);

    const size_t scratch_row = size_t(scratch_widest_elems(g)) * 2;
    const size_t logits_row = size_t(g.vocab) * 2;
    for (size_t t = 0; t < prefill.size(); ++t) {
        const MbBindOffsets offsets{.token_row = t,
                                    .logits_bytes = t * logits_row};
        bind_decode_dag_mb(*ctx, b, prefill[t], g, kp, vp, true, offsets);
        bind_scratch(*ctx, prefill[t], prefill_sched, scratch.data(), int(scratch.size()),
                     t * scratch_row);
        bind_decode_consts(*ctx, prefill[t], g, 4096, true);
        bind_prefill_gdn_state(*ctx, b, prefill[t], uint32_t(t & 1), (t & 1) == 0);
    }

    bool all_bound = true;
    for (const Dispatch& d : mb4) {
        const int n = required_slots(d.kind);
        for (int slot = 0; slot < n; ++slot) {
            if (!ctx->arg_slot_is_bound(d.ordinal, uint8_t(slot))) {
                std::printf("    missing ord=%d kind=%d slot=%d\n",
                            d.ordinal, int(d.kind), slot);
                all_bound = false;
            }
        }
        for (const auto& token_dag : prefill) {
            for (const Dispatch& d : token_dag) {
                const int n = required_slots(d.kind);
                for (int slot = 0; slot < n; ++slot)
                    all_bound &= ctx->arg_slot_is_bound(d.ordinal, uint8_t(slot));
            }
        }
    }
    expect(all_bound, "every required multi-batch argument-table slot is bound");

    const auto& row1 = prefill[1];
    const auto find_kind = [&](Kernel k) -> const Dispatch* {
        for (const Dispatch& d : row1) if (d.kind == k) return &d;
        return nullptr;
    };
    const Dispatch* embed = find_kind(Kernel::EmbedGather);
    const Dispatch* rope = find_kind(Kernel::Rope);
    const Dispatch* append = find_kind(Kernel::KvAppendPaged);
    const Dispatch* sdpa = find_kind(Kernel::SdpaPaged);
    const Dispatch* lm = find_kind(Kernel::QmvLmHead);
    const Dispatch* qmv = find_kind(Kernel::QmvQ);
    bool row_offsets = embed && rope && append && sdpa && lm && qmv;
    if (row_offsets) {
        row_offsets &=
            ctx->arg_slot_address(embed->ordinal, uint8_t(bind::Embed::TokenId)) ==
            b.io[int(IoSlot::TokenId)].gpu_address + 4;
        row_offsets &=
            ctx->arg_slot_address(rope->ordinal, uint8_t(bind::Rope::Position)) ==
            b.io[int(IoSlot::Position)].gpu_address + 4;
        row_offsets &=
            ctx->arg_slot_address(append->ordinal, uint8_t(bind::KvAppendPaged::WPage)) ==
            b.io[int(IoSlot::WPage)].gpu_address + 4;
        row_offsets &=
            ctx->arg_slot_address(sdpa->ordinal, uint8_t(bind::SdpaPaged::AttnMask)) ==
            b.io[int(IoSlot::AttnMask)].gpu_address +
                paged_attention_mask_pitch_bytes(g);
        row_offsets &=
            ctx->arg_slot_address(lm->ordinal, uint8_t(bind::Qmv::Out)) ==
            b.io[int(IoSlot::Logits)].gpu_address + logits_row;
        int qmv_index = -1;
        for (size_t i = 0; i < row1.size(); ++i)
            if (row1[i].ordinal == qmv->ordinal) qmv_index = int(i);
        int buffer_id = -1;
        for (const ScratchBind& sb : prefill_sched.per_dispatch[size_t(qmv_index)].binds)
            if (sb.bind_index == uint8_t(bind::Qmv::X)) buffer_id = sb.buffer_id;
        row_offsets &= buffer_id >= 0 &&
            ctx->arg_slot_address(qmv->ordinal, uint8_t(bind::Qmv::X)) ==
            scratch[size_t(buffer_id)].gpu_address + scratch_row;
    }
    expect(row_offsets,
           "per-token prefill tables address the matching IO, scratch, explicit-write, and logits rows");
    std::printf("\n==== decode_step_mb_test: %d passed, %d failed ====\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
