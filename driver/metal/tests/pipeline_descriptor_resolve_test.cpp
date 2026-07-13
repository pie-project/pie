// PTIR descriptor-resolver pure contract tests (metal_ptir_plan.md Phase 2,
// G2.1-style gates). Pure host, no Metal/Apple/checkpoint dependency —
// always builds and runs. Hand-builds minimal `Trace`/`InterpInstance`
// objects directly (no container/sidecar encoding needed — the resolver
// operates purely on the decoded `Trace` + live `ChannelState`).

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/descriptor_resolve.hpp"

using namespace pie::metal::pipeline;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// A minimal Trace: N channels (capacity 1, otherwise default), the given
// port bindings, no stages — sufficient for the resolver, which only reads
// `trace.ports` and the live channel state.
Trace make_trace(int n_channels, std::vector<cptir::PortBinding> ports) {
    Trace t;
    t.channels.resize(static_cast<std::size_t>(n_channels));
    for (int i = 0; i < n_channels; ++i) {
        t.channels[static_cast<std::size_t>(i)].id = static_cast<cptir::ChannelId>(i);
        t.channels[static_cast<std::size_t>(i)].capacity = 1;
    }
    t.ports = std::move(ports);
    return t;
}

InterpInstance make_inst(int n_channels) {
    InterpInstance inst;
    for (int i = 0; i < n_channels; ++i) {
        inst.channels.push_back(make_host_channel_state(DType::U32, 1, 1));
    }
    return inst;
}

void put(InterpInstance& inst, std::size_t channel, Value value) {
    inst.channels[channel] =
        make_host_channel_state(value.dtype, value.len(), 1);
    (void)inst.channels[channel]->push(std::move(value));
}

}  // namespace

int main() {
    std::printf("[is_device_geometry_trace]\n");

    // Ordinary structural channels alone do not trigger device geometry.
    {
        Trace t = make_trace(1, {{kPortPositions, 0, false}});
        expect(!is_device_geometry_trace(t), "Positions alone => runtime-owned geometry");
    }
    {
        Trace t = make_trace(1, {{kPortPages, 0, false}});
        expect(!is_device_geometry_trace(t), "Pages alone => runtime-owned geometry");
    }
    {
        Trace t = make_trace(1, {{kPortWSlot, 0, false}});
        expect(!is_device_geometry_trace(t), "WSlot alone => incomplete device geometry");
    }
    {
        Trace t = make_trace(1, {{kPortAttnMask, 0, false}});
        expect(!is_device_geometry_trace(t), "AttnMask alone => runtime-owned geometry");
    }

    // Runtime contract: WSlot/WOff plus channel-bound [B,P] Pages with P>1.
    {
        Trace t = make_trace(3, {{kPortPages, 0, false},
                                 {kPortWSlot, 1, false},
                                 {kPortWOff, 2, false}});
        t.channels[0].type.shape.dims = {2, 3};
        expect(is_device_geometry_trace(t), "write descriptors + Pages[B,P>1] => device geometry");
    }
    {
        Trace t = make_trace(2, {{kPortPages, 0, false}, {kPortWSlot, 1, false}});
        t.channels[0].type.shape.dims = {2, 1};
        expect(!is_device_geometry_trace(t), "Pages[B,1] => ordinary geometry");
    }
    // The canonical (runtime-owned) shape — token embed + KvLen-only
    // attention, plus a Readout channel — is NOT device geometry.
    {
        Trace t = make_trace(3, {{kPortEmbedTokens, 0, false},
                                {kPortKvLen, 1, false},
                                {kPortReadout, 2, false}});
        expect(!is_device_geometry_trace(t),
              "EmbedTokens/KvLen/Readout alone => NOT device geometry");
    }
    // A CONST-bound structure port does not count (host-prefilled on the wire).
    {
        Trace t = make_trace(1, {{kPortPositions, 0, true}});
        expect(!is_device_geometry_trace(t), "const Positions port does not trigger device geometry");
    }

    std::printf("[resolve_fire_geometry]\n");

    // Not-ready (W1.6): a channel-bound structure port whose ring is empty
    // fails the resolve — no dummy-run.
    {
        Trace t = make_trace(1, {{kPortPositions, 0, false}});
        InterpInstance inst = make_inst(1);
        cptir::FireGeometry fg;
        std::string err;
        const GeometryResolveResult typed =
            resolve_fire_geometry_typed(t, inst, 4, fg, &err);
        expect(
            typed.status == GeometryResolveStatus::NotReady &&
                typed.channel == 0 && contains(err, "not yet produced") &&
                !contains(err, "failed"),
            "not-ready classification is typed");
        err.clear();
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(!ok && contains(err, "not ready"), "not-ready positions channel fails resolve (" + err + ")");
    }

    // A present but malformed descriptor is permanently failed, never retryable.
    {
        Trace t = make_trace(1, {{kPortPositions, 0, false}});
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::f32({1.0f}));
        cptir::FireGeometry fg;
        std::string err;
        const GeometryResolveResult typed =
            resolve_fire_geometry_typed(t, inst, 4, fg, &err);
        expect(
            typed.status == GeometryResolveStatus::Failed &&
                contains(err, "unsupported dtype"),
            "malformed descriptor classification is permanently failed");
    }

    // Take-once / peek semantics: resolving is non-destructive — the actual
    // consume happens once, later, inside step()'s own port loop.
    {
        Trace t = make_trace(1, {{kPortPositions, 0, false}});
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::u32({5, 6, 7}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds when positions ready");
        expect(fg.position_ids.size() == 3 && fg.position_ids[0] == 5 && fg.position_ids[1] == 6 &&
                  fg.position_ids[2] == 7,
              "resolved positions == [5,6,7]");
        expect(inst.channels[0]->size() == 1,
              "resolve does not consume the channel (queue still has 1 entry)");
        const Value unchanged = inst.channels[0]->front();
        expect(unchanged.u.size() == 3 && unchanged.u[0] == 5,
              "channel front value unchanged after resolve (peek only)");
    }

    // CSR-prefix trim: the fixed-shape pages port is trimmed to
    // page_indptr's last (valid-prefix) entry.
    {
        Trace t = make_trace(2, {{kPortPages, 0, false}, {kPortPageIndptr, 1, false}});
        InterpInstance inst = make_inst(2);
        put(inst, 0, Value::u32({10, 11, 99, 99}));  // fixed shape[4]
        put(inst, 1, Value::u32({0, 2}));            // prefix len 2
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for CSR-prefix trim case (" + err + ")");
        expect(fg.kv_page_indices.size() == 2 && fg.kv_page_indices[0] == 10 &&
                  fg.kv_page_indices[1] == 11,
              "kv_page_indices trimmed to the CSR prefix [10,11] (stale tail dropped)");
        expect(fg.has_kv_family, "has_kv_family set true");
    }

    // Const request CSRs from the PTIR container remain available to the
    // descriptor resolver and keep each generated request independent.
    {
        ExecPlan plan;
        plan.trace = make_trace(
            4,
            {
                {kPortEmbedTokens, 0, false},
                {kPortEmbedIndptr, 0, true},
                {kPortPositions, 1, false},
                {kPortPages, 2, false},
                {kPortPageIndptr, 0, true},
                {kPortKvLen, 3, false},
            });
        plan.const_ports = {
            {
                .port = kPortEmbedIndptr,
                .value = Value::u32({0, 1, 2}),
            },
            {
                .port = kPortPageIndptr,
                .value = Value::u32({0, 3, 6}),
            },
        };
        InterpInstance inst = make_inst(4);
        put(inst, 0, Value::i32({3, 5}));
        put(inst, 1, Value::u32({7, 7}));
        put(inst, 2, Value::u32({5, 6, 0, 5, 6, 0}));
        put(inst, 3, Value::u32({7, 7}));
        cptir::FireGeometry fg;
        std::string err;
        const auto result =
            resolve_fire_geometry_typed(plan, inst, 4, fg, &err);
        expect(
            result.status == GeometryResolveStatus::Ready &&
                fg.qo_indptr ==
                    std::vector<std::uint32_t>({0, 1, 2}) &&
                fg.kv_page_indptr ==
                    std::vector<std::uint32_t>({0, 3, 6}) &&
                fg.kv_last_page_lens ==
                    std::vector<std::uint32_t>({3, 3}) &&
                fg.sampling_indices ==
                    std::vector<std::uint32_t>({0, 0}) &&
                fg.sampling_indptr ==
                    std::vector<std::uint32_t>({0, 1, 2}),
            "const CSRs preserve two request-local geometry/readout rows (" +
                err + ")");
    }

    // KvLen -> last_page_len math: ((len-1) % page) + 1, and 0 for len==0.
    {
        Trace t = make_trace(1, {{kPortKvLen, 0, false}});
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::u32({7, 8, 0, 1}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for kv_len case");
        expect(fg.kv_last_page_lens.size() == 4, "kv_last_page_lens has 4 entries");
        expect(fg.kv_last_page_lens[0] == 3, "last_page_len(7,4) == 3");
        expect(fg.kv_last_page_lens[1] == 4, "last_page_len(8,4) == 4 (exact page boundary)");
        expect(fg.kv_last_page_lens[2] == 0, "last_page_len(0,4) == 0 (empty extent)");
        expect(fg.kv_last_page_lens[3] == 1, "last_page_len(1,4) == 1");
    }

    // Default read-out (no Readout port): the last token of each lane.
    {
        Trace t = make_trace(2, {{kPortEmbedTokens, 0, false}, {kPortEmbedIndptr, 1, false}});
        InterpInstance inst = make_inst(2);
        put(inst, 0, Value::i32({100, 101, 102, 103, 104}));  // 5 tokens
        put(inst, 1, Value::u32({0, 2, 5}));                  // 2 lanes
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for default-readout case (" + err + ")");
        expect(fg.sampling_indices.size() == 2 && fg.sampling_indices[0] == 1 &&
                  fg.sampling_indices[1] == 2,
              "default readout keeps request-local last-token indices [1,2]");
        expect(fg.sampling_indptr.size() == 3 && fg.sampling_indptr[0] == 0 &&
                  fg.sampling_indptr[1] == 1 && fg.sampling_indptr[2] == 2,
              "default sampling_indptr == [0,1,2]");
    }

    // Explicit readout port overrides the default.
    {
        Trace t = make_trace(2, {{kPortEmbedTokens, 0, false}, {kPortReadout, 1, false}});
        InterpInstance inst = make_inst(2);
        put(inst, 0, Value::i32({1, 2, 3}));
        put(inst, 1, Value::u32({0, 2}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for explicit readout case (" + err + ")");
        expect(fg.sampling_indices.size() == 2 && fg.sampling_indices[0] == 0 &&
                  fg.sampling_indices[1] == 2,
              "explicit readout == [0,2]");
        expect(fg.sampling_indptr.size() == 2 && fg.sampling_indptr[0] == 0 &&
                  fg.sampling_indptr[1] == 2,
              "explicit readout indptr == [0,2]");
    }

    // Default position_ids (no Positions port): append order 0..nnz.
    {
        Trace t = make_trace(1, {{kPortEmbedTokens, 0, false}});
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::i32({7, 7, 7}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for default-positions case");
        expect(fg.position_ids.size() == 3 && fg.position_ids[0] == 0 &&
                  fg.position_ids[1] == 1 && fg.position_ids[2] == 2,
              "default position_ids == [0,1,2]");
    }

    // Default qo_indptr (no EmbedIndptr port): one lane over all tokens.
    {
        Trace t = make_trace(1, {{kPortEmbedTokens, 0, false}});
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::i32({9, 9, 9, 9}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for default-qo_indptr case");
        expect(fg.qo_indptr.size() == 2 && fg.qo_indptr[0] == 0 && fg.qo_indptr[1] == 4,
              "default qo_indptr == [0,4] (one lane over all tokens)");
    }

    // Explicit write descriptor (w_slot/w_off) port mapping.
    {
        Trace t = make_trace(2, {{kPortWSlot, 0, false}, {kPortWOff, 1, false}});
        InterpInstance inst = make_inst(2);
        put(inst, 0, Value::u32({3, 4}));
        put(inst, 1, Value::u32({1, 2}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for w_slot/w_off case");
        expect(fg.has_write_desc, "has_write_desc set true");
        expect(fg.w_page.size() == 2 && fg.w_page[0] == 3 && fg.w_page[1] == 4, "w_page == [3,4]");
        expect(fg.w_off.size() == 2 && fg.w_off[0] == 1 && fg.w_off[1] == 2, "w_off == [1,2]");
    }

    // Dense attention mask port mapping (unpacked Bool bytes).
    {
        Trace t = make_trace(1, {{kPortAttnMask, 0, false}});
        t.values.resize(1);
        cptir::Stage stage;
        cptir::Op mask_op;
        mask_op.code = cptir::OpCode::SlidingWindowMask;
        mask_op.result_id = 0;
        mask_op.result_count = 1;
        mask_op.imm = 4;
        mask_op.imm2 = 2;
        stage.ops.push_back(mask_op);
        stage.puts.push_back({.channel = 0, .value = 0});
        t.stages.push_back(std::move(stage));
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::boolean({1, 0, 1, 1}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(ok, "resolve succeeds for attn_mask case");
        expect(fg.has_mask, "has_mask set true");
        expect(fg.mask.size() == 4 && fg.mask[0] == 1 && fg.mask[1] == 0 && fg.mask[2] == 1 &&
                  fg.mask[3] == 1,
              "mask bytes == [1,0,1,1]");
        expect(
            fg.structured_mask.kind ==
                   cptir::StructuredMaskKind::SlidingWindow &&
                fg.structured_mask.key_len == 4 &&
                fg.structured_mask.window == 2,
            "semantic mask provenance accompanies its dense fallback");
    }

    // General mask SSA (including row_membership) has no special descriptor;
    // attention consumes the ordinary dense Bool channel.
    {
        Trace t = make_trace(1, {{kPortAttnMask, 0, false}});
        t.values.resize(1);
        cptir::Stage stage;
        cptir::Op general_mask;
        general_mask.code = cptir::OpCode::Cast;
        general_mask.result_id = 0;
        general_mask.result_count = 1;
        stage.ops.push_back(general_mask);
        stage.puts.push_back({.channel = 0, .value = 0});
        t.stages.push_back(std::move(stage));
        InterpInstance inst = make_inst(1);
        put(inst, 0, Value::boolean({1, 0, 0, 1, 1, 0}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(
            ok && fg.has_mask &&
                fg.mask == std::vector<std::uint8_t>({1, 0, 0, 1, 1, 0}) &&
                fg.structured_mask.kind ==
                    cptir::StructuredMaskKind::None,
            "row_membership-style general SSA uses dense Bool attention fallback");
    }

    // Not-ready mid-resolution: the first channel (embed_tokens) is ready,
    // but a LATER port (positions) is not — the resolver must still fail
    // (no partial/garbage geometry silently accepted).
    {
        Trace t = make_trace(2, {{kPortEmbedTokens, 0, false}, {kPortPositions, 1, false}});
        InterpInstance inst = make_inst(2);
        put(inst, 0, Value::i32({1, 2}));
        cptir::FireGeometry fg;
        std::string err;
        const bool ok = resolve_fire_geometry(t, inst, 4, fg, &err);
        expect(!ok && contains(err, "not ready"),
              "later not-ready port (positions) fails resolve even after an earlier port succeeds (" +
                  err + ")");
    }

    std::printf("[translate_kv_pages]\n");

    // Both Pages and WSlot translate through the same segment.
    {
        cptir::FireGeometry fg;
        fg.kv_page_indices = {0, 1, 2};
        fg.w_page = {2, 0};
        const std::vector<std::uint32_t> tr = {50, 51, 52};  // relative -> physical
        translate_kv_pages(tr.data(), tr.size(), fg);
        expect(fg.kv_page_indices.size() == 3 && fg.kv_page_indices[0] == 50 &&
                  fg.kv_page_indices[1] == 51 && fg.kv_page_indices[2] == 52,
              "kv_page_indices translated through the segment [50,51,52]");
        expect(fg.w_page.size() == 2 && fg.w_page[0] == 52 && fg.w_page[1] == 50,
              "w_page translated through the SAME segment [52,50]");
    }

    // An out-of-range relative index maps to physical page 0 (reserved-but-
    // unwritten / masked-only candidate), never left dangling.
    {
        cptir::FireGeometry fg;
        fg.kv_page_indices = {0, 5, 1};  // index 5 is past tr_len==3
        fg.w_page = {9};                // also past tr_len
        const std::vector<std::uint32_t> tr = {70, 71, 72};
        translate_kv_pages(tr.data(), tr.size(), fg);
        expect(fg.kv_page_indices[0] == 70 && fg.kv_page_indices[1] == 0 &&
                  fg.kv_page_indices[2] == 71,
              "out-of-range relative index (5) maps to physical page 0, in-range ids still translate");
        expect(fg.w_page[0] == 0, "out-of-range w_page relative index (9) maps to physical page 0");
    }

    // Zero is a valid relative index too (translates like any other).
    {
        cptir::FireGeometry fg;
        fg.kv_page_indices = {0};
        const std::vector<std::uint32_t> tr = {100};
        translate_kv_pages(tr.data(), tr.size(), fg);
        expect(fg.kv_page_indices[0] == 100, "relative index 0 translates through the segment");
    }

    std::printf("\n==== ptir_descriptor_resolve_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
