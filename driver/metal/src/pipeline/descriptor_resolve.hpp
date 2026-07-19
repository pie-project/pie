#pragma once

// PTIR pre-forward descriptor resolution (metal_ptir_plan.md Phase 2, W1.1) —
// the Metal mirror of CUDA's `driver/cuda/src/ptir/descriptor_resolve.hpp`
// and the host's `map_geometry` (runtime/engine/src/pipeline/fire/geometry.rs).
// For a program whose descriptor ports bind CHANNELS (device-produced
// geometry, e.g. the run-ahead beam epilogue), the driver reads the port
// channels' CURRENT cells BEFORE the forward and fills the standard
// per-request `FireGeometry` — so the forward runs on the ordinary
// batch/attention machinery with NO program-specific assembly.
//
// PROGRAM-AGNOSTIC (owner constraint): this is a 1:1 port→field copier
// applying two fixed contracts, NOT beam logic:
//   * CSR-prefix: for a CSR port pair (data, indptr), the indptr's LAST
//     element is the valid prefix length of the data port (channels keep
//     fixed shapes; the program densely packs live entries at the front).
//   * KvLen → last_page_len: `((len-1) % page) + 1` (port semantics).
// The port→field table is kept in explicit correspondence with CUDA's
// resolver and `map_geometry`.
//
// Metal channel cells use CPU-visible, device-bindable Shared storage, so
// descriptor resolution needs no readback. This is a PEEK against
// `InterpInstance` (the SAME authoritative committed-front-of-ring value
// `step()`'s own
// descriptor-port loop will later `take()`/`read()`). Reading here is
// non-destructive; the actual consume (ring-advance) for token-family ports
// (embed_tokens/positions/w_slot/w_off) happens exactly once, inside
// `step()`'s EXISTING port loop (unchanged) — this resolver never mutates
// channel state, so there is no double-take.
//
// Same-launch (synchronous) ordering makes the pre-forward read correct
// under run-ahead: fire t's epilogue channel puts happen-before fire t+1's
// descriptor reads, because `Context::launch` processes members
// strictly in order on the caller thread (no async queue in this
// increment).
//
// Resolution is typed: an empty descriptor cell is transient NotReady; malformed
// contents are Failed. Context combines NotReady with the endpoint poison word
// to distinguish a live producer from a permanently failed one.

#include <cstdint>
#include <string>
#include <vector>

#include "pie_native/ptir/descriptor.hpp"
#include "pie_native/ptir/fire_geometry.hpp"
#include "pipeline/interp.hpp"

namespace pie::metal::pipeline {

using namespace pie_native::ptir::descriptor;

enum class GeometryResolveStatus { Ready, NotReady, Failed };

struct GeometryResolveResult {
    GeometryResolveStatus status = GeometryResolveStatus::Ready;
    cptir::ChannelId channel = 0;
};

namespace detail {

inline const cptir::Op* producer(
    const Trace& trace, cptir::ValueId value) {
    for (const cptir::Stage& stage : trace.stages) {
        for (const cptir::Op& op : stage.ops) {
            if (value >= op.result_id &&
                value < op.result_id + op.result_count) {
                return &op;
            }
        }
    }
    return nullptr;
}

inline cptir::StructuredMaskDescriptor structured_mask_descriptor(
    const Trace& trace,
    cptir::ChannelId mask_channel) {
    const cptir::ChannelPut* selected = nullptr;
    for (const cptir::Stage& stage : trace.stages) {
        for (const cptir::ChannelPut& put : stage.puts) {
            if (put.channel == mask_channel) selected = &put;
        }
    }
    if (selected == nullptr) return {};
    cptir::ValueId value = selected->value;
    for (std::size_t depth = 0; depth <= trace.values.size(); ++depth) {
        const cptir::Op* op = producer(trace, value);
        if (op == nullptr) break;
        if (op->code == OpCode::Reshape && !op->args.empty()) {
            value = op->args[0];
            continue;
        }
        cptir::StructuredMaskDescriptor descriptor;
        switch (op->code) {
            case OpCode::CausalMask:
                descriptor.kind =
                    cptir::StructuredMaskKind::Causal;
                descriptor.key_len = op->imm;
                return descriptor;
            case OpCode::SlidingWindowMask:
                descriptor.kind =
                    cptir::StructuredMaskKind::SlidingWindow;
                descriptor.key_len = op->imm;
                descriptor.window = op->imm2;
                return descriptor;
            case OpCode::SinkWindowMask:
                descriptor.kind =
                    cptir::StructuredMaskKind::SinkWindow;
                descriptor.key_len = op->imm;
                descriptor.sink = op->imm2;
                descriptor.window = op->imm3;
                return descriptor;
            default:
                return {};
        }
    }
    return {};
}

// Bit-reinterpret a Value's lanes as u32, mirroring CUDA's raw-byte
// `as_u32` (a straight little-endian reinterpretation of the wire bytes,
// not a value-preserving numeric cast). Token/position/page ids are always
// non-negative in practice; I32's two's-complement bit pattern is preserved
// by `static_cast<uint32_t>`, exactly matching the CUDA memcpy semantics.
inline bool value_as_u32(const Value& v, std::vector<std::uint32_t>& out) {
    switch (v.dtype) {
        case DType::U32:
            out = v.u;
            return true;
        case DType::I32: {
            out.resize(v.i.size());
            for (std::size_t k = 0; k < out.size(); ++k) {
                out[k] = static_cast<std::uint32_t>(v.i[k]);
            }
            return true;
        }
        default:
            return false;
    }
}

// Peek a channel-bound port's CURRENT committed cell (the ring's front,
// non-destructive). Fails (W1.6) if the cell is not full — nothing will
// fill it on a solo device-geometry fire; the caller must fail the fire,
// never dummy-run.
inline GeometryResolveResult read_port_cell(
    InterpInstance& inst,
    cptir::ChannelId channel,
    std::vector<std::uint32_t>& out,
    std::string* err) {
    ChannelState& st = *inst.channels[channel];
    if (st.empty()) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " not ready (not yet produced)";
        }
        return {GeometryResolveStatus::NotReady, channel};
    }
    const Value value = st.front();
    if (!value_as_u32(value, out)) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " has an unsupported dtype for geometry resolution";
        }
        return {GeometryResolveStatus::Failed, channel};
    }
    return {GeometryResolveStatus::Ready, channel};
}

// Read the mask port's current cell as raw unpacked bytes (Bool channel: one
// byte per lane, 0/1 — the dense `[lanes, stride]` convention `FireGeometry
// ::mask` documents). Same not-ready/W1.6 contract as `read_port_cell`.
inline GeometryResolveResult read_mask_cell(
    InterpInstance& inst,
    cptir::ChannelId channel,
    std::vector<std::uint8_t>& out,
    std::string* err) {
    ChannelState& st = *inst.channels[channel];
    if (st.empty()) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " not ready (not yet produced)";
        }
        return {GeometryResolveStatus::NotReady, channel};
    }
    const Value v = st.front();
    if (v.dtype != DType::Bool) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " has an unsupported dtype for the attention mask";
        }
        return {GeometryResolveStatus::Failed, channel};
    }
    out = v.b;
    return {GeometryResolveStatus::Ready, channel};
}

}  // namespace detail

// Resolve a device-geometry program's descriptor-port channels into `out`.
// Only CHANNEL-bound ports are read; const ports are host-prefilled on the
// wire and untouched here. `page_size` applies the KvLen contract. Returns
// The typed form distinguishes transient readiness from permanent failure.
inline GeometryResolveResult resolve_fire_geometry_typed(
    const Trace& trace,
    const std::vector<ConstPortValue>& const_ports,
    InterpInstance& inst,
    std::uint32_t page_size,
    cptir::FireGeometry& out,
    std::string* err) {
    out = cptir::FireGeometry{};
    cptir::ChannelId ch[10] = {0};
    bool has[10] = {false};
    bool channel_bound[10] = {false};
    const Value* constant[10] = {nullptr};
    for (const cptir::PortBinding& pb : trace.ports) {
        if (pb.port > kPortAttnMask) continue;
        has[pb.port] = true;
        if (pb.is_const) {
            const auto found = std::find_if(
                const_ports.begin(),
                const_ports.end(),
                [&](const ConstPortValue& value) {
                    return value.port == pb.port;
                });
            if (found == const_ports.end()) {
                if (err != nullptr) {
                    *err = "ptir: const descriptor port payload is missing";
                }
                return {GeometryResolveStatus::Failed, 0};
            }
            constant[pb.port] = &found->value;
        } else {
            channel_bound[pb.port] = true;
            ch[pb.port] = pb.channel;
        }
    }
    auto read_u32_port = [&](std::uint8_t port,
                             std::vector<std::uint32_t>& values) {
        if (channel_bound[port]) {
            return detail::read_port_cell(
                inst, ch[port], values, err);
        }
        if (constant[port] == nullptr ||
            !detail::value_as_u32(*constant[port], values)) {
            if (err != nullptr) {
                *err =
                    "ptir: const descriptor port has an unsupported dtype";
            }
            return GeometryResolveResult{
                GeometryResolveStatus::Failed, 0};
        }
        return GeometryResolveResult{
            GeometryResolveStatus::Ready, 0};
    };
    auto read_bool_port = [&](std::uint8_t port,
                              std::vector<std::uint8_t>& values) {
        if (channel_bound[port]) {
            return detail::read_mask_cell(
                inst, ch[port], values, err);
        }
        if (constant[port] == nullptr ||
            constant[port]->dtype != DType::Bool) {
            if (err != nullptr) {
                *err =
                    "ptir: const attention mask has an unsupported dtype";
            }
            return GeometryResolveResult{
                GeometryResolveStatus::Failed, 0};
        }
        values = constant[port]->b;
        return GeometryResolveResult{
            GeometryResolveStatus::Ready, 0};
    };

    // -- token family --
    if (has[kPortEmbedTokens]) {
        const auto result =
            read_u32_port(kPortEmbedTokens, out.token_ids);
        if (result.status != GeometryResolveStatus::Ready) return result;
    }
    const std::uint32_t nnz = static_cast<std::uint32_t>(out.token_ids.size());

    if (has[kPortEmbedIndptr]) {
        const auto result =
            read_u32_port(kPortEmbedIndptr, out.qo_indptr);
        if (result.status != GeometryResolveStatus::Ready) return result;
    } else {
        out.qo_indptr = {0, nnz};  // one lane over all tokens
    }
    const std::size_t lanes = out.qo_indptr.size() > 0 ? out.qo_indptr.size() - 1 : 0;

    if (has[kPortPositions]) {
        const auto result =
            read_u32_port(kPortPositions, out.position_ids);
        if (result.status != GeometryResolveStatus::Ready) return result;
    } else {
        out.position_ids.resize(nnz);
        for (std::uint32_t i = 0; i < nnz; ++i) out.position_ids[i] = i;
    }

    // -- KV family (CSR-prefix contract) --
    if (has[kPortPageIndptr]) {
        const auto result =
            read_u32_port(kPortPageIndptr, out.kv_page_indptr);
        if (result.status != GeometryResolveStatus::Ready) return result;
    }
    if (has[kPortPages]) {
        const auto result =
            read_u32_port(kPortPages, out.kv_page_indices);
        if (result.status != GeometryResolveStatus::Ready) return result;
        // CSR-prefix: trim the fixed-shape data port to page_indptr's last entry.
        if (!out.kv_page_indptr.empty()) {
            const std::uint32_t nnz_pages = out.kv_page_indptr.back();
            if (nnz_pages <= out.kv_page_indices.size()) {
                out.kv_page_indices.resize(nnz_pages);
            }
        }
        out.has_kv_family = true;
    }
    if (has[kPortKvLen]) {
        std::vector<std::uint32_t> lens;
        const auto result =
            read_u32_port(kPortKvLen, lens);
        if (result.status != GeometryResolveStatus::Ready) return result;
        for (std::uint32_t len : lens) {
            out.kv_last_page_lens.push_back(last_page_len(len, page_size));
        }
    }

    // -- read-out --
    if (has[kPortReadout]) {
        const auto result =
            read_u32_port(kPortReadout, out.sampling_indices);
        if (result.status != GeometryResolveStatus::Ready) return result;
        out.sampling_indptr.push_back(0);
        if (lanes <= 1) {
            out.sampling_indptr.push_back(
                static_cast<std::uint32_t>(
                    out.sampling_indices.size()));
        } else if (out.sampling_indices.size() == lanes) {
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                out.sampling_indptr.push_back(
                    static_cast<std::uint32_t>(lane + 1));
            }
        } else {
            if (err != nullptr) {
                *err =
                    "ptir: multi-request readout needs one index per request";
            }
            return {GeometryResolveStatus::Failed, 0};
        }
    } else {
        out.sampling_indptr.push_back(0);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const std::uint32_t span =
                out.qo_indptr[lane + 1] - out.qo_indptr[lane];
            if (span != 0) {
                out.sampling_indices.push_back(span - 1);
            }
            out.sampling_indptr.push_back(static_cast<std::uint32_t>(out.sampling_indices.size()));
        }
    }

    // -- explicit KV write descriptor (w_slot/w_off → write_kv_explicit) --
    if (has[kPortWSlot]) {
        const auto result =
            read_u32_port(kPortWSlot, out.w_page);
        if (result.status != GeometryResolveStatus::Ready) return result;
        out.has_write_desc = true;
    }
    if (has[kPortWOff]) {
        const auto result =
            read_u32_port(kPortWOff, out.w_off);
        if (result.status != GeometryResolveStatus::Ready) return result;
    }

    // -- dense attention mask (→ pack_dense_mask) --
    if (has[kPortAttnMask]) {
        if (channel_bound[kPortAttnMask]) {
            out.structured_mask =
                detail::structured_mask_descriptor(
                    trace, ch[kPortAttnMask]);
        }
        const auto result =
            read_bool_port(kPortAttnMask, out.mask);
        if (result.status != GeometryResolveStatus::Ready) return result;
        out.has_mask = true;
        if (out.structured_mask &&
            out.structured_mask.key_len == 0 &&
            !out.token_ids.empty() &&
            out.mask.size() % out.token_ids.size() == 0) {
            out.structured_mask.key_len =
                static_cast<std::uint32_t>(
                    out.mask.size() / out.token_ids.size());
        }
    }
    return {GeometryResolveStatus::Ready, 0};
}

inline GeometryResolveResult resolve_fire_geometry_typed(
    const Trace& trace,
    InterpInstance& inst,
    std::uint32_t page_size,
    cptir::FireGeometry& out,
    std::string* err) {
    return resolve_fire_geometry_typed(
        trace, {}, inst, page_size, out, err);
}

inline bool resolve_fire_geometry(
    const Trace& trace,
    InterpInstance& inst,
    std::uint32_t page_size,
    cptir::FireGeometry& out,
    std::string* err) {
    return resolve_fire_geometry_typed(
               trace, {}, inst, page_size, out, err).status ==
           GeometryResolveStatus::Ready;
}

inline GeometryResolveResult resolve_fire_geometry_typed(
    const ExecPlan& plan,
    InterpInstance& inst,
    std::uint32_t page_size,
    cptir::FireGeometry& out,
    std::string* err) {
    return resolve_fire_geometry_typed(
        plan.trace,
        plan.const_ports,
        inst,
        page_size,
        out,
        err);
}

// WorkingSet page translation (kv_refact.md flattened-table model), the
// Metal mirror of CUDA's `ptir_dispatch.cu`'s inline translate step: a
// device-geometry program's channel-resolved `Pages`/`WSlot` values are
// WorkingSet-RELATIVE indexes — the guest never holds physical ids. `tr` is
// this instance's translation segment (`tr[relative_index]` -> physical
// page id), `tr_len` its length. An index at or past `tr_len` is a
// reserved-but-unwritten page (a masked-only attention candidate): mapped
// to page 0 (readable garbage the mask discards), never left dangling.
// Callers must skip calling this entirely for an EMPTY segment (that is the
// legacy/pass-through case — ids already physical, not relative).
inline void translate_kv_pages(const std::uint32_t* tr, std::size_t tr_len,
                               cptir::FireGeometry& fg) {
    auto translate = [&](std::vector<std::uint32_t>& ids) {
        for (std::uint32_t& v : ids) v = v < tr_len ? tr[v] : 0u;
    };
    translate(fg.kv_page_indices);
    translate(fg.w_page);
}

}  // namespace pie::metal::pipeline
