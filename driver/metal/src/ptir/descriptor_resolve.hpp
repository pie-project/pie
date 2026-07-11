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
// Metal has no device buffers to read back — a channel's "current cell" is
// already resident host memory (`ChannelState`), so this is a PEEK against
// `InterpInstance` (the SAME committed-front-of-ring value `step()`'s own
// descriptor-port loop will later `take()`/`read()`). Reading here is
// non-destructive; the actual consume (ring-advance) for token-family ports
// (embed_tokens/positions/w_slot/w_off) happens exactly once, inside
// `step()`'s EXISTING port loop (unchanged) — this resolver never mutates
// channel state, so there is no double-take.
//
// Same-launch (synchronous) ordering makes the pre-forward read correct
// under run-ahead: fire t's epilogue channel puts happen-before fire t+1's
// descriptor reads, because `MetalDriver::launch` processes members
// strictly in order on the caller thread (no async queue in this
// increment).
//
// NOT-READY IS AN ERROR (W1.6): on a solo device-geometry fire a descriptor
// channel that is not full will never fill (its producing fire failed) — the
// resolver fails the fire (poison that member only) rather than silently
// dummy-running.

#include <cstdint>
#include <string>
#include <vector>

#include "pie_native/ptir/fire_geometry.hpp"
#include "ptir/host_interp.hpp"

namespace pie_metal_driver::ptir_host {

// PtirPort tags (mirror interface/ptir/src/registry.rs `Port` / CUDA's
// descriptor_resolve.hpp — kept byte-identical across both drivers).
enum : std::uint8_t {
    kPortEmbedTokens = 0,
    kPortEmbedIndptr = 1,
    kPortPositions   = 2,
    kPortPages       = 3,
    kPortPageIndptr  = 4,
    kPortKvLen       = 5,
    kPortWSlot       = 6,
    kPortWOff        = 7,
    kPortReadout     = 8,
    kPortAttnMask    = 9,
};

// Device-geometry classification mirrors the runtime and CUDA detectors:
// explicit WSlot/WOff writes plus a channel-bound [B,P] Pages port with P>1.
// Ordinary decode may carry Positions/AttnMask channels while its page mapping
// remains runtime-owned on the wire; those channels alone are not sufficient.
inline bool is_device_geometry_trace(const Trace& trace) {
    bool has_write_desc = false;
    cptir::ChannelId pages_channel = 0;
    bool has_pages = false;
    for (const cptir::PortBinding& pb : trace.ports) {
        if (pb.is_const) continue;
        if (pb.port == kPortWSlot || pb.port == kPortWOff) {
            has_write_desc = true;
        } else if (pb.port == kPortPages) {
            pages_channel = pb.channel;
            has_pages = true;
        }
    }
    if (!has_write_desc || !has_pages || pages_channel >= trace.channels.size()) return false;
    const auto& dims = trace.channels[pages_channel].type.shape.dims;
    return dims.size() == 2 && dims[1] > 1;
}

namespace detail {

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
inline bool read_port_cell(InterpInstance& inst, cptir::ChannelId channel,
                           std::vector<std::uint32_t>& out, std::string* err) {
    ChannelState& st = *inst.channels[channel];
    if (st.queue.empty()) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " not ready (producing fire failed or not yet produced)";
        }
        return false;
    }
    if (!value_as_u32(st.queue.front(), out)) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " has an unsupported dtype for geometry resolution";
        }
        return false;
    }
    return true;
}

// Read the mask port's current cell as raw unpacked bytes (Bool channel: one
// byte per lane, 0/1 — the dense `[lanes, stride]` convention `FireGeometry
// ::mask` documents). Same not-ready/W1.6 contract as `read_port_cell`.
inline bool read_mask_cell(InterpInstance& inst, cptir::ChannelId channel,
                           std::vector<std::uint8_t>& out, std::string* err) {
    ChannelState& st = *inst.channels[channel];
    if (st.queue.empty()) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " not ready (producing fire failed or not yet produced)";
        }
        return false;
    }
    const Value& v = st.queue.front();
    if (v.dtype != DType::Bool) {
        if (err != nullptr) {
            *err = "ptir: descriptor channel " + std::to_string(channel) +
                   " has an unsupported dtype for the attention mask";
        }
        return false;
    }
    out = v.b;
    return true;
}

inline std::uint32_t last_page_len(std::uint32_t len, std::uint32_t page) {
    return (len == 0 || page == 0) ? 0 : ((len - 1) % page) + 1;
}

}  // namespace detail

// Resolve a device-geometry program's descriptor-port channels into `out`.
// Only CHANNEL-bound ports are read; const ports are host-prefilled on the
// wire and untouched here. `page_size` applies the KvLen contract. Returns
// false + `*err` on a not-ready channel (the caller must fail the fire).
inline bool resolve_fire_geometry(const Trace& trace, InterpInstance& inst,
                                  std::uint32_t page_size, cptir::FireGeometry& out,
                                  std::string* err) {
    out = cptir::FireGeometry{};
    cptir::ChannelId ch[10] = {0};
    bool has[10] = {false};
    for (const cptir::PortBinding& pb : trace.ports) {
        if (pb.is_const || pb.port > kPortAttnMask) continue;
        ch[pb.port] = pb.channel;
        has[pb.port] = true;
    }

    // -- token family --
    if (has[kPortEmbedTokens]) {
        if (!detail::read_port_cell(inst, ch[kPortEmbedTokens], out.token_ids, err)) return false;
    }
    const std::uint32_t nnz = static_cast<std::uint32_t>(out.token_ids.size());

    if (has[kPortEmbedIndptr]) {
        if (!detail::read_port_cell(inst, ch[kPortEmbedIndptr], out.qo_indptr, err)) return false;
    } else {
        out.qo_indptr = {0, nnz};  // one lane over all tokens
    }
    const std::size_t lanes = out.qo_indptr.size() > 0 ? out.qo_indptr.size() - 1 : 0;

    if (has[kPortPositions]) {
        if (!detail::read_port_cell(inst, ch[kPortPositions], out.position_ids, err)) return false;
    } else {
        out.position_ids.resize(nnz);
        for (std::uint32_t i = 0; i < nnz; ++i) out.position_ids[i] = i;
    }

    // -- KV family (CSR-prefix contract) --
    if (has[kPortPageIndptr]) {
        if (!detail::read_port_cell(inst, ch[kPortPageIndptr], out.kv_page_indptr, err)) return false;
    }
    if (has[kPortPages]) {
        if (!detail::read_port_cell(inst, ch[kPortPages], out.kv_page_indices, err)) return false;
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
        if (!detail::read_port_cell(inst, ch[kPortKvLen], lens, err)) return false;
        for (std::uint32_t len : lens) {
            out.kv_last_page_lens.push_back(detail::last_page_len(len, page_size));
        }
    }

    // -- read-out --
    if (has[kPortReadout]) {
        if (!detail::read_port_cell(inst, ch[kPortReadout], out.sampling_indices, err)) return false;
        out.sampling_indptr = {0, static_cast<std::uint32_t>(out.sampling_indices.size())};
    } else {
        out.sampling_indptr.push_back(0);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (out.qo_indptr[lane + 1] > out.qo_indptr[lane]) {
                out.sampling_indices.push_back(out.qo_indptr[lane + 1] - 1);
            }
            out.sampling_indptr.push_back(static_cast<std::uint32_t>(out.sampling_indices.size()));
        }
    }

    // -- explicit KV write descriptor (w_slot/w_off → write_kv_explicit) --
    if (has[kPortWSlot]) {
        if (!detail::read_port_cell(inst, ch[kPortWSlot], out.w_page, err)) return false;
        out.has_write_desc = true;
    }
    if (has[kPortWOff]) {
        if (!detail::read_port_cell(inst, ch[kPortWOff], out.w_off, err)) return false;
    }

    // -- dense attention mask (→ pack_dense_mask) --
    if (has[kPortAttnMask]) {
        if (!detail::read_mask_cell(inst, ch[kPortAttnMask], out.mask, err)) return false;
        out.has_mask = true;
    }
    return true;
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

}  // namespace pie_metal_driver::ptir_host
