#pragma once

// PTIR pre-forward descriptor resolution (plan W1.1) — the DEVICE mirror of the
// host's `map_geometry` (runtime/engine/src/pipeline/fire/geometry.rs). For a program whose
// descriptor ports bind CHANNELS (device-produced geometry, e.g. the run-ahead
// beam epilogue), the driver reads the port channels' current cells BEFORE the
// forward and fills the standard per-request geometry — so the forward runs on
// the ordinary batch/attention machinery with NO program-specific assembly.
//
// PROGRAM-AGNOSTIC (owner constraint): this is a 1:1 port→field copier applying
// two fixed contracts (plan §3.1), NOT beam logic:
//   * CSR-prefix: for a CSR port pair (data, indptr), the indptr's LAST element
//     is the valid prefix length of the data port (channels keep fixed shapes;
//     the program densely packs live entries at the front).
//   * KvLen → last_page_len: `((len-1) % page) + 1` (port semantics).
// The port→field table is kept in explicit correspondence with `map_geometry`.
//
// Same-stream ordering (plan §3.5) makes the D2H reads correct under run-ahead:
// fire t's epilogue channel puts happen-before fire t+1's descriptor reads.
// NOT-READY IS AN ERROR (W1.6): on a solo device-geometry fire a descriptor
// channel that is not full will never fill (its producing fire failed) — the
// resolver fails the fire rather than silently dummy-running.

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "pipeline/channel_registry.hpp"
#include "pie/driver/fire/descriptor.hpp"
#include "pie/driver/fire/geometry.hpp"
#include "pie/driver/launch/program.hpp"
#include "pie/driver/launch/query.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie::driver::launch (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie::driver::launch;
using namespace pie::driver::fire;
using namespace pie::driver::fire::descriptor;

namespace detail {

struct CachedPortCell {
    std::vector<std::uint8_t> bytes;
    std::uint8_t ready = 0;
};

using PortCellCache =
    std::unordered_map<std::uint32_t, CachedPortCell>;

// Read a channel-bound port's committed cell as raw bytes (D2H). Fails (W1.6) if
// the cell is not full — nothing will fill it on a solo device-geometry fire.
inline bool read_port_cell(ChannelView& view, ChannelId dense,
                           std::vector<std::uint8_t>& out, std::string* err,
                           const std::unordered_set<std::uint32_t>*
                               pending_slots = nullptr,
                           const PortCellCache* cached_cells = nullptr) {
    const std::uint32_t slot = view.slot(dense);
    if (cached_cells != nullptr) {
        const auto cached = cached_cells->find(slot);
        if (cached != cached_cells->end()) {
            if (cached->second.ready == 0) {
                if (err)
                    *err = "ptir: descriptor channel " +
                           std::to_string(dense) +
                           " not ready (producing fire failed or not yet produced)";
                return false;
            }
            out = cached->second.bytes;
            return true;
        }
    }
    if (pending_slots != nullptr && pending_slots->contains(slot)) {
        out.resize(view.cell_bytes(dense));
        cudaMemcpy(
            out.data(), view.pending_cell(dense), out.size(),
            cudaMemcpyDeviceToHost);
        return true;
    }
    if (!view.committed_full(dense)) {
        if (err)
            *err = "ptir: descriptor channel " + std::to_string(dense) +
                   " not ready (producing fire failed or not yet produced)";
        return false;
    }
    out.resize(view.cell_bytes(dense));
    view.read_committed(dense, out.data(), out.size());
    return true;
}

inline std::vector<std::uint32_t> as_u32(const std::vector<std::uint8_t>& bytes) {
    std::vector<std::uint32_t> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), out.size() * 4);
    return out;
}

}  // namespace detail

// Resolve a device-geometry program's descriptor-port channels into `out`.
// Only CHANNEL-bound ports are read; const ports were host-prefilled on the wire
// (plan §3.5). `page_size` applies the KvLen contract. Returns false + `*err` on
// a not-ready channel (the fire must be failed).
inline bool resolve_fire_geometry(const Trace& trace, ChannelView& view,
                                  std::uint32_t page_size, FireGeometry& out,
                                  std::string* err,
                                  bool allow_structured_masks = false,
                                  const std::unordered_set<std::uint32_t>*
                                      pending_slots = nullptr,
                                  const detail::PortCellCache*
                                      cached_cells = nullptr) {
    // Index the channel-bound ports by tag.
    // Sized by the HIGHEST port tag, not by the port COUNT: tags 10-14 are
    // reserved holes left by the rs-geometry rebalance, so the two are no
    // longer the same number.
    constexpr std::size_t kPortSlots = kPortRsFoldLen + 1;
    ChannelId ch[kPortSlots];
    bool has[kPortSlots] = {false};
    for (const PortBinding& pb : trace.ports) {
        if (pb.is_const || pb.port >= kPortSlots) continue;
        ch[pb.port] = pb.channel;
        has[pb.port] = true;
    }
    for (const PortBinding& pb : trace.ports) {
        if (!pb.is_const) continue;
        const auto values = detail::as_u32(pb.const_data);
        switch (pb.port) {
            case kPortEmbedTokens: out.token_ids = values; break;
            case kPortEmbedIndptr: out.qo_indptr = values; break;
            case kPortPositions: out.position_ids = values; break;
            case kPortPages:
                out.kv_page_indices = values;
                out.has_kv_family = true;
                break;
            case kPortPageIndptr: out.kv_page_indptr = values; break;
            case kPortKvLen:
                for (std::uint32_t len : values) {
                    out.kv_last_page_lens.push_back(
                        last_page_len(len, page_size));
                }
                break;
            case kPortReadout: out.sampling_indices = values; break;
            case kPortRsBufferPages:
                out.rs_buffer_slot_ids = values;
                out.has_rs_buffer_family = true;
                break;
            case kPortRsBufferIndptr: out.rs_buffer_slot_indptr = values; break;
            case kPortRsBufferLen: out.rs_buffer_lens = values; break;
            case kPortRsFoldLen:
                out.rs_fold_lens = values;
                out.has_rs_fold_len = true;
                break;
            case kPortRsWSlot:
                out.rs_w_slot = values;
                out.has_rs_write_desc = true;
                break;
            case kPortRsWOff: out.rs_w_off = values; break;
            default: break;
        }
    }

    // -- token family --
    if (has[kPortEmbedTokens]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortEmbedTokens], b, err, pending_slots,
                cached_cells)) return false;
        out.token_ids = detail::as_u32(b);
    }
    const std::uint32_t nnz = static_cast<std::uint32_t>(out.token_ids.size());

    if (has[kPortEmbedIndptr]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortEmbedIndptr], b, err, pending_slots,
                cached_cells)) return false;
        out.qo_indptr = detail::as_u32(b);
    } else if (out.qo_indptr.empty()) {
        out.qo_indptr = {0, nnz};  // one lane over all tokens
    }
    const std::size_t lanes = out.qo_indptr.size() > 0 ? out.qo_indptr.size() - 1 : 0;

    if (has[kPortPositions]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortPositions], b, err, pending_slots,
                cached_cells)) return false;
        out.position_ids = detail::as_u32(b);
    } else if (out.position_ids.empty()) {
        out.position_ids.resize(nnz);
        for (std::uint32_t i = 0; i < nnz; ++i) out.position_ids[i] = i;
    }

    // -- KV family (CSR-prefix contract) --
    if (has[kPortPageIndptr]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortPageIndptr], b, err, pending_slots,
                cached_cells)) return false;
        out.kv_page_indptr = detail::as_u32(b);
    }
    if (has[kPortPages]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortPages], b, err, pending_slots,
                cached_cells)) return false;
        out.kv_page_indices = detail::as_u32(b);
        if (!out.kv_page_indptr.empty()) {
            const auto& dims =
                trace.channels[ch[kPortPages]].type.shape.dims;
            if (dims.size() == 2 && dims[0] == lanes) {
                std::vector<std::uint32_t> packed;
                packed.reserve(out.kv_page_indptr.back());
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    const std::uint32_t count =
                        out.kv_page_indptr[lane + 1] -
                        out.kv_page_indptr[lane];
                    if (count > dims[1]) return false;
                    const std::size_t begin = lane * dims[1];
                    packed.insert(
                        packed.end(),
                        out.kv_page_indices.begin() + begin,
                        out.kv_page_indices.begin() + begin + count);
                }
                out.kv_page_indices = std::move(packed);
            } else {
                const std::uint32_t nnz_pages =
                    out.kv_page_indptr.back();
                if (nnz_pages <= out.kv_page_indices.size()) {
                    out.kv_page_indices.resize(nnz_pages);
                }
            }
        }
        out.has_kv_family = true;
    }
    if (has[kPortKvLen]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortKvLen], b, err, pending_slots,
                cached_cells)) return false;
        for (std::uint32_t len : detail::as_u32(b))
            out.kv_last_page_lens.push_back(last_page_len(len, page_size));
    }

    // -- recurrent-state buffered-slot family (same CSR-prefix contract as KV) --
    if (has[kPortRsBufferIndptr]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortRsBufferIndptr], b, err, pending_slots,
                cached_cells)) return false;
        out.rs_buffer_slot_indptr = detail::as_u32(b);
    }
    if (has[kPortRsBufferPages]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortRsBufferPages], b, err, pending_slots,
                cached_cells)) return false;
        out.rs_buffer_slot_ids = detail::as_u32(b);
        // CSR-prefix: channels keep a fixed shape, so the indptr's last
        // element is the live prefix length of the slab-id vector. A [rows,
        // stride] declaration is densely repacked per row, exactly as the KV
        // `pages` port is.
        if (!out.rs_buffer_slot_indptr.empty()) {
            const auto& dims =
                trace.channels[ch[kPortRsBufferPages]].type.shape.dims;
            const std::size_t rows = out.rs_buffer_slot_indptr.size() - 1;
            if (dims.size() == 2 && dims[0] == rows) {
                std::vector<std::uint32_t> packed;
                packed.reserve(out.rs_buffer_slot_indptr.back());
                for (std::size_t row = 0; row < rows; ++row) {
                    const std::uint32_t count =
                        out.rs_buffer_slot_indptr[row + 1] -
                        out.rs_buffer_slot_indptr[row];
                    if (count > dims[1]) return false;
                    const std::size_t begin = row * dims[1];
                    packed.insert(
                        packed.end(),
                        out.rs_buffer_slot_ids.begin() + begin,
                        out.rs_buffer_slot_ids.begin() + begin + count);
                }
                out.rs_buffer_slot_ids = std::move(packed);
            } else {
                const std::uint32_t nnz_slabs =
                    out.rs_buffer_slot_indptr.back();
                if (nnz_slabs <= out.rs_buffer_slot_ids.size()) {
                    out.rs_buffer_slot_ids.resize(nnz_slabs);
                }
            }
        }
        out.has_rs_buffer_family = true;
    }
    if (has[kPortRsBufferLen]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortRsBufferLen], b, err, pending_slots,
                cached_cells)) return false;
        out.rs_buffer_lens = detail::as_u32(b);
    }
    // Read LAST among the RS ports, and unconditionally: this is the one whose
    // value the host may never have seen, so there is no wire array to fall
    // back on if the cell is not ready.
    if (has[kPortRsFoldLen]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortRsFoldLen], b, err, pending_slots,
                cached_cells)) return false;
        out.rs_fold_lens = detail::as_u32(b);
        out.has_rs_fold_len = true;
    }
    if (has[kPortRsWSlot]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortRsWSlot], b, err, pending_slots,
                cached_cells)) return false;
        out.rs_w_slot = detail::as_u32(b);
        out.rs_w_slot.resize(nnz);
        out.has_rs_write_desc = true;
    }
    if (has[kPortRsWOff]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortRsWOff], b, err, pending_slots,
                cached_cells)) return false;
        out.rs_w_off = detail::as_u32(b);
        out.rs_w_off.resize(nnz);
    }

    // -- read-out --
    std::vector<std::uint32_t> readout = out.sampling_indices;
    if (has[kPortReadout]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortReadout], b, err, pending_slots,
                cached_cells)) return false;
        readout = detail::as_u32(b);
    }
    out.sampling_indices.clear();
    out.sampling_indptr.clear();
    out.sampling_indptr.push_back(0);
    if (readout.empty()) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (out.qo_indptr[lane + 1] > out.qo_indptr[lane]) {
                out.sampling_indices.push_back(
                    out.qo_indptr[lane + 1] -
                    out.qo_indptr[lane] - 1);
            }
            out.sampling_indptr.push_back(
                static_cast<std::uint32_t>(out.sampling_indices.size()));
        }
    } else {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const std::uint32_t begin = out.qo_indptr[lane];
            const std::uint32_t end = out.qo_indptr[lane + 1];
            for (std::uint32_t index : readout) {
                if (index >= begin && index < end) {
                    out.sampling_indices.push_back(index - begin);
                }
            }
            out.sampling_indptr.push_back(
                static_cast<std::uint32_t>(out.sampling_indices.size()));
        }
    }

    // -- explicit KV write descriptor (w_slot/w_off → write_kv_explicit) --
    if (has[kPortWSlot]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortWSlot], b, err, pending_slots,
                cached_cells)) return false;
        out.w_page = detail::as_u32(b);
        out.has_write_desc = true;
    }
    if (has[kPortWOff]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, ch[kPortWOff], b, err, pending_slots,
                cached_cells)) return false;
        out.w_off = detail::as_u32(b);
    }

    // -- dense attention mask (→ pack_dense_mask) --
    if (has[kPortAttnMask]) {
        out.structured_mask =
            structured_mask_descriptor(trace, ch[kPortAttnMask]);
        // Runtime-window APIs cannot encode an empty window. Keep window=0
        // valid by consuming the program's exact dense Bool result instead.
        const bool direct =
            allow_structured_masks &&
            (out.structured_mask.kind == StructuredMaskKind::Causal ||
             (out.structured_mask.kind ==
                  StructuredMaskKind::SlidingWindow &&
              out.structured_mask.window > 0) ||
             (out.structured_mask.kind ==
                  StructuredMaskKind::SinkWindow &&
              out.structured_mask.window > 0));
        if (!direct) {
            if (!detail::read_port_cell(
                    view, ch[kPortAttnMask], out.mask, err,
                    pending_slots, cached_cells)) {
                return false;
            }
            out.has_mask = true;
            if (out.structured_mask &&
                out.structured_mask.key_len == 0 &&
                !out.token_ids.empty() &&
                out.mask.size() % out.token_ids.size() == 0) {
                out.structured_mask.key_len = static_cast<std::uint32_t>(
                    out.mask.size() / out.token_ids.size());
            }
        }
    }
    return true;
}

/// Resolve ONLY the `rs_fold_len` port, for a program whose geometry is
/// otherwise host-composed. A commit fire is an ordinary wire program in every
/// respect except this one number, which the host may never have learned --
/// so promoting the whole program to device geometry to reach it would be a
/// far larger claim than the truth.
///
/// Absent port is NOT an error: a fire may carry a mix of rows, and only the
/// flagged ones need this. Composition refuses a flagged row whose program
/// left `has_rs_fold_len` false, which is where that mistake is caught.
inline bool resolve_rs_fold_len(
    const Trace& trace, ChannelView& view, FireGeometry& out,
    std::string* err,
    const std::unordered_set<std::uint32_t>* pending_slots = nullptr,
    const detail::PortCellCache* cached_cells = nullptr) {
    for (const PortBinding& candidate : trace.ports) {
        if (candidate.port != kPortRsFoldLen) continue;
        if (candidate.is_const) {
            out.rs_fold_lens = detail::as_u32(candidate.const_data);
            out.has_rs_fold_len = true;
            return true;
        }
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(
                view, candidate.channel, b, err, pending_slots,
                cached_cells)) {
            return false;
        }
        out.rs_fold_lens = detail::as_u32(b);
        out.has_rs_fold_len = true;
        return true;
    }
    return true;
}

inline bool resolve_attention_mask(
    const Trace& trace, ChannelView& view, FireGeometry& out,
    std::string* err, bool allow_structured_masks,
    const std::unordered_set<std::uint32_t>* pending_slots = nullptr,
    const detail::PortCellCache* cached_cells = nullptr) {
    const PortBinding* binding = nullptr;
    for (const PortBinding& candidate : trace.ports) {
        if (candidate.port != kPortAttnMask) continue;
        if (candidate.is_const) {
            if (err) *err = "ptir: device-derived attention mask must bind a channel";
            return false;
        }
        binding = &candidate;
        break;
    }
    if (binding == nullptr) {
        if (err) *err = "ptir: device-derived attention mask port is missing";
        return false;
    }

    out.structured_mask =
        structured_mask_descriptor(trace, binding->channel);
    const bool direct =
        allow_structured_masks &&
        (out.structured_mask.kind == StructuredMaskKind::Causal ||
         (out.structured_mask.kind == StructuredMaskKind::SlidingWindow &&
          out.structured_mask.window > 0) ||
         (out.structured_mask.kind == StructuredMaskKind::SinkWindow &&
          out.structured_mask.window > 0));
    if (direct) return true;

    if (!detail::read_port_cell(
            view, binding->channel, out.mask, err, pending_slots,
            cached_cells)) {
        return false;
    }
    out.has_mask = true;
    return true;
}

}  // namespace pie_cuda_driver::pipeline
