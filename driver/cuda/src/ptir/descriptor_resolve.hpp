#pragma once

// PTIR pre-forward descriptor resolution (plan W1.1) — the DEVICE mirror of the
// host's `map_geometry` (runtime/src/ptir/ptir_geometry.rs). For a program whose
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
#include <vector>

#include "ptir/channel_registry.hpp"
#include "ptir/fire_geometry.hpp"
#include "ptir/trace.hpp"

namespace pie_cuda_driver::ptir {

// PtirPort tags (mirror interface/ptir/src/registry.rs `Port`).
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

namespace detail {

// Read a channel-bound port's committed cell as raw bytes (D2H). Fails (W1.6) if
// the cell is not full — nothing will fill it on a solo device-geometry fire.
inline bool read_port_cell(ChannelView& view, ChannelId dense,
                           std::vector<std::uint8_t>& out, std::string* err) {
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

inline std::uint32_t last_page_len(std::uint32_t len, std::uint32_t page) {
    return (len == 0 || page == 0) ? 0 : ((len - 1) % page) + 1;
}

}  // namespace detail

// Resolve a device-geometry program's descriptor-port channels into `out`.
// Only CHANNEL-bound ports are read; const ports were host-prefilled on the wire
// (plan §3.5). `page_size` applies the KvLen contract. Returns false + `*err` on
// a not-ready channel (the fire must be failed).
inline bool resolve_fire_geometry(const Trace& trace, ChannelView& view,
                                  std::uint32_t page_size, FireGeometry& out,
                                  std::string* err) {
    // Index the channel-bound ports by tag.
    ChannelId ch[10];
    bool has[10] = {false};
    for (const PortBinding& pb : trace.ports) {
        if (pb.is_const || pb.port > kPortAttnMask) continue;
        ch[pb.port] = pb.channel;
        has[pb.port] = true;
    }

    // -- token family --
    if (has[kPortEmbedTokens]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortEmbedTokens], b, err)) return false;
        out.token_ids = detail::as_u32(b);
    }
    const std::uint32_t nnz = static_cast<std::uint32_t>(out.token_ids.size());

    if (has[kPortEmbedIndptr]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortEmbedIndptr], b, err)) return false;
        out.qo_indptr = detail::as_u32(b);
    } else {
        out.qo_indptr = {0, nnz};  // one lane over all tokens
    }
    const std::size_t lanes = out.qo_indptr.size() > 0 ? out.qo_indptr.size() - 1 : 0;

    if (has[kPortPositions]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortPositions], b, err)) return false;
        out.position_ids = detail::as_u32(b);
    } else {
        out.position_ids.resize(nnz);
        for (std::uint32_t i = 0; i < nnz; ++i) out.position_ids[i] = i;
    }

    // -- KV family (CSR-prefix contract) --
    if (has[kPortPageIndptr]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortPageIndptr], b, err)) return false;
        out.kv_page_indptr = detail::as_u32(b);
    }
    if (has[kPortPages]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortPages], b, err)) return false;
        out.kv_page_indices = detail::as_u32(b);
        // CSR-prefix: trim the fixed-shape data port to page_indptr's last entry.
        if (!out.kv_page_indptr.empty()) {
            const std::uint32_t nnz_pages = out.kv_page_indptr.back();
            if (nnz_pages <= out.kv_page_indices.size())
                out.kv_page_indices.resize(nnz_pages);
        }
        out.has_kv_family = true;
    }
    if (has[kPortKvLen]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortKvLen], b, err)) return false;
        for (std::uint32_t len : detail::as_u32(b))
            out.kv_last_page_lens.push_back(detail::last_page_len(len, page_size));
    }

    // -- read-out --
    if (has[kPortReadout]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortReadout], b, err)) return false;
        out.sampling_indices = detail::as_u32(b);
        out.sampling_indptr = {0, static_cast<std::uint32_t>(out.sampling_indices.size())};
    } else {
        out.sampling_indptr.push_back(0);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (out.qo_indptr[lane + 1] > out.qo_indptr[lane]) {
                out.sampling_indices.push_back(out.qo_indptr[lane + 1] - 1);
            }
            out.sampling_indptr.push_back(
                static_cast<std::uint32_t>(out.sampling_indices.size()));
        }
    }

    // -- explicit KV write descriptor (w_slot/w_off → write_kv_explicit) --
    if (has[kPortWSlot]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortWSlot], b, err)) return false;
        out.w_page = detail::as_u32(b);
        out.has_write_desc = true;
    }
    if (has[kPortWOff]) {
        std::vector<std::uint8_t> b;
        if (!detail::read_port_cell(view, ch[kPortWOff], b, err)) return false;
        out.w_off = detail::as_u32(b);
    }

    // -- dense attention mask (→ pack_dense_mask) --
    if (has[kPortAttnMask]) {
        if (!detail::read_port_cell(view, ch[kPortAttnMask], out.mask, err)) return false;
        out.has_mask = true;
    }
    return true;
}

}  // namespace pie_cuda_driver::ptir
