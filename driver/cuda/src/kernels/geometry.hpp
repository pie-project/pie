#pragma once

// PTIR M5 — Geometry as data, C1 final form (overview §5.1 / thrust-1-memory §5.M5).
//
// C1 has two forms:
//   * INTERIM — the host computes forward geometry (pages / kv_len / mask /
//     w_slot / w_off) and writes it into device buffers (the wire ForwardRequest
//     columns). Shipped: `kv_len` is a first-class named column
//     (schema.rs ForwardRequest::kv_len; the host derives it in
//     append_request_with_options).
//   * FINAL — the geometry is produced by a PREVIOUS PASS's KERNEL directly into
//     a device buffer the HOST NEVER READS, and the consuming forward binds that
//     device buffer instead of a host-fed scalar column (ForwardRequest
//     `kv_len_device`, the u64 device-handle column). The handshake exit
//     (thrust-1-memory §5.M5): a forward whose geometry was device-produced
//     matches the host-fed run BIT FOR BIT.
//
// This kernel is the FINAL-form producer for the length column: it derives
// `kv_len[r]` on-device from the paged-KV descriptor family the KV working-set
// projection already lays out (`kv_page_indptr` + `kv_last_page_lens`), exactly
// mirroring the host formula in `runtime/src/inference/request.rs`:
//
//     page_count = kv_page_indptr[r+1] - kv_page_indptr[r]
//     kv_len[r]  = page_count == 0 ? 0
//                                  : (page_count - 1) * page_size
//                                        + kv_last_page_lens[r]
//
// Frozen fork pages are counted FULL (W6): sub-page validity rides the attention
// mask, never this total. The parity test (`test_kv_len_geometry`) asserts the
// device-produced `kv_len` equals the host formula bit-for-bit over randomized
// page geometries — the standalone half of the M5 handshake (the full
// forward-binds-device-buffer integration rides the executor's late-bind
// read-seam, extended per thrust-1-memory §5.M5 item 1).

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Derive the per-request length column `kv_len[r]` (device-resident, C1 FINAL)
// from the paged-KV descriptors. `kv_page_indptr` is `[R+1]` (CSR page bounds,
// one leading 0), `kv_last_page_lens` is `[R]` (valid tokens in each request's
// final active page). `page_size` is the tokens-per-page cap. Writes `kv_len`
// `[R]`. One thread per request; bit-identical to the host reference.
void launch_derive_kv_len(
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::uint32_t page_size,
    std::uint32_t num_requests,
    std::uint32_t* kv_len,
    cudaStream_t stream);

// PTIR M5 / C1-FINAL — device-side working-set SLOT → PHYSICAL page-pool BlockId
// resolution, for the beam/§6.1 device-produced `pages` geometry (G2). The beam
// epilogue produces a `pages` channel of working-set SLOT ids into a device
// buffer the host never reads; the attention kernel needs PHYSICAL `kv_page_index`
// BlockIds (`arena/pool.rs` device-pool indices). The host resolution
// (`KvWorkingSet::resolve_read` → `Arena::blocks`, forward_prepare.rs) can't run
// without a host round-trip that defeats the run-ahead loop, so this resolves
// on-device: the runtime uploads a compact `slot_to_block` `[num_slots]`
// dictionary (host-authoritative, ONE per forward, NOT the per-beam geometry —
// so `pages` stays host-unread), and this kernel gathers
// `page_indices[i] = slot_to_block[pages[i]]` for the `count` (`[B·P]` flattened,
// padding dropped by np[b]) slot ids. Slot id 0 is a VALID slot — indexed
// directly, never special-cased. Mirrors `launch_derive_kv_len`: the read-seam
// binds this at the port boundary (co-design with the executor).
void launch_resolve_slot_to_block(
    const std::uint32_t* pages,
    const std::uint32_t* slot_to_block,
    std::uint32_t num_slots,
    std::uint32_t count,
    std::uint32_t* page_indices,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
