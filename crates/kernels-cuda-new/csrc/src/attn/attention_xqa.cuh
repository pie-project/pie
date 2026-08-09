//===-- attention_xqa.cuh - XQA's metadata build, split out -----*- CUDA -*-===//
//
// ONE `__global__`: `build_xqa_metadata`, the per-request page table and
// sequence-length build that XQA's decode reads back out of the attention
// workspace. It was `build_xqa_metadata_kernel` in
// `kernels-cuda/csrc/src/attn/attention_xqa.cu:55-83` -- **the last
// `__global__` in that archive** -- and it is the only device text that file
// ever held. No host code here.
//
// # A TOTAL split, and why this one deletes its `.cu` half
//
// `attn/attention_flashinfer.cuh` next door is a PARTIAL split: its `.cu`
// keeps three private kernels and includes this header for the fourth, so one
// text has two compilers. This file is the other case. The launcher --
// `prepare_attention_xqa_decode_bf16` -- was 44 lines of workspace arithmetic
// and a single `<<<num_requests, 128, 0, stream>>>`, it had **no shim entry
// and no C++ caller**, and its whole justification was that it was the only
// implementation of the `Prepare::FireWide` that
// `kernels-cuda-new/src/table/attn.rs`'s `attn::attention_xqa_decode_bf16_prepared`
// states. That implementation is now `driver-cuda/src/fire/xqa.rs`, in Rust,
// firing this text through NVRTC -- so the launcher is DELETED rather than
// kept, and `attention_xqa.cu` holds no device text and no launch at all.
//
// # What the kernel does
//
// XQA's decode entry point does not read a CSR. It wants a DENSE page table
// -- `num_requests * page_bucket` signed indices, one row per request, zero
// padded -- and a flat `seq_lens` array. The paged KV cache carries neither:
// it carries `kv_page_indptr` (a request's page range), `kv_page_indices` (the
// pages themselves, ragged) and `kv_last_page_lens` (how much of the final
// page is live). This kernel is the transform between those two shapes, and it
// is ours: FlashInfer's XQA csrc has no equivalent.
//
// One block per request on `grid.x`. `blockDim.x` is a pure STRIDE -- the page
// loop is `for (p = threadIdx.x; p < max_pages_per_seq; p += blockDim.x)` and
// the sequence length is written by lane 0 alone -- so every block width
// computes the same bytes. That is the property that makes the geometry
// unstatable rather than merely unstated; `families/attn.rs`' `ATTN_XQA_SIGS`
// carries the argument and `driver-cuda/src/fire/xqa.rs` carries the 128 with
// the deleted `<<<>>>`'s line number beside it.
//
// # `max_pages_per_seq` is the BUCKET, not the request's page count
//
// The launcher never passed the caller's `max_pages_per_seq` here. It passed
// `xqa_decode_page_bucket(max_pages_per_seq)` -- the next power of two at or
// above it, clamped at 4096 -- because the page table's row STRIDE has to
// match the one the decode dispatch computes when it reads the table back, and
// the decode hands XQA `page_bucket * page_size` as the maximum sequence
// length. Rounding to a power of two keeps that stride stable across the small
// per-step changes in `max_pages_per_seq` that would otherwise re-shape the
// buffer on every fire. The parameter is named `max_pages_per_seq` here
// because that is what it means TO THE KERNEL -- the width of the row it
// fills -- and the bucketing is a host decision that stays on the host.
//
// **The three buffers this writes into are carved out of
// `AttentionWorkspaceView::float_buffer`**, in the order `page_table`,
// `seq_lens`, then a 256-byte-aligned scratch the decode hands to XQA. The
// arithmetic is not here because it is host arithmetic, and it must agree
// digit for digit with the four surviving `detail::launch_attention_xqa_
// decode_bf16_gqa*_prepared` bodies, which recompute it themselves before
// reading the table back. `driver-cuda/src/fire/xqa.rs`' `carve` is the
// statement of it, with each `.cu` line cited.
//
// # Zero padding is load-bearing
//
// `p >= pages` writes `0`, not garbage and not a sentinel. XQA indexes the
// table for every column up to the row stride regardless of how many pages a
// request actually holds, and bounds it by `seq_lens[r]` instead -- so page 0
// is read and its contribution discarded. A row left uninitialised would have
// XQA dereference whatever the workspace held, which is a fault or a silent
// wrong answer depending on the last fire's allocation.
//
// # No external include, because NVRTC answers none
//
// The prelude and nothing else. The `.cu` this came from opened with
// `<algorithm>`, `<cmath>`, `<cstdint>`, `<cstdlib>`, `<stdexcept>`,
// `cuda_check.hpp` and -- through `#include <xqa/mha.cu>` -- the whole of
// FlashInfer's XQA translation unit. This kernel needed none of it: three
// integer widths from the prelude, and `std::size_t` for one index expression,
// which is `usize` here for the reason every other migrated header gives.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

// Pulled in by name rather than by `using namespace`, exactly as
// `attention_flashinfer.cuh` does and for the same reason: inside
// `attn::device` the `device::` qualifier resolves to THIS namespace, so a
// prelude name not re-exported here would stop resolving in any `.cu` that
// includes the header.
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::usize;

/// One block per request: scatter the request's ragged page list into a dense
/// zero-padded row, and reduce its page range to one sequence length.
///
/// `blockIdx.x` is the request. `threadIdx.x` strides the row, so the block
/// width is a launch stride and not an extent -- see the header.
__global__ void build_xqa_metadata(
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    i32* __restrict__ page_table,
    u32* __restrict__ seq_lens,
    int num_requests,
    int max_pages_per_seq,
    int page_size)
{
    const int r = static_cast<int>(blockIdx.x);
    if (r >= num_requests) return;
    const int page_lo = static_cast<int>(kv_page_indptr[r]);
    const int page_hi = static_cast<int>(kv_page_indptr[r + 1]);
    const int pages = page_hi > page_lo ? (page_hi - page_lo) : 0;
    for (int p = static_cast<int>(threadIdx.x); p < max_pages_per_seq;
         p += static_cast<int>(blockDim.x)) {
        const u32 src = (p < pages) ? kv_page_indices[page_lo + p] : 0u;
        page_table[static_cast<usize>(r) * static_cast<usize>(max_pages_per_seq) +
                   static_cast<usize>(p)] = static_cast<i32>(src);
    }
    if (threadIdx.x == 0) {
        const u32 last = pages > 0 ? kv_last_page_lens[r] : 0u;
        seq_lens[r] = pages > 0
            ? static_cast<u32>((pages - 1) * page_size) + last
            : 0u;
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
