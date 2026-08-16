//===-- attention_flashinfer.cuh - the score fold, split out ----*- CUDA -*-===//
//
// ONE `__global__`: `attn_score_fold_heads`, the per-head → per-request fold
// of a captured attention-score buffer. No host code. `attention_flashinfer.cu`
// includes this and keeps its `<<<>>>`, so the ahead-of-time build and NVRTC
// compile ONE text -- which is the whole point of the split, because two
// copies that agree today are two kernels that drift, each right for whichever
// half of the tree its tests exercise. `attn/kv_paged.cu` shipped fourteen of
// those for a week with every gate green (`new-horizon.md` §21.7), and the
// gate that could not see them compared NAMES: a split renames, so the two
// copies never shared one.
//
// # A PARTIAL split, and what stayed behind
//
// `attention_flashinfer.cu` is 1,674 lines of plan-cache plumbing, head_dim
// dispatch driven by `src/kernels.def`, and calls INTO FlashInfer, which does
// its own launching. It defines four `__global__`s and exactly one of them
// leaves:
//
//  * `k_attn_score_fold_heads` (`:811` before this edit) is `attn_score_fold_heads`
//    below. It is the only one a TABLE ROW names --
//    `kernels-cuda/src/table/attn.rs`'s `attn::attn_score_fold_heads` --
//    so it is the only one a JIT unit could ever be asked to compile.
//  * `k_attn_score_normalize`, `k_attn_prefill_score_normalize` and
//    `k_attn_prefill_score_fold` stay. Each is an internal step of a
//    FlashInfer DISPATCH -- `dispatch_attention_flashinfer_decode_capture_bf16`
//    launches the first immediately after `AttnHd<HD>::dispatch_decode_capture`
//    returns, on the same stream and over the same ragged CSR the vendored
//    kernel just filled. No symbol in `kernels::table` names any of them,
//    nothing outside this translation unit can call them, and a `.cuh` for a
//    kernel no row can name is text nobody compiles. They move when something
//    asks for them.
//
// So the file keeps its FlashInfer content and its three private kernels, and
// loses exactly the one that was blocked on being in a `.cu` with no `.cuh`.
//
// # What the kernel does
//
// Fold the per-head probability rows into one row per request.
//
// Eviction here is necessarily a per-REQUEST decision: the paged KV layout
// carries a single page list per request, so a per-head keep-set has nowhere
// to live. Quest already makes (and documents) the same collapse. Averaging
// rather than summing keeps the folded row a probability distribution -- it
// sums to 1 over the live prefix -- so a policy can threshold it in absolute
// terms.
//
// The folded CSR is not a second array: `score_indptr[r]` counts
// `num_q_heads * kv_len(r')` elements for every earlier request, so dividing
// it by `num_q_heads` is exactly the folded offset. Deriving it removes the
// chance of two CSRs disagreeing.
//
// It calls nothing. There are no `__device__` helpers to bring along, and the
// only names it needs beyond the language are three integer widths from the
// prelude.
//
// # No external include, because NVRTC answers none
//
// The prelude and nothing else. The `.cu` this came from opens with
// `<atomic>`, `<cstdio>`, `<cstdlib>`, the whole of `attention_flashinfer_common.cuh`
// and eleven FlashInfer headers; NVRTC resolves an `#include` against the
// carried header set alone and would answer none of them. `std::int32_t`,
// `std::uint32_t` and `std::size_t` are therefore `i32`, `u32`
// and `usize` here -- the same three types under nvcc, spelled so
// that `<cstdint>` is not needed to say them.
//
// There is no `#ifdef __CUDACC_RTC__` anywhere below, and §14.3 is why: the
// guard is exactly the thing that lets the two arms drift, and one text
// compiled twice is the property this file exists to have.
//
// # Linkage: SINGLE-INCLUDER, and it is not a template
//
// §21.6's measurement applies verbatim -- a `.cuh` holding a non-template
// `__global__` may be included by exactly ONE translation unit, because the
// host stub and the function both take external linkage and a second includer
// is a hard `multiple definition` at link even when it never launches it.
// Measured again for this header on this box, nvcc 13.0.88: two translation
// units that both include it fail at link with
//
//     multiple definition of `pie::attn::
//         attn_score_fold_heads(float const*, int const*, unsigned int const*,
//         unsigned int const*, int, int, float*)'
//
// plus the same for `__device_stub__…`, and the second unit launched nothing.
//
// The permitted includer is `attn/attention_flashinfer.cu`. It is the only
// file in the tree that launches this kernel -- one `<<<>>>`, at
// `attention_flashinfer.cu:829` -- and the SHARED text of this driver already
// has a home: `attention_flashinfer_common.cuh` is what the six
// `attention_flashinfer_hd<N>.cu` translation units include, and it is a
// different file on purpose. Nothing here belongs to a head_dim.
//
// THAT INCLUDER IS GONE, and so are the six. `driver-cuda/csrc` was deleted
// wholesale and every `attention_flashinfer_hd<N>.cu` with it, which leaves
// this header at ZERO includers and the "exactly one" rule satisfied
// vacuously. The measurement stands -- it was taken on this box and the link
// error is quoted above -- but it is now an argument with no subject, and the
// address in it is an invitation to recreate a deleted file. Do not.
//
// `attention_flashinfer_common.cuh` outlived its own six includers the same
// way and then went further. With no compiler left to answer to it was moved
// out of `csrc/` to `kernels-cuda/spec/` -- a directory whose whole promise
// was that no build step reads it -- and kept there for the sake of the
// citations that pointed into it. It has since been deleted with that
// directory. The `attention_flashinfer_common.cuh:NNN` citations survive it,
// in `attn/fa2.cuh`'s banner and across `kernels_cuda::attn::fa2`, and both
// of those say how to read one now: as provenance beside content that is
// written out, and not as a file to open.
//
// It stays a non-template because it has no honest parameter, which is
// `mla_paged.cuh`'s position on `write_mla` and `mxfp4_marlin.cuh`'s in the
// words *"a width parameter would be a lie that compiles."* Every buffer is
// `float` or page-table metadata; there is no element type to abstract over.
// The block width reaches the body as `blockDim.x` and the y-fanout as
// `gridDim.y`, both read at run time by a stride loop -- so a `template <int
// BLOCK>` would be a parameter the body never mentions, an arm that cannot
// differ from its sibling, and therefore a body change with no negative
// control available to it. §8 asks for evidence that a body change is inert;
// a parameter whose two instantiations are provably the same text cannot
// produce any.
//
// # No row, and the rule it would need
//
// The launcher is `attention_flashinfer.cu:828-829`:
//
//     const dim3 grid(static_cast<unsigned>(num_requests), 64u);
//     attn_score_fold_heads<<<grid, 256, 0, stream>>>(
//
// **`dim3(requests, 64)` at 256 threads, no shared memory.** Every rule in
// `kernels::LaunchRule` was read against it -- 40 variants of which
// `Unstated` is the ABSENCE of a rule, so 39 rules, the vocabulary having
// grown by `Single` (`<<<1, 256>>>`), `SingleWarp` (`<<<1, 32>>>`) and
// `PerRequest` while this split was being written -- and none states that
// shape. Three near misses, and
// each is the kind that does not announce itself:
//
//  * **`LaunchRule::PerRequest` is the nearest and it is one number away.**
//    `dim3(requests)` at 256 with nothing shared: `grid.x` is right, the
//    block is right, the absent smem is right, and `grid.y` is ONE where the
//    launcher writes 64. The body strides `i += blockDim.x * gridDim.y`, so
//    at `gridDim.y == 1` it still covers every key and still computes the
//    same float -- the difference is 64x fewer blocks and NOTHING ELSE. A
//    rule that is wrong only in latency is one no test in this tree could
//    fail on, which is the `combine_attn_outputs` argument `families/attn.rs`
//    makes and the reason this is a refusal rather than a row.
//  * `LaunchRule::PerRow` is that same shape read off the wrong axis --
//    `dim3(rows)`. A request count is not a row count (`Dims::requests`' own
//    doc), so on a prefill it would open a block per token over a buffer with
//    one entry per request.
//  * `LaunchRule::PagedScores` is the only other rule that opens a grid over
//    REQUESTS, and it is `dim3(requests, rows, q_heads)` at 128 with
//    `(head_dim + 128) * sizeof(float)` of dynamic shared memory. Three axes,
//    half the block, and a shared allocation this kernel does not want.
//
// The rule it needs is `LaunchRule::PerRequest` WITH A FIXED Y-FANOUT --
// `dim3(requests, 64)` at 256, nothing shared -- where the 64 is a launch
// constant of the grid-stride loop and not a dimension of anything. That is
// what no rule here carries: `Slab`'s cap of 1024 is the only other literal
// in the vocabulary and it caps a COMPUTED extent rather than standing in for
// one. Whoever adds it states that 64 and cites `attention_flashinfer.cu:829`
// for it, `runtime::launch`'s law being that a rule with no cited launcher is
// a guess. `tests/launch_rules.rs::mod transcribed::pins` pins the two lines
// above so the citation cannot rot.
//
// What is NOT a second refusal, any more: `Dims::requests` used to be zero at
// every call site `driver-cuda` had, and `bind/mod.rs` now fills it from
// `AttnCtx::num_requests`. A fire with no attention context still leaves it
// absent, and every reader of that field refuses on a zero.
//
// No unit is declared for this header, and that is deliberate rather than
// pending: `tests/units.rs::verdict` hard-fails a unit that declares no rows,
// because a cubin nothing can fire is cached under an architecture and
// satisfies nobody.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

/// One block per request on `grid.x`, the key axis split over `grid.y`:
/// average the `num_q_heads` score rows of a request into one folded row.
///
/// `blockIdx.y` and `gridDim.y` are a launch fanout, not an extent -- the
/// loop below covers `kv_len` for any `gridDim.y >= 1`.
__global__ void attn_score_fold_heads(
    const float* __restrict__ scores,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size,
    int num_q_heads,
    float* __restrict__ folded)
{
    const int request = static_cast<int>(blockIdx.x);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0 || num_q_heads <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;

    const usize base = static_cast<usize>(score_indptr[request]);
    const float* rows = scores + base;
    float* out = folded + base / static_cast<usize>(num_q_heads);
    const float inv_heads = 1.f / static_cast<float>(num_q_heads);

    for (int i = static_cast<int>(threadIdx.x) +
                 static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x);
         i < kv_len;
         i += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y)) {
        float total = 0.f;
        for (int h = 0; h < num_q_heads; ++h) {
            total += rows[static_cast<usize>(h) *
                              static_cast<usize>(kv_len) +
                          static_cast<usize>(i)];
        }
        out[i] = total * inv_heads;
    }
}

}  // namespace pie::attn
