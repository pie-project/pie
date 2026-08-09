//===-- pack_dense_mask.cuh - the custom-mask packers' device text --------===//
//
// Two `__global__`s and no host code at all. `pack_dense_mask.cu` includes
// this and keeps both `<<<>>>`, so the ahead-of-time build and NVRTC compile
// ONE text -- which is the whole point of the split, because two copies that
// agree today are two kernels that drift, each right for whichever half of
// the tree its tests exercise. `norm/altup_aux` shipped exactly that for a
// release with every test green.
//
// # What these compute
//
// FlashInfer's custom-mask prefill (`MaskMode::kCustom`) wants a BIT-PACKED
// `[qo_len x kv_len]` bitmap per request -- bit `q*kv_len + j`, read as
// `packed[bit/8] >> (bit%8) & 1` -- at a per-request BYTE offset carried in
// `mask_indptr`. Two things produce one:
//
//  * `pack_dense_mask` reads a program's DENSE `[TOTAL_Q, STRIDE]` byte-per-
//    cell mask -- what an `AttnMask` descriptor port emits -- and packs it.
//    The physical span `klen[l]` is contiguous in the logical grid because
//    pages are laid out page-major and `klen` counts only the live prefix, so
//    the logical index maps 1:1 to the mask column, mid-page holes included.
//  * `pack_structured_mask` materialises causal / sliding-window / sink
//    descriptors straight into the same ABI, with no dense tensor in between.
//
// Both give each thread whole output BYTES, which is not a tiling preference:
// a bit is not addressable, so two threads sharing a byte would be a
// read-modify-write race on every boundary.
//
// # `StructuredMaskParams` is mirrored here, and the mirror is CHECKED
//
// The struct lives in `pack_dense_mask.hpp`, which is a HOST interface
// header: `kernels-cuda/build.rs` compiles the generated `shim.cpp` with the
// host compiler against `csrc/src`, so the declaration of
// `attn::pack_structured_mask` and the `Ty::StructuredMasks` operand behind
// it must both stay reachable from plain C++. `new-horizon.md` §10.5's rule
// is that `.hpp` files do not convert, and NVRTC has no include path that
// could reach one anyway.
//
// `quant/mxfp4_marlin.cuh` met the same wall with an ENUM and answered it by
// taking the underlying type and mirroring the three values as named
// constants -- *"the mapping between this file and `mxfp4_marlin.hpp` is one
// grep and not three."* A struct has no underlying type to take, so the
// mirror is the struct, and the grep is replaced by something a reader cannot
// forget to run: `pack_dense_mask.cu` includes BOTH definitions and
// `static_assert`s size, alignment and all three field offsets against each
// other. A field added, reordered or widened on either side fails the
// ahead-of-time build with the two spellings named in the message. That is
// strictly more than the precedent, and it is what makes a second definition
// of a three-`u32` POD acceptable where a second definition of a kernel body
// never is.
//
// # `UINT32_MAX` became `0xffffffffu`
//
// The macro is `<cstdint>`'s, and NVRTC answers no external include -- 0 of
// 31, not even that one. The literal is the same value and the saturating
// compare it guards (`window > 0xffffffffu - key`) is unchanged: it exists
// because `key + window` overflows for the sink kind's open window, and an
// overflowed sum wraps to a small number, which reads as a CLOSED window and
// silently masks out the tokens it was meant to admit.
//
// # Which launcher becomes a row, and which does not
//
// BOTH DO, NOW. This section used to open *"Neither, and for two independent
// reasons that both have to move first"*, and it closed by saying the rows
// *"wait on a real argument or on an `instantiation()` that needs none."*
// Both reasons moved, and the second moved in exactly the way that sentence
// predicted. It is rewritten rather than deleted because the two halves came
// apart, and which half survived is the useful part.
//
//  * **Geometry — answered.** Both launch `<<<B, 128>>>`: one block per LANE,
//    a fixed 128 threads, a stride loop over that lane's output bytes. No
//    rule stated it, and naming `Rms` (one block per row, but at 256, asking
//    32 bytes of shared memory, and MEANING a reduction) or `Elementwise`
//    (`ceil(n/256)` over an extent nobody here has, since the byte count is
//    per-lane and derived on the device) would have been inventing a rule
//    under an existing name. `LaunchRule::PerRowNarrow` was ported from
//    `vision/gemma4_audio.cu:189` and states `rows x 128` exactly, so both
//    launchers are now reproduced digit for digit rather than approximated.
//
//  * **Instantiation — answered, and it was never NVRTC's limit.** This text
//    said `instantiation()` *"emits `path<elem>` and cannot name a plain
//    `__global__` at all"*. The first half was true of the emitter; the
//    second half was an inference about NVRTC, and it was measured false:
//    `nvrtcAddNameExpression` on a bare qualified path lowers a plain
//    `__global__` to `_ZN15pie_cuda_driver7kernels...` and
//    `cuModuleGetFunction` resolves it. A row states `DeviceKernel::PLAIN`
//    and `instantiation()` emits the path with no argument list. NVRTC
//    enforces the distinction in BOTH directions -- a template named with no
//    list is *"cannot determine which instance"*, a plain kernel named with
//    one is *"expected an expression"* -- so a row that gets it backwards
//    fails in `tests/units.rs` with NVRTC's own sentence.
//
//    So no parameter was invented here, which is what `mxfp4_marlin.cuh`
//    refused for `mxfp4_weight_to_gptq_w4` in the words *"a width parameter
//    would be a lie that compiles."* That refusal stands and was never the
//    obstacle it looked like.
//
// # What did NOT move: this header still has exactly one includer
//
// Naming and linkage were the same fact in the paragraph above and are two
// facts now. A non-template `__global__` in a `.cuh` still takes external
// linkage for itself and its host stub, so **this header may be included by
// exactly one translation unit**, which is `pack_dense_mask.cu`. A second
// includer is a hard `multiple definition` at link even when it never
// launches either kernel -- measured with nvcc on this box, not inferred.
//
// A header that wants a second includer should take a defaulted template
// parameter FOR THAT REASON and say so, the way `quant/dequant_wna16.cuh`
// and `layout/envelope.cuh` already do. That is now a linkage decision made
// on its own evidence, rather than a workaround for a naming gap.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

// Pulled in by name rather than by `using namespace`, so that `device::` here
// and in `pack_dense_mask.cu` means the same thing: inside `attn::device` the
// qualifier resolves to THIS namespace, and a prelude name not re-exported
// here would stop resolving in the `.cu` the moment it includes this header.
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::u64;
using ::pie_cuda_driver::kernels::device::u8;

/// The device mirror of `attn::StructuredMaskParams`, checked field for field
/// by `pack_dense_mask.cu`. See the header comment for why there are two.
///
/// `kind` is `StructuredMaskKind`'s numeric value: 1 causal, 2 sliding
/// window, 3 sink. It is read as a number rather than as an enum for the same
/// reason `mxfp4_marlin.cuh` takes an `int` -- the enum is host vocabulary
/// and a device header that restated it would be a second definition of a
/// contract, which is the drift this split exists to prevent.
struct StructuredMaskParams {
    u32 kind;
    u32 window;
    u32 sink;
};

/// One block per lane. Each thread packs a strided subset of the lane's
/// `qo_len * klen` bits.
///
/// `kvm_dense` is `[TOTAL_Q, STRIDE]` with one byte per cell (0/1).
/// `mask_indptr` is the per-lane BYTE offset into `packed` (`[LANES+1]`,
/// prefix-summed on the host from `ceil(qo_len[l]*klen[l]/8)`). `qo_indptr`
/// (`[LANES+1]`) gives each lane's query-row range. `packed` is pre-zeroed.
__global__ void pack_dense_mask(
    const u8* __restrict__ kvm_dense,    // [TOTAL_Q, STRIDE] bytes (0/1)
    const u32* __restrict__ klen,        // [LANES] physical span per lane
    const u32* __restrict__ qo_indptr,   // [LANES+1] query-row CSR
    const i32* __restrict__ mask_indptr, // [LANES+1] byte offsets
    u8* __restrict__ packed,             // out: bit-packed, pre-zeroed
    int B,
    int P_PAGE)                          // STRIDE (logical row stride)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int kl = static_cast<int>(klen[b]);
    const int qo_lo = static_cast<int>(qo_indptr[b]);
    const int qo_len = static_cast<int>(qo_indptr[b + 1]) - qo_lo;
    if (kl <= 0 || qo_len <= 0) return;
    const long long total_bits =
        static_cast<long long>(qo_len) * static_cast<long long>(kl);
    u8* out = packed + mask_indptr[b];
    // Each thread owns whole output BYTES to avoid RMW races on shared bytes.
    const int nbytes = static_cast<int>((total_bits + 7) / 8);
    for (int byte = threadIdx.x; byte < nbytes; byte += blockDim.x) {
        u8 acc = 0;
        const long long base = static_cast<long long>(byte) * 8;
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const long long gbit = base + bit;
            if (gbit < total_bits) {
                const int qi = static_cast<int>(gbit / kl);
                const int col = static_cast<int>(gbit % kl);
                const u8* row =
                    kvm_dense + static_cast<long long>(qo_lo + qi) * P_PAGE;
                if (row[col] != 0) acc |= static_cast<u8>(1u << bit);
            }
        }
        out[byte] = acc;
    }
}

/// One block per request: causal / sliding-window / sink descriptors straight
/// into the packed bitmap, with no dense tensor in between.
__global__ void pack_structured_mask(
    const u32* __restrict__ positions,
    const u32* __restrict__ klen,
    const u32* __restrict__ qo_indptr,
    const i32* __restrict__ mask_indptr,
    const StructuredMaskParams* __restrict__ masks,
    u8* __restrict__ packed,
    int B) {
    const int request = blockIdx.x;
    if (request >= B) return;
    const u32 keys = klen[request];
    const u32 query_begin = qo_indptr[request];
    const u32 queries =
        qo_indptr[request + 1] - query_begin;
    const u64 bits =
        static_cast<u64>(queries) * keys;
    const auto descriptor = masks[request];
    u8* output = packed + mask_indptr[request];
    const u32 bytes =
        static_cast<u32>((bits + 7) / 8);
    for (u32 byte = threadIdx.x;
         byte < bytes;
         byte += blockDim.x) {
        u8 value = 0;
        const u64 begin =
            static_cast<u64>(byte) * 8;
        #pragma unroll
        for (u32 bit = 0; bit < 8; ++bit) {
            const u64 index = begin + bit;
            if (index >= bits) break;
            const u32 query =
                static_cast<u32>(index / keys);
            const u32 key =
                static_cast<u32>(index % keys);
            const u32 position =
                positions[query_begin + query];
            // `key + window` overflows for the sink kind's open window, and
            // an overflowed sum wraps small -- which reads as a CLOSED window
            // and masks out exactly the tokens it was meant to admit. So the
            // sum saturates. `0xffffffffu` is `<cstdint>`'s `UINT32_MAX`,
            // written as a literal because NVRTC answers no include.
            const u32 key_plus_window =
                descriptor.window > 0xffffffffu - key
                    ? 0xffffffffu
                    : key + descriptor.window;
            const bool causal = key <= position;
            const bool in_window =
                causal && key_plus_window > position;
            const bool allowed = causal &&
                (descriptor.kind == 1 ||
                 (descriptor.kind == 2 && in_window) ||
                 (descriptor.kind == 3 &&
                  (key < descriptor.sink || in_window)));
            if (allowed) {
                value |= static_cast<u8>(1u << bit);
            }
        }
        output[byte] = value;
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
