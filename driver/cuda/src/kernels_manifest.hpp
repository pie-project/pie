//===-- kernels_manifest.hpp - C++ view of kernels.def ---------*- C++ -*-===//
//
// Host-side queries over the head_dim list in kernels.def, for the code that
// has to agree with what the attention kernels were built for -- config
// normalization, workspace sizing, and the fast-path admission checks.
//
// This header exists so those call sites cannot drift: each of them used to
// carry its own hand-written `hd == 64 || hd == 128 || ...`, which is exactly
// the duplication kernels.def is meant to remove. Deliberately free of CUDA so
// plain .cpp translation units can include it.
//
//===----------------------------------------------------------------------===//
#pragma once

namespace pie_cuda_driver {

/// Whether the attention kernels were instantiated for this head_dim.
constexpr bool attn_head_dim_instantiated(int head_dim) {
    return false
#define PIE_ATTN_HEAD_DIM(HD) || head_dim == (HD)
#include "kernels.def"
        ;
}

/// Whether the paged-decode kernels were instantiated for this GQA ratio
/// (num_q_heads / num_kv_heads). Ratios outside the set are routed through
/// the prefill kernel instead.
constexpr bool attn_decode_gqa_instantiated(int gqa) {
    return false
#define PIE_ATTN_DECODE_GQA(G) || gqa == (G)
#include "kernels.def"
        ;
}

/// Smallest instantiated head_dim that can hold `head_dim`, or `head_dim`
/// itself when none can -- callers then surface the dispatch error rather than
/// silently mis-sizing. Padding up is safe because the extra lanes are zero-
/// filled; it is what lets Phi-3-mini's 96 run on the 128 kernels.
constexpr int round_up_attn_head_dim(int head_dim) {
    int best = 0;
#define PIE_ATTN_HEAD_DIM(HD) \
    if ((HD) >= head_dim && (best == 0 || (HD) < best)) best = (HD);
#include "kernels.def"
    return best != 0 ? best : head_dim;
}

/// Space-separated instantiated head_dims, for error messages.
constexpr const char* attn_head_dim_list() {
    return
#define PIE_ATTN_HEAD_DIM(HD) " " #HD
#include "kernels.def"
        ;
}

/// Space-separated GQA group sizes the decode path was instantiated for.
constexpr const char* attn_decode_gqa_list() {
    return
#define PIE_ATTN_DECODE_GQA(G) " " #G
#include "kernels.def"
        ;
}

/// Whether a native MXFP4 expert GEMM is the right way to serve an MXFP4 MoE
/// on a device of this compute capability.
///
/// Not a hardware question. Blackwell has the FP4 unit, but on everything
/// older the answer is decided by which Marlin the driver was built with:
///
///   * the expert-indexed MoE kernel serves a whole layer in one launch, and
///     is sm80 -- with it the native path wins on Ampere onward;
///   * the dense, single-problem kernel alone does not qualify. It reaches the
///     tensor cores once per expert, so a 32-expert layer becomes 32 serial
///     launches: measured 67 tok/s against 747 for the routed dequant path it
///     would replace.
///
/// Two call sites must agree on this or a model fails to load with a message
/// about the target's capabilities: `context.cpp` publishes it as the
/// `native_mxfp4_moe` device fact the loader plans against, and
/// `loaded_model.cpp` passes the same bit into `DeviceTarget`. They disagreed
/// once already, and the symptom was a checkpoint that quietly took the slow
/// path with the fast kernels compiled in and unused.
constexpr bool device_supports_native_mxfp4_moe(int cc_major) {
#if defined(PIE_CUDA_HAS_MARLIN_MOE)
    return cc_major >= 8;
#elif defined(PIE_CUDA_HAS_MARLIN)
    return cc_major >= 10;
#else
    (void)cc_major;
    return false;
#endif
}

}  // namespace pie_cuda_driver
