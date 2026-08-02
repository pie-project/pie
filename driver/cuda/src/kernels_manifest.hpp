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

#include <cstdlib>

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
// The native MXFP4 lowering is what makes the Marlin MoE reachable, and it is
// exclusive: the routed-dequant slabs the per-route decode GEMV reads are not
// published alongside it, so choosing one gives up the other for the life of
// the model. Which one wins depends on the batch, not on the device --
// `driver/cuda/bench/moe_bench.cu` at gpt-oss's shape on an H100 measures the
// routed GEMV 2x AHEAD at one row, level around six, and 2.6x behind by
// thirty-two. A throughput fleet wants Marlin; a latency-bound single stream
// wants the GEMV, and on Hopper the difference end to end is 253 tok/s against
// 197. `PIE_CUDA_NATIVE_MXFP4_MOE=0` is how a deployment says which it is.
inline bool native_mxfp4_moe_opt_out() {
    static const bool off = [] {
        const char* v = std::getenv("PIE_CUDA_NATIVE_MXFP4_MOE");
        return v != nullptr && v[0] == '0';
    }();
    return off;
}

constexpr bool device_supports_native_mxfp4_moe(int cc_major) {
// The expert-indexed MoE GEMM is only half of what this answer promises.
// Saying yes makes the LOADER plan a marlin MXFP4 lowering, and that
// lowering's repack lives behind `PIE_CUDA_HAS_MARLIN`
// (`loader/transcode_engine.hpp`) with no MoE-specific alternative. The
// two options default apart — `PIE_CUDA_BUILD_MARLIN_MOE` is ON,
// `PIE_CUDA_BUILD_MARLIN` is OFF — so a DEFAULT build answered yes here
// and then threw "MarlinMxfp4Weight Repack requires Marlin" at load,
// which is to say it planned something it could not execute. Ask for
// both, and a build with only the MoE half falls back to the routed
// decode path instead of failing to load an MXFP4 checkpoint at all.
#if defined(PIE_CUDA_HAS_MARLIN_MOE) && defined(PIE_CUDA_HAS_MARLIN)
    return cc_major >= 8;
#elif defined(PIE_CUDA_HAS_MARLIN)
    return cc_major >= 10;
#else
    (void)cc_major;
    return false;
#endif
}

// The compile-time gate above, with the deployment's answer applied.
inline bool native_mxfp4_moe_enabled(int cc_major) {
    return device_supports_native_mxfp4_moe(cc_major) &&
           !native_mxfp4_moe_opt_out();
}

}  // namespace pie_cuda_driver
