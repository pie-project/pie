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

/// Whether routing a decode of this GQA ratio through FlashInfer is a win.
///
/// Only the instantiated ratios are, so this is `attn_decode_gqa_instantiated`
/// under the name the admission checks ask the question in. It lived beside
/// `AttentionWorkspace` while that class was in this crate; the class went
/// home to the driver and this stayed, because which kernels were built is a
/// fact about this crate and nothing the driver can answer.
constexpr bool flashinfer_decode_supports_gqa(int gqa) {
    return attn_decode_gqa_instantiated(gqa);
}

/// `PIE_CUDA_XQA_DECODE` override. Defaults to enabled; when false the driver
/// forces the FlashInfer paged decode kernel on every arch.
inline bool xqa_decode_enabled_by_env() { return true; }

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
// `crates/driver-cuda/csrc/bench/moe_bench.cu` at gpt-oss's shape on an H100 measures the
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

// `PIE_CUDA_NATIVE_MXFP4_MOE=1` forces the lowering back ON where the default
// declines it. Separate from "not opted out" on purpose: the Blackwell gate
// below is a correctness quarantine, and the only way to test a fix for it is
// to be able to ask for the quarantined path explicitly.
inline bool native_mxfp4_moe_opt_in() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_CUDA_NATIVE_MXFP4_MOE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

// Always no, and the vendored trees that could once have said yes are gone.
// Measured in §46: `third_party/marlin_moe`'s two exported functions had zero
// callers anywhere, and the capability this function answers was the ONLY
// stated reason the tree was still compiled -- 156 KB of CUDA on every default
// build. The chain behind that answer had four hops and the last one is a
// constant: this conjunct required `PIE_CUDA_HAS_MARLIN` as well, which
// defaults OFF while `PIE_CUDA_HAS_MARLIN_MOE` defaults ON (the comment this
// replaces explains that the conjunction was ADDED because the defaults
// disagreeing made a default build "plan something it could not execute"); and
// even with both ON, `model-loader/src/plan.rs:194` and `:214` hard-code
// `native_mxfp4_moe: false` in BOTH constructors, with
// `driver-cuda/src/weights/plan.rs:147` asserting it stays false because "the
// native GEMM would want a Marlin repack no kernel here implements".
//
// So no configuration of this tree ever reached the lowering. The function
// stays -- `:215` below still asks it, and a caller reading `false` from a
// named question is clearer than a caller that stopped asking -- but it is now
// the constant it always was in practice.
constexpr bool device_supports_native_mxfp4_moe(int cc_major) {
    (void)cc_major;
    return false;
}

// The Marlin MXFP4 MoE kernels produce WRONG VALUES on sm_100.
//
// Measured on a B200 with gpt-oss-20b: under this lowering decode emits
// uniform garbage ("nasquorashBR @@ Put ShortfacesInte Imper fmt Ind tass"),
// a 0% function-word rate against 39% for the same prompt on vLLM and
// SGLang. Under the routed-dequant GEMV the same build answers correctly at
// 314 tok/s. Bisected against CUDA graph capture: corrupt both captured and
// eager, correct both ways with the GEMV, so the lowering is the only
// variable. The generated kernel set is sm80-shaped
// (`sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`) and was never exercised above
// sm_90.
//
// This is quarantined rather than left to the deployment to discover because
// the gate above answers TRUE on Blackwell whether or not Marlin was built --
// so without this, every sm_100 deployment serves gpt-oss as garbage BY
// DEFAULT, and the failure is silent: the tokens arrive, at a plausible rate,
// and only reading them reveals it.
//
// UPDATE: the root cause is found, and it is NOT the kernel. Two bugs in the
// lowering ABOVE the GEMM, both architecture-independent, so they were
// corrupting sm_80 exactly as described here for sm_100:
//
//   1. the MXFP4 group scales were transposed twice -- the loader already
//      publishes them in Marlin's order and `mixtral.cpp` transposed them
//      again. Mean relative error against a host reference, measured with
//      `bench/marlin_moe_verify.cu` at gpt-oss's shape: 0.0017 correct,
//      0.9350 double-transposed, i.e. uncorrelated output.
//   2. `d_marlin_act`'s padding tail was never initialised, and `0 * NaN` is
//      NaN, so one NaN pattern poisoned a whole fp32-accumulated output row.
//
// The Marlin MoE kernel and the weight repack are themselves correct on
// sm_80 (0.0017-0.0023 mean rel err across E in {1,4,32}, top_k in {1,4}).
// After both fixes sm_80 measures 0/64 degenerate requests at 32-wide and
// 0/16 at concurrency 1, against 8-12/32 and 2/16 before.
//
// This quarantine is therefore very likely stale: the B200 garbage described
// above has the same signature and the same two causes. It is left in place
// only because there is no Blackwell here to verify on. RE-TEST sm_100 with
// `PIE_CUDA_NATIVE_MXFP4_MOE=1` and drop this if it is clean.
constexpr bool native_mxfp4_moe_known_broken(int cc_major) {
    return cc_major >= 10;
}

// The compile-time gate above, with the deployment's answer applied.
inline bool native_mxfp4_moe_enabled(int cc_major) {
    if (!device_supports_native_mxfp4_moe(cc_major)) return false;
    if (native_mxfp4_moe_opt_out()) return false;
    if (native_mxfp4_moe_known_broken(cc_major))
        return native_mxfp4_moe_opt_in();
    return true;
}

}  // namespace pie_cuda_driver
