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

// `#include <cstdlib>` WAS HERE, for the two
// `std::getenv("PIE_CUDA_NATIVE_MXFP4_MOE")` calls at the bottom of this file.
// Both are gone and so is the include -- `sources.rs`'s environment audit puts
// it best about a sibling file: *"the include coming back is the selector
// coming back"*. Nothing else in this header needs the C library; every
// remaining query is `constexpr` over `kernels.def`.

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

// ── THE `native_mxfp4_moe` CAPABILITY WAS HERE, ALL 122 LINES OF IT ──────
//
// Five functions -- `native_mxfp4_moe_enabled`,
// `device_supports_native_mxfp4_moe`, `native_mxfp4_moe_opt_out`,
// `native_mxfp4_moe_opt_in`, `native_mxfp4_moe_known_broken` -- and the two
// `std::getenv("PIE_CUDA_NATIVE_MXFP4_MOE")` reads inside two of them. They
// went with the vendored trees they existed to describe,
// `csrc/third_party/marlin` (500 KB) and `csrc/third_party/marlin_moe`
// (156 KB), and the deletion is §47's four-hop chain finished at the far end.
//
// # The consumer set, per piece, measured rather than assumed
//
// * `native_mxfp4_moe_enabled(cc_major)` -- the composed answer, and the only
//   one anything was ever meant to call. `grep -rn native_mxfp4_moe_enabled
//   crates/` outside this header returns NOTHING. The two call sites its own
//   documentation named -- *"`context.cpp` publishes it as the
//   `native_mxfp4_moe` device fact the loader plans against, and
//   `loaded_model.cpp` passes the same bit into `DeviceTarget`"* -- do not
//   exist: there is no `context.cpp` and no `loaded_model.cpp` anywhere in
//   `crates/`. The comment was true of a tree this one descends from, and a
//   citation to a deleted file is the most convincing kind of dead code
//   there is, because it reads as a live integration.
// * `device_supports_native_mxfp4_moe` -- already `return false;` before this
//   edit. It had been `#if defined(PIE_CUDA_HAS_MARLIN_MOE) &&
//   defined(PIE_CUDA_HAS_MARLIN)`, a conjunction whose two options DEFAULTED
//   APART (`PIE_CUDA_BUILD_MARLIN_MOE` ON, `PIE_CUDA_BUILD_MARLIN` OFF), so a
//   default build compiled 156 KB of CUDA and then answered *no* to the one
//   question that compilation existed to let it answer.
// * `native_mxfp4_moe_opt_out` / `native_mxfp4_moe_opt_in` -- the two
//   environment reads. Reachable only from `native_mxfp4_moe_enabled`, which
//   nothing called, so neither variable could change any behaviour of any
//   build. §36 audited six `getenv`s that were choosing kernels; these two
//   are the other failure, a knob wired to a function with no caller.
// * `native_mxfp4_moe_known_broken` -- the sm_100 quarantine. Same: reachable
//   only from `native_mxfp4_moe_enabled`. Its content is preserved below and
//   NOT deleted, because it is the only part of this block that was an open
//   question rather than an answer.
//
// Even with both build options ON the chain terminated in a constant:
// `model-loader/src/plan.rs:194` and `:214` hard-code `native_mxfp4_moe:
// false` in BOTH constructors, and `driver-cuda/src/weights/plan.rs:147`
// asserts it stays false because *"the native GEMM would want a Marlin repack
// no kernel here implements"*. No configuration of this tree ever reached the
// lowering, so nothing that ran depended on any of the five answers.
//
// ── THE OPEN QUESTION THIS DELETION MUST NOT SWALLOW ─────────────────────
//
// `native_mxfp4_moe_known_broken` quarantined sm_100 and its comment ended:
//
//     "This quarantine is therefore very likely stale: the B200 garbage
//      described above has the same signature and the same two causes. It is
//      left in place only because there is no Blackwell here to verify on.
//      RE-TEST sm_100 with `PIE_CUDA_NATIVE_MXFP4_MOE=1` and drop this if it
//      is clean."
//
// **That instruction cannot be followed as written any more**, because the
// variable it names is gone with the function that read it and the kernels it
// would have selected are gone with the vendored tree. Deleting it silently
// would consume a stated open question, so it is restated here in a form that
// survives the knob:
//
//   THE QUESTION. Do the Marlin MXFP4 MoE kernels produce correct values on
//   sm_100? Measured on a B200 with gpt-oss-20b, decode under this lowering
//   emitted uniform garbage ("nasquorashBR @@ Put ShortfacesInte Imper fmt Ind
//   tass"), a 0% function-word rate against 39% for the same prompt on vLLM
//   and SGLang, while the routed-dequant GEMV answered correctly at 314 tok/s
//   on the same build. Bisected against CUDA graph capture: corrupt both
//   captured and eager, correct both ways with the GEMV, so the lowering was
//   the only variable. The generated kernel set is sm80-shaped
//   (`sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`) and was never exercised above
//   sm_90.
//
//   WHY IT IS PROBABLY A NO-QUESTION. The root cause was found afterwards and
//   it is NOT the kernel. Two bugs in the lowering ABOVE the GEMM, both
//   architecture-independent, were corrupting sm_80 in exactly the way
//   described for sm_100: (1) the MXFP4 group scales were transposed twice --
//   the loader already publishes them in Marlin's order and the mixtral path
//   transposed them again; mean relative error against a host reference at
//   gpt-oss's shape was 0.0017 correct against 0.9350 double-transposed, i.e.
//   uncorrelated output. (2) `d_marlin_act`'s padding tail was never
//   initialised, and `0 * NaN` is NaN, so one NaN pattern poisoned a whole
//   fp32-accumulated output row. After both fixes sm_80 measured 0/64
//   degenerate requests at 32-wide and 0/16 at concurrency 1, against 8-12/32
//   and 2/16 before, and the kernel itself measured correct on sm_80 (0.0017
//   to 0.0023 mean relative error across E in {1,4,32}, top_k in {1,4}).
//
//   HOW TO ANSWER IT NOW. The question is only reachable by re-vendoring, so
//   it is a PRECONDITION on that work rather than a test anyone can run
//   today: whoever restores a native MXFP4 MoE lowering re-tests sm_100
//   FIRST, before re-adding any quarantine, and only re-adds one if it is
//   actually dirty. The two fixes above are the reason to expect it clean.
//   Everything needed to reconstruct the deleted text -- the five functions,
//   both `getenv` sites, the vendored trees, the `CMakeLists.txt` options and
//   the `kernels.def` shape list -- is in git history; `new-horizon.md` §47
//   holds the argument and this comment holds the measurement.
//
//   AND THE CHEAPER READING. The quarantine's own justification was that
//   "the gate above answers TRUE on Blackwell whether or not Marlin was
//   built, so without this every sm_100 deployment serves gpt-oss as garbage
//   BY DEFAULT". That gate answers `false` unconditionally now and the
//   lowering has no kernels behind it, so the failure mode the quarantine
//   guarded is unreachable by construction. What is open is not a risk; it is
//   an unmeasured fact about hardware nobody here has.

}  // namespace pie_cuda_driver
