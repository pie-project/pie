#pragma once

// ── STATUS: THIS HEADER DIES WITH THE ARCHIVE, AND ITS CONSUMERS ALREADY DID ──
//
// Kept in the pass that deleted the six `attention_xqa*.cu` on the EXISTENCE
// half of the two-consumer-sets rule: two files `#include` it, so deleting it
// would have been a compile break rather than a clean removal. The include
// edge is real --
//
//   driver-cuda/tests/oracle/llama_like_cfg/oracle.cpp:37
//   driver-cuda/tests/oracle/llama_like_prepare/oracle.cpp:35
//
// -- and it resolves through those oracles' `-I "$ROOT/crates/kernels-cuda/
// csrc/src"`. What was not checked at the time, and is the whole finding
// here: **NEITHER ORACLE CAN BE COMPILED.** Both `run.sh` open
// `set -euo pipefail` and then `cp "$SRC/model/llama_like/llama_like.cpp"`
// where `SRC="$ROOT/crates/driver-cuda/csrc/src"`. That directory was removed
// wholesale by commit `4569b9e4b` ("Delete crates/driver-cuda"); the `cp`
// fails and the script dies TEN LINES BEFORE the `g++` that would have read
// this header. Seventeen of the twenty-one oracles under
// `driver-cuda/tests/oracle/` are dead the same way, plus `store/` which dies
// at g++ on a missing input rather than at `cp`. The tree already knows:
// `tests/cublas_handle_parity.rs` states the policy -- the goldens are "a
// permanent record of behaviour that can be READ BUT NOT RE-DERIVED", and
// `run.sh` is kept "as the description of how it was taken rather than as a
// command anyone can issue."
//
// THE RULE, because it generalises and it cost a wrong answer here: **an
// `#include` from a translation unit that cannot be compiled is not a
// consumer.** An existence-consumer set is the set of compilers that will
// read the include, not the set of files that contain one. This is the same
// shape as "an absence guard over a deleted file is vacuously true forever":
// an edge that can never be traversed proves nothing, and it looks exactly
// like an edge that can.
//
// So the keep was right on the evidence gathered and wrong on the evidence
// not gathered, and the correct disposition is unchanged anyway: this file is
// `crates/kernels-cuda/csrc/src/**`, it goes when the archive goes at north
// star step 6, and it takes nothing live with it. What it carried that had to
// outlive it is discharged below and recorded in
// `driver-cuda/src/fire/xqa.rs`, which is in a crate that survives.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>

#include <cuda_runtime.h>

#include "attention_workspace_view.hpp"

namespace pie_cuda_driver::kernels::attn {

// FlashInfer XQA decode specializations currently compiled into Pie.
// This is a torch-free wrapper over FlashInfer's csrc/xqa decode kernel.
// It is intentionally narrow: unsupported shapes fall back to the existing
// FlashInfer decode/prefill paths.
bool xqa_decode_bf16_supported(int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int page_size,
                               int window_left,
                               float logits_soft_cap,
                               float sm_scale);

int xqa_decode_page_bucket(int max_pages_per_seq);
// `xqa_decode_bf16_warmup_current_device`,
// `xqa_decode_bf16_gqa5_warmup_current_device` and the unprepared
// `attention_xqa_decode_bf16` WERE declared here. All three launchers are
// deleted: no shim entry is generated for any of them, so the driver cannot
// call them, and nothing in `csrc` does either. The `.cu` carries each
// measurement.
//
// The paragraph that stood here explained the warmups -- that FlashInfer's xqa
// csrc sets the per-device max-dynamic-smem attribute from a once-per-process
// static initializer, which only covers whichever device is current when that
// static runs, so under TP>1 other ranks' devices may never get it, and that
// the fix is to call the warmup after `cudaSetDevice` on each rank before any
// graph capture.
//
// THAT HAZARD IS DISCHARGED. The paragraph used to end "That is a real hazard
// and it is now UNADDRESSED IN THIS TREE ... Kept as prose so the port that
// finishes XQA knows there is a `cudaFuncSetAttribute` per device owed." The
// port that finishes XQA has happened and it is owed nothing:
// `kernels-cuda-new/src/runtime/module.rs`'s `raise_dynamic_smem_cap` issues
// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`
// from inside `KernelModule::fire`, keyed
// `(cuCtxGetDevice(), CUfunction::addr())` against a high-water mark, for any
// launch whose `smem` exceeds 48 KiB. XQA's is 79,488
// (`driver-cuda/src/fire/xqa.rs::XQA_SMEM_BYTES`), so every XQA fire goes
// through it.
//
// THE KEY IS THE WHOLE ARGUMENT, so it is worth stating rather than citing.
// The defect was never "the attribute is not set", it was "it is set once, on
// whichever device happened to be current". `cuCtxGetDevice` reads the
// CALLING THREAD'S current context at the moment of the fire. Rank 1 fires on
// rank 1's context, misses rank 0's cache entry, and raises rank 1's device.
// A per-process static initializer cannot express that and a warmup entry
// point was the only way to work around it, which is why the warmups existed.
// They are not owed because the thing they worked around is gone.
// `driver-cuda/src/fire/xqa.rs:507-515` is the durable half of this record --
// durable because it is in a crate that outlives this one. `new-horizon.md`
// §50.9 is now a closed item and should be read with this paragraph.
//
// A REMAINING GAP, SO IT IS NOT READ AS CLOSED TOO: nothing raises the cap
// BEFORE a graph capture, because nothing raises it before a fire. The
// original prose asked for the warmup "before any graph capture" and that
// half is unaddressed -- the first fire inside a capture is the first fire.
// It is a different hazard from the TP>1 one and it belongs to whoever
// captures XQA, not to whoever ported it.
//
// `prepare_attention_xqa_decode_bf16` WAS declared here too, and it is the
// last thing to leave this archive that had a `__global__` behind it. Its
// kernel is `kernels-cuda-new/csrc/src/attn/attention_xqa.cuh`, JIT-compiled
// as the `attn/attention_xqa` unit; its host half is
// `driver-cuda/src/fire/xqa.rs::prepare_decode`. The row below still states
// `needs = Prepare::FireWide`, and the Rust is now the only thing that can
// discharge it -- as the C++ was, since it had no shim entry either.

void attention_xqa_decode_bf16_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale = -1.f);

}  // namespace pie_cuda_driver::kernels::attn
