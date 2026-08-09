//! Would three FlashInfer kernels vendor into `csrc/vendor/flashinfer/`, and
//! does NVRTC 13.0 compile them for `sm_89`?
//!
//! # The question
//!
//! Three table rows are classified `Wall::Library` — *"no device text in this
//! tree"* — in `examples/migration_status.rs`:
//!
//! | row | what it calls |
//! |---|---|
//! | `attn::merge_attention_states_bf16` | `::flashinfer::MergeStates<__nv_bfloat16, __nv_bfloat16>` |
//! | `comm::all_reduce_residual_rmsnorm_bf16` | `flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_launcher` |
//! | `ssm::flashinfer_mamba_ssu_bf16` | `flashinfer::mamba::invokeSelectiveStateUpdate` |
//!
//! That classification bundles two different claims — *"the kernel is a
//! library's"* and *"the text is not here"* — and for the first row the second
//! half is **false on its face**: `csrc/vendor/flashinfer/attention/cascade.cuh`
//! is already in this crate's own vendored tree, 791 lines, seven
//! `__global__`s, and `MODIFICATIONS` records it as one of the fourteen files
//! carried with **zero** guards. So this probe does not read; it compiles.
//!
//! # The method
//!
//! `new-horizon.md` §14 sets the bar a vendoring has to meet, and it is not
//! *"it compiles"*:
//!
//! * **strippable back to upstream byte for byte** — 33 guards over 206 added
//!   lines across 28 files today, every one marked `// PIE:`;
//! * every `#include` answered by the carried header set or by a shim, **never
//!   by the toolkit**;
//! * `NOTICE`, `MODIFICATIONS` and upstream's `LICENSE` carried beside it.
//!
//! So each candidate is measured on four axes, in this order, because a later
//! one is meaningless without the earlier:
//!
//! 1. **the closure** — which upstream files move, how many lines, and which
//!    are already vendored;
//! 2. **the externals** — every `#include` that leaves the FlashInfer tree,
//!    split into *carried by name* (a shim or a vendored stub answers it),
//!    *guarded in the baseline* (the 23 host includes §13.6 sorted out, which
//!    the existing 28 files already guard away), and **unanswered**;
//! 3. **NVRTC** — the real compile, with the real header set, at `sm_89`, with
//!    `--device-as-default-execution-space` and the same three float flags
//!    `runtime::nvrtc` uses, because a cubin built under different arithmetic
//!    is a different cubin;
//! 4. **the driver** — `cuModuleLoadData` plus one `cuModuleGetFunction` per
//!    lowered name, because a lowered name is a promise about a symbol and
//!    `cuModuleGetFunction` is what keeps it.
//!
//! # Nothing is migrated and nothing is copied into the repo
//!
//! This example **probes**. It adds no row, declares no shipped `Unit`, and
//! writes no file. The two candidates whose text is not in this tree are read
//! from the upstream tree CPM already fetched for the ahead-of-time build —
//! `target/*/build/kernels-cuda-*/out/kernels-cuda/build/_deps/flashinfer-src`,
//! the same tree `tests/flashinfer_decode.rs` reaches for and the same pin
//! (`v0.6.15`) `NOTICE` names — and staged **in memory**. Any guard a
//! candidate needs is a [`Guard`] in this file, applied to the text after it
//! is read, so the guard count this prints is auditable against the source
//! rather than asserted.
//!
//! # Why the upstream tree and not a fresh clone
//!
//! `NOTICE` says the vendored files *"were copied from the tree that command
//! produced … rather than from a fresh clone, so that the text carried here is
//! byte-for-byte the text the AOT build compiles."* A probe that measured a
//! different checkout would be measuring a different vendoring.
//!
//! ```text
//! PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 \
//!   cargo run -q -p kernels-cuda-new --features cuda-13 --example vendor_probe
//! ```

#[cfg(feature = "_cuda")]
fn main() {
    imp::main();
}

#[cfg(not(feature = "_cuda"))]
fn main() {
    println!(
        "vendor_probe needs NVRTC. Re-run with `--features cuda-13` (or \
         `cuda-12`): every question it asks is answered by a compile, and \
         there is nothing to report without a compiler."
    );
}

#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_lines)]
mod imp {
    use std::collections::{BTreeMap, BTreeSet};
    use std::ffi::{CStr, CString};
    use std::path::{Path, PathBuf};

    use cudarc::driver::sys as dr;
    use cudarc::nvrtc::sys as nv;
    use kernels_cuda_new::source::{self, ALL_HEADERS, LIBRARY, VENDOR};

    // -----------------------------------------------------------------------
    // the candidates
    // -----------------------------------------------------------------------

    /// One `#ifndef __CUDACC_RTC__` guard, applied to staged text in memory.
    ///
    /// The shape §14.2 requires: a marked edit that a strip reverses. `find`
    /// is matched exactly once — a `find` that matches zero times or more than
    /// once is reported as a defect of this file rather than silently skipped,
    /// because a guard that did not apply makes the count a lie in the
    /// direction that flatters the result.
    struct Guard {
        /// The upstream-relative file, e.g. `flashinfer/mamba/common.cuh`.
        file: &'static str,
        find: &'static str,
        replace: &'static str,
    }

    impl Guard {
        /// Lines this guard adds, which is what `MODIFICATIONS` counts.
        fn added_lines(&self) -> usize {
            self.replace.lines().count().saturating_sub(self.find.lines().count())
        }
    }

    /// A line the crate's own shim would have to gain.
    ///
    /// Distinct from a [`Guard`] and the distinction is the whole point:
    /// a guard is an edit to SOMEBODY ELSE'S source that a strip reverses, and
    /// `MODIFICATIONS` counts it against the rebase budget. A shim patch is an
    /// edit to a file this crate wrote, which costs nothing on a rebase and
    /// everything in review — `csrc/vendor/cstdint`'s own header argues at
    /// length that its macro surface is *"deliberately NOT here"*, so a line
    /// added to it is a decision, not a typo fix.
    struct ShimPatch {
        /// The carried header, by the name an `#include` spells.
        header: &'static str,
        /// Text appended before the trailing include guard is closed — which
        /// is legal here because the probe hands NVRTC the text directly and
        /// a trailing `#endif` mismatch would be a diagnostic, not a silence.
        append: &'static str,
    }

    /// One kernel under test.
    struct Candidate {
        /// The table row this is about.
        row: &'static str,
        /// Where the launcher lives, for the report.
        launcher: &'static str,
        /// Upstream roots whose transitive closure would have to move. Empty
        /// when the text is already vendored.
        roots: &'static [&'static str],
        /// The translation unit handed to NVRTC.
        source: &'static str,
        /// What to instantiate, and the label each answer is reported under.
        wanted: fn() -> Vec<(String, String)>,
        guards: &'static [Guard],
        /// Lines this crate's OWN shims would have to gain. Counted
        /// separately from `guards` — see [`ShimPatch`].
        patches: &'static [ShimPatch],
        /// Headers PIE does not have and would have to WRITE, each one a new
        /// entry in `DEVICE_HEADERS` and a new file in `csrc/vendor/`.
        ///
        /// This is the field that turns "vendors with N shims" from an
        /// estimate into a count: N is `shims.len()`, and the lines are the
        /// lines below. Each is the minimum that satisfies the names the
        /// closure's DEVICE code actually spells — §14.3's rule — and no more,
        /// because a shim that answers more than was asked is a promise this
        /// crate then has to keep.
        shims: &'static [Shim],
        /// Extra NVRTC options, appended after the five `runtime::nvrtc`
        /// always passes. Appended and not merged, which is what makes them
        /// meaningful: `nvrtc.rs:570-580` notes NVRTC reads the list in order
        /// and a later flag wins, so `-std=c++20` here overrides the shared
        /// `-std=c++17` exactly the way a `Unit::options` entry would, and
        /// `Unit::cache_key` spans the same strings so the override cannot be
        /// served a cubin built without it.
        options: &'static [&'static str],
    }

    /// A header this crate would have to write from nothing.
    struct Shim {
        /// The spelling the directive uses.
        name: &'static str,
        /// Why it cannot simply be guarded, in the terms §14.3 sets: which
        /// name reaches device code, and where.
        because: &'static str,
        text: &'static str,
    }

    /// **Row 1.** The text is already here, so nothing is staged.
    ///
    /// Both arms of `MergeStates`'s host `if` are named, at every head dim
    /// `DISPATCH_HEAD_DIM` instantiates. `vec_size` is
    /// `max(16 / sizeof(DTypeIn), HEAD_DIM / 32)` — 8 for bf16 up to head dim
    /// 256 and 16 at 512 — and `bdx` is `HEAD_DIM / vec_size`, `bdy` is
    /// `128 / bdx`. Those are read off `cascade.cuh:638-666` and not guessed;
    /// getting one wrong would name a template NVRTC never instantiates and
    /// `nvrtcGetLoweredName` would refuse it, which is the check.
    const MERGE: Candidate = Candidate {
        row: "attn::merge_attention_states_bf16",
        launcher: "kernels-cuda/csrc/src/attn/attention_merge_states.cu:37",
        roots: &[],
        source: r#"
#include <flashinfer/attention/cascade.cuh>
namespace fi = ::flashinfer;
"#,
        wanted: || {
            let mut out = Vec::new();
            // (head_dim, vec_size, bdx, bdy) -- cascade.cuh:641-651.
            for (head_dim, vec, bdx, bdy) in [(64u32, 8u32, 8u32, 16u32), (128, 8, 16, 8), (256, 8, 32, 4), (512, 16, 32, 4)] {
                out.push((
                    format!("MergeStatesLargeNumIndexSetsKernel head_dim={head_dim}"),
                    format!(
                        "fi::MergeStatesLargeNumIndexSetsKernel<{vec}, {bdx}, {bdy}, 4, __nv_bfloat16, __nv_bfloat16>"
                    ),
                ));
                out.push((
                    format!("MergeStatesKernel               head_dim={head_dim}"),
                    format!("fi::MergeStatesKernel<{vec}, __nv_bfloat16, __nv_bfloat16>"),
                ));
            }
            out
        },
        guards: &[],
        patches: &[],
        shims: &[],
        options: &[],
    };

    /// **Row 3.** A single-GPU SSM kernel, staged from upstream.
    ///
    /// The concrete types are `flashinfer_mamba.cu:16-28`'s, verbatim: the
    /// launcher `#define`s them as file-scope aliases before including the
    /// header, which is upstream's configuration mechanism for this file.
    /// `DIM`/`DSTATE`/`NTOKENS_MTP`/`PHILOX_ROUNDS` are the same four
    /// constants, and the launcher refuses any shape that does not match them.
    const MAMBA: Candidate = Candidate {
        row: "ssm::flashinfer_mamba_ssu_bf16",
        launcher: "kernels-cuda/csrc/src/ssm/flashinfer_mamba.cu:30",
        roots: &["flashinfer/mamba/selective_state_update.cuh"],
        source: r#"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef FLASHINFER_ENABLE_BF16
#define FLASHINFER_ENABLE_BF16 1
#endif
using state_t = nv_bfloat16;
using input_t = nv_bfloat16;
using weight_t = nv_bfloat16;
using matrixA_t = float;
using stateIndex_t = int32_t;
using cuSeqlensIndex_t = int32_t;
using numAcceptedIndex_t = int32_t;
using state_scale_t = void;
constexpr int DIM = 64;
constexpr int DSTATE = 128;
constexpr int NTOKENS_MTP = 1;
constexpr int PHILOX_ROUNDS = 0;
#include <flashinfer/mamba/selective_state_update.cuh>
namespace fm = ::flashinfer::mamba;
"#,
        wanted: Vec::new,
        guards: &[],
        patches: &[],
        shims: &[],
        options: &[],
    };

    /// **Row 2.** Multi-GPU, staged from upstream.
    ///
    /// `AllReduceFusionPattern::kARResidualRMSNorm` and `fp32_acc = true` are
    /// what `custom_all_reduce.cu:694-699` sets; `NRanks` is a template
    /// argument the launcher takes from `world_size_`, and 2 is the smallest
    /// value the dispatch instantiates.
    const TRTLLM: Candidate = Candidate {
        row: "comm::all_reduce_residual_rmsnorm_bf16",
        launcher: "kernels-cuda/csrc/src/comm/custom_all_reduce.cu:184-187, :661",
        roots: &["flashinfer/comm/trtllm_allreduce_fusion.cuh"],
        source: r#"
#include <cuda_bf16.h>
#include <flashinfer/comm/trtllm_allreduce_fusion.cuh>
namespace fa = ::flashinfer::trtllm_allreduce_fusion;
"#,
        wanted: || {
            // `kernels.def` fixes the pattern at `kARResidualRMSNorm` -- pie
            // reaches exactly one of upstream's ten -- and `custom_all_reduce.
            // cu:184-197` turns `fp32_acc` into `Fp32Acc`. `NRanks` stays
            // fully instantiated because TP world size is a deployment choice,
            // and `TriggerCompletionAtEnd` is upstream's own default-true axis
            // that `allreduce_fusion_kernel_launcher` picks from
            // `launch_with_pdl`.
            let mut out = Vec::new();
            for nranks in [2u32, 4, 8, 16] {
                for acc in ["false", "true"] {
                    out.push((
                        format!("oneshot_lamport  NRanks={nranks} Fp32Acc={acc}"),
                        format!(
                            "fa::allreduce_fusion_kernel_oneshot_lamport\
                             <fa::AllReduceFusionPattern::kARResidualRMSNorm, \
                             __nv_bfloat16, {nranks}, {acc}, true>"
                        ),
                    ));
                    out.push((
                        format!("twoshot_sync     NRanks={nranks} Fp32Acc={acc}"),
                        format!(
                            "fa::allreduce_fusion_kernel_twoshot_sync\
                             <fa::AllReduceFusionPattern::kARResidualRMSNorm, \
                             __nv_bfloat16, {nranks}, {acc}>"
                        ),
                    ));
                }
            }
            out
        },
        guards: &[
            // spdlog, deleted by one line. `flashinfer/logging.h` is included
            // at :10 and the file spells `FLASHINFER_LOG` exactly ZERO times
            // -- measured, not assumed -- so the entire third-party host
            // logging dependency is a dead include, and a guard is the whole
            // answer to two of the three unanswered externals.
            Guard {
                file: "flashinfer/comm/trtllm_allreduce_fusion.cuh",
                find: "#include \"../logging.h\"",
                replace: "// PIE: spdlog, and FLASHINFER_LOG is never spelled in this file.\n\
                          #ifndef __CUDACC_RTC__\n\
                          #include \"../logging.h\"\n\
                          #endif",
            },
            // `std::array<int, NRanks>` is a KERNEL PARAMETER at :1568-1569,
            // and this file includes neither <array> nor anything that names
            // it. Upstream reaches it transitively through libstdc++; NVRTC
            // has no transitive path, so the missing include is invisible
            // under nvcc and fatal here -- the same defect `conversion.cuh`
            // has, in a second file.
            Guard {
                file: "flashinfer/comm/trtllm_allreduce_fusion.cuh",
                find: "#include <cuda/std/optional>\n#include <tuple>",
                replace: "// PIE: std::array is a kernel parameter type at :1568.\n\
                          #include <array>\n\
                          #include <cuda/std/optional>\n\
                          #ifndef __CUDACC_RTC__\n\
                          #include <tuple>\n\
                          #endif",
            },
        ],
        patches: &[],
        shims: &[
            Shim {
                name: "array",
                because: "`std::array<int, NRanks>` is a BY-VALUE KERNEL PARAMETER of \
                          `allreduce_fusion_kernel_twoshot_sync` at \
                          `trtllm_allreduce_fusion.cuh:1568-1569`. §14.3 could not be \
                          more direct about this one: a name in a kernel signature \
                          reaches device code, so it is carried, and a guard would \
                          delete the kernel rather than compile it.",
                text: "\
#pragma once
namespace std {
template <class T, unsigned long N>
struct array {
  T __elems[N];
  __host__ __device__ constexpr T& operator[](unsigned long i) { return __elems[i]; }
  __host__ __device__ constexpr const T& operator[](unsigned long i) const { return __elems[i]; }
  __host__ __device__ constexpr T* data() { return __elems; }
  __host__ __device__ constexpr const T* data() const { return __elems; }
  __host__ __device__ constexpr unsigned long size() const { return N; }
  __host__ __device__ constexpr T* begin() { return __elems; }
  __host__ __device__ constexpr T* end() { return __elems + N; }
};
}
",
            },
            Shim {
                name: "cuda/std/optional",
                because: "`cuda::std::optional<int>` is a __device__ FUNCTION PARAMETER at \
                          `trtllm_allreduce_fusion.cuh:549-550` and `:591-593`, in the \
                          FP4 scaling-factor address maths. Carried, not guarded -- and \
                          this is the second CCCL name to reach device code across the \
                          three rows, which is the pattern worth naming.",
                text: "\
#pragma once
namespace cuda {
namespace std {
struct nullopt_t { explicit constexpr nullopt_t(int) {} };
inline constexpr nullopt_t nullopt{0};
template <class T>
class optional {
 public:
  __host__ __device__ constexpr optional() : __v{}, __has(false) {}
  __host__ __device__ constexpr optional(nullopt_t) : __v{}, __has(false) {}
  __host__ __device__ constexpr optional(T v) : __v(v), __has(true) {}
  __host__ __device__ constexpr explicit operator bool() const { return __has; }
  __host__ __device__ constexpr bool has_value() const { return __has; }
  __host__ __device__ constexpr T value() const { return __v; }
  __host__ __device__ constexpr const T& operator*() const { return __v; }
  __host__ __device__ constexpr T value_or(T d) const { return __has ? __v : d; }
 private:
  T __v;
  bool __has;
};
}  // namespace std
}  // namespace cuda
",
            },
        ],
        options: &[],
    };

    /// **Row 3, second reading.** The DEVICE subset, with the host dispatcher
    /// guarded away.
    ///
    /// [`MAMBA`] compiles the file the launcher includes, which drags in the
    /// whole host dispatch layer — `dispatchDimDstate`, `dispatchRatio`,
    /// `dispatchCtasPerHead`, `format_sequence` — built on
    /// `std::integer_sequence`, `std::ostringstream` and `std::clamp`. That is
    /// not what a JIT unit would compile. §11.2 already separates the two
    /// jobs, and §14.6 is the worked example: `scheduler.cuh`'s 1,710 lines of
    /// host planning were PORTED TO RUST and only the kernels were vendored.
    ///
    /// So this candidate asks the narrower and more useful question: with the
    /// host layer guarded out the way `scheduler.cuh`'s was, does the
    /// `__global__` itself compile and instantiate? The guards below are the
    /// bill for finding out, and they are exactly the shape §14.2 counts.
    const MAMBA_DEVICE: Candidate = Candidate {
        row: "ssm::flashinfer_mamba_ssu_bf16  [device subset]",
        launcher: "the simple MTP kernel alone; host dispatch guarded away",
        roots: &[
            "flashinfer/mamba/selective_state_update.cuh",
            "flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh",
        ],
        source: r#"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef FLASHINFER_ENABLE_BF16
#define FLASHINFER_ENABLE_BF16 1
#endif
// `flashinfer_mamba.cu:16-23`, verbatim. These are not decoration: upstream's
// `kernel_selective_state_update_mtp_simple.cuh` spells `cuSeqlensIndex_t` at
// :599 and `numAcceptedIndex_t` at :614 and declares NEITHER anywhere in the
// v0.6.15 tree -- the header is configured by typedefs the includer must have
// already made. Dropping them costs two "identifier is undefined" errors that
// look like a vendoring failure and are not.
using state_t = __nv_bfloat16;
using input_t = __nv_bfloat16;
using weight_t = __nv_bfloat16;
using matrixA_t = float;
using stateIndex_t = int32_t;
using cuSeqlensIndex_t = int32_t;
using numAcceptedIndex_t = int32_t;
using state_scale_t = void;
constexpr int DIM = 64;
constexpr int DSTATE = 128;
constexpr int NTOKENS_MTP = 1;
constexpr int PHILOX_ROUNDS = 0;
#include <flashinfer/mamba/selective_state_update.cuh>
#include <flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh>
namespace fm = ::flashinfer::mamba;
namespace fmm = ::flashinfer::mamba::mtp;

// The launcher calls `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicShared
// MemorySize, smem_size)` before every launch (`invoke_selective_state_update_
// mtp.cuh:294`), and `runtime/module.rs` has `cuLaunchKernel` and a `smem`
// field but NO `cuFuncSetAttribute`. Whether that absence bites depends
// entirely on one number: 48 KB. So the number is exported rather than
// reasoned about -- a `__device__` array the probe reads back through
// `cuModuleGetGlobal`, which makes the answer NVRTC's and not mine.
template <int CTAS_PER_HEAD>
struct PieSmem {
  static constexpr int NUM_WARPS = 4;
  static constexpr int ROWS = NUM_WARPS * fmm::simple_horiz::ROWS_PER_WARP;
  static constexpr int DIM_PER_CTA = DIM / CTAS_PER_HEAD;
  static constexpr int PASSES = DIM_PER_CTA / ROWS;
  static constexpr int STAGES = (PASSES == 1) ? 1 : 2;
  using type = fmm::SimpleStorage<input_t, state_t, NTOKENS_MTP, DIM_PER_CTA,
                                  fmm::padDstate<input_t>(DSTATE), ROWS, STAGES>;
  static constexpr unsigned long long value = sizeof(type);
};
__device__ unsigned long long pie_probe_smem[3] = {
    PieSmem<4>::value, PieSmem<2>::value, PieSmem<1>::value};
"#,
        wanted: || {
            // `flashinfer_mamba.cu:16-28`'s types, and the launcher's own
            // constants: DIM 64, DSTATE 128, NTOKENS_MTP 1, PHILOX_ROUNDS 0.
            // NUM_WARPS is 4 (`invoke_selective_state_update_mtp.cuh:251`).
            // HEADS_PER_GROUP and CTAS_PER_HEAD are the two arguments the HOST
            // dispatches over -- `{1,2,4,8,16,32,64}` and `{4,2,1}` -- so both
            // are swept here rather than picked, because a name expression is
            // fixed before it can see either.
            let mut out = Vec::new();
            for heads_per_group in [1u32, 8, 64] {
                for ctas_per_head in [4u32, 2, 1] {
                    out.push((
                        format!("simple_mtp  HEADS_PER_GROUP={heads_per_group} CTAS_PER_HEAD={ctas_per_head}"),
                        format!(
                            "fmm::selective_state_update_kernel_simple_mtp<__nv_bfloat16, \
                             __nv_bfloat16, float, __nv_bfloat16, int32_t, void, 1, 64, 128, \
                             {heads_per_group}, 0, 4, {ctas_per_head}>"
                        ),
                    ));
                }
            }
            out
        },
        guards: &[
            // The host dispatch layer, and the SM90 TMA kernel beside it.
            // `invoke_…mtp.cuh` is `invokeSelectiveStateUpdateMTP` — the
            // launcher §11.2 says belongs in Rust — and `…_stp.cuh` is gated
            // on `FLASHINFER_MAMBA_ENABLE_SM90`, which this build never
            // defines, so on sm_89 it contributes no kernel and only a
            // `<cuda/barrier>` dependency.
            Guard {
                file: "flashinfer/mamba/selective_state_update.cuh",
                find: "#include \"invoke_selective_state_update_mtp.cuh\"\n#include \
                       \"kernel_selective_state_update_stp.cuh\"",
                replace: "// PIE: the host dispatcher and the SM90 TMA kernel.\n\
                          #ifndef __CUDACC_RTC__\n\
                          #include \"invoke_selective_state_update_mtp.cuh\"\n\
                          #include \"kernel_selective_state_update_stp.cuh\"\n\
                          #endif",
            },
            // `<utility>` is `std::forward`, in `dispatchCtasPerHead`. Host.
            Guard {
                file: "flashinfer/mamba/selective_state_update.cuh",
                find: "#include <cstdint>\n#include <utility>",
                replace: "#include <cstdint>\n// PIE: host only.\n#ifndef __CUDACC_RTC__\n\
                          #include <utility>\n#endif",
            },
            // `common.cuh`'s tail: `format_sequence`, the three `dispatch*`
            // templates and `check_ptr_alignment_input_vars`. Every one is
            // called from the host launcher and none is `__device__`.
            Guard {
                file: "flashinfer/mamba/common.cuh",
                find: "// =============================================================================\n\
                       // Dispatch helpers\n\
                       // =============================================================================",
                replace: "// PIE: host-only dispatch, built on std::integer_sequence.\n\
                          #ifndef __CUDACC_RTC__\n\
                          // =============================================================================\n\
                          // Dispatch helpers\n\
                          // =============================================================================",
            },
            Guard {
                file: "flashinfer/mamba/common.cuh",
                find: "}\n\n}  // namespace flashinfer::mamba\n\n#endif  // FLASHINFER_MAMBA_COMMON_CUH_",
                replace: "}\n#endif\n\n}  // namespace flashinfer::mamba\n\n\
                          #endif  // FLASHINFER_MAMBA_COMMON_CUH_",
            },
            Guard {
                file: "flashinfer/mamba/common.cuh",
                find: "#include <cstdint>\n#include <sstream>\n#include <utility>",
                replace: "#include <cstdint>\n// PIE: host only.\n#ifndef __CUDACC_RTC__\n\
                          #include <sstream>\n#include <utility>\n#endif",
            },
            // Dead on sm_89 and measured to be so: every async copy in
            // `kernel_…_mtp_simple.cuh` is inline PTX -- `cp.async.cg.shared.
            // global` at :93, `cp.async.commit_group` at :266 and :492 -- and
            // the file names neither `cg::` nor `__pipeline_` anywhere. The
            // two headers are included and unused, which is what a guard is
            // for. The probe proves it the only way that counts: with both
            // registered as EMPTY, the kernel still compiles and loads.
            Guard {
                file: "flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh",
                find: "#include <cooperative_groups.h>\n\
                       #include <cooperative_groups/memcpy_async.h>\n\
                       #include <cuda_pipeline.h>",
                replace: "// PIE: unused on sm_89 -- every async copy here is inline PTX.\n\
                          #ifndef __CUDACC_RTC__\n\
                          #include <cooperative_groups.h>\n\
                          #include <cooperative_groups/memcpy_async.h>\n\
                          #include <cuda_pipeline.h>\n\
                          #endif",
            },
            // Not a guard so much as a repair, and worth its own line in
            // MODIFICATIONS for that reason. `conversion.cuh:90` uses
            // `std::numeric_limits` in __device__ code but includes neither
            // <limits> nor <cstdint>; upstream gets both transitively through
            // libstdc++ under nvcc. NVRTC has no transitive path to anything,
            // so the missing include is invisible upstream and fatal here.
            Guard {
                file: "flashinfer/mamba/conversion.cuh",
                find: "#pragma once\n#include <cuda.h>",
                replace: "#pragma once\n// PIE: NVRTC has no host include path; \
                          std::numeric_limits and int16_t are used at :90 and\n\
                          // reach here transitively under nvcc only.\n\
                          #include <cstdint>\n#include <limits>\n#include <cuda.h>",
            },
        ],
        patches: &[ShimPatch {
            header: "cstdint",
            // `common.cuh:73` and `:80` spell `__shfl_down_sync(UINT32_MAX, …)`.
            // The stub's own header says its macro surface is "deliberately
            // NOT here" because the attention closure named none of them, so
            // this is a decision to revisit and not an oversight to fix.
            append: "#ifndef UINT32_MAX\n#define UINT32_MAX 0xffffffffU\n#endif",
        }],
        shims: &[Shim {
            name: "limits",
            because: "`conversion.cuh:90` `numeric_limits<int16_t>::max()`, \
                      `ssu_mtp_common.cuh:172` `<float>::lowest()` and `:197` \
                      `<state_t>::max()` are all inside __device__ code, so §14.3 \
                      says CARRY, not guard -- an empty header compiles the file \
                      away, it does not answer it",
            // Only the three specialisations the closure's device code names,
            // and only the members it calls. `state_t` is `__nv_bfloat16` for
            // this row, whose `max()` is spelled through the bf16 shim's own
            // constructor from float.
            text: "\
#pragma once
namespace std {
template <class T> struct numeric_limits;
template <> struct numeric_limits<int16_t> {
  static __host__ __device__ constexpr int16_t max() { return 32767; }
  static __host__ __device__ constexpr int16_t lowest() { return -32768; }
};
template <> struct numeric_limits<float> {
  static __host__ __device__ constexpr float max() { return 3.40282347e+38f; }
  static __host__ __device__ constexpr float lowest() { return -3.40282347e+38f; }
  static __host__ __device__ constexpr float infinity() { return __int_as_float(0x7f800000); }
};
template <> struct numeric_limits<__nv_bfloat16> {
  static __host__ __device__ __nv_bfloat16 max() { return __nv_bfloat16(3.38953139e+38f); }
  static __host__ __device__ __nv_bfloat16 lowest() { return __nv_bfloat16(-3.38953139e+38f); }
};
template <> struct numeric_limits<__half> {
  static __host__ __device__ __half max() { return __half(65504.0f); }
  static __host__ __device__ __half lowest() { return __half(-65504.0f); }
};
}
",
        },
        Shim {
            name: "cuda/barrier",
            because: "`ssu_mtp_common.cuh:41` declares `using barrier_t =                       cuda::barrier<cuda::thread_scope_block>` at namespace scope,                       which a __device__ helper at :73 then takes by reference -- so                       the name is parsed on the way to EVERY kernel in the file,                       including the sm_89 one that never touches a barrier. §14.3                       says carry. This is the CCCL door §13.5 records as closed, and                       what follows is a measurement of how far it opens, not a port.",
            // The whole CCCL surface this closure spells, and nothing else:
            // the scope enumerator, the class template (declared far enough to
            // form the alias and bind a reference), and the one free function
            // at :73. `arrive` and `wait` are DECLARED and never defined --
            // legal, and the point: the simple MTP kernel never calls them, so
            // if this links, that absence is the proof.
            text: "#pragma once
namespace cuda {
enum thread_scope {
  thread_scope_thread = 0,
  thread_scope_block = 1,
  thread_scope_device = 2,
  thread_scope_system = 3
};
template <thread_scope Scope, class CompletionF = void>
class barrier {
 public:
  using arrival_token = unsigned long long;
  __host__ __device__ arrival_token arrive(unsigned update = 1);
  __host__ __device__ void wait(arrival_token&&) const;
  unsigned long long __pie_state;
};
namespace device {
__host__ __device__ inline unsigned long long* barrier_native_handle(
    barrier<thread_scope_block>& b) {
  return &b.__pie_state;
}
}  // namespace device
}  // namespace cuda
",
        }],
        // The kernel body needs C++20 -- see the report.
        options: &["-std=c++20"],
    };

    // -----------------------------------------------------------------------
    // staging: upstream text, in memory, under every spelling
    // -----------------------------------------------------------------------

    /// A staged header. `Header` is two `&'static str` and staged text is
    /// neither, so this owns its bytes and is leaked into a `&'static` only at
    /// the point NVRTC needs one.
    struct Staged {
        name: String,
        text: String,
    }

    /// Every quoted-or-angled `#include` spelling in `source`.
    ///
    /// Both bracket styles, because NVRTC matches `includeNames[]` against the
    /// literal directive string for either — which is `build.rs`'s `carried`
    /// module's measured finding and the reason 32 aliases exist in the
    /// current vendored tree.
    fn includes(source: &str) -> Vec<String> {
        source
            .lines()
            .filter_map(|line| {
                let rest = line.trim_start().strip_prefix('#')?.trim_start();
                let rest = rest.strip_prefix("include")?.trim_start();
                let close = match rest.chars().next()? {
                    '"' => '"',
                    '<' => '>',
                    _ => return None,
                };
                rest[1..].split(close).next().map(str::to_string)
            })
            .collect()
    }

    /// `a/b/../c` -> `a/c`, the way a preprocessor resolves a relative include.
    fn normalise(path: &str) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for seg in path.split('/') {
            match seg {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                other => parts.push(other),
            }
        }
        parts.join("/")
    }

    /// The transitive closure of `roots` inside `tree`, as upstream-relative
    /// paths, breadth-first and deterministic.
    fn closure(tree: &Path, roots: &[&str], guards: &[Guard]) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        let mut queue: Vec<String> = roots.iter().map(|r| (*r).to_string()).collect();
        while !queue.is_empty() {
            let rel = queue.remove(0);
            if seen.contains(&rel) {
                continue;
            }
            let Ok(mut text) = std::fs::read_to_string(tree.join(&rel)) else { continue };
            // The guards are applied BEFORE the closure is walked, not after,
            // and the difference is the whole cost estimate. A guard that
            // `#ifndef __CUDACC_RTC__`s away an `#include` means the included
            // file is never reached and therefore never vendored: guarding
            // `invoke_selective_state_update_mtp.cuh` out of the JIT unit does
            // not merely silence it, it removes it and the five files behind
            // it from the bill. Walking the raw text would charge this row for
            // 2,876 lines of host dispatch and TMA kernel that no sm_89 unit
            // ever compiles.
            for guard in guards.iter().filter(|g| g.file == rel) {
                text = text.replace(guard.find, guard.replace);
            }
            let text = strip_rtc_guarded(&text);
            seen.push(rel.clone());
            let dir = Path::new(&rel).parent().map_or(String::new(), |p| p.display().to_string());
            for spelling in includes(&text) {
                let beside = normalise(&format!("{dir}/{spelling}"));
                for candidate in [beside, normalise(&spelling)] {
                    if tree.join(&candidate).is_file() && !seen.contains(&candidate) {
                        queue.push(candidate);
                        break;
                    }
                }
            }
        }
        seen
    }

    /// Drop every `#ifndef __CUDACC_RTC__` region, the way NVRTC's own
    /// preprocessor will. Only the includes that SURVIVE that are a cost.
    ///
    /// Deliberately literal: it tracks `#ifndef __CUDACC_RTC__` and its
    /// matching `#endif` by nesting depth and understands nothing else, which
    /// is enough because that is the only spelling `MODIFICATIONS` uses and a
    /// cleverer reader would start disagreeing with NVRTC about something.
    fn strip_rtc_guarded(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut depth = 0usize;
        let mut skipping: Option<usize> = None;
        for line in text.lines() {
            let t = line.trim_start();
            let opens = t.starts_with("#if");
            let closes = t.starts_with("#endif");
            if opens {
                depth += 1;
                if skipping.is_none() && t.replace(' ', "").starts_with("#ifndef__CUDACC_RTC__") {
                    skipping = Some(depth);
                    continue;
                }
            }
            let leaving = closes && skipping == Some(depth);
            if closes {
                depth = depth.saturating_sub(1);
            }
            if leaving {
                skipping = None;
                continue;
            }
            if skipping.is_none() {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    /// Read the closure, apply the guards, and register each file under every
    /// spelling any directive in the staged set uses to reach it.
    ///
    /// The alias pass is `build.rs`'s, restated: NVRTC matches the literal
    /// string in the directive, so `flashinfer/mamba/common.cuh` reached as
    /// `"common.cuh"` from a sibling needs a second entry or the compile stops
    /// at that directive. Getting this wrong is the single easiest way to make
    /// a vendorable header look unvendorable.
    fn stage(tree: &Path, files: &[String], guards: &[Guard]) -> Result<Vec<Staged>, String> {
        let mut staged: Vec<Staged> = Vec::new();
        for rel in files {
            let mut text = std::fs::read_to_string(tree.join(rel))
                .map_err(|e| format!("{rel}: {e}"))?;
            for guard in guards.iter().filter(|g| g.file == *rel) {
                let hits = text.matches(guard.find).count();
                if hits != 1 {
                    return Err(format!(
                        "guard on `{rel}` matches {hits} times, not once -- the guard list in \
                         this file is wrong, and a guard that does not apply makes the count \
                         flatter than the truth"
                    ));
                }
                text = text.replace(guard.find, guard.replace);
            }
            staged.push(Staged { name: rel.clone(), text });
        }

        // Aliases, resolved against the staged set the way a preprocessor
        // would: beside the includer first, then from the tree root.
        let canonical: BTreeSet<String> = staged.iter().map(|s| s.name.clone()).collect();
        let mut aliases: BTreeMap<String, String> = BTreeMap::new();
        for entry in &staged {
            let dir =
                Path::new(&entry.name).parent().map_or(String::new(), |p| p.display().to_string());
            for spelling in includes(&entry.text) {
                if canonical.contains(&spelling) {
                    continue;
                }
                let beside = normalise(&format!("{dir}/{spelling}"));
                let target = if canonical.contains(&beside) {
                    beside
                } else if canonical.contains(&normalise(&spelling)) {
                    normalise(&spelling)
                } else {
                    continue;
                };
                if let Some(prior) = aliases.get(&spelling) {
                    if prior != &target {
                        return Err(format!(
                            "the spelling `{spelling}` reaches both `{prior}` and `{target}` -- \
                             NVRTC has one flat include namespace and would resolve it to \
                             whichever came first"
                        ));
                    }
                } else {
                    aliases.insert(spelling, target);
                }
            }
        }
        let extra: Vec<Staged> = aliases
            .into_iter()
            .map(|(spelling, target)| Staged {
                text: staged.iter().find(|s| s.name == target).unwrap().text.clone(),
                name: spelling,
            })
            .collect();
        staged.extend(extra);

        // A name the carried set already answers must NOT be staged a second
        // time. NVRTC takes `includeNames[]` as a flat namespace and refuses
        // the program outright — `NVRTC_ERROR_INVALID_INPUT`, from
        // `nvrtcCreateProgram`, before any diagnostic — when one name appears
        // twice. Measured here: the first run staged `flashinfer/utils.cuh`,
        // `vec_dtypes.cuh` and `exception.h` beside the vendored copies and
        // got exactly that.
        //
        // Dropping them is also the RIGHT answer and not just the one that
        // compiles: those three are already vendored WITH THEIR GUARDS, and a
        // pristine upstream copy shadowing the patched one would measure a
        // vendoring nobody would ship.
        staged.retain(|s| !ALL_HEADERS.iter().any(|h| h.name == s.name));
        Ok(staged)
    }

    // -----------------------------------------------------------------------
    // the compile
    // -----------------------------------------------------------------------

    /// What one candidate's compile produced.
    struct Compiled {
        millis: f64,
        cubin: Vec<u8>,
        /// `(label, expression, lowered name)` for every instantiation that
        /// came back.
        lowered: Vec<(String, String, String)>,
        /// Instantiations NVRTC compiled without and then would not name.
        missing: Vec<String>,
    }

    /// Compile `source` against `headers` exactly as `runtime::nvrtc` would.
    ///
    /// The three float flags are not decoration: they are in `Unit::cache_key`
    /// because the arithmetic a cubin was built under is part of what it
    /// answers, and a probe that dropped them would be measuring a cubin the
    /// driver never serves. `--device-as-default-execution-space` is §14.4's
    /// per-unit flag, which vendored FlashInfer needs and our own sources must
    /// never be given.
    fn compile(
        headers: &[(CString, CString)],
        source: &str,
        wanted: &[(String, String)],
        arch: &str,
        extra: &[&str],
    ) -> Result<Compiled, String> {
        let text_ptrs: Vec<*const i8> = headers.iter().map(|(t, _)| t.as_ptr()).collect();
        let name_ptrs: Vec<*const i8> = headers.iter().map(|(_, n)| n.as_ptr()).collect();
        let src = CString::new(source).map_err(|_| "a NUL in the source".to_string())?;
        let root = c"vendor_probe.cu";

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call, and the two arrays are
        // `headers.len()` long, which is the count passed.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                root.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        // Before the compile: NVRTC only instantiates a template it was asked
        // for by name, and one added after gets no mangled symbol and no code.
        let expressions: Vec<CString> =
            wanted.iter().map(|(_, e)| CString::new(e.as_str()).unwrap()).collect();
        for expression in &expressions {
            // SAFETY: `program` is live and the string outlives the call.
            unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };
        }

        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let extra: Vec<CString> =
            extra.iter().filter_map(|o| CString::new(*o).ok()).collect();
        let mut options = vec![
            gpu.as_ptr(),
            c"-std=c++17".as_ptr(),
            c"--fmad=false".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
            c"--device-as-default-execution-space".as_ptr(),
        ];
        options.extend(extra.iter().map(|o| o.as_ptr()));

        let started = std::time::Instant::now();
        // SAFETY: `program` came from a successful create; the options outlive it.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            let log = log_of(program);
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        let mut lowered = Vec::new();
        let mut missing = Vec::new();
        for (at, expression) in expressions.iter().enumerate() {
            let mut name: *const i8 = std::ptr::null();
            // SAFETY: `program` is live and compiled; `name` is an out-parameter
            // NVRTC fills with a pointer it owns.
            let code =
                unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut name) };
            if code == nv::nvrtcResult::NVRTC_SUCCESS && !name.is_null() {
                // SAFETY: NVRTC owns the string and it lives until the program
                // is destroyed, which has not happened yet.
                let mangled = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
                lowered.push((wanted[at].0.clone(), wanted[at].1.clone(), mangled));
            } else {
                missing.push(wanted[at].1.clone());
            }
        }

        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        let mut cubin = vec![0u8; size.max(1)];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked for.
        unsafe { nv::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
        cubin.truncate(size);
        // SAFETY: destroyed exactly once, and not used after.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };

        Ok(Compiled { millis, cubin, lowered, missing })
    }

    fn log_of(program: nv::nvrtcProgram) -> String {
        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        let mut log = vec![0u8; size.max(1)];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked for.
        unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
        CStr::from_bytes_until_nul(&log).map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned())
    }

    /// `cuModuleLoadData`, then one `cuModuleGetFunction` per lowered name.
    ///
    /// The half a cubin does not prove. NVRTC lowering a name says the
    /// template instantiated; the driver resolving it says the symbol survived
    /// into the image the device loads, and those come apart — a `__global__`
    /// whose every caller was eliminated is the ordinary way.
    fn resolve(cubin: &[u8], lowered: &[(String, String, String)]) -> Result<Vec<bool>, String> {
        // SAFETY: initialising the driver twice is defined and returns success.
        let code = unsafe { dr::cuInit(0) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(format!("cuInit: {code:?}"));
        }
        let mut device = 0;
        // SAFETY: out-parameter, driver initialised.
        unsafe { dr::cuDeviceGet(&raw mut device, 0) };
        let mut context: dr::CUcontext = std::ptr::null_mut();
        // SAFETY: out-parameter; the context is retained for the process and
        // released below.
        let code = unsafe { dr::cuDevicePrimaryCtxRetain(&raw mut context, device) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDevicePrimaryCtxRetain: {code:?}"));
        }
        // SAFETY: `context` was just retained.
        unsafe { dr::cuCtxSetCurrent(context) };

        let mut module: dr::CUmodule = std::ptr::null_mut();
        // SAFETY: the image is a cubin NVRTC just produced and is non-empty;
        // `cuModuleLoadData` reads its own header for the length.
        let code = unsafe { dr::cuModuleLoadData(&raw mut module, cubin.as_ptr().cast()) };
        if code != dr::CUresult::CUDA_SUCCESS {
            // SAFETY: released exactly once, balancing the retain above.
            unsafe { dr::cuDevicePrimaryCtxRelease_v2(device) };
            return Err(format!("cuModuleLoadData: {code:?}"));
        }

        // `pie_probe_smem`, if the TU exported one. Read from the loaded
        // module rather than computed here, so the answer is the one the
        // compiler and the loader agree on -- the same standard the mangled
        // names are held to two lines below.
        let mut probe: dr::CUdeviceptr = 0;
        let mut bytes = 0usize;
        // SAFETY: out-parameters; `module` is loaded. A missing symbol is a
        // returned error code, not undefined behaviour.
        let code = unsafe {
            dr::cuModuleGetGlobal_v2(
                &raw mut probe,
                &raw mut bytes,
                module,
                c"pie_probe_smem".as_ptr(),
            )
        };
        if code == dr::CUresult::CUDA_SUCCESS && bytes >= 8 {
            let mut values = vec![0u64; bytes / 8];
            // SAFETY: `values` holds `bytes` bytes and `probe` is a device
            // allocation of exactly that size, per `cuModuleGetGlobal`.
            let read = unsafe {
                dr::cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), probe, bytes)
            };
            if read == dr::CUresult::CUDA_SUCCESS {
                println!(
                    "\n    dynamic shared memory the launcher would request, per \
                     CTAS_PER_HEAD (4, 2, 1):"
                );
                for (at, value) in values.iter().enumerate() {
                    let ctas = [4, 2, 1].get(at).copied().unwrap_or(0);
                    println!(
                        "      CTAS_PER_HEAD={ctas}  {value} bytes  -- {} the 48 KB default, so \
                         cudaFuncSetAttribute is {}",
                        if *value > 49152 { "OVER" } else { "under" },
                        if *value > 49152 {
                            "REQUIRED and runtime/module.rs has no cuFuncSetAttribute"
                        } else {
                            "a no-op here"
                        }
                    );
                }
            }
        }

        let mut found = Vec::with_capacity(lowered.len());
        for (_, _, mangled) in lowered {
            let symbol = CString::new(mangled.as_str()).unwrap();
            let mut function: dr::CUfunction = std::ptr::null_mut();
            // SAFETY: `module` is loaded and the name outlives the call.
            let code =
                unsafe { dr::cuModuleGetFunction(&raw mut function, module, symbol.as_ptr()) };
            found.push(code == dr::CUresult::CUDA_SUCCESS && !function.is_null());
        }

        // SAFETY: unloaded exactly once; no entry borrowed from it escapes.
        unsafe { dr::cuModuleUnload(module) };
        // SAFETY: released exactly once, balancing the retain above.
        unsafe { dr::cuDevicePrimaryCtxRelease_v2(device) };
        Ok(found)
    }

    // -----------------------------------------------------------------------
    // the external-include census
    // -----------------------------------------------------------------------

    /// The 31 externals the CURRENT 28-file vendoring leaves, which §13.6
    /// sorted into *"shim the device headers, guard the 23 host ones"*.
    ///
    /// Spelled out because it is the denominator: an external a candidate adds
    /// that is already on this list costs nothing new, because the guard or
    /// the shim that answers it is already written and already tested. One
    /// that is not is real work, and the whole point of the census is to tell
    /// them apart.
    const BASELINE_EXTERNALS: &[&str] = &[
        "algorithm", "atomic", "bit", "boost/math/ccmath/fabs.hpp", "cmath",
        "cooperative_groups.h", "cstddef", "cstdint", "cuda.h", "cuda/cmath", "cuda/pipeline",
        "cuda/std/limits", "cuda_bf16.h", "cuda_device_runtime_api.h", "cuda_fp16.h", "cuda_fp4.h",
        "cuda_fp8.h", "cuda_runtime.h", "cuda_runtime_api.h", "driver_types.h", "exception",
        "iostream", "limits", "memory", "sstream", "stdexcept", "string", "tuple", "type_traits",
        "utility", "vector",
    ];

    /// How a candidate's external include is answered.
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum Answer {
        /// The carried set holds a file under exactly this name — a shim, or
        /// one of the six stubs §14.3 says must be CARRIED rather than guarded
        /// because their names reach device code.
        Carried,
        /// Not carried, but the existing vendoring already guards it away in
        /// the files it appears in. Costs a guard in the new file and nothing
        /// else.
        GuardedInBaseline,
        /// Neither. This is the number that decides a verdict.
        Unanswered,
    }

    fn answer(name: &str) -> Answer {
        if ALL_HEADERS.iter().any(|h| h.name == name) {
            Answer::Carried
        } else if BASELINE_EXTERNALS.contains(&name) {
            Answer::GuardedInBaseline
        } else {
            Answer::Unanswered
        }
    }

    // -----------------------------------------------------------------------
    // the two configurations, and why there are two
    // -----------------------------------------------------------------------

    /// Which substitute the unanswered includes get.
    ///
    /// §14.5's method, restated: *"run the experiment with a crutch first, so
    /// a failure of the SOURCE and a failure of the SUBSTITUTE cannot be
    /// confused. One typedef would have looked like 'FlashInfer does not
    /// JIT'."* That probe ran 28/28 with NVIDIA's real headers and 21/28 with
    /// the shims, and the seven-file gap was four missing typedefs rather than
    /// anything about FlashInfer.
    ///
    /// The same trap is live here and it points the other way: a candidate
    /// that refuses under the shipped header set might be refusing because the
    /// SOURCE needs something no shim can give (CCCL in device code — §13.5's
    /// closed door), or merely because a shim is four typedefs short. Those
    /// have opposite verdicts, and only two configurations tell them apart.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Config {
        /// Every unanswered external answered by an **empty** header.
        ///
        /// A guard is exactly an empty header applied at the includer, so this
        /// measures *"would `#ifndef __CUDACC_RTC__` be enough?"* If it
        /// compiles, the include is host-only or unreached and costs one
        /// guard. If it fails, the diagnostic names the identifiers that reach
        /// device code — which §14.3 says must be CARRIED, and carrying is
        /// where the real bill is.
        Stubs,
        /// Every unanswered external answered by the **real toolkit header**,
        /// read from `$CUDA_HOME/include` and `include/cccl`.
        ///
        /// The crutch. It is not a shippable configuration — §13.2 rejected
        /// `$CUDA_HOME` at build time and §3.2 is the whole argument for why —
        /// and that is the point: it answers *"is the SOURCE acceptable to
        /// NVRTC"* with the substitute question removed.
        Crutch,
    }

    impl Config {
        fn label(self) -> &'static str {
            match self {
                Config::Stubs => "empty stubs (would a guard do?)",
                Config::Crutch => "the crutch: real toolkit + CCCL",
            }
        }
    }

    /// Every file under `$CUDA_HOME/include`, registered under every name an
    /// `#include` could spell to reach it.
    ///
    /// Three roots, not one. `include/` is the ordinary search path; `cccl/`
    /// is CCCL's own root, which nvcc adds separately — `<cuda/barrier>` is
    /// `include/cccl/cuda/barrier` on this box and no directive spells the
    /// `cccl/` prefix. And every relative spelling is resolved the way
    /// [`stage`] resolves the FlashInfer ones, because NVRTC matches the
    /// literal directive string: `cooperative_groups/memcpy_async.h` says
    /// `#include "../cooperative_groups.h"`, and without that alias the crutch
    /// refuses at exactly the place the shims were supposed to be tested.
    fn toolkit() -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for root in ["/usr/local/cuda-13.0/include", "/usr/local/cuda-13.0/include/cccl"] {
            let base = Path::new(root);
            let mut stack = vec![base.to_path_buf()];
            while let Some(dir) = stack.pop() {
                let Ok(entries) = std::fs::read_dir(&dir) else { continue };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        // `cccl` is walked under its own root below; walking it
                        // here too would register every CCCL file with the
                        // prefix, which no directive spells.
                        if path.file_name().is_some_and(|n| n == "cccl") && root.ends_with("include")
                        {
                            continue;
                        }
                        stack.push(path);
                    } else if let Ok(text) = std::fs::read_to_string(&path) {
                        let Ok(rel) = path.strip_prefix(base) else { continue };
                        let name = rel.display().to_string();
                        if seen.insert(name.clone()) {
                            out.push((name, text));
                        }
                    }
                }
            }
        }

        // The alias pass, same rule as `stage`.
        let canonical: BTreeSet<String> = out.iter().map(|(n, _)| n.clone()).collect();
        let mut aliases: BTreeMap<String, String> = BTreeMap::new();
        for (name, text) in &out {
            let dir = Path::new(name).parent().map_or(String::new(), |p| p.display().to_string());
            for spelling in includes(text) {
                if canonical.contains(&spelling) || aliases.contains_key(&spelling) {
                    continue;
                }
                let beside = normalise(&format!("{dir}/{spelling}"));
                if canonical.contains(&beside) {
                    aliases.insert(spelling, beside);
                }
            }
        }
        let by_name: BTreeMap<&str, &str> =
            out.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
        let extra: Vec<(String, String)> = aliases
            .iter()
            .filter_map(|(spelling, target)| {
                by_name.get(target.as_str()).map(|t| (spelling.clone(), (*t).to_string()))
            })
            .collect();
        out.extend(extra);
        out
    }

    pub fn main() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        println!("NVRTC version:  {}", version());
        println!("architecture:   {arch}");
        println!(
            "carried set:    {} headers ({} library + {} vendored), {} bytes",
            ALL_HEADERS.len(),
            LIBRARY.len(),
            VENDOR.len(),
            ALL_HEADERS.iter().map(|h| h.text.len()).sum::<usize>()
        );

        let tree = upstream();
        match &tree {
            Some(path) => println!("upstream pin:   {}", path.display()),
            None => println!(
                "upstream pin:   NOT FOUND -- the two staged candidates cannot be measured. \
                 Build `kernels-cuda` once so CPM fetches flashinfer v0.6.15."
            ),
        }
        println!();

        let mut verdicts = Vec::new();
        let only: Vec<String> = std::env::args().skip(1).collect();
        for candidate in [&MERGE, &MAMBA, &MAMBA_DEVICE, &TRTLLM] {
            if !only.is_empty() && !only.iter().any(|o| candidate.row.contains(o.as_str())) {
                continue;
            }
            verdicts.push(run(candidate, tree.as_deref(), arch));
        }

        println!("\n{}", "=".repeat(78));
        println!("SUMMARY");
        println!("{}", "=".repeat(78));
        for (row, verdict) in &verdicts {
            println!("  {row:<42} {verdict}");
        }
        println!(
            "\nThis probe added no row, declared no shipped unit, and wrote no file.\n\
             A cubin and a resolved CUfunction prove the TEXT compiles and loads.\n\
             They prove nothing about whether a LaunchRule reproduces the C++ grid."
        );
    }

    /// Measure one candidate, and return `(row, verdict)`.
    fn run(candidate: &Candidate, tree: Option<&Path>, arch: &str) -> (String, String) {
        println!("{}", "=".repeat(78));
        println!("{}", candidate.row);
        println!("  launcher: {}", candidate.launcher);
        println!("{}", "-".repeat(78));

        // ---- 1. the closure ------------------------------------------------
        let mut staged: Vec<Staged> = Vec::new();
        let mut new_lines = 0usize;
        let mut new_bytes = 0usize;
        let mut guard_lines = 0usize;
        if candidate.roots.is_empty() {
            println!("  closure:  0 new files -- the text is ALREADY VENDORED.");
            let carried: Vec<&str> = VENDOR
                .iter()
                .filter(|h| h.name.contains("cascade"))
                .map(|h| h.name)
                .collect();
            println!(
                "            {} in csrc/vendor, {} lines, {} bytes",
                carried.first().copied().unwrap_or("?"),
                VENDOR
                    .iter()
                    .find(|h| h.name == "flashinfer/attention/cascade.cuh")
                    .map_or(0, |h| h.text.lines().count()),
                VENDOR
                    .iter()
                    .find(|h| h.name == "flashinfer/attention/cascade.cuh")
                    .map_or(0, |h| h.text.len())
            );
        } else {
            let Some(tree) = tree else {
                println!("  SKIPPED: no upstream tree on this box.");
                return (candidate.row.to_string(), "NOT MEASURED (no upstream tree)".into());
            };
            let files = closure(tree, candidate.roots, candidate.guards);
            let already: Vec<&String> =
                files.iter().filter(|f| ALL_HEADERS.iter().any(|h| h.name == **f)).collect();
            let fresh: Vec<&String> = files.iter().filter(|f| !already.contains(f)).collect();
            println!(
                "  closure:  {} files -- {} already vendored, {} would MOVE",
                files.len(),
                already.len(),
                fresh.len()
            );
            for rel in &fresh {
                let text = std::fs::read_to_string(tree.join(rel)).unwrap_or_default();
                new_lines += text.lines().count();
                new_bytes += text.len();
                println!("              {:>6} lines  {rel}", text.lines().count());
            }
            println!("              {new_lines:>6} lines TOTAL, {new_bytes} bytes");
            for rel in &already {
                println!("              (free)      {rel}");
            }

            guard_lines = candidate.guards.iter().map(Guard::added_lines).sum();
            match stage(tree, &files.iter().map(Clone::clone).collect::<Vec<_>>(), candidate.guards)
            {
                Ok(s) => staged = s,
                Err(why) => {
                    println!("  STAGING FAILED: {why}");
                    return (candidate.row.to_string(), "NOT MEASURED (staging)".into());
                }
            }
            let canonical = staged.iter().filter(|s| files.contains(&s.name)).count();
            println!(
                "  aliases:  {} extra spellings registered ({} canonical + {} alias = {} entries)",
                staged.len() - canonical,
                canonical,
                staged.len() - canonical,
                staged.len()
            );
            println!(
                "  guards:   {} guards, {guard_lines} added lines",
                candidate.guards.len()
            );
        }

        // ---- 2. the externals ----------------------------------------------
        let mut externals: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let inside: BTreeSet<&str> = staged
            .iter()
            .map(|s| s.name.as_str())
            .chain(ALL_HEADERS.iter().map(|h| h.name))
            .collect();
        let sources: Vec<(&str, &str)> = if staged.is_empty() {
            vec![("flashinfer/attention/cascade.cuh", VENDOR
                .iter()
                .find(|h| h.name == "flashinfer/attention/cascade.cuh")
                .map_or("", |h| h.text))]
        } else {
            staged.iter().map(|s| (s.name.as_str(), s.text.as_str())).collect()
        };
        for (name, text) in &sources {
            let dir = Path::new(name).parent().map_or(String::new(), |p| p.display().to_string());
            // On the GUARDED text, with `#ifndef __CUDACC_RTC__` regions
            // removed -- both the guards this candidate would add and the ones
            // upstream already ships. An include NVRTC's preprocessor never
            // reaches is not an include this vendoring has to answer, and
            // counting it would inflate the bill with names like
            // `<sstream>` that are already dead on arrival.
            for spelling in includes(&strip_rtc_guarded(text)) {
                let beside = normalise(&format!("{dir}/{spelling}"));
                if inside.contains(spelling.as_str()) || inside.contains(beside.as_str()) {
                    continue;
                }
                externals.entry(spelling).or_default().insert((*name).to_string());
            }
        }
        if !candidate.shims.is_empty() {
            println!(
                "\n  headers PIE would have to WRITE: {} ({} lines)",
                candidate.shims.len(),
                candidate.shims.iter().map(|s| s.text.lines().count()).sum::<usize>()
            );
            for shim in candidate.shims {
                println!("    <{}>  {} lines", shim.name, shim.text.lines().count());
                println!("        {}", shim.because.split_whitespace().collect::<Vec<_>>().join(" "));
            }
        }
        if !candidate.patches.is_empty() {
            println!("\n  lines this crate's OWN shims would gain: {}", candidate.patches.len());
            for patch in candidate.patches {
                println!(
                    "    <{}>  +{} lines",
                    patch.header,
                    patch.append.lines().count()
                );
            }
        }
        let mut unanswered = Vec::new();
        println!("\n  external includes it adds: {}", externals.len());
        for (name, from) in &externals {
            let verdict = match answer(name) {
                Answer::Carried => "answered -- CARRIED (shim or vendored stub)",
                Answer::GuardedInBaseline => "answered -- guarded in the baseline 28",
                Answer::Unanswered => {
                    unanswered.push(name.clone());
                    "NOT ANSWERED"
                }
            };
            println!(
                "    {name:<38} {verdict}\n        <- {}",
                from.iter().take(2).cloned().collect::<Vec<_>>().join(", ")
            );
        }
        println!(
            "  => {} carried, {} guarded-in-baseline, {} UNANSWERED",
            externals.keys().filter(|k| answer(k) == Answer::Carried).count(),
            externals.keys().filter(|k| answer(k) == Answer::GuardedInBaseline).count(),
            unanswered.len()
        );

        // ---- 3. NVRTC -------------------------------------------------------
        let wanted = (candidate.wanted)();
        if staged.is_empty() {
            // The text is already vendored: the shipped header set IS the
            // configuration, and there is nothing to substitute.
            let headers: Vec<(CString, CString)> = ALL_HEADERS
                .iter()
                .map(|h| (CString::new(h.text).unwrap(), CString::new(h.name).unwrap()))
                .collect();
            println!(
                "\n  NVRTC [the shipped header set]: {} headers, {} name expressions",
                headers.len(),
                wanted.len()
            );
            return report(candidate, &headers, &wanted, arch, new_lines, &unanswered);
        }

        let mut verdicts: Vec<String> = Vec::new();
        for config in [Config::Stubs, Config::Crutch] {
            // The crutch is built first because it also has to REPLACE this
            // crate's own shims. §14.5's separation only works if the two runs
            // differ in one thing: whose header answered. Leaving
            // `csrc/vendor/cuda_bf16.h` in place under the crutch would mix
            // NVIDIA's real `cooperative_groups/memcpy_async.h` against PIE's
            // 40-line `cooperative_groups.h`, and a failure there would name
            // neither the source nor the substitute.
            let real: BTreeMap<String, String> = match config {
                Config::Stubs => BTreeMap::new(),
                Config::Crutch => toolkit().into_iter().collect(),
            };
            let mut headers: Vec<(CString, CString)> = ALL_HEADERS
                .iter()
                .map(|h| {
                    let patched = candidate
                        .patches
                        .iter()
                        .filter(|p| p.header == h.name)
                        .fold(h.text.to_string(), |acc, p| format!("{acc}\n{}\n", p.append));
                    (CString::new(patched).unwrap(), CString::new(h.name).unwrap())
                })
                .collect();
            for entry in &staged {
                headers.push((
                    CString::new(entry.text.as_str()).unwrap_or_default(),
                    CString::new(entry.name.as_str()).unwrap(),
                ));
            }
            let taken: BTreeSet<String> =
                headers.iter().map(|(_, n)| n.to_string_lossy().into_owned()).collect();

            // BOTH configurations stub every external the carried set does not
            // answer -- including the ones this report calls
            // "guarded-in-baseline". That is not a shortcut, it is what the
            // baseline IS: `<utility>` is not answered by anything, it is
            // `#ifndef __CUDACC_RTC__`-ed away in the 28 files that reach it,
            // and a NEW file needs its OWN guard. An empty header is that
            // guard, applied from the outside, so the stub count IS the guard
            // count the vendoring would pay.
            //
            // The crutch then OVERWRITES the stub wherever NVIDIA actually
            // ships the header, which is what makes it a crutch: the CUDA
            // toolkit has `<cuda/barrier>` and `<cudaTypedefs.h>` and has
            // never had `<utility>`, so the difference between the two runs is
            // exactly "device headers real vs absent" and nothing else.
            let mut supplied = 0usize;
            for name in externals.keys() {
                if taken.contains(name) {
                    continue;
                }
                // A candidate's own shim wins over both, because it is the
                // thing being measured: the claim "vendors with N shims" is
                // only worth anything if the N are written out and compiled.
                let text = if let Some(shim) = candidate.shims.iter().find(|s| s.name == *name) {
                    shim.text.to_string()
                } else if let Some(t) = real.get(name) {
                    supplied += 1;
                    t.clone()
                } else {
                    "// PIE probe: empty stub, standing in for a guard".to_string()
                };
                let Ok(text) = CString::new(text) else { continue };
                headers.push((text, CString::new(name.as_str()).unwrap()));
            }
            // The crutch also has to answer whatever the REAL headers include,
            // which is most of CCCL. Registered wholesale, minus every name the
            // carried or staged set already owns -- a crutch that shadowed the
            // vendored, guarded FlashInfer files would measure a different tree.
            if config == Config::Crutch {
                for (name, text) in real {
                    if taken.contains(&name) || externals.contains_key(&name) {
                        continue;
                    }
                    let Ok(text) = CString::new(text) else { continue };
                    headers.push((text, CString::new(name).unwrap()));
                }
            }
            println!(
                "\n  NVRTC [{}]{}: {} headers, {} name expressions{}",
                config.label(),
                if candidate.options.is_empty() {
                    String::new()
                } else {
                    format!(" +{}", candidate.options.join(" "))
                },
                headers.len(),
                wanted.len(),
                if config == Config::Crutch {
                    format!(" ({supplied} of its externals answered by a REAL toolkit header)")
                } else {
                    String::new()
                }
            );
            let (_, verdict) = report(candidate, &headers, &wanted, arch, new_lines, &unanswered);
            verdicts.push(format!("{}: {verdict}", config.label()));
        }
        let _ = guard_lines;
        (candidate.row.to_string(), verdicts.join("  |  "))
    }

    /// Compile, resolve, and print — the tail shared by both configurations.
    fn report(
        candidate: &Candidate,
        headers: &[(CString, CString)],
        wanted: &[(String, String)],
        arch: &str,
        new_lines: usize,
        unanswered: &[String],
    ) -> (String, String) {
        match compile(headers, candidate.source, wanted, arch, candidate.options) {
            Ok(done) => {
                println!(
                    "    COMPILED in {:.0} ms, cubin {} bytes",
                    done.millis,
                    done.cubin.len()
                );
                if !done.missing.is_empty() {
                    println!("    {} instantiations got NO lowered name:", done.missing.len());
                    for m in &done.missing {
                        println!("      {m}");
                    }
                }
                if done.lowered.is_empty() {
                    println!("    (no instantiations asked for -- header acceptance only)");
                    return (
                        candidate.row.to_string(),
                        format!(
                            "headers accepted; {new_lines} new lines, {} unanswered externals",
                            unanswered.len()
                        ),
                    );
                }
                match resolve(&done.cubin, &done.lowered) {
                    Ok(found) => {
                        println!("\n    {:<44} {:<8} mangled name", "instantiation", "cuModule");
                        println!("    {}", "-".repeat(70));
                        let mut resolved = 0usize;
                        for ((label, _, mangled), ok) in done.lowered.iter().zip(&found) {
                            if *ok {
                                resolved += 1;
                            }
                            println!(
                                "    {label:<44} {:<8} {mangled}",
                                if *ok { "RESOLVES" } else { "MISSING" }
                            );
                        }
                        println!(
                            "\n    {resolved} of {} lowered names resolve through \
                             cuModuleGetFunction",
                            done.lowered.len()
                        );
                        (
                            candidate.row.to_string(),
                            format!(
                                "COMPILES + LOADS ({resolved}/{} symbols, {new_lines} new lines, \
                                 {} guards)",
                                done.lowered.len(),
                                candidate.guards.len()
                            ),
                        )
                    }
                    Err(why) => {
                        println!("    cubin loaded nowhere: {why}");
                        (candidate.row.to_string(), format!("compiles, will not load: {why}"))
                    }
                }
            }
            Err(log) => {
                println!("    REFUSED. NVRTC said:\n");
                let lines: Vec<&str> = log.lines().filter(|l| !l.trim().is_empty()).collect();
                // `PIE_PROBE_FULL_LOG=1` prints all of it. The default is 24
                // because a refusal's FIRST diagnostic is the one that names
                // the wall; the rest are the parser falling downhill.
                let cap =
                    if std::env::var_os("PIE_PROBE_FULL_LOG").is_some() { usize::MAX } else { 24 };
                for line in lines.iter().take(cap) {
                    println!("      {line}");
                }
                if lines.len() > cap {
                    println!("      ... and {} more lines", lines.len() - cap);
                }
                let distinct: BTreeSet<&str> =
                    lines.iter().filter(|l| l.contains("error")).copied().collect();
                println!(
                    "\n    {} error lines, {} distinct",
                    lines.iter().filter(|l| l.contains("error")).count(),
                    distinct.len()
                );
                (
                    candidate.row.to_string(),
                    format!("REFUSED ({} distinct errors)", distinct.len()),
                )
            }
        }
    }

    /// The upstream FlashInfer tree CPM fetched for the ahead-of-time build.
    ///
    /// The same walk `tests/flashinfer_decode.rs` does, and for its reason:
    /// `NOTICE` pins v0.6.15 and says the vendored files were copied from
    /// exactly this tree, so a probe measuring anything else measures a
    /// different vendoring.
    ///
    /// Every ancestor is tried rather than the first one holding a `target/`.
    /// The narrower walk found `.probe-scratch/ws/target` — a scratch
    /// workspace's own, which has no `kernels-cuda-*` build in it — and
    /// reported *"no upstream tree"* on a box that has seventeen. A locator
    /// that stops at the first candidate answers a question about the CWD.
    fn upstream() -> Option<PathBuf> {
        if let Ok(explicit) = std::env::var("PIE_FLASHINFER_INCLUDE") {
            let path = PathBuf::from(explicit);
            if path.join("flashinfer/attention/scheduler.cuh").exists() {
                return Some(path);
            }
        }
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if let Some(found) = in_target(&dir.join("target")) {
                return Some(found);
            }
            dir = dir.parent()?.to_path_buf();
        }
    }

    /// The flashinfer include tree under one `target/`, if a `kernels-cuda-*`
    /// build directory holds one.
    fn in_target(target: &Path) -> Option<PathBuf> {
        for profile in std::fs::read_dir(target).ok()?.flatten() {
            let build = profile.path().join("build");
            let Ok(entries) = std::fs::read_dir(&build) else { continue };
            for entry in entries.flatten() {
                if !entry.file_name().to_string_lossy().starts_with("kernels-cuda-") {
                    continue;
                }
                let include =
                    entry.path().join("out/kernels-cuda/build/_deps/flashinfer-src/include");
                if include.join("flashinfer/attention/scheduler.cuh").exists() {
                    return Some(include);
                }
            }
        }
        None
    }

    /// `libnvrtc`'s own version, so a report says which compiler answered.
    fn version() -> String {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: two live out-parameters, and the call takes nothing else.
        unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
        format!("{major}.{minor}")
    }

    /// Keeps `source`'s reachability walk visible to a reader of this file:
    /// the probe's staged set is checked the same way `Unit::header_set` is,
    /// so a candidate that resolves here resolves for the same reason a
    /// shipped unit would.
    #[allow(dead_code)]
    fn reachable_check(root: &str) -> Result<Vec<&'static str>, String> {
        source::reachable("vendor_probe", root, ALL_HEADERS)
    }
}
