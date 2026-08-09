//! `moe/flashinfer_moe.cu`'s HOST PROGRAM, in Rust — the whole file bar an
//! instantiation seam.
//!
//! # What this file was
//!
//! 817 lines with comments stripped, and this is its census:
//!
//! ```text
//! __global__      0      <<<             0      __device__          0
//! std::mutex      2      std::unordered_map     1
//! std::vector     7      std::string             7
//! cudaMalloc      1      cudaMemcpy              4      getenv       3
//! ```
//!
//! **Not one line of device code.** A workspace calculation, a tuning cache,
//! an autotuner and a dispatch — a host program that happened to have a `.cu`
//! extension. `std::mutex` and `std::unordered_map` are not things NVRTC
//! could compile; they are the evidence it was never NVRTC's to compile.
//!
//! The device text is elsewhere and is not ours:
//! `${flashinfer_SOURCE_DIR}/csrc/fused_moe/cutlass_backend/
//! cutlass_fused_moe_kernels.cuh`, 4,991 lines, 19 `__global__`, 8 `<<<>>>`,
//! 52 `cutlass::`, CPM-fetched upstream. **It is the last thing in this
//! family still compiled ahead of time, and that is a state, not a
//! settlement.** The principle has no exception any more —
//!
//! ```text
//!   Every CUDA kernel is compiled by NVRTC, at run time.
//!   All CPU-side code is Rust.
//!   No .cpp. No nvcc. No ahead-of-time CUDA build at all.
//! ```
//!
//! — so `pie_flashinfer_cutlass_moe`, its generated `_SM90_`/`_SM100_` lists
//! and the CPM fetch that feeds them are all things that have to go. Whether
//! this header can go through NVRTC is **measured, and the answer is mostly
//! yes**: a concrete sm90 ptr-array grouped GEMM compiles to 1,245,452 B of
//! PTX with exactly one `.entry`. §13.6's price of "a FlashInfer patch set
//! plus ~39 bit-exact device intrinsics" was quoted for FA2's prefill lattice
//! and **does not transfer** — CUTLASS cost three names in namespace `std`.
//! See the NVRTC section at the bottom of this doc.
//!
//! What this module is, is the other half, and it was never in question:
//! **where host code composes kernels — workspace arithmetic,
//! device-specific tuning — that host code is all Rust.**
//!
//! # What C++ still is, and why it must be
//!
//! `CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>` is a C++ class template
//! whose methods take `ActivationParams`, `QuantParams`, `LoraParams`,
//! `MoeMinLatencyParams` and `MOEParallelismConfig`. Rust cannot name that
//! type, cudarc cannot reach it, and NVRTC cannot compile it. Zero C++ is
//! impossible here; the honest minimum is an `extern "C"` seam that reports
//! FACTS and performs the two calls it is told to perform, deciding nothing.
//! That is [`seam`], five functions, and `csrc/src/moe/flashinfer_moe.cu` is
//! now those five functions and two standard headers — down from fourteen.
//!
//! Everything that *is* a decision came here: which tactic, when to tune,
//! what to cache and where, which rows are eligible, and whether the call is
//! declined.
//!
//! # The refusal is a type
//!
//! The C++ entry returned `bool`, and it is the single `bool`-returning row
//! in the generated shim. `false` did not mean "failed" — it meant **the
//! window declined**, and `model/src/qwen_3_5/forward/mod.rs:362` is only
//! correct because it means that. So the port answers [`Fused`], for
//! `fire/gemv.rs`'s reason: *"it declined"* cannot be spelled like *"it
//! ran"*. A runner that will not build, a seam that reports an error where
//! the C++ could not have thrown — those PANIC with the symbol named. **A
//! refusal is never a fallback.**
//!
//! # The three `getenv` sites: two policy, one dead
//!
//! §36's standing finding is that every knob measured so far selected between
//! arms that agree. These three do not even get that far.
//!
//! 1. `env_truthy(const char*)` — **not a knob at all.** It parsed a
//!    truthiness string, read no environment itself, and had **no caller in
//!    the file**: defined at line 38, zero call sites. Deleted, not ported.
//! 2. `env_int("PIE_MOE_FUSED_MAX_ROWS", 1024)` and the two `std::getenv`
//!    probes in `fused_window_overridden` — **policy, and policy that is OFF
//!    by construction.** See [`WINDOW`]. The variables are gone; the numbers
//!    and the reason the window is not enforced are not.
//!
//! A knob that cannot change an answer is deleted rather than ported, and a
//! genuine configuration becomes a constant here rather than a `getenv` in
//! Rust — which would move the problem instead of solving it.
//!
//! # What the tuning cache became
//!
//! The C++ had three layers and this file reproduces all three, at the same
//! granularity. **Nothing about the concurrency contract is changed
//! silently**; where it is uncomfortable it is named:
//!
//! | C++ | here | scope |
//! |---|---|---|
//! | `static std::array<RunnerState, 16>` keyed by `cudaGetDevice` | [`STATES`] | per device, process-wide |
//! | `std::unordered_map` under `std::mutex tune_mutex` | `RunnerState::tuned` | per device |
//! | function-local `static TuningCache` | [`DISK`] | **one, process-wide** |
//!
//! Two hazards are inherited rather than fixed, because fixing either would
//! be a behaviour change smuggled in under a port:
//!
//! * **The disk cache takes the FIRST device's signature.** `tactic_cache`
//!   was a function-local static initialised by whichever device reached it
//!   first, and `TuningCache::load` *deletes* a file whose signature does not
//!   match. On a heterogeneous multi-GPU box two devices would therefore
//!   thrash one `moe_tactics.txt`. [`DISK`] is a single `OnceLock` for
//!   exactly that reason: making it per-device would be a different program.
//! * **`setTactic` and `runMoe` are not atomic together.** `install_tactics`
//!   held `tune_mutex` while it installed the pair and released it before the
//!   run, so two threads on one device can interleave — A installs, B
//!   installs, A runs with B's tactic. Reproduced. A tactic is a performance
//!   choice, not a correctness one (the tuner never crosses an epilogue
//!   fusion, which is the only numerics-bearing field), so the race costs
//!   time and not answers.
//!
//! One difference is a real, disclosed change of behaviour: the C++ read
//! `pie_cuda_driver::cache_dir()`, the shell-published `[cache] dir`. This
//! crate has no Rust plumbing for that value yet — `layout::profile_cache::
//! ProfileCache::discover("")` and `serve/load.rs:641` pass `""` for the same
//! reason — so [`cache_path`] takes it as a parameter and the one call site
//! passes `""`. **On an engine that configures `[cache] dir`, the MoE tactic
//! file moves from there to `$XDG_CACHE_HOME/pie/`.** It is a cache; a miss
//! costs one sweep.
//!
//! # How it is reached
//!
//! `execution::RUST_SERVED` carries `moe::flashinfer_cutlass_moe_bf16`.
//! `abi::emit_c_shim` therefore drops its entry — which is what makes the C++
//! body deletable — and `abi::emit_rust_bindings` drops the declaration.
//! `emit_dispatch` writes no arm either, because all eighteen operands of the
//! `table::moe` row are `Source::Unbound`, exactly as for
//! `attn::mla_prepare_bf16` and `attn::write_mla_to_pages`. That is a
//! no-regression: there was no arm before this change. [`bind::service`] is
//! where the symbol is spelled, and the model compiler still cannot tell that
//! any of this happened — [`KernelSig`] is unchanged.
//!
//! # NVRTC: can `cutlass_fused_moe_kernels.cuh` go through NVRTC?
//!
//! **Measured, not argued. Yes for the GEMM; the remaining gaps are three
//! names, two PTX instructions and one open parameterisation.** Probes are
//! `nvrtc-probes/cutlass_moe_*.py` in the session state — `libnvrtc.so.13`,
//! CUDA 13.0, `nvrtcCompileProgram` + `nvrtcGetLoweredName` only, no CUDA
//! context, nothing built. The recipe is FA2's, reused verbatim:
//!
//! ```text
//!   -I kernels-cuda-new/csrc/{src,shim,vendor}  -I /usr/local/cuda/include
//!   -std=c++17  -default-device  --gpu-architecture=compute_{89,90a}
//! ```
//!
//! | probe | result |
//! |---|---|
//! | `cute/tensor.hpp` | **rc=0**, PTX 145,198 B |
//! | `cub/block/{block_scan,block_radix_rank,block_radix_sort}.cuh` | **rc=0**, PTX 110,083 B |
//! | CUTLASS grouped-GEMM stack @ `compute_90a` | **rc=0**, PTX 212,981 B |
//! | a CONCRETE sm90 ptr-array grouped GEMM | **rc=0**, lowered name, PTX **1,245,452 B, exactly 1 `.entry`** |
//! | `griddepcontrol.{wait,launch_dependents}` as inline PTX | **rc=0** |
//! | TRT-LLM's FINALIZE scatter epilogue | **unresolved — see below** |
//!
//! The concrete one is the claim that matters: `cutlass::device_kernel<
//! GemmUniversal<GroupProblemShape<Shape<int,int,int>>, CollectiveMma<
//! MainloopSm90ArrayTmaGmmaWarpSpecialized…>, …>>` — tile 128×128×64,
//! cluster 1×1×1, bf16×bf16→bf16, `KernelPtrArrayTmaWarpSpecializedCooperative`
//! — is the first line of `gemm_grouped/90/…group0.generated.cu` minus
//! TRT-LLM's epilogue, and it is **the mainloop with `make_tma_copy` on the
//! host at `sm90_mma_array_tma_gmma_ss_warpspecialized.hpp:251` and
//! `tensormaps_replace_global_address` on the device at `:656`.** NVRTC emits
//! it. §13.6's price — "a FlashInfer patch set plus ~39 bit-exact device
//! intrinsics" — **does not transfer to CUTLASS**, and the number below is
//! what it costs instead.
//!
//! **Three names in namespace `std`.** Every error in the CUTLASS stack, all
//! seven of them, was one of `std::is_pointer_v`
//! (`sm90_epilogue_array_tma_warpspecialized.hpp:497`), `std::max`
//! (`sm100_epilogue_array_tma_warpspecialized.hpp:242`) and `std::void_t`
//! (`linear_combination_bias_elementwise.h:77`) — three places where upstream
//! wrote `std::` where the file's own convention is `cute::` /
//! `cutlass::platform::`. A six-line prelude closed all seven. **CUTLASS needs
//! no patch set, only a shim row**, and `csrc/shim/type_traits` is where the
//! row goes. That is a different order of cost from a patch set and it should
//! not be quoted as one.
//!
//! **`cub` is not a wall either, and its own header says so.**
//! `cub/cub.cuh:17` `#error`s under `_CCCL_COMPILER(NVRTC)` with *"Include the
//! specific device header instead (e.g. `<cub/block/block_reduce.cuh>`)"*.
//! The four this file needs — `BlockScan`, `BlockRadixRank`, `BlockRadixSort`,
//! `BFEDigitExtractor` — compile as three includes. The umbrella is the only
//! thing unsupported.
//!
//! **Two PDL intrinsics are missing and cost two lines.**
//! `cudaGridDependencySynchronize` and
//! `cudaTriggerProgrammaticLaunchCompletion` are undefined under NVRTC — a
//! real gap, since 11 of the 19 `__global__`s call both. Replacing them with
//! `asm volatile("griddepcontrol.wait;" ::: "memory")` and
//! `asm volatile("griddepcontrol.launch_dependents;")` compiles clean. Note
//! this makes the earlier "0 inline PTX" reading a statement about the file
//! as written, not about the port.
//!
//! **What is NOT resolved: the FINALIZE scatter epilogue.** Two hand-built
//! parameterisations of `ScaledAccPerColBiasPerRowScaleScatter` both failed,
//! and **both failures are mine, not NVRTC's** — with
//! `PtrArrayTmaWarpSpecializedCooperative`,
//! `cutlass_extensions/…/sm90_visitor_scatter.hpp:301`'s
//! `constexpr int ThreadsMajor = size<1>(args.epi_tile) / VecSize;` reports
//! *"expression must have a constant value"*; with
//! `PtrArrayNoSmemWarpSpecialized` the epilogue `CollectiveBuilder` has no
//! matching specialisation. Neither is a missing header, a missing stdlib
//! name or a host-code rejection: both are ordinary template selection, the
//! kind nvcc would report identically for the same arguments. **This is the
//! one thing left to measure**, and the way to measure it is to instantiate
//! from the launcher's own `using` chain rather than by hand.
//!
//! **Incidental defect, already known.** `-I …/csrc/shim` must come AFTER the
//! toolkit include for CUTLASS: shim-first gives `shim/cuda_fp16.h(236):
//! invalid redeclaration of type name "__half"` and 83 cascading errors,
//! shim-last gives rc=0. That is the same shim-dtype incompleteness the XQA
//! probe recorded (9 errors → 2). Not a CUTLASS finding.
//!
//! **What none of this proves.** No output was compared against an
//! nvcc-built kernel, so nothing here is a claim about numerics — the phrase
//! "bit-exact" does not belong in any comment citing these probes. The tile,
//! cluster and alignment constants were read off the generated file but the
//! epilogue was substituted. `compute_89` and `compute_90a` only; sm100 is
//! unprobed. And a `.entry` in PTX is not a kernel that runs: `Params` (below)
//! is untouched by any of this.
//!
//! The reading that follows is what the source says, and it stands.
//!
//! **The host/device seam inside the header is clean and it is at a line.**
//! The 19 `__global__` are at `:210`–`:2556` plus three at `:4207`–`:4260`;
//! the host class bodies start at `:2600`. Every `std::vector` (20) and
//! `std::string` (5) in the file is above `:2616` — the workspace-size map,
//! the LoRA host bookkeeping and `GemmProfilerBackend`. **Not one is inside a
//! `__global__` or a `__device__`.** So `<algorithm> <memory> <numeric>
//! <random> <sstream>` are the HOST program's includes, and the host program
//! is the thing this module already replaced. The device half's own needs are
//! `<cuda.h> <cuda_fp16.h> <float.h> <math.h>`, `cub`, `curand`, CuTe and
//! CUTLASS — nothing NVRTC structurally refuses.
//!
//! **Nothing in the structural blocker list is present.** No inline PTX at
//! all (0 `asm`). No `-rdc`: `CMakeLists.txt` sets no
//! `CUDA_SEPARABLE_COMPILATION` anywhere and the target's only flags are
//! `--extended-lambda --expt-relaxed-constexpr`. Exactly one
//! `__launch_bounds__` (`:2149`), and its argument is
//! `constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256` (`:2056`) — a
//! literal, not a host-computed value. `<curand_kernel.h>` is real device
//! code but it is only reached from `prepareFakeRouterBuffers` /
//! `populateRandomBufferKernel` / `prepareMinLatencyBuffer` (`:4207`–`:4260`),
//! which belong to `GemmProfilerBackend` — UPSTREAM's autotuner, which this
//! module does not use because it has its own. `cub` is the one real device
//! dependency: `BlockScan`, `BlockRadixRank`, `BlockRadixSort`,
//! `BFEDigitExtractor`. It is CCCL, header-only, and CCCL ships NVRTC support
//! — but `kernels-cuda-new/csrc/shim/` does not impersonate it today, so a
//! carried-header set for cub is work this tree has not done.
//! `tensorrt_llm/common/envUtils.h` is included by the launcher and **no
//! `getEnv*` call appears in either file** — vestigial, so no §36 site hides
//! in here.
//!
//! **The instantiations are a fixed list, not a host-side lattice.**
//! `third_party/flashinfer_generated/gemm_grouped/{90,100}/` holds **240**
//! `PIE_INSTANTIATE_TMA_WS_MOE_GEMM(...)` lines across 35 units — 72 in 11
//! for sm90, 168 in 24 for sm100 — each with every template argument a
//! literal (arch, dtypes, epilogue tag, fusion, CTA M/N/K, CGA M/N/K, and
//! four bools). That is the GOOD case in the question: **rows**. The catch is
//! in `moe_gemm_tma_ws_instantiate.h`'s own comment — the compile-time
//! dispatch is EXHAUSTIVE, it names a launcher for every combination the
//! runtime could pick, so all of them must resolve at link time, and some are
//! not compilable at all (`PtrArrayTmaWarpSpecialized`/`NONE` and `void`/
//! `NONE` are type errors, upstream-verified, and get throwing stubs). Under
//! NVRTC "must resolve at link time" stops being the constraint it is here:
//! you compile the one instantiation you selected. The exhaustive dispatch
//! and the 240-line generator are both artifacts of ahead-of-time linking.
//!
//! **The hard part is not the kernels, it is `GemmUniversalAdapter`.** The
//! actual GEMM is launched by `moe_gemm_tma_ws_launcher.inl:701` as
//! `gemm.run(stream, nullptr, enable_pdl)` after `can_implement`,
//! `get_workspace_size` and `initialize(args, workspace)`. `run` reaches
//! `cutlass/cluster_launch.hpp:248`'s `cudaLaunchKernelExC(&config, kernel,
//! params)` where `kernel` is `(void const*) device_kernel<GemmKernel>` — **a
//! host function pointer that exists only because nvcc compiled the
//! `__global__` into the host TU.** The same shape governs 11 of the 19 glue
//! kernels, which are launched by `cudaLaunchKernelEx` with
//! `cudaLaunchAttributeProgrammaticStreamSerialization` rather than by
//! `<<<>>>` — which is why the file has 19 `__global__` and only 8 `<<<>>>`.
//!
//! That is a HOST problem and it has an exact driver-API answer:
//! `cuLaunchKernelEx` takes a `CUfunction`, `CU_LAUNCH_ATTRIBUTE_CLUSTER_
//! DIMENSION`, `CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION` and
//! `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, and
//! `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` is
//! the `cudaFuncSetAttribute` at `gemm_universal_adapter.h:279`. Every launch
//! attribute this path uses has a `CU`-prefixed twin. **What has no twin is
//! `GemmKernel::Params`.** `initialize()` runs `to_underlying_arguments`,
//! which for the sm90 ptr-array mainloop calls `make_tma_copy` on the HOST
//! (`sm90_mma_array_tma_gmma_ss_warpspecialized.hpp:251,257`) to build one
//! `CUtensorMap` per operand, which the device then patches per group
//! (`tensormaps_init` `:628`, `tensormaps_replace_global_address` `:656`).
//! `cuTensorMapEncodeTiled` is a driver API and Rust can call it — but the
//! swizzle, box and element-stride arguments `make_tma_copy` passes are
//! derived by CuTe's layout algebra at compile time, per tile shape. **That
//! is the §13.6-shaped cost, in a different currency: not ~39 device
//! intrinsics but a `Params` byte image and a TMA-descriptor derivation, per
//! instantiation.**
//!
//! **What the `Unit` would look like.** ONE unit, N `elem` instantiations —
//! not N units. The 240 generated lines are already exactly that list, and
//! `tactic_key`/[`TacticPair`] here already select one of them at run time
//! from a shape; an NVRTC unit rooted at the launcher header with the tile,
//! cluster, fusion and swap-AB tuple as `elem` parameters is the same
//! selection, moved from a link-time closure to a compile-at-first-use. It
//! would also DELETE the exhaustive dispatch, the throwing stubs and the
//! two-lists-must-stay-in-sync rule in `moe_gemm_tma_ws_instantiate.h`,
//! because there is no linker to satisfy. The autotuner in this module is
//! already the thing that would drive it.
//!
//! Ordered honestly, the remaining work is: (1) a cub carried-header set for
//! `shim/` — **measured as three specific includes, not the umbrella**, plus
//! three `std::` names and two `griddepcontrol` asm lines, (2) the 11
//! `cudaLaunchKernelEx` glue launches restated as `cuLaunchKernelEx` in Rust
//! — mechanical, and this module is where they would live, (3) `Params`,
//! which is the whole project and which **no probe has touched**. (1) and (2)
//! are not blocked on (3): the glue kernels are ordinary `__global__`s with no
//! CUTLASS in them at all, and they are 11 of the 19.
//!
//! [`bind::service`]: crate::bind::service
//! [`KernelSig`]: kernels::KernelSig

#![allow(clippy::print_stderr)]

use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys::{
    cudaError, cudaEventCreateWithFlags, cudaEventDestroy, cudaEventElapsedTime, cudaEventRecord,
    cudaEventSynchronize, cudaEvent_t, cudaFree, cudaGetDevice, cudaGetDeviceProperties_v2,
    cudaGetLastError, cudaMalloc, cudaMemcpyAsync, cudaMemcpyKind, cudaMemsetAsync,
    cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamNonBlocking, cudaStreamSynchronize,
    cudaStream_t,
};

use crate::bind::abi::MoeActivation;

// ───────────────────────────────────────────────────────────────────────────
// The seam
// ───────────────────────────────────────────────────────────────────────────

/// The five `extern "C"` entries `csrc/src/moe/flashinfer_moe.cu` still is.
///
/// Every one of them exists because it names a C++ type or enumerator that
/// has no spelling on this side — `CutlassMoeFCRunner`, `CutlassGemmConfig`,
/// `ActivationType`, the `std::vector` `getTactics` returns. None of them
/// decides anything, and none of them may throw: each catches, copies
/// `what()` into the caller's buffer and answers a status, because an
/// exception crossing the C ABI is undefined behaviour that in practice
/// reaches SIGABRT with no message.
pub mod seam {
    use core::ffi::{c_char, c_int, c_void};

    /// `ck::MoeGemmId::GEMM_1`.
    pub const GEMM_1: c_int = 1;
    /// `ck::MoeGemmId::GEMM_2`.
    pub const GEMM_2: c_int = 2;

    /// `EpilogueFusionType::NONE` — the topk sum is a separate fp32 pass.
    pub const FUSION_NONE: c_int = 0;
    /// `EpilogueFusionType::FINALIZE` — the topk sum folds into the GEMM2
    /// epilogue. See [`super::DEFAULTS`] for why it is not the default.
    pub const FUSION_FINALIZE: c_int = 1;
    /// An upstream enumerator the header has not been taught. Printed as
    /// `unknown`, exactly as the C++'s `fusion_name` tail did, and never
    /// selected across.
    pub const FUSION_UNKNOWN: c_int = 2;

    /// One `ce::CutlassGemmConfig`, flattened to the fields the host reads.
    ///
    /// Thirteen `int`s and no pointers, in `moe/flashinfer_moe.hpp`'s field
    /// order. `tile` is already the live one of the config's four tile enums
    /// — `sm_version` decides which, and printing the wrong one shows a
    /// constant `heuristic` for every tactic, so the seam picks it where the
    /// four field names exist.
    ///
    /// `occupancy` is `queryOccupancyForConfig`, **filled once at query
    /// time**. The C++ re-asked it inside every sweep; the query takes a
    /// config and nothing else, so the answer cannot depend on the problem.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Tactic {
        /// `FUSION_NONE` / `FUSION_FINALIZE` / `FUSION_UNKNOWN`.
        pub fusion: c_int,
        /// `is_tma_warp_specialized`, as 0/1.
        pub is_tma_warp_specialized: c_int,
        /// `swap_ab`, as 0/1.
        pub swap_ab: c_int,
        /// The SM this config was generated for.
        pub sm_version: c_int,
        /// The live tile enum, base-1000 digits. See [`super::shape_str`].
        pub tile: c_int,
        /// `mainloop_schedule`.
        pub mainloop_schedule: c_int,
        /// `epilogue_schedule`.
        pub epilogue_schedule: c_int,
        /// `cluster_shape`, base-1000 digits.
        pub cluster_shape: c_int,
        /// `dynamic_cluster_shape`, base-1000 digits.
        pub dynamic_cluster_shape: c_int,
        /// `fallback_cluster_shape`, base-1000 digits.
        pub fallback_cluster_shape: c_int,
        /// `split_k_factor`.
        pub split_k_factor: c_int,
        /// `stages`.
        pub stages: c_int,
        /// `queryOccupancyForConfig(cfg)`; `<= 0` means unusable here.
        pub occupancy: c_int,
    }

    unsafe extern "C" {
        /// Construct the bf16 runner. Null on failure, `what()` in `err`.
        ///
        /// There is deliberately no destructor: the C++ held its runners in a
        /// function-local `static std::array` whose destructors ran at
        /// process exit against a CUDA context that may already be gone, and
        /// [`super::STATES`] is a `OnceLock` that never drops. Neither ever
        /// freed one while the process could still use it, so the seam offers
        /// no way to.
        pub unsafe fn pie_moe_cutlass_create(err: *mut c_char, err_cap: usize) -> *mut c_void;

        /// Copy the tactic list for `gemm_id` into `out`, answering the FULL
        /// count (which may exceed `cap`) or `-1` with `err` set. `out` may
        /// be null when `cap` is 0, which is how the count is asked alone.
        pub unsafe fn pie_moe_cutlass_tactics(
            runner: *mut c_void,
            gemm_id: c_int,
            out: *mut Tactic,
            cap: c_int,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;

        /// `setTactic(gemm1_tactics[gemm1], gemm2_tactics[gemm2])` — indices
        /// into the same lists, in the same order. Non-zero with `err` set on
        /// failure, including an index out of range.
        pub unsafe fn pie_moe_cutlass_set_tactic(
            runner: *mut c_void,
            gemm1: c_int,
            gemm2: c_int,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;

        /// `getWorkspaceSize(...)`, and the ARCH PROBE: it is the call that
        /// throws when no TMA warp-specialized config has a compiled launcher
        /// for this SM. What a throw means is [`super::workspace_bytes`]'s
        /// decision, not the seam's.
        pub unsafe fn pie_moe_cutlass_workspace_size(
            runner: *mut c_void,
            activation: c_int,
            num_rows: c_int,
            hidden_size: c_int,
            inter_size: c_int,
            num_experts: c_int,
            experts_per_token: c_int,
            tp_size: c_int,
            tp_rank: c_int,
            out: *mut usize,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;

        /// `runMoe(...)` on `stream`, with the currently installed tactic.
        /// The seam validates no pointer; the caller already refused a null.
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn pie_moe_cutlass_run(
            runner: *mut c_void,
            activation: c_int,
            input: *const u16,
            token_selected_experts: *const i32,
            token_final_scales: *const f32,
            fc1_expert_weights: *const u16,
            fc2_expert_weights: *const u16,
            output: *mut u16,
            workspace: *mut u8,
            unpermuted_row_to_permuted_row: *mut i32,
            num_rows: c_int,
            hidden_size: c_int,
            inter_size: c_int,
            num_experts: c_int,
            experts_per_token: c_int,
            tp_size: c_int,
            tp_rank: c_int,
            stream: *mut c_void,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
    }
}

use seam::Tactic;

/// How much of a C++ `what()` is carried back across the seam.
const ERR_CAP: usize = 512;

/// A fresh, zeroed error buffer for one seam call.
const fn err_buf() -> [u8; ERR_CAP] {
    [0_u8; ERR_CAP]
}

/// The NUL-terminated text the seam wrote, as a `String`.
fn err_text(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

// ───────────────────────────────────────────────────────────────────────────
// Diagnostics
// ───────────────────────────────────────────────────────────────────────────

/// `constexpr bool log_enabled() { return false; }`.
///
/// A constant rather than an environment probe, and it was a constant in the
/// C++ too. §36's rule about knobs applies to the one that was never
/// introduced as much as to the two that were: flipping this recompiles, and
/// a build is a cheaper thing to be honest about than a variable nobody
/// remembers is set.
pub const LOG_ENABLED: bool = false;

/// `fusion_name` — `none` / `finalize` / `unknown`.
#[must_use]
pub const fn fusion_name(fusion: i32) -> &'static str {
    match fusion {
        seam::FUSION_NONE => "none",
        seam::FUSION_FINALIZE => "finalize",
        _ => "unknown",
    }
}

/// The tile/cluster enums encode their shape as base-1000 digits, so they are
/// unreadable as raw integers. `Undefined`(0) and `ChooseWithHeuristic`(1)
/// are not shapes and are reported as-is.
#[must_use]
pub fn shape_str(id: i32) -> String {
    if id == 0 {
        return "undef".to_owned();
    }
    if id == 1 {
        return "heuristic".to_owned();
    }
    format!("{}x{}x{}", id / 1_000_000, (id % 1_000_000) / 1000, id % 1000)
}

/// One tactic, as the C++'s `config_str` spelled it.
///
/// `tile=` is [`Tactic::tile`], which the seam has already narrowed by
/// `sm_version`. The C++ carried the warning and it is still live: print the
/// tile field for the wrong architecture and every tactic reports a constant
/// `heuristic`, which reads like a tuner that is not running.
#[must_use]
pub fn config_str(t: &Tactic) -> String {
    let mut s = String::with_capacity(160);
    let _ = write!(
        s,
        "fusion={} tma={} swap_ab={} sm={} tile={} mainloop={} epilogue={} \
         cluster={} dyn_cluster={} fallback_cluster={} split_k={} stages={}",
        fusion_name(t.fusion),
        t.is_tma_warp_specialized,
        t.swap_ab,
        t.sm_version,
        shape_str(t.tile),
        t.mainloop_schedule,
        t.epilogue_schedule,
        shape_str(t.cluster_shape),
        shape_str(t.dynamic_cluster_shape),
        shape_str(t.fallback_cluster_shape),
        t.split_k_factor,
        t.stages,
    );
    s
}

/// `log_config` — the selected tactic for one of the two GEMMs.
fn log_config(name: &str, t: &Tactic) {
    if !LOG_ENABLED {
        return;
    }
    eprintln!(
        "[pie-driver-cuda] FlashInfer MoE {name} tactic: {}",
        config_str(t)
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Default tactic selection
// ───────────────────────────────────────────────────────────────────────────

/// Why GEMM2's default epilogue is `NONE` and not `FINALIZE`.
///
/// **This constant exists to carry a measurement.** `FINALIZE` folds the
/// topk-weighted reduction and the unpermute into the GEMM epilogue, and in
/// isolation it is faster: **147.0 µs against 174.6 µs** at GLM's shapes
/// (M=128, H=6144, I=2048, E=8, topk=8). It is still not the default, because
/// of how it performs that reduction. Each expert's contribution to a token is
/// committed with `red.global.add.noftz.bf16x2` — a hardware reduction-add
/// straight to global memory, in bf16
/// (`cutlass_extensions/arch/copy_red_global.hpp`). So the topk sum
///
/// * accumulates in **bf16**, rounding after each of the topk terms, where
///   the unfused path accumulates in fp32, and
/// * adds them in whatever order the CTAs finish in, which is not the same
///   order twice.
///
/// The second point makes the whole engine irreproducible: the same prompt
/// decodes to different tokens on different runs whenever two logits land
/// within the resulting ~1 ulp. That was worth paying for a 16% cut of GEMM2
/// if it showed up end to end — it does not. **Median of 5 runs on
/// glm5.2-mini, 256 output tokens: 3518.8 vs 3515.8 tok/s at c=8 and 36824.6
/// vs 36931.3 at c=128**, i.e. a tie at both ends, because GEMM2 is a small
/// enough slice of the step that 27 µs/layer disappears into it. Determinism
/// and an fp32 reduction for nothing.
pub const DEFAULTS: i32 = seam::FUSION_NONE;

/// A tuning result, held as INDICES into the runner's candidate lists rather
/// than as configs, so that it is a pair of small integers and can be written
/// to (and matched against) an on-disk cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TacticPair {
    /// Index into the GEMM1 candidate list.
    pub gemm1: i32,
    /// Index into the GEMM2 candidate list.
    pub gemm2: i32,
}

impl Default for TacticPair {
    fn default() -> Self {
        Self {
            gemm1: -1,
            gemm2: -1,
        }
    }
}

/// The `supported_index`-th occupancy-viable candidate, optionally restricted
/// to one epilogue fusion; its list position is written to `out_index`.
///
/// The C++ re-asked `queryOccupancyForConfig` here; the seam already answered
/// it once per candidate and the answer does not depend on the problem, so
/// this reads [`Tactic::occupancy`].
fn first_supported(
    configs: &[Tactic],
    fusion: Option<i32>,
    supported_index: i32,
    name: &str,
    out_index: &mut i32,
) -> Option<Tactic> {
    let mut seen = 0_i32;
    let mut supported = 0_i32;
    let mut selected_index = -1_i32;
    let mut selected: Option<Tactic> = None;
    for (index, cfg) in configs.iter().enumerate() {
        let index = i32::try_from(index).unwrap_or(i32::MAX);
        if let Some(want) = fusion
            && cfg.fusion != want
        {
            continue;
        }
        seen += 1;
        if cfg.occupancy > 0 {
            if supported == supported_index {
                selected = Some(*cfg);
                selected_index = index;
                *out_index = index;
            }
            supported += 1;
        }
    }
    if LOG_ENABLED {
        let total = configs.len();
        if selected.is_some() {
            eprintln!(
                "[pie-driver-cuda] FlashInfer MoE {name} selected \
                 supported_index={supported_index} raw_index={selected_index} \
                 supported={supported} seen={seen} total={total}"
            );
        } else {
            eprintln!(
                "[pie-driver-cuda] FlashInfer MoE {name} no tactic for \
                 supported_index={supported_index} supported={supported} \
                 seen={seen} total={total}"
            );
        }
    }
    selected
}

// ───────────────────────────────────────────────────────────────────────────
// Per-device runner state
// ───────────────────────────────────────────────────────────────────────────

/// The C++'s `kMaxCudaDevices`. A device index at or past it is an error, not
/// a wrap: the array WAS the cache, and a modulo would have two GPUs sharing
/// one runner.
const MAX_DEVICES: usize = 16;

/// The seam handle, made shareable.
///
/// A raw pointer is neither `Send` nor `Sync`, and the C++ held this in a
/// process-wide `static` reached from every thread with no synchronisation of
/// its own. The `unsafe impl` below asserts exactly that contract and no more
/// — see the module header's note on `setTactic`/`runMoe` not being atomic
/// together, which is the one place it bites.
#[derive(Clone, Copy)]
struct Handle(*mut core::ffi::c_void);

// SAFETY: the pointee is CUTLASS's `CutlassMoeFCRunner`, whose mutable state
// is the two tactic pointers `setTactic` stores. The C++ shared one instance
// across threads on the same terms and this port changes nothing about that;
// the interleaving it permits is named in the module header rather than
// papered over with a lock the original did not hold.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

/// One device's runner, its candidate lists, its shape-blind default and its
/// in-memory memo — the C++'s `RunnerState`.
struct RunnerState {
    /// The seam handle.
    handle: Handle,
    /// The full GEMM1 candidate list, queried once at init so the autotuner
    /// does not have to re-ask.
    gemm1: Vec<Tactic>,
    /// The full GEMM2 candidate list.
    gemm2: Vec<Tactic>,
    /// The shape-blind pair installed at init.
    defaults: TacticPair,
    /// `std::unordered_map<std::uint64_t, TacticPair>` under
    /// `std::mutex tune_mutex`, at the same granularity: per device.
    tuned: Mutex<HashMap<u64, TacticPair>>,
}

/// `static std::array<RunnerState, 16>`, indexed by `cudaGetDevice`.
///
/// A `OnceLock` per slot is `std::call_once` plus the C++'s
/// `init_error` / `std::rethrow_exception` pair in one type: the first caller
/// builds, every later caller sees the same `Ok` or the same `Err` text.
static STATES: [OnceLock<Result<RunnerState, String>>; MAX_DEVICES] =
    [const { OnceLock::new() }; MAX_DEVICES];

/// `get_runner()` — this thread's device's runner, built on first use.
fn state() -> Result<&'static RunnerState, String> {
    let mut device: i32 = 0;
    let status = unsafe { cudaGetDevice(&raw mut device) };
    if status != cudaError::cudaSuccess {
        return Err(format!(
            "flashinfer CUTLASS MoE: cudaGetDevice failed: {status:?}"
        ));
    }
    let Ok(slot) = usize::try_from(device) else {
        return Err("flashinfer CUTLASS MoE: CUDA device index exceeds runner cache".to_owned());
    };
    if slot >= MAX_DEVICES {
        return Err("flashinfer CUTLASS MoE: CUDA device index exceeds runner cache".to_owned());
    }
    STATES[slot].get_or_init(build).as_ref().map_err(Clone::clone)
}

/// The whole candidate list for one GEMM.
///
/// Two seam calls where the C++ made one `getTactics`: the count, then the
/// fill. It happens once per device per process, and the alternative is a
/// guessed capacity that silently truncates the search space.
fn tactics(handle: *mut core::ffi::c_void, gemm_id: i32) -> Result<Vec<Tactic>, String> {
    let mut err = err_buf();
    let count =
        unsafe { seam::pie_moe_cutlass_tactics(handle, gemm_id, core::ptr::null_mut(), 0, err.as_mut_ptr().cast(), ERR_CAP) };
    if count < 0 {
        return Err(err_text(&err));
    }
    let len = usize::try_from(count).unwrap_or(0);
    let mut out = vec![Tactic::default(); len];
    if count > 0 {
        let filled = unsafe {
            seam::pie_moe_cutlass_tactics(
                handle,
                gemm_id,
                out.as_mut_ptr(),
                count,
                err.as_mut_ptr().cast(),
                ERR_CAP,
            )
        };
        if filled < 0 {
            return Err(err_text(&err));
        }
    }
    Ok(out)
}

/// `runner.setTactic(gemm1[pair.gemm1], gemm2[pair.gemm2])`.
fn set_tactic(handle: *mut core::ffi::c_void, pair: TacticPair) -> Result<(), String> {
    let mut err = err_buf();
    let rc = unsafe {
        seam::pie_moe_cutlass_set_tactic(
            handle,
            pair.gemm1,
            pair.gemm2,
            err.as_mut_ptr().cast(),
            ERR_CAP,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(err_text(&err))
    }
}

/// The body of `get_runner`'s `std::call_once` lambda, in order.
fn build() -> Result<RunnerState, String> {
    let mut err = err_buf();
    let handle = unsafe { seam::pie_moe_cutlass_create(err.as_mut_ptr().cast(), ERR_CAP) };
    if handle.is_null() {
        let text = err_text(&err);
        return Err(if text.is_empty() {
            "flashinfer CUTLASS MoE: runner not initialized".to_owned()
        } else {
            text
        });
    }
    let gemm1 = tactics(handle, seam::GEMM_1)?;
    let gemm2 = tactics(handle, seam::GEMM_2)?;

    // Default: first tactic the runner reports as occupancy-viable, with the
    // plain (NONE) epilogue on GEMM2. See `DEFAULTS` for the measurement that
    // rejected the alternative.
    let mut defaults = TacticPair::default();
    let best1 = first_supported(&gemm1, None, 0, "GEMM1", &mut defaults.gemm1);
    let mut best2 = first_supported(&gemm2, Some(DEFAULTS), 0, "GEMM2", &mut defaults.gemm2);
    if best2.is_none() {
        best2 = first_supported(&gemm2, None, 0, "GEMM2", &mut defaults.gemm2);
    }
    let (Some(best1), Some(best2)) = (best1, best2) else {
        return Err("flashinfer CUTLASS MoE: no supported BF16 tactics".to_owned());
    };
    log_config("GEMM1", &best1);
    log_config("GEMM2", &best2);
    set_tactic(handle, defaults)?;
    if defaults.gemm1 < 0 || defaults.gemm2 < 0 {
        return Err("flashinfer CUTLASS MoE: default tactic has no index".to_owned());
    }
    Ok(RunnerState {
        handle: Handle(handle),
        gemm1,
        gemm2,
        defaults,
        tuned: Mutex::new(HashMap::new()),
    })
}

// ───────────────────────────────────────────────────────────────────────────
// The problem, its buffers, and one run
// ───────────────────────────────────────────────────────────────────────────

/// `MOEParallelismConfig(max(1, tp_size), tp_rank, 1, 0)`'s first two fields.
///
/// The clamp is a HOST decision and lives here; the seam supplies the
/// structural `ep_size=1, ep_rank=0`, because pie runs no expert parallelism
/// and a value the caller cannot vary is not an argument.
///
/// Note it does **not** clamp `tp_rank`, and neither did the C++.
#[must_use]
pub const fn parallelism_config(tp_size: i32, tp_rank: i32) -> (i32, i32) {
    (if tp_size < 1 { 1 } else { tp_size }, tp_rank)
}

/// One MoE problem — everything the tactic key and the two seam calls read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeProblem {
    /// Tokens.
    pub num_rows: i32,
    /// Model width.
    pub hidden_size: i32,
    /// Per-expert intermediate width.
    pub inter_size: i32,
    /// Routed experts.
    pub num_experts: i32,
    /// Routes per token.
    pub experts_per_token: i32,
    /// Tensor-parallel size, UNCLAMPED as the caller gave it.
    pub tp_size: i32,
    /// Tensor-parallel rank.
    pub tp_rank: i32,
    /// The gated activation between the two grouped GEMMs.
    pub activation: MoeActivation,
}

/// The eight device pointers one run reads and writes.
#[derive(Clone, Copy)]
struct MoeBuffers {
    input: *const u16,
    token_selected_experts: *const i32,
    token_final_scales: *const f32,
    fc1_expert_weights: *const u16,
    fc2_expert_weights: *const u16,
    output: *mut u16,
    workspace: *mut u8,
    unpermuted_row_to_permuted_row: *mut i32,
}

/// `run_moe` — one `runMoe` on `stream`, with whatever tactic is installed.
///
/// # Safety
///
/// Every pointer in `b` must be a live device address of `p`'s shapes on the
/// current device, valid until the launch completes.
unsafe fn run_moe(
    handle: Handle,
    p: &MoeProblem,
    b: &MoeBuffers,
    stream: *mut core::ffi::c_void,
) -> Result<(), String> {
    let (tp_size, tp_rank) = parallelism_config(p.tp_size, p.tp_rank);
    let mut err = err_buf();
    let rc = unsafe {
        seam::pie_moe_cutlass_run(
            handle.0,
            p.activation as i32,
            b.input,
            b.token_selected_experts,
            b.token_final_scales,
            b.fc1_expert_weights,
            b.fc2_expert_weights,
            b.output,
            b.workspace,
            b.unpermuted_row_to_permuted_row,
            p.num_rows,
            p.hidden_size,
            p.inter_size,
            p.num_experts,
            p.experts_per_token,
            tp_size,
            tp_rank,
            stream,
            err.as_mut_ptr().cast(),
            ERR_CAP,
        )
    };
    if rc == 0 { Ok(()) } else { Err(err_text(&err)) }
}

// ───────────────────────────────────────────────────────────────────────────
// Shape-aware tactic selection
// ───────────────────────────────────────────────────────────────────────────

// `getTactics` returns **109 GEMM1 and 209 GEMM2** candidates, and on SM100
// every one of them reports positive occupancy -- so "first occupancy-viable"
// is really just "candidate 0", regardless of the problem. That is the wrong
// tile for almost every shape we run: a decode step is M=8 against N=4096,
// where a 256-row CTA tile wastes 97% of its rows, while a prefill at M=1024
// wants the widest tile available. CUTLASS ships no shape heuristic for
// grouped GEMM here (`ChooseWithHeuristic` is not among the returned
// configs), so measure.
//
// `setTactic` is two pointer stores, so the tactic can be chosen per call. We
// key on the problem shape, bucket M by power of two, and tune once per key
// by coordinate descent (sweep GEMM1 with GEMM2 fixed, then the reverse).
//
// Tuning runs entirely on private buffers and a private stream, sharing only
// the (read-only, long-since-written) expert weights with the caller. That
// matters because the shape we most want to tune -- decode -- is only ever
// seen from inside `cudaStreamBeginCapture`: capture takes the very first
// step of each bucket, with no eager pass in front of it. Borrowing the
// caller's stream or workspace would need cross-stream events, and those get
// swallowed into the graph. Being self-contained means tuning is just
// ordinary work on an unrelated stream, which capture does not observe.
// Timing calls (`cudaStreamSynchronize`, `cudaMalloc`) are unsafe API in the
// capture sense, but pie captures with `cudaStreamCaptureModeRelaxed`
// (`cuda_check.hpp`), which permits exactly that.

/// A candidate must be at least this much faster than the incumbent to
/// displace it.
pub const TACTIC_MARGIN: f32 = 0.98;

/// Warm-up runs before the timed ones.
const WARMUP: i32 = 3;

/// Timed runs per candidate; the best is kept.
const ITERS: i32 = 7;

/// The M bucket a row count tunes under.
///
/// Exact below and at 16 — decode concurrency lives there and the best tile
/// changes fast — power-of-two above, capped at `1 << 20`.
#[must_use]
pub const fn autotune_m_bucket(m: i32) -> i32 {
    if m <= 16 {
        return m;
    }
    let mut b = 16;
    while b < m && b < (1 << 20) {
        b <<= 1;
    }
    b
}

/// `tuning_cache.hpp`'s `tuning_hash`, spelled here so this module does not
/// need the C++ header for one fold. The constants are the algorithm's.
#[must_use]
pub const fn tuning_hash(h: u64, v: u64) -> u64 {
    h ^ (v
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(h << 6)
        .wrapping_add(h >> 2))
}

/// The cache key for a problem — the C++'s fold, in the C++'s order.
///
/// `tp_rank` is deliberately absent, as it was there: two ranks of one
/// tensor-parallel group run the same rectangle.
#[must_use]
pub fn tactic_key(p: &MoeProblem) -> u64 {
    let mut h = 0_u64;
    h = tuning_hash(h, autotune_m_bucket(p.num_rows) as u64);
    h = tuning_hash(h, p.hidden_size as u64);
    h = tuning_hash(h, p.inter_size as u64);
    h = tuning_hash(h, p.num_experts as u64);
    h = tuning_hash(h, p.experts_per_token as u64);
    h = tuning_hash(h, p.tp_size as u64);
    h = tuning_hash(h, p.activation as i32 as u64);
    h
}

/// `cudaGetLastError()`, discarded — the C++'s idiom for clearing a sticky
/// error after a failure that has already been handled.
fn clear_error() {
    let _ = unsafe { cudaGetLastError() };
}

/// The elapsed time of the fastest of [`ITERS`] runs, or `-1.0` if the tactic
/// fails.
///
/// Errors here are EXPECTED — a tile can be rejected for this problem even
/// when occupancy says otherwise — so they are swallowed and the tactic is
/// dropped from consideration.
///
/// # Safety
///
/// `b`'s pointers must be live device addresses for `p`'s shapes.
unsafe fn time_tactic(
    s: &RunnerState,
    p: &MoeProblem,
    b: &MoeBuffers,
    t: TacticPair,
    start: cudaEvent_t,
    stop: cudaEvent_t,
    stream: cudaStream_t,
) -> f32 {
    if set_tactic(s.handle.0, t).is_err() {
        clear_error();
        return -1.0;
    }
    let raw_stream = stream.cast::<core::ffi::c_void>();
    for _ in 0..WARMUP {
        if unsafe { run_moe(s.handle, p, b, raw_stream) }.is_err() {
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return -1.0;
        }
    }
    if unsafe { cudaStreamSynchronize(stream) } != cudaError::cudaSuccess {
        clear_error();
        return -1.0;
    }
    let mut best = 1e30_f32;
    for _ in 0..ITERS {
        let _ = unsafe { cudaEventRecord(start, stream) };
        if unsafe { run_moe(s.handle, p, b, raw_stream) }.is_err() {
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return -1.0;
        }
        let _ = unsafe { cudaEventRecord(stop, stream) };
        if unsafe { cudaEventSynchronize(stop) } != cudaError::cudaSuccess {
            clear_error();
            return -1.0;
        }
        let mut ms = 0.0_f32;
        if unsafe { cudaEventElapsedTime(&raw mut ms, start, stop) } != cudaError::cudaSuccess {
            clear_error();
            return -1.0;
        }
        best = best.min(ms);
    }
    best
}

/// Owns every allocation the tuner needs, so tuning never touches a buffer
/// the caller's stream might still be using.
///
/// The audit reports `TuneArena::{init, ctor, dtor}` dead and they are not:
/// `arena.init(...)` never matches a `name(` regex, and `Drop` has no call
/// site by construction. §43 records this. The Rust shape has the same
/// property and the same answer.
struct TuneArena {
    input: *mut core::ffi::c_void,
    experts: *mut core::ffi::c_void,
    scales: *mut core::ffi::c_void,
    output: *mut core::ffi::c_void,
    row_map: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    stream: cudaStream_t,
    start: cudaEvent_t,
    stop: cudaEvent_t,
}

impl TuneArena {
    /// An arena that owns nothing yet.
    const fn new() -> Self {
        Self {
            input: core::ptr::null_mut(),
            experts: core::ptr::null_mut(),
            scales: core::ptr::null_mut(),
            output: core::ptr::null_mut(),
            row_map: core::ptr::null_mut(),
            workspace: core::ptr::null_mut(),
            stream: core::ptr::null_mut(),
            start: core::ptr::null_mut(),
            stop: core::ptr::null_mut(),
        }
    }

    /// One `cudaMalloc`, with the C++'s swallow-and-report-false contract.
    fn alloc(bytes: usize, dst: &mut *mut core::ffi::c_void) -> bool {
        if unsafe { cudaMalloc(&raw mut *dst, bytes) } != cudaError::cudaSuccess {
            clear_error();
            *dst = core::ptr::null_mut();
            return false;
        }
        true
    }

    /// Allocate, create the private stream and events, and fill the routing.
    fn init(&mut self, p: &MoeProblem, workspace_bytes: usize) -> bool {
        let rows = usize::try_from(p.num_rows).unwrap_or(0);
        let hidden = usize::try_from(p.hidden_size).unwrap_or(0);
        let routes = rows * usize::try_from(p.experts_per_token).unwrap_or(0);
        let act_bytes = rows * hidden * size_of::<u16>();
        if !Self::alloc(act_bytes, &mut self.input)
            || !Self::alloc(act_bytes, &mut self.output)
            || !Self::alloc(routes * size_of::<i32>(), &mut self.experts)
            || !Self::alloc(routes * size_of::<i32>(), &mut self.row_map)
            || !Self::alloc(routes * size_of::<f32>(), &mut self.scales)
            || !Self::alloc(workspace_bytes, &mut self.workspace)
        {
            return false;
        }
        // `cudaEventCreate` is `cudaEventCreateWithFlags(.., cudaEventDefault)`,
        // and the timing the tuner needs is exactly what flag 0 leaves on.
        if unsafe { cudaStreamCreateWithFlags(&raw mut self.stream, cudaStreamNonBlocking) }
            != cudaError::cudaSuccess
            || unsafe { cudaEventCreateWithFlags(&raw mut self.start, 0) } != cudaError::cudaSuccess
            || unsafe { cudaEventCreateWithFlags(&raw mut self.stop, 0) } != cudaError::cudaSuccess
        {
            clear_error();
            eprintln!(
                "[pie-driver-cuda] FlashInfer MoE autotune stream/event setup failed"
            );
            return false;
        }

        // Activations only have to be finite -- a tensor-core GEMM's cost does
        // not depend on its values. Routing does matter, because it decides
        // how many rows land in each expert's group; round-robin is the
        // balanced case.
        let experts = usize::try_from(p.num_experts).unwrap_or(1).max(1);
        let host_experts: Vec<i32> = (0..routes)
            .map(|i| i32::try_from(i % experts).unwrap_or(0))
            .collect();
        let host_scales = vec![1.0_f32 / p.experts_per_token as f32; routes];
        let ok = unsafe { cudaMemsetAsync(self.input, 0x3C, act_bytes, self.stream) }
            == cudaError::cudaSuccess
            && unsafe {
                cudaMemcpyAsync(
                    self.experts,
                    host_experts.as_ptr().cast(),
                    routes * size_of::<i32>(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream,
                )
            } == cudaError::cudaSuccess
            && unsafe {
                cudaMemcpyAsync(
                    self.scales,
                    host_scales.as_ptr().cast(),
                    routes * size_of::<f32>(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream,
                )
            } == cudaError::cudaSuccess
            && unsafe { cudaStreamSynchronize(self.stream) } == cudaError::cudaSuccess;
        if !ok {
            clear_error();
            eprintln!(
                "[pie-driver-cuda] FlashInfer MoE autotune arena fill failed; \
                 using default tactic"
            );
        }
        ok
    }

    /// The arena's buffers as one run's operands, with the caller's expert
    /// weights — the only thing tuning shares, and it is read-only.
    fn buffers(&self, live: &MoeBuffers) -> MoeBuffers {
        MoeBuffers {
            input: self.input.cast::<u16>().cast_const(),
            token_selected_experts: self.experts.cast::<i32>().cast_const(),
            token_final_scales: self.scales.cast::<f32>().cast_const(),
            fc1_expert_weights: live.fc1_expert_weights,
            fc2_expert_weights: live.fc2_expert_weights,
            output: self.output.cast::<u16>(),
            workspace: self.workspace.cast::<u8>(),
            unpermuted_row_to_permuted_row: self.row_map.cast::<i32>(),
        }
    }
}

impl Drop for TuneArena {
    fn drop(&mut self) {
        if !self.start.is_null() {
            let _ = unsafe { cudaEventDestroy(self.start) };
        }
        if !self.stop.is_null() {
            let _ = unsafe { cudaEventDestroy(self.stop) };
        }
        if !self.stream.is_null() {
            let _ = unsafe { cudaStreamDestroy(self.stream) };
        }
        for p in [
            self.input,
            self.experts,
            self.scales,
            self.output,
            self.row_map,
            self.workspace,
        ] {
            if !p.is_null() {
                let _ = unsafe { cudaFree(p) };
            }
        }
        clear_error();
    }
}

/// Sweeps one of the two GEMMs, holding the other at the incumbent.
///
/// The winner is not simply the fastest sample. Dozens of these tactics land
/// within a percent of each other, so which one records the lowest time is
/// decided by noise — and since each tile shape accumulates in a different
/// order, letting noise pick meant the model's own output could change
/// between runs on a near-tied token. Instead every candidate within
/// [`TACTIC_MARGIN`] of the fastest is treated as tied, and the tie is broken
/// by the candidate's position in the list. That is stable as long as the
/// *set* of near-optimal tactics is stable, which it is; only the ordering
/// within it was noisy.
///
/// # Safety
///
/// `b`'s pointers must be live device addresses for `p`'s shapes.
#[allow(clippy::too_many_arguments)]
unsafe fn sweep(
    s: &RunnerState,
    configs: &[Tactic],
    is_gemm1: bool,
    p: &MoeProblem,
    b: &MoeBuffers,
    arena: &TuneArena,
    best: &mut TacticPair,
    best_ms: &mut f32,
    tried: &mut i32,
) {
    let mut best_idx = if is_gemm1 { best.gemm1 } else { best.gemm2 };
    let Ok(slot) = usize::try_from(best_idx) else {
        return;
    };
    if slot >= configs.len() {
        return;
    }
    // Tune within the chosen epilogue, never across it. Which epilogue GEMM2
    // runs decides how the topk sum is accumulated -- fp32 in a separate pass,
    // or bf16 reduction-adds in CTA completion order (see `DEFAULTS`) -- so it
    // is a numerics decision, and a tuner chasing a sub-percent timing
    // difference must not be able to reverse it.
    let fusion = configs[slot].fusion;
    let mut timings: Vec<(i32, f32)> = Vec::with_capacity(configs.len());
    let mut fastest = *best_ms;
    for (i, cfg) in configs.iter().enumerate() {
        if cfg.fusion != fusion || cfg.occupancy <= 0 {
            continue;
        }
        let i = i32::try_from(i).unwrap_or(i32::MAX);
        let mut cand = *best;
        if is_gemm1 {
            cand.gemm1 = i;
        } else {
            cand.gemm2 = i;
        }
        *tried += 1;
        let ms = unsafe { time_tactic(s, p, b, cand, arena.start, arena.stop, arena.stream) };
        if ms <= 0.0 {
            continue;
        }
        timings.push((i, ms));
        if fastest < 0.0 || ms < fastest {
            fastest = ms;
        }
    }
    if fastest <= 0.0 {
        return;
    }
    let cutoff = fastest / TACTIC_MARGIN;
    let mut chosen_ms = *best_ms;
    for &(i, ms) in &timings {
        if ms > cutoff {
            continue;
        }
        best_idx = i;
        chosen_ms = ms;
        break;
    }
    if is_gemm1 {
        best.gemm1 = best_idx;
    } else {
        best.gemm2 = best_idx;
    }
    *best_ms = chosen_ms;
}

/// Coordinate descent over the two candidate lists: GEMM1 with GEMM2 fixed,
/// then the reverse. Falls back to the device's default pair when the arena
/// cannot be built, which is the C++'s behaviour and not a silent retry.
///
/// # Safety
///
/// `live`'s expert-weight pointers must be live device addresses.
unsafe fn autotune(
    s: &RunnerState,
    p: &MoeProblem,
    live: &MoeBuffers,
    workspace_bytes: usize,
) -> TacticPair {
    let mut best = s.defaults;

    let mut arena = TuneArena::new();
    if !arena.init(p, workspace_bytes) {
        eprintln!(
            "[pie-driver-cuda] FlashInfer MoE autotune skipped for m={} h={} i={} e={} k={} \
             (arena setup failed)",
            p.num_rows, p.hidden_size, p.inter_size, p.num_experts, p.experts_per_token
        );
        return best;
    }

    let b = arena.buffers(live);
    let mut best_ms = unsafe { time_tactic(s, p, &b, best, arena.start, arena.stop, arena.stream) };
    let baseline_ms = best_ms;
    let mut tried = 0_i32;
    unsafe {
        sweep(
            s,
            &s.gemm1,
            true,
            p,
            &b,
            &arena,
            &mut best,
            &mut best_ms,
            &mut tried,
        );
        sweep(
            s,
            &s.gemm2,
            false,
            p,
            &b,
            &arena,
            &mut best,
            &mut best_ms,
            &mut tried,
        );
    }

    if LOG_ENABLED {
        let g1 = usize::try_from(best.gemm1).ok().and_then(|i| s.gemm1.get(i));
        let g2 = usize::try_from(best.gemm2).ok().and_then(|i| s.gemm2.get(i));
        eprintln!(
            "[pie-driver-cuda] FlashInfer MoE autotune m={} h={} i={} e={} k={}: \
             {:.1} us -> {:.1} us over {tried} tactics\n    GEMM1 {}\n    GEMM2 {}",
            p.num_rows,
            p.hidden_size,
            p.inter_size,
            p.num_experts,
            p.experts_per_token,
            baseline_ms * 1e3,
            best_ms * 1e3,
            g1.map_or_else(|| "none".to_owned(), config_str),
            g2.map_or_else(|| "none".to_owned(), config_str),
        );
    }
    best
}

// ───────────────────────────────────────────────────────────────────────────
// The on-disk tactic cache
// ───────────────────────────────────────────────────────────────────────────

/// The C++'s `TuningCache`, for this one file's use.
///
/// `tuning_cache.hpp` and `cache_root.hpp` ARE DELETED. This comment used to
/// read *"`tuning_cache.hpp` is NOT deleted — `gemm/gemm.cpp` still includes
/// it — so this is a second implementation of the same file format rather
/// than a move"*, and it was a trigger: `gemm/gemm.cpp` went (a 1,267-line
/// host program with zero `__global__`, and its dense autotuner is
/// `fire::gemm`), which left the two headers with **no includer anywhere**.
/// The C++ format is now described only by the two Rust re-implementations
/// that read and write it — this one and `fire::gemm`'s — so both carry it in
/// full, and the format is what follows.
///
/// Every observable is preserved: the signature is line 1, entries are
/// `%016llx %d %d`, writes APPEND (so concurrent ranks cannot truncate each
/// other, and a key written twice is harmless because the last line read
/// wins), and a file whose first line does not match is discarded AND removed
/// rather than replayed.
///
/// Still two implementations rather than one: this module is
/// `#[cfg(feature = "bridge")]` and the dense GEMM is not optional, so
/// `fire::gemm` cannot depend on it. The keys are on disk; neither copy may
/// move a constant.
struct DiskCache {
    /// Describes the hardware and candidate list the entries were measured
    /// against. Empty disables the cache, which is how a caller reports it
    /// could not identify the device.
    signature: String,
    /// Empty when no cache root could be derived at all.
    path: Option<PathBuf>,
    /// The loaded (and stored) entries.
    entries: HashMap<u64, (i32, i32)>,
}

impl DiskCache {
    /// Resolve the path and, when both it and the signature are usable, load.
    fn new(name: &str, signature: String, configured_dir: &str) -> Self {
        let path = cache_path(configured_dir, name, |k| std::env::var(k).ok());
        let mut cache = Self {
            signature,
            path,
            entries: HashMap::new(),
        };
        if !cache.signature.is_empty() && cache.path.is_some() {
            cache.load();
        }
        cache
    }

    /// The stored pair for `key`, if the file named one.
    fn lookup(&self, key: u64) -> Option<(i32, i32)> {
        self.entries.get(&key).copied()
    }

    /// Record `key` in memory and append it to the file.
    fn store(&mut self, key: u64, a: i32, b: i32) {
        self.entries.insert(key, (a, b));
        let (Some(path), false) = (self.path.as_ref(), self.signature.is_empty()) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        else {
            return;
        };
        // `std::ftell` on a freshly `fopen(.., "a")`d file answers the file's
        // size, so the C++'s `ftell(f) == 0` is "this file is new".
        let empty = file.metadata().map(|m| m.len() == 0).unwrap_or(false);
        if empty {
            let _ = writeln!(file, "{}", self.signature);
        }
        let _ = writeln!(file, "{key:016x} {a} {b}");
    }

    /// Read the file, honouring the signature.
    fn load(&mut self) {
        let Some(path) = self.path.clone() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let mut lines = text.lines();
        let matches = lines
            .next()
            .is_some_and(|first| first.trim_end_matches(['\n', '\r']) == self.signature);
        if matches {
            // `fscanf("%llx %d %d")` skips whitespace freely and stops at the
            // first token that does not parse, which is what this is.
            let mut fields = lines.flat_map(str::split_whitespace);
            while let (Some(k), Some(a), Some(b)) = (fields.next(), fields.next(), fields.next()) {
                let (Ok(k), Ok(a), Ok(b)) = (
                    u64::from_str_radix(k, 16),
                    a.parse::<i32>(),
                    b.parse::<i32>(),
                ) else {
                    break;
                };
                self.entries.insert(k, (a, b));
            }
        } else {
            // Entries measured against a different GPU or candidate list do
            // not name the kernels we would run today, so the file is worse
            // than nothing.
            self.entries.clear();
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Where the tactic file lives: the configured cache directory when the
/// engine sent one, else XDG, else `$HOME/.cache`. `None` when none of those
/// is set, which is a real configuration on a locked-down host and is why the
/// C++ returned an empty path rather than guessing.
///
/// The same derivation — and the same signature — as
/// [`crate::layout::profile_cache::cache_path`], down to treating an empty
/// string as unset (`xdg[0] != '\0'`). `env` is a parameter rather than a
/// call to [`std::env::var`] so this is testable without mutating the process
/// environment, which is unsound from a harness that runs threads in
/// parallel.
#[must_use]
pub fn cache_path(
    configured_dir: &str,
    name: &str,
    env: impl Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    if !configured_dir.is_empty() {
        return Some(Path::new(configured_dir).join(name));
    }
    if let Some(xdg) = env("XDG_CACHE_HOME").filter(|s| !s.is_empty()) {
        return Some(Path::new(&xdg).join("pie").join(name));
    }
    if let Some(home) = env("HOME").filter(|s| !s.is_empty()) {
        return Some(Path::new(&home).join(".cache").join("pie").join(name));
    }
    None
}

/// The tactic file's basename.
const CACHE_FILE: &str = "moe_tactics.txt";

/// ONE process-wide disk cache, taking the FIRST device's signature.
///
/// The C++ was a function-local `static TuningCache` inside `tactic_cache`,
/// so it was constructed once with whichever device reached it first, and
/// this is that — deliberately, not by accident. See the module header: a
/// per-device cache would put two signatures on one file, and `load` deletes
/// a file whose signature does not match, so two devices would repeatedly
/// delete each other's entries. **That first-device-signature behaviour is a
/// finding, not a fix**, and it is recorded here rather than quietly changed.
static DISK: OnceLock<Mutex<DiskCache>> = OnceLock::new();

/// `# pie-moe-tactics v1 sm<major><minor> gemm1=<n> gemm2=<n> dev=<name>`.
///
/// Empty when the device cannot be identified, which disables the cache —
/// the C++'s `return {}`.
fn tactic_cache_signature(s: &RunnerState) -> String {
    let mut device: i32 = 0;
    if unsafe { cudaGetDevice(&raw mut device) } != cudaError::cudaSuccess {
        clear_error();
        return String::new();
    }
    let mut prop = unsafe { core::mem::zeroed::<cudarc::runtime::sys::cudaDeviceProp>() };
    if unsafe { cudaGetDeviceProperties_v2(&raw mut prop, device) } != cudaError::cudaSuccess {
        clear_error();
        return String::new();
    }
    let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    format!(
        "# pie-moe-tactics v1 sm{}{} gemm1={} gemm2={} dev={name}",
        prop.major,
        prop.minor,
        s.gemm1.len(),
        s.gemm2.len(),
    )
}

/// Chooses (and on first sight of a shape, measures) the tactic pair for this
/// problem, then installs it on the runner.
///
/// The C++'s lock discipline exactly: `tune_mutex` is held across the memo
/// lookup, the disk lookup, the sweep and the final `setTactic`, and released
/// before the caller runs. See the module header for the interleaving that
/// permits.
///
/// # Safety
///
/// `b`'s expert-weight pointers must be live device addresses.
unsafe fn install_tactics(
    s: &RunnerState,
    p: &MoeProblem,
    b: &MoeBuffers,
    workspace_bytes: usize,
) -> Result<(), String> {
    let key = tactic_key(p);
    let mut tuned = s.tuned.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let disk = DISK.get_or_init(|| {
        Mutex::new(DiskCache::new(
            CACHE_FILE,
            tactic_cache_signature(s),
            // The C++ read `pie_cuda_driver::cache_dir()` here. This crate
            // has no Rust plumbing for `[cache] dir` yet -- `ProfileCache::
            // discover("")` is the same gap -- so the derivation falls
            // through to XDG. Disclosed in the module header.
            "",
        ))
    });
    let pair = if let Some(hit) = tuned.get(&key).copied() {
        hit
    } else {
        let mut disk = disk.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let cached = disk.lookup(key).filter(|&(g1, g2)| {
            g1 >= 0
                && g2 >= 0
                && usize::try_from(g1).is_ok_and(|i| i < s.gemm1.len())
                && usize::try_from(g2).is_ok_and(|i| i < s.gemm2.len())
        });
        let chosen = match cached {
            Some((gemm1, gemm2)) => TacticPair { gemm1, gemm2 },
            None => {
                let chosen = unsafe { autotune(s, p, b, workspace_bytes) };
                disk.store(key, chosen.gemm1, chosen.gemm2);
                chosen
            }
        };
        tuned.insert(key, chosen);
        chosen
    };
    set_tactic(s.handle.0, pair)
}

// ───────────────────────────────────────────────────────────────────────────
// The public surface — the five entry points the row and the model read
// ───────────────────────────────────────────────────────────────────────────

/// `flashinfer_cutlass_moe_enabled()` — the capability.
///
/// Still a bare `true`, and it still has to exist. §43's reachability audit
/// called it dead; it is not, because `model/src/qwen_3_5/forward/facts.rs`
/// (lines 78 and 185) cites it BY NAME as the reason `moe_cutlass_max_rows`
/// is non-zero — a fact the Rust side SHIPS. That citation is a claim about
/// this function's body, so the body stays until the claim moves.
#[must_use]
pub const fn enabled() -> bool {
    true
}

/// The fused path's row budget — the number `PIE_MOE_FUSED_MAX_ROWS` used to
/// default to, as a constant.
///
/// qwen3.5 sizes its CUTLASS workspace for this many rows
/// (`kFusedMoeMaxRows`); kimi and glm5 consult [`min_rows`] directly. **It is
/// not enforced here** — see [`WINDOW`].
#[must_use]
pub const fn max_rows() -> i32 {
    1024
}

/// The fused path's lower row bound — `PIE_MOE_FUSED_MIN_ROWS`'s default.
/// **Not enforced here**; see [`WINDOW`].
#[must_use]
pub const fn min_rows() -> i32 {
    0
}

/// A row window the dispatch would enforce, if one were configured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowWindow {
    /// Rows below this decline.
    pub min: i32,
    /// Rows above this decline.
    pub max: i32,
}

/// **There is no row window, and that is the C++'s answer reproduced, not a
/// simplification of it.**
///
/// # The verdict: policy, and policy that is OFF
///
/// The C++ read three environment variables here — `env_int`'s one `getenv`
/// for `PIE_MOE_FUSED_MAX_ROWS` and `fused_window_overridden`'s two — and
/// applied the window ONLY `if (fused_window_overridden())`, i.e. only when
/// one of the two variables was actually set. With neither set, both
/// comparisons were dead code. Its own comment says why enforcing the default
/// would be wrong, and the reasoning survives verbatim:
///
/// > Only an explicit override may narrow the window inside the runner. The
/// > callers already carry their own row caps (qwen3.5 sizes its workspace
/// > for `kFusedMoeMaxRows`, kimi and glm5 consult `min_rows` directly), so
/// > enforcing the default here would silently re-route their prefill batches
/// > to a different kernel — **a behaviour change dressed up as a fix**.
///
/// A refusal must reproduce the C++'s decision exactly, so this is `None` and
/// [`bf16`] declines on no row count at all. The numbers survive as
/// [`max_rows`] and [`min_rows`], which is where the callers read them.
///
/// # The open question the deletion does not get to consume
///
/// §36 requires that a stated open question outlive the variable it was
/// stated about. This is that question, from the C++, unchanged:
///
/// > `PIE_MOE_FUSED_MAX_ROWS` / `PIE_MOE_FUSED_MIN_ROWS` are documented in
/// > the header as the overrides for the fused path's row window, but both
/// > accessors returned a constant and never read the environment — so **the
/// > documented knobs did nothing and the window could not be swept against a
/// > measurement**.
///
/// Sweeping it still wants doing. Setting this to `Some(RowWindow { .. })` is
/// how, and it is a recompile rather than a variable so that the sweep and
/// the shipped behaviour cannot disagree about which arm ran.
///
/// # A knob for a function that no longer exists
///
/// `flashinfer_moe.hpp` also documented `PIE_MOE_GEMV_MAX_TOKENS`, for a
/// `moe_gemv_max_tokens(int fallback) { return fallback; }` that §43 deleted
/// as an identity function with an empty consumer set. The paragraph carried
/// one thing worth keeping, and it is a measurement: **the decode GEMV
/// dequantises int4 with the scalar FP32 ALU while the batched paths run on
/// tensor cores, so the crossover between them is far lower than a
/// weight-traffic model predicts.** That is the shape of any future window
/// here, and it is the reason a default cap would be a guess.
pub const WINDOW: Option<RowWindow> = None;

/// The workspace the fused path needs for this problem, in bytes; 0 when the
/// fused path cannot run it at all.
///
/// # The zero is the ARCH PROBE, not an error
///
/// `getWorkspaceSize` walks the TMA warp-specialized configs and throws when
/// none of them has a compiled launcher for this SM. The vendored generated
/// units cover sm80 and (behind `PIE_HAS_SM100`) sm100, so on Hopper every
/// config is unbacked and the query throws *"Could not find valid config when
/// calculating workspace size"* — which used to abort `load_model` outright.
/// Reporting zero is what the callers already handle: they leave `cutlass_ws`
/// empty and take the non-fused expert path, the same fallback the
/// sm100-without-Blackwell-kernels case was written for.
///
/// A runner that will not build reaches the same zero, because the C++ built
/// it inside the same `try`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn workspace_bytes(
    activation: MoeActivation,
    num_rows: i32,
    hidden_size: i32,
    inter_size: i32,
    num_experts: i32,
    experts_per_token: i32,
    tp_size: i32,
    tp_rank: i32,
) -> usize {
    if num_rows <= 0
        || hidden_size <= 0
        || inter_size <= 0
        || num_experts <= 0
        || experts_per_token <= 0
    {
        return 0;
    }
    let unavailable = |what: &str| {
        eprintln!(
            "[pie-driver-cuda] FlashInfer CUTLASS MoE unavailable on this device ({what}); \
             falling back to the unfused expert path"
        );
        0_usize
    };
    let s = match state() {
        Ok(s) => s,
        Err(what) => return unavailable(&what),
    };
    let (tp_size, tp_rank) = parallelism_config(tp_size, tp_rank);
    let mut bytes: usize = 0;
    let mut err = err_buf();
    let rc = unsafe {
        seam::pie_moe_cutlass_workspace_size(
            s.handle.0,
            activation as i32,
            num_rows,
            hidden_size,
            inter_size,
            num_experts,
            experts_per_token,
            tp_size,
            tp_rank,
            &raw mut bytes,
            err.as_mut_ptr().cast(),
            ERR_CAP,
        )
    };
    if rc != 0 {
        return unavailable(&err_text(&err));
    }
    if LOG_ENABLED {
        eprintln!(
            "[pie-driver-cuda] FlashInfer MoE workspace {} MiB \
             (rows={num_rows} hidden={hidden_size} inter={inter_size} \
             experts={num_experts} topk={experts_per_token})",
            bytes >> 20
        );
    }
    bytes
}

/// What [`bf16`] did.
///
/// A two-state answer rather than a `bool`, for `fire/gemv.rs`'s reason: the
/// C++ returned `bool` and its `false` meant **the window declined**, not
/// "the kernel failed". `qwen_3_5/forward/mod.rs` is only correct because it
/// means that, so *"it declined"* may not be spelled like *"it ran"*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[must_use]
pub enum Fused {
    /// The fused pipeline was issued on the caller's stream.
    Ran,
    /// Nothing was issued, `output` is untouched, and the caller must run its
    /// own expert path.
    Declined(Decline),
}

/// Why [`bf16`] declined — one variant per `return false` in the C++.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decline {
    /// One of the eight operands was null. Named, because a null here is a
    /// caller bug that the C++ turned into a silent fallback.
    NullOperand(&'static str),
    /// The row count fell outside a CONFIGURED window. Unreachable while
    /// [`WINDOW`] is `None`, which is the C++'s state; see [`WINDOW`].
    Window {
        /// The rows asked for.
        rows: i32,
        /// The window that refused them.
        window: RowWindow,
    },
    /// [`workspace_bytes`] reported 0 — the arch probe says the fused path
    /// has no compiled launcher here.
    NoWorkspace,
    /// The caller's workspace is smaller than the runner needs.
    WorkspaceTooSmall {
        /// What the caller offered.
        have: usize,
        /// What `getWorkspaceSize` asked for.
        need: usize,
    },
}

/// `moe::flashinfer_cutlass_moe_bf16` — permute, both grouped GEMMs, the
/// gated activation and the weighted finalize, as one call.
///
/// The leg decode actually takes:
/// `crates/model/src/qwen_3_5/forward/mod.rs:362` calls
/// `dsl::cuda::moe_fused_cutlass` unconditionally.
///
/// # Safety
///
/// Every pointer must be a device allocation of the stated shape on the
/// current device, live on `stream` until the launch completes. `workspace`
/// must be writable for `workspace_bytes`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn bf16(
    activation: MoeActivation,
    input: *const u16,
    token_selected_experts: *const i32,
    token_final_scales: *const f32,
    fc1_expert_weights: *const u16,
    fc2_expert_weights: *const u16,
    output: *mut u16,
    workspace: *mut u8,
    workspace_bytes_available: usize,
    unpermuted_row_to_permuted_row: *mut i32,
    num_rows: i32,
    hidden_size: i32,
    inter_size: i32,
    num_experts: i32,
    experts_per_token: i32,
    tp_size: i32,
    tp_rank: i32,
    stream: *mut core::ffi::c_void,
) -> Fused {
    for (ptr, name) in [
        (input.cast::<()>(), "input"),
        (token_selected_experts.cast::<()>(), "token_selected_experts"),
        (token_final_scales.cast::<()>(), "token_final_scales"),
        (fc1_expert_weights.cast::<()>(), "fc1_expert_weights"),
        (fc2_expert_weights.cast::<()>(), "fc2_expert_weights"),
        (output.cast_const().cast::<()>(), "output"),
        (workspace.cast_const().cast::<()>(), "workspace"),
        (
            unpermuted_row_to_permuted_row.cast_const().cast::<()>(),
            "unpermuted_row_to_permuted_row",
        ),
    ] {
        if ptr.is_null() {
            return Fused::Declined(Decline::NullOperand(name));
        }
    }
    // The row window is policy, not capacity, and it belongs here: a caller
    // that decided for itself would diverge from the others the moment the
    // window moved. Declining sends the caller to its own fallback path.
    if let Some(window) = WINDOW
        && (num_rows > window.max || num_rows < window.min)
    {
        return Fused::Declined(Decline::Window {
            rows: num_rows,
            window,
        });
    }
    let needed = workspace_bytes(
        activation,
        num_rows,
        hidden_size,
        inter_size,
        num_experts,
        experts_per_token,
        tp_size,
        tp_rank,
    );
    if needed == 0 {
        return Fused::Declined(Decline::NoWorkspace);
    }
    if workspace_bytes_available < needed {
        return Fused::Declined(Decline::WorkspaceTooSmall {
            have: workspace_bytes_available,
            need: needed,
        });
    }

    // `needed != 0` means `state()` already answered `Ok`, so this cannot be
    // the arch probe failing a second time. A failure here is a runner that
    // stopped existing between two calls, which is not a refusal -- it is a
    // broken invariant, and it panics with the symbol named.
    let s = state().unwrap_or_else(|what| {
        panic!("moe::flashinfer_cutlass_moe_bf16: runner vanished after a successful workspace query: {what}")
    });
    let problem = MoeProblem {
        num_rows,
        hidden_size,
        inter_size,
        num_experts,
        experts_per_token,
        tp_size,
        tp_rank,
        activation,
    };
    let buffers = MoeBuffers {
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        output,
        workspace,
        unpermuted_row_to_permuted_row,
    };
    // SAFETY: the caller's obligation, above.
    unsafe { install_tactics(s, &problem, &buffers, needed) }.unwrap_or_else(|what| {
        panic!("moe::flashinfer_cutlass_moe_bf16: setTactic failed: {what}")
    });
    // The runner leaves parts of its workspace untouched -- the permuted row
    // buffers are only filled for real routes, and the grouped GEMM rounds
    // each expert's row count up to a tile -- yet the padded rows are still
    // multiplied and, with a fused finalize epilogue, still scattered back.
    // Whatever the previous call left there therefore leaks into this one's
    // result. FlashInfer's own binding hides this by allocating a fresh
    // workspace per call, which a caching allocator hands back with the same
    // contents every time; pie keeps one workspace for the life of the model,
    // so the leftovers differ with every shape that ran before. That showed up
    // as the same prompt decoding to different tokens on different runs.
    let _ = unsafe {
        cudaMemsetAsync(
            workspace.cast::<core::ffi::c_void>(),
            0,
            needed,
            stream.cast(),
        )
    };
    // SAFETY: the caller's obligation, above.
    unsafe { run_moe(s.handle, &problem, &buffers, stream) }
        .unwrap_or_else(|what| panic!("moe::flashinfer_cutlass_moe_bf16: runMoe failed: {what}"));
    Fused::Ran
}

/// The routing front-end, on NVRTC.
///
/// The first of `cutlass_fused_moe_kernels.cuh`'s nineteen `__global__`s to
/// leave the ahead-of-time build. Four kernels, one unit, and the host walk
/// that drives them — which is this module, in Rust, where upstream had
/// `threeStepBuildExpertMapsSortFirstToken` and its three launchers.
///
/// # What moved, and what it replaced
///
/// | upstream, `cutlass_fused_moe_kernels.cuh`      | here                         |
/// | ---------------------------------------------- | ---------------------------- |
/// | `computeNumTokensPerBlock` `:585`              | [`tokens_per_block`]         |
/// | `blockExpertPrefixSum` `:646`                  | [`build_expert_maps`] step 1 |
/// | `globalExpertPrefixSum` `:764`                 | [`build_expert_maps`] step 2 |
/// | `mergeExpertPrefixSum` `:843`                  | [`build_expert_maps`] step 3 |
/// | `threeStepBuildExpertMapsSortFirstToken` `:898`| [`build_expert_maps`]        |
///
/// Each of the three launchers was a `cudaLaunchConfig_t`, a `cudaLaunchKernelEx`
/// and — for two of them — an `if` ladder over six function pointers
/// selecting a `cub::BlockScan` width. The ladder is gone: `expert_offsets.cuh`
/// takes its width off `blockDim.x`, so the six symbols became one and the
/// width is a number on a [`Launch`].
///
/// # The PDL attribute is NOT carried yet, and that is a stated gap
///
/// Upstream's five launchers all set
/// `cudaLaunchAttributeProgrammaticStreamSerialization` from a `bool
/// enable_pdl` threaded down from the entry point. This module fires through
/// `cuLaunchKernel`, which has no attribute list, so **PDL is off** on every
/// launch here.
///
/// That is a configuration these kernels support rather than a defect: the
/// device halves are inside upstream's own `#if __CUDA_ARCH__ >= 900` guard,
/// so on this sm_89 box they are not compiled at all, and on sm_90+ a
/// `griddepcontrol.wait` in a grid with no programmatic dependency is a
/// no-op. `enable_pdl == false` was always a legal call.
///
/// The fix, when it is wanted, is one shape and it is not this module's:
/// [`Launch`] gains a `pdl: bool` (or an attribute list), and
/// `runtime::module::fire` switches from `cuLaunchKernel` to
/// `cuLaunchKernelEx` with
/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`. Every CUDA
/// runtime launch attribute has a driver-API twin, so nothing about the
/// translation is unknown. It is not done here because `Launch` is shared
/// machinery and a launch geometry that grows a field for one caller is how
/// a struct nobody reads gets started.
pub mod routing {
    use kernels_cuda_new::runtime::{ArgValue, Args, Launch, Stream, cache};

    /// Phase one: per (expert, token-block) counts.
    const BLOCK: &str = "moe::expert_offsets_block_dev";
    /// Phase two, one element per thread.
    const SCAN: &str = "moe::expert_offsets_scan_dev";
    /// Phase two, strided, above 1024 elements.
    const SCAN_LARGE: &str = "moe::expert_offsets_scan_large_dev";
    /// Phase three: the scatter.
    const MERGE: &str = "moe::expert_offsets_merge_dev";

    /// The widest block phase two is launched at, and the point above which
    /// it switches to the strided form.
    ///
    /// `globalExpertPrefixSum` `:764` — the `if (num_elements <= 1024)` and
    /// the `config.blockDim = 1024` in the `else`.
    const MAX_BLOCK: i64 = 1024;

    /// `ceilDiv`, as `tensorrt_llm::common` spells it and both phases use it.
    ///
    /// `i64::div_ceil` rather than `(a + b - 1) / b`, which is what upstream
    /// expands to: the divisor is a block width in `[32, 1024]` at every call
    /// site, so the two agree, and the standard-library form cannot overflow
    /// on the numerator the way the fused form can.
    const fn ceil_div(a: i64, b: i64) -> i64 {
        a.div_ceil(b)
    }

    /// The tokens-per-block width phase one and phase three are launched at.
    ///
    /// `computeNumTokensPerBlock` (`:585`), transcribed exactly:
    ///
    /// ```text
    ///   for (w = 32; w <= 1024; w *= 2)
    ///     if (ceilDiv(num_tokens, w) * num_experts_per_node <= w) return w;
    ///   return 1024;
    /// ```
    ///
    /// Upstream's comment on why this predicate and not a simpler one, kept
    /// because it is the only statement of what the number is *for*:
    ///
    /// ```text
    ///   Note that both blockExpertPrefixSumKernel and
    ///   globalExpertPrefixSumKernel leverage cub::BlockScan, and their CTA
    ///   sizes are num_tokens_per_block and num_experts_per_node *
    ///   num_blocks_per_seq, respectively. computeNumTokensPerBlock tries to
    ///   find a minimum CTA size for both kernels, so that the block-level
    ///   cub::BlockScan can be efficient.
    /// ```
    ///
    /// So it is a joint minimisation over two launches, not a tile size — and
    /// the cub it names is gone from the kernels while the arithmetic it
    /// justifies is not. **That is deliberate**: the widths it walks are the
    /// six that `block_exclusive_sum_i32` needs anyway (every one a power of
    /// two in `[32, 1024]`, so every one a multiple of 32), and changing the
    /// ladder would be a performance change dressed as a cleanup, unmeasured.
    ///
    /// The `return 1024` fallback can leave phase two above `MAX_BLOCK`
    /// elements, which is exactly when [`build_expert_maps`] takes the
    /// strided form.
    #[must_use]
    pub const fn tokens_per_block(num_tokens: i64, num_experts_per_node: i64) -> i64 {
        let mut width = 32_i64;
        while width <= MAX_BLOCK {
            if ceil_div(num_tokens, width) * num_experts_per_node <= width {
                return width;
            }
            width *= 2;
        }
        MAX_BLOCK
    }

    /// Every device buffer [`build_expert_maps`] reads or writes.
    ///
    /// A struct rather than eight arguments because upstream's own signature
    /// is eleven pointers and two of the three launchers take a strict subset
    /// of them; naming them once is what makes the three calls below readable
    /// as three phases rather than three pointer lists.
    ///
    /// The three `blocked_*` members are SCRATCH — they exist only between
    /// phase one and phase three and are sized by
    /// `[num_experts_per_node, num_blocks_per_seq]`,
    /// `[num_experts_per_node, num_blocks_per_seq]` and
    /// `[num_experts_per_node, num_tokens]` respectively. The workspace that
    /// carves them is upstream's `getWsPtr(int{}, "blocked_expert_counts")`
    /// family at `:2844`, which this module does not own.
    #[derive(Clone, Copy, Debug)]
    pub struct Maps {
        /// `[num_tokens, num_experts_per_token]`, the router's output.
        pub token_selected_experts: *const i32,
        /// `[num_experts_per_node, num_blocks_per_seq]`, scratch.
        pub blocked_expert_counts: *mut i32,
        /// `[num_experts_per_node, num_blocks_per_seq]`, scratch.
        pub blocked_expert_counts_cumsum: *mut i32,
        /// `[num_experts_per_node, num_tokens]`, scratch, written sparsely.
        pub blocked_row_to_unpermuted_row: *mut i32,
        /// `[num_tokens * num_experts_per_token]`, the permuted expert ids.
        pub permuted_token_selected_experts: *mut i32,
        /// `[num_tokens * num_experts_per_token]`, the forward permutation.
        pub permuted_row_to_unpermuted_row: *mut i32,
        /// `[num_tokens * num_experts_per_token]`, the inverse permutation
        /// the finalize epilogue reads.
        pub unpermuted_row_to_permuted_row: *mut i32,
        /// `[num_experts_per_node + 1]` of `i64` — the array the grouped
        /// GEMM's `Params` is built from.
        pub expert_first_token_offset: *mut i64,
    }

    /// Build the expert maps and the first-token offsets, in three launches.
    ///
    /// `threeStepBuildExpertMapsSortFirstToken` (`:898-925`). Upstream calls
    /// `sync_check_cuda_error(stream)` between phases; that is a debug-build
    /// error check, not a synchronisation the algorithm needs — the three
    /// launches are ordered by the stream — so it has no counterpart here and
    /// its absence is not a missing barrier.
    ///
    /// # Panics
    ///
    /// On any drift between this driver and `families::moe`'s
    /// `EXPERT_OFFSETS_SIGS`, or a unit that will not compile. Every one of
    /// those is a bug in this tree rather than a condition a caller can meet.
    ///
    /// # Safety
    ///
    /// Every pointer in `maps` must be a live device allocation of the shape
    /// its field documents, on `stream`, for the duration of all three
    /// launches. `num_tokens`, `num_experts_per_node` and
    /// `num_experts_per_token` must be the shapes those allocations were
    /// sized by.
    pub unsafe fn build_expert_maps(
        maps: &Maps,
        num_tokens: i64,
        num_experts_per_node: i64,
        num_experts_per_token: i64,
        start_expert_id: i32,
        stream: *mut std::ffi::c_void,
    ) {
        let width = tokens_per_block(num_tokens, num_experts_per_node);
        let blocks_per_seq = ceil_div(num_tokens, width);

        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let (experts_u32, blocks_u32, width_u32) = (
            num_experts_per_node as u32,
            blocks_per_seq as u32,
            width as u32,
        );

        // `dim3 blocks(num_experts_per_node, num_blocks_per_seq)`,
        // `dim3 threads(num_tokens_per_block)`, `dynamicSmemBytes = 0`
        // — `:651-657`. The scratch is a static `__shared__` array sized for
        // the 1024-thread case, which is why 0 is right at every width.
        let phase_grid = Launch {
            grid: [experts_u32, blocks_u32, 1],
            block: [width_u32, 1, 1],
            smem: 0,
        };

        fire(
            BLOCK,
            phase_grid,
            &[
                ArgValue::Ptr(maps.token_selected_experts.cast_mut().cast()),
                ArgValue::Ptr(maps.blocked_expert_counts.cast()),
                ArgValue::Ptr(maps.blocked_row_to_unpermuted_row.cast()),
                ArgValue::I64(num_tokens),
                ArgValue::I64(num_experts_per_token),
                ArgValue::I32(start_expert_id),
            ],
            stream,
        );

        // `:766-800`. One block; the width is the smallest power of two in
        // `[32, 1024]` that covers the count matrix, and above 1024 elements
        // the strided form takes over at a fixed 1024.
        let elements = num_experts_per_node * blocks_per_seq;
        if elements <= MAX_BLOCK {
            let mut scan_width = 32_i64;
            while scan_width < elements {
                scan_width *= 2;
            }
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let scan = Launch {
                grid: [1, 1, 1],
                block: [scan_width as u32, 1, 1],
                smem: 0,
            };
            fire(
                SCAN,
                scan,
                &[
                    ArgValue::Ptr(maps.blocked_expert_counts.cast()),
                    ArgValue::Ptr(maps.blocked_expert_counts_cumsum.cast()),
                    ArgValue::Ptr(maps.expert_first_token_offset.cast()),
                    ArgValue::I64(num_experts_per_node),
                    ArgValue::I64(blocks_per_seq),
                ],
                stream,
            );
        } else {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let scan = Launch {
                grid: [1, 1, 1],
                block: [MAX_BLOCK as u32, 1, 1],
                smem: 0,
            };
            fire(
                SCAN_LARGE,
                scan,
                &[
                    ArgValue::Ptr(maps.blocked_expert_counts.cast()),
                    ArgValue::Ptr(maps.blocked_expert_counts_cumsum.cast()),
                    ArgValue::Ptr(maps.expert_first_token_offset.cast()),
                    ArgValue::I64(num_experts_per_node),
                    ArgValue::I64(blocks_per_seq),
                    ArgValue::I64(ceil_div(elements, MAX_BLOCK)),
                ],
                stream,
            );
        }

        // `:848-852` — phase one's grid at phase one's width. `threadIdx.x`
        // is a rank within an (expert, block) cell here, not a token index,
        // so the width has to match what phase one counted with.
        #[allow(clippy::cast_possible_truncation)]
        let num_tokens_i32 = num_tokens as i32;
        fire(
            MERGE,
            phase_grid,
            &[
                ArgValue::Ptr(maps.blocked_expert_counts.cast()),
                ArgValue::Ptr(maps.blocked_expert_counts_cumsum.cast()),
                ArgValue::Ptr(maps.blocked_row_to_unpermuted_row.cast()),
                ArgValue::Ptr(maps.permuted_token_selected_experts.cast()),
                ArgValue::Ptr(maps.permuted_row_to_unpermuted_row.cast()),
                ArgValue::Ptr(maps.unpermuted_row_to_permuted_row.cast()),
                ArgValue::I32(num_tokens_i32),
            ],
            stream,
        );
    }

    /// Resolve a row through the JIT table, bind the operands, launch.
    ///
    /// The same helper `fire/moe.rs` and `fire/gemv.rs` carry, for the same
    /// reason: `Args::bind` checks `values` against the row's signature, so a
    /// drift between a list above and `families::moe`'s `EXPERT_OFFSETS_SIGS`
    /// is a refusal here rather than a shifted argument at the kernel.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
    fn fire(
        symbol: &'static str,
        launch: Launch,
        values: &[ArgValue],
        stream: *mut std::ffi::c_void,
    ) {
        let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
            panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
        };
        let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
            panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
        };
        let module = match cache::module(index, unit) {
            Ok(module) => module,
            Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
        };
        let mut args = match Args::bind(sig, values) {
            Ok(args) => args,
            Err(why) => panic!("{symbol}: {why}"),
        };
        // SAFETY: the caller of `build_expert_maps` holds the stream live
        // across all three launches — the same assertion it made when it
        // handed the stream to `threeStepBuildExpertMapsSortFirstToken`.
        let stream = unsafe { Stream::from_runtime(stream) };
        if let Err(why) = module.fire(sig, launch, &mut args, stream) {
            panic!("{symbol}: {why}");
        }
    }
}

#[cfg(test)]
mod tests {
    //! What can be checked with no device: the pure arithmetic, the key, the
    //! formatting and the shape of the window decision.

    use super::*;

    #[test]
    fn m_buckets_are_exact_below_sixteen_and_powers_of_two_above() {
        for m in 0..=16 {
            assert_eq!(autotune_m_bucket(m), m, "exact below and at 16");
        }
        assert_eq!(autotune_m_bucket(17), 32);
        assert_eq!(autotune_m_bucket(32), 32);
        assert_eq!(autotune_m_bucket(33), 64);
        assert_eq!(autotune_m_bucket(1000), 1024);
        assert_eq!(autotune_m_bucket(1024), 1024);
        // Capped at `1 << 20`, so an absurd row count still keys somewhere.
        assert_eq!(autotune_m_bucket((1 << 20) + 1), 1 << 20);
    }

    fn problem(num_rows: i32) -> MoeProblem {
        MoeProblem {
            num_rows,
            hidden_size: 6144,
            inter_size: 2048,
            num_experts: 8,
            experts_per_token: 8,
            tp_size: 1,
            tp_rank: 0,
            activation: MoeActivation::Swiglu,
        }
    }

    #[test]
    fn the_key_buckets_m_and_ignores_rank() {
        // Same bucket, same key.
        assert_eq!(tactic_key(&problem(17)), tactic_key(&problem(31)));
        // Different bucket, different key.
        assert_ne!(tactic_key(&problem(16)), tactic_key(&problem(17)));
        // `tp_rank` is not folded in: two ranks of one group run the same
        // rectangle, which is why the C++ left it out.
        let mut other = problem(8);
        other.tp_rank = 3;
        assert_eq!(tactic_key(&problem(8)), tactic_key(&other));
        // The activation IS folded in -- it changes the epilogue.
        let mut geglu = problem(8);
        geglu.activation = MoeActivation::Geglu;
        assert_ne!(tactic_key(&problem(8)), tactic_key(&geglu));
    }

    #[test]
    fn the_hash_is_the_cpp_fold() {
        // `h ^= v + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)`, from zero.
        assert_eq!(tuning_hash(0, 0), 0x9e37_79b9_7f4a_7c15);
    }

    #[test]
    fn shapes_are_base_1000_digits() {
        assert_eq!(shape_str(0), "undef");
        assert_eq!(shape_str(1), "heuristic");
        assert_eq!(shape_str(128_256_064), "128x256x64");
        assert_eq!(shape_str(1_001_001), "1x1x1");
    }

    #[test]
    fn fusion_names_match_the_cpp_switch() {
        assert_eq!(fusion_name(seam::FUSION_NONE), "none");
        assert_eq!(fusion_name(seam::FUSION_FINALIZE), "finalize");
        assert_eq!(fusion_name(seam::FUSION_UNKNOWN), "unknown");
        assert_eq!(fusion_name(99), "unknown");
    }

    #[test]
    fn tp_size_is_clamped_and_rank_is_not() {
        assert_eq!(parallelism_config(0, 0), (1, 0));
        assert_eq!(parallelism_config(-4, 2), (1, 2));
        assert_eq!(parallelism_config(8, 7), (8, 7));
    }

    #[test]
    fn the_window_is_off_exactly_as_the_cpp_left_it() {
        // Not a style assertion: the C++ applied the window only when one of
        // `PIE_MOE_FUSED_{MAX,MIN}_ROWS` was set, so with neither set no row
        // count is refused. Enforcing the default would re-route the callers'
        // prefill batches. See `WINDOW`.
        assert!(WINDOW.is_none());
        // The numbers still ship, because the callers read them.
        assert_eq!(max_rows(), 1024);
        assert_eq!(min_rows(), 0);
    }

    #[test]
    fn the_default_epilogue_is_the_unfused_one() {
        // The FINALIZE epilogue is 147.0 us against NONE's 174.6 us in
        // isolation and is still not selected; `DEFAULTS` carries why.
        assert_eq!(DEFAULTS, seam::FUSION_NONE);
    }

    #[test]
    fn the_first_supported_tactic_skips_zero_occupancy_and_honours_fusion() {
        let configs = [
            Tactic {
                fusion: seam::FUSION_FINALIZE,
                occupancy: 4,
                ..Tactic::default()
            },
            Tactic {
                fusion: seam::FUSION_NONE,
                occupancy: 0,
                ..Tactic::default()
            },
            Tactic {
                fusion: seam::FUSION_NONE,
                occupancy: 2,
                tile: 128_256_064,
                ..Tactic::default()
            },
        ];
        let mut index = -1;
        let picked = first_supported(&configs, Some(seam::FUSION_NONE), 0, "GEMM2", &mut index);
        assert_eq!(index, 2, "the zero-occupancy NONE candidate is skipped");
        assert_eq!(picked.expect("one viable NONE candidate").tile, 128_256_064);

        // Nothing viable leaves `out_index` alone, which is what makes the
        // C++'s `default tactic has no index` check reachable.
        let none_viable = [Tactic {
            fusion: seam::FUSION_NONE,
            occupancy: 0,
            ..Tactic::default()
        }];
        let mut untouched = -1;
        assert!(first_supported(&none_viable, None, 0, "GEMM1", &mut untouched).is_none());
        assert_eq!(untouched, -1);
    }

    #[test]
    fn the_cache_path_is_the_profile_caches_derivation() {
        let path = cache_path("/cfg", CACHE_FILE, |_| None);
        assert_eq!(path.expect("configured"), Path::new("/cfg/moe_tactics.txt"));

        let xdg = cache_path(
            "",
            CACHE_FILE,
            |k| if k == "XDG_CACHE_HOME" { Some("/x".to_owned()) } else { None },
        );
        assert_eq!(xdg.expect("xdg"), Path::new("/x/pie/moe_tactics.txt"));

        let home = cache_path(
            "",
            CACHE_FILE,
            |k| if k == "HOME" { Some("/h".to_owned()) } else { None },
        );
        assert_eq!(home.expect("home"), Path::new("/h/.cache/pie/moe_tactics.txt"));

        // An empty string counts as unset, matching the C++'s `[0] != '\0'`.
        assert!(cache_path("", CACHE_FILE, |_| Some(String::new())).is_none());
        assert!(cache_path("", CACHE_FILE, |_| None).is_none());
    }

    #[test]
    fn a_decline_cannot_be_spelled_like_a_run() {
        // The whole reason this is not a `bool`.
        assert_ne!(Fused::Ran, Fused::Declined(Decline::NoWorkspace));
        assert_ne!(
            Fused::Declined(Decline::NoWorkspace),
            Fused::Declined(Decline::WorkspaceTooSmall {
                have: 1,
                need: 2
            })
        );
    }
}
