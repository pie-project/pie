//===-- cooperative_groups.h - the CCCL door, closed with a struct -------===//
//
// `cg::this_thread_block()`, `block.sync()`, and the two accessors that come
// free with them. This is what `#include <cooperative_groups.h>` resolves to
// when the compiler is NVRTC and the include path is a header set carried in
// the binary rather than a directory on a disk.
//
// # The 17 MB this file is instead of
//
// FlashInfer's attention closure -- 28 files and 17,981 lines reachable from
// `decode.cuh`, `prefill.cuh`, `mla.cuh` and the scheduler, at 0.6.15 -- is
// authored against exactly four doors into NVIDIA's CUDA C++ Core Libraries:
// three `#include <cooperative_groups.h>` and one `#include <cuda/std/limits>`.
// Behind those doors on this box is
// `/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl`: 13,691,725 bytes
// across 1691 files, 17 MB as it sits on disk.
//
// What the closure USES through them was measured, and it is a hand:
// `cg::this_thread_block()` at seven sites (four in `decode.cuh`, three in
// `prefill.cuh`), `block.sync()` at forty-nine, and `cg::this_grid()` at two.
// One method carries the forty-nine, and it is `__syncthreads()`.
//
// Forty-nine is decode plus prefill. The directory holds FIFTY: the extra one
// is in `attention/mla.cuh`, which is the third file to open this door and
// which nothing in the tree includes (`new-horizon.md` §23.7). The census is
// therefore right about the closure that COMPILES and short by one about the
// closure that is CARRIED, and both numbers are asserted by
// `tests/vendor_manifest.rs` so the distinction survives a FlashInfer bump.
//
// So this file exists, and CCCL never enters the header set. That is not a
// size optimisation -- an NVRTC compile that had to be handed 1691 files
// would be handed them from `$CUDA_HOME` at BUILD time, and this crate's
// whole claim is that it builds with no toolkit and runs with no headers on
// the machine. The 17 MB is not expensive; it is unavailable.
//
// # Why this was in THIS crate's csrc and not the archive crate's
//
// Every other device source `src/source.rs` carried was `include_str!`-ed out
// of `kernels-cuda/csrc/src` -- the ahead-of-time archive crate's tree,
// deleted at `85c6c674b` -- because while both the ahead-of-time and the JIT
// path had to run, one file is one contract and a copied `.cuh` is two
// contracts waiting to disagree. This file is not shared source. It is source
// that exists BECAUSE the compiler changed, and it must never be compiled by
// the compiler it was written to replace: that tree was on nvcc's `-I` line,
// and a file named `cooperative_groups.h` there either shadows NVIDIA's
// header or is shadowed by it, with nothing in either file saying which. It
// belongs where the compile it serves lives, which is here.
//
// # The name is the whole trick
//
// NVRTC resolves an `#include` by matching the directive's LITERAL spelling
// against the `includeNames[]` array handed to `nvrtcCreateProgram` -- angle
// brackets and quotes alike, since the array is a virtual filesystem and not
// a search path. `decode.cuh:18` says `#include <cooperative_groups.h>`, so
// the entry says `cooperative_groups.h`, so the upstream source is compiled
// UNMODIFIED and the resolution is ours. Nothing is patched, nothing is
// spliced, and a FlashInfer version bump does not have to be re-edited.
//
// `examples/header_probe.rs` measured the alternative on this box (L40S,
// NVRTC 13.0): with an empty header set, `<cooperative_groups.h>` fails with
// *"could not open source file ... (no directories in search list)"*. There
// is no third option where NVRTC finds the real one by itself.
//
// # What is deliberately NOT here
//
// **`this_grid()` and `grid_group`.** Loudly absent -- see below; the absence
// is the feature.
//
// **`tiled_partition<N>` and `thread_block_tile<N>`.** Checked and absent:
// neither appears anywhere in the attention closure. The tile showed up twice
// in this repository's reach when this was measured and neither was a reason
// to write one here. `air_top_p.cuh` and `sampling.cuh` used it, and they are
// not in the closure. So did the archive crate's
// `kernels-cuda/csrc/src/mlp/gaussian_topk.cu`, for a warp reduction, and its
// migration path is already written and is not this file: the prelude
// says so itself, in `pie_device.cuh`'s `block_sum` -- *"which is what
// `cg::tiled_partition<32>(...).shfl_down(...)` lowered to before"*. A shim
// that provided more than the measurement found would be a second
// implementation of CUDA that nobody asked for and that no test covers.
//
// # What a grid sync would cost to fake, and why nothing here fakes it
//
// `cg::this_grid()` appears twice in the closure, and a `grid.sync()` that
// merely did nothing would compile both and silently wrong-answer both:
//
//   * `decode.cuh:233`, in `SingleDecodeWithKVCacheKernel` -- a kernel this
//     driver never launches. Every decode row goes through
//     `BatchDecodeWithPagedKVCacheKernel`, which syncs no further than its
//     block.
//   * `mla.cuh:1061`, in `BatchMLAPagedAttentionKernel`, whose two stages are
//     separated by exactly that `grid.sync()`. FlashInfer launches it with
//     `cudaLaunchCooperativeKernel`.
//
// The second is the one that matters and it is not a header problem. A
// grid-wide barrier is a LAUNCH MODE: it requires the cooperative launch
// entry point and a grid small enough that every block is resident at once,
// so no block can be waiting on a block that has not been scheduled. No
// header can supply either -- a `sync()` that spun on a counter would
// deadlock the moment the grid outgrew the device, and one that returned
// immediately would let stage two read stage one's partial outputs.
//
// `kernels::LaunchRule` has no cooperative variant today; there is no rule a
// row could state and no launch path that would honour it. So the shim
// provides no `this_grid`, and a translation unit that reaches for one fails
// with a name error naming the identifier -- at compile time, on the machine
// that would have been wrong. When MLA's second stage is wanted, the work is
// a launch rule and a `cuLaunchCooperativeKernel` in `runtime::fire`, and the
// four lines that would go here are the last of it rather than the first.
//
// # THE CONDITION ABOVE IS MET, AND THESE ARE THOSE FOUR LINES
//
// `runtime::module::KernelModule::fire_ex` carries a third
// `CUlaunchAttribute` slot now -- `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`, the same
// mode `cuLaunchCooperativeKernel` sets, reached through `cuLaunchKernelEx`'s
// config struct rather than a second entry point. So the launch mode exists,
// and `grid_group` below is the last of the work rather than the first,
// exactly as this comment demanded.
//
// **The residency half is answered by measurement, not by a header.**
// `x::attn::mla_fa2` sizes its grid at `num_sm` blocks -- resident by
// construction, so no block can wait on a block that has not been scheduled.
// That is the fact that made a grid barrier safe here; it is not a property
// of this header and this header cannot check it. A caller that sizes its
// grid otherwise gets a hang, and neither the shim nor the driver will say
// so.
//
// `grid_group::sync()` lowers to `barrier.sync` over the whole grid through
// the cooperative launch's implicit grid barrier -- `bar.sync 0` is a BLOCK
// barrier and would be the silent wrong answer this comment spent forty lines
// refusing. The device-side primitive NVRTC already knows is
// `__threadfence()` plus the driver's cooperative-launch guarantee, and the
// spelling below is upstream's `cg::this_grid().sync()` mapped onto it.
//
// `decode.cuh:233`'s `SingleDecodeWithKVCacheKernel` is STILL never launched
// by this driver -- every decode row goes through
// `BatchDecodeWithPagedKVCacheKernel`, which syncs no further than its block.
// It compiles now where it did not before, and that is a widening of what
// compiles rather than of what runs.
//
//===----------------------------------------------------------------------===//
#pragma once

// Nothing is included, and nothing may be. A header in the set that reached
// for the prelude would create a diamond the includer never asked for --
// FlashInfer's `decode.cuh` has no idea `pie_device.cuh` exists and must not
// acquire a dependency on it by including NVIDIA's spelling of a header.
// Everything below is built out of what NVRTC already knows: `threadIdx`,
// `blockIdx`, `blockDim`, and `__syncthreads()`.

namespace cooperative_groups {

// NVIDIA's two device symbols for a grid barrier, declared rather than
// defined: `cicc` lowers the call and the driver resolves it at module load.
// See `grid_group::sync()` for the measurement and for what is not measured.
//
// Under `--relocatable-device-code=true` NVRTC's own builtin header declares
// both, as `__cudart_builtin__` over `enum cudaCGScope` -- declaring them
// again here is an incompatible redeclaration and the compile fails. RDC is
// also the only mode in which they can be RESOLVED, by a `cuLink` against
// `libcudadevrt.a`, so the two cases are one switch.
#if defined(__CUDACC_RDC__)
#define PIE_CG_SCOPE_GRID (::cudaCGScopeGrid)
#else
extern "C" __device__ unsigned long long cudaCGGetIntrinsicHandle(unsigned int scope);
extern "C" __device__ unsigned int cudaCGSynchronize(unsigned long long handle, unsigned int flags);
#define PIE_CG_SCOPE_GRID (1u)
#endif

/// The thread block as a group object, which is the only shape of it the
/// closure ever asks for.
///
/// Stateless on purpose. The real `thread_block` carries handles for the
/// partitioning and reduction machinery it can hand out; this one hands out
/// nothing, so it holds nothing, and `auto block = cg::this_thread_block()`
/// compiles to no register at all. That is also why every member is `const`
/// and `__forceinline__`: the call sites are inside attention inner loops,
/// and a group object that survived into the generated code would be a cost
/// the header it replaces did not have.
class thread_block {
public:
    /// The barrier the forty-nine reachable call sites are (fifty carried --
    /// see the census note above). `block.sync()` is
    /// `__syncthreads()` -- not approximately, exactly: the CUDA
    /// documentation defines `thread_block::sync()` as that instruction, and
    /// NVIDIA's own header lowers to it for a block-sized group.
    __device__ __forceinline__ void sync() const { __syncthreads(); }

    /// This thread's index within the block, x fastest, as
    /// `cooperative_groups` linearises it.
    ///
    /// Not measured in the closure -- the kernels there compute the same
    /// thing out of `threadIdx` by hand -- but free, in the sense that it is
    /// one expression with no way to be subtly wrong, and it is the accessor
    /// a caller reaches for first when it does not want to.
    __device__ __forceinline__ unsigned int thread_rank() const {
        return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    }

    /// The block's thread count, the denominator to `thread_rank`.
    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    /// `blockIdx`, under the group vocabulary's name for it.
    __device__ __forceinline__ dim3 group_index() const { return blockIdx; }

    /// `threadIdx`, likewise.
    __device__ __forceinline__ dim3 thread_index() const { return threadIdx; }
};

/// The block this thread is in.
///
/// Returns by value, and the value is empty, so this is the identity function
/// the optimiser deletes. It exists because the closure spells it seven
/// times and a name error at any one of them is a file that does not compile.
__device__ __forceinline__ thread_block this_thread_block() { return thread_block{}; }

/// The whole grid as a group object, valid only under a COOPERATIVE launch.
///
/// Stateless for `thread_block`'s reason: it hands out nothing, so it holds
/// nothing. What it is not is safe under an ordinary launch — see the header
/// note. `runtime::module::fire_ex` sets
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`, and the only caller that does is
/// `x::attn::mla_fa2`, whose grid is `num_sm` blocks and therefore resident by
/// construction.
class grid_group {
public:
    /// The grid-wide barrier separating `BatchMLAPagedAttentionKernel`'s two
    /// stages.
    ///
    /// **The two device symbols are NVIDIA's own**, the pair every real
    /// `cooperative_groups.h` lowers `this_grid().sync()` to:
    /// `cudaCGGetIntrinsicHandle(cudaCGScopeGrid)` for the handle and
    /// `cudaCGSynchronize(handle, 0)` for the barrier. Measured under NVRTC
    /// 13.0, `compute_90`, `-default-device`: **rc=0, and the PTX carries
    /// `.extern .func (.param .b32 func_retval0) cudaCGSynchronize`.**
    ///
    /// # Resolution at load is UNMEASURED, and §62 is why it is expected
    ///
    /// `.extern .func` in the PTX is a promise, not a link. Whether
    /// `cuModuleLoadData` resolves it needs a CUDA context, which no probe in
    /// this project takes.
    ///
    /// The precedent is exact and it is §62's: `cudaGraphSetConditional` is
    /// **also absent from `libcudadevrt.a`** — checked again here, and this
    /// archive contains no `cudaCG*` and no `cudaGraphSetConditional` either —
    /// and `cuModuleLoadData` resolved it anyway, because NVRTC and nvcc share
    /// `cicc` and there was never an nvcc-only lowering. The same mechanism
    /// should carry these two.
    ///
    /// **Should, not does.** If MLA's second stage fails at module load with
    /// an unresolved `cudaCGSynchronize`, this is the line, and the answer is
    /// a `-lcudadevrt` equivalent at `cuLinkAddFile` rather than anything in
    /// this header. Marked `CG_SYNC_RESOLUTION_UNMEASURED` so the integration
    /// pass finds the sentence and not the symptom.
    ///
    /// **`__syncthreads()` here would be the silent wrong answer** this
    /// header spent forty lines refusing: it would compile, run, and let
    /// stage two read stage one's partial outputs.
    __device__ __forceinline__ void sync() const {
        cudaCGSynchronize(cudaCGGetIntrinsicHandle(PIE_CG_SCOPE_GRID), 0u);
    }

    /// This thread's index within the grid, block-major.
    __device__ __forceinline__ unsigned long long thread_rank() const {
        const unsigned long long block =
            blockIdx.x + (unsigned long long)gridDim.x * (blockIdx.y + (unsigned long long)gridDim.y * blockIdx.z);
        const unsigned long long within =
            threadIdx.x + (unsigned long long)blockDim.x * (threadIdx.y + (unsigned long long)blockDim.y * threadIdx.z);
        return block * (blockDim.x * (unsigned long long)blockDim.y * blockDim.z) + within;
    }

    /// The grid's thread count.
    __device__ __forceinline__ unsigned long long size() const {
        return (unsigned long long)gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y *
               blockDim.z;
    }

    /// This CLUSTER's linear index in the grid — `%clusterid`, linearised.
    ///
    /// Added 2026-08-16 with [`cluster_group`] below; that class's banner
    /// carries the whole argument for the group, including what is measured
    /// and what is not.
    ///
    /// Under a launch that sets no cluster dimension every cluster is one
    /// block, so `%clusterid` is `blockIdx` and this returns what
    /// `blockIdx.x + ...` would. That is not a special case in the hardware:
    /// the default cluster shape is 1x1x1 and the registers report it.
    __device__ __forceinline__ unsigned int cluster_rank() const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        unsigned int x, y, z, nx, ny;
        asm("mov.u32 %0, %%clusterid.x;" : "=r"(x));
        asm("mov.u32 %0, %%clusterid.y;" : "=r"(y));
        asm("mov.u32 %0, %%clusterid.z;" : "=r"(z));
        asm("mov.u32 %0, %%nclusterid.x;" : "=r"(nx));
        asm("mov.u32 %0, %%nclusterid.y;" : "=r"(ny));
        return x + nx * (y + ny * z);
#else
        return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
#endif
    }

    /// How many clusters the grid holds — `%nclusterid`, linearised.
    __device__ __forceinline__ unsigned int num_clusters() const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        unsigned int x, y, z;
        asm("mov.u32 %0, %%nclusterid.x;" : "=r"(x));
        asm("mov.u32 %0, %%nclusterid.y;" : "=r"(y));
        asm("mov.u32 %0, %%nclusterid.z;" : "=r"(z));
        return x * y * z;
#else
        return gridDim.x * gridDim.y * gridDim.z;
#endif
    }
};

/// The thread-block cluster, as a group object — sm_90 and above.
///
/// # The site, and what it is for
///
/// Added 2026-08-16 for `flashinfer/comm/trtllm_allreduce_fusion.cuh`, which
/// reaches for it at two places, both inside
/// `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))`:
///
///   * `IndexHelper` — `grid.cluster_rank()`, `cluster.thread_rank()` and
///     `grid.num_clusters()`, which is how a token is spread across the
///     blocks of one cluster instead of one block.
///   * `FusedOp::rms_norm` — `cluster.num_blocks()`, `cluster.sync()` and
///     `cluster.map_shared_rank(&s_val, i)`, the distributed-shared-memory
///     reduction that finishes the sum of squares when a token spans more
///     than one block.
///
/// Below sm_90 upstream takes the `#else` in both places and this class is
/// never named, which is why the attention closure never needed it and why
/// `cooperative_groups.h` went without one until now.
///
/// # What is measured and what is not, stated plainly
///
/// **Measured:** every `asm` below assembles. NVRTC 13.0, `--gpu-architecture=
/// sm_90a`, carried set, no `-I`: rc=0. That covers the five special
/// registers, `barrier.cluster.arrive`/`wait` and `mapa`.
///
/// **Not measured:** whether any of it computes the right number, because
/// this box has ONE GPU and it is an sm_89 L40S, which cannot launch a
/// cluster at all. Nothing here has been run. The bodies are the PTX ISA's
/// definitions of the five registers and the two instructions, transcribed;
/// they are not a port of CCCL's `cooperative_groups/detail/cluster.h`,
/// which was not read.
///
/// **What that means for this crate today:** `kernels_cuda::comm`'s launcher
/// never sets a cluster dimension, so every launch it makes has the default
/// 1x1x1 cluster. Under that shape `num_blocks()` is 1, so `rms_norm`'s
/// cluster branch is skipped entirely and `map_shared_rank` is never called;
/// `thread_rank()` is `threadIdx` linearised and `cluster_rank()` is
/// `blockIdx` linearised, which is exactly the `#else` arm upstream takes
/// below sm_90. So on sm_90 the compiled kernel and the sm_89 one do the same
/// thing, and the parts of this class that could be wrong are the parts
/// nothing reaches. That is a property of the LAUNCHER, not of this file, and
/// it stops holding the moment anything sets `CU_LAUNCH_ATTRIBUTE_CLUSTER_
/// DIMENSION`.
class cluster_group {
public:
    /// This block's rank within the cluster — `%cluster_ctarank`.
    __device__ __forceinline__ unsigned int block_rank() const {
        unsigned int r;
        asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
        return r;
    }

    /// How many blocks the cluster holds — `%cluster_nctarank`.
    ///
    /// 1 under a launch that sets no cluster dimension, which is every launch
    /// this crate makes today. `rms_norm`'s `if (cluster.num_blocks() > 1)`
    /// is the reader that matters.
    __device__ __forceinline__ unsigned int num_blocks() const {
        unsigned int r;
        asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
        return r;
    }

    /// This thread's index within the cluster, block-major.
    ///
    /// `block_rank() * threads_per_block + thread_rank_in_block`, with the
    /// block's own rank linearised the way [`thread_block::thread_rank`] does
    /// it. `IndexHelper` uses this as `access_id_in_token`, so a wrong
    /// linearisation here is a wrong element address rather than a crash.
    __device__ __forceinline__ unsigned int thread_rank() const {
        const unsigned int within =
            threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        return block_rank() * (blockDim.x * blockDim.y * blockDim.z) + within;
    }

    /// The cluster's thread count.
    __device__ __forceinline__ unsigned int size() const {
        return num_blocks() * (blockDim.x * blockDim.y * blockDim.z);
    }

    /// The cluster-wide barrier.
    ///
    /// `barrier.cluster.arrive.release.aligned` then
    /// `barrier.cluster.wait.acquire.aligned` — the release/acquire pair, so
    /// that shared memory written before the barrier is visible to a peer
    /// block reading it through [`map_shared_rank`] after. The `.aligned`
    /// suffix asserts that every thread of every block in the cluster reaches
    /// this instruction, which is upstream's contract at both call sites
    /// (`rms_norm` calls it outside any divergent branch).
    ///
    /// The `"memory"` clobber is not decoration: without it the compiler may
    /// keep `s_val` in a register across the barrier, which is the one way a
    /// distributed-shared-memory reduction silently reads a stale value.
    __device__ __forceinline__ void sync() const {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }

    /// The address `addr` has in block `rank`'s shared memory — `mapa`.
    ///
    /// Distributed shared memory: every block of a cluster can address every
    /// other block's shared window, and `mapa` is the translation. `addr`
    /// must be a generic pointer into THIS block's shared memory; the result
    /// is a generic pointer into block `rank`'s.
    ///
    /// The generic form of the instruction is used rather than
    /// `mapa.shared::cluster`, because the call site
    /// (`FusedOp::rms_norm`'s `cluster.map_shared_rank(&s_val, i)`) hands over
    /// the address of a `__shared__` variable as a plain `float*`, which is
    /// generic.
    template <typename T>
    __device__ __forceinline__ T* map_shared_rank(T* addr, int rank) const {
        void* out;
        asm("mapa.u64 %0, %1, %2;"
            : "=l"(out)
            : "l"((void*)addr), "r"((unsigned int)rank));
        return (T*)out;
    }
};

/// The cluster this thread's block is in.
///
/// Stateless, like [`this_grid`] and [`this_thread_block`]: every member
/// reads a special register, so there is nothing to carry.
__device__ __forceinline__ cluster_group this_cluster() { return cluster_group{}; }

/// The grid this thread is in.
///
/// Spelled twice in the closure — `decode.cuh:233` in a kernel this driver
/// never launches, and `mla.cuh:1061` in `BatchMLAPagedAttentionKernel`,
/// which is the one that matters and the reason this exists at all.
__device__ __forceinline__ grid_group this_grid() { return grid_group{}; }

}  // namespace cooperative_groups

/// The conventional alias, at global scope.
///
/// NVIDIA's header does not define it -- every user writes
/// `namespace cg = cooperative_groups;` for itself, and FlashInfer writes it
/// inside `namespace flashinfer` in `decode.cuh:38` and `prefill.cuh:64`.
/// Defining it here does not collide with those: a namespace alias may be
/// redeclared in any scope as long as it names the same namespace, which this
/// does. It is here so that a translation unit which says `cg::` WITHOUT
/// declaring the alias -- a probe, a test, a kernel of ours -- gets the same
/// spelling as the sources this file was written for, rather than a second
/// convention.
namespace cg = cooperative_groups;
