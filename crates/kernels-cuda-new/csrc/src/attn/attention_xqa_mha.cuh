//===-- attention_xqa_mha.cuh - the one root the XQA lattice compiles ---===//
//
// Six units, one root, six option sets. `attn/attention_xqa.cu` and its five
// siblings in `kernels-cuda` are each a `#define` block followed by
// `#include <xqa/mha.cu>` and a host launcher; the `#define` block becomes
// `crate::unit::Unit::options`, the host launcher becomes Rust in
// `driver-cuda/src/fire/xqa.rs`, and what is left over -- the include, and
// two lines of ours -- is this file.
//
// # Why there is no `__global__` here
//
// There are three in the closure and none of them are ours.
// `xqa/mha.cu:2757` is `kernel_mha`, and it is already spelled the way a JIT
// wants:
//
// ```text
// xqa/mha.h:273        #define CUBIN_EXPORT extern "C"      // under GENERATE_CUBIN
// xqa/mha.cu:2757      CUBIN_EXPORT __global__ __launch_bounds__(256, nbCtaPerSM)
//                      void kernel_mha(...)
// ```
//
// XQA is JIT'd upstream. `GENERATE_CUBIN` is FlashInfer's own define, not
// ours: `xqa/mha.cu:2820` guards the host launcher `launchMHA` behind
// `#ifndef GENERATE_CUBIN`, and `xqa/mha_stdheaders.cuh` swaps the host
// standard library for self-defined `uint8_t`/`numeric_limits`/`min`/`max`
// under the same define. Compiling this closure with NVRTC is the mode
// upstream ships, not a mode we are inventing.
//
// # Measured, on this box
//
// NVRTC 13.0, L40S, `compute_89`, this root text verbatim, the option set of
// `attn/attention_xqa_gqa2.cu`:
//
// ```text
//   rc = 0, 0 errors, 1 .entry, PTX 327,395 bytes
//   .visible .entry kernel_mha_xqa_gqa2_bf16_p32_h128(
//   .global .align 4 .u32 pie_xqa_smem_size = 79488;
// ```
//
// and rc = 0 for every non-Hopper member of the lattice: HEAD_GRP_SIZE
// 2, 4, 5 and 8 at TOKENS_PER_PAGE 32, and HEAD_GRP_SIZE 2 at
// TOKENS_PER_PAGE 16. `pie_xqa_smem_size` is **79,488 in all five**;
// `sizeof(SharedMem)` turns out to depend on neither the head group nor the
// page size.
//
// The sixth member, `USE_SM90_MHA=1`, does NOT compile yet, and the reason is
// not the option set -- see "The Hopper arm is not ready" below.
//
// # The two lines of ours
//
// ## `kernel_mha` is renamed, and the rename is not decoration
//
// The archive renamed the HOST entry point six ways --
// `#define launchMHA launchMHA_xqa_gqa2_bf16_p32_h128`
// (`attn/attention_xqa_gqa2.cu:52`) and five more -- because six translation
// units defining the same `static` kernel symbols were about to be linked
// into one archive. Under NVRTC each unit is its own module and there is no
// linker to collide in, so it would be easy to read those `#define`s as dead
// on arrival.
//
// They are not. The collision MOVES rather than disappearing:
// `crate::unit::unit_of` resolves a symbol to a unit across the whole table,
// so six units that all export `kernel_mha` are six rows that cannot be told
// apart. The rename therefore survives the port and changes which name it
// renames -- from `launchMHA` (host, compiled out by `GENERATE_CUBIN`) to
// `kernel_mha` (device, the thing the table has to name). It is spelled as an
// option, `-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p32_h128`, and measured
// above: the `.entry` comes out under the new name and nothing else in the
// device closure refers to the old one (`kernel_mha` appears at
// `mha.cu:2757` and then only inside the `#ifndef GENERATE_CUBIN` host tail).
//
// ## `pie_xqa_smem_size` is the readable half of `configureKernel`
//
// `xqa/mha.cu:2955` is a host static initializer that does two things:
//
// ```text
// static uint32_t configureKernel() {
//   uint32_t size;
//   cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize));
//   cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
//   return size;
// }
// static uint32_t const hostSmemSize = configureKernel();   // mha.cu:2962
// ```
//
// The `cudaFuncSetAttribute` half is already in Rust and needs nothing here:
// `runtime::module::raise_dynamic_smem_cap` is called by `KernelModule::fire`
// whenever `Launch::smem` exceeds 48 KiB, keyed per `(CUdevice, CUfunction)`.
//
// The `cudaMemcpyFromSymbol` half needs a symbol to read, and upstream's is
// not readable:
//
// ```text
// xqa/mha.cu:409   CUBIN_EXPORT __device__ constexpr uint32_t smemSize = sizeof(SharedMem);
// ```
//
// `CUBIN_EXPORT` expands to `extern "C"`, which says upstream MEANT this to
// be read out of the cubin -- but `constexpr` at namespace scope is internal
// linkage, and the PTX shows it:
// `.global .align 4 .u32 smemSize = 79488;`, with no `.visible`. So this file
// re-exports it under a name of ours, which is a definition we control and a
// vendor patch we do not have to write.
//
// Measured honestly: the re-export does NOT come out `.visible` either --
// neither dropping the `const`, nor `nvrtcAddNameExpression("&pie_xqa_smem_size")`
// (which does return the lowered name `pie_xqa_smem_size`, rc = 0) changed
// the PTX. Whether `cuModuleGetGlobal` resolves a non-`.visible` `.global`
// could not be answered here: it needs a CUDA context, and the brief this
// file was written under forbids creating one. `fire/xqa.rs` therefore
// carries 79,488 as a measured constant and names this variable as the check
// that would catch it drifting, once `runtime::module` grows a
// `cuModuleGetGlobal` accessor -- which it does not have today, and which
// `attn/fa2.cuh`'s `ECHO_TEMPLATE` (`fa2.rs:527`) wants for the same reason.
//
// # The Hopper arm is not ready, and the option set is not why
//
// `USE_SM90_MHA=1` selects `xqa/mha_sm90.cu` instead of `xqa/mha.cu`.
// Measured at `compute_90a` with the `attn/attention_xqa_gqa8_sm90.cu` option
// set, device text only:
//
// ```text
//   xqa/mha_sm90.cu(1980): error: namespace "std" has no member "pair"
//   ...and 11 diagnostics cascading from that one line
// ```
//
// `std::pair` inside device code, which the header set has no answer for --
// a `<utility>` shim would be a new file and a new decision, not an option.
// And the archive unit compiles a second file first,
// `#include <xqa/tensorMap.cpp>` (`attention_xqa_gqa8_sm90.cu:56`), which is
// HOST code building `CUtensorMap`s through `cuTensorMapEncodeTiled`. It does
// not belong in a device unit at all; it is a second host-to-Rust port, and a
// larger one than `launchMHA`. Probing it anyway produced the other half of
// the answer: `csrc/shim/cuda.h` has no `CUtensorMap` and no
// `CUtensorMapDataType_enum`, so even `xqa/tensorMap.h`'s declarations do not
// parse.
//
// So: five of the six option sets are ready and the sixth is blocked on three
// separate things, none of which is a `-D`.
//
// # Every include below resolves, and NO TEST CHECKS THAT
//
// `csrc/vendor/xqa/` exists now: fifteen files, the transitive closure of
// upstream `xqa/mha.cu`'s quoted includes, 272 KB, carried automatically
// because `carried.rs` walks the directory -- *"the directory IS the set"*.
// They are carried as `xqa/mha.cuh`, `xqa/utils.cuh` and so on. Only the
// first of those names differs from upstream's, and only because this crate
// holds no `.cu`; see the include below for the check that made the rename
// safe.
//
// **The reachability gate does not cover this file.**
// `tests/layers.rs::every_include_reachable_from_a_unit_resolves` walks
// `source::quoted_includes`, which reads `#include "..."` and nothing else.
// Every include in this file is ANGLE-bracketed -- `<cuda_bf16.h>`,
// `<xqa/mha.cu>` -- so all five of them pass that test whether or not the
// header set carries them. `carried.rs`'s own header names the failure mode
// this leaves open:
//
// > It is an NVRTC *"could not open source file"* on a machine with a GPU, at
// > the first fire of whatever unit needed it, and the diagnostic names the
// > include rather than the omission.
//
// What closes it here is not a test, it is that the files are in the
// directory. That is the whole guarantee, and it is why the copy was the
// work rather than a list edit.
//
// # Measured through the real name resolution, not a simulation
//
// The numbers above were re-taken against `csrc/vendor/xqa/` as it now
// stands, with `csrc/{src,shim,vendor}` rooted so that NVRTC resolves the
// same literal names the carried set answers, and with the tree's own
// numerics contract (`--fmad=false --prec-div=true --prec-sqrt=true`,
// `runtime::nvrtc::options`) on the command line. All five non-Hopper members
// are rc = 0, and `pie_xqa_smem_size` is 79,488 in every one of them.
//
// (`compute_89` rather than `sm_89`: a probe wants PTX to read. A fire wants
// `sm_XY`, because only that makes NVRTC emit SASS -- `runtime::nvrtc` says
// so and no unit should be spelling either.)
//
// # THE JIT'D XQA WILL NOT MATCH THE ARCHIVE BIT FOR BIT
//
// Recorded here because this is the file where the two builds meet, and
// because the natural thing to write in a port's comment is that it
// reproduces what it replaced. It does not, and the difference is
// deliberate on our side.
//
// `runtime::nvrtc::options` passes `--fmad=false --prec-div=true
// --prec-sqrt=true` on every compile. The archive's CMake,
// `kernels-cuda/build.rs` and the surviving `cc::Build` pass none of the
// three, so nvcc built these same instantiations under its defaults -- and
// the default is `--fmad=true`. **The archive contracts multiply-adds into
// FMAs; the JIT refuses to.** Numerically the JIT is the stricter of the
// two, so the direction of the disagreement is known, but a
// bit-exactness claim against the archive would be false by construction.
// `new-horizon.md` §62.8.

#pragma once

// The dtype prelude, in this order. `xqa/mha.cu` reaches `__nv_bfloat16`,
// `__nv_bfloat162`, `__half`, `__half2` and `__nv_fp8x2_e4m3` without
// including a header for any of them -- upstream's build passes
// `-include cuda_fp16.h` and friends. Three includes here is the same thing
// said in text.
//
// TODAY THESE MUST RESOLVE TO THE TOOLKIT'S HEADERS, NOT `csrc/shim`'s.
// Measured: with `csrc/shim`'s the compile is 7 errors, all of them missing
// dtype intrinsics and constructors, and none of them an NVRTC limitation.
// The list is enumerated in `families/attn.rs`'s XQA lattice section. The
// shim is where they belong; two of the three headers carry written refusals
// of exactly these additions, which is why this file records the dependency
// instead of the next agent discovering it as seven mysterious errors.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// The unit. Everything that varies across the six is a `-D` in
// `Unit::options`; nothing varies here.
//
// `USE_SM90_MHA` is upstream's own switch name (`attn/attention_xqa_gqa8_sm90.cu:45`),
// and this is the one place the archive's six files disagreed about their
// include list rather than their defines.
//
// # `.cuh`, and upstream calls it `.cu`
//
// `kernels-cuda-new` holds no translation units -- 120 `.cuh` and no `.cu` --
// because that extension IS the device/host line this crate is drawing: a
// `.cu` is something nvcc compiles ahead of time, a `.cuh` is device text
// carried into NVRTC. So the vendored copy is `csrc/vendor/xqa/mha.cuh` and
// this directive names it, where `attn/attention_xqa_gqa2.cu:35` wrote
// `#include <xqa/mha.cu>`.
//
// The rename costs nothing because there is nothing to impersonate.
// `carried.rs`'s *"the name is the path, because the name is what resolves"*
// exists so that a header we do not own can be answered under the spelling
// its includer writes (`new-horizon.md` §13.4) -- and here the includer is
// this file, which is ours. Checked before renaming: **no file under
// `csrc/vendor/xqa/` includes a `.cu` by name**, across all fifteen. The
// only source that ever wrote `<xqa/mha.cu>` is the archive's six `.cu`
// files, which this port replaces.
//
// If `mha_sm90` is vendored later the same applies to it, and the same check
// applies first -- `xqa/tensorMap.cpp` is named by
// `attn/attention_xqa_gqa8_sm90.cu:37`, and if any upstream header reaches a
// `.cu` or `.cpp` by name then the spelling is upstream's and the decision is
// a different one.
#if USE_SM90_MHA
#include <xqa/mha_sm90.cuh>
#else
#include <xqa/mha.cuh>
#endif

// `configureKernel`'s readable half. See the header comment: upstream's
// `smemSize` is `constexpr`, so it has internal linkage and no `.visible`
// PTX directive; this gives the value a name a module lookup can be pointed
// at. It costs one `.u32` of constant data.
//
// `extern "C"` so the name in the module is the name written here, with no
// mangling for `runtime::module` to have to reproduce.
extern "C" __device__ unsigned int pie_xqa_smem_size = smemSize;
