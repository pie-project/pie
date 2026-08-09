//! Do `cuda_fp8.h`, `cuda_fp4.h`, `cuda/cmath` and `cuda/pipeline` compute
//! what NVIDIA's own headers compute?
//!
//! # The question this answers, and why compiling is not it
//!
//! Four files in `csrc/src` wear NVIDIA's filenames so that unmodified
//! FlashInfer source resolves them under NVRTC, where there is no include
//! path on disk and `examples/header_probe.rs` measured **0 of 31** external
//! directives answered against an empty header set. Every one of the four is
//! a chance to be quietly wrong:
//!
//! * a converter that rounds a subnormal the other way, or saturates one ulp
//!   late, produces a KV cache that is slightly off and a model that is
//!   slightly worse -- never a crash, never a stack trace;
//! * a lane swap in `cvt.rn.satfinite.e4m3x2.f32` -- the instruction packs
//!   TWO floats and a scalar conversion is that instruction with one lane
//!   discarded -- transposes every fp8 pair in the tree and still compiles;
//! * a `cuda::pipeline` that copies the right bytes and commits at the wrong
//!   time passes any single-threaded check ever written, and fails one launch
//!   in ten thousand on a customer's machine.
//!
//! So this probe does not ask whether the headers compile. It asks whether
//! they produce, bit for bit, what the vendor's implementation produces on
//! the same device from the same input -- and where no vendor implementation
//! exists on this machine, what the hardware itself produces.
//!
//! # Four gates, three kinds of reference
//!
//! | header | reference | what would slip past a weaker gate |
//! |---|---|---|
//! | `cuda_fp8.h` | `nvcc` against the real `<cuda_fp8.h>` | subnormals, saturation, NaN payloads, the packed lane order |
//! | `cuda_fp4.h` | `nvcc` against the real `<cuda_fp4.h>` | a storage width or an enumerator that disagrees across a TU boundary |
//! | `cuda/cmath` | the device's own `u32 / u32` | the `d == 1` case, where the 64-bit magic overflows and every quotient becomes zero |
//! | `cuda/pipeline` | `nvcc` against the toolkit's real `<cuda/pipeline>`, plus a host reference, plus two negative controls | a missing `wait_group` or a missing `__syncthreads()` |
//!
//! Both fp8 kernels are the SAME text compiled twice: `nvcc -cubin` against
//! `/usr/local/cuda/.../cuda_fp8.h`, and NVRTC against a [`Header`] array
//! built in this file out of `include_str!`. One source, two compilers, two
//! header sets, one device, no tolerance.
//!
//! The fp16 and bf16 types the fp8 classes convert through are supplied by
//! THIS FILE on the NVRTC side, as a nine-line ABI-compatible prelude, and by
//! `<cuda_fp16.h>`/`<cuda_bf16.h>` on the nvcc side. That is not a shortcut:
//! `csrc/src/cuda_fp16.h` and `cuda_bf16.h` belong to other work in this
//! crate, and the prelude here stands in for them so that
//! `__nv_fp8_e4m3(__half)` can be gated TODAY rather than after that lands.
//! The layouts are `unsigned short` on both sides, which is the whole of what
//! crosses the boundary being measured.
//!
//! # Why the negative controls are not optional
//!
//! Two of the pipeline rows are kernels that MUST fail. A staging test that
//! passes proves nothing unless the harness can see a race, and a race is
//! exactly the thing a quiet machine hides: the first control drops
//! `cp.async.wait_group`, the second keeps it and drops the `__syncthreads()`
//! that makes one thread's staged bytes visible to another. Both were
//! measured wrong on **300 of 300** launches. If either ever passes, this
//! probe has stopped measuring anything and every PASS above it is worthless.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example fp8_pipeline_probe
//! ```
//!
//! Needs `nvcc` on `PATH` or at `$CUDA_HOME/bin/nvcc`, and a device of
//! `sm_89` -- `cuda_fp8.h` refuses to compile below it, on purpose, because
//! that is where the `cvt` instruction it is written on begins. Exits
//! non-zero on any FAIL.

#[cfg(not(feature = "_cuda"))]
fn main() {
    // Declared with no `required-features` in `Cargo.toml`, which this file
    // does not own, so a default-feature `cargo test` compiles it. The gate
    // is here instead: layers 1 and 2 build with no CUDA at all, and a probe
    // that exists to show the toolkit is unnecessary at RUN time must not be
    // the thing that drags it in at BUILD time.
    println!(
        "fp8_pipeline_probe needs layer 3: cargo run -p kernels-cuda-new --features cuda-13 \
         --example fp8_pipeline_probe"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    std::process::exit(probe::run());
}

#[cfg(feature = "_cuda")]
#[expect(clippy::too_many_lines, reason = "a probe is a script; the report is the point")]
mod probe {
    use std::ffi::{CStr, CString, c_void};
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    use cudarc::driver::sys as dr;
    use cudarc::nvrtc::sys as nv;
    use cudarc::runtime::sys as rt;

    use kernels_cuda_new::source::{Header, as_nvrtc_arrays};

    // -----------------------------------------------------------------
    // the headers under test, carried the way they ship
    // -----------------------------------------------------------------

    /// `include_str!` and not a read: the bytes gated here are the bytes that
    /// ship, so a probe that passed against a file on disk while the binary
    /// carried something else would be measuring the wrong header.
    const CUDA_FP8: &str = include_str!("../csrc/src/cuda_fp8.h");
    const CUDA_FP4: &str = include_str!("../csrc/src/cuda_fp4.h");
    const CUDA_CMATH: &str = include_str!("../csrc/src/cuda/cmath");
    const CUDA_PIPELINE: &str = include_str!("../csrc/src/cuda/pipeline");

    /// The two shims this set already had, for the size table only. They are
    /// not under test here -- `cg_probe.rs` and the closure gate those -- but
    /// the CCCL comparison is a claim about the SET, and a set of two is a
    /// different claim from a set of four.
    const COOPERATIVE_GROUPS: &str = include_str!("../csrc/src/cooperative_groups.h");
    const CUDA_STD_LIMITS: &str = include_str!("../csrc/src/cuda/std/limits");

    /// CCCL as the toolkit ships it, measured on this machine at
    /// `/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl`. Constants
    /// rather than a walk of that tree, because the comparison must still
    /// read the same on a machine with no toolkit, and because the number is
    /// the one `.wiki/driver/new-horizon.md` §13.5 already quotes.
    const CCCL_BYTES: usize = 13_691_725;
    const CCCL_FILES: usize = 1_691;

    /// NVIDIA's fp8 and fp4 headers, likewise measured: `cuda_fp8.h` +
    /// `cuda_fp8.hpp` and `cuda_fp4.h` + `cuda_fp4.hpp`, which is what a
    /// vendoring decision would have had to carry.
    const NVIDIA_FP8_BYTES: usize = 19_143 + 100_356;
    const NVIDIA_FP4_BYTES: usize = 13_823 + 36_158;

    // -----------------------------------------------------------------
    // the kernels, written once and compiled twice
    // -----------------------------------------------------------------

    /// `__half` and `__nv_bfloat16` for the NVRTC side, at the ABI the fp8
    /// classes actually cross.
    ///
    /// Sixteen bits of storage, a constructor from the `_raw` struct and a
    /// conversion back to it -- which is the entire surface `cuda_fp8.h`
    /// borrows, as its own banner says. The macros are the ones NVIDIA's
    /// headers define and the ones the shim's interop is guarded on, so
    /// defining them here compiles exactly the blocks that would otherwise
    /// wait for `csrc/src/cuda_fp16.h` to exist.
    const FP16_PRELUDE: &str = concat!(
        "#define __CUDA_FP16_TYPES_EXIST__\n",
        "#define __CUDA_BF16_TYPES_EXIST__\n",
        "struct __half_raw { unsigned short x; };\n",
        "struct __half {\n",
        "    unsigned short __x;\n",
        "    __half() = default;\n",
        "    __device__ __half(const __half_raw r) { __x = r.x; }\n",
        "    __device__ operator __half_raw() const { __half_raw r; r.x = __x; return r; }\n",
        "};\n",
        "struct __nv_bfloat16_raw { unsigned short x; };\n",
        "struct __nv_bfloat16 {\n",
        "    unsigned short __x;\n",
        "    __nv_bfloat16() = default;\n",
        "    __device__ __nv_bfloat16(const __nv_bfloat16_raw r) { __x = r.x; }\n",
        "    __device__ operator __nv_bfloat16_raw() const {\n",
        "        __nv_bfloat16_raw r; r.x = __x; return r;\n",
        "    }\n",
        "};\n",
    );

    /// Every fp8 conversion the two trees were measured to reach, in three
    /// kernels, shared verbatim by both compilers.
    ///
    /// Written once rather than twice because the claim is that these calls
    /// MEAN the same thing in both worlds -- two copies would let a typo make
    /// that true by accident.
    const BODY_FP8: &str = concat!(
        "extern \"C\" __global__ void fp8_unpack(\n",
        "    const unsigned char* in,\n",
        "    unsigned short* cvt_e4m3, unsigned short* cvt_e5m2,\n",
        "    unsigned int* flt_e4m3, unsigned int* flt_e5m2,\n",
        "    unsigned short* hlf_e4m3, unsigned short* hlf_e5m2,\n",
        "    unsigned short* bf_e4m3, unsigned short* bf_e5m2)\n",
        "{\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i >= 256) { return; }\n",
        "    const __nv_fp8_storage_t s = (__nv_fp8_storage_t)in[i];\n",
        "    cvt_e4m3[i] = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3).x;\n",
        "    cvt_e5m2[i] = __nv_cvt_fp8_to_halfraw(s, __NV_E5M2).x;\n",
        "    __nv_fp8_e4m3 a; a.__x = s;\n",
        "    __nv_fp8_e5m2 b; b.__x = s;\n",
        "    flt_e4m3[i] = __float_as_uint(float(a));\n",
        "    flt_e5m2[i] = __float_as_uint(float(b));\n",
        "    hlf_e4m3[i] = static_cast<__half_raw>(__half(a)).x;\n",
        "    hlf_e5m2[i] = static_cast<__half_raw>(__half(b)).x;\n",
        "    bf_e4m3[i] = static_cast<__nv_bfloat16_raw>(__nv_bfloat16(a)).x;\n",
        "    bf_e5m2[i] = static_cast<__nv_bfloat16_raw>(__nv_bfloat16(b)).x;\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void fp8_pack(\n",
        "    const float* in,\n",
        "    unsigned char* cvt_e4m3, unsigned char* cvt_e5m2,\n",
        "    unsigned short* pair_e4m3, unsigned short* pair_e5m2,\n",
        "    unsigned char* cls_e4m3, unsigned char* cls_e5m2, int n)\n",
        "{\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i >= n) { return; }\n",
        "    const float v = in[i];\n",
        "    cvt_e4m3[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);\n",
        "    cvt_e5m2[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E5M2);\n",
        "    cls_e4m3[i] = __nv_fp8_e4m3(v).__x;\n",
        "    cls_e5m2[i] = __nv_fp8_e5m2(v).__x;\n",
        "    if ((i & 1) == 0) {\n",
        "        const float2 pair = make_float2(v, in[i + 1]);\n",
        "        pair_e4m3[i >> 1] = __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);\n",
        "        pair_e5m2[i >> 1] = __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E5M2);\n",
        "    }\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void fp8_from16(\n",
        "    const unsigned short* bits,\n",
        "    unsigned char* e4_half, unsigned char* e5_half,\n",
        "    unsigned char* e4_bf16, unsigned char* e5_bf16, int n)\n",
        "{\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i >= n) { return; }\n",
        "    __half_raw hr; hr.x = bits[i];\n",
        "    __nv_bfloat16_raw br; br.x = bits[i];\n",
        "    const __half h = static_cast<__half>(hr);\n",
        "    const __nv_bfloat16 t = static_cast<__nv_bfloat16>(br);\n",
        "    e4_half[i] = __nv_fp8_e4m3(h).__x;\n",
        "    e5_half[i] = __nv_fp8_e5m2(h).__x;\n",
        "    e4_bf16[i] = __nv_fp8_e4m3(t).__x;\n",
        "    e5_bf16[i] = __nv_fp8_e5m2(t).__x;\n",
        "}\n",
    );

    /// The fp4 ABI, which is all `cuda_fp4.h` claims: three storage widths,
    /// three class layouts, one enumerator, and that `__x` round-trips.
    ///
    /// There is no conversion here because the header refuses to carry one --
    /// `cvt.rn.satfinite.e2m1x2.f32` is `sm_100`+ and `ptxas` refuses the
    /// text at `sm_89`, so there is nothing on this box that could be
    /// compared. What CAN disagree between a shim and the vendor is exactly
    /// what is measured: a width, an alignment, an enumerator value.
    const BODY_FP4: &str = concat!(
        "extern \"C\" __global__ void fp4_facts(unsigned int* o)\n",
        "{\n",
        "    o[0] = (unsigned int)sizeof(__nv_fp4_storage_t);\n",
        "    o[1] = (unsigned int)sizeof(__nv_fp4x2_storage_t);\n",
        "    o[2] = (unsigned int)sizeof(__nv_fp4x4_storage_t);\n",
        "    o[3] = (unsigned int)sizeof(__nv_fp4_e2m1);\n",
        "    o[4] = (unsigned int)sizeof(__nv_fp4x2_e2m1);\n",
        "    o[5] = (unsigned int)sizeof(__nv_fp4x4_e2m1);\n",
        "    o[6] = (unsigned int)alignof(__nv_fp4_e2m1);\n",
        "    o[7] = (unsigned int)alignof(__nv_fp4x2_e2m1);\n",
        "    o[8] = (unsigned int)alignof(__nv_fp4x4_e2m1);\n",
        "    o[9] = (unsigned int)__NV_E2M1;\n",
        "    __nv_fp4_e2m1 a; a.__x = (__nv_fp4_storage_t)0xA5;\n",
        "    __nv_fp4x2_e2m1 b; b.__x = (__nv_fp4x2_storage_t)0x5A;\n",
        "    __nv_fp4x4_e2m1 c; c.__x = (__nv_fp4x4_storage_t)0x1234;\n",
        "    o[10] = (unsigned int)a.__x | ((unsigned int)b.__x << 8);\n",
        "    o[11] = (unsigned int)c.__x;\n",
        "    __nv_fp4_interpretation_t kind = __NV_E2M1;\n",
        "    o[12] = (unsigned int)kind;\n",
        "}\n",
    );

    /// A two-stage staged copy with a neighbour read, shared by both
    /// compilers.
    ///
    /// The neighbour read is the whole design. Every thread stages ITS OWN
    /// element and then reads the one 37 lanes along, so a result that is
    /// correct proves the staged tile became visible ACROSS threads, which is
    /// the property `cp.async.wait_group` alone does not give and the
    /// `__syncthreads()` in `consumer_wait` does. Thirty-seven because it is
    /// coprime to 256 and crosses a warp boundary; a neighbour inside the
    /// same warp would be hidden by lockstep.
    ///
    /// `cooperative_groups::this_thread_block()` on both sides, spelled the
    /// same, because both header sets have it: the toolkit's, and this
    /// crate's own `csrc/src/cooperative_groups.h`.
    const BODY_PIPE: &str = concat!(
        "#define STAGES 2\n",
        "#define TILE 256\n",
        "extern \"C\" __global__ void pipe_stage(\n",
        "    const float* __restrict__ g, float* __restrict__ out, int tiles)\n",
        "{\n",
        "    __shared__ float smem[STAGES][TILE];\n",
        "    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> state;\n",
        "    auto pipe = cuda::make_pipeline(cooperative_groups::this_thread_block(), &state);\n",
        "    const int t = (int)threadIdx.x;\n",
        "    int fetch = 0;\n",
        "    for (; fetch < STAGES && fetch < tiles; ++fetch) {\n",
        "        pipe.producer_acquire();\n",
        "        cuda::memcpy_async(&smem[fetch % STAGES][t], &g[fetch * TILE + t],\n",
        "                           sizeof(float), pipe);\n",
        "        pipe.producer_commit();\n",
        "    }\n",
        "    for (int c = 0; c < tiles; ++c) {\n",
        "        pipe.consumer_wait();\n",
        "        out[c * TILE + t] = smem[c % STAGES][(t + 37) % TILE];\n",
        "        pipe.consumer_release();\n",
        "        if (fetch < tiles) {\n",
        "            pipe.producer_acquire();\n",
        "            cuda::memcpy_async(&smem[fetch % STAGES][t], &g[fetch * TILE + t],\n",
        "                               sizeof(float), pipe);\n",
        "            pipe.producer_commit();\n",
        "            ++fetch;\n",
        "        }\n",
        "    }\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void pipe_stage16(\n",
        "    const float* __restrict__ g, float* __restrict__ out, int tiles)\n",
        "{\n",
        "    __shared__ float smem[STAGES][TILE * 4];\n",
        "    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> state;\n",
        "    auto pipe = cuda::make_pipeline(cooperative_groups::this_thread_block(), &state);\n",
        "    const int t = (int)threadIdx.x;\n",
        "    int fetch = 0;\n",
        "    for (; fetch < STAGES && fetch < tiles; ++fetch) {\n",
        "        pipe.producer_acquire();\n",
        "        cuda::memcpy_async(&smem[fetch % STAGES][t * 4], &g[fetch * TILE * 4 + t * 4],\n",
        "                           cuda::aligned_size_t<16>(16), pipe);\n",
        "        pipe.producer_commit();\n",
        "    }\n",
        "    for (int c = 0; c < tiles; ++c) {\n",
        "        pipe.consumer_wait();\n",
        "        const int nb = ((t + 37) % TILE) * 4;\n",
        "        for (int k = 0; k < 4; ++k) {\n",
        "            out[c * TILE * 4 + t * 4 + k] = smem[c % STAGES][nb + k];\n",
        "        }\n",
        "        pipe.consumer_release();\n",
        "        if (fetch < tiles) {\n",
        "            pipe.producer_acquire();\n",
        "            cuda::memcpy_async(&smem[fetch % STAGES][t * 4], &g[fetch * TILE * 4 + t * 4],\n",
        "                               cuda::aligned_size_t<16>(16), pipe);\n",
        "            pipe.producer_commit();\n",
        "            ++fetch;\n",
        "        }\n",
        "    }\n",
        "}\n",
    );

    /// The same staging loop with the synchronisation taken out, twice.
    ///
    /// Raw `cp.async` and no header at all, so that what is being measured is
    /// the HARNESS -- whether a neighbour read can see a race -- and not
    /// another statement of the shim. `unsync` drops the wait and the
    /// barrier; `nobarrier` keeps the wait and drops only the
    /// `__syncthreads()`, which is the subtler of the two mistakes and the
    /// one a single-threaded test cannot reach.
    const BODY_CONTROLS: &str = concat!(
        "#define STAGES 2\n",
        "#define TILE 256\n",
        "__device__ __forceinline__ void issue(float* dst, const float* src)\n",
        "{\n",
        "    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);\n",
        "    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 4;\\n\"\n",
        "                 :: \"r\"(d), \"l\"(src) : \"memory\");\n",
        "    asm volatile(\"cp.async.commit_group;\\n\" ::: \"memory\");\n",
        "}\n",
        "extern \"C\" __global__ void unsync(\n",
        "    const float* __restrict__ g, float* __restrict__ out, int tiles)\n",
        "{\n",
        "    __shared__ float smem[STAGES][TILE];\n",
        "    const int t = (int)threadIdx.x;\n",
        "    int fetch = 0;\n",
        "    for (; fetch < STAGES && fetch < tiles; ++fetch) {\n",
        "        issue(&smem[fetch % STAGES][t], &g[fetch * TILE + t]);\n",
        "    }\n",
        "    for (int c = 0; c < tiles; ++c) {\n",
        "        out[c * TILE + t] = smem[c % STAGES][(t + 37) % TILE];\n",
        "        if (fetch < tiles) { issue(&smem[fetch % STAGES][t], &g[fetch * TILE + t]); ++fetch; }\n",
        "    }\n",
        "}\n",
        "extern \"C\" __global__ void nobarrier(\n",
        "    const float* __restrict__ g, float* __restrict__ out, int tiles)\n",
        "{\n",
        "    __shared__ float smem[STAGES][TILE];\n",
        "    const int t = (int)threadIdx.x;\n",
        "    int fetch = 0;\n",
        "    for (; fetch < STAGES && fetch < tiles; ++fetch) {\n",
        "        issue(&smem[fetch % STAGES][t], &g[fetch * TILE + t]);\n",
        "    }\n",
        "    for (int c = 0; c < tiles; ++c) {\n",
        "        asm volatile(\"cp.async.wait_group 1;\\n\" ::: \"memory\");\n",
        "        out[c * TILE + t] = smem[c % STAGES][(t + 37) % TILE];\n",
        "        if (fetch < tiles) { issue(&smem[fetch % STAGES][t], &g[fetch * TILE + t]); ++fetch; }\n",
        "    }\n",
        "}\n",
    );

    /// `cuda::fast_mod_div` against the divide instruction, on the device, in
    /// one kernel.
    ///
    /// The comparison is done where the values are rather than shipped home,
    /// because the sweep is a quarter of a billion pairs and a round trip per
    /// pair would make the probe a benchmark of `cudaMemcpy`. What comes back
    /// is a mismatch count and the first offending triple, which is what a
    /// failure report needs.
    ///
    /// `atomicCAS` on the flag rather than a plain store, so that "first" is
    /// the first one FOUND rather than whichever thread wrote last -- the
    /// value of a first-differing input is that it is reproducible.
    const BODY_CMATH: &str = concat!(
        "extern \"C\" __global__ void divide(\n",
        "    const unsigned int* dividends, int count, unsigned int lo, unsigned int hi,\n",
        "    unsigned int* mismatches, unsigned int* first)\n",
        "{\n",
        "    const unsigned int d = lo + (unsigned int)(blockIdx.y * gridDim.x + blockIdx.x);\n",
        "    if (d > hi) { return; }\n",
        "    const cuda::fast_mod_div<unsigned int> fast(d);\n",
        "    for (int i = (int)threadIdx.x; i < count; i += (int)blockDim.x) {\n",
        "        const unsigned int n = dividends[i];\n",
        "        const unsigned int got = n / fast;\n",
        "        const unsigned int want = n / d;\n",
        "        if (got != want) {\n",
        "            atomicAdd(mismatches, 1u);\n",
        "            if (atomicCAS(&first[0], 0u, 1u) == 0u) {\n",
        "                first[1] = d; first[2] = n; first[3] = got; first[4] = want;\n",
        "            }\n",
        "        }\n",
        "    }\n",
        "}\n",
    );

    // -----------------------------------------------------------------
    // one row of the report
    // -----------------------------------------------------------------

    /// A check, its size, and -- when it failed -- the input that broke it.
    struct Row {
        what: &'static str,
        inputs: usize,
        passed: bool,
        /// The first disagreement: what went in, what the shim said, what the
        /// reference said. All in hex, because a converter's bug is a bit
        /// pattern and a decimal rendering of a NaN hides which one it is.
        detail: String,
    }

    impl Row {
        fn pass(what: &'static str, inputs: usize) -> Self {
            Self { what, inputs, passed: true, detail: String::new() }
        }
        fn fail(what: &'static str, inputs: usize, detail: String) -> Self {
            Self { what, inputs, passed: false, detail }
        }
    }

    /// Compare two byte slices as fixed-width little-endian words.
    ///
    /// Returns the row directly so that every conversion check reads the same
    /// at the call site, and so that "first differing" means the same thing
    /// in all sixteen of them: lowest index, not lowest address.
    fn compare_words(
        what: &'static str,
        width: usize,
        got: &[u8],
        want: &[u8],
        input: impl Fn(usize) -> String,
    ) -> Row {
        let count = got.len() / width;
        for i in 0..count {
            let g = &got[i * width..(i + 1) * width];
            let w = &want[i * width..(i + 1) * width];
            if g != w {
                return Row::fail(
                    what,
                    count,
                    format!("in {} shim {} nvcc {}", input(i), hex_le(g), hex_le(w)),
                );
            }
        }
        Row::pass(what, count)
    }

    /// Little-endian bytes as the number they are: `0x3c00`, not `0x003c`.
    fn hex_le(bytes: &[u8]) -> String {
        let mut value = 0u64;
        for (i, b) in bytes.iter().enumerate() {
            value |= u64::from(*b) << (8 * i);
        }
        format!("0x{value:0width$x}", width = bytes.len() * 2)
    }

    // -----------------------------------------------------------------
    // the run
    // -----------------------------------------------------------------

    pub fn run() -> i32 {
        let Some(arch) = kernels_cuda_new::runtime::cache::arch() else {
            println!("no CUDA device is current; this probe needs one to compare on");
            return 1;
        };

        println!("fp8 / fp4 / cmath / pipeline parity probe -- four shims against the vendor\n");
        println!("  device        {} ({arch})", device_name());
        println!("  NVRTC         {}", nvrtc_version());
        let Some(nvcc) = find_nvcc() else {
            println!(
                "\n  no `nvcc` on PATH or at $CUDA_HOME/bin/nvcc.\n\
                 The reference path IS the check -- a hand-written converter compared\n\
                 against itself proves nothing -- so there is nothing to report without it."
            );
            return 1;
        };
        println!("  nvcc          {}", nvcc.display());

        let fp8 = Header { name: "cuda_fp8.h", text: CUDA_FP8 };
        let fp4 = Header { name: "cuda_fp4.h", text: CUDA_FP4 };
        let cmath = Header { name: "cuda/cmath", text: CUDA_CMATH };
        let pipeline = Header { name: "cuda/pipeline", text: CUDA_PIPELINE };
        let cg = Header { name: "cooperative_groups.h", text: COOPERATIVE_GROUPS };

        let mut rows: Vec<Row> = Vec::new();
        let mut compile_failures = 0usize;

        println!("\ncompiling:\n");

        // -------------------------------------------------------------
        // fp8
        // -------------------------------------------------------------
        let fp8_reference = format!(
            "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n\n{BODY_FP8}"
        );
        let fp8_under_test = format!("{FP16_PRELUDE}#include <cuda_fp8.h>\n\n{BODY_FP8}");

        let fp8_pair = build_pair(
            "cuda_fp8.h",
            &nvcc,
            arch,
            &fp8_reference,
            &fp8_under_test,
            &[fp8],
            &mut compile_failures,
        );

        // -------------------------------------------------------------
        // fp4
        // -------------------------------------------------------------
        let fp4_reference = format!("#include <cuda_fp4.h>\n\n{BODY_FP4}");
        let fp4_under_test = format!("#include <cuda_fp4.h>\n\n{BODY_FP4}");
        let fp4_pair = build_pair(
            "cuda_fp4.h",
            &nvcc,
            arch,
            &fp4_reference,
            &fp4_under_test,
            &[fp4],
            &mut compile_failures,
        );

        // -------------------------------------------------------------
        // pipeline
        // -------------------------------------------------------------
        let pipe_reference = format!(
            "#include <cooperative_groups.h>\n#include <cuda/pipeline>\n\n{BODY_PIPE}"
        );
        let pipe_under_test = format!(
            "#include <cooperative_groups.h>\n#include <cuda/pipeline>\n\n{BODY_PIPE}"
        );
        let pipe_pair = build_pair(
            "cuda/pipeline",
            &nvcc,
            arch,
            &pipe_reference,
            &pipe_under_test,
            &[pipeline, cg],
            &mut compile_failures,
        );

        // -------------------------------------------------------------
        // cmath and the controls: NVRTC only
        // -------------------------------------------------------------
        let cmath_built = compile_with_nvrtc(
            &format!("#include <cuda/cmath>\n\n{BODY_CMATH}"),
            arch,
            &[cmath],
        );
        match &cmath_built {
            Ok(built) => println!("  under test  NVRTC  cuda/cmath           {:8.1} ms", built.millis),
            Err(why) => {
                println!("  under test  NVRTC  cuda/cmath  REFUSED:\n{why}");
                compile_failures += 1;
            }
        }
        let controls = compile_with_nvrtc(BODY_CONTROLS, arch, &[]);
        match &controls {
            Ok(built) => {
                println!("  control     NVRTC  no headers           {:8.1} ms", built.millis);
            }
            Err(why) => {
                println!("  control     NVRTC  REFUSED:\n{why}");
                compile_failures += 1;
            }
        }

        if compile_failures > 0 {
            println!("\n{compile_failures} translation unit(s) refused to compile; nothing was measured.");
            return 1;
        }

        // -------------------------------------------------------------
        // fp8: the sweeps
        // -------------------------------------------------------------
        if let Some((reference, under_test)) = fp8_pair.as_ref() {
            match fp8_checks(reference, under_test) {
                Ok(mut got) => rows.append(&mut got),
                Err(why) => {
                    rows.push(Row::fail("cuda_fp8.h (all)", 0, why));
                }
            }
        }
        if let Some((reference, under_test)) = fp4_pair.as_ref() {
            match fp4_checks(reference, under_test) {
                Ok(row) => rows.push(row),
                Err(why) => rows.push(Row::fail("cuda_fp4.h ABI", 0, why)),
            }
        }
        if let Ok(built) = &cmath_built {
            match cmath_checks(&built.image) {
                Ok(mut got) => rows.append(&mut got),
                Err(why) => rows.push(Row::fail("cuda::fast_mod_div", 0, why)),
            }
        }
        if let (Some((reference, under_test)), Ok(controls)) = (pipe_pair.as_ref(), &controls) {
            match pipeline_checks(reference, under_test, &controls.image) {
                Ok(mut got) => rows.append(&mut got),
                Err(why) => rows.push(Row::fail("cuda::pipeline", 0, why)),
            }
        }

        // -------------------------------------------------------------
        // the report
        // -------------------------------------------------------------
        println!("\n{:<40} {:>12}  {:>6}  first differing input / detail", "check", "inputs", "result");
        println!("{}", "-".repeat(118));
        let mut failures = 0usize;
        for row in &rows {
            if !row.passed {
                failures += 1;
            }
            println!(
                "{:<40} {:>12}  {:>6}  {}",
                row.what,
                thousands(row.inputs),
                if row.passed { "PASS" } else { "FAIL" },
                row.detail
            );
        }
        println!("{}", "-".repeat(118));

        size_table();
        closure_note(arch);

        if failures == 0 {
            println!(
                "\nPARITY: {} of {} checks bit-identical. The four shims answer\n\
                 `<cuda_fp8.h>`, `<cuda_fp4.h>`, `<cuda/cmath>` and `<cuda/pipeline>` with what\n\
                 the vendor's own headers and this device's own instructions answer -- over every\n\
                 fp8 byte pattern, every 16-bit half and bfloat pattern, a million random floats,\n\
                 eight hundred million division pairs, and a staged pipeline whose neighbour read\n\
                 the two negative controls prove would have caught a missing barrier.",
                rows.len(),
                rows.len()
            );
            0
        } else {
            println!(
                "\nPARITY FAILED on {failures} of {} checks. `first differing input` is the\n\
                 bit pattern to reproduce with; `shim` is this crate's header and `nvcc` is the\n\
                 vendor's. Do not loosen the comparison -- both paths ran the same source on the\n\
                 same device, so the only correct difference is none.",
                rows.len()
            );
            1
        }
    }

    /// The closure, end to end -- reported, deliberately not gated.
    ///
    /// The table above holds four headers to bit-parity against the vendor.
    /// This asks the other question: does the tree that includes them
    /// actually compile? It walks `csrc/src` and `csrc/vendor` off disk,
    /// hands NVRTC every file under both its path-relative and its bare name,
    /// and includes the fifteen closure roots.
    ///
    /// Off disk, not `include_str!`, for a reason: the vendored tree is six
    /// hundred files and moves under other hands. And the result is PRINTED,
    /// never counted as a FAIL, for the same reason -- it depends on
    /// `cuda_fp16.h`, `cuda_bf16.h`, `cooperative_groups.h`, `cuda/std/limits`
    /// and every vendored `.cuh`, none of which this file owns. A red row here
    /// would say "someone else's header moved", which is not what a parity
    /// probe is for. When it was last measured it compiled, with the two
    /// typedefs named below supplied by this function rather than by the
    /// fp16/bf16 shims.
    fn closure_note(arch: &str) {
        const ROOTS: [&str; 15] = [
            "attention/cascade.cuh",
            "attention/decode.cuh",
            "attention/default_decode_params.cuh",
            "attention/default_prefill_params.cuh",
            "attention/mask.cuh",
            "attention/mla.cuh",
            "attention/prefill.cuh",
            "attention/scheduler.cuh",
            "attention/state.cuh",
            "attention/variants.cuh",
            "fastdiv.cuh",
            "layout.cuh",
            "page.cuh",
            "pos_enc.cuh",
            "utils.cuh",
        ];

        println!("\nthe closure, end to end (reported, not gated):\n");

        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // In priority order, and the order is the point: this crate's own
        // shims answer first, the vendored tree second, and the AOT crate's
        // `pie_*.cuh` last -- the same three sources `src/source.rs` draws
        // `DEVICE_HEADERS` from, and the same precedence.
        let roots = [
            crate_dir.join("csrc/src"),
            crate_dir.join("csrc/vendor"),
            crate_dir.join("../kernels-cuda/csrc/src"),
        ];
        let mut owned: Vec<(String, String)> = Vec::new();
        let mut claimed = std::collections::HashSet::new();
        for root in &roots {
            if !root.is_dir() {
                println!("  {} is not there -- nothing to compile", root.display());
                return;
            }
            let mut found = Vec::new();
            gather(root, root, &mut found);
            found.sort();
            for (name, text) in found {
                if claimed.insert(name.clone()) {
                    owned.push((name, text));
                }
            }
        }

        // NVRTC matches the literal spelling, so a file reached as
        // `"cp_async.cuh"` from a sibling and as `"../cp_async.cuh"` from a
        // subdirectory is three names for one text. An include path would
        // have done this; there is no include path.
        let mut aliases: Vec<(String, String)> = Vec::new();
        for (name, text) in &owned {
            let mut spellings = vec![format!("../{name}")];
            if let Some((_, bare)) = name.rsplit_once('/') {
                spellings.push(bare.to_string());
                spellings.push(format!("../{bare}"));
            }
            for spelling in spellings {
                if claimed.insert(spelling.clone()) {
                    aliases.push((spelling, text.clone()));
                }
            }
        }
        owned.extend(aliases);

        // `Header` holds `&'static str` because the type is the design: text
        // that came out of `include_str!` and therefore out of the binary.
        // Disk text has to be leaked to satisfy it, and that friction is the
        // type doing its job -- this is the one place in the crate that reads
        // a header from a filesystem, and it has to say so out loud to compile.
        let headers: Vec<Header> = owned
            .iter()
            .map(|(n, t)| Header {
                name: Box::leak(n.clone().into_boxed_str()),
                text: Box::leak(t.clone().into_boxed_str()),
            })
            .collect();

        // `nv_half` and `nv_bfloat16` are NVIDIA typedefs that `page.cuh:232`
        // reaches for and the sibling fp16/bf16 shims do not carry yet. They
        // are declared here, in the translation unit, because the two headers
        // that should declare them are not this file's to edit.
        let source = format!(
            "#include <cuda_fp16.h>\n             #include <cuda_bf16.h>\n             typedef __half nv_half;\n             typedef __nv_bfloat16 nv_bfloat16;\n             {}\n             __global__ void closure_touch(int* o) {{ o[0] = 1; }}\n",
            ROOTS.iter().map(|r| format!("#include <flashinfer/{r}>")).collect::<Vec<_>>().join("\n")
        );

        // `-default-device` is not decoration and not this probe cheating.
        // NVRTC compiles device code only, so every unannotated function is a
        // host function and a hard error -- and FlashInfer's headers are half
        // host launchers: `cascade.cuh:573`, `decode.cuh:668`,
        // `scheduler.cuh`, all `cudaError_t` entry points a JIT caller never
        // calls. Without the flag the closure reports 200-odd of them and
        // nothing about the shims. `runtime/nvrtc.rs::options` does not pass
        // it today; that is a decision for whoever owns that file, recorded
        // here because this is where it first shows.
        match compile_with_nvrtc_opts(&source, arch, &headers, &[c"-default-device"]) {
            Ok(built) => println!(
                "  {} roots, {} carried headers, `-default-device`, no include path,\n  \
                 no toolkit -- COMPILES in {:.0} ms",
                ROOTS.len(),
                headers.len(),
                built.millis
            ),
            Err(why) => {
                if std::env::var_os("PIE_CLOSURE_LOG").is_some() {
                    println!("{why}");
                }
                let first = why
                    .lines()
                    .find(|l| l.contains("error"))
                    .unwrap_or_else(|| why.lines().next().unwrap_or(""));
                println!(
                    "  {} roots, {} carried headers -- BLOCKED, first error:\n    {}",
                    ROOTS.len(),
                    headers.len(),
                    first.trim()
                );
                println!(
                    "  Not a parity failure and not counted as one. Find the header that owes\n                       that name before reading anything into it."
                );
            }
        }
    }

    /// Every file under `root`, named relative to `base`, with `/` separators.
    fn gather(base: &std::path::Path, dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                gather(base, &path, out);
            } else if let (Ok(rel), Ok(text)) =
                (path.strip_prefix(base), std::fs::read_to_string(&path))
            {
                let name = rel.to_string_lossy().replace('\\', "/");
                if !name.contains("MODIFICATIONS") && !name.contains("LICENSE") {
                    out.push((name, text));
                }
            }
        }
    }

    /// Compile one body twice and report both, returning the loaded modules.
    fn build_pair(
        what: &str,
        nvcc: &PathBuf,
        arch: &str,
        reference_source: &str,
        under_test_source: &str,
        headers: &[Header],
        failures: &mut usize,
    ) -> Option<(Images, Images)> {
        let reference = match compile_with_nvcc(nvcc, what, reference_source, arch) {
            Ok(built) => {
                println!("  reference   nvcc   {what:<20} {:8.1} ms", built.millis);
                built
            }
            Err(why) => {
                println!("  reference   nvcc   {what:<20} REFUSED: {why}");
                *failures += 1;
                return None;
            }
        };
        let under_test = match compile_with_nvrtc(under_test_source, arch, headers) {
            Ok(built) => {
                println!("  under test  NVRTC  {what:<20} {:8.1} ms", built.millis);
                built
            }
            Err(why) => {
                println!("  under test  NVRTC  {what:<20} REFUSED:\n{why}");
                *failures += 1;
                return None;
            }
        };
        Some((Images { image: reference.image }, Images { image: under_test.image }))
    }

    /// A compiled cubin, kept until the modules that read it are built.
    struct Images {
        image: Vec<u8>,
    }

    // -----------------------------------------------------------------
    // fp8: all 256 patterns, 2^20 floats, all 65,536 sixteen-bit patterns
    // -----------------------------------------------------------------

    /// The floats every packing check runs on: every special that has ever
    /// been a converter bug, then a million from a counter-based PRNG.
    ///
    /// Specials first so that a failure names one of them rather than a
    /// random exponent -- and they are the list they are because each one is
    /// a documented edge of E4M3 or E5M2: the maxima 448 and 57344, the
    /// midpoints 464 and 61440 above which round-to-nearest reaches infinity
    /// and `satfinite` must clamp instead, the minimum normals `2^-6` and
    /// `2^-14`, the minimum subnormals `2^-9` and `2^-16`, and the halfway
    /// points between subnormals where round-half-to-even decides.
    fn float_inputs() -> Vec<f32> {
        let mut values: Vec<f32> = vec![
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffff_ffff),
            f32::from_bits(0x7fbf_ffff),
            1.0,
            -1.0,
            448.0,
            -448.0,
            448.000_03,
            456.0,
            464.0,
            464.000_03,
            480.0,
            512.0,
            -464.0,
            -480.0,
            1.0e30,
            -1.0e30,
            57344.0,
            -57344.0,
            57345.0,
            61439.0,
            61440.0,
            61441.0,
            65504.0,
            0.015_625,
            0.007_812_5,
            0.003_906_25,
            0.001_953_125,
            0.000_976_562_5,
            0.000_488_281_25,
            -0.001_953_125,
            0.002_929_687_5,
            0.000_244_140_62,
            f32::from_bits(0x3800_0000),
            f32::from_bits(0x3780_0000),
            f32::from_bits(0x0000_0001),
            f32::from_bits(0x8000_0001),
            f32::from_bits(0x007f_ffff),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        // 2^20 draws from a counter PRNG. Reproducible on purpose: a probe
        // that reported a failing input from a seed nobody can restate would
        // be reporting a rumour.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..(1usize << 20) {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let mixed = (state >> 32) as u32 ^ (state as u32).rotate_left(13);
            values.push(f32::from_bits(mixed));
        }
        if values.len() % 2 == 1 {
            values.push(0.0);
        }
        values
    }

    fn fp8_checks(reference: &Images, under_test: &Images) -> Result<Vec<Row>, String> {
        let want = Module::load(&reference.image)?;
        let got = Module::load(&under_test.image)?;
        let mut rows = Vec::new();

        // -- all 256 byte patterns, unpacked six ways
        let patterns: Vec<u8> = (0..=255u8).collect();
        let a = unpack(&want, &patterns)?;
        let b = unpack(&got, &patterns)?;
        let byte = |i: usize| format!("0x{i:02x}");
        rows.push(compare_words("__nv_cvt_fp8_to_halfraw e4m3", 2, &b.0, &a.0, byte));
        rows.push(compare_words("__nv_cvt_fp8_to_halfraw e5m2", 2, &b.1, &a.1, byte));
        rows.push(compare_words("float(__nv_fp8_e4m3)", 4, &b.2, &a.2, byte));
        rows.push(compare_words("float(__nv_fp8_e5m2)", 4, &b.3, &a.3, byte));
        rows.push(compare_words("__half(__nv_fp8_e4m3)", 2, &b.4, &a.4, byte));
        rows.push(compare_words("__half(__nv_fp8_e5m2)", 2, &b.5, &a.5, byte));
        rows.push(compare_words("__nv_bfloat16(__nv_fp8_e4m3)", 2, &b.6, &a.6, byte));
        rows.push(compare_words("__nv_bfloat16(__nv_fp8_e5m2)", 2, &b.7, &a.7, byte));

        // -- a million floats and every special, packed four ways
        let floats = float_inputs();
        let a = pack(&want, &floats)?;
        let b = pack(&got, &floats)?;
        let scalar = {
            let floats = floats.clone();
            move |i: usize| format!("0x{:08x}", floats[i].to_bits())
        };
        let paired = {
            let floats = floats.clone();
            move |i: usize| {
                format!("0x{:08x},0x{:08x}", floats[2 * i].to_bits(), floats[2 * i + 1].to_bits())
            }
        };
        rows.push(compare_words("__nv_cvt_float_to_fp8 e4m3", 1, &b.0, &a.0, scalar.clone()));
        rows.push(compare_words("__nv_cvt_float_to_fp8 e5m2", 1, &b.1, &a.1, scalar.clone()));
        rows.push(compare_words("__nv_cvt_float2_to_fp8x2 e4m3", 2, &b.2, &a.2, paired.clone()));
        rows.push(compare_words("__nv_cvt_float2_to_fp8x2 e5m2", 2, &b.3, &a.3, paired));
        rows.push(compare_words("__nv_fp8_e4m3(float)", 1, &b.4, &a.4, scalar.clone()));
        rows.push(compare_words("__nv_fp8_e5m2(float)", 1, &b.5, &a.5, scalar));

        // -- every 16-bit pattern, as a half and as a bfloat
        let bits: Vec<u16> = (0..=u16::MAX).collect();
        let a = from16(&want, &bits)?;
        let b = from16(&got, &bits)?;
        let short = |i: usize| format!("0x{i:04x}");
        rows.push(compare_words("__nv_fp8_e4m3(__half)", 1, &b.0, &a.0, short));
        rows.push(compare_words("__nv_fp8_e5m2(__half)", 1, &b.1, &a.1, short));
        rows.push(compare_words("__nv_fp8_e4m3(__nv_bfloat16)", 1, &b.2, &a.2, short));
        rows.push(compare_words("__nv_fp8_e5m2(__nv_bfloat16)", 1, &b.3, &a.3, short));

        Ok(rows)
    }

    /// Six outputs per byte pattern, from both compilers' `fp8_unpack`.
    type Unpacked = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    fn unpack(module: &Module, patterns: &[u8]) -> Result<Unpacked, String> {
        let n = patterns.len();
        let input = Device::upload(patterns)?;
        let outs: Vec<Device> = (0..8)
            .map(|i| Device::alloc(n * if (2..4).contains(&i) { 4 } else { 2 }))
            .collect::<Result<_, _>>()?;
        let mut args = vec![input.ptr];
        args.extend(outs.iter().map(|d| d.ptr));
        module.launch("fp8_unpack", 1, u32::try_from(n).unwrap(), &args, &[])?;
        let read = |i: usize, width: usize| outs[i].download(n * width);
        Ok((
            read(0, 2)?,
            read(1, 2)?,
            read(2, 4)?,
            read(3, 4)?,
            read(4, 2)?,
            read(5, 2)?,
            read(6, 2)?,
            read(7, 2)?,
        ))
    }

    /// Six outputs per float, from both compilers' `fp8_pack`.
    type Packed = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    fn pack(module: &Module, values: &[f32]) -> Result<Packed, String> {
        let n = values.len();
        let input = Device::upload(values)?;
        let scalar_e4 = Device::alloc(n)?;
        let scalar_e5 = Device::alloc(n)?;
        let pair_e4 = Device::alloc(n)?;
        let pair_e5 = Device::alloc(n)?;
        let class_e4 = Device::alloc(n)?;
        let class_e5 = Device::alloc(n)?;
        let args = [
            input.ptr,
            scalar_e4.ptr,
            scalar_e5.ptr,
            pair_e4.ptr,
            pair_e5.ptr,
            class_e4.ptr,
            class_e5.ptr,
        ];
        let count = i32::try_from(n).unwrap();
        module.launch("fp8_pack", 256, u32::try_from(n.div_ceil(256)).unwrap(), &args, &[count])?;
        Ok((
            scalar_e4.download(n)?,
            scalar_e5.download(n)?,
            pair_e4.download(n)?,
            pair_e5.download(n)?,
            class_e4.download(n)?,
            class_e5.download(n)?,
        ))
    }

    /// Four outputs per 16-bit pattern, from both compilers' `fp8_from16`.
    type From16 = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    fn from16(module: &Module, bits: &[u16]) -> Result<From16, String> {
        let n = bits.len();
        let input = Device::upload(bits)?;
        let outs: Vec<Device> = (0..4).map(|_| Device::alloc(n)).collect::<Result<_, _>>()?;
        let mut args = vec![input.ptr];
        args.extend(outs.iter().map(|d| d.ptr));
        let count = i32::try_from(n).unwrap();
        module.launch("fp8_from16", 256, u32::try_from(n.div_ceil(256)).unwrap(), &args, &[count])?;
        Ok((outs[0].download(n)?, outs[1].download(n)?, outs[2].download(n)?, outs[3].download(n)?))
    }

    // -----------------------------------------------------------------
    // fp4: the ABI, which is the whole claim
    // -----------------------------------------------------------------

    fn fp4_checks(reference: &Images, under_test: &Images) -> Result<Row, String> {
        let want = fp4_facts(&Module::load(&reference.image)?)?;
        let got = fp4_facts(&Module::load(&under_test.image)?)?;
        let names = [
            "sizeof(__nv_fp4_storage_t)",
            "sizeof(__nv_fp4x2_storage_t)",
            "sizeof(__nv_fp4x4_storage_t)",
            "sizeof(__nv_fp4_e2m1)",
            "sizeof(__nv_fp4x2_e2m1)",
            "sizeof(__nv_fp4x4_e2m1)",
            "alignof(__nv_fp4_e2m1)",
            "alignof(__nv_fp4x2_e2m1)",
            "alignof(__nv_fp4x4_e2m1)",
            "__NV_E2M1",
            "__x round trip (8+8 bit)",
            "__x round trip (16 bit)",
            "__nv_fp4_interpretation_t",
        ];
        for (i, name) in names.iter().enumerate() {
            if got[i] != want[i] {
                return Ok(Row::fail(
                    "cuda_fp4.h ABI",
                    names.len(),
                    format!("in {name} shim 0x{:x} nvcc 0x{:x}", got[i], want[i]),
                ));
            }
        }
        Ok(Row::pass("cuda_fp4.h ABI", names.len()))
    }

    fn fp4_facts(module: &Module) -> Result<Vec<u32>, String> {
        let out = Device::alloc(13 * 4)?;
        module.launch("fp4_facts", 1, 1, &[out.ptr], &[])?;
        let bytes = out.download(13 * 4)?;
        Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    // -----------------------------------------------------------------
    // cmath: a quarter of a billion divisions
    // -----------------------------------------------------------------

    /// Divisors 1..=4096, plus the ones that break a magic if anything does:
    /// `1` itself, where `floor(2^64/d)` is not a 64-bit number; the powers of
    /// two either side of a word; and `0xFFFFFFFF`, where the magic is 2 and
    /// the quotient is 0 or 1.
    fn cmath_checks(image: &[u8]) -> Result<Vec<Row>, String> {
        let module = Module::load(image)?;

        let mut dividends: Vec<u32> = vec![
            0,
            1,
            2,
            3,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0001,
            0xffff_fffe,
            0xffff_ffff,
            0xffff,
            0x1_0000,
            0x1_0001,
        ];
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..65_536 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            dividends.push((state >> 32) as u32);
        }
        let count = dividends.len();
        let input = Device::upload(&dividends)?;

        let mut rows = Vec::new();
        for (label, lo, hi) in [
            ("cuda::fast_mod_div d=1..4096", 1u32, 4096u32),
            ("cuda::fast_mod_div d=2^31..+4k", 0x8000_0000u32, 0x8000_0fffu32),
            ("cuda::fast_mod_div d=top 4k", 0xffff_f000u32, 0xffff_ffffu32),
        ] {
            let mismatches = Device::alloc(4)?;
            let first = Device::alloc(5 * 4)?;
            mismatches.zero(4)?;
            first.zero(5 * 4)?;
            let divisors = (hi - lo + 1) as usize;
            module.launch_2d(
                "divide",
                256,
                u32::try_from(divisors).unwrap(),
                &[input.ptr, mismatches.ptr, first.ptr],
                &[i32::try_from(count).unwrap()],
                &[lo, hi],
            )?;
            let bad = u32::from_le_bytes(mismatches.download(4)?.try_into().unwrap());
            let pairs = divisors * count;
            if bad == 0 {
                rows.push(Row::pass(label, pairs));
            } else {
                let f: Vec<u32> = first
                    .download(5 * 4)?
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                rows.push(Row::fail(
                    label,
                    pairs,
                    format!(
                        "in d=0x{:08x} n=0x{:08x} shim 0x{:08x} device 0x{:08x}",
                        f[1], f[2], f[3], f[4]
                    ),
                ));
            }
        }
        Ok(rows)
    }

    // -----------------------------------------------------------------
    // pipeline: the bytes, the synchronisation, and the two controls
    // -----------------------------------------------------------------

    /// Threads per block, and elements per staged tile.
    const TILE: usize = 256;
    /// Tiles staged per launch: enough that a two-stage pipeline wraps
    /// thirty-two times, so a stage that is released too early has thirty-two
    /// chances per launch to be caught.
    const TILES: usize = 64;
    /// Launches. A race that shows up one launch in a thousand is a race that
    /// ships; two thousand launches of a 64-tile pipeline is 128,000 stage
    /// transitions per kernel, which is the number that makes a PASS mean
    /// something.
    const RUNS: usize = 2_000;
    /// Launches of the cross-compiler comparison, and of each negative
    /// control. Fewer than `RUNS` because each of these runs two kernels or
    /// reads back a second buffer, and a race that survives five hundred
    /// launches of a deliberately broken kernel is not a race at all.
    const CROSS_RUNS: usize = 100;
    const CONTROL_RUNS: usize = 500;

    fn pipeline_checks(
        reference: &Images,
        under_test: &Images,
        controls: &[u8],
    ) -> Result<Vec<Row>, String> {
        let want_module = Module::load(&reference.image)?;
        let got_module = Module::load(&under_test.image)?;
        let control_module = Module::load(controls)?;

        let n = TILE * TILES;
        let source: Vec<f32> = (0..n).map(|i| i as f32).collect();
        // What a correct pipeline must produce: tile `c`, lane `t`, reads the
        // element 37 lanes along IN THAT TILE. Computed on the host, so a
        // wrong answer that both kernels agree on is still a FAIL.
        let host: Vec<f32> = (0..TILES)
            .flat_map(|c| (0..TILE).map(move |t| (c * TILE + (t + 37) % TILE) as f32))
            .collect();

        let input = Device::upload(&source)?;
        let output = Device::alloc(n * 4)?;
        let mut rows = Vec::new();

        // -- the shim, against the host reference, RUNS times
        let mut bad_runs = 0usize;
        let mut detail = String::new();
        for run in 0..RUNS {
            output.zero(n * 4)?;
            got_module.launch(
                "pipe_stage",
                u32::try_from(TILE).unwrap(),
                1,
                &[input.ptr, output.ptr],
                &[i32::try_from(TILES).unwrap()],
            )?;
            let bytes = output.download(n * 4)?;
            let got: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            if got != host {
                bad_runs += 1;
                if detail.is_empty() {
                    let at = got.iter().zip(&host).position(|(a, b)| a != b).unwrap_or(0);
                    detail = format!(
                        "run {run} tile {} lane {} shim 0x{:08x} host 0x{:08x}",
                        at / TILE,
                        at % TILE,
                        got[at].to_bits(),
                        host[at].to_bits()
                    );
                }
            }
        }
        rows.push(if bad_runs == 0 {
            Row::pass("cuda::pipeline vs host reference", RUNS * n)
        } else {
            Row::fail("cuda::pipeline vs host reference", RUNS * n, detail)
        });

        // -- the shim against the toolkit's real cuda::pipeline, same source
        //
        // Both sides re-run every launch rather than once: two staging loops
        // that agree on one launch and diverge on the ninetieth are exactly
        // what a synchronisation difference looks like.
        let want_out = Device::alloc(n * 4)?;
        let mut cross = Row::pass("cuda::pipeline vs libcu++ (nvcc)", CROSS_RUNS * n);
        for run in 0..CROSS_RUNS {
            want_out.zero(n * 4)?;
            output.zero(n * 4)?;
            for (module, out) in [(&want_module, &want_out), (&got_module, &output)] {
                module.launch(
                    "pipe_stage",
                    u32::try_from(TILE).unwrap(),
                    1,
                    &[input.ptr, out.ptr],
                    &[i32::try_from(TILES).unwrap()],
                )?;
            }
            let row = compare_words(
                "cuda::pipeline vs libcu++ (nvcc)",
                4,
                &output.download(n * 4)?,
                &want_out.download(n * 4)?,
                |i| format!("run {run} tile {} lane {}", i / TILE, i % TILE),
            );
            if !row.passed {
                cross = Row::fail("cuda::pipeline vs libcu++ (nvcc)", CROSS_RUNS * n, row.detail);
                break;
            }
        }
        rows.push(cross);

        // -- the 16-byte cp.async.cg path, through aligned_size_t<16>
        let wide = TILE * 4 * TILES;
        let wide_source: Vec<f32> = (0..wide).map(|i| i as f32).collect();
        let wide_host: Vec<f32> = (0..TILES)
            .flat_map(|c| {
                (0..TILE).flat_map(move |t| {
                    (0..4).map(move |k| (c * TILE * 4 + ((t + 37) % TILE) * 4 + k) as f32)
                })
            })
            .collect();
        let wide_in = Device::upload(&wide_source)?;
        let wide_got = Device::alloc(wide * 4)?;
        let wide_want = Device::alloc(wide * 4)?;
        for (module, out) in [(&got_module, &wide_got), (&want_module, &wide_want)] {
            out.zero(wide * 4)?;
            module.launch(
                "pipe_stage16",
                u32::try_from(TILE).unwrap(),
                1,
                &[wide_in.ptr, out.ptr],
                &[i32::try_from(TILES).unwrap()],
            )?;
        }
        let wide_got_bytes = wide_got.download(wide * 4)?;
        let wide_host_bytes: Vec<u8> =
            wide_host.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        rows.push(compare_words(
            "cuda::memcpy_async aligned_size_t<16>",
            4,
            &wide_got_bytes,
            &wide_host_bytes,
            |i| format!("element {i}"),
        ));
        rows.push(compare_words(
            "  ... same, vs libcu++ (nvcc)",
            4,
            &wide_got_bytes,
            &wide_want.download(wide * 4)?,
            |i| format!("element {i}"),
        ));

        // -- the controls, which must FAIL
        for (name, label) in [
            ("unsync", "control: no wait_group (must fail)"),
            ("nobarrier", "control: no __syncthreads (must fail)"),
        ] {
            let mut wrong = 0usize;
            for _ in 0..RUNS.min(CONTROL_RUNS) {
                output.zero(n * 4)?;
                control_module.launch(
                    name,
                    u32::try_from(TILE).unwrap(),
                    1,
                    &[input.ptr, output.ptr],
                    &[i32::try_from(TILES).unwrap()],
                )?;
                let bytes = output.download(n * 4)?;
                let got: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                if got != host {
                    wrong += 1;
                }
            }
            let runs = RUNS.min(CONTROL_RUNS);
            rows.push(if wrong > 0 {
                let mut row = Row::pass(label, runs * n);
                row.detail = format!("caught on {wrong} of {runs} launches");
                row
            } else {
                Row::fail(
                    label,
                    runs * n,
                    format!(
                        "{runs} launches with the synchronisation removed all agreed with the \
                         host -- this harness can no longer see a race, so every PASS above it \
                         is unproven"
                    ),
                )
            });
        }

        Ok(rows)
    }

    // -----------------------------------------------------------------
    // the size table
    // -----------------------------------------------------------------

    fn size_table() {
        let four = CUDA_FP8.len() + CUDA_FP4.len() + CUDA_CMATH.len() + CUDA_PIPELINE.len();
        let set = four + COOPERATIVE_GROUPS.len() + CUDA_STD_LIMITS.len();
        println!("\nwhat the four headers cost:\n");
        println!("  csrc/src/cuda_fp8.h                    {:>10} B", thousands(CUDA_FP8.len()));
        println!("  csrc/src/cuda_fp4.h                    {:>10} B", thousands(CUDA_FP4.len()));
        println!("  csrc/src/cuda/cmath                    {:>10} B", thousands(CUDA_CMATH.len()));
        println!("  csrc/src/cuda/pipeline                 {:>10} B", thousands(CUDA_PIPELINE.len()));
        println!("  {:<38} {:>10} B", "", "----------");
        println!("  these four                             {:>10} B", thousands(four));
        println!(
            "  csrc/src/cooperative_groups.h          {:>10} B",
            thousands(COOPERATIVE_GROUPS.len())
        );
        println!(
            "  csrc/src/cuda/std/limits               {:>10} B",
            thousands(CUDA_STD_LIMITS.len())
        );
        println!("  the whole shim set                     {:>10} B", thousands(set));
        println!();
        println!(
            "  CCCL, as the toolkit ships it          {:>10} B  in {} files",
            thousands(CCCL_BYTES),
            thousands(CCCL_FILES)
        );
        println!(
            "  NVIDIA cuda_fp8.h + .hpp               {:>10} B",
            thousands(NVIDIA_FP8_BYTES)
        );
        println!(
            "  NVIDIA cuda_fp4.h + .hpp               {:>10} B",
            thousands(NVIDIA_FP4_BYTES)
        );
        let vendor = CCCL_BYTES + NVIDIA_FP8_BYTES + NVIDIA_FP4_BYTES;
        println!("  {:<38} {:>10} B", "", "----------");
        println!("  what carrying them would have cost     {:>10} B", thousands(vendor));
        println!(
            "\n  {}x smaller, and none of it is NVIDIA's text -- which is the licence\n  \
             question this approach exists not to have to answer.",
            vendor / set.max(1)
        );
    }

    /// `13,691,725`, because a seven-digit number without separators is a
    /// number nobody checks.
    fn thousands(value: usize) -> String {
        let digits = value.to_string();
        let mut out = String::new();
        for (i, c) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i).is_multiple_of(3) {
                out.push(',');
            }
            out.push(c);
        }
        out
    }

    // -----------------------------------------------------------------
    // compiling, both ways
    // -----------------------------------------------------------------

    /// One compiled kernel image, and what it cost.
    struct Built {
        image: Vec<u8>,
        millis: f64,
    }

    /// Compile with the machine's `nvcc`, against its own headers.
    ///
    /// Shelling out, and reading the toolkit, is exactly what the shipped
    /// crate refuses to do -- and it is right for a probe: this path exists
    /// to produce the answer the shims are held to, so it must be the
    /// vendor's implementation and not another statement of theirs.
    ///
    /// The files land in `OUT_DIR`, which is this build's own scratch inside
    /// `target/`.
    fn compile_with_nvcc(
        nvcc: &PathBuf,
        what: &str,
        source: &str,
        arch: &str,
    ) -> Result<Built, String> {
        let scratch = PathBuf::from(env!("OUT_DIR")).join("fp8_pipeline_probe");
        std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
        let stem = what.replace(['/', '.'], "_");
        let cu = scratch.join(format!("{stem}.cu"));
        let cubin = scratch.join(format!("{stem}.cubin"));
        std::fs::write(&cu, source).map_err(|e| e.to_string())?;

        let started = Instant::now();
        let out = Command::new(nvcc)
            .arg(format!("-arch={arch}"))
            .args(["-std=c++17", "--cubin", "-o"])
            .arg(&cubin)
            .arg(&cu)
            .output()
            .map_err(|e| format!("could not run nvcc: {e}"))?;
        let millis = started.elapsed().as_secs_f64() * 1e3;
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
        }
        let image = std::fs::read(&cubin).map_err(|e| e.to_string())?;
        Ok(Built { image, millis })
    }

    /// Compile with NVRTC, against a header set built in this file.
    ///
    /// The set is the whole point: `nvrtcCreateProgram` is handed the header
    /// TEXTS and their NAMES, and NVRTC matches an `#include` against those
    /// names literally. Nothing is read from a directory, and there is no
    /// directory to read -- which is why the names here are NVIDIA's
    /// (`cuda_fp8.h`, `cuda/pipeline`) rather than anything of ours.
    ///
    /// `sm_XY` and not `compute_XY`: the reference is a cubin for this
    /// device, so the thing under test has to be one too, or the comparison
    /// would include a difference in who ran the back end.
    fn compile_with_nvrtc(source: &str, arch: &str, headers: &[Header]) -> Result<Built, String> {
        compile_with_nvrtc_opts(source, arch, headers, &[])
    }

    /// The same, plus options the parity path deliberately does not pass.
    fn compile_with_nvrtc_opts(
        source: &str,
        arch: &str,
        headers: &[Header],
        extra: &[&'static CStr],
    ) -> Result<Built, String> {
        let src = CString::new(source).map_err(|_| "a NUL in the probe source")?;
        let name = c"fp8_pipeline_probe.cu";
        let (texts, names) = as_nvrtc_arrays(headers)?;
        let text_ptrs: Vec<_> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call, and the two arrays are the
        // same length -- the whole of `nvrtcCreateProgram`'s contract. The
        // header set is an in-memory filesystem: nothing is read from disk,
        // which is the property this probe is here to keep honest.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                name.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let mut options = vec![gpu.as_ptr(), c"-std=c++17".as_ptr()];
        options.extend(extra.iter().map(|o| o.as_ptr()));

        let started = Instant::now();
        // SAFETY: the program is live and the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;
        let log = program_log(program);

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        let mut size = 0;
        // SAFETY: the program compiled, so a cubin exists; `size` is live.
        let code = unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: as above.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(format!("nvrtcGetCUBINSize: {code:?}"));
        }
        let mut image = vec![0u8; size];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked
        // for.
        let code = unsafe { nv::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) };
        // SAFETY: destroyed exactly once, after the last read out of it.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcGetCUBIN: {code:?}"));
        }
        Ok(Built { image, millis })
    }

    /// Whatever NVRTC had to say, whether or not it compiled.
    fn program_log(program: nv::nvrtcProgram) -> String {
        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        let mut log = vec![0u8; size.max(1)];
        // SAFETY: the buffer is the size NVRTC asked for.
        unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
        CStr::from_bytes_until_nul(&log)
            .map_or_else(|_| String::new(), |s| s.to_string_lossy().trim().to_string())
    }

    /// `nvcc`, wherever this machine keeps it.
    fn find_nvcc() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(path) = std::env::var("PATH") {
            candidates.extend(std::env::split_paths(&path).map(|dir| dir.join("nvcc")));
        }
        for root in ["CUDA_HOME", "CUDA_PATH"] {
            if let Ok(dir) = std::env::var(root) {
                candidates.push(PathBuf::from(dir).join("bin").join("nvcc"));
            }
        }
        candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
        candidates.into_iter().find(|c| c.is_file())
    }

    // -----------------------------------------------------------------
    // running
    // -----------------------------------------------------------------

    /// A loaded cubin, and the functions looked up out of it on demand.
    ///
    /// Its own type rather than `runtime::KernelModule`, for the reason
    /// `mma_probe`'s is: that one is keyed on rows and units these kernels do
    /// not have, being `extern "C"` on purpose so that both compilers'
    /// outputs are found by the same string. The unload in `Drop` matters
    /// here more than there -- five modules are live at once.
    struct Module {
        module: dr::CUmodule,
    }

    impl Module {
        fn load(image: &[u8]) -> Result<Self, String> {
            ensure_context()?;
            let mut module: dr::CUmodule = std::ptr::null_mut();
            // SAFETY: the image is a cubin this process just produced and
            // outlives the call; `module` is a live out-parameter.
            let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData: {code:?}"));
            }
            Ok(Self { module })
        }

        fn function(&self, name: &str) -> Result<dr::CUfunction, String> {
            let symbol = CString::new(name).map_err(|_| "a NUL in a kernel name")?;
            let mut function: dr::CUfunction = std::ptr::null_mut();
            // SAFETY: the module came from a successful load and the name is
            // NUL-terminated.
            let code =
                unsafe { dr::cuModuleGetFunction(&raw mut function, self.module, symbol.as_ptr()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleGetFunction({name}): {code:?}"));
            }
            Ok(function)
        }

        /// Launch with pointer arguments first, then `i32` scalars -- the
        /// order every kernel in this file declares.
        fn launch(
            &self,
            name: &str,
            threads: u32,
            blocks: u32,
            pointers: &[*mut c_void],
            scalars: &[i32],
        ) -> Result<(), String> {
            self.launch_2d(name, threads, blocks, pointers, scalars, &[])
        }

        /// As `launch`, with an optional second grid dimension and a pair of
        /// trailing `u32` scalars -- which is the `divide` kernel's shape:
        /// one block per divisor, and the divisor range passed in.
        fn launch_2d(
            &self,
            name: &str,
            threads: u32,
            blocks: u32,
            pointers: &[*mut c_void],
            scalars: &[i32],
            range: &[u32],
        ) -> Result<(), String> {
            let function = self.function(name)?;
            let mut owned_pointers: Vec<*mut c_void> = pointers.to_vec();
            let mut owned_scalars: Vec<i32> = scalars.to_vec();
            let mut owned_range: Vec<u32> = range.to_vec();

            // The `divide` kernel's signature is (ptr, i32, u32, u32, ptr,
            // ptr): the range sits between the count and the outputs, so the
            // parameter list is assembled rather than concatenated.
            let mut params: Vec<*mut c_void> = Vec::new();
            if owned_range.is_empty() {
                for p in &mut owned_pointers {
                    params.push((&raw mut *p).cast());
                }
                for s in &mut owned_scalars {
                    params.push((&raw mut *s).cast());
                }
            } else {
                params.push((&raw mut owned_pointers[0]).cast());
                params.push((&raw mut owned_scalars[0]).cast());
                params.push((&raw mut owned_range[0]).cast());
                params.push((&raw mut owned_range[1]).cast());
                for p in &mut owned_pointers[1..] {
                    params.push((&raw mut *p).cast());
                }
            }

            // A grid wider than 65,535 in y is illegal, and the divisor
            // sweeps are 4,096 blocks, so x carries the count and y carries
            // the overflow.
            let (grid_x, grid_y) =
                if blocks > 65_535 { (65_535, blocks.div_ceil(65_535)) } else { (blocks, 1) };

            // SAFETY: the function came from a live module; every allocation
            // behind `pointers` outlives the synchronise below; `params`
            // borrows locals that outlive the call.
            let code = unsafe {
                dr::cuLaunchKernel(
                    function,
                    grid_x,
                    grid_y,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuLaunchKernel({name}): {code:?}"));
            }
            // SAFETY: no arguments, and a fault inside the kernel surfaces
            // here rather than at the copy that follows.
            let code = unsafe { rt::cudaDeviceSynchronize() };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaDeviceSynchronize after {name}: {code:?}"));
            }
            Ok(())
        }
    }

    impl Drop for Module {
        fn drop(&mut self) {
            // SAFETY: the handle came from `cuModuleLoadData`, every launch
            // that named it has been synchronised, and nothing else holds it.
            unsafe { dr::cuModuleUnload(self.module) };
        }
    }

    /// One device allocation, freed when it goes out of scope.
    struct Device {
        ptr: *mut c_void,
    }

    impl Device {
        fn alloc(bytes: usize) -> Result<Self, String> {
            ensure_context()?;
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { rt::cudaMalloc(&raw mut ptr, bytes) };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMalloc({bytes}): {code:?}"));
            }
            Ok(Self { ptr })
        }

        fn upload<T>(values: &[T]) -> Result<Self, String> {
            let bytes = std::mem::size_of_val(values);
            let owned = Self::alloc(bytes)?;
            // SAFETY: the destination is `bytes` long by construction and the
            // source is the slice's own storage.
            let code = unsafe {
                rt::cudaMemcpy(
                    owned.ptr,
                    values.as_ptr().cast(),
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMemcpy H2D: {code:?}"));
            }
            Ok(owned)
        }

        fn download(&self, bytes: usize) -> Result<Vec<u8>, String> {
            let mut out = vec![0u8; bytes];
            // SAFETY: both sides are `bytes` long -- the device side by the
            // allocation that produced this handle.
            let code = unsafe {
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    self.ptr,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMemcpy D2H: {code:?}"));
            }
            Ok(out)
        }

        /// Zero the buffer between runs.
        ///
        /// Not hygiene: a staging kernel that failed to write a lane would
        /// otherwise be checked against whatever the PREVIOUS launch left
        /// there, which for a repeated identical launch is the right answer.
        /// That is the one way this probe could pass a pipeline that never
        /// ran.
        fn zero(&self, bytes: usize) -> Result<(), String> {
            // SAFETY: the allocation is at least `bytes` long.
            let code = unsafe { rt::cudaMemset(self.ptr, 0, bytes) };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMemset: {code:?}"));
            }
            Ok(())
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            // SAFETY: the pointer came from `cudaMalloc` and nothing else
            // holds it; every launch that read it has been synchronised.
            unsafe { rt::cudaFree(self.ptr) };
        }
    }

    /// A context the driver API can load a module into.
    ///
    /// The runtime API creates the primary context lazily and pushes it onto
    /// the calling thread, which is why a `cudaFree(null)` is enough. The
    /// explicit retain is the fallback for the case where it is not -- and it
    /// is a real case rather than defensiveness: `cuModuleLoadData` with no
    /// current context fails with `CUDA_ERROR_INVALID_CONTEXT`, which reads
    /// like a broken cubin.
    fn ensure_context() -> Result<(), String> {
        // SAFETY: a null pointer is the documented no-op that forces runtime
        // initialisation.
        unsafe { rt::cudaFree(std::ptr::null_mut()) };
        let mut current: dr::CUcontext = std::ptr::null_mut();
        // SAFETY: `current` is a live out-parameter.
        unsafe { dr::cuCtxGetCurrent(&raw mut current) };
        if !current.is_null() {
            return Ok(());
        }
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a live out-parameter; the driver is initialised
        // by the runtime call above.
        unsafe { dr::cuDeviceGet(&raw mut device, 0) };
        let mut context: dr::CUcontext = std::ptr::null_mut();
        // SAFETY: `context` is live and `device` came from `cuDeviceGet`.
        let code = unsafe { dr::cuDevicePrimaryCtxRetain(&raw mut context, device) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDevicePrimaryCtxRetain: {code:?}"));
        }
        // SAFETY: `context` came from a successful retain.
        let code = unsafe { dr::cuCtxSetCurrent(context) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(format!("cuCtxSetCurrent: {code:?}"));
        }
        Ok(())
    }

    /// What the driver calls this GPU, so the report names the machine it was
    /// measured on.
    fn device_name() -> String {
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a live out-parameter; `arch()` has already
        // initialised the driver by the time this is called.
        if unsafe { dr::cuDeviceGet(&raw mut device, 0) } != dr::CUresult::CUDA_SUCCESS {
            return "unknown".to_string();
        }
        let mut name = [0u8; 128];
        // SAFETY: the buffer is 128 bytes and that is what is claimed.
        let code = unsafe {
            dr::cuDeviceGetName(name.as_mut_ptr().cast(), i32::try_from(name.len()).unwrap(), device)
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return "unknown".to_string();
        }
        CStr::from_bytes_until_nul(&name)
            .map_or_else(|_| "unknown".to_string(), |s| s.to_string_lossy().into_owned())
    }

    /// `libnvrtc`'s own version, because whether `<cuda_fp8.h>` resolves is a
    /// property of it and not of the toolkit beside it.
    fn nvrtc_version() -> String {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both are live out-parameters for the call's duration.
        let code = unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
        if code == nv::nvrtcResult::NVRTC_SUCCESS {
            format!("{major}.{minor}")
        } else {
            format!("unavailable ({code:?})")
        }
    }
}
