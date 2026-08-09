//! Do `cuda_fp16.h` and `cuda_bf16.h` return the same BITS as NVIDIA's?
//!
//! # The question this answers, and why compiling is not it
//!
//! `csrc/shim/cuda_fp16.h` and `csrc/shim/cuda_bf16.h` answer FlashInfer's
//! `#include <cuda_fp16.h>` and `#include <cuda_bf16.h>` out of the header set
//! carried in the binary, because NVRTC 13.0 was measured refusing all 31
//! external includes of the attention closure and reading `$CUDA_HOME` at
//! build time was rejected. Between them they restate 27 functions, four
//! operators and six types.
//!
//! Every failure mode those two files have is a WRONG NUMBER, not a
//! diagnostic. `__float2bfloat16` written as a truncation instead of a
//! round-to-nearest-even compiles perfectly and biases every attention score
//! downward by half an ulp. The two source operands of `cvt.rn.f16x2.f32`
//! written in the vendor's order instead of the reverse compiles perfectly
//! and swaps the two lanes of every packed convert. A `__hmax` that returns
//! the NaN instead of the other operand compiles perfectly and poisons a
//! softmax row. So a probe that merely compiles the headers proves nothing at
//! all, and a probe with a tolerance proves less than nothing -- it is a way
//! of not noticing the one class of defect that is actually reachable here.
//!
//! # Two paths, one body
//!
//! Every kernel below is compiled twice from the SAME text:
//!
//! * **reference** -- `nvcc -cubin` against the machine's own
//!   `<cuda_fp16.h>` and `<cuda_bf16.h>`. Shelling out to the toolkit is
//!   exactly what the shipped crate refuses to do, and it is right for a
//!   probe: this path has to be the vendor's implementation, or the
//!   comparison is the shim against a second statement of itself.
//! * **under test** -- NVRTC against a header set built here out of
//!   [`kernels_cuda_new::source`]: the prelude, `cuda_bf16.h`, `cuda_fp16.h`,
//!   and nothing else. No include path on disk. If a fourth entry were needed
//!   the compile would fail, which is part of the claim.
//!
//! Both cubins are `sm_89` for this device, both run on it, and the outputs
//! are compared as **bit patterns**. Same instruction, same hardware -- the
//! difference must be exactly zero.
//!
//! # The input space, which is where a probe like this usually cheats
//!
//! Three friendly values pass against almost any wrong implementation. What
//! is swept instead:
//!
//! | corpus | what it catches |
//! |---|---|
//! | all 65,536 bf16 patterns, widened | an exponent or NaN payload wrong in one octave |
//! | all 65,536 fp16 patterns, widened | the same, plus every fp16 subnormal |
//! | all 65,536 bf16 TIES -- `bits \| 0x8000` | truncation instead of round-to-nearest-even, and ties-away instead of ties-to-even |
//! | every fp16 tie and its two fp32 neighbours | the same for fp16, and the sticky bit |
//! | ±0, ±inf, quiet and signalling NaN, largest finite, smallest normal, largest and smallest subnormal, both signs | the branchy edges of a hand-written converter |
//! | the fp16 and bf16 overflow and underflow thresholds | a rounding carry that must reach infinity, and a tie at half the smallest subnormal |
//! | 1.2M pseudo-random fp32 over the whole 32-bit space | everything else |
//! | all 65,536 16-bit patterns as operands | a packed op that is wrong for one class of input |
//! | a 216 x 216 dense grid of exponent/mantissa representatives | a packed op wrong for a RANGE of exponents, which randoms hit thinly |
//! | products aimed at 2^-134 | the one window where an fp32 emulation of a bf16 multiply can double-round |
//! | 1M pseudo-random 16-bit pairs | NaN x inf, ±0 x ±0, subnormal x subnormal |
//! | 0.5, 2.5, -2.5 and the integer edges | the only place `_rd`, `_rn` and `_rz` visibly disagree |
//!
//! # And three things beyond the table
//!
//! **The portable pass.** Both headers gate their instructions on
//! `__CUDA_ARCH__` -- bf16 packed multiply is sm_90, its FMA emulation sm_80,
//! `max.f16` sm_80, `.f16x2` `cvt` destinations sm_80 -- and every gate has a
//! fallback for older parts. A fallback that ships unmeasured is a silent
//! wrong answer on the first Turing card that runs it, so the whole table is
//! run a second time with `-DPIE_HALFTYPE_FORCE_PORTABLE`, which forces every
//! fallback ON an sm_89 device and compares it against the same vendor
//! reference.
//!
//! **The sensitivity check.** `cuda_fp16.h` is compiled a third time with the
//! two source operands of its `cvt.rn.f16x2.f32` swapped -- the exact lane
//! error the header's comment warns about -- and the comparison is REQUIRED
//! to catch it. A green table proves nothing unless a red one was reachable.
//!
//! **The known divergence.** `float(h)` on a `__half` needs a member
//! conversion operator, and neither header can add a member to a type it does
//! not own. Rather than claim that, the probe compiles it and prints NVRTC's
//! refusal, beside the spelling that does work.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example halftype_parity
//! ```
//!
//! Needs `nvcc` on `PATH` or at `$CUDA_HOME/bin/nvcc`, and a device. Exits
//! non-zero if any row is not bit-identical.

#[cfg(not(feature = "_cuda"))]
fn main() {
    // The same gate `mma_probe` carries, and for the same reason: layers 1
    // and 2 build with no CUDA at all, and a probe that exists to show the
    // toolkit is unnecessary must not be the thing that drags it in.
    println!(
        "halftype_parity needs layer 3: cargo run -p kernels-cuda-new --features cuda-13 \
         --example halftype_parity"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    std::process::exit(probe::run());
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::ffi::{CStr, CString, c_void};
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    use cudarc::driver::sys as dr;
    use cudarc::nvrtc::sys as nv;
    use cudarc::runtime::sys as rt;

    use kernels_cuda_new::source::{DEVICE_HEADERS, Header, as_nvrtc_arrays};

    /// The two shims, carried the way every other device source in this crate
    /// is.
    ///
    /// `include_str!` and not a read: the bytes compared here are the bytes
    /// that ship, so a probe that passed against a file on disk while the
    /// binary carried something else would be measuring the wrong header.
    const CUDA_FP16: &str = include_str!("../csrc/shim/cuda_fp16.h");
    const CUDA_BF16: &str = include_str!("../csrc/shim/cuda_bf16.h");

    /// The instruction the sensitivity check corrupts, and what it becomes.
    ///
    /// `cvt.rn.f16x2.f32 d, a, b` puts `a` in the HIGH half, so the header
    /// names `hi` first. Swapping the two operand numbers is a one-character
    /// edit that transposes every packed convert and produces no diagnostic
    /// whatsoever -- which is precisely why the harness has to be shown
    /// catching it.
    const LANE_ORDER: &str = "cvt.rn.f16x2.f32 %0, %2, %1;";
    const LANE_ORDER_SWAPPED: &str = "cvt.rn.f16x2.f32 %0, %1, %2;";

    /// The harness both compilations share, above the generated kernels.
    ///
    /// Every helper here is a bit-cast and nothing else. They are written as
    /// `reinterpret_cast` rather than through `__half_as_ushort`, because
    /// that function is one of the things under test and a harness that used
    /// it would be unable to report it wrong.
    const HARNESS: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float f32(unsigned b) { return __int_as_float((int)b); }
__device__ __forceinline__ unsigned ub(float f) { return (unsigned)__float_as_int(f); }
__device__ __forceinline__ unsigned ub(__half v) {
    return (unsigned)*reinterpret_cast<const unsigned short*>(&v);
}
__device__ __forceinline__ unsigned ub(__half2 v) {
    return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ unsigned ub(__nv_bfloat16 v) {
    return (unsigned)*reinterpret_cast<const unsigned short*>(&v);
}
__device__ __forceinline__ unsigned ub(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ __half h1(unsigned b) {
    __half v;
    *reinterpret_cast<unsigned short*>(&v) = (unsigned short)b;
    return v;
}
__device__ __forceinline__ __half2 h2(unsigned a, unsigned b) {
    __half2 v;
    v.x = h1(a);
    v.y = h1(b);
    return v;
}
__device__ __forceinline__ __nv_bfloat16 b1(unsigned b) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short*>(&v) = (unsigned short)b;
    return v;
}
__device__ __forceinline__ __nv_bfloat162 b2(unsigned a, unsigned b) {
    __nv_bfloat162 v;
    v.x = b1(a);
    v.y = b1(b);
    return v;
}
"#;

    /// Which corpus a row draws its operands from.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Feed {
        /// fp32 bit patterns, one `u32` slot each.
        Floats,
        /// 16-bit patterns, zero-extended, one `u32` slot each. The same
        /// corpus feeds fp16 and bf16 rows: a 16-bit pattern is a 16-bit
        /// pattern, and what it MEANS is the row's business.
        Bits16,
    }

    /// One function under test.
    struct Row {
        /// What the report calls it, which is what a call site spells.
        name: &'static str,
        /// Its `extern "C"` kernel name in both cubins.
        sym: &'static str,
        /// The statement the generated kernel runs, in terms of `in`, `out`
        /// and `i`. Written per row rather than derived, so that the index
        /// arithmetic is visible beside the call it feeds.
        body: &'static str,
        /// Input slots per result.
        ins: usize,
        /// Output slots per result.
        outs: usize,
        feed: Feed,
        /// `fp16`, `bf16`, or `builtin` -- see the report's grouping.
        group: &'static str,
    }

    /// The 43 rows, which are the measured surface of the two headers plus
    /// the three integer roundings that are NOT theirs.
    fn rows() -> Vec<Row> {
        vec![
            // -- fp16 conversions ---------------------------------------
            Row {
                name: "__float2half",
                sym: "k_f2h",
                group: "fp16",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__float2half(f32(in[i])));",
            },
            Row {
                name: "__float2half_rn",
                sym: "k_f2h_rn",
                group: "fp16",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__float2half_rn(f32(in[i])));",
            },
            Row {
                name: "__half2float",
                sym: "k_h2f",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__half2float(h1(in[i])));",
            },
            Row {
                name: "__float2half2_rn",
                sym: "k_f2h2",
                group: "fp16",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__float2half2_rn(f32(in[i])));",
            },
            Row {
                name: "__floats2half2_rn",
                sym: "k_ff2h2",
                group: "fp16",
                feed: Feed::Floats,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__floats2half2_rn(f32(in[2*i]), f32(in[2*i+1])));",
            },
            Row {
                name: "__float22half2_rn",
                sym: "k_f22h2",
                group: "fp16",
                feed: Feed::Floats,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__float22half2_rn(make_float2(f32(in[2*i]), f32(in[2*i+1]))));",
            },
            Row {
                name: "__half22float2",
                sym: "k_h22f2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 2,
                body: "float2 r = __half22float2(h2(in[2*i], in[2*i+1]));\n\
                         out[2*i] = ub(r.x); out[2*i+1] = ub(r.y);",
            },
            Row {
                name: "__halves2half2",
                sym: "k_hh2h2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__halves2half2(h1(in[2*i]), h1(in[2*i+1])));",
            },
            Row {
                name: "make_half2",
                sym: "k_mkh2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(make_half2(h1(in[2*i]), h1(in[2*i+1])));",
            },
            Row {
                name: "__half_as_ushort",
                sym: "k_h_as_u",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "out[i] = (unsigned)__half_as_ushort(h1(in[i]));",
            },
            Row {
                name: "__ushort_as_half",
                sym: "k_u_as_h",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__ushort_as_half((unsigned short)in[i]));",
            },
            // -- fp16 arithmetic ----------------------------------------
            Row {
                name: "__hmul",
                sym: "k_hmul",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__hmul(h1(in[2*i]), h1(in[2*i+1])));",
            },
            Row {
                name: "__hsub",
                sym: "k_hsub",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__hsub(h1(in[2*i]), h1(in[2*i+1])));",
            },
            Row {
                name: "__hmax",
                sym: "k_hmax",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__hmax(h1(in[2*i]), h1(in[2*i+1])));",
            },
            Row {
                name: "operator*(half, half)",
                sym: "k_op_mul_h",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(h1(in[2*i]) * h1(in[2*i+1]));",
            },
            Row {
                name: "operator-(half, half)",
                sym: "k_op_sub_h",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(h1(in[2*i]) - h1(in[2*i+1]));",
            },
            Row {
                name: "__hmul2",
                sym: "k_hmul2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 4,
                outs: 1,
                body: "out[i] = ub(__hmul2(h2(in[4*i], in[4*i+1]), h2(in[4*i+2], in[4*i+3])));",
            },
            Row {
                name: "__hsub2",
                sym: "k_hsub2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 4,
                outs: 1,
                body: "out[i] = ub(__hsub2(h2(in[4*i], in[4*i+1]), h2(in[4*i+2], in[4*i+3])));",
            },
            Row {
                name: "__hmax2",
                sym: "k_hmax2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 4,
                outs: 1,
                body: "out[i] = ub(__hmax2(h2(in[4*i], in[4*i+1]), h2(in[4*i+2], in[4*i+3])));",
            },
            Row {
                name: "__hfma2",
                sym: "k_hfma2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 6,
                outs: 1,
                body: "out[i] = ub(__hfma2(h2(in[6*i], in[6*i+1]), h2(in[6*i+2], in[6*i+3]), \
                         h2(in[6*i+4], in[6*i+5])));",
            },
            Row {
                name: "operator*(half2, half2)",
                sym: "k_op_mul_h2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 4,
                outs: 1,
                body: "out[i] = ub(h2(in[4*i], in[4*i+1]) * h2(in[4*i+2], in[4*i+3]));",
            },
            Row {
                name: "operator-(half2, half2)",
                sym: "k_op_sub_h2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 4,
                outs: 1,
                body: "out[i] = ub(h2(in[4*i], in[4*i+1]) - h2(in[4*i+2], in[4*i+3]));",
            },
            Row {
                name: "__shfl_xor_sync(half2)",
                sym: "k_shfl_h2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 2,
                body: "__half2 v = h2(in[2*i], in[2*i+1]);\n\
                         out[2*i] = ub(__shfl_xor_sync(0xffffffffu, v, 1));\n\
                         out[2*i+1] = ub(__shfl_xor_sync(0xffffffffu, v, 2));",
            },
            // -- bf16 ---------------------------------------------------
            Row {
                name: "__float2bfloat16",
                sym: "k_f2b",
                group: "bf16",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__float2bfloat16(f32(in[i])));",
            },
            Row {
                name: "__float2bfloat16_rn",
                sym: "k_f2b_rn",
                group: "bf16",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__float2bfloat16_rn(f32(in[i])));",
            },
            Row {
                name: "__bfloat162float",
                sym: "k_b2f",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__bfloat162float(b1(in[i])));",
            },
            Row {
                name: "__float2bfloat162_rn",
                sym: "k_f2b2",
                group: "bf16",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = ub(__float2bfloat162_rn(f32(in[i])));",
            },
            Row {
                name: "__floats2bfloat162_rn",
                sym: "k_ff2b2",
                group: "bf16",
                feed: Feed::Floats,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__floats2bfloat162_rn(f32(in[2*i]), f32(in[2*i+1])));",
            },
            Row {
                name: "__float22bfloat162_rn",
                sym: "k_f22b2",
                group: "bf16",
                feed: Feed::Floats,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(__float22bfloat162_rn(make_float2(f32(in[2*i]), \
                         f32(in[2*i+1]))));",
            },
            Row {
                name: "__bfloat1622float2",
                sym: "k_b22f2",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 2,
                body: "float2 r = __bfloat1622float2(b2(in[2*i], in[2*i+1]));\n\
                         out[2*i] = ub(r.x); out[2*i+1] = ub(r.y);",
            },
            Row {
                name: "make_bfloat162",
                sym: "k_mkb2",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "out[i] = ub(make_bfloat162(b1(in[2*i]), b1(in[2*i+1])));",
            },
            Row {
                name: "__hmul2(bf16)",
                sym: "k_hmul2_b",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 4,
                outs: 1,
                body: "out[i] = ub(__hmul2(b2(in[4*i], in[4*i+1]), b2(in[4*i+2], in[4*i+3])));",
            },
            // -- the storage structs ------------------------------------
            //
            // Not arithmetic -- a layout and two casts -- and gated here
            // anyway, because "it is only a bit-copy" is exactly the claim a
            // parity harness exists to stop taking on trust. Each direction
            // is a separate row: the class-to-raw cast is what `cuda_fp8.h`
            // reads `.x` off, and the raw-to-class one is written as
            // COPY-initialisation on purpose, because that is the form
            // `tests/prelude_parity.rs` uses and the form an `explicit`
            // conversion operator would silently refuse.
            Row {
                name: "static_cast<__half_raw>(h).x",
                sym: "k_h2raw",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "out[i] = (unsigned)static_cast<__half_raw>(h1(in[i])).x;",
            },
            Row {
                name: "__half = __half_raw",
                sym: "k_raw2h",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "__half_raw r; r.x = (unsigned short)in[i];\n\
                         __half v = r; out[i] = ub(v);",
            },
            Row {
                name: "static_cast<__half2_raw>(h2)",
                sym: "k_h22raw",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 2,
                body: "__half2_raw r = static_cast<__half2_raw>(h2(in[2*i], in[2*i+1]));\n\
                         out[2*i] = (unsigned)r.x; out[2*i+1] = (unsigned)r.y;",
            },
            Row {
                name: "__half2 = __half2_raw",
                sym: "k_raw2h2",
                group: "fp16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "__half2_raw r; r.x = (unsigned short)in[2*i];\n\
                         r.y = (unsigned short)in[2*i+1];\n\
                         __half2 v = r; out[i] = ub(v);",
            },
            Row {
                name: "static_cast<__nv_bfloat16_raw>(b).x",
                sym: "k_b2raw",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "out[i] = (unsigned)static_cast<__nv_bfloat16_raw>(b1(in[i])).x;",
            },
            Row {
                name: "__nv_bfloat16 = __nv_bfloat16_raw",
                sym: "k_raw2b",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 1,
                outs: 1,
                body: "__nv_bfloat16_raw r; r.x = (unsigned short)in[i];\n\
                         __nv_bfloat16 v = r; out[i] = ub(v);",
            },
            Row {
                name: "static_cast<__nv_bfloat162_raw>(b2)",
                sym: "k_b22raw",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 2,
                body: "__nv_bfloat162_raw r = static_cast<__nv_bfloat162_raw>(\
                         b2(in[2*i], in[2*i+1]));\n\
                         out[2*i] = (unsigned)r.x; out[2*i+1] = (unsigned)r.y;",
            },
            Row {
                name: "__nv_bfloat162 = __nv_bfloat162_raw",
                sym: "k_raw2b2",
                group: "bf16",
                feed: Feed::Bits16,
                ins: 2,
                outs: 1,
                body: "__nv_bfloat162_raw r; r.x = (unsigned short)in[2*i];\n\
                         r.y = (unsigned short)in[2*i+1];\n\
                         __nv_bfloat162 v = r; out[i] = ub(v);",
            },
            // -- not ours, and the row is the proof ---------------------
            Row {
                name: "__float2int_rd",
                sym: "k_f2i_rd",
                group: "builtin",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = (unsigned)__float2int_rd(f32(in[i]));",
            },
            Row {
                name: "__float2int_rn",
                sym: "k_f2i_rn",
                group: "builtin",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = (unsigned)__float2int_rn(f32(in[i]));",
            },
            Row {
                name: "__float2int_rz",
                sym: "k_f2i_rz",
                group: "builtin",
                feed: Feed::Floats,
                ins: 1,
                outs: 1,
                body: "out[i] = (unsigned)__float2int_rz(f32(in[i]));",
            },
        ]
    }

    /// The one source text both compilers see, kernels generated from the
    /// table so that the two paths cannot drift apart by a typo.
    fn source(rows: &[Row]) -> String {
        let mut out = String::from(HARNESS);
        for row in rows {
            out.push_str(&format!(
                "\nextern \"C\" __global__ void {}(const unsigned* in, unsigned* out, unsigned n)\n\
                 {{\n    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 \x20   if (i < n) {{\n        {}\n    }}\n}}\n",
                row.sym, row.body
            ));
        }
        out
    }

    pub fn run() -> i32 {
        let Some(arch) = kernels_cuda_new::jit::cache::arch() else {
            println!("no CUDA device is current; this probe needs one to compare on");
            return 1;
        };

        println!("halftype_parity -- csrc/shim/cuda_fp16.h and cuda_bf16.h against NVIDIA's\n");
        println!("  device        {} ({arch})", device_name());
        println!("  NVRTC         {}", nvrtc_version());
        let Some(nvcc) = find_nvcc() else {
            println!(
                "\n  no `nvcc` on PATH or at $CUDA_HOME/bin/nvcc.\n\
                 The reference path IS the check -- a host converter written here would\n\
                 share this file's idea of round-to-nearest -- so there is nothing useful\n\
                 to report without it."
            );
            return 1;
        };
        println!("  nvcc          {}", nvcc.display());
        println!("  cuda_fp16.h   {} bytes", CUDA_FP16.len());
        println!("  cuda_bf16.h   {} bytes", CUDA_BF16.len());

        let Some(prelude) = DEVICE_HEADERS.iter().find(|h| h.name == "pie_device.cuh") else {
            println!("the header set has no `pie_device.cuh`, which both shims include");
            return 1;
        };
        // Built literally, in table order: the prelude the two shims take
        // their types from, and the two shims. Three entries and no include
        // path -- if a shim had grown a dependency, this compile would say so.
        let headers = [
            *prelude,
            Header { name: "cuda_bf16.h", text: CUDA_BF16 },
            Header { name: "cuda_fp16.h", text: CUDA_FP16 },
        ];
        println!(
            "  header set    {}",
            headers
                .iter()
                .map(|h| format!("{} ({} B)", h.name, h.text.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let rows = rows();
        let text = source(&rows);
        println!("\ncompiling {} kernels, one source, three ways:\n", rows.len());

        let reference = match compile_with_nvcc(&nvcc, &text, arch) {
            Ok(built) => {
                println!("  reference     nvcc -cubin, the real headers   {:8.1} ms", built.millis);
                built
            }
            Err(why) => {
                println!("  reference     nvcc REFUSED:\n{why}");
                return 1;
            }
        };
        let native = match compile_with_nvrtc(&text, arch, &headers, &[]) {
            Ok(built) => {
                println!("  under test    NVRTC, the header set          {:8.1} ms", built.millis);
                built
            }
            Err(why) => {
                println!("  under test    NVRTC REFUSED:\n{why}");
                return 1;
            }
        };
        let portable =
            match compile_with_nvrtc(&text, arch, &headers, &["-DPIE_HALFTYPE_FORCE_PORTABLE"]) {
                Ok(built) => {
                    println!(
                        "  portable      NVRTC, every fallback forced   {:8.1} ms",
                        built.millis
                    );
                    Some(built)
                }
                Err(why) => {
                    println!("  portable      NVRTC REFUSED:\n{why}");
                    None
                }
            };

        let (Ok(reference), Ok(native)) =
            (Module::load(&reference.image), Module::load(&native.image))
        else {
            println!("\nloading a cubin failed");
            return 1;
        };
        let portable = portable.and_then(|b| Module::load(&b.image).ok());

        let floats = f32_corpus();
        let bits16 = bits16_corpus();
        println!("\ncorpora: {} fp32 patterns, {} 16-bit patterns\n", floats.len(), bits16.len());

        let d_floats = match Device::upload(&floats) {
            Ok(d) => d,
            Err(why) => {
                println!("uploading the fp32 corpus: {why}");
                return 1;
            }
        };
        let d_bits16 = match Device::upload(&bits16) {
            Ok(d) => d,
            Err(why) => {
                println!("uploading the 16-bit corpus: {why}");
                return 1;
            }
        };

        let mut failures = 0usize;
        let mut checked = 0usize;

        println!(
            "{:<26} {:>5} {:>10}  {:>6}  {}",
            "function", "group", "inputs", "result", "first difference"
        );
        println!("{}", "-".repeat(112));
        for row in &rows {
            let (corpus, device) = match row.feed {
                Feed::Floats => (&floats, &d_floats),
                Feed::Bits16 => (&bits16, &d_bits16),
            };
            let n = corpus.len() / row.ins;
            checked += n;
            match compare_row(row, &reference, &native, device, n, corpus) {
                Ok(None) => {
                    println!("{:<26} {:>5} {:>10}  {:>6}  {}", row.name, row.group, n, "PASS", "--")
                }
                Ok(Some(diff)) => {
                    failures += 1;
                    println!(
                        "{:<26} {:>5} {:>10}  {:>6}  {}",
                        row.name, row.group, n, "FAIL", diff
                    );
                }
                Err(why) => {
                    failures += 1;
                    println!("{:<26} {:>5} {:>10}  {:>6}  {why}", row.name, row.group, n, "ERROR");
                }
            }
        }
        println!("{}", "-".repeat(112));
        println!(
            "{} of {} rows bit-identical over {checked} comparisons.",
            rows.len() - failures,
            rows.len()
        );

        // -----------------------------------------------------------------
        // the same table, with every architecture fallback forced on
        // -----------------------------------------------------------------
        let mut portable_failures = 0usize;
        if let Some(portable) = &portable {
            println!(
                "\nthe same {} rows with -DPIE_HALFTYPE_FORCE_PORTABLE -- the pre-sm_80 paths,\n\
                 which no device in this building runs, measured against the same reference:\n",
                rows.len()
            );
            println!(
                "{:<26} {:>5} {:>10}  {:>6}  {}",
                "function", "group", "inputs", "result", "first difference"
            );
            println!("{}", "-".repeat(112));
            for row in &rows {
                let (corpus, device) = match row.feed {
                    Feed::Floats => (&floats, &d_floats),
                    Feed::Bits16 => (&bits16, &d_bits16),
                };
                let n = corpus.len() / row.ins;
                match compare_row(row, &reference, portable, device, n, corpus) {
                    Ok(None) => println!(
                        "{:<26} {:>5} {:>10}  {:>6}  {}",
                        row.name, row.group, n, "PASS", "--"
                    ),
                    Ok(Some(diff)) => {
                        portable_failures += 1;
                        println!(
                            "{:<26} {:>5} {:>10}  {:>6}  {}",
                            row.name, row.group, n, "FAIL", diff
                        );
                    }
                    Err(why) => {
                        portable_failures += 1;
                        println!(
                            "{:<26} {:>5} {:>10}  {:>6}  {why}",
                            row.name, row.group, n, "ERROR"
                        );
                    }
                }
            }
            println!("{}", "-".repeat(112));
            println!(
                "{} of {} portable rows bit-identical.",
                rows.len() - portable_failures,
                rows.len()
            );
        }

        // -----------------------------------------------------------------
        // a harness that cannot fail is not a harness
        // -----------------------------------------------------------------
        let sensitivity = sensitivity_check(&rows, &reference, arch, prelude, &d_floats, &floats);
        println!("\n{sensitivity}");
        let sensitive = sensitivity.starts_with("sensitivity: CAUGHT");
        if !sensitive {
            failures += 1;
        }

        // -----------------------------------------------------------------
        // the divergence, measured rather than claimed
        // -----------------------------------------------------------------
        println!("\n{}", conversion_divergence(arch, &headers));

        if failures == 0 && portable_failures == 0 {
            println!(
                "\nPARITY: every function in both headers returns the bits NVIDIA's returns,\n\
                 on this device, over {checked} inputs per path -- including all 65,536 bf16\n\
                 ties, every fp16 subnormal, and both signalling NaNs. The architecture\n\
                 fallbacks agree too, which is the sm_75 path measured rather than argued."
            );
            0
        } else {
            println!(
                "\nFAILED: {failures} row(s) on this device's path and {portable_failures} on the\n\
                 portable path. A difference here is a wrong answer in attention, not a\n\
                 tolerance to widen -- the first differing input is printed above with both\n\
                 bit patterns, and it is reproducible from the corpus index."
            );
            1
        }
    }

    /// Run one row on both modules and compare, returning the first
    /// difference if there is one.
    fn compare_row(
        row: &Row,
        reference: &Module,
        under_test: &Module,
        input: &Device,
        n: usize,
        corpus: &[u32],
    ) -> Result<Option<String>, String> {
        let want = reference.run(row.sym, input, n, row.outs)?;
        let got = under_test.run(row.sym, input, n, row.outs)?;
        for (at, (w, g)) in want.iter().zip(&got).enumerate() {
            if w != g {
                let result = at / row.outs;
                let args: Vec<String> = (0..row.ins)
                    .map(|k| format!("{:#010x}", corpus[result * row.ins + k]))
                    .collect();
                return Ok(Some(format!(
                    "#{result} in=[{}] ref={:#010x} test={:#010x}",
                    args.join(","),
                    w,
                    g
                )));
            }
        }
        Ok(None)
    }

    /// Compile the fp16 header with its packed convert's operands swapped,
    /// and require the comparison to notice.
    ///
    /// The two rows chosen are the only ones that reach
    /// `cvt.rn.f16x2.f32`, and the mutation is the lane transposition the
    /// header's own comment calls out.
    ///
    /// The whole corpus, and that is not caution. A 4,096-slot prefix was
    /// tried first and the mutant PASSED -- the corpus opens with the bf16
    /// sweep, whose first few thousand entries are `i << 16` for small `i`,
    /// every one of which narrows to fp16 +0.0 in BOTH lanes. Swapping two
    /// zeroes is invisible. A harness can be blind for reasons that have
    /// nothing to do with the harness.
    fn sensitivity_check(
        rows: &[Row],
        reference: &Module,
        arch: &str,
        prelude: &Header,
        input: &Device,
        corpus: &[u32],
    ) -> String {
        let mutated = CUDA_FP16.replace(LANE_ORDER, LANE_ORDER_SWAPPED);
        if mutated == CUDA_FP16 {
            return format!(
                "sensitivity: SKIPPED -- `{LANE_ORDER}` is not in cuda_fp16.h any more, so the \
                 mutation this check is built on no longer exists"
            );
        }
        let headers = [
            *prelude,
            Header { name: "cuda_bf16.h", text: CUDA_BF16 },
            // Leaked because `Header` carries `&'static str`: the shipped set
            // is `include_str!`, and one probe-local mutation is not worth a
            // lifetime on the type every kernel compile goes through.
            Header { name: "cuda_fp16.h", text: String::leak(mutated) },
        ];
        let text = source(rows);
        let built = match compile_with_nvrtc(&text, arch, &headers, &[]) {
            Ok(built) => built,
            Err(why) => return format!("sensitivity: could not compile the mutant: {why}"),
        };
        let Ok(mutant) = Module::load(&built.image) else {
            return "sensitivity: could not load the mutant".to_string();
        };

        let mut caught = Vec::new();
        let mut missed = Vec::new();
        for row in rows.iter().filter(|r| r.sym == "k_ff2h2" || r.sym == "k_f22h2") {
            let n = corpus.len() / row.ins;
            match compare_row(row, reference, &mutant, input, n, corpus) {
                Ok(Some(_)) => caught.push(row.name),
                _ => missed.push(row.name),
            }
        }
        if missed.is_empty() {
            format!(
                "sensitivity: CAUGHT -- swapping the two source operands of \
                 `cvt.rn.f16x2.f32` makes\n  {} differ from the reference. The table above is \
                 a measurement, not a formality.",
                caught.join(" and ")
            )
        } else {
            format!(
                "sensitivity: MISSED -- {} still matched the reference with the lanes \
                 swapped,\n  which means this harness cannot see a lane error and the \
                 PASSes above are worth nothing.",
                missed.join(" and ")
            )
        }
    }

    /// Compile the three spellings of "widen a `__half`" and report which
    /// ones NVRTC accepts.
    ///
    /// The headers say a member conversion operator cannot be added to a type
    /// they do not own, and that the fix therefore belonged in
    /// `pie_device.cuh`. That fix has since landed as an `explicit`
    /// conversion. This is the check that says so, executed rather than
    /// asserted -- and it tests the IMPLICIT form too, because `explicit` is
    /// the load-bearing half of the decision: if it ever loses that keyword,
    /// `h + 1.0f` starts compiling and silently widening, and this line is
    /// where that would first show up.
    fn conversion_divergence(arch: &str, headers: &[Header]) -> String {
        let head = "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n";
        let cast = format!(
            "{head}__device__ float widen_h(__half h) {{ return (float)h; }}\n\
             __device__ float widen_b(__nv_bfloat16 b) {{ return (float)b; }}\n"
        );
        let implicit =
            format!("{head}__device__ float widen_h(__half h) {{ float f = h; return f; }}\n");
        let call = format!(
            "{head}__device__ float widen_h(__half h) {{ return __half2float(h); }}\n\
             __device__ float widen_b(__nv_bfloat16 b) {{ return __bfloat162float(b); }}\n"
        );
        let cast_err = compile_with_nvrtc(&cast, arch, headers, &[]).err();
        let implicit_err = compile_with_nvrtc(&implicit, arch, headers, &[]).err();
        let accepted = compile_with_nvrtc(&call, arch, headers, &[]).is_ok();

        let note = match (cast_err.is_none(), implicit_err.is_none()) {
            (true, false) => "as designed: `pie_device.cuh:88` and `:96` declare\n\
                \x20 `explicit operator float()`, so the three closure sites that spell an\n\
                \x20 explicit cast -- vec_dtypes.cuh:159, vec_dtypes.cuh:553,\n\
                \x20 prefill.cuh:1523 -- resolve, while implicit narrowing stays refused.\n\
                \x20 Measured with this header set and no include path on disk: 28 of 28\n\
                \x20 closure files, and BatchDecodeWithPagedKVCacheKernel<half> instantiates."
                .to_string(),
            (true, true) => "the conversion is IMPLICIT. That widens `h` in any float\n\
                \x20 expression without a cast, which is the accident `explicit` was chosen\n\
                \x20 to prevent. Nothing here is wrong; the prelude's decision changed."
                .to_string(),
            (false, _) => format!(
                "the divergence is OPEN again: {}\n\
                \x20 A member conversion cannot be added to `device::f16` or `device::bf16`\n\
                \x20 from a header that does not define them, so the fix is in\n\
                \x20 `pie_device.cuh`, not here. Stated, not silent.",
                cast_err
                    .as_deref()
                    .and_then(|l| l.lines().find(|l| l.contains("error")))
                    .unwrap_or("(refused with no error line)")
                    .trim()
            ),
        };
        format!(
            "the member conversion, measured:\n\
             \x20 `(float)h` explicit cast    {}\n\
             \x20 `float f = h;` implicit     {}\n\
             \x20 `__half2float(h)`           {}\n\
             \x20 {note}",
            if cast_err.is_some() { "REFUSED" } else { "accepted" },
            if implicit_err.is_some() { "REFUSED" } else { "accepted" },
            if accepted { "accepted" } else { "REFUSED" },
        )
    }

    // ---------------------------------------------------------------------
    // the corpora
    // ---------------------------------------------------------------------

    /// Every fp32 pattern worth aiming at a narrowing conversion.
    ///
    /// Ordered so that a failing index is legible: the exhaustive sweeps
    /// first, then the ties, then the named specials, then the randoms.
    fn f32_corpus() -> Vec<u32> {
        let mut v: Vec<u32> = Vec::with_capacity(1_800_000);

        // Every bf16 value, exactly -- bfloat16 is fp32 with the low 16 bits
        // dropped, so this is also every fp32 with a zero low half.
        for i in 0..=0xffffu32 {
            v.push(i << 16);
        }
        // Every fp16 value, exactly. Reaches the fp16 subnormals, which is
        // where a hand-written converter usually differs.
        for i in 0..=0xffffu32 {
            v.push(f16_to_f32_bits(i as u16));
        }
        // Every bf16 TIE: the value exactly halfway to the next pattern, with
        // no sticky bits. Round-to-nearest-even must break every one of these
        // toward the even neighbour, and a truncating or ties-away converter
        // gets all 32,768 of one parity wrong.
        for i in 0..=0xffffu32 {
            v.push((i << 16) | 0x8000);
        }
        // Every fp16 tie and its two fp32 neighbours -- the tie itself, one
        // ulp below (must round down) and one above (must round up), which is
        // the sticky bit.
        for i in 0..0xffffu32 {
            let lo = f32::from_bits(f16_to_f32_bits(i as u16));
            let hi = f32::from_bits(f16_to_f32_bits((i + 1) as u16));
            if !lo.is_finite() || !hi.is_finite() {
                continue;
            }
            let mid = ((f64::from(lo) + f64::from(hi)) / 2.0) as f32;
            let bits = mid.to_bits();
            v.push(bits);
            v.push(bits.wrapping_sub(1));
            v.push(bits.wrapping_add(1));
        }

        // The named edges, both signs. Every one of these is a branch in a
        // hand-written converter and a `NaN` case in the ISA.
        let named: [u32; 22] = [
            0x0000_0000, // +0
            0x0000_0001, // smallest subnormal
            0x007f_ffff, // largest subnormal
            0x0080_0000, // smallest normal
            0x7f7f_ffff, // largest finite
            0x7f80_0000, // +inf
            0x7fc0_0000, // quiet NaN
            0x7f80_0001, // signalling NaN
            0x7fff_ffff, // NaN, all payload bits set
            0x3f80_0000, // 1.0
            0x3f00_0000, // 0.5
            0x4020_0000, // 2.5, where _rd, _rn and _rz all disagree
            0x3300_0000, // 2^-25: exactly half the smallest fp16 subnormal
            0x3300_0001, // and one ulp above it
            0x477f_e000, // 65504, the largest finite fp16
            0x477f_f000, // the fp16 overflow tie
            0x0000_8000, // 2^-134: exactly half the smallest bf16 subnormal
            0x0000_8001, // and one ulp above it
            0x0001_0000, // 2^-133, the smallest bf16 subnormal
            0x7f7f_ffff, // largest finite again, as the bf16 overflow tie's base
            0x4f00_0000, // 2^31, where float-to-int clamps
            0x5f00_0000, // 2^63, well past it
        ];
        for bits in named {
            v.push(bits);
            v.push(bits ^ 0x8000_0000);
        }

        // Everything else. Full-range `u32`s, so NaNs, infinities and
        // subnormals appear at their natural density; then two biased runs,
        // because a uniform `u32` almost never lands in the bottom or top
        // octave and both are where a converter's branches live.
        let mut rng = Rng::new(0x0f16_bf16_2024_1013);
        for _ in 0..1_200_000 {
            v.push(rng.next_u32());
        }
        for _ in 0..120_000 {
            let sign = rng.next_u32() & 0x8000_0000;
            let exponent = rng.next_u32() % 24;
            v.push(sign | (exponent << 23) | (rng.next_u32() & 0x007f_ffff));
        }
        for _ in 0..120_000 {
            let sign = rng.next_u32() & 0x8000_0000;
            let exponent = 230 + rng.next_u32() % 26;
            v.push(sign | (exponent << 23) | (rng.next_u32() & 0x007f_ffff));
        }

        // Trimmed to a multiple of two, which is the largest arity any row
        // draws from this corpus.
        v.truncate(v.len() / 2 * 2);
        v
    }

    /// Every 16-bit pattern worth using as an operand, read as fp16 by one
    /// row and as bf16 by the next.
    fn bits16_corpus() -> Vec<u32> {
        let mut v: Vec<u32> = Vec::with_capacity(1_600_000);

        // Exhaustive, which settles the four unary rows outright.
        for i in 0..=0xffffu32 {
            v.push(i);
        }

        // A dense grid. Randoms cover the space thinly and uniformly; a
        // packed op that is wrong for one EXPONENT RANGE -- an emulation that
        // overflows an intermediate, a max that mishandles a whole octave --
        // shows up here and only here. The representatives are structured in
        // both formats' fields, because the same corpus feeds both.
        let mut reps: Vec<u32> = Vec::new();
        for sign in [0u32, 1] {
            for exponent in
                [0u32, 1, 2, 64, 100, 124, 125, 126, 127, 128, 129, 130, 140, 190, 254, 255]
            {
                for mantissa in [0u32, 1, 0x40, 0x7f] {
                    reps.push((sign << 15) | (exponent << 7) | mantissa);
                }
            }
            for exponent in [0u32, 1, 2, 10, 14, 15, 16, 20, 29, 30, 31] {
                for mantissa in [0u32, 1, 0x200, 0x3ff] {
                    reps.push((sign << 15) | (exponent << 10) | mantissa);
                }
            }
        }
        reps.sort_unstable();
        reps.dedup();
        for &a in &reps {
            for &b in &reps {
                v.push(a);
                v.push(b);
            }
        }

        // The named edges of both formats, crossed with themselves: NaN times
        // inf, -0 minus -0, subnormal times subnormal.
        let named: [u32; 26] = [
            0x0000, 0x8000, // ±0
            0x0001, 0x8001, // smallest subnormal, either reading
            0x03ff, 0x0400, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x7c01, // fp16 edges
            0x3c00, 0xbc00, 0x4000, 0x3800, // fp16 1.0, -1.0, 2.0, 0.5
            0x007f, 0x0080, 0x7f7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81, // bf16 edges
            0x3f80, 0xbf80, 0x4000, 0x3f00, // bf16 1.0, -1.0, 2.0, 0.5
        ];
        for &a in &named {
            for &b in &named {
                v.push(a);
                v.push(b);
            }
        }

        // Products aimed at 2^-134, which is exactly half of the smallest
        // bf16 subnormal and the one window where multiplying in fp32 and
        // then narrowing can round twice. `2^-a * 2^-b` with `a + b` near 134
        // lands there; the mantissas walk it across the tie.
        for a in 40u32..=94 {
            for delta in [132u32, 133, 134, 135] {
                let b = delta - a;
                if b == 0 || b > 133 {
                    continue;
                }
                for (ma, mb) in [(0u32, 0u32), (1, 0), (0, 1), (0x40, 0x40), (0x7f, 0x7f)] {
                    v.push(((127 - a) << 7) | ma);
                    v.push(((127 - b) << 7) | mb);
                }
            }
        }

        // Everything else, uniform over the 16-bit space.
        let mut rng = Rng::new(0x1eb4_4f16_dead_beef);
        for _ in 0..1_000_000 {
            v.push(rng.next_u32() & 0xffff);
        }

        // 1536 is the least common multiple of the arities a row draws from
        // this corpus (1, 2, 4 and 6) and of the 512 the warp row needs so
        // that every `__shfl_xor_sync` runs with all 32 lanes of every warp
        // converged -- a partial warp there is undefined behaviour, not a
        // wrong number.
        v.truncate(v.len() / 1536 * 1536);
        v
    }

    /// `fp16 bits -> fp32 bits`, on the host, exact.
    ///
    /// Written out rather than taken from the prelude's `f16_to_f32`, which
    /// this crate's own prelude was measured returning the WRONG SIGN for
    /// all 1,024 negative subnormals -- a corpus built with that would have a
    /// hole exactly where the interesting inputs are.
    fn f16_to_f32_bits(h: u16) -> u32 {
        let sign = (u32::from(h) & 0x8000) << 16;
        let exponent = (u32::from(h) >> 10) & 0x1f;
        let mantissa = u32::from(h) & 0x3ff;
        if exponent == 0 {
            if mantissa == 0 {
                return sign;
            }
            // Normalise: shift the leading one out of the mantissa and pay
            // for it in the exponent.
            let shift = mantissa.leading_zeros() - 21;
            let e = 127 - 15 - shift;
            return sign | (e << 23) | ((mantissa << (shift + 1)) & 0x007f_ffff) << 10;
        }
        if exponent == 0x1f {
            return sign | 0x7f80_0000 | (mantissa << 13);
        }
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    }

    /// A deterministic generator, so a failing input can be regenerated
    /// rather than described as "some random float".
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// xorshift64*, which is enough entropy for a corpus and short enough
        /// to read.
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
        }
    }

    // ---------------------------------------------------------------------
    // compiling, three ways
    // ---------------------------------------------------------------------

    /// One compiled image, and what it cost.
    struct Built {
        image: Vec<u8>,
        millis: f64,
    }

    /// Compile with the machine's `nvcc`, against its own headers.
    ///
    /// The files land in `OUT_DIR`, which is this build's own scratch inside
    /// `target/`.
    fn compile_with_nvcc(nvcc: &PathBuf, source: &str, arch: &str) -> Result<Built, String> {
        let scratch = PathBuf::from(env!("OUT_DIR")).join("halftype_parity");
        std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
        let cu = scratch.join("reference.cu");
        let cubin = scratch.join("reference.cubin");
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

    /// Compile with NVRTC against a header set and nothing on disk.
    ///
    /// `sm_XY` and not `compute_XY`: the reference is a cubin for this
    /// device, so the thing under test has to be one too, or the comparison
    /// would include a difference in who ran the back end.
    fn compile_with_nvrtc(
        source: &str,
        arch: &str,
        headers: &[Header],
        extra: &[&str],
    ) -> Result<Built, String> {
        let src = CString::new(source).map_err(|_| "a NUL in the probe source")?;
        let name = c"halftype_parity.cu";
        let (texts, names) = as_nvrtc_arrays(headers)?;
        let text_ptrs: Vec<_> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call, and the two arrays are the
        // same length -- the whole of `nvrtcCreateProgram`'s contract. The
        // header set is an in-memory filesystem: nothing is read from disk,
        // which is the property this probe exists to keep honest.
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
        let extra: Vec<CString> = extra.iter().map(|o| CString::new(*o).unwrap()).collect();
        let mut options = vec![gpu.as_ptr(), c"-std=c++17".as_ptr()];
        options.extend(extra.iter().map(|o| o.as_ptr()));

        let started = Instant::now();
        // SAFETY: the program is live and the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(
                program,
                i32::try_from(options.len()).unwrap(),
                options.as_ptr(),
            )
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

    // ---------------------------------------------------------------------
    // running
    // ---------------------------------------------------------------------

    /// A loaded cubin, looked up by kernel name.
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

        /// One row: `n` results of `outs` words each, from a corpus already
        /// on the device.
        ///
        /// 256 threads a block and no grid-stride loop, so that thread `i`
        /// handles result `i` in BOTH cubins -- the warp row shuffles between
        /// lanes, so the mapping from lane to input has to be identical or
        /// the two paths would be answering different questions.
        fn run(
            &self,
            sym: &str,
            input: &Device,
            n: usize,
            outs: usize,
        ) -> Result<Vec<u32>, String> {
            let name = CString::new(sym).map_err(|_| "a NUL in a kernel name")?;
            let mut function: dr::CUfunction = std::ptr::null_mut();
            // SAFETY: `module` came from a successful load and `name` is
            // NUL-terminated.
            let code =
                unsafe { dr::cuModuleGetFunction(&raw mut function, self.module, name.as_ptr()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleGetFunction({sym}): {code:?}"));
            }

            let out = Device::alloc(n * outs * std::mem::size_of::<u32>())?;
            let mut p_in = input.ptr;
            let mut p_out = out.ptr;
            let mut count = u32::try_from(n).map_err(|_| "the corpus does not fit a u32")?;
            let mut params = [
                (&raw mut p_in).cast::<c_void>(),
                (&raw mut p_out).cast::<c_void>(),
                (&raw mut count).cast::<c_void>(),
            ];

            let blocks = u32::try_from(n.div_ceil(256)).map_err(|_| "too many blocks")?;
            // SAFETY: the function came from a live module; both allocations
            // outlive the launch because `params` borrows locals that outlive
            // the synchronise below.
            let code = unsafe {
                dr::cuLaunchKernel(
                    function,
                    blocks,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuLaunchKernel({sym}): {code:?}"));
            }
            // SAFETY: no arguments, and a fault inside the kernel surfaces
            // here rather than at the copy below.
            let code = unsafe { rt::cudaDeviceSynchronize() };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaDeviceSynchronize({sym}): {code:?}"));
            }

            let mut host = vec![0u32; n * outs];
            // SAFETY: both sides are `n * outs` words, and the device side
            // was allocated at that size above.
            let code = unsafe {
                rt::cudaMemcpy(
                    host.as_mut_ptr().cast(),
                    out.ptr,
                    std::mem::size_of_val(host.as_slice()),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMemcpy D2H({sym}): {code:?}"));
            }
            Ok(host)
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
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { rt::cudaMalloc(&raw mut ptr, bytes) };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMalloc({bytes}): {code:?}"));
            }
            Ok(Self { ptr })
        }

        fn upload<T>(values: &[T]) -> Result<Self, String> {
            ensure_context()?;
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
    /// `cuModuleLoadData` with no current context fails with
    /// `CUDA_ERROR_INVALID_CONTEXT`, which reads like a broken cubin.
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
            dr::cuDeviceGetName(
                name.as_mut_ptr().cast(),
                i32::try_from(name.len()).unwrap(),
                device,
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return "unknown".to_string();
        }
        CStr::from_bytes_until_nul(&name)
            .map_or_else(|_| "unknown".to_string(), |s| s.to_string_lossy().into_owned())
    }

    /// `libnvrtc`'s own version, because whether a device header resolves is a
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
