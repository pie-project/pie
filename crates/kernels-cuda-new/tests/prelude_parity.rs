//! The prelude's conversions, swept exhaustively against an independent
//! implementation.
//!
//! # The defect this exists because of
//!
//! `pie_device.cuh` widened fp16 with
//!
//! ```text
//! return __int_as_float(s) + (m == 0 ? 0.f : 2^-24 * m);
//! ```
//!
//! which reads as right and is not. `__int_as_float(s)` is `-0.0f` when the
//! sign bit is set, and **`-0.0 + x` is `+x`** for every positive `x`. So all
//! 1,023 negative fp16 subnormals widened POSITIVE, and `-0.0` widened to
//! `+0.0`: a sign flip on the smallest magnitudes, in a header carried by
//! every JIT unit in this crate.
//!
//! It survived a whole migration. Every test that read a normal number
//! passed, because the defect is confined to one exponent — `e == 0` — and
//! nothing sampled it. Reverting the fix and re-running this file reports
//! **1,024 mismatches, the first at `0x8000`**, which is exactly `-0.0`.
//!
//! # Why the reference is written here rather than borrowed
//!
//! The obvious reference is `__half2float` out of `cuda_fp16.h`. This file
//! does not use it, for two reasons that are the same reason: the crate
//! deliberately carries no NVIDIA header (see `new-horizon.md` §13), and a
//! test whose oracle is the thing being replaced cannot outlive the
//! replacement. The oracle below is fifteen lines of Rust written from the
//! IEEE-754 binary16 definition — an INDEPENDENT implementation, which is
//! what makes an agreement worth something.
//!
//! # Exhaustive, because sampling is what failed
//!
//! All 65,536 patterns, not a sample. There are only 65,536, the sweep costs
//! one launch, and the defect above is precisely what a sample misses.

#![cfg(feature = "_cuda")]

use std::ffi::{CStr, CString};

use cudarc::driver::sys as dr;
use cudarc::nvrtc::sys as nv;
use kernels_cuda_new::runtime::cache;
use kernels_cuda_new::source;

/// IEEE-754 binary16 → binary32, from the definition.
///
/// Every case spelled out rather than folded together, because the folding is
/// what went wrong in the header this checks.
fn widen(raw: u16) -> f32 {
    let sign = f32::from_bits(u32::from(raw & 0x8000) << 16);
    let exponent = (raw >> 10) & 0x1f;
    let mantissa = u32::from(raw & 0x03ff);
    match exponent {
        // Zero and subnormal. `2^-24 * m` is exact for m in [0, 1023], and the
        // sign is applied by COPYING it rather than by adding a signed zero.
        0 => f32::from_bits((mantissa as f32 * 2.0f32.powi(-24)).to_bits() | sign.to_bits()),
        // Infinity and NaN.
        31 => f32::from_bits(sign.to_bits() | 0x7f80_0000 | (mantissa << 13)),
        // Normal: rebias 15 → 127 and shift the mantissa into place.
        _ => f32::from_bits(
            sign.to_bits() | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
        ),
    }
}

/// A kernel that widens every fp16 pattern and writes the f32 bits.
const SWEEP: &str = r#"
#include "pie_device.cuh"
namespace pie_cuda_driver { namespace kernels { namespace probe {
__global__ void widen_all(unsigned int* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536u) return;
    device::f16 v; v.raw = (unsigned short)i;
    out[i] = __float_as_int(device::f16_to_f32(v));
}
}}}
"#;

#[test]
fn every_fp16_pattern_widens_to_the_same_bits() {
    let Some(arch) = cache::arch() else {
        eprintln!("SKIP every_fp16_pattern_widens_to_the_same_bits: no CUDA device is current");
        return;
    };
    cache::bind_context().expect("a primary context");

    let cubin = compile(SWEEP, arch, "::pie_cuda_driver::kernels::probe::widen_all");
    let device = run(&cubin.image, &cubin.lowered);

    let mut mismatches = Vec::new();
    for raw in 0u32..=0xffff {
        #[allow(clippy::cast_possible_truncation)]
        let expected = widen(raw as u16).to_bits();
        let got = device[raw as usize];
        // Two NaNs with different payloads are both NaN; nothing else is
        // allowed to differ by a bit.
        let both_nan = (expected & 0x7fff_ffff) > 0x7f80_0000 && (got & 0x7fff_ffff) > 0x7f80_0000;
        if expected != got && !both_nan {
            mismatches.push((raw, expected, got));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of 65536 fp16 patterns widen to the wrong bits; first three: {:x?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(3)]
    );
}

/// The sign of a zero survives, stated on its own so a failure names it.
///
/// A subset of the sweep above, and worth its own assertion because `-0.0` is
/// the one value whose corruption is invisible in every arithmetic that
/// follows it — until a division produces the wrong infinity.
#[test]
fn negative_zero_stays_negative() {
    assert_eq!(widen(0x8000).to_bits(), 0x8000_0000, "the oracle itself must keep the sign");
    assert_eq!(widen(0x0000).to_bits(), 0x0000_0000);
    assert!(widen(0x8001) < 0.0, "the smallest negative subnormal is negative");
}

/// A compiled probe: the image, and the mangled name its entry got.
///
/// Both, because they are useless apart — and because the mangled name is the
/// one thing about a JIT that must never be guessed. An earlier version of
/// this file spelled the Itanium mangling by hand and got the namespace's
/// length prefix wrong, which `cuModuleGetFunction` reported as
/// `CUDA_ERROR_NOT_FOUND`: a true statement about a mistake nothing in it
/// names. `nvrtcGetLoweredName` is what `runtime::nvrtc` asks, and it is what
/// this asks.
struct Probe {
    image: Vec<u8>,
    lowered: String,
}

/// Compile `source` against the crate's shipped header set and return a cubin.
fn compile(source: &str, arch: &str, instantiation: &str) -> Probe {
    let (texts, names) = source::as_nvrtc_arrays(source::DEVICE_HEADERS).expect("no NULs");
    let text_ptrs: Vec<*const i8> = texts.iter().map(|c| c.as_ptr()).collect();
    let name_ptrs: Vec<*const i8> = names.iter().map(|c| c.as_ptr()).collect();
    let src = CString::new(source).unwrap();
    let unit = CString::new("prelude_parity.cu").unwrap();
    let mut program: nv::nvrtcProgram = std::ptr::null_mut();
    // SAFETY: every pointer outlives the call, and the two arrays are the
    // same length as the count passed with them.
    let code = unsafe {
        nv::nvrtcCreateProgram(
            &raw mut program,
            src.as_ptr(),
            unit.as_ptr(),
            i32::try_from(text_ptrs.len()).unwrap(),
            text_ptrs.as_ptr(),
            name_ptrs.as_ptr(),
        )
    };
    assert_eq!(code, nv::nvrtcResult::NVRTC_SUCCESS, "nvrtcCreateProgram");

    let expression = CString::new(instantiation).unwrap();
    // SAFETY: `program` is live and the expression outlives the call.
    unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };

    let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
    let std17 = CString::new("--std=c++17").unwrap();
    let options = [gpu.as_ptr(), std17.as_ptr()];
    // SAFETY: `program` is live; the options outlive the call.
    let code = unsafe {
        nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
    };
    if code != nv::nvrtcResult::NVRTC_SUCCESS {
        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        let mut log = vec![0u8; size.max(1)];
        // SAFETY: the buffer is the size NVRTC just asked for.
        unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
        panic!("{}", String::from_utf8_lossy(&log));
    }

    let mut size = 0;
    // SAFETY: `program` compiled successfully and `size` is a live slot.
    unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
    let mut image = vec![0u8; size];
    // SAFETY: the buffer is the size NVRTC just asked for.
    unsafe { nv::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) };

    // BEFORE the destroy: the lowered name points into the program's own
    // storage, so it is copied here rather than borrowed out.
    let mut lowered_ptr: *const i8 = std::ptr::null();
    // SAFETY: `program` is live and was compiled with this expression added.
    let code =
        unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut lowered_ptr) };
    assert_eq!(code, nv::nvrtcResult::NVRTC_SUCCESS, "nvrtcGetLoweredName");
    // SAFETY: NVRTC returned a NUL-terminated name valid until the destroy.
    let lowered = unsafe { CStr::from_ptr(lowered_ptr) }.to_string_lossy().into_owned();

    // SAFETY: destroyed exactly once, after the cubin and the name are copied.
    unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
    Probe { image, lowered }
}

/// Load `cubin`, launch its sweep over all 65,536 patterns, and read back.
fn run(cubin: &[u8], lowered: &str) -> Vec<u32> {
    const N: usize = 65536;
    let mut module: dr::CUmodule = std::ptr::null_mut();
    // SAFETY: `cubin` is a complete image produced by NVRTC above.
    let code = unsafe { dr::cuModuleLoadData(&raw mut module, cubin.as_ptr().cast()) };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuModuleLoadData");

    let entry_name = CString::new(lowered).unwrap();
    let mut function: dr::CUfunction = std::ptr::null_mut();
    // SAFETY: `module` is live and the name outlives the call.
    let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, entry_name.as_ptr()) };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuModuleGetFunction");

    let mut buffer: dr::CUdeviceptr = 0;
    // SAFETY: `buffer` is a live out-parameter; the size is non-zero.
    unsafe { dr::cuMemAlloc_v2(&raw mut buffer, N * 4) };
    let mut argument = buffer;
    let mut args: [*mut std::ffi::c_void; 1] = [(&raw mut argument).cast()];
    // SAFETY: the kernel takes exactly the one pointer `args` supplies, and
    // the grid covers 65,536 threads with the kernel guarding the tail.
    let code = unsafe {
        dr::cuLaunchKernel(
            function,
            256,
            1,
            1,
            256,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuLaunchKernel");

    let mut out = vec![0u32; N];
    // SAFETY: no work outstanding beyond the launch above.
    unsafe { dr::cuCtxSynchronize() };
    // SAFETY: the allocation and the destination are both `N * 4` bytes.
    unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), buffer, N * 4) };
    // SAFETY: freed and unloaded exactly once, with nothing outstanding.
    unsafe { dr::cuMemFree_v2(buffer) };
    // SAFETY: no function of this module is in flight.
    unsafe { dr::cuModuleUnload(module) };
    out
}

/// The honest-named headers are self-contained.
///
/// # What replaced a probe, and why
///
/// `csrc/src` carries two spellings of the same arithmetic on purpose —
/// `new-horizon.md` §13.4 — and for a while it carried two IMPLEMENTATIONS,
/// which is a one-definition-rule violation waiting for the first translation
/// unit that quantises a KV page and runs an fp8 attention variant together.
/// `build.rs` walks the directory and carries everything in it, so both
/// spellings are in every header set and nothing prevents that unit from
/// existing.
///
/// `pie_fp8.cuh` and `pie_half2.cuh` are therefore forwarders now, and the
/// probe that used to exercise their bodies was deleted rather than rewritten:
/// its subject had been merged away, and its coverage is strictly subsumed by
/// `halftype_parity` (35 of 35 rows over 32,945,058 comparisons) and
/// `fp8_pipeline_probe` (28 of 28 over every fp8 byte pattern). A probe that
/// pokes at private details of a deleted implementation is worse than no probe.
///
/// What it did prove, and what this keeps, is narrower and real: **one
/// `#include` of an honest name yields a working environment.** `cuda_fp8.h`
/// deliberately includes nothing and takes `__half` on faith, because every
/// FlashInfer file that reaches it includes `<cuda_fp16.h>` first; our own
/// `.cu` files are not FlashInfer and include one header expecting it to work.
/// Encapsulating that ordering is what the honest name is FOR, and this is the
/// check that it does.
#[test]
fn one_include_of_an_honest_name_is_enough() {
    let Some(arch) = cache::arch() else {
        eprintln!("SKIP one_include_of_an_honest_name_is_enough: no CUDA device is current");
        return;
    };
    cache::bind_context().expect("a primary context");

    // Each source names ONE header and then uses the whole environment it is
    // supposed to bring: the vendor types, the conversions, and the prelude
    // the conversions are written over.
    let cases = [
        (
            "pie_fp8.cuh",
            r#"
#include "pie_fp8.cuh"
namespace device = ::pie_cuda_driver::kernels::device;
__global__ void probe(unsigned char* out, const float* in) {
    __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(in[0], __NV_SATFINITE, __NV_E4M3);
    __half back = __half_raw(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
    device::f16 mine; mine.raw = __half_as_ushort(back);
    out[0] = (unsigned char)(device::f16_to_f32(mine) > 0.f ? code : 0);
}
"#,
        ),
        (
            "pie_half2.cuh",
            r#"
#include "pie_half2.cuh"
namespace device = ::pie_cuda_driver::kernels::device;
__global__ void probe(float* out, const unsigned int* in) {
    __half2 a = *reinterpret_cast<const __half2*>(&in[0]);
    __half2 b = __float2half2_rn(2.0f);
    __half2 r = __hsub2(__hfma2(a, b, b), b);
    float2 wide = __half22float2(r);
    device::f16 lane; lane.raw = __half_as_ushort(a.x);
    out[0] = wide.x + wide.y + device::f16_to_f32(lane);
}
"#,
        ),
    ];

    for (header, source) in cases {
        // A refusal panics inside `compile` with NVRTC's own log, which names
        // the identifier that was missing — the diagnosis a caller of the
        // honest name would get, arriving here instead of on a GPU.
        let built = compile(source, arch, "::probe");
        assert!(!built.image.is_empty(), "{header} compiled to nothing");
        assert!(
            built.lowered.contains("probe"),
            "{header}: the entry did not survive as {}",
            built.lowered
        );
    }
}
