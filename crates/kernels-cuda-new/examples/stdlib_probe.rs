//! Which of FlashInfer's 31 external includes does NVRTC supply by itself?
//!
//! # Why this is the decisive measurement
//!
//! [`header_probe`] settled the small question — this NVRTC ships **no** CUDA
//! device headers, so `cuda_fp8.h` and friends must be replaced by shims. This
//! settles the large one.
//!
//! Our FlashInfer closure is 28 files and 17,981 lines, reached from fifteen
//! headers that `kernels-cuda/csrc/src` includes by name. Walking its
//! `#include` graph leaves **31 directives that point outside the tree**, and
//! whether attention can ever be JIT-compiled comes down to which of them the
//! compiler already answers:
//!
//! * **Answered by NVRTC** — free. Nothing to carry, nothing to shim.
//! * **Host-only** — used inside `#ifndef __CUDACC_RTC__` or in plan code a
//!   device compile never reaches. These need a GUARD upstream, not a header.
//! * **Device, unanswered** — the real bill. Each is a shim we write, and the
//!   only acceptable gate on a shim that substitutes arithmetic under someone
//!   else's kernel is bit-parity, not "it compiled".
//!
//! Guessing which bucket a header falls in is how an estimate goes wrong by an
//! order of magnitude in either direction. This asks the compiler.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example stdlib_probe
//! ```
//!
//! [`header_probe`]: ../header_probe/index.html

use std::ffi::{CStr, CString};

use cudarc::nvrtc::sys as nv;

/// Every directive that leaves the FlashInfer closure, as the walk found them.
///
/// Grouped by what they are rather than alphabetically, because the grouping
/// is the finding: the probe's job is to confirm or refute the grouping, and a
/// sorted list would hide which bucket moved.
const EXTERNAL: &[(&str, &str)] = &[
    // The C++ standard library. Every one of these is reached from host code
    // in FlashInfer -- `FLASHINFER_CUDA_CALL`'s `std::ostringstream`, the
    // plan functions' `std::vector` -- but `type_traits` and `cstdint` are
    // also read by device templates, so which bucket they land in is exactly
    // what has to be measured rather than assumed.
    ("std", "algorithm"),
    ("std", "atomic"),
    ("std", "bit"),
    ("std", "cmath"),
    ("std", "cstddef"),
    ("std", "cstdint"),
    ("std", "exception"),
    ("std", "iostream"),
    ("std", "limits"),
    ("std", "memory"),
    ("std", "sstream"),
    ("std", "stdexcept"),
    ("std", "string"),
    ("std", "tuple"),
    ("std", "type_traits"),
    ("std", "utility"),
    ("std", "vector"),
    // The CUDA host API. NVRTC predefines the built-in variables and types a
    // kernel uses, so these may be answerable or may simply be unnecessary --
    // the distinction matters, because "unnecessary" is a guard upstream and
    // "answerable" is nothing at all.
    ("cuda-host", "cuda.h"),
    ("cuda-host", "cuda_runtime.h"),
    ("cuda-host", "cuda_runtime_api.h"),
    ("cuda-host", "cuda_device_runtime_api.h"),
    ("cuda-host", "driver_types.h"),
    // The device headers. THIS is the bill: FlashInfer names `__nv_bfloat16`
    // and `__half` directly and calls 39 distinct intrinsics on them, so a
    // prelude that defines its own bf16 does not substitute -- the type is
    // matched by name.
    ("device", "cuda_fp16.h"),
    ("device", "cuda_bf16.h"),
    ("device", "cuda_fp8.h"),
    ("device", "cuda_fp4.h"),
    // CCCL. Two are already shimmed in this crate at a cost of 14,637 bytes
    // against CCCL's 13,691,725; these two are what remains of that door.
    ("cccl", "cooperative_groups.h"),
    ("cccl", "cuda/std/limits"),
    ("cccl", "cuda/cmath"),
    ("cccl", "cuda/pipeline"),
    // The outlier, and worth naming: a CUTLASS utility reaches for Boost.
    ("other", "boost/math/ccmath/fabs.hpp"),
];

fn main() {
    let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
    println!("NVRTC {} · {arch}\n", version());
    println!("Which of FlashInfer's 31 external includes does NVRTC answer alone?\n");

    let mut answered: Vec<&str> = Vec::new();
    let mut unanswered: Vec<(&str, &str)> = Vec::new();

    let mut group = "";
    for (kind, header) in EXTERNAL {
        if *kind != group {
            println!("  -- {kind} --");
            group = kind;
        }
        let source = format!("#include <{header}>\n__global__ void k() {{}}\n");
        match compile(&source, arch) {
            Ok(millis) => {
                println!("     ANSWERED  {header:<34} {millis:5.0} ms");
                answered.push(header);
            }
            Err(_) => {
                println!("     no        {header}");
                unanswered.push((kind, header));
            }
        }
    }

    let device: Vec<&str> =
        unanswered.iter().filter(|(k, _)| *k == "device").map(|(_, h)| *h).collect();
    let cccl: Vec<&str> = unanswered.iter().filter(|(k, _)| *k == "cccl").map(|(_, h)| *h).collect();
    let host: Vec<&str> = unanswered
        .iter()
        .filter(|(k, _)| *k == "std" || *k == "cuda-host" || *k == "other")
        .map(|(_, h)| *h)
        .collect();

    println!("\n{} of {} answered by NVRTC alone.\n", answered.len(), EXTERNAL.len());
    println!("The bill for JIT-compiling attention, by kind:");
    println!(
        "  shim (device arithmetic, must be bit-exact) : {} -- {}",
        device.len(),
        list(&device)
    );
    println!("  shim (CCCL door, 2 already written)         : {} -- {}", cccl.len(), list(&cccl));
    println!(
        "  guard upstream (#ifndef __CUDACC_RTC__)     : {} -- {}",
        host.len(),
        list(&host)
    );
    println!(
        "\nThe third row is a patch to someone else's source and the first is\n\
         arithmetic we take responsibility for. They are not the same kind of\n\
         work and an estimate that adds them is wrong."
    );
}

/// A comma-joined list, or a word saying there is nothing to list.
fn list(items: &[&str]) -> String {
    if items.is_empty() { "none".to_string() } else { items.join(", ") }
}

/// `libnvrtc`'s own version, which is what decides every answer above.
fn version() -> String {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both are live out-parameters for the call's duration.
    let code = unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
    if code == nv::nvrtcResult::NVRTC_SUCCESS {
        format!("{major}.{minor}")
    } else {
        format!("unavailable ({code:?})")
    }
}

/// Compile `source` for `arch` with an EMPTY header set.
///
/// Empty is the point: supplying the header would prove it exists on this
/// disk, which is not in question.
fn compile(source: &str, arch: &str) -> Result<f64, String> {
    let src = CString::new(source).expect("no NULs in a probe");
    let name = CString::new("probe.cu").unwrap();
    let mut program: nv::nvrtcProgram = std::ptr::null_mut();
    // SAFETY: both strings outlive the call; a header count of zero means the
    // two array pointers are never read.
    let code = unsafe {
        nv::nvrtcCreateProgram(
            &raw mut program,
            src.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if code != nv::nvrtcResult::NVRTC_SUCCESS {
        return Err(format!("nvrtcCreateProgram: {code:?}"));
    }

    let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
    let std17 = CString::new("--std=c++17").unwrap();
    let options = [gpu.as_ptr(), std17.as_ptr()];

    let started = std::time::Instant::now();
    // SAFETY: `program` came from a successful create; the options outlive it.
    let code = unsafe {
        nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
    };
    let millis = started.elapsed().as_secs_f64() * 1e3;

    let mut size = 0;
    // SAFETY: `program` is live and `size` is a live out-parameter.
    unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
    let mut log = vec![0u8; size.max(1)];
    // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked for.
    unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
    // SAFETY: destroyed exactly once, and not used after.
    unsafe { nv::nvrtcDestroyProgram(&raw mut program) };

    let log = CStr::from_bytes_until_nul(&log)
        .map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned());
    if code == nv::nvrtcResult::NVRTC_SUCCESS { Ok(millis) } else { Err(log) }
}
