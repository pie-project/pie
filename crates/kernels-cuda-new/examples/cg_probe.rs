//! Does a hand-written `cooperative_groups.h` actually close the CCCL door —
//! and does it stay SHUT where it is supposed to?
//!
//! # The question this answers
//!
//! [`kernels_cuda_new::source`] states the rule: *no include path on disk;
//! includes resolve against a header set carried in the binary, or they do
//! not resolve at all.* `examples/header_probe.rs` measured what that costs
//! on this box (L40S, NVRTC 13.0) and the answer for FlashInfer was two
//! refusals — `<cooperative_groups.h>` and `<cuda/std/limits>`, both *"could
//! not open source file … (no directories in search list)"*.
//!
//! Those two directives are the whole of FlashInfer's reach into NVIDIA's
//! CUDA C++ Core Libraries: 13,691,725 bytes across 1691 files on this box,
//! 17 MB as it sits on disk, which cannot be carried without reading
//! `$CUDA_HOME` at build time and giving up a toolkit-free build. And what
//! the attention closure USES through them was measured and is a hand:
//! `cg::this_thread_block()`, `block.sync()`, and one
//! `numeric_limits<float>::infinity()`.
//!
//! So `csrc/shim/cooperative_groups.h` and `csrc/shim/cuda/std/limits` were
//! written — carrying NVIDIA's own spellings, so the upstream source is
//! compiled unmodified — and this example is the claim being cashed. It
//! proves three things, and the third is the one that decays if nobody
//! checks it:
//!
//! 1. The shim COMPILES the surface the measurement found, through NVRTC,
//!    with a header set of exactly two files.
//! 2. `cuda::std::numeric_limits<float>::infinity()` resolves the same way.
//! 3. `cg::this_grid()` is still REFUSED. A grid-wide barrier is a launch
//!    mode — `cudaLaunchCooperativeKernel` and a resident grid — not
//!    something a header can fake, and a `grid.sync()` that did nothing
//!    would turn `BatchMLAPagedAttentionKernel`'s two-stage merge into a
//!    silent wrong answer. A probe that only checked what works would let
//!    someone add that no-op later and find out in a logits diff.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example cg_probe
//! ```

use std::ffi::{CStr, CString};

use cudarc::nvrtc::sys as nv;
use kernels_cuda_new::source::{Header, as_nvrtc_arrays};

/// The header set under test: the two files that stand in for CCCL.
///
/// Built here rather than taken from [`kernels_cuda_new::source`] because a
/// probe must be able to state its own set — this one is deliberately NOT
/// `DEVICE_HEADERS`, since the point is what these two files alone resolve.
/// The names are the literal spellings NVRTC matches against the directive,
/// which is why one of them has a directory in it and no extension.
const SHIM: &[Header] = &[
    Header {
        name: "cooperative_groups.h",
        text: include_str!("../csrc/shim/cooperative_groups.h"),
    },
    Header { name: "cuda/std/limits", text: include_str!("../csrc/shim/cuda/std/limits") },
];

/// What CCCL costs, measured on this box with `du -sb` and `find | wc -l`
/// against `/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl`.
///
/// A constant and not a `std::fs::metadata` walk, because this crate does not
/// read the toolkit — that is the property being defended, and an example
/// that stat'd `/usr/local/cuda` to print a number would be the first line of
/// code in the crate to need one.
const CCCL_BYTES: usize = 13_691_725;

/// The file count behind the same two directives.
const CCCL_FILES: usize = 1691;

/// What a probe is expected to do — and the expectation is the test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// The shim supplies this, so NVRTC must accept it.
    Compiles,
    /// The shim deliberately does not, so NVRTC must refuse it BY NAME.
    Refused,
}

/// One translation unit, what it is for, and which way it has to come out.
struct Probe {
    /// What appears in the report.
    what: &'static str,
    /// A unit that uses the surface for real. A declaration alone would
    /// resolve the header and prove nothing about what is in it.
    source: &'static str,
    /// The verdict this probe asserts.
    expect: Expect,
}

const PROBES: &[Probe] = &[
    Probe {
        what: "cg::this_thread_block + block.sync + thread_rank",
        // The exact surface measured across `decode.cuh` and `prefill.cuh`:
        // seven `this_thread_block()` sites, forty-nine `block.sync()` calls.
        // Angle brackets, because that is what FlashInfer writes and the
        // whole question is whether NVRTC matches `includeNames[]` against a
        // `<...>` directive the same way it does a quoted one.
        source: concat!(
            "#include <cooperative_groups.h>\n",
            "namespace cg = cooperative_groups;\n",
            "__global__ void k(float* p) {\n",
            "    auto block = cg::this_thread_block();\n",
            "    p[block.thread_rank()] = 1.0f;\n",
            "    block.sync();\n",
            "    p[block.thread_rank()] += float(block.size() + block.group_index().x\n",
            "                                    + block.thread_index().x);\n",
            "    block.sync();\n",
            "}\n"
        ),
        expect: Expect::Compiles,
    },
    Probe {
        what: "cuda::std::numeric_limits<float>::infinity  (mla.cuh x4)",
        source: concat!(
            "#include <cuda/std/limits>\n",
            "__global__ void k(float* lse, int n) {\n",
            "    for (int i = threadIdx.x; i < n; i += blockDim.x) {\n",
            "        lse[i] = -cuda::std::numeric_limits<float>::infinity();\n",
            "    }\n",
            "}\n"
        ),
        expect: Expect::Compiles,
    },
    Probe {
        what: "cg::this_grid + grid.sync  (MUST be refused)",
        // `mla.cuh:1061`, reduced. If this ever compiles, someone has added a
        // grid barrier that a launch cannot honour.
        source: concat!(
            "#include <cooperative_groups.h>\n",
            "namespace cg = cooperative_groups;\n",
            "__global__ void k(float* p) {\n",
            "    auto grid = cg::this_grid();\n",
            "    grid.sync();\n",
            "    p[threadIdx.x] = 1.0f;\n",
            "}\n"
        ),
        expect: Expect::Refused,
    },
    Probe {
        what: "both headers in one unit  (the diamond)",
        // Two doors, one translation unit, and `cooperative_groups.h`
        // included twice on purpose: `#pragma once` is what makes that one
        // definition, and it is the reason the set can be handed whole to a
        // compile without knowing who includes whom.
        source: concat!(
            "#include <cooperative_groups.h>\n",
            "#include <cuda/std/limits>\n",
            "#include <cooperative_groups.h>\n",
            "namespace flashinfer {\n",
            "namespace cg = cooperative_groups;\n",
            "__global__ void k(float* p) {\n",
            "    auto block = cg::this_thread_block();\n",
            "    p[block.thread_rank()] = -cuda::std::numeric_limits<float>::infinity();\n",
            "    block.sync();\n",
            "}\n",
            "}\n"
        ),
        expect: Expect::Compiles,
    },
];

fn main() {
    let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
    let bytes: usize = SHIM.iter().map(|h| h.text.len()).sum();

    println!("NVRTC version: {}", version());
    println!("architecture:  {arch}");
    println!("header set:    {} files, {bytes} bytes\n", SHIM.len());
    println!("Does the shim carry what FlashInfer reaches for -- and only that?\n");

    let mut wrong = 0usize;
    for probe in PROBES {
        let result = compile(probe.source, arch, SHIM);
        let (verdict, note) = match &result {
            Ok(millis) => ("OK      ", format!("{millis:6.0} ms")),
            Err(log) => ("REFUSED ", first_line(log)),
        };
        let got = if result.is_ok() { Expect::Compiles } else { Expect::Refused };
        let mark = if got == probe.expect {
            " "
        } else {
            wrong += 1;
            "!"
        };
        println!("  {mark} {verdict} {:<56} {note}", probe.what);
    }

    println!();
    if wrong == 0 {
        println!(
            "Every verdict is the expected one. `this_grid` is still a name error,\n\
             which is what keeps a grid barrier a LAUNCH decision -- \
             `kernels::LaunchRule`\n\
             has no cooperative variant, and until it does a `grid.sync()` \
             that compiled\n\
             would be a wrong answer nothing reports."
        );
    } else {
        println!(
            "{wrong} of {} probes came out the other way. A `Compiles` that refused \
             means the\nshim is missing something the closure uses; a `Refused` that \
             compiled means the\nshim grew something no launch path can honour.",
            PROBES.len()
        );
    }

    println!();
    println!(
        "HEADER SET {bytes} bytes in {} files  vs  CCCL {CCCL_BYTES} bytes in \
         {CCCL_FILES} files  ({}x smaller, and it needs no toolkit to carry)",
        SHIM.len(),
        CCCL_BYTES / bytes.max(1)
    );

    if wrong != 0 {
        std::process::exit(1);
    }
}

/// `libnvrtc`'s own version, so a run of this can be read next to the
/// `header_probe` run that motivated it.
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

/// Compile `source` for `arch` against `headers` and nothing else.
///
/// Written out here rather than borrowed from `header_probe`, which is a
/// sibling example and not a library — and the difference from that one is
/// the point of this file: there, the header count is zero to ask what NVRTC
/// finds by itself; here it is the shim, to ask what the shim supplies.
///
/// The arrays come from [`as_nvrtc_arrays`], the same function the runtime's
/// compile path uses, so this measures the mechanism rather than a lookalike
/// of it. They are held in locals for the whole call because NVRTC copies
/// neither the pointers nor what they point at.
fn compile(source: &str, arch: &str, headers: &[Header]) -> Result<f64, String> {
    let (texts, names) = as_nvrtc_arrays(headers)?;
    let text_ptrs: Vec<_> = texts.iter().map(|t| t.as_ptr()).collect();
    let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();

    let src = CString::new(source).expect("no NULs in a probe");
    let unit = CString::new("cg_probe.cu").unwrap();
    let mut program: nv::nvrtcProgram = std::ptr::null_mut();
    // SAFETY: the source, the name and both arrays outlive the call, and the
    // arrays are as long as the count says.
    let code = unsafe {
        nv::nvrtcCreateProgram(
            &raw mut program,
            src.as_ptr(),
            unit.as_ptr(),
            i32::try_from(headers.len()).unwrap(),
            text_ptrs.as_ptr(),
            name_ptrs.as_ptr(),
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

/// The first line of a diagnosis that says something, so a refusal fits on a
/// row of the report — and so that the `this_grid` row shows the name error
/// itself rather than a count of errors.
fn first_line(log: &str) -> String {
    log.lines()
        .find(|line| line.contains("error") || line.contains("catastrophic"))
        .unwrap_or_else(|| log.lines().next().unwrap_or("(no log)"))
        .trim()
        .to_string()
}
