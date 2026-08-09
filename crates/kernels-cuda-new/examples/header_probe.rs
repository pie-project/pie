//! What does NVRTC on THIS machine resolve by itself, and what must be
//! carried?
//!
//! # The question this answers
//!
//! [`kernels_cuda_new::source`] states the rule the crate is built on — *no
//! include path on disk; includes resolve against a header set carried in the
//! binary* — and then lists what is deliberately absent from that set:
//! `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`, `mma.h`. Seven files in
//! `kernels-cuda/csrc/src` are blocked on exactly those, because they use
//! hardware instructions no prelude can restate: `__nv_cvt_fp8_*` is an
//! instruction, `wmma::fragment` is a layout the compiler knows by NAME, and
//! `__hfma2`/`__hsub2` are packed-half arithmetic.
//!
//! So the migration has a fork in it, and which branch to take depends on one
//! fact about the local NVRTC:
//!
//! * **If NVRTC resolves them itself** — recent `libnvrtc` ships the device
//!   headers inside the library, reachable with no `headers[]` entry at all —
//!   then the seven files unblock for free, and the crate stays toolkit-free
//!   at build time because nothing is embedded.
//! * **If it does not**, the headers must be carried, which means reading them
//!   from `$CUDA_HOME` at BUILD time. That is a real cost: this crate's whole
//!   claim is that it needs no toolkit to build, so carrying them has to be an
//!   opt-in feature rather than the default.
//!
//! Guessing costs an architecture. This measures.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example header_probe
//! ```

use std::ffi::{CStr, CString};

use cudarc::nvrtc::sys as nv;

/// One thing to try compiling, and why anyone cares.
struct Probe {
    /// What appears in the report.
    what: &'static str,
    /// A translation unit that uses the feature for real — a declaration alone
    /// would resolve the header and prove nothing about the instruction.
    source: &'static str,
}

const PROBES: &[Probe] = &[
    Probe {
        what: "baseline (no include at all)",
        source: "__global__ void k(float* p) { p[threadIdx.x] = 1.0f; }\n",
    },
    Probe {
        what: "cuda_fp16.h -- __hfma2/__hsub2 (dequant_fp4, dequant_wna16)",
        source: concat!(
            "#include <cuda_fp16.h>\n",
            "__global__ void k(__half2* p, __half2 a, __half2 b) {\n",
            "    p[threadIdx.x] = __hsub2(__hfma2(a, b, p[threadIdx.x]), b);\n",
            "}\n"
        ),
    },
    Probe {
        what: "cuda_bf16.h -- __nv_bfloat162 arithmetic",
        source: concat!(
            "#include <cuda_bf16.h>\n",
            "__global__ void k(__nv_bfloat162* p, __nv_bfloat162 a) {\n",
            "    p[threadIdx.x] = __hmul2(p[threadIdx.x], a);\n",
            "}\n"
        ),
    },
    Probe {
        what: "cuda_fp8.h -- __nv_cvt_fp8 (kv_paged, dequant_fp8, naive_paged)",
        source: concat!(
            "#include <cuda_fp8.h>\n",
            "__global__ void k(__nv_fp8_storage_t* out, float x) {\n",
            "    out[threadIdx.x] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);\n",
            "}\n"
        ),
    },
    Probe {
        what: "mma.h -- wmma::fragment (moe_dispatch, moe_grouped_gemm)",
        source: concat!(
            "#include <mma.h>\n",
            "using namespace nvcuda;\n",
            "__global__ void k(const __half* a, const __half* b, float* c) {\n",
            "    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;\n",
            "    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;\n",
            "    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;\n",
            "    wmma::fill_fragment(acc, 0.0f);\n",
            "    wmma::load_matrix_sync(fa, a, 16);\n",
            "    wmma::load_matrix_sync(fb, b, 16);\n",
            "    wmma::mma_sync(acc, fa, fb, acc);\n",
            "    wmma::store_matrix_sync(c, acc, 16, wmma::mem_row_major);\n",
            "}\n"
        ),
    },
    Probe {
        what: "cooperative_groups.h -- the CCCL door (flashinfer decode/prefill/mla)",
        source: concat!(
            "#include <cooperative_groups.h>\n",
            "namespace cg = cooperative_groups;\n",
            "__global__ void k(float* p) {\n",
            "    cg::thread_block block = cg::this_thread_block();\n",
            "    block.sync();\n",
            "    p[threadIdx.x] = 1.0f;\n",
            "}\n"
        ),
    },
    Probe {
        what: "cuda/std/limits -- libcu++ (flashinfer mla)",
        source: concat!(
            "#include <cuda/std/limits>\n",
            "__global__ void k(float* p) {\n",
            "    p[threadIdx.x] = cuda::std::numeric_limits<float>::infinity();\n",
            "}\n"
        ),
    },
];

fn main() {
    let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
    println!("NVRTC version: {}", version());
    println!("architecture:  {arch}\n");
    println!("Does NVRTC resolve these WITHOUT a headers[] entry?\n");

    let mut carried = Vec::new();
    for probe in PROBES {
        match compile(probe.source, arch) {
            Ok(millis) => println!("  OK       {:<62} {millis:6.0} ms", probe.what),
            Err(log) => {
                println!("  REFUSED  {:<62} {}", probe.what, first_line(&log));
                carried.push(probe.what);
            }
        }
    }

    println!();
    if carried.is_empty() {
        println!(
            "Every probe resolved. This NVRTC ships the device headers, so the\n\
             seven blocked files need no vendoring and no build-time toolkit:\n\
             they become entries in a unit table and nothing else."
        );
    } else {
        println!(
            "{} of {} must be CARRIED. This NVRTC does not ship them, so the\n\
             header set has to widen -- which means reading $CUDA_HOME at build\n\
             time, behind an opt-in feature, because this crate builds\n\
             toolkit-free by default and that has to stay true.",
            carried.len(),
            PROBES.len()
        );
    }
}

/// `libnvrtc`'s own version, which is what decides the answer above.
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
/// Empty is the whole point: a probe that supplied the header would prove the
/// file exists on this disk, which is not in question. What is in question is
/// whether NVRTC finds it with nothing pointed at it.
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

/// The first line of a diagnosis that says something, so a refusal fits on a
/// row of the report.
fn first_line(log: &str) -> String {
    log.lines()
        .find(|line| line.contains("error") || line.contains("catastrophic"))
        .unwrap_or_else(|| log.lines().next().unwrap_or("(no log)"))
        .trim()
        .to_string()
}
