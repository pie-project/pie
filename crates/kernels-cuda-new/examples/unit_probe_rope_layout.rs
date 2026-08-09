//! Do `rope`'s and `layout`'s units compile under NVRTC, and does every row
//! they declare come back with a lowered name?
//!
//! # The question this answers
//!
//! A `Unit` is a claim with two halves, and the halves fail independently. The
//! first is *this text compiles with no include path on disk* — the migration's
//! central bet, because NVRTC resolves `#include "pie_device.cuh"` against a
//! header set carried in the binary and against nothing else, and the
//! `stdlib_probe` measured zero of thirty-one standard headers answering. The
//! second is *every row names a template this text actually instantiates*.
//! A row can be spelled wrong in three ways that all compile — a stale
//! namespace, a template that lost its last type parameter, a symbol whose
//! header moved — and every one of them presents identically on a GPU: the
//! unit compiles, the cubin loads, and the first fire of that symbol reports
//! `NoLoweredName` from inside a decode loop.
//!
//! So this asks both, per unit, before any GPU is involved in anything but
//! `nvrtcCompileProgram`. `runtime::nvrtc::compile` is the same call
//! `runtime::cache` makes on a first fire, so a green line here is the
//! measurement that fire would have made.
//!
//! # The six headers with no unit
//!
//! `layout` split ten `.cu` files and rowed three. The other six carry device
//! text no `LaunchRule` can state — a `gridDim.y`, a 3-D grid, a
//! one-block-whatever-the-rectangle launch, transposed axes, a host alignment
//! choice — and `Unit::compile_with` refuses a unit with no instantiations,
//! because an empty cubin cached under (unit, arch) satisfies no fire and is
//! never recomputed. They are still in [`kernels_cuda_new::source::DEVICE_HEADERS`],
//! though, and a header that is carried but never compiled is a header that
//! rots: it can lose a `__device__` annotation nvcc forgives and NVRTC does
//! not — which is exactly what happened to `yarn_original_ramp_bounds` — and
//! nothing would notice until the day a rule arrived and someone added a row.
//! So they are compiled here as roots with zero name expressions, through raw
//! NVRTC, which has no such refusal.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_rope_layout
//! ```

#[cfg(not(feature = "_cuda"))]
fn main() {
    // `Cargo.toml` gives the four older probes `required-features = ["_cuda"]`
    // and this one is not in that list, so a feature-free `cargo test` -- which
    // BUILDS every example -- would fail on `use cudarc`. A `#[cfg]` pair costs
    // a `main` that does nothing; the alternative costs the one build this
    // crate most wants to keep working. See the report: one `[[example]]` entry
    // retires this.
    eprintln!(
        "unit_probe_rope_layout needs a CUDA feature: \
         cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_rope_layout"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    probe::run();
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::ffi::{CStr, CString};

    use cudarc::nvrtc::sys as nv;
    use kernels_cuda_new::runtime::nvrtc;
    use kernels_cuda_new::source::{self, DEVICE_HEADERS};
    use kernels_cuda_new::unit::Unit;

    /// The headers this family carries with no unit on them, and why each has
    /// none. Compiled as roots with no name expressions: the question is
    /// whether the TEXT still parses, not whether anything instantiates.
    const UNROWED: &[(&str, &str, &str)] = &[
        (
            "layout/envelope",
            include_str!("../csrc/src/layout/envelope.cuh"),
            "six of seven launch dim3(x, num_kv_heads) and read blockIdx.y",
        ),
        (
            "layout/gather_tokens",
            include_str!("../csrc/src/layout/gather_tokens.cuh"),
            "dim3(num_ops, 1, num_layers), plus a host stride-alignment choice",
        ),
        (
            "layout/geometry",
            include_str!("../csrc/src/layout/geometry.cuh"),
            "fits Elementwise exactly -- but the driver composes it, no fire states it",
        ),
        (
            "layout/graph_pad",
            include_str!("../csrc/src/layout/graph_pad.cuh"),
            "<<<1, padding>>> once; RouteRows would race dims.rows blocks on one CSR",
        ),
        (
            "layout/slot_ops",
            include_str!("../csrc/src/layout/slot_ops.cuh"),
            "one 2-D grid, one <<<1, 256>>> copy RouteRows would repeat per row",
        ),
        (
            "layout/split_gate_up",
            include_str!("../csrc/src/layout/split_gate_up.cuh"),
            "grid axes are the TRANSPOSE of ElementwiseRows -- a body change, not a row",
        ),
    ];

    pub fn run() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("header set:    {} carried headers\n", DEVICE_HEADERS.len());

        let units: Vec<&Unit> = kernels_cuda_new::x::rope::UNITS
            .iter()
            .chain(kernels_cuda_new::families::layout::UNITS)
            .collect();

        println!("UNITS -- every row must come back with a lowered name\n");
        println!("  {:<26} {:>5} {:>9} {:>11}  {}", "unit", "rows", "ms", "cubin B", "verdict");
        println!("  {}", "-".repeat(74));

        let mut failures = 0usize;
        let (mut rows_total, mut bytes_total) = (0usize, 0usize);
        for unit in &units {
            match nvrtc::compile(unit, arch) {
                Ok(compiled) => {
                    // The check the GPU would otherwise make at first fire:
                    // NVRTC answers `nvrtcGetLoweredName` for an expression it
                    // instantiated and refuses one it did not, and `compile`
                    // turns that refusal into `NoLoweredName` -- so a full
                    // `lowered` is the proof that every row names a template
                    // this root really has.
                    let resolved = compiled.lowered.len();
                    let ok = resolved == unit.rows.len() && !compiled.cubin.is_empty();
                    println!(
                        "  {:<26} {:>5} {:>9.0} {:>11}  {}",
                        unit.name,
                        unit.rows.len(),
                        compiled.elapsed.as_secs_f64() * 1e3,
                        compiled.cubin.len(),
                        if ok { "OK" } else { "INCOMPLETE" }
                    );
                    if !compiled.log.trim().is_empty() {
                        for line in compiled.log.lines() {
                            println!("      | {line}");
                        }
                    }
                    if ok {
                        rows_total += resolved;
                        bytes_total += compiled.cubin.len();
                    } else {
                        failures += 1;
                    }
                }
                Err(why) => {
                    println!("  {:<26} {:>5} {:>9} {:>11}  FAILED", unit.name, unit.rows.len(), "-", "-");
                    for line in why.to_string().lines().take(12) {
                        println!("      | {line}");
                    }
                    failures += 1;
                }
            }
        }
        println!(
            "  {}\n  {:<26} {:>5} {:>9} {:>11}",
            "-".repeat(74),
            "total",
            rows_total,
            "",
            bytes_total
        );

        println!("\nCARRIED, NOT ROWED -- does the text still parse with no include path?\n");
        println!("  {:<26} {:>9}  {}", "header", "ms", "why no row");
        println!("  {}", "-".repeat(96));
        for (name, root, why) in UNROWED {
            match compile_root(name, root, arch) {
                Ok(millis) => println!("  {name:<26} {millis:>9.0}  {why}"),
                Err(log) => {
                    println!("  {name:<26} {:>9}  FAILED", "-");
                    for line in log.lines().take(12) {
                        println!("      | {line}");
                    }
                    failures += 1;
                }
            }
        }

        println!();
        if failures == 0 {
            println!(
                "{} units, {rows_total} rows, {} headers with no unit: all compiled, every row lowered.",
                units.len(),
                UNROWED.len()
            );
        } else {
            println!("{failures} failed. A row that does not lower is a fire that reports NoLoweredName.");
            std::process::exit(1);
        }
    }

    /// `libnvrtc`'s own version, because a lowered name is a mangling and a
    /// mangling is a compiler's.
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

    /// Compile a root against the carried header set with NO name expressions.
    ///
    /// The one thing `runtime::nvrtc` will not do -- it refuses a compile with
    /// no instantiations, so that an empty cubin never reaches the module
    /// cache. That refusal is right for a cache and wrong for a probe, and a
    /// probe is what this is: the question here is whether a header the JIT
    /// CARRIES but no unit compiles has quietly stopped parsing.
    ///
    /// The header arrays are held for the whole call on purpose. NVRTC copies
    /// neither the pointers nor the text behind them.
    fn compile_root(name: &str, root: &str, arch: &str) -> Result<f64, String> {
        let (texts, names) = source::as_nvrtc_arrays(DEVICE_HEADERS)?;
        let text_ptrs: Vec<*const i8> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<*const i8> = names.iter().map(|n| n.as_ptr()).collect();

        let src = CString::new(root).map_err(|_| format!("{name}: a NUL in the source"))?;
        let unit_name = CString::new(name).map_err(|_| format!("{name}: a NUL in the name"))?;
        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string and both arrays outlive the call, and their
        // lengths agree with the count.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                unit_name.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        // The same flags `runtime::nvrtc::options` states, and for the same
        // reason: `--fmad=false` and the precise divide/sqrt are what make a
        // JIT-compiled kernel bit-identical to its nvcc twin, and a probe
        // compiled under looser flags would be measuring a different program.
        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let flags = [
            gpu.as_ptr(),
            c"--std=c++17".as_ptr(),
            c"--fmad=false".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
        ];

        let started = std::time::Instant::now();
        // SAFETY: `program` came from a successful create; the flags outlive it.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(flags.len()).unwrap(), flags.as_ptr())
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
}
