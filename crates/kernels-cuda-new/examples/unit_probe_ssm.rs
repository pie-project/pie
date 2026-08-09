//! Does every `ssm` unit compile under NVRTC, and does every row resolve?
//!
//! # The question this answers
//!
//! A `Unit` claims two things a table cannot check: that its root compiles at
//! run time against the header set this crate carries, and that every
//! [`kernels_cuda_new::device::DeviceKernel`] row in it names a template that
//! root actually holds. Both are claims about NVRTC, and `cargo test` cannot
//! make them — the crate's own suite runs with no GPU and no `libnvrtc`, so
//! its unit tests check spelling and agreement and stop there.
//!
//! That gap is where the migration's worst failure lives, and it has a name.
//! `537294a7a` moved `norm/altup_aux`'s six `__global__`s into a header, and
//! `altup_aux.cu` kept its own copy — two definitions, both compiling, each
//! right for whichever half its tests exercised, for a whole release.
//! Compilation is the only thing that catches a row whose template path is a
//! typo, whose element type has no `Elem` specialisation, or whose header
//! reaches for a standard include NVRTC does not ship. So this probe compiles.
//!
//! It does NOT fire anything. A cubin and a lowered name prove the row can be
//! reached; they say nothing about whether its `LaunchRule` computes the grid
//! the C++ launcher did, or whether `Args::bind` can marshal its operands.
//! Those are separate measurements and this example does not claim them.
//!
//! # What it prints
//!
//! One row per unit: how many instantiations were asked for, how many came
//! back with a lowered name, how long `nvrtcCompileProgram` took, and how big
//! the cubin is. A unit that compiles but loses a name is a FAILURE and says
//! so — NVRTC answers `NVRTC_ERROR_INVALID_INPUT` from `nvrtcGetLoweredName`
//! for an expression it never instantiated, and it does that silently at
//! compile time.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_ssm
//! ```
//!
//! # Why the whole file is behind a `cfg`
//!
//! `cargo test -p kernels-cuda-new` with no features builds every example,
//! and `cudarc` is only in the dependency graph under `_cuda`. The clean fix
//! is a `[[example]]` entry with `required-features = ["_cuda"]` in
//! `Cargo.toml`; this file cannot add one, so it gates itself and leaves a
//! `main` that explains the absence rather than a link error that does not.

#[cfg(feature = "_cuda")]
fn main() {
    imp::main();
}

#[cfg(not(feature = "_cuda"))]
fn main() {
    println!(
        "unit_probe_ssm needs NVRTC. Re-run with `--features cuda-13` (or \
         `cuda-12`): the probe's whole content is a compile, and there is \
         nothing to report without a compiler."
    );
}

#[cfg(feature = "_cuda")]
mod imp {
    use std::ffi::{CStr, CString};

    use cudarc::nvrtc::sys as nv;
    use kernels_cuda_new::source::{self, DEVICE_HEADERS};
    use kernels_cuda_new::unit::Unit;

    /// What one unit's compile produced.
    struct Report {
        rows: usize,
        /// Rows that came back from `nvrtcGetLoweredName`. Anything less than
        /// `rows` is a row naming a template its unit does not hold.
        lowered: usize,
        millis: f64,
        cubin_bytes: usize,
    }

    pub fn main() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("headers:       {} carried\n", DEVICE_HEADERS.len());

        // `families::ssm::UNITS` stood here; §5 step 5 took the family into
        // fn-world and the five units are `unit!`-generated now. Same five
        // `Unit`s, same texts, same rows — only the file they are written in
        // changed, which is what makes this probe still meaningful.
        let units = kernels_cuda_new::x::ssm::UNITS;
        println!(
            "{:<28} {:>5} {:>8} {:>10} {:>12}",
            "unit", "rows", "lowered", "ms", "cubin bytes"
        );
        println!("{}", "-".repeat(68));

        let mut failed = 0usize;
        for unit in units {
            match compile(unit, arch) {
                Ok(report) => {
                    let ok = report.lowered == report.rows && report.cubin_bytes > 0;
                    if !ok {
                        failed += 1;
                    }
                    println!(
                        "{:<28} {:>5} {:>8} {:>10.0} {:>12}  {}",
                        unit.name,
                        report.rows,
                        report.lowered,
                        report.millis,
                        report.cubin_bytes,
                        if ok { "OK" } else { "MISSING NAMES" }
                    );
                }
                Err(log) => {
                    failed += 1;
                    println!(
                        "{:<28} {:>5} {:>8} {:>10} {:>12}  FAILED",
                        unit.name,
                        unit.rows.len(),
                        0,
                        "-",
                        "-"
                    );
                    for line in log.lines().take(12) {
                        println!("    {line}");
                    }
                }
            }
        }

        println!();
        let rows: usize = units.iter().map(|u| u.rows.len()).sum();
        if failed == 0 {
            println!(
                "{} units, {rows} rows: every unit compiled and every row got a \
                 lowered name.\nThe cubins are proof the templates exist and \
                 instantiate; they are NOT proof\nthat any row's LaunchRule \
                 reproduces its C++ launcher's grid.",
                units.len()
            );
        } else {
            println!("{failed} of {} units did not answer.", units.len());
            std::process::exit(1);
        }
    }

    /// Compile one unit exactly as `runtime::nvrtc` does — same headers, same
    /// float flags, same name expressions.
    ///
    /// The flags are copied rather than called because `runtime::nvrtc`'s
    /// compile is private to the fire path. They are not decoration:
    /// `--fmad=false`, `--prec-div=true` and `--prec-sqrt=true` are in
    /// `Unit::cache_key` because the arithmetic a cubin was built under is
    /// part of what it answers, and a probe compiling without them would be
    /// measuring a different cubin than the one the driver serves.
    fn compile(unit: &Unit, arch: &str) -> Result<Report, String> {
        let (texts, names) = source::as_nvrtc_arrays(DEVICE_HEADERS)?;
        let text_ptrs: Vec<*const i8> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<*const i8> = names.iter().map(|n| n.as_ptr()).collect();

        let src = CString::new(unit.root).map_err(|_| "a NUL in the root".to_string())?;
        let root_name = CString::new(format!("{}.cuh", unit.name)).unwrap();
        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call, and the two arrays are
        // `DEVICE_HEADERS.len()` long, which is the count passed.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                root_name.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        // One name expression per row, added BEFORE the compile: NVRTC only
        // instantiates a template it was asked for by name, and a row added
        // after the fact gets no mangled symbol and no code.
        let instantiations = unit.instantiations();
        let expressions: Vec<CString> = instantiations
            .iter()
            .map(|i| CString::new(i.as_str()).expect("no NULs in an instantiation"))
            .collect();
        for expression in &expressions {
            // SAFETY: `program` is live and the string outlives the call.
            let code = unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS {
                return Err(format!("nvrtcAddNameExpression: {code:?}"));
            }
        }

        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let mut options = vec![
            gpu.as_ptr(),
            c"-std=c++17".as_ptr(),
            c"--fmad=false".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
        ];
        let extra: Vec<CString> = unit
            .options
            .iter()
            .map(|o| CString::new(*o).expect("no NULs in an option"))
            .collect();
        options.extend(extra.iter().map(|o| o.as_ptr()));

        let started = std::time::Instant::now();
        // SAFETY: `program` came from a successful create; the options outlive it.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            let log = log_of(program);
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        // The lowered names, one per row. Asked for BEFORE the program is
        // destroyed, because NVRTC owns the strings it returns.
        let mut lowered = 0usize;
        for (at, expression) in expressions.iter().enumerate() {
            let mut name: *const i8 = std::ptr::null();
            // SAFETY: `program` is live and compiled; `name` is an out-parameter
            // NVRTC fills with a pointer it owns.
            let code =
                unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut name) };
            if code == nv::nvrtcResult::NVRTC_SUCCESS && !name.is_null() {
                lowered += 1;
            } else {
                println!("    row {at} has no lowered name: {}", instantiations[at]);
            }
        }

        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        let mut cubin = vec![0u8; size.max(1)];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked for.
        unsafe { nv::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
        // SAFETY: destroyed exactly once, and not used after.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };

        Ok(Report { rows: unit.rows.len(), lowered, millis, cubin_bytes: size })
    }

    /// The program log, as a `String`.
    fn log_of(program: nv::nvrtcProgram) -> String {
        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        let mut log = vec![0u8; size.max(1)];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked for.
        unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
        CStr::from_bytes_until_nul(&log)
            .map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned())
    }

    /// `libnvrtc`'s own version, so a report says which compiler answered.
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
}
