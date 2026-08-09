//! Does every heavy `attn` unit COMPILE under NVRTC, and does every row of it
//! come back with a mangled symbol?
//!
//! # The question this answers
//!
//! The migration's claim per unit is three things at once: the extracted
//! `.cuh` is NVRTC-clean, the row set names templates that source actually
//! holds, and the result is a cubin — not PTX the driver would JIT a second
//! time at load. Any one of the three can fail on its own and none of them is
//! visible from a host build:
//!
//! * A header that includes `<cstdint>` compiles under nvcc and is refused
//!   here, because NVRTC ships no C++ standard library. Measured: 0 of 31
//!   standard headers answered, which is why the prelude exists.
//! * A row whose `template_path` has a typo, or whose unit is the wrong one,
//!   compiles fine and then returns no lowered name — a defect that surfaces
//!   as `NoLoweredName` at first fire, on a GPU, in production.
//! * A `--gpu-architecture=compute_XY` would emit PTX and `nvrtcGetCUBIN`
//!   would refuse it. Asking for the CUBIN is how that stays true.
//!
//! So this probe asks all three per unit, and prints what each cost.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_attn_heavy
//! ```
//!
//! # Why it is its own example
//!
//! `header_probe` asks what NVRTC resolves with an EMPTY header set — a
//! question about the toolkit. This asks whether OUR units compile against
//! OUR carried set, which is a question about the migration. Two questions,
//! two exit codes: this one fails the process when a unit or a row does, so
//! it can be a gate.

//! # Gating
//!
//! The body is behind `_cuda` and the entry point is not, because
//! `cargo test -p kernels-cuda-new` with no features still BUILDS every
//! example — and an example that needs `cudarc` would break the featureless
//! build of a crate whose whole claim is that it builds without a toolkit.
//! The other probes state the same requirement as `required-features` in
//! `Cargo.toml`; this one states it in its own source, which is the same
//! contract in the file that has to honour it.

#[cfg(not(feature = "_cuda"))]
fn main() {
    println!(
        "unit_probe_attn_heavy needs NVRTC: \
         cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_attn_heavy"
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
    use kernels_cuda_new::families::attn;
    use kernels_cuda_new::source;
    use kernels_cuda_new::unit::Unit;

    /// The units this probe covers: the heavy `attn` files' own, and only those.
    ///
    /// Named explicitly rather than filtered out of `unit::UNITS`, because a
    /// filter on a name prefix would silently start covering the small half's
    /// units the moment they land — and a gate that quietly widens is a gate
    /// whose failures belong to someone else.
    fn units() -> &'static [Unit] {
        attn::UNITS_HEAVY
    }

    pub fn run() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        let (texts, names) = source::as_nvrtc_arrays(source::DEVICE_HEADERS).expect("carried headers");
        let header_texts: Vec<*const i8> = texts.iter().map(|t| t.as_ptr()).collect();
        let header_names: Vec<*const i8> = names.iter().map(|n| n.as_ptr()).collect();

        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("headers:       {} carried\n", source::DEVICE_HEADERS.len());
        println!("  {:<24} {:>5}  {:>9}  {:>11}  {}", "unit", "rows", "compile", "cubin", "result");
        println!("  {}", "-".repeat(74));

        let mut failures = Vec::new();
        for unit in units() {
            match compile(unit, arch, &header_texts, &header_names) {
                Ok(report) => {
                    println!(
                        "  {:<24} {:>5}  {:>7.0} ms  {:>9} B  every row lowered",
                        unit.name, report.rows, report.millis, report.cubin
                    );
                }
                Err(why) => {
                    println!("  {:<24} {:>5}  {:>32}", unit.name, unit.rows.len(), first_line(&why));
                    failures.push((unit.name, why));
                }
            }
        }

        println!();
        if failures.is_empty() {
            let rows: usize = units().iter().map(|unit| unit.rows.len()).sum();
            println!(
                "{} units compiled to cubin, {rows} rows asked for, {rows} lowered \
                 names returned.",
                units().len(),
            );
        } else {
            for (name, why) in &failures {
                println!("---- {name} ----\n{why}");
            }
            std::process::exit(1);
        }
    }

    /// What one unit's compile cost and produced.
    struct Report {
        rows: usize,
        millis: f64,
        cubin: usize,
    }

    /// `libnvrtc`'s own version.
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

    /// Compile one unit against the carried headers, ask for every row's mangled
    /// name, and take the cubin.
    ///
    /// The order is NVRTC's and cannot be rearranged: name expressions are added
    /// BEFORE the compile, because that is what makes the instantiation exist;
    /// lowered names are read AFTER it and BEFORE the program is destroyed,
    /// because they point into the program's own storage.
    fn compile(
        unit: &Unit,
        arch: &str,
        header_texts: &[*const i8],
        header_names: &[*const i8],
    ) -> Result<Report, String> {
        let src = CString::new(unit.root).map_err(|_| "the root contains a NUL".to_string())?;
        let name = CString::new(format!("{}.cuh", unit.name)).unwrap();
        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every pointer outlives the call, and the two header arrays are
        // the same length as the count passed with them.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                name.as_ptr(),
                i32::try_from(header_texts.len()).unwrap(),
                header_texts.as_ptr(),
                header_names.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        let wanted: Vec<CString> = unit
            .rows
            .iter()
            .map(|row| CString::new(row.instantiation()).expect("an instantiation has no NUL"))
            .collect();
        for expression in &wanted {
            // SAFETY: `program` is live and the expression outlives the call.
            let code = unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS {
                // SAFETY: destroyed exactly once, and not used after.
                unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
                return Err(format!("nvrtcAddNameExpression({expression:?}): {code:?}"));
            }
        }

        // The float contract `runtime::nvrtc::options` passes, restated: a probe
        // that compiled under different arithmetic would answer a question about
        // a cubin nothing will ever serve.
        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let mut options = vec![
            gpu.as_ptr(),
            c"-std=c++17".as_ptr(),
            c"--fmad=false".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
        ];
        let extra: Vec<CString> =
            unit.options.iter().map(|o| CString::new(*o).expect("no NUL")).collect();
        options.extend(extra.iter().map(|o| o.as_ptr()));

        let started = std::time::Instant::now();
        // SAFETY: `program` came from a successful create; every option outlives
        // the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;
        let log = log_of(program);
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        // Every row, or the unit is a lie about what it compiles.
        for (row, expression) in unit.rows.iter().zip(&wanted) {
            let mut lowered: *const i8 = std::ptr::null();
            // SAFETY: `program` is live and compiled; `lowered` borrows from it
            // and is read before the program is destroyed.
            let code =
                unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut lowered) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS || lowered.is_null() {
                // SAFETY: destroyed exactly once, and not used after.
                unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
                return Err(format!(
                    "{} asked for `{}` and NVRTC returned no lowered name ({code:?})",
                    row.sig.symbol,
                    expression.to_string_lossy()
                ));
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

        if size == 0 {
            return Err("compiled and produced no cubin -- the arch was virtual".to_string());
        }
        Ok(Report { rows: unit.rows.len(), millis, cubin: size })
    }

    /// The program log, whether or not the compile succeeded.
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

    /// The first line of a diagnosis that says something, so a refusal fits on a
    /// row of the table.
    fn first_line(log: &str) -> String {
        log.lines()
            .find(|line| line.contains("error") || line.contains("catastrophic"))
            .unwrap_or_else(|| log.lines().next().unwrap_or("(no log)"))
            .trim()
            .to_string()
    }
}
