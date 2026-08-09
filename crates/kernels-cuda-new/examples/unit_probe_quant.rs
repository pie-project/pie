//! Does every `quant` unit compile under NVRTC, and does every row it states
//! resolve to a symbol?
//!
//! # The question this answers
//!
//! A `Unit` is a claim with two halves, and the halves fail differently.
//! The first — *this text compiles* — fails loudly: NVRTC returns a log.
//! The second — *this instantiation names something the text defines* —
//! fails SILENTLY. `nvrtcAddNameExpression` accepts any string. A template
//! path with a typo, an element type that is not a type, a kernel that was
//! renamed in the header and not in the row: all of them compile, and the
//! only trace is that `nvrtcGetLoweredName` has nothing to hand back. The
//! row then fires at run time against a symbol that was never in the cubin.
//!
//! So this asks for every row's instantiation BEFORE the compile — the only
//! point at which NVRTC takes one — and reads every lowered name back after
//! it, before the program is destroyed. A missing one is a defect in the
//! ROW, not in the C++, and it is reported as such.
//!
//! # Why it is an example and not only a test
//!
//! `tests/units.rs` runs the same gate over every family, which is what a
//! regression gate should do and exactly the wrong shape for migrating one.
//! While `quant` was being split, the interesting output was per unit, per
//! row, with the compile log — and a test that aborts the run on another
//! family's failure hides it. This narrows the loop to the family being
//! moved and prints the numbers on success as well, because compile
//! milliseconds are a cold-start budget and the only place they appear is
//! here.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_quant
//! ```
//!
//! # The two headers with no unit
//!
//! `quant/dequant_wna16.cuh` and `quant/transcode.cuh` carry kernels no
//! `LaunchRule` fits and no `DeviceKernel::instantiation` can name — the
//! reasons are enumerated in `x::quant`. They still get compiled
//! here, as roots with no name expressions, because `build.rs` carries EVERY
//! `.cuh` under `csrc/src` into the header set: text that is carried and
//! never compiled is a `<cstdint>` waiting for whichever unit first includes
//! it. `transcode.cuh` is included by nothing today; it was still converted,
//! and this is what checks that the conversion holds.
//!
//! Requires a CUDA feature; without one the crate has no `cudarc` and this
//! prints why. That is a `#[cfg]` and not a `required-features` entry
//! because `Cargo.toml` belongs to another owner — see the migration report.

#[cfg(not(feature = "_cuda"))]
fn main() {
    eprintln!(
        "unit_probe_quant needs a CUDA runtime chosen at compile time.\n\
         Run it as: cargo run -p kernels-cuda-new --features cuda-13 \
         --example unit_probe_quant"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    probe::main();
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::ffi::{CStr, CString};

    use cudarc::nvrtc::sys as nv;
    use kernels_cuda_new::x::quant;
    use kernels_cuda_new::runtime::cache;
    use kernels_cuda_new::source;

    /// The device text `quant` carries that no unit compiles, and the header
    /// each one is named by.
    ///
    /// Compiled with an empty row set, which is why they are not `Unit`s:
    /// `tests/units.rs` refuses a unit with no rows, and it is right to —
    /// a cubin nothing can look a symbol up in would be cached under an
    /// architecture and satisfy nothing. Compiling them HERE is not the same
    /// claim; it is only "this text is still NVRTC-clean".
    const UNROWED: &[(&str, &str)] = &[
        (
            "quant/dequant_wna16",
            include_str!("../csrc/src/quant/dequant_wna16.cuh"),
        ),
        (
            "quant/transcode",
            include_str!("../csrc/src/quant/transcode.cuh"),
        ),
    ];

    /// One unit's outcome, for a report that says something when it passes.
    struct Outcome {
        unit: &'static str,
        rows: usize,
        millis: f64,
        cubin: usize,
    }

    pub fn main() {
        let Some(arch) = cache::arch() else {
            eprintln!("unit_probe_quant: no CUDA device is current, so there is no arch to compile for");
            std::process::exit(1);
        };

        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("carried headers: {}\n", source::DEVICE_HEADERS.len());

        let mut done: Vec<Outcome> = Vec::new();
        let mut failed: Vec<String> = Vec::new();

        for unit in quant::UNITS {
            let expressions: Vec<String> =
                unit.rows.iter().map(|row| row.instantiation()).collect();
            match compile(unit.name, unit.root, &expressions, unit.options, arch) {
                Ok((millis, cubin, unresolved)) if unresolved.is_empty() && cubin > 0 => {
                    done.push(Outcome {
                        unit: unit.name,
                        rows: unit.rows.len(),
                        millis,
                        cubin,
                    });
                }
                // A compile that produced no lowered name for a row is the
                // defect this file exists for, and it is named by SYMBOL --
                // the row's word, not the mangled one, because the row is
                // what has to change.
                Ok((_, cubin, unresolved)) => {
                    let names: Vec<&str> = unresolved
                        .iter()
                        .map(|&at| unit.rows[at].sig.symbol)
                        .collect();
                    failed.push(format!(
                        "{}: compiled to {cubin} B, but {} row(s) produced no lowered \
                         name -- the instantiation names something the source does not \
                         define: {}",
                        unit.name,
                        names.len(),
                        names.join(", ")
                    ));
                }
                Err(log) => failed.push(format!("{}\n{log}", unit.name)),
            }
        }

        report(&done);

        println!("carried, unrowed -- compiled for cleanliness, not for symbols:");
        for &(name, root) in UNROWED {
            match compile(name, root, &[], &[], arch) {
                Ok((millis, cubin, _)) => {
                    println!("  OK       {name:<34} {millis:>7.0} ms {cubin:>8} B");
                }
                Err(log) => {
                    println!("  REFUSED  {name:<34} {}", first_line(&log));
                    failed.push(format!("{name}\n{log}"));
                }
            }
        }
        println!();

        if failed.is_empty() {
            println!(
                "{} unit(s), {} row(s): every unit compiled and every row resolved.",
                done.len(),
                done.iter().map(|o| o.rows).sum::<usize>()
            );
        } else {
            println!("{} FAILURE(S):\n\n{}", failed.len(), failed.join("\n\n"));
            std::process::exit(1);
        }
    }

    /// The table. Printed on success too — the numbers are the point.
    fn report(done: &[Outcome]) {
        println!("{:<34} {:>5} {:>9} {:>10}", "unit", "rows", "compile", "cubin");
        println!("{}", "-".repeat(62));
        for outcome in done {
            println!(
                "{:<34} {:>5} {:>7.0} ms {:>8} B",
                outcome.unit, outcome.rows, outcome.millis, outcome.cubin
            );
        }
        println!(
            "{}\n{} units, {} rows, {} bytes of cubin\n",
            "-".repeat(62),
            done.len(),
            done.iter().map(|o| o.rows).sum::<usize>(),
            done.iter().map(|o| o.cubin).sum::<usize>()
        );
    }

    /// Compile `root` for `arch`, asking for every expression in `wanted`.
    ///
    /// Returns the wall time, the cubin size, and the INDICES of the
    /// expressions NVRTC would not name. Deliberately not
    /// `runtime::nvrtc::compile`, for the reason `tests/units.rs` gives: this
    /// is a test of the SOURCES, and going through the crate's own compiler
    /// would make a failure ambiguous between a bad `.cuh` and a bad compile
    /// path. The option list is the one `runtime::nvrtc::options` builds,
    /// plus whatever the unit states.
    fn compile(
        name: &str,
        root: &str,
        wanted: &[String],
        extra: &[&'static str],
        arch: &str,
    ) -> Result<(f64, usize, Vec<usize>), String> {
        let (texts, names) = source::as_nvrtc_arrays(source::DEVICE_HEADERS)?;
        let text_ptrs: Vec<*const i8> = texts.iter().map(|c| c.as_ptr()).collect();
        let name_ptrs: Vec<*const i8> = names.iter().map(|c| c.as_ptr()).collect();

        let root = CString::new(root).map_err(|_| "the root contains a NUL".to_string())?;
        let file = CString::new(format!("{name}.cuh")).map_err(|_| "NUL in a name")?;

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every pointer outlives the call, and the arrays are the
        // length passed with them.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                root.as_ptr(),
                file.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        // Before the compile: the only point at which NVRTC takes one.
        let expressions: Vec<CString> = wanted
            .iter()
            .map(|e| CString::new(e.as_str()).expect("an instantiation has no NUL"))
            .collect();
        for expression in &expressions {
            // SAFETY: `program` is live and the string outlives the call.
            unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };
        }

        let mut options: Vec<CString> = vec![
            CString::new(format!("--gpu-architecture={arch}")).unwrap(),
            c"-std=c++17".to_owned(),
            c"--fmad=false".to_owned(),
            c"--prec-div=true".to_owned(),
            c"--prec-sqrt=true".to_owned(),
        ];
        options.extend(extra.iter().map(|o| CString::new(*o).expect("an option has no NUL")));
        let option_ptrs: Vec<*const i8> = options.iter().map(|c| c.as_ptr()).collect();

        let started = std::time::Instant::now();
        // SAFETY: `program` is live; the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(
                program,
                i32::try_from(option_ptrs.len()).unwrap(),
                option_ptrs.as_ptr(),
            )
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;

        let log = {
            let mut size = 0;
            // SAFETY: `program` is live and `size` is a live out-parameter.
            unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
            let mut buffer = vec![0u8; size.max(1)];
            // SAFETY: the buffer is the size NVRTC just asked for.
            unsafe { nv::nvrtcGetProgramLog(program, buffer.as_mut_ptr().cast()) };
            CStr::from_bytes_until_nul(&buffer)
                .map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned())
        };

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: destroyed once, after the log has been copied out.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        let mut unresolved: Vec<usize> = Vec::new();
        for (at, expression) in expressions.iter().enumerate() {
            let mut lowered: *const i8 = std::ptr::null();
            // SAFETY: `program` compiled and the expression was added before.
            let code =
                unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut lowered) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS || lowered.is_null() {
                unresolved.push(at);
            }
        }

        let mut size = 0;
        // SAFETY: `program` compiled successfully and `size` is a live slot.
        unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        let mut cubin = vec![0u8; size];
        // SAFETY: the buffer is the size NVRTC just asked for.
        unsafe { nv::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
        // SAFETY: destroyed once, after everything has been copied out.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };

        // A warning on a SUCCESSFUL compile is where a `__CUDA_ARCH__`-guarded
        // mistake shows up, and it is the only trace of a kernel that compiles
        // clean and fires wrong. Printed, never swallowed.
        if !log.trim().is_empty() {
            println!("  {name} compiled with something to say:\n{log}");
        }

        Ok((millis, cubin.len(), unresolved))
    }

    /// `libnvrtc`'s own version, which decides what the compile even means.
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

    /// The first line of a diagnosis that says something, so a refusal fits
    /// on a row of the report.
    fn first_line(log: &str) -> String {
        log.lines()
            .find(|line| line.contains("error") || line.contains("catastrophic"))
            .unwrap_or_else(|| log.lines().next().unwrap_or("(no log)"))
            .trim()
            .to_string()
    }
}
