//! Does the small half of `attn` compile under NVRTC, and does every row it
//! states resolve to a symbol?
//!
//! # The question this answers
//!
//! A [`kernels_cuda_new::unit::Unit`] is three claims, and Rust checks none of
//! them: that the root is device text NVRTC accepts, that every `#include` in
//! it resolves against the carried set, and that each row names an
//! instantiation the compiler will actually produce a mangled symbol for. A
//! misspelled `template_path`, an element type the template does not take, or
//! a `.cuh` that reaches for `<cstdint>` all compile perfectly on the host and
//! fail at the first fire — on a machine with a GPU, which is the slowest
//! place to find out.
//!
//! `tests/units.rs` is the permanent gate and walks every unit in the crate.
//! This example is the MIGRATION's gate: it walks the small half of `attn`
//! alone, prints a per-unit table, and additionally compiles the two `.cuh`
//! files that were split but got no rows — `attn/head_dim_pad` and
//! `attn/split_packed` — because the claim being made about them is precise
//! and worth measuring. They are NVRTC-clean; what they lack is a launch rule
//! this backend evaluates, and the difference between *"the device text is not
//! ready"* and *"the rule is not ported"* is the difference between a
//! migration that is stuck and one that is waiting.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_attn_small
//! ```
//!
//! # Why this file carries a `cfg` fence
//!
//! Every example in this crate that touches `cudarc` is declared in
//! `Cargo.toml` with `required-features = ["_cuda"]`, because `cargo test`
//! with no features builds every example and an example naming
//! [`kernels_cuda_new::runtime`] does not exist in a feature-free build. This
//! one has no such entry: the migration that wrote it owns three files and
//! `Cargo.toml` is not one of them, and two agents editing the manifest at
//! once is a merge conflict in the one file that has to parse for anything to
//! build at all. `unit_probe_norm.rs` reached the same fence for the same
//! reason -- the manifest entry is the fix, and deleting both fences is what
//! landing it looks like.

#[cfg(not(feature = "_cuda"))]
fn main() {
    eprintln!(
        "unit_probe_attn_small asks NVRTC to compile things and needs a CUDA backend:\n  \
         cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_attn_small"
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
    use kernels_cuda_new::families::attn;
    use kernels_cuda_new::runtime::cache;
    use kernels_cuda_new::{source, unit};

    /// One compile's outcome, for a report that says something when it passes.
    struct Outcome {
        unit: &'static str,
        rows: usize,
        millis: f64,
        cubin: usize,
    }

    /// A `.cuh` that was split out but states no rows, and the rule it is waiting
    /// for.
    ///
    /// Compiled with ZERO name expressions. A template nothing instantiates is
    /// still PARSED, so this proves the text is NVRTC-acceptable — the whole of
    /// the migration except the row.
    struct Ruleless {
        name: &'static str,
        root: &'static str,
        waiting_on: &'static str,
    }

    const RULELESS: &[Ruleless] = &[
        Ruleless {
            name: "attn/head_dim_pad",
            root: include_str!("../csrc/src/attn/head_dim_pad.cuh"),
            waiting_on: "LaunchRule::PerHead",
        },
        Ruleless {
            name: "attn/split_packed",
            root: include_str!("../csrc/src/attn/split_packed.cuh"),
            waiting_on: "LaunchRule::SplitPacked",
        },
    ];

    pub fn main() {
        let Some(arch) = cache::arch() else {
            eprintln!("no CUDA device is current -- nothing to compile for");
            std::process::exit(1);
        };
        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("headers:       {} carried\n", source::DEVICE_HEADERS.len());

        let mut done: Vec<Outcome> = Vec::new();
        let mut failed: Vec<String> = Vec::new();

        for u in attn::UNITS_SMALL {
            match compile(u.name, u.root, &u.instantiations(), u.options, arch) {
                Ok(outcome) => done.push(outcome),
                Err(why) => failed.push(format!("{}\n{why}", u.name)),
            }
        }

        println!("{:<26} {:>5} {:>10} {:>11}", "unit", "rows", "compile", "cubin");
        println!("{}", "-".repeat(55));
        for outcome in &done {
            println!(
                "{:<26} {:>5} {:>7.0} ms {:>9} B",
                outcome.unit, outcome.rows, outcome.millis, outcome.cubin
            );
        }
        println!("{}", "-".repeat(55));
        println!(
            "{} units, {} rows, {} bytes of cubin\n",
            done.len(),
            done.iter().map(|o| o.rows).sum::<usize>(),
            done.iter().map(|o| o.cubin).sum::<usize>()
        );

        println!("split, NVRTC-clean, waiting on a launch rule -- no rows, no unit:\n");
        for r in RULELESS {
            match compile(r.name, r.root, &[], &[], arch) {
                Ok(outcome) => println!(
                    "  OK       {:<26} {:>7.0} ms   waiting on {}",
                    outcome.unit, outcome.millis, r.waiting_on
                ),
                Err(why) => failed.push(format!("{}\n{why}", r.name)),
            }
        }

        // Every row of the small half must resolve through the crate's own
        // lookup, not merely through the table this example read. A unit that
        // compiles and whose symbol `unit_of` cannot find is a kernel with no way
        // to be fired, and the failure would otherwise surface as "unknown
        // kernel" at the first statement that names it.
        println!("\nrow lookup:");
        for u in attn::UNITS_SMALL {
            for row in u.rows {
                match unit::unit_of(row.sig.symbol) {
                    Some((_, found)) if found.name == u.name => {}
                    Some((_, found)) => failed
                        .push(format!("{} resolves to `{}`, not `{}`", row.sig.symbol, found.name, u.name)),
                    None => failed.push(format!("{} resolves to no unit", row.sig.symbol)),
                }
                println!("  {:<34} -> {}", row.sig.symbol, row.instantiation());
            }
        }

        if failed.is_empty() {
            println!("\nall {} rows resolve.", attn::UNITS_SMALL.iter().map(|u| u.rows.len()).sum::<usize>());
        } else {
            println!("\n{} failure(s):\n\n{}", failed.len(), failed.join("\n\n"));
            std::process::exit(1);
        }
    }

    /// `libnvrtc`'s own version — the compiler whose answers these are.
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

    /// Compile one root against the carried headers, asking for `instantiations`.
    ///
    /// Deliberately not `runtime::nvrtc::compile`, for `tests/units.rs`' reason:
    /// this is a test of the SOURCES, and going through the crate's own compiler
    /// would make a failure ambiguous between a bad `.cuh` and a bad compile path.
    /// The option list is the same one `runtime::nvrtc::options` builds.
    fn compile(
        name: &'static str,
        root: &str,
        instantiations: &[String],
        extra: &[&str],
        arch: &str,
    ) -> Result<Outcome, String> {
        let (texts, names) = source::as_nvrtc_arrays(source::DEVICE_HEADERS)?;
        let text_ptrs: Vec<*const i8> = texts.iter().map(|c| c.as_ptr()).collect();
        let name_ptrs: Vec<*const i8> = names.iter().map(|c| c.as_ptr()).collect();

        let root = CString::new(root).map_err(|_| "the root contains a NUL".to_string())?;
        let file = CString::new(format!("{name}.cuh")).map_err(|_| "NUL in a name")?;

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every pointer outlives the call, and the arrays are the length
        // passed with them.
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

        // Every instantiation, BEFORE the compile — that is the only point at
        // which NVRTC will accept one, and the lowered name is only readable
        // after the compile and before the program is destroyed.
        let expressions: Vec<CString> = instantiations
            .iter()
            .map(|i| CString::new(i.as_str()).expect("an instantiation has no NUL"))
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
            // SAFETY: destroyed exactly once, after the log has been copied out.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        // A row that compiled but produced no lowered name is the defect this
        // whole probe exists for: the template path or the element type names
        // something the source does not define, and NVRTC says nothing about it
        // until the symbol is asked for.
        let mut unresolved: Vec<String> = Vec::new();
        for expression in &expressions {
            let mut lowered: *const i8 = std::ptr::null();
            // SAFETY: `program` compiled and the expression was added before it.
            let code =
                unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut lowered) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS || lowered.is_null() {
                unresolved.push(expression.to_string_lossy().into_owned());
            }
        }

        let mut size = 0;
        // SAFETY: `program` compiled successfully and `size` is a live slot.
        unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        let mut cubin = vec![0u8; size];
        // SAFETY: the buffer is the size NVRTC just asked for.
        unsafe { nv::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
        // SAFETY: destroyed exactly once, after everything has been copied out.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };

        if !unresolved.is_empty() {
            return Err(format!(
                "compiled, but {} row(s) produced no lowered name -- the instantiation \
                 names something the source does not define: {}",
                unresolved.len(),
                unresolved.join(", ")
            ));
        }
        if cubin.is_empty() && !expressions.is_empty() {
            return Err("compiled to an empty cubin".to_string());
        }
        if !log.trim().is_empty() {
            // A warning on a successful compile is where a `__CUDA_ARCH__`-guarded
            // mistake shows up, and it is the only trace of a kernel that compiles
            // clean and fires wrong.
            println!("  {name} compiled with something to say:\n{log}");
        }

        Ok(Outcome { unit: name, rows: expressions.len(), millis, cubin: cubin.len() })
    }
}
