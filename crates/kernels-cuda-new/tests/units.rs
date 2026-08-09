//! Every declared unit compiles, and every row it declares resolves — or it
//! declines by name, loudly, and is counted as unverified.
//!
//! # The gate the migration is held to
//!
//! A `Unit` is a claim with three parts: that its root is device text NVRTC
//! accepts, that every `#include` in it resolves against the carried set, and
//! that each of its rows names a template instantiation the compiler can
//! actually produce a mangled symbol for. Nothing in Rust checks any of the
//! three. A row with a misspelled `template_path`, an element type the
//! template does not accept, or a `.cuh` that reaches for `<cstdint>` all
//! compile perfectly on the host and fail at the first fire — on a machine
//! with a GPU, which is the slowest place to find out.
//!
//! So this file compiles them. Every unit in [`unit::UNITS`], every row,
//! `nvrtcAddNameExpression` before the compile and `nvrtcGetLoweredName`
//! after, and a cubin at the end.
//!
//! # Why it is one test and not one per family
//!
//! It walks `UNITS`, so a family that migrates a `.cuh` and adds a `Unit` is
//! covered the moment the row lands — no test to remember to extend, and no
//! per-family probe to keep in step with a per-family module. The failure
//! message names the unit, the row and NVRTC's own diagnosis, which is what a
//! migration needs to act on.
//!
//! # The fourth part of the claim, and why it is a skip and not a failure
//!
//! `Unit` now states a toolchain floor. A unit that needs NVRTC 13.3 —
//! `moe_grouped_gemm_tile` is the finished, measured, exact one waiting on
//! this — cannot be compiled by the 13.0 this box loads, and there are only
//! three things this gate could do about it:
//!
//! * **fail**, which is what would happen today, for the whole crate, on
//!   every machine that is not on the newest toolkit. That makes a unit
//!   undeclarable until every box is upgraded, which is how finished work
//!   ends up stranded as a comment.
//! * **skip silently**, which is worse, and the rest of this header is why.
//! * **skip loudly and count it**, which is what happens below.
//!
//! # A skipped unit is UNCOMPILED, not merely unlaunched
//!
//! This is the warning that shapes the whole reporting end of this file, from
//! the agent who migrated the vision family: `unit_probe_vision` proved that
//! NVRTC only PARSES an uninstantiated template. A skipped unit is a step
//! weaker still — nothing parsed it at all. A `__tile_global__` that stopped
//! parsing after an edit would be invisible here and would surface on the
//! first 13.3 machine to run this, with the edit long since merged.
//!
//! So a skip is printed to the process's real stderr, past libtest's capture,
//! with the unit named and both versions in the line — because libtest
//! swallows a passing test's output, and a skip that is only visible under
//! `--nocapture` is a silent skip wearing a report's clothes.
//!
//! # What it does NOT check
//!
//! That the kernel computes the right thing. A cubin proves the source is
//! acceptable and the instantiation exists; it says nothing about arithmetic,
//! and `new-horizon.md` §8 is clear that a kernel body change needs its own
//! parity evidence. This gate is the floor, not the ceiling.

#![cfg(feature = "_cuda")]

use std::ffi::{CStr, CString};
use std::io::Write;

use cudarc::nvrtc::sys as nv;
use kernels_cuda_new::device::DeviceKernel;
use kernels_cuda_new::runtime::{cache, nvrtc};
use kernels_cuda_new::unit::{Demands, Toolchain, Unit};
use kernels_cuda_new::{source, unit};

/// One unit's outcome, for a report that says something when it passes too.
struct Outcome {
    unit: &'static str,
    rows: usize,
    millis: f64,
    cubin: usize,
}

/// What the gate found for one unit.
///
/// Three arms and not a `Result`, because a skip is neither: it is the third
/// answer this file exists to be able to give, and collapsing it into either
/// neighbour is exactly the mistake — folded into `Ok` it reads as a pass,
/// folded into `Err` it makes a unit undeclarable until every machine is
/// upgraded.
enum Verdict {
    /// NVRTC produced a cubin and every row got a lowered name.
    Compiled(Outcome),
    /// The unit's floor is above the NVRTC this process loaded, so it declined
    /// by name. **Nothing about this unit was checked.**
    Skipped {
        unit: &'static str,
        needs: Toolchain,
        have: Toolchain,
    },
    /// The unit should have compiled and did not.
    Failed { unit: &'static str, why: String },
}

#[test]
fn every_unit_compiles_and_every_row_resolves() {
    let Some(arch) = cache::arch() else {
        eprintln!("SKIP every_unit_compiles_and_every_row_resolves: no CUDA device is current");
        return;
    };

    let verdicts: Vec<Verdict> =
        unit::UNITS.iter().map(|unit| verdict(unit, unit.demands(), arch)).collect();
    report(&verdicts, arch);

    let failed: Vec<String> = verdicts
        .iter()
        .filter_map(|v| match v {
            Verdict::Failed { unit, why } => Some(format!("{unit}\n{why}")),
            _ => None,
        })
        .collect();
    assert!(
        failed.is_empty(),
        "{} unit(s) will not compile:\n\n{}",
        failed.len(),
        failed.join("\n\n")
    );

    // The denominator, asserted rather than assumed.
    //
    // This gate can now FILTER, and a filter that empties its own set passes
    // by having nothing left to check. The banner says so out loud, but a
    // banner is read by a person and this is read by CI, so: the walk covers
    // every unit, and at least one of them was actually compiled. A run in
    // which every declared unit skipped is a run that verified nothing, and on
    // a box where not one unit compiles this crate cannot serve a single fire
    // — reporting that as a pass is the failure mode the whole skip mechanism
    // is under suspicion of.
    assert_eq!(
        verdicts.len(),
        unit::UNITS.len(),
        "the walk dropped a unit before any of the above was decided"
    );
    let compiled = verdicts.iter().filter(|v| matches!(v, Verdict::Compiled(_))).count();
    assert!(
        compiled > 0,
        "all {} declared units skipped: NOT ONE was compiled, so this test proves nothing about \
         any of them. A green run here means the gate verified the empty set.",
        verdicts.len()
    );
}

/// **A row that states no template arguments lowers to a symbol with none,
/// and it is checked against the row that states some.**
///
/// [`every_unit_compiles_and_every_row_resolves`] already fails a row that
/// produces no lowered name, so this does not repeat that. What it adds is
/// the thing the lowered name is EVIDENCE of, which a resolution alone does
/// not distinguish: that a row stating [`DeviceKernel::PLAIN`] named a
/// `__global__` with no template parameter list, and a row stating an `elem`
/// named an instantiation of one.
///
/// The distinguishing mark is the Itanium ABI's, and it is not a heuristic:
/// a function template's mangling carries its arguments in an `I...E` bracket
/// immediately after the name, and a plain function's does not. So
/// `…5plainEPii` against `…7oneflagILb1EEEvPi` is the whole test, and it runs
/// over every declared row rather than over an example.
///
/// **Both denominators are asserted.** A tree with no plain row would make
/// the first half vacuous and a tree with no templated row would make the
/// second half vacuous, and either would pass while checking nothing — the
/// same emptiness [`every_unit_compiles_and_every_row_resolves`] guards with
/// `compiled > 0`.
#[test]
fn a_row_with_no_elem_lowers_to_a_symbol_with_no_template_arguments() {
    let Some(arch) = cache::arch() else {
        eprintln!(
            "SKIP a_row_with_no_elem_lowers_to_a_symbol_with_no_template_arguments: \
             no CUDA device is current"
        );
        return;
    };

    let mut plain = 0usize;
    let mut templated = 0usize;
    let mut units_touched = 0usize;
    for unit in unit::UNITS {
        units_touched += 1;
        let rows: Vec<&DeviceKernel> = unit.rows.iter().collect();
        let compiled = match nvrtc::compile_rows(unit, arch, &rows) {
            Ok(compiled) => compiled,
            Err(nvrtc::CompileError::Toolchain { needs, have, .. }) => {
                // The same third answer the walk above gives, for the same
                // reason: a floor this box does not meet is not a defect in
                // the row. Undo the count, so a skipped unit cannot stand in
                // for one that was checked.
                units_touched -= 1;
                eprintln!("SKIP {}: needs NVRTC {needs}, this box loads {have}", unit.name);
                continue;
            }
            Err(why) => panic!("{} does not compile: {why}", unit.name),
        };
        for row in unit.rows {
            let leaf = row.template_path.rsplit("::").next().expect("a path has a leaf");
            let (_, mangled) = compiled
                .lowered
                .iter()
                .find(|(symbol, _)| *symbol == row.sig.symbol)
                .unwrap_or_else(|| panic!("{} produced no lowered name", row.sig.symbol));
            assert!(
                mangled.contains(leaf),
                "{}: `{mangled}` does not name `{leaf}`",
                row.sig.symbol
            );
            let bracket = format!("{leaf}I");
            if row.is_plain() {
                plain += 1;
                assert!(
                    !mangled.contains(&bracket),
                    "{} states no template arguments, but NVRTC lowered it to `{mangled}`, \
                     which mangles some",
                    row.sig.symbol
                );
                assert!(
                    !row.instantiation().contains('<'),
                    "{} asked for an argument list it says it does not have",
                    row.sig.symbol
                );
            } else {
                templated += 1;
                assert!(
                    mangled.contains(&bracket),
                    "{} states `{}`, but NVRTC lowered it to `{mangled}`, which mangles no \
                     template arguments -- so the row named a plain `__global__` while \
                     claiming an element type",
                    row.sig.symbol,
                    row.elem
                );
            }
        }
    }

    assert!(units_touched > 0, "every unit skipped, so nothing above ran");
    assert_eq!(
        plain + templated,
        unit::UNITS.iter().map(|u| u.rows.len()).sum::<usize>(),
        "the walk dropped a row before deciding its shape"
    );
    assert!(
        plain > 0,
        "no row states `DeviceKernel::PLAIN`, so this test proves nothing about plain kernels"
    );
    assert!(
        templated > 0,
        "no row states an `elem`, so the contrast this test draws is between one thing \
         and nothing"
    );
    println!(
        "{units_touched} unit(s) walked: {plain} plain row(s) lowered without template \
         arguments, {templated} templated row(s) lowered with them"
    );
}

/// A floor this machine does not meet skips, and the skip says both versions.
///
/// **The floor is measured, not chosen.** No 13.3 toolkit exists on this box —
/// `/usr/local` holds 12.5 and 13.0 — so a literal `13.3` here would be a
/// number that is only true of one machine. `nvrtcVersion` is asked what is
/// loaded and the synthetic floor is one minor above it, which is above
/// whatever is loaded wherever this runs. Nothing in [`unit::UNITS`] states a
/// floor, so without a synthetic one the entire skip path would be dead code
/// that first executes on the day it matters.
///
/// Both directions, because only the pair is evidence: the same real unit,
/// with a floor this machine does not meet, is skipped and produces no cubin;
/// with the floor it does meet — the loaded version itself, which is the
/// inclusive boundary — it compiles exactly as it did before any of this
/// existed.
#[test]
fn a_floor_this_machine_does_not_meet_skips_and_names_both_versions() {
    let Some(arch) = cache::arch() else {
        eprintln!("SKIP a_floor_this_machine_does_not_meet_skips: no CUDA device is current");
        return;
    };
    let have = nvrtc::version().expect("this crate cannot compile a unit without NVRTC");
    let unreachable = Toolchain::new(have.major, have.minor + 1);
    let unit = &unit::UNITS[0];

    match verdict(unit, Demands { floor: unreachable, ..Demands::DEFAULT }, arch) {
        Verdict::Skipped { unit: named, needs, have: found } => {
            assert_eq!(named, unit.name, "a skip names the unit");
            assert_eq!(needs, unreachable);
            assert_eq!(found, have, "and reports what was found, not what was wanted");
            let line = format!("skipped, needs {needs}, have {found}");
            assert!(line.contains(&format!("needs {unreachable}")), "{line}");
            assert!(line.contains(&format!("have {have}")), "{line}");
            println!("  the line the gate prints: {} {line}", unit.name);
        }
        Verdict::Compiled(_) => panic!(
            "`{}` was compiled by NVRTC {have} while claiming to need {unreachable} -- a JIT \
             failure is a refusal, never a fallback",
            unit.name
        ),
        Verdict::Failed { why, .. } => {
            panic!("a version gap must not present as a compile failure: {why}")
        }
    }

    // The floor that IS met -- the loaded version itself, so this also pins
    // the boundary as inclusive.
    match verdict(unit, Demands { floor: have, ..Demands::DEFAULT }, arch) {
        Verdict::Compiled(outcome) => {
            assert!(outcome.cubin > 0, "a met floor compiles exactly as before");
            assert_eq!(outcome.rows, unit.rows.len());
        }
        Verdict::Skipped { needs, .. } => {
            panic!("NVRTC {have} meets a floor of {needs} and must not skip it")
        }
        Verdict::Failed { why, .. } => panic!("{why}"),
    }

    // And the crate's own compile path -- not this file's copy of it --
    // declines rather than quietly handing the source to the older compiler.
    let rows: Vec<&kernels_cuda_new::device::DeviceKernel> = unit.rows.iter().collect();
    match nvrtc::compile_under(unit, arch, &rows, unit.header_set(), unreachable) {
        Err(nvrtc::CompileError::Toolchain { unit: named, needs, have: found }) => {
            assert_eq!((named, needs, found), (unit.name, unreachable, have));
        }
        Err(other) => panic!("a version gap has its own variant, got {other:?}"),
        Ok(compiled) => panic!(
            "`{}` produced {} bytes of cubin out of a compiler it declared too old",
            unit.name,
            compiled.cubin.len()
        ),
    }
}

/// The mutation check: a unit that should have compiled and did not is a
/// FAILURE, and never a skip.
///
/// The one way this whole mechanism could be worse than nothing is by
/// converting real breakage into a quiet "skipped". So the gate is run against
/// units that are deliberately wrong, in the two shapes wrong takes here, and
/// each is required to come back `Failed`:
///
/// * a root NVRTC rejects — the migration's everyday failure;
/// * rows that name templates this root does not define — the defect
///   `every_row_is_in_the_unit_its_file_names` guards against without a GPU,
///   which surfaces here as a compile with no lowered names.
///
/// Each is checked twice: with no floor, and with a floor this machine MEETS.
/// The second is the one that matters — it proves the floor check is a gate in
/// front of the compile rather than a replacement for it.
#[test]
fn a_unit_that_should_have_compiled_and_did_not_is_a_failure_not_a_skip() {
    let Some(arch) = cache::arch() else {
        eprintln!("SKIP a_unit_that_should_have_compiled_and_did_not: no CUDA device is current");
        return;
    };
    let have = nvrtc::version().expect("NVRTC is loaded");
    let met = Demands { floor: have, ..Demands::DEFAULT };

    let real = unit::UNITS[0];
    let mutants = [
        ("a root NVRTC rejects", Unit { root: REJECTED, ..real }),
        (
            "rows the root does not define",
            Unit { rows: unit::UNITS[unit::UNITS.len() - 1].rows, ..real },
        ),
    ];

    for (what, mutant) in mutants {
        for demands in [Demands::DEFAULT, met] {
            match verdict(&mutant, demands, arch) {
                Verdict::Failed { why, .. } => {
                    assert!(!why.is_empty(), "a failure carries NVRTC's own diagnosis");
                }
                Verdict::Compiled(outcome) => panic!(
                    "{what}: `{}` compiled to {} bytes, so this mutation proves nothing -- \
                     pick a mutation NVRTC actually rejects",
                    mutant.name, outcome.cubin
                ),
                Verdict::Skipped { needs, have, .. } => panic!(
                    "{what}: reported as skipped (needs {needs}, have {have}) rather than \
                     failed -- the floor check is swallowing real breakage, which is the \
                     one way this mechanism is worse than not having it"
                ),
            }
        }
    }

    // The hard edge that used to be an `assert!` in the middle of the walk: a
    // unit with no rows is a cubin nothing can fire. Still a failure, and now
    // one that lets the other 43 units report before it is raised.
    let rowless = Unit { rows: &[], ..real };
    match verdict(&rowless, Demands::DEFAULT, arch) {
        Verdict::Failed { why, .. } => assert!(why.contains("no rows"), "{why}"),
        _ => panic!("a unit with no rows compiles to a cubin that satisfies no fire"),
    }
}

/// A root NVRTC rejects, for the mutation check.
///
/// The prelude include is real, so what is being mutated is the unit's own
/// text rather than its ability to resolve a header — a missing include and a
/// broken kernel are different failures and this file already has a test for
/// the other one.
const REJECTED: &str = "#include \"pie_device.cuh\"\n\nthis is not device text;\n";

/// Compile one unit, or decline it, and say which.
///
/// The floor is asked about through [`nvrtc::admits`] — the crate's own
/// comparison, the same one `nvrtc::compile_with` makes before it creates a
/// program. Deliberately not a `>=` written out here: a gate that decided for
/// itself what "meets the floor" means could skip a unit the compiler would
/// have accepted, and an over-eager skip is an unverified kernel that reads
/// like a pass.
///
/// `demands` is a parameter rather than read off the unit, so that the two
/// tests above can hand it a floor no declared unit states. The crate's
/// demands table is empty today — nothing declared needs more than the NVRTC
/// this box loads — so without that seam the skip path would first run on the
/// day it mattered.
fn verdict(unit: &Unit, demands: Demands, arch: &str) -> Verdict {
    match nvrtc::admits(unit.name, demands.floor) {
        Ok(()) => {}
        Err(nvrtc::CompileError::Toolchain { unit, needs, have }) => {
            return Verdict::Skipped { unit, needs, have };
        }
        Err(other) => {
            return Verdict::Failed { unit: unit.name, why: other.to_string() };
        }
    }

    // A unit with no rows would compile to a cubin nothing can fire, which
    // `nvrtc::compile` refuses for the same reason: it would be cached under
    // this architecture and satisfy nothing.
    if unit.rows.is_empty() {
        return Verdict::Failed {
            unit: unit.name,
            why: format!("unit `{}` declares no rows", unit.name),
        };
    }

    match compile(unit, arch, demands.headers.set()) {
        Ok(outcome) => Verdict::Compiled(outcome),
        Err(why) => Verdict::Failed { unit: unit.name, why },
    }
}

/// The report, and the shout.
///
/// Printed on success as well, because the numbers are the point: a unit whose
/// compile takes a second is a cold-start stall, and the only place that shows
/// up is here. The skip count is in the summary line whether or not it is
/// zero — a reader who has to notice the ABSENCE of a line has not been told
/// anything.
fn report(verdicts: &[Verdict], arch: &str) {
    let have = nvrtc::version().map_or_else(|_| "unknown".to_string(), |v| v.to_string());
    println!("\n{:<34} {:>5} {:>9} {:>10}", "unit", "rows", "compile", "cubin");
    println!("{}", "-".repeat(62));
    for verdict in verdicts {
        match verdict {
            Verdict::Compiled(outcome) => println!(
                "{:<34} {:>5} {:>7.0} ms {:>8} B",
                outcome.unit, outcome.rows, outcome.millis, outcome.cubin
            ),
            Verdict::Skipped { unit, needs, have } => {
                println!("{unit:<34} {:>5} {:>19}", "-", format!("SKIPPED needs {needs}, have {have}"));
            }
            Verdict::Failed { unit, .. } => println!("{unit:<34} {:>5} {:>19}", "-", "FAILED"),
        }
    }

    let compiled: Vec<&Outcome> = verdicts
        .iter()
        .filter_map(|v| match v {
            Verdict::Compiled(outcome) => Some(outcome),
            _ => None,
        })
        .collect();
    let skipped: Vec<(&str, Toolchain, Toolchain)> = verdicts
        .iter()
        .filter_map(|v| match v {
            Verdict::Skipped { unit, needs, have } => Some((*unit, *needs, *have)),
            _ => None,
        })
        .collect();
    println!(
        "{}\n{} units compiled, {} rows, {} bytes of cubin, for {arch} on NVRTC {have}\n\
         {} of {} units SKIPPED\n",
        "-".repeat(62),
        compiled.len(),
        compiled.iter().map(|o| o.rows).sum::<usize>(),
        compiled.iter().map(|o| o.cubin).sum::<usize>(),
        skipped.len(),
        verdicts.len(),
    );

    if !skipped.is_empty() {
        let rule = format!("!! {}", "=".repeat(72));
        let named: String = skipped
            .iter()
            .map(|(unit, needs, have)| {
                format!("!!     {unit:<40} skipped, needs {needs}, have {have}\n")
            })
            .collect();
        shout(&format!(
            "{rule}\n\
             !! {} of {} DECLARED UNITS WERE SKIPPED AND ARE THEREFORE UNVERIFIED\n\
             !! this box loads NVRTC {have}, and each of these needs a newer one:\n\
             {named}\
             !!\n\
             !! a skipped unit is UNCOMPILED, not merely unlaunched: nothing parsed\n\
             !! its source, so a kernel that stopped parsing is invisible here and\n\
             !! surfaces on the first machine with a new enough toolkit. Do not read\n\
             !! a green run of this test as covering the units listed above.\n\
             {rule}",
            skipped.len(),
            verdicts.len(),
        ));
    }
}

/// Say something libtest cannot swallow.
///
/// `println!` and `eprintln!` both go through the capture libtest installs for
/// the duration of a test, and a PASSING test's capture is thrown away — which
/// is exactly the run a skip has to be visible in. Writing to the process's
/// own stderr handle goes past that, so the line appears in a plain
/// `cargo test` with no `--nocapture` and no failure.
///
/// Used only for the skip banner. Everything else belongs in the report, where
/// it is read when someone is looking.
fn shout(what: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "{what}");
    let _ = stderr.flush();
}

/// Compile one unit for `arch`, asking for every row's instantiation, against
/// the header set the unit chose.
///
/// Deliberately not `runtime::nvrtc::compile`: this is a test of the SOURCES,
/// and going through the crate's own compiler would make a failure ambiguous
/// between a bad `.cuh` and a bad compile path. The option list is the same
/// one `runtime::nvrtc::options` builds, plus the unit's own — including
/// `--device-as-default-execution-space` where a unit states it.
///
/// The header set is a parameter for the same reason it is a parameter on
/// `compile_with`: it is the unit's choice, and a gate that compiled every
/// unit against the library set would be testing something other than what
/// `nvrtc::compile` will do.
fn compile(unit: &Unit, arch: &str, headers: &[source::Header]) -> Result<Outcome, String> {
    let (texts, names) = source::as_nvrtc_arrays(headers)?;
    let text_ptrs: Vec<*const i8> = texts.iter().map(|c| c.as_ptr()).collect();
    let name_ptrs: Vec<*const i8> = names.iter().map(|c| c.as_ptr()).collect();

    let root = CString::new(unit.root).map_err(|_| "the root contains a NUL".to_string())?;
    let file = CString::new(format!("{}.cuh", unit.name)).map_err(|_| "NUL in a name")?;

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
    let expressions: Vec<CString> = unit
        .rows
        .iter()
        .map(|row| CString::new(row.instantiation()).expect("an instantiation has no NUL"))
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
    options.extend(unit.options.iter().map(|o| CString::new(*o).expect("an option has no NUL")));
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
    // whole test exists for: the template path or the element type names
    // something the source does not define, and NVRTC says nothing about it
    // until the symbol is asked for.
    let mut unresolved: Vec<&str> = Vec::new();
    for (row, expression) in unit.rows.iter().zip(&expressions) {
        let mut lowered: *const i8 = std::ptr::null();
        // SAFETY: `program` compiled and the expression was added before it.
        let code =
            unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut lowered) };
        if code != nv::nvrtcResult::NVRTC_SUCCESS || lowered.is_null() {
            unresolved.push(row.sig.symbol);
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
    if cubin.is_empty() {
        return Err("compiled to an empty cubin".to_string());
    }
    if !log.trim().is_empty() {
        // A warning on a successful compile is where a `__CUDA_ARCH__`-guarded
        // mistake shows up, and it is the only trace of a kernel that compiles
        // clean and fires wrong.
        println!("  {} compiled with something to say:\n{log}", unit.name);
    }

    Ok(Outcome { unit: unit.name, rows: unit.rows.len(), millis, cubin: cubin.len() })
}

/// **A drifted row is REFUSED by the compiler the JIT actually uses.**
///
/// # The instrument this falsifies, and why it needed falsifying
///
/// `abi::device_typecheck` writes one `static_assert` per row, comparing the
/// operand list the row states against the type of the `__global__` it names,
/// and `Unit::source` appends that text to the root so `nvrtcCompileProgram`
/// has to agree with it. Two sentences in the portable `kernels` crate — the
/// ones under `Ty::Fp8Kind` and `Ty::KvScheme`, read by Metal, Vulkan, WGPU
/// and CPU as well as here — have claimed for a long time that the rows are
/// ASSERTED rather than assumed. Until this walk they were assumed: the
/// emitter's only caller was a hand-run example, over one row list, producing
/// a file for a compiler this tree no longer contains.
///
/// **A translation unit with no assertions in it compiles exactly like one
/// that checks everything.** So a green `every_unit_compiles_and_every_row_resolves`
/// is not evidence that anything is being checked, and this test is the
/// evidence. It is a control and a mutant:
///
/// * the REAL row must compile — otherwise a red below would only mean the
///   appendix is broken;
/// * the SAME row with one operand's `Ty` swapped must NOT compile;
/// * and the diagnosis must be OUR `static_assert`, naming the row. A "must
///   not compile" test satisfied by its own setup failing is the decoy this
///   requirement exists to rule out, so a generic NVRTC failure is not
///   accepted as a pass.
///
/// # Why `attn::write_kv_fp8_per_tensor`
///
/// It is one of the two rows in the tree whose operands name `Ty::Fp8Kind`,
/// which is the failure the whole instrument is ordered around — a
/// `cuLaunchKernel` cell one byte wide against a four-byte parameter
/// mis-marshals every argument after it, silently. It is also
/// `DeviceKernel::PLAIN`, and `PLAIN` is the string `"(no template
/// arguments)"`: before this change the emitter demanded an element type of
/// every row, found a `(` in that sentinel, and refused the whole slice — so
/// the `sizeof(__nv_fp8_interpretation_t)` assertion was generated into a
/// string that was then discarded with the error. This row being checked
/// here is the widening, measured.
///
/// # The drift
///
/// `d: I32` spelled `F32`. Both are four bytes and both cross as one 32-bit
/// cell, so nothing between the row and the launch can tell them apart. The
/// C++ can, and that is the entire argument for compiling this text.
#[test]
fn a_drifted_row_is_refused_by_the_compiler_the_jit_uses() {
    use kernels::{KernelSig, kernel, operands};

    let Some(arch) = cache::arch() else {
        eprintln!("SKIP a_drifted_row_is_refused_by_the_compiler_the_jit_uses: no CUDA device");
        return;
    };

    const SYMBOL: &str = "attn::write_kv_fp8_per_tensor";
    let (_, unit) = unit::unit_of(SYMBOL).unwrap_or_else(|| {
        panic!("`{SYMBOL}` is in no unit -- this test's subject has moved and it is now checking nothing")
    });
    let real = unit.row(SYMBOL).expect("the unit that hosts it holds it");

    // THE CONTROL. One row, so the compile that follows is about this row and
    // not about the other thirty in the unit.
    let checked = unit.typecheck(&[real]).unwrap_or_else(|why| panic!("{}: {why}", unit.name));
    assert_eq!(
        checked.checked, 1,
        "`{SYMBOL}` produced no assertion, so the mutant below cannot fail for the \
         reason this test claims:\n{:?}",
        checked.skipped
    );
    assert!(
        checked.text.contains("sizeof(::__nv_fp8_interpretation_t) == 4"),
        "the fp8 width assertion is not in the text this test compiles:\n{}",
        checked.text
    );
    match nvrtc::compile_rows(unit, arch, &[real]) {
        Ok(_) => {}
        Err(nvrtc::CompileError::Toolchain { needs, have, .. }) => {
            eprintln!("SKIP a_drifted_row_is_refused: needs NVRTC {needs}, this box loads {have}");
            return;
        }
        Err(why) => panic!(
            "the REAL `{SYMBOL}` does not compile with its own typecheck appended, so \
             every red this test could report would be the instrument's rather than a \
             row's:\n{why}\n\n{}",
            checked.text
        ),
    }

    // THE MUTANT. Everything the emitter reads is identical except `d`.
    static DRIFTED: KernelSig = kernel!(write_kv_fp8_per_tensor "attn::write_kv_fp8_per_tensor",
        file = Some("attn/kv_paged.cuh"),
        operands = operands![
            k_curr: Bf16s,
            v_curr: Bf16s,
            k_pages: U8sMut,
            v_pages: U8sMut,
            qo_indptr: U32s,
            kv_page_indices: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            r: I32,
            page_size: I32,
            h_kv: I32,
            d: F32,
            fp8_kind: Fp8Kind,
        ]);

    // Pinned to the original, so a later edit to the row cannot leave this
    // mutant quietly testing a different shape -- or nothing.
    assert_eq!(
        DRIFTED.operands.len(),
        real.sig.operands.len(),
        "the mutant and `{SYMBOL}` no longer have the same arity, which any emitter \
         would catch: the point is a same-width substitution"
    );
    let differ: Vec<&str> = real
        .sig
        .operands
        .iter()
        .zip(DRIFTED.operands)
        .filter(|(a, b)| a.ty != b.ty)
        .map(|(a, _)| a.name)
        .collect();
    assert_eq!(differ, vec!["d"], "the mutant must differ in exactly `d`");

    let mutant =
        DeviceKernel { sig: &DRIFTED, template_path: real.template_path, elem: real.elem };
    let Err(why) = nvrtc::compile_rows(unit, arch, &[&mutant]) else {
        let text = unit.typecheck(&[&mutant]).expect("emitted");
        panic!(
            "THE DRIFTED `{SYMBOL}` COMPILED. The typecheck appended to every unit is \
             not distinguishing an `int` parameter from a `float` one, which means it \
             distinguishes nothing and every green it has ever shown is a \
             decoration:\n\n{}",
            text.text
        );
    };
    let why = why.to_string();
    assert!(
        why.contains(SYMBOL) || why.contains("static assert") || why.contains("static_assert"),
        "the mutant failed, but not on the assertion this test is about -- a compile \
         that breaks for any other reason satisfies a `must not compile` test while \
         proving nothing:\n{why}"
    );
}

/// A `*mut bf16` destination, asserted as `bf16*` — and REFUSED as `f16*`.
///
/// # What is on trial
///
/// [`kernels::Ty::Bf16sMut`] and [`kernels::Ty::F16sMut`]. They were added
/// because `kernels::Ty` had `Bf16s` and `F16s` and no written halves, so
/// every bf16 OUTPUT in the tree said `Ty::BufMut`, whose `cpp()` is `void*`
/// — and `abi::self_describing` declines `void*` outright, because every
/// object pointer converts to it and an assertion against it holds for every
/// possible kernel.
///
/// # The tag has moved, and this test's copies are gone with it
///
/// This test was written around the sentence *"nothing produces the new kinds
/// yet — `x::abi`'s `ptr_abi!(bf16, …)` still tags `*mut bf16` `Ty::BufMut`
/// while already declaring its `Abi::CPP` to be the exact string
/// `Ty::Bf16sMut.cpp()` returns, so a variant added and left there is a
/// decoration: it renders, and nothing ever compiles it."* It stated the rows
/// **the way `ptr_abi!` WOULD state them** once the tag moved, because no
/// real row could.
///
/// The tag moved. `unit::rows()`'s `norm::tanh_bf16` now states `x:
/// Bf16sMut` itself, so the two `WRITES_*` copies became a SECOND POPULATION
/// — a fixture outliving the gap it stood in for, free to drift from the
/// thing it was standing in for with nothing to notice. They are deleted and
/// the controls below compile the real rows.
///
/// **The two mutants stay synthetic and always must.** A mutant is by
/// definition not a real row: if `unit::rows()` ever stated
/// `norm::tanh_bf16`'s `x` as `Ty::F16sMut`, this test would be asserting
/// that a correct row is refused.
///
/// # The count the old text carried, corrected
///
/// It said *"122 fn-world rows carry such a destination at 170 operand
/// positions, and 12 more carry an f16 one at 13; those 183 are the dominant
/// part of the fifth of operand positions the JIT's typecheck leaves
/// unasserted."* The shape of that sentence was right — rows and positions
/// are different quantities and it distinguished them — and every value in it
/// was wrong, produced by an extractor that stopped at the first `]` of an
/// operand list and so read one row per `fn`. Re-derived at `d737aad29` with
/// each row's own `where [T = …]` substituted first:
///
/// **172 rows at 269 positions** — 252 bf16 (`*mut bf16` 245,
/// `Option<NonNull<bf16>>` 7) and 17 f16 (`*mut f16` 11,
/// `Option<NonNull<f16>>` 6), six rows carrying both. Tree-wide that took
/// operand positions asserted from 1843/2207 (83%) to 2112/2207 (95%) and
/// rows fully checked from 61 to 226.
/// `tests/device_typecheck_types.rs`'s
/// `the_written_sixteen_bit_positions_are_two_hundred_and_sixty_nine`
/// re-derives 269 at run time; this is a note, not the check.
///
/// # Why this pair and not a synthetic kernel
///
/// `norm::tanh_bf16` and `norm::tanh_f16` are ONE `__global__` —
/// `template <class T> __global__ void tanh_inplace(T* __restrict__ x, int n)`
/// at `csrc/src/norm/altup_aux.cuh:189` — at two instantiations. So the
/// mutant of each is the OTHER's correct type. That is the sharpest control
/// available: `f16*` is not a nonsense spelling that a translation unit might
/// reject for some incidental reason, it is a type this very template really
/// has at the instantiation next door. A red here can only be the operand.
///
/// It also proves the two directions independently. A checker that spelled
/// every sixteen-bit destination `bf16*` would pass the first half of this
/// test and fail the second.
///
/// # And the whole unit, which is a different population
///
/// One row per compile is what makes a red attributable, and it is also a
/// population of one. The last step compiles this unit's ENTIRE row set — the
/// text the JIT itself hands `nvrtcCreateProgram` — and requires the number
/// of `bf16*` destination assertions in it to equal the number of
/// `Ty::Bf16sMut` operands the unit's rows state. A tag that rendered
/// correctly for the hand-picked row and was dropped for the rest would pass
/// everything above.
#[test]
fn a_written_bf16_is_asserted_as_bf16_by_the_jit() {
    use kernels::{KernelSig, kernel, operands};

    let Some(arch) = cache::arch() else {
        eprintln!("SKIP a_written_bf16_is_asserted_as_bf16_by_the_jit: no CUDA device");
        return;
    };

    const BF16: &str = "norm::tanh_bf16";
    const F16: &str = "norm::tanh_f16";
    const BF16_CPP: &str = "::pie_cuda_driver::kernels::device::bf16*";
    const F16_CPP: &str = "::pie_cuda_driver::kernels::device::f16*";

    // Each row claiming its SIBLING's destination format. Synthetic, and
    // permanently so -- see the header.
    static BF16_CLAIMS_F16: KernelSig = kernel!(tanh_inplace "norm::tanh_bf16",
        file = Some("norm/altup_aux.cuh"),
        operands = operands![
            x: F16sMut,
            n: I32,
        ]);
    static F16_CLAIMS_BF16: KernelSig = kernel!(tanh_inplace "norm::tanh_f16",
        file = Some("norm/altup_aux.cuh"),
        operands = operands![
            x: Bf16sMut,
            n: I32,
        ]);

    let (_, unit) = unit::unit_of(BF16).unwrap_or_else(|| {
        panic!("`{BF16}` is in no unit -- this test's subject has moved and it is now checking nothing")
    });
    assert!(
        unit.hosts(F16),
        "`{BF16}` and `{F16}` are no longer in one unit, so the two halves below \
         would be compiling different translation units and the symmetry this \
         test rests on is gone"
    );

    // THE REAL ROWS CARRY THE WRITTEN KINDS. This loop used to pin the
    // opposite -- it required the tree's own tag to still be `Ty::BufMut` and
    // said "if it is already `{want:?}`, delete this test's copy and check the
    // real row". This is that instruction, carried out: the copies are gone
    // and the assertion is inverted, so a revert to `BufMut` fails here with
    // a message rather than silently un-asserting 269 positions.
    for (symbol, want) in [(BF16, kernels::Ty::Bf16sMut), (F16, kernels::Ty::F16sMut)] {
        let real = unit.row(symbol).expect("the unit that hosts it holds it");
        assert_eq!(
            real.sig.operands.len(),
            2,
            "`{symbol}`'s arity moved; the mutants below are pinned to two operands"
        );
        assert_eq!(real.sig.operands[0].name, "x", "`{symbol}`'s first operand is not `x`");
        assert_eq!(
            real.sig.operands[0].ty, want,
            "`{symbol}`'s `x` is stated {:?}. `x::abi`'s `ptr_abi!` tag is the only \
             thing that sets it, and if it is back to `Ty::BufMut` then `void*` is \
             being asserted -- a type every object pointer converts to, so nothing \
             downstream would fail",
            real.sig.operands[0].ty
        );
    }

    // AND THE MUTANTS DIFFER FROM THEM IN EXACTLY `x`. With the copies gone
    // this is the only thing tying the synthetic sigs to the tree; without it
    // a mutant is free to drift and its refusal stops being about the operand
    // it names.
    for (sig, symbol, want) in [
        (&BF16_CLAIMS_F16, BF16, kernels::Ty::F16sMut),
        (&F16_CLAIMS_BF16, F16, kernels::Ty::Bf16sMut),
    ] {
        let real = unit.row(symbol).expect("held");
        assert_eq!(real.sig.symbol, sig.symbol, "the mutant is a different row");
        assert_eq!(real.sig.operands.len(), sig.operands.len(), "the mutant's arity drifted");
        let differ: Vec<(&str, kernels::Ty, kernels::Ty)> = real
            .sig
            .operands
            .iter()
            .zip(sig.operands)
            .inspect(|(a, b)| assert_eq!(a.name, b.name, "`{symbol}`: operands reordered"))
            .filter(|(a, b)| a.ty != b.ty)
            .map(|(a, b)| (a.name, a.ty, b.ty))
            .collect();
        assert_eq!(
            differ.len(),
            1,
            "`{symbol}`'s mutant must differ from the real row in exactly one \
             operand and differs in {differ:?}"
        );
        assert_eq!(differ[0].0, "x", "`{symbol}`'s mutant drifts the wrong operand");
        assert_eq!(differ[0].2, want, "`{symbol}`'s mutant no longer claims the sibling's kind");
    }

    // THE CONTROLS. Each REAL row under its own format, one row per compile,
    // so the result is about this row and not about the twenty others in the
    // unit.
    for (symbol, want) in [(BF16, BF16_CPP), (F16, F16_CPP)] {
        let row = unit.row(symbol).expect("held");
        let checked = unit.typecheck(&[row]).unwrap_or_else(|why| panic!("{}: {why}", unit.name));
        assert_eq!(
            checked.checked, 1,
            "`{symbol}` is not fully checked under the written kind, so the mutant \
             below cannot fail for the reason this test claims:\n{:?}",
            checked.skipped
        );
        assert_eq!(
            checked.asserted, checked.positions,
            "an operand of `{symbol}` went unasserted:\n{:?}",
            checked.skipped
        );
        assert!(
            checked.text.contains(&format!("{want}>")),
            "the destination is not asserted as `{want}`:\n{}",
            checked.text
        );
        match nvrtc::compile_rows(unit, arch, &[row]) {
            Ok(_) => {}
            Err(nvrtc::CompileError::Toolchain { needs, have, .. }) => {
                eprintln!("SKIP a_written_bf16_is_asserted: needs NVRTC {needs}, this box loads {have}");
                return;
            }
            Err(why) => panic!(
                "`{symbol}` does NOT compile with its destination asserted as \
                 `{want}`. Either `Ty::{:?}`'s C++ spelling is wrong or the kernel \
                 does not take what this test says it takes -- and until this is \
                 green the mutant's red below proves nothing:\n{why}\n\n{}",
                row.sig.operands[0].ty, checked.text
            ),
        }
    }

    // THE MUTANTS. Each row claiming the other's format.
    for (sig, symbol, sibling) in
        [(&BF16_CLAIMS_F16, BF16, F16), (&F16_CLAIMS_BF16, F16, BF16)]
    {
        let real = unit.row(symbol).expect("held");
        let row = DeviceKernel { sig, template_path: real.template_path, elem: real.elem };
        let Err(why) = nvrtc::compile_rows(unit, arch, &[&row]) else {
            let text = unit.typecheck(&[&row]).expect("emitted");
            panic!(
                "`{symbol}` COMPILED with its destination asserted as `{sibling}`'s \
                 format. The two sixteen-bit formats are the same WIDTH, so nothing \
                 else in this tree would ever have said so -- and `Ty::Bf16sMut` and \
                 `Ty::F16sMut` are two kinds instead of one `Ty::U16sMut` for exactly \
                 this reason:\n\n{}",
                text.text
            );
        };
        let why = why.to_string();
        assert!(
            why.contains(symbol) || why.contains("static assert") || why.contains("static_assert"),
            "`{symbol}` failed, but not on the assertion this test is about -- a \
             compile that breaks for any other reason satisfies a `must not compile` \
             test while proving nothing:\n{why}"
        );
    }

    // THE WHOLE UNIT, which is the population the JIT actually compiles. Every
    // row, one translation unit, and the count of destination assertions in it
    // derived from the ROWS rather than read off the text it is checking.
    let all: Vec<&DeviceKernel> = unit.rows.iter().collect();
    let want_bf16 = unit
        .rows
        .iter()
        .flat_map(|r| r.sig.operands)
        .filter(|o| o.ty == kernels::Ty::Bf16sMut)
        .count();
    assert!(
        want_bf16 > 1,
        "`{}` states {want_bf16} written-bf16 destinations, so this step is the \
         same population as the control above rather than a wider one",
        unit.name
    );
    let whole = unit.typecheck(&all).unwrap_or_else(|why| panic!("{}: {why}", unit.name));
    assert_eq!(
        whole.text.matches(&format!("{BF16_CPP}>")).count(),
        want_bf16,
        "`{}` states {want_bf16} `Ty::Bf16sMut` operands and its typecheck asserts \
         `{BF16_CPP}` at a different number of them -- a kind that renders for one \
         hand-picked row and is dropped for the rest passes every check above",
        unit.name
    );
    match nvrtc::compile_rows(unit, arch, &all) {
        Ok(_) => {}
        Err(nvrtc::CompileError::Toolchain { needs, have, .. }) => {
            eprintln!("SKIP a_written_bf16_is_asserted (whole unit): needs {needs}, have {have}");
        }
        Err(why) => panic!(
            "`{}` does not compile with all {want_bf16} of its bf16 destinations \
             asserted, though the single-row control above did:\n{why}\n\n{}",
            unit.name, whole.text
        ),
    }
}
