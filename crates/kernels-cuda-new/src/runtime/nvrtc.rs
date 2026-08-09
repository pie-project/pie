//! Layer 3's compile half: a unit's text goes in, a cubin and a mangled name
//! per row come out.
//!
//! # Why the instantiation set is a Rust table and not a C++ file
//!
//! The ahead-of-time build stated which template instantiations exist by
//! TAKING THEIR ADDRESSES in a generated translation unit, so the set lived in
//! C++: adding one meant a `.def` entry, an explicit-instantiation line, a
//! CMake regex and a rebuild of every unit that read the manifest. This states
//! the same set the other way round. `nvrtcAddNameExpression` takes an
//! instantiation as a STRING, `nvrtcGetLoweredName` answers with the mangled
//! symbol, and `cuModuleGetFunction` finds it — so a row in [`crate::device`]
//! is the whole statement, and nothing in between is a file a human wrote.
//!
//! That is the measured claim this crate exists to make good on, and this
//! module is where it is cashed: without it a row would be a description of a
//! kernel that does not exist until some C++ says it does, rather than the
//! input that brings one into being.
//!
//! # One compile per unit, many name expressions per compile
//!
//! The granularity is deliberate and it is the only tuning decision in the
//! file. A compile per KERNEL multiplies NVRTC's fixed cost — the front end
//! re-parses the same headers once per row — and a compile of the whole tree
//! is invalidated by any edit anywhere in it. A unit is the middle: the rows
//! that share a root share a compile, and an edit invalidates exactly the
//! roots that can see it, which is the same line [`crate::unit::Unit`] draws
//! for the cache key.
//!
//! # Three orderings, and only two of them are checked
//!
//! A name expression must be added BEFORE `nvrtcCompileProgram` — NVRTC
//! records the expressions and only then knows which templates to instantiate
//! and what to mangle — and a lowered name must be read AFTER it.
//! `NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION` and
//! `NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION` are how the library says
//! so, at run time, on a machine with a GPU.
//!
//! The third has no error code and no diagnostic: a lowered name must be read
//! and COPIED before the program is destroyed, because NVRTC owns those
//! strings and frees them with it. Keeping the pointer is a use-after-free
//! whose usual symptom is a symbol that merely appears to be missing from the
//! module — a nonsense mangled name is exactly what a `cuModuleGetFunction`
//! miss looks like. [`Program`] is the type that arranges it: every error path
//! below returns early, and each one would otherwise leak the program and the
//! strings it owns while the compiler happily agreed the code was fine.
//!
//! # What is deliberately not here
//!
//! No module, no cache, no launch. A compile is a pure function of a unit, an
//! architecture and a header set; `runtime::cache` owns the per-(unit, arch)
//! `OnceLock` and the loaded module. Keeping the split means an offline cache
//! builder can call [`compile`] with nothing but an `sm_XY` string — no
//! context, no device, no handle — and get the same bytes the process would.

use std::ffi::{CStr, CString, c_char};
use std::time::{Duration, Instant};

use cudarc::nvrtc::sys as nvrtc;

use crate::device::DeviceKernel;
use crate::source::Header;
use crate::unit::{Toolchain, Unit};

/// One unit, compiled.
pub struct Compiled {
    /// The image, ready for `runtime::cache` to hand `cuModuleLoadData`.
    pub cubin: Vec<u8>,
    /// `(row symbol, mangled name)`, in the order the rows were asked for.
    ///
    /// Both halves, because neither is derivable from the other: the symbol is
    /// what a trace says and the mangled name is what the module holds, and
    /// the mapping between them is exactly what NVRTC was asked for. A loader
    /// given only mangled names would have to demangle to find out which row
    /// it just resolved.
    pub lowered: Vec<(&'static str, String)>,
    /// What the compile alone cost.
    ///
    /// The compile ALONE, which is the distinction worth keeping: `runtime`'s
    /// module cache times the compile plus the load itself, because what an
    /// operator wants to know is what the first fire cost. This is the half
    /// that an offline cache builder is deciding about — whether persisting
    /// this cubin buys back more than it costs to write.
    pub elapsed: Duration,
    /// NVRTC's log, captured whether or not the compile succeeded.
    ///
    /// Empty when NVRTC had nothing to say. **Success is the interesting
    /// case**, and the reason this field exists at all: a mistake under
    /// `#if __CUDA_ARCH__ >= …` in a branch this architecture does not take
    /// compiles clean and fires wrong, and a warning on the way past is the
    /// only place it is visible. Thrown away on success, that signal only
    /// exists on a machine someone happened to be watching.
    ///
    /// Data, not a report: whether to say it out loud belongs to the caller
    /// that knows whether this compile was a cache miss or a warm-up.
    pub log: String,
}

/// Why a unit would not compile.
///
/// Its own enum rather than the crate's unified `runtime::Error`, and
/// `runtime::error`'s header states the reason from the other side: compiling
/// is a thing done ON PURPOSE, so a caller that asked for a compile wants the
/// compiler's diagnosis — which name would not resolve, which include did not
/// land — and not a launch's summary of it. `runtime::cache` renders this into
/// `Error::Compile { unit, why }` at the boundary where the reader stops being
/// someone who asked for a compile and starts being a dispatcher that fired a
/// kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// NVRTC rejected the source, or an expression naming an instantiation.
    ///
    /// Deterministic by construction, and that is a property of this crate
    /// rather than an assumption: the source is `include_str!`-ed into the
    /// binary and the rows are a static table, so nothing about the input can
    /// differ between two calls in one process. A rejection today is a
    /// rejection forever, so nothing about it is worth retrying and a negative
    /// answer is safe to remember — the same distinction
    /// `driver-cuda`'s `program::compile::FailureKind` draws, with only one
    /// side reachable from here.
    Nvrtc(String),
    /// A call failed for a reason that is not about the source — the call's
    /// name, and NVRTC's numeric result.
    ///
    /// Not "the library is missing", which is worth stating because the name
    /// suggests it: `cudarc`'s dynamic loading PANICS when `libnvrtc.so`
    /// cannot be found, before any result code exists to wrap. That is the
    /// reason layer 3 is a feature rather than a graceful fallback — a process
    /// that cannot open NVRTC cannot fire a kernel whose only existence is
    /// text, so there is nothing to degrade to.
    Driver(&'static str, i32),
    /// A row's instantiation compiled and NVRTC has no lowered name for it.
    ///
    /// A row naming a template that does not exist is caught by the compile;
    /// this is the narrower case where the expression parsed, named something,
    /// and that something was never instantiated — a template whose primary
    /// declaration is visible and whose definition is behind an `#if`, say.
    NoLoweredName {
        /// The row that named it.
        symbol: &'static str,
        /// The expression that produced nothing.
        instantiation: String,
    },
    /// The compile was refused before NVRTC was asked.
    ///
    /// Everything this file decides on its own: an empty row list, an
    /// architecture that is not real, and a NUL in text that a C string cannot
    /// carry. All of them are decided before any FFI, which is what lets them
    /// reach a caller on a machine with no CUDA at all — and what lets this
    /// file's tests assert them with no GPU.
    ///
    /// The empty case is new here, and the original's lack of it is the point.
    /// In the driver a compile with no instantiations was legal and useful — a
    /// probe asking whether the SOURCE compiles at all. In this crate the
    /// caller is `runtime::cache`, which stores what it gets under (unit,
    /// arch) for the life of the process, so an empty compile poisons the
    /// cache with a cubin that can satisfy no fire and is never recomputed.
    /// Refusing costs a probe that has other ways to ask, and prevents a
    /// failure that presents as every symbol in a unit going missing at once.
    Refused(String),
    /// This machine's NVRTC is older than the unit says it needs.
    ///
    /// **Deliberately not [`CompileError::Refused`], which means a bad
    /// architecture or an empty row list.** A version gap and a bad argument
    /// are different facts and must not share a name, and the difference is
    /// not cosmetic:
    ///
    /// * a `Refused` is a statement about the REQUEST and is true on every
    ///   machine there is. `compute_89` is not an architecture in Ohio either.
    ///   It is deterministic in the strong sense the enum's header claims, and
    ///   the fix is to change the row or the caller.
    /// * this is a statement about THIS PROCESS's compiler, and is false on
    ///   the next box. Nothing is wrong with the unit; the fix is to install a
    ///   toolkit, and the message says which one.
    ///
    /// That difference is exactly what `tests/units.rs` keys on to skip rather
    /// than fail, and it is why a caller must be able to tell them apart by
    /// matching rather than by reading a string. It also bounds what may be
    /// remembered: `runtime::cache` may cache this for the life of the process
    /// — the loaded `libnvrtc` cannot change under a running program — and
    /// **an on-disk cache must not persist it**, because the machine it
    /// describes is the one thing about a JIT that can change between runs.
    ///
    /// A refusal, never a fallback: a unit whose floor is not met is not
    /// quietly compiled by the older compiler and not silently dropped from a
    /// fire. It declines by name, saying which version it needed and which it
    /// found.
    Toolchain {
        /// The unit that declined.
        unit: &'static str,
        /// The floor it states.
        needs: Toolchain,
        /// What `nvrtcVersion` answered in this process.
        have: Toolchain,
    },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Nvrtc(log) => write!(f, "{log}"),
            CompileError::Driver(call, code) => write!(f, "{call} failed with {code}"),
            CompileError::NoLoweredName { symbol, instantiation } => write!(
                f,
                "`{symbol}` names `{instantiation}`, which NVRTC compiled and did not instantiate"
            ),
            CompileError::Refused(why) => write!(f, "{why}"),
            CompileError::Toolchain { unit, needs, have } => write!(
                f,
                "`{unit}` needs NVRTC {needs} and this process loaded {have} -- a unit \
                 whose floor this machine does not meet declines by name rather than \
                 being compiled by an older compiler"
            ),
        }
    }
}

impl std::error::Error for CompileError {}

/// The NVRTC this process loaded, as a [`Toolchain`].
///
/// `nvrtcVersion` is the only call in this file that happens BEFORE a program
/// exists, and it is the only one that asks about the machine rather than
/// about the source. Every probe in `examples/` already prints this number;
/// what was missing was a caller that acts on it.
///
/// No patch level, because there is none to have: `nvrtcVersion` fills a major
/// and a minor, and the 13.0.88 this box loads answers `(13, 0)`.
///
/// # Errors
///
/// [`CompileError::Driver`] when the call fails. Not "the library is missing" —
/// `cudarc` PANICS before any result code exists when `libnvrtc.so` cannot be
/// opened, which is the reason layer 3 is a feature rather than a fallback.
pub fn version() -> Result<Toolchain, CompileError> {
    let mut major = 0i32;
    let mut minor = 0i32;
    // SAFETY: both out-parameters are live `i32`s for the duration of the
    // call, which is the whole of `nvrtcVersion`'s contract.
    let code = unsafe { nvrtc::nvrtcVersion(&raw mut major, &raw mut minor) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Driver("nvrtcVersion", code as i32));
    }
    Ok(Toolchain::new(
        u32::try_from(major).unwrap_or(0),
        u32::try_from(minor).unwrap_or(0),
    ))
}

/// Whether the NVRTC this process loaded may compile a unit whose floor is
/// `floor`.
///
/// The one place the comparison happens, so that `compile_with` and
/// `tests/units.rs` cannot disagree about what "meets the floor" means — the
/// gate skipping a unit the compiler would in fact have accepted is a silently
/// unverified kernel, which is the failure mode the whole skip mechanism is
/// under suspicion of.
///
/// **A unit with no floor never calls `nvrtcVersion`.** That is not a
/// micro-optimisation: it is what keeps this crate's no-GPU refusal tests
/// honest — they assert an empty row list is refused on a machine with no
/// driver at all — and what keeps the 44 units declared today on exactly the
/// code path they were on before this existed.
///
/// # Errors
///
/// [`CompileError::Toolchain`] when the floor is above what is loaded, and
/// [`CompileError::Driver`] when the version cannot be asked for at all.
pub fn admits(unit: &'static str, floor: Toolchain) -> Result<(), CompileError> {
    if floor.is_any() {
        return Ok(());
    }
    let have = version()?;
    if floor.met_by(have) {
        Ok(())
    } else {
        Err(CompileError::Toolchain { unit, needs: floor, have })
    }
}

/// Compile `unit` for `arch` against the header set the unit asked for.
///
/// The whole of what `runtime::cache` calls, and deliberately the narrowest
/// signature that can serve it: everything else a compile depends on — the
/// text, the rows, the headers, the flags, the floor — is in the binary, so
/// two processes built from one tree compile one unit to one cubin.
///
/// `arch` is an `sm_XY` string. See [`options`] for why not `compute_XY`.
///
/// The header set used to be [`crate::source::DEVICE_HEADERS`], spelled here,
/// which meant a unit compiling vendored source could not go through this
/// function at all — `tests/flashinfer_decode.rs` reached for [`compile_with`]
/// instead, which no launch path calls. [`Unit::header_set`] is that decision
/// moved to where the unit can state it.
///
/// # Errors
///
/// [`CompileError`]. Two kinds, and they are not the same kind of thing:
/// [`CompileError::Toolchain`] says this machine's compiler is too old, which
/// is a fact about the machine; everything else is drift between the rows and
/// the templates, which is a fact about the binary and is worth remembering
/// rather than retrying.
pub fn compile(unit: &Unit, arch: &str) -> Result<Compiled, CompileError> {
    let rows: Vec<&DeviceKernel> = unit.rows.iter().collect();
    compile_with(unit, arch, &rows, unit.header_set())
}

/// Compile a chosen subset of a unit's rows.
///
/// The seam a test hands a row that is deliberately wrong, or one row out of a
/// unit to show that the rest of the table is not what made the compile
/// succeed. `rows` need not be the unit's own — nothing checks the
/// relationship, because the check that matters is whether NVRTC can
/// instantiate the expression against this root, and that is the answer being
/// asked for.
///
/// # Errors
///
/// As [`compile`], plus [`CompileError::Refused`] for an empty `rows`.
pub fn compile_rows(
    unit: &Unit,
    arch: &str,
    rows: &[&DeviceKernel],
) -> Result<Compiled, CompileError> {
    compile_with(unit, arch, rows, unit.header_set())
}

/// Compile against a doctored header set — the seam every test that proves
/// something about include resolution goes through.
///
/// Taking the header set as a parameter is what makes the claim in
/// [`crate::source`] testable rather than merely stated: hand this an empty
/// slice and a root with a `#include` in it, and the compile fails with
/// NVRTC's own "cannot open source file" — which is the proof that the
/// directive resolves against the set carried in the binary and NOT against
/// some `/usr/local/cuda/include` that happens to be on the machine.
///
/// The two arrays are built by [`crate::source::as_nvrtc_arrays`] rather than
/// here, and that is not merely reuse: NVRTC copies NEITHER the pointers nor
/// the text behind them, so every `CString` must outlive
/// `nvrtcCreateProgram`. A helper that returns the owned vectors makes the
/// lifetime visible in the code — the vectors are alive because this function
/// still holds them — where a helper that returned raw pointers would make it
/// a comment, and a wrong comment is how that class of bug survives review.
///
/// [`crate::unit::Unit::name`] is handed to NVRTC as the program's name, which
/// is the string its diagnostics are prefixed with. Deliberately the unit's
/// own name and not an invented `pie_norm_device.cu`: a compile error, an
/// `Error::Compile`, a cache key and a row's `KernelSig::file` then all say
/// `norm/altup_aux`, and a human reading one of them can find the others.
///
/// # Errors
///
/// [`CompileError`].
pub fn compile_with(
    unit: &Unit,
    arch: &str,
    rows: &[&DeviceKernel],
    headers: &[Header],
) -> Result<Compiled, CompileError> {
    compile_under(unit, arch, rows, headers, unit.floor())
}

/// Compile under a stated toolchain floor — the seam a test hands a floor this
/// machine cannot meet.
///
/// Exactly what [`compile_with`] is for the header set, and for the same
/// reason: the interesting case is the one no declared unit exhibits, so if
/// the only way to reach the check were to declare a unit that needs 13.3, the
/// check would be untested until the day it mattered. Handed a floor, this
/// answers the question a 13.3 box would answer, on this one.
///
/// The check is **before `nvrtcCreateProgram`** — before the source is copied,
/// before a name expression is added, before any of the work whose failure
/// would be reported as a diagnostic about the source. A version gap must
/// reach the caller as a version gap.
///
/// # Errors
///
/// As [`compile_with`], plus [`CompileError::Toolchain`] when `floor` is above
/// the NVRTC this process loaded.
pub fn compile_under(
    unit: &Unit,
    arch: &str,
    rows: &[&DeviceKernel],
    headers: &[Header],
    floor: Toolchain,
) -> Result<Compiled, CompileError> {
    // First, and before the program exists. A unit this compiler cannot
    // compile declines here rather than being handed to it and rejected in its
    // own words -- 13.0 answers `--enable-tile` with "unknown option", which
    // is a diagnostic about a flag when the fact is a missing compiler.
    admits(unit.name, floor)?;
    if rows.is_empty() {
        return Err(CompileError::Refused(format!(
            "`{}` was asked for a cubin with no instantiations in it, which would \
             be cached under this architecture and satisfy no fire",
            unit.name
        )));
    }
    let options = options(arch, unit.options)?;

    let started = Instant::now();
    let root = CString::new(unit.root)
        .map_err(|_| CompileError::Refused(format!("`{}`'s source contains a NUL", unit.name)))?;
    let name = CString::new(unit.name)
        .map_err(|_| CompileError::Refused(format!("the unit name `{}` has a NUL", unit.name)))?;

    // The header set, as the two parallel arrays NVRTC resolves quoted
    // includes against. Held for the whole call, deliberately: see this
    // function's docs.
    let (header_texts, header_names) =
        crate::source::as_nvrtc_arrays(headers).map_err(CompileError::Refused)?;
    let text_ptrs: Vec<*const c_char> = header_texts.iter().map(|t| t.as_ptr()).collect();
    let name_ptrs: Vec<*const c_char> = header_names.iter().map(|n| n.as_ptr()).collect();
    let count = i32::try_from(text_ptrs.len())
        .map_err(|_| CompileError::Refused("more headers than NVRTC can take".into()))?;

    let mut handle: nvrtc::nvrtcProgram = std::ptr::null_mut();
    // SAFETY: every string outlives the call, and the two arrays are the same
    // length -- which is the whole of `nvrtcCreateProgram`'s contract for
    // them. An in-memory virtual filesystem: nothing is read from disk, so a
    // `#include` resolves identically on a machine with a CUDA toolkit and on
    // one without.
    let code = unsafe {
        nvrtc::nvrtcCreateProgram(
            &raw mut handle,
            root.as_ptr(),
            name.as_ptr(),
            count,
            if text_ptrs.is_empty() { std::ptr::null() } else { text_ptrs.as_ptr() },
            if name_ptrs.is_empty() { std::ptr::null() } else { name_ptrs.as_ptr() },
        )
    };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Driver("nvrtcCreateProgram", code as i32));
    }
    let program = Program(handle);

    let wanted: Vec<(&'static str, CString)> = rows
        .iter()
        .map(|row| {
            let expr = CString::new(row.instantiation()).map_err(|_| {
                CompileError::Refused(format!(
                    "`{}` names an instantiation with a NUL in it",
                    row.sig.symbol
                ))
            })?;
            Ok((row.sig.symbol, expr))
        })
        .collect::<Result<_, CompileError>>()?;
    // Named BEFORE the compile, and therefore here rather than beside the
    // lookup they feed: NVRTC records the expressions and only then knows
    // which templates to instantiate and what to mangle.
    for (symbol, expr) in &wanted {
        // SAFETY: the program is live and `expr` outlives the call.
        let code = unsafe { nvrtc::nvrtcAddNameExpression(program.0, expr.as_ptr()) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
            return Err(CompileError::Nvrtc(format!(
                "`{symbol}` names `{}`, which NVRTC would not accept as an expression",
                expr.to_string_lossy()
            )));
        }
    }

    let option_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: the program is live; every option outlives the call. NVRTC takes
    // the array as `char* const*` and does not write through it.
    let code = unsafe {
        nvrtc::nvrtcCompileProgram(
            program.0,
            i32::try_from(option_ptrs.len()).expect("five options fit an i32"),
            option_ptrs.as_ptr().cast_mut(),
        )
    };
    let log = program.log();
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        let log = log.unwrap_or_else(|| "NVRTC rejected the source and offered no log".into());
        if let Some(diagnosis) = tile_header_mismatch(&log) {
            return Err(CompileError::Nvrtc(format!("{log}\n\n{diagnosis}")));
        }
        return Err(CompileError::Nvrtc(log));
    }
    let log = log.unwrap_or_default();

    // The mangled names, read while the program is still alive, and copied.
    let mut lowered = Vec::with_capacity(wanted.len());
    for (symbol, expr) in &wanted {
        let mut mangled: *const c_char = std::ptr::null();
        // SAFETY: the program compiled; `expr` is one of the expressions added
        // above; `mangled` is a live out-parameter.
        let code =
            unsafe { nvrtc::nvrtcGetLoweredName(program.0, expr.as_ptr(), &raw mut mangled) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS || mangled.is_null() {
            return Err(CompileError::NoLoweredName {
                symbol,
                instantiation: expr.to_string_lossy().into_owned(),
            });
        }
        // SAFETY: NVRTC returns a NUL-terminated string owned by the program,
        // which is still alive here. It is copied, because it is not after
        // `Drop`.
        let mangled = unsafe { CStr::from_ptr(mangled) }.to_string_lossy().into_owned();
        lowered.push((*symbol, mangled));
    }

    let mut size = 0usize;
    // SAFETY: the program compiled; `size` is a live out-parameter.
    let code = unsafe { nvrtc::nvrtcGetCUBINSize(program.0, &raw mut size) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Driver("nvrtcGetCUBINSize", code as i32));
    }
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` is exactly the size NVRTC just reported.
    let code = unsafe { nvrtc::nvrtcGetCUBIN(program.0, cubin.as_mut_ptr().cast()) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Driver("nvrtcGetCUBIN", code as i32));
    }
    unassembled_tile_ir(unit.name, &cubin)?;

    Ok(Compiled { cubin, lowered, elapsed: started.elapsed(), log })
}

/// Refuse an image that is Tile IR rather than SASS.
///
/// # The one failure this layer could not see
///
/// Every other way a unit can fail is loud: NVRTC rejects the source, or
/// `admits` declines the compiler, or a name expression resolves to nothing.
/// A TILE unit under a tile-capable NVRTC fails in none of those ways and is
/// still broken. Measured on this box with NVRTC 13.3.33 and a bf16 tile
/// `mma`:
///
/// ```text
///   nvrtcCompileProgram      rc = 0
///   nvrtcGetCUBIN            47,560 bytes, `.note.nv.tkinfo` and NO `.text`
///   cuModuleLoadData         SUCCESS
///   cuModuleGetFunction      CUDA_ERROR_NOT_FOUND
/// ```
///
/// A tile kernel does not compile to SASS. NVRTC emits **Tile IR**, and
/// something downstream has to assemble it: a driver new enough to do it at
/// load — 580.159.03 is not, and loads the image without assembling — or
/// `tileiras` over `nvrtcGetTileIR`'s output before the cubin is cached.
///
/// So without this check a tile unit compiles clean, is CACHED, loads, and
/// fails at the first launch, one layer away from anything that could explain
/// it. With it, the compile refuses and says what to install.
///
/// # Why a byte scan and not an ELF parse
///
/// This is a guard, not a loader. It answers one question — did NVRTC hand
/// back something with executable code in it — and the two section names are
/// unambiguous enough to answer it by looking. A parse would be more precise
/// about a case that does not arise: nothing else in this crate produces a
/// cubin, and an image with neither marker is left alone rather than guessed
/// at.
///
/// `.wiki/driver/new-horizon.md` §23.18 has the end-to-end transcript of the
/// path that does work, and the note beside [`crate::unit::DEMANDS`] says why
/// a [`Toolchain`] floor alone is not enough to make a tile unit sound.
fn unassembled_tile_ir(unit: &str, cubin: &[u8]) -> Result<(), CompileError> {
    let has = |needle: &[u8]| cubin.windows(needle.len()).any(|w| w == needle);
    if has(b".note.nv.tkinfo") && !has(b".text.") {
        return Err(CompileError::Refused(format!(
            "`{unit}` compiled to Tile IR, not SASS: the image carries \
             `.note.nv.tkinfo` and no `.text`, so it would load and then answer \
             `cuModuleGetFunction` with NOT_FOUND at the first launch. A tile \
             unit needs its Tile IR assembled -- `tileiras` over \
             `nvrtcGetTileIR`, with CUDA_ROOT set, or a driver new enough to \
             assemble at load. See new-horizon.md 23.18"
        )));
    }
    Ok(())
}

/// Recognise the one CuTile failure whose message names nothing that caused
/// it, and say what did.
///
/// # The trap
///
/// The tile frontend does not know what a `__nv_bfloat16` is. It learns it
/// from a marker in the RUNTIME header: CUDA 13.3's `cuda_bf16.h` tags the
/// struct `__NV_TL_BUILTIN__`, which the frontend expands to
/// `__tile_builtin__`. Under 13.0 headers that marker site does not exist, so
/// a 2-byte struct lowers as `tile<2 x i8>` and tile codegen aborts:
///
/// ```text
///   cuda_tile.h(1364): error: Internal Compiler Error (tile codegen):
///                      "Unexpected element type in tile!"
/// ```
///
/// Nothing in that message is about headers, versions, or bf16.
///
/// The pieces are four independently versioned pip wheels with no cross-check
/// between them -- `nvidia-cuda-nvcc`, `-nvrtc`, `-tileiras` and
/// `-cuda-runtime`. Only the last one carries the marker, and it is the one
/// nothing forces you to upgrade, so the default outcome of a partial upgrade
/// is a frontend that speaks tile over headers that do not.
///
/// # Why this is worth code rather than a doc
///
/// It is already a doc. `.wiki/driver/new-horizon.md` has the A/B that proves
/// it -- adding `__tile_builtin__` by hand to the 13.0 header makes the same
/// source compile -- and the analysis cost a day and a withdrawn bug report.
/// It then caught the author of that analysis a THIRD time, with the wiki
/// page open, because the ICE looks like a compiler bug and reads like one.
///
/// A message is cheaper than a memory.
///
/// # The check it recommends
///
/// `cuda_tf32.h` ships only in 13.3 and later. Its presence beside
/// `cuda_bf16.h` is a one-`ls` proxy for the marker and needs no parsing, and
/// grepping the marker itself is the exact test when the proxy is ambiguous.
fn tile_header_mismatch(log: &str) -> Option<String> {
    let ice = log.contains("Unexpected element type in tile!");
    let tile_codegen = log.contains("tile codegen");
    if !(ice || (tile_codegen && log.contains("Internal Compiler Error"))) {
        return None;
    }

    Some(
        "This is almost certainly NOT a compiler bug. It is a version skew \
         between the tile frontend and the CUDA RUNTIME headers.\n\n\
         A 16-bit type only becomes a tile element because CUDA 13.3's \
         `cuda_bf16.h` / `cuda_fp16.h` mark it `__NV_TL_BUILTIN__`, which the \
         frontend expands to `__tile_builtin__`. Under 13.0 headers that \
         marker site does not exist, the 2-byte struct lowers as `tile<2 x \
         i8>`, and tile codegen aborts with the message above -- which names \
         neither headers nor bf16.\n\n\
         Check the runtime headers on the include path, not NVRTC's version:\n\
         \n\
         \x20   ls  <include>/cuda_tf32.h          # ships only in 13.3+\n\
         \x20   grep -c __NV_TL_BUILTIN__ <include>/cuda_bf16.h   # 0 is the bug\n\
         \n\
         The four wheels version independently and nothing cross-checks them: \
         nvidia-cuda-nvcc, -nvrtc, -tileiras and -cuda-runtime. Only the last \
         carries the marker. See new-horizon.md on the 16-bit header trap."
            .to_string(),
    )
}

/// The compile options, in the order NVRTC is handed them.
///
/// # The real architecture, not a virtual one
///
/// `--gpu-architecture=sm_XY`, so NVRTC returns a cubin. `compute_XY` returns
/// PTX, which the driver JIT-compiles a SECOND time at `cuModuleLoadData` —
/// outside CUDA minor-version compatibility, so every host would have to carry
/// a driver at least as new as the toolkit that produced the PTX, and the load
/// would pay a compile the JIT was supposed to have already done. `sm_` is
/// therefore checked rather than assumed: a `compute_XY` slipping through
/// fails later and more obscurely, at `nvrtcGetCUBIN`, which reports only that
/// there is no cubin.
///
/// The prefix test and not an exact shape, because `sm_90a` and `sm_100f` are
/// real architectures — the architecture-specific and family-specific
/// variants — and a stricter pattern would refuse a GPU rather than a mistake.
///
/// # The float flags are a contract, not a tuning knob
///
/// `--fmad=false --prec-div=true --prec-sqrt=true`. Contracting a
/// multiply-add, or taking a fast reciprocal, moves a lane by more than the
/// tolerance a replay is held to and turns a tie into a different argmax
/// winner — these are what make a CPU reference and the GPU agree on a token.
///
/// They are also HALF of a fact: [`crate::unit::Unit::cache_key`] restates the
/// same three flags as a string, because a cubin compiled under different
/// arithmetic is a different ANSWER and must not be served for this key. The
/// two spellings live in two files because layer 2 may not depend on `cudarc`
/// and therefore cannot ask this function. That duplication is paid on
/// purpose, and the way it is kept honest is that the list here is written out
/// literally and checked literally by `the_options_are_the_contract` below —
/// so changing a flag is an edit that prompts changing the other.
///
/// # Errors
///
/// [`CompileError::Refused`] for an architecture that is not an `sm_XY`.
fn options(arch: &str, extra: &[&str]) -> Result<Vec<CString>, CompileError> {
    if !arch.starts_with("sm_") {
        return Err(CompileError::Refused(format!(
            "`{arch}` is not a real architecture: only `sm_XY` makes NVRTC emit SASS, \
             and a virtual `compute_XY` would hand the driver PTX to JIT a second time \
             at load"
        )));
    }
    let gpu = CString::new(format!("--gpu-architecture={arch}"))
        .map_err(|_| CompileError::Refused(format!("the architecture `{arch}` has a NUL")))?;
    let mut all = vec![
        gpu,
        c"-std=c++17".to_owned(),
        c"--fmad=false".to_owned(),
        c"--prec-div=true".to_owned(),
        c"--prec-sqrt=true".to_owned(),
    ];
    // The unit's own, appended rather than merged: NVRTC reads the list in
    // order and a later flag wins, so a unit can only ever ADD to the shared
    // contract or override it deliberately — and `Unit::cache_key` spans the
    // same strings, so an override cannot be served a cubin built without it.
    for option in extra {
        all.push(CString::new(*option).map_err(|_| {
            CompileError::Refused(format!("the option `{option}` contains a NUL"))
        })?);
    }
    Ok(all)
}

/// An NVRTC program, destroyed on the way out.
///
/// A newtype rather than a bare handle because every error path in
/// [`compile_with`] returns early, and each one of them would otherwise leak
/// the program AND the mangled-name strings it owns. The type is what makes
/// "read the lowered names before the program dies" a thing the borrow checker
/// arranges rather than a rule in a comment.
struct Program(nvrtc::nvrtcProgram);

impl Program {
    /// NVRTC's log, or `None` when it had nothing to say.
    ///
    /// Called on both paths. On failure it is the only account of what is
    /// wrong; on success it is the warnings, which is where a mistake that
    /// compiled anyway shows up. `None` rather than a placeholder string, so
    /// that the two callers can each say the thing that is true for them — a
    /// successful compile with an empty log is not "NVRTC offered no log", it
    /// is a clean compile.
    fn log(&self) -> Option<String> {
        let mut size = 0usize;
        // SAFETY: the program is live; `size` is a live out-parameter.
        if unsafe { nvrtc::nvrtcGetProgramLogSize(self.0, &raw mut size) }
            != nvrtc::nvrtcResult::NVRTC_SUCCESS
            || size <= 1
        {
            return None;
        }
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` is exactly the size NVRTC just reported.
        if unsafe { nvrtc::nvrtcGetProgramLog(self.0, buf.as_mut_ptr().cast()) }
            != nvrtc::nvrtcResult::NVRTC_SUCCESS
        {
            return Some("NVRTC has a log for this program and would not hand it over".into());
        }
        buf.pop();
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: the handle came from `nvrtcCreateProgram` and nothing else
        // holds it. Every string `nvrtcGetLoweredName` returned was copied
        // before this runs, which is what makes destroying it here safe.
        unsafe { nvrtc::nvrtcDestroyProgram(&raw mut self.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompileError, admits, compile_rows, compile_under, options, unassembled_tile_ir, version,
    };
    use crate::device::DeviceKernel;
    use crate::families::norm::ALTUP_AUX as NORM_ALTUP_AUX;
    use crate::unit::Toolchain;

    /// A compile with no rows is refused, and refused before NVRTC is
    /// touched — which is why this test runs on a machine with no GPU, no
    /// driver and no `libnvrtc.so`.
    ///
    /// A REAL unit, so that what is shown to be refused is the argument and
    /// not a fixture that was never compilable. It has rows; it is the subset
    /// that is empty, which is the shape the failure actually takes: a caller
    /// filtering rows by a predicate that matched none, then caching the
    /// result for the process and finding every symbol in the unit missing at
    /// once.
    #[test]
    fn a_compile_with_no_rows_is_refused() {
        assert!(!NORM_ALTUP_AUX.rows.is_empty(), "the unit under test is not the empty case");

        let no_rows: Vec<&DeviceKernel> = Vec::new();
        match compile_rows(&NORM_ALTUP_AUX, "sm_89", &no_rows) {
            Err(CompileError::Refused(why)) => {
                assert!(why.contains("norm/altup_aux"), "a refusal names the unit: {why}");
            }
            other => panic!("an empty row list must be refused, got {:?}", other.err()),
        }
    }

    /// A virtual architecture is refused for the same reason and in the same
    /// place: before any FFI, so the answer is the same on every machine.
    ///
    /// Left to NVRTC this is barely an error — the source compiles, and only
    /// the request for a cubin fails, reporting that there is none rather than
    /// that PTX was asked for.
    #[test]
    fn a_virtual_architecture_is_refused() {
        let rows: Vec<&DeviceKernel> = NORM_ALTUP_AUX.rows.iter().collect();
        match compile_rows(&NORM_ALTUP_AUX, "compute_89", &rows) {
            Err(CompileError::Refused(why)) => {
                assert!(why.contains("compute_89"), "a refusal names the architecture: {why}");
            }
            other => panic!("`compute_89` must be refused, got {:?}", other.err()),
        }
        assert!(options("compute_90", &[]).is_err());
        assert!(options("", &[]).is_err());
    }

    /// The options are the arithmetic contract, spelled out.
    ///
    /// Written as a literal list rather than derived from the code, so that
    /// changing a flag is an edit HERE — which is the prompt to change the
    /// float-contract string in [`crate::unit::Unit::cache_key`] too. A flag
    /// that moved silently would serve a cubin compiled under the old
    /// arithmetic for a key computed under the new, which is a wrong answer
    /// rather than a slow one.
    #[test]
    fn the_options_are_the_contract() {
        let options = options("sm_89", &[]).expect("a real architecture");
        let spelled: Vec<&str> =
            options.iter().map(|o| o.to_str().expect("options are ASCII")).collect();
        assert_eq!(
            spelled,
            [
                "--gpu-architecture=sm_89",
                "-std=c++17",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ]
        );
    }

    /// A unit's own options are appended, and the shared contract is intact
    /// in front of them.
    ///
    /// The order is the guarantee: NVRTC reads the list left to right and a
    /// later flag wins, so a unit can add to the contract or deliberately
    /// override it, and can never accidentally lose it by being listed first.
    /// `--device-as-default-execution-space` is the case — vendored upstream
    /// source needs it and nothing authored here may have it, because on our
    /// own sources it would compile an unannotated HOST helper onto the device
    /// silently instead of reporting it.
    #[test]
    fn a_units_own_options_come_after_the_shared_contract() {
        let options = options("sm_89", &["--device-as-default-execution-space"])
            .expect("a real architecture");
        let spelled: Vec<&str> =
            options.iter().map(|o| o.to_str().expect("options are ASCII")).collect();
        assert_eq!(spelled[..5], [
            "--gpu-architecture=sm_89",
            "-std=c++17",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ]);
        assert_eq!(spelled[5], "--device-as-default-execution-space");
        assert_eq!(spelled.len(), 6);
    }

    /// The architecture-specific variants are architectures, not mistakes.
    ///
    /// `sm_90a` is how a Hopper WGMMA kernel is compiled at all, so a check
    /// that insisted on digits after `sm_` would refuse the hardware this
    /// crate most wants to reach.
    #[test]
    fn architecture_specific_variants_are_accepted() {
        for arch in ["sm_80", "sm_90a", "sm_100f", "sm_120"] {
            let options = options(arch, &[]).unwrap_or_else(|e| panic!("{arch} is real: {e}"));
            assert_eq!(options[0].to_str().expect("ASCII"), format!("--gpu-architecture={arch}"));
        }
    }

    /// A floor this machine does not meet declines by name, and says both
    /// numbers.
    ///
    /// The floor is derived from what `nvrtcVersion` actually answers rather
    /// than written down: this box loads 13.0 and no 13.3 toolkit exists on
    /// it, so a literal `13.3` would be a number chosen to be true of one
    /// machine. `have.minor + 1` is above whatever is loaded, wherever this
    /// runs.
    #[test]
    fn a_floor_above_the_loaded_nvrtc_declines_by_name() {
        let have = version().expect("this crate cannot compile without NVRTC anyway");
        let unreachable = Toolchain::new(have.major, have.minor + 1);

        match admits("norm/altup_aux", unreachable) {
            Err(CompileError::Toolchain { unit, needs, have: found }) => {
                assert_eq!(unit, "norm/altup_aux", "a decline names the unit");
                assert_eq!(needs, unreachable);
                assert_eq!(found, have, "and reports what it found, not what it wanted");
            }
            other => panic!("a floor above the loaded NVRTC must decline, got {other:?}"),
        }

        // The floor that IS met, and the boundary: `met_by` is inclusive, so
        // the version loaded compiles a unit that asks for exactly it.
        assert!(admits("norm/altup_aux", have).is_ok(), "the loaded version meets its own floor");
        assert!(admits("norm/altup_aux", Toolchain::ANY).is_ok());
    }

    /// A version gap is not a [`CompileError::Refused`], and the compile path
    /// declines before it creates a program.
    ///
    /// The distinction is the whole point of the variant: `Refused` means the
    /// request is wrong on every machine, this means the machine is wrong for
    /// the request. A caller that had to tell them apart by reading a string
    /// would be one string edit from treating a missing toolkit as a broken
    /// row.
    #[test]
    fn a_version_gap_is_not_a_refusal() {
        let have = version().expect("NVRTC is loaded");
        let rows: Vec<&DeviceKernel> = NORM_ALTUP_AUX.rows.iter().collect();
        let unreachable = Toolchain::new(have.major + 1, 0);

        let why = declined(
            compile_under(&NORM_ALTUP_AUX, "sm_89", &rows, NORM_ALTUP_AUX.header_set(), unreachable),
            "a unit whose floor is not met is not compiled by an older compiler",
        );
        assert!(
            matches!(why, CompileError::Toolchain { .. }),
            "a version gap has its own variant: {why:?}"
        );
        assert!(!matches!(why, CompileError::Refused(_)), "and is not a bad-argument refusal");

        let rendered = why.to_string();
        assert!(rendered.contains("norm/altup_aux"), "{rendered}");
        assert!(rendered.contains(&unreachable.to_string()), "which version it needed: {rendered}");
        assert!(rendered.contains(&have.to_string()), "and which it found: {rendered}");

        // The neighbouring failure, for contrast: same unit, same machine, a
        // bad architecture -- and a different variant with no version in it.
        let refused = declined(
            compile_under(
                &NORM_ALTUP_AUX,
                "compute_89",
                &rows,
                NORM_ALTUP_AUX.header_set(),
                Toolchain::ANY,
            ),
            "a virtual architecture is refused",
        );
        assert!(matches!(refused, CompileError::Refused(_)), "{refused:?}");

        // And the ordering: a floor that is not met wins over an empty row
        // list, because it is checked before anything else -- including
        // before the program exists.
        let none: Vec<&DeviceKernel> = Vec::new();
        let first = declined(
            compile_under(&NORM_ALTUP_AUX, "sm_89", &none, NORM_ALTUP_AUX.header_set(), unreachable),
            "both are wrong",
        );
        assert!(matches!(first, CompileError::Toolchain { .. }), "{first:?}");
    }

    /// The error out of a compile that must not have produced a cubin.
    ///
    /// `Result::expect_err` wants `Compiled: Debug`, and `Compiled` holds a
    /// cubin -- a `Debug` on it would print a megabyte of bytes at the one
    /// moment a reader wants a sentence.
    fn declined(result: Result<super::Compiled, CompileError>, why: &str) -> CompileError {
        match result {
            Err(error) => error,
            Ok(compiled) => panic!("{why}, and instead it produced {} bytes", compiled.cubin.len()),
        }
    }

    /// The version this process loaded is a real one, and asking twice gives
    /// the same answer.
    ///
    /// A weak assertion on purpose — the number is the machine's, and a test
    /// that hard-coded 13.0 would fail on the 13.3 box this whole seam exists
    /// to prepare for. What is worth checking is that the query works and is
    /// not a placeholder: a `(0, 0)` would silently meet every floor.
    #[test]
    fn the_loaded_nvrtc_reports_a_version() {
        let have = version().expect("NVRTC is loaded");
        assert_eq!(have, version().expect("and stays loaded"));
        assert!(have.major >= 11, "NVRTC has reported a major version since 7.0: {have}");
        assert!(!have.is_any(), "`any` is the absence of a floor, never a version");
        println!("nvrtcVersion says {have}");
    }

    /// The Tile IR guard, against images shaped like the two NVRTC produces.
    ///
    /// Not synthetic in the part that matters: the section names are the ones
    /// `cuobjdump -elf` prints for a real tile cubin (`.note.nv.tkinfo`, no
    /// `.text`) and for a real SASS one (`.text.<mangled>`), measured on this
    /// box with NVRTC 13.3.33.
    #[test]
    fn tile_ir_is_refused_and_sass_is_not() {
        // What `nvrtcGetCUBIN` returns for a `__tile_global__`: a note section
        // and no code. It loads. It has no entry point.
        let tile_ir = b"\x7fELF...\x00.note.nv.tkinfo\x00.nv.info\x00".to_vec();
        let err = unassembled_tile_ir("moe/moe_grouped_gemm_tile", &tile_ir)
            .expect_err("an image with tkinfo and no .text must be refused");
        let said = format!("{err:?}");
        assert!(
            said.contains("Tile IR") && said.contains("tileiras"),
            "the refusal has to name what happened and what to install: {said}"
        );

        // What it returns for every unit this crate declares today.
        let sass = b"\x7fELF...\x00.text._ZN3pie6kernel1kEv\x00.nv.info\x00".to_vec();
        assert!(
            unassembled_tile_ir("norm/rmsnorm", &sass).is_ok(),
            "an ordinary cubin must pass untouched"
        );

        // And the AOT-assembled form, which carries BOTH -- `nvcc --tilecubin`
        // output has `.note.nv.tkinfo` next to real `.text`. It runs, so it
        // must not be refused.
        let assembled = b"\x7fELF\x00.note.nv.tkinfo\x00.text._ZN1kEv\x00".to_vec();
        assert!(
            unassembled_tile_ir("moe/moe_grouped_gemm_tile", &assembled).is_ok(),
            "tkinfo WITH .text is an assembled tile cubin and this box runs those"
        );
    }
}

#[cfg(test)]
mod tile_header_trap {
    use super::tile_header_mismatch;

    /// The real message, copied from a failing build on this box: nvcc 13.3.73
    /// with `nvidia-cuda-runtime` still at 13.0 headers, compiling the tree's
    /// own `norm/rmsnorm_tile.cuh` -- a kernel known to be correct, which is
    /// what makes the ICE so convincing.
    const REAL_ICE: &str = r#"/opt/cu13/include/crt/cuda_tile.h(1364): error: Internal Compiler Error (tile codegen): "Unexpected element type in tile!"
Compilation aborted."#;

    #[test]
    fn the_ice_gets_a_cause_attached() {
        let d = tile_header_mismatch(REAL_ICE).expect(
            "the bf16 tile-codegen ICE must be recognised -- it is the one              CuTile failure whose message names nothing that caused it",
        );

        // The cause, in the words that make it searchable.
        assert!(d.contains("__NV_TL_BUILTIN__"), "the marker must be named");
        assert!(
            d.contains("NOT a compiler bug"),
            "the message must say this plainly. A day and a withdrawn bug              report went into learning it, and the ICE reads like a compiler              bug to everyone who meets it"
        );

        // And the check, which is the part that ends the debugging session.
        assert!(
            d.contains("cuda_tf32.h"),
            "the message must give the one-`ls` proxy; a version number does              not answer the question because the RUNTIME wheel is the one              that matters and it versions independently"
        );
        assert!(
            d.contains("grep -c __NV_TL_BUILTIN__"),
            "and the exact test for when the proxy is ambiguous"
        );
    }

    /// Each recognition branch, pinned on its own.
    ///
    /// The first version of this test used only the real transcript, which
    /// carries BOTH signatures -- so deleting either branch left it passing.
    /// A gate that survives the removal of what it guards is not a gate, and
    /// these two inputs are exactly the discriminating ones.
    #[test]
    fn each_branch_is_pinned_separately() {
        // The message alone, as a future release might phrase it with no
        // "Internal Compiler Error" banner.
        assert!(
            tile_header_mismatch("error: \"Unexpected element type in tile!\"").is_some(),
            "the element-type message must be recognised on its own -- it is \
             the specific symptom of an unmarked 16-bit type and the banner \
             around it is not load-bearing"
        );

        // And the banner alone, for a tile codegen ICE whose detail text
        // changes. Same cause, worth the same pointer.
        assert!(
            tile_header_mismatch(
                "cuda_tile.h(902): error: Internal Compiler Error (tile codegen): \"???\""
            )
            .is_some(),
            "a tile-codegen ICE with different detail text must still get the \
             pointer; the header skew is by far the likeliest cause of any of \
             them and the message costs nothing when it is wrong"
        );
    }

    /// The diagnosis must not fire on ordinary source errors. A wrong
    /// suggestion is worse than none -- it sends the reader to reinstall
    /// wheels over a typo.
    #[test]
    fn ordinary_failures_are_left_alone() {
        for log in [
            r#"kernel.cu(12): error: identifier "foo" is undefined"#,
            r#"kernel.cu(3): error: no instance of function template "cuda::tiles::store" matches the argument list"#,
            r#"cuda_tile.h(55): error: #error "This file needs C++20 features""#,
            "",
        ] {
            assert!(
                tile_header_mismatch(log).is_none(),
                "the header-trap diagnosis fired on an unrelated failure: {log}"
            );
        }
    }

    /// The C++20 case above is deliberately in that list and deserves saying
    /// out loud: it is a REAL and common tile misconfiguration, and it already
    /// says exactly what to do. Attaching a header-version essay to a message
    /// that is already actionable would make this guard noise.
    #[test]
    fn a_self_explaining_error_is_not_decorated() {
        let clear = r#"cuda_tile.h(55): error: #error "This file needs C++20 features. Please compile with c++20 or later dialect""#;
        assert!(
            tile_header_mismatch(clear).is_none(),
            "an error that already names its own fix must be left alone"
        );
    }
}
