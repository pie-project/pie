//! Tier A: the templates, compiled at run time and instantiated by name.
//!
//! The offline path (`abi::emit_device_typecheck`, compiled by nvcc) states
//! its instantiation set by taking addresses. This is the same set stated the
//! other way: `nvrtcAddNameExpression` takes an instantiation as a STRING,
//! `nvrtcGetLoweredName` answers with the mangled symbol, and
//! `cuModuleGetFunction` finds it. Nothing in between is a file a human
//! wrote.
//!
//! # Why the header is a Rust string
//!
//! [`SOURCE`] is `include_str!` of `csrc/src/norm/altup_aux.cuh`, so the
//! template lives in the BINARY rather than on disk. Two consequences, and
//! the second is the one that matters:
//!
//! * The driver has no path to get wrong, no install layout to agree with,
//!   and no way to load a header that does not match the rows it was built
//!   with — the two are in the same binary or neither is.
//! * `#include` is not available, so the header may not have one. That is
//!   already this tree's rule for anything NVRTC sees:
//!   `program::compile`'s own docs say *"a `#include` appearing in an emitted
//!   source is a bug in the emitter, and the right place to find that out is
//!   a compile error rather than a search path that silently resolves it
//!   against whatever CUDA toolkit is installed."* `altup_aux.cuh` is the
//!   first AUTHORED source held to it, which is why it spells its own bf16
//!   conversions instead of including `cuda_bf16.h`.
//!
//! It is also what keeps a toolkit-free RUN: reaching for `cuda_bf16.h` here
//! would put the CUDA headers on the critical path of every process that
//! fires a kernel.
//!
//! # The compile is cached, because it is not free
//!
//! One NVRTC compile of the whole family, once per (source, architecture) —
//! not one per kernel and not one per fire. [`Family::compile`] reports what
//! it cost so a caller can decide whether to persist the cubin the way
//! `program::cache` already persists PTIR's.

use std::ffi::{CStr, CString};

use cudarc::driver::sys as dr;
use cudarc::nvrtc::sys as nvrtc;
use kernels::KernelSig;
use kernels_cuda_new::device::{ALTUP_AUX as ENTRIES, DeviceKernel};

use super::device::{Error, KernelModule};
use super::headers;

/// The templates themselves, carried in the binary.
///
/// The path is relative to this file, which is what makes a header that moves
/// a compile error here rather than a missing file at run time.
pub const SOURCE: &str = include_str!("../../../kernels-cuda-new/csrc/src/norm/altup_aux.cuh");

/// The name NVRTC gives the program in its diagnostics.
const UNIT: &str = "pie_norm_device.cu";

/// The float flags, as one string, for [`Family::cache_key`].
///
/// Spelled beside the flags themselves so a change to one is visibly a change
/// to the other: these decide the arithmetic, so a cubin compiled under
/// different ones is a different answer and must not be served for this key.
const FLOAT_CONTRACT: &str = "fmad=false,prec-div=true,prec-sqrt=true";

/// Why a family could not be compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// NVRTC rejected the source, or an expression naming an instantiation.
    ///
    /// Deterministic by construction: the source is in the binary and the
    /// expressions are in the table, so a rejection today is a rejection
    /// forever. Nothing about it is worth retrying — the same distinction
    /// `program::compile::FailureKind` draws, with only one side reachable.
    Nvrtc(String),
    /// The library could not be loaded, or a call failed for a reason that
    /// is not about the source.
    Driver(&'static str, i32),
    /// A row's instantiation compiled and NVRTC has no lowered name for it.
    ///
    /// A row naming a template that does not exist is caught by the compile;
    /// this is the narrower case where the expression parsed, named
    /// something, and that something was not instantiated.
    NoLoweredName {
        /// The row that named it.
        symbol: &'static str,
        /// The expression that produced nothing.
        instantiation: String,
    },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Nvrtc(log) => write!(f, "{log}"),
            CompileError::Driver(call, code) => write!(f, "{call} failed with {code}"),
            CompileError::NoLoweredName {
                symbol,
                instantiation,
            } => write!(
                f,
                "`{symbol}` names `{instantiation}`, which NVRTC compiled and did not instantiate"
            ),
        }
    }
}

impl std::error::Error for CompileError {}

/// A compiled family: the cubin, and the mangled name of every row's
/// instantiation.
pub struct Compiled {
    /// The image, ready for [`KernelModule::load_mangled`].
    pub cubin: Vec<u8>,
    /// `(row symbol, mangled name)`, in the table's order.
    pub lowered: Vec<(&'static str, String)>,
    /// What the compile cost, for a caller deciding whether to cache it.
    pub elapsed: std::time::Duration,
}

/// One compilable unit: a source, the rows that instantiate it, and the name
/// NVRTC calls it in a diagnostic.
///
/// **A family is data now.** Tier A had one, so `Family` was a unit struct
/// with `SOURCE` and `ENTRIES` reached through `use` — which made the second
/// family a copy of the first's associated functions rather than a row in a
/// table. Stage C is four more families; each is a `Unit` and nothing else.
///
/// The three fields are exactly what `compile_unit` takes and no more. A unit
/// that needed a fact not listed here would be a signal the fact belongs in
/// [`headers::DEVICE_HEADERS`] — which every unit shares, because a shared
/// device header is shared across families or it is not shared at all.
#[derive(Clone, Copy)]
pub struct Unit {
    /// The name NVRTC gives this program in its diagnostics.
    pub unit: &'static str,
    /// The templates, carried in the binary.
    pub source: &'static str,
    /// The instantiations wanted out of it.
    pub rows: &'static [DeviceKernel],
}

impl Unit {
    /// Compile this unit for `arch`, instantiating exactly what its rows name.
    ///
    /// # Errors
    ///
    /// [`CompileError`]. A failure here is drift between the rows and the
    /// templates, not a condition a machine can be in.
    pub fn compile(&self, arch: &str) -> Result<Compiled, CompileError> {
        self.compile_rows(arch, self.rows)
    }

    /// [`Unit::compile`] over an arbitrary row list, so a test can hand it a
    /// row that is deliberately wrong.
    ///
    /// # Errors
    ///
    /// As [`Unit::compile`].
    pub fn compile_rows(
        &self,
        arch: &str,
        rows: &[DeviceKernel],
    ) -> Result<Compiled, CompileError> {
        self.compile_with(arch, rows, headers::DEVICE_HEADERS)
    }

    /// [`Unit::compile_rows`] against an arbitrary header set, so a test can
    /// take the header array away and watch the include fail to resolve.
    ///
    /// # Errors
    ///
    /// As [`Unit::compile`].
    pub fn compile_with(
        &self,
        arch: &str,
        rows: &[DeviceKernel],
        header_set: &[headers::Header],
    ) -> Result<Compiled, CompileError> {
        let wanted: Vec<(&'static str, String)> = rows
            .iter()
            .map(|k| (k.sig.symbol, k.instantiation()))
            .collect();
        compile_unit(arch, self.unit, self.source, &wanted, header_set)
    }

    /// Compile for `arch` and load the result.
    ///
    /// # Errors
    ///
    /// [`CompileError`] if the templates and the rows disagree, or [`Error`]
    /// if the image will not load.
    pub fn load(&self, arch: &str) -> Result<(KernelModule, std::time::Duration), FamilyError> {
        let compiled = self.compile(arch).map_err(FamilyError::Compile)?;
        let sigs: Vec<&'static KernelSig> = self.rows.iter().map(|k| k.sig).collect();
        let module = KernelModule::load_mangled(&compiled.cubin, &sigs, &compiled.lowered)
            .map_err(FamilyError::Load)?;
        Ok((module, compiled.elapsed))
    }

    /// The key a compiled unit may be cached under.
    ///
    /// **Everything that can change the cubin is in it, and that is the whole
    /// specification.** `program::cache`'s own header records what the
    /// alternative costs — a cubin keyed on less than what produced it is
    /// served after the thing it was not keyed on changes — and
    /// `driver-metal/src/program/cache.rs` keys its pipelines on the RESOLVED
    /// text for the same reason.
    ///
    /// Since Stage B the resolved text is no longer one file. NVRTC resolves
    /// `#include "pie_device.cuh"` against
    /// [`headers::DEVICE_HEADERS`], so an edit to a header changes what
    /// compiles while leaving [`Unit::source`] byte-identical — which is
    /// exactly the shape of a stale-cache bug.
    ///
    /// Four components, each because it can move on its own: the unit, the
    /// architecture (a cubin is per-`sm_XY`), the float flags — §6.5 of
    /// `new-horizon.md` calls them a contract, and `--fmad=false` is what
    /// keeps a reduction's last bit — and the header digest, on top of
    /// [`cache::disk_key`]'s own fingerprint of the source.
    #[must_use]
    pub fn cache_key(&self, arch: &str) -> String {
        self.cache_key_with(arch, headers::DEVICE_HEADERS)
    }

    /// [`Unit::cache_key`] over an arbitrary header set, so a test can show
    /// that changing a header changes the key.
    #[must_use]
    pub fn cache_key_with(&self, arch: &str, header_set: &[headers::Header]) -> String {
        let identity = format!(
            "tier-a/{}/{arch}/{FLOAT_CONTRACT}/h{:016x}",
            self.unit,
            headers::digest(header_set)
        );
        crate::program::cache::disk_key(&identity, self.source)
    }
}

/// Every unit the driver compiles at run time.
///
/// Adding a family is adding a line here, which is the point: the alternative
/// is four copies of `Unit`'s methods with different constants in them.
pub static UNITS: &[Unit] = &[NORM, NORM_ELEMENTWISE];

/// `norm`'s AltUp auxiliaries.
pub const NORM: Unit = Unit {
    unit: "pie_norm_device.cu",
    source: SOURCE,
    rows: ENTRIES,
};

/// `norm`'s pointwise pair: `residual_add` and `scalar_mul`.
///
/// The second unit, and the first one that is a LINE rather than a module.
/// Its launchers were `(n + 255) / 256` blocks of 256 and an empty-`n` guard,
/// which is `LaunchRule::Elementwise` — so nothing about them was ported;
/// the rule states what they stated and the C++ is gone.
pub const NORM_ELEMENTWISE: Unit = Unit {
    unit: "pie_norm_elementwise.cu",
    source: headers::sources::NORM_ELEMENTWISE,
    rows: kernels_cuda_new::device::ELEMENTWISE,
};

/// The `norm` device family, as the free functions Tier A wrote.
///
/// [`NORM`] is the unit; this is the spelling `tier_a_pilot` and the rest of
/// Tier A already use. Every method forwards, so there is one implementation
/// and the older name is a view of it rather than a second copy.
pub struct Family;

impl Family {
    /// The rows this family compiles.
    #[must_use]
    pub fn rows() -> &'static [DeviceKernel] {
        NORM.rows
    }

    /// [`Unit::compile`] for [`NORM`].
    ///
    /// # Errors
    ///
    /// [`CompileError`].
    pub fn compile(arch: &str) -> Result<Compiled, CompileError> {
        NORM.compile(arch)
    }

    /// [`Unit::compile_rows`] for [`NORM`].
    ///
    /// # Errors
    ///
    /// [`CompileError`].
    pub fn compile_rows(arch: &str, rows: &[DeviceKernel]) -> Result<Compiled, CompileError> {
        NORM.compile_rows(arch, rows)
    }

    /// [`Unit::compile_with`] for [`NORM`].
    ///
    /// # Errors
    ///
    /// [`CompileError`].
    pub fn compile_with(
        arch: &str,
        rows: &[DeviceKernel],
        header_set: &[headers::Header],
    ) -> Result<Compiled, CompileError> {
        NORM.compile_with(arch, rows, header_set)
    }

    /// [`Unit::load`] for [`NORM`].
    ///
    /// # Errors
    ///
    /// [`FamilyError`].
    pub fn load(arch: &str) -> Result<(KernelModule, std::time::Duration), FamilyError> {
        NORM.load(arch)
    }

    /// [`Unit::cache_key`] for [`NORM`].
    #[must_use]
    pub fn cache_key(arch: &str) -> String {
        NORM.cache_key(arch)
    }

    /// [`Unit::cache_key_with`] for [`NORM`].
    #[must_use]
    pub fn cache_key_with(arch: &str, header_set: &[headers::Header]) -> String {
        NORM.cache_key_with(arch, header_set)
    }
}

/// One NVRTC compile: a source, the instantiations wanted out of it, and the
/// headers its includes resolve against.
///
/// **The family is a caller, not a concept here.** This was `Family::compile`
/// with `SOURCE` and `ENTRIES` spelled inline, which made a second family a
/// copy of forty lines of FFI rather than a [`Unit`] and a string.
///
/// Granularity is §6.4's: **one compile per unit, many name expressions per
/// compile.** Per-kernel compiles multiply NVRTC's fixed cost; whole-tree
/// compiles invalidate on any edit.
///
/// `instantiations` is `(row symbol, expression)`, and the symbol is only
/// carried so a refusal can name the row that asked. Passing none is legal
/// and is what a probe does when the question is whether the SOURCE compiles
/// at all.
///
/// # Errors
///
/// [`CompileError`].
pub fn compile_unit(
    arch: &str,
    unit_name: &str,
    source_text: &str,
    instantiations: &[(&'static str, String)],
    header_set: &[headers::Header],
) -> Result<Compiled, CompileError> {
    let started = std::time::Instant::now();
    let source = CString::new(source_text)
        .map_err(|_| CompileError::Nvrtc("the template source contains a NUL".into()))?;
    let unit = CString::new(unit_name)
        .map_err(|_| CompileError::Nvrtc("the unit name contains a NUL".into()))?;

    let mut program: nvrtc::nvrtcProgram = std::ptr::null_mut();
    // The header set, as the two parallel arrays NVRTC resolves quoted
    // includes against. Held for the whole call: NVRTC copies neither the
    // pointers nor what they point at.
    let (header_texts, header_names) =
        headers::as_nvrtc_arrays(header_set).map_err(CompileError::Nvrtc)?;
    let header_ptrs: Vec<*const std::ffi::c_char> =
        header_texts.iter().map(|t| t.as_ptr()).collect();
    let name_ptrs: Vec<*const std::ffi::c_char> = header_names.iter().map(|n| n.as_ptr()).collect();
    let count = i32::try_from(header_ptrs.len())
        .map_err(|_| CompileError::Nvrtc("more headers than NVRTC can take".into()))?;
    // SAFETY: every string outlives the call, and the two arrays are the same
    // length -- which is the whole of `nvrtcCreateProgram`'s contract for
    // them. An in-memory virtual filesystem: nothing is read from disk, so a
    // `#include` resolves identically on a machine with a CUDA toolkit and on
    // one without.
    let code = unsafe {
        nvrtc::nvrtcCreateProgram(
            &raw mut program,
            source.as_ptr(),
            unit.as_ptr(),
            count,
            if header_ptrs.is_empty() {
                std::ptr::null()
            } else {
                header_ptrs.as_ptr()
            },
            if name_ptrs.is_empty() {
                std::ptr::null()
            } else {
                name_ptrs.as_ptr()
            },
        )
    };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Driver("nvrtcCreateProgram", code as i32));
    }
    let program = Program(program);

    // The instantiations, named BEFORE the compile: NVRTC records each
    // expression and only then knows which templates to instantiate and what
    // to mangle. Adding one afterwards is accepted and answers nothing, which
    // is why this loop is here rather than beside the lookup it feeds.
    let wanted: Vec<(&'static str, CString)> = instantiations
        .iter()
        .map(|(symbol, expr)| {
            let expr = CString::new(expr.as_str()).map_err(|_| {
                CompileError::Nvrtc(format!("`{symbol}` names an expression with a NUL"))
            })?;
            Ok((*symbol, expr))
        })
        .collect::<Result<_, CompileError>>()?;
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

    let gpu = CString::new(format!("--gpu-architecture={arch}")).expect("no NUL");
    let options: [&CStr; 5] = [
        &gpu,
        c"-std=c++17",
        c"--fmad=false",
        c"--prec-div=true",
        c"--prec-sqrt=true",
    ];
    let option_ptrs: Vec<*const std::ffi::c_char> = options.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: the program is live; every option outlives the call.
    let code = unsafe {
        nvrtc::nvrtcCompileProgram(
            program.0,
            i32::try_from(option_ptrs.len()).expect("five options fit an i32"),
            option_ptrs.as_ptr().cast_mut(),
        )
    };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Nvrtc(program.log()));
    }

    // The mangled names, read BEFORE the program is destroyed: NVRTC owns the
    // strings and frees them with it.
    let mut lowered = Vec::with_capacity(wanted.len());
    for (symbol, expr) in &wanted {
        let mut name: *const std::ffi::c_char = std::ptr::null();
        // SAFETY: the program compiled; `expr` is one of the expressions added
        // above; `name` is a live out-parameter.
        let code = unsafe { nvrtc::nvrtcGetLoweredName(program.0, expr.as_ptr(), &raw mut name) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS || name.is_null() {
            return Err(CompileError::NoLoweredName {
                symbol,
                instantiation: expr.to_string_lossy().into_owned(),
            });
        }
        // SAFETY: NVRTC returns a NUL-terminated string owned by the program,
        // which is still alive here. It is copied, because it is not after
        // `Drop`.
        let mangled = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
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

    Ok(Compiled {
        cubin,
        lowered,
        elapsed: started.elapsed(),
    })
}

/// Either half of getting a family onto the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FamilyError {
    /// The templates and the rows disagree.
    Compile(CompileError),
    /// The image would not load, or a mangled name is not in it.
    Load(Error),
}

impl std::fmt::Display for FamilyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FamilyError::Compile(e) => write!(f, "{e}"),
            FamilyError::Load(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FamilyError {}

/// An NVRTC program, destroyed on the way out.
///
/// A newtype rather than a bare handle because every error path above returns
/// early, and each one of them would otherwise leak the program AND the
/// mangled-name strings it owns.
struct Program(nvrtc::nvrtcProgram);

impl Program {
    /// NVRTC's log, which is where a compilation failure says what is wrong.
    fn log(&self) -> String {
        let mut size = 0usize;
        // SAFETY: the program is live; `size` is a live out-parameter.
        if unsafe { nvrtc::nvrtcGetProgramLogSize(self.0, &raw mut size) }
            != nvrtc::nvrtcResult::NVRTC_SUCCESS
            || size <= 1
        {
            return "NVRTC rejected the source and offered no log".into();
        }
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` is exactly the size NVRTC just reported.
        if unsafe { nvrtc::nvrtcGetProgramLog(self.0, buf.as_mut_ptr().cast()) }
            != nvrtc::nvrtcResult::NVRTC_SUCCESS
        {
            return "NVRTC rejected the source and would not say why".into();
        }
        buf.pop();
        String::from_utf8_lossy(&buf).into_owned()
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

/// The driver's own `cuModuleGetFunction`, re-exported for the loader.
///
/// Here rather than in `device.rs` because looking a MANGLED name up is this
/// module's concern: the name came from NVRTC, and nothing else in the crate
/// has one.
pub(super) fn function_by_name(
    module: dr::CUmodule,
    mangled: &str,
) -> Result<dr::CUfunction, Error> {
    let Ok(c_name) = CString::new(mangled) else {
        return Err(Error::Invalid(format!(
            "the mangled name `{mangled}` contains a NUL"
        )));
    };
    let mut function: dr::CUfunction = std::ptr::null_mut();
    // SAFETY: `module` is loaded and `c_name` outlives the call.
    let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, c_name.as_ptr()) };
    if code == dr::CUresult::CUDA_SUCCESS {
        Ok(function)
    } else {
        Err(Error::Driver {
            call: "cuModuleGetFunction",
            code,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ENTRIES, SOURCE};

    /// Every `#include` the compiled source reaches names a header the
    /// binary carries.
    ///
    /// Not style, and it replaces a stricter rule that Stage B retired.
    /// `nvrtcCreateProgram` used to be called with zero headers and zero
    /// include names, so an `#include` here was a run-time compile failure on
    /// a machine that may have no CUDA headers at all — and the test forbade
    /// the directive outright. It is now called with
    /// [`headers::DEVICE_HEADERS`], so the rule is narrower: an include is
    /// fine, an include the SET does not carry is the same failure as before.
    ///
    /// Angle includes stay forbidden. Those name a compiler's own headers,
    /// which is exactly the reach for `/usr/local/cuda/include` that would
    /// cost the toolkit-free run.
    #[test]
    fn the_template_source_includes_only_what_the_binary_carries() {
        for included in super::headers::quoted_includes(SOURCE) {
            assert!(
                super::headers::DEVICE_HEADERS
                    .iter()
                    .any(|h| h.name == included),
                "{}: `#include \"{included}\"` names no header in the set, and \
                 NVRTC resolves against the set and nothing else",
                super::UNIT,
            );
        }
        for (n, line) in SOURCE.lines().enumerate() {
            assert!(
                !line.starts_with("#include <"),
                "{}:{}: `{}` -- an angle include is the compiler's own header \
                 path, which this source is compiled without",
                super::UNIT,
                n + 1,
                line.trim()
            );
        }
    }

    /// Every row's template and tag appear in the source that will be
    /// compiled.
    ///
    /// A weaker check than the compile itself and a much faster one: it runs
    /// with no GPU and no NVRTC, and it catches the common edit — a template
    /// renamed in the header and not in the table.
    ///
    /// Searched over the WHOLE translation unit, which since Stage B is the
    /// root plus every header it can include. Searching only the root would
    /// have started failing the moment the element types moved into
    /// `pie_device.cuh` — and the honest reading of that failure is
    /// not "the test broke" but "the test was asking about a file when it
    /// meant to ask about a compile".
    #[test]
    fn every_row_names_something_the_source_defines() {
        let unit: String = std::iter::once(SOURCE)
            .chain(super::headers::DEVICE_HEADERS.iter().map(|h| h.text))
            .collect::<Vec<_>>()
            .join("\n");
        for k in ENTRIES {
            let template = k.template_path.rsplit("::").next().expect("a leaf name");
            let elem = k.elem.rsplit("::").next().expect("a leaf name");
            assert!(
                unit.contains(&format!("__global__ void {template}(")),
                "{}: the source defines no `__global__ void {template}`",
                k.sig.symbol
            );
            assert!(
                unit.contains(&format!("struct {elem} {{")),
                "{}: the source defines no element type `{elem}`",
                k.sig.symbol
            );
            // A format with no `Elem` specialisation compiles as a template
            // and fails to instantiate, which is a much worse error to read
            // than this one.
            assert!(
                unit.contains(&format!("struct Elem<{elem}> {{")),
                "{}: `{elem}` has no `Elem` specialisation, so no kernel can widen it",
                k.sig.symbol
            );
        }
    }

    /// A cubin is keyed on everything that produced it, and a header is part
    /// of that.
    ///
    /// The failure this forbids is specific and silent: edit
    /// `pie_device.cuh`, leave `altup_aux.cuh` alone, and a key computed
    /// from the root source alone is unchanged — so the next run loads a
    /// cubin compiled from the header's old text and every kernel quietly
    /// widens bf16 the way it used to.
    #[test]
    fn the_cache_key_spans_the_header_set() {
        use super::headers::Header;

        let base = super::Family::cache_key("sm_89");
        assert_eq!(base, super::Family::cache_key("sm_89"), "and is stable");
        assert_ne!(
            base,
            super::Family::cache_key("sm_90"),
            "a cubin is per-architecture, not portable"
        );

        let edited = [Header {
            name: super::headers::DEVICE_HEADERS[0].name,
            text: "// a header that changed while the root did not",
        }];
        assert_ne!(
            base,
            super::Family::cache_key_with("sm_89", &edited),
            "editing a header must move the key even though SOURCE is identical"
        );

        assert_ne!(
            base,
            super::Family::cache_key_with("sm_89", &[]),
            "and so must removing one"
        );
    }

    /// An instantiation is fully qualified. NVRTC resolves a name expression
    /// in no particular scope, so an unqualified one is a lookup that depends
    /// on where the compiler happened to be.
    #[test]
    fn every_instantiation_is_rooted_at_the_global_namespace() {
        for k in ENTRIES {
            let inst = k.instantiation();
            assert!(
                inst.starts_with("::"),
                "{}: `{inst}` is not rooted",
                k.sig.symbol
            );
            assert!(
                inst.contains("<::"),
                "{}: `{inst}`'s argument is not rooted",
                k.sig.symbol
            );
        }
    }
}
