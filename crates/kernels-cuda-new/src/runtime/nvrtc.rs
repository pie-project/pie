use std::ffi::{CStr, CString, c_char};
use std::time::{Duration, Instant};

use cudarc::nvrtc::sys as nvrtc;

use crate::jit::Toolchain;
use crate::source::Header;

/// Why a unit would not compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// NVRTC rejected the source, or an expression naming an instantiation.
    Nvrtc(String),
    /// A call failed for a reason that is not about the source — the call's
    Driver(&'static str, i32),
    /// A row's instantiation compiled and NVRTC has no lowered name for it.
    NoLoweredName {
        /// The row that named it.
        symbol: &'static str,
        /// The expression that produced nothing.
        instantiation: String,
    },
    /// The compile was refused before NVRTC was asked.
    Refused(String),
    /// This machine's NVRTC is older than the unit says it needs.
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
pub fn version() -> Result<Toolchain, CompileError> {
    let mut major = 0i32;
    let mut minor = 0i32;
    // SAFETY: both out-parameters are live `i32`s for the duration of the
    let code = unsafe { nvrtc::nvrtcVersion(&raw mut major, &raw mut minor) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(CompileError::Driver("nvrtcVersion", code as i32));
    }
    Ok(Toolchain::new(u32::try_from(major).unwrap_or(0), u32::try_from(minor).unwrap_or(0)))
}

/// Whether the NVRTC this process loaded may compile a unit whose floor is
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

/// One compile, described by everything that decides its answer.
pub struct Job<'a> {
    /// What a diagnostic calls it.
    pub name: &'static str,
    /// The translation unit, root and any appendix already joined.
    pub source: String,
    /// The target architecture, as `sm_XY`.
    pub arch: &'a str,
    /// Options beyond the ones every compile here carries.
    pub options: &'a [&'a str],
    /// The carried headers `#include`s resolve against.
    pub headers: &'a [Header],
    /// The oldest NVRTC that may answer.
    pub floor: Toolchain,
    /// The instantiations to ask for, as C++ expressions.
    pub wanted: &'a [String],
    /// Whether the cubin must be device-linked before it will load.
    pub device_link: bool,
}

/// One compile, done.
pub struct Built {
    /// The image.
    pub cubin: Vec<u8>,
    /// The mangled name of each of [`Job::wanted`], in that order.
    pub lowered: Vec<String>,
    /// What the compile alone cost.
    pub elapsed: Duration,
    /// NVRTC's log.
    pub log: String,
}

/// Compile one translation unit and resolve the names it was asked for.
///
/// The one place NVRTC is driven. A unit compile and a per-symbol compile
/// differ only in what they put in [`Job::wanted`].
pub fn compile_text(job: &Job<'_>) -> Result<Built, CompileError> {
    let unit = job.name;
    let arch = job.arch;
    let headers = job.headers;
    admits(unit, job.floor)?;
    if job.wanted.is_empty() {
        return Err(CompileError::Refused(format!(
            "`{unit}` was asked for a cubin with no instantiations in it, which would \
             be cached under this architecture and satisfy no fire"
        )));
    }
    let options = options(arch, job.options)?;

    let started = Instant::now();
    let root = CString::new(job.source.clone())
        .map_err(|_| CompileError::Refused(format!("`{unit}`'s source contains a NUL")))?;
    let name = CString::new(unit)
        .map_err(|_| CompileError::Refused(format!("the unit name `{unit}` has a NUL")))?;

    let (header_texts, header_names) =
        crate::source::as_nvrtc_arrays(headers).map_err(CompileError::Refused)?;
    let text_ptrs: Vec<*const c_char> = header_texts.iter().map(|t| t.as_ptr()).collect();
    let name_ptrs: Vec<*const c_char> = header_names.iter().map(|n| n.as_ptr()).collect();
    let count = i32::try_from(text_ptrs.len())
        .map_err(|_| CompileError::Refused("more headers than NVRTC can take".into()))?;

    let mut handle: nvrtc::nvrtcProgram = std::ptr::null_mut();
    // SAFETY: every string outlives the call, and the two arrays are the same
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

    let wanted: Vec<CString> = job
        .wanted
        .iter()
        .map(|expr| {
            CString::new(expr.as_str()).map_err(|_| {
                CompileError::Refused(format!(
                    "`{unit}` names the instantiation `{expr}`, which has a NUL in it"
                ))
            })
        })
        .collect::<Result<_, CompileError>>()?;
    for expr in &wanted {
        // SAFETY: the program is live and `expr` outlives the call.
        let code = unsafe { nvrtc::nvrtcAddNameExpression(program.0, expr.as_ptr()) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
            return Err(CompileError::Nvrtc(format!(
                "`{unit}` names `{}`, which NVRTC would not accept as an expression",
                expr.to_string_lossy()
            )));
        }
    }

    let option_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: the program is live; every option outlives the call. NVRTC takes
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

    let mut lowered = Vec::with_capacity(wanted.len());
    for expr in &wanted {
        let mut mangled: *const c_char = std::ptr::null();
        // SAFETY: the program compiled; `expr` is one of the expressions added
        let code =
            unsafe { nvrtc::nvrtcGetLoweredName(program.0, expr.as_ptr(), &raw mut mangled) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS || mangled.is_null() {
            return Err(CompileError::NoLoweredName {
                symbol: unit,
                instantiation: expr.to_string_lossy().into_owned(),
            });
        }
        // SAFETY: NVRTC returns a NUL-terminated string owned by the program,
        let mangled = unsafe { CStr::from_ptr(mangled) }.to_string_lossy().into_owned();
        lowered.push(mangled);
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
    // A leading `&` is how this crate spells a variable's name expression;
    // a function is named bare. See `attention_mla_fa2.cuh`, which records
    // `nvrtcAddNameExpression` refusing `smem_bytes_mla<KT>` and accepting
    // `&smem_bytes_mla<KT>`.
    let wants_function = job.wanted.iter().any(|w| !w.trim_start().starts_with('&'));
    unassembled_tile_ir(unit, &cubin, wants_function)?;
    if job.device_link {
        cubin = device_link(unit, &cubin)?;
    }

    Ok(Built { cubin, lowered, elapsed: started.elapsed(), log })
}

/// `libcudadevrt.a`, which a cooperative kernel's `grid.sync()` needs.
///
/// `cooperative_groups::this_grid()` lowers to `cudaCGGetIntrinsicHandle`, an
/// extern device function `nvcc` resolves in its device-link step and NVRTC
/// does not have: the cubin loads and `ptxas` reports
/// `Unresolved extern function`. `cuLink` is that step, at runtime.
fn cudadevrt() -> Result<std::ffi::CString, CompileError> {
    let roots = ["CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(std::path::PathBuf::from)
        .chain(["/usr/local/cuda", "/opt/cuda"].into_iter().map(std::path::PathBuf::from));
    for root in roots {
        for lib in ["lib64", "lib"] {
            let path = root.join(lib).join("libcudadevrt.a");
            if path.exists() {
                return std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
                    CompileError::Refused("the path to `libcudadevrt.a` contains a NUL".into())
                });
            }
        }
    }
    Err(CompileError::Refused(
        "a cooperative unit needs `libcudadevrt.a` and none was found under CUDA_ROOT, \
         CUDA_HOME, CUDA_PATH, /usr/local/cuda or /opt/cuda"
            .into(),
    ))
}

/// Link one relocatable cubin against the CUDA device runtime.
fn device_link(unit: &str, relocatable: &[u8]) -> Result<Vec<u8>, CompileError> {
    use cudarc::driver::sys as dr;

    let devrt = cudadevrt()?;
    let mut state: dr::CUlinkState = std::ptr::null_mut();
    // SAFETY: no options, so both arrays are null and the count is zero.
    let code = unsafe {
        dr::cuLinkCreate_v2(0, std::ptr::null_mut(), std::ptr::null_mut(), &raw mut state)
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(CompileError::Driver("cuLinkCreate", code as i32));
    }

    struct Link(dr::CUlinkState);
    impl Drop for Link {
        fn drop(&mut self) {
            // SAFETY: created above and destroyed exactly once.
            unsafe { dr::cuLinkDestroy(self.0) };
        }
    }
    let link = Link(state);

    let name = std::ffi::CString::new(unit).unwrap_or_default();
    // SAFETY: `relocatable` outlives the call and its length is its own.
    let code = unsafe {
        dr::cuLinkAddData_v2(
            link.0,
            dr::CUjitInputType::CU_JIT_INPUT_CUBIN,
            relocatable.as_ptr().cast_mut().cast(),
            relocatable.len(),
            name.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(CompileError::Driver("cuLinkAddData", code as i32));
    }

    // SAFETY: `devrt` names a file that exists and outlives the call.
    let code = unsafe {
        dr::cuLinkAddFile_v2(
            link.0,
            dr::CUjitInputType::CU_JIT_INPUT_LIBRARY,
            devrt.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(CompileError::Driver("cuLinkAddFile", code as i32));
    }

    let mut image: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut size = 0usize;
    // SAFETY: the state holds both inputs; both out-parameters are live.
    let code = unsafe { dr::cuLinkComplete(link.0, &raw mut image, &raw mut size) };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(CompileError::Driver("cuLinkComplete", code as i32));
    }
    if image.is_null() || size == 0 {
        return Err(CompileError::Refused(format!("`{unit}` linked to an empty image")));
    }
    // SAFETY: `cuLinkComplete` reports an image of `size` bytes owned by the
    // link state, which `link` keeps alive across this copy.
    let linked = unsafe { std::slice::from_raw_parts(image.cast::<u8>(), size) }.to_vec();
    Ok(linked)
}

/// Refuse an image that is Tile IR rather than SASS.
///
/// `wants_function` is what makes a missing `.text` a defect. The symptom this
/// recognises is a kernel that would answer `cuModuleGetFunction` with
/// NOT_FOUND, and a compile that asked for no kernel has none to lose: an
/// image holding only `__device__` variables is `.text`-less because there is
/// nothing to put there. The smem echoes are compiled exactly that way — a
/// `sizeof` is readable without paying to codegen the 360-symbol kernel that
/// would otherwise have to come with it — so without this the check refuses a
/// correct image and names the wrong cause.
fn unassembled_tile_ir(
    unit: &str,
    cubin: &[u8],
    wants_function: bool,
) -> Result<(), CompileError> {
    let has = |needle: &[u8]| cubin.windows(needle.len()).any(|w| w == needle);
    if wants_function && has(b".note.nv.tkinfo") && !has(b".text.") {
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

/// Assemble Tile IR into a loadable cubin, by running `tileiras`.
pub fn assemble_tile_ir(
    tile_ir: &[u8],
    arch: &str,
    tileiras: &std::path::Path,
) -> Result<Vec<u8>, CompileError> {
    use std::io::Write as _;

    if !tileiras.exists() {
        return Err(CompileError::Refused(format!(
            "`tileiras` not found at {}. It ships in its own wheel, \
             `nvidia-cuda-tileiras`, versioned independently of nvcc, nvrtc \
             and the runtime headers -- nothing cross-checks the four",
            tileiras.display()
        )));
    }

    let in_tree = tileiras
        .parent()
        .and_then(|bin| bin.parent())
        .is_some_and(|root| root.join("include").is_dir() || root.join("nvvm").is_dir());
    if !in_tree && std::env::var_os("CUDA_ROOT").is_none() {
        return Err(CompileError::Refused(format!(
            "`tileiras` at {} is not inside a CUDA toolkit and CUDA_ROOT is \
             unset, so it cannot find one. It would fail with `error: failed \
             to compile Tile IR program` and nothing else -- the same message \
             a malformed input gets. Set CUDA_ROOT to the toolkit root, or \
             run the copy that lives in it",
            tileiras.display()
        )));
    }

    let stem = std::env::temp_dir().join(format!(
        "pie-tileir-{}-{:p}",
        std::process::id(),
        tile_ir.as_ptr()
    ));
    let input = stem.with_extension("tilebc");
    let output = stem.with_extension("cubin");
    let cleanup = || {
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    };

    let write = std::fs::File::create(&input).and_then(|mut f| f.write_all(tile_ir));
    if let Err(e) = write {
        cleanup();
        return Err(CompileError::Refused(format!(
            "could not stage Tile IR at {}: {e}. `tileiras` reads a file and \
             offers no stdin, so a writable temp directory is a requirement \
             of this path rather than a convenience",
            input.display()
        )));
    }

    let run = std::process::Command::new(tileiras)
        .arg(format!("--gpu-name={arch}"))
        .arg("-o")
        .arg(&output)
        .arg(&input)
        .output();

    let run = match run {
        Ok(r) => r,
        Err(e) => {
            cleanup();
            return Err(CompileError::Refused(format!(
                "could not run {}: {e}",
                tileiras.display()
            )));
        }
    };

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr).trim().to_string();
        cleanup();
        return Err(CompileError::Refused(format!(
            "`tileiras` refused the Tile IR for {arch}: {stderr}. With the \
             toolkit reachable -- checked above -- this is the input, not the \
             environment"
        )));
    }

    let cubin = std::fs::read(&output);
    cleanup();
    let cubin = cubin.map_err(|e| {
        CompileError::Refused(format!(
            "`tileiras` reported success and left nothing readable at {}: {e}",
            output.display()
        ))
    })?;

    // `true`: assembling Tile IR is done to obtain kernels, so an image that
    // still has no `.text` is `tileiras` having reported a success it did not
    // deliver — the one case this check was written for.
    unassembled_tile_ir("assembled tile ir", &cubin, true)?;
    Ok(cubin)
}

/// The compile options, in the order NVRTC is handed them.
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
    for option in extra {
        all.push(
            CString::new(*option).map_err(|_| {
                CompileError::Refused(format!("the option `{option}` contains a NUL"))
            })?,
        );
    }
    Ok(all)
}

/// An NVRTC program, destroyed on the way out.
struct Program(nvrtc::nvrtcProgram);

impl Program {
    /// NVRTC's log, or `None` when it had nothing to say.
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
        unsafe { nvrtc::nvrtcDestroyProgram(&raw mut self.0) };
    }
}
#[cfg(test)]
mod tile_header_trap {
    use super::tile_header_mismatch;

    /// The real message, copied from a failing build on this box: nvcc 13.3.73
    const REAL_ICE: &str = r#"/opt/cu13/include/crt/cuda_tile.h(1364): error: Internal Compiler Error (tile codegen): "Unexpected element type in tile!"
Compilation aborted."#;

    #[test]
    fn the_ice_gets_a_cause_attached() {
        let d = tile_header_mismatch(REAL_ICE).expect(
            "the bf16 tile-codegen ICE must be recognised -- it is the one \
             CuTile failure whose message names nothing that caused it",
        );

        assert!(d.contains("__NV_TL_BUILTIN__"), "the marker must be named");
        assert!(
            d.contains("NOT a compiler bug"),
            "the message must say this plainly. A day and a withdrawn bug \
             report went into learning it, and the ICE reads like a compiler \
             bug to everyone who meets it"
        );

        assert!(
            d.contains("cuda_tf32.h"),
            "the message must give the one-`ls` proxy; a version number does \
             not answer the question because the RUNTIME wheel is the one \
             that matters and it versions independently"
        );
        assert!(
            d.contains("grep -c __NV_TL_BUILTIN__"),
            "and the exact test for when the proxy is ambiguous"
        );
    }

    /// Each recognition branch, pinned on its own.
    #[test]
    fn each_branch_is_pinned_separately() {
        assert!(
            tile_header_mismatch("error: \"Unexpected element type in tile!\"").is_some(),
            "the element-type message must be recognised on its own -- it is \
             the specific symptom of an unmarked 16-bit type and the banner \
             around it is not load-bearing"
        );

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
    #[test]
    fn a_self_explaining_error_is_not_decorated() {
        let clear = r#"cuda_tile.h(55): error: #error "This file needs C++20 features. Please compile with c++20 or later dialect""#;
        assert!(
            tile_header_mismatch(clear).is_none(),
            "an error that already names its own fix must be left alone"
        );
    }
}

#[cfg(test)]
mod tile_assembly {
    use super::{CompileError, assemble_tile_ir};
    use std::path::{Path, PathBuf};

    fn refusal(e: CompileError) -> String {
        match e {
            CompileError::Refused(m) => m,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// A missing assembler must say WHICH of the four wheels is missing. The
    #[test]
    fn a_missing_assembler_names_its_wheel() {
        let e = refusal(
            assemble_tile_ir(b"", "sm_89", Path::new("/nonexistent/tileiras")).unwrap_err(),
        );
        assert!(e.contains("nvidia-cuda-tileiras"), "the refusal must name the wheel: {e}");
        assert!(
            e.contains("nothing cross-checks the four"),
            "and must say why the version is not obvious: {e}"
        );
    }

    /// The trap this function exists to make impossible: an assembler outside
    #[test]
    fn an_unrooted_assembler_is_refused_before_it_can_lie() {
        let dir = std::env::temp_dir().join(format!("pie-unrooted-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let fake = dir.join("tileiras");
        std::fs::write(&fake, b"#!/bin/sh\nexit 0\n").expect("write");

        // SAFETY-adjacent: this test is the only reader of CUDA_ROOT here and
        let saved = std::env::var_os("CUDA_ROOT");
        unsafe { std::env::remove_var("CUDA_ROOT") };
        let got = assemble_tile_ir(b"not really tile ir", "sm_89", &fake);
        if let Some(v) = saved {
            unsafe { std::env::set_var("CUDA_ROOT", v) };
        }
        let _ = std::fs::remove_dir_all(&dir);

        let e = refusal(got.unwrap_err());
        assert!(
            e.contains("CUDA_ROOT is \nunset") || e.contains("CUDA_ROOT is unset"),
            "the refusal must name CUDA_ROOT: {e}"
        );
        assert!(
            e.contains("the same message \na malformed input gets")
                || e.contains("the same message a malformed input gets"),
            "and must say why this is checked up front rather than reported \
             from the exit code: the two failures are indistinguishable \
             afterwards. Got: {e}"
        );
    }

    /// Find a real `tileiras`, or `None`. Deliberately not a hard-coded path:
    fn real_tileiras() -> Option<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(r) = std::env::var_os("CUDA_ROOT") {
            roots.push(PathBuf::from(r));
        }
        roots.push(PathBuf::from("/usr/local/cuda"));
        for r in roots {
            let p = r.join("bin/tileiras");
            if p.exists() {
                return Some(p);
            }
        }
        std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).map(|d| d.join("tileiras")).collect::<Vec<_>>())
            .unwrap_or_default()
            .into_iter()
            .find(|p| p.exists())
    }

    /// Garbage in must be refused, and the refusal must say it is the input.
    #[test]
    fn a_real_assembler_refuses_garbage_and_blames_the_input() {
        let Some(tileiras) = real_tileiras() else {
            assert!(
                std::env::var_os("PIE_REQUIRE_TILEIRAS").is_none(),
                "PIE_REQUIRE_TILEIRAS is set and no `tileiras` was found on \
                 CUDA_ROOT, /usr/local/cuda or PATH. Either it is missing or \
                 it is somewhere this test does not look -- both are worth a \
                 failure when the path is meant to be covered"
            );
            eprintln!(
                "SKIPPED (no tileiras on CUDA_ROOT, /usr/local/cuda or PATH). \
                 Set PIE_REQUIRE_TILEIRAS to make this a failure."
            );
            return;
        };
        let e = refusal(
            assemble_tile_ir(b"this is not tile bytecode", "sm_89", &tileiras).unwrap_err(),
        );
        assert!(
            e.contains("this is the input, not the \nenvironment")
                || e.contains("this is the input, not the environment"),
            "with the toolkit reachable the refusal must point at the input: {e}"
        );
    }
}
