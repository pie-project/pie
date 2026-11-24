#![allow(unsafe_code)]
use std::ffi::{CStr, CString, c_char};

use cudarc::nvrtc::sys as nvrtc;

use crate::jit::{Fault, Toolchain};
use crate::source::Header;

pub(crate) fn version() -> Result<Toolchain, Fault> {
    let mut major = 0i32;
    let mut minor = 0i32;

    // SAFETY: both out-pointers are locals that live across the call.
    let code = unsafe { nvrtc::nvrtcVersion(&raw mut major, &raw mut minor) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(Fault::Device {
            call: "nvrtcVersion",
            code: code as i32,
        });
    }
    Ok(Toolchain::new(
        u32::try_from(major).unwrap_or(0),
        u32::try_from(minor).unwrap_or(0),
    ))
}

fn admits(unit: &'static str, floor: Toolchain) -> Result<(), Fault> {
    if floor.is_any() {
        return Ok(());
    }
    let have = version()?;
    if floor.met_by(have) {
        Ok(())
    } else {
        Err(Fault::Compile {
            unit,
            log: format!("needs NVRTC {floor} and this process loaded {have}"),
        })
    }
}

pub(crate) struct Job<'a> {
    pub name: &'static str,
    pub source: String,
    pub arch: &'a str,
    pub options: &'a [&'a str],
    pub headers: &'a [Header],
    pub floor: Toolchain,
    pub wanted: &'a [String],
}

pub(crate) struct Built {
    pub cubin: Vec<u8>,
    pub lowered: Vec<String>,
    pub log: String,
}

pub(crate) fn compile_text(job: &Job<'_>) -> Result<Built, Fault> {
    let unit = job.name;
    let arch = job.arch;
    let headers = job.headers;
    admits(unit, job.floor)?;
    let options = options(arch, job.options)?;

    let root = CString::new(job.source.clone()).map_err(|_| Fault::Compile {
        unit,
        log: format!("`{unit}`'s source contains a NUL"),
    })?;
    let name = CString::new(unit).map_err(|_| Fault::Compile {
        unit,
        log: format!("the unit name `{unit}` has a NUL"),
    })?;

    let (header_texts, header_names) =
        crate::source::as_nvrtc_arrays(headers).map_err(|log| Fault::Compile { unit, log })?;
    let text_ptrs: Vec<*const c_char> = header_texts.iter().map(|t| t.as_ptr()).collect();
    let name_ptrs: Vec<*const c_char> = header_names.iter().map(|n| n.as_ptr()).collect();
    let count = i32::try_from(text_ptrs.len()).map_err(|_| Fault::Compile {
        unit,
        log: "more headers than NVRTC can take".into(),
    })?;

    let mut handle: nvrtc::nvrtcProgram = std::ptr::null_mut();

    // SAFETY: `root` and `name` are NUL-terminated CStrings; each array holds
    // `count` pointers into the CStrings of `header_texts`/`header_names`,
    // all of which outlive the call (NVRTC copies them) and `handle` is a
    // local out-pointer. An empty array is passed as null, as NVRTC asks.
    let code = unsafe {
        nvrtc::nvrtcCreateProgram(
            &raw mut handle,
            root.as_ptr(),
            name.as_ptr(),
            count,
            if text_ptrs.is_empty() {
                std::ptr::null()
            } else {
                text_ptrs.as_ptr()
            },
            if name_ptrs.is_empty() {
                std::ptr::null()
            } else {
                name_ptrs.as_ptr()
            },
        )
    };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(Fault::Device {
            call: "nvrtcCreateProgram",
            code: code as i32,
        });
    }
    let program = Program(handle);

    let wanted: Vec<CString> = job
        .wanted
        .iter()
        .map(|expr| {
            CString::new(expr.as_str()).map_err(|_| Fault::Compile {
                unit,
                log: format!("`{unit}` names the instantiation `{expr}`, which has a NUL in it"),
            })
        })
        .collect::<Result<_, Fault>>()?;
    for expr in &wanted {
        // SAFETY: `program` holds the handle nvrtcCreateProgram just made and
        // `expr` is a CString that outlives the call.
        let code = unsafe { nvrtc::nvrtcAddNameExpression(program.0, expr.as_ptr()) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
            return Err(Fault::Compile {
                unit,
                log: format!(
                    "`{unit}` names `{}`, which NVRTC would not accept as an expression",
                    expr.to_string_lossy()
                ),
            });
        }
    }

    let option_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();

    // SAFETY: `program` is live; `option_ptrs` holds `len` pointers into the
    // CStrings of `options`, which outlive the call. NVRTC only reads them,
    // so the `cast_mut` the C signature asks for is never written through.
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
        return Err(Fault::Compile { unit, log });
    }
    let log = log.unwrap_or_default();

    let mut lowered = Vec::with_capacity(wanted.len());
    for expr in &wanted {
        let mut mangled: *const c_char = std::ptr::null();

        // SAFETY: `program` is live and compiled, `expr` was added to it
        // before the compile, and `mangled` is a local out-pointer.
        let code =
            unsafe { nvrtc::nvrtcGetLoweredName(program.0, expr.as_ptr(), &raw mut mangled) };
        if code != nvrtc::nvrtcResult::NVRTC_SUCCESS || mangled.is_null() {
            return Err(Fault::Compile {
                unit,
                log: format!(
                    "`{}` compiled and NVRTC would not lower a name for it",
                    expr.to_string_lossy()
                ),
            });
        }

        // SAFETY: NVRTC returned success and a non-null pointer to a
        // NUL-terminated name it owns until `program` is destroyed, which is
        // after this copy.
        let mangled = unsafe { CStr::from_ptr(mangled) }
            .to_string_lossy()
            .into_owned();
        lowered.push(mangled);
    }

    let mut size = 0usize;

    // SAFETY: `program` is live and compiled; `size` is a local out-pointer.
    let code = unsafe { nvrtc::nvrtcGetCUBINSize(program.0, &raw mut size) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(Fault::Device {
            call: "nvrtcGetCUBINSize",
            code: code as i32,
        });
    }
    let mut cubin = vec![0u8; size];

    // SAFETY: `cubin` has the `size` bytes NVRTC just reported for this same
    // program, so the copy stays in bounds.
    let code = unsafe { nvrtc::nvrtcGetCUBIN(program.0, cubin.as_mut_ptr().cast()) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(Fault::Device {
            call: "nvrtcGetCUBIN",
            code: code as i32,
        });
    }

    Ok(Built {
        cubin,
        lowered,
        log,
    })
}

/// `sm_XY`, its arch-specific `sm_XYa` or its family `sm_XYf`: each makes
/// NVRTC emit SASS.
fn real(arch: &str) -> bool {
    let Some(rest) = arch.strip_prefix("sm_") else {
        return false;
    };
    let digits = rest.strip_suffix(['a', 'f']).unwrap_or(rest);
    digits.len() >= 2 && digits.bytes().all(|b| b.is_ascii_digit())
}

fn options(arch: &str, extra: &[&str]) -> Result<Vec<CString>, Fault> {
    if !real(arch) {
        return Err(Fault::Compile {
            unit: "nvrtc",
            log: format!(
                "`{arch}` is not a real architecture: only `sm_XY` (or its `sm_XYa`/`sm_XYf` \
                 twin) makes NVRTC emit SASS, and a virtual `compute_XY` would hand the \
                 driver PTX to JIT a second time at load"
            ),
        });
    }
    let gpu = CString::new(format!("--gpu-architecture={arch}")).map_err(|_| Fault::Compile {
        unit: "nvrtc",
        log: format!("the architecture `{arch}` has a NUL"),
    })?;
    let mut all = vec![
        gpu,
        c"-std=c++17".to_owned(),
        c"--fmad=false".to_owned(),
        c"--prec-div=true".to_owned(),
        c"--prec-sqrt=true".to_owned(),
    ];
    for option in extra {
        all.push(CString::new(*option).map_err(|_| Fault::Compile {
            unit: "nvrtc",
            log: format!("the option `{option}` contains a NUL"),
        })?);
    }
    Ok(all)
}

struct Program(nvrtc::nvrtcProgram);

impl Program {
    fn log(&self) -> Option<String> {
        let mut size = 0usize;

        // SAFETY: `self.0` is the live handle this `Program` owns; `size` is a
        // local out-pointer.
        if unsafe { nvrtc::nvrtcGetProgramLogSize(self.0, &raw mut size) }
            != nvrtc::nvrtcResult::NVRTC_SUCCESS
            || size <= 1
        {
            return None;
        }
        let mut buf = vec![0u8; size];

        // SAFETY: `buf` has the `size` bytes NVRTC just reported for this log,
        // terminator included.
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
        // SAFETY: `Program` is built only from a handle nvrtcCreateProgram
        // returned, owns it alone, and drops it once.
        unsafe { nvrtc::nvrtcDestroyProgram(&raw mut self.0) };
    }
}
