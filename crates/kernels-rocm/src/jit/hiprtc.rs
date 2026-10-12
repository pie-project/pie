use std::ffi::{CStr, CString, c_char};

use crate::hip::sys;
use crate::jit::{Fault, Toolchain};
use crate::source::Header;

fn api() -> Result<&'static sys::Api, Fault> {
    sys::api().map_err(|log| Fault::Compile {
        unit: "hiprtc",
        log,
    })
}

pub(crate) fn version() -> Result<Toolchain, Fault> {
    let mut major = 0i32;
    let mut minor = 0i32;

    let code = unsafe { (api()?.rtc.hiprtcVersion)(&raw mut major, &raw mut minor) };
    if code != sys::HIPRTC_SUCCESS {
        return Err(Fault::Device {
            call: "hiprtcVersion",
            code,
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
            log: format!("needs hiprtc {floor} and this process loaded {have}"),
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
    pub code: Vec<u8>,
    pub lowered: Vec<String>,
    pub log: String,
}

pub(crate) fn compile_text(job: &Job<'_>) -> Result<Built, Fault> {
    let unit = job.name;
    let api = api()?;
    admits(unit, job.floor)?;
    if job.wanted.is_empty() {
        return Err(Fault::Compile {
            unit,
            log: format!(
                "`{unit}` was asked for a code object with no instantiations in it, which \
                 would be cached under this architecture and satisfy no fire"
            ),
        });
    }
    let options = options(job.arch, job.options)?;

    let root = CString::new(format!("{COMPAT}{}", job.source)).map_err(|_| Fault::Compile {
        unit,
        log: format!("`{unit}`'s source contains a NUL"),
    })?;
    let name = CString::new(unit).map_err(|_| Fault::Compile {
        unit,
        log: format!("the unit name `{unit}` has a NUL"),
    })?;

    let (header_texts, header_names) =
        crate::source::as_rtc_arrays(job.headers).map_err(|log| Fault::Compile { unit, log })?;
    let text_ptrs: Vec<*const c_char> = header_texts.iter().map(|t| t.as_ptr()).collect();
    let name_ptrs: Vec<*const c_char> = header_names.iter().map(|n| n.as_ptr()).collect();
    let count = i32::try_from(text_ptrs.len()).map_err(|_| Fault::Compile {
        unit,
        log: "more headers than hiprtc can take".into(),
    })?;

    let mut handle: sys::Program = std::ptr::null_mut();

    let code = unsafe {
        (api.rtc.hiprtcCreateProgram)(
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
    if code != sys::HIPRTC_SUCCESS {
        return Err(Fault::Device {
            call: "hiprtcCreateProgram",
            code,
        });
    }
    let program = Program(handle, api);

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
        let code = unsafe { (api.rtc.hiprtcAddNameExpression)(program.0, expr.as_ptr()) };
        if code != sys::HIPRTC_SUCCESS {
            return Err(Fault::Compile {
                unit,
                log: format!(
                    "`{unit}` names `{}`, which hiprtc would not accept as an expression",
                    expr.to_string_lossy()
                ),
            });
        }
    }

    let option_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();

    let code = unsafe {
        (api.rtc.hiprtcCompileProgram)(
            program.0,
            i32::try_from(option_ptrs.len()).expect("a handful of options fit an i32"),
            option_ptrs.as_ptr(),
        )
    };
    let log = program.log();
    if code != sys::HIPRTC_SUCCESS {
        let log = log.unwrap_or_else(|| "hiprtc rejected the source and offered no log".into());
        return Err(Fault::Compile { unit, log });
    }
    let log = log.unwrap_or_default();

    let mut lowered = Vec::with_capacity(wanted.len());
    for expr in &wanted {
        let mut mangled: *const c_char = std::ptr::null();

        let code = unsafe {
            (api.rtc.hiprtcGetLoweredName)(program.0, expr.as_ptr(), &raw mut mangled)
        };
        if code != sys::HIPRTC_SUCCESS || mangled.is_null() {
            return Err(Fault::Compile {
                unit,
                log: format!(
                    "`{}` compiled and hiprtc would not lower a name for it",
                    expr.to_string_lossy()
                ),
            });
        }

        let mangled = unsafe { CStr::from_ptr(mangled) }
            .to_string_lossy()
            .into_owned();
        lowered.push(mangled);
    }

    let mut size = 0usize;

    let code = unsafe { (api.rtc.hiprtcGetCodeSize)(program.0, &raw mut size) };
    if code != sys::HIPRTC_SUCCESS {
        return Err(Fault::Device {
            call: "hiprtcGetCodeSize",
            code,
        });
    }
    let mut image = vec![0u8; size];

    let code = unsafe { (api.rtc.hiprtcGetCode)(program.0, image.as_mut_ptr().cast()) };
    if code != sys::HIPRTC_SUCCESS {
        return Err(Fault::Device {
            call: "hiprtcGetCode",
            code,
        });
    }

    Ok(Built {
        code: image,
        lowered,
        log,
    })
}

/// Put ahead of every unit: the CUDA-to-HIP names and the 32-lane warp, in
/// `shim/pie_hip_compat.h`. It shifts the unit's line numbers by one.
const COMPAT: &str = "#include \"pie_hip_compat.h\"\n";

/// Options NVRTC takes that clang has no spelling for; the carried units
/// still name them, so they are dropped here rather than in every root.
const NVRTC_ONLY: &[&str] = &["--device-as-default-execution-space"];

/// The ROCm include root, for the few device headers hiprtc does not carry
/// itself (`hip/hip_fp8.h`, `hip/hip_cooperative_groups.h`).
fn include_root() -> String {
    let root = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_owned());
    format!("-I{root}/include")
}

fn options(arch: &str, extra: &[&str]) -> Result<Vec<CString>, Fault> {
    if !arch.starts_with("gfx") {
        return Err(Fault::Compile {
            unit: "hiprtc",
            log: format!("`{arch}` is not an AMD GPU architecture: hiprtc emits for `gfxNNN`"),
        });
    }
    let nul = |what: &str| Fault::Compile {
        unit: "hiprtc",
        log: format!("{what} contains a NUL"),
    };
    let mut all = vec![
        CString::new(format!("--offload-arch={arch}")).map_err(|_| nul("the architecture"))?,
        c"-std=c++17".to_owned(),
        c"-O3".to_owned(),
        // NVRTC's `--fmad=false`: no contraction of a*b+c into an fma, so a
        // kernel's rounding matches its reference.
        c"-ffp-contract=off".to_owned(),
        CString::new(include_root()).map_err(|_| nul("ROCM_PATH"))?,
    ];
    for option in extra {
        if NVRTC_ONLY.contains(option) {
            continue;
        }
        all.push(CString::new(*option).map_err(|_| nul(&format!("the option `{option}`")))?);
    }
    Ok(all)
}

struct Program(sys::Program, &'static sys::Api);

impl Program {
    fn log(&self) -> Option<String> {
        let mut size = 0usize;

        if unsafe { (self.1.rtc.hiprtcGetProgramLogSize)(self.0, &raw mut size) }
            != sys::HIPRTC_SUCCESS
            || size <= 1
        {
            return None;
        }
        let mut buf = vec![0u8; size];

        if unsafe { (self.1.rtc.hiprtcGetProgramLog)(self.0, buf.as_mut_ptr().cast()) }
            != sys::HIPRTC_SUCCESS
        {
            return Some("hiprtc has a log for this program and would not hand it over".into());
        }
        while buf.last() == Some(&0) {
            buf.pop();
        }
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { (self.1.rtc.hiprtcDestroyProgram)(&raw mut self.0) };
    }
}
