//! The compile step: one NVRTC program per (unit, arch), name expressions
//! for the wanted instantiations, SASS out — plus the device-link pass a
//! relocatable unit needs and the diagnostics for the two ways a cubin can
//! be quietly wrong (Tile IR without SASS, tile-frontend header skew).

use std::ffi::{CStr, CString, c_char};

use cudarc::nvrtc::sys as nvrtc;

use crate::jit::{Fault, Toolchain};
use crate::source::Header;

pub(crate) fn version() -> Result<Toolchain, Fault> {
    let mut major = 0i32;
    let mut minor = 0i32;

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
    pub device_link: bool,
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
    if job.wanted.is_empty() {
        return Err(Fault::Compile {
            unit,
            log: format!(
                "`{unit}` was asked for a cubin with no instantiations in it, which would \
                 be cached under this architecture and satisfy no fire"
            ),
        });
    }
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
            return Err(Fault::Compile {
                unit,
                log: format!("{log}\n\n{diagnosis}"),
            });
        }
        return Err(Fault::Compile { unit, log });
    }
    let log = log.unwrap_or_default();

    let mut lowered = Vec::with_capacity(wanted.len());
    for expr in &wanted {
        let mut mangled: *const c_char = std::ptr::null();

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

        let mangled = unsafe { CStr::from_ptr(mangled) }
            .to_string_lossy()
            .into_owned();
        lowered.push(mangled);
    }

    let mut size = 0usize;

    let code = unsafe { nvrtc::nvrtcGetCUBINSize(program.0, &raw mut size) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(Fault::Device {
            call: "nvrtcGetCUBINSize",
            code: code as i32,
        });
    }
    let mut cubin = vec![0u8; size];

    let code = unsafe { nvrtc::nvrtcGetCUBIN(program.0, cubin.as_mut_ptr().cast()) };
    if code != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(Fault::Device {
            call: "nvrtcGetCUBIN",
            code: code as i32,
        });
    }

    let wants_function = job.wanted.iter().any(|w| !w.trim_start().starts_with('&'));
    unassembled_tile_ir(unit, &cubin, wants_function)?;
    if job.device_link {
        cubin = device_link(unit, &cubin)?;
    }

    Ok(Built {
        cubin,
        lowered,
        log,
    })
}

fn cudadevrt() -> Result<CString, Fault> {
    let roots = ["CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(std::path::PathBuf::from)
        .chain(
            ["/usr/local/cuda", "/opt/cuda"]
                .into_iter()
                .map(std::path::PathBuf::from),
        );
    for root in roots {
        for lib in ["lib64", "lib"] {
            let path = root.join(lib).join("libcudadevrt.a");
            if path.exists() {
                return CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
                    Fault::Compile {
                        unit: "nvrtc",
                        log: "the path to `libcudadevrt.a` contains a NUL".into(),
                    }
                });
            }
        }
    }
    Err(Fault::Compile {
        unit: "nvrtc",
        log: "a cooperative unit needs `libcudadevrt.a` and none was found under CUDA_ROOT, \
              CUDA_HOME, CUDA_PATH, /usr/local/cuda or /opt/cuda"
            .into(),
    })
}

fn device_link(unit: &'static str, relocatable: &[u8]) -> Result<Vec<u8>, Fault> {
    use cudarc::driver::sys as dr;

    let devrt = cudadevrt()?;

    crate::jit::cache::bind_context().map_err(|_| Fault::Compile {
        unit,
        log: format!(
            "`{unit}` is relocatable device code and its link step needs a current \
             CUDA context, which no device on this machine would give"
        ),
    })?;
    let mut state: dr::CUlinkState = std::ptr::null_mut();

    let code = unsafe {
        dr::cuLinkCreate_v2(
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &raw mut state,
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Fault::Device {
            call: "cuLinkCreate",
            code: code as i32,
        });
    }

    struct Link(dr::CUlinkState);
    impl Drop for Link {
        fn drop(&mut self) {
            unsafe { dr::cuLinkDestroy(self.0) };
        }
    }
    let link = Link(state);

    let name = CString::new(unit).unwrap_or_default();

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
        return Err(Fault::Device {
            call: "cuLinkAddData",
            code: code as i32,
        });
    }

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
        return Err(Fault::Device {
            call: "cuLinkAddFile",
            code: code as i32,
        });
    }

    let mut image: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut size = 0usize;

    let code = unsafe { dr::cuLinkComplete(link.0, &raw mut image, &raw mut size) };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Fault::Device {
            call: "cuLinkComplete",
            code: code as i32,
        });
    }
    if image.is_null() || size == 0 {
        return Err(Fault::Compile {
            unit,
            log: format!("`{unit}` linked to an empty image"),
        });
    }

    let linked = unsafe { std::slice::from_raw_parts(image.cast::<u8>(), size) }.to_vec();
    Ok(linked)
}

fn unassembled_tile_ir(
    unit: &'static str,
    cubin: &[u8],
    wants_function: bool,
) -> Result<(), Fault> {
    let has = |needle: &[u8]| cubin.windows(needle.len()).any(|w| w == needle);
    if wants_function && has(b".note.nv.tkinfo") && !has(b".text.") {
        return Err(Fault::Compile {
            unit,
            log: format!(
                "`{unit}` compiled to Tile IR, not SASS: the image carries \
                 `.note.nv.tkinfo` and no `.text`, so it would load and then answer \
                 `cuModuleGetFunction` with NOT_FOUND at the first launch. A tile \
                 unit needs its Tile IR assembled -- `tileiras` over \
                 `nvrtcGetTileIR`, with CUDA_ROOT set, or a driver new enough to \
                 assemble at load. See new-horizon.md 23.18"
            ),
        });
    }
    Ok(())
}

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

fn options(arch: &str, extra: &[&str]) -> Result<Vec<CString>, Fault> {
    if !arch.starts_with("sm_") {
        return Err(Fault::Compile {
            unit: "nvrtc",
            log: format!(
                "`{arch}` is not a real architecture: only `sm_XY` makes NVRTC emit SASS, \
                 and a virtual `compute_XY` would hand the driver PTX to JIT a second time \
                 at load"
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

        if unsafe { nvrtc::nvrtcGetProgramLogSize(self.0, &raw mut size) }
            != nvrtc::nvrtcResult::NVRTC_SUCCESS
            || size <= 1
        {
            return None;
        }
        let mut buf = vec![0u8; size];

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
        unsafe { nvrtc::nvrtcDestroyProgram(&raw mut self.0) };
    }
}
