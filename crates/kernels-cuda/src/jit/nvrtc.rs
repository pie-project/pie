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

    let wants_function = job.wanted.iter().any(|w| !w.trim_start().starts_with('&'));
    unassembled_tile_ir(unit, &cubin, wants_function)?;

    Ok(Built {
        cubin,
        lowered,
        log,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::{Features, Root, Toolchain};

    /// The units that include the fp8 shim directly, each named with a kernel
    /// that converts fp8 and one that does not. The bf16 one is the point:
    /// NVRTC lowers the whole unit, so a shim that refuses an architecture
    /// takes every kernel in the unit with it, not only the fp8 ones.
    const FP8_UNITS: &[(&str, &[&str])] = &[
        (
            "attn/kv_append.cu",
            &[
                "::pie::attn::kv_append_explicit<::pie::true_type::value>",
                "::pie::attn::kv_append_fp8_per_tensor",
                "::pie::attn::kv_append_per_token_head<::pie::true_type::value>",
            ],
        ),
        (
            "attn/kv_dequant.cu",
            &[
                "::pie::attn::dequant_fp8_pages_active",
                "::pie::attn::dequant_fp8_per_token_head_pages_active<::pie::bf16>",
            ],
        ),
        (
            "linear/act_quant.cu",
            &["::pie::linear::quant_act_fp8_per_group"],
        ),
    ];

    fn lowers(file: &'static str, arch: &str, extra: &[&str]) -> Result<(), String> {
        let (_, wanted) = FP8_UNITS
            .iter()
            .find(|(name, _)| *name == file)
            .expect("listed above");
        let root = Root::of(file).expect("a carried unit");
        let mut options: Vec<&str> = root.options.to_vec();
        options.extend_from_slice(extra);
        let wanted: Vec<String> = wanted.iter().map(|&w| w.to_owned()).collect();
        compile_text(&Job {
            name: root.name,
            source: root.text.to_owned(),
            arch,
            options: &options,
            headers: root.header_set(),
            floor: Toolchain::ANY,
            wanted: &wanted,
        })
        .map(|_| ())
        .map_err(|why| why.to_string())
    }

    fn nvrtc_is_here() -> bool {
        // SAFETY: probing for the library only tries to load it; no symbol
        // from it is called.
        let present = unsafe { nvrtc::is_culib_present() };
        if !present {
            eprintln!("skipped: libnvrtc is not loadable on this box");
        }
        present
    }

    /// What the arch-tier units are held to past their non-template code:
    /// nothing but a B200 or an H100 compiles them at run time, where a unit
    /// that will not compile declines to the portable path in silence, and
    /// NVRTC checks a template's body only where something instantiates it.
    /// Each names at least every `__global__` of its unit, as the launches
    /// spell them.
    const INSTANTIATED: &[(&str, &[&str])] = &[
        (
            "linear/gemm_sm100.cu",
            &[
                "::pie::linear::tc5_gemm<0, 1, 1, 0, 0, 128, 4, 0, 1, 0, 0, 0, 0>",
                "::pie::linear::tc5_finalize",
                "::pie::linear::tc5_finalize_routed",
            ],
        ),
        (
            "attn/prefill_sm100.cu",
            &[
                "::pie::attn::fmha::tc5_fmha_prefill<128, 128, 3, 128>",
                "::pie::attn::fmha::tc5_fmha_prefill<256, 64, 3, 256>",
                "::pie::attn::fmha::tc5_fmha_prefill_wide<512, 128, 0, 2, 3>",
                "::pie::attn::fmha::tc5_fmha_prefill_wide<256, 128, 0, 2, 5>",
            ],
        ),
        (
            "attn/compressed/prefill_sm100.cu",
            &[
                "::pie::attn::compressed::sm100::sparse<true>",
                "::pie::attn::compressed::sm100::sparse<false>",
                "::pie::attn::compressed::sm100::sparse<true, 1, 1>",
                "::pie::attn::compressed::sm100::stage_pages<1>",
            ],
        ),
        (
            "attn/index/logits_sm100.cu",
            &[
                "::pie::attn::index::index_logits_tc5<32>",
                "::pie::attn::index::index_logits_tc5<64>",
                "::pie::attn::index::index_select_rows<true>",
                "::pie::attn::index::index_select_rows<false>",
            ],
        ),
        (
            "attn/index/logits_fp4_sm100.cu",
            &["::pie::attn::index::index_logits_fp4<64>"],
        ),
        (
            "attn/mla/sparse_sm100.cu",
            &[
                "::pie::attn::mla::mla_sparse<64, false>",
                "::pie::attn::mla::mla_sparse<64, true>",
                "::pie::attn::mla::mla_sparse_fold",
                "::pie::attn::mla::mla_sparse_e4<64>",
            ],
        ),
        (
            "ssm/gdn_chunk_sm100.cu",
            &["::pie::ssm::sm100::gdn_chunked"],
        ),
        (
            "linear/moe/select_fp4_sm100.cu",
            &["::pie::linear::moe_select_mxfp4_cvt<2, 1, 2, 4, 3, 1>"],
        ),
        (
            "elemwise/hc_sm100.cu",
            // (hidden: DeepSeek-V4-Flash's and V4's, the held e4m3 operand or
            // not): the forms `hc::fold_pre` launches.
            &[
                "::pie::elemwise::hc_fold_pre_tc5<4096, true>",
                "::pie::elemwise::hc_fold_pre_tc5<4096, false>",
                "::pie::elemwise::hc_fold_pre_tc5<7168, true>",
                "::pie::elemwise::hc_fold_pre_tc5<7168, false>",
            ],
        ),
        (
            "linear/moe/mega_sm100.cu",
            // (kind: W4A8 or bf16, block width 32, Glu code, weights first,
            // the five-stage ring 32 fits): every form `moe::mega` launches.
            &[
                "::pie::linear::mega_moe<1, 32, 2, 1, 5>",
                "::pie::linear::mega_moe<0, 32, 1, 0, 5>",
                "::pie::linear::mega_moe<0, 32, 1, 1, 5>",
                "::pie::linear::mega_moe<0, 32, 2, 0, 5>",
                "::pie::linear::mega_moe<0, 32, 2, 1, 5>",
            ],
        ),
        (
            "linear/moe/route_sm100.cu",
            &["::pie::linear::moe_gate_tc5<1, 64, 2, 2, 0, true>"],
        ),
        (
            "linear/moe/select_sm90.cu",
            &[
                "::pie::linear::moe_select_fp8a8_wgmma<64>",
                "::pie::linear::moe_select_bf16_wgmma<16, 1>",
                "::pie::linear::moe_select_bf16_wgmma<32, 2>",
                "::pie::linear::moe_select_fp4_wgmma<1, false, 3>",
                "::pie::linear::moe_select_bf16_wsgemm<4, 1>",
            ],
        ),
        (
            "linear/w8a16_sm90.cu",
            &["::pie::linear::fp8_w8a16_wgmma<128, 3>"],
        ),
        (
            "linear/act_quant.cu",
            &["::pie::linear::act_quant_nvfp4<0, false>"],
        ),
    ];

    /// The `__global__` functions `text` defines, by name.
    fn globals(text: &str) -> Vec<&str> {
        let mut names = Vec::new();
        for (at, _) in text.match_indices("__global__") {
            let line = text[..at].rsplit('\n').next().unwrap_or("");
            if line.trim_start().starts_with("//") {
                continue;
            }
            let mut rest = &text[at + "__global__".len()..];
            loop {
                rest = rest.trim_start();
                let word = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                if word == 0 {
                    break;
                }
                let (name, after) = rest.split_at(word);
                if !after.trim_start().starts_with('(') {
                    rest = after;
                    continue;
                }
                if name != "__launch_bounds__" {
                    names.push(name);
                    break;
                }
                let mut depth = 0;
                let close = after
                    .char_indices()
                    .find(|&(_, c)| {
                        depth += i32::from(c == '(') - i32::from(c == ')');
                        c == ')' && depth == 0
                    })
                    .map_or(after.len(), |(i, _)| i + 1);
                rest = &after[close..];
            }
        }
        names
    }

    /// The plain target a unit's tier names, by its file's suffix: the
    /// arch-specific twin's chip for `_sm100` and `_sm90`, sm_80 (the
    /// oldest a unit is compiled for) for `_sm80` and the generic units.
    fn tier(file: &str) -> &'static str {
        if file.contains("_sm100") {
            "sm_100"
        } else if file.contains("_sm90") {
            "sm_90"
        } else {
            "sm_80"
        }
    }

    /// Every unit under `kernels/` compiles for each target it is
    /// registered for, as `cache::load` would compile it there: its
    /// options, its headers, its NVRTC floor and the target
    /// [`Root::target`] picks for its tier's chip. A unit with no
    /// instantiation asked of it is checked for its text and non-template
    /// code; each in [`INSTANTIATED`] is lowered as its launch names it.
    #[test]
    fn every_unit_lowers() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        if !nvrtc_is_here() {
            return;
        }
        let nvrtc = version().expect("libnvrtc answers its version");
        // A unit another unit includes is a header: it compiles as part of
        // its includers, under their options, not on its own.
        let included: std::collections::BTreeSet<&str> = crate::source::LIBRARY
            .iter()
            .flat_map(|h| h.text.lines())
            .filter_map(|line| line.strip_prefix("#include \"")?.split('"').next())
            .collect();
        let mut failed = Vec::new();
        let mut jobs: Vec<(Root, String, Option<&str>)> = Vec::new();
        for header in crate::source::LIBRARY {
            if included.contains(header.name) {
                continue;
            }
            let Some(root) = Root::of(header.name) else {
                continue;
            };
            let wanted = INSTANTIATED
                .iter()
                .find(|(file, _)| *file == header.name)
                .map_or(&[][..], |(_, wanted)| *wanted);
            if root.specific {
                for name in globals(root.text) {
                    let named = |w: &&str| {
                        w.contains(&format!("::{name}<")) || w.ends_with(&format!("::{name}"))
                    };
                    if !wanted.iter().any(named) {
                        failed.push(format!(
                            "{}: `{name}` is instantiated nowhere in INSTANTIATED",
                            header.name
                        ));
                    }
                }
            }
            let mut plains = vec![tier(header.name)];
            if root.tcgen05_twin {
                plains.push("sm_100");
            }
            for plain in plains {
                let Some(arch) = root.target(plain, Features::of(plain, nvrtc).tcgen05) else {
                    failed.push(format!(
                        "{}: {plain} has no arch-specific twin",
                        header.name
                    ));
                    continue;
                };
                jobs.push((root, arch.clone(), None));
                jobs.extend(wanted.iter().map(|&w| (root, arch.clone(), Some(w))));
            }
        }
        let next = AtomicUsize::new(0);
        let refused = Mutex::new(Vec::new());
        let workers = std::thread::available_parallelism().map_or(8, usize::from);
        std::thread::scope(|scope| {
            for _ in 0..workers.min(jobs.len()) {
                scope.spawn(|| {
                    while let Some((root, arch, wanted)) =
                        jobs.get(next.fetch_add(1, Ordering::Relaxed))
                    {
                        let wanted: Vec<String> = wanted.iter().map(|&w| w.to_owned()).collect();
                        let built = compile_text(&Job {
                            name: root.name,
                            source: root.text.to_owned(),
                            arch,
                            options: root.options,
                            headers: root.header_set(),
                            floor: root.floor,
                            wanted: &wanted,
                        });
                        if let Err(why) = built {
                            let what = wanted.first().map_or("", String::as_str);
                            refused
                                .lock()
                                .expect("a worker panicked")
                                .push(format!("{} {what} at {arch}: {why}", root.file));
                        }
                    }
                });
            }
        });
        let mut refused = refused.into_inner().expect("a worker panicked");
        refused.sort();
        failed.extend(refused);
        assert!(failed.is_empty(), "{}", failed.join("\n\n"));
    }

    #[test]
    fn the_arch_specific_and_family_targets_are_real_and_a_virtual_one_is_not() {
        for real in [
            "sm_80", "sm_100", "sm_100a", "sm_103a", "sm_120a", "sm_100f",
        ] {
            assert!(options(real, &[]).is_ok(), "{real} emits SASS");
        }
        for not in ["compute_100a", "sm_", "sm_a", "sm_100b", "sm_10x"] {
            assert!(options(not, &[]).is_err(), "{not} is no target");
        }
    }

    /// Every tcgen05 wrapper, issued once: what the arch-specific target
    /// must take and the plain one must refuse.
    const TCGEN05_PROBE: &str = r#"
#include "prim/mma_sm100.cuh"
namespace pie {
__global__ void tc5_probe(const __grid_constant__ tc5::TmaMap map, const void* src, unsigned* out) {
    using namespace tc5;
    __shared__ alignas(1024) unsigned char tile[1024];
    __shared__ u64 bar;
    __shared__ u32 slot;
    if (threadIdx.x == 0) {
        bar_init(&bar, 1);
        bar_init_fence();
        bar_arrive_expect_tx(&bar, 1024);
        tma_load_2d(tile, &bar, &map, 0, 0);
        bulk_load(tile, src, 512, &bar);
        bar_wait(&bar, 0);
    }
    tmem_alloc(&slot, 32);
    tmem_relinquish();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 t = slot;
    const u64 d = sdesc_sw128(smem_addr(tile));
    if (threadIdx.x == 0) {
        tc_cp_sf_atom(t, sdesc_sf(smem_addr(tile)));
        mma_f16(t, d, d, idesc_dense(kF16Bf16, kF16Bf16, 128, 32), false);
        mma_f8f6f4(t, d, d, idesc_dense(kE4M3, kE4M3, 128, 32), true);
        mma_mxf8f6f4(t, d, d, idesc_scaled(kE4M3, kE2M1, 128, 32, true), true, t, t);
        mma_nvf4(t, d, d, idesc_scaled(kNvE2M1, kNvE2M1, 128, 32, false), true, t, t);
        tc_commit(&bar);
    }
    u32 v[32];
    tmem_ld32(t, v);
    tmem_wait_ld();
    out[threadIdx.x] = v[threadIdx.x % 32];
    tmem_dealloc(t, 32);
}
}
"#;

    /// tcgen05 exists at the arch-specific target alone: the wrappers lower
    /// at sm_100a and sm_103a, and the plain sm_100 refuses their PTX.
    #[test]
    fn the_tcgen05_wrappers_lower_at_the_arch_specific_targets_only() {
        if !nvrtc_is_here() || !Features::NVRTC.met_by(version().unwrap_or(Toolchain::ANY)) {
            return;
        }
        let wanted = vec!["::pie::tc5_probe".to_owned()];
        let job = |arch| Job {
            name: "tc5_probe",
            source: TCGEN05_PROBE.to_owned(),
            arch,
            options: &[],
            headers: crate::source::DEVICE_HEADERS,
            floor: Features::NVRTC,
            wanted: &wanted,
        };
        for arch in ["sm_100a", "sm_103a"] {
            if let Err(why) = compile_text(&job(arch)) {
                panic!("the tcgen05 unit would not lower at {arch}: {why}");
            }
        }
        assert!(
            compile_text(&job("sm_100")).is_err(),
            "the plain target took tcgen05"
        );
    }

    /// sm_80 has no fp8 cvt and takes the portable path; sm_89 is where the
    /// instruction arrives; sm_120 is the box this crate is developed on.
    /// Needs libnvrtc, not a device: the compile targets an architecture.
    #[test]
    fn the_fp8_shim_lowers_below_sm_89() {
        if !nvrtc_is_here() {
            return;
        }
        for arch in ["sm_80", "sm_89", "sm_120"] {
            for (file, _) in FP8_UNITS {
                if let Err(why) = lowers(file, arch, &[]) {
                    panic!("`{file}` would not lower at {arch}: {why}");
                }
            }
        }
    }

    /// The portable path is what sm_80 runs, so it has to lower on the
    /// architecture that has the hardware cvt too: that is how it is held
    /// against the instruction on a box that has one.
    #[test]
    fn the_portable_fp8_path_lowers_where_the_hardware_one_does() {
        if !nvrtc_is_here() {
            return;
        }
        for (file, _) in FP8_UNITS {
            if let Err(why) = lowers(file, "sm_89", &["-DPIE_HALFTYPE_FORCE_PORTABLE"]) {
                panic!("`{file}` would not lower portably at sm_89: {why}");
            }
        }
    }

    /// The affine-u4 arms are chosen by shape, never by capability, so
    /// sm_90 keeps every arm sm_89 has: no `__CUDA_ARCH__` gate in those
    /// units, the prim headers or the shims may admit 890 and refuse 900 (#680).
    #[test]
    fn no_arch_gate_on_the_affine_u4_path_refuses_sm_90() {
        use crate::source::{LIBRARY, SHIM};

        let admits = |op: &str, bound: u32, arch: u32| match op {
            ">=" => arch >= bound,
            "<=" => arch <= bound,
            "==" => arch == bound,
            "!=" => arch != bound,
            ">" => arch > bound,
            _ => arch < bound,
        };
        let units = [
            "linear/affine.cu",
            "linear/dequant.cu",
            "linear/quant.cuh",
            "linear/codec.cuh",
        ]
        .map(|file| Root::of(file).expect("a carried unit"));
        let scanned = units.iter().map(|root| (root.name, root.text)).chain(
            SHIM.iter()
                .chain(LIBRARY.iter().filter(|h| h.name.starts_with("prim/")))
                .map(|h| (h.name, h.text)),
        );
        let mut gates = 0;
        for (name, text) in scanned {
            for (at, _) in text.match_indices("__CUDA_ARCH__") {
                let rest = text[at + "__CUDA_ARCH__".len()..].trim_start();
                let Some(op) = [">=", "<=", "==", "!=", ">", "<"]
                    .into_iter()
                    .find(|op| rest.starts_with(op))
                else {
                    continue;
                };
                let digits: String = rest[op.len()..]
                    .trim_start()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let Ok(bound) = digits.parse::<u32>() else {
                    continue;
                };
                gates += 1;
                assert!(
                    !admits(op, bound, 890) || admits(op, bound, 900),
                    "`{name}` gates on `__CUDA_ARCH__ {op} {bound}`, which admits sm_89 and \
                     refuses sm_90"
                );
            }
        }
        assert!(gates > 0, "the scan read no `__CUDA_ARCH__` gate at all");
    }
}
