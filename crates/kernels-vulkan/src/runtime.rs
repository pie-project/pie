//! Compiling a point from Slang at run time, instead of shelling `slangc` for
//! every point of every tier at build time.
//!
//! ## What moved, and what did not
//!
//! The compiler is the same compiler. `slangc` is a thin program over
//! `libslang`, and the options this passes are the ones [`crate::build`]'s
//! command line passed: `-target spirv`, `-stage compute`, `-entry main`,
//! `-fvk-use-entrypoint-name`, `-emit-spirv-directly`, `-allow-glsl`, `-I` the
//! tree, `-DPIE_ENTRYPOINT=...`, the point's own defines, and `-O0` for the
//! cooperative-matrix tier against `-O2` elsewhere. Same library, same
//! switches, same SPIR-V. What changed is WHEN, and therefore WHICH points get
//! compiled: a build compiled all 681 the tree declares, a run compiles the
//! ones a model fires.
//!
//! ## Why this backend was the odd one
//!
//! `Fire::at`'s first argument names a SOURCE on the other three planes — a
//! `.cuh`, a `.metal`, a `.wgsl` — and named a compiled ARTIFACT here, a
//! `.spv` whose stem carried the device tier. That is what
//! `crate::module::path` was for: given an entrypoint and the device's
//! ceiling, walk down [`Capability::PREFERENCE`] until you find a stem this
//! BUILD happened to produce, and hand the body that string.
//!
//! So a body had to know a fact about the build in order to name the thing it
//! wanted to run, and the tier — which is a property of the DEVICE — was
//! resolved by string search over a table of file names. With the compile at
//! run time the tier is what it always was, a set of defines, and it goes
//! where the point's other defines already are: [`Root::options`].
//!
//! ## The cost is real and it is paid once
//!
//! Measured on this tree, `quant/qmm_t.slang` (the largest, 461 declared
//! points) compiles in **~182 ms per point**, and the second point costs the
//! same as the first — it is per-point work, not a one-time session warmup.
//! That is why this goes through [`kernels::jit`] rather than calling Slang
//! directly: the shared layer memoises per point in the process and, past
//! that, on disk, so a second run of the same model pays deserialisation
//! instead. The disk cache is keyed on the source text, so editing a shader
//! invalidates exactly the points that read it.
//!
//! ## What it costs to ship
//!
//! `libslang` is ~46 MB against the ~5.5 MB of SPIR-V it replaces. That is the
//! honest trade and it is not obviously a win on size alone. It buys three
//! things: a build that no longer needs `slangc` on `PATH` to produce a
//! WORKING binary, a `.slang` tree that can gain a point without a build
//! script learning about it, and — the reason the other three planes look like
//! this — bodies that name a source and let the compile be someone else's job.

#[cfg(feature = "runtime")]
use kernels::jit::Root;

include!(concat!(env!("OUT_DIR"), "/sources.rs"));

/// The `.slang` text under this tree-relative path.
///
/// The path is what a body writes in [`crate::routine::Fire::at`], and what a
/// `#include` inside a shader spells, so one lookup answers both.
#[must_use]
pub fn source(file: &str) -> Option<&'static str> {
    SOURCES
        .binary_search_by_key(&file, |&(name, _)| name)
        .ok()
        .map(|i| SOURCES[i].1)
}

/// Every `.slang` in the tree, as `(path, text)`.
///
/// Sorted by path, which is what [`source`] binary-searches.
#[must_use]
pub fn sources() -> &'static [(&'static str, &'static str)] {
    SOURCES
}

/// The carried tree, on disk, where a compiler can walk its `#include`s.
///
/// Written once per process into the same cache root [`kernels::jit`] keys its
/// images under, and CONTENT-ADDRESSED: the directory name is a digest of
/// every source, so a tree that changes lands somewhere else and a stale one
/// is never read. Two processes racing write the same bytes to the same paths.
///
/// # Errors
///
/// If the cache root cannot be determined or written.
#[cfg(feature = "runtime")]
pub fn materialise() -> Result<&'static std::path::Path, native::Failed> {
    use native::Failed;
    use std::sync::OnceLock;

    static TREE: OnceLock<Result<std::path::PathBuf, String>> = OnceLock::new();

    TREE.get_or_init(|| {
        let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
        for (name, text) in SOURCES {
            for byte in name.as_bytes().iter().chain(text.as_bytes()) {
                digest ^= u64::from(*byte);
                digest = digest.wrapping_mul(0x1000_0000_01b3);
            }
        }

        let root = std::env::var_os("PIE_CACHE")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from))
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
            })
            .ok_or_else(|| {
                "no PIE_CACHE, XDG_CACHE_HOME or HOME to hold the shader tree".to_string()
            })?
            .join("pie")
            .join(format!("slang/{digest:016x}"));

        for (name, text) in SOURCES {
            let path = root.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            // Written unconditionally rather than only when absent: the cost
            // is a 392 KB write once per process, and the alternative is
            // trusting that whatever is already at this path is what the
            // digest says it is.
            std::fs::write(&path, text)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        Ok(root)
    })
    .as_ref()
    .map(|p| p.as_path())
    .map_err(|e| Failed(e.clone()))
}

/// The runtime Slang compiler, as [`kernels::jit`] sees it.
///
/// Only under `native`, for the reason the feature exists: the portable half
/// of this crate is a signature table, and linking a shader compiler into
/// `model-ir` would undo that. Without the feature the sources above are still
/// carried — declaring is not compiling — and nothing can turn them into
/// SPIR-V.
#[cfg(feature = "runtime")]
pub mod native {
    use super::{Root, source};
    use crate::Capability;
    use kernels::jit::Compiles;
    use shader_slang as slang;
    use slang::Downcast;

    /// The Vulkan compile plane.
    ///
    /// A unit type and not a value: [`kernels::jit`] keys its per-point slots
    /// on this type, so one of these per backend is the whole point.
    #[derive(Debug, Clone, Copy)]
    pub struct Slang;

    /// What a fire is issued against, once a point is compiled.
    ///
    /// The SPIR-V words, not a `vk::ShaderModule`: this crate does not own a
    /// device — `driver-vulkan` does — and a module is a device object. The
    /// shared layer memoises the expensive half (Slang, ~182 ms) and the
    /// driver keeps its pipeline cache over the cheap half
    /// (`vkCreateShaderModule`, which is a copy and a header check).
    pub type Spirv = Vec<u32>;

    /// Slang's diagnostics, as a value.
    ///
    /// A `String` because [`kernels::jit`] requires `Clone` and Slang's own
    /// error is not — and because what a caller does with it is print it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Failed(pub String);

    impl core::fmt::Display for Failed {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl Compiles for Slang {
        /// The tier, which is what a Vulkan point's includes resolve
        /// *differently* under: `@coopmat` pulls in `linalg` and the
        /// cooperative-matrix types, `@fp16` real `float16_t`.
        type Headers = Capability;
        /// Slang's own release. A point compiled by an older one is a point
        /// compiled by a different compiler, which is what the floor is for.
        type Toolchain = &'static str;
        type Entry = Spirv;
        type Error = Failed;

        /// `"spirv"`, and the constant is the honest answer rather than a
        /// placeholder.
        ///
        /// [`kernels::jit::resolve`] reads `None` as "there is no device", and
        /// refuses — which is right for CUDA, where an image is built for the
        /// architecture in front of it and there is nothing to build without
        /// one. Vulkan has no such dependence: SPIR-V is the portable form,
        /// that is what it is FOR, and the device-specific compile happens
        /// inside the driver at `vkCreateComputePipelines`. A point compiles
        /// the same on a box with no GPU at all, which is what lets this
        /// crate's fixtures compile the tree in CI.
        ///
        /// So the term is a constant, and it stays IN the cache key: a key
        /// without it would collide across backends that share a cache root.
        fn arch() -> Option<&'static str> {
            Some("spirv")
        }

        fn headers_key(headers: Self::Headers) -> (&'static str, u64) {
            (headers.tag(), headers as u64)
        }

        /// The census is the declaration, so it is what a point is checked
        /// against — before the cache, which is why this is `admits` rather
        /// than a guard inside [`Self::compile`].
        ///
        /// `build.rs` passed `-DPIE_ENTRYPOINT=<name>` and this passes the same
        /// define, but NO SHADER IN THE TREE READS IT — 37 sources, zero
        /// occurrences. The name's only job is to be the string
        /// `vulkan_use_entry_point_name` stamps on the module's `main`, so a
        /// misspelled point compiles perfectly and produces a module under a name
        /// nothing will ever ask for.
        ///
        /// That was invisible while the build enumerated every point from the
        /// tree's own `// pie:instantiate` lines: a name that was not in that list
        /// was never compiled, so there was nothing to be wrong. Composing the
        /// name at a fire removes that check, and this puts it back where it can
        /// still be made — against `CENSUS`, which is those same directives, and
        /// is written whether or not this build has a compiler.
        ///
        /// Found by `an_unknown_point_refuses`, which asked for
        /// `..._gs_999_b_9_bm_9_bn_9` and got a valid SPIR-V module — and then
        /// kept getting one from the CACHE after the check was added in the
        /// wrong place, which is what `admits` exists to make impossible.
        fn admits(_root: &Root<Self>, point: &str) -> Result<(), Self::Error> {
            // The one way past this check, and it exists so that
            // `a_cached_unknown_point_still_refuses` can put a bogus image in
            // the cache the way the original accident did. Nothing else reads
            // it: a guard that cannot be defeated on purpose cannot be TESTED
            // for being defeated by accident.
            if std::env::var_os("PIE_VULKAN_SKIP_CENSUS").is_some() {
                return Ok(());
            }
            if crate::module::CENSUS.contains(&point) {
                Ok(())
            } else {
                Err(Failed(format!(
                    "`{point}` is not declared by any `// pie:instantiate` line \
                     in the shader tree, so nothing stamps a function under \
                     that name"
                )))
            }
        }

        /// Compile `point` out of `root`, as `slangc` did at build time.
        fn compile(
            root: &Root<Self>,
            point: &str,
            _arch: &str,
        ) -> Result<(Vec<u8>, String), Self::Error> {
            let words = compile_spirv(root, point)?;
            let mut image = Vec::with_capacity(words.len() * 4);
            for w in &words {
                image.extend_from_slice(&w.to_le_bytes());
            }
            // The entrypoint survives compilation, because
            // `vulkan_use_entry_point_name` is what makes it survive: without
            // it Slang emits `main` and the name a body asked for is lost.
            Ok((image, point.to_string()))
        }

        /// The image, as words.
        ///
        /// SPIR-V is a `u32` stream and `vkCreateShaderModule` wants it as
        /// one, so the alignment is checked HERE, once, rather than at each
        /// fire. A cached image is bytes on disk; this is where it becomes a
        /// module again.
        fn load(
            _root: &Root<Self>,
            image: &[u8],
            _mangled: &str,
        ) -> Result<Self::Entry, Self::Error> {
            if image.len() % 4 != 0 {
                return Err(Failed(format!(
                    "SPIR-V is a stream of 32-bit words and this image is {} bytes",
                    image.len()
                )));
            }
            let words: Vec<u32> = image
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            match words.first() {
                // 0x07230203, little-endian, as the spec's first word.
                Some(&0x0723_0203) => Ok(words),
                Some(&other) => Err(Failed(format!(
                    "not SPIR-V: first word is {other:#010x}, want 0x07230203"
                ))),
                None => Err(Failed("empty image".to_string())),
            }
        }
    }

    /// One Slang compile, with the options `build.rs` used to pass `slangc`.
    fn compile_spirv(root: &Root<Slang>, point: &str) -> Result<Vec<u32>, Failed> {
        let global = slang::GlobalSession::new()
            .ok_or_else(|| Failed("cannot create a Slang global session".to_string()))?;

        // `-O0` for the cooperative-matrix tier, `-O2` elsewhere. Not a
        // preference: ggml #15344 is a spirv-opt miscompile of coopmat code,
        // and Slang runs spirv-opt through the same library `slangc` does, so
        // it is the same miscompile. The build script carried this and the
        // reason travels with it.
        let optimization = if root.headers == Capability::Coopmat {
            slang::OptimizationLevel::None
        } else {
            slang::OptimizationLevel::High
        };

        let mut options = slang::CompilerOptions::default()
            .optimization(optimization)
            .vulkan_use_entry_point_name(true)
            .emit_spirv_directly(true)
            .matrix_layout_row(true)
            .macro_define("PIE_ENTRYPOINT", point);
        for define in root.options {
            let (name, value) = define.split_once('=').unwrap_or((define, "1"));
            options = options.macro_define(name, value);
        }

        let target = slang::TargetDesc::default()
            .format(slang::CompileTarget::Spirv)
            .profile(global.find_profile("spirv_1_5"));
        let targets = [target];

        // # The tree goes to disk, once, and why that is not a retreat
        //
        // `shader-slang` 0.1.0 loads a module by NAME through a search path;
        // it has no source-string entry point. That is not merely a gap in the
        // binding: a `.slang` root `#include`s others, and resolving those is
        // the compiler walking a search path itself. Handing it one file's
        // text would still leave it looking for `common/bf16.slang` on disk.
        //
        // So the sources are carried in the BINARY and materialised into the
        // cache directory on first use. The deployment property is the one
        // `crate::module` argues for and it is unchanged: what ships is the
        // executable, nothing has to be installed beside it, and a machine
        // that never had the source tree can still compile a point. The
        // directory is a CACHE — derived, reproducible, and re-created if it
        // is deleted — rather than an input someone has to put there.
        let tree = crate::runtime::materialise()?;
        let tree = std::ffi::CString::new(tree.to_string_lossy().as_ref())
            .map_err(|e| Failed(format!("the cache path is not a C string: {e}")))?;
        let paths = [tree.as_ptr()];

        let session = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&paths)
            .options(&options);
        let session = global
            .create_session(&session)
            .ok_or_else(|| Failed("cannot create a Slang session".to_string()))?;

        let module = session
            .load_module(root.file)
            .map_err(|e| Failed(format!("{}: {e}", root.file)))?;
        let entry = module
            .find_entry_point_by_name("main")
            .ok_or_else(|| Failed(format!("{}: no `main` entry point", root.file)))?;
        let program = session
            .create_composite_component_type(&[module.downcast().clone(), entry.downcast().clone()])
            .map_err(|e| Failed(format!("{}: {e}", root.file)))?;
        let linked = program
            .link()
            .map_err(|e| Failed(format!("{point}: {e}")))?;
        let code = linked
            .entry_point_code(0, 0)
            .map_err(|e| Failed(format!("{point}: {e}")))?;

        let bytes = code.as_slice();
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// A root for one `.slang` file, at one tier, with one point's defines.
    ///
    /// This is what replaces `crate::module::path`. A body named an ARTIFACT
    /// there — `affine_qmm_t_..._bm_32_bn_32.coopmat.spv`, a string it got by
    /// asking which stems the build happened to emit — and names a SOURCE
    /// here. The axes that were spelled into that file name are `options`, and
    /// the tier is `tier`.
    ///
    /// `options` is `&'static` because a `Root` is, and because the set is
    /// finite: `crate::quant`'s composers build these from the same axis
    /// constants they build names from.
    #[must_use]
    pub fn root(
        file: &'static str,
        tier: Capability,
        options: &'static [&'static str],
    ) -> Option<Root<Slang>> {
        Some(Root {
            name: file,
            text: source(file)?,
            file,
            options,
            headers: tier,
            floor: "2026.14.1",
        })
    }
}
