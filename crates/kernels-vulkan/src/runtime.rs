#[cfg(feature = "runtime")]
use kernels::jit::Root;

include!(concat!(env!("OUT_DIR"), "/sources.rs"));

#[must_use]
pub fn source(file: &str) -> Option<&'static str> {
    SOURCES
        .binary_search_by_key(&file, |&(name, _)| name)
        .ok()
        .map(|i| SOURCES[i].1)
}

#[must_use]
pub fn sources() -> &'static [(&'static str, &'static str)] {
    SOURCES
}

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

            std::fs::write(&path, text)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        Ok(root)
    })
    .as_ref()
    .map(|p| p.as_path())
    .map_err(|e| Failed(e.clone()))
}

#[cfg(feature = "runtime")]
pub mod native {
    use super::{Root, source};
    use crate::Capability;
    use kernels::jit::Compiles;
    use shader_slang as slang;
    use slang::Downcast;

    #[derive(Debug, Clone, Copy)]
    pub struct Slang;

    pub type Spirv = Vec<u32>;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Failed(pub String);

    impl core::fmt::Display for Failed {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl Compiles for Slang {
        type Headers = Capability;

        type Toolchain = &'static str;
        type Entry = Spirv;
        type Error = Failed;

        fn arch() -> Option<&'static str> {
            Some("spirv")
        }

        fn headers_key(headers: Self::Headers) -> (&'static str, u64) {
            (headers.tag(), headers as u64)
        }

        fn admits(_root: &Root<Self>, point: &str) -> Result<(), Self::Error> {
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

            Ok((image, point.to_string()))
        }

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
                Some(&0x0723_0203) => Ok(words),
                Some(&other) => Err(Failed(format!(
                    "not SPIR-V: first word is {other:#010x}, want 0x07230203"
                ))),
                None => Err(Failed("empty image".to_string())),
            }
        }
    }

    fn compile_spirv(root: &Root<Slang>, point: &str) -> Result<Vec<u32>, Failed> {
        let global = slang::GlobalSession::new()
            .ok_or_else(|| Failed("cannot create a Slang global session".to_string()))?;

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
