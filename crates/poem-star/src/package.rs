//! A model package: its files, each evaluated with the globals of its stage
//! and frozen, so a deployment's layout, forward and formats are functions
//! a trace or an import calls.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use starlark::environment::{FrozenModule, Globals, GlobalsBuilder, LibraryExtension, Module};
use starlark::eval::{Evaluator, ReturnFileLoader};
use starlark::syntax::{AstModule, Dialect};
use starlark::values::FrozenHeapName;

/// What a file of a package may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `model.star`: a deployment's dims and the weights they lay out.
    Layout,
    /// `forward.star`: the caches a deployment holds and the forward its
    /// rows run.
    Forward,
    /// `formats.star`: the formats a deployment's checkpoints are read in.
    Formats,
    /// Any other file: pure helpers every stage may load.
    Lib,
}

impl Stage {
    /// The stage a package file named `file` is evaluated in.
    #[must_use]
    pub fn of(file: &str) -> Stage {
        match file {
            "model.star" => Stage::Layout,
            "forward.star" => Stage::Forward,
            "formats.star" => Stage::Formats,
            _ => Stage::Lib,
        }
    }

    fn globals(self) -> &'static Globals {
        static LIB: LazyLock<Globals> = LazyLock::new(|| base().build());
        static LAYOUT: LazyLock<Globals> =
            LazyLock::new(|| base().with(crate::layout::layout).build());
        static FORWARD: LazyLock<Globals> = LazyLock::new(|| {
            base()
                .with(crate::forward::forward)
                .with(crate::forward::ops)
                .build()
        });
        static FORMATS: LazyLock<Globals> =
            LazyLock::new(|| base().with(crate::formats::formats).build());
        match self {
            Stage::Layout => &LAYOUT,
            Stage::Forward => &FORWARD,
            Stage::Formats => &FORMATS,
            Stage::Lib => &LIB,
        }
    }
}

/// What every stage may use: Starlark's own library with structs, the
/// dtypes and float rounding.
fn base() -> GlobalsBuilder {
    GlobalsBuilder::extended_by(&[
        LibraryExtension::StructType,
        LibraryExtension::Map,
        LibraryExtension::Filter,
        LibraryExtension::Print,
    ])
    .with(crate::layout::numbers)
    .with(crate::layout::dtypes)
}

/// A model package, its files frozen.
pub struct Package {
    name: String,
    modules: BTreeMap<String, FrozenModule>,
}

impl Package {
    /// The package `name` of `files` (file name, source).
    pub fn new(name: &str, files: &[(&str, &str)]) -> anyhow::Result<Package> {
        let sources: BTreeMap<&str, &str> = files.iter().copied().collect();
        let mut modules = BTreeMap::new();
        for file in sources.keys() {
            freeze(name, file, &sources, &mut modules, &mut Vec::new())?;
        }
        Ok(Package {
            name: name.to_string(),
            modules,
        })
    }

    /// The package in the directory `dir`, named after it: every `.star`
    /// file in it.
    pub fn from_dir(dir: &std::path::Path) -> anyhow::Result<Package> {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("{} names no package", dir.display()))?
            .to_string();
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let Some(file) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if file.ends_with(".star") {
                files.push((file.to_string(), std::fs::read_to_string(&path)?));
            }
        }
        let files: Vec<(&str, &str)> = files
            .iter()
            .map(|(f, s)| (f.as_str(), s.as_str()))
            .collect();
        Package::new(&name, &files)
    }

    /// The package's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The frozen file `file`.
    pub(crate) fn module(&self, file: &str) -> anyhow::Result<&FrozenModule> {
        self.modules
            .get(file)
            .ok_or_else(|| anyhow::anyhow!("the package `{}` has no `{file}`", self.name))
    }
}

fn freeze(
    package: &str,
    file: &str,
    sources: &BTreeMap<&str, &str>,
    modules: &mut BTreeMap<String, FrozenModule>,
    loading: &mut Vec<String>,
) -> anyhow::Result<()> {
    if modules.contains_key(file) {
        return Ok(());
    }
    if loading.iter().any(|f| f == file) {
        anyhow::bail!(
            "`{package}/{file}` loads itself through {}",
            loading.join(" → ")
        );
    }
    let source = sources
        .get(file)
        .ok_or_else(|| anyhow::anyhow!("the package `{package}` has no `{file}` to load"))?;
    let ast = AstModule::parse(
        &format!("{package}/{file}"),
        (*source).to_string(),
        &Dialect::Extended,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    loading.push(file.to_string());
    for load in ast.loads() {
        let loaded = load.module_id;
        if Stage::of(loaded) != Stage::Lib {
            anyhow::bail!(
                "`{package}/{file}` loads `{loaded}`, a stage of its own; a stage loads only \
                 helper files"
            );
        }
        freeze(package, loaded, sources, modules, loading)?;
    }
    loading.pop();
    let loads: Vec<(String, &FrozenModule)> = ast
        .loads()
        .into_iter()
        .map(|load| (load.module_id.to_string(), &modules[load.module_id]))
        .collect();
    let loads_map = loads.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let loader = ReturnFileLoader {
        modules: &loads_map,
    };
    let globals = Stage::of(file).globals();
    let frozen = Module::with_temp_heap(|module| -> anyhow::Result<FrozenModule> {
        {
            let mut eval = Evaluator::new(&module);
            eval.set_loader(&loader);
            eval.eval_module(ast, globals)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        module
            .freeze_named(FrozenHeapName::user(format!("{package}/{file}")))
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    })?;
    modules.insert(file.to_string(), frozen);
    Ok(())
}
