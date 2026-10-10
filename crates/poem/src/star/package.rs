//! A model package: its files, each evaluated with the globals of its stage
//! and frozen, so a deployment's layout, forward and formats are functions
//! a trace or an import calls.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use starlark::environment::{FrozenModule, Globals, GlobalsBuilder, LibraryExtension, Module};
use starlark::eval::{Evaluator, ReturnFileLoader};
use starlark::syntax::{AstModule, Dialect};
use starlark::values::FrozenHeapName;

use crate::star::manifest::Manifest;

/// What a file of a package may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `package.poem`: the models a package holds and the deployments of
    /// them it lists.
    Manifest,
    /// `model.poem`: a deployment's dims and the weights they lay out.
    Layout,
    /// `forward.poem`: the caches a deployment holds and the forward its
    /// rows run.
    Forward,
    /// `formats.poem`: the formats a deployment's checkpoints are read in.
    Formats,
    /// Any other file: pure helpers every stage may load.
    Lib,
}

/// The prefix a library shared between packages is loaded under: a stage
/// file of `//lib/<name>/` is evaluated in its stage, and only that stage's
/// files may load it.
pub const LIBRARY: &str = "//lib/";

impl Stage {
    /// The stage a package file named `file` is evaluated in.
    #[must_use]
    pub fn of(file: &str) -> Stage {
        match file.rsplit('/').next().unwrap_or(file) {
            "package.poem" => Stage::Manifest,
            "model.poem" => Stage::Layout,
            "forward.poem" => Stage::Forward,
            "formats.poem" => Stage::Formats,
            _ => Stage::Lib,
        }
    }

    fn globals(self) -> &'static Globals {
        static LIB: LazyLock<Globals> = LazyLock::new(|| base().build());
        static MANIFEST: LazyLock<Globals> =
            LazyLock::new(|| base().with(crate::star::manifest::manifest).build());
        static LAYOUT: LazyLock<Globals> = LazyLock::new(|| {
            base()
                .with(crate::star::layout::layout)
                .with(crate::star::generative::generative)
                .build()
        });
        static FORWARD: LazyLock<Globals> = LazyLock::new(|| {
            base()
                .with(crate::star::forward::forward)
                .with(crate::star::forward::ops)
                .build()
        });
        static FORMATS: LazyLock<Globals> =
            LazyLock::new(|| base().with(crate::star::formats::formats).build());
        match self {
            Stage::Manifest => &MANIFEST,
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
    .with(crate::star::layout::numbers)
    .with(crate::star::layout::dtypes)
}

/// The version of the builtins a package is written against. An artifact
/// records the one its package was imported under, and a build serves it only
/// at the same one: a package is code, and code written against other
/// builtins does not mean what it meant.
pub const API: u32 = 2;

/// The attribute prefix an artifact carries its package under.
pub const ATTRIBUTE: &str = "pie.package/";

/// A model package, its files frozen.
pub struct Package {
    name: String,
    sources: BTreeMap<String, String>,
    modules: BTreeMap<String, FrozenModule>,
    manifest: Manifest,
}

impl Package {
    /// The package `name` of `files` (file name, source).
    pub fn new(name: &str, files: &[(&str, &str)]) -> anyhow::Result<Package> {
        let sources: BTreeMap<&str, &str> = files.iter().copied().collect();
        let mut modules = BTreeMap::new();
        for file in sources.keys() {
            freeze(name, file, &sources, &mut modules, &mut Vec::new())?;
        }
        let manifest = match modules.get("package.poem") {
            Some(module) => Manifest::of(name, module)?,
            None => Manifest::default(),
        };
        Ok(Package {
            name: name.to_string(),
            sources: sources
                .iter()
                .map(|(f, s)| ((*f).to_string(), (*s).to_string()))
                .collect(),
            modules,
            manifest,
        })
    }

    /// The package an artifact carries in its attributes `(key, text)`, if
    /// it carries one; refused if it was written against other builtins.
    pub fn from_attributes<'a>(
        attributes: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> anyhow::Result<Option<Package>> {
        let mut name = None;
        let mut api = None;
        let mut files = Vec::new();
        for (key, text) in attributes {
            let Some(key) = key.strip_prefix(ATTRIBUTE) else {
                continue;
            };
            match key {
                "name" => name = Some(text),
                "api" => api = Some(text),
                _ => match key.strip_prefix("files/") {
                    Some(file) => files.push((file, text)),
                    None => anyhow::bail!("an artifact's package states `{ATTRIBUTE}{key}`"),
                },
            }
        }
        let Some(name) = name else {
            if files.is_empty() && api.is_none() {
                return Ok(None);
            }
            anyhow::bail!("an artifact carries a package's files and not its name");
        };
        if api != Some(API.to_string().as_str()) {
            anyhow::bail!(
                "the artifact's package `{name}` is written against builtins version {} and \
                 this build's are version {API}; import the checkpoint again",
                api.unwrap_or("(none)")
            );
        }
        Package::new(name, &files).map(Some)
    }

    /// The attributes an artifact carries this package in.
    #[must_use]
    pub fn attributes(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::from([
            (format!("{ATTRIBUTE}name"), self.name.clone()),
            (format!("{ATTRIBUTE}api"), API.to_string()),
        ]);
        for (file, source) in &self.sources {
            out.insert(format!("{ATTRIBUTE}files/{file}"), source.clone());
        }
        out
    }

    /// What the package states of itself in `package.poem`.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
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
        let shared = loaded.starts_with(LIBRARY) && Stage::of(loaded) == Stage::of(file);
        if Stage::of(loaded) != Stage::Lib && !shared {
            anyhow::bail!(
                "`{package}/{file}` loads `{loaded}`, a stage of its own; a stage loads only \
                 helper files and a library's files of its own stage"
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
            .freeze_named(FrozenHeapName::User(Box::new(format!("{package}/{file}"))))
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    })?;
    modules.insert(file.to_string(), frozen);
    Ok(())
}
