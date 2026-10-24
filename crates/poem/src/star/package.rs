//! A model package: its files, each evaluated with the DSL's globals and
//! frozen, so a deployment's layout, forward and formats are functions a
//! trace or an import calls. A file's name says nothing of what it holds: the
//! package's entries (`MODELS`, `layout`, `caches`, `forward`, `formats`, …)
//! are found in whichever one file defines each, so a package may be one file
//! or several. What a stage's builtins read (the trace being recorded, the
//! checkpoint being read) is there only while the runtime calls that stage's
//! entry, and a builtin called outside it says so.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use starlark::environment::{FrozenModule, Globals, GlobalsBuilder, LibraryExtension, Module};
use starlark::eval::{Evaluator, ReturnFileLoader};
use starlark::syntax::{AstModule, Dialect};
use starlark::values::{FrozenHeapName, OwnedFrozenValue};

use crate::star::manifest::Manifest;

/// The globals every file of a package is evaluated with: Starlark's own
/// library, and every stage's builtins.
fn globals() -> &'static Globals {
    static ALL: LazyLock<Globals> = LazyLock::new(|| {
        base()
            .with(crate::star::manifest::manifest)
            .with(crate::star::layout::layout)
            .with(crate::star::generative::generative)
            .with(crate::star::forward::forward)
            .with(crate::star::forward::ops)
            .with(crate::star::formats::formats)
            .build()
    });
    &ALL
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

/// A model package, its files frozen.
pub struct Package {
    name: String,
    sources: BTreeMap<String, String>,
    modules: BTreeMap<String, FrozenModule>,
    /// What each file binds by `load`, which is another file's, not its own.
    loaded: BTreeMap<String, BTreeSet<String>>,
    manifest: Manifest,
}

impl Package {
    /// The package `name` of `files` (file name, source).
    pub fn new(name: &str, files: &[(&str, &str)]) -> anyhow::Result<Package> {
        let sources: BTreeMap<&str, &str> = files.iter().copied().collect();
        let mut modules = BTreeMap::new();
        let mut loaded = BTreeMap::new();
        for file in sources.keys() {
            freeze(
                name,
                file,
                &sources,
                &mut modules,
                &mut loaded,
                &mut Vec::new(),
            )?;
        }
        let manifest = match defining(name, &modules, &loaded, "MODELS")? {
            Some(file) => Manifest::of(name, &modules[file])?,
            None => Manifest::default(),
        };
        Ok(Package {
            name: name.to_string(),
            sources: sources
                .iter()
                .map(|(f, s)| ((*f).to_string(), (*s).to_string()))
                .collect(),
            modules,
            loaded,
            manifest,
        })
    }

    pub fn files(&self) -> impl Iterator<Item = (&str, &str)> {
        self.sources.iter().map(|(f, s)| (f.as_str(), s.as_str()))
    }

    /// A digest of the name and files: what an artifact records of its package.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.name.as_bytes());
        for (file, source) in &self.sources {
            hasher.update(&[0]);
            hasher.update(file.as_bytes());
            hasher.update(&[0]);
            hasher.update(source.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// What the package states of itself in its `MODELS` and `DEPLOYMENTS`.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The package's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The entry `name` (`layout`, `forward`, …): what the one file that
    /// defines it defines, or none if no file does.
    pub(crate) fn entry(&self, name: &str) -> anyhow::Result<Option<OwnedFrozenValue>> {
        Ok(
            match defining(&self.name, &self.modules, &self.loaded, name)? {
                Some(file) => Some(self.modules[file].get(name)?),
                None => None,
            },
        )
    }

    /// The entry `name`, which the package must define.
    pub(crate) fn required(&self, name: &str) -> anyhow::Result<OwnedFrozenValue> {
        self.entry(name)?
            .ok_or_else(|| anyhow::anyhow!("the package `{}` defines no `{name}`", self.name))
    }
}

fn freeze(
    package: &str,
    file: &str,
    sources: &BTreeMap<&str, &str>,
    modules: &mut BTreeMap<String, FrozenModule>,
    loaded: &mut BTreeMap<String, BTreeSet<String>>,
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
        loaded
            .entry(file.to_string())
            .or_default()
            .extend(load.symbols.keys().map(|local| (*local).to_string()));
        freeze(package, load.module_id, sources, modules, loaded, loading)?;
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
    let globals = globals();
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

/// The prefix of the libraries every package may load: helpers, whose names
/// are never a package's entries.
const LIBRARY: &str = "//lib/";

/// The one file of the package's own (`modules` but its libraries) that
/// defines `name`, or none; two that do are refused, for an entry is the
/// package's, not a file's.
fn defining<'m>(
    package: &str,
    modules: &'m BTreeMap<String, FrozenModule>,
    loaded: &BTreeMap<String, BTreeSet<String>>,
    name: &str,
) -> anyhow::Result<Option<&'m str>> {
    let mut found = modules
        .iter()
        .filter(|(file, module)| {
            !file.starts_with(LIBRARY)
                && module.names().any(|n| n.as_str() == name)
                && !loaded.get(*file).is_some_and(|names| names.contains(name))
        })
        .map(|(file, _)| file.as_str());
    let first = found.next();
    if let Some(second) = found.next() {
        anyhow::bail!(
            "the package `{package}` defines `{name}` in both `{}` and `{second}`; an entry is \
             defined once",
            first.expect("a first before a second")
        );
    }
    Ok(first)
}
