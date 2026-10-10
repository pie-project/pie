//! The catalog: a set of model packages and every deployment they list, each
//! under the name it is served by. A build embeds the packages it ships; a
//! test reads the repository's from disk, through [`repository`]. Nothing
//! here names a model: what a model is, which deployments it lists, and what
//! it asks of a template and a tokenizer, its package states.

use std::path::Path;
use std::sync::{Arc, LazyLock, OnceLock};

use checkpoint::contract::ModelContract;
use poem::generative::{Diffusion, Generative};
use poem::star::{Deploy, Model, Package, Published};
use poem::{Dtype, Platform, Trace};

/// Why a model does not serve a deployment.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Refused(pub String);

/// One deployment of one model, under the name it is served by.
#[derive(Clone)]
pub struct Deployment {
    pub name: String,
    pub package: Arc<Package>,
    pub model: Model,
    pub deploy: Deploy,
}

impl Deployment {
    /// The deployment `deploy` of `model`, which `package` holds.
    #[must_use]
    pub fn of(package: Arc<Package>, model: &Model, deploy: Deploy) -> Deployment {
        Deployment {
            name: model.name(&deploy),
            package,
            model: model.clone(),
            deploy,
        }
    }

    /// Whether the model serves this deployment on `platform`: the parts and
    /// the drafter are ones it has, the model builds at that precision, and
    /// its trace splits across the ranks.
    pub fn check(&self, platform: Platform) -> Result<(), Refused> {
        self.model
            .admits(&self.deploy)
            .map_err(|why| Refused(format!("{why:#}")))?;
        self.try_trace(platform).map(drop)
    }

    /// The trace each of the deployment's ranks runs.
    pub fn try_trace(&self, platform: Platform) -> Result<Trace, Refused> {
        let trace = self
            .package
            .trace(&self.model.id, &self.deploy, &self.name, platform)
            .map_err(|why| Refused(format!("{why:#}")))?;
        crate::shard::shard(trace, self.deploy.tp).map_err(|why| Refused(why.to_string()))
    }

    /// The trace each of the deployment's ranks runs, for a deployment the
    /// catalog lists, which always builds.
    #[must_use]
    pub fn trace(&self, platform: Platform) -> Trace {
        self.try_trace(platform)
            .unwrap_or_else(|why| panic!("`{}` does not build: {why}", self.name))
    }

    /// The contract reading the checkpoint `src` into this deployment's
    /// weights.
    pub fn contract(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, poem::import::Error> {
        whole(self.deploy.tp)?;
        self.package
            .import(&self.model.id, &self.deploy, src, platform)
    }

    /// The generative facts the model states for this deployment.
    #[must_use]
    pub fn generative(&self) -> Option<Generative> {
        self.package
            .generative(&self.model.id, &self.deploy)
            .unwrap_or_else(|why| panic!("`{}` states no generative facts: {why:#}", self.name))
    }

    /// The canvas the model states for this deployment.
    #[must_use]
    pub fn diffusion(&self) -> Option<Diffusion> {
        self.package
            .diffusion(&self.model.id, &self.deploy)
            .unwrap_or_else(|why| panic!("`{}` states no canvas: {why:#}", self.name))
    }

    /// The marker groups this deployment's tokenizer must hold: the model's
    /// and those of each part it serves.
    #[must_use]
    pub fn markers(&self) -> Vec<Vec<String>> {
        self.model.tokenizer.markers_for(&self.deploy.parts)
    }
}

/// An import reads a whole checkpoint, which one rank of a split row does not
/// land: its share is banded out of a stamped artifact instead.
pub fn whole(tp: u32) -> Result<(), poem::import::Error> {
    if tp == 1 {
        return Ok(());
    }
    Err(poem::import::Error::Illegible {
        name: String::new(),
        detail: format!(
            "an import states the WHOLE checkpoint and this contract is built for {tp} \
             ranks; nothing has banded the file it is reading"
        ),
    })
}

/// The rank counts [`Catalog::splits`] tries each listed deployment at.
pub const RANKS: [u32; 3] = [2, 4, 8];

/// A set of packages and the deployments they list.
pub struct Catalog {
    packages: Vec<Arc<Package>>,
    deployments: Vec<Deployment>,
    splits: OnceLock<Vec<Deployment>>,
}

impl Catalog {
    /// The catalog of `packages`, in their order.
    pub fn new(packages: Vec<Package>) -> Result<Catalog, Refused> {
        let packages: Vec<Arc<Package>> = packages.into_iter().map(Arc::new).collect();
        let mut deployments = Vec::new();
        for package in &packages {
            for (id, deploy) in &package.manifest().deployments {
                let model = package.manifest().model(id).ok_or_else(|| {
                    Refused(format!(
                        "the package `{}` lists a deployment of `{id}`, which it holds no model of",
                        package.name()
                    ))
                })?;
                deployments.push(Deployment::of(Arc::clone(package), model, deploy.clone()));
            }
        }
        Ok(Catalog {
            packages,
            deployments,
            splits: OnceLock::new(),
        })
    }

    /// The catalog of packages given as `(name, [(file, source)])`: the form
    /// a build embeds them in.
    pub fn from_files(packages: &[(&str, &[(&str, &str)])]) -> Result<Catalog, Refused> {
        let packages = packages
            .iter()
            .map(|(name, files)| {
                Package::new(name, files)
                    .map_err(|why| Refused(format!("the package `{name}` does not load: {why:#}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Catalog::new(packages)
    }

    /// The catalog of the packages under `root`: each directory holding a
    /// `package.poem`, with every `.poem` file in it, and the libraries under
    /// `root/lib/` as `//lib/…`, which every package may load.
    pub fn from_dir(root: &Path) -> Result<Catalog, Refused> {
        let io = |what: &str, why: std::io::Error| Refused(format!("{what}: {why}"));
        let mut libraries = Vec::new();
        let lib = root.join("lib");
        if lib.is_dir() {
            walk(&lib, "//lib/", &mut libraries)?;
        }
        libraries.sort();
        let mut packages = Vec::new();
        let dirs = std::fs::read_dir(root).map_err(|why| io(&format!("read {root:?}"), why))?;
        for dir in dirs {
            let dir = dir
                .map_err(|why| io(&format!("read {root:?}"), why))?
                .path();
            if !dir.join("package.poem").is_file() {
                continue;
            }
            let name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| Refused(format!("{dir:?} is no package name")))?
                .to_string();
            let mut files = Vec::new();
            let entries =
                std::fs::read_dir(&dir).map_err(|why| io(&format!("read {dir:?}"), why))?;
            for entry in entries {
                let path = entry
                    .map_err(|why| io(&format!("read {dir:?}"), why))?
                    .path();
                let Some(file) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if file.ends_with(".poem") {
                    let text = std::fs::read_to_string(&path)
                        .map_err(|why| io(&format!("read {path:?}"), why))?;
                    files.push((file.to_string(), text));
                }
            }
            files.sort();
            files.extend(libraries.iter().cloned());
            packages.push((name, files));
        }
        packages.sort();
        let packages = packages
            .iter()
            .map(|(name, files)| {
                let files: Vec<(&str, &str)> = files
                    .iter()
                    .map(|(f, s)| (f.as_str(), s.as_str()))
                    .collect();
                Package::new(name, &files)
                    .map_err(|why| Refused(format!("the package `{name}` does not load: {why:#}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Catalog::new(packages)
    }

    /// Every package, in order.
    #[must_use]
    pub fn packages(&self) -> &[Arc<Package>] {
        &self.packages
    }

    /// The package holding the model `id`, if one does.
    #[must_use]
    pub fn package_of(&self, id: &str) -> Option<&Arc<Package>> {
        self.packages
            .iter()
            .find(|package| package.manifest().model(id).is_some())
    }

    /// The model `id` and the package holding it.
    #[must_use]
    pub fn model(&self, id: &str) -> Option<(&Arc<Package>, &Model)> {
        self.packages
            .iter()
            .find_map(|package| package.manifest().model(id).map(|model| (package, model)))
    }

    /// Every model every package holds, whole and miniature alike.
    pub fn models(&self) -> impl Iterator<Item = (&Arc<Package>, &Model)> {
        self.packages
            .iter()
            .flat_map(|package| package.manifest().models.iter().map(move |m| (package, m)))
    }

    /// Every deployment the packages list, package by package in the order
    /// each lists them. Each runs on one rank; [`Catalog::splits`] are the
    /// same deployments across more.
    pub fn deployments(&self) -> impl Iterator<Item = &Deployment> {
        self.deployments.iter()
    }

    /// Every listed deployment at every count of [`RANKS`] its model splits
    /// it across, a deployment the compiler's sharding pass refuses left out.
    pub fn splits(&self) -> impl Iterator<Item = &Deployment> {
        self.splits
            .get_or_init(|| {
                self.deployments
                    .iter()
                    .flat_map(|whole| {
                        RANKS.into_iter().filter_map(|tp| {
                            let deploy = Deploy {
                                tp,
                                ..whole.deploy.clone()
                            };
                            let split =
                                Deployment::of(Arc::clone(&whole.package), &whole.model, deploy);
                            split.check(Platform::Cuda).is_ok().then_some(split)
                        })
                    })
                    .collect()
            })
            .iter()
    }

    /// The deployment `name` names: one the packages list, or one of its
    /// splits.
    #[must_use]
    pub fn deployment(&self, name: &str) -> Option<&Deployment> {
        self.deployments()
            .find(|d| d.name == name)
            .or_else(|| self.splits().find(|d| d.name == name))
    }

    /// The deployment `name` spells of any model a package holds, listed or
    /// not.
    #[must_use]
    pub fn parse(&self, name: &str) -> Option<Deployment> {
        self.packages.iter().find_map(|package| {
            let (model, deploy) = package.manifest().parse(name)?;
            Some(Deployment::of(Arc::clone(package), model, deploy))
        })
    }

    /// Every drafter a package states it was published for.
    pub fn published(&self) -> impl Iterator<Item = &Published> {
        self.packages
            .iter()
            .flat_map(|package| package.manifest().published.iter())
    }

    /// Every listed deployment, the most likely to read a checkpoint first:
    /// whole models before miniatures, which read a prefix of their whole
    /// model's planes and would claim the whole checkpoint too; then the
    /// richest deployment, since a checkpoint is served with every part and
    /// drafter it carries unless a config leaves them off; the packages'
    /// order breaks ties.
    #[must_use]
    pub fn candidates(&self) -> Vec<&Deployment> {
        let mut candidates: Vec<&Deployment> = self.deployments().collect();
        candidates.sort_by_key(|d| {
            (
                d.model.mini,
                std::cmp::Reverse(d.deploy.parts.len() + usize::from(d.deploy.drafter.is_some())),
            )
        });
        candidates
    }

    /// Every listed deployment with the contract reading `src` into it, in
    /// [`Catalog::candidates`] order.
    pub fn fits<'a>(
        &'a self,
        src: &'a ztensor::Source,
        platform: Platform,
    ) -> impl Iterator<Item = (&'a Deployment, Result<ModelContract, poem::import::Error>)> + 'a
    {
        self.candidates()
            .into_iter()
            .map(move |d| (d, d.contract(src, platform)))
    }

    /// The listed deployment that reads `src` as it is stored.
    pub fn identify(&self, src: &ztensor::Source, platform: Platform) -> Result<&str, Unmatched> {
        let mut misses: Vec<(String, String)> = Vec::new();
        for deployment in self.candidates() {
            match deployment.contract(src, platform) {
                Ok(contract) => match requantizes(&contract) {
                    None => return Ok(&deployment.name),
                    Some(plane) => misses.push((
                        deployment.name.clone(),
                        format!(
                            "reads this checkpoint only by re-quantizing `{plane}` from the form \
                             it is stored in; a second quantization is taken by `--deployment`, \
                             not by identification"
                        ),
                    )),
                },
                Err(why) => misses.push((deployment.name.clone(), why.to_string())),
            }
        }
        Err(Unmatched { misses })
    }

    /// The trace of the model `id`, listed or not, at weights `w` and kv `kv`
    /// on one rank, named `id`: the small geometries the engines' tests serve.
    #[must_use]
    pub fn trace_of(&self, id: &str, w: Dtype, kv: Dtype, platform: Platform) -> Trace {
        let package = self
            .package_of(id)
            .unwrap_or_else(|| panic!("no package holds `{id}`"));
        package
            .trace(id, &one_rank(w, kv), id, platform)
            .unwrap_or_else(|why| panic!("`{id}` does not trace: {why:#}"))
    }

    /// The contract reading `src` into the model `id`, listed or not, at
    /// weights `w` and kv `kv` on one rank.
    pub fn import_of(
        &self,
        id: &str,
        w: Dtype,
        kv: Dtype,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, poem::import::Error> {
        let package = self
            .package_of(id)
            .unwrap_or_else(|| panic!("no package holds `{id}`"));
        package.import(id, &one_rank(w, kv), src, platform)
    }

    /// The package holding the model `id` with `source` appended to its
    /// `file`, stating `function` in place of the one the file states, which
    /// stays as `whole_<function>`: a part of a model traced or read alone,
    /// by the package's own functions.
    pub fn replacing(
        &self,
        id: &str,
        file: &str,
        function: &str,
        source: &str,
    ) -> Result<Package, String> {
        let package = self
            .package_of(id)
            .ok_or_else(|| format!("no package holds `{id}`"))?;
        let prefix = format!("{}files/", poem::star::ATTRIBUTE);
        let stated = format!("def {function}(");
        let files: Vec<(String, String)> = package
            .attributes()
            .into_iter()
            .filter_map(|(key, text)| {
                let name = key.strip_prefix(&prefix)?.to_string();
                let text = if name == file {
                    text.replace(&stated, &format!("def whole_{function}(")) + "\n" + source
                } else {
                    text
                };
                Some((name, text))
            })
            .collect();
        let files: Vec<(&str, &str)> = files
            .iter()
            .map(|(f, s)| (f.as_str(), s.as_str()))
            .collect();
        Package::new(package.name(), &files).map_err(|why| format!("{why:#}"))
    }
}

fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, String)>) -> Result<(), Refused> {
    let entries = std::fs::read_dir(dir).map_err(|why| Refused(format!("read {dir:?}: {why}")))?;
    for entry in entries {
        let path = entry
            .map_err(|why| Refused(format!("read {dir:?}: {why}")))?
            .path();
        let Some(file) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path.is_dir() {
            walk(&path, &format!("{prefix}{file}/"), out)?;
        } else if file.ends_with(".poem") {
            let text = std::fs::read_to_string(&path)
                .map_err(|why| Refused(format!("read {path:?}: {why}")))?;
            out.push((format!("{prefix}{file}"), text));
        }
    }
    Ok(())
}

/// A deployment at weights `w` and kv `kv` on one rank, with no part or
/// drafter.
#[must_use]
pub fn one_rank(w: Dtype, kv: Dtype) -> Deploy {
    Deploy {
        weights: vec![w],
        kv,
        tp: 1,
        parts: Vec::new(),
        drafter: None,
    }
}

/// The plane an import of `contract` would quantize a second time: one
/// stored quantized and published quantized again.
#[must_use]
pub fn requantizes(contract: &ModelContract) -> Option<String> {
    use checkpoint::types::Encoding;
    contract.tensors.iter().find_map(|stored| {
        let name = stored.name.strip_suffix(".stored")?;
        if !matches!(stored.encoding, Encoding::Quant(_)) {
            return None;
        }
        let published = contract.tensors.iter().find(|t| t.name == name)?;
        matches!(published.encoding, Encoding::Quant(_)).then(|| name.to_string())
    })
}

/// No listed deployment reads a checkpoint, and why each does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmatched {
    pub misses: Vec<(String, String)>,
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "this checkpoint matches no deployment this build ships")?;
        for (deployment, why) in &self.misses {
            write!(f, "\n  {deployment}: {why}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unmatched {}

/// Where the repository keeps its packages, beside the crates.
pub const REPOSITORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

static REPOSITORY_CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    Catalog::from_dir(Path::new(REPOSITORY))
        .unwrap_or_else(|why| panic!("the repository's packages under {REPOSITORY}: {why}"))
});

/// The repository's catalog, read from `models/` on disk: what the tests
/// serve. A build ships its own, embedded by the runtime.
pub fn repository() -> &'static Catalog {
    &REPOSITORY_CATALOG
}

/// The deployment `name` of the repository's catalog, listed or split.
#[must_use]
pub fn deployment(name: &str) -> Option<&'static Deployment> {
    repository().deployment(name)
}

/// Every deployment the repository's packages list.
pub fn deployments() -> impl Iterator<Item = &'static Deployment> {
    repository().deployments()
}

/// Every listed deployment of the repository's catalog split across ranks.
pub fn splits() -> impl Iterator<Item = &'static Deployment> {
    repository().splits()
}

/// The package of the repository's catalog holding the model `id`.
#[must_use]
pub fn package_of(id: &str) -> Option<&'static Arc<Package>> {
    repository().package_of(id)
}

/// [`Catalog::trace_of`] over the repository's catalog.
#[must_use]
pub fn trace_of(id: &str, w: Dtype, kv: Dtype, platform: Platform) -> Trace {
    repository().trace_of(id, w, kv, platform)
}

/// [`Catalog::import_of`] over the repository's catalog.
pub fn import_of(
    id: &str,
    w: Dtype,
    kv: Dtype,
    src: &ztensor::Source,
    platform: Platform,
) -> Result<ModelContract, poem::import::Error> {
    repository().import_of(id, w, kv, src, platform)
}

/// [`Catalog::replacing`] over the repository's catalog.
pub fn replacing(id: &str, file: &str, function: &str, source: &str) -> Result<Package, String> {
    repository().replacing(id, file, function, source)
}

/// [`Catalog::identify`] over the repository's catalog.
pub fn identify(src: &ztensor::Source, platform: Platform) -> Result<&'static str, Unmatched> {
    repository().identify(src, platform)
}
