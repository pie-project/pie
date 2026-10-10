//! The catalog: model packages and every deployment they list, each under
//! the name it is served by. A build embeds its packages; the tests read the
//! repository's through [`repository`]. Nothing here names a model.

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
    #[must_use]
    pub fn of(package: Arc<Package>, model: &Model, deploy: Deploy) -> Deployment {
        Deployment {
            name: model.name(&deploy),
            package,
            model: model.clone(),
            deploy,
        }
    }

    /// The model admits the deployment and its trace builds and shards.
    pub fn check(&self, platform: Platform) -> Result<(), Refused> {
        self.model
            .admits(&self.deploy)
            .map_err(|why| Refused(format!("{why:#}")))?;
        self.try_trace(platform).map(drop)
    }

    pub fn try_trace(&self, platform: Platform) -> Result<Trace, Refused> {
        let trace = self
            .package
            .trace(&self.model.id, &self.deploy, &self.name, platform)
            .map_err(|why| Refused(format!("{why:#}")))?;
        crate::shard::shard(trace, self.deploy.tp).map_err(|why| Refused(why.to_string()))
    }

    /// [`Deployment::try_trace`], panicking: for a listed deployment, which builds.
    #[must_use]
    pub fn trace(&self, platform: Platform) -> Trace {
        self.try_trace(platform)
            .unwrap_or_else(|why| panic!("`{}` does not build: {why}", self.name))
    }

    pub fn contract(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, poem::import::Error> {
        whole(self.deploy.tp)?;
        self.package
            .import(&self.model.id, &self.deploy, src, platform)
    }

    pub fn try_generative(&self) -> Result<Option<Generative>, Refused> {
        self.package
            .generative(&self.model.id, &self.deploy)
            .map_err(|why| Refused(format!("{why:#}")))
    }

    pub fn try_diffusion(&self) -> Result<Option<Diffusion>, Refused> {
        self.package
            .diffusion(&self.model.id, &self.deploy)
            .map_err(|why| Refused(format!("{why:#}")))
    }

    /// [`Deployment::try_generative`], panicking: for a listed deployment.
    #[must_use]
    pub fn generative(&self) -> Option<Generative> {
        self.try_generative()
            .unwrap_or_else(|why| panic!("`{}` states no generative facts: {why}", self.name))
    }

    /// [`Deployment::try_diffusion`], panicking: for a listed deployment.
    #[must_use]
    pub fn diffusion(&self) -> Option<Diffusion> {
        self.try_diffusion()
            .unwrap_or_else(|why| panic!("`{}` states no canvas: {why}", self.name))
    }

    /// The marker groups the tokenizer must hold for the parts served.
    #[must_use]
    pub fn markers(&self) -> Vec<Vec<String>> {
        self.model.tokenizer.markers_for(&self.deploy.parts)
    }
}

/// An import reads a whole checkpoint; a split rank bands its share out of an
/// artifact instead.
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

/// The rank counts [`Catalog::splits`] tries.
pub const RANKS: [u32; 3] = [2, 4, 8];

pub struct Catalog {
    packages: Vec<Arc<Package>>,
    deployments: Vec<Deployment>,
    splits: OnceLock<Vec<Deployment>>,
}

impl Catalog {
    pub fn new(packages: Vec<Package>) -> Result<Catalog, Refused> {
        let mut catalog = Catalog::empty();
        for package in packages {
            catalog.add(Arc::new(package))?;
        }
        Ok(catalog)
    }

    fn empty() -> Catalog {
        Catalog {
            packages: Vec::new(),
            deployments: Vec::new(),
            splits: OnceLock::new(),
        }
    }

    /// Adds `package`, refused if it holds a model or lists a deployment
    /// another package already does.
    fn add(&mut self, package: Arc<Package>) -> Result<(), Refused> {
        let manifest = package.manifest();
        if let Some(taken) = manifest.models.iter().find(|m| self.model(&m.id).is_some()) {
            return Err(Refused(format!(
                "the package `{}` holds `{}`, which another package already holds",
                package.name(),
                taken.id
            )));
        }
        let mut deployments = Vec::new();
        for (id, deploy) in &manifest.deployments {
            let model = manifest.model(id).ok_or_else(|| {
                Refused(format!(
                    "the package `{}` lists a deployment of `{id}`, which it holds no model of",
                    package.name()
                ))
            })?;
            deployments.push(Deployment::of(Arc::clone(&package), model, deploy.clone()));
        }
        self.packages.push(package);
        self.deployments.extend(deployments);
        Ok(())
    }

    /// The catalog of `tree`, a package that does not load left out and reported.
    pub fn of_tree(tree: &Tree) -> (Catalog, Vec<Refused>) {
        let mut catalog = Catalog::empty();
        let mut refused = Vec::new();
        for (name, files) in &tree.packages {
            let files: Vec<(&str, &str)> = files
                .iter()
                .chain(&tree.library)
                .map(|(f, s)| (f.as_str(), s.as_str()))
                .collect();
            let added = Package::new(name, &files)
                .map_err(|why| Refused(format!("the package `{name}` does not load: {why:#}")))
                .and_then(|package| catalog.add(Arc::new(package)));
            if let Err(why) = added {
                refused.push(why);
            }
        }
        (catalog, refused)
    }

    pub fn from_tree(tree: &Tree) -> Result<Catalog, Refused> {
        let (catalog, refused) = Catalog::of_tree(tree);
        match refused.into_iter().next() {
            None => Ok(catalog),
            Some(why) => Err(why),
        }
    }

    pub fn from_dir(root: &Path) -> Result<Catalog, Refused> {
        Catalog::from_tree(&Tree::read(root)?)
    }

    #[must_use]
    pub fn packages(&self) -> &[Arc<Package>] {
        &self.packages
    }

    #[must_use]
    pub fn package(&self, name: &str) -> Option<&Arc<Package>> {
        self.packages.iter().find(|p| p.name() == name)
    }

    #[must_use]
    pub fn package_of(&self, id: &str) -> Option<&Arc<Package>> {
        self.packages
            .iter()
            .find(|package| package.manifest().model(id).is_some())
    }

    #[must_use]
    pub fn model(&self, id: &str) -> Option<(&Arc<Package>, &Model)> {
        self.packages
            .iter()
            .find_map(|package| package.manifest().model(id).map(|model| (package, model)))
    }

    pub fn models(&self) -> impl Iterator<Item = (&Arc<Package>, &Model)> {
        self.packages
            .iter()
            .flat_map(|package| package.manifest().models.iter().map(move |m| (package, m)))
    }

    /// The listed deployments, each on one rank.
    pub fn deployments(&self) -> impl Iterator<Item = &Deployment> {
        self.deployments.iter()
    }

    /// The listed deployments at every count of [`RANKS`] they shard across.
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

    /// A listed deployment or one of its splits, by name.
    #[must_use]
    pub fn deployment(&self, name: &str) -> Option<&Deployment> {
        self.deployments()
            .find(|d| d.name == name)
            .or_else(|| self.splits().find(|d| d.name == name))
    }

    /// The deployment `name` spells, listed or not.
    #[must_use]
    pub fn parse(&self, name: &str) -> Option<Deployment> {
        self.packages.iter().find_map(|package| {
            let (model, deploy) = package.manifest().parse(name)?;
            Some(Deployment::of(Arc::clone(package), model, deploy))
        })
    }

    pub fn published(&self) -> impl Iterator<Item = &Published> {
        self.packages
            .iter()
            .flat_map(|package| package.manifest().published.iter())
    }

    /// The listed deployments, the most likely to read a checkpoint first: whole
    /// models before miniatures (which read a prefix of the whole's planes), then
    /// the most parts and drafter, then package order.
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

    /// The listed deployment that reads `src` as stored.
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

    /// The trace of the model `id`, listed or not, on one rank: the engines' test geometries.
    #[must_use]
    pub fn trace_of(&self, id: &str, w: Dtype, kv: Dtype, platform: Platform) -> Trace {
        let package = self
            .package_of(id)
            .unwrap_or_else(|| panic!("no package holds `{id}`"));
        package
            .trace(id, &one_rank(w, kv), id, platform)
            .unwrap_or_else(|why| panic!("`{id}` does not trace: {why:#}"))
    }

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

    /// The package of `id` with `source` appended to `file`, whose `function` is
    /// renamed `whole_<function>`: a part of a model traced or read alone.
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
        let stated = format!("def {function}(");
        let files: Vec<(String, String)> = package
            .files()
            .map(|(name, text)| {
                let text = if name == file {
                    text.replace(&stated, &format!("def whole_{function}(")) + "\n" + source
                } else {
                    text.to_string()
                };
                (name.to_string(), text)
            })
            .collect();
        let files: Vec<(&str, &str)> = files
            .iter()
            .map(|(f, s)| (f.as_str(), s.as_str()))
            .collect();
        Package::new(package.name(), &files).map_err(|why| format!("{why:#}"))
    }
}

/// Packages as a directory holds them: each one's files, and the `//lib/`
/// files every package may load.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tree {
    pub packages: Vec<(String, Vec<(String, String)>)>,
    pub library: Vec<(String, String)>,
}

pub const LIBRARY: &str = "//lib/";

impl Tree {
    pub fn read(root: &Path) -> Result<Tree, Refused> {
        let io = |what: &str, why: std::io::Error| Refused(format!("{what}: {why}"));
        let mut library = Vec::new();
        let lib = root.join("lib");
        if lib.is_dir() {
            walk(&lib, LIBRARY, &mut library)?;
        }
        library.sort();
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
                if file.ends_with(".poem") && path.is_file() {
                    let text = std::fs::read_to_string(&path)
                        .map_err(|why| io(&format!("read {path:?}"), why))?;
                    files.push((file.to_string(), text));
                }
            }
            files.sort();
            packages.push((name, files));
        }
        packages.sort();
        Ok(Tree { packages, library })
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

/// The plane an import would quantize a second time, if any.
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

pub const REPOSITORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

static REPOSITORY_CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    Catalog::from_dir(Path::new(REPOSITORY))
        .unwrap_or_else(|why| panic!("the repository's packages under {REPOSITORY}: {why}"))
});

/// The repository's `models/`, read from disk: what the tests serve.
pub fn repository() -> &'static Catalog {
    &REPOSITORY_CATALOG
}

#[must_use]
pub fn deployment(name: &str) -> Option<&'static Deployment> {
    repository().deployment(name)
}

pub fn deployments() -> impl Iterator<Item = &'static Deployment> {
    repository().deployments()
}

pub fn splits() -> impl Iterator<Item = &'static Deployment> {
    repository().splits()
}

#[must_use]
pub fn package_of(id: &str) -> Option<&'static Arc<Package>> {
    repository().package_of(id)
}

#[must_use]
pub fn trace_of(id: &str, w: Dtype, kv: Dtype, platform: Platform) -> Trace {
    repository().trace_of(id, w, kv, platform)
}

pub fn import_of(
    id: &str,
    w: Dtype,
    kv: Dtype,
    src: &ztensor::Source,
    platform: Platform,
) -> Result<ModelContract, poem::import::Error> {
    repository().import_of(id, w, kv, src, platform)
}

pub fn replacing(id: &str, file: &str, function: &str, source: &str) -> Result<Package, String> {
    repository().replacing(id, file, function, source)
}

pub fn identify(src: &ztensor::Source, platform: Platform) -> Result<&'static str, Unmatched> {
    repository().identify(src, platform)
}
