//! The catalog this build serves: `$PIE_HOME/models/`, which [`install`] seeds
//! from the packages the binary embeds and keeps whole. A seeded package
//! records its seed's digest beside it (`.{name}.seeded`, or `.seeded` in a
//! directory): untouched, it follows the binary; edited, it is kept, with a
//! warning once the built-in it came from moves on.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};

pub use poem_compiler::catalog::{Catalog, Deployment, Refused, Tree};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/packages.rs"));
}

pub const SEEDED: &str = ".seeded";

const LIB: &str = "lib";

static INSTALLED: OnceLock<(PathBuf, Vec<(String, Seeded)>)> = OnceLock::new();

#[must_use]
pub fn embedded_tree() -> Tree {
    Tree {
        packages: embedded::PACKAGES
            .iter()
            .map(|(name, files)| {
                (
                    (*name).to_string(),
                    files
                        .iter()
                        .map(|(f, s)| ((*f).to_string(), (*s).to_string()))
                        .collect(),
                )
            })
            .collect(),
        library: embedded::LIBRARY
            .iter()
            .map(|(f, s)| ((*f).to_string(), (*s).to_string()))
            .collect(),
    }
}

/// The embedded packages alone: what serves until [`install`] runs.
#[must_use]
pub fn embedded() -> Catalog {
    Catalog::from_tree(&embedded_tree())
        .unwrap_or_else(|why| panic!("this build's packages do not load: {why}"))
}

/// What seeding found a directory to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seeded {
    Written,
    /// Already the binary's version.
    Current,
    /// An older seed nobody edited, now the binary's version.
    Updated,
    /// Edited by hand and kept; `behind` once the built-in has moved on since.
    Edited {
        behind: bool,
    },
    /// Made by hand under a built-in's name, with no seed to compare to; kept.
    Foreign,
}

/// Seeds `models` and makes it the directory the catalog reads, once; a
/// later call returns what the first found.
pub fn install(models: &Path) -> &'static [(String, Seeded)] {
    &INSTALLED
        .get_or_init(|| {
            let outcome = match seed(models) {
                Ok(outcome) => outcome,
                Err(why) => {
                    tracing::warn!(
                        ?models,
                        %why,
                        "the models directory could not be seeded; this build's embedded packages serve"
                    );
                    Vec::new()
                }
            };
            for (name, seeded) in &outcome {
                match seeded {
                    Seeded::Written => tracing::info!(package = name, ?models, "seeded"),
                    Seeded::Current => {}
                    Seeded::Updated => {
                        tracing::info!(package = name, ?models, "brought up to this build's")
                    }
                    Seeded::Edited { behind: false } | Seeded::Foreign => {
                        tracing::info!(package = name, ?models, "made by hand; serving it")
                    }
                    Seeded::Edited { behind: true } => tracing::warn!(
                        package = name,
                        ?models,
                        "edited by hand from an older built-in, which has changed since; delete \
                         it to take this build's, or carry the edit over"
                    ),
                }
            }
            (models.to_path_buf(), outcome)
        })
        .1
}

#[must_use]
pub fn installed() -> Option<&'static Path> {
    INSTALLED.get().map(|(models, _)| models.as_path())
}

/// Each directory the tree seeds: every package, and `lib`.
fn seeds(tree: &Tree) -> Vec<(String, Vec<(String, String)>)> {
    let mut seeds = tree.packages.clone();
    if !tree.library.is_empty() {
        seeds.push((
            LIB.to_string(),
            tree.library
                .iter()
                .map(|(path, text)| {
                    let below = path
                        .strip_prefix(poem_compiler::catalog::LIBRARY)
                        .unwrap_or(path);
                    (below.to_string(), text.clone())
                })
                .collect(),
        ));
    }
    seeds
}

fn digest(files: &[(String, String)]) -> String {
    let mut sorted: Vec<&(String, String)> = files.iter().collect();
    sorted.sort();
    let mut hasher = blake3::Hasher::new();
    for (path, text) in sorted {
        hasher.update(path.as_bytes());
        hasher.update(&[0]);
        hasher.update(text.as_bytes());
        hasher.update(&[0]);
    }
    hasher.finalize().to_hex().to_string()
}

/// The `.poem` files under `dir`, by path below it.
fn on_disk(dir: &Path) -> std::io::Result<Vec<(String, String)>> {
    fn walk(dir: &Path, below: &str, out: &mut Vec<(String, String)>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let Some(file) = path.file_name().and_then(|f| f.to_str()) else {
                continue;
            };
            if path.is_dir() {
                walk(&path, &format!("{below}{file}/"), out)?;
            } else if file.ends_with(".poem") {
                out.push((format!("{below}{file}"), std::fs::read_to_string(&path)?));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, "", &mut out)?;
    Ok(out)
}

/// Writes `files` and the seed marker as a fresh directory beside `dir`,
/// then swaps it in, so a seed that dies partway leaves `dir` as it was.
fn write(dir: &Path, files: &[(String, String)], seeded: &str) -> std::io::Result<()> {
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("package");
    let parent = dir.parent().unwrap_or(Path::new("."));
    let fresh = parent.join(format!(".{name}.{}.seeding", std::process::id()));
    let _ = std::fs::remove_dir_all(&fresh);
    std::fs::create_dir_all(&fresh)?;
    for (path, text) in files {
        let target = fresh.join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(target, text)?;
    }
    std::fs::write(fresh.join(SEEDED), seeded)?;
    let old = parent.join(format!(".{name}.{}.replaced", std::process::id()));
    if dir.is_dir() {
        std::fs::rename(dir, &old)?;
    }
    if let Err(why) = std::fs::rename(&fresh, dir) {
        if old.is_dir() {
            let _ = std::fs::rename(&old, dir);
        }
        return Err(why);
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Where a seed lands: a package of one file as `{name}.poem` beside the
/// others, its digest in `.{name}.seeded`; `lib`, and a package of several
/// files, as a directory holding its digest in `.seeded`.
enum Place {
    File { path: PathBuf, marker: PathBuf },
    Dir(PathBuf),
}

impl Place {
    fn of(models: &Path, name: &str, files: &[(String, String)]) -> Place {
        match files {
            [(file, _)] if name != LIB && *file == format!("{name}.poem") => Place::File {
                path: models.join(file),
                marker: models.join(format!(".{name}{SEEDED}")),
            },
            _ => Place::Dir(models.join(name)),
        }
    }

    fn held(&self) -> std::io::Result<Option<Vec<(String, String)>>> {
        match self {
            Place::File { path, .. } if path.is_file() => {
                let file = path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or_default();
                Ok(Some(vec![(
                    file.to_string(),
                    std::fs::read_to_string(path)?,
                )]))
            }
            Place::Dir(dir) if dir.is_dir() => on_disk(dir).map(Some),
            _ => Ok(None),
        }
    }

    fn marker(&self) -> PathBuf {
        match self {
            Place::File { marker, .. } => marker.clone(),
            Place::Dir(dir) => dir.join(SEEDED),
        }
    }

    /// Writes `files` and the marker; a file is written beside its place
    /// and renamed over it, so a seed that dies partway leaves it whole.
    fn write(&self, files: &[(String, String)], seeded: &str) -> std::io::Result<()> {
        match self {
            Place::Dir(dir) => write(dir, files, seeded),
            Place::File { path, marker } => {
                let fresh = path.with_extension(format!("poem.{}.seeding", std::process::id()));
                std::fs::write(&fresh, &files[0].1)?;
                std::fs::rename(&fresh, path)?;
                std::fs::write(marker, seeded)
            }
        }
    }
}

/// An earlier build's seed of `name` as a directory, where it now ships as
/// one file: gone if nobody edited it, else what it was found to be.
fn earlier(models: &Path, name: &str, ships: &str) -> std::io::Result<Option<Seeded>> {
    let dir = models.join(name);
    if !dir.is_dir() {
        return Ok(None);
    }
    let held = digest(&on_disk(&dir)?);
    match std::fs::read_to_string(dir.join(SEEDED)).ok() {
        Some(seeded) if seeded == held => {
            std::fs::remove_dir_all(&dir)?;
            Ok(None)
        }
        Some(seeded) => Ok(Some(Seeded::Edited {
            behind: seeded != ships,
        })),
        None => Ok(Some(Seeded::Foreign)),
    }
}

fn seed(models: &Path) -> std::io::Result<Vec<(String, Seeded)>> {
    std::fs::create_dir_all(models)?;
    let mut outcome = Vec::new();
    for (name, files) in seeds(&embedded_tree()) {
        let place = Place::of(models, &name, &files);
        let ships = digest(&files);
        if matches!(place, Place::File { .. })
            && let Some(kept) = earlier(models, &name, &ships)?
        {
            outcome.push((name, kept));
            continue;
        }
        let seeded = match place.held()? {
            None => {
                place.write(&files, &ships)?;
                Seeded::Written
            }
            Some(held) => {
                let held = digest(&held);
                let marker = std::fs::read_to_string(place.marker()).ok();
                if held == ships {
                    if marker.as_deref() != Some(ships.as_str()) {
                        std::fs::write(place.marker(), &ships)?;
                    }
                    Seeded::Current
                } else {
                    match marker.as_deref() {
                        None => Seeded::Foreign,
                        Some(seeded) if seeded == held => {
                            place.write(&files, &ships)?;
                            Seeded::Updated
                        }
                        Some(seeded) => Seeded::Edited {
                            behind: seeded != ships,
                        },
                    }
                }
            }
        };
        outcome.push((name, seeded));
    }
    Ok(outcome)
}

static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    let Some(models) = installed() else {
        return embedded();
    };
    let tree = match Tree::read(models) {
        Ok(tree) => tree,
        Err(why) => {
            tracing::error!(?models, %why, "the models directory does not read; this build's embedded packages serve");
            return embedded();
        }
    };
    let (catalog, refused) = Catalog::of_tree(&tree);
    for why in refused.iter().chain(&template_faults(&catalog)) {
        tracing::error!(?models, %why, "under the models directory");
    }
    catalog
});

/// Each model whose template names a format or a setting the runtime does
/// not ship. The catalog still lists it; registering it refuses.
pub fn template_faults(catalog: &Catalog) -> Vec<Refused> {
    catalog
        .models()
        .filter_map(|(package, model)| {
            crate::model::template_of(model).check().err().map(|why| {
                Refused(format!(
                    "`{}` in the package `{}`: {why}",
                    model.id,
                    package.name()
                ))
            })
        })
        .collect()
}

pub fn catalog() -> &'static Catalog {
    &CATALOG
}

#[must_use]
pub fn deployment(name: &str) -> Option<&'static Deployment> {
    catalog().deployment(name)
}

pub fn deployments() -> impl Iterator<Item = &'static Deployment> {
    catalog().deployments()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeding_every_case() {
        let home = tempfile::tempdir().unwrap();
        let models = home.path().join("models");

        let first = seed(&models).unwrap();
        assert!(first.iter().all(|(_, s)| *s == Seeded::Written));
        assert!(models.join("qwen_3.poem").is_file());
        assert!(models.join(LIB).join(SEEDED).is_file());

        let again = seed(&models).unwrap();
        assert!(
            again.iter().all(|(_, s)| *s == Seeded::Current),
            "{again:?}"
        );

        let qwen = models.join("qwen_3.poem");
        let marker = models.join(".qwen_3.seeded");
        let digest_of = |text: &str| digest(&[("qwen_3.poem".to_string(), text.to_string())]);
        std::fs::write(&qwen, "# an older build's\n").unwrap();
        std::fs::write(&marker, digest_of("# an older build's\n")).unwrap();
        let updated = seed(&models).unwrap();
        assert!(
            updated
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Updated),
            "{updated:?}"
        );
        assert_ne!(
            std::fs::read_to_string(&qwen).unwrap(),
            "# an older build's\n"
        );

        std::fs::write(&qwen, "# mine\n").unwrap();
        let edited = seed(&models).unwrap();
        assert!(
            edited
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Edited { behind: false }),
            "{edited:?}"
        );
        assert_eq!(std::fs::read_to_string(&qwen).unwrap(), "# mine\n");

        std::fs::write(&marker, "an older build's digest").unwrap();
        let behind = seed(&models).unwrap();
        assert!(
            behind
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Edited { behind: true }),
            "{behind:?}"
        );

        std::fs::remove_file(&marker).unwrap();
        let foreign = seed(&models).unwrap();
        assert!(
            foreign
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Foreign),
            "{foreign:?}"
        );

        std::fs::remove_file(models.join("gemma_4.poem")).unwrap();
        let back = seed(&models).unwrap();
        assert!(
            back.iter()
                .any(|(n, s)| n == "gemma_4" && *s == Seeded::Written)
        );
    }

    #[test]
    fn an_earlier_seed_directory_gives_way_to_its_one_file() {
        let home = tempfile::tempdir().unwrap();
        let models = home.path().join("models");
        seed(&models).unwrap();
        // An earlier build seeded `glm_4/` and `mimo_v2/` as directories;
        // nobody touched the first, the second was edited.
        for name in ["glm_4", "mimo_v2"] {
            std::fs::remove_file(models.join(format!("{name}.poem"))).unwrap();
            std::fs::remove_file(models.join(format!(".{name}.seeded"))).unwrap();
            let dir = models.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.poem"), "# an older build's\n").unwrap();
            std::fs::write(dir.join(SEEDED), digest(&on_disk(&dir).unwrap())).unwrap();
        }
        std::fs::write(models.join("mimo_v2").join("model.poem"), "# mine\n").unwrap();
        let outcome = seed(&models).unwrap();
        assert!(
            !models.join("glm_4").exists(),
            "the untouched directory is gone"
        );
        assert!(models.join("glm_4.poem").is_file());
        assert!(
            outcome
                .iter()
                .any(|(n, s)| n == "mimo_v2" && *s == Seeded::Edited { behind: true }),
            "{outcome:?}"
        );
        assert!(models.join("mimo_v2").is_dir() && !models.join("mimo_v2.poem").exists());
        Tree::read(&models).expect("the seeded tree reads");
    }
}
