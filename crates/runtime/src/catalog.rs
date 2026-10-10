//! The catalog this build serves: the packages under `$PIE_HOME/models/`,
//! which [`install`] seeds from the ones the binary embeds (the repository's
//! `models/` at build time) and keeps complete. What a model is, its package
//! states; the runtime only reads packages.
//!
//! Each seeded directory records the digest of what was seeded in
//! `.seeded`. A directory nobody edited is brought up to the binary's
//! version when the binary changes; one that was edited is kept, and a
//! warning says the built-in it started from has moved on since.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};

pub use poem_compiler::catalog::{Catalog, Deployment, Refused, Tree};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/packages.rs"));
}

/// The file a seeded directory records its seed's digest in.
pub const SEEDED: &str = ".seeded";

/// The directory under `models/` the libraries live in.
const LIB: &str = "lib";

static MODELS_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The tree this build embeds.
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

/// The packages this build embeds, alone.
#[must_use]
pub fn embedded() -> Catalog {
    Catalog::from_tree(&embedded_tree())
        .unwrap_or_else(|why| panic!("this build's packages do not load: {why}"))
}

/// What seeding `models/` found, per directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seeded {
    /// Written for the first time.
    Written,
    /// Already the binary's version.
    Current,
    /// An older seed nobody edited, brought up to the binary's version.
    Updated,
    /// Edited by hand; left as it is. The built-in has moved on since the
    /// seed it was edited from when `behind` is set.
    Edited { behind: bool },
}

/// Seeds `models` from the embedded tree and installs it as the directory
/// the catalog reads. Set once, before the catalog is first read; a later
/// call seeds again but does not move the catalog.
pub fn install(models: &Path) -> Vec<(String, Seeded)> {
    let outcome = match seed(models) {
        Ok(outcome) => outcome,
        Err(why) => {
            tracing::warn!(
                ?models,
                %why,
                "the models directory could not be seeded; this build's embedded packages serve"
            );
            return Vec::new();
        }
    };
    for (name, seeded) in &outcome {
        match seeded {
            Seeded::Written => tracing::info!(package = name, ?models, "seeded"),
            Seeded::Current => {}
            Seeded::Updated => {
                tracing::info!(
                    package = name,
                    ?models,
                    "brought up to this build's version"
                );
            }
            Seeded::Edited { behind: false } => {
                tracing::info!(package = name, ?models, "edited by hand; serving the edit");
            }
            Seeded::Edited { behind: true } => tracing::warn!(
                package = name,
                ?models,
                "edited by hand from an older built-in, which has changed since; delete the \
                 directory to take this build's, or carry the edit over"
            ),
        }
    }
    let _ = MODELS_DIR.set(models.to_path_buf());
    outcome
}

/// The models directory installed, if one was.
#[must_use]
pub fn installed() -> Option<&'static Path> {
    MODELS_DIR.get().map(PathBuf::as_path)
}

/// Every directory the embedded tree seeds: each package by name with its
/// files, and `lib` with the libraries under their path below it.
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

/// The digest of a directory's files, `(path below the directory, text)`,
/// in path order.
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

/// The `.poem` files under `dir`, `(path below `dir`, text)`, walked whole.
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

/// Writes `files` under `dir`, every other `.poem` file there removed, and
/// records `seeded` as the directory's seed.
fn write(dir: &Path, files: &[(String, String)], seeded: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let stale = on_disk(dir)?;
    for (path, _) in &stale {
        if !files.iter().any(|(wanted, _)| wanted == path) {
            std::fs::remove_file(dir.join(path))?;
        }
    }
    for (path, text) in files {
        let target = dir.join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(target, text)?;
    }
    std::fs::write(dir.join(SEEDED), seeded)
}

fn seed(models: &Path) -> std::io::Result<Vec<(String, Seeded)>> {
    std::fs::create_dir_all(models)?;
    let mut outcome = Vec::new();
    for (name, files) in seeds(&embedded_tree()) {
        let dir = models.join(&name);
        let ships = digest(&files);
        let seeded = if !dir.is_dir() {
            write(&dir, &files, &ships)?;
            Seeded::Written
        } else {
            let held = digest(&on_disk(&dir)?);
            let marker = std::fs::read_to_string(dir.join(SEEDED)).ok();
            if held == ships {
                if marker.as_deref() != Some(ships.as_str()) {
                    std::fs::write(dir.join(SEEDED), &ships)?;
                }
                Seeded::Current
            } else if marker.as_deref() == Some(held.as_str()) {
                write(&dir, &files, &ships)?;
                Seeded::Updated
            } else {
                Seeded::Edited {
                    behind: marker.as_deref() != Some(ships.as_str()),
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
    for why in refused {
        tracing::error!(?models, %why, "a package under the models directory does not load and is left out");
    }
    catalog
});

/// The catalog this build serves.
pub fn catalog() -> &'static Catalog {
    &CATALOG
}

/// The deployment `name` of this build's catalog, listed or split.
#[must_use]
pub fn deployment(name: &str) -> Option<&'static Deployment> {
    catalog().deployment(name)
}

/// Every deployment this build's packages list.
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

        // A fresh directory is written whole, lib included.
        let first = seed(&models).unwrap();
        assert!(first.iter().all(|(_, s)| *s == Seeded::Written));
        assert!(models.join("qwen_3").join("package.poem").is_file());
        assert!(models.join(LIB).join(SEEDED).is_file());

        // Seeding again finds everything current.
        let again = seed(&models).unwrap();
        assert!(
            again.iter().all(|(_, s)| *s == Seeded::Current),
            "{again:?}"
        );

        // An older seed nobody edited: the marker says so, and it is updated.
        let qwen = models.join("qwen_3");
        std::fs::write(qwen.join("forward.poem"), "# an older build's\n").unwrap();
        std::fs::write(qwen.join(SEEDED), digest(&on_disk(&qwen).unwrap())).unwrap();
        let updated = seed(&models).unwrap();
        assert!(
            updated
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Updated),
            "{updated:?}"
        );
        assert_ne!(
            std::fs::read_to_string(qwen.join("forward.poem")).unwrap(),
            "# an older build's\n"
        );

        // An edit of the current seed is kept and is not behind.
        std::fs::write(qwen.join("forward.poem"), "# mine\n").unwrap();
        let edited = seed(&models).unwrap();
        assert!(
            edited
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Edited { behind: false }),
            "{edited:?}"
        );
        assert_eq!(
            std::fs::read_to_string(qwen.join("forward.poem")).unwrap(),
            "# mine\n"
        );

        // An edit of an older seed is kept and is behind.
        std::fs::write(qwen.join(SEEDED), "an older build's digest").unwrap();
        let behind = seed(&models).unwrap();
        assert!(
            behind
                .iter()
                .any(|(n, s)| n == "qwen_3" && *s == Seeded::Edited { behind: true }),
            "{behind:?}"
        );

        // A missing package comes back.
        std::fs::remove_dir_all(models.join("gemma_4")).unwrap();
        let back = seed(&models).unwrap();
        assert!(
            back.iter()
                .any(|(n, s)| n == "gemma_4" && *s == Seeded::Written)
        );
    }
}
