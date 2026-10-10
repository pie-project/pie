//! The catalog this build serves: every package under the repository's
//! `models/` at build time, embedded, shadowed by the packages under the
//! installed models directory (`$PIE_HOME/models`) once one is installed.
//! What a model is, its package states; the runtime only reads packages.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};

pub use poem_compiler::catalog::{Catalog, Deployment, Refused};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/packages.rs"));
}

static MODELS_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The models directory whose packages shadow the embedded ones. Set once,
/// before the catalog is first read; a later call is ignored.
pub fn install(models: &Path) {
    let _ = MODELS_DIR.set(models.to_path_buf());
}

/// The models directory installed, if one was.
#[must_use]
pub fn installed() -> Option<&'static Path> {
    MODELS_DIR.get().map(PathBuf::as_path)
}

/// The packages this build embeds, alone.
pub fn embedded() -> Catalog {
    Catalog::from_files(embedded::PACKAGES)
        .unwrap_or_else(|why| panic!("this build's packages do not load: {why}"))
}

static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    let catalog = embedded();
    match installed() {
        Some(models) => catalog
            .shadowed_by(models)
            .unwrap_or_else(|why| panic!("the packages under {models:?} do not load: {why}")),
        None => catalog,
    }
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
