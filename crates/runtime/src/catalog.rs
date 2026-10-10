//! The catalog this build ships: every package under the repository's
//! `models/` at build time, embedded, and the deployments they list. What a
//! model is, its package states; the runtime only reads the packages.

use std::sync::LazyLock;

pub use poem_compiler::catalog::{Catalog, Deployment, Refused};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/packages.rs"));
}

static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    Catalog::from_files(embedded::PACKAGES)
        .unwrap_or_else(|why| panic!("this build's packages do not load: {why}"))
});

/// The catalog this build ships.
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
