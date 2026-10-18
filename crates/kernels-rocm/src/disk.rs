use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const CODE_OBJECTS: &str = "code-objects";

static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

pub fn install(root: Option<&Path>) {
    let _ = ROOT.set(root.map(Path::to_path_buf));
}

#[must_use]
pub fn dir(name: &str) -> Option<PathBuf> {
    ROOT.get()?.as_ref().map(|root| root.join(name))
}
