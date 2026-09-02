//! Cache root for NVRTC cubins and the measured cuBLASLt algorithm table.
//! The root is stated by `[cache] dir`, never discovered from the
//! environment; not installed means caching is off (a miss, never an error).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Every NVRTC artifact this deployment keeps. Shared with the engine's
/// guest-program plane; each side names its files apart (`{key:016x}.cubin`
/// vs `{key:016x}-{region}.cubin`) and stores the full key inside the file,
/// so a digest collision is a miss rather than the wrong cubin.
pub const CUBINS: &str = "cubins";

/// The measured cuBLASLt algorithm table's directory. Not under [`CUBINS`]:
/// this is a measured result, not compiler output.
pub const GEMM_ALGOS: &str = "gemm-algos";

static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// State this process's cache root, once. The first call wins; later ones
/// are dropped. `None` installs the feature off explicitly.
pub fn install(root: Option<&Path>) {
    let _ = ROOT.set(root.map(Path::to_path_buf));
}

/// One named subdirectory of the root, or `None` when nothing is stored.
/// Not created here: writers create their own parent on the way past.
#[must_use]
pub fn dir(name: &str) -> Option<PathBuf> {
    ROOT.get()?.as_ref().map(|root| root.join(name))
}
