//! **Where this crate keeps what it will not compute twice**, and the one way
//! it learns where that is.
//!
//! Two caches hang off the root below: the NVRTC cubins
//! [`jit::cache`](crate::jit::cache) resolves, and the measured cuBLASLt
//! algorithm table [`linear::dense`](crate::linear::dense) keeps. Both used to
//! resolve their own path from `$XDG_CACHE_HOME`, else `$HOME/.cache` — which
//! put them outside `$PIE_HOME`, and therefore outside what `pie cache list`
//! can see and `pie cache clear` can reclaim. A cache only one of those knows
//! about is a cache that either cannot be reclaimed or is reclaimed by
//! surprise, which is the argument `worker::state` was written around.
//!
//! **THE ROOT IS STATED, NOT DISCOVERED** (article 9). `[cache] dir` reaches
//! the shell typed on its boot document, and the shell installs it here before
//! it compiles anything. Nothing in this crate reads the environment.
//!
//! Never installed is the feature OFF — the same answer `[cache] dir` gives
//! for a deployment that named no directory. Every kernel then compiles
//! through NVRTC and nothing is stored, which costs time and never an answer:
//! every failure of a cache under here is a miss and never an error.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// **Every NVRTC artifact this deployment keeps, in one directory.**
///
/// Shared with the engine's guest-program plane, which writes its own cubins
/// here rather than into the `ptir-cuda` it had to itself: the two producers
/// store the same kind of thing — a cubin ELF image `cuModuleLoadData` loads —
/// and a directory named after either one of them would be a lie about the
/// other's half of its contents.
///
/// The two name their files apart without having to agree on anything: this
/// crate writes `{key:016x}.cubin` and the program plane writes
/// `{key:016x}-{region}.cubin`. Both also store the full key inside the file
/// and compare it on the way back out, so a digest collision is a miss rather
/// than the wrong cubin.
pub const CUBINS: &str = "cubins";

/// The measured cuBLASLt algorithm table's directory.
///
/// Not under [`CUBINS`], and the distinction is the point: what lands there is
/// compiler output, and this is a heuristic RESULT — no NVRTC ran to produce
/// it, and deleting it costs a re-measurement rather than a re-compile.
pub const GEMM_ALGOS: &str = "gemm-algos";

static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// State this process's cache root, once.
///
/// The first call wins and later ones are dropped, which is what makes this
/// safe to call from every door that can start a compile — a shell booting, a
/// test opening its device — without any of them having to know whether it is
/// the first. `None` installs the feature off explicitly.
pub fn install(root: Option<&Path>) {
    let _ = ROOT.set(root.map(Path::to_path_buf));
}

/// One named subdirectory of the root, or `None` when nothing is stored.
///
/// The directory is not created here: a reader that finds nothing must not
/// leave a directory behind, so the writers create their own parent on the way
/// past.
#[must_use]
pub fn dir(name: &str) -> Option<PathBuf> {
    ROOT.get()?.as_ref().map(|root| root.join(name))
}
