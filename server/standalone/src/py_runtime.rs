//! Python WASM runtime install — fetches the
//! `componentize-py-runtime` core module and unpacks it under
//! `$PIE_HOME/py-runtime/` so Python inferlets have something to
//! link against.
//!
//! Mirrors `sdk/tools/bakery/src/bakery/py_runtime.py` (the canonical
//! installer used by the Python `pie config init` and `Server.__aenter__`
//! paths). Both call paths land at the same on-disk layout, so a tarball
//! pulled by either is reused by the other.
//!
//! Pinned `RUNTIME_URL` matches bakery's; bumping requires a coordinated
//! change with `runtime/program/python` on the Rust side.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

const RUNTIME_URL: &str =
    "https://registry.pie-project.org/api/v1/runtimes/python3.14/0.3.0/download";

/// Where the host loader expects to find the runtime tree.
pub fn runtime_dir() -> PathBuf {
    pie::path::get_pie_home().join("py-runtime")
}

/// Sentinel file the host loader links against. Its presence is what
/// `is_installed` checks for — if it's there, the rest of the tree is
/// almost certainly intact.
fn sentinel() -> PathBuf {
    runtime_dir()
        .join("shared")
        .join("componentize-py-runtime.wasm")
}

pub fn is_installed() -> bool {
    sentinel().is_file()
}

/// Install the runtime if it isn't already. Returns the runtime
/// directory either way.
///
/// `quiet` suppresses the progress line on the engine startup path
/// where the user shouldn't be asked to read a download bar. When
/// invoked manually via `pie config init`, callers pass `quiet=false`
/// so the user sees the progress.
pub fn ensure_installed(quiet: bool) -> Result<PathBuf> {
    let dir = runtime_dir();
    if is_installed() {
        return Ok(dir);
    }

    let pie_home = pie::path::get_pie_home();
    std::fs::create_dir_all(&pie_home)
        .map_err(|e| anyhow!("create {pie_home:?}: {e}"))?;

    if !quiet {
        eprintln!("Downloading Python WASM runtime from {RUNTIME_URL}…");
    }
    let blob = fetch()?;

    if !quiet {
        eprintln!("Extracting runtime to {}…", pie_home.display());
    }
    extract(&blob, &pie_home)?;

    if !is_installed() {
        bail!(
            "Python runtime download completed but {dir:?} is incomplete \
             (missing {})",
            sentinel().display()
        );
    }
    Ok(dir)
}

/// Best-effort install for the engine startup path. Logs and swallows
/// failures so a missing network doesn't block users who aren't
/// running Python inferlets at all.
pub fn ensure_installed_best_effort() {
    if is_installed() {
        return;
    }
    if let Err(e) = ensure_installed(/*quiet=*/ true) {
        eprintln!(
            "warning: could not auto-install Python WASM runtime ({e}); \
             Python inferlets will fail to instantiate until you run \
             `pie config init` manually."
        );
    }
}

/// Synchronous tarball download. Uses `reqwest::blocking` — we're
/// either in a non-async context (`pie config init`) or this is a
/// best-effort startup install where briefly blocking the main
/// thread is fine (matches Python's `asyncio.to_thread` behavior).
fn fetch() -> Result<Vec<u8>> {
    let resp = reqwest::blocking::Client::new()
        .get(RUNTIME_URL)
        .send()
        .map_err(|e| anyhow!("GET {RUNTIME_URL}: {e}"))?
        .error_for_status()
        .map_err(|e| anyhow!("GET {RUNTIME_URL}: {e}"))?;
    resp.bytes()
        .map(|b| b.to_vec())
        .map_err(|e| anyhow!("read response body: {e}"))
}

/// xz-decompress + untar into `dest`. Mirrors what bakery's Python
/// equivalent does via `lzma.decompress` + `tarfile.extractall`.
fn extract(blob: &[u8], dest: &Path) -> Result<()> {
    let mut decoder = xz2::read::XzDecoder::new(blob);
    let mut tar = tar::Archive::new(&mut decoder);
    tar.unpack(dest)
        .map_err(|e| anyhow!("extract tarball into {dest:?}: {e}"))?;
    Ok(())
}
