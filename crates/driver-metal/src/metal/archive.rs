//! Where compiled pipelines are kept between runs.
//!
//! Compiling the load's kernels from MSL takes seconds, and almost none of it
//! is parsing -- it is the back end turning each function into a GPU binary.
//! Metal 4 will hand that binary back on a later run if it was written to an
//! archive, which turns a cold start into a lookup.
//!
//! # The archive is keyed, not named
//!
//! An archive holds the binaries for one exact batch. Serving one built from
//! different sources would be a silent miscompile rather than a slow start,
//! so the filename IS the key: [`Batch::key`] over the entry points asked for
//! and the resolved text of every file they came out of, salted with the GPU
//! and the language version. Editing a kernel changes the key, which misses,
//! which recompiles.
//!
//! That is also why nothing here ever overwrites or updates an archive. A key
//! either matches or it does not.
//!
//! # Which means old ones pile up
//!
//! Every edit to a kernel source strands the archive keyed to the version
//! before it, at a few megabytes each, in a directory nothing else prunes.
//! [`Archives::prune`] deletes what has not been touched in [`MAX_AGE`] --
//! by modification time and not by creation, so an archive still being served
//! on every start is not deleted for being old.
//!
//! # Absence is not failure
//!
//! Every operation here degrades to "no cache" rather than to an error. There
//! is no `HOME`, the directory cannot be created, the disk is full: none of
//! those stop a pipeline from being compiled, and turning any of them into a
//! failed load would trade a working slow start for a broken one. The one
//! thing that must never be silent is the opposite case -- an archive that
//! IS found and is wrong -- and the key is what rules that out.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The environment variable that relocates the cache, or disables it.
///
/// Set to a path, that path is used. Set to empty, there is no cache at all
/// -- which is the only way to ask for that, and is what a benchmark of cold
/// compilation needs.
pub const CACHE_ENV: &str = "PIE_METAL_PSO_CACHE";

/// The extension every archive written here carries.
///
/// [`Archives::prune`] deletes only files that end in it, so a directory that
/// is also something else does not lose the something else.
pub const EXTENSION: &str = "mtl4archive";

/// How long an unused archive is kept.
///
/// Two weeks, which is long enough that a branch switched away from and back
/// still starts warm, and short enough that a directory of stale archives is
/// bounded rather than merely growing slowly.
pub const MAX_AGE: Duration = Duration::from_secs(14 * 24 * 60 * 60);

/// The directory pipeline archives are read from and written to.
///
/// `None` for the directory means there is no cache. That is a state the type
/// holds rather than refuses, because every caller's response to it is the
/// same: compile, and do not try to save the result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Archives {
    dir: Option<PathBuf>,
}

impl Archives {
    /// The cache directory this machine should use.
    ///
    /// [`CACHE_ENV`] wins if set, including when it is set to empty -- that is
    /// how caching is turned off. Otherwise the per-user cache directory, and
    /// no cache at all if there is no `HOME` to put it under.
    #[must_use]
    pub fn discover() -> Self {
        Self::from_env(
            std::env::var_os(CACHE_ENV).map(PathBuf::from),
            std::env::var_os("HOME").map(PathBuf::from),
        )
    }

    /// [`Archives::discover`] over values the caller supplies, for tests.
    #[must_use]
    pub fn from_env(override_dir: Option<PathBuf>, home: Option<PathBuf>) -> Self {
        if let Some(dir) = override_dir {
            return Self::new(if dir.as_os_str().is_empty() {
                None
            } else {
                Some(dir)
            });
        }
        Self::new(
            home.filter(|home| !home.as_os_str().is_empty())
                .map(|home| home.join("Library/Caches/pie-metal")),
        )
    }

    /// An explicit directory, or `None` for no cache.
    #[must_use]
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self { dir }
    }

    /// The directory, if there is one.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The file an archive with this key would live at.
    ///
    /// Creates the directory, because the caller's next move is either to read
    /// this path or to write it and both need it to exist. `None` if there is
    /// no cache or the directory could not be made -- see the module docs on
    /// why that is not an error.
    #[must_use]
    pub fn path(&self, key: u64) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        std::fs::create_dir_all(dir).ok()?;
        Some(dir.join(format!("psos-{key:016x}.{EXTENSION}")))
    }

    /// Delete archives untouched for longer than `max_age`.
    ///
    /// Returns how many were deleted. Errors are swallowed one file at a time:
    /// an archive another process holds open, or one owned by another user, is
    /// a file to skip rather than a reason to stop pruning the rest.
    pub fn prune(&self, max_age: Duration) -> usize {
        let Some(dir) = self.dir.as_ref() else {
            return 0;
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return 0;
        };
        let now = SystemTime::now();
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != EXTENSION) {
                continue;
            }
            // Modified, not created: an archive served on every start is not
            // stale, and its modification time is what a served archive keeps
            // current.
            let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) else {
                continue;
            };
            // A file dated in the future has an unreadable age, not a huge
            // one. `duration_since` erroring is that case, and it is left.
            if now.duration_since(modified).is_ok_and(|age| age > max_age)
                && std::fs::remove_file(&path).is_ok()
            {
                removed += 1;
            }
        }
        removed
    }
}

impl Default for Archives {
    fn default() -> Self {
        Self::discover()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pie-archive-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn an_empty_override_is_how_caching_is_turned_off() {
        let archives = Archives::from_env(Some(PathBuf::new()), Some(PathBuf::from("/home/x")));
        assert_eq!(archives.dir(), None);
        assert_eq!(archives.path(7), None);
    }

    #[test]
    fn the_override_beats_the_home_directory() {
        let archives = Archives::from_env(
            Some(PathBuf::from("/tmp/elsewhere")),
            Some(PathBuf::from("/home/x")),
        );
        assert_eq!(archives.dir(), Some(Path::new("/tmp/elsewhere")));
    }

    #[test]
    fn no_home_and_no_override_is_no_cache() {
        assert_eq!(Archives::from_env(None, None).dir(), None);
        assert_eq!(Archives::from_env(None, Some(PathBuf::new())).dir(), None);
    }

    #[test]
    fn the_key_is_the_filename() {
        let dir = scratch("name");
        let archives = Archives::new(Some(dir.clone()));
        let path = archives.path(0x0123_4567_89ab_cdef).expect("path");
        assert_eq!(
            path,
            dir.join("psos-0123456789abcdef.mtl4archive"),
            "the key is the whole name, zero padded, so two keys cannot share a file"
        );
        assert!(dir.is_dir(), "asking for a path creates the directory");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_deletes_only_old_archives_and_only_archives() {
        let dir = scratch("prune");
        let archives = Archives::new(Some(dir.clone()));
        let old = archives.path(1).expect("old");
        let new = archives.path(2).expect("new");
        let other = dir.join("notes.txt");
        for path in [&old, &new, &other] {
            std::fs::write(path, b"x").expect("write");
        }
        // Nothing is old yet, so nothing goes.
        assert_eq!(archives.prune(MAX_AGE), 0);

        // A zero max age makes everything old -- except the file that is not
        // an archive, which is the half worth asserting.
        assert_eq!(archives.prune(Duration::ZERO), 2);
        assert!(!old.exists() && !new.exists());
        assert!(other.exists(), "prune touches only {EXTENSION} files");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pruning_nothing_is_not_an_error() {
        assert_eq!(Archives::new(None).prune(Duration::ZERO), 0);
        assert_eq!(
            Archives::new(Some(PathBuf::from("/nonexistent/pie/cache"))).prune(Duration::ZERO),
            0
        );
    }
}
