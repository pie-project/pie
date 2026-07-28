//! The hand-written device text an emitted kernel is built around, as it stood
//! when the C++ oracle was dumped.
//!
//! `compiler/codegen/runtime/{cuda,metal}/` is device source the kernel side
//! owns. It is `include_str!`ed verbatim into every emitted kernel, and it
//! moves for reasons the compiler has no say in: `3aeb4162a` shipped a decode
//! parallelism fix that grew the CUDA prologue by 1,266 bytes and
//! `fused_block0.cuh` by 1,006. The golden dumps are the output of a C++
//! program that has since been deleted, so they can only ever be re-read, never
//! re-taken — held against today's device text they would decay into a suite
//! that forbids anyone from touching a kernel, and the first person to hit that
//! would regenerate the dumps and throw the oracle away.
//!
//! So the oracle carries its own inputs. A device file that has moved since the
//! dump was taken keeps its oracle-era text beside the dump under
//! `golden-*/oracle-inputs/`, and an emitted source is rewritten back onto that
//! text before the diff. Everything the *emitter* decides is still compared byte
//! for byte; the bytes the kernel side decides are not compared at all, because
//! the oracle has nothing left to say about them. The rewrite cannot hide an
//! emitter change: it only fires where the live file was spliced in whole and
//! verbatim, so dropping, duplicating, truncating or reordering a block still
//! lands in the diff — and `oracle_inputs_are_live_files_that_moved` refuses a
//! stale or invented entry.

// Each test binary uses the part of this it needs.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The oracle-era text of the device files that have moved since the dump.
pub struct OracleInputs {
    /// `(live text, oracle-era text)`, longest first so that a file which
    /// contains another cannot be half-rewritten.
    substitutions: Vec<(String, String)>,
    runtime_dir: PathBuf,
    oracle_dir: PathBuf,
}

fn read_dir_sorted(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| path.is_file())
        .collect();
    paths.sort();
    paths
}

impl OracleInputs {
    /// `runtime_dir` is the live device source; `oracle_dir` is the oracle-era
    /// copy of whichever of those files have since changed.
    pub fn load(runtime_dir: PathBuf, oracle_dir: PathBuf) -> Self {
        let mut substitutions = Vec::new();
        for path in read_dir_sorted(&oracle_dir) {
            let name = path.file_name().expect("named file").to_owned();
            let live_path = runtime_dir.join(&name);
            let live = std::fs::read_to_string(&live_path).unwrap_or_else(|error| {
                panic!(
                    "{} has an oracle-era copy but no live file ({error})",
                    live_path.display()
                )
            });
            let oracle = std::fs::read_to_string(&path).expect("readable oracle-era text");
            substitutions.push((live, oracle));
        }
        substitutions.sort_by_key(|(live, _)| core::cmp::Reverse(live.len()));
        Self {
            substitutions,
            runtime_dir,
            oracle_dir,
        }
    }

    /// Put an emitted kernel back onto the device text the oracle was taken
    /// with. A no-op for every backend whose device source has not moved.
    pub fn rewrite(&self, source: &str) -> String {
        let mut source = source.to_string();
        for (live, oracle) in &self.substitutions {
            if source.contains(live.as_str()) {
                source = source.replace(live.as_str(), oracle);
            }
        }
        source
    }

    /// What `compare` should tell whoever just changed a kernel.
    pub fn hint(&self) -> String {
        format!(
            "if a file under {} changed, its previous text belongs in {} — the \
             dump is only comparable against the inputs it was taken with, and \
             regenerating it would destroy the oracle instead",
            self.runtime_dir.display(),
            self.oracle_dir.display()
        )
    }

    /// Every entry stands in for a live file that really has moved. A copy that
    /// matched its live file would silently absorb the next drift as well.
    pub fn assert_entries_are_live_files_that_moved(&self) {
        for path in read_dir_sorted(&self.oracle_dir) {
            let name = path.file_name().expect("named file");
            let live_path = self.runtime_dir.join(name);
            let live = std::fs::read_to_string(&live_path).unwrap_or_else(|error| {
                panic!(
                    "{} has an oracle-era copy but no live file ({error})",
                    live_path.display()
                )
            });
            let oracle = std::fs::read_to_string(&path).expect("readable oracle-era text");
            assert_ne!(
                live,
                oracle,
                "{} is byte-identical to its live file, so it stands in for \
                 nothing and would quietly swallow the next device change; \
                 delete it",
                path.display()
            );
        }
        assert_eq!(
            self.substitutions.len(),
            read_dir_sorted(&self.oracle_dir).len(),
            "every oracle-era input has to be loaded"
        );
    }

    /// The live device files, so a caller can check the emitter still splices
    /// each of them in whole — which is what makes the rewrite safe.
    pub fn live_files(&self) -> Vec<(String, String)> {
        read_dir_sorted(&self.runtime_dir)
            .into_iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .expect("named file")
                    .to_string_lossy()
                    .into_owned();
                let text = std::fs::read_to_string(&path).expect("readable device source");
                (name, text)
            })
            .collect()
    }
}

/// Every hand-written device file under `compiler/codegen/runtime/`, across
/// backends, longest first.
pub fn live_device_text(runtime_root: &Path) -> Vec<String> {
    let mut backends = read_dir_all(runtime_root);
    backends.sort();
    let mut texts: Vec<String> = backends
        .iter()
        .flat_map(|backend| read_dir_sorted(backend))
        .map(|path| std::fs::read_to_string(&path).expect("readable device source"))
        .collect();
    texts.sort_by_key(|text| core::cmp::Reverse(text.len()));
    texts
}

/// Drop the hand-written device text from an emitted kernel, leaving the bytes
/// the compiler generated.
///
/// A pin over the whole source would move every time a kernel engineer touched
/// a `.cuh` — churn that says nothing about the compiler and buries the diff
/// that does. What the emitter splices those blocks into is pinned here; that it
/// still splices them at all is `every_live_device_file_is_spliced_into_a_kernel_verbatim`.
pub fn elide_device_text(source: &str, device: &[String]) -> String {
    let mut source = source.to_string();
    for text in device {
        if source.contains(text.as_str()) {
            source = source.replace(text.as_str(), "");
        }
    }
    source
}

fn read_dir_all(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| path.is_dir())
        .collect()
}
