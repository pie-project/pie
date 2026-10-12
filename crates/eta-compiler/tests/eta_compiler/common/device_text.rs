#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub struct OracleInputs {
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

    pub fn rewrite(&self, source: &str) -> String {
        let mut source = source.to_string();
        for (live, oracle) in &self.substitutions {
            if source.contains(live.as_str()) {
                source = source.replace(live.as_str(), oracle);
            }
        }
        source
    }

    pub fn hint(&self) -> String {
        format!(
            "if a file under {} changed, its previous text belongs in {} — the \
             dump is only comparable against the inputs it was taken with, and \
             regenerating it would destroy the oracle instead",
            self.runtime_dir.display(),
            self.oracle_dir.display()
        )
    }

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
