use std::path::{Path, PathBuf};

mod model;
mod runner;

pub use runner::Runner;

pub struct Build {
    pub dir: PathBuf,
    pub hidden: u32,
    pub intermediate: u32,
    pub ane: u32,
    pub buckets: Vec<u32>,
    pub layers: Vec<u32>,
    input: String,
    output: String,
}

#[derive(Clone, Copy)]
pub struct Job {
    pub layer: u32,
    pub rows: u32,
    pub stage: u64,
    pub done: u64,
}

impl Build {
    pub fn find(fingerprint: &str, asked: &str) -> Option<Build> {
        if !asked.is_empty() {
            return Build::read(Path::new(asked), fingerprint);
        }
        let root = PathBuf::from(std::env::var("HOME").ok()?).join(".cache/pie/ane");
        let mut found = Vec::new();
        for model in std::fs::read_dir(root).ok()?.flatten() {
            let Ok(builds) = std::fs::read_dir(model.path()) else {
                continue;
            };
            for build in builds.flatten() {
                if let Some(build) = Build::read(&build.path(), fingerprint) {
                    let when = std::fs::metadata(build.dir.join("meta.json"))
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::UNIX_EPOCH);
                    found.push((when, build));
                }
            }
        }
        found.sort_by_key(|(when, _)| *when);
        found.pop().map(|(_, build)| build)
    }

    fn read(dir: &Path, fingerprint: &str) -> Option<Build> {
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).ok()?).ok()?;
        let int = |key: &str| meta.get(key)?.as_u64().and_then(|v| u32::try_from(v).ok());
        let ints = |key: &str| -> Option<Vec<u32>> {
            meta.get(key)?
                .as_array()?
                .iter()
                .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()))
                .collect()
        };
        let text = |key: &str| meta.get(key)?.as_str().map(str::to_string);
        if text("fingerprint")? != fingerprint {
            return None;
        }
        Some(Build {
            dir: dir.to_path_buf(),
            hidden: int("hidden")?,
            intermediate: int("intermediate")?,
            ane: int("ane")?,
            buckets: ints("buckets")?,
            layers: ints("layers")?,
            input: text("input")?,
            output: text("output")?,
        })
    }

    #[must_use]
    pub fn max_rows(&self) -> u32 {
        self.buckets.iter().copied().max().unwrap_or(0)
    }
}
