//! The engine's config: the `[cerebras]` boot table the worker writes from `[engine]`. Every
//! table refuses an unknown key or a mistyped value: a misspelt setting is an error, not a
//! silently different engine.

use std::path::PathBuf;

use kernels_cerebras::spatial::toolchain::Toolchain;
use serde::{Deserialize, Serialize};

/// The wafer generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    Wse2,
    #[default]
    Wse3,
}

/// The `[cerebras]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub target: Target,
    /// The CS system's `host:port`; none runs the fabric simulator.
    pub cmaddr: Option<String>,
    /// Simulator threads.
    pub num_threads: u32,
    /// Where compiled programs and cut tiles are kept (the worker's engine cache).
    pub cache_dir: Option<PathBuf>,
    /// The most steps a greedy lane's token loop runs past its row (0: off; loops start at 8
    /// and double while a sequence uses them up). None: the engine's default.
    pub lookahead: Option<usize>,
    /// Whether frames settle asynchronously: each step lands on the completion sink once its
    /// epilogue ran. None / false: each frame runs whole inside its submit.
    #[serde(rename = "async")]
    pub asynchronous: Option<bool>,
    /// The most steps a posted token loop runs past its row (default: the program's plan's).
    pub posted_loop: Option<usize>,
    pub wafer: Wafer,
    pub toolchain: Toolchain,
    /// Bench and experiment settings; a load with any logs them.
    pub debug: Option<Debug>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            target: Target::Wse3,
            cmaddr: None,
            num_threads: 16,
            cache_dir: None,
            lookahead: None,
            asynchronous: None,
            posted_loop: None,
            wafer: Wafer::default(),
            toolchain: Toolchain::default(),
            debug: None,
        }
    }
}

/// `[cerebras.wafer]`: how the wafer program is chosen, built and fed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Wafer {
    /// The family's preset by name (else the first that builds).
    pub preset: Option<String>,
    pub build: Build,
    pub tiles: Tiles,
}

/// Whether a load compiles a program it has not cached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Build {
    /// Compile it (minutes for a whole wafer) and cache it.
    #[default]
    OnLoad,
    /// Refuse: a chip session serves only what was compiled before.
    Never,
}

/// Where a load's cut of the checkpoint lives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tiles {
    /// Kept in the cache (one cut a model: a new one replaces it).
    #[default]
    Cache,
    /// Cut for this load and dropped once placed.
    Memory,
}

/// `[cerebras.debug]`: bench and experiment settings, none of which a served model needs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Debug {
    /// Open this linked program (its `artifact/`) instead of building one.
    pub program: Option<PathBuf>,
    /// Place these staged tiles (`<dir>` or `<dir>:<file prefix>`) instead of cutting the
    /// checkpoint.
    pub weights: Option<String>,
    /// Overrides of the preset's keys (`K=V,...`).
    pub set: String,
    /// Kernel levers past the preset's (`K:V,...`).
    pub levers: String,
    /// Feed the prompt through the token loop (the program must be built for it).
    pub force_prompt: bool,
    /// `false`: one greedy loop at a time instead of posted back to back.
    pub posted: Option<bool>,
    /// `false`: a loop runs the lookahead alone, not bounded by the lane's pages.
    pub loop_room: Option<bool>,
}

impl Config {
    /// The config in a boot document.
    pub fn parse(bytes: &[u8]) -> Result<Config, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| format!("the cerebras boot config is not utf-8: {e}"))?;
        let doc: toml::Table = text
            .parse()
            .map_err(|e| format!("the cerebras boot config is not TOML: {e}"))?;
        let table = doc.get("cerebras").cloned().unwrap_or_else(|| toml::Value::Table(Default::default()));
        table
            .try_into()
            .map_err(|e| format!("[cerebras]: {e}"))
    }

    /// The boot document of this config.
    pub fn render(&self) -> Result<String, String> {
        let mut doc = toml::Table::new();
        doc.insert(
            "cerebras".into(),
            toml::Value::try_from(self).map_err(|e| e.to_string())?,
        );
        toml::to_string(&doc).map_err(|e| e.to_string())
    }

    pub fn debug(&self) -> Debug {
        self.debug.clone().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(text: &str) -> Result<Config, String> {
        Config::parse(text.as_bytes())
    }

    #[test]
    fn defaults_and_round_trip() {
        let c = of("").unwrap();
        assert_eq!(c, Config::default());
        let c = of("[cerebras]\ncmaddr = \"10.0.0.1:9000\"\nlookahead = 32\nasync = true\n\
                    [cerebras.wafer]\npreset = \"9b\"\nbuild = \"never\"\n\
                    [cerebras.toolchain]\ncompile_timeout = 600\n")
        .unwrap();
        assert_eq!(c.lookahead, Some(32));
        assert_eq!(c.wafer.build, Build::Never);
        assert_eq!(c.toolchain.compile_timeout, Some(600));
        assert_eq!(Config::parse(c.render().unwrap().as_bytes()).unwrap(), c);
    }

    /// The config is the whole configuration: neither this crate nor kernels-cerebras reads a
    /// `PIE_CEREBRAS_*` variable or a source-tree path at run time.
    #[test]
    fn nothing_reads_the_environment() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut hits = Vec::new();
        for dir in [root.join("src"), root.join("../kernels-cerebras/src")] {
            let mut stack = vec![dir];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).unwrap().flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                        continue;
                    }
                    let text = std::fs::read_to_string(&p).unwrap_or_default();
                    for (n, line) in text.lines().enumerate() {
                        let code = line.split("//").next().unwrap_or("");
                        let env = ["var(", "var_os("].iter().any(|v| {
                            code.contains(&format!("env::{v}\"PIE_CEREBRAS"))
                        });
                        if env || code.contains("\"tools/wafer") {
                            hits.push(format!("{}:{}: {line}", p.display(), n + 1));
                        }
                    }
                }
            }
        }
        assert!(hits.is_empty(), "{hits:#?}");
    }

    #[test]
    fn unknown_keys_and_bad_values_are_refused() {
        assert!(of("[cerebras]\nprogram = \"p\"\n").is_err());
        assert!(of("[cerebras.wafer]\nbuild = \"sometimes\"\n").is_err());
        assert!(of("[cerebras.debug]\nposted_loop = 4\n").is_err());
        assert!(of("[cerebras]\nlookahead = \"8\"\n").is_err());
    }
}
