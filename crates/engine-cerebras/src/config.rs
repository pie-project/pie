//! The engine's config: the `[cerebras]` boot table the worker writes from `[engine]`. Every
//! table refuses an unknown key or a mistyped value: a misspelt setting is an error, not a
//! silently different engine.

use std::path::PathBuf;

use kernels_cerebras::compile::toolchain::Toolchain;
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
    /// Simulator threads (none: the host's available parallelism).
    pub num_threads: Option<u32>,
    /// Where compiled programs and cut tiles are kept (the worker's engine cache).
    pub cache_dir: Option<PathBuf>,
    /// The most steps a greedy lane's token loop runs past its row (0: off). None: the engine's
    /// default.
    pub lookahead: Option<usize>,
    /// Whether frames settle asynchronously, each step landing on the completion sink. None /
    /// false: each frame runs whole inside its submit.
    #[serde(rename = "async")]
    pub asynchronous: Option<bool>,
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
            num_threads: None,
            cache_dir: None,
            lookahead: None,
            asynchronous: None,
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
    pub build: Build,
    pub tiles: Tiles,
    /// A stream reading's rows a class a step: [`DEFAULT_STREAM_ROWS`] unless stated.
    pub stream_rows: Option<u32>,
    /// The annealing placer's moves a block after the search for a stream or unit program
    /// (kernels-cerebras `place::anneal`): off unless stated.
    pub anneal: Option<u64>,
}

/// The rows a class a stream program is compiled for when `stream_rows` is not stated.
pub const DEFAULT_STREAM_ROWS: u32 = 64;

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
    /// Open this linked program (its `artifact/`) instead of building one; or a stream reading's
    /// program directory (its `host.json`), linked in `link`.
    pub program: Option<PathBuf>,
    /// A stream program's link (cslc's output), else `<program>/out`.
    pub link: Option<PathBuf>,
    /// `"host"`: a stream reading's steps on the host forward (`reading::HostFabric`); else the
    /// configured system or the simulator.
    pub stream_fabric: Option<String>,
    /// Place these staged tiles (`<dir>` or `<dir>:<file prefix>`) instead of cutting the
    /// checkpoint.
    pub weights: Option<String>,
    /// Overrides of the plan's keys (`K=V,...`).
    pub set: String,
    /// Kernel levers past the plan's (`K:V,...`).
    pub levers: String,
}

impl Config {
    /// The config in a boot document.
    pub fn parse(bytes: &[u8]) -> Result<Config, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| format!("the cerebras boot config is not utf-8: {e}"))?;
        let doc: toml::Table = text
            .parse()
            .map_err(|e| format!("the cerebras boot config is not TOML: {e}"))?;
        let table = doc
            .get("cerebras")
            .cloned()
            .unwrap_or_else(|| toml::Value::Table(Default::default()));
        table.try_into().map_err(|e| format!("[cerebras]: {e}"))
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
        let c = of(
            "[cerebras]\ncmaddr = \"10.0.0.1:9000\"\nlookahead = 32\nasync = true\n\
                    [cerebras.wafer]\nbuild = \"never\"\n\
                    [cerebras.toolchain]\ncompile_timeout = 600\n\
                    [cerebras.debug]\nset = \"G2=8\"\n",
        )
        .unwrap();
        assert_eq!(c.lookahead, Some(32));
        assert_eq!(c.wafer.build, Build::Never);
        assert_eq!(c.debug().set, "G2=8");
        assert_eq!(c.toolchain.compile_timeout, Some(600));
        assert_eq!(Config::parse(c.render().unwrap().as_bytes()).unwrap(), c);
    }

    #[test]
    fn unknown_keys_and_bad_values_are_refused() {
        assert!(of("[cerebras]\nprogram = \"p\"\n").is_err());
        assert!(of("[cerebras.wafer]\nbuild = \"sometimes\"\n").is_err());
        assert!(of("[cerebras.debug]\nposted_loop = 4\n").is_err());
        assert!(of("[cerebras]\nlookahead = \"8\"\n").is_err());
    }
}
