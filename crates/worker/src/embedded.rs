//! The worker inside an app's own process: no gateway and no controller.
//! Clients reach the runtime through in-process sessions that carry
//! `pie serve`'s MessagePack frames.

use std::path::Path;

use anyhow::{Result, anyhow};
use serde::Deserialize;

pub use crate::boot::{Embedded, Engine, Loaded, Summary};
use crate::config::{ByteSize, Config, EngineKind, Runahead};

/// The knobs an app sets (`PieServer.Configuration` in Swift and Kotlin, the
/// browser's boot config), laid onto a worker [`Config`] sized for a phone or
/// a tab by [`Settings::config`].
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// False boots the runtime with no engine, for programs that never run a
    /// forward pass.
    pub engine: bool,
    /// See [`crate::config::EngineConfig::max_context`].
    pub max_context: u32,
    /// See [`crate::config::MemoryConfig::budget_device`]: on a phone, what
    /// the app leaves the model of the memory it may wire.
    pub budget_device: Option<ByteSize>,
    /// See [`crate::config::MemoryConfig::yield_share`]: `yield`.
    #[serde(rename = "yield")]
    pub yield_share: Option<f64>,
    pub sandbox_memory_mb: u64,
    pub max_concurrent_processes: Option<usize>,
    /// See [`crate::config::RuntimeConfig::frame_size`]. An app serves few
    /// lanes, so it defaults to the most waves a frame takes
    /// ([`Runahead::STEPS_MAX`]), the bound the worker config checks it by.
    pub frame_size: u32,
    /// See [`crate::config::RuntimeConfig::frame_dispatch_depth`].
    pub frame_dispatch_depth: u32,
    /// See [`crate::config::EngineConfig::measure`]: `false` times nothing
    /// at load, each rate served by the estimate, named in the log.
    pub measure: bool,
    pub verbose: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            engine: true,
            max_context: 4096,
            budget_device: None,
            yield_share: None,
            sandbox_memory_mb: 512,
            max_concurrent_processes: None,
            frame_size: u32::from(Runahead::STEPS_MAX),
            frame_dispatch_depth: 2,
            measure: true,
            verbose: false,
        }
    }
}

impl Settings {
    /// TOML, or the same document as JSON.
    pub fn parse(text: &str) -> Result<Self> {
        let parsed = if text.trim_start().starts_with('{') {
            serde_json::from_str(text).map_err(|e| e.to_string())
        } else {
            toml::from_str(text).map_err(|e: toml::de::Error| e.message().to_string())
        };
        parsed.map_err(|e| anyhow!("parse the settings: {e}"))
    }

    /// The worker config that serves `artifact` from `home`: this build's
    /// engine with these budgets, and a sandbox with no network or files.
    pub fn config(&self, artifact: &Path, home: &Path) -> Result<Config> {
        let kind = native_kind();
        let mut document = toml::Table::new();
        let mut model = toml::Table::new();
        model.insert("name".into(), "default".into());
        model.insert(
            "source".into(),
            artifact
                .to_str()
                .ok_or_else(|| anyhow!("{} is not UTF-8", artifact.display()))?
                .into(),
        );
        let mut engine = toml::Table::new();
        engine.insert("type".into(), kind.as_str().into());
        let device = match kind {
            EngineKind::Cuda => "cuda:0".to_string(),
            kind => format!("{}:0", kind.as_str()),
        };
        engine.insert("device".into(), device.into());
        document.insert("model".into(), model.into());
        document.insert("engine".into(), engine.into());
        let mut config = Config::parse(&document.to_string())?;

        config.engine.options = self.engine_options(kind, home);
        config.engine.max_context = Some(self.max_context);
        config.engine.measure = self.measure;
        config.engine.memory.budget_device = self.budget_device;
        config.engine.memory.yield_share = self.yield_share;
        config.engine.memory.yields()?;
        config.server.worker_threads = 2;
        config.server.verbose = self.verbose;
        config.server.max_upload = ByteSize::from_mib(256);
        config.runtime.frame_size = self.frame_size;
        config.runtime.frame_dispatch_depth = Some(self.frame_dispatch_depth);
        config.runtime.max_concurrent_processes = self.max_concurrent_processes;
        let sandbox = &mut config.sandbox;
        sandbox.max_instances = Some(16);
        sandbox.max_memory = ByteSize::from_mib(self.sandbox_memory_mb);
        sandbox.warm_memory = ByteSize::default();
        sandbox.warm_slots = 0;
        sandbox.allow_fs = false;
        sandbox.scratch_dir = home.join("scratch");
        sandbox.allow_network = false;
        sandbox.network_allowed_hosts.clear();
        config.validate()?;
        Ok(config)
    }

    fn engine_options(&self, kind: EngineKind, home: &Path) -> toml::Table {
        let mut options = toml::Table::new();
        let mut set = |key: &str, value: toml::Value| {
            options.insert(key.into(), value);
        };
        if kind == EngineKind::Vulkan {
            let cache = home.join("vulkan-pipelines.bin");
            set(
                "pipeline_cache",
                cache.to_string_lossy().into_owned().into(),
            );
        }
        options
    }
}

/// The engine this build serves on: Metal on Apple, else CUDA, else Vulkan,
/// else WebGPU.
fn native_kind() -> EngineKind {
    if cfg!(all(feature = "metal", target_vendor = "apple")) {
        EngineKind::Metal
    } else if cfg!(feature = "engine-cuda") {
        EngineKind::Cuda
    } else if cfg!(feature = "vulkan") {
        EngineKind::Vulkan
    } else {
        EngineKind::Wgpu
    }
}

/// Hands `session` one MessagePack `ClientMessage`.
pub fn send_frame(session: u32, frame: &[u8]) -> Result<()> {
    let message: client_api::ClientMessage =
        rmp_serde::from_slice(frame).map_err(|e| anyhow!("client frame: {e}"))?;
    runtime::server::send_client_message(session, message)
}

/// Up to `max` server frames, waiting at most `max_wait_ms` for the first.
pub async fn recv_frames(session: u32, max_wait_ms: u64, max: usize) -> Result<Vec<Vec<u8>>> {
    let messages = runtime::server::recv_messages(session, max_wait_ms, max).await?;
    messages
        .iter()
        .map(|message| rmp_serde::to_vec_named(message).map_err(anyhow::Error::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What an app that states nothing boots with is a config the worker
    /// takes: each default sits inside the bound `Config::validate` holds it to.
    #[test]
    fn the_default_settings_lay_a_valid_config() {
        let config = Settings::default()
            .config(Path::new("model.zt"), Path::new("home"))
            .expect("the default settings validate");
        assert_eq!(config.runtime.frame_size, u32::from(Runahead::STEPS_MAX));
    }
}
