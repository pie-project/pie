//! The worker inside an app's own process: no gateway and no controller.
//! Clients reach the runtime through in-process sessions that carry
//! `pie serve`'s MessagePack frames.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

pub use crate::boot::{Embedded, Engine, Loaded, Summary};
use crate::config::{ByteSize, Config, EngineKind};

/// The knobs an app sets (`PieServer.Configuration` in Swift and Kotlin, the
/// browser's boot config), laid onto a worker [`Config`] sized for a phone or
/// a tab by [`Settings::config`].
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// False boots the runtime with no engine, for programs that never run a
    /// forward pass.
    pub engine: bool,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
    pub max_total_pages: u32,
    pub max_state_slots: u32,
    pub max_model_len: u32,
    pub gpu_mem_utilization: f64,
    pub sandbox_memory_mb: u64,
    pub max_concurrent_processes: Option<usize>,
    pub frame_size: u32,
    pub frame_dispatch_depth: u32,
    pub verbose: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            engine: true,
            max_forward_tokens: 512,
            max_forward_requests: 8,
            max_total_pages: 512,
            max_state_slots: 64,
            max_model_len: 4096,
            gpu_mem_utilization: 0.9,
            sandbox_memory_mb: 512,
            max_concurrent_processes: None,
            frame_size: 8,
            frame_dispatch_depth: 2,
            verbose: false,
        }
    }
}

impl Settings {
    /// TOML, or the same document as JSON.
    pub fn parse(text: &str) -> Result<Self> {
        if text.trim_start().starts_with('{') {
            return serde_json::from_str(text).context("parse the settings");
        }
        toml::from_str(text).context("parse the settings")
    }

    /// The worker config that serves `artifact` from `home`: this build's
    /// engine with these budgets, and a sandbox with no network or files.
    pub fn config(&self, artifact: &Path, home: &Path) -> Result<Config> {
        let kind = native_kind();
        let mut document = toml::Table::new();
        let mut model = toml::Table::new();
        model.insert("name".into(), "default".into());
        model.insert(
            "model".into(),
            artifact
                .to_str()
                .ok_or_else(|| anyhow!("{} is not UTF-8", artifact.display()))?
                .into(),
        );
        let mut engine = toml::Table::new();
        engine.insert("type".into(), kind.as_str().into());
        engine.insert("device".into(), format!("{}:0", kind.as_str()).into());
        model.insert("engine".into(), engine.into());
        document.insert("model".into(), model.into());
        let mut config = Config::parse(&document.to_string())?;

        config.model.engine.options = self.engine_options(kind, home);
        config.server.worker_threads = 2;
        config.server.verbose = self.verbose;
        config.server.max_upload = ByteSize::from_mib(256);
        config.runtime.frame_size = self.frame_size;
        config.runtime.frame_dispatch_depth = self.frame_dispatch_depth;
        config.runtime.max_concurrent_processes = self.max_concurrent_processes;
        let sandbox = &mut config.sandbox;
        sandbox.max_instances = 16;
        sandbox.max_memory = ByteSize::from_mib(self.sandbox_memory_mb);
        sandbox.warm_memory = ByteSize::default();
        sandbox.warm_slots = 0;
        sandbox.allow_fs = false;
        sandbox.fs_scratch_dir = home.join("scratch");
        sandbox.allow_network = false;
        sandbox.network_allowed_hosts.clear();
        Ok(config)
    }

    fn engine_options(&self, kind: EngineKind, home: &Path) -> toml::Table {
        let mut options = toml::Table::new();
        let mut set = |key: &str, value: toml::Value| {
            options.insert(key.into(), value);
        };
        set("gpu_mem_utilization", self.gpu_mem_utilization.into());
        set(
            "max_forward_tokens",
            i64::from(self.max_forward_tokens).into(),
        );
        set(
            "max_forward_requests",
            i64::from(self.max_forward_requests).into(),
        );
        set("max_state_slots", i64::from(self.max_state_slots).into());
        set("max_model_len", i64::from(self.max_model_len).into());
        let pages = i64::from(self.max_total_pages).into();
        match kind {
            EngineKind::Metal => {
                set("total_pages", pages);
                set("kv_page_size", 16.into());
            }
            EngineKind::Vulkan => {
                set("max_total_pages", pages);
                let cache = home.join("vulkan-pipelines.bin");
                set(
                    "pipeline_cache",
                    cache.to_string_lossy().into_owned().into(),
                );
            }
            _ => set("max_total_pages", pages),
        }
        options
    }
}

/// The engine this build serves on: Metal on Apple, else Vulkan, else WebGPU.
fn native_kind() -> EngineKind {
    if cfg!(all(feature = "metal", target_vendor = "apple")) {
        EngineKind::Metal
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
