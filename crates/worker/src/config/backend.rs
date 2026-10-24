use std::path::PathBuf;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use units::ByteSize;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaEngineOptions {
    pub kv_page_size: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub max_model_len: Option<u32>,
    pub max_forward_tokens: Option<u32>,
    pub max_forward_requests: Option<u32>,
    #[serde(skip)]
    pub device: String,
    #[serde(skip)]
    pub verbose: bool,
    pub diagnostics: Option<String>,
    pub nccl_transport: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetalEngineOptions {
    pub kv_page_size: Option<u32>,
    pub max_forward_tokens: Option<u32>,
    pub max_forward_requests: Option<u32>,
    pub max_model_len: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub diagnostics: Option<String>,
    #[serde(skip)]
    pub device: String,
    #[serde(skip)]
    pub verbose: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VulkanEngineOptions {
    /// Which physical device, in `vkEnumeratePhysicalDevices` order: the
    /// number in `[engine] device` (`vulkan:1`).
    #[serde(skip)]
    pub device_index: u32,
    pub max_forward_tokens: Option<u32>,
    pub max_forward_requests: Option<u32>,
    pub max_model_len: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub validation: bool,
    pub pipeline_cache: Option<PathBuf>,
}

impl VulkanEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        limits_are_positive(self.max_forward_tokens, self.max_forward_requests)
    }
}

/// A stated fire ceiling of zero admits nothing.
fn limits_are_positive(tokens: Option<u32>, requests: Option<u32>) -> Result<()> {
    ensure!(
        tokens != Some(0),
        "engine.max_forward_tokens must be > 0; omit it for the engine's own"
    );
    ensure!(
        requests != Some(0),
        "engine.max_forward_requests must be > 0; omit it for the engine's own"
    );
    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WgpuEngineOptions {
    /// Which adapter, in enumeration order among the reachable ones: the
    /// number in `[engine] device` (`wgpu:1`).
    #[serde(skip)]
    pub adapter_index: u32,
    pub backends: Option<String>,
    pub power_preference: String,
    pub max_forward_tokens: Option<u32>,
    pub max_forward_requests: Option<u32>,
    pub max_model_len: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub pipeline_cache: Option<PathBuf>,
    pub device_memory: Option<ByteSize>,
}

impl Default for WgpuEngineOptions {
    fn default() -> Self {
        Self {
            adapter_index: 0,
            backends: None,
            power_preference: "high-performance".to_string(),
            max_forward_tokens: None,
            max_forward_requests: None,
            max_model_len: None,
            max_state_slots: None,
            pipeline_cache: None,
            device_memory: None,
        }
    }
}

impl WgpuEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.power_preference.as_str(),
                "high-performance" | "low-power" | "none"
            ),
            "engine.power_preference must be one of \"high-performance\", \
             \"low-power\", \"none\"; got {:?}",
            self.power_preference
        );
        limits_are_positive(self.max_forward_tokens, self.max_forward_requests)?;
        if let Some(memory) = self.device_memory {
            ensure!(
                memory.as_bytes() > 0,
                "engine.device_memory must be > 0; \
                 omit it to read the adapter's own answer"
            );
        }
        Ok(())
    }
}

/// The XLA engine: a PJRT plugin (libtpu on a TPU host) driving one device.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct XlaEngineOptions {
    /// The PJRT plugin to load. Omit to find one: `PIE_XLA_PLUGIN`, then
    /// `TPU_LIBRARY_PATH`, then `libtpu.so` on the loader's path.
    pub plugin: Option<PathBuf>,
    /// Which of the plugin's addressable devices this engine drives: the
    /// number in `[engine] device` (`tpu:1`).
    #[serde(skip)]
    pub device_index: u32,
    /// The most tokens one fire carries. Omit for the engine's own.
    pub max_forward_tokens: Option<u32>,
    /// The most requests one fire carries. Omit for the engine's own.
    pub max_forward_requests: Option<u32>,
    /// Recurrent-state seats (hybrid models).
    pub max_state_slots: Option<u32>,
    /// The longest sequence (prompt + output) one lane holds, in tokens.
    /// Omit for the engine default (4096).
    pub max_model_len: Option<u32>,
    /// Trace and compile the fire shapes a server meets first at load, so
    /// no early request waits on a compile. Slower to load.
    pub prewarm: bool,
}

impl XlaEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        limits_are_positive(self.max_forward_tokens, self.max_forward_requests)
    }
}

/// The Cerebras engine: the fabric simulator, or a CS system reached by
/// `cmaddr`, driving one wafer.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CerebrasEngineOptions {
    /// The wafer generation the program is compiled for: `"wse2"` or
    /// `"wse3"`.
    pub target: String,
    /// The CS system's `host:port`. Omit to run the fabric simulator.
    pub cmaddr: Option<String>,
    /// Host threads the simulator and the host-side staging use.
    pub num_threads: u32,
    /// The most tokens one fire carries. Omit for the engine's own.
    pub max_forward_tokens: Option<u32>,
    /// The most requests one fire carries. Omit for the engine's own.
    pub max_forward_requests: Option<u32>,
    /// Recurrent-state seats (hybrid models).
    pub max_state_slots: Option<u32>,
    /// The longest sequence (prompt + output) one lane holds, in tokens.
    /// Omit for the engine default (4096).
    pub max_model_len: Option<u32>,
    /// Trace and compile the first fire shapes at load.
    pub prewarm: bool,
}

impl Default for CerebrasEngineOptions {
    fn default() -> Self {
        Self {
            target: "wse3".to_string(),
            cmaddr: None,
            num_threads: 16,
            max_forward_tokens: None,
            max_forward_requests: None,
            max_state_slots: None,
            max_model_len: None,
            prewarm: false,
        }
    }
}

impl CerebrasEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.target.as_str(), "wse2" | "wse3"),
            "engine.target must be one of \"wse2\", \"wse3\"; got {:?}",
            self.target
        );
        ensure!(self.num_threads > 0, "engine.num_threads must be > 0");
        limits_are_positive(self.max_forward_tokens, self.max_forward_requests)
    }
}

impl CudaEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        if let Some(size) = self.kv_page_size {
            ensure!(
                size > 0,
                "engine.kv_page_size must be > 0; \
                 omit it to let the memory planner derive one"
            );
        }
        Ok(())
    }
}
