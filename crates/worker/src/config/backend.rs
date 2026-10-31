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
    /// An expert-parallel group rebalances its experts across its ranks
    /// every this many steps (0: never), from the loads of the last
    /// `expert_load_window` steps, each rewriting at most
    /// `expert_rebalance_moves` slots a layer and only where that lowers its
    /// busiest rank by `expert_rebalance_gain` of the mean.
    pub expert_rebalance_steps: Option<u32>,
    pub expert_load_window: Option<u32>,
    pub expert_rebalance_moves: Option<u32>,
    pub expert_rebalance_gain: Option<f64>,
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
    /// The spatial engine (the model forward as a resident wafer program,
    /// instead of the phase programs) on this compiled program's output
    /// directory, run on `cmaddr` (else the simulator). Omit, with
    /// `spatial_serve`, for the cerebras shell.
    pub spatial_program: Option<String>,
    /// Steps a greedy lane's token loop runs past its row in one wafer
    /// launch (0: off); omit for the engine's default.
    pub spatial_lookahead: Option<u32>,
    /// Who runs epilogues the wafer's guest VM can: `"off"` (the host, the
    /// default), `"check"` (the VM contract played on the host) or `"wafer"`.
    pub spatial_guest: Option<String>,
    /// A Hugging Face directory the wafer program's weights are read from;
    /// omit for the model's checkpoint.
    pub spatial_ckpt: Option<String>,
    /// The spatial engine's program built by PIE from the model's plan: the
    /// lanes, page size, pipeline, eta VM image, ... (the engine's
    /// `[cerebras.spatial.serve]`; `engine-cerebras` `spatial::build`).
    /// Used when `spatial_program` is not set; the
    /// compile is cached, so later starts reuse it.
    pub spatial_serve: Option<toml::Table>,
    /// The guest placement's forecast: the model's token in cycles (omit:
    /// from its size), the share of it a table's `pre` may take and `main`
    /// may add before the epilogue stays on the host.
    pub spatial_token_cycles: Option<u64>,
    pub spatial_pre_margin: Option<f64>,
    pub spatial_main_fraction: Option<f64>,
    /// The engine's switches (`[cerebras.switches]`: placement, serving,
    /// toolchain; `engine-cerebras` `switches`), each winning over its old
    /// `PIE_CEREBRAS_*` variable.
    pub switches: Option<toml::Table>,
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
            spatial_program: None,
            spatial_lookahead: None,
            spatial_guest: None,
            spatial_ckpt: None,
            spatial_serve: None,
            spatial_token_cycles: None,
            spatial_pre_margin: None,
            spatial_main_fraction: None,
            switches: None,
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
