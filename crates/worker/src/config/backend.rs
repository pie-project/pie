use std::path::PathBuf;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use units::ByteSize;

use super::removed::{
    DeviceMemory, KvPageSize, MaxForwardRequests, MaxForwardTokens, MaxModelLen, MaxStateSlots,
    Removed,
};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaEngineOptions {
    #[serde(skip_serializing)]
    pub kv_page_size: Removed<KvPageSize>,
    #[serde(skip_serializing)]
    pub max_state_slots: Removed<MaxStateSlots>,
    #[serde(skip_serializing)]
    pub max_model_len: Removed<MaxModelLen>,
    #[serde(skip_serializing)]
    pub max_forward_tokens: Removed<MaxForwardTokens>,
    #[serde(skip_serializing)]
    pub max_forward_requests: Removed<MaxForwardRequests>,
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
    #[serde(skip_serializing)]
    pub kv_page_size: Removed<KvPageSize>,
    #[serde(skip_serializing)]
    pub max_state_slots: Removed<MaxStateSlots>,
    #[serde(skip_serializing)]
    pub max_model_len: Removed<MaxModelLen>,
    #[serde(skip_serializing)]
    pub max_forward_tokens: Removed<MaxForwardTokens>,
    #[serde(skip_serializing)]
    pub max_forward_requests: Removed<MaxForwardRequests>,
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
    #[serde(skip_serializing)]
    pub kv_page_size: Removed<KvPageSize>,
    #[serde(skip_serializing)]
    pub max_state_slots: Removed<MaxStateSlots>,
    #[serde(skip_serializing)]
    pub max_model_len: Removed<MaxModelLen>,
    #[serde(skip_serializing)]
    pub max_forward_tokens: Removed<MaxForwardTokens>,
    #[serde(skip_serializing)]
    pub max_forward_requests: Removed<MaxForwardRequests>,
    pub validation: bool,
    pub pipeline_cache: Option<PathBuf>,
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
    #[serde(skip_serializing)]
    pub kv_page_size: Removed<KvPageSize>,
    #[serde(skip_serializing)]
    pub max_state_slots: Removed<MaxStateSlots>,
    #[serde(skip_serializing)]
    pub max_model_len: Removed<MaxModelLen>,
    #[serde(skip_serializing)]
    pub max_forward_tokens: Removed<MaxForwardTokens>,
    #[serde(skip_serializing)]
    pub max_forward_requests: Removed<MaxForwardRequests>,
    pub pipeline_cache: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub device_memory: Removed<DeviceMemory>,
}

impl Default for WgpuEngineOptions {
    fn default() -> Self {
        Self {
            adapter_index: 0,
            backends: None,
            power_preference: "high-performance".to_string(),
            kv_page_size: Removed::default(),
            max_state_slots: Removed::default(),
            max_model_len: Removed::default(),
            max_forward_tokens: Removed::default(),
            max_forward_requests: Removed::default(),
            pipeline_cache: None,
            device_memory: Removed::default(),
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
    #[serde(skip_serializing)]
    pub kv_page_size: Removed<KvPageSize>,
    #[serde(skip_serializing)]
    pub max_state_slots: Removed<MaxStateSlots>,
    #[serde(skip_serializing)]
    pub max_model_len: Removed<MaxModelLen>,
    #[serde(skip_serializing)]
    pub max_forward_tokens: Removed<MaxForwardTokens>,
    #[serde(skip_serializing)]
    pub max_forward_requests: Removed<MaxForwardRequests>,
    /// Trace and compile the fire shapes a server meets first at load, so
    /// no early request waits on a compile. Slower to load.
    pub prewarm: bool,
    /// Device memory the load leaves past its weights and kv pool for XLA's
    /// scratch. Omit for the chip's own (1/16 of its HBM).
    pub device_headroom: Option<ByteSize>,
    /// The tokens one sequence is expected to hold, its prompt and output
    /// budget: a hybrid's state slots are balanced against the kv pages a
    /// seat of this many tokens takes rather than one at `max_model_len`.
    /// The pool still holds one sequence at `max_model_len`, and one that
    /// outgrows its seat meets kv pressure as any other. Omit for
    /// `max_model_len`.
    pub seat_tokens: Option<u32>,
}

/// The Cerebras engine: the model forward as a wafer program on a CS system reached by `cmaddr`,
/// or the fabric simulator. A load plans the model's trace, compiles the program and cuts the
/// checkpoint for it (both cached under the engine cache), so the model is all it needs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CerebrasEngineOptions {
    /// The wafer generation the program is compiled for: `"wse2"` or
    /// `"wse3"`.
    pub target: String,
    /// The CS system's `host:port`. Omit to run the fabric simulator.
    pub cmaddr: Option<String>,
    /// Host threads the simulator uses.
    pub num_threads: u32,
    #[serde(skip_serializing)]
    pub kv_page_size: Removed<KvPageSize>,
    #[serde(skip_serializing)]
    pub max_state_slots: Removed<MaxStateSlots>,
    #[serde(skip_serializing)]
    pub max_model_len: Removed<MaxModelLen>,
    #[serde(skip_serializing)]
    pub max_forward_tokens: Removed<MaxForwardTokens>,
    #[serde(skip_serializing)]
    pub max_forward_requests: Removed<MaxForwardRequests>,
    /// Steps a greedy lane's token loop runs past its row in one wafer launch (0: off); omit
    /// for the engine's default.
    pub lookahead: Option<u32>,
    /// Whether frames settle asynchronously (each step lands once its epilogue ran); omit for
    /// the engine's default (off).
    #[serde(rename = "async")]
    pub asynchronous: Option<bool>,
    /// Steps a posted token loop runs past its row; omit for the program's own.
    pub posted_loop: Option<u32>,
    /// `[engine.wafer]`: `preset` (the family's preset by name, else the first that builds),
    /// `build` (`"on_load"`, or `"never"`: refuse a program not compiled before), `tiles`
    /// (`"cache"`, or `"memory"`: cut for each load). Checked by the engine at boot.
    pub wafer: Option<toml::Table>,
    /// `[engine.toolchain]`: where the SDK image's toolchain is when off its default layout
    /// (`cslc`, `cslang_root`, `python`, `sdk_lib`, `compile_timeout`).
    pub toolchain: Option<toml::Table>,
    /// `[engine.debug]`: bench and experiment settings (`program`, `weights`, `set`, `levers`,
    /// `force_prompt`, `posted`, `loop_room`); a load with any logs them.
    pub debug: Option<toml::Table>,
}

impl Default for CerebrasEngineOptions {
    fn default() -> Self {
        Self {
            target: "wse3".to_string(),
            cmaddr: None,
            num_threads: 16,
            kv_page_size: Removed::default(),
            max_state_slots: Removed::default(),
            max_model_len: Removed::default(),
            max_forward_tokens: Removed::default(),
            max_forward_requests: Removed::default(),
            lookahead: None,
            asynchronous: None,
            posted_loop: None,
            wafer: None,
            toolchain: None,
            debug: None,
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
        Ok(())
    }
}
