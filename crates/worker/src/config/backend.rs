use std::path::PathBuf;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use super::units::ByteSize;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaNativeEngineOptions {
    pub gpu_mem_utilization: f64,
    pub kv_page_size: Option<u32>,
    pub max_total_pages: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub max_model_len: Option<u32>,
    pub max_forward_tokens: Option<u32>,
    pub max_forward_requests: Option<u32>,
    #[serde(skip)]
    pub device: String,
    #[serde(skip)]
    pub verbose: bool,
    pub graphs: Option<String>,
    pub recording: Option<String>,
    pub pad: Option<bool>,
    pub bodies: Option<bool>,
    pub bodies_mem: Option<u32>,
    pub fallback_copy: Option<bool>,
    pub grouped: Option<bool>,
    pub diagnostics: Option<String>,
    pub nccl_transport: Option<String>,
    pub side_streams: Option<u32>,
}

impl Default for CudaNativeEngineOptions {
    fn default() -> Self {
        Self {
            gpu_mem_utilization: 0.90,
            kv_page_size: None,
            max_total_pages: None,
            max_state_slots: None,
            max_model_len: None,
            max_forward_tokens: None,
            max_forward_requests: None,
            device: String::new(),
            verbose: false,
            graphs: None,
            recording: None,
            pad: None,
            bodies: None,
            bodies_mem: None,
            fallback_copy: None,
            grouped: None,
            diagnostics: None,
            nccl_transport: None,
            side_streams: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetalEngineOptions {
    pub kv_page_size: u32,
    pub total_pages: u32,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
    pub max_model_len: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub gpu_mem_utilization: f64,
    pub diagnostics: Option<String>,
    pub tuning: toml::Table,
    #[serde(skip)]
    pub device: String,
    #[serde(skip)]
    pub verbose: bool,
}

impl Default for MetalEngineOptions {
    fn default() -> Self {
        Self {
            kv_page_size: 32,
            total_pages: 1024,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
            max_model_len: None,
            max_state_slots: None,
            gpu_mem_utilization: 0.90,
            diagnostics: None,
            tuning: toml::Table::new(),
            device: "metal:0".to_string(),
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VulkanEngineOptions {
    pub device_index: u32,
    pub gpu_mem_utilization: f64,
    pub max_total_pages: Option<u32>,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
    pub max_model_len: Option<u32>,
    pub max_state_slots: Option<u32>,
    pub validation: bool,
    pub pipeline_cache: Option<PathBuf>,
}

impl Default for VulkanEngineOptions {
    fn default() -> Self {
        Self {
            device_index: 0,
            gpu_mem_utilization: 0.90,
            max_total_pages: None,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
            max_model_len: None,
            max_state_slots: None,
            validation: false,
            pipeline_cache: None,
        }
    }
}

impl VulkanEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        ensure!(
            self.max_forward_tokens > 0,
            "engine.max_forward_tokens must be > 0"
        );
        ensure!(
            self.max_forward_requests > 0,
            "engine.max_forward_requests must be > 0"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WgpuEngineOptions {
    pub adapter_index: u32,
    pub backends: Option<String>,
    pub power_preference: String,
    pub gpu_mem_utilization: f64,
    pub max_total_pages: Option<u32>,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
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
            gpu_mem_utilization: 0.90,
            max_total_pages: None,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
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
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        ensure!(
            matches!(
                self.power_preference.as_str(),
                "high-performance" | "low-power" | "none"
            ),
            "engine.power_preference must be one of \"high-performance\", \
             \"low-power\", \"none\"; got {:?}",
            self.power_preference
        );
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        ensure!(
            self.max_forward_tokens > 0,
            "engine.max_forward_tokens must be > 0"
        );
        ensure!(
            self.max_forward_requests > 0,
            "engine.max_forward_requests must be > 0"
        );
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
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct XlaEngineOptions {
    /// The PJRT plugin to load. Omit to find one: `PIE_XLA_PLUGIN`, then
    /// `TPU_LIBRARY_PATH`, then `libtpu.so` on the loader's path.
    pub plugin: Option<PathBuf>,
    /// Which of the plugin's addressable devices this engine drives.
    pub device_index: u32,
    /// The fraction of device memory planned against: weights and the kv
    /// and state pools.
    pub mem_utilization: f64,
    /// The kv pool, in pages. Omit for the engine's own default.
    pub max_total_pages: Option<u32>,
    /// The most tokens one fire carries.
    pub max_forward_tokens: u32,
    /// The most requests one fire carries.
    pub max_forward_requests: u32,
    /// Recurrent-state seats (hybrid models).
    pub max_state_slots: Option<u32>,
    /// The longest sequence (prompt + output) one lane holds, in tokens.
    /// Omit for the engine default (4096).
    pub max_model_len: Option<u32>,
}

impl Default for XlaEngineOptions {
    fn default() -> Self {
        Self {
            plugin: None,
            device_index: 0,
            mem_utilization: 0.90,
            max_total_pages: None,
            max_forward_tokens: 2048,
            max_forward_requests: 64,
            max_state_slots: None,
            max_model_len: None,
        }
    }
}

impl XlaEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.mem_utilization.is_finite()
                && self.mem_utilization > 0.0
                && self.mem_utilization <= 1.0,
            "engine.mem_utilization must be finite and in (0.0, 1.0]"
        );
        if let Some(pages) = self.max_total_pages {
            ensure!(pages > 0, "engine.max_total_pages must be > 0");
        }
        ensure!(
            self.max_forward_tokens > 0,
            "engine.max_forward_tokens must be > 0"
        );
        ensure!(
            self.max_forward_requests > 0,
            "engine.max_forward_requests must be > 0"
        );
        Ok(())
    }
}

impl CudaNativeEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
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
