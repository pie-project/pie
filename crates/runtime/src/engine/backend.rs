use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};
use engine::Engine;
use engine::transfer::MemoryDomain;
use eta_ir::registry::PortMask;

pub type EngineBox = Box<dyn Engine>;

#[derive(Debug, Clone, Copy)]
pub struct SchedulerLimits {
    pub max_forward_requests: usize,
    pub max_forward_tokens: usize,
    pub max_page_refs: usize,
    pub max_context: usize,
}

#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub num_kv_pages: usize,
    pub limits: SchedulerLimits,
    pub device_geometry_port_mask: PortMask,
    pub device_domain: MemoryDomain,
}

impl EngineSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

pub mod open {
    #[cfg(any(
        feature = "cuda",
        feature = "vulkan",
        feature = "wgpu",
        all(feature = "metal", target_vendor = "apple")
    ))]
    use super::{EngineBox, Result};

    #[cfg(feature = "cuda")]
    pub fn cuda(boot: engine_cuda::DeviceBoot) -> Result<EngineBox> {
        engine_cuda::open(boot, crate::engine::load::contract_for, |name| {
            models::sku(name).map(|sku| sku.classify)
        })
        .map(|engine| Box::new(engine) as EngineBox)
        .map_err(::anyhow::Error::msg)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_group(mut boots: Vec<engine_cuda::DeviceBoot>) -> Result<(EngineBox, usize)> {
        match boots.len() {
            0 => Err(super::anyhow!("a cuda group requires at least one rank")),
            1 => Ok((cuda(boots.remove(0))?, 1)),
            ranks => engine_cuda::open_group(boots, crate::engine::load::contract_for, |name| {
                models::sku(name).map(|sku| sku.classify)
            })
            .map(|group| (Box::new(group) as EngineBox, ranks))
            .map_err(::anyhow::Error::msg),
        }
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    pub fn metal(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_metal::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    #[cfg(feature = "vulkan")]
    pub fn vulkan(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_vulkan::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_wgpu::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }
}

#[cfg(feature = "cuda")]
pub use engine_cuda::comm::Transport;
#[cfg(feature = "cuda")]
pub use engine_cuda::{DeviceBoot, Diagnostics, Graphs, Knobs, Recording, World, ordinal_of};

mod remote;

pub use remote::{RemoteDisconnectHandle, RemoteEngine};

#[cfg(feature = "cuda")]
#[must_use]
pub fn envelopes_resolved() -> u64 {
    engine_cuda::Shell::envelopes_resolved()
}

struct EngineRegistration {
    spec: EngineSpec,
    backend: Option<EngineBox>,
}

fn registry() -> &'static RwLock<Vec<Option<EngineRegistration>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Option<EngineRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_engine_backend(mut spec: EngineSpec, backend: EngineBox) -> usize {
    let mut engines = registry().write().unwrap();
    let id = engines.len();
    spec.device_domain = backend
        .device_facts()
        .map_or(MemoryDomain::HostPinned, |facts| facts.domain);
    engines.push(Some(EngineRegistration {
        spec,
        backend: Some(backend),
    }));
    id
}

pub fn get_spec(engine_id: usize) -> Result<EngineSpec> {
    registry()
        .read()
        .unwrap()
        .get(engine_id)
        .and_then(|d| d.as_ref().map(|r| r.spec.clone()))
        .ok_or_else(|| anyhow!("unknown engine {engine_id}"))
}

pub fn take_engine_backend(engine_id: usize) -> Result<EngineBox> {
    let mut engines = registry().write().unwrap();
    let Some(Some(engine)) = engines.get_mut(engine_id) else {
        return Err(anyhow!("unknown engine {engine_id}"));
    };
    engine
        .backend
        .take()
        .ok_or_else(|| anyhow!("engine {engine_id} has no backend installed"))
}

pub fn unregister_engine(engine_id: usize) -> Result<()> {
    let mut engines = registry().write().unwrap();
    let Some(slot) = engines.get_mut(engine_id) else {
        return Err(anyhow!("unknown engine {engine_id}"));
    };
    slot.take();
    Ok(())
}
