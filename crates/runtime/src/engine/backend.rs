use std::sync::{Mutex, OnceLock};

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
    /// The rows one prefill fire takes best (0: up to `max_forward_tokens`).
    pub prefill_rows: usize,
    /// The most lanes a step of prompts only (no lane decoding) carries
    /// (0: as many as `max_forward_requests`), what the engine holds a body
    /// for ([`::engine::FireLimits::prefill_lanes`]).
    pub prefill_lanes: usize,
    /// The most rows one fire reads out: the engine's ceiling under the
    /// rows its fixed readout store holds ([`Self::read_out_by`]). 0: up to
    /// `max_forward_tokens`.
    pub max_readouts: usize,
    /// The rows the engine's fixed readout store holds
    /// ([`::engine::ReadoutFacts::store_rows`]; 0: no store). A request whose
    /// sequence lanes alone read out more is refused at admission, by name
    /// (`scheduler::batch::AdmissionLimits`); a float lane's rows are not
    /// held to the store (`FireRequest::sequence_readouts`).
    pub readout_store: usize,
    /// The most buffered recurrent rows the lanes of one step replay,
    /// summed ([`::engine::FireLimits::replayed_rows`]; 0: no bound past
    /// each fire's rung).
    pub max_replayed_rows: usize,
    /// The step compositions the load stated (CONTRACT-BC 24): the step
    /// builder forms none outside them, a member whose joining would leave
    /// the set going in a later step and a lone member outside it refused at
    /// admission by name. Every served load states one (bootstrap computes
    /// it from the load's facts and holds the engine's to it); `None` only
    /// where a test builds limits by hand.
    pub shapes: Option<&'static ::engine::StepShapes>,
}

impl SchedulerLimits {
    /// The limits with the rows one fire reads out bounded as the engine
    /// states its readouts ([`::engine::ReadoutFacts::fire_rows`]): a fixed
    /// store's rows under the engine's ceiling, which fires are cut to
    /// ([`crate::scheduler::batch`], the frame's waves). An engine that
    /// states no store leaves them as they are.
    #[must_use]
    pub fn read_out_by(mut self, readouts: &::engine::ReadoutFacts) -> Self {
        let ceiling = u32::try_from(self.max_readouts).unwrap_or(u32::MAX);
        self.max_readouts = readouts.fire_rows(ceiling) as usize;
        self.readout_store = readouts.store_rows.map_or(0, |rows| rows.max(1) as usize);
        self
    }

    /// The facts of a load stating `chunks` under these limits, as the step
    /// set is computed from them ([`::engine::StepFacts::of`]): the limits
    /// after [`Self::read_out_by`], so the readout bound is the one fires
    /// are cut to, its lanes carrying `attachments` ([`::engine::Attachments::served`]).
    #[must_use]
    pub fn step_facts(
        &self,
        chunks: &::engine::ChunkFacts,
        attachments: ::engine::Attachments,
    ) -> ::engine::StepFacts {
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        ::engine::StepFacts::of(
            chunks,
            count(self.max_forward_requests),
            count(self.max_forward_tokens),
            count(self.max_readouts),
            attachments,
        )
    }
}

#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub limits: SchedulerLimits,
    pub device_geometry_port_mask: PortMask,
    pub device_domain: MemoryDomain,
    /// The host ops the engine serves, as its load stated them
    /// ([`::engine::Capabilities::host_ops`]).
    pub host_ops: ::engine::HostOps,
    /// The engine's kind ([`::engine::Engine::kind`]), as bind names it.
    pub kind: &'static str,
    /// The backend the host emits a guest program's kernels for
    /// ([`::engine::caps::DeviceFacts::codegen_backend`]), where the engine
    /// takes host-emitted kernels: a registration's codegen runs off the
    /// engine's thread with it.
    pub codegen_backend: Option<String>,
}

impl EngineSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

pub mod open {
    #[cfg(any(
        feature = "engine-cuda",
        feature = "vulkan",
        feature = "wgpu",
        feature = "xla",
        feature = "cerebras",
        all(feature = "metal", target_vendor = "apple")
    ))]
    use super::{EngineBox, Result};

    #[cfg(feature = "engine-cuda")]
    pub fn cuda(boot: engine_cuda::DeviceBoot) -> Result<EngineBox> {
        engine_cuda::open(boot, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    #[cfg(feature = "engine-cuda")]
    pub fn cuda_group(mut boots: Vec<engine_cuda::DeviceBoot>) -> Result<(EngineBox, usize)> {
        match boots.len() {
            0 => Err(super::anyhow!("a cuda group requires at least one rank")),
            1 => Ok((cuda(boots.remove(0))?, 1)),
            ranks => engine_cuda::open_group(boots, crate::engine::load::contract_for)
                .map(|group| (Box::new(group) as EngineBox, ranks))
                .map_err(::anyhow::Error::msg),
        }
    }

    /// The ranks of an expert-parallel group, one engine a device, each
    /// loaded at once with the others.
    #[cfg(feature = "engine-cuda")]
    pub fn cuda_lockstep(boots: Vec<engine_cuda::DeviceBoot>) -> Result<Vec<EngineBox>> {
        engine_cuda::lockstep::open_lockstep(boots, crate::engine::load::contract_for)
            .map(|ranks| {
                ranks
                    .into_iter()
                    .map(|rank| Box::new(rank) as EngineBox)
                    .collect()
            })
            .map_err(::anyhow::Error::msg)
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

    #[cfg(feature = "xla")]
    pub fn xla(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_xla::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    #[cfg(feature = "cerebras")]
    pub fn cerebras(config_bytes: &[u8]) -> Result<EngineBox> {
        // The `[cerebras]` table the worker writes opens the wafer-program engine.
        engine_cerebras::open_engine(config_bytes, crate::engine::load::contract_for)
            .map_err(::anyhow::Error::msg)
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu_on_device(
        config_bytes: &[u8],
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<EngineBox> {
        engine_wgpu::open_with_device(
            config_bytes,
            crate::engine::load::contract_for,
            adapter,
            device,
            queue,
        )
        .map(|engine| Box::new(engine) as EngineBox)
        .map_err(::anyhow::Error::msg)
    }
}

#[cfg(feature = "engine-cuda")]
pub use engine_cuda::comm::Transport;
#[cfg(feature = "engine-cuda")]
pub use engine_cuda::{DeviceBoot, Diagnostics, Knobs, QueueMasks, World, ordinal_of};

struct EngineRegistration {
    spec: EngineSpec,
    backend: Option<EngineBox>,
    /// What the engine counts while it serves, held from load on.
    live: ::engine::live::Live,
    /// The rungs it stated eager at load
    /// ([`::engine::chunk::ChunkFacts::eager_rungs`]): read by the model
    /// status, and by nothing that cuts a pass.
    eager_rungs: Vec<u32>,
}

fn registry() -> &'static Mutex<Vec<Option<EngineRegistration>>> {
    static REGISTRY: OnceLock<Mutex<Vec<Option<EngineRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn register_engine_backend(
    mut spec: EngineSpec,
    backend: EngineBox,
    eager_rungs: Vec<u32>,
) -> usize {
    let mut engines = registry().lock().unwrap();
    let id = engines.len();
    spec.device_domain = backend
        .device_facts()
        .map_or(MemoryDomain::HostPinned, |facts| facts.domain);
    spec.kind = backend.kind();
    spec.codegen_backend = crate::engine::verbs::codegen_backend(&backend).map(str::to_owned);
    let live = backend.live();
    engines.push(Some(EngineRegistration {
        spec,
        backend: Some(backend),
        live,
        eager_rungs,
    }));
    id
}

/// The rungs each registered engine stated eager at load, in engine order:
/// a fire there runs uncaptured, as stated.
#[must_use]
pub fn eager_rungs() -> Vec<Vec<u32>> {
    registry()
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .map(|r| r.eager_rungs.clone())
        .collect()
}

/// What each registered engine counts while it serves, in engine order:
/// every engine serves the one model the worker loads.
#[must_use]
pub fn lives() -> Vec<::engine::live::Live> {
    registry()
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .map(|r| r.live.clone())
        .collect()
}

/// What one fire carries at most on every registered engine: the fewest
/// tokens, then the fewest lanes, any of them states. None before an engine
/// is registered.
#[must_use]
pub fn fire_limits() -> Option<(usize, usize)> {
    let engines = registry().lock().unwrap();
    smallest_fire(engines.iter().flatten().map(|r| r.spec.limits))
}

fn smallest_fire(limits: impl Iterator<Item = SchedulerLimits>) -> Option<(usize, usize)> {
    limits
        .map(|limits| (limits.max_forward_tokens, limits.max_forward_requests))
        .reduce(|(tokens, lanes), (t, l)| (tokens.min(t), lanes.min(l)))
}

pub fn get_spec(engine_id: usize) -> Result<EngineSpec> {
    registry()
        .lock()
        .unwrap()
        .get(engine_id)
        .and_then(|d| d.as_ref().map(|r| r.spec.clone()))
        .ok_or_else(|| anyhow!("unknown engine {engine_id}"))
}

/// Refuses a program of `package` on engine `engine_id` by name where it
/// needs a host op the engine's load did not state it serves
/// ([`::engine::HostOps::needed_by`]): asked at the bind, before anything is
/// registered or any step submitted, so a beam search on an engine without
/// `beam_list` fails at its start, not at its finish.
pub fn refuse_unserved_host_ops(
    engine_id: usize,
    package: &eta_compiler::codegen::launch::LaunchPackage,
) -> Result<()> {
    let needed = ::engine::HostOps::needed_by(package);
    if needed.is_empty() {
        return Ok(());
    }
    let spec = get_spec(engine_id)?;
    match spec.host_ops.unserved(&needed) {
        Some(op) => Err(anyhow!(
            "the {} engine does not serve `{}`, which this program needs (a beam group reads its \
             chosen beam's page list at its finish): refused at its bind, before any step",
            spec.kind,
            op.name()
        )),
        None => Ok(()),
    }
}

pub fn take_engine_backend(engine_id: usize) -> Result<EngineBox> {
    let mut engines = registry().lock().unwrap();
    let Some(Some(engine)) = engines.get_mut(engine_id) else {
        return Err(anyhow!("unknown engine {engine_id}"));
    };
    engine
        .backend
        .take()
        .ok_or_else(|| anyhow!("engine {engine_id} has no backend installed"))
}

/// Forgets the boot's engines, so the next boot's first is 0 again.
pub(crate) fn reset() {
    registry().lock().unwrap().clear();
}

pub fn unregister_engine(engine_id: usize) -> Result<()> {
    let mut engines = registry().lock().unwrap();
    let Some(slot) = engines.get_mut(engine_id) else {
        return Err(anyhow!("unknown engine {engine_id}"));
    };
    slot.take();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fire_is_what_every_engine_carries() {
        let limits = |tokens, lanes| SchedulerLimits {
            max_forward_requests: lanes,
            max_forward_tokens: tokens,
            max_page_refs: 0,
            max_context: 0,
            prefill_rows: 0,
            prefill_lanes: 0,
            max_readouts: 0,
            readout_store: 0,
            max_replayed_rows: 0,
            shapes: None,
        };
        assert_eq!(smallest_fire(std::iter::empty()), None, "no engine yet");
        assert_eq!(
            smallest_fire([limits(5120, 256)].into_iter()),
            Some((5120, 256))
        );
        assert_eq!(
            smallest_fire([limits(5120, 64), limits(2048, 256)].into_iter()),
            Some((2048, 64)),
            "each the smallest any engine states"
        );
    }
}
