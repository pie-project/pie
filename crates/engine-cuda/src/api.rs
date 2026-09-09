use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout, Step,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use engine::transfer::{KvCopy, MemoryDomain, StateCopy};
use eta_ir::registry::{GeometryClass, ModelProfile, Port, PortMask};
use eta_ir::types::Dtype;
use model_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder, VoxelLadder};
use model_ir::Trace;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Graphs, Knobs, Lane, Seated, Shell};

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

pub type ClassifyFor = fn(&str) -> Option<model_ir::ClassifyFn>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct World {
    pub rank: u32,
    pub size: u32,
}

impl Default for World {
    fn default() -> World {
        World { rank: 0, size: 1 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    pub ordinal: i32,
    pub world: World,
    pub comm: Option<Arc<crate::comm::Comm>>,
    pub graphs: Graphs,
    pub knobs: Knobs,
    pub cache_dir: Option<std::path::PathBuf>,
    pub adapter_dir: Option<std::path::PathBuf>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            ordinal: 0,
            world: World::default(),
            comm: None,
            graphs: Graphs::default(),
            knobs: Knobs::default(),
            cache_dir: None,
            adapter_dir: None,
        }
    }
}

struct PendingStep {
    readout: Vec<Readout>,
    settled: crate::serve::Settled,
}

pub struct Cuda {
    boot: DeviceBoot,
    contract_for: ContractFor,
    classify_for: ClassifyFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    channels: BTreeMap<ChannelId, Arc<crate::program::Endpoint>>,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<PendingStep>)>,
    adapters: BTreeMap<InstanceId, crate::Binding>,
}

impl Cuda {
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor, classify_for: ClassifyFor) -> Cuda {
        crate::serve::diag::publish(&boot.knobs.diagnostics);
        Cuda {
            boot,
            contract_for,
            classify_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            channels: BTreeMap::new(),
            sink: None,
            pending: None,
            adapters: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn shell(&self) -> Option<&Shell> {
        self.shell.as_ref()
    }

    pub fn shell_mut(&mut self) -> Option<&mut Shell> {
        self.shell.as_mut()
    }

    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    pub fn open(&mut self, slot: u32) -> EngineResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    #[must_use]
    pub fn rank(&self) -> u32 {
        self.boot.world.rank
    }

    #[must_use]
    pub fn endpoint(&self, id: ChannelId) -> Option<Arc<crate::program::Endpoint>> {
        self.channels.get(&id).cloned()
    }

    pub fn adopt_channel(
        &mut self,
        id: ChannelId,
        endpoint: Arc<crate::program::Endpoint>,
    ) -> EngineResult<RegisteredChannel> {
        if self.channels.contains_key(&id) {
            return Err(Error::Program(format!(
                "channel {id} is already registered on this engine"
            )));
        }
        self.channels.insert(id, endpoint);
        Ok(RegisteredChannel {
            id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    pub fn channel_predictions(&mut self) -> EngineResult<Vec<(u64, Vec<crate::program::Cursor>)>> {
        Ok(self.loaded_mut()?.channel_predictions())
    }

    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id,
            })
    }

    fn settle(
        &self,
        trace: &Trace,
        path: &Path,
        residency: &engine::Residency,
    ) -> EngineResult<(ModelContract, crate::experts::Plan)> {
        let contract = (self.contract_for)(trace, path).map_err(Error::Load)?;
        let prospect =
            crate::weights::prospect(trace, &contract, path, self.target()).map_err(fault)?;
        let plan = crate::experts::Plan::cut(
            &prospect.ranking,
            crate::experts::Budgets {
                device: residency.device_weight_budget,
                host: residency.host_weight_budget,
            },
        )
        .map_err(fault)?;
        let sourced = plan.spill_demand() > 0
            && crate::checkpoint_serving::Serving::open(path, trace).is_some();
        residency.admit_tiers(engine::load::Tiers {
            device: plan.device_demand(),
            host: plan.host_demand(),
            spilled: plan.spill_demand(),
            sourced,
        })?;
        Ok((contract, plan))
    }

    fn target(&self) -> checkpoint::plan::StorageTarget {
        checkpoint::plan::StorageTarget::for_backend(
            checkpoint::types::BackendKind::Cuda,
            self.boot.world.rank,
            self.boot.world.size,
        )
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the cuda engine has no model loaded".into()))
    }
}

fn fault(fault: Fault) -> Error {
    match fault {
        Fault::Runtimeless | Fault::Device { .. } => Error::Device(fault.to_string()),
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Param { .. }
        | Fault::Unbound { .. }
        | Fault::Unlowered { .. }
        | Fault::Golden { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::OutOfMemory { need, have } => Error::Exhausted {
            resource: "device memory",
            wanted: need,
            available: have,
        },
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Fragmented { .. } | Fault::Integrity { .. } => Error::Device(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Draftless { .. }
        | Fault::DraftWord { .. }
        | Fault::Scoreless { .. }
        | Fault::ScoreWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) | Fault::PatchPayload { .. } | Fault::VoxelPayload { .. } => {
            Error::Invalid(fault.to_string())
        }
    }
}

fn extents(stated: &BindExtents) -> eta_exec::Extents {
    eta_exec::Extents {
        kv_len: stated.kv_len,
        page_count: stated.page_count,
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

#[must_use]
pub(crate) fn default_lattice(max_tokens: u32) -> Vec<u32> {
    let mut lattice: Vec<u32> =
        core::iter::successors(Some(LATTICE_FLOOR), |point| point.checked_mul(2))
            .take_while(|point| *point < max_tokens)
            .collect();
    lattice.push(max_tokens);
    lattice
}

pub(crate) const LATTICE_FLOOR: u32 = 1;

#[must_use]
pub(crate) fn lattice(stated: Vec<u32>, max_tokens: u32) -> Vec<u32> {
    if stated.is_empty() {
        default_lattice(max_tokens)
    } else {
        stated
    }
}

fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let declares_patches = trace.values.iter().any(|decl| {
        matches!(&decl.ty, model_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(model_ir::RowAxis::Patches))
    });
    if !declares_patches {
        return None;
    }

    let max_patches = budgets
        .max_patches
        .unwrap_or_else(|| budgets.max_tokens.min(DERIVED_PATCH_CEILING))
        .max(PATCH_LATTICE_FLOOR);
    let mut buckets = Vec::new();
    let mut rung = PATCH_LATTICE_FLOOR;
    while rung < max_patches {
        buckets.push(rung);
        rung = rung.saturating_mul(2);
    }
    buckets.push(max_patches);
    Some(PatchLadder {
        max_images: budgets
            .max_images
            .unwrap_or_else(|| max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

#[must_use]
pub fn voxel_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<VoxelLadder> {
    const DERIVED_VOXEL_CEILING: u32 = 65_536;

    let declares_voxels = trace.values.iter().any(|decl| {
        matches!(&decl.ty, model_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(model_ir::RowAxis::Voxels))
    });
    if !declares_voxels {
        return None;
    }
    let max_voxels = budgets.max_voxels.unwrap_or(DERIVED_VOXEL_CEILING).max(1);
    Some(VoxelLadder {
        max_voxels,
        buckets: Vec::new(),
        max_clips: budgets
            .max_clips
            .unwrap_or(budgets.max_lanes)
            .clamp(1, max_voxels),
    })
}

fn profile(shell: &Shell, budgets: &LoadBudgets) -> EngineResult<ModelProfile> {
    let trace = shell.trace();
    let layers = trace
        .nodes
        .iter()
        .filter_map(|node| node.layer)
        .max()
        .map_or(0, |top| top + 1);
    let vocab = match shell.out_width() {
        Ok(width) => u32::try_from(width).unwrap_or(u32::MAX),
        Err(_) if shell.readout_seam().is_some() => 0,
        Err(why) => return Err(fault(why)),
    };
    let (has_pixels, pixels_width) = shell.pixels_facts();
    Ok(ModelProfile {
        vocab,
        page_size: budgets.page_size,
        num_layers: layers,
        activation: Dtype::F32,
        has_mtp_logits: shell.drafts(),
        mtp_depth: shell.mtp_depth(),
        draft_block: trace.drafter.map_or(0, |d| d.rows),
        draft_mask_token: trace.drafter.map_or(0, |d| d.mask_token),
        draft_bidirectional: trace.drafter.is_some_and(|d| d.bidirectional),
        draft_proposals_from: trace.drafter.map_or(1, |d| d.proposals_from),
        has_value_head: false,
        has_attn_score: shell.observes_scores(),
        has_attn_page_mask: false,
        has_lora: true,
        has_velocity: shell.velocity_width().is_some(),
        velocity_width: shell.velocity_width().unwrap_or(0),
        has_pixels,
        pixels_width,
        kernels: Vec::new(),
    })
}

fn adapter_of(
    shell: &mut Shell,
    program: engine::program::ProgramId,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::Binding>> {
    let Some(sink) = shell.program_adapter_sink(program).map_err(fault)? else {
        return Ok(None);
    };
    let seats = shell.bank_seats();
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        let wire = seeds
            .iter()
            .find(|(seeded, _)| seeded == channel)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| {
                Error::Load(format!(
                    "this program's `lora` sink reads its `{}` plane out of channel \
                     {channel} and this bind seeded nothing into it; an adapter's \
                     weights are the seed, because the fire path never reads the cell \
                     (alto adapter §6.1), so an unseeded plane is a correction of zero \
                     that nobody asked for",
                    role.bank()
                ))
            })?;
        built.extend(crate::adapter::planes_of(*role, site, wire, &seats).map_err(fault)?);
    }
    let planes: Vec<crate::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| crate::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .bind_adapter(crate::AdapterSource::Own {
            instance,
            planes: &planes,
        })
        .map(Some)
        .map_err(fault)
}

impl Engine for Cuda {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this cuda engine already has a model loaded; one shell per engine".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            frames_in_flight,
        } = request;
        let trace = model_ir::fuse::residual_norm(trace);
        let trace = if self.boot.knobs.diagnostics.fuse_chains {
            model_ir::fuse::residual_chains(trace)
        } else {
            trace
        };

        if !self.boot.graphs.records() {
            eprintln!(
                "engine-cuda: serving without CUDA graph capture ([engine] graphs = \
{:?}, not \"on\"): every fire launches eagerly, which costs per-step host time; \
intended for diagnostics, not serving",
                self.boot.graphs
            );
        }

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the cuda shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        refuse_an_artifact_for_another_deployment(
            &path,
            trace.platform.backend(),
            width_free(&trace.name),
        )?;
        let (contract, plan) = self.settle(&trace, &path, &residency)?;

        let patches = patch_ladder(&trace, &budgets);
        let voxels = voxel_ladder(&trace, &budgets);
        let classify = (self.classify_for)(&trace.name).ok_or_else(|| {
            Error::Load(format!(
                "this build ships no classifier for {:?}",
                trace.name
            ))
        })?;
        let mut shell = Shell::load(Boot {
            classify,
            trace,
            contract: &contract,
            checkpoint: &path,
            budget: bake_budgets(&budgets),
            patches,
            voxels,
            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            pages: budgets.pages,
            ordinal: if ordinal >= 0 {
                ordinal
            } else {
                self.boot.ordinal
            },
            graphs: self.boot.graphs,
            knobs: self.boot.knobs.clone(),
            cache_dir: self.boot.cache_dir.as_deref(),
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            residency: plan,
            deferred_tier: residency.deferred_tier,
            world: self.boot.world,
            comm: self
                .boot
                .comm
                .as_ref()
                .map_or(core::ptr::null_mut(), |comm| comm.raw()),
        })
        .map_err(fault)?;

        shell.mount_adapters(self.boot.adapter_dir.clone());

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let (pool_committed_bytes, pool_high_water_bytes, elastic_page_bytes, elastic_budget_pages) =
            shell.elastic();
        let weights_from_cache = shell.weights_from_cache();
        let weights_resident = shell.weights_resident();
        let paging = shell.paging();
        let state_rows = shell
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, model_ir::CacheRow::State { .. }));
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cuda".to_string(),
                domain: MemoryDomain::CudaDevice(u32::try_from(shell.ordinal()).unwrap_or(0)),
                sms: shell.sms(),
                unified_memory: false,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: Some("cuda".to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: shell
                    .banks()
                    .iter()
                    .map(|&(_, adapters, _)| adapters)
                    .min()
                    .unwrap_or(0),
                elastic_page_bytes,
                elastic_budget_pages,
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            ports: PortMask::DEVICE_GEOMETRY
                .with(Port::AttnMask)
                .with(Port::RsFoldLen),
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: true,
            bidirectional_attention: true,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                weights_resident,
                weights_from_cache,
                arena_bytes,
                pool_bytes,
                input_bytes,
                pool_committed_bytes,
                pool_high_water_bytes,
            },
            caps,
        })
    }

    fn register_adapter(
        &mut self,
        registration: &engine::adapter::AdapterRegistration,
    ) -> EngineResult<()> {
        let planes: Vec<crate::AdapterPlane<'_>> = registration
            .planes
            .iter()
            .map(|plane| crate::AdapterPlane {
                bank: plane.bank.as_str(),
                bytes: &plane.bytes,
            })
            .collect();
        self.loaded_mut()?
            .register_adapter(registration.id, &planes)
            .map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(engine::Serves {
            device_channel_commit: self
                .caps
                .as_ref()
                .is_some_and(|caps| caps.device_channel_commit),
            rs_verbs: self.caps.as_ref().is_some_and(|caps| caps.rs_verbs),
            bidirectional: self
                .caps
                .as_ref()
                .is_some_and(|caps| caps.bidirectional_attention),
        })?;
        let id = self.next_frame;
        self.next_frame = self.next_frame.wrapping_add(1);
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut settled = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            let (ticket, step_settled) = match self.fire_step(step, at) {
                Ok(both) => both,
                Err(error) => {
                    self.pending = None;
                    let airborne = self.shell.as_ref().map_or(0, Shell::airborne_steps);
                    return Err(attributed(error, airborne));
                }
            };
            steps.push(ticket);
            settled.push(step_settled);
        }
        self.pending = Some((id, settled));
        Ok(FrameTicket { id, steps })
    }

    fn settles_asynchronously(&self) -> bool {
        true
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has already carved \
                 over its arena rectangles",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: frame {id} has been submitted since, and \
                 the out seam is one arena rectangle that every fire carves over. Ask \
                 for a frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut settled) = self.pending.take().expect("checked just above");
        let shell = self.loaded_mut()?;
        for step in &mut settled {
            shell
                .read_out_rows(&mut step.settled, &step.readout)
                .map_err(fault)?;
        }
        for (receipt, step) in ticket.steps.iter_mut().zip(&settled) {
            receipt.readouts = readouts_of(step);
        }
        self.pending = Some((ticket.id, settled));
        Ok(())
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.loaded_mut()?
            .register_program(registration)
            .map_err(fault)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        if self.channels.contains_key(&registration.id) {
            return Err(Error::Program(format!(
                "channel {} is already registered on this engine",
                registration.id
            )));
        }
        let numel = registration
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1);
        let device_only = registration.host_role == eta_ir::container::HostRole::None;
        let cell_bytes = u32::try_from(if device_only {
            crate::program::launch::native_cell_bytes(registration.dtype.program_dtype(), numel)
        } else {
            eta_exec::wire_cell_bytes(registration.dtype.program_dtype(), numel)
        })
        .map_err(|_| {
            Error::Program(format!(
                "channel {}'s cell is wider than a u32 counts",
                registration.id
            ))
        })?;
        let capacity = registration.capacity.max(1);
        let endpoint = Arc::new(
            crate::program::Endpoint::open(registration.host_role, cell_bytes, capacity)
                .map_err(fault)?,
        );
        let mirror = (!device_only).then(|| HostMirror {
            mirror: endpoint.mirror_host(),
            words: endpoint.words_host(),
            cell_bytes,
            capacity,
        });
        self.channels.insert(registration.id, endpoint);
        Ok(RegisteredChannel {
            id: registration.id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
                "this load resolves {:?} on the device, so it binds at most {:?} and \
                 not {:?}",
                caps.ports, caps.geometry, binding.geometry
            )));
        }
        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        let adopted: Vec<Option<Arc<crate::program::Endpoint>>> = binding
            .channels
            .iter()
            .map(|id| self.channels.get(id).cloned())
            .collect();
        let shell = self.loaded_mut()?;
        let id = shell
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
                &adopted,
                &binding.channels,
            )
            .map_err(fault)?;
        let landed = adapter_of(shell, binding.program, id, &seeds);
        let bound = match landed {
            Ok(bound) => bound,
            Err(why) => {
                let _ = shell.close_program_instance(id);
                return Err(why);
            }
        };
        if let Some(bound) = bound {
            self.adapters.insert(id, bound);
        }
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        let held = self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        if let Some(held) = held {
            shell.release_adapter(held);
        }
        shell.close_program_instance(id).map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.channels.remove(&id);
        Ok(())
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.instance(instance)?
            .publish(channel, cell)
            .map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.instance(instance)?.take(channel).map_err(fault)
    }

    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        for (at, move_) in copy.moves.iter().enumerate() {
            if move_.src_token_offset != 0 || move_.dst_token_offset != 0 {
                return Err(Error::Invalid(format!(
                    "state move {at} names a token offset, and a recurrent bank is a folded \
                     summary of a prefix rather than an array of per-token entries — this \
                     engine moves whole slots"
                )));
            }
        }
        let shell = self.loaded_mut()?;
        for move_ in &copy.moves {
            shell
                .copy_state(move_.src_slot_id, move_.dst_slot_id)
                .map_err(fault)?;
        }
        Ok(())
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let ordinal = self
            .caps
            .as_ref()
            .map(|caps| caps.device.domain)
            .and_then(MemoryDomain::ordinal);
        let served = matches!(
            (copy.src, copy.dst),
            (MemoryDomain::CudaDevice(src), MemoryDomain::CudaDevice(dst))
                if Some(src) == ordinal && Some(dst) == ordinal
        );
        if !served {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "cuda",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let mut moves: Vec<crate::store::Move> =
            Vec::with_capacity(copy.src_page_ids.len() + copy.moves.len());
        for (src, dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            moves.push(crate::store::Move {
                src_page: *src,
                src_token: 0,
                dst_page: *dst,
                dst_token: 0,
                tokens: page_size,
            });
        }
        for (at, cell) in copy.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(Error::Invalid(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                )));
            }
            if cell.src_page_id == cell.dst_page_id
                && cell.src_token_offset == cell.dst_token_offset
            {
                continue;
            }
            let run = moves.last_mut().filter(|run| {
                run.src_page == cell.src_page_id
                    && run.dst_page == cell.dst_page_id
                    && run.src_token + run.tokens == cell.src_token_offset
                    && run.dst_token + run.tokens == cell.dst_token_offset
                    && run.src_token + run.tokens < page_size
            });
            match run {
                Some(run) => run.tokens += 1,
                None => moves.push(crate::store::Move {
                    src_page: cell.src_page_id,
                    src_token: cell.src_token_offset,
                    dst_page: cell.dst_page_id,
                    dst_token: cell.dst_token_offset,
                    tokens: 1,
                }),
            }
        }
        for run in &moves {
            if run.src_page != run.dst_page {
                continue;
            }
            let (lo, hi) = (
                u32::min(run.src_token, run.dst_token),
                u32::max(run.src_token, run.dst_token),
            );
            if hi - lo < run.tokens {
                return Err(Error::Invalid(format!(
                    "a kv move of {} tokens reads page {} from token {} and writes the same \
                     page at token {} — the two ends overlap, and a device copy whose ends \
                     overlap is undefined rather than a shift",
                    run.tokens, run.src_page, run.src_token, run.dst_token
                )));
            }
        }
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }

}

impl Cuda {
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        let done = self.sink.as_ref().map(|sink| crate::serve::Done {
            at,
            sink: std::sync::Arc::clone(sink),
        });
        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        if !self.adapters.is_empty() {
            for attachment in &submission.attachments {
                let Some(bound) = self.adapters.get(&attachment.instance) else {
                    continue;
                };
                if let Some(slot) = lane_adapters.get_mut(attachment.lane as usize) {
                    *slot = Some(bound.slot);
                }
            }
        }
        let shell = self.loaded_mut()?;
        let mut words: Vec<u64> = submission.lanes.iter().map(|lane| lane.word).collect();
        for (at, slot) in lane_adapters.iter().enumerate() {
            if slot.is_none() {
                continue;
            }
            words[at] = shell.adapted_word(words[at]).ok_or_else(|| {
                Error::Invalid(format!(
                    "lane {at} is attached to an instance that bound an adapter, and \
                     this load's model text has no corrected class for its fact word \
                     {:#x}: the text declares no `linear.lora_correct` arm, or the arm's \
                     window is not decided by one fact. A lane that asked for a \
                     correction and got the base model is the one wrong answer this \
                     axis refuses to give",
                    words[at]
                ))
            })?;
        }
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                if !lane.positions.is_empty() {
                    return Err(Error::Unsupported {
                        verb: "explicit lane positions",
                        engine: "cuda",
                    });
                }
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    kv_less: lane.kv_less,
                    translation: &lane.kv.translation,
                    mask: lane.mask.as_ref(),
                    adapter: lane_adapters[at].or(lane.adapter),
                    drafts: lane.drafts,
                    captures_scores: lane.captures_scores,
                    bidirectional: lane.bidirectional,
                    self_cond: lane.self_cond.as_ref(),
                    rs: lane.rs.clone(),
                    rs_reset: lane.rs_reset,
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    stream: lane.stream as u8,
                    group: lane.group,
                    peer: lane.peer,
                    ports: &lane.ports,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        for attachment in &submission.attachments {
            let Some(lane) = submission.lanes.get(attachment.lane as usize) else {
                continue;
            };
            if lane.channels.is_empty() {
                continue;
            }
            if let Some(why) =
                shell.program_ticket_disagreement(attachment.instance, &lane.channels)
            {
                return Err(Error::Program(format!(
                    "this fire's channel predictions and the engine's disagree: {why}"
                )));
            }
        }

        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                instance: attachment.instance,
                at: attachment.at,
            })
            .collect();

        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            let Some(element) = shell.patch_element() else {
                return Err(fault(crate::error::Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::Towerless {
                        lane: submission.media[0].lane,
                    },
                ))));
            };
            staged.reserve(submission.media.len());
            for row in &submission.media {
                staged.push(patch_bytes(&row.patches, element).map_err(|why| {
                    Error::Unsupported {
                        verb: why,
                        engine: "cuda",
                    }
                })?);
            }
        }
        let mut voxel_bytes: Vec<Vec<u8>> = Vec::new();
        if !submission.voxels.is_empty() {
            let Some(element) = shell.voxel_element() else {
                return Err(fault(crate::error::Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::Vaeless {
                        lane: submission.voxels[0].lane,
                    },
                ))));
            };
            voxel_bytes.reserve(submission.voxels.len());
            for row in &submission.voxels {
                voxel_bytes.push(crate::voxels::port_bytes(&row.payload, element).map_err(
                    |why| Error::Unsupported {
                        verb: why,
                        engine: "cuda",
                    },
                )?);
            }
        }
        let voxels: Vec<crate::serve::Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&voxel_bytes)
            .map(|(row, payload)| crate::serve::Clips {
                lane: row.lane,
                clips: &row.clips,
                payload,
            })
            .collect();
        let media: Vec<crate::serve::Media<'_>> = submission
            .media
            .iter()
            .zip(&staged)
            .map(|(row, patches)| crate::serve::Media {
                lane: row.lane,
                rows: &row.rows,
                patches,
                routes: &row.routes,
                positions: &row.positions,
                embed_rows: &row.embed_rows,
                embed_weights: &row.embed_weights,
                token_positions: &row.token_positions,
            })
            .collect();

        let settled = {
            use engine::frame::Shell as FrameShell;
            let prepared = FrameShell::prepare(
                shell,
                crate::serve::StepView {
                    lanes: &seated,
                    attachments: &attached,
                    media: &media,
                    voxels: &voxels,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            shell.settle_step(enqueued, done).map_err(fault)?
        };

        Ok((
            FireTicket {
                id,
                readouts: Vec::new(),
            },
            PendingStep {
                readout: submission
                    .lanes
                    .iter()
                    .map(|lane| lane.readout.clone())
                    .collect(),
                settled,
            },
        ))
    }
}

fn patch_bytes(
    patches: &[f32],
    element: model_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        model_ir::Dtype::Bf16 => Ok(patches
            .iter()
            .flat_map(|&v| crate::adapter::bf16_bits(v).to_le_bytes())
            .collect()),
        model_ir::Dtype::F32 => Ok(patches.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err(
            "a media submission against a plan whose activation element is neither \
                  `bf16` nor `f32`, which is the pair every tower in this catalog computes in",
        ),
    }
}

fn kv_copy_direction(src: MemoryDomain, dst: MemoryDomain) -> &'static str {
    match (src, dst) {
        (MemoryDomain::HostPinned, MemoryDomain::HostPinned) => {
            "`copy_kv` host-pinned to host-pinned, which is the caller's own memmove"
        }
        (MemoryDomain::HostPinned, _) => {
            "`copy_kv` out of host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (_, MemoryDomain::HostPinned) => {
            "`copy_kv` into host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (MemoryDomain::CudaDevice(_), MemoryDomain::CudaDevice(_)) => {
            "`copy_kv` between two CUDA ordinals, which needs a peer mapping this load has not \
             opened"
        }
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

fn attributed(error: Error, airborne: u64) -> Error {
    if airborne == 0 {
        return error;
    }
    let note = format!(
        " (detected while {airborne} earlier step(s) were still airborne; device \
         faults are asynchronous, so this may name the step that DETECTED the \
         fault rather than the step that caused it)"
    );
    match error {
        Error::Device(why) => Error::Device(format!("{why}{note}")),
        Error::Program(why) => Error::Program(format!("{why}{note}")),
        other => other,
    }
}

fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let mut out = Vec::with_capacity(step.readout.len());
    for (lane, want) in step.readout.iter().enumerate() {
        let scores = step.settled.scores.get(lane).cloned().unwrap_or_default();
        let values = step.settled.logits.get(lane).cloned().unwrap_or_default();
        let rows = step.settled.rows.get(lane).copied().unwrap_or(0);
        let width = if rows == 0 {
            0
        } else {
            u32::try_from(values.len() / rows as usize).unwrap_or(u32::MAX)
        };
        if let Some((pixels, clips)) = step.settled.pixels.get(lane).filter(|(_, c)| !c.is_empty())
        {
            let voxels: usize = clips
                .iter()
                .map(|[t, h, w]| *t as usize * *h as usize * *w as usize)
                .sum();
            out.push(LaneReadout {
                rows: u32::try_from(voxels).unwrap_or(u32::MAX),
                width: u32::try_from(pixels.len() / voxels.max(1)).unwrap_or(u32::MAX),
                values: pixels.clone(),
                scores,
                seam: engine::fire::ReadoutSeam::Pixels,
                clips: clips.clone(),
            });
            continue;
        }
        out.push(match want {
            Readout::None => LaneReadout {
                scores,
                ..LaneReadout::default()
            },
            Readout::Last | Readout::Rows(_) => LaneReadout {
                rows,
                width,
                values,
                scores,
                seam: step
                    .settled
                    .seams
                    .get(lane)
                    .copied()
                    .unwrap_or(engine::fire::ReadoutSeam::Logits),
                clips: Vec::new(),
            },
        });
    }
    out
}

// SAFETY: `Shell` holds raw device handles inline, none of which is `Send`
// to the compiler. Sound because every verb takes `&mut self`, so exactly
// one thread touches a shell at a time.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

#[cfg(test)]
mod tests {
    use super::patch_ladder;
    use engine::load::Budgets as LoadBudgets;
    use model_compiler::PATCH_LATTICE_FLOOR;
    use model_ir::{Def, Dim, Dtype, RuntimeInput, Trace, Ty, ValueDecl};

    fn trace_with(shape: Vec<Dim>) -> Trace {
        Trace {
            name: "gate".into(),
            platform: model_ir::Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values: vec![ValueDecl {
                def: Def::Input(RuntimeInput::Tokens),
                ty: Ty::Tensor {
                    shape,
                    dtype: Dtype::I32,
                },
            }],
            nodes: Vec::new(),
            seams: Vec::new(),
            drafter: None,
        }
    }

    #[test]
    fn api_every_case() {
        a_tower_plan_derives_a_ladder_from_nothing_but_its_own_declaration();
        a_stated_ceiling_wins_and_still_gets_its_rungs();
    }

    fn a_tower_plan_derives_a_ladder_from_nothing_but_its_own_declaration() {
        let ladder = patch_ladder(
            &trace_with(vec![Dim::Patches, Dim::Const(768)]),
            &LoadBudgets::default(),
        )
        .expect("a plan that states patch rows gets a ladder");
        assert_eq!(
            ladder.max_patches, 4096,
            "two whole images at the native grid"
        );
        assert_eq!(
            ladder.buckets,
            vec![64, 128, 256, 512, 1024, 2048, 4096],
            "rungs double from the patch lattice's floor to the ceiling"
        );
        assert_eq!(
            ladder.max_images,
            4096 / PATCH_LATTICE_FLOOR,
            "as many images as the ceiling holds at the smallest whole image"
        );

        for shape in [vec![Dim::Images], vec![Dim::ImagesPlus(1)]] {
            assert!(patch_ladder(&trace_with(shape), &LoadBudgets::default()).is_some());
        }
    }

    fn a_stated_ceiling_wins_and_still_gets_its_rungs() {
        let budgets = LoadBudgets {
            max_patches: Some(1024),
            max_images: Some(3),
            ..LoadBudgets::default()
        };
        let ladder = patch_ladder(&trace_with(vec![Dim::Patches]), &budgets).expect("a ladder");
        assert_eq!(ladder.max_patches, 1024);
        assert_eq!(ladder.max_images, 3);
        assert_eq!(ladder.buckets, vec![64, 128, 256, 512, 1024]);

        let tiny = LoadBudgets {
            max_patches: Some(8),
            ..LoadBudgets::default()
        };
        let raised = patch_ladder(&trace_with(vec![Dim::Patches]), &tiny).expect("a ladder");
        assert_eq!(raised.max_patches, PATCH_LATTICE_FLOOR);
        assert_eq!(raised.buckets, vec![PATCH_LATTICE_FLOOR]);
    }
}

fn width_free(sku: &str) -> &str {
    match sku.rsplit_once("-tp") {
        Some((base, width)) if !width.is_empty() && width.bytes().all(|b| b.is_ascii_digit()) => {
            base
        }
        _ => sku,
    }
}

fn refuse_an_artifact_for_another_deployment(
    path: &std::path::Path,
    backend: &str,
    sku: &str,
) -> EngineResult<()> {
    let stamp = match checkpoint::file::serve::stamp_of(path) {
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,
        Err(why) => return Err(Error::Load(why.to_string())),
    };
    let deployment = checkpoint::serving::Stamp::of(backend, sku);
    stamp
        .check(&deployment)
        .map_err(|mismatch| Error::Load(mismatch.refuse(&path.display().to_string())))
}

#[cfg(test)]
mod serving_stamp_tests {
    use super::refuse_an_artifact_for_another_deployment as refuse;
    use checkpoint::file::emit::{self, Object};
    use checkpoint::serving::Stamp;
    use std::collections::BTreeMap;

    fn artifact(dir: &std::path::Path, backend: &str, sku: &str) -> std::path::PathBuf {
        let path = dir.join(format!("{backend}-{sku}.zt"));
        let bytes = vec![7u8; 8192];
        emit::write(
            &path,
            &Stamp::of(backend, sku),
            &BTreeMap::new(),
            4096,
            &[Object::leaf("embed", vec![8192], ztensor::Leaf::U8, &bytes)],
            |o, p, _| panic!("{o}/{p} is not streamed here"),
        )
        .expect("the fixture artifact writes");
        path
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cuda_stamp_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_artifact_for_another_shell_is_refused_before_anything_is_opened() {
        let dir = tmp("cross");
        let foreign = artifact(&dir, "metal", "qwen_3");
        let why =
            refuse(&foreign, "cuda", "qwen_3").expect_err("a metal artifact is not servable here");
        let said = format!("{why}");
        for wanted in [
            "backend",
            "\"metal\"",
            "\"cuda\"",
            "pie model import --force",
        ] {
            assert!(
                said.contains(wanted),
                "the refusal does not say {wanted:?}: {said}"
            );
        }
        refuse(&artifact(&dir, "cuda", "qwen_3"), "cuda", "qwen_3")
            .expect("a cuda artifact serves on cuda");
        std::fs::remove_dir_all(&dir).ok();
    }
}
