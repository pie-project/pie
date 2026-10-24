#![allow(unsafe_code)]
use std::collections::BTreeMap;
use std::path::Path;
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
use engine::transfer::{KvCopy, MemoryDomain, StateCopy, StateDirection};
use eta_ir::registry::{GeometryClass, ModelProfile, Port, PortMask};
use eta_ir::types::Dtype;
use poem_compiler::{Budget, PATCH_LATTICE_FLOOR, PatchLadder, VoxelLadder};
use poem_ir::Trace;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Knobs, Lane, Seated, Shell};

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

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

#[derive(Debug, Clone, Default)]
pub struct DeviceBoot {
    pub ordinal: i32,
    pub world: World,
    pub comm: Option<Arc<crate::comm::Comm>>,
    pub knobs: Knobs,
    pub cache_dir: Option<std::path::PathBuf>,
    pub adapter_dir: Option<std::path::PathBuf>,
}

struct PendingStep {
    readout: Vec<Readout>,
    settled: crate::serve::Settled,
}

pub struct Cuda {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    channels: BTreeMap<ChannelId, Arc<crate::program::Endpoint>>,
    sink: Option<engine::CompletionSink>,
    tap_sink: Option<engine::TapSink>,
    pending: Option<(FrameId, Vec<PendingStep>)>,
    adapters: BTreeMap<InstanceId, crate::Binding>,
    replay: Option<crate::replay::Replay>,
}

impl Cuda {
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Cuda {
        crate::serve::diag::publish(&boot.knobs.diagnostics);
        Cuda {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            channels: BTreeMap::new(),
            sink: None,
            tap_sink: None,
            pending: None,
            adapters: BTreeMap::new(),
            replay: None,
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
        Fault::Runtimeless | Fault::Device { .. } | Fault::Unsupported(_) => {
            Error::Device(fault.to_string())
        }
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
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

/// The backend the host emits fused regions for: the HIP dialect (wave64)
/// on ROCm.
#[cfg(not(feature = "rocm"))]
pub const CODEGEN: &str = "cuda";
#[cfg(feature = "rocm")]
pub const CODEGEN: &str = "hip";

/// Powers of two from the floor up to `max_tokens`, with `steps - 1` more
/// between each pair; from `fine_from` rows (0 never), at least `fine_steps`
/// a doubling. A prefill's GEMMs run its bucket's
/// rows, so a 2K prompt padded to 4096 costs a second prompt, while buckets
/// this tall are few.
#[must_use]
pub(crate) fn default_lattice(
    max_tokens: u32,
    steps: u32,
    fine_from: u32,
    fine_steps: u32,
) -> Vec<u32> {
    let steps = steps.max(1);
    let mut lattice: Vec<u32> = Vec::new();
    let push = |point: f64, lattice: &mut Vec<u32>| -> bool {
        if point >= f64::from(max_tokens) {
            return false;
        }
        let point = point as u32;
        if lattice.last() != Some(&point) {
            lattice.push(point);
        }
        true
    };
    for i in 0u32.. {
        let point = (f64::from(LATTICE_FLOOR) * 2f64.powf(f64::from(i) / f64::from(steps))).ceil();
        if fine_from > 0 && point >= f64::from(fine_from) {
            break;
        }
        if !push(point, &mut lattice) {
            lattice.push(max_tokens);
            return lattice;
        }
    }
    let fine = steps.max(fine_steps);
    for i in 0u32.. {
        let point = (f64::from(fine_from) * 2f64.powf(f64::from(i) / f64::from(fine))).ceil();
        if !push(point, &mut lattice) {
            break;
        }
    }
    lattice.push(max_tokens);
    lattice
}

pub(crate) const LATTICE_FLOOR: u32 = 1;

#[must_use]
pub(crate) fn lattice(
    stated: Vec<u32>,
    max_tokens: u32,
    steps: u32,
    fine_from: u32,
    fine_steps: u32,
) -> Vec<u32> {
    if stated.is_empty() {
        default_lattice(max_tokens, steps, fine_from, fine_steps)
    } else {
        stated
    }
}

/// The rows a fire takes when neither `max_forward_tokens` nor the chip's
/// profile states them.
pub(crate) const DEFAULT_FORWARD_TOKENS: u32 = 8192;

/// The load's budgets for `trace`: its logit readouts sized to what the
/// trace drafts, not the general 16 a lane, so a vocab-wide head carves only
/// the rows a fire can read out.
fn bake_budgets(budgets: &LoadBudgets, trace: &Trace) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
        readouts_per_lane: poem_compiler::arena::readouts_per_lane(trace),
    }
}

pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let declares_patches = trace.values.iter().any(|decl| {
        matches!(&decl.ty, poem_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Patches))
    });
    if !declares_patches {
        return None;
    }

    let max_patches = budgets
        .max_patches
        .unwrap_or_else(|| {
            let tokens = match budgets.max_tokens {
                0 => DEFAULT_FORWARD_TOKENS,
                tokens => tokens,
            };
            tokens.min(DERIVED_PATCH_CEILING)
        })
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
            .unwrap_or(max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

#[must_use]
pub fn voxel_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<VoxelLadder> {
    const DERIVED_VOXEL_CEILING: u32 = 65_536;

    let declares_voxels = trace.values.iter().any(|decl| {
        matches!(&decl.ty, poem_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Voxels))
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

    fn export_kv_handle(&self) -> Option<engine::KvHandle> {
        self.caps.as_ref()?.kv_handle.clone()
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

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the cuda shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        refuse_an_artifact_for_another_deployment(
            &path,
            trace.platform.backend(),
            width_free(&trace.name),
        )?;
        let (contract, plan) = self.settle(&trace, &path, &residency)?;
        let mut budgets = budgets;
        if budgets.context_declared && residency.device_kv_budget.is_none() {
            let fits = crate::store::declared_context_that_fits(
                &trace,
                budgets.page_size,
                budgets.max_context,
                if budgets.max_tokens == 0 {
                    16_384
                } else {
                    budgets.max_tokens
                },
                residency.device_memory_share,
                plan.device_demand(),
            )
            .map_err(fault)?;
            if fits < budgets.max_context {
                eprintln!(
                    "engine-cuda: the model declares a {}-token context and one sequence at it \
                     takes more than half of what `[model] device_memory_share` leaves past the \
                     weights; serving {fits} tokens. State `[engine] max_model_len` to choose.",
                    budgets.max_context
                );
                budgets.max_context = fits;
            }
        }

        let patches = patch_ladder(&trace, &budgets);
        let voxels = voxel_ladder(&trace, &budgets);
        let budget = bake_budgets(&budgets, &trace);
        let mut shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget,
            patches,
            voxels,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            kv_budget: residency.device_kv_budget,
            memory_share: residency.device_memory_share,
            ordinal: if ordinal >= 0 {
                ordinal
            } else {
                self.boot.ordinal
            },
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
                .map_or(core::ptr::null(), Arc::as_ptr),
        })
        .map_err(fault)?;

        shell.mount_adapters(self.boot.adapter_dir.clone());
        let host_kv_pages = shell
            .seat_host_kv(
                residency.host_kv_budget,
                residency.kv_seats.unwrap_or(budgets.max_lanes),
                budgets.max_context,
                self.boot.world.size,
            )
            .map_err(fault)?;
        // Every page the load serves is backed now, so serving maps nothing:
        // a dying process holding the driver's lock stalls no fire of this
        // one's.
        shell.back_whole().map_err(fault)?;
        let kv_handle = if residency.kv_export {
            Some(shell.export_kv(self.boot.world).map_err(fault)?)
        } else {
            None
        };
        let disk_kv_pages = shell
            .seat_disk_kv(
                residency.disk_kv_budget,
                residency.disk_kv_dir.as_deref(),
                self.boot.world,
            )
            .map_err(fault)?;

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
            .any(|row| matches!(row, poem_ir::CacheRow::State { .. }));
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
                codegen_backend: Some(CODEGEN.to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                state_slot_bytes: shell.state_slot_bytes(),
                buffer_pages: shell.buffer_pages(),
                adapter_banks: shell
                    .banks()
                    .iter()
                    .map(|&(_, adapters, _)| adapters)
                    .min()
                    .unwrap_or(0),
                elastic_page_bytes,
                elastic_budget_pages,
                window_pages: paging.window.map_or(0, |_| {
                    u32::try_from(paging.window_pages()).unwrap_or(u32::MAX)
                }),
                window_tokens: paging.window.map_or(0, |window| window.tokens),
                host_kv_pages,
                host_window_pages: shell.host_window_pages(),
                host_state_slots: shell.host_state_slots(),
                host_buffer_pages: shell.host_buffer_pages(),
                disk_kv_pages,
                one_frame_when_short: shell.one_frame_when_short(),
                seated_first: shell.seated_first(),
            },
            limits: FireLimits {
                max_lanes: shell.budget().max_lanes,
                max_tokens: shell.budget().max_tokens,
                max_page_refs: paging
                    .pages_per_slot
                    .saturating_mul(shell.budget().max_lanes),
                max_context: paging.context(),
                prefill_rows: 0,
                max_readouts: u32::try_from(poem_compiler::arena::readouts_ceiling(shell.budget()))
                    .unwrap_or(u32::MAX),
                max_scoring_lanes: shell.scoring_lanes(),
            },
            profile,
            ports: PortMask::DEVICE_GEOMETRY
                .with(Port::AttnMask)
                .with(Port::Fold),
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: host_kv_pages > 0,
                host_to_device: host_kv_pages > 0,
                host_to_host: false,
            },
            kv_handle,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: true,
            rs_snapshots: if state_rows && shell.lands_snapshots() {
                crate::store::rs::SNAPSHOTS
            } else {
                0
            },
            bidirectional_attention: true,
            adapter_attachments: true,
            facts: shell.trace().facts.clone(),
        };

        self.replay = crate::replay::Replay::of(&shell);
        if let Some(sink) = &self.tap_sink {
            shell.on_tapped(sink.clone());
        }
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
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::Serves::NONE, engine::Serves::of),
        )?;
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
            // A causal encoder-decoder's long prompts: their leading rows
            // through the encoder alone first, unannounced, then the step
            // with each such prompt's last window (`crate::replay`).
            let split = self.replay.and_then(|replay| replay.split(step));
            if let Some((head, _)) = &split
                && let Err(error) = self.fire_step(head, at, false)
            {
                self.pending = None;
                let airborne = self.shell.as_ref().map_or(0, Shell::airborne_steps);
                return Err(attributed(error, airborne));
            }
            let step = split.as_ref().map_or(step, |(_, tail)| tail);
            let (ticket, step_settled) = match self.fire_step(step, at, true) {
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

    /// One rank's planes hold a band of each layer's heads, so a
    /// tensor-parallel rank taps nothing.
    fn tap_groups(&self) -> Option<engine::TapGroups> {
        self.shell.as_ref()?.tap_groups()
    }

    fn on_tapped(&mut self, sink: engine::TapSink) {
        if let Some(shell) = self.shell.as_mut() {
            shell.on_tapped(sink.clone());
        }
        self.tap_sink = Some(sink);
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

    fn publish_landed(&mut self) -> EngineResult<()> {
        match self.shell.as_mut() {
            Some(shell) => shell.reap_landed_guests().map_err(fault),
            None => Ok(()),
        }
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
        let stream = self.loaded_mut()?.stream();
        let endpoint = Arc::new(
            crate::program::Endpoint::open(registration.host_role, cell_bytes, capacity, stream)
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
        let stream = self.loaded_mut()?.stream();
        self.instance(instance)?
            .publish(channel, cell, stream)
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
        let pairs = |moves: &[engine::transfer::StateMove]| -> Vec<(u32, u32)> {
            moves
                .iter()
                .map(|m| (m.src_slot_id, m.dst_slot_id))
                .collect()
        };
        let (states, pages) = (pairs(&copy.moves), pairs(&copy.pages));
        let to_host = match copy.direction {
            StateDirection::DeviceToDevice => {
                for &(src, dst) in &states {
                    shell.copy_state(src, dst).map_err(fault)?;
                }
                return shell.copy_pages(&pages).map_err(fault);
            }
            StateDirection::DeviceToHost => true,
            StateDirection::HostToDevice => false,
        };
        let device_host = |pairs: Vec<(u32, u32)>| -> Vec<(u32, u32)> {
            if to_host {
                pairs
            } else {
                pairs
                    .into_iter()
                    .map(|(host, device)| (device, host))
                    .collect()
            }
        };
        shell
            .copy_state_host(to_host, &device_host(states))
            .map_err(fault)?;
        shell
            .copy_pages_host(to_host, &device_host(pages))
            .map_err(fault)
    }

    fn host_kv_layout(&self) -> Option<state::PageLayout> {
        self.shell.as_ref()?.host_kv_layout(1)
    }

    fn read_host_kv(&mut self, slots: &[u32]) -> EngineResult<Vec<u8>> {
        self.loaded_mut()?.read_host_kv(slots).map_err(fault)
    }

    fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> EngineResult<()> {
        self.loaded_mut()?
            .write_host_kv(slots, bytes)
            .map_err(fault)
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        // A tensor-parallel group names its device by its first rank's
        // ordinal; to each rank, a move between "the device" and a host or
        // disk tier is a move out of or into its own.
        let tier = |domain| matches!(domain, MemoryDomain::HostPinned | MemoryDomain::LocalDisk);
        let own = |domain| match (domain, device) {
            (MemoryDomain::CudaDevice(_), Some(device)) => device,
            _ => domain,
        };
        let copy = &match (copy.src, copy.dst) {
            (src, dst) if tier(dst) && !tier(src) => KvCopy {
                src: own(src),
                ..copy.clone()
            },
            (src, dst) if tier(src) && !tier(dst) => KvCopy {
                dst: own(dst),
                ..copy.clone()
            },
            _ => copy.clone(),
        };
        if let Some((tier, out, pages, slots)) = device.and_then(|device| copy.tier_move(device)) {
            // The windowed pairs are stated in the copy's direction; the pool
            // takes them as (device, host).
            let windowed: Vec<(u32, u32)> = copy
                .windowed
                .iter()
                .map(|&(src, dst)| if out { (src, dst) } else { (dst, src) })
                .collect();
            if tier == MemoryDomain::LocalDisk && !windowed.is_empty() {
                return Err(Error::Unsupported {
                    verb: "windowed kv pages to or from disk",
                    engine: "cuda",
                });
            }
            let shell = self.loaded_mut()?;
            return match tier {
                MemoryDomain::LocalDisk => shell.spill_kv(out, pages, slots),
                _ => shell.swap_kv(out, pages, slots, &windowed),
            }
            .map_err(fault);
        }
        if device.is_none_or(|device| copy.src != device || copy.dst != device) {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "cuda",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = copy.runs(page_size)?;
        self.loaded_mut()?
            .copy_kv(&moves, !copy.moves.is_empty())
            .map_err(fault)
    }
}

impl Cuda {
    /// `submission` fired; its completion told the sink where `announce`.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
        announce: bool,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        let done = self
            .sink
            .as_ref()
            .filter(|_| announce)
            .map(|sink| crate::serve::Done {
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
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    positions: &lane.positions,
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    kv_less: lane.kv_less,
                    translation: &lane.kv.translation,
                    window: &lane.kv.window,
                    window_copies: &lane.kv.window_copies,
                    mask: lane.mask.as_ref(),
                    adapter: lane_adapters[at].or(lane.adapter),
                    drafts: lane.drafts,
                    captures_scores: lane.captures_scores,
                    bidirectional: lane.bidirectional,
                    self_cond: lane.self_cond.as_ref(),
                    rs: lane.rs.clone(),
                    rs_reset: lane.rs_reset,
                    rs_slot: lane.rs_slot,
                    rs_snapshots: &lane.rs_snapshots,
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    stream: lane.stream as u8,
                    group: lane.group,
                    peer: lane.peer,
                    ports: &lane.ports,
                    attn_classes: lane.attn_classes.as_ref(),
                    reports: lane.reports,
                    tap: lane.tap.as_ref(),
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
                return Err(fault(crate::error::Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless {
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
                return Err(fault(crate::error::Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Vaeless {
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
                grids: &row.grids,
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
    patches: &[u8],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::U8 => Ok(patches.to_vec()),
        _ => Err("a media submission against a plan whose pixel rows are not u8 bytes"),
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

// SAFETY: `Shell` holds raw device handles and `Cell`s inline, none of
// which the compiler lets cross threads. The handles name driver objects that
// any thread may use once it binds the context, and every verb that touches
// the device, the stream or a `Cell` takes `&mut self` (and `bind_thread`
// binds the context first), so one thread drives a shell at a time.
unsafe impl Send for Cuda {}

#[cfg(test)]
mod tests {
    use super::patch_ladder;
    use engine::load::Budgets as LoadBudgets;
    use poem_compiler::PATCH_LATTICE_FLOOR;
    use poem_ir::{Def, Dim, Dtype, RuntimeInput, Trace, Ty, ValueDecl};

    fn trace_with(shape: Vec<Dim>) -> Trace {
        Trace {
            facts: Default::default(),
            name: "gate".into(),
            platform: poem_ir::Platform::Cuda,
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

fn width_free(deployment: &str) -> &str {
    match deployment.rsplit_once("-tp") {
        Some((base, width)) if !width.is_empty() && width.bytes().all(|b| b.is_ascii_digit()) => {
            base
        }
        _ => deployment,
    }
}

fn refuse_an_artifact_for_another_deployment(
    path: &std::path::Path,
    backend: &str,
    deployment: &str,
) -> EngineResult<()> {
    let stamp = match checkpoint::file::serve::stamp_of(path) {
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,
        Err(why) => return Err(Error::Load(why.to_string())),
    };
    let wanted = checkpoint::serving::Stamp::of(backend, deployment);
    stamp
        .check(&wanted)
        .map_err(|mismatch| Error::Load(mismatch.refuse(&path.display().to_string())))
}

#[cfg(test)]
mod serving_stamp_tests {
    use super::refuse_an_artifact_for_another_deployment as refuse;
    use checkpoint::file::emit::{self, Object};
    use checkpoint::serving::Stamp;
    use std::collections::BTreeMap;

    fn artifact(dir: &std::path::Path, backend: &str, deployment: &str) -> std::path::PathBuf {
        let path = dir.join(format!("{backend}-{deployment}.zt"));
        let bytes = vec![7u8; 8192];
        emit::write(
            &path,
            &Stamp::of(backend, deployment),
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
