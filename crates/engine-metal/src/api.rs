use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
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
use model_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use model_ir::Trace;

use crate::error::Fault;
use crate::experts;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Landed, Lane, Seated, Shell, StepView};
use crate::settle::Done;
use crate::weights::AdapterPlane;

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    pub gpu_mem_utilization: f64,
    pub adapter_dir: Option<std::path::PathBuf>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            gpu_mem_utilization: crate::store::accounting::DEFAULT_GPU_MEM_UTILIZATION,
            adapter_dir: None,
        }
    }
}

struct PendingStep {
    readout: Vec<Readout>,
    landed: Landed,
    rows: Option<Vec<Vec<f32>>>,
}

pub struct Metal {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<PendingStep>)>,
    adapters: BTreeMap<InstanceId, crate::adapter::Binding>,
}

impl Metal {
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Metal {
        Metal {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
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

    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id,
            })
    }

    #[allow(dead_code)]
    fn loaded(&self) -> EngineResult<&Shell> {
        self.shell
            .as_ref()
            .ok_or_else(|| Error::Load("the metal engine has no model loaded".into()))
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the metal engine has no model loaded".into()))
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
        (MemoryDomain::MetalPrivate, _) | (_, MemoryDomain::MetalPrivate) => {
            "`copy_kv` with a Metal PRIVATE end, and every reservation this load made is \
             Shared — there are no private pages here to read or write"
        }
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

fn fault(fault: Fault) -> Error {
    match fault {
        Fault::Deviceless | Fault::Device { .. } => Error::Device(fault.to_string()),
        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Backing { .. }
        | Fault::Mapped { .. }
        | Fault::Param { .. }
        | Fault::Recipe(_)
        | Fault::Shader { .. }
        | Fault::Unbound { .. } => Error::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Fragmented { .. } => Error::Device(fault.to_string()),
        Fault::Unaffine { .. } | Fault::Unstructured { .. } => Error::Load(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. }
        | Fault::Positions { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => Error::Invalid(fault.to_string()),
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Scoreless { .. } | Fault::ScoreWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) => Error::Invalid(fault.to_string()),
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
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

fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

#[must_use]
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
            .unwrap_or(max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

#[must_use]
pub fn voxel_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<model_compiler::VoxelLadder> {
    const DERIVED_VOXEL_CEILING: u32 = 65_536;

    let declares_voxels = trace.values.iter().any(|decl| {
        matches!(&decl.ty, model_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(model_ir::RowAxis::Voxels))
    });
    if !declares_voxels {
        return None;
    }
    let max_voxels = budgets.max_voxels.unwrap_or(DERIVED_VOXEL_CEILING).max(1);
    Some(model_compiler::VoxelLadder {
        max_voxels,
        buckets: Vec::new(),
        max_clips: budgets
            .max_clips
            .unwrap_or(budgets.max_lanes)
            .clamp(1, max_voxels),
    })
}

fn patch_bytes(
    patches: &[f32],
    element: model_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        model_ir::Dtype::Bf16 => Ok(patches
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect()),
        model_ir::Dtype::F32 => Ok(patches.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err(
            "a media submission against a plan whose activation element is neither \
                  `bf16` nor `f32`, which is the pair every tower in this catalog computes in",
        ),
    }
}

fn voxel_bytes(
    payload: &[f32],
    element: model_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        model_ir::Dtype::Bf16 => Ok(payload
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect()),
        model_ir::Dtype::F32 => Ok(payload.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err(
            "a voxel submission against a plan whose voxel element is neither `bf16` \
                  nor `f32`, which is the pair every VAE in this catalog computes in",
        ),
    }
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn profile(shell: &Shell, budgets: &LoadBudgets) -> EngineResult<ModelProfile> {
    let trace = shell.trace();
    let layers = trace
        .nodes
        .iter()
        .filter_map(|node| node.layer)
        .max()
        .map_or(0, |top| top + 1);
    let width = u32::try_from(shell.out_width().map_err(fault)?).unwrap_or(u32::MAX);
    let seam = shell.readout_seam();
    let vocab = match seam {
        engine::fire::ReadoutSeam::Logits => width,
        _ => 0,
    };
    Ok(ModelProfile {
        vocab,
        page_size: budgets.page_size,
        num_layers: layers,
        activation: Dtype::F32,
        has_mtp_logits: shell.drafts(),
        mtp_depth: shell.mtp_depth(),
        draft_block: shell.trace().drafter.map_or(0, |d| d.rows),
        draft_mask_token: shell.trace().drafter.map_or(0, |d| d.mask_token),
        draft_bidirectional: shell.trace().drafter.is_some_and(|d| d.bidirectional),
        draft_proposals_from: shell.trace().drafter.map_or(1, |d| d.proposals_from),
        has_value_head: false,
        has_attn_score: shell.observes_scores(),
        has_attn_page_mask: false,
        has_lora: true,
        has_velocity: seam == engine::fire::ReadoutSeam::Velocity,
        velocity_width: match seam {
            engine::fire::ReadoutSeam::Velocity => width,
            _ => 0,
        },
        has_pixels: shell.pixels_width().is_some(),
        pixels_width: shell.pixels_width().unwrap_or(0),
        kernels: Vec::new(),
    })
}

fn adapter_of(
    shell: &mut Shell,
    program: engine::program::ProgramId,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::adapter::Binding>> {
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
                     weights are the seed, because the fire path never reads the cell, \
                     so an unseeded plane is a correction of zero \
                     that nobody asked for",
                    role.bank()
                ))
            })?;
        built.extend(crate::adapter::planes_of(*role, site, wire, &seats).map_err(fault)?);
    }
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .bind_adapter(crate::adapter::Source::Own {
            instance,
            planes: &planes,
        })
        .map(Some)
        .map_err(fault)
}

impl Engine for Metal {
    fn kind(&self) -> &'static str {
        "metal"
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
                "this metal engine already has a model loaded; one shell per engine".into(),
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

        if ordinal > 0 {
            return Err(Error::unsupported("metal", "device ordinal selection"));
        }

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the metal shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        let planes = crate::weights::attachments(&trace, &contract, &path).map_err(fault)?;
        let gathered = crate::gather::Plan::of(
            &trace,
            &planes,
            residency.device_weight_budget,
            budgets.max_tokens,
        )
        .map_err(fault)?;
        let mut residency_plan =
            experts::Plan::beside(&trace, &planes, residency.device_weight_budget, gathered)
                .map_err(fault)?;

        {
            let working_set = crate::device::Context::bind().map_err(fault)?.working_set();
            let util = self.boot.gpu_mem_utilization;
            let paging = crate::store::kv::Paging::of(
                budgets.page_size,
                budgets.max_context,
                budgets.slots,
                u64::from(budgets.pages),
            )
            .map_err(|error| fault(Fault::from(error)))?;
            let kv_pool = crate::store::pool_demand(&trace, paging).map_err(fault)?;

            let acct = crate::store::accounting::Accounting::of(
                working_set,
                util,
                residency_plan.device_demand(),
                kv_pool,
            );
            if acct.admit(residency.device_weight_budget, util).is_err() {
                let headroom = acct.weight_headroom();
                let regathered =
                    crate::gather::Plan::of(&trace, &planes, Some(headroom), budgets.max_tokens);
                let regathered = match regathered {
                    Ok(plan) => plan,
                    Err(_) => {
                        return Err(fault(
                            acct.admit(residency.device_weight_budget, util)
                                .expect_err("admit errored above"),
                        ));
                    }
                };
                match experts::Plan::beside(&trace, &planes, Some(headroom), regathered) {
                    Ok(shrunk) => {
                        let re = crate::store::accounting::Accounting::of(
                            working_set,
                            util,
                            shrunk.device_demand(),
                            kv_pool,
                        );
                        re.admit(Some(headroom), util).map_err(fault)?;
                        residency_plan = shrunk;
                    }
                    Err(_) => {
                        return Err(fault(
                            acct.admit(residency.device_weight_budget, util)
                                .expect_err("admit errored above"),
                        ));
                    }
                }
            }
        }

        residency.admit(residency_plan.device_demand(), residency_plan.host_demand())?;
        let streams = residency_plan.streams();

        let patches = patch_ladder(&trace, &budgets);
        let voxels = voxel_ladder(&trace, &budgets);
        let mut shell = Shell::load(Boot {
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
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            residency: residency_plan,
        })
        .map_err(fault)?;

        shell.mount_adapters(self.boot.adapter_dir.clone());

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let weights_warm = shell.weights_warm();

        let paging = shell.paging();
        let state_rows = shell
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, model_ir::CacheRow::State { .. }));
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "metal".to_string(),
                domain: MemoryDomain::MetalShared,
                sms: 0,
                unified_memory: true,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: shell.max_buffer(),
                codegen_backend: Some("metal".to_string()),
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
                elastic_page_bytes: 0,
                elastic_budget_pages: 0,
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
            device_channel_commit: false,
            rs_verbs: shell.serves_rs_verbs(),
            bidirectional_attention: true,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                weights_resident: !streams,
                weights_from_cache: weights_warm,
                arena_bytes,
                pool_bytes,
                input_bytes,
                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        let planes: Vec<AdapterPlane<'_>> = registration
            .planes
            .iter()
            .map(|plane| AdapterPlane {
                bank: plane.bank.as_str(),
                bytes: &plane.bytes,
            })
            .collect();
        self.loaded_mut()?
            .register_adapter(registration.id, &planes)
            .map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(engine::fire::Serves {
            device_channel_commit: false,
            rs_verbs: self.caps.as_ref().is_some_and(|caps| caps.rs_verbs),
            bidirectional: true,
        })?;
        let id = self.next_frame;
        self.next_frame = self.next_frame.wrapping_add(1);
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut pending = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            match self.fire_step(step, at) {
                Ok((ticket, step_pending)) => {
                    steps.push(ticket);
                    pending.push(step_pending);
                }
                Err(error) => {
                    self.pending = None;
                    return Err(error);
                }
            }
        }
        let mut ticket = FrameTicket { id, steps };
        self.pending = Some((id, pending));
        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    fn settles_asynchronously(&self) -> bool {
        self.shell
            .as_ref()
            .is_some_and(|shell| shell.frames_in_flight() > 1)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has already taken \
                 its readout seats",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: frame {id} has been submitted since, and \
                 a step's rows live in the readout seat its arm owns. Ask for a \
                 frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut pending) = self.pending.take().expect("checked just above");
        let mut refused = None;
        match self.shell.as_mut() {
            None => {
                refused = Some(Error::Load("the metal engine has no model loaded".into()));
            }
            Some(shell) => {
                for step in &mut pending {
                    if step.rows.is_some() {
                        continue;
                    }
                    match shell.rows_of(&step.landed) {
                        Ok(rows) => step.rows = Some(rows),
                        Err(why) => {
                            refused = Some(fault(why));
                            break;
                        }
                    }
                }
            }
        }
        self.pending = Some((ticket.id, pending));
        if let Some(error) = refused {
            return Err(error);
        }
        let seam = self
            .shell
            .as_ref()
            .map_or(engine::fire::ReadoutSeam::Logits, Shell::readout_seam);
        let (_, pending) = self.pending.as_ref().expect("just put back");
        for (receipt, step) in ticket.steps.iter_mut().zip(pending) {
            receipt.readouts = readouts_of(step, seam);
        }
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            let _ = shell.reap();
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
        if registration.host_role != eta_ir::container::HostRole::None {
            return Err(self.unsupported("register_channel"));
        }
        let shape = crate::program::ChannelShape {
            capacity: registration.capacity.max(1),
            numel: registration
                .shape
                .iter()
                .map(|&dim| dim as usize)
                .product::<usize>()
                .max(1),
            dtype: registration.dtype.program_dtype(),
        };
        self.loaded_mut()?
            .register_shared_channel(registration.id, shape)
            .map_err(fault)?;
        Ok(RegisteredChannel {
            id: registration.id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
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
        let shell = self.loaded_mut()?;
        let id = shell
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
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
        if let Some(bound) = held.as_ref() {
            shell.release_adapter(bound);
        }
        shell.close_program_instance(id).map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        if self.loaded_mut()?.close_shared_channel(id) {
            return Ok(());
        }
        Err(self.unsupported("close_channel"))
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

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let served = self
            .caps
            .as_ref()
            .is_some_and(|caps| copy.src == caps.device.domain && copy.dst == caps.device.domain);
        if !served {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "metal",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = crate::store::Move::plan(copy, page_size).map_err(Error::Invalid)?;
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
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
        let moves: Vec<(u32, u32)> = copy
            .moves
            .iter()
            .map(|move_| (move_.src_slot_id, move_.dst_slot_id))
            .collect();
        self.loaded_mut()?.copy_state(&moves).map_err(fault)
    }
}

impl Metal {
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
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
            let Some(element) = self.loaded()?.patch_element() else {
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
                        engine: "metal",
                    }
                })?);
            }
        }
        let mut voxel_payloads: Vec<Vec<u8>> = Vec::with_capacity(submission.voxels.len());
        if !submission.voxels.is_empty() && self.loaded()?.voxel_element().is_none() {
            return Err(Error::Invalid(
                "this step submits VAE clips and the loaded plan states no voxel row; a \
                 model with no autoencoder has nothing to decode them with"
                    .into(),
            ));
        }
        let voxel_element = self
            .loaded()?
            .voxel_element()
            .unwrap_or(model_ir::Dtype::Bf16);
        for row in &submission.voxels {
            voxel_payloads.push(voxel_bytes(&row.payload, voxel_element).map_err(|why| {
                Error::Unsupported {
                    verb: why,
                    engine: "metal",
                }
            })?);
        }
        let clips: Vec<crate::serve::Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&voxel_payloads)
            .map(|(row, payload)| crate::serve::Clips {
                lane: row.lane,
                boxes: &row.clips,
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

        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        for (index, lane) in submission.lanes.iter().enumerate() {
            let listed = matches!(lane.readout, Readout::Rows(_));
            let served = attached
                .iter()
                .any(|a| a.lane as usize == index && a.at == engine::fire::Boundary::Epilogue);
            if listed && !served {
                return Err(Error::unsupported("metal", "row-selected readout"));
            }
        }

        let sink = self.sink.clone();
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
                    stream: lane.stream as u8,
                    group: lane.group,
                    ports: &lane.ports,
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    captures_scores: lane.captures_scores,
                    mask: lane.mask.as_ref(),
                    bidirectional: lane.bidirectional,
                    self_cond: lane.self_cond.as_ref(),
                    adapter: lane_adapters[at].or(lane.adapter),
                    positions: &lane.positions,
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    translation: &lane.kv.translation,
                    rs: &lane.rs,
                    rs_reset: lane.rs_reset,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        let landed = {
            use engine::frame::Shell as FrameShell;
            let done = sink.map(|sink| Done { at, sink });
            let prepared = FrameShell::prepare(
                shell,
                StepView {
                    lanes: &seated,
                    attachments: &attached,
                    media: &media,
                    clips: &clips,
                    done,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            FrameShell::settle(shell, enqueued).map_err(fault)?
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
                landed,
                rows: None,
            },
        ))
    }
}

fn readouts_of(step: &PendingStep, seam: engine::fire::ReadoutSeam) -> Vec<LaneReadout> {
    let rows: &[Vec<f32>] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| match policy {
            Readout::None | Readout::Rows(_) => LaneReadout::default(),
            Readout::Last => {
                let values = rows.get(lane).cloned().unwrap_or_default();
                LaneReadout {
                    rows: 1,
                    width: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    seam,
                    values,
                    ..LaneReadout::default()
                }
            }
        })
        .collect()
}

// SAFETY: `Engine` requires `Send + Sync`; a loaded `Shell` holds retained Objective-C
// objects `objc2` doesn't mark `Send`, plus `RefCell`s written through `&self`. Sound
// because every verb that reaches the shell takes `&mut self`; `kind`/`device_facts`
// (the only `&self` verbs) never touch the shell.
unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}
