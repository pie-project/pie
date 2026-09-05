use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, MemoryDomain, StateCopy};
use eta_ir::registry::{GeometryClass, ModelProfile, PortMask};
use eta_ir::types::Dtype;
use model_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use model_ir::Trace;

use crate::error::Fault;
use crate::serve::{Boot, Landed, Lane, Seated, Shell, StepView};
use crate::settle::Done;

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    pub adapter_index: u32,

    pub backends: Option<String>,

    pub gpu_mem_utilization: f64,

    pub power_preference: String,

    pub pipeline_cache: Option<PathBuf>,

    pub device_memory: Option<u64>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            adapter_index: 0,
            backends: None,
            gpu_mem_utilization: crate::boot::DEFAULT_GPU_MEM_UTILIZATION,
            power_preference: crate::boot::DEFAULT_POWER_PREFERENCE.to_string(),
            pipeline_cache: None,
            device_memory: None,
        }
    }
}

struct PendingStep {
    readout: Vec<Readout>,

    landed: Landed,

    rows: Option<Vec<Vec<f32>>>,
}

pub struct Wgpu {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,

    sink: Option<engine::CompletionSink>,

    pending: Option<(FrameId, Vec<PendingStep>)>,

    programs: crate::program::Plane,

    adapters: BTreeMap<InstanceId, crate::adapter::Binding>,
}

impl Wgpu {
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Wgpu {
        Wgpu {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            sink: None,
            pending: None,
            programs: crate::program::Plane::default(),
            adapters: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn boot(&self) -> &DeviceBoot {
        &self.boot
    }

    #[must_use]
    pub fn contract_for(&self) -> ContractFor {
        self.contract_for
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

    #[allow(dead_code)]
    fn loaded(&self) -> EngineResult<&Shell> {
        self.shell
            .as_ref()
            .ok_or_else(|| Error::Load("the wgpu engine has no model loaded".into()))
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the wgpu engine has no model loaded".into()))
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
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

fn refusal(refusal: crate::program::Refusal) -> Error {
    match refusal {
        crate::program::Refusal::Closed { what, id } => Error::Closed { what, id },
        crate::program::Refusal::Program(why) => Error::Program(why),
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
        Fault::Program { .. } => Error::Program(fault.to_string()),
        Fault::Wgpu { .. } | Fault::NoDevice { .. } => Error::Device(fault.to_string()),
        Fault::Fire(_) => Error::Invalid(fault.to_string()),

        Fault::Residency(_) => Error::Impossible(fault.to_string()),
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
    let vocab = u32::try_from(shell.out_width().map_err(fault)?).unwrap_or(u32::MAX);
    Ok(ModelProfile {
        vocab,
        page_size: budgets.page_size,
        num_layers: layers,

        activation: Dtype::F32,
        has_mtp_logits: shell.drafts(),
        mtp_depth: shell.mtp_depth(),
        has_value_head: false,
        has_attn_score: false,
        has_attn_page_mask: false,

        has_lora: true,
        kernels: Vec::new(),
    })
}

fn adapter_of(
    shell: &mut Shell,
    package: &eta_compiler::codegen::launch::LaunchPackage,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::adapter::Binding>> {
    let Some(sink) = crate::adapter::sink_of(package).map_err(fault)? else {
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
    let planes: Vec<crate::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| crate::AdapterPlane {
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

impl Engine for Wgpu {
    fn kind(&self) -> &'static str {
        "wgpu"
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
                "this wgpu engine already has a model loaded; one shell per engine".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,

            frames_in_flight: _,
        } = request;

        if ordinal > 0 {
            return Err(Error::unsupported("wgpu", "device ordinal selection"));
        }

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the wgpu shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        let patches = patch_ladder(&trace, &budgets);
        let shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget: bake_budgets(&budgets),
            patches,

            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            pages: budgets.pages,

            runahead: engine::runahead::Runahead::of(1),
            device: &self.boot,
            residency,
        })
        .map_err(fault)?;
        let weights_resident = !shell.weights_stream();

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();

        let paging = shell.paging();

        let state_rows = shell
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, model_ir::CacheRow::State { .. }));
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "wgpu".to_string(),
                domain: MemoryDomain::WgpuDevice(self.boot.adapter_index),
                sms: shell.cores(),
                unified_memory: false,

                fp8_native: false,
                native_mxfp4_moe: false,

                storage_alignment: 256,

                storage_max_tile_bytes: shell.max_buffer(),

                codegen_backend: None,
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: u32::try_from(shell.banks().len()).unwrap_or(u32::MAX),

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

            ports: PortMask::DECODE_ENVELOPE.with(eta_ir::registry::Port::RsFoldLen),
            geometry: GeometryClass::DecodeEnvelope,

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
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,

                weights_resident,
                weights_from_cache: false,
                arena_bytes,
                pool_bytes,
                input_bytes,

                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(engine::fire::Serves {
            device_channel_commit: false,
            rs_verbs: self.caps.as_ref().is_some_and(|caps| caps.rs_verbs),
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
            let fired = self
                .fire_step(step, at)
                .and_then(|(ticket, mut step_pending)| {
                    if !step.attachments.is_empty() {
                        let shell = self.loaded_mut()?;
                        let rows = shell.rows_of(&step_pending.landed).map_err(fault)?;

                        let drafts = shell.draft_rows_of(&step_pending.landed);
                        let mtp_width = shell.mtp_width();
                        step_pending.rows = Some(rows);
                        let landed = step_pending.landed;
                        self.run_attachments(
                            step,
                            &readouts_of(&step_pending),
                            drafts.as_deref(),
                            mtp_width,
                            &landed,
                        )?;
                    }
                    Ok((ticket, step_pending))
                });
            match fired {
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

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.programs.register(registration).map_err(refusal)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        self.programs
            .register_channel(registration)
            .map_err(refusal)
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
                "this load resolves {:?} on the device, so it binds at most {:?} and not {:?}",
                caps.ports, caps.geometry, binding.geometry
            )));
        }
        #[cfg(feature = "wgpu")]
        let device = self.shell.as_ref().map(crate::serve::Shell::device);
        let bound = self
            .programs
            .bind(
                binding,
                #[cfg(feature = "wgpu")]
                device,
            )
            .map_err(refusal)?;

        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        let landed = match self.programs.package(binding.program) {
            Some(package) => {
                let package = package.clone();
                match self.shell.as_mut() {
                    Some(shell) => adapter_of(shell, &package, bound.id, &seeds),
                    None => Ok(None),
                }
            }
            None => Ok(None),
        };
        match landed {
            Ok(Some(binding)) => {
                self.adapters.insert(bound.id, binding);
            }
            Ok(None) => {}
            Err(why) => {
                let _ = self.programs.close_instance(bound.id);
                return Err(why);
            }
        }
        Ok(bound)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        if let (Some(held), Some(shell)) = (self.adapters.remove(&id), self.shell.as_mut()) {
            shell.release_adapter(&held);
        }
        self.programs.close_instance(id).map_err(refusal)
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
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

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.programs.close_channel(id).map_err(refusal)
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.programs
            .publish(instance, channel, cell)
            .map_err(refusal)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.programs.take(instance, channel).map_err(refusal)
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
                refused = Some(Error::Load("the wgpu engine has no model loaded".into()));
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
        let (_, pending) = self.pending.as_ref().expect("just put back");
        for (receipt, step) in ticket.steps.iter_mut().zip(pending) {
            receipt.readouts = readouts_of(step);
        }
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            let _ = shell.reap();
        }
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
                engine: "wgpu",
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
                    "state move {at} names a token offset; this engine moves whole slots"
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
}

impl Wgpu {
    fn run_attachments(
        &mut self,
        step: &Step,
        readouts: &[LaneReadout],
        drafts: Option<&[Vec<f32>]>,
        mtp_width: u32,
        landed: &crate::serve::Landed,
    ) -> EngineResult<()> {
        let Wgpu {
            shell, programs, ..
        } = self;
        for attachment in &step.attachments {
            let readout = readouts.get(attachment.lane as usize);
            let (logits, rows, vocab) = match readout {
                Some(lane) if lane.rows > 0 => {
                    (Some(lane.values.as_slice()), lane.rows, lane.width)
                }
                _ => (None, 0, 0),
            };

            let mtp_logits = drafts
                .and_then(|lanes| lanes.get(attachment.lane as usize))
                .filter(|values| !values.is_empty() && mtp_width > 0)
                .map(Vec::as_slice);

            #[cfg(feature = "wgpu")]
            let (device, seat) = match shell.as_ref() {
                Some(shell) => (
                    Some(shell.device()),
                    shell
                        .readout_row(landed, attachment.lane)
                        .map(|(seat, at, width)| crate::guest::run::Readout { seat, at, width }),
                ),
                None => (None, None),
            };
            programs
                .fire(
                    attachment.instance,
                    logits,
                    rows,
                    vocab,
                    mtp_logits,
                    mtp_width,
                    #[cfg(feature = "wgpu")]
                    device,
                    #[cfg(feature = "wgpu")]
                    seat,
                )
                .map_err(refusal)?;
        }
        if let Some(shell) = shell.as_mut() {
            shell.forget_seat(landed);
        }
        Ok(())
    }

    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
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
                        engine: "wgpu",
                    }
                })?);
            }
        }
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

        let mut resolved: Vec<Option<Vec<u32>>> = vec![None; submission.lanes.len()];
        for attachment in &submission.attachments {
            let Some(lane) = submission.lanes.get(attachment.lane as usize) else {
                return Err(Error::Invalid(format!(
                    "attachment names lane {} of a {}-lane step",
                    attachment.lane,
                    submission.lanes.len()
                )));
            };
            resolved[attachment.lane as usize] = self
                .programs
                .envelope_tokens(attachment.instance, lane.tokens.len())
                .map_err(refusal)?;
        }

        let mut verbs: Vec<engine::fire::RsVerb> = submission
            .lanes
            .iter()
            .map(|lane| lane.rs.clone())
            .collect();
        for attachment in &submission.attachments {
            let verb = &mut verbs[attachment.lane as usize];
            let len = match verb {
                engine::fire::RsVerb::Buffer { fold, .. }
                | engine::fire::RsVerb::Window { fold, .. } => fold,
                engine::fire::RsVerb::FoldBuffered { len, .. } => len,
                engine::fire::RsVerb::Fold => continue,
            };
            if matches!(len, engine::fire::FoldLen::Device(_))
                && let Some(n) = self
                    .programs
                    .envelope_fold_len(attachment.instance)
                    .map_err(refusal)?
            {
                *len = engine::fire::FoldLen::Host(n);
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
                        tokens: resolved[at].as_deref().unwrap_or(&lane.tokens),
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    captures_scores: lane.captures_scores,

                    mask: lane.mask.as_ref(),

                    adapter: lane_adapters[at].or(lane.adapter),
                    positions: &lane.positions,

                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },

                    translation: &lane.kv.translation,

                    rs: &verbs[at],
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
                    attachments: &[],
                    media: &media,
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

fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let rows: &[Vec<f32>] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| {
            let count = match policy {
                Readout::None => return LaneReadout::default(),
                Readout::Last => 1,
                Readout::Rows(list) => list.len().max(1),
            };
            let values = rows.get(lane).cloned().unwrap_or_default();
            LaneReadout {
                rows: u32::try_from(count).unwrap_or(u32::MAX),
                width: u32::try_from(values.len() / count).unwrap_or(u32::MAX),
                values,
                ..LaneReadout::default()
            }
        })
        .collect()
}

unsafe impl Send for Wgpu {}
unsafe impl Sync for Wgpu {}
