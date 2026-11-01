//! The engine layer: PIE's `Engine` over a wafer program. It loads the
//! model (plans or reads its program, opens the wafer), then serves fires:
//! batching lanes into launches, keeping lanes' KV pages (and the host
//! tier), routing epilogues to the host interpreter or the wafer's VM.
//! Everything about the program itself (its shape, symbols, tables, page
//! layout) comes from the program layer ([`super::program`],
//! [`super::component`]) through the driver ([`super::wafer`]).

mod fire;
mod guest_vm;
mod kv;
mod launch;
mod posted;

use std::path::Path;

use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout};
use engine::load::{Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::MemoryDomain;
use eta_ir::registry::{GeometryClass, PortMask};

use super::SpatialBoot;
use fire::PagedAhead;
use kv::Kv;

use super::{build, ckpt, guest4, model, wafer};

use crate::api::{ContractFor, DeviceBoot, bake_budgets, fault, patch_ladder, refusal};
use crate::serve::{Boot, Shell};

/// The spatial engine: the model forward in the wafer program,
/// guest programs on the host (or the wafer's VM), PIE's KV pages.
pub struct Spatial {
    programs: crate::program::Plane,
    boot: DeviceBoot,
    spatial: SpatialBoot,
    contract_for: ContractFor,
    /// A dry shell: the trace, its classes, paging and readout, no device.
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    readout: Option<(String, u32)>,
    model: Option<wafer::Wafer>,
    kv: Kv,
    /// Programs whose guest translation was told, and those the guest VM
    /// can run, by program id.
    guest_probed: std::collections::HashSet<u64>,
    guest_programs: std::collections::HashMap<u64, std::sync::Arc<guest4::lane::Program>>,
    /// Programs the VM could take that the cost forecast keeps on the host,
    /// and why.
    guest_costly: std::collections::HashMap<u64, String>,
    /// The model's parameter count (the forecast's default token time).
    params: f64,
    /// The guest VM a lane's instance runs on, while it does.
    guest_vms: std::collections::HashMap<InstanceId, guest4::lane::CheckVm>,
    /// The instance whose program the wafer head's VM holds (`guest`
    /// "wafer"): its homes live there while it does.
    vm_owner: Option<InstanceId>,
    /// A multi-lane VM: the program it holds and each lane's request.
    vm_program: Option<u64>,
    vm_banks: Vec<Option<InstanceId>>,
    /// A per-lane-program VM: each lane's program as placed (empty until
    /// the first placement).
    vm_bank_progs: Vec<Option<u64>>,
    /// Each bank's last fire (`paged_fires`): who an eviction takes first.
    vm_bank_used: Vec<u64>,
    /// Evicted instances' bank windows, word for word, until they place
    /// again (or are gone).
    vm_parked: std::collections::HashMap<InstanceId, guest4::slots::Parked>,
    /// Fires run (the model-time log's count).
    fires: u64,
    /// Epilogue steps the guest VM ran.
    guest_steps: u64,
    /// Rows of history replayed into the model's cache (the fallback for a
    /// model that owns no paged KV), and fires run on paged KV.
    replayed: u64,
    paged_fires: u64,
    /// Rows re-prefilled into a lane's own pages.
    refilled: u64,
    /// Greedy rows a token loop answered, and the loops run.
    ahead_hits: u64,
    ahead_runs: u64,
    /// A paged lane's token loop run ahead on the wafer, by sequence slot:
    /// the slot its next row writes, then each row's expected token and the
    /// argmax it already has (its K/V written on the lane's own pages).
    paged_ahead: std::collections::HashMap<u32, PagedAhead>,
    /// The token loop's answer row, reused fire to fire.
    one_hot: fire::OneHotRow,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<Vec<LaneReadout>>)>,
    /// Steps posted and not yet run (an asynchronous engine), the token
    /// loop continuation on the wafer, the steps the wafer's thread already
    /// reported landed, and the loops started (each loop's epoch).
    posted: std::collections::VecDeque<posted::Posted>,
    in_flight: Option<posted::InFlight>,
    reported: std::collections::HashSet<(u64, u32)>,
    next_loop: u64,
}

impl Spatial {
    #[must_use]
    pub fn new(boot: DeviceBoot, spatial: SpatialBoot, contract_for: ContractFor) -> Spatial {
        Spatial {
            programs: crate::program::Plane::default(),
            boot,
            spatial,
            contract_for,
            shell: None,
            caps: None,
            readout: None,
            model: None,
            kv: Kv::default(),
            guest_probed: std::collections::HashSet::new(),
            guest_programs: std::collections::HashMap::new(),
            guest_costly: std::collections::HashMap::new(),
            params: 0.0,
            guest_vms: std::collections::HashMap::new(),
            vm_owner: None,
            vm_program: None,
            vm_banks: Vec::new(),
            vm_bank_progs: Vec::new(),
            vm_bank_used: Vec::new(),
            vm_parked: std::collections::HashMap::new(),
            fires: 0,
            guest_steps: 0,
            replayed: 0,
            paged_fires: 0,
            refilled: 0,
            ahead_hits: 0,
            ahead_runs: 0,
            paged_ahead: std::collections::HashMap::new(),
            one_hot: fire::OneHotRow::default(),
            next_fire: 0,
            next_frame: 0,
            sink: None,
            pending: None,
            posted: std::collections::VecDeque::new(),
            in_flight: None,
            reported: std::collections::HashSet::new(),
            next_loop: 0,
        }
    }

    /// The wafer program's directory: named (`program`), or built from the
    /// model's plan (`serve`); checked against the model (dims, layers,
    /// vocabulary, and what the placement assumes: the head, eps, q/k
    /// norms, theta).
    fn program_dir(&self, spec: &model::ModelSpec, page_size: usize) -> EngineResult<String> {
        let program = match &self.spatial.program {
            Some(dir) => Some(dir.clone()),
            None => match &self.spatial.serve {
                // PIE builds the program from the model's plan.
                Some(t) => {
                    let cfg = build::ServeConfig::from_toml(t, page_size)
                        .map_err(|e| Error::Load(e.to_string()))?;
                    Some(
                        build::build_for(spec, &cfg, self.boot.target, self.boot.cmaddr.is_some())
                            .map_err(|e| Error::Load(format!("building the spatial program: {e}")))?
                            .display()
                            .to_string(),
                    )
                }
                None => None,
            },
        };
        let Some(program) = program else {
            return Err(Error::Load(
                "[cerebras.spatial] names no wafer program (`program` or `serve`)".into(),
            ));
        };
        let out_json = Path::new(&program).join("out.json");
        let json: serde_json::Value = std::fs::read_to_string(&out_json)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .ok_or_else(|| Error::Load(format!("{}: no readable out.json", out_json.display())))?;
        spec.check_program(model::out_json_params(&json))
            .map_err(|e| Error::Load(e.to_string()))?;
        Ok(program)
    }

    fn model(&mut self) -> EngineResult<&mut wafer::Wafer> {
        self.model
            .as_mut()
            .ok_or_else(|| Error::Load("the spatial engine has no model loaded".into()))
    }
}

impl Engine for Spatial {
    fn kind(&self) -> &'static str {
        "cerebras-spatial"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this spatial engine already has a model loaded".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal: _,
            frames_in_flight: _,
        } = request;
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the spatial engine loads a checkpoint path".into(),
            ));
        };
        // The dry shell reads neither contract nor checkpoint; it wants one.
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;
        let patches = patch_ladder(&trace, &budgets);
        let voxels = poem_compiler::VoxelLadder::of(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        let slots =
            poem_compiler::state_slots(budgets.slots, &trace, crate::api::DEFAULT_STATE_SLOTS);
        // The model as the spatial kernels see it: refused here unless its
        // structure is one they implement (spatial/plan.rs).
        let spec = model::ModelSpec::read(&trace).map_err(|e| Error::Load(e.to_string()))?;
        // The weights land by the trace's params (the shell takes the trace).
        let weights_trace = trace.clone();
        self.params = {
            let (h, hd) = (spec.hidden as f64, spec.head_dim as f64);
            let attn = h * hd * (2 * spec.q_heads + 2 * spec.kv_heads) as f64;
            let heads = if spec.tied { 1.0 } else { 2.0 };
            spec.layers as f64 * (attn + 3.0 * h * spec.inter as f64)
                + heads * spec.vocab as f64 * h
        };
        let shell = Shell::load_dry(
            Boot {
                patches,
                trace,
                contract: &contract,
                checkpoint: &path,
                budget: bake_budgets(&budgets),
                page_size: budgets.page_size,
                context: budgets.max_context,
                slots,
                kv_budget: residency.device_kv_budget,
                memory_share: residency.device_memory_share,
                device: &self.boot,
            },
            voxels,
            false,
        )
        .map_err(fault)?;
        self.readout = shell.readout_output();
        // The wafer program: compiled, built from the plan, or named.
        let ckpt = self
            .spatial
            .ckpt
            .clone()
            .unwrap_or_else(|| Path::new(&path).display().to_string());
        let program = &self.program_dir(&spec, budgets.page_size as usize)?;
        let platform = match &self.boot.cmaddr {
            Some(cmaddr) => crate::sdk::Platform::System {
                cmaddr: cmaddr.clone(),
            },
            None => crate::sdk::Platform::Simulator(crate::sdk::Simulator {
                num_threads: self.boot.num_threads,
                ..crate::sdk::Simulator::new(self.boot.target)
            }),
        };
        let theta = Some(f64::from(spec.rope_theta));
        // The weights through PIE's checkpoint path (the load's contract
        // over its checkpoint); `ckpt` names a Hugging Face directory
        // instead.
        let weights = match &self.spatial.ckpt {
            Some(dir) => ckpt::Ckpt::open(Path::new(dir)),
            None => ckpt::Ckpt::landed(&weights_trace, &contract, Path::new(&path)),
        }
        .map_err(|e| Error::Load(format!("the spatial wafer model's weights: {e}")))?;
        let w = wafer::Wafer::open(program.into(), weights, platform, theta)
            .map_err(|e| Error::Load(format!("the spatial wafer model: {e}")))?;
        tracing::info!(
            program,
            ckpt,
            vocab = w.vocab,
            hidden = w.hidden,
            "spatial wafer model loaded"
        );
        // A batched build owns the KV as PIE's pages.
        let paged_model = w.paged.is_some();
        self.model = Some(w);
        self.kv = Kv {
            paged: paged_model,
            ..Kv::default()
        };

        let trace_name = shell.trace().name.clone();
        let paging = shell.paging();
        self.kv.page_size = paging.page_size;
        // A batched wafer build's pool is PIE's: the same page size, and at
        // least the pages PIE hands out.
        if let Some(w) = &self.model
            && let Some(shape) = w.paged
            && (paging.page_size as usize != shape.page_size
                || paging.pages() > shape.pages() as u64)
        {
            return Err(Error::Load(format!(
                "the wafer program's KV pool is {} pages of {}; this load pages {} of {} \
                     (set the page size and the KV budget to fit it)",
                shape.pages(),
                shape.page_size,
                paging.pages(),
                paging.page_size
            )));
        }
        // The host tier: `host_kv_budget` of whole pages, on a wafer build
        // whose pages the host can read and write.
        self.kv.host_pages = match &self.model {
            Some(w) => match (w.paged, w.page_bytes, residency.host_kv_budget) {
                (Some(_), Some(bytes), Some(budget)) if bytes > 0 => {
                    u32::try_from(budget / bytes as u64).unwrap_or(u32::MAX)
                }
                _ => 0,
            },
            _ => 0,
        };
        if self.kv.host_pages > 0 {
            tracing::info!(
                pages = self.kv.host_pages,
                "spatial: the host keeps suspended lanes' kv pages"
            );
        }
        let state_rows = shell.has_state();
        let wafer_pool = self.model.as_ref().and_then(|w| w.paged);
        // The context the model holds: the wafer program's positions (a
        // lane's pages on a batched build).
        let model_context = self
            .model
            .as_ref()
            .map_or(u32::MAX, |w| u32::try_from(w.positions).unwrap_or(u32::MAX));
        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cerebras".to_string(),
                domain: MemoryDomain::CerebrasDevice(self.boot.ordinal),
                sms: 1,
                unified_memory: false,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: None,
            },
            pools: PoolFacts {
                // A batched wafer build reads at most E of a lane's pages on a
                // row: pages below rows x E fill every row E times over, so no
                // lane can name more.
                kv_pages: u32::try_from(paging.pages())
                    .unwrap_or(u32::MAX)
                    .min(wafer_pool.map_or(u32::MAX, |s| (s.rows * s.entries) as u32)),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                buffer_pages: shell.buffer_pages(),
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: 0,
                host_kv_pages: self.kv.host_pages,
                ..PoolFacts::default()
            },
            limits: FireLimits {
                // The model holds one sequence's cache; a fire's lanes run one
                // after another, each cache brought back from its tokens.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging
                    .pages_per_slot
                    .min(wafer_pool.map_or(u32::MAX, |s| s.max_page_refs() as u32)),
                max_context: paging.context().min(model_context),
                prefill_rows: 0,
                max_readouts: budgets.max_tokens,
            },
            has_lora: false,
            has_attn_page_mask: false,
            rotates_by_table: false,
            draft_block: 0,
            draft_mask_token: 0,
            draft_proposals_from: 1,
            // A paged model attends what each row's mask says; a model of
            // one sequence a cache runs causal attention only, so it takes
            // no mask port.
            ports: if paged_model {
                PortMask::DEVICE_GEOMETRY.with(eta_ir::registry::Port::AttnMask)
            } else {
                PortMask::DEVICE_GEOMETRY
            },
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                // Pages are tokens here: a copy copies them. A wafer build
                // whose pages the host can read parks a suspended lane's
                // pages on the host (never attended there).
                device_to_device: true,
                device_to_host: self.kv.host_pages > 0,
                host_to_device: self.kv.host_pages > 0,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: shell.serves_rs_verbs(),
            rs_snapshots: 0,
            bidirectional_attention: false,
            adapter_attachments: false,
            facts: shell.trace().facts.clone(),
            inputs: shell.input_decls(),
            classes: shell.classes(),
        };
        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes: 0,
                weights_resident: true,
                weights_from_cache: false,
                arena_bytes: 0,
                pool_bytes: 0,
                input_bytes: 0,
                pool_committed_bytes: 0,
                pool_high_water_bytes: 0,
            },
            caps,
        })
    }

    fn copy_kv(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        copy.validate()?;
        self.drain();
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            tracing::debug!(target: "spatial_trace", "spatial-trace copy_kv: {copy:?}");
        }
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        let host = engine::transfer::MemoryDomain::HostPinned;
        if self.kv.host_pages > 0 && Some(copy.src) == device && copy.dst == host {
            return self.page_out(copy);
        }
        if self.kv.host_pages > 0 && copy.src == host && Some(copy.dst) == device {
            return self.page_in(copy);
        }
        if Some(copy.src) != device || Some(copy.dst) != device {
            return Err(Error::Unsupported {
                verb: "`copy_kv` between domains other than this load's device",
                engine: "cerebras-spatial",
            });
        }
        let size = self.kv.page_size.max(1);
        if self.model.as_ref().is_some_and(|w| w.paged.is_some()) {
            let runs = copy.runs(size)?;
            launch::copy_paged(self.model()?, &runs)?;
        }
        for run in copy.runs(size)? {
            let src: Vec<Option<u32>> = self
                .kv
                .pages
                .get(&run.src_page)
                .cloned()
                .unwrap_or_else(|| vec![None; size as usize]);
            let dst = self
                .kv
                .pages
                .entry(run.dst_page)
                .or_insert_with(|| vec![None; size as usize]);
            self.kv.copied.insert(run.dst_page);
            for k in 0..run.tokens as usize {
                if let (Some(&cell), Some(slot)) = (
                    src.get(run.src_token as usize + k),
                    dst.get_mut(run.dst_token as usize + k),
                ) {
                    *slot = cell;
                }
            }
        }
        Ok(())
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
                "this load binds at most {:?} and not {:?}",
                caps.geometry, binding.geometry
            )));
        }
        let bound = self.programs.bind(binding).map_err(refusal)?;
        self.probe_guest(bound.id, bound.program);
        Ok(bound)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.drain();
        self.guest_vms.remove(&id);
        self.programs.close_instance(id).map_err(refusal)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.drain();
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

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn settles_asynchronously(&self) -> bool {
        self.asynchronous()
    }

    fn publish_landed(&mut self) -> EngineResult<()> {
        self.pump();
        Ok(())
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::fire::Serves::NONE, engine::fire::Serves::of),
        )?;
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            tracing::debug!(target: "spatial_trace",
                "spatial-trace frame: {} step(s), attachments {:?}",
                frame.steps.len(),
                frame.steps.first().map(|s| s.attachments.len())
            );
        }
        if self.settling() {
            posted::post(&mut self.posted, id, &frame.steps);
            self.pump();
            // Readouts come on the lanes' channels: a step lands when its
            // epilogue has run.
            return Ok(FrameTicket {
                id,
                steps: frame
                    .steps
                    .iter()
                    .map(|_| FireTicket {
                        id: FireId(self.next_fire),
                        readouts: Vec::new(),
                    })
                    .collect(),
            });
        }
        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut readouts = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            let read = self.fire_step(step)?;
            if let Some(sink) = &self.sink {
                sink(
                    engine::StepDone {
                        frame: id,
                        step: index as u32,
                    },
                    engine::StepOutcome::Committed,
                );
            }
            steps.push(FireTicket {
                id: FireId(self.next_fire),
                readouts: read.clone(),
            });
            readouts.push(read);
        }
        self.pending = Some((id, readouts));
        Ok(FrameTicket { id, steps })
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        if let Some((id, readouts)) = &self.pending
            && *id == ticket.id
        {
            for (receipt, read) in ticket.steps.iter_mut().zip(readouts) {
                receipt.readouts = read.clone();
            }
        }
        Ok(())
    }
}
