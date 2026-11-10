//! The engine layer: PIE's `Engine` over the wafer program. It loads the
//! model (opens the compiled program, places the prep's tiles), then serves
//! fires: each lane's rows through the program's one-sequence cache (brought
//! to the lane's history first), the guest passes in the host interpreter,
//! the lanes' readouts.

mod fire;
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
use kv::Kv;

use super::{artifact, wafer};

use crate::api::{ContractFor, DeviceBoot, bake_budgets, fault, patch_ladder, refusal};
use crate::serve::{Boot, Shell};

/// The spatial engine: the model forward in the wafer program, guest
/// programs on the host.
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
    /// Fires run (the model-time log's count).
    fires: u64,
    /// Rows of history replayed into the model's cache.
    replayed: u64,
    /// Greedy rows a token loop answered, and the loops run.
    ahead_hits: u64,
    ahead_runs: u64,
    /// The posted token loop on the wafer: the token it was fed and its
    /// position.
    loop_posted: Option<(u32, usize)>,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<Vec<LaneReadout>>)>,
    /// Steps posted and not yet run (an asynchronous engine).
    posted: std::collections::VecDeque<posted::Posted>,
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
            fires: 0,
            replayed: 0,
            ahead_hits: 0,
            ahead_runs: 0,
            loop_posted: None,
            next_fire: 0,
            next_frame: 0,
            sink: None,
            pending: None,
            posted: std::collections::VecDeque::new(),
        }
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
        // The wafer program: compiled outside the engine, its artifact beside it.
        let program = Path::new(&self.spatial.program);
        if !artifact::is_artifact_program(program) {
            return Err(Error::Load(format!(
                "{}: not a compiled wafer program (no artifact/program.json): wafer-compile \
                 build --plan <plan.json> --out <dir>, then link.sh <dir>",
                program.display()
            )));
        }
        let platform = match &self.boot.cmaddr {
            Some(cmaddr) => crate::sdk::Platform::System {
                cmaddr: cmaddr.clone(),
            },
            None => crate::sdk::Platform::Simulator(crate::sdk::Simulator {
                num_threads: self.boot.num_threads,
                ..crate::sdk::Simulator::new(self.boot.target)
            }),
        };
        let vocab = self
            .readout
            .as_ref()
            .map_or(usize::MAX, |(_, w)| *w as usize);
        let w = wafer::Wafer::open(program.to_path_buf(), platform, vocab)
            .map_err(|e| Error::Load(format!("the spatial wafer model: {e}")))?;
        tracing::info!(
            program = %program.display(),
            vocab = w.vocab,
            hidden = w.hidden,
            "spatial wafer model loaded"
        );
        // The context the model holds: the wafer program's positions.
        let model_context = u32::try_from(w.positions).unwrap_or(u32::MAX);
        self.model = Some(w);
        self.kv = Kv::default();

        let trace_name = shell.trace().name.clone();
        let paging = shell.paging();
        self.kv.page_size = paging.page_size;
        let state_rows = shell.has_state();
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
                // The pages hold tokens here (the program's cache is replayed from them).
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                buffer_pages: shell.buffer_pages(),
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: 0,
                ..PoolFacts::default()
            },
            limits: FireLimits {
                // The model holds one sequence's cache; a fire's lanes run one
                // after another, each cache brought back from its tokens.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot,
                max_context: paging.context().min(model_context),
                prefill_rows: 0,
                max_readouts: budgets.max_tokens,
            },
            has_lora: false,
            has_attn_page_mask: false,
            draft_block: 0,
            draft_mask_token: 0,
            draft_proposals_from: 1,
            // One sequence a cache runs causal attention only: no mask port.
            ports: PortMask::DEVICE_GEOMETRY,
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                // Pages are tokens here: a copy copies them.
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
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

    /// Recurrent states replay from their tokens (the program holds one
    /// sequence's): a copy is not a device move here.
    fn copy_state(&mut self, copy: &engine::transfer::StateCopy) -> EngineResult<()> {
        copy.device_only("cerebras-spatial")?;
        Err(Error::Unsupported {
            verb: "`copy_state` on the wafer program (its state replays from tokens)",
            engine: "cerebras-spatial",
        })
    }

    fn copy_kv(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let t0 = std::time::Instant::now();
        self.drain();
        tracing::info!(
            pages = copy.src_page_ids.len(),
            drain_ms = t0.elapsed().as_secs_f64() * 1e3,
            "spatial: copy_kv"
        );
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            tracing::debug!(target: "spatial_trace", "spatial-trace copy_kv: {copy:?}");
        }
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        if Some(copy.src) != device || Some(copy.dst) != device {
            return Err(Error::Unsupported {
                verb: "`copy_kv` between domains other than this load's device",
                engine: "cerebras-spatial",
            });
        }
        let size = self.kv.page_size.max(1);
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
        self.programs.bind(binding).map_err(refusal)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.drain();
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
