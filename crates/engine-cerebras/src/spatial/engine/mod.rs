//! The engine layer: PIE's `Engine` over the wafer program. It loads the
//! model (plans its trace, builds or finds the program, cuts or finds the
//! checkpoint's tiles, opens the program and places them), then serves
//! fires: each lane's rows through the program's one-sequence cache (brought
//! to the lane's history first), the guest passes in the host interpreter,
//! the lanes' readouts.

mod fire;
mod kv;
mod launch;
mod posted;

use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, PoolFacts};
use engine::chunk::{ChunkFacts, DegenerateBytes};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout};
use engine::load::{Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::MemoryDomain;
use eta_ir::registry::{GeometryClass, PortMask};

use kv::Kv;

use super::config::{Config, Debug, Target};
use super::facts::Facts;
use super::{prepare, wafer};

use crate::api::{ContractFor, refusal};

/// The spatial engine: the model forward in the wafer program, guest
/// programs on the host.
pub struct Spatial {
    programs: crate::program::Plane,
    cfg: Config,
    debug: Debug,
    contract_for: ContractFor,
    /// The trace's classes, inputs, readout and paging.
    facts: Option<Facts>,
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
    pub fn new(cfg: Config, contract_for: ContractFor) -> Spatial {
        Spatial {
            programs: crate::program::Plane::default(),
            debug: cfg.debug(),
            cfg,
            contract_for,
            facts: None,
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

/// Recurrent-state seats when the budgets state none.
const DEFAULT_STATE_SLOTS: u32 = 4;

impl Engine for Spatial {
    fn kind(&self) -> &'static str {
        "cerebras-spatial"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.facts.is_some() {
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
        // The program keeps one sequence's cache and every lane's as host
        // tokens: there is no device pool to back with a slot file or to
        // map out to another worker (CONTRACT-BC 22).
        if let Some(dir) = &residency.disk_kv_dir {
            return Err(Error::Load(format!(
                "the spatial engine has no disk kv tier (disk_kv_dir {} is stated)",
                dir.display()
            )));
        }
        if residency.kv_export {
            return Err(Error::Load(
                "the spatial engine exports no kv planes (kv_export is stated)".into(),
            ));
        }
        if residency.host_kv_budget.is_some_and(|bytes| bytes > 0) {
            tracing::warn!(
                "the spatial engine states no host kv tier: host_kv_budget is not used"
            );
        }
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the spatial engine loads a checkpoint path".into(),
            ));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;
        // the program serves the context the config states (else as many positions as it holds)
        let positions = budgets.context_declared.then_some(budgets.max_context);
        let prepared = prepare::prepare(&self.cfg, &trace, &contract, &path, positions)
            .map_err(|e| Error::Load(format!("the wafer program: {e}")))?;
        let slots = poem_compiler::state_slots(budgets.slots, &trace, DEFAULT_STATE_SLOTS);
        let facts = Facts::of(trace, &budgets, slots).map_err(Error::Load)?;
        self.readout = facts.readout();
        let platform = match &self.cfg.cmaddr {
            Some(cmaddr) => crate::sdk::Platform::System {
                cmaddr: cmaddr.clone(),
            },
            None => crate::sdk::Platform::Simulator(crate::sdk::Simulator {
                num_threads: self.cfg.num_threads,
                ..crate::sdk::Simulator::new(match self.cfg.target {
                    Target::Wse2 => crate::sdk::Target::Wse2,
                    Target::Wse3 => crate::sdk::Target::Wse3,
                })
            }),
        };
        let vocab = self
            .readout
            .as_ref()
            .map_or(usize::MAX, |(_, w)| *w as usize);
        let w = wafer::Wafer::open(
            prepared.program.clone(),
            platform,
            vocab,
            prepared.weights.clone(),
            self.cfg.toolchain.sdk_lib(),
            self.debug.force_prompt,
        )
        .map_err(|e| Error::Load(format!("the spatial wafer model: {e}")))?;
        tracing::info!(
            program = %prepared.program.display(),
            vocab = w.vocab,
            hidden = w.hidden,
            "spatial wafer model loaded"
        );
        // The context the model holds: the wafer program's positions.
        let model_context = u32::try_from(w.positions).unwrap_or(u32::MAX);
        self.model = Some(w);
        self.kv = Kv::default();

        let trace_name = facts.trace().name.clone();
        let paging = facts.paging();
        self.kv.page_size = paging.page_size;
        let state_rows = facts.has_state();
        let pools = PoolFacts {
            // The pages hold tokens here (the program's cache is replayed from them).
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            state_slots: if state_rows { paging.slots } else { 0 },
            // recurrent state replays from tokens: no buffered verbs
            buffer_pages: 0,
            state_slot_bytes: facts.state_slot_bytes(),
            adapter_banks: 0,
            ..PoolFacts::default()
        };
        // Today's pools as pinned chunks, one slot a chunk, device tier only:
        // a kv page is its tokens on the host (a 32-bit token a position),
        // never a device span the runtime could place, evict or stream.
        let chunks = ChunkFacts::degenerate(
            &pools,
            &DegenerateBytes {
                kv_page: u64::from(paging.page_size) * 4,
                buffer_page: 0,
            },
        );
        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cerebras".to_string(),
                domain: MemoryDomain::CerebrasDevice(0),
                sms: 1,
                unified_memory: false,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: None,
            },
            pools,
            chunks,
            limits: FireLimits {
                max_unfolded_rows: 0,
                // The model holds one sequence's cache; a fire's lanes run one
                // after another, each cache brought back from its tokens.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot,
                max_context: paging.context().min(model_context),
                prefill_rows: 0,
                prefill_lanes: 0,
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
            kv_handle: None,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: false,
            rs_snapshots: 0,
            bidirectional_attention: false,
            adapter_attachments: false,
            facts: facts.trace().facts.clone(),
            inputs: facts.input_decls(),
            classes: facts.classes(),
        };
        self.facts = Some(facts);
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
        let _ = copy;
        Err(Error::Unsupported {
            verb: "`copy_state` on the wafer program (its state replays from tokens)",
            engine: "cerebras-spatial",
        })
    }

    /// The load states its pools pinned (`ChunkFacts::degenerate`), so the
    /// runtime moves its kv through `copy_kv`; there is no shared pool and
    /// no host or disk tier to move slots between.
    fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        let _ = copy;
        Err(Error::Unsupported {
            verb: "`copy_slots` on the wafer program (its pools are pinned, device tier only: \
                   kv moves through `copy_kv`)",
            engine: "cerebras-spatial",
        })
    }

    fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> EngineResult<Vec<u8>> {
        let _ = slots;
        Err(Error::Unsupported {
            verb: "`read_tier_slots` on the wafer program (it states no host tier)",
            engine: "cerebras-spatial",
        })
    }

    fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> EngineResult<()> {
        let _ = (slots, bytes);
        Err(Error::Unsupported {
            verb: "`write_tier_slots` on the wafer program (it states no host tier)",
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
