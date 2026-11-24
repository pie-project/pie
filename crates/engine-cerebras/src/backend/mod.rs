//! PIE's `Engine` over the wafer program. It loads the model (plans its trace, builds or finds
//! the program and the checkpoint's tiles, places them), then serves fires: each lane's rows
//! through the program's cache, the guest passes in the host interpreter, the lanes' readouts.

mod fire;
mod kv;
mod launch;
mod posted;
mod stream;

use engine::caps::{Capabilities, DeviceFacts, FireLimits, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::chunk::{ChunkFacts, DegenerateBytes};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout};
use engine::load::{Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{DeviceReach, MemoryDomain};
use engine::{Channels, Completion, Engine, Programs, Tiers};
use eta_ir::registry::{GeometryClass, PortMask};

use kv::Kv;

use crate::config::{Config, Debug, Target};
use crate::facts::Facts;
use crate::{prepare, wafer};

use crate::ContractFor;
use crate::guest::refusal;

/// The Cerebras engine: the model forward in the wafer program, guest programs on the host.
pub struct CerebrasEngine {
    programs: crate::guest::Plane,
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
    /// The posted token loop on the wafer: the token it was fed and its position.
    loop_posted: Option<(u32, usize)>,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<Vec<LaneReadout>>)>,
    /// Steps posted and not yet run (an asynchronous engine).
    posted: std::collections::VecDeque<posted::Posted>,
    /// A stream reading's program (a diffusion transformer's denoise), when the load is one.
    stream: Option<stream::Streams>,
}

impl CerebrasEngine {
    #[must_use]
    pub fn new(cfg: Config, contract_for: ContractFor) -> CerebrasEngine {
        CerebrasEngine {
            programs: crate::guest::Plane::default(),
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
            stream: None,
        }
    }

    /// The system the program runs on: the configured CS system, else the simulator.
    fn platform(&self) -> EngineResult<crate::sdk::Platform> {
        Ok(match &self.cfg.cmaddr {
            Some(cmaddr) => crate::sdk::Platform::System {
                cmaddr: cmaddr.clone(),
            },
            None => {
                let threads = match self.cfg.num_threads {
                    Some(n) => n,
                    None => crate::sdk::host_threads().map_err(Error::Load)?,
                };
                crate::sdk::Platform::Simulator(crate::sdk::Simulator::new(
                    match self.cfg.target {
                        Target::Wse2 => crate::sdk::Target::Wse2,
                        Target::Wse3 => crate::sdk::Target::Wse3,
                    },
                    threads,
                ))
            }
        })
    }

    fn model(&mut self) -> EngineResult<&mut wafer::Wafer> {
        self.model
            .as_mut()
            .ok_or_else(|| Error::Load("the cerebras engine has no model loaded".into()))
    }
}

/// What Pie's scheduler is told of the device (every load alike): one compute unit, host pools
/// (pages hold tokens, recurrent state replays from them), no device alignment or tile bound.
fn device_facts() -> DeviceFacts {
    DeviceFacts {
        backend: "cerebras".to_string(),
        domain: MemoryDomain::Device {
            ordinal: 0,
            reach: DeviceReach::Private,
        },
        sms: 1,
        unified_memory: false,
        fp8_native: false,
        native_mxfp4_moe: false,
        storage_alignment: 256,
        storage_max_tile_bytes: u64::MAX,
        codegen_backend: None,
    }
}

/// Recurrent-state seats when the budgets state none: one a lane the budgets run together.
fn default_state_slots(budgets: &engine::load::Budgets) -> u32 {
    budgets.max_lanes.max(1)
}

impl Engine for CerebrasEngine {
    fn kind(&self) -> &'static str {
        "cerebras-spatial"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn disk_kv_tier(&self) -> bool {
        false
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.facts.is_some() {
            return Err(Error::Load(
                "this cerebras engine already has a model loaded".into(),
            ));
        }
        crate::chip::calibrate().map_err(Error::Load)?;
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal: _,
            frames_in_flight: _,
        } = request;
        // The program's cache is replayed from host tokens: there is no device pool to back with a
        // slot file or map out to another worker.
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
            tracing::warn!("the spatial engine states no host kv tier: host_kv_budget is not used");
        }
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the cerebras engine loads a checkpoint path".into(),
            ));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;
        // a stream reading's program (its host.json), or a directory of them, one a reading
        if let Some(programs) = self
            .debug
            .program
            .as_deref()
            .and_then(|d| stream::programs_in(d, self.debug.link.as_deref()))
        {
            return self.load_stream(trace, &path, &contract, &budgets, &programs);
        }
        // the program serves the context the config states (else as many positions as it holds)
        let positions = budgets.context_declared.then_some(budgets.max_context);
        let slots =
            poem_compiler::state_slots(budgets.slots, &trace, default_state_slots(&budgets));
        let platform = self.platform()?;
        let w = match prepare::prepare(&self.cfg, &trace, &contract, &path, positions) {
            Ok(prepared) => {
                let facts = Facts::of(trace.clone(), &budgets, slots).map_err(Error::Load)?;
                let vocab = facts
                    .readout()
                    .as_ref()
                    .map_or(usize::MAX, |(_, w)| *w as usize);
                let w = wafer::Wafer::open(
                    prepared.program.clone(),
                    platform,
                    vocab,
                    prepared.weights.clone(),
                    self.cfg.toolchain.sdk_lib(),
                )
                .map_err(|e| Error::Load(format!("the wafer model: {e}")))?;
                // a program without the token loop serves only with the lookahead off: refused here
                if !w.token_loop
                    && self
                        .cfg
                        .lookahead
                        .unwrap_or(fire::default_lookahead(w.posted_loop))
                        > 0
                {
                    return Err(Error::Load(format!(
                        "{}: built without the token loop (a WL build), and the lookahead is on \
                         ([cerebras] lookahead = 0 serves it without one)",
                        prepared.program.display()
                    )));
                }
                tracing::info!(
                    program = %prepared.program.display(),
                    vocab = w.vocab,
                    hidden = w.hidden,
                    "spatial wafer model loaded"
                );
                w
            }
            Err(e) => {
                let host = self.debug.stream_fabric.as_deref() == Some("host");
                let declared = trace.facts.declared("reading").is_some();
                // a model the decode templates do not run: its decode step as stream units
                let units = if declared || self.debug.program.is_some() || host {
                    None
                } else {
                    match prepare::unit_program(
                        &self.cfg,
                        &trace,
                        positions,
                        prepare::Lanes::Upto(budgets.max_lanes.max(1)),
                    ) {
                        Ok(p) => Some(p),
                        Err(u) => {
                            tracing::warn!(templates = %e, units = %u, "cerebras: no decode program");
                            None
                        }
                    }
                };
                if let Some((program, link, held)) = units {
                    tracing::info!(decode = %e, "cerebras: the decode templates refuse; the decode step runs as stream units");
                    let w = wafer::Wafer::open_units(
                        program.clone(),
                        link,
                        platform,
                        self.cfg.toolchain.sdk_lib(),
                        trace.clone(),
                        contract.clone(),
                        path.clone(),
                        held as usize,
                    )
                    .map_err(|e| Error::Load(format!("the unit decode program: {e}")))?;
                    tracing::info!(program = %program.display(), vocab = w.vocab, positions = w.positions, "spatial unit decode model loaded");
                    w
                } else {
                    // a model the decode program does not run: its stream readings, compiled here
                    let streamed = if self.debug.program.is_none() {
                        prepare::stream_programs(&self.cfg, &trace, host, budgets.max_voxels)
                            .map_err(Error::Load)?
                    } else {
                        None
                    };
                    if let Some(programs) = streamed {
                        tracing::info!(
                            decode = %e,
                            "cerebras: no decode program; the model's stream readings are served"
                        );
                        return self.load_stream(trace, &path, &contract, &budgets, &programs);
                    }
                    return Err(Error::Load(format!("the wafer program: {e}")));
                }
            }
        };
        let facts = Facts::of(trace, &budgets, slots).map_err(Error::Load)?;
        self.readout = facts.readout();
        // The context the model holds: the wafer program's positions.
        let model_context = u32::try_from(w.positions).unwrap_or(u32::MAX);
        // a unit program's pages are Pie's (its K / V columns' own, a lane's ids its pages)
        let wafer_pages = w.paging;
        // (a unit program's state slots its lane slots)
        let wafer_lanes = w.lanes;
        let posted_loop = w.posted_loop;
        self.model = Some(w);
        self.kv = Kv::default();
        let lookahead = self.lookahead();
        let dispatch_depth = fire::dispatch_depth(posted_loop, lookahead);
        tracing::info!(
            posted_loop,
            lookahead,
            dispatch_depth,
            "cerebras: posted loops"
        );

        let trace_name = facts.trace().name.clone();
        let paging = facts.paging();
        self.kv.page_size = paging.page_size;
        let state_rows = facts.has_state();
        let pools = PoolFacts {
            // The pages hold tokens here (the program's cache is replayed from them).
            kv_pages: wafer_pages.map_or(
                u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                |(_, pages, _)| pages,
            ),
            kv_page_size: wafer_pages.map_or(paging.page_size, |(_, _, size)| size),
            state_slots: match (state_rows, wafer_lanes) {
                (false, _) => 0,
                (true, Some(b)) => u32::try_from(b).unwrap_or(u32::MAX),
                (true, None) => paging.slots,
            },
            // recurrent state replays from tokens: no buffered verbs
            buffer_pages: 0,
            state_slot_bytes: facts.state_slot_bytes(),
            adapter_banks: 0,
            ..PoolFacts::default()
        };
        // Pinned chunks, one slot a chunk, device tier only: a kv page is its tokens on the host,
        // never a device span the runtime could place, evict or stream.
        let chunks = ChunkFacts::degenerate(
            &pools,
            &DegenerateBytes {
                kv_page: u64::from(pools.kv_page_size) * 4,
                buffer_page: 0,
            },
        );
        let caps = Capabilities {
            device: device_facts(),
            pools,
            chunks,
            limits: FireLimits {
                // The model holds one sequence's cache; a fire's lanes run one after another.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: wafer_pages.map_or(paging.pages_per_slot, |(_, pages, _)| pages),
                max_context: wafer_pages
                    .map_or(paging.context().min(model_context), |(_, p, s)| p * s),
                prefill_rows: 0,
                max_readouts: budgets.max_tokens,
                dispatch_depth,
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

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        refuse_adapters(frame)?;
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
            // Readouts come on the lanes' channels: a step lands when its epilogue has run.
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

    fn completion(&mut self) -> Option<&mut dyn Completion> {
        if self.asynchronous() {
            Some(self)
        } else {
            None
        }
    }

    fn programs(&mut self) -> Option<&mut dyn Programs> {
        Some(self)
    }

    fn channels(&mut self) -> Option<&mut dyn Channels> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn Tiers> {
        Some(self)
    }
}

impl Tiers for CerebrasEngine {
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
    fn copy(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        let _ = copy;
        Err(Error::Unsupported {
            verb: "`copy_slots` on the wafer program (its pools are pinned, device tier only: \
                   kv moves through `copy_kv`)",
            engine: "cerebras-spatial",
        })
    }

    fn read(&mut self, slots: &[(engine::chunk::ShapeId, Vec<u32>)]) -> EngineResult<Vec<u8>> {
        let _ = slots;
        Err(Error::Unsupported {
            verb: "`read_tier_slots` on the wafer program (it states no host tier)",
            engine: "cerebras-spatial",
        })
    }

    fn write(
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
}

impl Programs for CerebrasEngine {
    fn register(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.programs.register(registration).map_err(refusal)
    }

    fn bind(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
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

    fn close(&mut self, id: InstanceId) -> EngineResult<()> {
        self.drain();
        self.programs.close_instance(id).map_err(refusal)
    }

    fn adapter(&mut self, registration: &engine::adapter::AdapterRegistration) -> EngineResult<()> {
        Err(no_adapter(registration.id))
    }
}

impl Channels for CerebrasEngine {
    fn open(&mut self, registration: &ChannelRegistration) -> EngineResult<RegisteredChannel> {
        self.programs
            .register_channel(registration)
            .map_err(refusal)
    }

    fn close(&mut self, id: ChannelId) -> EngineResult<()> {
        self.drain();
        self.programs.close_channel(id).map_err(refusal)
    }

    fn publish(&mut self, instance: InstanceId, channel: u32, cell: &[u8]) -> EngineResult<bool> {
        self.programs
            .publish(instance, channel, cell)
            .map_err(refusal)
    }

    fn take(&mut self, instance: InstanceId, channel: u32) -> EngineResult<Option<Vec<u8>>> {
        self.programs.take(instance, channel).map_err(refusal)
    }
}

impl Completion for CerebrasEngine {
    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn publish_landed(&mut self) -> EngineResult<()> {
        self.pump();
        Ok(())
    }
}

/// The refusal of adapter `id`: the wafer's programs serve none, so a lane asking for one is
/// refused by name, never run without it.
fn no_adapter(id: u32) -> Error {
    Error::invalid(format!("cerebras serves no adapter: adapter {id}"))
}

/// A frame whose lane names an adapter, refused.
fn refuse_adapters(frame: &FrameSubmission) -> EngineResult<()> {
    for step in &frame.steps {
        if let Some(id) = step.lanes.iter().find_map(|l| l.adapter) {
            return Err(no_adapter(id));
        }
    }
    Ok(())
}
