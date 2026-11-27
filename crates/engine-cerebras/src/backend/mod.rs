//! PIE's `Engine` over the wafer program. It loads the model (plans its trace, builds or finds
//! the program and the checkpoint's tiles, places them), then serves fires: each lane's rows
//! through the program's cache, the guest passes in the host interpreter, the lanes' readouts.

mod fire;
mod kv;
mod launch;
mod pool;
mod posted;
mod stream;
mod window;

use engine::caps::{Capabilities, DeviceFacts, FireLimits, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout};
use engine::load::{Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{DeviceReach, MemoryDomain};
use engine::{Channels, Completion, Engine, Programs, Tiers};
use eta_ir::registry::{GeometryClass, PortMask};

use kv::Kv;
use window::Windows;

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
    /// Windowed layers' rings and window buffer.
    windows: Windows,
    /// Fires run (the model-time log's count).
    fires: u64,
    /// Rows of history replayed into the model's cache.
    replayed: u64,
    /// Greedy rows a token loop answered, and the loops run.
    ahead_hits: u64,
    ahead_runs: u64,
    /// The posted token loop on the wafer: the token it was fed and its position.
    loop_posted: Option<(u32, usize)>,
    /// The most steps a greedy lane's token loop runs past its row, as the
    /// load decided ([`fire::lookahead_of`]).
    lookahead: usize,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<Vec<LaneReadout>>)>,
    /// Steps posted and not yet run (an asynchronous engine).
    posted: std::collections::VecDeque<posted::Posted>,
    /// A stream reading's program (a diffusion transformer's denoise), when the load is one.
    stream: Option<stream::Streams>,
    /// A unit program's pages, as its lanes' kv refs name them.
    wafer_pages: Option<pool::WaferPages>,
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
            windows: Windows::default(),
            fires: 0,
            replayed: 0,
            ahead_hits: 0,
            ahead_runs: 0,
            loop_posted: None,
            lookahead: 0,
            next_fire: 0,
            next_frame: 0,
            sink: None,
            pending: None,
            posted: std::collections::VecDeque::new(),
            stream: None,
            wafer_pages: None,
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
        // what it planned with before forms were stated: every form
        library_forms: engine::LibraryForms::ALL,
    }
}

/// Recurrent-state seats when the budgets state none: one a lane the budgets run together.
fn default_state_slots(budgets: &engine::load::Budgets) -> u32 {
    budgets.max_lanes.max(1)
}

impl Engine for CerebrasEngine {
    /// Unsupported on Cerebras, by name: it keys no host cover (no committed
    /// covers) and has no keyed load-time measurements to skip (out of scope
    /// for this work), so it never skips.
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    /// Cerebras times nothing at load: none.
    fn measured_at_load(&self) -> Vec<String> {
        Vec::new()
    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "cerebras-spatial"
    }

    /// The fire the wafer program is built for; a stated context is its
    /// positions, built into the program.
    fn fire_ceiling(&self) -> Option<engine::load::FireCeiling> {
        Some(engine::load::FireCeiling {
            lanes: 4,
            tokens: 64,
            page_size: None,
            context_binds: true,
        })
    }

    fn live(&self) -> engine::live::Live {
        self.programs.live()
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
            steps,
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
        // What the trace and budgets alone decide, before anything is compiled: the float output
        // a trace reads at its readout rows bounds the wafer's vocabulary, wherever it has one. A
        // decode model (no stream reading declared, which is what routes a load to its stream
        // readings) without one could serve no fire.
        let slots =
            poem_compiler::state_slots(budgets.slots, &trace, default_state_slots(&budgets));
        let decode = trace.facts.declared("reading").is_none() && self.debug.program.is_none();
        let readout = if decode {
            Facts::of(trace.clone(), &budgets, slots)
                .map_err(Error::Load)?
                .readout()
        } else {
            Facts::of(trace.clone(), &budgets, slots)
                .ok()
                .and_then(|facts| facts.readout())
        };
        if decode && readout.is_none() {
            return Err(Error::Load(
                "the trace reads no float output at its readout rows (`logits`, else its one \
                 float readout): the cerebras engine serves none"
                    .into(),
            ));
        }
        let width = readout.map(|(_, width)| width);
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;
        // a stream reading's program (its host.json), or a directory of them, one a reading
        if let Some(programs) = self
            .debug
            .program
            .as_deref()
            .and_then(|d| stream::programs_in(d, self.debug.link.as_deref()))
        {
            return self.load_stream(trace, &path, &contract, &budgets, &steps, &programs);
        }
        // the program serves the context the config states (else as many positions as it holds)
        let positions = budgets.context_declared.then_some(budgets.max_context);
        let platform = self.platform()?;
        let w = match prepare::prepare(&self.cfg, &trace, &contract, &path, positions) {
            Ok(prepared) => {
                let vocab = width.map_or(usize::MAX, |w| w as usize);
                let w = wafer::Wafer::open(
                    prepared.program.clone(),
                    platform,
                    vocab,
                    prepared.weights.clone(),
                    self.cfg.toolchain.sdk_lib(),
                )
                .map_err(|e| Error::Load(format!("the wafer model: {e}")))?;
                // Whether greedy lanes run the program's token loop, and how far: a lookahead
                // stated for a program without one is refused here.
                self.lookahead =
                    fire::lookahead_of(self.cfg.lookahead, w.token_loop, w.posted_loop)
                        .map_err(|e| Error::Load(format!("{}: {e}", prepared.program.display())))?;
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
                    self.lookahead =
                        fire::lookahead_of(self.cfg.lookahead, w.token_loop, w.posted_loop)
                            .map_err(Error::Load)?;
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
                        return self
                            .load_stream(trace, &path, &contract, &budgets, &steps, &programs);
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
        // One shared pool on the host: a kv page is its tokens (a 32-bit token a position),
        // never a device span the runtime could place, evict or stream. A unit program's pages
        // are Pie's (its K / V columns' own, a lane's ids its pages).
        let windowed = facts.windowed_layers().map_err(Error::Load)?;
        let lanes = budgets.max_lanes.max(1);
        let page_size = wafer_pages.map_or(paging.page_size, |(_, _, size)| size);
        self.kv.page_size = page_size;
        let pool = pool::Pool {
            page_size,
            kv_pages: wafer_pages.map_or(paging.pages(), |(_, pages, _)| u64::from(pages)),
            // each seat's record and a snapshot's, and a copy-on-write one a lane a fire runs
            records: paging.slots.saturating_mul(2).saturating_add(lanes),
            state_bytes: if facts.has_state() {
                facts.state_slot_bytes()
            } else {
                0
            },
            buffer_pages: u64::from(paging.slots)
                * u64::from(budgets.max_tokens.div_ceil(paging.page_size.max(1)) + 2),
            windowed,
            // a unit program's pages and state slots are its own on the wafer
            wafer: wafer_pages
                .zip(wafer_lanes)
                .map(|((_, pages, _), lanes)| pool::WaferIds {
                    pages,
                    state_slots: u32::try_from(lanes).unwrap_or(u32::MAX),
                }),
        };
        self.wafer_pages = pool.wafer_pages();
        let chunks = pool.facts();
        chunks
            .validate()
            .map_err(|e| Error::Load(format!("the spatial engine's pool: {e}")))?;
        if let Some(ids) = pool.wafer
            && chunks.total_chunks <= 1
        {
            return Err(Error::Load(format!(
                "the unit program's {} pages and {} state slots hold no chunk of {} bytes \
                 (its widest slot): the pool would name ids the wafer does not hold",
                ids.pages, ids.state_slots, chunks.chunk_bytes
            )));
        }
        self.windows = Windows::new(
            pool.windowed.iter().map(|&(_, window)| window).collect(),
            page_size,
        );
        let pools = PoolFacts::default();
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
                prefill_lanes: 0,
                max_readouts: budgets.max_tokens,
                // Each lane's rung bounds what it replays.
                replayed_rows: 0,
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
            rs_verbs: false,
            windowed_state: !self.windows.is_empty(),
            rs_snapshots: 0,
            bidirectional_attention: false,
            adapter_attachments: false,
            facts: facts.trace().facts.clone(),
            inputs: facts.input_decls(),
            classes: facts.classes(),
            // Cerebras states no bound on one submission.
            guest_cost: engine::GuestCost::default(),
            // No `beam_list`: a beam group's bind is refused by name.
            host_ops: engine::HostOps::NONE,
            mixed_lanes: engine::MixedLanes::Any,
            // one frame at a time: no run-ahead
            frames_in_flight: 1,
            steps: engine::StatedSteps::default(),
        };

        // The step compositions the runtime's builder forms over this load's
        // facts (CONTRACT-BC 24). Interim: computed once the caps are; Cerebras records
        // no bodies (out of scope for this set).
        let mut caps = caps;
        caps.steps = engine::StatedSteps::of(
            &steps,
            &engine::StepFacts::stated(&caps, &engine::ReadoutFacts::default()),
        );
        tracing::info!(
            "engine-cerebras: step set: {}; its arm walk does not key on the set yet, so none \
             of the {} is checked against what it records",
            caps.steps.shapes.census(),
            caps.steps.shapes.len(),
        );
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

    /// A frame's submit, its refusal counted ([`engine::live::Tally::Refused`]).
    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        let submitted = self.submit_frame(frame);
        if submitted.is_err() {
            self.programs.live().tally(engine::live::Tally::Refused, 1);
        }
        submitted
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

    /// Every load: a synchronous one still tells the sink, inside
    /// `submit`, of each step whose lanes faulted ([`Completion::asynchronous`]).
    fn completion(&mut self) -> Option<&mut dyn Completion> {
        Some(self)
    }

    fn programs(&mut self) -> Option<&mut dyn Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn Programs> {
        Some(self)
    }

    fn channels(&mut self) -> Option<&mut dyn Channels> {
        Some(self)
    }

    fn channels_ref(&self) -> Option<&dyn Channels> {
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

    /// The wafer program copies no slot between tiers: nothing to wait for.
    fn settle(&mut self) -> EngineResult<()> {
        Ok(())
    }

    /// Refused as [`Tiers::copy`] is: the wafer program moves no slot
    /// between tiers.
    fn copy_landing(&mut self, copy: &engine::chunk::SlotCopy, _token: u64) -> EngineResult<bool> {
        self.copy(copy)?;
        Ok(true)
    }

    fn copy(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        let _ = copy;
        Err(Error::Unsupported {
            verb: "`copy_slots` on the wafer program (its pool has no host or disk tier: \
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

    /// The program adopted in the returned compile, off the engine's
    /// thread.
    fn prepare(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let registration = registration.clone();
        Ok(Box::new(move || {
            engine::Prepared(Box::new(crate::guest::Plane::adopt(&registration)))
        }))
    }

    /// The adopted program's id; a compile that refused refuses here.
    fn finish(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let Ok(adopted) = prepared
            .0
            .downcast::<Result<eta_exec::ExecPlan, crate::guest::Refusal>>()
        else {
            return Err(engine::Error::Program(
                "a prepared program Cerebras did not prepare".to_string(),
            ));
        };
        self.programs.finish(*adopted).map_err(refusal)
    }

    fn close_program(&mut self, program: ProgramId) -> EngineResult<()> {
        self.programs.close_program(program).map_err(refusal)
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

    fn close(&mut self, id: InstanceId) -> EngineResult<u64> {
        self.drain();
        self.programs.close_instance(id).map_err(refusal)?;
        Ok(0)
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

    /// Whether a step lands after `submit` returns ([`Self::on_complete`]
    /// told each), as the config says.
    fn asynchronous(&self) -> bool {
        CerebrasEngine::asynchronous(self)
    }
}

impl CerebrasEngine {
    /// The frame `submit` runs: posted on an asynchronous engine, else run
    /// whole here.
    fn submit_frame(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
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
                refused: Vec::new(),
            });
        }
        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut readouts = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            let (read, outcome) = self.fire_step(step)?;
            if let Some(sink) = &self.sink {
                sink(
                    engine::StepDone {
                        frame: id,
                        step: index as u32,
                    },
                    outcome,
                );
            }
            steps.push(FireTicket {
                id: FireId(self.next_fire),
                readouts: read.clone(),
            });
            readouts.push(read);
        }
        self.pending = Some((id, readouts));
        Ok(FrameTicket::of(id, steps, None))
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
