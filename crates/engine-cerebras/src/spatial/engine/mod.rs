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
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::chunk::{ChunkFacts, DegeneratePools};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout};
use engine::load::{Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::MemoryDomain;
use eta_ir::registry::{GeometryClass, PortMask};

use kv::Kv;

use super::config::{Config, Target};
use super::facts::Facts;
use super::{prepare, wafer};

use crate::api::{ContractFor, refusal};

/// The spatial engine: the model forward in the wafer program, guest
/// programs on the host.
pub struct Spatial {
    programs: crate::program::Plane,
    cfg: Config,
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
    /// The most steps a greedy lane's token loop runs past its row, as the
    /// load decided ([`fire::lookahead_of`]).
    lookahead: usize,
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
            lookahead: 0,
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

    fn live(&self) -> engine::live::Live {
        self.programs.live()
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
            steps,
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
            tracing::warn!("the spatial engine states no host kv tier: host_kv_budget is not used");
        }
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the spatial engine loads a checkpoint path".into(),
            ));
        };
        // What the trace and budgets alone decide, before anything is
        // compiled: the float output the readout rows read, which bounds
        // the wafer's vocabulary. A trace without one could serve no fire.
        let slots = poem_compiler::state_slots(budgets.slots, &trace, DEFAULT_STATE_SLOTS);
        let facts = Facts::of(trace, &budgets, slots).map_err(Error::Load)?;
        let (readout, width) = facts.readout().ok_or_else(|| {
            Error::Load(
                "the trace reads no float output at its readout rows (`logits`, else its one \
                 float readout): the spatial engine serves none"
                    .into(),
            )
        })?;
        let contract = (self.contract_for)(facts.trace(), &path).map_err(Error::Load)?;
        // the program serves the context the config states (else as many positions as it holds)
        let positions = budgets.context_declared.then_some(budgets.max_context);
        let prepared = prepare::prepare(&self.cfg, facts.trace(), &contract, &path, positions)
            .map_err(|e| Error::Load(format!("the wafer program: {e}")))?;
        // Whether greedy lanes run the program's token loop, and how far:
        // decided before the wafer opens.
        self.lookahead = fire::lookahead_of(self.cfg.lookahead, prepared.plan.contract.token_loop)
            .map_err(Error::Load)?;
        self.readout = Some((readout, width));
        let platform = match &self.cfg.cmaddr {
            Some(cmaddr) => crate::sdk::Platform::System {
                cmaddr: cmaddr.clone(),
            },
            None => crate::sdk::Platform::Simulator(
                crate::sdk::Simulator::new(match self.cfg.target {
                    Target::Wse2 => crate::sdk::Target::Wse2,
                    Target::Wse3 => crate::sdk::Target::Wse3,
                })
                .map_err(|e| Error::Load(format!("the fabric simulator: {e}")))?,
            ),
        };
        let vocab = width as usize;
        let w = wafer::Wafer::open(
            prepared.program.clone(),
            platform,
            vocab,
            prepared.weights.clone(),
            self.cfg.toolchain.sdk_lib(),
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
        // Pinned pools as chunks, one slot a chunk, device tier only: a kv
        // page is its tokens on the host (a 32-bit token a position), never
        // a device span the runtime could place, evict or stream.
        let chunks = ChunkFacts::degenerate(&DegeneratePools {
            // The pages hold tokens here (the program's cache is replayed from them).
            kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
            kv_page_size: paging.page_size,
            kv_page_bytes: u64::from(paging.page_size) * 4,
            state_slots: if state_rows { paging.slots } else { 0 },
            state_slot_bytes: facts.state_slot_bytes(),
            // recurrent state replays from tokens: no buffered verbs
            ..DegeneratePools::default()
        });
        self.kv.ref_from = chunks
            .shape(engine::chunk::DEGENERATE_KV)
            .and_then(|shape| Some((shape.pinned?, shape.slots_per_chunk)))
            .map_or(0, |(pinned, per)| pinned.device.first * per - pinned.ids_from);
        let pools = PoolFacts::default();
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
                // what it planned with before forms were stated: every form
                library_forms: engine::LibraryForms::ALL,
            },
            pools,
            chunks,
            limits: FireLimits {
                // The model holds one sequence's cache; a fire's lanes run one
                // after another, each cache brought back from its tokens.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot,
                max_context: paging.context().min(model_context),
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: budgets.max_tokens,
                // Each lane's rung bounds what it replays.
                replayed_rows: 0,
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
            windowed_state: false,
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
        // no bodies (degenerate facts, out of scope for this set).
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
    fn settle_copies(&mut self) -> EngineResult<()> {
        Ok(())
    }

    /// The load states its pools pinned (`ChunkFacts::degenerate`), so the
    /// runtime moves its kv through `copy_kv`; there is no shared pool and
    /// no host or disk tier to move slots between.
    /// Refused as [`Engine::copy_slots`] is: the wafer program moves no
    /// slot between tiers.
    fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        _token: u64,
    ) -> EngineResult<bool> {
        self.copy_slots(copy)?;
        Ok(true)
    }

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

    /// Every registration: its ring on the host, mirrored where the host
    /// reads or writes it.
    fn holds_channel(&self, registration: &ChannelRegistration) -> bool {
        let _ = registration;
        true
    }

    /// The program adopted in the returned compile, off the engine's
    /// thread.
    fn prepare_program(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let registration = registration.clone();
        Ok(Box::new(move || {
            engine::Prepared(Box::new(crate::program::Plane::adopt(&registration)))
        }))
    }

    /// The adopted program's id; a compile that refused refuses here.
    fn finish_program(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let Ok(adopted) = prepared
            .0
            .downcast::<Result<eta_exec::ExecPlan, crate::program::Refusal>>()
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

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<u64> {
        self.drain();
        self.programs.close_instance(id).map_err(refusal)?;
        Ok(0)
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
}

impl Spatial {
    /// The frame `submit` runs: posted on an asynchronous engine, else run
    /// whole here.
    fn submit_frame(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
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

#[cfg(test)]
mod tests {
    use engine::Engine;
    use engine::fire::{FrameSubmission, Lane, Step};
    use engine::live::Tally;
    use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
    use poem::{Dtype, ForwardHybrid, Input, StateRows, Weight, ops};

    use super::Spatial;

    fn no_model(
        _: &poem_ir::Trace,
        _: &std::path::Path,
    ) -> std::result::Result<checkpoint::contract::ModelContract, String> {
        Err("no model in this test".into())
    }

    /// A frame the engine refuses (here: a lane before any load) is counted
    /// on the engine's live record, which the runtime reads.
    #[test]
    fn a_refused_frame_is_counted() {
        let mut spatial = Spatial::new(crate::spatial::Config::default(), no_model);
        let live = spatial.live();
        let refused = spatial.submit(&FrameSubmission::of(Step {
            lanes: vec![Lane {
                tokens: vec![1],
                ..Lane::default()
            }],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        }));
        assert!(refused.is_err());
        assert_eq!(live.tallied(Tally::Refused), 1);
    }

    const VOCAB: u32 = 16;

    /// A model whose one readout is an i32 argmax: no float output for the
    /// readout rows.
    struct IdsOnly;

    impl ForwardHybrid for IdsOnly {
        fn states(&self) -> StateRows {
            StateRows::new()
        }
        fn forward(&self, inputs: Input) {
            let table = Weight::sym("embed", [u64::from(VOCAB), 8], Dtype::Bf16);
            let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
            let x = ops::gather::gather_rows(&x, &inputs.readout_rows());
            poem::output("ids", &ops::sampling::argmax(&[&x]), None);
        }
    }

    fn compiles_nothing(
        _: &poem_ir::Trace,
        _: &std::path::Path,
    ) -> std::result::Result<checkpoint::contract::ModelContract, String> {
        panic!("a load the trace alone refuses reaches no import, plan or compile")
    }

    /// A trace with no float readout is refused at load, before the import
    /// contract, the plan or a compile, by name; it does not load with an
    /// unbounded vocabulary to fail at every fire.
    #[test]
    fn a_trace_without_a_float_readout_is_refused_at_load() {
        let trace = poem::trace_hybrid("ids-only", &IdsOnly, poem_ir::Platform::Cerebras);
        let mut spatial = Spatial::new(crate::spatial::Config::default(), compiles_nothing);
        let refused = spatial
            .load(LoadRequest {
                trace,
                checkpoint: Checkpoint::Path("/nonexistent/ids-only.zt".into()),
                budgets: Budgets {
                    max_lanes: 1,
                    max_tokens: 4,
                    buckets: vec![4],
                    max_adapters: 0,
                    page_size: 16,
                    max_context: 64,
                    context_declared: false,
                    slots: 1,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                residency: Residency::default(),
                ordinal: 0,
                frames_in_flight: 1,
                steps: Default::default(),
            })
            .expect_err("the load is refused");
        assert!(refused.to_string().contains("no float output"), "{refused}");
    }
}
