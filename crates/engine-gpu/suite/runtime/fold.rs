//! A buffering lane whose fold length is the device's replays what one
//! folding the same count on the host replays, bit for bit, when the
//! runtime's count lags: the device-folding fires of one set are pipelined
//! (the next submitted before the previous settles), each stating the
//! `folded` the runtime held at submit, before the report of the fold it
//! carries was heard. And a fork of that set, fired while its parent's
//! device fold is unsettled, replays what the host fork replays. Each
//! device-folding fire reports the slots its fold froze through the
//! completion sink: as its step lands on an engine that settles off the
//! submit, and before `submit` returns on one that settles inside it (one
//! frame in flight, where the engine then does). qwen35-tiny (recurrent
//! layers); the fold count rides a decode-envelope program's `fold` port.

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::{ChunkFacts, Growth, ShapeId, SlotKind};
use engine::fire::{
    Attachment, FoldLen, FrameSubmission, FrameTicket, KvDelta, Lane, Readout, RsReset, RsVerb,
    Step,
};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration};
use engine::{StepDone, StepOutcome};
use engine_gpu::store::layout::{FIXED, LIST, RS};
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, GeometryClass, Port, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem_ir::Trace;

use super::outputs::contract_for;
use super::slots::Slots;
use crate::backend;
use crate::model_logits::{fixture, word};

const MODEL: &str = "qwen35-tiny";

/// The slots each buffering fire's fold freezes.
const N: u32 = 4;
/// The prompt's rows (folded) and each buffering fire's.
const PROMPT: u32 = 16;
const ROWS: u32 = 8;
const PAGE: u32 = 16;
/// The sets the test serves, forks included.
const SETS: u32 = 7;

/// A program that tells each fire's tokens and fold length, from host
/// channels: channel 0 the lane's `ROWS` tokens, channel 1 its fold.
fn fold_program() -> TraceContainer {
    let channel = |shape: Shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::U32),
        capacity: 2,
        host_role: HostRole::Writer,
        seeded: false,
    };
    let indptr: Vec<u8> = [0u32, ROWS].iter().flat_map(|w| w.to_le_bytes()).collect();
    TraceContainer {
        names: Vec::new(),
        channels: vec![channel(Shape::vector(ROWS)), channel(Shape::vector(1))],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            },
            PortBinding {
                port: Port::EmbedIndptr,
                source: PortSource::Const {
                    dtype: EtaDtype::U32,
                    shape: Shape::vector(2),
                    data: indptr,
                },
            },
            PortBinding {
                port: Port::Fold,
                source: PortSource::Channel(1),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::ChanTake(0), Op::ChanTake(1)],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The fold program bound on the engine, a decode envelope, its tables and
/// cells in `slots`.
fn bind_fold(engine: &mut backend::Engine, slots: &mut Slots) -> InstanceId {
    let bound = eta_ir::validate::bind(fold_program(), vec![Vec::new()], EngineCaps::default())
        .expect("the fold program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0xf01d,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the fold program registers");
    let channels: Vec<ChannelId> = [ROWS, 1]
        .iter()
        .enumerate()
        .map(|(at, &width)| {
            let id = ChannelId((1 << 20) + at as u64);
            slots.register(
                engine,
                ChannelRegistration {
                    id,
                    shape: vec![width],
                    dtype: ChanDType::Concrete(EtaDtype::U32),
                    host_role: HostRole::Writer,
                    seeded: false,
                    extern_dir: None,
                    capacity: 2,
                    extern_name: Vec::new(),
                    slots: Vec::new(),
                },
            );
            id
        })
        .collect();
    let binding = slots.binding(
        engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: Vec::new(),
            geometry: GeometryClass::DecodeEnvelope,
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            extents: engine::program::BindExtents {
                row_count: ROWS,
                token_count: ROWS,
                sampled_rows: ROWS,
                query_len: ROWS,
                ..engine::program::BindExtents::default()
            },
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            pair_generation: 0,
        },
        &channels,
    );
    engine
        .bind_instance(&binding)
        .expect("the fold program binds an instance")
        .id
}

/// Slot refs out of chunks of the test's own: a shape fills a chunk, then
/// takes the next one up.
struct Refs {
    facts: ChunkFacts,
    chunk: u32,
    end: u32,
    open: Vec<(ShapeId, u32, u32)>,
}

impl Refs {
    fn take(&mut self, id: ShapeId) -> u32 {
        let per = self
            .facts
            .shape(id)
            .unwrap_or_else(|| panic!("{MODEL} lays out shape {id}"))
            .slots_per_chunk;
        let at = match self.open.iter().position(|&(shape, ..)| shape == id) {
            Some(at) => at,
            None => {
                self.open.push((id, 0, 0));
                self.open.len() - 1
            }
        };
        if self.open[at].1 == self.open[at].2 {
            assert!(self.chunk < self.end, "the test's chunks are committed");
            self.open[at] = (id, self.chunk * per, (self.chunk + 1) * per);
            self.chunk += 1;
        }
        self.open[at].1 += 1;
        self.open[at].1 - 1
    }
}

fn tokens(seed: u32) -> Vec<u32> {
    (0..ROWS)
        .map(|at| (seed * 29 + at * 37 + 11) % 256)
        .collect()
}

/// One sequence: its kv pages, the rs-list slots its buffer holds from the
/// prompt's end on, its record (`[run, cow]` a piece), and the slot copies
/// its next lane makes (a fork's).
struct Set {
    slot: u32,
    held: u32,
    kv: Vec<u32>,
    rs: Vec<u32>,
    record: Vec<[u32; 2]>,
    copies: Vec<(ShapeId, Vec<[u32; 2]>)>,
}

impl Set {
    /// A set of its own slots, its prompt of `PROMPT` rows folded.
    fn new(model: &mut Model, slot: u32) -> Set {
        let kv = (0..(PROMPT + 3 * ROWS).div_ceil(PAGE))
            .map(|_| model.refs.take(LIST))
            .collect();
        let rs = (0..model.rs_pages).map(|_| model.refs.take(RS)).collect();
        let record = (0..model.pieces)
            .map(|_| {
                let piece = model.refs.take(FIXED);
                [piece, piece]
            })
            .collect();
        let mut set = Set {
            slot,
            held: 0,
            kv,
            rs,
            record,
            copies: Vec::new(),
        };
        // Every set's prompt is the same: sets are told apart by their
        // fires alone.
        let prompt: Vec<u32> = (0..PROMPT).map(|at| (at * 53 + 5) % 256).collect();
        let lane = Lane {
            rs: RsVerb::Fold,
            rs_reset: RsReset::Fresh,
            ..set.rows(&model.trace, prompt)
        };
        model.fire(lane);
        set.held = PROMPT;
        set
    }

    /// A lane of `tokens` after the rows the set holds.
    fn rows(&self, trace: &Trace, tokens: Vec<u32>) -> Lane {
        let upto = self.held + tokens.len() as u32;
        Lane {
            slot: self.slot,
            word: word(trace, tokens.len() as u32),
            kv: KvDelta {
                held: self.held,
                ..KvDelta::default()
            },
            readout: Readout::Last,
            tables: vec![(LIST, self.kv[..upto.div_ceil(PAGE) as usize].to_vec())],
            record: engine::fire::LaneRecord::Pieces(vec![(FIXED, self.record.clone())]),
            tokens,
            ..Lane::default()
        }
    }

    /// A buffering lane of `ROWS` rows whose state holds the slots below
    /// `folded`, folding `fold`; it reports under `reports` (0: no report).
    fn lane(&mut self, trace: &Trace, seed: u32, folded: u32, fold: FoldLen, reports: u64) -> Lane {
        let lane = Lane {
            rs: RsVerb::Buffer {
                pages: self.rs.clone(),
                base: PROMPT,
                folded,
                fold,
                lineage: engine::fire::Lineage::Set,
            },
            rs_reset: RsReset::Held,
            slot_copies: std::mem::take(&mut self.copies),
            reports,
            ..self.rows(trace, tokens(seed))
        };
        self.held += ROWS;
        // A fork's record is its own from its first fire on.
        for piece in &mut self.record {
            piece[0] = piece[1];
        }
        lane
    }

    /// A fork of the set as it stands: the parent's record copied onto one
    /// of its own at its first fire, and the kv page and rs slots its rows
    /// land in copied to slots of its own.
    fn fork(&self, model: &mut Model, slot: u32) -> Set {
        let mut kv = self.kv.clone();
        let page = (self.held / PAGE) as usize;
        let mut kv_copies = Vec::new();
        if page < kv.len() {
            let fresh = model.refs.take(LIST);
            kv_copies.push([kv[page], fresh]);
            kv[page] = fresh;
        }
        let record = self
            .record
            .iter()
            .map(|&[_, cow]| [cow, model.refs.take(FIXED)])
            .collect();
        let landed = ((self.held - PROMPT) / model.rs_tokens) as usize;
        let mut rs = self.rs.clone();
        let rs_copies: Vec<[u32; 2]> = (landed..rs.len())
            .map(|at| {
                let fresh = model.refs.take(RS);
                let old = std::mem::replace(&mut rs[at], fresh);
                [old, fresh]
            })
            .collect();
        Set {
            slot,
            held: self.held,
            kv,
            rs,
            record,
            copies: vec![(LIST, kv_copies), (RS, rs_copies)],
        }
    }
}

/// The loaded miniature, the slots its sets take and what its completion
/// sink heard.
struct Model {
    engine: backend::Engine,
    trace: Trace,
    refs: Refs,
    pieces: u32,
    rs_tokens: u32,
    rs_pages: u32,
    /// Whether the engine settles off the submit: its sink then hears
    /// every step. Else a frame settles inside `submit`, whose result
    /// answers for it, and the sink hears only what a step's lanes wrote.
    asynchronous: bool,
    heard: mpsc::Receiver<(StepDone, StepOutcome)>,
    seen: Vec<(StepDone, StepOutcome)>,
}

impl Model {
    fn submit(&mut self, lane: Lane, attach: Option<InstanceId>) -> FrameTicket {
        let ticket = self
            .engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![lane],
                attachments: attach
                    .map(|instance| Attachment {
                        lane: 0,
                        lanes: 1,
                        instance,
                        adapter_only: false,
                        scratch: ::engine::ScratchPlace::None,
                    })
                    .into_iter()
                    .collect(),
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the frame fires");
        if !self.asynchronous {
            // What a frame settled inside `submit` tells, it has told by
            // now: nothing heard later counts.
            self.seen.extend(self.heard.try_iter());
        }
        ticket
    }

    /// The frame's last-row logits, as bits.
    fn settled(&mut self, ticket: &mut FrameTicket) -> Vec<u32> {
        self.engine.settle_frame(ticket).expect("the frame settles");
        ticket.steps[0].readouts[0]
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Fires a host lane and settles it: it reports no fold.
    fn fire(&mut self, lane: Lane) -> Vec<u32> {
        let mut ticket = self.submit(lane, None);
        let bits = self.settled(&mut ticket);
        assert_eq!(
            self.outcome(&ticket),
            self.asynchronous.then(StepOutcome::committed),
            "a fire whose fold lengths the host states reports no fold"
        );
        bits
    }

    /// Publishes `lane`'s tokens and the fold length on the program's
    /// channels.
    fn tell(&mut self, instance: InstanceId, lane: &Lane) {
        let tokens: Vec<u8> = lane.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        for (channel, bytes) in [(0, tokens), (1, N.to_le_bytes().to_vec())] {
            assert!(
                self.engine
                    .publish_channel(instance, channel, &bytes)
                    .expect("the cell publishes"),
                "the ring had room"
            );
        }
    }

    /// What the completion sink heard of `ticket`'s step: once it lands,
    /// on an engine that settles off the submit; else by the time its
    /// `submit` returned, where a step that wrote nothing tells nothing.
    fn outcome(&mut self, ticket: &FrameTicket) -> Option<StepOutcome> {
        loop {
            if let Some((_, outcome)) = self.seen.iter().find(|(at, _)| at.frame == ticket.id) {
                return Some(outcome.clone());
            }
            if !self.asynchronous {
                return None;
            }
            let heard = self
                .heard
                .recv_timeout(Duration::from_secs(60))
                .expect("the step's completion reaches the sink");
            self.seen.push(heard);
        }
    }

    /// The slots the fold of `ticket`'s step froze, as its one lane
    /// reported them under `tag`.
    fn reported(&mut self, ticket: &FrameTicket, tag: u64) -> Option<u32> {
        match self.outcome(ticket) {
            Some(StepOutcome {
                wrote: writes,
                faulted,
            }) if !writes.is_empty() && faulted.is_empty() => {
                assert_eq!(writes.len(), 1, "one lane reports: {writes:?}");
                let report = &writes[0];
                assert_eq!(report.tag, tag, "the report carries the lane's tag");
                assert_eq!(report.end, None, "this engine tells no end");
                assert!(
                    report.tokens.is_empty() && report.pages.is_empty(),
                    "this engine places no rows: {report:?}"
                );
                report.fold
            }
            other => panic!("tag {tag}: the step reported {other:?}, not what its fold froze"),
        }
    }

    /// What a set's state and buffer answer once every fold has settled: a
    /// host lane that replays from the exact count and folds nothing.
    fn probe(&mut self, set: &mut Set, folded: u32) -> Vec<u32> {
        let lane = set.lane(&self.trace, 99, folded, FoldLen::Host(0), 0);
        self.fire(lane)
    }
}

/// The miniature loaded with `frames` in flight.
fn load(frames: u8) -> Option<(Model, InstanceId)> {
    load_under(frames, None, PAGE)
}

/// [`load`] for fires of up to `tokens` rows, a set's kv pages for as many
/// rows past its own, on a device whose platform states `bound` on one GPU
/// submission where given (the boot's, `test-limits`).
fn load_under(
    frames: u8,
    bound: Option<Duration>,
    tokens: u32,
) -> Option<(Model, InstanceId)> {
    let boot = crate::backend::DeviceBoot {
        submit_bound: bound,
        ..crate::gpu::device_boot()
    };
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return None;
    }
    let fixture = fixture(MODEL, None);
    let mut engine = backend::Engine::new(boot, contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: tokens,
                buckets: vec![tokens],
                max_adapters: 0,
                page_size: PAGE,
                max_context: 64.max(2 * tokens),
                context_declared: false,
                slots: SETS + 1,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: frames,
        })
        .expect("the miniature loads");
    let asynchronous = engine.settles_asynchronously();
    let (tx, heard) = mpsc::channel();
    let tx = Mutex::new(tx);
    engine.on_complete(Arc::new(move |at, outcome| {
        let _ = tx.lock().expect("the sink's sender").send((at, outcome));
    }));

    let facts = loaded.caps.chunks.clone();
    let pieces = match facts.shape(FIXED).map(|shape| shape.kind) {
        Some(SlotKind::Fixed { pieces }) => pieces,
        other => panic!("{MODEL}'s record shape is {other:?}"),
    };
    // The record's bytes the load states are the model's state rows, which
    // its pool lays in `pieces` fixed slots: a record's, not its slots'.
    let laid = facts.shape(FIXED).map_or(0, |shape| shape.slot_bytes) * u64::from(pieces);
    assert!(
        facts.record_bytes > 0 && facts.record_bytes <= laid,
        "{MODEL} states a record of {} bytes in {laid} bytes of slots",
        facts.record_bytes
    );
    let rs_tokens = match facts.shape(RS).map(|shape| shape.kind) {
        Some(SlotKind::List { tokens, .. }) => tokens,
        other => panic!("{MODEL}'s rs list is {other:?}"),
    };
    let rs_pages = (3 * ROWS).div_ceil(rs_tokens) + 1;
    // Segments of the test's own above the committed ones: the guest's two
    // chunks, then each shape's (a set's kv pages and a fork's page, its
    // rs slots and its record).
    let per = |id| facts.shape(id).map_or(1, |shape| shape.slots_per_chunk);
    let need = 2
        + (SETS * ((PROMPT + 3 * ROWS).max(tokens + 1).div_ceil(PAGE) + 1)).div_ceil(per(LIST))
        + (SETS * rs_pages).div_ceil(per(RS))
        + (SETS * pieces).div_ceil(per(FIXED));
    let segment = facts.segment_chunks;
    // Where the pool grows, the test's segments are committed past the
    // load's. A pool laid whole at its load (`Growth::Fixed`, as wgpu lays
    // it) has none past them: its chunks lie from the first segment past the
    // seats' runs, below the activation span's segment (the last).
    let (first, commit, most) = if facts.segments.growth == Growth::Fixed {
        let seats = 1 + engine
            .loaded()
            .expect("the shell stands")
            .pool_layout()
            .seat_chunks();
        let first = seats.div_ceil(segment) * segment;
        (
            first,
            (first + need).div_ceil(segment),
            facts.segments.max - 1,
        )
    } else {
        (
            facts.segments.committed * segment,
            facts.segments.committed + need.div_ceil(segment),
            facts.segments.max,
        )
    };
    assert!(
        most >= commit,
        "the test's chunks span {commit} segments, the pool has {most} for them"
    );
    engine.commit_segments(commit).expect("the segments commit");
    eprintln!(
        "{MODEL}: {need} chunks from {first}, {segment} a segment; {pieces} piece(s) a record, \
         {rs_tokens} rows an rs slot"
    );
    let mut slots = Slots::of(&facts, first, first + 1);
    let instance = bind_fold(&mut engine, &mut slots);
    let model = Model {
        engine,
        trace: fixture.trace,
        refs: Refs {
            chunk: first + 2,
            end: commit * segment,
            facts,
            open: Vec::new(),
        },
        pieces,
        rs_tokens,
        rs_pages,
        asynchronous,
        heard,
        seen: Vec::new(),
    };
    Some((model, instance))
}

/// Two frames in flight: the engine settles off the submit, and a fire is
/// staged behind one still in flight.
#[test]
fn a_device_fold_replays_what_a_host_fold_replays() {
    replays_what_a_host_fold_replays(2);
}

/// One frame in flight: an engine that then settles each frame inside
/// `submit` has told a fold's report by the time `submit` returns.
#[test]
fn a_device_fold_settled_inside_submit_replays_what_a_host_fold_replays() {
    replays_what_a_host_fold_replays(1);
}

fn replays_what_a_host_fold_replays(frames: u8) {
    let Some((mut model, instance)) = load(frames) else {
        return;
    };
    let trace = model.trace.clone();

    // Two buffering fires folding N each. The host's counts are known at
    // submit, so the runtime's `folded` is exact.
    let mut host = Set::new(&mut model, 0);
    let a = host.lane(&trace, 11, PROMPT, FoldLen::Host(N), 0);
    model.fire(a);
    let b = host.lane(&trace, 12, PROMPT + N, FoldLen::Host(N), 0);
    let rows = model.fire(b);
    let want = model.probe(&mut host, PROMPT + 2 * N);
    assert!(
        !want.is_empty() && want.iter().all(|&bits| f32::from_bits(bits).is_finite()),
        "the reference reads finite logits"
    );
    // The device's ride the program's `fold` port: the second is submitted
    // before the first's report is read (before the first settles, where
    // the engine settles off the submit), stating the `folded` the runtime
    // held then.
    let mut device = Set::new(&mut model, 1);
    let a = device.lane(&trace, 11, PROMPT, FoldLen::Device, 101);
    model.tell(instance, &a);
    let first = model.submit(a, Some(instance));
    let b = device.lane(&trace, 12, PROMPT, FoldLen::Device, 102);
    model.tell(instance, &b);
    let mut second = model.submit(b, Some(instance));
    // The first frame's rows went with the second's submit; its report is
    // the sink's. A step whose guest pass runs behind its forward hands
    // the host no rows.
    let got = model.settled(&mut second);
    assert!(
        got.is_empty() || got == rows,
        "the second fire replays from where the first left the record"
    );
    assert_eq!(
        model.reported(&first, 101),
        Some(N),
        "the first fire's fold"
    );
    assert_eq!(
        model.reported(&second, 102),
        Some(N),
        "the second fire's own fold, from where the first left the record"
    );
    assert!(
        model.probe(&mut device, PROMPT + 2 * N) == want,
        "a pipelined device fold leaves what the host fold leaves"
    );

    // The same lagging `folded` with no position of the record's own (a
    // host count keeps none) folds slots twice: the probe tells them apart.
    let mut twice = Set::new(&mut model, 6);
    let a = twice.lane(&trace, 11, PROMPT, FoldLen::Host(N), 0);
    model.fire(a);
    let b = twice.lane(&trace, 12, PROMPT, FoldLen::Host(N), 0);
    model.fire(b);
    assert!(
        model.probe(&mut twice, PROMPT + 2 * N) != want,
        "slots replayed twice leave another state: the probe sees a lagging `folded`"
    );

    // A fork of a set whose device fold is unsettled, fired at once.
    let mut host = Set::new(&mut model, 2);
    let a = host.lane(&trace, 21, PROMPT, FoldLen::Host(N), 0);
    model.fire(a);
    let mut child = host.fork(&mut model, 3);
    let c = child.lane(&trace, 22, PROMPT + N, FoldLen::Host(N), 0);
    let rows = model.fire(c);
    let want = model.probe(&mut child, PROMPT + 2 * N);
    let mut device = Set::new(&mut model, 4);
    let a = device.lane(&trace, 21, PROMPT, FoldLen::Device, 201);
    model.tell(instance, &a);
    let first = model.submit(a, Some(instance));
    let mut child = device.fork(&mut model, 5);
    let c = child.lane(&trace, 22, PROMPT, FoldLen::Device, 202);
    model.tell(instance, &c);
    let mut second = model.submit(c, Some(instance));
    let got = model.settled(&mut second);
    assert!(
        got.is_empty() || got == rows,
        "the fork replays from where its parent's fold left the record it copied"
    );
    assert_eq!(model.reported(&first, 201), Some(N), "the parent's fold");
    assert_eq!(
        model.reported(&second, 202),
        Some(N),
        "the fork's own fold, from where its parent's left the record it copied"
    );
    assert!(
        model.probe(&mut child, PROMPT + 2 * N) == want,
        "a fork of a set whose device fold is unsettled leaves what the host fork leaves"
    );
    eprintln!(
        "{}: {frames} frame(s) in flight, reports heard {}: pipelined and forked device folds \
         reported {N} a fire and left the host folds' state over {} logits",
        backend::NAME,
        if model.asynchronous {
            "as each step landed"
        } else {
            "by the time each submit returned"
        },
        want.len()
    );
}

/// A lane sharing its set replays the buffered slots its mask admits
/// (`Lineage::Admitted`), every run of them in slot order. Two sets hold the
/// same prompt and the same eight buffered rows; a masked lane after them
/// admits every slot, on one set as one run and on the other as two runs
/// side by side (a zero-width hidden run between). Its recurrence replays
/// the same rows either way, so the two land the same logits, bit for bit;
/// replaying only the first run, the second set's lane ran its recurrence
/// over rows the replay never wrote.
#[test]
fn a_two_run_lineage_replays_both_runs() {
    let Some((mut model, _)) = load(1) else {
        return;
    };
    let trace = model.trace.clone();
    let masked = trace.facts.word(&poem::Request::new(ROWS, true));
    let mut last = |slot: u32, runs: Vec<u32>| {
        let mut set = Set::new(&mut model, slot);
        let buffer = set.lane(&trace, 11, PROMPT, FoldLen::Host(0), 0);
        model.fire(buffer);
        let mut lane = set.lane(&trace, 12, PROMPT, FoldLen::Host(0), 0);
        let RsVerb::Buffer { lineage, .. } = &mut lane.rs else {
            unreachable!("a buffering lane")
        };
        *lineage = engine::fire::Lineage::Admitted;
        lane.word = masked;
        lane.kv.folded = PROMPT;
        let total = u64::from(ROWS + ROWS);
        lane.mask = Some(engine::fire::Masking::Extent(engine::fire::Mask::new(
            runs, total,
        )));
        model.fire(lane)
    };
    let one = last(0, vec![0, 2 * ROWS]);
    let two = last(1, vec![0, 3, 0, 2 * ROWS - 3]);
    assert!(
        !one.is_empty() && one.iter().all(|&bits| f32::from_bits(bits).is_finite()),
        "the one-run lane reads finite logits"
    );
    assert!(
        one == two,
        "a lineage of two runs side by side replays what one run over the same slots replays"
    );
}

/// Under a stated submit bound small enough that one layer of a 512-row
/// prompt step of qwen35-tiny (attention and gated-delta recurrent layers)
/// is past it, the step is cut by rows (`engine_gpu::submit::row_pieces`),
/// its recurrent state carried piece to piece in its record, and serves
/// what it served whole under a roomy bound: its last logits, and a decode
/// after it read from the record the pieces left. The tight load is booted
/// under a stated bound, then held to one between a row's layer and the
/// step's at the ratio its rates step measured alone (`Shell::hold_to`: a
/// desktop GPU's idle probe swings 1x to 7x from frame to frame).
#[test]
fn a_hybrid_prompt_past_the_bound_a_layer_at_a_time_is_cut_by_rows_and_serves_alike() {
    use engine_gpu::submit::{Admitted, StepRows};
    const LONG: u32 = 512;
    let step = StepRows {
        rows: LONG,
        lanes: 1,
        keys: LONG,
        prompt: true,
    };
    let prompt: Vec<u32> = (0..LONG).map(|at| (at * 37 + 11) % 256).collect();
    // The prompt, the load's first fire, on a set of its own held to
    // `bound`, then a decode of its argmax held to a roomy one: the logits
    // of both, and the fires the prompt's step took past its own (its
    // leading row pieces: fire ids run one a fire from 1).
    let serve = |model: &mut Model, bound: f64| -> (Vec<f32>, Vec<f32>, u64) {
        let kv: Vec<u32> = (0..(LONG + 1).div_ceil(PAGE))
            .map(|_| model.refs.take(LIST))
            .collect();
        let record: Vec<[u32; 2]> = (0..model.pieces)
            .map(|_| {
                let piece = model.refs.take(FIXED);
                [piece, piece]
            })
            .collect();
        let trace = model.trace.clone();
        let lane = |tokens: Vec<u32>, held: u32, rs_reset: RsReset| Lane {
            slot: 0,
            word: word(&trace, tokens.len() as u32),
            kv: KvDelta {
                held,
                ..KvDelta::default()
            },
            readout: Readout::Last,
            tables: vec![(LIST, kv.clone())],
            record: engine::fire::LaneRecord::Pieces(vec![(FIXED, record.clone())]),
            rs: RsVerb::Fold,
            rs_reset,
            tokens,
            ..Lane::default()
        };
        let mut fire = |lane: Lane, bound: f64| {
            model.engine.loaded_mut().expect("loaded").hold_to(bound);
            let mut ticket = model.submit(lane, None);
            let bits = model.settled(&mut ticket);
            let values: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();
            (ticket.steps[0].id.0, values)
        };
        let (at, prompted) = fire(lane(prompt.clone(), 0, RsReset::Fresh), bound);
        let next = crate::model_logits::argmax(&prompted);
        let (after, decoded) = fire(lane(vec![next], LONG, RsReset::Held), 2e9);
        assert_eq!(after, at + 1, "the decode is the fire after the prompt's last");
        (prompted, decoded, at - 1)
    };
    let held_of = |model: &Model| {
        model
            .engine
            .loaded()
            .expect("loaded")
            .submit_bound()
            .expect("the bound the boot states")
            .clone()
    };
    let Some((mut roomy, _)) = load_under(1, Some(Duration::from_secs(2)), LONG) else {
        return;
    };
    let held = held_of(&roomy);
    assert!(matches!(held.admit(step), Admitted::Whole { .. }), "{:?}", held.admit(step));
    let (whole, whole_decode, pieces) = serve(&mut roomy, 2e9);
    assert_eq!(pieces, 0, "a step within the bound is fired whole");
    drop(roomy);
    let Some((mut cut, _)) = load_under(1, Some(Duration::from_secs(2)), LONG) else {
        return;
    };
    // A bound between one row's layer and the whole step's as `admit` reads
    // them once held (at the ratio the rates step measured alone: the
    // load's probe read the clock of a device idle since its boot, an L40S
    // at ~210 MHz, and a hold drops that scale): a layer of the step past
    // it, a row's within.
    cut.engine.loaded_mut().expect("loaded").hold_to(2e9);
    let held = held_of(&cut);
    let layer = |rows: u32| held.step_ns(StepRows { rows, ..step }).1;
    let tight = (layer(1) * layer(LONG)).sqrt();
    eprintln!(
        "a layer of 1 row {:.3} ms, of {LONG} rows {:.3} ms; the bound {:.3} ms",
        layer(1) / 1e6,
        layer(LONG) / 1e6,
        tight / 1e6,
    );
    cut.engine.loaded_mut().expect("loaded").hold_to(tight);
    let held = held_of(&cut);
    cut.engine.loaded_mut().expect("loaded").hold_to(2e9);
    assert!(
        matches!(held.admit(step), Admitted::Refused { .. }),
        "a layer of the step is past the bound: {:?}",
        held.admit(step)
    );
    let piece_rows = held.rows_a_piece(step, LONG).expect("a piece of fewer rows fits");
    assert!(piece_rows < LONG);
    // Pieces of at most `piece_rows` rows, fewer as the pieces landed raise
    // the ratio each next one is cut at.
    let (pieced, pieced_decode, pieces) = serve(&mut cut, tight);
    assert!(
        pieces >= u64::from(LONG.div_ceil(piece_rows) - 1),
        "the step went out in {pieces} + 1 pieces, of at most {piece_rows} rows"
    );
    let cosine = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b)).max(f64::MIN_POSITIVE)
    };
    for (what, a, b) in [("prompt", &whole, &pieced), ("decode", &whole_decode, &pieced_decode)] {
        assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()), "{what}: finite");
        let (wa, wb) = (crate::model_logits::argmax(a), crate::model_logits::argmax(b));
        assert_eq!(wa, wb, "{what}: the same argmax whole and in pieces");
        let cos = cosine(a, b);
        assert!(cos > 0.999, "{what}: logits whole and in pieces at cosine {cos}");
        eprintln!("{what}: cosine {cos:.6} whole against {pieces} + 1 pieces of at most {piece_rows} rows");
    }
}

/// Under a stated submit bound small enough that one layer of a step of two
/// 256-row qwen35-tiny prompts is past it, the step is cut lane by lane,
/// then by rows (`engine_gpu::submit::next_cut`): each prompt lane its own
/// sequence of pieces, its recurrent state carried in its own record, both
/// lanes' last rows in the step's last piece. Both prompts' logits, and a
/// decode of both after, are what the step served whole.
#[test]
fn a_step_of_two_hybrid_prompts_past_the_bound_is_cut_lane_by_lane_and_serves_alike() {
    use engine_gpu::submit::{Admitted, StepRows};
    const EACH: u32 = 256;
    let prompts: [Vec<u32>; 2] = [
        (0..EACH).map(|at| (at * 37 + 11) % 256).collect(),
        (0..EACH).map(|at| (at * 53 + 5) % 256).collect(),
    ];
    let step = StepRows {
        rows: 2 * EACH,
        lanes: 2,
        keys: EACH,
        prompt: true,
    };
    // Both prompts in one step held to `bound`, the load's first fire,
    // then a decode of both held to a roomy one: each lane's logits of
    // both, and the fires the prompts' step took past its own.
    let serve = |model: &mut Model, bound: f64| -> (Vec<Vec<f32>>, Vec<Vec<f32>>, u64) {
        let trace = model.trace.clone();
        let sets: Vec<(Vec<u32>, Vec<[u32; 2]>)> = (0..2)
            .map(|_| {
                let kv = (0..(EACH + 1).div_ceil(PAGE))
                    .map(|_| model.refs.take(LIST))
                    .collect();
                let record = (0..model.pieces)
                    .map(|_| {
                        let piece = model.refs.take(FIXED);
                        [piece, piece]
                    })
                    .collect();
                (kv, record)
            })
            .collect();
        let lane = |at: usize, tokens: Vec<u32>, held: u32, rs_reset: RsReset| Lane {
            slot: at as u32,
            word: word(&trace, tokens.len() as u32),
            kv: KvDelta {
                held,
                ..KvDelta::default()
            },
            readout: Readout::Last,
            tables: vec![(LIST, sets[at].0.clone())],
            record: engine::fire::LaneRecord::Pieces(vec![(FIXED, sets[at].1.clone())]),
            rs: RsVerb::Fold,
            rs_reset,
            tokens,
            ..Lane::default()
        };
        let mut fire = |lanes: Vec<Lane>, bound: f64| {
            model.engine.loaded_mut().expect("loaded").hold_to(bound);
            let mut ticket = model
                .engine
                .submit(&FrameSubmission::of(Step {
                    lanes,
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                }))
                .unwrap_or_else(|why| panic!("the step fires: {why}"));
            model.engine.settle_frame(&mut ticket).expect("the step settles");
            let step = &ticket.steps[0];
            (
                step.id.0,
                step.readouts.iter().map(|r| r.values.clone()).collect::<Vec<_>>(),
            )
        };
        let (at, prompted) = fire(
            (0..2)
                .map(|at| lane(at, prompts[at].clone(), 0, RsReset::Fresh))
                .collect(),
            bound,
        );
        assert_eq!(prompted.len(), 2, "both lanes read out");
        let (after, decoded) = fire(
            (0..2)
                .map(|at| {
                    let next = crate::model_logits::argmax(&prompted[at]);
                    lane(at, vec![next], EACH, RsReset::Held)
                })
                .collect(),
            2e9,
        );
        assert_eq!(after, at + 1, "the decode is the fire after the prompts' last");
        (prompted, decoded, at - 1)
    };
    let held_of = |model: &Model| {
        model
            .engine
            .loaded()
            .expect("loaded")
            .submit_bound()
            .expect("the bound the boot states")
            .clone()
    };
    let Some((mut roomy, _)) = load_under(1, Some(Duration::from_secs(2)), 2 * EACH) else {
        return;
    };
    let (whole, whole_decode, pieces) = serve(&mut roomy, 2e9);
    assert_eq!(pieces, 0, "a step within the bound is fired whole");
    drop(roomy);
    let Some((mut cut, _)) = load_under(1, Some(Duration::from_secs(2)), 2 * EACH) else {
        return;
    };
    // A bound between one row's layer and the two-prompt step's as `admit`
    // reads them once held (at the ratio the rates step measured alone: a
    // hold drops the scale the load's probe read off a device idle since
    // its boot): a layer of the step past it, a row's within it.
    cut.engine.loaded_mut().expect("loaded").hold_to(2e9);
    let held = held_of(&cut);
    let layer = |s: StepRows| held.step_ns(s).1;
    let one = StepRows {
        rows: 1,
        lanes: 1,
        keys: EACH,
        prompt: true,
    };
    let tight = (layer(one) * layer(step)).sqrt();
    eprintln!(
        "a layer of 1 row {:.3} ms, of two {EACH}-row prompts {:.3} ms; the bound {:.3} ms",
        layer(one) / 1e6,
        layer(step) / 1e6,
        tight / 1e6,
    );
    cut.engine.loaded_mut().expect("loaded").hold_to(tight);
    let held = held_of(&cut);
    assert!(
        matches!(held.admit(step), Admitted::Refused { .. }),
        "a layer of the step is past the bound: {:?}",
        held.admit(step)
    );
    let (pieced, pieced_decode, pieces) = serve(&mut cut, tight);
    assert!(pieces >= 2, "each prompt lane cut off in a piece at least: {pieces}");
    let cosine = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b)).max(f64::MIN_POSITIVE)
    };
    for lane in 0..2 {
        for (what, a, b) in [
            ("prompt", &whole[lane], &pieced[lane]),
            ("decode", &whole_decode[lane], &pieced_decode[lane]),
        ] {
            assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()), "lane {lane} {what}: finite");
            let (wa, wb) = (crate::model_logits::argmax(a), crate::model_logits::argmax(b));
            assert_eq!(wa, wb, "lane {lane} {what}: the same argmax whole and in pieces");
            let cos = cosine(a, b);
            assert!(cos > 0.999, "lane {lane} {what}: logits at cosine {cos}");
            eprintln!("lane {lane} {what}: cosine {cos:.6} whole against {pieces} + 1 pieces");
        }
    }
}
