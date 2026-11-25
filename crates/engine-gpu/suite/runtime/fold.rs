//! A buffering lane whose fold length is the device's replays what one
//! folding the same count on the host replays, bit for bit, when the
//! runtime's count lags: the device-folding fires of one set are pipelined
//! (the next submitted before the previous settles), each stating the
//! `folded` the runtime held at submit, before the report of the fold it
//! carries was heard. Each device-folding fire reports the slots its fold
//! froze through the completion sink: as its step lands on an engine that
//! settles off the submit, and before `submit` returns on one that settles
//! inside it (one frame in flight). And under a stated submit bound, a step
//! of hybrid prompts past it is cut and serves what it served whole.
//! qwen35-tiny (recurrent layers); the fold count rides a decode-envelope
//! program's `fold` port.

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use engine::Engine;
use engine::channel::ChannelId;
use engine::chunk::{ChunkFacts, Growth, ShapeId, SlotKind};
use engine::fire::{
    Attachment, FoldLen, FrameSubmission, FrameTicket, KvDelta, Lane, Readout, RsReset, RsVerb,
    Step,
};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, InstanceId};
use engine::{StepDone, StepOutcome};
use engine_gpu::submit::{Admitted, StepRows};
use engine_gpu::store::layout::{FIXED, LIST, RS};
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Port, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem_ir::Trace;

use super::outputs::{channel, contract_for, registration_of};
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
/// The sets a load serves (the cut step's lanes).
const SETS: u32 = 2;

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
            PortBinding { port: Port::EmbedTokens, source: PortSource::Channel(0) },
            PortBinding {
                port: Port::EmbedIndptr,
                source: PortSource::Const { dtype: EtaDtype::U32, shape: Shape::vector(2), data: indptr },
            },
            PortBinding { port: Port::Fold, source: PortSource::Channel(1) },
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
    let program = engine::Programs::register(engine, &registration_of(&[], fold_program(), 0xf01d))
        .expect("the fold program registers");
    let channels: Vec<ChannelId> = [ROWS, 1]
        .iter()
        .enumerate()
        .map(|(at, &width)| {
            let id = ChannelId((1 << 20) + at as u64);
            slots.register(engine, channel(id, vec![width], EtaDtype::U32, HostRole::Writer, 2));
            id
        })
        .collect();
    let binding = slots.binding(
        engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            geometry: GeometryClass::DecodeEnvelope,
            extents: engine::program::BindExtents {
                row_count: ROWS,
                token_count: ROWS,
                sampled_rows: ROWS,
                query_len: ROWS,
                ..engine::program::BindExtents::default()
            },
            ..InstanceBinding::default()
        },
        &channels,
    );
    engine::Programs::bind(engine, &binding).expect("the fold program binds an instance").id
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
        let per = self.facts.shape(id).unwrap_or_else(|| panic!("{MODEL} lays out shape {id}")).slots_per_chunk;
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
    (0..ROWS).map(|at| (seed * 29 + at * 37 + 11) % 256).collect()
}

/// One sequence: its kv pages, the rs-list slots its buffer holds from the
/// prompt's end on, and its record (`[run, cow]` a piece).
struct Set {
    slot: u32,
    held: u32,
    kv: Vec<u32>,
    rs: Vec<u32>,
    record: Vec<[u32; 2]>,
}

impl Set {
    /// A set of its own slots for `rows` rows, nothing held.
    fn of(model: &mut Model, slot: u32, rows: u32) -> Set {
        let kv = (0..rows.div_ceil(PAGE)).map(|_| model.refs.take(LIST)).collect();
        let rs = (0..model.rs_pages).map(|_| model.refs.take(RS)).collect();
        let record = (0..model.pieces)
            .map(|_| {
                let piece = model.refs.take(FIXED);
                [piece, piece]
            })
            .collect();
        Set { slot, held: 0, kv, rs, record }
    }

    /// A set whose prompt of `PROMPT` rows is folded: every set's the same,
    /// so sets are told apart by their fires alone.
    fn new(model: &mut Model, slot: u32) -> Set {
        let mut set = Set::of(model, slot, PROMPT + 3 * ROWS);
        let prompt: Vec<u32> = (0..PROMPT).map(|at| (at * 53 + 5) % 256).collect();
        let lane = set.rows(&model.trace, prompt, RsReset::Fresh);
        model.fire(lane);
        set.held = PROMPT;
        set
    }

    /// A folding lane of `tokens` after the rows the set holds.
    fn rows(&self, trace: &Trace, tokens: Vec<u32>, rs_reset: RsReset) -> Lane {
        let upto = self.held + tokens.len() as u32;
        Lane {
            slot: self.slot,
            word: word(trace, tokens.len() as u32),
            kv: KvDelta { held: self.held, ..KvDelta::default() },
            readout: Readout::Last,
            tables: vec![(LIST, self.kv[..upto.div_ceil(PAGE) as usize].to_vec())],
            record: engine::fire::LaneRecord::Pieces(vec![(FIXED, self.record.clone())]),
            rs: RsVerb::Fold,
            rs_reset,
            tokens,
            ..Lane::default()
        }
    }

    /// A buffering lane of `ROWS` rows whose state holds the slots below
    /// `folded`, folding `fold`; it reports under `reports` (0: no report).
    fn lane(&mut self, trace: &Trace, seed: u32, folded: u32, fold: FoldLen, reports: u64) -> Lane {
        let pages = self.rs.clone();
        let lineage = engine::fire::Lineage::Set;
        let lane = Lane {
            rs: RsVerb::Buffer { pages, base: PROMPT, folded, fold, lineage },
            reports,
            ..self.rows(trace, tokens(seed), RsReset::Held)
        };
        self.held += ROWS;
        lane
    }
}

/// The loaded miniature, the slots its sets take and what its completion
/// sink heard.
struct Model {
    engine: backend::Engine,
    trace: Trace,
    refs: Refs,
    pieces: u32,
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
        let attachments = attach
            .map(|instance| Attachment { lane: 0, lanes: 1, instance, adapter_only: false, scratch: ::engine::ScratchPlace::None })
            .into_iter()
            .collect();
        let step = Step { lanes: vec![lane], attachments, media: Vec::new(), voxels: Vec::new(), shape: None };
        let ticket = self.engine.submit(&FrameSubmission::of(step)).expect("the frame fires");
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
        ticket.steps[0].readouts[0].values.iter().map(|v| v.to_bits()).collect()
    }

    /// Fires a host lane and settles it: it reports no fold.
    fn fire(&mut self, lane: Lane) -> Vec<u32> {
        let mut ticket = self.submit(lane, None);
        let bits = self.settled(&mut ticket);
        let outcome = self.outcome(&ticket);
        assert_eq!(outcome, self.asynchronous.then(StepOutcome::committed), "a host-folding fire reports no fold");
        bits
    }

    /// Publishes `lane`'s tokens and the fold length on the program's
    /// channels.
    fn tell(&mut self, instance: InstanceId, lane: &Lane) {
        let tokens: Vec<u8> = lane.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        for (channel, bytes) in [(0, tokens), (1, N.to_le_bytes().to_vec())] {
            let published = engine::Channels::publish(&mut self.engine, instance, channel, &bytes);
            assert!(published.expect("the cell publishes"), "the ring had room");
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
            let heard = self.heard.recv_timeout(Duration::from_secs(60));
            self.seen.push(heard.expect("the step's completion reaches the sink"));
        }
    }

    /// The slots the fold of `ticket`'s step froze, as its one lane
    /// reported them under `tag`.
    fn reported(&mut self, ticket: &FrameTicket, tag: u64) -> Option<u32> {
        match self.outcome(ticket) {
            Some(StepOutcome { wrote, faulted, .. }) if wrote.len() == 1 && faulted.is_empty() => {
                assert_eq!(wrote[0].tag, tag, "the report carries the lane's tag");
                wrote[0].fold
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

    /// The bound the load holds its steps to, held to `bound`.
    fn held(&mut self, bound: f64) -> engine_gpu::submit::SubmitBound {
        let shell = self.engine.loaded_mut().expect("loaded");
        shell.hold_to(bound);
        shell.submit_bound().expect("the bound the boot states").clone()
    }
}

/// The miniature loaded with `frames` in flight for fires of up to
/// `tokens` rows, on a device whose platform states `bound` on one GPU
/// submission where given (the boot's, `test-limits`).
fn load(frames: u8, bound: Option<Duration>, tokens: u32) -> Option<(Model, InstanceId)> {
    let boot = crate::backend::DeviceBoot { submit_bound: bound, ..crate::gpu::device_boot() };
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
                page_size: PAGE,
                max_context: 64.max(2 * tokens),
                slots: SETS + 1,
                ..Budgets::default()
            },
            residency: Residency { device: crate::gpu::share(), ..Residency::default() },
            ordinal: 0,
            frames_in_flight: frames,
            steps: Default::default(),
        })
        .expect("the miniature loads");
    let asynchronous = engine::Completion::asynchronous(&engine);
    let (tx, heard) = mpsc::channel();
    let tx = Mutex::new(tx);
    engine::Completion::on_complete(&mut engine, Arc::new(move |at, outcome| {
        let _ = tx.lock().expect("the sink's sender").send((at, outcome));
    }));

    let facts = loaded.caps.chunks.clone();
    let Some(SlotKind::Fixed { pieces }) = facts.shape(FIXED).map(|shape| shape.kind) else {
        panic!("{MODEL}'s record shape is fixed");
    };
    let Some(SlotKind::List { tokens: rs_tokens, .. }) = facts.shape(RS).map(|shape| shape.kind) else {
        panic!("{MODEL}'s rs list is a list");
    };
    let rs_pages = (3 * ROWS).div_ceil(rs_tokens) + 1;
    // Segments of the test's own above the committed ones: the guest's two
    // chunks, then each shape's (a set's kv pages, its rs slots and its
    // record).
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
        let seats = 1 + engine.loaded().expect("the shell stands").pool_layout().seat_chunks();
        let first = seats.div_ceil(segment) * segment;
        (first, (first + need).div_ceil(segment), facts.segments.max - 1)
    } else {
        let committed = facts.segments.committed;
        (committed * segment, committed + need.div_ceil(segment), facts.segments.max)
    };
    assert!(most >= commit, "the test's chunks span {commit} segments, the pool has {most} for them");
    engine::Pool::commit(&mut engine, commit).expect("the segments commit");
    let mut slots = Slots::of(&facts, first, first + 1);
    let instance = bind_fold(&mut engine, &mut slots);
    let model = Model {
        engine,
        trace: fixture.trace,
        refs: Refs { chunk: first + 2, end: commit * segment, facts, open: Vec::new() },
        pieces,
        rs_pages,
        asynchronous,
        heard,
        seen: Vec::new(),
    };
    Some((model, instance))
}

/// Two frames in flight (the engine settles off the submit, a fire staged
/// behind one still in flight), then one (an engine that then settles each
/// frame inside `submit` has told a fold's report by the time it returns).
#[test]
fn a_device_fold_replays_what_a_host_fold_replays() {
    for frames in [2, 1] {
        let Some((mut model, instance)) = load(frames, None, PAGE) else {
            return;
        };
        let trace = model.trace.clone();
        // Two buffering fires folding N each. The host's counts are known
        // at submit, so the runtime's `folded` is exact.
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
        // The device's ride the program's `fold` port: the second is
        // submitted before the first's report is read, stating the `folded`
        // the runtime held then.
        let mut device = Set::new(&mut model, 1);
        let a = device.lane(&trace, 11, PROMPT, FoldLen::Device, 101);
        model.tell(instance, &a);
        let first = model.submit(a, Some(instance));
        let b = device.lane(&trace, 12, PROMPT, FoldLen::Device, 102);
        model.tell(instance, &b);
        let mut second = model.submit(b, Some(instance));
        // A step whose guest pass runs behind its forward hands the host
        // no rows.
        let got = model.settled(&mut second);
        assert!(got.is_empty() || got == rows, "{frames}: the second fire replays from where the first left");
        assert_eq!(model.reported(&first, 101), Some(N), "{frames}: the first fire's fold");
        assert_eq!(model.reported(&second, 102), Some(N), "{frames}: the second fire's own fold");
        assert!(
            model.probe(&mut device, PROMPT + 2 * N) == want,
            "{frames} frame(s) in flight: a pipelined device fold leaves what the host fold leaves"
        );
    }
}

/// Under a stated submit bound small enough that one layer of a step of two
/// 256-row qwen35-tiny prompts (attention and gated-delta layers) is past
/// it, the step is cut lane by lane, then by rows (`engine_gpu::submit::
/// next_cut`), each lane's recurrent state carried in its own record: both
/// prompts' logits, and a decode of both after, are what the step served
/// whole under a roomy bound. The tight load is held to a bound between a
/// row's layer and the step's at the ratio its rates step measured alone
/// (`Shell::hold_to`: a desktop GPU's idle probe swings 1x to 7x).
#[test]
fn a_step_of_hybrid_prompts_past_the_bound_is_cut_and_serves_alike() {
    const EACH: u32 = 256;
    let step = StepRows { rows: 2 * EACH, lanes: 2, keys: EACH, prompt: true };
    // Both prompts in one step held to `bound`, then a decode of both held
    // to a roomy one: each lane's logits of both, and the fires the
    // prompts' step took past its own.
    let serve = |model: &mut Model, bound: f64| -> (Vec<Vec<f32>>, u64) {
        let trace = model.trace.clone();
        let mut sets: Vec<Set> = (0..SETS).map(|slot| Set::of(model, slot, EACH + 1)).collect();
        let mut fire = |lanes: Vec<Lane>, bound: f64| {
            model.held(bound);
            let step = Step { lanes, attachments: Vec::new(), media: Vec::new(), voxels: Vec::new(), shape: None };
            let mut ticket = model.engine.submit(&FrameSubmission::of(step)).expect("the step fires");
            model.engine.settle_frame(&mut ticket).expect("the step settles");
            let step = &ticket.steps[0];
            (step.id.0, step.readouts.iter().map(|r| r.values.clone()).collect::<Vec<_>>())
        };
        let prompts = sets
            .iter()
            .map(|set| {
                let prompt = (0..EACH).map(|at| (at * (37 + 16 * set.slot) + 11) % 256).collect();
                set.rows(&trace, prompt, RsReset::Fresh)
            })
            .collect();
        let (at, mut logits) = fire(prompts, bound);
        assert_eq!(logits.len(), 2, "both lanes read out");
        let decodes = sets
            .iter_mut()
            .zip(&logits)
            .map(|(set, row)| {
                set.held = EACH;
                set.rows(&trace, vec![crate::model_logits::argmax(row)], RsReset::Held)
            })
            .collect();
        let (after, decoded) = fire(decodes, 2e9);
        assert_eq!(after, at + 1, "the decode is the fire after the prompts' last");
        logits.extend(decoded);
        (logits, at - 1)
    };
    let Some((mut roomy, _)) = load(1, Some(Duration::from_secs(2)), 2 * EACH) else {
        return;
    };
    let (whole, pieces) = serve(&mut roomy, 2e9);
    assert_eq!(pieces, 0, "a step within the bound is fired whole");
    drop(roomy);
    let Some((mut cut, _)) = load(1, Some(Duration::from_secs(2)), 2 * EACH) else {
        return;
    };
    let held = cut.held(2e9);
    let layer = |s: StepRows| held.step_ns(s).1;
    let tight = (layer(StepRows { rows: 1, lanes: 1, ..step }) * layer(step)).sqrt();
    let held = cut.held(tight);
    assert!(matches!(held.admit(step), Admitted::Refused { .. }), "a layer of the step is past the bound");
    let (pieced, pieces) = serve(&mut cut, tight);
    assert!(pieces >= 2, "each prompt lane cut off in a piece at least: {pieces}");
    let cosine = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b)).max(f64::MIN_POSITIVE)
    };
    for (at, (a, b)) in whole.iter().zip(&pieced).enumerate() {
        let what = format!("lane {} {}", at % 2, if at < 2 { "prompt" } else { "decode" });
        assert!(a.iter().chain(b).all(|v| v.is_finite()), "{what}: finite");
        let (wa, wb) = (crate::model_logits::argmax(a), crate::model_logits::argmax(b));
        assert_eq!(wa, wb, "{what}: the same argmax whole and in pieces");
        let cos = cosine(a, b);
        assert!(cos > 0.999, "{what}: logits whole and in pieces at cosine {cos}");
    }
}
