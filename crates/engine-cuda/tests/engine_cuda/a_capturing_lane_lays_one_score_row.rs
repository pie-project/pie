#![cfg(feature = "device")]

//! A fire of three lanes, one capturing its attention scores, lays one score
//! row (in its reading program's span) and a fire capturing none lays none.
//! The row a program reads, at key grain (`attn.scores`) and page grain
//! (`attn.scores.pages`), is what the CPU oracle answers for the lane, at
//! its prefill and at a decode after it.

use crate::common_dit::{NAME, Rig, Weights};
use engine::Engine;
use engine::fire::{Attachment, FrameSubmission, Lane, Step};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Trace, Weight, ops};

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
/// The Rig's load: 64 keys a lane, in pages of 16.
const CONTEXT: u32 = 64;
const PAGE: u32 = 16;
const PAGES: u32 = CONTEXT / PAGE;

/// Two attention layers over one kv row each, the scores captured.
struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(KV_HEADS) * u64::from(HEAD_DIM);
        spec.kv("kv0", Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec.kv("kv1", Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let q_plane = u64::from(Q_HEADS) * u64::from(HEAD_DIM);
        let kv_plane = u64::from(KV_HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let mut x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        for layer in 0..2 {
            let kv = format!("{KV_ROW}{layer}");
            let q = ops::linear::matmul(&x, &w(&format!("{layer}.q"), q_plane, WIDTH));
            let k = ops::linear::matmul(&x, &w(&format!("{layer}.k"), kv_plane, WIDTH));
            let v = ops::linear::matmul(&x, &w(&format!("{layer}.v"), kv_plane, WIDTH));
            ops::attn::kv_append(
                &k,
                Some(&v),
                inputs.kv(&kv),
                &inputs.write_page(&kv),
                &inputs.write_offset(&kv),
            );
            let a = ops::attn::attend(
                &q, &inputs, &kv, Q_HEADS, KV_HEADS, HEAD_DIM, SM_SCALE, None, None, None, false,
                None, None, false, true,
            );
            let h = ops::linear::matmul(&a, &w(&format!("{layer}.o"), WIDTH, q_plane));
            x = ops::elemwise::residual_add(&x, &h);
        }
        let logits = ops::linear::lm_head(&x, &table);
        poem::output("logits", &logits, None);
    }
}

fn reader(width: u32) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::matrix(1, width),
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

/// An epilogue putting its lane's key row and page row on its channels.
fn reading() -> TraceContainer {
    let output = |index: u16, width: u32| Op::Output {
        index,
        lane: 0,
        shape: Shape::matrix(1, width),
        dtype: EtaDtype::F32,
    };
    TraceContainer {
        names: vec!["attn.scores".to_string(), "attn.scores.pages".to_string()],
        channels: vec![reader(CONTEXT), reader(PAGES)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                output(0, CONTEXT),
                output(1, PAGES),
                Op::ChanPut { chan: 0, value: 0 },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn lane(trace: &Trace, slot: u32, tokens: &[u32], captures: bool) -> Lane {
    let request = Request::new(tokens.len() as u32, false).capturing_scores(captures);
    Lane {
        slot,
        word: trace.facts.word(&request),
        tokens: tokens.to_vec(),
        ..Lane::default()
    }
}

fn fire(rig: &mut Rig, lanes: Vec<Lane>, attachments: Vec<Attachment>) {
    let mut ticket = rig
        .engine
        .submit(&FrameSubmission::of(Step {
            lanes,
            attachments,
            media: Vec::new(),
            voxels: Vec::new(),
        }))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
}

fn captured(rig: &Rig) -> u32 {
    rig.engine
        .shell()
        .expect("the engine holds a load")
        .captured()
        .0
}

/// The oracle's key row and page row for one step of its lane.
fn oracle(
    cpu: &mut engine_cpu::Forward,
    trace: &Trace,
    tokens: &[u32],
    start: u32,
) -> [Vec<f32>; 2] {
    let word = trace
        .facts
        .word(&Request::new(tokens.len() as u32, false).capturing_scores(true));
    let outs = cpu.step(tokens, start, word).expect("the oracle steps");
    let floats = |name: &str| match outs.iter().find(|out| out.name == name) {
        Some(engine_cpu::forward::Output {
            values: engine_cpu::forward::Values::F32(v),
            ..
        }) => v.clone(),
        _ => panic!("the oracle reads out no `{name}`"),
    };
    [floats("attn.scores"), floats("attn.scores.pages")]
}

fn assert_row(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: width");
    for (at, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-3 + 2e-2 * w.abs(),
            "{what} at {at}: device {g} against the oracle's {w}"
        );
    }
}

#[test]
fn a_capturing_lane_lays_one_score_row() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid(NAME, &Micro, Platform::Cuda);
    let weights = Weights::random(&trace, 0x5c0e);
    let mut rig = Rig::load_plan(trace.clone(), &weights, 64, Vec::new());
    assert_eq!(rig.reads_out("attn.scores"), Some(CONTEXT));
    assert_eq!(rig.reads_out("attn.scores.pages"), Some(PAGES));

    let program = rig.register(reading(), 0x5c);
    // A program reading a lane's scores states its rows a member: the key
    // row with its table entry, and the page row.
    let extents = engine::program::BindExtents {
        row_count: 7,
        token_count: 7,
        sampled_rows: 1,
        query_len: 7,
        ..engine::program::BindExtents::default()
    };
    let facts = rig.engine.fire_facts(program, &extents).expect("its facts");
    let row = u64::from(CONTEXT) * 4 + 4 + u64::from(PAGES) * 4;
    assert!(
        facts.a_lane >= row,
        "a reading member states {} bytes a lane, and its rows are {row}",
        facts.a_lane
    );
    let instance = rig.bind_readers(program, &[vec![1, CONTEXT], vec![1, PAGES]], 7);

    let prompt: Vec<u32> = (0..7).map(|at| (at * 13 + 5) % VOCAB).collect();
    let second: Vec<u32> = (0..5).map(|at| (at * 7 + 3) % VOCAB).collect();
    let third: Vec<u32> = (0..3).map(|at| (at * 11 + 1) % VOCAB).collect();
    let mut cpu =
        engine_cpu::Forward::new(trace.clone(), None, &|name, _| weights.get(name).to_vec())
            .expect("the oracle loads")
            .at_context(CONTEXT, PAGE);

    // Three lanes, the first capturing: one row.
    fire(
        &mut rig,
        vec![
            lane(&trace, 0, &prompt, true),
            lane(&trace, 1, &second, false),
            lane(&trace, 2, &third, false),
        ],
        vec![Attachment {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }],
    );
    assert_eq!(captured(&rig), 1, "one capturing lane, one row");
    let [keys, pages] = oracle(&mut cpu, &trace, &prompt, 0);
    assert_row(&rig.take(instance, 0), &keys, "the prefill's key row");
    assert_row(&rig.take(instance, 1), &pages, "the prefill's page row");
    let mass: f32 = keys.iter().sum();
    assert!(
        (mass - 2.0).abs() < 1e-3,
        "two layers, a mass of one each: {mass}"
    );

    // The other lanes decode, none capturing: no row.
    fire(
        &mut rig,
        vec![lane(&trace, 1, &[9], false), lane(&trace, 2, &[17], false)],
        Vec::new(),
    );
    assert_eq!(captured(&rig), 0, "a fire capturing nothing lays no row");

    // The capturing lane decodes beside a plain one: its row again.
    fire(
        &mut rig,
        vec![lane(&trace, 1, &[21], false), lane(&trace, 0, &[33], true)],
        vec![Attachment {
            lane: 1,
            lanes: 1,
            instance,
            adapter_only: false,
        }],
    );
    assert_eq!(captured(&rig), 1, "one capturing lane, one row");
    let [keys, pages] = oracle(&mut cpu, &trace, &[33], prompt.len() as u32);
    assert_row(&rig.take(instance, 0), &keys, "the decode's key row");
    assert_row(&rig.take(instance, 1), &pages, "the decode's page row");
}
