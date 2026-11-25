#![cfg(feature = "device")]
//! A plain lane runs its class's plain body whether or not the load seats
//! adapters; a fire with an adapted lane, alone or beside a plain one, walks
//! eager, as the load states, and each lane answers what it answers alone.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::Readout;
use engine::live::Tally;
use engine_cuda::adapter::{planes_of, sink_of};
use engine_cuda::{AdapterPlane, AdapterSource, Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::Shape;
use poem::{
    Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, fact, ops,
    trace_hybrid,
};
use poem_compiler::Budget;
use poem_ir::Builtin;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const PROMPT: u32 = 24;
const STEPS: u32 = 4;
const SLOTS: u64 = 4;
const RANK: u64 = 16;
const BANK_A: &str = "layer.0.lora_a";
const BANK_B: &str = "layer.0.lora_b";

/// One attention layer whose output an adapted row corrects.
struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, false,
            None, None, false, false,
        );
        let o = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let bank_a = Weight::sym(BANK_A, [SLOTS, RANK, WIDTH], Dtype::Bf16).registered();
        let bank_b = Weight::sym(BANK_B, [SLOTS, WIDTH, RANK], Dtype::Bf16).registered();
        let adapted = fact::has(Builtin::Adapted);
        let o = ops::linear::lora_correct(
            &x.on(adapted.clone()),
            &bank_a,
            &bank_b,
            &inputs.adapter_routes(),
            &o.on(adapted),
        );
        let h = ops::elemwise::residual_add(&o, &x);
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, lanes: u32) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    Shell::load(Boot {
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::On, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(trace, &contract, path, Budget::new(lanes, 64), 256, 8)
    })
    .expect("the shell loads")
}

/// The adapter a program's `lora` sink binds: its planes seeded into the
/// channels the sink reads, landed as the engine's bind lands them. The
/// load states no adapter capacity.
fn bind_through_a_sink(shell: &mut Shell) -> u32 {
    let chan = |shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(eta_ir::types::Dtype::F32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: true,
    };
    let (rank, width) = (RANK as u32, WIDTH as u32);
    let container = TraceContainer {
        names: vec!["lora".to_string()],
        channels: vec![
            chan(Shape::new(&[1, rank, width]).expect("A is [layers, rank, hidden]")),
            chan(Shape::new(&[1, width, rank]).expect("B is [layers, hidden, rank]")),
            chan(Shape::vector(1)),
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::ChanRead(2),
                Op::SinkCall { name: 0, args: vec![0, 1, 2] },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let bound = crate::bound::logits(container, VOCAB).expect("the lora prologue binds");
    let package = eta_compiler::codegen::launch::build(&bound, &compile_bound(&bound));
    let sink = sink_of(&package).expect("the sink reads").expect("the program has a sink");
    let seeds = [plane(RANK * WIDTH, 7, 0.5), plane(WIDTH * RANK, 13, 0.5)];
    let seats = shell.bank_seats();
    let site = sink.site().expect("the sink names one site or none");
    let built: Vec<(String, Vec<u8>)> = sink
        .planes
        .iter()
        .flat_map(|(role, channel)| {
            planes_of(*role, site, &seeds[*channel as usize], &seats).expect("a plane fits")
        })
        .collect();
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane { bank, bytes })
        .collect();
    shell
        .bind_adapter(AdapterSource::Own { instance: engine::InstanceId(1), planes: &planes })
        .expect("a load stating no capacity binds a sink's adapter")
        .slot
}

/// An f32 plane of `count` values from a seeded ramp, `scale` wide.
fn plane(count: u64, seed: u32, scale: f32) -> Vec<u8> {
    (0..count as u32)
        .flat_map(|at| {
            let unit = ((at.wrapping_mul(2_654_435_761).wrapping_add(seed)) >> 8) as f32
                / (1u32 << 24) as f32;
            ((unit * 2.0 - 1.0) * scale).to_le_bytes()
        })
        .collect()
}

fn worst(a: &[f32], b: &[f32]) -> (f32, f32) {
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let span = b.iter().copied().fold(f32::MIN, f32::max) - b.iter().copied().fold(f32::MAX, f32::min);
    (worst, span)
}

#[test]
fn an_adapted_lane_beside_a_plain_one_answers_as_alone() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = trace_hybrid("adapter-data-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4111).write(dir.path());
    let mut shell = shell(trace.clone(), &path, 2);
    let adapter = bind_through_a_sink(&mut shell);

    let pages: Vec<Vec<u32>> = (0..8u32)
        .map(|slot| {
            let per = 256 / PAGE;
            (slot * per..(slot + 1) * per).collect()
        })
        .collect();
    let word = |rows: u32, adapted: bool| {
        trace
            .facts
            .word(&Request::new(rows, false).adapted(adapted))
    };
    // `(slot, tokens, held, adapter)` a lane.
    type Row<'a> = (u32, &'a [u32], u32, Option<u32>);
    let fire = |shell: &mut Shell, lanes: &[Row<'_>]| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(slot, tokens, held, adapter)| Seated {
                pages: &pages[slot as usize],
                held: Some(held),
                adapter,
                ..Seated::of(Lane {
                    slot,
                    word: word(tokens.len() as u32, adapter.is_some()),
                    tokens,
                })
            })
            .collect();
        let want: Vec<Readout> = lanes.iter().map(|_| Readout::Last).collect();
        shell.fire_rows(&seated, &want).expect("the fire serves")
    };
    let prompt = |lane: u32| -> Vec<u32> {
        (0..PROMPT)
            .map(|t| (t * 7 + 11 + lane * 13) % VOCAB)
            .collect()
    };

    // Lane 0 plain, lane 1 adapted: batched in slots 0 and 1, alone in 2
    // and 3; the adapted lane's prompt without its adapter in slot 4.
    let adapter_of = |lane: u32| (lane == 1).then_some(adapter);
    let mut tokens = [[0u32; 2]; 2];
    for lane in 0..2u32 {
        for (side, slot) in [(0usize, lane), (1, 2 + lane)] {
            shell.open(slot).expect("a slot opens");
            let rows = fire(&mut shell, &[(slot, &prompt(lane), 0, adapter_of(lane))]);
            tokens[lane as usize][side] = argmax(&rows[0]);
        }
    }
    shell.open(4).expect("a slot opens");
    let bare = fire(&mut shell, &[(4, &prompt(1), 0, None)]).swap_remove(0);

    // Every fire with the adapter walks eager, counted as stated.
    let mut adapted_fires = 2u64;
    for step in 0..STEPS {
        let held = PROMPT + step;
        let next: Vec<[u32; 1]> = (0..2).map(|lane| [tokens[lane][0]]).collect();
        let hits = shell.body_stats().tally.hits;
        let batched = fire(
            &mut shell,
            &[
                (0, &next[0][..], held, adapter_of(0)),
                (1, &next[1][..], held, adapter_of(1)),
            ],
        );
        adapted_fires += 1;
        assert_eq!(shell.body_stats().tally.hits, hits, "step {step}: the mixed fire replayed a body");
        for lane in 0..2u32 {
            let token = [tokens[lane as usize][1]];
            let hits = shell.body_stats().tally.hits;
            let alone = fire(&mut shell, &[(2 + lane, &token, held, adapter_of(lane))]).swap_remove(0);
            let replayed = shell.body_stats().tally.hits > hits;
            assert_eq!(replayed, lane == 0, "step {step}: the plain lane alone replays its body");
            adapted_fires += u64::from(lane == 1);
            let (worst, span) = worst(&batched[lane as usize], &alone);
            eprintln!("step {step} lane {lane}: worst {worst} of a span {span}");
            assert!(
                worst <= span * 1e-2,
                "step {step}: lane {lane} batched answers otherwise than alone ({worst} of {span})"
            );
            tokens[lane as usize] = [argmax(&batched[lane as usize]), argmax(&alone)];
        }
    }

    // Two lanes of one prompt, both adapted, answer as one does alone.
    for slot in [5, 6, 7] {
        shell.open(slot).expect("a slot opens");
    }
    let both = fire(
        &mut shell,
        &[(5, &prompt(1), 0, Some(adapter)), (6, &prompt(1), 0, Some(adapter))],
    );
    let alone = fire(&mut shell, &[(7, &prompt(1), 0, Some(adapter))]).swap_remove(0);
    adapted_fires += 2;
    for (at, lane) in both.iter().enumerate() {
        let (worst, span) = worst(lane, &alone);
        eprintln!("all adapted, lane {at}: worst {worst} of a span {span}");
        assert!(worst <= span * 1e-2, "lane {at} ({worst} of {span})");
    }

    // The adapter is read: the adapted lane's prompt without it answers
    // otherwise than with it.
    let with = alone;
    let (moved, span) = worst(&with, &bare);
    eprintln!("the adapter moves the prompt's logits {moved} of a span {span}");
    assert!(moved > span * 5e-2, "the adapter moves the logits ({moved} of {span})");

    eprintln!("census: {:?}", shell.body_stats().tally);
    assert_eq!(shell.tallied(Tally::Eager), adapted_fires, "every adapted fire is stated eager");
    assert_eq!(shell.tallied(Tally::Unrecorded), 0, "no fire walked unstated");
    assert_eq!(shell.tallied(Tally::Compiled), 0, "no serving fire captured a body");
    assert_eq!(shell.tallied(Tally::BuiltInFire), 0, "no fire built a kernel");
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(at, _)| at as u32)
}
