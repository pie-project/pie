#![cfg(feature = "device")]
//! A lane's attention mask is data, not a class: a masked beam lane and plain
//! decode lanes share one fire and one body, and each answers what it answers
//! alone (contract A: a token parts only on a near-tie). A model opening the
//! mask arms as many bodies as the same model with no mask.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::{Mask, Masking, Readout};
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::contract::{Drift, near_tie};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const PROMPT: u32 = 40;
/// Plain lanes beside the beam.
const PLAIN: u32 = 3;
const LANES: u32 = PLAIN + 1;
const STEPS: u32 = 6;

/// One attention layer, its mask open or not.
struct Micro {
    mask: bool,
}

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
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, self.mask,
            None, None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

/// The shell over `lanes` lanes a fire, each with a slot pair and one
/// slot more.
fn shell(trace: poem::Trace, path: &Path, graphs: Graphs, lanes: u32) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(lanes, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: 2 * lanes + 1,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((graphs, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
        steps: engine::StepRules::runtime(),
    })
    .expect("the shell loads");
    shell
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(at, _)| at)
}

/// The beam's mask over `columns` keys: every key but a stretch of its
/// prompt (`8..16`), so a lane reading no mask answers otherwise.
fn beam_mask(columns: u32) -> Masking {
    Masking::Extent(Mask::new(
        vec![0, 8, 8, columns - 16],
        u64::from(columns),
    ))
}

/// Each lane's slot pair: batched in slot `l`, alone in slot `LANES + l`.
fn serve(graphs: Graphs) {
    let trace = trace_hybrid("mask-data-micro", &Micro { mask: true }, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2207).write(dir.path());
    let mut shell = shell(trace.clone(), &path, graphs, LANES);
    let pages: Vec<Vec<u32>> = (0..2 * LANES + 1)
        .map(|slot| {
            let per = 256 / PAGE;
            (slot * per..(slot + 1) * per).collect()
        })
        .collect();
    let word = |rows: u32| trace.facts.word(&Request::new(rows, false));
    let prompt = |lane: u32| -> Vec<u32> {
        (0..PROMPT)
            .map(|t| (t * 7 + 11 + lane * 13) % VOCAB)
            .collect()
    };
    let beam = PLAIN;
    // `(slot, tokens, held, mask)` a lane.
    type Row<'a> = (u32, &'a [u32], u32, Option<&'a Masking>);
    let fire = |shell: &mut Shell, lanes: &[Row<'_>]| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(slot, tokens, held, mask)| Seated {
                pages: &pages[slot as usize],
                held: Some(held),
                mask,
                ..Seated::of(Lane {
                    slot,
                    word: word(tokens.len() as u32),
                    tokens,
                })
            })
            .collect();
        let want: Vec<Readout> = lanes.iter().map(|_| Readout::Last).collect();
        shell.fire_rows(&seated, &want).expect("the fire serves")
    };

    let mut last = [[0u32; 2]; LANES as usize];
    for lane in 0..LANES {
        for (side, slot) in [(0usize, lane), (1, LANES + lane)] {
            shell.open(slot).expect("a slot opens");
            let tokens = prompt(lane);
            let rows = fire(&mut shell, &[(slot, &tokens, 0, None)]);
            last[lane as usize][side] = argmax(&rows[0]) as u32;
        }
    }
    let replays_before = shell.body_stats().tally.hits;
    let first_beam_token = last[beam as usize][1];
    let mut first_beam_alone = Vec::new();
    let (mut same, mut ties) = (0u32, 0u32);
    let mut parted = [false; LANES as usize];
    for step in 0..STEPS {
        let held = PROMPT + step;
        let mask = beam_mask(held + 1);
        let next: Vec<[u32; 1]> = (0..LANES as usize).map(|l| [last[l][0]]).collect();
        let rows: Vec<Row<'_>> = (0..LANES)
            .map(|l| {
                (
                    l,
                    &next[l as usize][..],
                    held,
                    (l == beam).then_some(&mask),
                )
            })
            .collect();
        let batched = fire(&mut shell, &rows);
        if matches!(graphs, Graphs::On) {
            let key = shell
                .replayed_body()
                .expect("the mixed fire replays a body")
                .to_string();
            assert!(
                !key.contains("~gm"),
                "the mixed fire replayed {key}, a masked twin"
            );
        }
        for lane in 0..LANES as usize {
            let token = [last[lane][1]];
            let alone = fire(
                &mut shell,
                &[(
                    LANES + lane as u32,
                    &token,
                    held,
                    (lane as u32 == beam).then_some(&mask),
                )],
            )
            .swap_remove(0);
            if step == 0 && lane as u32 == beam {
                first_beam_alone.clone_from(&alone);
            }
            let (a, b) = (&alone, &batched[lane]);
            let (ta, tb) = (argmax(a), argmax(b));
            if !parted[lane] && ta != tb {
                let gap = {
                    let mut top = [f32::NEG_INFINITY; 2];
                    for &v in a {
                        if v > top[0] {
                            top = [v, top[0]];
                        } else if v > top[1] {
                            top[1] = v;
                        }
                    }
                    top[0] - top[1]
                };
                let parting = near_tie(gap, &[ta, tb], Drift::Rows { a, b }, None);
                eprintln!("lane {lane} parts at step {step}: {parting:?}");
                assert!(
                    parting.tie,
                    "lane {lane} (beam: {}) parts at step {step} on no near-tie",
                    lane as u32 == beam
                );
                parted[lane] = true;
                ties += 1;
            } else if !parted[lane] {
                same += 1;
            }
            last[lane] = [tb as u32, ta as u32];
        }
    }
    eprintln!("{graphs:?}: {same} lane steps the same token, {ties} parted on a near-tie");
    if matches!(graphs, Graphs::On) {
        assert!(
            shell.body_stats().tally.hits > replays_before,
            "the decode steps replayed bodies"
        );
    }

    // The beam's mask is read: its prompt in one more slot, the beam's
    // first step stating none, answers otherwise than the beam alone did.
    let spare = 2 * LANES;
    shell.open(spare).expect("a slot opens");
    fire(&mut shell, &[(spare, &prompt(beam), 0, None)]);
    let token = [first_beam_token];
    let unmasked = fire(&mut shell, &[(spare, &token, PROMPT, None)]).swap_remove(0);
    let moved = unmasked
        .iter()
        .zip(&first_beam_alone)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        moved > 1e-3,
        "the beam's first step reads its mask: stating none moves its logits ({moved})"
    );
}

#[test]
fn a_masked_beam_beside_plain_decodes_answers_as_each_alone() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    serve(Graphs::Off);
    serve(Graphs::On);
}

/// Opening the mask adds no body: the masks are data each body reads.
#[test]
fn a_mask_adds_no_body() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let census = |mask: bool| {
        let trace = trace_hybrid("mask-data-micro", &Micro { mask }, Platform::Cuda);
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = Weights::random(&trace, 2207).write(dir.path());
        let shell = shell(trace, &path, Graphs::On, LANES);
        let stats = shell.body_stats();
        (stats.census.bodies, stats.tally.armed_at_load)
    };
    let (masked, plain) = (census(true), census(false));
    eprintln!("bodies masked {masked:?} plain {plain:?}");
    assert!(plain.0 > 0, "the plain load arms bodies");
    assert_eq!(
        masked, plain,
        "a model opening the mask arms the bodies (resident, armed) the maskless one does"
    );
}

/// A prompt and its decodes answer alike whether the model opens the mask or
/// not: a lane stating none reads through the masked arms as the plain ones.
#[test]
fn a_maskless_lane_answers_through_the_masked_arms_as_through_the_plain_ones() {
    maskless_through_masked_arms(LANES);
}

/// As [`a_maskless_lane_answers_through_the_masked_arms_as_through_the_plain_ones`]
/// on a load of one lane a fire: every fire's mask entries (one lane stating
/// none) are the ones the load's arming fires staged before the load zeroed
/// the workspace, and the first fire served copies them again.
#[test]
fn a_lone_maskless_lane_answers_through_the_masked_arms_as_through_the_plain_ones() {
    maskless_through_masked_arms(1);
}

fn maskless_through_masked_arms(lanes: u32) {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let serve = |mask: bool| -> Vec<Vec<f32>> {
        let trace = trace_hybrid("mask-data-micro", &Micro { mask }, Platform::Cuda);
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = Weights::random(&trace, 2207).write(dir.path());
        let mut shell = shell(trace.clone(), &path, Graphs::On, lanes);
        let pages: Vec<u32> = (0..256 / PAGE).collect();
        let word = |rows: u32| trace.facts.word(&Request::new(rows, false));
        shell.open(0).expect("a slot opens");
        let mut out = Vec::new();
        let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
        let mut held = 0;
        for tokens in [prompt, vec![5], vec![9], vec![17]] {
            let seated = [Seated {
                pages: &pages,
                held: Some(held),
                ..Seated::of(Lane { slot: 0, word: word(tokens.len() as u32), tokens: &tokens })
            }];
            out.push(shell.fire_rows(&seated, &[Readout::Last]).expect("fires").swap_remove(0));
            held += tokens.len() as u32;
        }
        out
    };
    let (masked, plain) = (serve(true), serve(false));
    for (step, (a, b)) in masked.iter().zip(&plain).enumerate() {
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let span = b.iter().copied().fold(f32::MIN, f32::max) - b.iter().copied().fold(f32::MAX, f32::min);
        eprintln!("step {step}: worst {worst} of a span {span}");
        assert!(a.iter().all(|x| x.is_finite()), "step {step}: the masked arms answer a non-finite logit");
        assert!(worst <= span * 1e-2, "step {step}: the masked arms answer otherwise ({worst})");
    }
}
