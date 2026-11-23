#![cfg(target_vendor = "apple")]
//! A lane's attention mask is data, not a class: a masked beam lane and
//! plain decode lanes share one fire, and each answers what it answers alone
//! (contract A: a token parts only on a near-tie). The paged decode reads
//! each row's mask word once, and a key's bit only past the lane's boundary,
//! so a lane stating no mask reads every key in the same arm.

use engine::fire::{Mask, Masking};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use eta_compiler::contract::{Drift, near_tie};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, argmax, fixture_from};

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const PROMPT: u32 = 40;
/// Plain lanes beside the beam.
const PLAIN: u32 = 3;
const LANES: u32 = PLAIN + 1;
/// Each lane batched and alone, one more for the unmasked beam and one for
/// the mixed fire's prompt.
const SLOTS: u32 = 2 * LANES + 2;
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

fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(LANES + 1, 64),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: SLOTS,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

/// The beam's mask over `columns` keys: every key but a stretch of its
/// prompt (`8..16`), so a lane reading no mask answers otherwise.
fn beam_mask(columns: u32) -> Masking {
    Masking::Extent(Mask::new(vec![0, 8, 8, columns - 16], u64::from(columns)))
}

fn prompt(lane: u32) -> Vec<u32> {
    (0..PROMPT).map(|t| (t * 7 + 11 + lane * 13) % VOCAB).collect()
}

/// `(slot, tokens, held, mask)` a lane.
type Row<'a> = (u32, &'a [u32], u32, Option<&'a Masking>);

fn fire(shell: &mut Shell, trace: &poem::Trace, lanes: &[Row<'_>]) -> Vec<Vec<f32>> {
    let seated: Vec<Seated<'_>> = lanes
        .iter()
        .map(|&(slot, tokens, held, mask)| Seated {
            held: Some(held),
            mask,
            ..Seated::of(Lane {
                slot,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens,
            })
        })
        .collect();
    shell.fire_seated(&seated).expect("the fire serves")
}

/// The gap between a row's two largest logits.
fn gap(row: &[f32]) -> f32 {
    let mut top = [f32::NEG_INFINITY; 2];
    for &v in row {
        if v > top[0] {
            top = [v, top[0]];
        } else if v > top[1] {
            top[1] = v;
        }
    }
    top[0] - top[1]
}

#[test]
fn a_masked_beam_beside_plain_decodes_answers_as_each_alone() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let trace = trace_hybrid("mask-data-micro", &Micro { mask: true }, Platform::Metal);
    let fixture = fixture_from(trace.clone(), "mask-data-micro");
    let mut shell = load(&fixture);
    let beam = PLAIN;

    // Each lane's prompt in its pair of slots: batched in `l`, alone in
    // `LANES + l`.
    let mut last = [[0u32; 2]; LANES as usize];
    for lane in 0..LANES {
        for (side, slot) in [(0usize, lane), (1, LANES + lane)] {
            shell.open(slot).expect("a slot opens");
            let rows = fire(&mut shell, &trace, &[(slot, &prompt(lane), 0, None)]);
            last[lane as usize][side] = argmax(&rows[0]);
        }
    }
    let first_beam_token = last[beam as usize][1];
    let mut first_beam_alone = Vec::new();
    let (mut same, mut ties) = (0u32, 0u32);
    let mut parted = [false; LANES as usize];
    for step in 0..STEPS {
        let held = PROMPT + step;
        let mask = beam_mask(held + 1);
        let next: Vec<[u32; 1]> = (0..LANES as usize).map(|l| [last[l][0]]).collect();
        let rows: Vec<Row<'_>> = (0..LANES)
            .map(|l| (l, &next[l as usize][..], held, (l == beam).then_some(&mask)))
            .collect();
        let batched = fire(&mut shell, &trace, &rows);
        for lane in 0..LANES as usize {
            let token = [last[lane][1]];
            let alone = fire(
                &mut shell,
                &trace,
                &[(LANES + lane as u32, &token, held, (lane as u32 == beam).then_some(&mask))],
            )
            .swap_remove(0);
            if step == 0 && lane as u32 == beam {
                first_beam_alone.clone_from(&alone);
            }
            let (a, b) = (&alone, &batched[lane]);
            let (ta, tb) = (argmax(a) as usize, argmax(b) as usize);
            if !parted[lane] && ta != tb {
                let parting = near_tie(gap(a), &[ta, tb], Drift::Rows { a, b }, None);
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
    eprintln!("{same} lane steps the same token, {ties} parted on a near-tie");

    // A mixed fire: a prompt between the plain decodes and the beam, so the
    // decoding class is out of order and its window is copied or split. The
    // beam reads its own mask rows either way: it answers as alone.
    if !parted[beam as usize] {
        let held = PROMPT + STEPS;
        let mask = beam_mask(held + 1);
        let prompted = SLOTS - 1;
        shell.open(prompted).expect("a slot opens");
        let fresh = prompt(LANES + 1);
        let next: Vec<[u32; 1]> = (0..LANES as usize).map(|l| [last[l][0]]).collect();
        let mut rows: Vec<Row<'_>> = vec![(0, &next[0][..], held, None), (prompted, &fresh, 0, None)];
        rows.extend((1..LANES).map(|l| (l, &next[l as usize][..], held, (l == beam).then_some(&mask))));
        let mixed = fire(&mut shell, &trace, &rows);
        let token = [last[beam as usize][1]];
        let alone = fire(&mut shell, &trace, &[(LANES + beam, &token, held, Some(&mask))]).swap_remove(0);
        let got = &mixed[rows.len() - 1];
        let (ta, tb) = (argmax(&alone) as usize, argmax(got) as usize);
        let worst = alone.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("the beam in a mixed fire: {worst} from alone");
        assert!(
            ta == tb || near_tie(gap(&alone), &[ta, tb], Drift::Rows { a: &alone, b: got }, None).tie,
            "the beam in a mixed fire answers otherwise than alone, on no near-tie"
        );
    }

    // The beam's mask is read: its prompt in one more slot, the beam's
    // first step stating none, answers otherwise than the beam alone did.
    let spare = 2 * LANES;
    shell.open(spare).expect("a slot opens");
    fire(&mut shell, &trace, &[(spare, &prompt(beam), 0, None)]);
    let token = [first_beam_token];
    let unmasked = fire(&mut shell, &trace, &[(spare, &token, PROMPT, None)]).swap_remove(0);
    let moved = unmasked
        .iter()
        .zip(&first_beam_alone)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Against the row's own span: the seeded planes are small, and a lane
    // fired alone answers the same bits every time, so any move is the mask.
    let span = first_beam_alone.iter().copied().fold(f32::MIN, f32::max)
        - first_beam_alone.iter().copied().fold(f32::MAX, f32::min);
    eprintln!("stating no mask moves the beam's first step {moved} of a span {span}");
    assert!(
        moved > span * 0.05,
        "the beam's first step reads its mask: stating none moves its logits ({moved} of {span})"
    );
}

/// A prompt and its decodes answer alike whether the model opens the mask or
/// not: a lane stating none reads through the masked arms as the plain ones.
#[test]
fn a_maskless_lane_answers_through_the_masked_arms_as_through_the_plain_ones() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let serve = |mask: bool| -> Vec<Vec<f32>> {
        let name = if mask { "mask-data-micro" } else { "mask-data-micro-plain" };
        let trace = trace_hybrid(name, &Micro { mask }, Platform::Metal);
        let fixture = fixture_from(trace.clone(), name);
        let mut shell = load(&fixture);
        shell.open(0).expect("a slot opens");
        let mut out = Vec::new();
        let mut held = 0;
        for tokens in [prompt(0), vec![5], vec![9], vec![17]] {
            out.push(fire(&mut shell, &trace, &[(0, &tokens, held, None)]).swap_remove(0));
            held += tokens.len() as u32;
        }
        out
    };
    let (masked, plain) = (serve(true), serve(false));
    for (step, (a, b)) in masked.iter().zip(&plain).enumerate() {
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let span = b.iter().copied().fold(f32::MIN, f32::max) - b.iter().copied().fold(f32::MAX, f32::min);
        eprintln!("step {step}: worst {worst} of a span {span}");
        assert!(worst <= span * 1e-2, "step {step}: the masked arms answer otherwise ({worst})");
    }
}
