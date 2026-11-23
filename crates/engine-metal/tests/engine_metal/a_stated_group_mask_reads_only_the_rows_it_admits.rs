#![cfg(target_vendor = "apple")]

//! A kv-less lane states which of its group's rows each row attends (a
//! group mask: an encoder's or a vision tower's document isolation), and the
//! ragged kernel reads it. Metal records no body: every fire encodes its own
//! dispatch, the mask arm chosen from the fire's own stated masks. A
//! maskless fire, a masked one, then a maskless one again, each over three
//! lanes, land the host reference with exactly the rows each states hidden:
//! a masked fire never reads a maskless arm, and a maskless one after it
//! never reads its masks.

use engine::fire::{Mask, Masking};
use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};
use poem_ir::ops::attn::RaggedMask;
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};

use super::a_scattered_pool_serves_the_same_logits::{fixture_from, load_with, seeded};

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const ROWS: [usize; 3] = [5, 13, 29];
/// The softmax's scale: small, so a row attends its group nearly evenly
/// (q, k and v are one row; at the usual scale a row attends itself
/// alone), and the rows a mask hides change its answer.
const SCALE: f32 = 0.01;

/// q, k and v the tokens' embedding rows, attended over each lane's group.
struct Grouped;

impl ForwardHybrid for Grouped {
    fn states(&self) -> StateRows {
        StateRows::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), WIDTH], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let groups = inputs.group_indptr();
        let o = ops::attn::ragged(&x, &x, &x, &groups, &groups, HEAD_DIM, SCALE, RaggedMask::Stated);
        let read = ops::gather::gather_rows(&o, &inputs.readout_rows());
        poem::output("logits", &read, None);
    }
}

/// Row `i` of lane `at` attends key row `j`: itself always, else a pattern
/// that differs lane by lane, so a lane reading another's bits shows.
fn admits(at: usize, i: usize, j: usize) -> bool {
    i == j || (i * 7 + j * 3 + at) % 4 != 0
}

/// Row `i`'s mask over the lane's `rows` key rows, as runs: hidden,
/// admitted, hidden, ...
fn row_mask(at: usize, i: usize, rows: usize) -> Mask {
    let mut runs = Vec::new();
    let (mut admitted, mut run) = (false, 0u32);
    for j in 0..rows {
        if admits(at, i, j) == admitted {
            run += 1;
        } else {
            runs.push(run);
            admitted = !admitted;
            run = 1;
        }
    }
    runs.push(run);
    Mask::new(runs, rows as u64)
}

/// The host's answer: each head of each row over the rows it admits.
fn reference(x: &[Vec<f32>], admitted: &dyn Fn(usize, usize) -> bool) -> Vec<f32> {
    let n = x.len();
    let heads = WIDTH as usize / HEAD_DIM as usize;
    let d = HEAD_DIM as usize;
    let scale = SCALE;
    let mut out = vec![0f32; n * WIDTH as usize];
    for i in 0..n {
        for h in 0..heads {
            let head = |row: &[f32]| row[h * d..(h + 1) * d].to_vec();
            let q = head(&x[i]);
            let scores: Vec<(usize, f32)> = (0..n)
                .filter(|&j| admitted(i, j))
                .map(|j| (j, head(&x[j]).iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * scale))
                .collect();
            let top = scores.iter().map(|s| s.1).fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = scores.iter().map(|s| (s.1 - top).exp()).sum();
            for &(j, s) in &scores {
                let p = (s - top).exp() / total;
                for (k, v) in head(&x[j]).iter().enumerate() {
                    out[i * WIDTH as usize + h * d + k] += p * v;
                }
            }
        }
    }
    out
}

/// The largest difference of two rows, against the larger row's largest
/// value: the row's own scale.
fn apart(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().chain(b).fold(0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let off = apart(got, want);
    assert!(off < 2e-2, "{what}: off by {off} of the row's scale");
}

#[test]
fn a_stated_group_mask_reads_only_the_rows_it_admits() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let trace = poem::trace_hybrid("group-mask-micro", &Grouped, Platform::Metal);
    let table: Vec<f32> = seeded(&trace)
        .into_iter()
        .find(|(name, ..)| name.ends_with("embed"))
        .map(|(.., bytes)| {
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect()
        })
        .expect("the embedding table is seeded");
    let fixture = fixture_from(trace.clone(), "group-mask-micro");
    let mut shell: Shell = load_with(&fixture, 256);
    let w = WIDTH as usize;
    let tokens: Vec<Vec<u32>> = ROWS
        .iter()
        .enumerate()
        .map(|(at, &n)| (0..n as u32).map(|t| (t * 5 + 3 * at as u32 + 1) % VOCAB).collect())
        .collect();
    let rows_of = |tokens: &[u32]| -> Vec<Vec<f32>> {
        tokens.iter().map(|&t| table[t as usize * w..][..w].to_vec()).collect()
    };
    let masks: Vec<Masking> = ROWS
        .iter()
        .enumerate()
        .map(|(at, &n)| Masking::Rows((0..n).map(|i| row_mask(at, i, n)).collect()))
        .collect();
    for at in 0..ROWS.len() {
        shell.open(at as u32).expect("the slot opens");
    }
    // A fire reads one row a lane: rows 0, 2 and 4, each through the
    // maskless, masked, maskless sequence.
    for row in [0u32, 2, 4] {
    let read = [row];
    for (fire, masked) in [false, true, false].into_iter().enumerate() {
        let lanes: Vec<Seated<'_>> = (0..ROWS.len())
            .map(|at| Seated {
                replay_stated: None,
                kv_less: true,
                readout: Some(read.as_slice()),
                mask: masked.then(|| &masks[at]),
                ..Seated::of(Lane {
                    slot: at as u32,
                    word: trace.facts.word(&Request::new(ROWS[at] as u32, false)),
                    tokens: &tokens[at],
                })
            })
            .collect();
        let got = shell.fire_seated(&lanes).expect("the fire runs");
        assert_eq!(got.len(), ROWS.len(), "fire {fire}: every lane read out");
        for (at, got) in got.iter().enumerate() {
            let x = rows_of(&tokens[at]);
            let want = if masked {
                reference(&x, &|i, j| admits(at, i, j))
            } else {
                reference(&x, &|_, _| true)
            };
            let r = row as usize;
            close(
                got,
                &want[r * w..(r + 1) * w],
                &format!(
                    "row {row}, fire {fire} ({}), lane {at}",
                    if masked { "masked" } else { "maskless" }
                ),
            );
        }
    }
    }
    // The masks must matter: a masked row differs from its maskless one.
    let x = rows_of(&tokens[0]);
    let (all, hidden) = (reference(&x, &|_, _| true), reference(&x, &|i, j| admits(0, i, j)));
    let differ = apart(&all[..w], &hidden[..w]);
    assert!(
        differ > 0.1,
        "row 0's mask hides rows that change its answer by far more than the tolerance: {differ}"
    );
}
