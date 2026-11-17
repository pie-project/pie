//! A kv-less lane of a bidirectional layer states which of its group's rows
//! each row attends (a group mask, the ragged kernel's custom arm), after a
//! load that armed what it keeps, between two fires of the same lanes that
//! state none: the masked fire lands the host reference with the same rows
//! hidden, not the one with every row of the group, and each maskless fire
//! lands every row, so no body made of one answers the other. Each runs as
//! the engine states it encodes such rows (`Tally::Eager`), recording
//! nothing after ready, and where the backend records, building nothing:
//! the load named the mask's arm.

use std::collections::BTreeMap;

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Mask, Masking, Readout, Step};
use engine::live::Tally;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine_gpu::serve::Serve;
use poem::{Dtype, ForwardHybrid, Input, RaggedMask, Request, StateRows, Weight, ops};

use super::outputs::{NAME, contract_for, write};
use crate::backend;

const WIDTH: u32 = 32;
const VOCAB: u32 = 64;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
/// Every row a lane holds is read out: the fire's rows within what a load
/// of four lanes reads out.
const ROWS: [usize; 3] = [4, 5, 7];

/// One bidirectional layer over each attention group's rows, a lane its
/// own group: its rows' embeddings attend one another, then project back
/// onto the residual.
struct Encoder;

impl ForwardHybrid for Encoder {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", HEADS * HEAD_DIM, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", HEADS * HEAD_DIM, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", HEADS * HEAD_DIM, WIDTH));
        let perm = inputs.row_permutation();
        let indptr = inputs.group_indptr();
        let o = ops::attn::ragged(
            &ops::gather::pack_rows(&q, &perm),
            &ops::gather::pack_rows(&k, &perm),
            &ops::gather::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::gather::unpack_rows(&o, &perm);
        let y = ops::linear::matmul(&o, &w("o", WIDTH, HEADS * HEAD_DIM));
        let out = ops::elemwise::residual_add(&x, &y);
        let out = ops::gather::gather_rows(&out, &inputs.readout_rows());
        poem::output("logits", &out, None);
    }
}

fn bf(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

fn planes() -> BTreeMap<String, (Vec<u64>, Vec<f32>)> {
    let mut state = 0x9a5c_u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let mut planes = BTreeMap::new();
    for (name, out, inner) in [
        ("embed", VOCAB, WIDTH),
        ("q", HEADS * HEAD_DIM, WIDTH),
        ("k", HEADS * HEAD_DIM, WIDTH),
        ("v", HEADS * HEAD_DIM, WIDTH),
        ("o", WIDTH, HEADS * HEAD_DIM),
    ] {
        let scale = if name == "embed" {
            1.0
        } else {
            1.0 / (inner as f32).sqrt()
        };
        let values = (0..out * inner).map(|_| bf(unit() * scale)).collect();
        planes.insert(
            name.to_string(),
            (vec![u64::from(out), u64::from(inner)], values),
        );
    }
    planes
}

/// Row `i` of lane `at` attends key row `j`: itself always, else a pattern
/// that differs lane by lane, so a lane reading another's bits shows.
fn admits(at: usize, i: usize, j: usize) -> bool {
    i == j || (i * 7 + j * 3 + at) % 3 != 0
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

fn matmul(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc = x[r * k + i].mul_add(w[c * k + i], acc);
            }
            y[r * n + c] = bf(acc);
        }
    }
    y
}

/// The layer on the host over one lane's `tokens`, row `i` attending key
/// row `j` where `admits(i, j)`.
fn reference(
    planes: &BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    tokens: &[u32],
    admits: &dyn Fn(usize, usize) -> bool,
) -> Vec<f32> {
    let (w, hd, heads) = (WIDTH as usize, HEAD_DIM as usize, HEADS as usize);
    let width = heads * hd;
    let rows = tokens.len();
    let get = |name: &str| planes[name].1.as_slice();
    let x: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| get("embed")[t as usize * w..][..w].iter().copied())
        .collect();
    let q = matmul(&x, rows, w, get("q"), width);
    let k = matmul(&x, rows, w, get("k"), width);
    let v = matmul(&x, rows, w, get("v"), width);
    let mut o = vec![0f32; rows * width];
    for h in 0..heads {
        for i in 0..rows {
            let qi = &q[i * width + h * hd..][..hd];
            let scores: Vec<f32> = (0..rows)
                .map(|j| {
                    if !admits(i, j) {
                        return f32::NEG_INFINITY;
                    }
                    let kj = &k[j * width + h * hd..][..hd];
                    qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * SM_SCALE
                })
                .collect();
            let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
            let mass: f32 = weights.iter().sum();
            for d in 0..hd {
                let mut acc = 0f32;
                for j in 0..rows {
                    acc += bf(weights[j] / mass) * v[j * width + h * hd + d];
                }
                o[i * width + h * hd + d] = bf(acc);
            }
        }
    }
    let y = matmul(&o, rows, width, get("o"), w);
    y.iter().zip(&x).map(|(a, b)| bf(a + b)).collect()
}

/// How far `got` lies from `want`, relative, at its worst cell.
fn worst(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
        .fold(0.0, f32::max)
}

#[test]
fn a_masked_group_fire_after_arming_reads_only_the_rows_it_admits() {
    let boot = backend::DeviceBoot::default();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-group-mask-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    let planes = planes();
    write(&path, &planes);
    let trace = poem::trace_hybrid(NAME, &Encoder, backend::PLATFORM);
    let mut engine = backend::Engine::new(boot, contract_for);
    engine
        .load(LoadRequest {
            trace: trace.clone(),
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 64,
                buckets: vec![16, 32, 64],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: backend::ALLOWANCE,
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("the layer loads");

    let tokens: Vec<Vec<u32>> = ROWS
        .iter()
        .enumerate()
        .map(|(at, &n)| {
            (0..n)
                .map(|i| ((i * 11 + at * 5 + 3) % VOCAB as usize) as u32)
                .collect()
        })
        .collect();
    let masked: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(at, tokens)| reference(&planes, tokens, &|i, j| admits(at, i, j)))
        .collect();
    let whole: Vec<Vec<f32>> = tokens
        .iter()
        .map(|tokens| reference(&planes, tokens, &|_, _| true))
        .collect();
    let mut fire = |mask: bool| {
        let lanes = tokens
            .iter()
            .enumerate()
            .map(|(at, tokens)| {
                let rows = tokens.len();
                Lane {
                    slot: at as u32,
                    word: trace.facts.word(&Request::new(rows as u32, false)),
                    tokens: tokens.clone(),
                    readout: Readout::Rows((0..rows as u32).collect()),
                    group: Some(at as u32),
                    kv_less: true,
                    mask: mask
                        .then(|| Masking::Rows((0..rows).map(|i| row_mask(at, i, rows)).collect())),
                    ..Lane::default()
                }
            })
            .collect();
        let eager = engine.live().tallied(Tally::Eager);
        let mut ticket = engine
            .submit(&FrameSubmission::of(Step {
                lanes,
                attachments: Vec::new(),
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the group fire fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let live = engine.live();
        assert_eq!(
            live.tallied(Tally::Eager),
            eager + 1,
            "a {} group fire runs as the engine states it encodes such rows",
            if mask { "masked" } else { "maskless" }
        );
        assert_eq!(live.tallied(Tally::RecordedAfterReady), 0);
        assert_eq!(live.tallied(Tally::Unrecorded), 0);
        // The load named and built both arms of the ragged kernel. A
        // backend that keeps no recording binds each fire's views as it
        // first meets them, which is no pipeline: its count is said.
        let built = live.tallied(Tally::BuiltAfterReady);
        if <backend::Api as Serve>::RECORDS {
            assert_eq!(built, 0, "a group fire built or made what the load did not");
        } else {
            eprintln!("{} objects made after ready, as the backend binds", built);
        }
        let readouts = &ticket.steps[0].readouts;
        assert_eq!(readouts.len(), ROWS.len(), "a readout a lane");
        readouts
            .iter()
            .map(|readout| readout.values.clone())
            .collect::<Vec<_>>()
    };

    for (at, (masked, whole)) in masked.iter().zip(&whole).enumerate() {
        assert!(
            worst(masked, whole) > 4.0e-2,
            "lane {at}: the mask hides no row the whole group's reference reads"
        );
    }
    let whole_fire = |got: Vec<Vec<f32>>, after: &str| {
        for (at, (got, want)) in got.iter().zip(&whole).enumerate() {
            let off = worst(got, want);
            assert!(
                off <= 4.0e-2,
                "lane {at}: the maskless fire {after} lies {off} from every row of its group"
            );
        }
    };
    whole_fire(fire(false), "before a masked one");
    let got = fire(true);
    for (at, ((got, want), wrong)) in got.iter().zip(&masked).zip(&whole).enumerate() {
        assert_eq!(got.len(), want.len(), "lane {at}: every row read out");
        let (off, unmasked) = (worst(got, want), worst(got, wrong));
        eprintln!(
            "masked lane {at}: {off:.4} from its masked reference, {unmasked:.4} from the \
             whole group's"
        );
        assert!(
            off <= 4.0e-2,
            "lane {at}: the masked fire lies {off} from its masked reference ({unmasked} from \
             every row of its group)"
        );
    }
    whole_fire(fire(false), "after a masked one");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
