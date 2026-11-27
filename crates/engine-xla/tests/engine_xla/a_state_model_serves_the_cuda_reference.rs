//! A Mamba-2 and a KDA miniature served on XLA against the CUDA reference:
//! engine-cuda's golden-logits probe (`golden_logits.rs`) seeds each
//! model's checkpoint by plane name and serves it over a fixed script of
//! prefills and greedy decodes; here the same seeded planes are served
//! over the same script, and each step's readout row is set beside the
//! CUDA one's top tokens. The two backends sum in other orders, so the
//! rows are compared by their top tokens and logits, not bit for bit.
//!
//! Run by hand (ignored in the suite):
//!
//! ```text
//! PIE_XLA_REF=<file of `cuda <model> <step> id:logit ...` lines> \
//!   cargo test --release -p engine-xla --test engine_xla \
//!   a_state_model_serves_the_cuda_reference -- --ignored --nocapture
//! ```
//!
//! It prints one `xla <model> <step> id:logit ...` line a step (the top
//! five). With `PIE_XLA_REF` a decode takes the reference's top token, so
//! one parting does not move every step after it, and each step's top
//! tokens and their logits are checked against the reference's.

use std::collections::BTreeMap;

use poem_ir::{ParamSource, Trace};

use crate::common_llm::{Model, Refs, Seq};

/// The models: Mamba-2 mixers beside attention and routed experts; KDA
/// linear attention beside full.
const MODELS: &[&str] = &["nemotron3-nano-mini", "glm53-flash-mini"];

/// The probe's context.
const CONTEXT: u32 = 1024;

#[derive(Clone, Copy)]
enum Step {
    Prefill(u32),
    Decode,
}

/// The probe's script: 37 + 3 + 250 + 3 + 250 + 3 = 546 rows.
const SCRIPT: &[Step] = &[
    Step::Prefill(37),
    Step::Decode,
    Step::Decode,
    Step::Decode,
    Step::Prefill(250),
    Step::Decode,
    Step::Decode,
    Step::Decode,
    Step::Prefill(250),
    Step::Decode,
    Step::Decode,
    Step::Decode,
];

/// The top tokens compared a step.
const TOP: usize = 5;

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The span an index plane's entries keep below (the probe's).
const INDEX_SPAN: u64 = 8;

/// `trace`'s checkpoint planes seeded by name as the probe seeds them
/// (`seeded_planes`), written as a checkpoint: float planes near one for
/// norms and small otherwise, index planes small and distinct a row.
fn write_seeded(path: &std::path::Path, trace: &Trace) {
    // `PIE_XLA_REF_SET=<plane>=<value>[;...]`: those planes every element
    // that value in place of their seeding (an ablation, matched on the
    // reference side).
    let set: BTreeMap<String, f32> = std::env::var("PIE_XLA_REF_SET")
        .map(|v| {
            v.split(';')
                .filter_map(|kv| {
                    let (k, val) = kv.split_once('=')?;
                    Some((k.trim().to_string(), val.trim().parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    // A packed plane's scales are made with it when the artifact is
    // imported from these floats (`common_llm`): none of their own here.
    let made: std::collections::BTreeSet<String> = planes
        .iter()
        .filter(|param| {
            !matches!(
                param.dtype,
                poem::Dtype::Bf16 | poem::Dtype::F32 | poem::Dtype::I64 | poem::Dtype::I32
            )
        })
        .flat_map(|param| {
            [
                poem::scales_name(&param.name),
                poem::biases_name(&param.name),
                poem::global_scales_name(&param.name),
            ]
        })
        .collect();
    planes.retain(|param| !made.contains(&param.name));
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    let mut writer = ztensor::Writer::create(path).expect("the fixture writes");
    for param in planes {
        let shape = poem::Weight::of_plane(param).shape;
        let count: usize = shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let bits = seeded_bits(&param.name, count, trace);
        let mut at = 0usize;
        let mut next = || {
            at += 1;
            bits[at - 1]
        };
        let unit = |bits: u64| (bits as f32 / (1u64 << 31) as f32) - 0.5;
        let fixed = set.get(&param.name).copied();
        let value = |bits: u64| {
            if let Some(v) = fixed {
                v
            } else if norm {
                1.0 + 0.05 * unit(bits)
            } else {
                0.08 * unit(bits)
            }
        };
        let (leaf, bytes): (ztensor::Leaf, Vec<u8>) = match param.dtype {
            poem::Dtype::F32 => (
                ztensor::Leaf::F32,
                (0..count)
                    .flat_map(|_| value(next()).to_le_bytes())
                    .collect(),
            ),
            poem::Dtype::I64 | poem::Dtype::I32 => {
                let cols = param.shape.last().copied().unwrap_or(1).max(1);
                let ids = (0..count as u64).map(|at| (at / cols + at % cols) % INDEX_SPAN);
                if param.dtype == poem::Dtype::I64 {
                    (
                        ztensor::Leaf::I64,
                        ids.flat_map(|id| id.to_le_bytes()).collect(),
                    )
                } else {
                    (
                        ztensor::Leaf::I32,
                        ids.flat_map(|id| (id as u32).to_le_bytes()).collect(),
                    )
                }
            }
            // bf16, and any plane the load encodes from bf16
            _ => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
        };
        writer
            .add(param.name.as_str(), shape, leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the fixture is written");
}

/// The seed after `n` LCG steps from `seed` (jump-ahead by squaring).
fn skip(seed: u64, n: u64) -> u64 {
    let (mut a, mut c) = (6_364_136_223_846_793_005u64, 1u64);
    let (mut ra, mut rc) = (1u64, 0u64);
    let mut n = n;
    while n > 0 {
        if n & 1 == 1 {
            ra = ra.wrapping_mul(a);
            rc = rc.wrapping_mul(a).wrapping_add(c);
        }
        c = c.wrapping_mul(a).wrapping_add(c);
        a = a.wrapping_mul(a);
        n >>= 1;
    }
    seed.wrapping_mul(ra).wrapping_add(rc)
}

/// `len` draws of plane `name`'s sequence from draw `start` on.
fn draws(name: &str, start: u64, len: usize) -> impl Iterator<Item = u64> {
    let mut seed = skip(fnv(name), start);
    (0..len).map(move |_| {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        seed >> 33
    })
}

/// Plane `name`'s `count` seed draws: its own sequence; or, under
/// `PIE_XLA_REF_JOINED=1`, where the CUDA plan joins planes this plan keeps
/// apart (gpt-oss: `attn.qkv` rows q, k, v; `mlp.gate_up` each expert's
/// gate rows then up rows; their biases alike), its slices of the joined
/// plane's sequence, as golden_logits seeds the joined plane.
fn seeded_bits(name: &str, count: usize, trace: &Trace) -> Vec<u64> {
    let joined = std::env::var("PIE_XLA_REF_JOINED").is_ok_and(|v| v == "1");
    let shape_of = |leg: &str| -> Option<Vec<u64>> {
        trace
            .params
            .iter()
            .find(|p| p.name == leg)
            .map(|p| poem::Weight::of_plane(p).shape)
    };
    let rows_of = |leg: &str| shape_of(leg).and_then(|s| s.first().copied()).unwrap_or(0);
    if joined && let Some((layer, leg)) = name.rsplit_once('.') {
        let attn = |l: &str| format!("{layer}.{l}");
        let (q, k) = (rows_of(&attn("q")), rows_of(&attn("k")));
        let qkv_at = match leg {
            "q" | "q_bias" => Some(0),
            "k" | "k_bias" => Some(q),
            "v" | "v_bias" => Some(q + k),
            _ => None,
        };
        if layer.ends_with(".attn")
            && let Some(row) = qkv_at
        {
            let bias = leg.ends_with("_bias");
            let width = if bias {
                1
            } else {
                count as u64 / rows_of(&attn(leg)).max(1)
            };
            let seed = attn(if bias { "qkv_bias" } else { "qkv" });
            return draws(&seed, row * width, count).collect();
        }
        if layer.ends_with(".mlp") && matches!(leg, "gate" | "up" | "gate_bias" | "up_bias") {
            let bias = leg.ends_with("_bias");
            let shape = shape_of(name).unwrap_or_default();
            let (experts, inter) = (shape[0] as usize, shape[1] as usize);
            let width = if bias { 1 } else { shape[2] as usize };
            let seed = attn(if bias { "gate_up_bias" } else { "gate_up" });
            let half = if leg.starts_with("up") { inter } else { 0 };
            return (0..experts)
                .flat_map(|e| {
                    draws(
                        &seed,
                        ((e * 2 * inter + half) * width) as u64,
                        inter * width,
                    )
                })
                .collect();
        }
    }
    draws(name, 0, count).collect()
}

/// A row's top `TOP` tokens and logits, highest first.
fn top(row: &[f32]) -> Vec<(u32, f32)> {
    let mut at: Vec<u32> = (0..row.len() as u32).collect();
    at.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    at.into_iter()
        .take(TOP)
        .map(|i| (i, row[i as usize]))
        .collect()
}

/// The reference's top tokens a step, per model, from `PIE_XLA_REF`.
fn reference() -> BTreeMap<(String, usize), Vec<(u32, f32)>> {
    let Ok(path) = std::env::var("PIE_XLA_REF") else {
        return BTreeMap::new();
    };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|why| panic!("{path}: {why}"));
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some("cuda"), Some(model), Some(step)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let Ok(step) = step.parse::<usize>() else {
            continue;
        };
        let tops: Vec<(u32, f32)> = parts
            .filter_map(|p| {
                let (id, logit) = p.split_once(':')?;
                Some((id.parse().ok()?, logit.parse().ok()?))
            })
            .collect();
        out.insert((model.to_string(), step), tops);
    }
    out
}

#[test]
#[ignore = "run by hand against a CUDA reference"]
fn a_state_model_serves_the_cuda_reference() {
    let reference = reference();
    let models: Vec<String> = std::env::var("PIE_XLA_REF_MODELS").map_or_else(
        |_| MODELS.iter().map(|m| (*m).to_string()).collect(),
        |v| v.split(',').map(str::to_string).collect(),
    );
    let mut failed = Vec::new();
    for name in &models {
        crate::common_llm::set_knob("TOKENS", 256);
        // `PIE_XLA_REF_LAYERS=<n>`: the model cut to its first n layers,
        // compared against a reference labelled `<model>-l<n>`.
        let layers: Option<u32> = std::env::var("PIE_XLA_REF_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok());
        let label = layers.map_or_else(|| name.clone(), |n| format!("{name}-l{n}"));
        let mut model = Model::load_written(name, layers, CONTEXT, &write_seeded);
        let mut refs = Refs::of(&model);
        // `PIE_XLA_REF_SLOT=<n>`: the lane in slot n (a fresh slot answers
        // the same whichever it is).
        let slot: u32 = std::env::var("PIE_XLA_REF_SLOT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut seq = Seq::new(slot);
        let mut fed = 0u32;
        let mut last = None;
        for (at, step) in SCRIPT.iter().enumerate() {
            let tokens: Vec<u32> = match *step {
                Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
                Step::Decode => vec![last.expect("a decode follows a step")],
            };
            fed += tokens.len() as u32;
            let rows_fed = tokens.len() as u32;
            // `PIE_XLA_REF_TOKENWISE=1`: a prompt fired a token at a time
            // (the decode path), its last row read.
            let rows = if std::env::var_os("PIE_XLA_REF_TOKENWISE").is_some() && tokens.len() > 1 {
                let mut last_rows = Vec::new();
                for &tok in &tokens {
                    let lane = seq.lane(&model, &mut refs, vec![tok]);
                    last_rows = model.fire(vec![lane]);
                    seq.held += 1;
                }
                last_rows
            } else {
                let lane = seq.lane(&model, &mut refs, tokens);
                let rows = model.fire(vec![lane]);
                seq.held += rows_fed;
                rows
            };
            let mine = top(&rows[0]);
            let line: Vec<String> = mine.iter().map(|(id, l)| format!("{id}:{l:.4}")).collect();
            eprintln!("xla {label} {at} {}", line.join(" "));
            let want = reference.get(&(label.clone(), at));
            // a decode takes the reference's top token where there is one
            last = Some(want.and_then(|w| w.first()).map_or(mine[0].0, |w| w.0));
            if let Some(want) = want {
                // The same top token; or, where the reference's first two
                // stand nearer than XLA's pick parts from its first (a tie
                // the backends' summing orders break either way), the
                // reference's top within that gap of XLA's.
                let row = &rows[0];
                let gap = match want.get(1) {
                    Some(second) => want[0].1 - second.1,
                    None => 0.0,
                };
                let same = mine[0].0 == want[0].0;
                // XLA's pick one of the reference's tokens tied at its top
                // logit, or within the reference's top-2 gap of it
                let tied = want
                    .iter()
                    .any(|&(id, l)| id == mine[0].0 && l == want[0].1);
                let tie = !same && (tied || (mine[0].1 - row[want[0].0 as usize]) <= gap);
                let worst = want
                    .iter()
                    .map(|&(id, l)| (row[id as usize] - l).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "check {label} {at}: top {} vs {} (reference gap {gap:.4}) worst top-{TOP} logit diff {worst:.4}: {}",
                    mine[0].0,
                    want[0].0,
                    if same {
                        "same"
                    } else if tie {
                        "tie"
                    } else {
                        "PARTED"
                    }
                );
                if !(same || tie) {
                    failed.push(format!("{label} step {at}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "parted from the reference: {failed:?}");
}
