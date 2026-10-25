//! MiniMax-M2 shrunk to a micro, served by one rank and by `N` simulated
//! tensor-parallel ranks, each a thread holding its share of every weight the
//! compiler cuts, their sums and gathers meeting in memory: every rank's
//! logits are the ones one rank answers, at 2, 4 and 8 ranks, through a
//! prefill and the decode after it. Its q and k are RMS-normed across all
//! their heads, which the ranks split: each norms its share by the row's sum
//! of squares, its own summed across the ranks (two kv heads, so from 4 ranks
//! up a pair of ranks holds each kv head alike and the sum counts it twice).
//!
//! Every activation is held f32 (the weights bf16): a sum the ranks reorder
//! moves the logits by f32 roundings only, far under what a share normed by
//! its own sum, or a kv head counted once too often, misses by.

use std::sync::Arc;

use engine_cpu::forward::{Output, Values};
use engine_cpu::{Exchange, Forward, Group};
use poem::star::{Deploy, Package};
use poem_ir::{Dtype, Platform, Request, Trace};

const MODEL: &str = "minimax-m25-mini";

/// MiniMax-M2's package at a micro's widths: every dimension its mini states
/// but the layers and experts made small, the forward as it is.
fn micro() -> Package {
    let package = poem_compiler::catalog::package_of(MODEL).expect("the catalog holds MiniMax-M2");
    let small = [
        ("hidden = 3072,", "hidden = 256,"),
        ("q_heads = 48,", "q_heads = 8,"),
        ("kv_heads = 8,", "kv_heads = 2,"),
        ("head_dim = 128,", "head_dim = 32,"),
        ("rotary_dim = 64,", "rotary_dim = 16,"),
        ("inter = 1536,", "inter = 64,"),
        ("200064", "512"),
    ];
    let files: Vec<(String, String)> = package
        .files()
        .map(|(file, source)| {
            let mut source = source.to_string();
            if source.contains("def dims(") {
                for (from, to) in small {
                    assert!(source.contains(from), "the package states `{from}`");
                    source = source.replace(from, to);
                }
            }
            (file.to_string(), source)
        })
        .collect();
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(f, s)| (f.as_str(), s.as_str()))
        .collect();
    Package::new(package.name(), &files).unwrap_or_else(|why| panic!("{why:#}"))
}

fn deploy(tp: u32) -> Deploy {
    Deploy {
        tp,
        ..poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16)
    }
}

/// The trace each of `tp` ranks runs.
fn lowered(package: &Package, tp: u32) -> Trace {
    let deploy = deploy(tp);
    let trace = package
        .trace(MODEL, &deploy, MODEL, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the micro traces: {why:#}"));
    let mut trace = poem_compiler::catalog::lowered(trace, &deploy)
        .unwrap_or_else(|why| panic!("the micro splits {tp} ways: {why}"));
    // Every activation held f32, so the ranks' sums differ from one rank's
    // only by their order, not by bf16 roundings of the terms.
    for decl in &mut trace.values {
        if let poem_ir::Ty::Tensor { dtype, .. } = &mut decl.ty
            && *dtype == Dtype::Bf16
        {
            *dtype = Dtype::F32;
        }
    }
    trace
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A whole plane's values by its name: norms near one, the rest small.
fn plane(name: &str, count: usize) -> Vec<f32> {
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            if norm { 1.0 + 0.5 * unit } else { 0.2 * unit }
        })
        .collect()
}

fn word(trace: &Trace, tokens: usize) -> u64 {
    (trace.facts).word(&Request::new(tokens as u32, false).capturing_scores(true))
}

const PROMPT: [u32; 9] = [7, 300, 41, 511, 2, 96, 257, 13, 404];
const NEXT: u32 = 77;

/// A step's logits and its attention scores, a layer after another.
type Read = [Vec<f32>; 2];

fn read(outs: Vec<Output>) -> Read {
    let floats = |name: &str| match outs.iter().find(|out| out.name == name) {
        Some(Output {
            values: Values::F32(v),
            ..
        }) => v.clone(),
        _ => panic!("the step reads out no `{name}`"),
    };
    [floats("logits"), floats("attn.scores")]
}

/// The prefill's and the decode's reads of one forward, `group` its rank.
fn steps(trace: &Trace, group: Option<Group>) -> [Read; 2] {
    let mut me = Forward::new(trace.clone(), group, &plane).expect("the micro loads");
    let first = me
        .step(&PROMPT, 0, word(trace, PROMPT.len()))
        .expect("the prefill runs");
    let second = me
        .step(&[NEXT], PROMPT.len() as u32, word(trace, 1))
        .expect("the decode runs");
    [read(first), read(second)]
}

fn group(trace: &Trace, ranks: usize) -> Vec<[Read; 2]> {
    let exchange = Exchange::new(ranks);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..ranks)
            .map(|rank| {
                let exchange = Arc::clone(&exchange);
                scope.spawn(move || steps(trace, Some(Group { rank, exchange })))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a rank ran"))
            .collect()
    })
}

/// How far `got` is from `want`: the root of the mean squared difference
/// over the root of `want`'s mean square.
fn apart(got: &[f32], want: &[f32]) -> f32 {
    let diff: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
    let norm: f32 = want.iter().map(|b| b * b).sum();
    (diff / norm).sqrt()
}

#[test]
fn a_tensor_parallel_group_answers_what_one_rank_answers() {
    let package = micro();
    let whole = lowered(&package, 1);
    let want = steps(&whole, None);
    for (s, [logits, scores]) in want.iter().enumerate() {
        assert_eq!(logits.len(), 512, "a micro's vocabulary");
        assert!(logits.iter().all(|v| v.is_finite()), "one rank's logits");
        let keys = PROMPT.len() + s;
        assert_eq!(scores.len(), 4 * 8 * 2048, "a run per layer and head");
        for run in scores.chunks(2048) {
            let mass: f32 = run[..keys].iter().sum();
            assert!(
                (mass - 1.0).abs() < 1e-4 && run[keys..].iter().all(|&v| v == 0.0),
                "a head's mass is over the keys it holds: {mass}"
            );
        }
    }
    for ranks in [2u32, 4, 8] {
        let split = lowered(&package, ranks);
        let count = |op: &str| {
            (split.nodes.iter())
                .filter(|n| poem_ir::Operands::name(&n.op) == op)
                .count()
        };
        // q's and k's norms, a layer each.
        assert_eq!(
            count("elemwise.rmsnorm_by"),
            2 * 4,
            "{ranks} ranks: each layer norms q and k by their summed squares"
        );
        assert!(
            (split.params.iter())
                .any(|p| p.name == "layers.0.q_norm"
                    && matches!(p.shard, poem_ir::Shard::Cut { .. })),
            "{ranks} ranks: q's norm is cut with q's heads"
        );
        let got = group(&split, ranks as usize);
        for (r, got) in got.iter().enumerate() {
            for (s, step) in ["prefill", "decode"].iter().enumerate() {
                for (k, what) in ["logits", "scores"].iter().enumerate() {
                    let off = apart(&got[s][k], &want[s][k]);
                    assert!(
                        off < 1e-4,
                        "{ranks} ranks: rank {r}'s {step} {what} are {off} apart from one rank's"
                    );
                }
            }
        }
    }
}
