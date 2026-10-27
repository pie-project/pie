//! Qwen3 0.6B shrunk to a micro, whose head reads its token embedding tied,
//! served by one rank and by `N` simulated tensor-parallel ranks: the
//! compiler cuts the tied table by its rows (the vocabulary), so each rank
//! looks up the tokens its band holds, zeros for the rest, the ranks' rows
//! summed, and reads out its band of the logits, gathered after. Every
//! rank's logits are the ones one rank answers, at 2, 4 and 8 ranks, through
//! a prefill and the decode after it.

use std::sync::Arc;

use engine_cpu::forward::{Output, Values};
use engine_cpu::{Exchange, Forward, Group};
use poem::star::{Deploy, Package};
use poem_ir::{Dtype, Platform, Request, Trace};

const MODEL: &str = "qwen3-0.6b";

/// Qwen3's package at a micro's widths: the 0.6B's tied table and forward,
/// its layers, heads and vocabulary made small.
fn micro() -> Package {
    let package = poem_compiler::catalog::package_of(MODEL).expect("the catalog holds Qwen3");
    let small = [
        (
            "\"qwen3-0.6b\": dims(hidden = 1024, layers = 28, q_heads = 16, kv_heads = 8, inter = 3072, tied = True),",
            "\"qwen3-0.6b\": dims(hidden = 256, layers = 2, q_heads = 8, kv_heads = 2, inter = 64, tied = True),",
        ),
        (
            "qwen3(\"qwen3-0.6b\", layers = 28),",
            "qwen3(\"qwen3-0.6b\", layers = 2),",
        ),
        ("head_dim = 128,", "head_dim = 32,"),
        ("151936", "512"),
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

/// The trace each of `tp` ranks runs, every activation held f32.
fn lowered(package: &Package, tp: u32) -> Trace {
    let deploy = deploy(tp);
    let trace = package
        .trace(MODEL, &deploy, MODEL, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the micro traces: {why:#}"));
    let mut trace = poem_compiler::catalog::lowered(trace, &deploy)
        .unwrap_or_else(|why| panic!("the micro splits {tp} ways: {why}"));
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
    trace.facts.word(&Request::new(tokens as u32, false))
}

const PROMPT: [u32; 9] = [7, 300, 41, 511, 2, 96, 257, 13, 404];
const NEXT: u32 = 77;

fn logits(outs: Vec<Output>) -> Vec<f32> {
    match outs.into_iter().find(|out| out.name == "logits") {
        Some(Output {
            values: Values::F32(v),
            ..
        }) => v,
        _ => Vec::new(),
    }
}

/// The prefill's and the decode's logits of one forward, `group` its rank.
fn steps(trace: &Trace, group: Option<Group>) -> [Vec<f32>; 2] {
    let mut me = Forward::new(trace.clone(), group, &plane).expect("the micro loads");
    let first = me
        .step(&PROMPT, 0, word(trace, PROMPT.len()))
        .expect("the prefill runs");
    let second = me
        .step(&[NEXT], PROMPT.len() as u32, word(trace, 1))
        .expect("the decode runs");
    [logits(first), logits(second)]
}

fn group(trace: &Trace, ranks: usize) -> Vec<[Vec<f32>; 2]> {
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
fn a_tied_table_cut_by_its_rows_answers_what_one_rank_answers() {
    let package = micro();
    let whole = lowered(&package, 1);
    let want = steps(&whole, None);
    for logits in &want {
        assert_eq!(logits.len(), 512, "a micro's vocabulary");
        assert!(logits.iter().all(|v| v.is_finite()), "one rank's logits");
    }
    for ranks in [2u32, 4, 8] {
        let split = lowered(&package, ranks);
        let embed = (split.params.iter())
            .find(|p| p.name == "embed")
            .expect("the tied table");
        assert!(
            matches!(embed.shard, poem_ir::Shard::Cut { axis: 0, .. }),
            "{ranks} ranks: the tied table is cut by its rows, not {:?}",
            embed.shard
        );
        let got = group(&split, ranks as usize);
        for (r, got) in got.iter().enumerate() {
            for (s, step) in ["prefill", "decode"].iter().enumerate() {
                let off = apart(&got[s], &want[s]);
                assert!(
                    off < 1e-4,
                    "{ranks} ranks: rank {r}'s {step} logits are {off} apart from one rank's"
                );
            }
        }
    }
}
