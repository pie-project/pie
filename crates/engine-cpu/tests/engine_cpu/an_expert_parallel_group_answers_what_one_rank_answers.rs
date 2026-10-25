//! The MoE micro served by one rank holding every expert and by `N`
//! simulated ranks, each a thread holding its slots of every bank, their
//! dispatches and combines meeting in memory: every rank's logits are the
//! bits one rank answers for its prompt, at 2, 4 and 8 ranks, whether the
//! other ranks bring prompts of their own or idle through the step serving
//! the first's routes, through a prefill and the decode after it.

use engine_cpu::{Exchange, Forward, Group};
use poem_ir::{Dtype, Platform, Request, Trace};

const MICRO: &str = "qwen3-a3b-micro";

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A plane's values by its name, as engine-cuda's EP micro writes its
/// checkpoint: the same for one rank's whole bank and a rank's slots.
fn plane(name: &str, count: usize) -> Vec<f32> {
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            f32::from_bits(value.to_bits() & 0xffff_0000)
        })
        .collect()
}

fn prompt(len: u32, seed: u32) -> Vec<u32> {
    (0..len).map(|t| (t * 7 + 11 + seed * 131) % 2048).collect()
}

/// A lane's steps: a prompt, then one token decoded after it.
struct Steps {
    prompt: Vec<u32>,
    next: u32,
}

impl Steps {
    fn of(rank: u32) -> Steps {
        Steps {
            prompt: prompt(37 + 15 * rank, rank + 1),
            next: 5 + rank,
        }
    }
}

fn word(trace: &Trace, tokens: usize) -> u64 {
    trace.facts.word(&Request::new(tokens as u32, false))
}

/// The two steps' logits on one rank holding every expert.
fn alone(whole: &Trace, steps: &Steps) -> [Vec<f32>; 2] {
    let mut one = Forward::new(whole.clone(), None, &plane).expect("the micro loads");
    let p = &steps.prompt;
    let first = one
        .step(p, 0, word(whole, p.len()))
        .expect("the prefill runs");
    let second =
        (one.step(&[steps.next], p.len() as u32, word(whole, 1))).expect("the decode runs");
    [first, second]
}

/// Each rank's two steps' logits, rank `r` bringing `steps[r]` (none
/// where it idles: an empty step each time, serving its peers).
fn group(lowered: &Trace, steps: &[Option<Steps>]) -> Vec<[Vec<f32>; 2]> {
    let exchange = Exchange::new(steps.len());
    std::thread::scope(|scope| {
        let ranks: Vec<_> = (steps.iter().enumerate())
            .map(|(rank, steps)| {
                let exchange = exchange.clone();
                scope.spawn(move || {
                    let group = Group { rank, exchange };
                    let mut me = (Forward::new(lowered.clone(), Some(group), &plane))
                        .expect("the rank loads");
                    let (p, next): (&[u32], &[u32]) = match steps {
                        Some(s) => (&s.prompt, std::slice::from_ref(&s.next)),
                        None => (&[], &[]),
                    };
                    let first = me
                        .step(p, 0, word(lowered, p.len()))
                        .expect("the prefill runs");
                    let second =
                        (me.step(next, p.len() as u32, word(lowered, 1))).expect("the decode runs");
                    [first, second]
                })
            })
            .collect();
        ranks
            .into_iter()
            .map(|h| h.join().expect("a rank ran"))
            .collect()
    })
}

fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: the vocabularies differ");
    assert!(
        want.iter().all(|v| v.is_finite()),
        "{what}: one rank's logits are not all finite"
    );
    let spread = want.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - want.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(spread > 1e-3, "{what}: the logits are flat");
    let differ = (got.iter().zip(want))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let worst = (got.iter().zip(want))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        differ, 0,
        "{what}: {differ} logits differ, worst by {worst} over a spread of {spread}"
    );
}

#[test]
fn an_expert_parallel_group_answers_what_one_rank_answers() {
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    for ranks in [2u32, 4, 8] {
        let lowered = poem_compiler::expert::lower(whole.clone(), ranks, 0)
            .unwrap_or_else(|why| panic!("the micro's MoE lowers at {ranks}: {why}"));
        let want: Vec<_> = (0..ranks).map(|r| alone(&whole, &Steps::of(r))).collect();

        let got = group(
            &lowered,
            &(0..ranks).map(|r| Some(Steps::of(r))).collect::<Vec<_>>(),
        );
        for (r, (got, want)) in got.iter().zip(&want).enumerate() {
            for (s, step) in ["prefill", "decode"].iter().enumerate() {
                same(
                    &got[s],
                    &want[s],
                    &format!("{ranks} ranks: rank {r}'s {step} beside the others'"),
                );
            }
        }

        let idle: Vec<_> = (0..ranks).map(|r| (r == 0).then(|| Steps::of(0))).collect();
        let got = group(&lowered, &idle);
        for (s, step) in ["prefill", "decode"].iter().enumerate() {
            same(
                &got[0][s],
                &want[0][s],
                &format!("{ranks} ranks: rank 0's {step} with the others idle"),
            );
        }
        assert!(
            got[1..].iter().all(|steps| steps.iter().all(Vec::is_empty)),
            "{ranks} ranks: an idle rank reads out nothing"
        );
    }
}
