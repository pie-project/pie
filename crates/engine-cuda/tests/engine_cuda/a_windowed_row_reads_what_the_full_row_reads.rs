#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine_cuda::{Boot, Lane, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;
use poem_ir::{CacheRow, Trace};

const DEPLOYMENT: &str = "gemma4-e4b-mini-l6-bf16-kv-bf16";
const CHUNK: u32 = 256;

fn logits(trace: Trace, path: &Path, prompt: &[u32], decodes: u32) -> Vec<Vec<f32>> {
    served(DEPLOYMENT, trace, path, prompt, decodes).0
}

/// The trace with every windowed kv row sized to the context.
fn full_rows(windowed: &Trace) -> Trace {
    let mut full = windowed.clone();
    for row in &mut full.caches {
        if let CacheRow::Kv { window, .. } = row {
            *window = None;
        }
    }
    full
}

/// `got` answers each fire within 1e-3 of `want`: a windowed row reads the
/// same keys as the full row, laid on other pages.
fn agree(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len());
    for (step, (got, want)) in got.iter().zip(want).enumerate() {
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 1e-3,
            "{what} fire {step}: the windowed rows answer {worst} away from the full ones"
        );
    }
}

/// `deployment`'s logits over `prompt` in chunks and `decodes` greedy-free
/// decodes, and what its windowed fires did and the side streams it forks.
fn served(
    name: &str,
    trace: Trace,
    path: &Path,
    prompt: &[u32],
    decodes: u32,
) -> (Vec<Vec<f32>>, engine_cuda::serve::RingSeen, usize) {
    let deployment = poem_compiler::catalog::deployment(name).expect("the catalog ships the mini");
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        knobs: engine_cuda::Knobs {
            pin: Some((engine_cuda::Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(trace, &contract, path, Budget::new(1, CHUNK), 1024, 1)
    })
    .expect("the shell loads");
    shell.open(0).expect("the slot opens");
    let mut fire = |tokens: &[u32]| {
        let facts = deployment.trace(poem::Platform::Cuda).facts;
        let word = facts.word(&Request::new(tokens.len() as u32, false));
        let rows = shell
            .fire(&[Lane {
                slot: 0,
                word,
                tokens,
            }])
            .expect("the fire runs");
        rows.into_iter().next().expect("one readout")
    };
    let mut out: Vec<Vec<f32>> = prompt.chunks(CHUNK as usize).map(&mut fire).collect();
    for step in 0..decodes {
        out.push(fire(&[(step * 7919) % 4096]));
    }
    drop(fire);
    (out, shell.ring_seen(), shell.side_streams())
}

// #699: a windowed row keeps only its window, every page behind it reading
// a zeroed null page and the rest cycling through a ring, and it answers
// what the same row sized to the context answers — past the point where
// the ring has wrapped over pages the window left behind.
#[test]
fn a_windowed_row_reads_what_the_full_row_reads() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let windowed = poem_compiler::catalog::deployment(DEPLOYMENT)
        .expect("the mini gemma")
        .trace(Platform::Cuda);
    assert!(
        windowed.caches.iter().any(|row| matches!(
            row,
            CacheRow::Kv {
                window: Some(512),
                ..
            }
        )),
        "the sliding rows are declared windowed"
    );
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&windowed, 699).write(dir.path());
    let prompt: Vec<u32> = (0..3 * CHUNK).map(|at| (at * 31 + 5) % 4096).collect();
    let want = logits(full_rows(&windowed), &path, &prompt, 96);
    let got = logits(windowed, &path, &prompt, 96);
    agree(&got, &want, DEPLOYMENT);
}

// CONTRACT-BC 19: a fire's fold lands after every read its lanes' attention
// makes, on every stream it forks: a fire whose lanes read ring pages that
// its own fold then overwrites (its own rows past its ring on own-row pages,
// folded onto ring pages it read) and fires that write their rows on their
// ring directly answer what the full rows answer, on loads whose fires fork
// regions onto side streams.
#[test]
fn a_ring_fold_lands_after_every_read_of_its_fire() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let mut forked = 0usize;
    for name in [DEPLOYMENT] {
        let trace = poem_compiler::catalog::deployment(name)
            .expect("the mini")
            .trace(Platform::Cuda);
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = Weights::random(&trace, 701).write(dir.path());
        let prompt: Vec<u32> = (0..3 * CHUNK).map(|at| (at * 31 + 5) % 4096).collect();
        let (want, _, _) = served(name, full_rows(&trace), &path, &prompt, 64);
        let (got, seen, sides) = served(name, trace, &path, &prompt, 64);
        eprintln!("{name}: {seen:?}, {sides} side stream(s)");
        assert!(seen.direct > 0, "{name}: some fire writes its ring directly");
        assert!(seen.own > 0, "{name}: some fire takes own-row pages");
        assert!(
            seen.read_then_folded > 0,
            "{name}: some fire folds onto ring pages it read"
        );
        forked += sides;
        agree(&got, &want, name);
    }
    assert!(forked > 0, "a load whose fires fork regions onto side streams");
}

/// `served`, its decodes reading a mask (every key but a stretch of the
/// prompt) through the masked arms.
fn served_masked(trace: Trace, path: &Path, prompt: &[u32], decodes: u32) -> Vec<Vec<f32>> {
    use engine::fire::{Mask, Masking, Readout};
    let deployment = poem_compiler::catalog::deployment(DEPLOYMENT).expect("the mini");
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        knobs: engine_cuda::Knobs {
            pin: Some((engine_cuda::Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(trace, &contract, path, Budget::new(1, CHUNK), 1024, 1)
    })
    .expect("the shell loads");
    shell.open(0).expect("the slot opens");
    let facts = deployment.trace(poem::Platform::Cuda).facts;
    let mut out = Vec::new();
    for chunk in prompt.chunks(CHUNK as usize) {
        let word = facts.word(&Request::new(chunk.len() as u32, false));
        let rows = shell
            .fire(&[Lane { slot: 0, word, tokens: chunk }])
            .expect("the fire runs");
        out.push(rows.into_iter().next().expect("one readout"));
    }
    for step in 0..decodes {
        let held = prompt.len() as u32 + step;
        let columns = held + 1;
        let mask = Masking::Extent(Mask::new(vec![0, 8, 8, columns - 16], u64::from(columns)));
        let token = [(step * 7919) % 4096];
        let word = facts.word(&Request::new(1, false));
        let seated = engine_cuda::serve::Seated::masked(Lane { slot: 0, word, tokens: &token }, &mask);
        let rows = shell
            .fire_rows(&[seated], &[Readout::Last])
            .expect("the masked fire runs");
        out.push(rows.into_iter().next().expect("one readout"));
    }
    out
}

// CONTRACT-BC 19: a lane reading a mask of its own through the masked arms
// reads its windowed rows from ring + buffer as from the full rows.
#[test]
fn a_masked_lane_reads_its_ring_as_its_full_rows() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem_compiler::catalog::deployment(DEPLOYMENT)
        .expect("the mini gemma")
        .trace(Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 703).write(dir.path());
    let prompt: Vec<u32> = (0..3 * CHUNK).map(|at| (at * 31 + 5) % 4096).collect();
    let want = served_masked(full_rows(&trace), &path, &prompt, 48);
    let got = served_masked(trace, &path, &prompt, 48);
    agree(&got, &want, DEPLOYMENT);
}
