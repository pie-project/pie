#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine_cuda::{Boot, Lane, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;
use poem_ir::{CacheRow, Trace};

const DEPLOYMENT: &str = "gemma4-e4b-mini-l6-bf16-kv-bf16";
const CHUNK: u32 = 256;

fn logits(trace: Trace, path: &Path, prompt: &[u32], decodes: u32, ringed: bool) -> Vec<Vec<f32>> {
    served(DEPLOYMENT, trace, path, prompt, decodes, ringed).0
}

/// `deployment`'s logits over `prompt` in chunks and `decodes` greedy-free
/// decodes, and what its windowed fires did and the side streams it forks.
fn served(
    name: &str,
    trace: Trace,
    path: &Path,
    prompt: &[u32],
    decodes: u32,
    ringed: bool,
) -> (Vec<Vec<f32>>, engine_cuda::serve::RingSeen, usize) {
    let deployment = poem_compiler::catalog::deployment(name).expect("the catalog ships the mini");
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(1, CHUNK),
        patches: None,
        page_size: 16,
        context: 1024,
        slots: 1,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((engine_cuda::Graphs::Off, None)),
            diagnostics: engine_cuda::serve::diag::Diagnostics {
                windowed_state: ringed,
                ..engine_cuda::serve::diag::Diagnostics::default()
            },
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
    shell.open(0).expect("the slot opens");
    assert_eq!(
        !shell.windowed_layers().is_empty(),
        ringed,
        "the load keeps rings exactly where it is asked to"
    );
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
    let mut full = windowed.clone();
    for row in &mut full.caches {
        if let CacheRow::Kv { window, .. } = row {
            *window = None;
        }
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&windowed, 699).write(dir.path());
    let prompt: Vec<u32> = (0..3 * CHUNK).map(|at| (at * 31 + 5) % 4096).collect();
    let want = logits(full, &path, &prompt, 96, false);
    let got = logits(windowed, &path, &prompt, 96, false);
    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 1e-3,
            "fire {step}: the windowed rows answer {worst} away from the full ones"
        );
    }
}

// CONTRACT-BC 19: the windowed rows served as ring + buffer (each row's
// window a ring of pages in the lane's record, the fire's own rows past it
// on own-row pages and folded onto the ring after) answer what the window
// lists answer, bit for bit: the same rows, read by the same kernels.
#[test]
fn a_ringed_window_reads_what_the_window_list_reads() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let windowed = poem_compiler::catalog::deployment(DEPLOYMENT)
        .expect("the mini gemma")
        .trace(Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&windowed, 699).write(dir.path());
    let prompt: Vec<u32> = (0..3 * CHUNK).map(|at| (at * 31 + 5) % 4096).collect();
    let want = logits(windowed.clone(), &path, &prompt, 96, false);
    let got = logits(windowed, &path, &prompt, 96, true);
    assert_eq!(got.len(), want.len());
    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        let apart = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(apart, 0, "fire {step}: {apart} logits apart from the window list's");
    }
}

// CONTRACT-BC 19: a fire's fold lands after every read its lanes' attention
// makes, on every stream it forks: a fire whose lanes read ring pages that
// its own fold then overwrites (its own rows past its ring on own-row pages,
// folded onto ring pages it read) and fires that write their rows on their
// ring directly answer what the window lists answer, bit for bit, on loads
// whose fires fork regions onto side streams.
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
        let (want, _, _) = served(name, trace.clone(), &path, &prompt, 64, false);
        let (got, seen, sides) = served(name, trace, &path, &prompt, 64, true);
        eprintln!("{name}: {seen:?}, {sides} side stream(s)");
        assert!(seen.direct > 0, "{name}: some fire writes its ring directly");
        assert!(seen.own > 0, "{name}: some fire takes own-row pages");
        assert!(
            seen.read_then_folded > 0,
            "{name}: some fire folds onto ring pages it read"
        );
        forked += sides;
        for (step, (got, want)) in got.iter().zip(&want).enumerate() {
            let apart = got
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(apart, 0, "{name} fire {step}: {apart} logits apart from the window list's");
        }
    }
    assert!(forked > 0, "a load whose fires fork regions onto side streams");
}

/// `served`, its decodes reading a mask (every key but a stretch of the
/// prompt) through the masked arms.
fn served_masked(trace: Trace, path: &Path, prompt: &[u32], decodes: u32, ringed: bool) -> Vec<Vec<f32>> {
    use engine::fire::{Mask, Masking, Readout};
    let deployment = poem_compiler::catalog::deployment(DEPLOYMENT).expect("the mini");
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(1, CHUNK),
        patches: None,
        page_size: 16,
        context: 1024,
        slots: 1,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((engine_cuda::Graphs::Off, None)),
            diagnostics: engine_cuda::serve::diag::Diagnostics {
                windowed_state: ringed,
                ..engine_cuda::serve::diag::Diagnostics::default()
            },
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
// reads its windowed rows from ring + buffer as from the window lists.
#[test]
fn a_masked_lane_reads_its_ring_as_its_window_list() {
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
    let want = served_masked(trace.clone(), &path, &prompt, 48, false);
    let got = served_masked(trace, &path, &prompt, 48, true);
    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        let apart = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(apart, 0, "fire {step}: {apart} logits apart from the window list's");
    }
}
