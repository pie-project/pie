#![cfg(feature = "cuda")]

mod common_encoder;

use std::path::Path;

use common_encoder::{Weights, contract_for};
use engine_cuda::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};
use model_ir::{CacheRow, Trace};

const SKU: &str = "gemma4-e4b-mini-l6-bf16-kv-bf16";
const CHUNK: u32 = 256;

fn logits(trace: Trace, path: &Path, prompt: &[u32], decodes: u32) -> Vec<Vec<f32>> {
    let sku = models::sku(SKU).expect("the catalog ships the mini gemma");
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        classify: sku.classify,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(1, CHUNK),
        patches: None,
        profile: None,
        page_size: 16,
        context: 1024,
        slots: 1,
        pages: 64,
        ordinal: 0,
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    shell.open(0).expect("the slot opens");
    let mut fire = |tokens: &[u32]| {
        let word = (sku.classify)(&Request::new(tokens.len() as u32, false));
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
    out
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
    let windowed = (models::sku(SKU).expect("the mini gemma").trace)(Platform::Cuda);
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
    let want = logits(full, &path, &prompt, 96);
    let got = logits(windowed, &path, &prompt, 96);
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
