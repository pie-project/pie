//! A model loaded after another was dropped, in the same process, is
//! fitted to the card as it was before: what the dropped shell's buffers
//! left kept for reuse goes back to the driver, so neither the fit nor
//! another process on the card counts it as taken. The kept buffers once
//! stayed (5.5 GiB after the mega-MoE test's two loads), and a fit at a
//! tight `device_memory_share` that followed found nothing under it.
#![cfg(feature = "device")]
#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use engine_cuda::{Boot, Knobs, Lane, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;

const DEPLOYMENT: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;

/// What the tight load's share leaves it under the ceiling past what the
/// card already holds: its weights and a few sequences, so the fit sizes
/// the pool to the card.
const HEADROOM: u64 = 4 << 30;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

fn device_bytes() -> (u64, u64) {
    use engine_cuda::device::rt;

    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    assert_eq!(asked, rt::cudaError::cudaSuccess, "cudaMemGetInfo answers");
    (free as u64, total as u64)
}

#[test]
fn a_model_loaded_after_another_fits_as_in_a_fresh_process() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!("no Qwen3.5-0.8B snapshot in the hugging face cache (set PIE_SMOKE_SNAPSHOT)");
        return;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("{checkpoint:?} holds no tensor container");
        return;
    };
    let deployment =
        poem_compiler::catalog::deployment(DEPLOYMENT).expect("the catalog ships the deployment");
    let trace = deployment.trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = deployment
        .contract(&source, Platform::Cuda)
        .expect("the deployment's import contract fits its own checkpoint");
    drop(source);

    let (free, total) = device_bytes();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a card's byte count is far inside f64's exact integer range"
    )]
    let tight = ((total - free + HEADROOM) as f64 / total as f64).min(1.0);
    let load = |lanes: u32, tokens: u32, share: f64| {
        Shell::load(Boot {
            voxels: None,
            deferred_tier: false,
            residency: engine_cuda::experts::Plan::default(),
            trace: trace.clone(),
            contract: &contract,
            checkpoint: &checkpoint,
            budget: Budget::new(lanes, tokens),
            patches: None,
            page_size: PAGE,
            context: CONTEXT,
            slots: 256,
            kv_budget: None,
            memory_share: share,
            ordinal: 0,
            knobs: Knobs {
                chip: Some(engine_cuda::chip::pinned(engine_cuda::Graphs::Off, None)),
                ..Knobs::default()
            },
            cache_dir: None,
            runahead: engine::runahead::Runahead::F1,
            world: engine_cuda::World::default(),
            expert: engine_cuda::World::default(),
            comm: core::ptr::null_mut(),
        })
    };

    // The tight load before the other: the fit a fresh process makes.
    let first = load(16, 2048, tight).expect("the tight share holds the model");
    let fresh = first.paging();
    drop(first);
    let (after_first, _) = device_bytes();

    // Another load, wider and fired, then dropped: its weights, arena and
    // the buffers its fire grew are all dropped with it.
    let mut other = load(32, 4096, engine::load::DEFAULT_DEVICE_MEMORY_SHARE)
        .expect("the wide load fits the card");
    other.open(0).expect("a slot opens");
    let tokens: Vec<u32> = (0..300).map(|at| 1000 + at * 37).collect();
    other
        .fire(&[Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(300, false)),
            tokens: &tokens,
        }])
        .expect("the wide load fires");
    drop(other);
    let (after_other, _) = device_bytes();
    eprintln!(
        "free {} MiB after the tight load dropped, {} MiB after the wide one; {} MiB kept",
        after_first >> 20,
        after_other >> 20,
        engine_cuda::device::held::kept_bytes() >> 20
    );

    let again = load(16, 2048, tight)
        .expect("the tight share holds the model again, after another was dropped");
    assert_eq!(
        again.paging(),
        fresh,
        "the tight load is fitted otherwise after another load was dropped"
    );
    drop(again);
    assert_eq!(
        engine_cuda::device::held::kept_bytes(),
        0,
        "a dropped shell's buffers stay kept with nothing left to reuse them"
    );
}
