#![cfg(feature = "device")]
#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Knobs, Shell};
use poem::Platform;
use poem_compiler::Budget;
use poem_compiler::catalog::Deployment;

const DEPLOYMENT: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PAGE: u32 = 16;
const LANES: u32 = 16;
const TOKENS: u32 = 8192;
const CONTEXT: u32 = 32768;

// The card shrinks until it no longer loads. What it costs to hold the
// weight tier and one sequence depends on the runtime (what a CUDA or a ROCm
// context keeps on the device before a load begins) as much as on the
// model, so no one share is short on every card: the test finds the
// shortest share the card loads at, within `STEP`, and asks what gave way
// there. A share this large holds the whole working set on any card that
// runs the test; a share this small holds no sequence.
const ROOMY: u64 = 16_000_000_000;
const CRAMPED: u64 = 1_000_000_000;
const STEP: u64 = 64 * 1024 * 1024;

fn total_bytes() -> u64 {
    use engine_cuda::device::rt;

    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    assert_eq!(asked, rt::cudaError::cudaSuccess, "cudaMemGetInfo answers");
    total as u64
}

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

#[expect(
    clippy::cast_precision_loss,
    reason = "a card's byte count is far inside f64's exact integer range"
)]
fn load(
    deployment: &Deployment,
    contract: &ModelContract,
    checkpoint: &Path,
    share: u64,
    total: u64,
) -> Option<Shell> {
    Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace: deployment.trace(Platform::Cuda),
        contract,
        checkpoint,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 256,
        kv_budget: None,
        memory_share: share as f64 / total as f64,
        ordinal: 0,
        knobs: Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .ok()
}

#[test]
fn a_short_card_keeps_the_declared_context() {
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
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = deployment
        .contract(&source, Platform::Cuda)
        .expect("the deployment's import contract fits its own checkpoint");
    drop(source);
    let total = total_bytes();
    let at = |share: u64| load(deployment, &contract, &checkpoint, share, total);

    let roomy = at(ROOMY).expect("a roomy card loads");
    assert_eq!(
        roomy.budget().max_tokens,
        TOKENS,
        "a roomy card holds the whole working set"
    );
    drop(roomy);
    assert!(at(CRAMPED).is_none(), "a cramped card holds no sequence");

    let (mut short, mut long) = (CRAMPED, ROOMY);
    while long - short > STEP {
        let mid = short + (long - short) / 2;
        if at(mid).is_some() {
            long = mid;
        } else {
            short = mid;
        }
    }
    let shell = at(long).expect("the shortest share found loads again");

    let served = shell.budget().max_tokens;
    assert!(
        served < TOKENS,
        "at the shortest share the card loads, the token budget gave way: {served} of {TOKENS}"
    );
    assert!(
        served >= 1024,
        "and the halving stopped at its floor, not below: {served}"
    );
    let paging = shell.paging();
    assert_eq!(
        paging.context(),
        CONTEXT,
        "the declared context is the sequence's ceiling, and a sequence spans many \
         forwards; only the per-forward budget shrinks"
    );
    assert_eq!(paging.pages_per_slot, CONTEXT / PAGE);
    assert!(
        paging.pages() >= u64::from(paging.pages_per_slot),
        "and the pool seats at least one sequence at it: {} pages",
        paging.pages()
    );
}
