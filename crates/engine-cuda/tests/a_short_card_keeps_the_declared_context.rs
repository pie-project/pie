#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use engine_cuda::{Boot, Knobs, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PAGE: u32 = 16;
const LANES: u32 = 16;
const TOKENS: u32 = 8192;
const CONTEXT: u32 = 32768;

// A tenth of a 32 GB card: enough for the weight tier and a sequence, not
// for an 8192-token working set, so the boot halves the token budget.
const FRACTION: f64 = 0.10;

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
    let sku = models::sku(SKU).expect("the catalog ships the SKU");
    let trace = (sku.trace)(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = sku
        .contract(&source, Platform::Cuda)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        classify: sku.classify,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 256,
        pages: 256 * CONTEXT / PAGE,
        ordinal: 0,
        graphs: engine_cuda::Graphs::On,
        knobs: Knobs {
            gpu_mem_utilization: FRACTION,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("a tenth of the card still holds one sequence");

    let served = shell.budget().max_tokens;
    assert!(
        served < TOKENS,
        "the fraction is tight enough that the token budget halved: {served} of {TOKENS}"
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
