#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use engine_cuda::device::elastic::budget_bytes;
use engine_cuda::{Boot, Knobs, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PAGE: u32 = 16;
const LANES: u32 = 16;
const TOKENS: u32 = 2048;
const CONTEXT: u32 = 2048;

// Tight enough that the fit sizes the pool to the card and hands it
// everything left under the ceiling, as a serve on a short card does.
const FRACTION: f64 = 0.15;

// What a fire allocated beside the pool on the L40S of issue #672, past the
// reserve the fit held for the guests' programs: enough to leave the pool
// no room under the ceiling.
const TAKEN_PAST_THE_RESERVE: u64 = 256 << 20;

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
    use cudarc::runtime::sys as rt;

    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    assert_eq!(asked, rt::cudaError::cudaSuccess, "cudaMemGetInfo answers");
    (free as u64, total as u64)
}

#[test]
fn a_fitted_pool_is_pledged_what_it_declared() {
    // Issue #672: the fit declared the pool from what the card had left and
    // the runtime admitted a c64 decode frame against it; a fire then took
    // room beside the pool, and the commit under the frame recalibrated the
    // pool's budget to what was free, refusing the growth as retryable past
    // static admission with "wanted 14889779200, 15699279872 available".
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

    let mut shell = Shell::load(Boot {
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
        graphs: engine_cuda::Graphs::Off,
        knobs: Knobs {
            gpu_mem_utilization: FRACTION,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("a short card still holds one sequence");

    let paging = shell.paging();
    assert!(
        paging.slots >= 2,
        "the pool declares recurrent slots to commit: {}",
        paging.slots
    );

    // A fire takes the room the fit left under the ceiling, and more: the
    // pool's recalibration now finds nothing free under the operator's
    // fraction, while the card itself has plenty.
    let (free, total) = device_bytes();
    let headroom = budget_bytes(free, total, FRACTION);
    let taken = engine_cuda::device::Buffer::zeroed(
        usize::try_from(headroom + TAKEN_PAST_THE_RESERVE).expect("a buffer the card holds"),
    )
    .expect("the card holds what the fraction left");
    let (free, total) = device_bytes();
    assert_eq!(
        budget_bytes(free, total, FRACTION),
        0,
        "nothing is left under the ceiling"
    );

    // Every slot the pool declared is inside the watermark the runtime
    // admits against, so each opens: the slab it maps was the fit's promise,
    // not the recalibration's to withdraw.
    for slot in 0..paging.slots {
        shell.open(slot).unwrap_or_else(|refusal| {
            panic!(
                "slot {slot} of the {} declared was refused past static admission: {refusal}",
                paging.slots
            )
        });
    }
    drop(taken);
}
