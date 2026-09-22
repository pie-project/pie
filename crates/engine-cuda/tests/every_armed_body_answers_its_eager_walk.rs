use std::path::{Path, PathBuf};

use engine_cuda::{Boot, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PAGE: u32 = 16;

const LANES: u32 = 16;
const TOKENS: u32 = 512;

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

fn ready(what: &str) -> Option<Shell> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
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
        context: 512,
        slots: LANES,
        pages: LANES * 512 / PAGE,
        ordinal: 0,
        graphs: engine_cuda::Graphs::On,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    Some(shell)
}

#[test]
fn every_armed_body_answers_its_eager_walk() {
    let Some(mut shell) = ready("every_armed_body_answers_its_eager_walk") else {
        return;
    };
    let armed = shell.armed().map_or(0, |armed| armed.armed);
    assert!(
        armed > 0,
        "the lattice armed no body, so there is nothing to answer for"
    );
    let verified = shell
        .verify_bodies()
        .expect("every armed body answers its eager walk");
    assert_eq!(
        verified, armed,
        "the golden walked {verified} bodies where the arming pass captured {armed}"
    );
}
