use std::path::{Path, PathBuf};

use engine_cuda::{Boot, Shell};
use poem::Platform;
use poem_compiler::Budget;

/// A hybrid deployment, and a plain-attention one whose masked, adapted
/// and score-capturing classes arm one wide body a bucket.
const HYBRID: (&str, &str) = ("qwen35-d0.8b-bf16-kv-bf16", "Qwen3.5-0.8B");
const ATTENTION: (&str, &str) = ("qwen3-0.6b-bf16-kv-bf16", "Qwen3-0.6B");

const PAGE: u32 = 16;

const LANES: u32 = 16;
const TOKENS: u32 = 512;

fn snapshot(model: &str, stated: &str) -> Option<PathBuf> {
    if let Ok(stated) = std::env::var(stated) {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = Path::new(&home).join(format!(
        ".cache/huggingface/hub/models--Qwen--{model}/snapshots"
    ));
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

fn ready(what: &str, (row, model): (&str, &str), stated: &str) -> Option<Shell> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot(model, stated) else {
        eprintln!(
            "skipping {what}: no {model} snapshot in the hugging face cache \
             (set {stated})"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let deployment =
        poem_compiler::catalog::deployment(row).expect("the catalog ships the deployment");
    let trace = deployment.trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = deployment
        .contract(&source, Platform::Cuda)
        .expect("the deployment's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        page_size: PAGE,
        context: 512,
        slots: LANES,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    Some(shell)
}

fn answers(shell: &mut Shell) {
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

#[test]
fn every_armed_body_answers_its_eager_walk() {
    let what = "every_armed_body_answers_its_eager_walk";
    if let Some(mut shell) = ready(what, HYBRID, "PIE_SMOKE_SNAPSHOT") {
        answers(&mut shell);
    }
}

/// A wide body is keyed by every class of its tier and replays for any
/// subset of them: the golden walks it at full width and with each class
/// off the plain tier alone.
#[test]
fn every_armed_wide_body_answers_its_eager_walk() {
    let what = "every_armed_wide_body_answers_its_eager_walk";
    let Some(mut shell) = ready(what, ATTENTION, "PIE_QWEN3_SNAPSHOT") else {
        return;
    };
    let wide = shell.armed().map_or(0, |armed| {
        armed.kinds[engine_cuda::serve::Kind::Wide.at()].0
    });
    assert!(wide > 0, "the lattice armed no wide body");
    answers(&mut shell);
}
