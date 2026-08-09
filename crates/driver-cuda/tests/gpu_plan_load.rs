//! The load path through `model-loader`'s plan, on a real checkpoint.
//!
//! What this holds: a compiled plan executes into DEVICE memory, the tensors
//! it names are where the plan says they are, and the fused projections the
//! shell used to build by hand come out of the plan instead.

#![cfg(all(feature = "cuda-13", feature = "abi"))]

use std::path::PathBuf;

use driver_cuda::gpu::weights::plan::{compile_load_plan, cuda_storage_target};
use driver_cuda::gpu::weights::stage::stage_plan_weights;

/// A cached HF snapshot, or `None` to skip.
fn snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{repo}"))
        .join("snapshots");
    std::fs::read_dir(snaps).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        p.join("model.safetensors").is_file().then_some(p)
    })
}

fn descriptor() -> Option<String> {
    let p = PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/\
         7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad/qwen3_descriptor.json",
    );
    std::fs::read_to_string(p).ok()
}

#[test]
fn a_checkpoint_stages_into_device_memory_through_its_plan() {
    let (Some(snap), Some(desc)) = (snapshot("Qwen--Qwen3-0.6B"), descriptor()) else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(&snap)
        .expect("the checkpoint parses");
    let target = cuda_storage_target(0, 1);
    let (plan, _moe) =
        compile_load_plan(&snap, &meta, &target, &desc).expect("the plan compiles");

    // THE JOINS ARE IN THE PLAN. `Projections::Fused` is what the CUDA
    // GEMMs want, and the shell used to satisfy it by reading q/k/v back
    // off the device and re-uploading their concatenation.
    let fused = plan
        .tensors
        .iter()
        .filter(|t| t.name.contains("qkv_proj.fused"))
        .count();
    assert!(
        fused > 0,
        "the plan carries no fused qkv; the driver would have to build them"
    );

    let alloc = driver_cuda::gpu::device::Allocator::new();
    let staged = stage_plan_weights(&plan, &snap, &alloc).expect("the plan executes");

    assert!(
        staged.spans.len() >= plan.tensors.len(),
        "every tensor the plan names is staged: {} spans for {} tensors",
        staged.spans.len(),
        plan.tensors.len()
    );
    for (name, span) in &staged.spans {
        assert!(!span.ptr.is_null(), "{name} has no address");
        assert!(span.bytes > 0, "{name} is empty");
    }
    // The arena is one allocation, so the whole model is contiguous and the
    // spans are offsets into it — the property that makes this cheaper than
    // a per-tensor `cudaMalloc`.
    assert_eq!(
        staged.owned.len(),
        1,
        "a resident plan should leave nothing outside the arena"
    );
}

/// A rank of a tensor-parallel group reads only its own band.
///
/// The whole of what makes a load sharded: `tp_rank`/`tp_size` in the target.
/// Every family states its splits in terms of them, so a rank compiles a plan
/// that reads its bands out of the checkpoint and sizes its arena to them —
/// the driver never slices a tensor itself.
///
/// One GPU is enough to hold this. What needs two is the COLLECTIVE that puts
/// the shards back together, and that is orchestration, not layout.
#[test]
fn a_rank_of_a_tp_group_plans_a_smaller_arena() {
    let (Some(snap), Some(desc)) = (snapshot("Qwen--Qwen3-0.6B"), descriptor()) else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(&snap)
        .expect("the checkpoint parses");
    let whole = compile_load_plan(&snap, &meta, &cuda_storage_target(0, 1), &desc)
        .expect("the unsharded plan compiles")
        .0;
    let half = compile_load_plan(&snap, &meta, &cuda_storage_target(0, 2), &desc)
        .expect("rank 0 of 2 compiles")
        .0;
    let other = compile_load_plan(&snap, &meta, &cuda_storage_target(1, 2), &desc)
        .expect("rank 1 of 2 compiles")
        .0;

    assert!(
        half.memory.persistent_bytes < whole.memory.persistent_bytes,
        "a rank of two plans {} bytes where one rank plans {}",
        half.memory.persistent_bytes,
        whole.memory.persistent_bytes
    );
    // The two ranks are the same SIZE and different CONTENT: same layout,
    // different file offsets. If they were byte-identical the split would be
    // naming a band without moving it.
    assert_eq!(
        half.memory.persistent_bytes, other.memory.persistent_bytes,
        "the two ranks of a symmetric split are equally wide"
    );
    assert_ne!(
        half.instrs, other.instrs,
        "the ranks read the same bytes; nothing was actually split"
    );
}
