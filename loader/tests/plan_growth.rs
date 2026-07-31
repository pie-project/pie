//! How the plan grows with the model.
//!
//! Every golden is two layers, which is enough to pin a plan's *shape* but says
//! nothing about how it scales: a compiler that emitted one instruction per
//! tensor and one that coalesced a whole layer look identical at two layers if
//! neither has anything to coalesce across. Loading a 30-to-80-layer model is
//! the only case that actually happens, so the growth is worth a test rather
//! than an extrapolation.
//!
//! Rather than hand-author a large contract, this expands the stored two-layer
//! Llama one by cloning its layer-0 tensors under new indices. The result is
//! the real contract at a real size, so nothing here can drift from what the
//! goldens compile.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::contract::ModelContract;
use pie_loader::plan::compile as compile_load_plan;
use pie_loader::plan::{
    CUDA_TILE_MAP_MASK, FUSION_FP8_TO_MXFP4, LoadPlan, StorageInstr, StorageTarget,
};
use pie_loader::types::{BackendKind, CheckpointFormat, DType, Encoding, FileId, TensorId};

/// A stored Llama contract, with its two layers replayed up to `layers`.
///
/// Sharding is stated by the contract, not the target, so the sharded plan has
/// to grow the sharded contract -- the same fixture the tp golden compiles.
fn llama_contract(name: &str, layers: usize) -> ModelContract {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/contracts")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path).expect("stored llama contract");
    let stored: serde_json::Value = serde_json::from_str(&text).expect("contract parses");

    let tensors = stored["tensors"].as_array().expect("tensor list");
    let mut grown: Vec<serde_json::Value> = Vec::new();
    for tensor in tensors {
        let name = tensor["name"].as_str().unwrap_or_default();
        // A layer tensor is replayed; everything else (embeddings, the final
        // norm, the head) exists once however many layers there are, and it is
        // exactly that fixed part which decides whether per-tensor bookkeeping
        // amortises.
        if !name.starts_with("model.layers.0.") {
            if !name.starts_with("model.layers.1.") {
                grown.push(tensor.clone());
            }
            continue;
        }
        for layer in 0..layers {
            let cloned = tensor
                .to_string()
                .replace("model.layers.0.", &format!("model.layers.{layer}."));
            grown.push(serde_json::from_str(&cloned).expect("cloned tensor parses"));
        }
    }

    serde_json::from_value(serde_json::json!({
        "alignment": stored["alignment"],
        "tensors": grown,
    }))
    .expect("grown contract parses")
}

/// The checkpoint the contract above reads, at the same size.
fn llama_checkpoint(layers: usize) -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim, intermediate, vocab) = (256, 8, 2, 32, 704, 512);
    let bf16 = Encoding::Raw(DType::BF16);
    let mut tensors: Vec<RawTensor> = Vec::new();
    let mut offset = 0_u64;
    // Captures rather than taking the accumulators as parameters: the calls
    // below are the readable part of this fixture and a four-argument helper
    // wraps every one of them across four lines.
    let mut push = |name: String, shape: Vec<i64>| {
        let elements: i64 = shape.iter().product();
        let span_bytes = u64::try_from(elements).unwrap() * DType::BF16.bytes();
        tensors.push(RawTensor {
            id: TensorId(tensors.len() as u32),
            name,
            file_id: FileId(0),
            file_offset: offset,
            span_bytes,
            shape,
            encoding: bf16.clone(),
        });
        offset += span_bytes;
    };

    push("model.embed_tokens.weight".into(), vec![vocab, hidden]);
    for layer in 0..layers {
        let p = format!("model.layers.{layer}");
        push(format!("{p}.input_layernorm.weight"), vec![hidden]);
        push(
            format!("{p}.self_attn.q_proj.weight"),
            vec![heads * head_dim, hidden],
        );
        push(
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv_heads * head_dim, hidden],
        );
        push(
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv_heads * head_dim, hidden],
        );
        push(
            format!("{p}.self_attn.o_proj.weight"),
            vec![hidden, heads * head_dim],
        );
        push(format!("{p}.post_attention_layernorm.weight"), vec![hidden]);
        push(
            format!("{p}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            format!("{p}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            format!("{p}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
    }
    push("model.norm.weight".into(), vec![hidden]);
    push("lm_head.weight".into(), vec![vocab, hidden]);

    // Tests run in parallel and several want the same layer count, so the
    // scratch name has to be unique per call: two threads racing on one
    // temporary would leave a reader looking at a file another had renamed
    // away, which reports as a plan that reads past the end of its checkpoint.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!("pie_plan_growth_{layers}.bin"));
    let scratch = path.with_extension(format!(
        "{}.{}.part",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    // Sparse, not written. Verification only checks that every source extent
    // lands inside the file it names, so the bytes are never read -- and an
    // 80-layer fixture is 113 MB, which is real memory on a tmpfs `/tmp` and
    // enough to exhaust a quota over a few runs. `set_len` costs zero blocks.
    std::fs::File::create(&scratch)
        .and_then(|file| file.set_len(offset))
        .expect("size the checkpoint");
    std::fs::rename(&scratch, &path).expect("publish checkpoint");

    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: path.to_string_lossy().into_owned(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    }
}

/// The same target the CUDA goldens compile against.
fn target(tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank,
        tp_size,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        native_mxfp4_moe: false,
        fusion_mask: FUSION_FP8_TO_MXFP4,
        encode_scratch_dtype: DType::BF16,
        block_scale_rows: 128,
    }
}

/// The instruction histogram the loader itself renders, so this test counts
/// what an operator dump would show rather than a second table of names.
fn histogram(plan: &LoadPlan) -> BTreeMap<String, u64> {
    let stats: serde_json::Value =
        serde_json::from_str(&pie_loader::dump::plan_stats_json(plan)).expect("stats are JSON");
    stats["instruction_kinds"]
        .as_object()
        .expect("histogram")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or_default()))
        .collect()
}

fn compile(layers: usize, world: u32) -> LoadPlan {
    let (name, rank) = if world > 1 {
        ("llama_dense_cuda_tp1_of_2", 1)
    } else {
        ("llama_dense_cuda", 0)
    };
    let metadata = llama_checkpoint(layers);
    let contract = llama_contract(name, layers);
    compile_load_plan(&metadata, &contract, target(rank, world))
        .unwrap_or_else(|err| panic!("{name} at {layers} layers: compiling failed: {err}"))
}

/// The marginal cost of a layer between successive model sizes, which is the
/// growth rate with the fixed part -- embeddings, final norm, head -- divided
/// out rather than assumed away.
fn per_layer(counts: &[(usize, usize)]) -> Vec<f64> {
    counts
        .windows(2)
        .map(|w| (w[1].1 - w[0].1) as f64 / (w[1].0 - w[0].0) as f64)
        .collect()
}

/// The plan grows linearly in the model, at one instruction per twelve.
///
/// Measured 2 -> 8 -> 32 -> 80 layers: 31, 103, 391, 968 instructions, and a
/// marginal cost of exactly 12 per layer at every step. Twelve is not a tuning
/// number, it is six materialised tensors -- the fused qkv, the fused gate/up,
/// two norms, `o_proj` and `down_proj` -- each costing one `Allocate` and one
/// `Finalize`, which is the floor: a loader cannot give a tensor a buffer and a
/// name in fewer than two instructions.
///
/// This is the test the goldens cannot be: at two layers a compiler that
/// emitted one instruction per *byte range* and one that coalesced a whole
/// model are within a few instructions of each other.
#[test]
fn a_layer_costs_a_constant_number_of_instructions() {
    let sizes = [2_usize, 8, 32, 80];
    let totals: Vec<(usize, usize)> = sizes
        .iter()
        .map(|&layers| (layers, compile(layers, 1).instrs.len()))
        .collect();

    let marginals = per_layer(&totals);
    for (window, marginal) in marginals.iter().enumerate() {
        assert!(
            (marginal - 12.0).abs() <= 0.05,
            "a layer cost {marginal} instructions between {} and {} layers, not 12: {totals:?}",
            sizes[window],
            sizes[window + 1]
        );
    }
}

/// Bookkeeping is one `Allocate` and one `Finalize` per tensor, and nothing else
/// grows with the model.
///
/// Those two are 79% of the instructions in every golden, which invites the
/// question of whether they can be compressed. They cannot: the count is
/// exactly the number of tensors the plan produces, at every size, so there is
/// no batching left to do. What this pins is that it stays exactly that -- an
/// extra per-tensor instruction would be invisible in a golden and obvious here.
#[test]
fn bookkeeping_is_exactly_one_allocate_and_one_finalize_per_tensor() {
    for layers in [2_usize, 80] {
        let plan = compile(layers, 1);
        let counts = histogram(&plan);
        let tensors = plan.tensors.len() as u64;
        assert_eq!(
            counts.get("Finalize").copied().unwrap_or_default(),
            tensors,
            "{layers} layers: a plan that publishes {tensors} tensors finalized a different number"
        );
        assert!(
            counts.get("Allocate").copied().unwrap_or_default() >= tensors,
            "{layers} layers: fewer allocations than tensors"
        );
    }
}

/// Whole-model coalescing does not decay with size: an eighty-layer model is
/// still *one* copy, unless its own byte budget says otherwise.
///
/// At 80 layers the plan carries two bulk copies rather than one, and the
/// reason is not that `hoist-bulk-arena-writes` gave up -- it is that the model
/// is 107 MiB and `max_tile_bytes` is 64. Raising the budget collapses it back
/// to one, which is what separates a cap doing its job from a pass that stopped
/// working, and is the kind of thing a golden at two layers cannot see at all.
#[test]
fn an_eighty_layer_model_is_one_arena_copy_unless_the_tile_budget_says_otherwise() {
    let capped = compile(80, 1);
    assert_eq!(
        histogram(&capped)
            .get("BulkExtentWrite")
            .copied()
            .unwrap_or_default(),
        2,
        "80 layers at a 64 MiB tile budget should split into two copies"
    );

    let metadata = llama_checkpoint(80);
    let contract = llama_contract("llama_dense_cuda", 80);
    let mut generous = target(0, 1);
    generous.max_tile_bytes = 1 << 30;
    let plan = compile_load_plan(&metadata, &contract, generous).expect("compiling failed");
    assert_eq!(
        histogram(&plan)
            .get("BulkExtentWrite")
            .copied()
            .unwrap_or_default(),
        1,
        "with room for the whole model the arena should take one copy"
    );
}

/// Sharding is what stops a model being one copy, and its cost is also flat.
///
/// A rank reads a band of every tensor, so the runs are no longer adjacent and
/// the single arena copy fragments: 27 instructions per layer against 12. Two
/// of those are the strided reads -- `o_proj` and `down_proj`, the row-parallel
/// pair, the only tensors whose shard cuts across rows. That pair is what the
/// CUDA executor's `cudaMemcpy2D` path exists for, so its count per layer is
/// worth pinning: it is the multiplier on that path's cost.
#[test]
fn a_sharded_load_pays_two_strided_reads_per_layer_at_any_size() {
    let mut totals = Vec::new();
    for layers in [2_usize, 8, 32, 80] {
        let plan = compile(layers, 2);
        let strided = plan
            .instrs
            .iter()
            .filter(|instr| match instr {
                StorageInstr::ExtentWrite { source, .. } => !source.stride.is_byte_run(),
                _ => false,
            })
            .count();
        assert_eq!(
            strided,
            2 * layers,
            "{layers} layers: {strided} strided reads, expected two per layer"
        );
        totals.push((layers, plan.instrs.len()));
    }

    for marginal in per_layer(&totals) {
        assert!(
            (marginal - 27.0).abs() <= 0.7,
            "a sharded layer cost {marginal} instructions, not 27: {totals:?}"
        );
    }
}
