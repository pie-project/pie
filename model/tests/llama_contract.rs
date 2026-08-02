//! The llama lineage's contract, pinned and proved.
//!
//! Two claims, in rising order of strength:
//!
//! 1. **The contract is pinned.** The authored contract for a fixture
//!    checkpoint is compared byte-for-byte against a committed golden, so a
//!    change to the author cannot happen quietly. Regenerate after an
//!    intended change with `UPDATE_GOLDEN=1 cargo test -p pie-model
//!    --features contract`.
//! 2. **The contract compiles and verifies.** The authored contract goes
//!    through `pie_loader::plan::compile` and the marshalled-view verifier —
//!    the same pipeline a driver boot runs — so the pin is of something the
//!    loader accepts, not just of plausible JSON.
//!
//! The fixture checkpoint is the same dense decoder
//! `loader/tests/golden_plans.rs` compiles its llama goldens from: q/k/v
//! fusion, grouped-query attention (k/v shard differently from q), a gated
//! MLP, an embedding and head of tied shape.
//!
//! **Migration note.** `loader/tests/golden/contracts/llama_dense_cuda.json`
//! is a dump of the C++ author, and predates its per-projection views into
//! the fused banks; this port follows the current C++ source, so the two
//! goldens agree except for those views. The authoritative differential —
//! C++ author vs this author over the same snapshot — runs where the CUDA
//! driver builds, via `pie_cuda_author_contract`.

#![cfg(feature = "contract")]

use std::path::PathBuf;

use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::plan::StorageTarget;
use pie_loader::plan::compile as compile_load_plan;
use pie_loader::types::{BackendKind, CheckpointFormat, DType, Encoding, FileId, TensorId};
use pie_loader::verify::ContractView;

use pie_model_common::facts::ModelFacts;
use pie_model_common::policy::Policy;
use pie_model::contract::author;

// ── the fixture checkpoint ──────────────────────────────────────────

/// Accumulates tensors at packed, ascending file offsets, the way a real
/// safetensors shard is laid out.
struct Checkpoint {
    tensors: Vec<RawTensor>,
    offset: u64,
}

impl Checkpoint {
    fn new() -> Self {
        Self {
            tensors: Vec::new(),
            offset: 0,
        }
    }

    fn push(&mut self, name: &str, shape: &[i64], encoding: Encoding) -> &mut Self {
        let elements: i64 = shape.iter().product();
        let span_bytes = match &encoding {
            Encoding::Raw(dtype) => u64::try_from(elements).unwrap() * dtype.bytes(),
            Encoding::Quant(spec) => {
                u64::try_from(elements).unwrap() * u64::from(spec.bits_per_element) / 8
            }
        };
        self.tensors.push(RawTensor {
            id: TensorId(self.tensors.len() as u32),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: self.offset,
            span_bytes,
            shape: shape.to_vec(),
            encoding,
        });
        self.offset += span_bytes;
        self
    }

    /// Write a file of the right size and point the metadata at it, so the
    /// verifier's source-extent checks run against something real.
    fn finish(self, name: &str) -> CheckpointMetadata {
        let path = std::env::temp_dir().join(format!(
            "pie_model_contract_{}_{}.safetensors",
            name,
            std::process::id()
        ));
        if std::fs::metadata(&path).map(|meta| meta.len()).ok() != Some(self.offset) {
            let staging = path.with_extension(format!("{:?}.partial", std::thread::current().id()));
            std::fs::write(&staging, vec![0u8; self.offset as usize])
                .expect("write fixture checkpoint");
            std::fs::rename(&staging, &path).expect("publish fixture checkpoint");
        }
        CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: path.to_string_lossy().into_owned(),
                size_bytes: self.offset,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: self.tensors,
        }
    }
}

fn bf16() -> Encoding {
    Encoding::Raw(DType::BF16)
}

fn llama_checkpoint() -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim, intermediate, vocab) = (256, 8, 2, 32, 704, 512);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[vocab, hidden], bf16());
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        ck.push(&format!("{p}.input_layernorm.weight"), &[hidden], bf16());
        ck.push(
            &format!("{p}.self_attn.q_proj.weight"),
            &[heads * head_dim, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv_heads * head_dim, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv_heads * head_dim, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.self_attn.o_proj.weight"),
            &[hidden, heads * head_dim],
            bf16(),
        );
        ck.push(
            &format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.mlp.gate_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.mlp.up_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.mlp.down_proj.weight"),
            &[hidden, intermediate],
            bf16(),
        );
    }
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.push("lm_head.weight", &[vocab, hidden], bf16());
    ck.finish("llama")
}

// ── the harness ─────────────────────────────────────────────────────

fn facts() -> ModelFacts {
    ModelFacts {
        model_type: "llama3".to_string(),
        num_hidden_layers: 2,
        head_dim: 32,
        ..Default::default()
    }
}

fn target(tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank,
        tp_size,
        preferred_alignment: 256,
        max_tile_bytes: 64 << 20,
        ..StorageTarget::default()
    }
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.contract.json"))
}

/// Author, pin against the golden, then compile and verify what was pinned.
fn check(name: &str, target: &StorageTarget) {
    let metadata = llama_checkpoint();
    let facts = facts();
    let policy = Policy::default();
    let contract = author(&facts, &metadata, target, &policy)
        .expect("authoring failed")
        .expect("llama3 must have an author");

    let mut fresh = serde_json::to_string_pretty(&contract).expect("serialize contract");
    fresh.push('\n');
    let path = golden_path(name);
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("create golden directory");
        std::fs::write(&path, &fresh).expect("write golden");
    } else {
        let stored = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!(
                "{name}: cannot read {}: {err}\n\
                 If this contract is new, regenerate with UPDATE_GOLDEN=1.",
                path.display()
            )
        });
        assert_eq!(
            stored, fresh,
            "{name}: the authored contract changed; regenerate with UPDATE_GOLDEN=1 \
             if the change is intended"
        );
    }

    // The pin is of something the loader accepts: same compile + verify a
    // driver boot runs.
    let plan = compile_load_plan(&metadata, &contract, target.clone())
        .unwrap_or_else(|err| panic!("{name}: compiling failed: {err}"));
    if let Err(violations) =
        pie_loader_capi::view::verify_marshalled(&plan, Some(&ContractView::of(&contract)))
    {
        let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
        panic!(
            "{name}: the plan does not honour its contract:\n  {}",
            listed.join("\n  ")
        );
    }
}

// ── the tests ───────────────────────────────────────────────────────

#[test]
fn llama_dense_cuda() {
    check("llama_dense_cuda", &target(0, 1));
}

/// Rank 1 of 2, not rank 0: rank 0 of a sharded load often coincides with
/// the unsharded plan for the leading slice of every tensor, so it is the
/// weakest rank to pin.
#[test]
fn llama_dense_cuda_tp1_of_2() {
    check("llama_dense_cuda_tp1_of_2", &target(1, 2));
}

/// The shape of what was authored, stated as assertions a reader can check
/// against the C++ author without a JSON diff: fused banks first, each bank
/// followed by views for its parts, then the untouched tensors.
#[test]
fn the_dense_join_publishes_banks_then_views() {
    let metadata = llama_checkpoint();
    let contract = author(&facts(), &metadata, &target(0, 1), &Policy::default())
        .unwrap()
        .unwrap();
    let names: Vec<&str> = contract
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert_eq!(
        &names[..8],
        &[
            "model.layers.0.self_attn.qkv_proj.fused.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.1.self_attn.qkv_proj.fused.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
        ],
        "qkv banks lead, each followed by its views"
    );
    // Every source tensor is reachable under a published name: the six
    // projections as views, everything else directly.
    for raw in metadata.weights() {
        assert!(
            names.contains(&raw.name.as_str()),
            "{} is not published under its own name",
            raw.name
        );
    }
    // GQA shapes survived the join: 8 q heads + 2 kv heads of 32 = 384 rows.
    let qkv = &contract.tensors[0];
    assert_eq!(qkv.shape.as_deref(), Some(&[384, 256][..]));
}

/// Under TP the parts shard before the join, and the declared shapes are
/// this rank's, not the checkpoint's.
#[test]
fn tp_shards_each_part_before_the_join() {
    let metadata = llama_checkpoint();
    let contract = author(&facts(), &metadata, &target(1, 2), &Policy::default())
        .unwrap()
        .unwrap();
    let qkv = &contract.tensors[0];
    // (8 q heads + 2 kv heads) of 32, halved: 128 + 32 + 32 = 192 rows.
    assert_eq!(qkv.shape.as_deref(), Some(&[192, 256][..]));
    // o_proj is row-parallel on axis 1 and keeps its row count.
    let o_proj = contract
        .tensors
        .iter()
        .find(|tensor| tensor.name == "model.layers.0.self_attn.o_proj.weight")
        .expect("o_proj is published");
    assert_eq!(o_proj.shape.as_deref(), Some(&[256, 128][..]));
}
