//! Every ported family's contract, pinned and proved.
//!
//! One fixture checkpoint per family, shaped to reach the passes that make
//! the family worth porting — the GDN blocked shards, the Mamba unit folds,
//! the MXFP4 triplets, the W4A16 stacks — and one `check` per interesting
//! point in (tp, policy) space. Each check pins the authored contract
//! against a committed golden and then pushes it through
//! `pie_loader::plan::compile` + the marshalled-view verifier, the same
//! pipeline a driver boot runs.
//!
//! Regenerate after an intended change:
//! `UPDATE_GOLDEN=1 cargo test -p pie-model --features contract --test family_contracts`
//!
//! The authoritative C++ differential (same snapshot, same request, both
//! authors) still runs where the CUDA driver builds; these goldens pin the
//! Rust author against *itself* so a change cannot happen quietly.

#![cfg(feature = "contract")]

use std::path::PathBuf;

use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::plan::{
    CUDA_TILE_MAP_MASK, METAL_TILE_MAP_MASK, StorageTarget, compile as compile_load_plan,
};
use pie_loader::types::{BackendKind, CheckpointFormat, DType, Encoding, FileId, TensorId};
use pie_loader::verify::ContractView;

use pie_model_common::facts::ModelFacts;
use pie_model_common::policy::{Mxfp4MoeRequest, Naming, Policy, Projections, RuntimeQuant};
use pie_model::contract::author;

// ── fixture machinery ───────────────────────────────────────────────

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

    fn finish(self, name: &str) -> CheckpointMetadata {
        let path = std::env::temp_dir().join(format!(
            "pie_model_family_{}_{}.safetensors",
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

fn f32enc() -> Encoding {
    Encoding::Raw(DType::F32)
}

fn u8enc() -> Encoding {
    Encoding::Raw(DType::U8)
}

fn target(tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank,
        tp_size,
        preferred_alignment: 256,
        max_tile_bytes: 64 << 20,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    }
}

fn facts(model_type: &str, layers: u32) -> ModelFacts {
    ModelFacts {
        model_type: model_type.to_string(),
        num_hidden_layers: layers,
        ..Default::default()
    }
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.contract.json"))
}

/// Author, pin against the golden, then compile and verify what was pinned.
fn check(
    name: &str,
    metadata: &CheckpointMetadata,
    facts: &ModelFacts,
    target: &StorageTarget,
    policy: &Policy,
) {
    let contract = author(facts, metadata, target, policy)
        .unwrap_or_else(|err| panic!("{name}: authoring failed: {err}"))
        .unwrap_or_else(|| panic!("{name}: no author for {}", facts.model_type));

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

    let plan = compile_load_plan(metadata, &contract, target.clone())
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

/// A dense attention block plus a gated MLP under `prefix`, the shapes GQA
/// needs to make k/v shard differently from q.
fn dense_layer(
    ck: &mut Checkpoint,
    prefix: &str,
    hidden: i64,
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
    intermediate: i64,
) {
    ck.push(
        &format!("{prefix}input_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}self_attn.q_proj.weight"),
        &[heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}self_attn.k_proj.weight"),
        &[kv_heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}self_attn.v_proj.weight"),
        &[kv_heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}self_attn.o_proj.weight"),
        &[hidden, heads * head_dim],
        bf16(),
    );
    ck.push(
        &format!("{prefix}post_attention_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}mlp.gate_proj.weight"),
        &[intermediate, hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}mlp.up_proj.weight"),
        &[intermediate, hidden],
        bf16(),
    );
    ck.push(
        &format!("{prefix}mlp.down_proj.weight"),
        &[hidden, intermediate],
        bf16(),
    );
}

// ── gemma4: nested decoder, router-scale fold ───────────────────────

fn gemma4_checkpoint() -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim, intermediate) = (64, 4, 2, 16, 96);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[128, hidden], bf16());
    let p = "model.language_model.layers.0.";
    dense_layer(&mut ck, p, hidden, heads, kv_heads, head_dim, intermediate);
    ck.push(&format!("{p}router.scale"), &[hidden], bf16());
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("gemma4")
}

#[test]
fn gemma4_dense_cuda() {
    let mut facts = facts("gemma4_text", 1);
    facts.head_dim = 16;
    check(
        "gemma4_dense_cuda",
        &gemma4_checkpoint(),
        &facts,
        &target(0, 1),
        &Policy::default(),
    );
}

// ── csm: fp32 checkpoint, bf16 kernels ──────────────────────────────

fn csm_checkpoint() -> CheckpointMetadata {
    let hidden = 32;
    let mut ck = Checkpoint::new();
    ck.push("backbone.embed_tokens.weight", &[64, hidden], f32enc());
    ck.push("backbone.layers.0.mlp.w1.weight", &[48, hidden], f32enc());
    ck.push("backbone.norm.weight", &[hidden], f32enc());
    ck.push("depth_decoder.weight", &[hidden, hidden], f32enc());
    ck.finish("csm")
}

#[test]
fn csm_cuda() {
    check(
        "csm_cuda",
        &csm_checkpoint(),
        &facts("csm", 1),
        &target(0, 1),
        &Policy::default(),
    );
}

// ── glm5: FP8 kv_b_proj dequant + per-expert stacks ─────────────────

fn glm5_checkpoint() -> CheckpointMetadata {
    let (hidden, intermediate, experts) = (64, 32, 2);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[128, hidden], bf16());
    let p = "model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
    // The FP8 pair the family dequantizes in the loader: DeepSeek-style
    // square 32x32 blocks, stated by the shape ratio alone.
    ck.push(
        &format!("{p}self_attn.kv_b_proj.weight"),
        &[64, 32],
        Encoding::Raw(DType::F8E4M3),
    );
    ck.push(
        &format!("{p}self_attn.kv_b_proj.weight_scale_inv"),
        &[2, 1],
        f32enc(),
    );
    for expert in 0..experts {
        let e = format!("{p}mlp.experts.{expert}.");
        ck.push(
            &format!("{e}gate_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{e}up_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{e}down_proj.weight"),
            &[hidden, intermediate],
            bf16(),
        );
    }
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("glm5")
}

#[test]
fn glm5_cuda() {
    let mut facts = facts("glm_moe_dsa", 1);
    facts.num_experts = 2;
    check(
        "glm5_cuda",
        &glm5_checkpoint(),
        &facts,
        &target(0, 1),
        &Policy::default(),
    );
}

// ── gpt_oss: MXFP4 triplets, three lowerings ────────────────────────

fn gpt_oss_checkpoint() -> CheckpointMetadata {
    let (hidden, experts, intermediate) = (64, 2, 64);
    let mut ck = Checkpoint::new();
    let p = "model.layers.0.mlp.experts";
    ck.push(
        &format!("{p}.gate_up_proj_blocks"),
        &[experts, 2 * intermediate, hidden / 32, 16],
        u8enc(),
    );
    ck.push(
        &format!("{p}.gate_up_proj_scales"),
        &[experts, 2 * intermediate, hidden / 32],
        u8enc(),
    );
    ck.push(
        &format!("{p}.gate_up_proj_bias"),
        &[experts, 2 * intermediate],
        bf16(),
    );
    ck.push(
        &format!("{p}.down_proj_blocks"),
        &[experts, hidden, intermediate / 32, 16],
        u8enc(),
    );
    ck.push(
        &format!("{p}.down_proj_scales"),
        &[experts, hidden, intermediate / 32],
        u8enc(),
    );
    ck.push(&format!("{p}.down_proj_bias"), &[experts, hidden], bf16());
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("gpt_oss_fam")
}

#[test]
fn gpt_oss_routed_cuda() {
    let mut facts = facts("gpt_oss", 1);
    facts.num_experts = 2;
    check(
        "gpt_oss_routed_cuda",
        &gpt_oss_checkpoint(),
        &facts,
        &target(0, 1),
        &Policy::default(),
    );
}

#[test]
fn gpt_oss_native_cuda() {
    let mut facts = facts("gpt_oss", 1);
    facts.num_experts = 2;
    let mut target = target(0, 1);
    target.native_mxfp4_moe = true;
    let policy = Policy {
        moe_request: Mxfp4MoeRequest::NativeGemm,
        ..Policy::default()
    };
    check(
        "gpt_oss_native_cuda",
        &gpt_oss_checkpoint(),
        &facts,
        &target,
        &policy,
    );
}

#[test]
fn gpt_oss_streamed_cuda() {
    let mut facts = facts("gpt_oss", 1);
    facts.num_experts = 2;
    let policy = Policy {
        stream_routed_experts: true,
        ..Policy::default()
    };
    check(
        "gpt_oss_streamed_cuda",
        &gpt_oss_checkpoint(),
        &facts,
        &target(0, 1),
        &policy,
    );
}

// ── qwen3_5: GDN blocked shards + fp32 widening ─────────────────────

fn qwen3_5_checkpoint() -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim) = (64, 4, 2, 16);
    let (k_dim, v_dim) = (32, 32);
    let conv_dim = 2 * k_dim + v_dim;
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[128, hidden], bf16());
    // Layer 0: a Gated DeltaNet mixer.
    let la = "model.layers.0.linear_attn.";
    ck.push(
        &format!("{la}in_proj_qkv.weight"),
        &[conv_dim, hidden],
        bf16(),
    );
    ck.push(&format!("{la}in_proj_z.weight"), &[v_dim, hidden], bf16());
    ck.push(&format!("{la}in_proj_b.weight"), &[8, hidden], bf16());
    ck.push(&format!("{la}in_proj_a.weight"), &[8, hidden], bf16());
    ck.push(&format!("{la}conv1d.weight"), &[conv_dim, 1, 4], bf16());
    ck.push(&format!("{la}conv1d.bias"), &[conv_dim], bf16());
    ck.push(&format!("{la}out_proj.weight"), &[hidden, v_dim], bf16());
    // A_log ships bf16 here, so the widening pass has a cast to state.
    ck.push(&format!("{la}A_log"), &[8], bf16());
    ck.push(&format!("{la}dt_bias"), &[8], bf16());
    ck.push(&format!("{la}norm.weight"), &[v_dim], f32enc());
    // Layer 1: full attention, so the dense join has work.
    dense_layer(
        &mut ck,
        "model.layers.1.",
        hidden,
        heads,
        kv_heads,
        head_dim,
        96,
    );
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("qwen3_5")
}

#[test]
fn qwen3_5_dense_cuda() {
    let mut facts = facts("qwen3_5", 2);
    facts.head_dim = 16;
    check(
        "qwen3_5_dense_cuda",
        &qwen3_5_checkpoint(),
        &facts,
        &target(0, 1),
        &Policy::default(),
    );
}

/// Rank 1 of 2: the GDN `[K|K|V]` blocked shard is this family's whole
/// reason to have its own contract, and it only exists at tp > 1.
#[test]
fn qwen3_5_dense_cuda_tp1_of_2() {
    let mut facts = facts("qwen3_5", 2);
    facts.head_dim = 16;
    check(
        "qwen3_5_dense_cuda_tp1_of_2",
        &qwen3_5_checkpoint(),
        &facts,
        &target(1, 2),
        &Policy::default(),
    );
}

// ── qwen3_5_moe: shared-expert join + per-expert stacks ─────────────

fn qwen3_5_moe_checkpoint() -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim, intermediate, experts) = (64, 4, 2, 16, 32, 2);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[128, hidden], bf16());
    let p = "model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
    ck.push(
        &format!("{p}self_attn.q_proj.weight"),
        &[heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}self_attn.k_proj.weight"),
        &[kv_heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}self_attn.v_proj.weight"),
        &[kv_heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}self_attn.o_proj.weight"),
        &[hidden, heads * head_dim],
        bf16(),
    );
    ck.push(&format!("{p}mlp.gate.weight"), &[experts, hidden], bf16());
    for expert in 0..experts {
        let e = format!("{p}mlp.experts.{expert}.");
        ck.push(
            &format!("{e}gate_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{e}up_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{e}down_proj.weight"),
            &[hidden, intermediate],
            bf16(),
        );
    }
    ck.push(
        &format!("{p}mlp.shared_expert.gate_proj.weight"),
        &[intermediate, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}mlp.shared_expert.up_proj.weight"),
        &[intermediate, hidden],
        bf16(),
    );
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("qwen3_5_moe")
}

#[test]
fn qwen3_5_moe_cuda() {
    let mut facts = facts("qwen3_5_moe", 1);
    facts.head_dim = 16;
    facts.num_experts = 2;
    check(
        "qwen3_5_moe_cuda",
        &qwen3_5_moe_checkpoint(),
        &facts,
        &target(0, 1),
        &Policy::default(),
    );
}

// ── nemotron_h: Mamba unit folds + packed expert views ──────────────

fn nemotron_h_checkpoint() -> CheckpointMetadata {
    // heads=4 of head_dim=8 → intermediate 32; groups=2 of state=8 →
    // group_state 16; conv carries [x | B | C] = 64 rows; in_proj stacks
    // [z | x | B | C | dt] = 100 rows.
    let (hidden, heads, head_dim, group_state) = (64, 4, 8, 16);
    let intermediate = heads * head_dim;
    let conv_dim = intermediate + 2 * group_state;
    let in_rows = 2 * intermediate + 2 * group_state + heads;
    let mut ck = Checkpoint::new();
    ck.push(
        "language_model.backbone.embed_tokens.weight",
        &[128, hidden],
        bf16(),
    );
    let mp = "language_model.backbone.layers.0.mixer.";
    ck.push(&format!("{mp}in_proj.weight"), &[in_rows, hidden], bf16());
    ck.push(&format!("{mp}conv1d.weight"), &[conv_dim, 1, 4], bf16());
    ck.push(&format!("{mp}conv1d.bias"), &[conv_dim], bf16());
    ck.push(&format!("{mp}A_log"), &[heads], f32enc());
    ck.push(&format!("{mp}D"), &[heads], f32enc());
    ck.push(&format!("{mp}dt_bias"), &[heads], f32enc());
    ck.push(&format!("{mp}norm.weight"), &[intermediate], bf16());
    ck.push(
        &format!("{mp}out_proj.weight"),
        &[hidden, intermediate],
        bf16(),
    );
    // Layer 1: the MoE mixer, one packed slab's worth of experts.
    let ep = "language_model.backbone.layers.1.mixer.experts";
    for expert in 0..2 {
        ck.push(
            &format!("{ep}.{expert}.up_proj.weight"),
            &[32, hidden],
            bf16(),
        );
        ck.push(
            &format!("{ep}.{expert}.down_proj.weight"),
            &[hidden, 32],
            bf16(),
        );
    }
    ck.push("language_model.backbone.norm_f.weight", &[hidden], bf16());
    ck.finish("nemotron_h")
}

#[test]
fn nemotron_h_cuda_tp1_of_2() {
    let mut facts = facts("nemotron_h", 2);
    facts.num_experts = 2;
    facts.mamba_groups = 2;
    check(
        "nemotron_h_cuda_tp1_of_2",
        &nemotron_h_checkpoint(),
        &facts,
        &target(1, 2),
        &Policy::default(),
    );
}

// ── kimi_k2: MLA joins + W4A16 expert stacks ────────────────────────

fn kimi_checkpoint() -> CheckpointMetadata {
    let (hidden, intermediate) = (64, 32);
    let mut ck = Checkpoint::new();
    ck.push(
        "language_model.model.embed_tokens.weight",
        &[128, hidden],
        bf16(),
    );
    let p = "language_model.model.layers.0.";
    ck.push(
        &format!("{p}self_attn.q_a_proj.weight"),
        &[32, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}self_attn.kv_a_proj_with_mqa.weight"),
        &[48, hidden],
        bf16(),
    );
    for expert in 0..2 {
        let e = format!("{p}mlp.experts.{expert}.");
        for half in ["gate_proj", "up_proj"] {
            ck.push(
                &format!("{e}{half}.weight_packed"),
                &[intermediate, hidden / 8],
                Encoding::Raw(DType::I32),
            );
            ck.push(
                &format!("{e}{half}.weight_scale"),
                &[intermediate, hidden / 32],
                bf16(),
            );
        }
        ck.push(
            &format!("{e}down_proj.weight_packed"),
            &[hidden, intermediate / 8],
            Encoding::Raw(DType::I32),
        );
        ck.push(
            &format!("{e}down_proj.weight_scale"),
            &[hidden, intermediate / 32],
            bf16(),
        );
    }
    ck.push("language_model.model.norm.weight", &[hidden], bf16());
    ck.push("lm_head.weight", &[128, hidden], bf16());
    ck.finish("kimi")
}

#[test]
fn kimi_k2_cuda() {
    check(
        "kimi_k2_cuda",
        &kimi_checkpoint(),
        &facts("kimi_k2", 1),
        &target(0, 1),
        &Policy::default(),
    );
}

// ── kimi_k3: A_log bands + MXFP4 stacks with GEMV republish ─────────

fn kimi_k3_checkpoint() -> CheckpointMetadata {
    let (hidden, latent, intermediate) = (64, 64, 32);
    let mut ck = Checkpoint::new();
    ck.push(
        "language_model.model.embed_tokens.weight",
        &[128, hidden],
        bf16(),
    );
    let p = "language_model.model.layers.0.";
    // A KDA layer: 8 real heads in a 16-entry padded gate bank.
    ck.push(&format!("{p}self_attn.A_log"), &[16], f32enc());
    ck.push(&format!("{p}self_attn.b_proj.weight"), &[8, hidden], bf16());
    let moe = format!("{p}block_sparse_moe.");
    for expert in 0..2 {
        let e = format!("{moe}experts.{expert}.");
        for half in ["w1", "w3"] {
            ck.push(
                &format!("{e}{half}.weight_packed"),
                &[intermediate, latent / 2],
                u8enc(),
            );
            ck.push(
                &format!("{e}{half}.weight_scale"),
                &[intermediate, latent / 32],
                u8enc(),
            );
        }
        ck.push(
            &format!("{e}w2.weight_packed"),
            &[latent, intermediate / 2],
            u8enc(),
        );
        ck.push(
            &format!("{e}w2.weight_scale"),
            &[latent, intermediate / 32],
            u8enc(),
        );
    }
    ck.push("language_model.model.norm.weight", &[hidden], bf16());
    ck.finish("kimi_k3")
}

#[test]
fn kimi_k3_cuda() {
    let mut facts = facts("kimi_k3", 1);
    facts.num_experts = 2;
    check(
        "kimi_k3_cuda",
        &kimi_k3_checkpoint(),
        &facts,
        &target(0, 1),
        &Policy::default(),
    );
}

// ── deepseek_v4: E8M0 block scales, expert stacks and groups ────────

fn deepseek_v4_checkpoint() -> CheckpointMetadata {
    let (hidden, intermediate) = (64, 32);
    let mut ck = Checkpoint::new();
    ck.push("embed_tokens.weight", &[128, hidden], bf16());
    // The bare `layers.` spelling this family ships.
    let p = "layers.0.";
    // A dense FP8 projection with its square block scale, outside the FFN.
    ck.push(
        &format!("{p}attn.wq.weight"),
        &[64, 64],
        Encoding::Raw(DType::F8E4M3),
    );
    ck.push(&format!("{p}attn.wq.scale"), &[2, 2], u8enc());
    for expert in 0..2 {
        let e = format!("{p}ffn.experts.{expert}.");
        for half in ["w1", "w3"] {
            ck.push(
                &format!("{e}{half}.weight"),
                &[intermediate, hidden / 2],
                Encoding::Raw(DType::I8),
            );
            ck.push(
                &format!("{e}{half}.scale"),
                &[intermediate, hidden / 32],
                u8enc(),
            );
        }
        ck.push(
            &format!("{e}w2.weight"),
            &[hidden, intermediate / 2],
            Encoding::Raw(DType::I8),
        );
        ck.push(
            &format!("{e}w2.scale"),
            &[hidden, intermediate / 32],
            u8enc(),
        );
    }
    ck.push("norm.weight", &[hidden], bf16());
    ck.finish("deepseek_v4")
}

#[test]
fn deepseek_v4_eager_cuda() {
    check(
        "deepseek_v4_eager_cuda",
        &deepseek_v4_checkpoint(),
        &facts("deepseek_v4", 1),
        &target(0, 1),
        &Policy::default(),
    );
}

#[test]
fn deepseek_v4_streamed_cuda() {
    let policy = Policy {
        stream_routed_experts: true,
        ..Policy::default()
    };
    check(
        "deepseek_v4_streamed_cuda",
        &deepseek_v4_checkpoint(),
        &facts("deepseek_v4", 1),
        &target(0, 1),
        &policy,
    );
}

// ═══ The Metal point in policy space: Naming::Mlx, bind in place ═══

fn metal_target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Metal,
        tp_rank: 0,
        tp_size: 1,
        preferred_alignment: 256,
        max_tile_bytes: 64 << 20,
        tile_map_mask: METAL_TILE_MAP_MASK,
        ..StorageTarget::default()
    }
}

fn mlx_policy() -> Policy {
    Policy {
        naming: Naming::Mlx,
        projections: Projections::InPlace,
        ..Policy::default()
    }
}

fn f16enc() -> Encoding {
    Encoding::Raw(DType::F16)
}

/// One MLX affine-U4 g64 triplet: `[rows, cols/8]` U32 beside `[rows,
/// cols/64]` F16 scales and biases.
fn mlx_triplet(ck: &mut Checkpoint, base: &str, rows: i64, cols: i64) {
    ck.push(
        &format!("{base}.weight"),
        &[rows, cols / 8],
        Encoding::Raw(DType::U32),
    );
    ck.push(&format!("{base}.scales"), &[rows, cols / 64], f16enc());
    ck.push(&format!("{base}.biases"), &[rows, cols / 64], f16enc());
}

fn llama_mlx_checkpoint() -> CheckpointMetadata {
    let (hidden, intermediate, vocab) = (64, 128, 128);
    let mut ck = Checkpoint::new();
    // Tied: an embed_tokens and no lm_head, quantized like everything else.
    mlx_triplet(&mut ck, "model.embed_tokens", vocab, hidden);
    let p = "model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        mlx_triplet(&mut ck, &format!("{p}self_attn.{proj}"), hidden, hidden);
    }
    ck.push(
        &format!("{p}post_attention_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    mlx_triplet(&mut ck, &format!("{p}mlp.gate_proj"), intermediate, hidden);
    mlx_triplet(&mut ck, &format!("{p}mlp.up_proj"), intermediate, hidden);
    mlx_triplet(&mut ck, &format!("{p}mlp.down_proj"), hidden, intermediate);
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("llama_mlx")
}

#[test]
fn llama_mlx_metal() {
    let mut facts = facts("llama3", 1);
    facts.quant_bits = 4;
    facts.quant_group_size = 64;
    check(
        "llama_mlx_metal",
        &llama_mlx_checkpoint(),
        &facts,
        &metal_target(),
        &mlx_policy(),
    );
}

/// The same decoder as [`llama_mlx_checkpoint`], unquantized — what a stock
/// BF16 release ships, and what Metal cannot bind as it stands: its matvecs
/// read MLX affine, and even the embedding gather wants scales.
fn llama_bf16_checkpoint() -> CheckpointMetadata {
    let (hidden, intermediate, vocab) = (64, 128, 128);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[vocab, hidden], bf16());
    let p = "model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        ck.push(&format!("{p}self_attn.{proj}.weight"), &[hidden, hidden], bf16());
    }
    ck.push(
        &format!("{p}post_attention_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    ck.push(&format!("{p}mlp.gate_proj.weight"), &[intermediate, hidden], bf16());
    ck.push(&format!("{p}mlp.up_proj.weight"), &[intermediate, hidden], bf16());
    ck.push(&format!("{p}mlp.down_proj.weight"), &[hidden, intermediate], bf16());
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("llama_bf16")
}

/// Offline quantization: the request that turns the BF16 release above into
/// something these kernels read.
///
/// Every rank-2 `.weight` becomes an affine-U4 `Encode` and the rank-1 norms
/// stay values — the same arm gpt-oss applies unconditionally to the BF16 half
/// of its published checkpoint, and the same `Encode` a serve boot would run.
/// `pie model build --backend metal --quant int4` is this contract with the
/// transform executed on the host instead of at load.
#[test]
fn llama_mlx_metal_int4() {
    check(
        "llama_mlx_metal_int4",
        &llama_bf16_checkpoint(),
        &facts("llama3", 1),
        &metal_target(),
        &Policy {
            runtime_quant: RuntimeQuant::Int4,
            ..mlx_policy()
        },
    );
}

/// A requantization this lowering has no encoder for is refused, not ignored.
///
/// The three CUDA schemes are the ones that used to be plumbed through to an
/// author that never read the field: silently authoring an unquantized
/// contract for `--quant int8` would hand back an artifact whose name says one
/// thing and whose bytes say another.
#[test]
fn metal_refuses_a_requantization_it_cannot_encode() {
    for quant in [RuntimeQuant::Int8, RuntimeQuant::Mxfp4, RuntimeQuant::Fp8] {
        // gpt-oss is in the list because its lowering encodes whatever the
        // request says -- so the refusal is the only thing the request can do
        // there, and an author that ignored it would be the odd one out.
        for (family, checkpoint) in [
            ("llama3", llama_bf16_checkpoint()),
            ("gpt_oss", gptoss_mlx_checkpoint()),
        ] {
            let err = author(
                &facts(family, 1),
                &checkpoint,
                &metal_target(),
                &Policy {
                    runtime_quant: quant,
                    ..mlx_policy()
                },
            )
            .expect_err(&format!("{family} runtime_quant={quant:?} should be refused"));
            let text = err.to_string();
            assert!(text.contains("no encoder here"), "{family} {quant:?}: {text}");
            assert!(
                text.contains("int4"),
                "{family} {quant:?} refusal names the alternative: {text}"
            );
        }
    }
}

/// A BF16 release with NO quantization requested is refused while the
/// checkpoint is still in view.
///
/// `llama_mlx_metal_int4` above is the same checkpoint with `--quant int4`, and
/// it authors fine — the encode supplies what the driver needs. Without that
/// request nothing does: `push_quant` asks for `.weight`/`.scales`/`.biases`
/// unconditionally, so the artifact used to look well-formed and then fail deep
/// in the loader with `llama bind: unstaged weight embed_tokens.scales`, which
/// names an internal staging table rather than the problem. The refusal has to
/// name the two ways out, since both are real.
#[test]
fn an_unquantized_checkpoint_is_refused_unless_it_is_being_quantized() {
    let mut facts = facts("llama3", 1);
    facts.quant_bits = 4;
    facts.quant_group_size = 64;

    let err = author(
        &facts,
        &llama_bf16_checkpoint(),
        &metal_target(),
        &mlx_policy(),
    )
    .expect_err("bf16 with no --quant has no Metal binding and must be refused");
    let msg = err.to_string();
    assert!(msg.contains("scales"), "names what is missing: {msg}");
    assert!(msg.contains("int4"), "names the offline encode: {msg}");
    assert!(msg.contains("4bit"), "names the pre-quantized repos: {msg}");
}

/// The same decoder again, in F16 — the width many older releases ship.
///
/// Not a cosmetic variant: the driver's `encode_mlx_affine_u4` is handed a byte
/// width, not a dtype, so it reads every 2-byte element as BF16. Encoding an
/// F16 weight where it lies would quantize correct values offline and misread
/// bits at load, which is the one thing this feature must not do. The contract
/// casts first, and the golden is where that stays true.
fn llama_f16_checkpoint() -> CheckpointMetadata {
    let (hidden, intermediate, vocab) = (64, 128, 128);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[vocab, hidden], f16enc());
    let p = "model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], f16enc());
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        ck.push(
            &format!("{p}self_attn.{proj}.weight"),
            &[hidden, hidden],
            f16enc(),
        );
    }
    ck.push(
        &format!("{p}mlp.down_proj.weight"),
        &[hidden, intermediate],
        f16enc(),
    );
    ck.push("model.norm.weight", &[hidden], f16enc());
    ck.finish("llama_f16")
}

#[test]
fn llama_mlx_metal_int4_from_f16() {
    check(
        "llama_mlx_metal_int4_from_f16",
        &llama_f16_checkpoint(),
        &facts("llama3", 1),
        &metal_target(),
        &Policy {
            runtime_quant: RuntimeQuant::Int4,
            ..mlx_policy()
        },
    );
}

/// An already-quantized checkpoint is not quantized again.
///
/// The U32-triplet arm is checked before the encode arm, and the order is the
/// whole guarantee: reversed, `--quant int4` over an MLX checkpoint would try
/// to encode packed codes as if they were values. Nothing in a type would
/// catch that, so the property is asserted — no widening tensor appears, and
/// every projection is still the affine the checkpoint shipped.
///
/// This is what makes `optimize --quant int4` idempotent: run twice, the second
/// artifact is the same size as the first and gates identically.
#[test]
fn an_mlx_checkpoint_is_not_requantized_by_int4() {
    let mut facts = facts("llama3", 1);
    facts.quant_bits = 4;
    facts.quant_group_size = 64;
    let contract = author(
        &facts,
        &llama_mlx_checkpoint(),
        &metal_target(),
        &Policy {
            runtime_quant: RuntimeQuant::Int4,
            ..mlx_policy()
        },
    )
    .expect("authoring an MLX checkpoint with int4 succeeds")
    .expect("llama has an author");

    assert!(
        !contract.tensors.iter().any(|t| t.name.ends_with(".bf16")),
        "an already-quantized checkpoint grew a widening tensor: {:?}",
        contract
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
    );
    let projections = contract.tensors.iter().filter(|t| {
        t.name.ends_with(".weight") && (t.name.contains("self_attn.") || t.name.contains("mlp."))
    });
    let mut seen = 0;
    for tensor in projections {
        seen += 1;
        assert!(
            matches!(&tensor.encoding, Encoding::Quant(spec) if spec.bits_per_element == 4),
            "{} lost the affine the checkpoint shipped: {:?}",
            tensor.name,
            tensor.encoding
        );
    }
    assert!(seen >= 7, "fixture should carry every projection, saw {seen}");
}

fn qwen3_5_mlx_checkpoint() -> CheckpointMetadata {
    let hidden = 64;
    let mut ck = Checkpoint::new();
    // The mlx_lm spelling: `language_model.model.*`, words swapped.
    mlx_triplet(&mut ck, "language_model.model.embed_tokens", 128, hidden);
    let p = "language_model.model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        mlx_triplet(&mut ck, &format!("{p}self_attn.{proj}"), hidden, hidden);
    }
    ck.push(&format!("{p}linear_attn.A_log"), &[8], f32enc());
    ck.push("language_model.model.norm.weight", &[hidden], bf16());
    ck.finish("qwen3_5_mlx")
}

#[test]
fn qwen3_5_mlx_metal() {
    let mut facts = facts("qwen3_5", 1);
    facts.quant_bits = 4;
    facts.quant_group_size = 64;
    check(
        "qwen3_5_mlx_metal",
        &qwen3_5_mlx_checkpoint(),
        &facts,
        &metal_target(),
        &mlx_policy(),
    );
}

fn gemma4_mlx_checkpoint() -> CheckpointMetadata {
    let hidden = 64;
    let mut ck = Checkpoint::new();
    mlx_triplet(&mut ck, "model.language_model.embed_tokens", 128, hidden);
    for layer in 0..2 {
        let p = format!("model.language_model.layers.{layer}.");
        ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            mlx_triplet(&mut ck, &format!("{p}self_attn.{proj}"), hidden, hidden);
        }
        ck.push(&format!("{p}self_attn.k_norm.weight"), &[16], bf16());
    }
    ck.push("model.language_model.norm.weight", &[hidden], bf16());
    ck.finish("gemma4_mlx")
}

/// Layer 1 is KV-shared: its k/v projections and k-norm ship in the file
/// and must not be declared — the golden is what pins that they are not.
#[test]
fn gemma4_mlx_metal() {
    let mut facts = facts("gemma4_text", 2);
    facts.quant_bits = 4;
    facts.quant_group_size = 64;
    facts.num_kv_shared_layers = 1;
    check(
        "gemma4_mlx_metal",
        &gemma4_mlx_checkpoint(),
        &facts,
        &metal_target(),
        &mlx_policy(),
    );
}

fn gptoss_mlx_checkpoint() -> CheckpointMetadata {
    // The PUBLISHED layout: BF16 attention the loader quantizes on the way
    // in, plus MXFP4 `_blocks`/`_scales`/`_bias` expert triplets whose
    // gate/up halves are interleaved row by row.
    let (hidden, experts, intermediate) = (64, 2, 64);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[128, hidden], bf16());
    let p = "model.layers.0.";
    ck.push(&format!("{p}input_layernorm.weight"), &[hidden], bf16());
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        ck.push(
            &format!("{p}self_attn.{proj}.weight"),
            &[hidden, hidden],
            bf16(),
        );
    }
    ck.push(&format!("{p}self_attn.sinks"), &[4], bf16());
    ck.push(&format!("{p}mlp.router.weight"), &[experts, hidden], bf16());
    ck.push(&format!("{p}mlp.router.bias"), &[experts], bf16());
    let e = format!("{p}mlp.experts.");
    ck.push(
        &format!("{e}gate_up_proj_blocks"),
        &[experts, 2 * intermediate, hidden / 32, 16],
        u8enc(),
    );
    ck.push(
        &format!("{e}gate_up_proj_scales"),
        &[experts, 2 * intermediate, hidden / 32],
        u8enc(),
    );
    ck.push(
        &format!("{e}gate_up_proj_bias"),
        &[experts, 2 * intermediate],
        bf16(),
    );
    ck.push(
        &format!("{e}down_proj_blocks"),
        &[experts, hidden, intermediate / 32, 16],
        u8enc(),
    );
    ck.push(
        &format!("{e}down_proj_scales"),
        &[experts, hidden, intermediate / 32],
        u8enc(),
    );
    ck.push(&format!("{e}down_proj_bias"), &[experts, hidden], bf16());
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.push("lm_head.weight", &[128, hidden], bf16());
    ck.finish("gptoss_mlx")
}

#[test]
fn gptoss_mlx_metal() {
    let mut facts = facts("gpt_oss", 1);
    facts.num_experts = 2;
    check(
        "gptoss_mlx_metal",
        &gptoss_mlx_checkpoint(),
        &facts,
        &metal_target(),
        &mlx_policy(),
    );
}
