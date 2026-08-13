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
    CUDA_TILE_MAP_MASK, HOST_TILE_MAP_MASK, METAL_TILE_MAP_MASK, StorageInstr, StorageTarget,
    TileMapKind,
    compile as compile_load_plan,
};
use pie_loader::types::{
    BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantGranularity, QuantScheme,
    RepackLayout, ScaleForm, TensorId, Visibility,
};
use pie_loader::verify::ContractView;

use pie_model::contract::{author, author_with_policy};
use pie_model_common::facts::ModelFacts;
use pie_model_common::policy::{
    Mxfp4MoePolicy, Mxfp4MoeRequest, Naming, Policy, Projections, RuntimeQuant,
};

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
            let file = std::fs::File::create(&staging).expect("create fixture checkpoint");
            file.set_len(self.offset).expect("size fixture checkpoint");
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
    qwen3_5_checkpoint_with(bf16(), "qwen3_5")
}

/// `file_tag` keys the fixture's temp file: two variants of this checkpoint
/// have different sizes, so sharing one name would race the other tests.
fn qwen3_5_checkpoint_with(qkv_encoding: Encoding, file_tag: &str) -> CheckpointMetadata {
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
        qkv_encoding,
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
    ck.finish(file_tag)
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

/// An FP8 GDN weight *without* its `weight_scale_inv` companion cannot be
/// dequantized, and the CUDA forward has no quant-scale port for it — so the
/// author must refuse the checkpoint rather than let it load and serve noise.
#[test]
fn qwen3_5_refuses_fp8_gdn_without_scales() {
    let mut facts = facts("qwen3_5", 2);
    facts.head_dim = 16;
    let checkpoint = qwen3_5_checkpoint_with(Encoding::Raw(DType::F8E4M3), "qwen3_5_fp8");
    for tp_size in [1, 2] {
        let err = author(&facts, &checkpoint, &target(0, tp_size), &Policy::default())
            .expect_err(&format!("tp={tp_size}: an unpaired FP8 GDN weight must be refused"));
        let msg = err.to_string();
        assert!(
            msg.contains("weight_scale_inv") && msg.contains("cannot be dequantized"),
            "tp={tp_size}: unexpected error: {msg}"
        );
    }
}

// ── qwen3_5: FP8 GDN load-time dequantization ───────────────────────

/// The E4M3 byte for a small positive value that E4M3 represents exactly.
fn fp8_e4m3(value: f64) -> u8 {
    let exponent = value.log2().floor() as i32;
    let mantissa = (value / f64::from(exponent).exp2() - 1.0) * 8.0;
    (((exponent + 7) as u8) << 3) | (mantissa.round() as u8)
}

/// f32 → bf16 bits, round to nearest even. The fixture values are chosen so
/// the products are exact in bf16, so the rounding never actually fires — it
/// is here so a wrong fixture value fails loudly instead of subtly.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// The FP8 payload element at flat index `i`: 1..=7, exactly representable in
/// E4M3. Period 7 stays out of phase with the 4-wide scale blocks, so a
/// misindexed block cannot reproduce the right sequence.
fn gdn_value(i: usize) -> f32 {
    (i % 7 + 1) as f32
}

/// The block scale at flat index `j`: powers of two, so `value * scale` is
/// exact in bf16 and any factor applied to the wrong block is off by at
/// least a factor of two.
fn gdn_scale(j: usize) -> f32 {
    [0.5, 2.0, 4.0, 0.25, 8.0][j % 5]
}

/// An FP8-GDN checkpoint with *real bytes*: every `linear_attn` projection
/// ships E4M3 beside a bf16 `weight_scale_inv` in 4x4 blocks, and the file
/// holds the values `gdn_value`/`gdn_scale` state so the host executor can be
/// checked against arithmetic done here. K = V = 8 with 4-row blocks keeps
/// every tp=2 band on a block boundary; everything else is zero-filled bf16.
fn qwen3_5_fp8_gdn_checkpoint() -> CheckpointMetadata {
    let hidden = 8i64;
    let (k_dim, v_dim) = (8i64, 8i64);
    let conv_dim = 2 * k_dim + v_dim;
    let block = 4i64;
    let fp8 = Encoding::Raw(DType::F8E4M3);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[16, hidden], bf16());
    let la = "model.layers.0.linear_attn.";
    ck.push(
        &format!("{la}in_proj_qkv.weight"),
        &[conv_dim, hidden],
        fp8.clone(),
    );
    ck.push(
        &format!("{la}in_proj_qkv.weight_scale_inv"),
        &[conv_dim / block, hidden / block],
        bf16(),
    );
    ck.push(&format!("{la}in_proj_z.weight"), &[v_dim, hidden], fp8.clone());
    ck.push(
        &format!("{la}in_proj_z.weight_scale_inv"),
        &[v_dim / block, hidden / block],
        bf16(),
    );
    ck.push(&format!("{la}in_proj_b.weight"), &[2, hidden], bf16());
    ck.push(&format!("{la}in_proj_a.weight"), &[2, hidden], bf16());
    ck.push(&format!("{la}conv1d.weight"), &[conv_dim, 1, 4], bf16());
    ck.push(&format!("{la}conv1d.bias"), &[conv_dim], bf16());
    ck.push(&format!("{la}out_proj.weight"), &[hidden, v_dim], fp8);
    ck.push(
        &format!("{la}out_proj.weight_scale_inv"),
        &[hidden / block, v_dim / block],
        bf16(),
    );
    ck.push(&format!("{la}A_log"), &[2], bf16());
    ck.push(&format!("{la}dt_bias"), &[2], bf16());
    ck.push(&format!("{la}norm.weight"), &[v_dim], f32enc());
    dense_layer(&mut ck, "model.layers.1.", hidden, 2, 2, 4, 16);
    ck.push("model.norm.weight", &[hidden], bf16());
    let metadata = ck.finish("qwen3_5_fp8_gdn");

    let mut data = vec![0u8; usize::try_from(metadata.files[0].size_bytes).unwrap()];
    for raw in &metadata.tensors {
        let start = usize::try_from(raw.file_offset).unwrap();
        if raw.encoding == Encoding::Raw(DType::F8E4M3) {
            for i in 0..usize::try_from(raw.span_bytes).unwrap() {
                data[start + i] = fp8_e4m3(f64::from(gdn_value(i)));
            }
        } else if raw.name.ends_with(".weight_scale_inv") {
            for j in 0..usize::try_from(raw.span_bytes / 2).unwrap() {
                let bits = bf16_bits(gdn_scale(j));
                data[start + 2 * j..start + 2 * j + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
    // Staged and renamed, not written in place: two tests build this fixture
    // concurrently, and a truncate-then-write under a concurrent reader is a
    // short read.
    let path = std::path::Path::new(&metadata.files[0].path);
    let staging = path.with_extension(format!("{:?}.partial", std::thread::current().id()));
    std::fs::write(&staging, &data).expect("stage fp8 GDN fixture bytes");
    std::fs::rename(&staging, path).expect("publish fp8 GDN fixture bytes");
    metadata
}

/// The bf16 bytes the dequant must produce for the local tensor whose
/// element (r, c) reads global element (rows[r], cols[c]) of a full
/// `[_, full_cols]` FP8 weight with `[_, full_cols / 4]`-shaped scales.
fn gdn_expected(rows: &[i64], cols: &[i64], full_cols: i64) -> Vec<u8> {
    let block = 4i64;
    let scale_cols = full_cols / block;
    let mut out = Vec::with_capacity(rows.len() * cols.len() * 2);
    for &gr in rows {
        for &gc in cols {
            let value = gdn_value(usize::try_from(gr * full_cols + gc).unwrap());
            let scale =
                gdn_scale(usize::try_from((gr / block) * scale_cols + gc / block).unwrap());
            out.extend_from_slice(&bf16_bits(value * scale).to_le_bytes());
        }
    }
    out
}

/// The load plan dequantizes FP8 GDN weights to the mathematically expected
/// bf16 values — `weight = fp8 * block_scale` — at tp=1 and on both ranks of
/// tp=2, executed end to end by the host executor over real checkpoint
/// bytes. The tp=2 ranks also prove the `[K|K|V]` bands and the row/column
/// shards slice the *scales* consistently with the weights.
#[test]
fn qwen3_5_fp8_gdn_dequantizes_to_expected_bf16() {
    let mut facts = facts("qwen3_5", 2);
    facts.head_dim = 4;
    let checkpoint = qwen3_5_fp8_gdn_checkpoint();
    let la = "model.layers.0.linear_attn.";
    let all8: Vec<i64> = (0..8).collect();

    for (tp_rank, tp_size) in [(0, 1), (0, 2), (1, 2)] {
        let target = target(tp_rank, tp_size);
        let contract = author(&facts, &checkpoint, &target, &Policy::default())
            .expect("authoring failed")
            .expect("no author for qwen3_5");
        for leaf in ["in_proj_qkv", "in_proj_z", "out_proj"] {
            let entry = contract
                .tensors
                .iter()
                .find(|t| t.name == format!("{la}{leaf}.weight"))
                .unwrap_or_else(|| panic!("{leaf}: no published weight"));
            assert_eq!(
                entry.encoding,
                Encoding::Raw(DType::BF16),
                "{leaf}: the GDN path binds bf16"
            );
            let scales = contract
                .tensors
                .iter()
                .find(|t| t.name == format!("{la}{leaf}.weight_scale_inv"))
                .unwrap_or_else(|| panic!("{leaf}: no declared scale factors"));
            assert_eq!(
                scales.visibility,
                Visibility::Internal,
                "{leaf}: factors are load-time only, not a bind name"
            );
        }
        let plan = compile_load_plan(&checkpoint, &contract, target.clone())
            .expect("compiling failed");
        if let Err(violations) = pie_loader_capi::view::verify_marshalled(
            &plan,
            Some(&ContractView::of(&contract)),
        ) {
            let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
            panic!("tp{tp_size} rank {tp_rank}: {}", listed.join("\n  "));
        }
        let scaled: Vec<_> = plan
            .instrs
            .iter()
            .filter_map(|instr| match instr {
                StorageInstr::TileMap {
                    kind: TileMapKind::Scale,
                    transform,
                    ..
                } => Some(transform),
                _ => None,
            })
            .collect();
        assert_eq!(
            scaled.len(),
            3,
            "tp{tp_size} rank {tp_rank}: one per-block Scale per FP8 projection"
        );
        for transform in scaled {
            assert_eq!(transform.scale_blocks, vec![4, 4]);
            assert_eq!(transform.from, Some(QuantScheme::Fp8E4M3));
        }

        // The CUDA plan above pins what the driver will run; the host replay
        // executes the same contract compiled against the host's own mask.
        let mut host_target = target.clone();
        host_target.tile_map_mask = HOST_TILE_MAP_MASK;
        let host_plan = compile_load_plan(&checkpoint, &contract, host_target)
            .expect("compiling host plan failed");
        let storage = pie_loader::executor::host::execute_plan(
            &host_plan,
            std::path::Path::new(""),
        )
        .expect("host execution failed");
        // This rank's global rows/cols per projection: at tp=2 the qkv rows
        // are the rank's half of each of the [K|K|V] bands, z splits rows,
        // out_proj splits columns.
        let half = |base: i64| -> Vec<i64> {
            (base + i64::from(tp_rank) * 4..base + i64::from(tp_rank) * 4 + 4).collect()
        };
        let (qkv_rows, z_rows, out_cols) = if tp_size == 1 {
            ((0..24).collect::<Vec<i64>>(), all8.clone(), all8.clone())
        } else {
            (
                [half(0), half(8), half(16)].concat(),
                half(0),
                half(0),
            )
        };
        for (leaf, rows, cols, full_cols) in [
            ("in_proj_qkv", &qkv_rows, &all8, 8),
            ("in_proj_z", &z_rows, &all8, 8),
            ("out_proj", &all8, &out_cols, 8),
        ] {
            let name = format!("{la}{leaf}.weight");
            let bytes = storage
                .tensors
                .get(&name)
                .unwrap_or_else(|| panic!("{name}: not materialized"));
            assert_eq!(
                bytes,
                &gdn_expected(rows, cols, full_cols),
                "tp{tp_size} rank {tp_rank}: {name} dequantized wrong"
            );
        }
    }
}

/// A tp that splits inside a scale block has no representable scale slice —
/// K = V = 8 in 4-row blocks cannot split 4 ways — and must be refused
/// rather than approximated.
#[test]
fn qwen3_5_fp8_gdn_refuses_misaligned_tp() {
    let mut facts = facts("qwen3_5", 2);
    facts.head_dim = 4;
    let checkpoint = qwen3_5_fp8_gdn_checkpoint();
    let err = author(&facts, &checkpoint, &target(0, 4), &Policy::default())
        .expect_err("tp=4 splits inside a 4-row scale block and must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("scale-block boundaries"),
        "unexpected error: {msg}"
    );
}

/// The GDN fp32 widening casts a rank slice of `A_log` at tp > 1, and the
/// executor sizes a Cast's file source from the extent's dim counts times the
/// dtype width. A byte-run stride handed to a dtype-typed source view makes
/// that product double `span_bytes` and the driver throws
/// "Cast source byte size mismatch" — so measure it the executor's way here.
#[test]
fn qwen3_5_tp2_cast_source_extents_match_their_span() {
    let mut facts = facts("qwen3_5", 2);
    facts.head_dim = 16;
    let checkpoint = qwen3_5_checkpoint();
    for rank in 0..2 {
        let target = target(rank, 2);
        let contract = author(&facts, &checkpoint, &target, &Policy::default())
            .expect("authoring failed")
            .expect("no author for qwen3_5");
        let plan =
            compile_load_plan(&checkpoint, &contract, target).expect("compiling failed");
        let mut casts = 0;
        for instr in &plan.instrs {
            let StorageInstr::TileMap {
                kind: TileMapKind::Cast,
                source: Some(source),
                ..
            } = instr
            else {
                continue;
            };
            casts += 1;
            let elements: i64 = source.stride.dims.iter().map(|dim| dim.count).product();
            let extent_bytes = u64::try_from(elements).unwrap() * source.dtype.bytes();
            assert_eq!(
                extent_bytes, source.span_bytes,
                "rank {rank}: Cast source extent is {extent_bytes} bytes but \
                 span_bytes is {}",
                source.span_bytes
            );
        }
        assert!(
            casts > 0,
            "rank {rank}: fixture no longer produces a Cast with a file source"
        );
    }
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

/// `experts` of `None` is a dense-only K3, with no `block_sparse_moe` names at
/// all; `Some(enc)` gives it two routed experts whose packed tensors carry
/// `enc` — `u8enc()` for the MXFP4 layout the family really ships, anything
/// else for a checkpoint that spells the names and means something different.
///
/// `tag` names the fixture and is required, not derived: K3 has several, they
/// must not share a temp file, and a new one that inherited another's path
/// would be verified against the wrong bytes and report the mismatch as a
/// defect in whichever case lost the race. A required argument cannot be
/// forgotten.
///
/// `intermediate` is a parameter because the streamed group shards *inside*
/// the expert: at 32 the tp=2 slice is 16, and `down.weight_scale`'s
/// `[latent, local_inter / 32]` collapses to a zero-width axis. Any tp>1 case
/// needs at least 64.
fn kimi_k3_checkpoint_at(
    intermediate: i64,
    experts: Option<Encoding>,
    tag: &str,
) -> CheckpointMetadata {
    kimi_k3_checkpoint_sized(64, 64, intermediate, 8, experts, tag)
}

/// [`kimi_k3_checkpoint_at`] with the two widths the expert shapes are built
/// from left open, so one case can use the model's real ones.
fn kimi_k3_checkpoint_sized(
    hidden: i64,
    latent: i64,
    intermediate: i64,
    heads: i64,
    experts: Option<Encoding>,
    tag: &str,
) -> CheckpointMetadata {
    let mut ck = Checkpoint::new();
    ck.push(
        "language_model.model.embed_tokens.weight",
        &[128, hidden],
        bf16(),
    );
    let p = "language_model.model.layers.0.";
    // A KDA layer: `heads` real heads in a gate bank the checkpoint rounded
    // up. `heads` is a parameter because every rank has to divide it, so it
    // bounds the tp degrees a fixture can reach — the padding ratio is not
    // load-bearing, only that the band has to find its heads inside it.
    ck.push(&format!("{p}self_attn.A_log"), &[heads * 2], f32enc());
    ck.push(
        &format!("{p}self_attn.b_proj.weight"),
        &[heads, hidden],
        bf16(),
    );
    if let Some(packed) = experts {
        let moe = format!("{p}block_sparse_moe.");
        for expert in 0..2 {
            let e = format!("{moe}experts.{expert}.");
            for half in ["w1", "w3"] {
                ck.push(
                    &format!("{e}{half}.weight_packed"),
                    &[intermediate, latent / 2],
                    packed.clone(),
                );
                ck.push(
                    &format!("{e}{half}.weight_scale"),
                    &[intermediate, latent / 32],
                    packed.clone(),
                );
            }
            ck.push(
                &format!("{e}w2.weight_packed"),
                &[latent, intermediate / 2],
                packed.clone(),
            );
            ck.push(
                &format!("{e}w2.weight_scale"),
                &[latent, intermediate / 32],
                packed.clone(),
            );
        }
    }
    ck.push("language_model.model.norm.weight", &[hidden], bf16());
    ck.finish(&format!("kimi_k3_{tag}"))
}

fn kimi_k3_checkpoint() -> CheckpointMetadata {
    kimi_k3_checkpoint_at(32, Some(u8enc()), "narrow")
}

fn kimi_k3_facts() -> ModelFacts {
    let mut facts = facts("kimi_k3", 1);
    facts.num_experts = 2;
    facts
}

fn streamed() -> Policy {
    Policy {
        stream_routed_experts: true,
        ..Policy::default()
    }
}

#[test]
fn kimi_k3_cuda() {
    check(
        "kimi_k3_cuda",
        &kimi_k3_checkpoint(),
        &kimi_k3_facts(),
        &target(0, 1),
        &Policy::default(),
    );
}

/// Streaming *and* a TP split, which no other streamed golden covers.
///
/// The two existing streamed goldens run tp_size=1, so nothing until now
/// pinned what a group does to the rank's slice. K3's group shards inside the
/// expert — gate/up on their output axis, down on its input axis — so at
/// tp 1-of-2 every declared shape here is a half-width one, and a group that
/// forgot to shard would pin as a full-width slot instead of failing.
#[test]
fn kimi_k3_streamed_cuda_tp1_of_2() {
    check(
        "kimi_k3_streamed_cuda_tp1_of_2",
        &kimi_k3_checkpoint_at(64, Some(u8enc()), "wide"),
        &kimi_k3_facts(),
        &target(1, 2),
        &streamed(),
    );
}

/// The knob stops lying: a K3 that cannot stream says so at authoring time.
///
/// `stream_routed_experts` reaches the driver as a boolean, and the driver
/// builds its slab only when the contract declared groups — so a family that
/// declares none accepts the request, logs nothing, and loads the whole expert
/// bank resident, at a size the refusal message itself states — this is not a
/// difference anyone discovers from a log line that was never printed.
///
/// All three ways of declaring nothing are refused, because they arrive
/// differently: no expert names at all, and a config whose expert count is
/// zero, both reach the end of the pass having pushed nothing, while names
/// that are not MXFP4 would otherwise leave *some* layers streamed and the
/// rest resident. The zero-count case is here because the pass has no explicit
/// test for it — an empty per-expert loop reads no shapes, so the layer is
/// skipped and the same refusal catches it.
#[test]
fn kimi_k3_refuses_a_streaming_request_it_cannot_serve() {
    let no_experts = ModelFacts {
        num_experts: 0,
        ..kimi_k3_facts()
    };
    for (case, facts, checkpoint, names) in [
        (
            "no routed experts",
            kimi_k3_facts(),
            kimi_k3_checkpoint_at(64, None, "dense"),
            "no routed experts to stream",
        ),
        (
            "a config that declares zero experts",
            no_experts,
            kimi_k3_checkpoint_at(64, Some(u8enc()), "wide"),
            "no routed experts to stream",
        ),
        (
            "experts that are not MXFP4",
            kimi_k3_facts(),
            kimi_k3_checkpoint_at(64, Some(bf16()), "bf16experts"),
            "not MXFP4",
        ),
    ] {
        let err = author(&facts, &checkpoint, &target(1, 2), &streamed())
            .expect_err(&format!("kimi_k3 with {case} must refuse the knob"));
        let text = err.to_string();
        assert!(
            text.contains("stream_routed_experts was requested"),
            "{case}: names the knob: {text}"
        );
        assert!(text.contains(names), "{case}: names the reason: {text}");
    }
}

/// Every scale's declared extent follows from the weight it names.
///
/// This is the assertion the goldens cannot be. A golden is written *from* the
/// author's output, so a wrong `channel_axis` would have been recorded
/// faithfully and every mutation against it would still have passed — the
/// mistake and its record are the same artifact. The residency-parity test
/// does not close it either: it proves the two sides agree, and two identical
/// mistakes agree. Here the expected extent is derived from the **weight's**
/// shape, so a wrong axis fails on arithmetic and fails at generation time.
///
/// `entries` must be the whole set a `scales.of` could name — the group's own
/// tensors, or the contract's — so a dangling reference is caught, not skipped.
fn assert_scales_follow_their_weights(
    entries: &[pie_loader::contract::TensorContract],
    what: &str,
) {
    let mut checked = 0;
    for entry in entries {
        let (name, shape) = (&entry.name, entry.shape.clone().expect("a declared shape"));
        let Some(s) = entry.scales.as_ref() else {
            continue;
        };
        assert_eq!(
            s.granularity,
            QuantGranularity::PerGroup,
            "{what}: {name}: this check only describes PerGroup"
        );
        assert_eq!(
            s.form,
            ScaleForm::RawE8M0,
            "{what}: {name}: MXFP4 exponents are raw E8M0 bytes; F32Factors would have the \
             driver expand bytes it should read directly, and nothing checks this at runtime"
        );
        let weight = entries
            .iter()
            .find(|other| other.name == s.of)
            .and_then(|other| other.shape.clone())
            .unwrap_or_else(|| panic!("{what}: {name} scales '{}', which is not declared", s.of));

        // MXFP4 packs two codes per byte along the axis it also groups along,
        // so the grouped axis is the last one. If a later layout separates
        // them this fires rather than quietly computing against the wrong axis.
        let grouped = s.channel_axis as usize + 1;
        assert_eq!(
            grouped,
            weight.len() - 1,
            "{what}: {name}: MXFP4 groups along the packed axis, which is the last of a \
             rank-{} weight; channel_axis {} points at axis {grouped}",
            weight.len(),
            s.channel_axis
        );
        assert_eq!(
            shape.len(),
            weight.len(),
            "{what}: {name}: a scale has one entry per group, so it keeps the weight's rank"
        );
        assert_eq!(
            &shape[..grouped],
            &weight[..grouped],
            "{what}: {name}: the extents before the grouped axis must match the weight"
        );
        assert_eq!(
            weight[grouped] * 2,
            shape[grouped] * i64::from(s.group_size),
            "{what}: {name}: the weight holds {} codes on axis {grouped}, but the scale \
             declares {} entries of {}",
            weight[grouped] * 2,
            shape[grouped],
            s.group_size
        );
        checked += 1;
    }
    assert!(checked > 0, "{what}: nothing carried scales — vacuous");
}

/// The slot a rank actually pages in, at the widths Kimi-K3 really has.
///
/// Every other K3 case here is toy-width, so until now nothing checked this
/// contract against the model it exists for. The expert count stays small on
/// purpose — a slot is per expert per rank, so the count does not enter the
/// arithmetic, and 896 experts at these widths is a fixture in the gigabytes.
///
/// The assertions here are the ones a golden cannot make. A golden is written
/// *from* the author's output, so any defect it records is a defect it then
/// pins: re-blessing turns it green again. Everything below is derived from
/// something else — the checkpoint's own widths, the weight a scale names, or
/// the source a half is built from.
#[test]
fn kimi_k3_streams_real_width_slots() {
    // moonshotai/Kimi-K3, measured: hidden 7168, routed_expert_hidden_size
    // 3584, moe_intermediate_size 3072, 96 KDA heads in a 128-entry A_log.
    const HIDDEN: i64 = 7168;
    const LATENT: i64 = 3584;
    const INTERMEDIATE: i64 = 3072;
    const HEADS: i64 = 96;
    // (tp_size, bytes in one expert's four leaves on one rank). Literals,
    // computed away from this code: they are what the feature was priced on.
    const SLOT_BYTES: [(u32, i64); 4] = [
        (1, 17_547_264),
        (4, 4_386_816),
        (8, 2_193_408),
        (16, 1_096_704),
    ];

    let checkpoint = kimi_k3_checkpoint_sized(
        HIDDEN,
        LATENT,
        INTERMEDIATE,
        HEADS,
        Some(u8enc()),
        "realwidth",
    );
    for (tp, expected_bytes) in SLOT_BYTES {
        let rank = target(tp - 1, tp);
        let contract = author(&kimi_k3_facts(), &checkpoint, &rank, &streamed())
            .unwrap_or_else(|err| panic!("tp{tp}: authoring failed: {err}"))
            .expect("kimi_k3 has an author");

        // The whole feature is that the bank is NOT resident. Publishing the
        // experts as well as the group would page them in *and* keep them —
        // 1.4465 TB of it — and every shape below would still be right.
        let resident: Vec<&str> = contract
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|name| name.contains("block_sparse_moe"))
            .collect();
        assert!(
            resident.is_empty(),
            "tp{tp}: streaming left {} expert tensors resident beside the slab: {resident:?}",
            resident.len()
        );

        assert_eq!(
            contract.groups.len(),
            1,
            "tp{tp}: one group per MoE layer, and this fixture has one layer"
        );
        let group = &contract.groups[0];
        assert_eq!(group.arity, 2, "tp{tp}: arity is the expert count");

        let local = INTERMEDIATE / i64::from(tp);
        let expected: [(&str, Vec<i64>); 4] = [
            ("gate_up.weight_packed", vec![local, 2, LATENT / 2]),
            ("gate_up.weight_scale", vec![local, 2, LATENT / 32]),
            ("down.weight_packed", vec![LATENT, local / 2]),
            ("down.weight_scale", vec![LATENT, local / 32]),
        ];
        let mut bytes = 0i64;
        for (name, want) in &expected {
            let got = group
                .tensors
                .iter()
                .find(|t| t.name == *name)
                .unwrap_or_else(|| panic!("tp{tp}: the group declares no {name}"));
            let shape = got.shape.clone().expect("a declared shape");
            assert_eq!(&shape, want, "tp{tp}: {name}");
            // Every leaf is Raw(U8), so the element count is the byte count.
            assert_eq!(
                got.encoding,
                Encoding::Raw(DType::U8),
                "tp{tp}: {name} must stay packed"
            );
            bytes += shape.iter().product::<i64>();
        }
        assert_eq!(
            bytes, expected_bytes,
            "tp{tp}: one expert's slot is {bytes} B on this rank, expected {expected_bytes}"
        );
        assert_eq!(
            group.tensors.len(),
            expected.len(),
            "tp{tp}: the slot is exactly these four leaves — anything else is bytes paged in \
             per expert that this arithmetic did not price"
        );

        // Gate over up, and the two leaves must agree on it. w1 and w3 have
        // identical declared shapes, so no shape or byte assertion above can
        // see them swapped — and a swap gives a slot whose gate rows carry
        // up's scales, which is wrong output rather than a wrong size.
        for (leaf, gate, up) in [
            (
                "gate_up.weight_packed",
                "w1.weight_packed",
                "w3.weight_packed",
            ),
            ("gate_up.weight_scale", "w1.weight_scale", "w3.weight_scale"),
        ] {
            let expr = serde_json::to_string(
                &group
                    .tensors
                    .iter()
                    .find(|t| t.name == leaf)
                    .expect("a declared leaf")
                    .expr,
            )
            .expect("serialize expr");
            let (at_gate, at_up) = (expr.find(gate), expr.find(up));
            assert!(
                at_gate.is_some() && at_up.is_some(),
                "tp{tp}: {leaf} is not built from {gate} and {up}: {expr}"
            );
            assert!(
                at_gate < at_up,
                "tp{tp}: {leaf} joins {up} before {gate}; the decode GEMV reads row 2i as \
                 gate and 2i+1 as up, so the halves are not interchangeable"
            );
        }

        // `assert_scales_follow_their_weights` skips a tensor that carries no
        // scales, and its vacuity guard is satisfied by whichever leaf still
        // has them — so name the two that must, or dropping one is invisible.
        for leaf in ["gate_up.weight_scale", "down.weight_scale"] {
            let got = group
                .tensors
                .iter()
                .find(|t| t.name == leaf)
                .expect("a declared leaf");
            assert!(
                got.scales.is_some(),
                "tp{tp}: {leaf} declares no scales, so nothing says which weight it scales"
            );
        }
        assert_scales_follow_their_weights(&group.tensors, &format!("tp{tp} streamed"));

        // Declared shapes are a claim; the plan is whether the expressions can
        // actually produce them at this rank.
        let plan = compile_load_plan(&checkpoint, &contract, rank.clone())
            .unwrap_or_else(|err| panic!("tp{tp}: compiling failed: {err}"));
        if let Err(violations) =
            pie_loader_capi::view::verify_marshalled(&plan, Some(&ContractView::of(&contract)))
        {
            let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
            panic!(
                "tp{tp}: the plan does not honour its contract:\n  {}",
                listed.join("\n  ")
            );
        }
    }
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
    ck.push(&format!("{p}attn.attn_sink"), &[4], f32enc());
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

fn deepseek_v4_imported_checkpoint() -> CheckpointMetadata {
    // The import preserves typed E8M0 scales; safetensors presents them as U8.
    let (hidden, intermediate) = (4096, 2048);
    let mut ck = Checkpoint::new();
    ck.push("embed.weight", &[129280, hidden], bf16());
    ck.push("head.weight", &[129280, hidden], bf16());
    ck.push("hc_head_base", &[4], bf16());
    ck.push("hc_head_fn", &[4, 16384], bf16());
    ck.push("hc_head_scale", &[1], bf16());
    let p = "layers.0.";
    ck.push(&format!("{p}attn_norm.weight"), &[hidden], bf16());
    ck.push(&format!("{p}ffn_norm.weight"), &[hidden], bf16());
    ck.push(&format!("{p}attn.attn_sink"), &[64], f32enc());
    ck.push(&format!("{p}attn.q_norm.weight"), &[1024], bf16());
    ck.push(&format!("{p}attn.kv_norm.weight"), &[512], bf16());
    for (name, weight, scale) in [
        ("attn.wq_a", [1024, 4096], [8, 32]),
        ("attn.wq_b", [32768, 1024], [256, 8]),
        ("attn.wkv", [512, 4096], [4, 32]),
        ("attn.wo_a", [8192, 4096], [64, 32]),
        ("attn.wo_b", [4096, 8192], [32, 64]),
        ("ffn.shared_experts.w1", [2048, 4096], [16, 32]),
        ("ffn.shared_experts.w2", [4096, 2048], [32, 16]),
        ("ffn.shared_experts.w3", [2048, 4096], [16, 32]),
    ] {
        ck.push(
            &format!("{p}{name}.weight"),
            &weight,
            Encoding::Raw(DType::F8E4M3),
        );
        ck.push(
            &format!("{p}{name}.scale"),
            &scale,
            Encoding::Raw(DType::E8M0),
        );
    }
    ck.push(&format!("{p}ffn.gate.weight"), &[256, hidden], bf16());
    ck.push(
        &format!("{p}ffn.gate.tid2eid"),
        &[129280, 6],
        Encoding::Raw(DType::I64),
    );
    for stem in ["hc_attn", "hc_ffn"] {
        ck.push(&format!("{p}{stem}_base"), &[24], bf16());
        ck.push(&format!("{p}{stem}_fn"), &[24, 16384], bf16());
        ck.push(&format!("{p}{stem}_scale"), &[3], bf16());
    }
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
                Encoding::Raw(DType::E8M0),
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
            Encoding::Raw(DType::E8M0),
        );
    }

    // The C4 indexer and both compressor shapes are separate real header
    // classes even though they are outside the one-layer expert fixture.
    for (layer, ape, width) in [(2, [4, 1024], 1024), (3, [128, 512], 512)] {
        let a = format!("layers.{layer}.attn.compressor.");
        ck.push(&format!("{a}ape"), &ape, f32enc());
        ck.push(&format!("{a}norm.weight"), &[512], bf16());
        ck.push(&format!("{a}wgate.weight"), &[width, hidden], bf16());
        ck.push(&format!("{a}wkv.weight"), &[width, hidden], bf16());
    }
    let indexer = "layers.2.attn.indexer.";
    ck.push(
        &format!("{indexer}weights_proj.weight"),
        &[64, hidden],
        bf16(),
    );
    ck.push(
        &format!("{indexer}wq_b.weight"),
        &[8192, 1024],
        Encoding::Raw(DType::F8E4M3),
    );
    ck.push(
        &format!("{indexer}wq_b.scale"),
        &[64, 8],
        Encoding::Raw(DType::E8M0),
    );
    let ic = format!("{indexer}compressor.");
    ck.push(&format!("{ic}ape"), &[4, 256], f32enc());
    ck.push(&format!("{ic}norm.weight"), &[128], bf16());
    ck.push(&format!("{ic}wgate.weight"), &[256, hidden], bf16());
    ck.push(&format!("{ic}wkv.weight"), &[256, hidden], bf16());
    ck.push("layers.3.ffn.gate.bias", &[256], f32enc());

    // MTP is not bound by the decoder, but its block-FP8 pairs still pass
    // through the family-wide scale normalizer.
    let m = "mtp.0.";
    for (name, weight, scale) in [
        ("attn.wq_a", [1024, 4096], [8, 32]),
        ("attn.wq_b", [32768, 1024], [256, 8]),
        ("attn.wkv", [512, 4096], [4, 32]),
        ("attn.wo_a", [8192, 4096], [64, 32]),
        ("attn.wo_b", [4096, 8192], [32, 64]),
        ("e_proj", [4096, 4096], [32, 32]),
        ("h_proj", [4096, 4096], [32, 32]),
        ("ffn.shared_experts.w1", [2048, 4096], [16, 32]),
        ("ffn.shared_experts.w2", [4096, 2048], [32, 16]),
        ("ffn.shared_experts.w3", [2048, 4096], [16, 32]),
    ] {
        ck.push(
            &format!("{m}{name}.weight"),
            &weight,
            Encoding::Raw(DType::F8E4M3),
        );
        ck.push(
            &format!("{m}{name}.scale"),
            &scale,
            Encoding::Raw(DType::E8M0),
        );
    }
    for name in [
        "attn_norm.weight",
        "ffn_norm.weight",
        "enorm.weight",
        "hnorm.weight",
        "norm.weight",
    ] {
        ck.push(&format!("{m}{name}"), &[hidden], bf16());
    }
    ck.push(&format!("{m}attn.q_norm.weight"), &[1024], bf16());
    ck.push(&format!("{m}attn.kv_norm.weight"), &[512], bf16());
    ck.push(&format!("{m}attn.attn_sink"), &[64], f32enc());
    ck.push(&format!("{m}ffn.gate.weight"), &[256, hidden], bf16());
    ck.push(&format!("{m}ffn.gate.bias"), &[256], f32enc());
    for half in ["w1", "w3"] {
        ck.push(
            &format!("{m}ffn.experts.0.{half}.weight"),
            &[intermediate, hidden / 2],
            Encoding::Raw(DType::I8),
        );
        ck.push(
            &format!("{m}ffn.experts.0.{half}.scale"),
            &[intermediate, hidden / 32],
            Encoding::Raw(DType::E8M0),
        );
    }
    ck.push(
        &format!("{m}ffn.experts.0.w2.weight"),
        &[hidden, intermediate / 2],
        Encoding::Raw(DType::I8),
    );
    ck.push(
        &format!("{m}ffn.experts.0.w2.scale"),
        &[hidden, intermediate / 32],
        Encoding::Raw(DType::E8M0),
    );
    for stem in ["hc_attn", "hc_ffn"] {
        ck.push(&format!("{m}{stem}_base"), &[24], bf16());
        ck.push(&format!("{m}{stem}_fn"), &[24, 16384], bf16());
        ck.push(&format!("{m}{stem}_scale"), &[3], bf16());
    }
    ck.push(&format!("{m}hc_head_base"), &[4], bf16());
    ck.push(&format!("{m}hc_head_fn"), &[4, 16384], bf16());
    ck.push(&format!("{m}hc_head_scale"), &[1], bf16());
    ck.push("norm.weight", &[hidden], bf16());
    ck.finish("deepseek_v4_imported")
}

fn assert_imported_block_scale_decodes(contract: &pie_loader::contract::ModelContract) {
    let expected = [
        "layers.0.attn.wq_a.scale",
        "layers.0.attn.wq_b.scale",
        "layers.0.attn.wkv.scale",
        "layers.0.attn.wo_a.scale",
        "layers.0.attn.wo_b.scale",
        "layers.0.ffn.shared_experts.w1.scale",
        "layers.0.ffn.shared_experts.w2.scale",
        "layers.0.ffn.shared_experts.w3.scale",
        "layers.2.attn.indexer.wq_b.scale",
        "mtp.0.attn.wq_a.scale",
        "mtp.0.attn.wq_b.scale",
        "mtp.0.attn.wkv.scale",
        "mtp.0.attn.wo_a.scale",
        "mtp.0.attn.wo_b.scale",
        "mtp.0.e_proj.scale",
        "mtp.0.h_proj.scale",
        "mtp.0.ffn.shared_experts.w1.scale",
        "mtp.0.ffn.shared_experts.w2.scale",
        "mtp.0.ffn.shared_experts.w3.scale",
    ];
    for name in expected {
        let scale = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("imported block scale '{name}' is published"));
        assert_eq!(scale.encoding, Encoding::Raw(DType::U8), "{name}");
        assert_eq!(
            scale.scales.as_ref().map(|scales| scales.form),
            Some(ScaleForm::F32Factors),
            "{name}"
        );
        let weight_name = format!("{}.weight", name.trim_end_matches(".scale"));
        let weight = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == weight_name)
            .unwrap_or_else(|| panic!("block scale '{name}' keeps its companion weight"));
        assert_eq!(weight.encoding, Encoding::Raw(DType::F8E4M3), "{name}");
    }

    // The MTP routed experts are not decoder groups, but they prove that the
    // companion-type guard does not reinterpret MXFP4 scales as FP8 factors.
    for half in ["w1", "w2", "w3"] {
        let name = format!("mtp.0.ffn.experts.0.{half}.scale");
        let scale = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("routed scale '{name}' is published"));
        assert_eq!(scale.encoding, Encoding::Raw(DType::E8M0), "{name}");
        assert!(scale.scales.is_none(), "{name}");
    }
}

#[test]
fn deepseek_v4_attention_sink_survives_contract_authoring() {
    let metadata = deepseek_v4_checkpoint();
    for policy in [
        Policy::default(),
        Policy {
            stream_routed_experts: true,
            ..Policy::default()
        },
    ] {
        let contract = author(&facts("deepseek_v4", 1), &metadata, &target(0, 1), &policy)
            .expect("DeepSeek-V4 contract authoring succeeds")
            .expect("DeepSeek-V4 has a contract author");
        let sink = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.attn.attn_sink")
            .expect("DeepSeek-V4 attention sink survives contract authoring");
        assert_eq!(sink.shape, Some(vec![4]));
        assert_eq!(sink.encoding, f32enc());
        assert_eq!(sink.visibility, Visibility::Public);
    }
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
fn deepseek_v4_eager_imported_mxfp4_metadata_compiles() {
    let checkpoint = deepseek_v4_imported_checkpoint();
    let target = target(0, 4);
    let (contract, resolved) = author_with_policy(
        &facts("deepseek_v4", 1),
        &checkpoint,
        &target,
        &Policy::default(),
    )
    .expect("authoring succeeds")
    .expect("deepseek_v4 has an author");

    assert_eq!(resolved, Mxfp4MoePolicy::EagerBf16);
    assert_imported_block_scale_decodes(&contract);
    let eager = [
        "layers.0.ffn.experts.gate_up.weight",
        "layers.0.ffn.experts.down.weight",
    ];
    for name in eager {
        let tensor = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("eager expert stack '{name}' is published"));
        assert_eq!(tensor.encoding, Encoding::Raw(DType::BF16), "{name}");
    }
    assert!(
        contract
            .tensors
            .iter()
            .all(|tensor| !tensor.name.starts_with("layers.0.ffn.experts.marlin.")),
        "a non-native target must not publish native Marlin expert banks"
    );

    let plan = compile_load_plan(&checkpoint, &contract, target)
        .expect("the eager contract compiles against imported MXFP4 metadata");
    for name in eager {
        assert!(
            plan.tensors.iter().any(|tensor| tensor.name == name),
            "the eager expert stack '{name}' reaches the bindable plan"
        );
    }
}

#[test]
fn deepseek_v4_native_cuda_keeps_imported_mxfp4_packed() {
    let checkpoint = deepseek_v4_imported_checkpoint();
    let mut target = target(0, 4);
    target.native_mxfp4_moe = true;
    let (contract, resolved) = author_with_policy(
        &facts("deepseek_v4", 1),
        &checkpoint,
        &target,
        &Policy::default(),
    )
    .expect("authoring succeeds")
    .expect("deepseek_v4 has an author");

    assert_eq!(resolved, Mxfp4MoePolicy::NativeGemm);
    assert_imported_block_scale_decodes(&contract);
    for half in ["gate", "up", "down"] {
        let weight_name = format!("layers.0.ffn.experts.marlin.{half}.weight");
        let weight = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == weight_name)
            .unwrap_or_else(|| panic!("native {half} weight is published"));
        assert!(
            matches!(weight.encoding, Encoding::Quant(_)),
            "native {half} must remain packed, got {:?}",
            weight.encoding
        );
        let scale = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == format!("layers.0.ffn.experts.marlin.{half}.scale"))
            .unwrap_or_else(|| panic!("native {half} scale is published"));
        assert_eq!(scale.encoding, Encoding::Raw(DType::U8));
        assert_eq!(
            scale
                .scales
                .as_ref()
                .map(|scales| (&scales.of, scales.group_size, scales.form)),
            Some((&weight_name, 32, ScaleForm::RawE8M0))
        );
    }
    assert!(contract.tensors.iter().all(|tensor| {
        !tensor.name.starts_with("layers.0.ffn.experts.")
            || tensor.encoding != Encoding::Raw(DType::BF16)
    }));
    for name in [
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "layers.0.hc_attn_base",
        "layers.0.hc_attn_fn",
        "layers.0.hc_attn_scale",
        "layers.0.hc_ffn_base",
        "layers.0.hc_ffn_fn",
        "layers.0.hc_ffn_scale",
    ] {
        let tensor = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("HC function '{name}' is published"));
        assert_eq!(tensor.encoding, Encoding::Raw(DType::F32), "{name}");
    }

    let plan = compile_load_plan(&checkpoint, &contract, target)
        .expect("the native contract compiles against imported E8M0 metadata");
    let repacks: Vec<_> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Repack,
                transform,
                ..
            } => transform.repack,
            _ => None,
        })
        .collect();
    assert_eq!(repacks.len(), 6);
    assert_eq!(
        repacks
            .iter()
            .filter(|repack| repack.layout == RepackLayout::MarlinMxfp4Weight)
            .count(),
        3
    );
    assert_eq!(
        repacks
            .iter()
            .filter(|repack| repack.layout == RepackLayout::MarlinMxfp4Scale)
            .count(),
        3
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

#[test]
fn deepseek_v4_streamed_imported_mxfp4_metadata_compiles() {
    let checkpoint = deepseek_v4_imported_checkpoint();
    let policy = Policy {
        stream_routed_experts: true,
        ..Policy::default()
    };
    let target = target(0, 4);
    let contract = author(&facts("deepseek_v4", 1), &checkpoint, &target, &policy)
        .expect("authoring succeeds")
        .expect("deepseek_v4 has an author");

    assert_imported_block_scale_decodes(&contract);
    compile_load_plan(&checkpoint, &contract, target)
        .expect("the streamed contract compiles against imported MXFP4 metadata");
}

#[test]
fn deepseek_v4_does_not_reinterpret_a_mixed_expert_triplet() {
    let mut checkpoint = deepseek_v4_imported_checkpoint();
    let w3 = checkpoint
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "layers.0.ffn.experts.0.w3.weight")
        .expect("the fixture has expert w3");
    w3.encoding = Encoding::Raw(DType::U8);

    for (mode, policy) in [
        ("eager", Policy::default()),
        (
            "streamed",
            Policy {
                stream_routed_experts: true,
                ..Policy::default()
            },
        ),
    ] {
        let contract = author(
            &facts("deepseek_v4", 1),
            &checkpoint,
            &target(0, 4),
            &policy,
        )
        .expect("authoring succeeds")
        .expect("deepseek_v4 has an author");
        assert!(
            contract
                .tensors
                .iter()
                .all(|tensor| tensor.name != "layers.0.ffn.experts.gate_up.weight"),
            "{mode}: a mixed triplet must not become an eager MXFP4 stack"
        );
        assert!(
            contract
                .groups
                .iter()
                .all(|group| group.name != "layers.0.ffn.experts"),
            "{mode}: a mixed triplet must not become a streamed MXFP4 group"
        );
        let published = contract
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.ffn.experts.0.w3.weight")
            .expect("the unclaimed source is published unchanged");
        assert_eq!(published.encoding, Encoding::Raw(DType::U8), "{mode}");
    }
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
        ck.push(
            &format!("{p}self_attn.{proj}.weight"),
            &[hidden, hidden],
            bf16(),
        );
    }
    ck.push(
        &format!("{p}post_attention_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}mlp.gate_proj.weight"),
        &[intermediate, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}mlp.up_proj.weight"),
        &[intermediate, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}mlp.down_proj.weight"),
        &[hidden, intermediate],
        bf16(),
    );
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
            .expect_err(&format!(
                "{family} runtime_quant={quant:?} should be refused"
            ));
            let text = err.to_string();
            assert!(
                text.contains("no encoder here"),
                "{family} {quant:?}: {text}"
            );
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
    assert!(
        seen >= 7,
        "fixture should carry every projection, saw {seen}"
    );
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
