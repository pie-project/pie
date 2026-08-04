//! The §8-3 differential: the C++ author against the Rust author, same
//! snapshot, same request, contract against contract.
//!
//! This is the test the whole migration is proved by
//! (`plan/model-in-rust.md` §8): `pie_cuda_author_contract` — the CUDA
//! driver's author-only entry, no device needed — authors each family's
//! contract over a synthetic snapshot on disk, `pie_model::contract::author`
//! authors from the same facts and policy, and the two serialize
//! byte-identically or the test names the first divergence. Equal contracts
//! mean equal plans mean equal cache keys mean identical materialized
//! weights, by construction.
//!
//! The fixtures are the same shapes `model/tests/family_contracts.rs` pins
//! goldens over, written here as *real* files because the C++ side parses
//! `config.json` and the safetensors headers itself — which is the point:
//! the C++ author's inputs are read the C++ way, and the Rust facts mirror
//! `config.cpp`'s key chains rather than inventing their own.

#![cfg(feature = "driver-cuda")]

use std::path::Path;

use pie_loader::contract::ModelContract;
use pie_loader::plan::StorageTarget;
use pie_loader::types::BackendKind;
use pie_model::common::facts::ModelFacts;
use pie_model::common::policy::{Mxfp4MoeRequest, Policy};
use pie_worker::contract_author::{AuthorRequest, author_contract};

/// One tensor of a synthetic snapshot.
struct Spec {
    name: String,
    shape: Vec<i64>,
    dtype: &'static str,
}

fn t(name: impl Into<String>, shape: &[i64], dtype: &'static str) -> Spec {
    Spec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
    }
}

fn dtype_width(dtype: &str) -> u64 {
    match dtype {
        "BF16" | "F16" => 2,
        "F32" | "I32" | "U32" => 4,
        "U8" | "I8" | "F8_E4M3" => 1,
        other => panic!("unsupported fixture dtype {other}"),
    }
}

/// Write `specs` as one real safetensors file plus `config` as config.json.
fn write_snapshot(dir: &Path, config: &str, specs: &[Spec]) {
    let mut header = String::from("{");
    let mut offset = 0u64;
    for (index, spec) in specs.iter().enumerate() {
        let elements: i64 = spec.shape.iter().product();
        let nbytes = elements as u64 * dtype_width(spec.dtype);
        if index > 0 {
            header.push(',');
        }
        let dims: Vec<String> = spec.shape.iter().map(ToString::to_string).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            spec.name,
            spec.dtype,
            dims.join(","),
            offset + nbytes
        ));
        offset += nbytes;
    }
    header.push('}');
    let mut file = Vec::with_capacity(8 + header.len() + offset as usize);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend(std::iter::repeat_n(0u8, offset as usize));
    std::fs::create_dir_all(dir).expect("create snapshot dir");
    std::fs::write(dir.join("model.safetensors"), file).expect("write snapshot");
    std::fs::write(dir.join("config.json"), config).expect("write config");
}

/// The facts as `config.cpp` would compute them, mirrored key chain by key
/// chain — including the MLA override that redefines `head_dim` as
/// `qk_nope + qk_rope`.
fn facts_from_config(config: &serde_json::Value) -> ModelFacts {
    // A multimodal release nests the decoder's facts under `text_config`;
    // `config.cpp` reads through the same view.
    let text = config.get("text_config");
    let get = |key: &str| {
        config
            .get(key)
            .or_else(|| text.and_then(|t| t.get(key)))
            .and_then(serde_json::Value::as_i64)
    };
    let model_type = config["model_type"]
        .as_str()
        .expect("model_type")
        .to_string();
    let quant_method = config
        .get("quantization_config")
        .and_then(|quant| quant.get("quant_method"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let num_hidden_layers = get("num_hidden_layers").expect("num_hidden_layers") as u32;
    let num_experts = get("num_local_experts")
        .or_else(|| get("num_experts"))
        .or_else(|| get("n_routed_experts"))
        .unwrap_or(0) as u32;
    let mut head_dim = get("head_dim").unwrap_or_else(|| {
        get("hidden_size").expect("hidden_size") / get("num_attention_heads").expect("heads")
    });
    let (nope, rope) = (
        get("qk_nope_head_dim").unwrap_or(0),
        get("qk_rope_head_dim").unwrap_or(0),
    );
    if get("v_head_dim").unwrap_or(0) > 0 && nope > 0 && rope > 0 {
        head_dim = nope + rope;
    }
    let mamba_groups = get("n_groups")
        .or_else(|| get("mamba_n_groups"))
        .unwrap_or(0) as u32;
    ModelFacts {
        model_type,
        quant_method,
        num_hidden_layers,
        num_experts,
        head_dim: head_dim as u32,
        mamba_groups,
        ..Default::default()
    }
}

struct Case {
    name: &'static str,
    config: String,
    specs: Vec<Spec>,
    tp: (u32, u32),
    moe_request: u32,
    stream: bool,
    native_mxfp4_moe: bool,
}

impl Case {
    fn new(name: &'static str, config: String, specs: Vec<Spec>) -> Self {
        Self {
            name,
            config,
            specs,
            tp: (0, 1),
            moe_request: 0,
            stream: false,
            native_mxfp4_moe: false,
        }
    }
}

/// Author both ways and hold the two contracts to byte equality.
fn diff(case: &Case) {
    let dir = std::env::temp_dir().join(format!(
        "pie_author_diff_{}_{}",
        case.name,
        std::process::id()
    ));
    write_snapshot(&dir, &case.config, &case.specs);

    // The C++ author, through the same entry `pie model optimize` was
    // designed to call.
    let cpp: ModelContract = author_contract(&AuthorRequest {
        snapshot_dir: &dir,
        runtime_quant: "",
        mxfp4_moe_request: case.moe_request,
        component: 0,
        stream_routed_experts: case.stream,
        tp_rank: case.tp.0,
        tp_size: case.tp.1,
        fp8_native: false,
        native_mxfp4_moe: case.native_mxfp4_moe,
    })
    .unwrap_or_else(|err| panic!("{}: C++ author failed: {err}", case.name));

    // The Rust author, from the same facts and the same policy point.
    let config: serde_json::Value =
        serde_json::from_str(&case.config).expect("fixture config parses");
    let facts = facts_from_config(&config);
    let metadata = pie_loader::checkpoint::read::parse_checkpoint_metadata(&dir)
        .unwrap_or_else(|err| panic!("{}: reading the snapshot back: {err}", case.name));
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: case.tp.0,
        tp_size: case.tp.1,
        preferred_alignment: 256,
        native_mxfp4_moe: case.native_mxfp4_moe,
        ..StorageTarget::default()
    };
    let policy = Policy {
        moe_request: match case.moe_request {
            0 => Mxfp4MoeRequest::Auto,
            1 => Mxfp4MoeRequest::RoutedDecode,
            2 => Mxfp4MoeRequest::NativeGemm,
            3 => Mxfp4MoeRequest::EagerBf16,
            other => panic!("bad moe request {other}"),
        },
        stream_routed_experts: case.stream,
        ..Policy::default()
    };
    let rust = pie_model::contract::author(&facts, &metadata, &target, &policy)
        .unwrap_or_else(|err| panic!("{}: Rust author failed: {err}", case.name))
        .unwrap_or_else(|| panic!("{}: no Rust author for {}", case.name, facts.model_type));

    let cpp_text = serde_json::to_string_pretty(&cpp).expect("serialize");
    let rust_text = serde_json::to_string_pretty(&rust).expect("serialize");
    if cpp_text != rust_text {
        let divergence = cpp_text
            .lines()
            .zip(rust_text.lines())
            .enumerate()
            .find(|(_, (c, r))| c != r)
            .map(|(line, (c, r))| format!("line {}:\n  c++ : {c}\n  rust: {r}", line + 1))
            .unwrap_or_else(|| {
                format!(
                    "c++ has {} lines, rust has {}",
                    cpp_text.lines().count(),
                    rust_text.lines().count()
                )
            });
        let dump = |tag: &str, text: &str| {
            let path = dir.join(format!("{tag}.contract.json"));
            std::fs::write(&path, text).ok();
            path.display().to_string()
        };
        panic!(
            "{}: the two authors disagree; first divergence at {divergence}\n\
             full dumps: {} vs {}",
            case.name,
            dump("cpp", &cpp_text),
            dump("rust", &rust_text),
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ── the fixtures, shaped as in model/tests/family_contracts.rs ──────

fn dense_layer(
    specs: &mut Vec<Spec>,
    p: &str,
    hidden: i64,
    heads: i64,
    kv: i64,
    hd: i64,
    inter: i64,
) {
    specs.push(t(format!("{p}input_layernorm.weight"), &[hidden], "BF16"));
    specs.push(t(
        format!("{p}self_attn.q_proj.weight"),
        &[heads * hd, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.k_proj.weight"),
        &[kv * hd, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.v_proj.weight"),
        &[kv * hd, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.o_proj.weight"),
        &[hidden, heads * hd],
        "BF16",
    ));
    specs.push(t(
        format!("{p}post_attention_layernorm.weight"),
        &[hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}mlp.gate_proj.weight"),
        &[inter, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}mlp.up_proj.weight"),
        &[inter, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}mlp.down_proj.weight"),
        &[hidden, inter],
        "BF16",
    ));
}

fn llama_case(tp: (u32, u32)) -> Case {
    let (hidden, heads, kv, hd, inter, vocab) = (256, 8, 2, 32, 704, 512);
    let mut specs = vec![t("model.embed_tokens.weight", &[vocab, hidden], "BF16")];
    for layer in 0..2 {
        dense_layer(
            &mut specs,
            &format!("model.layers.{layer}."),
            hidden,
            heads,
            kv,
            hd,
            inter,
        );
    }
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    specs.push(t("lm_head.weight", &[vocab, hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"llama3","architectures":["LlamaForCausalLM"],
            "num_hidden_layers":2,"hidden_size":{hidden},"head_dim":{hd},
            "num_attention_heads":{heads},"num_key_value_heads":{kv},
            "intermediate_size":{inter},"vocab_size":{vocab},
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6,"tie_word_embeddings":false}}"#
    );
    let mut case = Case::new(if tp.1 > 1 { "llama_tp2" } else { "llama" }, config, specs);
    case.tp = tp;
    case
}

#[test]
fn llama_dense() {
    diff(&llama_case((0, 1)));
}

#[test]
fn llama_dense_tp1_of_2() {
    diff(&llama_case((1, 2)));
}

#[test]
fn gemma4_dense() {
    let (hidden, heads, kv, hd, inter) = (64, 4, 2, 16, 96);
    let mut specs = vec![t("model.embed_tokens.weight", &[128, hidden], "BF16")];
    let p = "model.language_model.layers.0.";
    dense_layer(&mut specs, p, hidden, heads, kv, hd, inter);
    specs.push(t(format!("{p}router.scale"), &[hidden], "BF16"));
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"gemma4_text","architectures":["Gemma4ForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":{hd},
            "num_attention_heads":{heads},"num_key_value_heads":{kv},
            "intermediate_size":{inter},"vocab_size":128,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("gemma4", config, specs));
}

#[test]
fn qwen3_5_dense_tp1_of_2() {
    let (hidden, heads, kv, hd) = (64, 4, 2, 16);
    let (k_dim, v_dim) = (32, 32);
    let conv_dim = 2 * k_dim + v_dim;
    let mut specs = vec![t("model.embed_tokens.weight", &[128, hidden], "BF16")];
    let la = "model.layers.0.linear_attn.";
    specs.push(t(
        format!("{la}in_proj_qkv.weight"),
        &[conv_dim, hidden],
        "BF16",
    ));
    specs.push(t(format!("{la}in_proj_z.weight"), &[v_dim, hidden], "BF16"));
    specs.push(t(format!("{la}in_proj_b.weight"), &[8, hidden], "BF16"));
    specs.push(t(format!("{la}in_proj_a.weight"), &[8, hidden], "BF16"));
    specs.push(t(format!("{la}conv1d.weight"), &[conv_dim, 1, 4], "BF16"));
    specs.push(t(format!("{la}conv1d.bias"), &[conv_dim], "BF16"));
    specs.push(t(format!("{la}out_proj.weight"), &[hidden, v_dim], "BF16"));
    specs.push(t(format!("{la}A_log"), &[8], "BF16"));
    specs.push(t(format!("{la}dt_bias"), &[8], "BF16"));
    specs.push(t(format!("{la}norm.weight"), &[v_dim], "F32"));
    dense_layer(&mut specs, "model.layers.1.", hidden, heads, kv, hd, 96);
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"qwen3_5","architectures":["Qwen3_5ForCausalLM"],
            "num_hidden_layers":2,"hidden_size":{hidden},"head_dim":{hd},
            "num_attention_heads":{heads},"num_key_value_heads":{kv},
            "intermediate_size":96,"vocab_size":128,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    let mut case = Case::new("qwen3_5_tp2", config, specs);
    case.tp = (1, 2);
    diff(&case);
}

#[test]
fn gpt_oss_routed() {
    let (hidden, experts, inter) = (64, 2i64, 64);
    let mut specs = Vec::new();
    let p = "model.layers.0.mlp.experts";
    specs.push(t(
        format!("{p}.gate_up_proj_blocks"),
        &[experts, 2 * inter, hidden / 32, 16],
        "U8",
    ));
    specs.push(t(
        format!("{p}.gate_up_proj_scales"),
        &[experts, 2 * inter, hidden / 32],
        "U8",
    ));
    specs.push(t(
        format!("{p}.gate_up_proj_bias"),
        &[experts, 2 * inter],
        "BF16",
    ));
    specs.push(t(
        format!("{p}.down_proj_blocks"),
        &[experts, hidden, inter / 32, 16],
        "U8",
    ));
    specs.push(t(
        format!("{p}.down_proj_scales"),
        &[experts, hidden, inter / 32],
        "U8",
    ));
    specs.push(t(format!("{p}.down_proj_bias"), &[experts, hidden], "BF16"));
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"gpt_oss","architectures":["GptOssForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":2,
            "intermediate_size":{inter},"vocab_size":128,
            "num_local_experts":{experts},"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("gpt_oss_routed", config, specs));
}

#[test]
fn deepseek_v4_eager() {
    let (hidden, inter) = (64, 32);
    let mut specs = vec![t("embed_tokens.weight", &[128, hidden], "BF16")];
    let p = "layers.0.";
    specs.push(t(format!("{p}attn.wq.weight"), &[64, 64], "F8_E4M3"));
    specs.push(t(format!("{p}attn.wq.scale"), &[2, 2], "U8"));
    for expert in 0..2 {
        let e = format!("{p}ffn.experts.{expert}.");
        for half in ["w1", "w3"] {
            specs.push(t(format!("{e}{half}.weight"), &[inter, hidden / 2], "I8"));
            specs.push(t(format!("{e}{half}.scale"), &[inter, hidden / 32], "U8"));
        }
        specs.push(t(format!("{e}w2.weight"), &[hidden, inter / 2], "I8"));
        specs.push(t(format!("{e}w2.scale"), &[hidden, inter / 32], "U8"));
    }
    specs.push(t("norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"deepseek_v4","architectures":["DeepseekV4ForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":4,
            "intermediate_size":{inter},"vocab_size":128,
            "n_routed_experts":2,"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("deepseek_v4", config, specs));
}

#[test]
fn phi3_fused_splits() {
    let (hidden, kv_rows, inter) = (64i64, 32i64, 96i64);
    let mut specs = vec![t("model.embed_tokens.weight", &[128, hidden], "BF16")];
    let p = "model.layers.0.";
    specs.push(t(format!("{p}input_layernorm.weight"), &[hidden], "BF16"));
    specs.push(t(
        format!("{p}self_attn.qkv_proj.weight"),
        &[hidden + 2 * kv_rows, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.o_proj.weight"),
        &[hidden, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}mlp.gate_up_proj.weight"),
        &[2 * inter, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}mlp.down_proj.weight"),
        &[hidden, inter],
        "BF16",
    ));
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"phi3","architectures":["Phi3ForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":2,
            "intermediate_size":{inter},"vocab_size":128,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("phi3", config, specs));
}

#[test]
fn glm5_fp8_and_expert_stacks() {
    let (hidden, inter, experts) = (64i64, 32i64, 2i64);
    let mut specs = vec![t("model.embed_tokens.weight", &[128, hidden], "BF16")];
    let p = "model.layers.0.";
    specs.push(t(format!("{p}input_layernorm.weight"), &[hidden], "BF16"));
    specs.push(t(
        format!("{p}self_attn.kv_b_proj.weight"),
        &[64, 32],
        "F8_E4M3",
    ));
    specs.push(t(
        format!("{p}self_attn.kv_b_proj.weight_scale_inv"),
        &[2, 1],
        "F32",
    ));
    for expert in 0..experts {
        let e = format!("{p}mlp.experts.{expert}.");
        specs.push(t(format!("{e}gate_proj.weight"), &[inter, hidden], "BF16"));
        specs.push(t(format!("{e}up_proj.weight"), &[inter, hidden], "BF16"));
        specs.push(t(format!("{e}down_proj.weight"), &[hidden, inter], "BF16"));
    }
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"glm_moe_dsa","architectures":["GlmMoeDsaForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":2,
            "intermediate_size":{inter},"vocab_size":128,
            "n_routed_experts":{experts},"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("glm5", config, specs));
}

#[test]
fn qwen3_5_moe_stacks_and_shared_expert() {
    let (hidden, heads, kv, hd, inter, experts) = (64i64, 4i64, 2i64, 16i64, 32i64, 2i64);
    let mut specs = vec![t("model.embed_tokens.weight", &[128, hidden], "BF16")];
    let p = "model.layers.0.";
    specs.push(t(format!("{p}input_layernorm.weight"), &[hidden], "BF16"));
    specs.push(t(
        format!("{p}self_attn.q_proj.weight"),
        &[heads * hd, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.k_proj.weight"),
        &[kv * hd, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.v_proj.weight"),
        &[kv * hd, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.o_proj.weight"),
        &[hidden, heads * hd],
        "BF16",
    ));
    specs.push(t(format!("{p}mlp.gate.weight"), &[experts, hidden], "BF16"));
    for expert in 0..experts {
        let e = format!("{p}mlp.experts.{expert}.");
        specs.push(t(format!("{e}gate_proj.weight"), &[inter, hidden], "BF16"));
        specs.push(t(format!("{e}up_proj.weight"), &[inter, hidden], "BF16"));
        specs.push(t(format!("{e}down_proj.weight"), &[hidden, inter], "BF16"));
    }
    specs.push(t(
        format!("{p}mlp.shared_expert.gate_proj.weight"),
        &[inter, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}mlp.shared_expert.up_proj.weight"),
        &[inter, hidden],
        "BF16",
    ));
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"qwen3_5_moe","architectures":["Qwen3_5MoeForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":{hd},
            "num_attention_heads":{heads},"num_key_value_heads":{kv},
            "intermediate_size":{inter},"vocab_size":128,
            "num_experts":{experts},"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("qwen3_5_moe", config, specs));
}

#[test]
fn nemotron_h_mamba_tp1_of_2() {
    let (hidden, heads, head_dim, group_state) = (64i64, 4i64, 8i64, 16i64);
    let intermediate = heads * head_dim;
    let conv_dim = intermediate + 2 * group_state;
    let in_rows = 2 * intermediate + 2 * group_state + heads;
    let mut specs = vec![t(
        "language_model.backbone.embed_tokens.weight",
        &[128, hidden],
        "BF16",
    )];
    let mp = "language_model.backbone.layers.0.mixer.";
    specs.push(t(format!("{mp}in_proj.weight"), &[in_rows, hidden], "BF16"));
    specs.push(t(format!("{mp}conv1d.weight"), &[conv_dim, 1, 4], "BF16"));
    specs.push(t(format!("{mp}conv1d.bias"), &[conv_dim], "BF16"));
    specs.push(t(format!("{mp}A_log"), &[heads], "F32"));
    specs.push(t(format!("{mp}D"), &[heads], "F32"));
    specs.push(t(format!("{mp}dt_bias"), &[heads], "F32"));
    specs.push(t(format!("{mp}norm.weight"), &[intermediate], "BF16"));
    specs.push(t(
        format!("{mp}out_proj.weight"),
        &[hidden, intermediate],
        "BF16",
    ));
    let ep = "language_model.backbone.layers.1.mixer.experts";
    for expert in 0..2 {
        specs.push(t(
            format!("{ep}.{expert}.up_proj.weight"),
            &[32, hidden],
            "BF16",
        ));
        specs.push(t(
            format!("{ep}.{expert}.down_proj.weight"),
            &[hidden, 32],
            "BF16",
        ));
    }
    specs.push(t(
        "language_model.backbone.norm_f.weight",
        &[hidden],
        "BF16",
    ));
    let config = format!(
        r#"{{"model_type":"nemotron_h","architectures":["NemotronHForCausalLM"],
            "num_hidden_layers":2,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":2,
            "intermediate_size":96,"vocab_size":128,
            "num_experts":2,"num_experts_per_tok":1,
            "mamba_n_groups":2,"mamba_num_heads":{heads},
            "mamba_head_dim":{head_dim},"ssm_state_size":8,"conv_kernel":4,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    let mut case = Case::new("nemotron_h_tp2", config, specs);
    case.tp = (1, 2);
    diff(&case);
}

fn kimi_family_specs(prefix: &str) -> Vec<Spec> {
    let (hidden, inter) = (64i64, 32i64);
    let mut specs = vec![t(
        format!("{prefix}model.embed_tokens.weight"),
        &[128, hidden],
        "BF16",
    )];
    let p = format!("{prefix}model.layers.0.");
    specs.push(t(
        format!("{p}self_attn.q_a_proj.weight"),
        &[32, hidden],
        "BF16",
    ));
    specs.push(t(
        format!("{p}self_attn.kv_a_proj_with_mqa.weight"),
        &[48, hidden],
        "BF16",
    ));
    for expert in 0..2 {
        let e = format!("{p}mlp.experts.{expert}.");
        for half in ["gate_proj", "up_proj"] {
            specs.push(t(
                format!("{e}{half}.weight_packed"),
                &[inter, hidden / 8],
                "I32",
            ));
            specs.push(t(
                format!("{e}{half}.weight_scale"),
                &[inter, hidden / 32],
                "BF16",
            ));
        }
        specs.push(t(
            format!("{e}down_proj.weight_packed"),
            &[hidden, inter / 8],
            "I32",
        ));
        specs.push(t(
            format!("{e}down_proj.weight_scale"),
            &[hidden, inter / 32],
            "BF16",
        ));
    }
    specs.push(t(format!("{prefix}model.norm.weight"), &[hidden], "BF16"));
    specs.push(t("lm_head.weight", &[128, hidden], "BF16"));
    specs
}

fn mla_config(model_type: &str) -> String {
    format!(
        r#"{{"model_type":"{model_type}","architectures":["X"],
            "num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":4,"num_key_value_heads":4,
            "intermediate_size":32,"vocab_size":128,
            "n_routed_experts":2,"num_experts_per_tok":2,
            "qk_nope_head_dim":8,"qk_rope_head_dim":8,"v_head_dim":8,
            "kv_lora_rank":16,"q_lora_rank":16,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    )
}

#[test]
fn kimi_k2_w4a16_stacks() {
    diff(&Case::new(
        "kimi_k2",
        mla_config("kimi_k2"),
        kimi_family_specs("language_model."),
    ));
}

#[test]
fn deepseek_v2_mla_joins() {
    diff(&Case::new(
        "deepseek_v2",
        mla_config("deepseek_v2"),
        kimi_family_specs(""),
    ));
}

#[test]
fn deepseek_v4_streamed() {
    let (hidden, inter) = (64i64, 32i64);
    let mut specs = vec![t("embed_tokens.weight", &[128, hidden], "BF16")];
    let p = "layers.0.";
    for expert in 0..2 {
        let e = format!("{p}ffn.experts.{expert}.");
        for half in ["w1", "w3"] {
            specs.push(t(format!("{e}{half}.weight"), &[inter, hidden / 2], "I8"));
            specs.push(t(format!("{e}{half}.scale"), &[inter, hidden / 32], "U8"));
        }
        specs.push(t(format!("{e}w2.weight"), &[hidden, inter / 2], "I8"));
        specs.push(t(format!("{e}w2.scale"), &[hidden, inter / 32], "U8"));
    }
    specs.push(t("norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"deepseek_v4","architectures":["DeepseekV4ForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":4,
            "intermediate_size":{inter},"vocab_size":128,
            "n_routed_experts":2,"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    let mut case = Case::new("deepseek_v4_streamed", config, specs);
    case.stream = true;
    diff(&case);
}

#[test]
fn gpt_oss_streamed() {
    let (hidden, experts, inter) = (64i64, 2i64, 64i64);
    let mut specs = Vec::new();
    let p = "model.layers.0.mlp.experts";
    specs.push(t(
        format!("{p}.gate_up_proj_blocks"),
        &[experts, 2 * inter, hidden / 32, 16],
        "U8",
    ));
    specs.push(t(
        format!("{p}.gate_up_proj_scales"),
        &[experts, 2 * inter, hidden / 32],
        "U8",
    ));
    specs.push(t(
        format!("{p}.gate_up_proj_bias"),
        &[experts, 2 * inter],
        "BF16",
    ));
    specs.push(t(
        format!("{p}.down_proj_blocks"),
        &[experts, hidden, inter / 32, 16],
        "U8",
    ));
    specs.push(t(
        format!("{p}.down_proj_scales"),
        &[experts, hidden, inter / 32],
        "U8",
    ));
    specs.push(t(format!("{p}.down_proj_bias"), &[experts, hidden], "BF16"));
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"gpt_oss","architectures":["GptOssForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":2,
            "intermediate_size":{inter},"vocab_size":128,
            "num_local_experts":{experts},"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    let mut case = Case::new("gpt_oss_streamed", config, specs);
    case.stream = true;
    diff(&case);
}

#[test]
fn gpt_oss_native_marlin() {
    let (hidden, experts, inter) = (64i64, 2i64, 64i64);
    let mut specs = Vec::new();
    let p = "model.layers.0.mlp.experts";
    specs.push(t(
        format!("{p}.gate_up_proj_blocks"),
        &[experts, 2 * inter, hidden / 32, 16],
        "U8",
    ));
    specs.push(t(
        format!("{p}.gate_up_proj_scales"),
        &[experts, 2 * inter, hidden / 32],
        "U8",
    ));
    specs.push(t(
        format!("{p}.gate_up_proj_bias"),
        &[experts, 2 * inter],
        "BF16",
    ));
    specs.push(t(
        format!("{p}.down_proj_blocks"),
        &[experts, hidden, inter / 32, 16],
        "U8",
    ));
    specs.push(t(
        format!("{p}.down_proj_scales"),
        &[experts, hidden, inter / 32],
        "U8",
    ));
    specs.push(t(format!("{p}.down_proj_bias"), &[experts, hidden], "BF16"));
    specs.push(t("model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"gpt_oss","architectures":["GptOssForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":16,
            "num_attention_heads":4,"num_key_value_heads":2,
            "intermediate_size":{inter},"vocab_size":128,
            "num_local_experts":{experts},"num_experts_per_tok":2,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    let mut case = Case::new("gpt_oss_native", config, specs);
    case.moe_request = 2;
    case.native_mxfp4_moe = true;
    diff(&case);
}

#[test]
fn csm_f32_narrowing() {
    let hidden = 32i64;
    let specs = vec![
        t("backbone.embed_tokens.weight", &[64, hidden], "F32"),
        t("backbone.layers.0.mlp.w1.weight", &[48, hidden], "F32"),
        t("backbone.norm.weight", &[hidden], "F32"),
        t("depth_decoder.weight", &[hidden, hidden], "F32"),
    ];
    let config = format!(
        r#"{{"model_type":"csm","architectures":["CsmForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},"head_dim":8,
            "num_attention_heads":4,"num_key_value_heads":4,
            "intermediate_size":48,"vocab_size":64,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("csm", config, specs));
}

#[test]
fn kimi_k3_bands_and_mxfp4_stacks() {
    let (hidden, latent, inter) = (64i64, 64i64, 32i64);
    let mut specs = vec![t(
        "language_model.model.embed_tokens.weight",
        &[128, hidden],
        "BF16",
    )];
    let p = "language_model.model.layers.0.";
    specs.push(t(format!("{p}self_attn.A_log"), &[16], "F32"));
    specs.push(t(
        format!("{p}self_attn.b_proj.weight"),
        &[8, hidden],
        "BF16",
    ));
    let moe = format!("{p}block_sparse_moe.");
    for expert in 0..2 {
        let e = format!("{moe}experts.{expert}.");
        for half in ["w1", "w3"] {
            specs.push(t(
                format!("{e}{half}.weight_packed"),
                &[inter, latent / 2],
                "U8",
            ));
            specs.push(t(
                format!("{e}{half}.weight_scale"),
                &[inter, latent / 32],
                "U8",
            ));
        }
        specs.push(t(
            format!("{e}w2.weight_packed"),
            &[latent, inter / 2],
            "U8",
        ));
        specs.push(t(
            format!("{e}w2.weight_scale"),
            &[latent, inter / 32],
            "U8",
        ));
    }
    specs.push(t("language_model.model.norm.weight", &[hidden], "BF16"));
    let config = format!(
        r#"{{"model_type":"kimi_k3","architectures":["KimiK3ForCausalLM"],
            "num_hidden_layers":1,"hidden_size":{hidden},
            "num_attention_heads":4,"num_key_value_heads":4,
            "intermediate_size":{inter},"vocab_size":128,
            "n_routed_experts":2,"num_experts_per_tok":2,
            "qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,
            "kv_lora_rank":16,"q_lora_rank":16,
            "max_position_embeddings":2048,"rope_theta":10000.0,
            "rms_norm_eps":1e-6}}"#
    );
    diff(&Case::new("kimi_k3", config, specs));
}

/// The real-checkpoint sweep: point `PIE_DIFF_SNAPSHOTS` at a colon-
/// separated list of snapshot directories and every one is authored both
/// ways from its own `config.json` and its own tensor table — no synthetic
/// shapes, the shapes HuggingFace actually ships.
#[test]
fn real_snapshots_from_env() {
    let Some(list) = std::env::var_os("PIE_DIFF_SNAPSHOTS") else {
        eprintln!("PIE_DIFF_SNAPSHOTS unset; the real-checkpoint sweep is a no-op");
        return;
    };
    for snapshot in list.to_string_lossy().split(':').filter(|s| !s.is_empty()) {
        let dir = Path::new(snapshot);
        let config_text = std::fs::read_to_string(dir.join("config.json"))
            .unwrap_or_else(|err| panic!("{snapshot}: read config.json: {err}"));
        let config: serde_json::Value = serde_json::from_str(&config_text)
            .unwrap_or_else(|err| panic!("{snapshot}: parse config.json: {err}"));
        let facts = facts_from_config(&config);
        eprintln!("== {} ({}) ==", facts.model_type, snapshot);

        let cpp: ModelContract = author_contract(&AuthorRequest {
            snapshot_dir: dir,
            runtime_quant: "",
            mxfp4_moe_request: 0,
            component: 0,
            stream_routed_experts: false,
            tp_rank: 0,
            tp_size: 1,
            fp8_native: false,
            native_mxfp4_moe: false,
        })
        .unwrap_or_else(|err| panic!("{snapshot}: C++ author failed: {err}"));

        let metadata = pie_loader::checkpoint::read::parse_checkpoint_metadata(dir)
            .unwrap_or_else(|err| panic!("{snapshot}: reading the snapshot: {err}"));
        let target = StorageTarget {
            backend: BackendKind::Cuda,
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let rust = pie_model::contract::author(&facts, &metadata, &target, &Policy::default())
            .unwrap_or_else(|err| panic!("{snapshot}: Rust author failed: {err}"))
            .unwrap_or_else(|| panic!("{snapshot}: no Rust author for {}", facts.model_type));

        let cpp_text = serde_json::to_string_pretty(&cpp).expect("serialize");
        let rust_text = serde_json::to_string_pretty(&rust).expect("serialize");
        if cpp_text != rust_text {
            let divergence = cpp_text
                .lines()
                .zip(rust_text.lines())
                .enumerate()
                .find(|(_, (c, r))| c != r)
                .map(|(line, (c, r))| format!("line {}:\n  c++ : {c}\n  rust: {r}", line + 1))
                .unwrap_or_else(|| {
                    format!(
                        "c++ has {} lines, rust has {}",
                        cpp_text.lines().count(),
                        rust_text.lines().count()
                    )
                });
            std::fs::write("/tmp/diff_cpp.json", &cpp_text).ok();
            std::fs::write("/tmp/diff_rust.json", &rust_text).ok();
            panic!(
                "{snapshot}: the two authors disagree; first divergence at {divergence}\n\
                 dumps: /tmp/diff_cpp.json /tmp/diff_rust.json"
            );
        }
        eprintln!("   {} tensors, identical", rust.tensors.len());
    }
}
