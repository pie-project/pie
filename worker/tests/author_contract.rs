//! The author-only FFI, end to end: the CUDA driver's family hook authors a
//! contract for a synthetic checkpoint, the view crosses back into Rust, and
//! the loader compiles it into a plan — all without touching a device.
//!
//! The checkpoint is the smallest llama the family hook accepts, written by
//! hand: real HF names, consistent shapes, zero-filled BF16 payloads. What
//! the test pins is the seam, not the family — that authoring runs outside a
//! load, that the marshaled contract reads back, and that what reads back is
//! a contract the compiler accepts against the same checkpoint.

#![cfg(feature = "driver-cuda")]

use std::path::Path;

use pie_worker::contract_author::{AuthorRequest, author_contract};

fn write_tiny_llama(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 16,
  "vocab_size": 256,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "max_position_embeddings": 2048,
  "torch_dtype": "bfloat16",
  "tie_word_embeddings": false
}"#,
    )
    .unwrap();

    let mut tensors: Vec<(String, Vec<i64>)> = vec![
        ("model.embed_tokens.weight".into(), vec![256, 64]),
        ("model.norm.weight".into(), vec![64]),
        ("lm_head.weight".into(), vec![256, 64]),
    ];
    for layer in 0..2 {
        let at = |name: &str| format!("model.layers.{layer}.{name}");
        tensors.extend([
            (at("input_layernorm.weight"), vec![64]),
            (at("self_attn.q_proj.weight"), vec![64, 64]),
            (at("self_attn.k_proj.weight"), vec![32, 64]),
            (at("self_attn.v_proj.weight"), vec![32, 64]),
            (at("self_attn.o_proj.weight"), vec![64, 64]),
            (at("post_attention_layernorm.weight"), vec![64]),
            (at("mlp.gate_proj.weight"), vec![128, 64]),
            (at("mlp.up_proj.weight"), vec![128, 64]),
            (at("mlp.down_proj.weight"), vec![64, 128]),
        ]);
    }

    let mut header = String::from("{");
    let mut at = 0u64;
    for (index, (name, shape)) in tensors.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let bytes: i64 = shape.iter().product::<i64>() * 2;
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":{shape:?},\"data_offsets\":[{at},{}]}}",
            at + bytes as u64
        ));
        at += bytes as u64;
    }
    header.push('}');

    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + at as usize, 0);
    std::fs::write(dir.join("model.safetensors"), &file).unwrap();
}

#[test]
fn the_driver_authors_a_contract_without_a_device() {
    let dir = std::env::temp_dir().join(format!("pie_author_ffi_{}", std::process::id()));
    write_tiny_llama(&dir);

    let contract = author_contract(&AuthorRequest {
        snapshot_dir: &dir,
        runtime_quant: "",
        mxfp4_moe_request: 0,
        component: 0,
        stream_routed_experts: false,
        tp_rank: 0,
        tp_size: 1,
        fp8_native: false,
        native_mxfp4_moe: false,
    })
    .expect("the family hook authors against this checkpoint");
    assert!(
        !contract.tensors.is_empty(),
        "an authored contract declares tensors"
    );

    // The strongest statement available without a device: the contract the
    // driver wrote compiles against the same checkpoint it was written from.
    let metadata = pie_loader::checkpoint::read::parse_checkpoint_metadata(&dir).unwrap();
    let target = pie_loader::plan::StorageTarget {
        backend: pie_loader::types::BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        block_scale_rows: 128,
        ..pie_loader::plan::StorageTarget::default()
    };
    let plan = pie_loader::plan::compile(&metadata, &contract, target)
        .expect("the authored contract compiles");
    assert!(!plan.instrs.is_empty());
    std::fs::remove_dir_all(dir).ok();
}
