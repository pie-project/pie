//! The two family-aware offline paths, end to end on real files.
//!
//! `build` authors the serve contract, materializes it through the
//! streaming executor and the disk spool, and writes the *runtime* tensors —
//! the fused QKV banks are the proof, because no checkpoint ships one.
//! `import` on an F32 snapshot exercises the same spool from the other
//! side: the decoded set is the whole model, which is exactly the case the
//! spool exists for, and the artifact must hold the BF16 the engine reads.
//!
//! Both run against a synthetic llama snapshot written from scratch — real
//! safetensors bytes on disk, no fixtures — so the tests cover the
//! config-facts read, the authoring, the plan, the executor and the writer
//! as one path.

use std::path::Path;

use pie_loader::checkpoint::zt::{parse_checkpoint, verify_checkpoint};
use pie_loader::types::{DType, Encoding};

/// A dense llama-shaped snapshot: `config.json` plus one real safetensors
/// file of zeroed `dtype` weights.
fn write_snapshot(dir: &Path, dtype: &str) {
    let (hidden, heads, kv_heads, head_dim, intermediate, vocab) =
        (64i64, 4i64, 2i64, 16i64, 96i64, 128i64);
    let mut tensors: Vec<(String, Vec<i64>)> = vec![
        ("model.embed_tokens.weight".into(), vec![vocab, hidden]),
        ("model.norm.weight".into(), vec![hidden]),
        ("lm_head.weight".into(), vec![vocab, hidden]),
    ];
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        tensors.extend([
            (format!("{p}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{p}.self_attn.q_proj.weight"),
                vec![heads * head_dim, hidden],
            ),
            (
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_heads * head_dim, hidden],
            ),
            (
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_heads * head_dim, hidden],
            ),
            (
                format!("{p}.self_attn.o_proj.weight"),
                vec![hidden, heads * head_dim],
            ),
            (format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            (
                format!("{p}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{p}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{p}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }

    let width = match dtype {
        "BF16" | "F16" => 2u64,
        "F32" => 4u64,
        other => panic!("unsupported fixture dtype {other}"),
    };
    let mut header = String::from("{");
    let mut offset = 0u64;
    for (index, (name, shape)) in tensors.iter().enumerate() {
        let elements: i64 = shape.iter().product();
        let nbytes = elements as u64 * width;
        if index > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
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
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"llama3","architectures":["LlamaForCausalLM"],
            "num_hidden_layers":2,"head_dim":16,"num_attention_heads":4,
            "num_key_value_heads":2,"hidden_size":64,"vocab_size":128,
            "intermediate_size":96,"max_position_embeddings":2048,
            "rope_theta":10000.0,"rms_norm_eps":1e-6,
            "tie_word_embeddings":false}"#,
    )
    .expect("write config");
}

/// build writes runtime tensors: the fused banks exist, their views
/// exist, and every digest verifies.
#[test]
fn build_materializes_the_serve_contract() {
    let staging = tempfile::tempdir().expect("staging");
    write_snapshot(staging.path(), "BF16");
    let store = tempfile::tempdir().expect("store");
    let artifact = store.path().join("optimized.zt");

    pie_bin::ops::model::build::run(pie_bin::ops::model::build::BuildArgs {
        source: staging.path().to_string_lossy().into_owned(),
        quant: None,
        fp8_native: false,
        moe: None,
        // The flag's own default. This test asserts CUDA's fused-bank names
        // below, which is the layout `cuda` authors -- Metal's schemas produce
        // in-place projections under MLX names instead.
        backend: "cuda".to_string(),
        out: Some(artifact.clone()),
        dry_run: false,
    })
    .expect("build failed");

    let parsed = parse_checkpoint(&artifact).expect("parse artifact");
    let names: Vec<&str> = parsed.weights().map(|t| t.name.as_str()).collect();
    // The bank no checkpoint ships, and the per-projection views into it.
    for expected in [
        "model.layers.0.self_attn.qkv_proj.fused.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_up_proj.fused.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "lm_head.weight",
    ] {
        assert!(names.contains(&expected), "{expected} missing: {names:?}");
    }
    // The spool left nothing behind.
    assert!(
        !store.path().join("optimized.spool.tmp").exists(),
        "the spool was not cleaned up"
    );
    let verified = verify_checkpoint(&artifact).expect("digests verify");
    assert_eq!(verified as usize, parsed.tensors.len());
}

/// import on an F32 snapshot decodes *everything* — the whole-model case
/// the streaming executor and the disk spool exist for — and the artifact
/// holds BF16 under the original names.
#[test]
fn import_streams_a_fully_decoded_model_through_the_spool() {
    let staging = tempfile::tempdir().expect("staging");
    write_snapshot(staging.path(), "F32");
    let store = tempfile::tempdir().expect("store");
    let artifact = store.path().join("converted.zt");

    pie_bin::ops::model::import::run(pie_bin::ops::model::import::ImportArgs {
        source: staging.path().to_string_lossy().into_owned(),
        out: Some(artifact.clone()),
        dry_run: false,
        force: false,
        max_shard_size: None,
        delete_source: false,
    })
    .expect("import failed");

    let parsed = parse_checkpoint(&artifact).expect("parse artifact");
    for tensor in parsed.weights() {
        assert_eq!(
            tensor.encoding,
            Encoding::Raw(DType::BF16),
            "{} was not normalized to BF16",
            tensor.name
        );
    }
    assert_eq!(parsed.weights().count(), 21, "every tensor came through");
    assert!(
        !store.path().join("converted.spool.tmp").exists(),
        "the spool was not cleaned up"
    );
    let verified = verify_checkpoint(&artifact).expect("digests verify");
    assert_eq!(verified as usize, parsed.tensors.len());
}
