//! The two family-aware offline paths, end to end on real files.
//!
//! `build` authors the serve contract, materializes it through the streaming
//! executor and the disk spool, and writes the *runtime* tensors under the
//! backend's naming. It used to write fused QKV banks too, and those were what
//! proved a rewrite had happened; `build_policy` authors `Projections::InPlace`
//! for every backend now -- a fusion is a view, and a file has no pointers --
//! so the banks are asserted ABSENT here and the rewrite is proved from the
//! artifact's own metadata instead.
//! `import` on an F32 snapshot exercises the same spool from the other
//! side: the decoded set is the whole model, which is exactly the case the
//! spool exists for, and the artifact must hold the BF16 the engine reads.
//!
//! Both run against a synthetic llama snapshot written from scratch — real
//! safetensors bytes on disk, no fixtures — so the tests cover the
//! config-facts read, the authoring, the plan, the executor and the writer
//! as one path.

use std::path::Path;

use model_loader::checkpoint::zt::{parse_checkpoint, verify_checkpoint};
use model_loader::types::{DType, Encoding};

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

/// build writes the serve contract's tensors, and NOT the fused banks.
///
/// # It used to assert the banks, and that was the old contract
///
/// This test named `model.layers.0.self_attn.qkv_proj.fused.weight` and
/// `model.layers.1.mlp.gate_up_proj.fused.weight` and required them present,
/// on the reading that a build emits runtime layout and CUDA's runtime layout
/// is fused. `build_policy` in `src/ops/model/build.rs` authors
/// `Projections::InPlace` for every backend now, and its doc is where the
/// reasoning lives -- the short version being that a fusion is a VIEW, so
/// persisting it persists the bank AND the projections that alias it, which
/// on Qwen3-0.6B was 56 extra tensors, 587 MB of file and 560 MiB of resident
/// VRAM for a concatenation measured at no difference.
///
/// So the assertion inverts rather than relaxes. The projections must be
/// there, under their checkpoint names, because that is what the load path
/// joins from -- and the banks must be ABSENT, because their presence is the
/// defect. Requiring the absence is what makes this a gate rather than a
/// deletion: a backend that starts persisting a fusion again fails here and
/// says so, which is what the old assertion did in the other direction.
#[test]
fn build_materializes_the_serve_contract() {
    let staging = tempfile::tempdir().expect("staging");
    write_snapshot(staging.path(), "BF16");
    let store = tempfile::tempdir().expect("store");
    let artifact = store.path().join("optimized.zt");

    pie::ops::model::build::run(pie::ops::model::build::BuildArgs {
        source: staging.path().to_string_lossy().into_owned(),
        quant: None,
        fp8_native: false,
        moe: None,
        // The flag's own default. `cuda` is what selects HF naming below --
        // Metal's and Vulkan's schemas rename for MLX's binder instead. It no
        // longer selects a fused layout, because no backend's BUILD does.
        backend: "cuda".to_string(),
        out: Some(artifact.clone()),
        dry_run: false,
        as_id: None,
    })
    .expect("build failed");

    let parsed = parse_checkpoint(&artifact).expect("parse artifact");
    let names: Vec<&str> = parsed.weights().map(|t| t.name.as_str()).collect();
    // The projections the load path joins from, under HF names, plus the
    // untied head this row states so that the tied path is not the only one a
    // cheap test covers.
    for expected in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "lm_head.weight",
    ] {
        assert!(names.contains(&expected), "{expected} missing: {names:?}");
    }
    // And the banks are NOT here. `build_policy`'s rule stated as a test: a
    // build must not materialize a tensor that another materialized tensor is
    // a view of.
    for bank in names.iter().filter(|n| n.contains(".fused.")) {
        panic!(
            "`{bank}` is a fused bank the projections alias, and a file has no \
             pointers -- persisting it writes the same bytes twice. See \
             `build_policy` in src/ops/model/build.rs. All of: {names:?}"
        );
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

    pie::ops::model::import::run(pie::ops::model::import::ImportArgs {
        source: staging.path().to_string_lossy().into_owned(),
        out: Some(artifact.clone()),
        dry_run: false,
        force: false,
        max_shard_size: None,
        delete_source: false,
        consume_source: false,
        keep_source: false,
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

/// A BUILT artifact is still the model it was built from.
///
/// The open question this settles. `build` rewrites a checkpoint into runtime
/// layout, so the worry was that the rewrite destroys the thing identity is
/// matched against, leaving a built artifact that can never be identified and
/// therefore never served without an operator naming it.
///
/// It does not, and the reason is structural rather than lucky. A manifest
/// states the rows a MODEL has and `Manifest::check` walks only those, so a
/// tensor the model does not mention cannot fault; and every row it does ask
/// about is a checkpoint-named projection that the rewrite leaves at its
/// checkpoint extents.
///
/// # What the rewrite is, now that it is not a fusion
///
/// This test used to prove it was looking at a REWRITTEN artifact by finding
/// `qkv_proj.fused.weight` in it -- "the bank is the proof, as it is next
/// door". `build_policy` authors `Projections::InPlace` for every backend now
/// and no bank is written, so that proof is gone and the test would otherwise
/// have become vacuous: a plain copy of the snapshot would pass it.
///
/// The replacement is the artifact's own header. A `.zt` built by this command
/// carries a `pie.model/1` descriptor and a source-encoding summary that a
/// converted snapshot does not, so "this went through `build`" is asked of the
/// metadata rather than inferred from a tensor name -- which is the more
/// direct question anyway, and one that does not move when a layout decision
/// does.
///
/// If a future backend fused DESTRUCTIVELY -- writing a bank and dropping the
/// projections -- this test is what fails, and the fix would be for `build` to
/// write the id it identified into the artifact rather than for identification
/// to learn about banks.
#[test]
fn a_built_artifact_still_identifies_as_the_row_it_was_built_from() {
    let staging = tempfile::tempdir().expect("staging");
    write_snapshot(staging.path(), "BF16");
    let store = tempfile::tempdir().expect("store");
    let artifact = store.path().join("optimized.zt");

    pie::ops::model::build::run(pie::ops::model::build::BuildArgs {
        source: staging.path().to_string_lossy().into_owned(),
        quant: None,
        fp8_native: false,
        moe: None,
        backend: "cuda".to_string(),
        out: Some(artifact.clone()),
        dry_run: false,
        as_id: None,
    })
    .expect("build failed");

    let meta = parse_checkpoint(&artifact).expect("parse artifact");
    // Not vacuous: this is the artifact `build` wrote, not the snapshot it was
    // given. The snapshot is safetensors with no `__meta__/` namespace at all,
    // so a metadata object existing is the rewrite having happened.
    let objects: Vec<&str> = meta.meta_objects().map(|o| o.name.as_str()).collect();
    assert!(
        !objects.is_empty(),
        "the artifact carries no `__meta__/` objects, so this is not something \
         `build` wrote and the test asserts nothing about the rewrite"
    );
    let row = model::catalog::identify(&meta, &model::catalog::Override::None)
        .expect("a built artifact is still identifiable");
    assert_eq!(
        row.id(),
        "test-tiny-llama",
        "the rewrite changed which model the artifact is"
    );
}
