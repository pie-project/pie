//! The family-blind offline path, end to end on real files.
//!
//! `import` on an F32 snapshot exercises the streaming executor and the disk
//! spool from the side they exist for: the decoded set is the whole model,
//! and the artifact must hold the BF16 the runtime reads.
//!
//! `build_materializes_the_serve_contract` and
//! `a_built_artifact_still_identifies_as_the_row_it_was_built_from` STOOD
//! HERE. Both exercised `pie model build`, which authored a
//! `model_legacy::contract` and materialized it offline; R3 deleted the
//! command with the contract, so their subject is gone. Nothing replaces
//! them, and that is the point: an engine produces its weights from the
//! checkpoint through the SKU's own import table at load, so there is no
//! offline rewrite left to prove anything about.
//!
//! The run below goes against a synthetic llama snapshot written from
//! scratch — real safetensors bytes on disk, no fixtures — so it covers the
//! plan, the executor and the writer as one path.

use std::path::Path;

use checkpoint::file::zt;
use checkpoint::types::{DType, Encoding};

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

// `build_materializes_the_serve_contract` STOOD HERE, and R3 took the
// test at `6393b8ddb` while leaving these twenty lines of its doc
// attached to the test below it. That is the defect clippy's
// `empty_line_after_doc_comment` was pointing at: a deletion that took
// an item and left its prose to glue itself onto the next one.
//
// WHAT IT ESTABLISHED, kept because the measurement is not recoverable
// from anything left in the tree:
// build writes the serve contract's tensors, and NOT the fused banks.
//
// # It used to assert the banks, and that was the old contract
//
// This test named `model.layers.0.self_attn.qkv_proj.fused.weight` and
// `model.layers.1.mlp.gate_up_proj.fused.weight` and required them present,
// on the reading that a build emits runtime layout and CUDA's runtime layout
// is fused. `build_policy` in src/ops/model/build.rs, deleted at R3 authors
// `Projections::InPlace` for every backend now, and its doc is where the
// reasoning lives -- the short version being that a fusion is a VIEW, so
// persisting it persists the bank AND the projections that alias it, which
// on Qwen3-0.6B was 56 extra tensors, 587 MB of file and 560 MiB of resident
// VRAM for a concatenation measured at no difference.
//
// So the assertion inverts rather than relaxes. The projections must be
// there, under their checkpoint names, because that is what the load path
// joins from -- and the banks must be ABSENT, because their presence is the
// defect. Requiring the absence is what makes this a gate rather than a
// deletion: a backend that starts persisting a fusion again fails here and
// says so, which is what the old assertion did in the other direction.

/// import on an F32 snapshot decodes *everything* — the whole-model case
/// the streaming executor and the disk spool exist for — and the artifact
/// holds BF16 under the original names.
///
/// **"THE ORIGINAL NAMES" IS NOW A CLAIM WITH A CONDITION ON IT** (§M-4a).
/// Since the promotion gate widened past `Cast{Quant}`/`Repack` to the whole
/// landing, a source some SKU in this build CLAIMS comes out under that SKU's
/// plane names — the fusion, the band, the rename and the fold are performed
/// here, so the artifact holds `layer.0.qg_proj` where the checkpoint held
/// `model.layers.0.self_attn.q_proj.weight`. This fixture is two layers of
/// 64 hidden over a 128-token vocabulary and no row in `models::catalog()`
/// has that geometry, so `conversion_contract` answers `None`, nothing is
/// promoted, and every name below is the source's own.
///
/// That is asserted rather than assumed, because it is exactly the shape of
/// gate that passes for the wrong reason: a fixture that quietly started
/// matching a SKU would keep the count at 21 and change every name, and the
/// encoding sweep above would not notice. The day a row claims this snapshot,
/// this assertion is what says so.
///
/// **AND `no_prepare: true` IS LOAD-BEARING NOW, NOT SCENERY** (§M-4g). An
/// import that is going to prepare the artifact for THIS box refuses a source
/// no SKU in this build claims, because `prepare` swallows its own refusals
/// and so cannot be the party that says nothing will ever serve the file.
/// `--no-prepare` is the path that legitimately converts one — a shelf
/// converted for machines whose build ships the row — and that is the path
/// this test is on. The line below it says the artifact is not served here; it
/// is now also what says the conversion is allowed to happen at all. The other
/// side of the same fact is
/// `an_import_that_will_prepare_refuses_a_source_no_sku_claims`.
#[test]
fn import_streams_a_fully_decoded_model_through_the_spool() {
    let staging = tempfile::tempdir().expect("staging");
    write_snapshot(staging.path(), "F32");
    let store = tempfile::tempdir().expect("store");
    let artifact = store.path().join("converted.zt");

    // The prepare arm reads the box's config through these; `no_prepare`
    // below keeps this import conversion-only, so bare defaults suffice.
    let global = bootstrap::GlobalArgs {
        config: None,
        log_level: "info".into(),
        metrics_addr: None,
    };
    pie::ops::model::import::run(pie::ops::model::import::ImportArgs {
        source: staging.path().to_string_lossy().into_owned(),
        aux: None,
        out: Some(artifact.clone()),
        dry_run: false,
        force: false,
        max_shard_size: None,
        delete_source: false,
        consume_source: false,
        keep_source: false,
        // The artifact is parsed below, not served: baking a tier for it
        // would want an engine and a config this test rightly has neither of.
        no_prepare: true,
        // And this is the CONVERSION half of the command, which is the half
        // §M-3's rebuild door skips — `--prepare-only` returns before the
        // first line of what this test is about.
        prepare_only: false,
    }, &global)
    .expect("import failed");

    let parsed = zt::parse(&artifact).expect("parse artifact");
    for tensor in parsed.weights() {
        assert_eq!(
            tensor.encoding,
            Encoding::Raw(DType::Bf16),
            "{} was not normalized to BF16",
            tensor.name
        );
    }
    assert_eq!(parsed.weights().count(), 21, "every tensor came through");
    let names: Vec<&str> = parsed.weights().map(|t| t.name.as_str()).collect();
    for expected in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ] {
        assert!(
            names.contains(&expected),
            "no SKU claims this snapshot, so `{expected}` is still spelled the \
             source's way; the artifact holds {names:?}"
        );
    }
    assert!(
        !store.path().join("converted.spool.tmp").exists(),
        "the spool was not cleaned up"
    );
    let verified = zt::verify(&artifact).expect("digests verify");
    assert_eq!(verified as usize, parsed.tensors.len());
}

/// **THE SAME SNAPSHOT, WITHOUT `--no-prepare`, IS A REFUSAL** (§M-4g).
///
/// The fixture above converts a checkpoint no catalog row claims and is right
/// to: it passes `no_prepare`, which is the operator saying this box is not
/// making the model servable. Drop that one flag and the command is promising
/// something it cannot deliver — since §M-4a the import runs the SKU's own
/// contract to produce the planes a serving load binds, so with no SKU there
/// are no planes, and the artifact it would write is one no serve boot and no
/// `--prepare-only` on any box with this catalog can open.
///
/// The two tests are one property read from both sides, which is what keeps
/// the green one green for a reason rather than by luck: the discriminator is
/// `--no-prepare` and nothing else — not `--out` (both pass one; a `[model]`
/// key takes a path as readily as a store name), not the geometry, not the
/// destination.
#[test]
fn an_import_that_will_prepare_refuses_a_source_no_sku_claims() {
    let staging = tempfile::tempdir().expect("staging");
    write_snapshot(staging.path(), "F32");
    let store = tempfile::tempdir().expect("store");
    let artifact = store.path().join("converted.zt");

    let global = bootstrap::GlobalArgs {
        config: None,
        log_level: "info".into(),
        metrics_addr: None,
    };
    // `Answer` states an outcome for the CLI to render and carries no `Debug`,
    // so the Ok arm is named rather than unwrapped through one.
    let answered = pie::ops::model::import::run(
        pie::ops::model::import::ImportArgs {
            source: staging.path().to_string_lossy().into_owned(),
            aux: None,
            out: Some(artifact.clone()),
            dry_run: false,
            force: false,
            max_shard_size: None,
            delete_source: false,
            consume_source: false,
            keep_source: false,
            // The one difference from the test above, and the whole of it.
            no_prepare: false,
            prepare_only: false,
        },
        &global,
    );
    let Err(why) = answered else {
        panic!("an import that will prepare wrote an artifact nothing in this build can serve");
    };

    let said = format!("{why:#}");
    for expected in ["no SKU this build ships claims", "--no-prepare"] {
        assert!(
            said.contains(expected),
            "the refusal has to name the source, the fact and the fix; it said {said:?}"
        );
    }
    assert!(
        !artifact.exists(),
        "the refusal has to come before the write, not after it"
    );
}
