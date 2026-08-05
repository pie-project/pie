//! Every fact an author reads comes out of the descriptor.
//!
//! `ModelFacts` is the whole of what an authoring call knows beyond the
//! checkpoint's own tensor table, and since the request stopped carrying it
//! field by field, [`ModelFacts::from_descriptor`] is the only way to obtain
//! one. So the property worth pinning is *coverage*: a field that the
//! descriptor does not supply is a field silently stuck at its default, and
//! the defaults are load-bearing — `quant_bits` of 0 means "4 bits" to
//! `push_mlx_affine_declared`, and `num_kv_shared_layers` of 0 means "declare
//! every layer's k/v".
//!
//! The test is written so that adding a field to `ModelFacts` breaks it:
//! the expected value is spelled out per field in a struct literal, which
//! does not compile when a field is added.

#![cfg(feature = "contract")]

use pie_model_common::facts::ModelFacts;

/// A descriptor with a distinct, non-default value in every field the
/// projection reads. Distinct so a mapping that crosses two fields shows up
/// as a swap rather than as two equal numbers.
const FULL: &str = r#"{
    "version": "pie.model/1",
    "model_type": "nemotron_h",
    "quant_method": "awq",
    "num_hidden_layers": 41,
    "num_experts": 43,
    "head_dim": 47,
    "mamba_n_groups": 53,
    "tie_word_embeddings": false,
    "quant_bits": 8,
    "quant_group_size": 59,
    "num_kv_shared_layers": 61
}"#;

#[test]
fn every_fact_is_read_from_the_descriptor() {
    let facts = ModelFacts::from_descriptor(FULL.as_bytes()).expect("a valid descriptor");
    // A struct literal, not field-by-field asserts: a new `ModelFacts` field
    // fails to compile here, which is the point. An author that starts
    // reading a fact the descriptor never supplies is the failure this
    // prevents, and it is invisible at runtime — the field just holds its
    // default.
    assert_eq!(
        facts,
        ModelFacts {
            model_type: "nemotron_h".to_string(),
            quant_method: "awq".to_string(),
            num_hidden_layers: 41,
            num_experts: 43,
            head_dim: 47,
            mamba_groups: 53,
            tied_embeddings: false,
            quant_bits: 8,
            quant_group_size: 59,
            num_kv_shared_layers: 61,
        }
    );
}

/// A descriptor written by an older pie leaves new fields at their defaults
/// rather than failing: the version is the schema's, and the schema only
/// grows.
#[test]
fn a_sparse_descriptor_falls_back_to_defaults() {
    let sparse = r#"{"version":"pie.model/1","model_type":"llama"}"#;
    let facts = ModelFacts::from_descriptor(sparse.as_bytes()).expect("a valid descriptor");
    assert_eq!(facts.model_type, "llama");
    assert_eq!(facts.num_hidden_layers, 0);
    // The one non-zero default, and it is not zero-by-accident: HF defaults
    // `tie_word_embeddings` to true, so a config that omits it means true.
    assert!(facts.tied_embeddings);
    assert_eq!(
        ModelFacts {
            model_type: "llama".to_string(),
            ..ModelFacts::default()
        },
        facts
    );
}

/// A negative in an `i32` schema field is "unset", not a count.
///
/// `head_dim` is `-1` in the descriptor for every family that does not state
/// one, because the C++ struct the schema is generated from spells absence
/// that way. Read as a `u32` it would become 4294967295 and reach a shard
/// rule as a head dimension.
#[test]
fn an_unset_i32_field_does_not_become_a_huge_u32() {
    let unset = r#"{"version":"pie.model/1","model_type":"llama","head_dim":-1}"#;
    let facts = ModelFacts::from_descriptor(unset.as_bytes()).expect("a valid descriptor");
    assert_eq!(facts.head_dim, 0, "a -1 head_dim leaked through as a count");
}

/// The version is checked, because a newer one may have changed what a name
/// means — and a fact read under the wrong meaning is exactly the failure the
/// descriptor exists to prevent.
#[test]
fn a_descriptor_of_another_version_is_refused() {
    for doc in [
        r#"{"version":"pie.model/2","model_type":"llama"}"#,
        r#"{"model_type":"llama"}"#,
        "not json at all",
    ] {
        let err = ModelFacts::from_descriptor(doc.as_bytes())
            .expect_err("accepted a descriptor it cannot vouch for");
        let text = err.to_string();
        assert!(
            text.contains("descriptor"),
            "the error does not say what was wrong: {text}"
        );
    }
}
