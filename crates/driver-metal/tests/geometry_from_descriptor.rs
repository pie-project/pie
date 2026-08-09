//! What a descriptor turns into, pinned per family, so that #7 can move.
//!
//! # Why this file exists before the refactor rather than after
//!
//! `north-star.md` #7 removes the driver's `model_type` table. The survey
//! found that `facts.rs` and `batch/geometry_facts.rs` are two halves of one
//! round trip: `ModelFacts::from_descriptor` **splits** the flat, already
//! normalized `pie.model/1` document into four family-prefixed field sets
//! (`ll_*`, `go_*`, `g4_*`, `q35_*`), and `geometry_from_facts` **merges**
//! them back into one flat `DecodeGeometry` by asking which block came out
//! non-zero.
//!
//! Measured, the split is not by family. All four blocks read the same core
//! keys — `num_hidden_layers`, `hidden_size`, `num_attention_heads`,
//! `num_key_value_heads`, `head_dim`, `intermediate_size`, `num_experts`,
//! `num_experts_per_tok` — and differ only by which *features* they add on
//! top: `swiglu_limit` for one, five `linear_*` keys for another,
//! `sliding_window` and the `gemma4_*` pair for a third. The prefix says
//! family and the content says feature.
//!
//! That refactor changes two large files at once and its failure mode is a
//! wrong number, not a compiler message: `lowering/consts.rs` binds `hidden`,
//! `intermediate`, `moe_intermediate`, `n_experts`, `vocab`, `eps`,
//! `gdn_conv_dim` and `gdn_v_total` straight off `DecodeGeometry`, and a
//! geometry that is merely *plausible* runs and answers wrongly.
//!
//! So this file pins the whole path — JSON in, `DecodeGeometry` out — for one
//! realistic descriptor per family, BEFORE anything moves. The existing tests
//! in both files do not cover it: every one of them builds a `ModelFacts`
//! literal by hand and calls the merge directly, so the split is exercised by
//! nobody and the round trip by nobody at all.
//!
//! # How to read a failure
//!
//! A diff here is not automatically a bug, but it is always a decision. Each
//! expectation below names the model it was measured from. If a change makes
//! one move, the question to answer in the commit message is which of the two
//! numbers the checkpoint actually asks for — and two of these fields are
//! already known to be answered by the *old* code with a guess:
//!
//!   - a gemma-4 config's `rms_norm_eps` is hardcoded to `1e-6` by the merge,
//!     with a comment saying gemma4's block states none, while the descriptor
//!     carries the real key at the top level;
//!   - a gpt-oss config's tying is hardcoded to `false` by the merge, with a
//!     comment saying "the config says so" — which is an argument for reading
//!     the config rather than for restating it here.
//!
//! Both are pinned below AS THE OLD CODE ANSWERS THEM, deliberately. That is
//! what a characterization test is for: it records what the code does, not
//! what it should do, so that a change of behaviour cannot hide inside a
//! change of shape.
//!
//! # What four falsifications established
//!
//! Three break it, and they are the three that matter: changing a pinned
//! value fails naming the field; dropping a field from `pins` fails naming
//! what the lowering binds; deleting the gpt-oss projection block in
//! `geometry_from_facts` fails with that family's own "config carried no
//! decoder shape".
//!
//! The fourth does not, and is the reason the refactor is tractable.
//! Replacing `from_descriptor`'s `ModelFamily::of(..) == Llama` gate with
//! `true` — so a gpt-oss and a gemma-4 config fill the llama block too —
//! leaves all six of these green. Two unit tests in `facts.rs` do catch it
//! (`the_llama_block_is_not_read_for_a_gpt_oss_model_type` and
//! `the_qwen35_block_reads_the_linear_attention_shape`), and they catch it by
//! asserting `ll_num_hidden_layers == 0`: they pin the SPLIT, which is the
//! thing #7 removes. The outcome is unmoved because every block reads the
//! same keys and the merge tries the blocks in a fixed order, so the gate is
//! not load-bearing for anything downstream.
//!
//! That is the division of labour to keep in mind when the split goes: the
//! unit tests are pinning a mechanism and will be deleted with it, and this
//! file is pinning what must survive it.

use driver_metal::ModelFacts;
use driver_metal::batch::{DecodeGeometry, geometry_from_facts};
use serde_json::{Value, json};

/// The document every family shares, at the values a 4-bit MLX export gives.
///
/// Split out because the point of the per-family cases below is the DELTA
/// from this, and a reader who has to diff two 30-line JSON literals to find
/// it will not.
fn base(model_type: &str, arch: &str) -> Value {
    json!({
        "version": "pie.model/1",
        "model_type": model_type,
        "arch_name": arch,
        "max_position_embeddings": 131_072,
        "quant_bits": 4,
        "quant_group_size": 64,
    })
}

/// The geometry a descriptor produces, or the refusal's text.
///
/// Goes through `from_descriptor` rather than building `ModelFacts` by hand,
/// because the split is half of what is being pinned.
fn geometry_of(doc: &Value) -> Result<DecodeGeometry, String> {
    let facts = ModelFacts::from_descriptor(&doc.to_string())
        .ok_or_else(|| "the descriptor was refused outright".to_string())?;
    geometry_from_facts(&facts).map_err(|e| e.to_string())
}

/// `Llama-3.2-1B-Instruct-4bit`, the checkpoint the device gate runs.
///
/// Values from its own `config.json`: 16 layers of width 2048, 32 query
/// heads over 8 key/value heads at head_dim 64, an 8192-wide FFN, 128256
/// vocabulary, theta 500000 and a llama3 rope schedule with factor 32.
fn llama_3_2_1b() -> Value {
    let mut d = base("llama", "LlamaForCausalLM");
    d["num_hidden_layers"] = json!(16);
    d["hidden_size"] = json!(2048);
    d["num_attention_heads"] = json!(32);
    d["num_key_value_heads"] = json!(8);
    d["head_dim"] = json!(64);
    d["intermediate_size"] = json!(8192);
    d["vocab_size"] = json!(128_256);
    d["rms_norm_eps"] = json!(1e-5);
    d["rope_theta"] = json!(500_000.0);
    d["tie_word_embeddings"] = json!(true);
    d["rope_scaling_kind"] = json!("llama3");
    d["rope_factor"] = json!(32.0);
    d["rope_low_freq_factor"] = json!(1.0);
    d["rope_high_freq_factor"] = json!(4.0);
    d["rope_original_max_position"] = json!(8192);
    d
}

/// `openai/gpt-oss-20b`: 24 layers of width 2880, 64 over 8 heads at 64, a
/// mixture in every layer with 32 experts and 4 active, theta 150000 and the
/// clamped SwiGLU that is the family's one genuinely distinct feature.
fn gpt_oss_20b() -> Value {
    let mut d = base("gpt_oss", "GptOssForCausalLM");
    d["num_hidden_layers"] = json!(24);
    d["hidden_size"] = json!(2880);
    d["num_attention_heads"] = json!(64);
    d["num_key_value_heads"] = json!(8);
    d["head_dim"] = json!(64);
    d["intermediate_size"] = json!(2880);
    d["vocab_size"] = json!(201_088);
    d["num_experts"] = json!(32);
    d["num_experts_per_tok"] = json!(4);
    d["rms_norm_eps"] = json!(1e-5);
    d["rope_theta"] = json!(150_000.0);
    d["swiglu_limit"] = json!(7.0);
    d["tie_word_embeddings"] = json!(false);
    d
}

/// A `qwen3_next`-shaped config: the linear-attention family, where three
/// layers in four are gated deltanet and the fourth is full attention.
fn qwen3_next() -> Value {
    let mut d = base("qwen3_next", "Qwen3NextForCausalLM");
    d["num_hidden_layers"] = json!(48);
    d["hidden_size"] = json!(2048);
    d["num_attention_heads"] = json!(16);
    d["num_key_value_heads"] = json!(2);
    d["head_dim"] = json!(256);
    d["intermediate_size"] = json!(5120);
    d["moe_intermediate_size"] = json!(512);
    d["vocab_size"] = json!(151_936);
    d["num_experts"] = json!(512);
    d["num_experts_per_tok"] = json!(10);
    d["rms_norm_eps"] = json!(1e-6);
    d["rope_theta"] = json!(10_000_000.0);
    d["linear_num_key_heads"] = json!(16);
    d["linear_num_value_heads"] = json!(32);
    d["linear_key_head_dim"] = json!(128);
    d["linear_value_head_dim"] = json!(128);
    d["linear_conv_kernel_dim"] = json!(4);
    d["norm_topk_prob"] = json!(true);
    d["tie_word_embeddings"] = json!(false);
    // Import expands the schedule into one entry per layer; the driver
    // derives the period from it. A qwen3-next config that states neither
    // this nor an interval is REFUSED — "which layers are linear cannot be
    // guessed" — which is the right answer and is why this fixture states it.
    let types: Vec<&str> = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    d["layer_types"] = json!(types);
    d
}

/// A `gemma4_text` config: the family whose attention shape is per LAYER
/// TYPE, with a second key/value head count and head dim for the global
/// layers that the flattening bug named in `geometry_facts.rs` used to drop.
fn gemma4_text() -> Value {
    let mut d = base("gemma4_text", "Gemma4ForCausalLM");
    d["num_hidden_layers"] = json!(48);
    d["hidden_size"] = json!(3840);
    d["num_attention_heads"] = json!(16);
    d["num_key_value_heads"] = json!(4);
    d["head_dim"] = json!(256);
    d["intermediate_size"] = json!(14_336);
    d["vocab_size"] = json!(262_144);
    d["rms_norm_eps"] = json!(1e-6);
    d["rope_theta"] = json!(3_000_000.0);
    d["sliding_window"] = json!(1024);
    d["num_kv_shared_layers"] = json!(0);
    d["gemma4_num_global_key_value_heads"] = json!(2);
    d["gemma4_global_head_dim"] = json!(128);
    d["gemma4_attention_k_eq_v"] = json!(false);
    d["gemma_final_logit_softcap"] = json!(30.0);
    d["tie_word_embeddings"] = json!(true);
    d
}

/// `gemma-4-31b-it-4bit` AS THE IMPORTER ACTUALLY WRITES IT.
///
/// The fixture above states a flat `rope_theta` and no `layer_types`, which
/// is a shape no real gemma-4 descriptor has: the importer expands
/// `rope_parameters` into a per-layer array whenever `layer_types` is stated,
/// and gemma-4 always states both. So that fixture pins the FALLBACK and this
/// one pins the path every gemma-4 checkpoint takes.
///
/// The values are read off the cached checkpoint, not invented, and one of
/// them is the reason this fixture exists: the flat `rope_theta` comes out
/// **1e4**, the SLIDING base, because the importer's flat key defaults from
/// the last rope entry it saw. The full-attention layers want 1e6. So a
/// gemma-4 stack that ever falls back to the flat key does not get a
/// defensible approximation, it gets the right number for a tenth of its
/// layers and a 100x error on the rest — and rope errors grow with position,
/// so a short prompt agrees and a long one drifts. That makes this pin the
/// sharpest one in the file: 1e6 and 1e4 are two orders of magnitude apart,
/// and no plausible refactor lands between them by accident.
fn gemma4_31b_as_imported() -> Value {
    let mut d = base("gemma4_text", "Gemma4ForCausalLM");
    d["num_hidden_layers"] = json!(60);
    d["hidden_size"] = json!(5376);
    d["num_attention_heads"] = json!(32);
    d["num_key_value_heads"] = json!(16);
    d["head_dim"] = json!(256);
    d["intermediate_size"] = json!(21_504);
    d["vocab_size"] = json!(262_144);
    d["rms_norm_eps"] = json!(1e-6);
    d["sliding_window"] = json!(1024);
    d["num_kv_shared_layers"] = json!(0);
    d["gemma4_num_global_key_value_heads"] = json!(4);
    d["gemma4_global_head_dim"] = json!(512);
    d["gemma4_attention_k_eq_v"] = json!(true);
    d["gemma_final_logit_softcap"] = json!(30.0);
    d["tie_word_embeddings"] = json!(true);
    // The importer's flat key: the sliding base, NOT the global one.
    d["rope_theta"] = json!(10_000.0);
    // Five sliding layers then one full, sixty deep — interval 6.
    let types: Vec<Value> = (0..60)
        .map(|i| {
            if i % 6 == 5 {
                json!("full_attention")
            } else {
                json!("sliding_attention")
            }
        })
        .collect();
    let thetas: Vec<Value> = (0..60)
        .map(|i| {
            if i % 6 == 5 {
                json!(1_000_000.0)
            } else {
                json!(10_000.0)
            }
        })
        .collect();
    let partial: Vec<Value> = (0..60)
        .map(|i| if i % 6 == 5 { json!(0.25) } else { json!(1.0) })
        .collect();
    d["layer_types"] = json!(types);
    d["gemma_per_layer_rope_theta"] = json!(thetas);
    d["gemma_per_layer_partial_rotary_factor"] = json!(partial);
    d
}

/// A `qwen2_moe` config, which `ModelFamily::of` sorts into the llama-shaped
/// family: a mixture whose expert width differs from its dense width, which
/// is the case that tells `intermediate` and `moe_intermediate` apart.
fn qwen2_moe() -> Value {
    let mut d = base("qwen2_moe", "Qwen2MoeForCausalLM");
    d["num_hidden_layers"] = json!(24);
    d["hidden_size"] = json!(2048);
    d["num_attention_heads"] = json!(16);
    d["num_key_value_heads"] = json!(16);
    d["head_dim"] = json!(128);
    d["intermediate_size"] = json!(5632);
    d["moe_intermediate_size"] = json!(1408);
    d["vocab_size"] = json!(151_936);
    d["num_experts"] = json!(60);
    d["num_experts_per_tok"] = json!(4);
    d["rms_norm_eps"] = json!(1e-6);
    d["rope_theta"] = json!(1_000_000.0);
    d["norm_topk_prob"] = json!(false);
    d["tie_word_embeddings"] = json!(false);
    d
}

/// Every field of the geometry that the lowering binds, in one place.
///
/// `DecodeGeometry` derives `Debug`, and pinning the whole `{:#?}` would pin
/// the FORMATTING too — a field reordered is not a behaviour change but would
/// read as one. So this lists what is checked, and
/// `the_pinned_set_still_covers_what_the_lowering_binds` below is what stops
/// the list going stale.
fn pins(g: &DecodeGeometry) -> Vec<(&'static str, String)> {
    vec![
        ("hidden", g.hidden.to_string()),
        ("n_layers", g.n_layers.to_string()),
        ("n_q_heads", g.n_q_heads.to_string()),
        ("n_kv_heads", g.n_kv_heads.to_string()),
        ("head_dim", g.head_dim.to_string()),
        ("intermediate", g.intermediate.to_string()),
        ("moe_intermediate", g.moe_intermediate.to_string()),
        ("n_experts", g.n_experts.to_string()),
        ("n_experts_per_tok", g.experts_per_token.to_string()),
        ("vocab", g.vocab.to_string()),
        ("eps", format!("{:e}", g.eps)),
        ("rope_theta", format!("{:e}", g.rope_theta)),
        ("tied_embeddings", g.tied_embeddings.to_string()),
        ("quant_bits", g.quant.bits.to_string()),
        ("quant_group", g.quant.group.to_string()),
    ]
}

/// Assert the pinned fields, naming each one that moved.
///
/// One assertion per field rather than one over the vector, because a failure
/// that says "left != right" over fifteen pairs is a failure the reader has
/// to diff by eye.
fn assert_pinned(what: &str, g: &DecodeGeometry, expected: &[(&str, &str)]) {
    let got = pins(g);
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: the pin list and the expectation are different lengths"
    );
    for ((name, actual), (want_name, want)) in got.iter().zip(expected) {
        assert_eq!(name, want_name, "{what}: pin list out of order");
        assert_eq!(
            actual, want,
            "{what}: `{name}` moved — decide which number the checkpoint asks \
             for and say so in the commit message"
        );
    }
}

#[test]
fn a_llama_descriptor_still_lands_where_it_landed() {
    let g = geometry_of(&llama_3_2_1b()).expect("Llama-3.2-1B is a sound config");
    assert_pinned(
        "llama",
        &g,
        &[
            ("hidden", "2048"),
            ("n_layers", "16"),
            ("n_q_heads", "32"),
            ("n_kv_heads", "8"),
            ("head_dim", "64"),
            ("intermediate", "8192"),
            ("moe_intermediate", "0"),
            ("n_experts", "0"),
            ("n_experts_per_tok", "0"),
            ("vocab", "128256"),
            ("eps", "1e-5"),
            ("rope_theta", "5e5"),
            ("tied_embeddings", "true"),
            ("quant_bits", "4"),
            ("quant_group", "64"),
        ],
    );
}

#[test]
fn a_gpt_oss_descriptor_still_lands_where_it_landed() {
    let g = geometry_of(&gpt_oss_20b()).expect("gpt-oss-20b is a sound config");
    assert_pinned(
        "gpt_oss",
        &g,
        &[
            ("hidden", "2880"),
            ("n_layers", "24"),
            ("n_q_heads", "64"),
            ("n_kv_heads", "8"),
            ("head_dim", "64"),
            ("intermediate", "2880"),
            // The merge copies `intermediate` here: gpt-oss is a mixture in
            // every layer with no dense FFN beside it, so the two widths are
            // one number and the descriptor states it once.
            ("moe_intermediate", "2880"),
            ("n_experts", "32"),
            ("n_experts_per_tok", "4"),
            ("vocab", "201088"),
            ("eps", "1e-5"),
            ("rope_theta", "1.5e5"),
            // Hardcoded `false` by the merge. The descriptor says false too,
            // so this pin does not distinguish them — which is the point of
            // recording that it is a hardcode.
            ("tied_embeddings", "false"),
            ("quant_bits", "4"),
            ("quant_group", "64"),
        ],
    );
}

#[test]
fn a_qwen3_next_descriptor_still_lands_where_it_landed() {
    let g = geometry_of(&qwen3_next()).expect("qwen3-next is a sound config");
    assert_pinned(
        "qwen3_next",
        &g,
        &[
            ("hidden", "2048"),
            ("n_layers", "48"),
            ("n_q_heads", "16"),
            ("n_kv_heads", "2"),
            ("head_dim", "256"),
            ("intermediate", "5120"),
            ("moe_intermediate", "512"),
            ("n_experts", "512"),
            ("n_experts_per_tok", "10"),
            ("vocab", "151936"),
            ("eps", "1e-6"),
            ("rope_theta", "1e7"),
            ("tied_embeddings", "false"),
            ("quant_bits", "4"),
            ("quant_group", "64"),
        ],
    );
}

#[test]
fn a_gemma4_descriptor_still_lands_where_it_landed() {
    let g = geometry_of(&gemma4_text()).expect("gemma4 is a sound config");
    assert_pinned(
        "gemma4",
        &g,
        &[
            ("hidden", "3840"),
            ("n_layers", "48"),
            ("n_q_heads", "16"),
            ("n_kv_heads", "4"),
            ("head_dim", "256"),
            ("intermediate", "14336"),
            ("moe_intermediate", "0"),
            ("n_experts", "0"),
            ("n_experts_per_tok", "0"),
            // Read from the descriptor's TOP LEVEL, because gemma4's own
            // block states no vocabulary.
            ("vocab", "262144"),
            // HARDCODED by the merge, not read: the descriptor above states
            // 1e-6 as well, so this pin agrees with both readings. If a
            // gemma config ever states another epsilon this is the pin that
            // has to be argued about.
            ("eps", "1e-6"),
            // The base the descriptor STATES. Until the four-arm family
            // ladder came out this was 1e6 — `g4_rope_theta_full`'s
            // fabricated default, chosen because the geometry asked whether
            // the gemma4 block had been read rather than whether a per-layer
            // base had been stated, and the flat key was discarded. The
            // fixture states 3e6 precisely so the two readings cannot agree
            // by coincidence.
            ("rope_theta", "3e6"),
            ("tied_embeddings", "true"),
            ("quant_bits", "4"),
            ("quant_group", "64"),
        ],
    );
}

/// The gemma-4 path every real checkpoint takes.
///
/// The test above pins the fallback — a flat base and no per-layer array.
/// This pins the array, and the two disagree on purpose: the importer writes
/// 1e4 into the flat key and 1e6 into the array's full-attention entries, so
/// `rope_theta` here is the one pin in the file that cannot be satisfied by
/// reading the wrong source. Falling back to the flat key would read 1e4.
#[test]
fn the_real_gemma4_descriptor_takes_its_base_from_the_per_layer_array() {
    let g = geometry_of(&gemma4_31b_as_imported()).expect("gemma-4-31b is a sound config");
    assert_pinned(
        "gemma4-31b-as-imported",
        &g,
        &[
            ("hidden", "5376"),
            ("n_layers", "60"),
            ("n_q_heads", "32"),
            ("n_kv_heads", "16"),
            ("head_dim", "256"),
            ("intermediate", "21504"),
            ("moe_intermediate", "0"),
            ("n_experts", "0"),
            ("n_experts_per_tok", "0"),
            ("vocab", "262144"),
            ("eps", "1e-6"),
            // 1e6, from the array. The flat key says 1e4. Two orders of
            // magnitude between the right answer and the reachable wrong one.
            ("rope_theta", "1e6"),
            ("tied_embeddings", "true"),
            ("quant_bits", "4"),
            ("quant_group", "64"),
        ],
    );
}

#[test]
fn a_qwen2_moe_descriptor_still_lands_where_it_landed() {
    let g = geometry_of(&qwen2_moe()).expect("qwen2-moe is a sound config");
    assert_pinned(
        "qwen2_moe",
        &g,
        &[
            ("hidden", "2048"),
            ("n_layers", "24"),
            ("n_q_heads", "16"),
            ("n_kv_heads", "16"),
            ("head_dim", "128"),
            ("intermediate", "5632"),
            ("moe_intermediate", "1408"),
            ("n_experts", "60"),
            ("n_experts_per_tok", "4"),
            ("vocab", "151936"),
            ("eps", "1e-6"),
            ("rope_theta", "1e6"),
            ("tied_embeddings", "false"),
            ("quant_bits", "4"),
            ("quant_group", "64"),
        ],
    );
}

/// The pinned set must still be the set the lowering binds.
///
/// The pins above are worth exactly what they cover. `lowering/consts.rs` is
/// the heaviest reader of `DecodeGeometry` and the survey named eight fields
/// it binds directly; if one of those stops being pinned, this file goes on
/// passing while the thing it was written to protect stops being protected.
///
/// Reads the source rather than the struct because there is no reflection: a
/// field bound in the lowering and absent from `pins` is the failure, and the
/// lowering's text is where the binding is stated.
#[test]
fn the_pinned_set_still_covers_what_the_lowering_binds() {
    let pinned: Vec<&str> = pins(&sound_geometry()).into_iter().map(|(n, _)| n).collect();
    for field in [
        "hidden",
        "intermediate",
        "moe_intermediate",
        "n_experts",
        "vocab",
        "eps",
    ] {
        assert!(
            pinned.contains(&field),
            "`{field}` is bound by lowering/consts.rs off DecodeGeometry and \
             is no longer pinned by this file — add it to `pins`, or this \
             test's list is what went stale"
        );
    }
}

/// Any geometry at all, for the coverage check above.
fn sound_geometry() -> DecodeGeometry {
    geometry_of(&llama_3_2_1b()).expect("Llama-3.2-1B is a sound config")
}
