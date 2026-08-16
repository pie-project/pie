//! Every catalog row against the checkpoint it claims to be, number for
//! number.
//!
//! This is the safety net for the whole refactor, and it is the same
//! technique the repo already trusts: `tests/differential.rs` held the
//! Rust normalizer to the C++ one it replaced, over this exact corpus,
//! and that is what made deleting the C++ tree a mechanical act rather
//! than a leap. The 58 configs come back here as the oracle for a second
//! replacement.
//!
//! # THE ORACLE IS THE CHECKPOINT'S OWN WORDS
//!
//! It began as a differential against `deployment_cuda::deployment_from`
//! — the eleven per-family derivations this refactor deletes — and that
//! was the right oracle while both existed. It is the WRONG one now,
//! and not merely because the subject is deleted: the derivation and
//! the row would have been two readings of the same `config.json`, so
//! agreement would have proved they were transcribed alike, never that
//! either was RIGHT.
//!
//! So the oracle is the file. For every corpus `config.json` a row
//! claims, every number the config states is compared against the row's
//! answer to the same question. `hidden_size: 1024` in the file and
//! `hidden: 1024` in the row, or the test fails and names both.
//!
//! This is the machinery honest-cost 2 of the design asked for. A
//! `config.json` is at least SELF-CONSISTENT — it is what the
//! checkpoint says about itself — and moving those numbers into `const`
//! rows means a typo becomes a quietly wrong model. Manifest matching
//! catches a typo that contradicts a TENSOR; this catches one that
//! contradicts the publisher. Between them there is nowhere for a
//! mistyped digit to hide.
//!
//! # What a failure means
//!
//! Not "the test is wrong". A row is a set of constants transcribed
//! from a published checkpoint, and a mismatch is a transcription
//! error until proven otherwise. The rare other case — the row is
//! right and the config is describing something else, as when a
//! `head_dim` is absent and implied — is handled by naming the field
//! here rather than by loosening the comparison.
//!
//! # What it does NOT check
//!
//! Rows with no config in the corpus. The corpus is 58 files and the
//! catalog is larger, because a corpus is what someone happened to dump
//! and a catalog is what the tree claims to serve. Coverage of the
//! corpus is asserted below; coverage of the catalog is the manifest's
//! job, not this file's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use model::catalog::{self, Deployed};
use model::deployment::Deployment;
use serde_json::Value;

/// The corpus, which lives with `driver-cuda` because it moved there
/// when the C++ tree was deleted.
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/driver-cuda/tests/hf_config_dump/corpus")
}

/// Every `(stem, path)` in the corpus, sorted so a failure names the
/// same file on every machine.
fn corpus() -> Vec<(String, PathBuf)> {
    let mut entries: Vec<(String, PathBuf)> = std::fs::read_dir(corpus_dir())
        .expect("the corpus directory is checked in beside the oracle")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), p))
        .collect();
    entries.sort();
    entries
}

/// Which row a corpus file is a config of.
///
/// A hand-written map, and it is hand-written for a reason worth
/// stating: the whole point of the refactor is that a `config.json`
/// string is NOT how a model is identified any more, so a differential
/// test that derived the mapping from `model_type` would be testing the
/// thing it is supposed to be retiring. In production the answer comes
/// from [`catalog::identify`] matching TENSORS. Here there are no
/// tensors — a corpus of configs has no checkpoint behind it — so the
/// correspondence is stated, once, by the person who transcribed the
/// rows.
///
/// A file absent from this map is a file no row claims, which is a
/// legitimate answer for the 27 synthetic configs: they exist to
/// exercise branches of a PARSER, and a parser is what is going away.
fn claimed_by() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        // ── Real checkpoints, each transcribed into a row.
        ("Qwen--Qwen3-0.6B", "qwen3-0.6b"),
        ("Qwen--Qwen3-1.7B", "qwen3-1.7b"),
        ("meta-llama--Llama-3.2-1B-Instruct", "llama-3.2-1b"),
        ("microsoft--Phi-3-mini-4k-instruct", "phi-3-mini-4k"),
        ("allenai--Olmo-3-1025-7B", "olmo-3-7b"),
        ("openai--gpt-oss-20b", "gpt-oss-20b"),
        ("google--gemma-3n-E2B-it", "gemma-3n-e2b"),
        ("google--gemma-3n-E4B-it", "gemma-3n-e4b"),
        // The E4B pair is the 0.8B pair's reason again: `google/gemma-4-E4B`
        // and its instruction tune ship byte-identical `text_config`
        // blocks, so no manifest match could tell them apart and one row
        // claims both.
        ("google--gemma-4-E4B-it", "gemma-4-e4b"),
        ("google--gemma-4-E4B", "gemma-4-e4b"),
        ("google--gemma-4-26B-A4B-it", "gemma-4-26b-a4b"),
        ("google--embeddinggemma-300m", "embeddinggemma-300m"),
        ("Qwen--Qwen3.5-4B", "qwen3.5-4b"),
        ("Qwen--Qwen3.5-9B", "qwen3.5-9b"),
        ("Qwen--Qwen3.5-35B-A3B", "qwen3.5-35b-a3b"),
        // ONE row for two files, twice over, and for two different
        // reasons worth keeping apart.
        //
        // The 0.8B pair is decision 1: the base model and its instruct
        // tune have byte-identical geometry, so no manifest match could
        // ever tell them apart. Two rows would be a distinction the
        // identifier cannot make, which is exactly the `Ambiguous` arm
        // the design promises to avoid by not creating the ambiguity.
        ("Qwen--Qwen3.5-0.8B-Base", "qwen3.5-0.8b-base"),
        ("Qwen--Qwen3.5-0.8B", "qwen3.5-0.8b-base"),
        // The 27B pair is decision 2: quantization is policy, not
        // identity. If these two ever disagree, a `Deployment` has
        // learned about an encoding and something has gone wrong
        // upstream of here.
        ("Qwen--Qwen3.6-27B", "qwen3.6-27b"),
        ("Qwen--Qwen3.6-27B-FP8", "qwen3.6-27b"),
    ])
}

/// Real checkpoints in the corpus that deliberately have no row.
///
/// Named individually rather than swept under a prefix, because the
/// cost of a closed catalog (honest dependency 1 of the design) is that
/// "we do not serve this" has to be a STATEMENT. A config sitting in the
/// corpus unclaimed and unexplained is indistinguishable from a row
/// someone forgot to write.
fn not_served() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        (
            "Qwen--Qwen1.5-MoE-A2.7B",
            "qwen-1.5 predates every generation in the catalog; it is in the \
             corpus because it exercised the normalizer's shared-expert branch",
        ),
        (
            "Qwen--Qwen3-VL-2B-Instruct",
            "a vision-language checkpoint whose text tower pie has never \
             served on its own; the row would claim tensors nothing places",
        ),
        (
            "google--gemma-4-E4B-it-assistant",
            "four layers with `num_kv_shared_layers: 4`, so every layer in it \
             attends KV an earlier layer wrote and there is no earlier layer: \
             its pages come from the E4B backbone it rides beside. A \
             `Deployment` describes ONE stack and has nowhere to say that, so \
             a row would have to land every `kv_source` on itself and serve a \
             stack attending its own empty cache",
        ),
    ])
}

/// Rows that identify their checkpoint and refuse to fire it.
///
/// A third category, and it is the one the `Result` on
/// [`catalog::Variant::deployment`] exists for. A row here is NOT
/// unclaimed: its manifest matches, its load shape is complete and its
/// contract authors, so a load reaches the door and is turned away with
/// a reason. The differential still holds every number it CAN answer to
/// the config — which is the whole load shape — and skips the
/// deployment half, because there is no deployment to compare.
///
/// Being a named list rather than a `matches!` on the error is
/// deliberate: a row that starts refusing without anyone deciding it
/// should is exactly the regression this file is for.
///
/// EMPTY today, and the empty list is doing work. Its one entry was
/// `google--gemma-4-26B-A4B-it`, excused for `attention_k_eq_v` and a
/// 128-expert block; both legs are traced now, so the differential
/// compares its deployment like any other row's. Kept rather than
/// deleted because the category is real -- `deployment` returns a
/// `Result` and a build that cannot provision a store still has to turn
/// a load away at the door -- and because `every_unfired_row_actually_refuses`
/// is what noticed the excuse had gone stale rather than letting it
/// silently skip half a comparison.
fn identified_but_unfired() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::new()
}

/// The object a checkpoint's own numbers live in.
///
/// HuggingFace nests the decoder under `text_config` whenever a
/// checkpoint ships a tower beside it — gemma-4 and Qwen3-VL both do —
/// and states the same keys at the root when it does not. Two spellings
/// of one place, so this returns the one that HAS a layer count rather
/// than guessing from the family.
///
/// Preferring `text_config` is deliberate. A multimodal config states
/// `num_hidden_layers` at BOTH levels, and the root's is the vision
/// tower's: reading it would compare a 27-layer decoder against a
/// 16-layer encoder and call the row wrong.
fn decoder_object(doc: &Value) -> &Value {
    match doc.get("text_config") {
        Some(t) if t.get("num_hidden_layers").is_some() => t,
        _ => doc,
    }
}

/// An unsigned number a config states under any of several spellings,
/// or `None` if it states none of them.
///
/// `None` is NOT a failure. HuggingFace omits what it considers
/// implied — `head_dim` is absent whenever it is `hidden_size /
/// num_attention_heads`, `num_key_value_heads` is absent on a
/// multi-head model — and a row states the resolved value. Comparing
/// against an absent key would fail every such row for being complete.
///
/// The spellings are tried IN ORDER and the first present one wins, so
/// a config carrying both `n_routed_experts` and a legacy `num_experts`
/// is read the way its own family reads it.
fn stated_u32(o: &Value, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|k| o.get(*k).and_then(Value::as_u64))
        .and_then(|v| u32::try_from(v).ok())
}

/// The same, for a value a config states as a float.
fn stated_f64(o: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| o.get(*k).and_then(Value::as_f64))
}

/// One comparison, deferred so a whole config reports at once.
///
/// A row with three transcription errors should print three lines. The
/// alternative — `assert_eq!` per field — prints the first and hides the
/// rest behind a re-run, which is how a transcription pass turns into
/// five.
struct Compare {
    stem: String,
    id: &'static str,
    rows: Vec<String>,
}

impl Compare {
    fn new(stem: &str, id: &'static str) -> Self {
        Self {
            stem: stem.to_string(),
            id,
            rows: Vec::new(),
        }
    }

    /// A number the config states, against the row's answer.
    ///
    /// Skipped entirely when the config states nothing — see
    /// [`stated_u32`].
    fn u32(&mut self, field: &str, stated: Option<u32>, row: u32) {
        if let Some(want) = stated
            && want != row
        {
            self.rows
                .push(format!("  {field}: config={want} row={row}"));
        }
    }

    /// A float, compared with a RELATIVE tolerance.
    ///
    /// `1e-6` is `0.000001` in JSON and `1e-6f32` is `9.99999975e-7`, so
    /// an exact comparison fails every norm epsilon in the corpus for
    /// being correctly rounded. The tolerance is on the ratio rather
    /// than the difference because these span nine orders of magnitude:
    /// an absolute epsilon that passes `rope_theta: 1000000.0` would
    /// pass `norm_eps: 1e-5` against `1e-6`, which is a real defect —
    /// gemma-2 and llama-3 differ by exactly that.
    fn f32(&mut self, field: &str, stated: Option<f64>, row: f32) {
        let Some(want) = stated else { return };
        let (want, got) = (want, f64::from(row));
        let scale = want.abs().max(got.abs()).max(f64::MIN_POSITIVE);
        if (want - got).abs() / scale > 1e-5 {
            self.rows
                .push(format!("  {field}: config={want} row={got}"));
        }
    }

    /// A boolean the config states.
    fn bool(&mut self, field: &str, stated: Option<bool>, row: bool) {
        if let Some(want) = stated
            && want != row
        {
            self.rows
                .push(format!("  {field}: config={want} row={row}"));
        }
    }

    /// The report, or `None` when every stated number matched.
    fn finish(self) -> Option<String> {
        (!self.rows.is_empty()).then(|| {
            format!(
                "{} (row '{}'): the row disagrees with the checkpoint's own \
                 config.json\n{}",
                self.stem,
                self.id,
                self.rows.join("\n")
            )
        })
    }
}

/// Every claimed corpus config agrees with the row that claims it.
///
/// The one test the whole refactor rests on.
#[test]
fn a_row_states_what_its_checkpoint_states() {
    let claims = claimed_by();
    let unfired = identified_but_unfired();
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (stem, path) in corpus() {
        let Some(id) = claims.get(stem.as_str()).copied() else {
            continue;
        };
        let Some(row) = catalog::find(id) else {
            failures.push(format!(
                "{stem}: claimed by row '{id}', which is not in the catalog — \
                 the map above and the table have drifted"
            ));
            continue;
        };
        let text = std::fs::read_to_string(&path).expect("the corpus is checked in");
        let doc: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{stem}: not JSON: {e}"));
                continue;
            }
        };
        // A refusal is a failure UNLESS the row is one of the named few
        // that identify a checkpoint and decline to fire it. Those still
        // get their load shape compared: half the row's transcription is
        // readable without a `Deployment`, and leaving it uncompared
        // would make a refusing row the one place a mistyped digit could
        // hide.
        let dep = match row.deployment(Deployed::single()) {
            Ok(d) => Some(d),
            Err(_) if unfired.contains_key(stem.as_str()) => None,
            Err(r) => {
                failures.push(format!("{stem}: row '{id}' refused to deploy: {r:?}"));
                continue;
            }
        };
        checked += 1;
        if let Some(report) = compare(&stem, id, &doc, dep.as_ref(), row.load_shape()) {
            failures.push(report);
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(
        checked >= 10,
        "only {checked} configs were compared; the map has gone stale and the \
         net is not catching anything"
    );
}

/// Every number a corpus config states, against the row's answer to the
/// same question.
///
/// # Why these fields and not others
///
/// A `Deployment` holds answers of two kinds. Some are TRANSCRIBED —
/// the layer count, the widths, the head counts, the epsilon, the
/// rotary base — and a config states them, so this compares them. The
/// rest are DECIDED: `PrefillStyle`, `AttnOutput`, `KvStyle` and
/// `NormPlacement` are choices about how this build serves a shape, and
/// a `config.json` has no opinion on them. Comparing those against a
/// file would mean inventing a mapping here, which is the eleven
/// deleted derivations coming back in the test directory.
///
/// So this catches transcription errors, which is the failure mode the
/// catalog introduces, and says nothing about the projections, which
/// are covered where they are written.
/// The half of the comparison a `Deployment` answers.
///
/// Split out so a row that identifies its checkpoint and refuses to
/// fire it — see [`identified_but_unfired`] — is still held to the
/// config for everything it CAN answer.
fn compare_deployment(c: &mut Compare, o: &Value, dep: &Deployment) {
    c.u32("layers", stated_u32(o, &["num_hidden_layers"]), dep.layers);
    c.u32("hidden", stated_u32(o, &["hidden_size"]), dep.shape.hidden);
    c.u32(
        "q_heads",
        stated_u32(o, &["num_attention_heads"]),
        dep.shape.q_heads,
    );
    c.u32(
        "kv_heads",
        stated_u32(o, &["num_key_value_heads"]),
        dep.shape.kv_heads,
    );
    c.u32("head_dim", stated_u32(o, &["head_dim"]), dep.shape.head_dim);
    // `intermediate_size` is the width of ONE MLP, and which MLP that is
    // depends on whether the stack has a dense one.
    //
    // `Geometry::intermediate` means the DENSE block, which is why
    // gpt-oss states zero there and 2880 in `moe_intermediate`: every
    // layer of it is the router's, and `widest_mlp()` takes the max of
    // the two to size the one buffer. Its config still spells the expert
    // width `intermediate_size`, because that is the only MLP it has —
    // there is no `moe_intermediate_size` key in it at all.
    //
    // So a row with no dense block is held to the config on the field
    // the config is describing. Comparing it against `intermediate`
    // reported `config=2880 row=0` and named neither the row nor the
    // config as wrong, which is what a mapping from one key to one field
    // does when a key means two things.
    let all_routed = dep.shape.intermediate == 0 && dep.shape.moe_intermediate > 0;
    if all_routed {
        c.u32(
            "intermediate",
            stated_u32(o, &["intermediate_size"]),
            dep.shape.moe_intermediate,
        );
    } else {
        c.u32(
            "intermediate",
            stated_u32(o, &["intermediate_size"]),
            dep.shape.intermediate,
        );
    }
    c.u32("vocab", stated_u32(o, &["vocab_size"]), dep.shape.vocab);
    c.u32(
        "max_model_len",
        stated_u32(o, &["max_position_embeddings"]),
        dep.advertised.max_model_len,
    );
    c.f32(
        "norm_eps",
        stated_f64(o, &["rms_norm_eps", "layer_norm_eps", "norm_eps"]),
        dep.norm_eps,
    );

    // THE ROTARY BASE, per layer, against the one the config states.
    //
    // Compared on the layers that USE it. A sliding-window layer in a
    // gemma-3 stack runs a different base from a full-attention one —
    // `rope_local_base_freq` against `rope_theta` — so this checks that
    // at least one layer carries the stated global base rather than
    // demanding every layer carry it. A row that put the LOCAL base on
    // every layer fails, which is the transcription error worth
    // catching; a row that alternates correctly passes.
    if let Some(theta) = stated_f64(o, &["rope_theta"])
        && !dep.attention.is_empty()
    {
        let matched = dep
            .attention
            .iter()
            .any(|a| (f64::from(a.rope_theta) - theta).abs() / theta.abs().max(1.0) <= 1e-5);
        if !matched {
            let seen: Vec<String> = dep
                .attention
                .iter()
                .map(|a| a.rope_theta.to_string())
                .collect();
            c.rows.push(format!(
                "  rope_theta: config={theta} and no layer of the row uses it \
                 (row layers use {seen:?})"
            ));
        }
    }

    // THE ROPE RESCALING, which is the four numbers `rope_theta` cannot
    // say.
    //
    // This is the check that would have caught the regression that
    // prompted it: `driver-metal` read `rope_scaling.{factor,
    // low_freq_factor, high_freq_factor,
    // original_max_position_embeddings}` off the deleted `pie.model/1`
    // descriptor, and when the descriptor went, nothing filled them.
    // `DecodeGeometry` kept four zeroes, and a zero factor reads as "no
    // rescaling" — so every Llama-3.1/3.2/3.3 would have attended past
    // its trained 8192 with the wrong wavelengths. That model does not
    // crash. It degrades, fluently, which is the failure mode this whole
    // file exists to convert into a red build.
    //
    // Both directions are checked. A row that states a rescaling the
    // config does not is as wrong as one that omits the config's.
    compare_rope_scaling(c, o, dep);
}

/// A config's rope rescaling, as the four-or-five numbers it states.
///
/// `rope_scaling` is the long-standing spelling and `rope_parameters` the
/// newer one; a config states one or neither, so this reads both. A kind
/// of `default` is the ABSENCE of a rescaling written out longhand, which
/// is why it returns `None` rather than a zeroed rescaling.
fn stated_rope_scaling(o: &Value) -> Option<(String, Value)> {
    let r = o.get("rope_scaling").or_else(|| o.get("rope_parameters"))?;
    let obj = r.as_object()?;
    let kind = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    if kind == "default" {
        return None;
    }
    Some((kind.to_owned(), r.clone()))
}

fn compare_rope_scaling(c: &mut Compare, o: &Value, dep: &Deployment) {
    use model::deployment::RopeScaling;

    let stated = stated_rope_scaling(o);
    let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64);
    let flag = |v: &Value, k: &str| v.get(k).and_then(Value::as_bool);

    match (stated, dep.rope_scaling) {
        (None, None) => {}
        (None, Some(row)) => c.rows.push(format!(
            "  rope_scaling: the config states none and the row states {row:?} — a row \
             that rescales a ladder its checkpoint does not is as wrong as one that \
             forgets to"
        )),
        (Some((kind, _)), None) => c.rows.push(format!(
            "  rope_scaling: config states `{kind}` and the row states none, so a driver \
             builds the unrescaled ladder and the model degrades past its trained context \
             rather than failing"
        )),
        (
            Some((kind, v)),
            Some(RopeScaling::Piecewise {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            }),
        ) => {
            if kind != "llama3" {
                c.rows
                    .push(format!("  rope_scaling.kind: config={kind} row=llama3"));
                return;
            }
            c.f32("rope_scaling.factor", num(&v, "factor"), factor);
            c.f32(
                "rope_scaling.low_freq_factor",
                num(&v, "low_freq_factor"),
                low_freq_factor,
            );
            c.f32(
                "rope_scaling.high_freq_factor",
                num(&v, "high_freq_factor"),
                high_freq_factor,
            );
            c.u32(
                "rope_scaling.original_max_position",
                num(&v, "original_max_position_embeddings").map(|x| x as u32),
                original_max_position,
            );
        }
        (
            Some((kind, v)),
            Some(RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                attention_factor,
                original_max_position,
                truncate,
            }),
        ) => {
            if kind != "yarn" {
                c.rows
                    .push(format!("  rope_scaling.kind: config={kind} row=yarn"));
                return;
            }
            c.f32("rope_scaling.factor", num(&v, "factor"), factor);
            c.f32("rope_scaling.beta_fast", num(&v, "beta_fast"), beta_fast);
            c.f32("rope_scaling.beta_slow", num(&v, "beta_slow"), beta_slow);
            c.u32(
                "rope_scaling.original_max_position",
                num(&v, "original_max_position_embeddings").map(|x| x as u32),
                original_max_position,
            );
            // Compared only when the config states it. Most omit it, and
            // the row then carries HF's `0.1 * ln(factor) + 1`; that
            // formula is checked against OLMo 3, which does state its
            // answer, in `olmo_3`'s own tests.
            c.f32(
                "rope_scaling.attention_factor",
                num(&v, "attention_factor"),
                attention_factor,
            );
            // Compared only when the config states it, for the same reason
            // as `attention_factor` -- but the two omissions mean different
            // things. An absent `attention_factor` is a formula the row
            // evaluates; an absent `truncate` is HF's `true`, and the whole
            // corpus omits it except gpt-oss, which is the family whose
            // ramp would move if a row copied the majority.
            c.bool("rope_scaling.truncate", flag(&v, "truncate"), truncate);
        }
    }
}

fn compare(
    stem: &str,
    id: &'static str,
    doc: &Value,
    dep: Option<&Deployment>,
    shape: model::catalog::LoadShape,
) -> Option<String> {
    let o = decoder_object(doc);
    let mut c = Compare::new(stem, id);

    if let Some(dep) = dep {
        compare_deployment(&mut c, o, dep);
    }
    // THE LOAD SHAPE, which is the other half of the row and is read by
    // the authoring pass rather than by a driver. `head_dim` appears in
    // both and is compared in both deliberately: `LoadShape::head_dim`
    // is the UNPADDED width a TP split cuts on and `Geometry::head_dim`
    // is what a kernel is handed, and a row that lets them drift splits
    // a head in half.
    c.u32(
        "load_shape.head_dim",
        stated_u32(o, &["head_dim"]),
        shape.head_dim,
    );
    c.u32(
        "load_shape.layers",
        stated_u32(o, &["num_hidden_layers"]),
        shape.layers,
    );
    c.u32(
        "load_shape.n_experts",
        stated_u32(o, &["num_experts", "num_local_experts", "n_routed_experts"]),
        shape.n_experts,
    );
    c.u32(
        "load_shape.mamba_groups",
        stated_u32(o, &["mamba_n_groups", "n_groups"]),
        shape.mamba_groups,
    );
    // `tie_word_embeddings` is the one boolean a config states that a
    // row must match, and it is the one whose default is a trap: absent
    // means TRUE for most families and FALSE for a few, which is why the
    // deleted normalizer carried a `TIE_BY_DEFAULT` table of its own. An
    // absent key is SKIPPED here rather than defaulted — a row's answer
    // is the transcription of what the checkpoint ships, one embedding
    // table or two, and that is a manifest question rather than a
    // config one.
    c.bool(
        "tied_embeddings",
        o.get("tie_word_embeddings").and_then(Value::as_bool),
        shape.tied_embeddings,
    );

    c.finish()
}

/// The comparison catches a transcribed digit that the config
/// contradicts.
///
/// A test of the NET rather than of a row, and it is here because a
/// differential that silently compares nothing is worse than no
/// differential: `stated_u32` returning `None` for a misspelled key
/// would make every field pass. This drives one field wrong and one
/// field absent through the same path.
#[test]
fn the_comparison_catches_a_wrong_digit_and_skips_an_absent_one() {
    let doc: Value = serde_json::from_str(
        r#"{"num_hidden_layers": 28, "hidden_size": 1024, "rms_norm_eps": 1e-6}"#,
    )
    .unwrap();
    let o = decoder_object(&doc);

    let mut c = Compare::new("fixture", "row");
    c.u32("layers", stated_u32(o, &["num_hidden_layers"]), 28);
    c.u32("vocab", stated_u32(o, &["vocab_size"]), 151_936);
    c.f32("norm_eps", stated_f64(o, &["rms_norm_eps"]), 1e-6);
    assert!(
        c.finish().is_none(),
        "a matching row, and an absent key, must not report"
    );

    let mut c = Compare::new("fixture", "row");
    c.u32("layers", stated_u32(o, &["num_hidden_layers"]), 24);
    let report = c.finish().expect("a wrong layer count must report");
    assert!(report.contains("config=28 row=24"), "{report}");
    assert!(report.contains("fixture"), "{report}");
}

/// The float comparison is RELATIVE, so `1e-6` in JSON matches `1e-6f32`
/// and `1e-5` does not.
///
/// The exact case that motivated it: gemma-2 runs `1e-6` and llama-3
/// runs `1e-5`, a difference an absolute tolerance sized for
/// `rope_theta: 1000000.0` cannot see.
#[test]
fn the_epsilon_comparison_separates_1e5_from_1e6() {
    let mut c = Compare::new("fixture", "row");
    c.f32("norm_eps", Some(1e-6), 1e-6);
    c.f32("rope_theta", Some(1_000_000.0), 1e6);
    assert!(
        c.finish().is_none(),
        "a correctly-rounded f32 must match its JSON literal"
    );

    let mut c = Compare::new("fixture", "row");
    c.f32("norm_eps", Some(1e-5), 1e-6);
    assert!(
        c.finish().is_some(),
        "1e-5 and 1e-6 are different models, not rounding"
    );
}

/// A multimodal config's decoder is read from `text_config`, not from
/// the root.
///
/// The failure this forbids is quiet and total: a gemma-4 config states
/// `num_hidden_layers` twice, and the root's belongs to the vision
/// tower. Reading it compares a 16-layer encoder against a 35-layer
/// decoder and reports the ROW as wrong.
#[test]
fn a_multimodal_config_is_read_at_its_text_tower() {
    let nested: Value = serde_json::from_str(
        r#"{"num_hidden_layers": 16, "text_config": {"num_hidden_layers": 35}}"#,
    )
    .unwrap();
    assert_eq!(
        stated_u32(decoder_object(&nested), &["num_hidden_layers"]),
        Some(35)
    );

    // A flat config has no `text_config`, and one that carries a
    // `text_config` with no layer count in it (a tokenizer stub) is not
    // a decoder either.
    let flat: Value = serde_json::from_str(r#"{"num_hidden_layers": 28}"#).unwrap();
    assert_eq!(
        stated_u32(decoder_object(&flat), &["num_hidden_layers"]),
        Some(28)
    );
    let stub: Value =
        serde_json::from_str(r#"{"num_hidden_layers": 28, "text_config": {"eos": 1}}"#).unwrap();
    assert_eq!(
        stated_u32(decoder_object(&stub), &["num_hidden_layers"]),
        Some(28)
    );
}

/// The spellings are tried in order, and the first present one wins.
#[test]
fn an_expert_count_is_read_under_whichever_name_its_family_uses() {
    let each = [
        (r#"{"num_experts": 128}"#, 128),
        (r#"{"num_local_experts": 8}"#, 8),
        (r#"{"n_routed_experts": 256}"#, 256),
    ];
    for (raw, want) in each {
        let doc: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(
            stated_u32(
                &doc,
                &["num_experts", "num_local_experts", "n_routed_experts"]
            ),
            Some(want),
            "{raw}"
        );
    }
    // A dense config states none of them, and `None` means "compare
    // nothing" rather than "zero experts".
    let dense: Value = serde_json::from_str(r#"{"hidden_size": 4096}"#).unwrap();
    assert_eq!(
        stated_u32(
            &dense,
            &["num_experts", "num_local_experts", "n_routed_experts"]
        ),
        None
    );
}

/// Every id in the map above is a real row.
///
/// Separate from the differential so a typo in the map reports as a typo
/// rather than as a model disagreement.
#[test]
fn the_map_names_only_rows_that_exist() {
    let missing: Vec<&str> = claimed_by()
        .values()
        .copied()
        .filter(|id| catalog::find(id).is_none())
        .collect();
    assert!(missing.is_empty(), "ids claimed by no row: {missing:?}");
}

/// Every corpus file is either claimed by a row or is a parser fixture.
///
/// The second category is finite and named, so a REAL checkpoint added
/// to the corpus without a row fails here instead of being silently
/// unserved. `synthetic--*` files exercise branches of the normalizer
/// that is going away; `amd-quark--`, `np-cr--`, `dacorvo--` and
/// `tiny-random--` are randomly-initialised test artifacts with no
/// published geometry to transcribe.
#[test]
fn every_real_config_is_claimed_or_explained() {
    let claims = claimed_by();
    let excused = not_served();
    let unexplained: Vec<String> = corpus()
        .into_iter()
        .map(|(stem, _)| stem)
        .filter(|stem| !claims.contains_key(stem.as_str()))
        .filter(|stem| !excused.contains_key(stem.as_str()))
        .filter(|stem| !stem.starts_with("synthetic--"))
        .filter(|stem| !stem.starts_with("tiny-random--"))
        .filter(|stem| !stem.starts_with("amd-quark--"))
        .filter(|stem| !stem.starts_with("np-cr--"))
        .filter(|stem| !stem.starts_with("dacorvo--"))
        .collect();
    assert!(
        unexplained.is_empty(),
        "real checkpoints in the corpus that no row claims: {unexplained:?}\n\
         Either transcribe a row for each, or add it to `not_served` with a \
         sentence saying why it is not served."
    );
}

/// The two lists do not overlap.
///
/// A stem in both would mean someone wrote a row and then also excused
/// the file, and the excuse would win silently in the differential above
/// — the model would go uncompared while looking claimed.
#[test]
fn nothing_is_both_claimed_and_excused() {
    let claims = claimed_by();
    let both: Vec<&str> = not_served()
        .keys()
        .copied()
        .filter(|stem| claims.contains_key(stem))
        .collect();
    assert!(
        both.is_empty(),
        "claimed by a row AND excused from having one: {both:?}"
    );

    // The unfired list is the opposite shape: every stem in it MUST be
    // claimed, because it names a row's refusal and a stem no row claims
    // has no refusal to name.
    let unclaimed: Vec<&str> = identified_but_unfired()
        .keys()
        .copied()
        .filter(|stem| !claims.contains_key(stem))
        .collect();
    assert!(
        unclaimed.is_empty(),
        "listed as identified-but-unfired while no row claims them: {unclaimed:?}"
    );
    let excused: Vec<&str> = identified_but_unfired()
        .keys()
        .copied()
        .filter(|stem| not_served().contains_key(stem))
        .collect();
    assert!(
        excused.is_empty(),
        "both excused from having a row and named as one: {excused:?}"
    );
}

/// A row excused from firing really does refuse.
///
/// Without this the excuse is a one-way ratchet: a leg gets traced, the
/// row starts deploying, and the list above quietly keeps skipping the
/// half of the comparison that just became available. A stale excuse is
/// a differential that compares less than it says it does.
#[test]
fn every_unfired_row_actually_refuses() {
    let claims = claimed_by();
    for (stem, why) in identified_but_unfired() {
        let id = claims.get(stem).copied().expect("checked above");
        let row = catalog::find(id).expect("checked above");
        assert!(
            row.deployment(Deployed::single()).is_err(),
            "'{id}' is excused from firing because {why}, and it deploys — \
             delete the entry and let the differential compare its deployment"
        );
    }
}

/// The corpus is still the size the oracle recorded.
#[test]
fn the_corpus_is_the_one_the_oracle_saw() {
    assert_eq!(
        corpus().len(),
        58,
        "the corpus changed size; `tests/differential.rs` asserts against the \
         same directory and will need the same update"
    );
}
