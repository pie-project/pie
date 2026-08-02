//! The Rust normalizer against the C++ one it replaces, field for field.
//!
//! `pie.model/1` only works if this crate answers "what is this model made of"
//! the same way `driver/cuda/src/model/config.cpp` does. Not approximately —
//! a single defaulting rule that differs breaks a model, and it breaks it
//! *later*, once the C++ side is deleted and there is nothing left to disagree
//! with.
//!
//! So the check is a comparison against recorded output of the real thing.
//! `driver/cuda/tests/hf_config_dump/` builds `parse_hf_config` behind a JSON
//! dumper with a host compiler (no CUDA), runs it over 55 configs — 28 real,
//! 27 synthetic for the branches a model cache does not happen to contain —
//! and checks in the results. This test re-normalizes the same inputs here and
//! compares.
//!
//! The goldens record what `config.cpp` *does*, including where that is wrong:
//! `attention_has_sinks` is set for `deepseek_v4` and then unconditionally
//! overwritten, so DeepSeek-V4 has no sinks. This test asserts the port
//! reproduces that. Fixing it means editing the C++, regenerating the goldens
//! and updating both sides together — never one side quietly diverging.

use std::path::{Path, PathBuf};

use serde_json::Value;

fn here() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn oracle_dir() -> PathBuf {
    here().join("../../driver/cuda/tests/hf_config_dump")
}

/// Every (name, config path, golden path) in the corpus.
fn corpus() -> Vec<(String, PathBuf, PathBuf)> {
    let dir = oracle_dir();
    let mut entries: Vec<_> = std::fs::read_dir(dir.join("corpus"))
        .expect("the corpus directory is checked in beside the oracle")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .map(|e| {
            let name = e.path().file_stem().unwrap().to_string_lossy().into_owned();
            let golden = dir.join("golden").join(format!("{name}.json"));
            (name, e.path(), golden)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Compares two JSON values, collecting every difference rather than the first.
///
/// Numbers compare at `f32`: the C++ side stores these as `float` and prints
/// them promoted to `double`, so `1e-5f` arrives as `9.999999747378752e-06`.
/// Comparing the printed text would fail on formatting; comparing as `f64`
/// would fail on the promotion. `f32` is the width both sides actually hold.
fn diff(path: &str, actual: &Value, expected: &Value, out: &mut Vec<String>) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            let mut keys: Vec<&String> = a.keys().chain(e.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                match (a.get(key), e.get(key)) {
                    (Some(a), Some(e)) => diff(&child, a, e, out),
                    (None, Some(e)) => out.push(format!("{child}: missing, expected {e}")),
                    (Some(a), None) => out.push(format!("{child}: unexpected {a}")),
                    (None, None) => unreachable!(),
                }
            }
        }
        (Value::Array(a), Value::Array(e)) => {
            if a.len() != e.len() {
                out.push(format!("{path}: length {} != {}", a.len(), e.len()));
                return;
            }
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                diff(&format!("{path}[{i}]"), a, e, out);
            }
        }
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64(), e.as_f64());
            match (a, e) {
                (Some(a), Some(e)) if a as f32 == e as f32 => {}
                (Some(a), Some(e)) if a.is_nan() && e.is_nan() => {}
                _ => out.push(format!("{path}: {a:?} != {e:?}")),
            }
        }
        (a, e) if a == e => {}
        (a, e) => out.push(format!("{path}: {a} != {e}")),
    }
}

/// The Rust normalizer reproduces the C++ one on every config in the corpus.
#[test]
fn the_port_matches_the_normalizer_it_replaces() {
    let entries = corpus();
    assert!(
        entries.len() >= 50,
        "the corpus should hold 55 configs, found {} — is \
         driver/cuda/tests/hf_config_dump/corpus checked out?",
        entries.len()
    );

    let mut failures = Vec::new();
    for (name, config, golden) in &entries {
        let raw = std::fs::read_to_string(config).unwrap();
        let root: Value = serde_json::from_str(&raw).unwrap();
        let expected: Value = serde_json::from_str(
            &std::fs::read_to_string(golden)
                .unwrap_or_else(|_| panic!("{name}: no golden; regenerate them")),
        )
        .unwrap();

        // The C++ side reports a refusal as `{"error": ...}`. Where it refuses,
        // this must refuse too — a port that accepts what the original rejects
        // is as wrong as one that computes a different number.
        let cxx_refused = expected.get("error").is_some();
        match pie_model_config::normalize(&root, name) {
            Ok(_) if cxx_refused => {
                failures.push(format!("{name}: accepted, but the C++ side refuses it"))
            }
            Err(err) if !cxx_refused => failures.push(format!(
                "{name}: refused ({err}), but the C++ side accepts it"
            )),
            Err(_) => {}
            Ok(cfg) => {
                let actual = serde_json::to_value(&cfg).unwrap();
                let mut out = Vec::new();
                diff("", &actual, &expected, &mut out);
                if !out.is_empty() {
                    failures.push(format!("{name}:\n    {}", out.join("\n    ")));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} configs normalize differently:\n\n{}",
        failures.len(),
        entries.len(),
        failures.join("\n\n")
    );
}

/// The corpus reaches the branches it was built to reach.
///
/// A differential over inputs that all take the same path would pass while
/// covering nothing, so this asserts the goldens actually contain the
/// distinguishing outputs — one per hard branch. It guards the *corpus*, not
/// the port.
#[test]
fn the_corpus_exercises_the_branches_it_claims_to() {
    let dir = oracle_dir().join("golden");
    let load = |name: &str| -> Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap()
    };

    // MLA head_dim synthesis fires — and is gated on model_type.
    assert_eq!(load("synthetic--deepseek-v2")["head_dim"], 64);
    assert_eq!(load("synthetic--mla-shape-unlisted-arch")["head_dim"], 16);
    // The outer name wins over the inner `kimi_linear`.
    assert_eq!(load("synthetic--kimi-k3")["model_type"], "kimi_k3");
    assert_eq!(
        load("synthetic--kimi-linear-standalone")["model_type"],
        "kimi_linear"
    );
    // Each layer_types source produces a distinguishable schedule.
    assert_eq!(
        load("synthetic--nemotron-h")["layer_types"],
        serde_json::json!(["mamba", "attention", "moe", "mlp"])
    );
    assert_eq!(
        load("synthetic--gemma2")["layer_types"][0],
        "sliding_attention"
    );
    assert_eq!(
        load("synthetic--gemma3")["layer_types"][2],
        "full_attention"
    );
    assert_eq!(
        load("synthetic--kimi-k3")["layer_types"][1],
        "linear_attention"
    );
    // YaRN derives its factors rather than reading them.
    assert!(
        load("synthetic--rope-yarn")["rope_mla_softmax_mscale"]
            .as_f64()
            .unwrap()
            > 1.0
    );
    assert_eq!(
        load("synthetic--rope-llama3")["rope_scaling_kind"],
        "llama3"
    );
    // Defaulting rules.
    assert_eq!(load("synthetic--defaults-bare")["num_key_value_heads"], 8);
    assert_eq!(load("synthetic--defaults-bare")["head_dim"], 16);
    assert_eq!(load("synthetic--qwen2")["attention_bias"], true);
    assert_eq!(load("synthetic--gemma2")["tie_word_embeddings"], true);
    // A real config with a per-layer intermediate_size list.
    assert!(
        !load("google--gemma-3n-E2B-it")["gemma3n_per_layer_intermediate"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    // The two keys carried purely for the Metal driver reach the descriptor
    // with values, so the comparison says something about them rather than
    // matching two defaults.
    assert_eq!(
        load("synthetic--gemma4-metal-keys")["gemma4_global_head_dim"],
        256
    );
    assert_eq!(
        load("synthetic--gemma4-metal-keys")["gemma4_double_wide_mlp"],
        true
    );
    // And the per-attention-type rope is expanded per layer, which is how a
    // reader recovers the full/sliding thetas without re-reading nested JSON.
    assert_eq!(
        load("synthetic--gemma4-metal-keys")["gemma_per_layer_rope_theta"],
        serde_json::json!([10000.0, 10000.0, 10000.0, 1000000.0])
    );

    // The nested sub-configs are populated, not null.
    assert!(load("synthetic--csm")["csm"].is_object());
    assert_eq!(
        load("synthetic--csm")["csm"]["codec"]["upsampling_ratios"],
        serde_json::json!([8, 6, 5, 4])
    );
    assert!(load("google--gemma-4-E4B-it")["gemma_vision"].is_object());
    assert!(load("Qwen--Qwen3-VL-2B-Instruct")["qwen3_vl_vision"].is_object());
}

/// The `deepseek_v4` sinks bug is reproduced, deliberately.
///
/// `config.cpp:306` sets it; `config.cpp:331` overwrites it. If someone fixes
/// the C++ without regenerating the goldens, or "fixes" the port on its own,
/// this is what says so.
#[test]
fn the_dead_deepseek_v4_sinks_assignment_is_reproduced() {
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(oracle_dir().join("golden/synthetic--deepseek-v4.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        golden["attention_has_sinks"], false,
        "the golden no longer records the overwrite — if config.cpp was fixed, \
         update this test with it"
    );

    let raw =
        std::fs::read_to_string(oracle_dir().join("corpus/synthetic--deepseek-v4.json")).unwrap();
    let cfg = pie_model_config::normalize(&serde_json::from_str(&raw).unwrap(), "dsv4").unwrap();
    assert!(
        !cfg.attention_has_sinks,
        "the port diverged from the C++ side by fixing the bug on its own"
    );
    // The rest of the DSV4 block did land, so this is the overwrite and not a
    // block that failed to run.
    assert_eq!(cfg.dsv4_o_lora_rank, 32);
}
