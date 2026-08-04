//! The JSON accessors `config.cpp` reads HuggingFace configs through.
//!
//! Deliberately a transliteration of `hf_config_json.hpp` rather than idiomatic
//! serde. The normalizer's behavior is defined by these accessors' quirks — a
//! `null` being indistinguishable from an absent key, an `int` satisfying a
//! `float` read, a type mismatch throwing rather than defaulting — and a
//! rewrite in a more natural style would change results in ways no reviewer
//! could see. `tests/differential.rs` is what says this transliteration is
//! faithful.

use anyhow::{Result, bail};
use serde_json::Value;

/// The object the text tower's hyperparameters live in.
///
/// Multimodal checkpoints nest them under `text_config` (or `llm_config`) while
/// keeping the architecture name and a stub `model_type` at the top level.
pub fn text_config_view(root: &Value) -> &Value {
    for key in ["text_config", "llm_config"] {
        if let Some(nested @ Value::Object(_)) = root.get(key) {
            return nested;
        }
    }
    root
}

/// `model_type`, or empty when absent or null.
pub fn model_type(value: &Value) -> String {
    match value.get("model_type") {
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Whether `rope_parameters` is keyed by attention type rather than flat.
///
/// Gemma-4 nests one entry per layer type; everything else stores the
/// parameters directly. The test is the shape of the first value.
pub fn is_nested_rope_parameters(rope_parameters: &Value) -> bool {
    match rope_parameters {
        Value::Object(map) => map.values().next().is_some_and(Value::is_object),
        _ => false,
    }
}

/// `rope_parameters`, only when it is the flat kind.
pub fn flat_rope_parameters(text: &Value) -> Option<&Value> {
    match text.get("rope_parameters") {
        Some(v @ Value::Object(_)) if !is_nested_rope_parameters(v) => Some(v),
        _ => None,
    }
}

/// Where RoPE scaling is configured: `rope_scaling`, else flat
/// `rope_parameters`.
pub fn flat_rope_config(text: &Value) -> Option<&Value> {
    match text.get("rope_scaling") {
        Some(v @ Value::Object(_)) => Some(v),
        _ => flat_rope_parameters(text),
    }
}

/// A key that must be present. Missing is an error; a *wrong type* is too.
pub fn require_i32(value: &Value, key: &str, path: &str) -> Result<i32> {
    match value.get(key) {
        None => bail!("config.json ({path}): missing key '{key}'"),
        Some(found) => as_i32(found, key, path),
    }
}

/// A key that may be absent. **`null` counts as absent** — that is
/// `json_get_opt`'s rule, and HF relies on it (`final_logit_softcapping: null`
/// means "no cap", not "cap of zero").
fn present<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value.get(key) {
        None | Some(Value::Null) => None,
        Some(found) => Some(found),
    }
}

fn as_i32(found: &Value, key: &str, path: &str) -> Result<i32> {
    // A float satisfies an int read by truncation, as `get<int>()` does.
    if let Some(n) = found.as_i64() {
        return Ok(n as i32);
    }
    if let Some(n) = found.as_f64() {
        return Ok(n as i32);
    }
    bail!("config.json ({path}): '{key}' is not a number")
}

pub fn opt_i32(value: &Value, key: &str, default: i32, path: &str) -> Result<i32> {
    match present(value, key) {
        None => Ok(default),
        Some(found) => as_i32(found, key, path),
    }
}

pub fn opt_f32(value: &Value, key: &str, default: f32, path: &str) -> Result<f32> {
    match present(value, key) {
        None => Ok(default),
        Some(found) => match found.as_f64() {
            // The C++ side stores these as `float`, so the narrowing is part
            // of the observable result and happens here rather than later.
            Some(n) => Ok(n as f32),
            None => bail!("config.json ({path}): '{key}' is not a number"),
        },
    }
}

pub fn opt_bool(value: &Value, key: &str, default: bool, path: &str) -> Result<bool> {
    match present(value, key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => bail!("config.json ({path}): '{key}' is not a boolean"),
    }
}

pub fn opt_string(value: &Value, key: &str, default: &str, path: &str) -> Result<String> {
    match present(value, key) {
        None => Ok(default.to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => bail!("config.json ({path}): '{key}' is not a string"),
    }
}

/// An array of ints, or `None` when the key is absent or not an array.
pub fn opt_i32_array(value: &Value, key: &str, path: &str) -> Result<Option<Vec<i32>>> {
    let Some(Value::Array(items)) = value.get(key) else {
        return Ok(None);
    };
    items
        .iter()
        .map(|item| as_i32(item, key, path))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// An array of floats, or `None` when the key is absent or not an array.
pub fn opt_f32_array(value: &Value, key: &str, path: &str) -> Result<Option<Vec<f32>>> {
    let Some(Value::Array(items)) = value.get(key) else {
        return Ok(None);
    };
    items
        .iter()
        .map(|item| match item.as_f64() {
            Some(n) => Ok(n as f32),
            None => bail!("config.json ({path}): '{key}' holds a non-number"),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// An array of strings, or `None` when the key is absent or not an array.
pub fn opt_string_array(value: &Value, key: &str, path: &str) -> Result<Option<Vec<String>>> {
    let Some(Value::Array(items)) = value.get(key) else {
        return Ok(None);
    };
    items
        .iter()
        .map(|item| match item {
            Value::String(s) => Ok(s.clone()),
            _ => bail!("config.json ({path}): '{key}' holds a non-string"),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// A nested object, or `None` when the key is absent or not one.
pub fn object<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value.get(key) {
        Some(found @ Value::Object(_)) => Some(found),
        _ => None,
    }
}
