//! The planner profile cache's key: what makes two calibration runs
//! comparable.
//!
//! The winning lattice shape is written to `~/.cache` and looked up on later
//! boots; the key must capture everything that changes the answer and nothing
//! that drifts. So `budget_bytes` is excluded — it moves a few MiB between
//! boots — and lives on the stored shape ([`ProfileShape::budget_bytes`]).

use std::fmt::Write as _;

/// Bump whenever the meaning of a stored field changes.
///
/// A document not carrying this exact version is refused rather than partially
/// interpreted.
pub const SCHEMA_VERSION: i32 = 2;

/// Everything that must match for a cached plan to apply.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProfileKey {
    /// `cudaDeviceProp::name`, verbatim.
    pub gpu_name: String,
    /// Compute capability major version.
    pub compute_major: i32,
    /// Compute capability minor version.
    pub compute_minor: i32,
    /// `multiProcessorCount`.
    pub sm_count: i32,
    /// The resolved KV format's name, not the config alias.
    pub kv_cache_dtype: String,
    /// Tensor-parallel rank count.
    pub tp_size: i32,
    /// The checkpoint's architecture family.
    pub model_type: String,
    /// Model hidden size.
    pub hidden_size: i32,
    /// Layer count.
    pub num_hidden_layers: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// KV head count, unsharded.
    pub num_key_value_heads: i32,
    /// The kernel-facing head dim (`head_dim_kernel`).
    pub head_dim: i32,
}

/// A cached planner result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProfileShape {
    /// The profile family that won the sweep.
    pub policy_profile: String,
    /// KV page size the winning layout used.
    pub kv_page_size: i32,
    /// Token capacity of the winning forward buffer.
    pub max_forward_tokens: i32,
    /// Request capacity of the winning forward buffer.
    pub max_forward_requests: i32,
    /// The planner budget the sweep ran inside.
    ///
    /// Not part of [`ProfileKey`] (it drifts by a few MiB between boots and
    /// would miss every time); recorded so a reader can decide the entry no
    /// longer applies when VRAM pressure or a requantised checkpoint changes
    /// the memory situation the key can't see.
    pub budget_bytes: u64,
}

/// A field as it was found in a stored document.
///
/// Matching is JSON-type-strict: a `sm_count` stored as `"132"` or `132.0`
/// does not match the integer `132`. This enum keeps that distinction visible.
#[derive(Debug, Clone, PartialEq)]
pub enum StoredField {
    /// The field is absent from the document.
    Missing,
    /// Present as JSON `null`.
    Null,
    /// Present as a JSON boolean.
    Bool(bool),
    /// Present as a JSON integer.
    Int(i64),
    /// Present as a JSON number that is not an integer.
    Float(f64),
    /// Present as a JSON string.
    Str(String),
}

impl StoredField {
    fn matches_str(&self, expected: &str) -> bool {
        matches!(self, StoredField::Str(s) if s == expected)
    }

    fn matches_int(&self, expected: i32) -> bool {
        matches!(self, StoredField::Int(v) if *v == i64::from(expected))
    }
}

/// The twelve key fields.
pub const KEY_FIELDS: [&str; 12] = [
    "gpu_name",
    "compute_major",
    "compute_minor",
    "sm_count",
    "kv_cache_dtype",
    "tp_size",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
];

impl ProfileKey {
    /// Does a stored key object describe the same configuration?
    ///
    /// Every field must be present, of the right JSON type, and equal; a
    /// missing field is a mismatch, not a wildcard.
    pub fn matches(&self, lookup: impl Fn(&str) -> StoredField) -> bool {
        lookup("gpu_name").matches_str(&self.gpu_name)
            && lookup("compute_major").matches_int(self.compute_major)
            && lookup("compute_minor").matches_int(self.compute_minor)
            && lookup("sm_count").matches_int(self.sm_count)
            && lookup("kv_cache_dtype").matches_str(&self.kv_cache_dtype)
            && lookup("tp_size").matches_int(self.tp_size)
            && lookup("model_type").matches_str(&self.model_type)
            && lookup("hidden_size").matches_int(self.hidden_size)
            && lookup("num_hidden_layers").matches_int(self.num_hidden_layers)
            && lookup("num_attention_heads").matches_int(self.num_attention_heads)
            && lookup("num_key_value_heads").matches_int(self.num_key_value_heads)
            && lookup("head_dim").matches_int(self.head_dim)
    }

    /// The key as canonical JSON.
    ///
    /// Keys are emitted alphabetically, not in declaration order, to match
    /// `nlohmann`'s sorted `std::map` output. Serialised by hand so the key
    /// order can't be flipped by a sibling crate enabling `preserve_order`.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut o = String::with_capacity(256);
        o.push('{');
        let mut fields: Vec<(&str, Field<'_>)> = vec![
            ("gpu_name", Field::Str(&self.gpu_name)),
            ("compute_major", Field::Int(self.compute_major)),
            ("compute_minor", Field::Int(self.compute_minor)),
            ("sm_count", Field::Int(self.sm_count)),
            ("kv_cache_dtype", Field::Str(&self.kv_cache_dtype)),
            ("tp_size", Field::Int(self.tp_size)),
            ("model_type", Field::Str(&self.model_type)),
            ("hidden_size", Field::Int(self.hidden_size)),
            ("num_hidden_layers", Field::Int(self.num_hidden_layers)),
            ("num_attention_heads", Field::Int(self.num_attention_heads)),
            ("num_key_value_heads", Field::Int(self.num_key_value_heads)),
            ("head_dim", Field::Int(self.head_dim)),
        ];
        fields.sort_by_key(|(name, _)| *name);
        for (i, (name, value)) in fields.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write_json_string(&mut o, name);
            o.push(':');
            match value {
                Field::Str(s) => write_json_string(&mut o, s),
                Field::Int(v) => {
                    let _ = write!(o, "{v}");
                }
            }
        }
        o.push('}');
        o
    }
}

enum Field<'a> {
    Str(&'a str),
    Int(i32),
}

/// Escape a string the way `nlohmann::json::dump()` does.
///
/// Does not escape `/`, and leaves non-ASCII UTF-8 as literal bytes rather
/// than `\u` pairs — both are observable in the cache file.
fn write_json_string(o: &mut String, s: &str) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ProfileKey {
        ProfileKey {
            gpu_name: "NVIDIA H100 80GB HBM3".into(),
            compute_major: 9,
            compute_minor: 0,
            sm_count: 132,
            kv_cache_dtype: "bf16".into(),
            tp_size: 1,
            model_type: "llama".into(),
            hidden_size: 8192,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
        }
    }

    /// A lookup that agrees with `key()` on every field.
    fn exact(k: &ProfileKey) -> impl Fn(&str) -> StoredField + '_ {
        move |name| match name {
            "gpu_name" => StoredField::Str(k.gpu_name.clone()),
            "kv_cache_dtype" => StoredField::Str(k.kv_cache_dtype.clone()),
            "model_type" => StoredField::Str(k.model_type.clone()),
            "compute_major" => StoredField::Int(k.compute_major.into()),
            "compute_minor" => StoredField::Int(k.compute_minor.into()),
            "sm_count" => StoredField::Int(k.sm_count.into()),
            "tp_size" => StoredField::Int(k.tp_size.into()),
            "hidden_size" => StoredField::Int(k.hidden_size.into()),
            "num_hidden_layers" => StoredField::Int(k.num_hidden_layers.into()),
            "num_attention_heads" => StoredField::Int(k.num_attention_heads.into()),
            "num_key_value_heads" => StoredField::Int(k.num_key_value_heads.into()),
            "head_dim" => StoredField::Int(k.head_dim.into()),
            _ => StoredField::Missing,
        }
    }

    #[test]
    fn an_exact_document_matches() {
        let k = key();
        assert!(k.matches(exact(&k)));
    }

    #[test]
    fn every_field_is_load_bearing() {
        let k = key();
        for dropped in KEY_FIELDS {
            let base = exact(&k);
            let lookup = |name: &str| {
                if name == dropped {
                    StoredField::Missing
                } else {
                    base(name)
                }
            };
            assert!(
                !k.matches(lookup),
                "{dropped} is missing and the key still matched"
            );
        }
    }

    #[test]
    fn a_number_stored_as_a_string_does_not_match() {
        let k = key();
        let base = exact(&k);
        let lookup = |n: &str| {
            if n == "sm_count" {
                StoredField::Str("132".into())
            } else {
                base(n)
            }
        };
        assert!(!k.matches(lookup));
    }

    #[test]
    fn a_number_stored_as_a_float_does_not_match() {
        let k = key();
        let base = exact(&k);
        let lookup = |n: &str| {
            if n == "sm_count" {
                StoredField::Float(132.0)
            } else {
                base(n)
            }
        };
        assert!(!k.matches(lookup), "132.0 is not is_number_integer()");
    }

    #[test]
    fn nulls_and_bools_never_match_anything() {
        let k = key();
        for field in KEY_FIELDS {
            for bad in [
                StoredField::Null,
                StoredField::Bool(true),
                StoredField::Bool(false),
            ] {
                let base = exact(&k);
                let lookup = |n: &str| if n == field { bad.clone() } else { base(n) };
                assert!(!k.matches(lookup), "{field} as {bad:?} matched");
            }
        }
    }

    #[test]
    fn a_string_stored_as_a_number_does_not_match() {
        let k = key();
        let base = exact(&k);
        let lookup = |n: &str| {
            if n == "model_type" {
                StoredField::Int(0)
            } else {
                base(n)
            }
        };
        assert!(!k.matches(lookup));
    }

    #[test]
    fn json_keys_come_out_alphabetically_not_in_declaration_order() {
        let json = key().to_json();
        let mut expected = KEY_FIELDS;
        expected.sort_unstable();
        let mut at = 0;
        for name in expected {
            let needle = format!("\"{name}\":");
            let found = json[at..].find(&needle).map(|i| i + at);
            assert!(found.is_some(), "{name} missing or out of order in {json}");
            at = found.unwrap();
        }
        assert!(json.starts_with("{\"compute_major\":"), "{json}");
        assert!(json.ends_with("\"tp_size\":1}"), "{json}");
    }

    #[test]
    fn json_has_no_spaces_matching_a_compact_dump() {
        let json = key().to_json();
        assert!(!json.contains(", "), "{json}");
        assert!(!json.contains(": "), "{json}");
    }

    #[test]
    fn strings_are_escaped_the_way_nlohmann_escapes_them() {
        let k = ProfileKey {
            gpu_name: "a\"b\\c\nd\te\u{1}f".into(),
            ..Default::default()
        };
        let json = k.to_json();
        assert!(
            json.contains(r#""gpu_name":"a\"b\\c\nd\te\u0001f""#),
            "{json}"
        );
    }

    #[test]
    fn a_forward_slash_is_not_escaped_and_utf8_stays_literal() {
        let k = ProfileKey {
            gpu_name: "a/b".into(),
            model_type: "日本".into(),
            ..Default::default()
        };
        let json = k.to_json();
        assert!(json.contains(r#""gpu_name":"a/b""#), "{json}");
        assert!(json.contains(r#""model_type":"日本""#), "{json}");
    }

    #[test]
    fn negative_numbers_round_trip() {
        let k = ProfileKey {
            compute_major: -1,
            ..Default::default()
        };
        assert!(k.to_json().contains(r#""compute_major":-1"#));
    }

    #[test]
    fn the_schema_version_is_the_one_the_cpp_writes() {
        assert_eq!(SCHEMA_VERSION, 2);
    }
}
