//! Behavioural parity with the C++ `parse_pie_model_descriptor` —
//! gate-hf-config's read side.
//!
//! The oracle in `tests/oracle/hf_descriptor/` runs `check_descriptor.sh`'s
//! pipeline over the 56-config corpus — Rust normalize → `pie.model/1` →
//! C++ read (`descriptor.cpp`) → 134-field dump — and flattens every field
//! to a type-tagged row. This test replays the same pipeline with the PORT
//! as the reader (the normalizer is called in-process; it is the same code
//! the oracle's `descriptor` binary wraps) and requires the transcripts to
//! be byte-identical, refusals included.
//!
//! Run `tests/oracle/hf_descriptor/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++ reader's** output.
//!
//! Floats cross as f32 bit patterns: nlohmann's shortest-repr and Rust's
//! `{}` disagree on text, never on value, and the tag makes that a
//! non-question.

use std::fmt::Write as _;
use std::path::PathBuf;

use model::descriptor::{
    ATTN_HEAD_DIMS, DescriptorError, parse_pie_model_descriptor,
};
use serde_json::Value;

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x93ec90d54fe7c002;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 9608;

const SEP: char = '\u{1f}';

/// The corpus MOVED here when `crates/driver-cuda` was deleted.
///
/// It is the oracle this port was checked against — 58 real `config.json`
/// files and a transcript of what the C++ parse made of each — and it
/// outlives the C++ tree on purpose. The C++ dumper that produced the
/// transcript is beside it, unbuilt: what is being kept is the ANSWER,
/// and a recorded answer does not need its producer to still compile.
fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/hf_config_dump/corpus")
}

/// The corpus, in the oracle's (byte-lexicographic glob) order.
fn corpus() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("corpus directory")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension().and_then(|e| e.to_str()) == Some("json")).then_some(p)
        })
        .collect();
    files.sort();
    files
}

fn tag(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("b:{}", u8::from(*b)),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                format!("i:{n}")
            } else {
                // The C++ dump wrote an f32 through a double; converting the
                // parsed double back to f32 is that exact narrowing.
                #[allow(clippy::cast_possible_truncation)]
                let bits = (n.as_f64().expect("a number") as f32).to_bits();
                format!("f:{bits}")
            }
        }
        Value::String(s) => format!("s:{s}"),
        Value::Array(_) | Value::Object(_) => unreachable!("containers are walked"),
    }
}

fn walk(out: &mut String, name: &str, prefix: &str, v: &Value) {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for k in keys {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                walk(out, name, &p, &map[k]);
            }
        }
        Value::Array(items) => {
            let _ = writeln!(out, "{name}{SEP}{prefix}.len{SEP}i:{}", items.len());
            for (i, e) in items.iter().enumerate() {
                walk(out, name, &format!("{prefix}.{i}"), e);
            }
        }
        other => {
            let _ = writeln!(out, "{name}{SEP}{prefix}{SEP}{}", tag(other));
        }
    }
}

/// Normalize a corpus config into its descriptor — the same call the
/// oracle's `descriptor` binary makes.
fn descriptor_of(path: &std::path::Path) -> Value {
    let raw = std::fs::read_to_string(path).expect("corpus config");
    let root: Value = serde_json::from_str(&raw).expect("corpus config parses");
    model::config::descriptor(&root, path.to_str().expect("utf-8 path"))
        .expect("the corpus normalizes")
}

fn refusal_row(out: &mut String, case: &str, err: &DescriptorError) {
    let _ = writeln!(out, "{case}{SEP}error{SEP}{}", err.0);
}

fn transcript() -> String {
    let mut out = String::new();
    let corpus = corpus();
    for path in &corpus {
        let name = path.file_name().and_then(|n| n.to_str()).expect("file name");
        let desc = descriptor_of(path);
        let cfg = parse_pie_model_descriptor(&desc.to_string())
            .expect("the corpus descriptors read back");
        let dumped = serde_json::to_value(&cfg).expect("HfConfig serializes");
        walk(&mut out, name, "", &dumped);
    }

    // The refusal cases, built from the first corpus descriptor.
    let base = descriptor_of(&corpus[0]);

    let mut v2 = base.clone();
    v2["version"] = Value::String("pie.model/2".into());
    let err = parse_pie_model_descriptor(&v2.to_string()).expect_err("foreign version refuses");
    refusal_row(&mut out, "refuse-version", &err);

    let mut missing = base.clone();
    missing.as_object_mut().expect("object").remove("hidden_size");
    let err = parse_pie_model_descriptor(&missing.to_string()).expect_err("missing key refuses");
    refusal_row(&mut out, "refuse-missing", &err);

    let mut rope = base;
    rope["rope_scaling_kind"] = Value::String("yarn_v3".into());
    let err = parse_pie_model_descriptor(&rope.to_string()).expect_err("unknown rope refuses");
    refusal_row(&mut out, "refuse-rope", &err);

    out
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_port_reproduces_the_cpp_transcript() {
    let text = transcript();
    let rows = text.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — corpus or shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("hf_descriptor_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}

/// [`ATTN_HEAD_DIMS`] restates `kernels.def`'s `PIE_ATTN_HEAD_DIM` rows —
/// a build property of the kernels crate the reader recomputes
/// `head_dim_kernel` from. The restatement is checked against the `.def`
/// rather than believed: a head dim added there and not here would round
/// Phi-3's 96 differently on the two sides of the rewrite.
#[test]
fn the_head_dim_list_matches_kernels_def() {
    let def = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kernels-cuda/csrc/src/kernels.def"),
    )
    .expect("kernels.def");
    let mut dims: Vec<i32> = def
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("PIE_ATTN_HEAD_DIM("))
        .filter_map(|l| {
            l.strip_prefix("PIE_ATTN_HEAD_DIM(")?
                .split(')')
                .next()?
                .parse::<i32>()
                .ok()
        })
        .collect();
    dims.sort_unstable();
    dims.dedup();
    assert!(
        dims.len() >= 4,
        "the scan found {} head dims, so its shape assumption broke",
        dims.len()
    );
    assert_eq!(dims, ATTN_HEAD_DIMS, "kernels.def and ATTN_HEAD_DIMS diverged");
}
