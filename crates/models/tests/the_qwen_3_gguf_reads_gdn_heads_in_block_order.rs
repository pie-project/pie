//! A GGUF's GDN layers read into the order the scan pairs heads by: `in_ba`
//! is `[beta, alpha]`, beta first, and a "v-grouped" GGUF's tiled v-heads are
//! gathered back into block order.

mod sparse_gguf;

use poem::Dtype;
use sparse_gguf::{Kv, Scratch, reads};

fn arch() -> (&'static str, Kv) {
    ("general.architecture", Kv::Str("qwen35".into()))
}

/// The recorded read of `weight`.
fn read_of<'a>(log: &'a [String], weight: &str) -> &'a str {
    log.iter()
        .find(|r| r.contains(&format!("name: \"{weight}\"")))
        .unwrap_or_else(|| panic!("no read of `{weight}`"))
}

#[test]
fn in_ba_concatenates_beta_first() {
    let scratch = Scratch::new("qwen3-in-ba");
    let log = reads(&scratch.0, "qwen35-tiny", Dtype::Bf16, &[arch()]);
    let read = read_of(&log, "layer.0.in_ba");
    let beta = read
        .find("blk.0.ssm_beta.weight")
        .expect("in_ba reads ssm_beta");
    let alpha = read
        .find("blk.0.ssm_alpha.weight")
        .expect("in_ba reads ssm_alpha");
    assert!(beta < alpha, "in_ba lands beta first: {read}");
}

#[test]
fn a_v_grouped_gguf_gathers_its_tiled_heads_into_block_order() {
    let scratch = Scratch::new("qwen3-v-grouped");
    let plain = reads(&scratch.0, "qwen35-tiny", Dtype::Bf16, &[arch()]);
    assert!(
        !read_of(&plain, "layer.0.dt_bias").contains("Gather"),
        "a block-stored GGUF is read as stored"
    );
    let grouped = reads(
        &scratch.0,
        "qwen35-tiny",
        Dtype::Bf16,
        &[arch(), ("prism.hadamard.gdn_v_grouped", Kv::Bool(true))],
    );
    // qwen35-tiny: 8 v-heads over 4 k-heads, two each. Block position `p`
    // draws tiled head `(p % 2) * 4 + p / 2`.
    let read = read_of(&grouped, "layer.0.dt_bias");
    assert!(
        read.contains("indices: [0, 4, 1, 5, 2, 6, 3, 7]"),
        "the tiled v-heads are gathered into block order: {read}"
    );
}
