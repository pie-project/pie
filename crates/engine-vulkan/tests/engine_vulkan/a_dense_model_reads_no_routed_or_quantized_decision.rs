//! A decision is read where a dispatch consults it (`Tuning`'s accessors),
//! so `pie tune` calls measured only what its check models read
//! (`profile::search::Line::measured`). A dense bf16 model, loaded and
//! served a prompt and decodes, reads no routed (`kernel.moe.*`) or
//! quantized (`kernel.qmm`) decision. Control: Qwen3-A3B's mixture
//! miniature with affine u4/64 weights (a routed matmul batches over sorted
//! tiles only where its experts are quantized) served the same reads its
//! routed decisions, so the test sees a read where one happens.

use poem::Dtype;

use crate::model_logits::{CONTEXT, Fixture, fixture, fixture_deployed, load, serve_seats};

/// The decision ids `fixture`'s model reads, loaded and served; `None`
/// where no device comes up.
fn read_by(fixture: &Fixture) -> Option<Vec<&'static str>> {
    let mut shell = load(&fixture, CONTEXT)?;
    serve_seats(&mut shell, &fixture.trace, 2);
    Some(shell.tuning().profile().read())
}

const MIXTURE: &str = "qwen3-a3b-micro";

#[test]
fn a_dense_model_reads_no_routed_or_quantized_decision() {
    let Some(read) = read_by(&fixture("qwen3-micro-text", None)) else {
        return;
    };
    eprintln!("a dense bf16 model read {read:?}");
    let foreign: Vec<&&str> = read
        .iter()
        .filter(|id| id.starts_with("kernel.moe") || id.starts_with("kernel.qmm"))
        .collect();
    assert!(foreign.is_empty(), "a dense bf16 model read {foreign:?}");
    let package = poem_compiler::catalog::package_of(MIXTURE).expect("the model is listed");
    let deploy = poem_compiler::catalog::one_rank(Dtype::U4g64, Dtype::Bf16);
    let mixture = fixture_deployed(package, MIXTURE, &deploy, "-u4g64");
    let routed = read_by(&mixture).expect("the device came up once");
    eprintln!("a u4/64 mixture read {routed:?}");
    assert!(
        routed.iter().any(|id| id.starts_with("kernel.moe")),
        "the mixture reads its routed decisions: {routed:?}"
    );
}
