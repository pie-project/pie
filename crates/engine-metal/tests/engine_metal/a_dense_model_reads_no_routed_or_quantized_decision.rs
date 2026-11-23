#![cfg(target_vendor = "apple")]

//! A decision is read where a dispatch consults it (`Tuning`'s accessors),
//! so `pie tune` calls measured only what its check models read
//! (`profile::search::Line::measured`). A dense bf16 model, walked and
//! served a prompt and a decode, reads no routed (moe), recurrent (gdn) or
//! quantized (qmm, its tensor-op tiles) decision. It does read the Neural
//! Engine split's (`kernel.ane.*`): the load consults them for its MLP.
//! Control: a mixture of affine u4/64 experts (Qwen3-30B-A3B's miniature)
//! served the same reads its routed (moe) decisions, so the test sees a
//! read where one happens.

use engine_metal::serve::Seated;
use engine_metal::Lane;

use super::a_scattered_pool_serves_the_same_logits::{
    Fixture, deployed_fixture, fixture_of, load_with, word,
};

/// The decision ids `fixture`'s model reads, walked and served a prompt and
/// a decode.
fn read_by(fixture: &Fixture) -> Vec<&'static str> {
    let mut shell = load_with(fixture, 256);
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the fixed pipelines build");
    let prompt: Vec<u32> = (0..12u32).map(|t| (t * 7 + 11) % 256).collect();
    shell.open(0).expect("the slot opens");
    shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(0),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, prompt.len() as u32),
                tokens: &prompt,
            })
        }])
        .expect("the prompt serves");
    shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(prompt.len() as u32),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, 1),
                tokens: &[5],
            })
        }])
        .expect("a decode serves");
    shell.tuning().profile().read()
}

#[test]
fn a_dense_model_reads_no_routed_or_quantized_decision() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let read = read_by(&fixture_of("qwen3-micro-text", None));
    let kernel: Vec<&&str> = read.iter().filter(|id| id.starts_with("kernel.")).collect();
    eprintln!("a dense bf16 model read {kernel:?}");
    let foreign: Vec<&&str> = read
        .iter()
        .filter(|id| {
            ["kernel.moe", "kernel.gdn", "kernel.qmm", "kernel.mpp_"]
                .iter()
                .any(|prefix| id.starts_with(prefix))
        })
        .collect();
    assert!(foreign.is_empty(), "a dense bf16 model read {foreign:?}");
    let routed = read_by(&deployed_fixture("qwen3-30b-a3b-mini-u4g64-kv-bf16"));
    eprintln!(
        "a mixture read {:?}",
        routed.iter().filter(|id| id.starts_with("kernel.")).collect::<Vec<_>>()
    );
    assert!(
        routed.iter().any(|id| id.starts_with("kernel.moe")),
        "the mixture reads its routed decisions: {routed:?}"
    );
}
