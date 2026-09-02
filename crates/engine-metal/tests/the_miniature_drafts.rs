//! **THE DRAFT HEAD IS THERE, AND THE TRUNK DOES NOT NOTICE IT.**
//!
//! `dsv4-flash-mtp` is the miniature with DeepSeek-V4-Flash's own `nextn`
//! head over it — `mlx-community/DeepSeek-V4-Flash-MTP-bf16`, restated by
//! `scripts/dsv4_mtp_companion.py` and landed by `pie model import <mini>
//! --aux <overlay>`. This file asks the two things an engine can ask before
//! the runtime's verify loop is pointed at it:
//!
//! 1. the load advertises a draft head (`Shell::drafts`), and a lane that
//!    asks for drafts fires — prefill and decode — with finite logits;
//! 2. the TRUNK's logits with the draft arm running are the trunk's logits
//!    without it, bit for bit. The head reads the streams and writes its own
//!    seam; a trunk that changed under it would be a draft window leaking
//!    into the readout.
//!
//! Acceptance is the runtime's to measure (`tests/inferlets/test_eagle.py`).
//! `PIE_DSV4_MTP_ARTIFACT` names the artifact; unset, the test looks in
//! `/tmp/warmstream/dsv4-mini-mtp.zt` and skips by name when it is absent.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_miniature_drafts -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "dsv4-flash-mtp-u4g64-u2g64-mxfp4-kv-bf16";
const PROMPT: &[u32] = &[0, 671, 6102, 294, 8760, 344, 270, 4593, 294];
const STEPS: usize = 6;

fn artifact() -> Option<PathBuf> {
    let path = std::env::var("PIE_DSV4_MTP_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/warmstream/dsv4-mini-mtp.zt"));
    path.is_file().then_some(path)
}

fn word(query_len: u32, drafts: bool) -> u64 {
    models::deepseek_v4::forward::Facts::of(&Request::new(query_len, false).drafting(drafts)).word()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "{what} produced a non-finite logit"
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(spread > 1e-3, "{what} logits span {spread}, which nothing wrote");
}

/// Prefill the prompt and decode `STEPS` greedy tokens in `slot`, every
/// fire asking for drafts or not — the rows of trunk logits.
fn run(shell: &mut Shell, slot: u32, drafts: bool) -> Vec<Vec<f32>> {
    shell.open(slot).expect("the slot opens");
    let mut rows = Vec::with_capacity(STEPS + 1);
    let got = shell
        .fire(&[Lane {
            slot,
            word: word(PROMPT.len() as u32, drafts),
            tokens: PROMPT,
        }])
        .expect("the prefill fires");
    rows.push(got.into_iter().next().expect("one row"));
    for step in 0..STEPS {
        let fed = [argmax(rows.last().expect("a row"))];
        let got = shell
            .fire(&[Lane {
                slot,
                word: word(1, drafts),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} (drafts={drafts}) fires: {why}"));
        rows.push(got.into_iter().next().expect("one row"));
    }
    rows
}

#[test]
fn the_draft_head_fires_and_the_trunk_is_unchanged() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!("not asked: no dsv4 mtp artifact (PIE_DSV4_MTP_ARTIFACT, or /tmp/warmstream/dsv4-mini-mtp.zt)");
        return;
    };
    let sku = models::sku(SKU).expect("the catalog ships the drafting mini row");
    let trace = (sku.trace)(Platform::Metal);
    let source = ztensor_compat::index(&artifact).expect("the artifact opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Metal)
        .unwrap_or_else(|why| panic!("the artifact holds every plane of {SKU}: {why}"));
    drop(source);

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &artifact,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        pages: (4) * (512) / (16),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the drafting shell loads");
    eprintln!(
        "loaded {SKU} in {:.1}s; drafts advertised: {}",
        booted.elapsed().as_secs_f64(),
        shell.drafts()
    );
    assert!(shell.drafts(), "the load declares a draft head and advertises none");
    assert_eq!(
        shell.mtp_depth(),
        models::deepseek_v4::model::DRAFT_DEPTH,
        "the load's token plane is as deep as the text chains"
    );

    let plain = run(&mut shell, 0, false);
    let drafted = run(&mut shell, 1, true);
    for (step, (a, b)) in plain.iter().zip(drafted.iter()).enumerate() {
        finite(a, &format!("plain step {step}"));
        finite(b, &format!("drafting step {step}"));
        assert_eq!(
            a, b,
            "the trunk's logits at step {step} changed when the draft head ran beside it"
        );
    }
    let tokens: Vec<u32> = drafted.iter().map(|r| argmax(r)).collect();
    eprintln!("drafting run tokens {tokens:?} — identical to the plain run, {STEPS} decodes");
}
