//! **THE BLOCK DRAFTER'S PLANES BIND, AND THE TRUNK DOES NOT NOTICE THEM.**
//!
//! `qwen36-27b-dflash` is `mlx-community/Qwen3.6-27B-4bit` with
//! `z-lab/Qwen3.6-27B-DFlash` landed over it by `pie model import <target>
//! --aux <drafter>`. Unlike the chained heads, this drafter is five decoder
//! layers of its OWN geometry (32 q heads / 8 kv / head dim 128 against the
//! trunk's 24 / 4 / 256) fed by a fusion of five tapped trunk hidden states,
//! and it runs over rows the trunk is guarded away from.
//!
//! This asks the two things an engine can ask before an inferlet drives a
//! draft pass at it:
//!
//! 1. the artifact loads — every one of the drafter's 58 planes binds, its
//!    five kv rows are carved, and the load advertises a draft head;
//! 2. **the trunk's logits with the drafter's context arm running are the
//!    trunk's logits without it, bit for bit.** That arm fuses five taps and
//!    writes ten projections into the drafter's kv rows on every drafting
//!    fire; a trunk that moved under it would mean the fusion had reached
//!    the residual stream, which is exactly the bug the two-stream reading
//!    of this architecture invites.
//!
//! What is NOT asked here is the draft pass itself: its rows carry a
//! bidirectional block mask, which is the guest's to state, so it belongs to
//! the inferlet's test rather than this one.
//!
//! ```text
//! PIE_DFLASH_ARTIFACT=~/.pie/models/<hash>/<hash>.qwen36-27b-dflash-u4g64-kv-bf16.metal.zt \
//!   cargo test -p engine-metal --release --test the_block_drafter_loads_and_leaves_the_trunk_alone -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen36-27b-dflash-u4g64-kv-bf16";
const PROMPT: &[u32] = &[9707, 11, 847, 829, 374, 264, 1602, 2613, 3364];
const STEPS: usize = 4;

/// The artifact, named or found in the store by the SKU its name carries.
fn artifact() -> Option<PathBuf> {
    if let Ok(named) = std::env::var("PIE_DFLASH_ARTIFACT") {
        let path = PathBuf::from(shellexpand(&named));
        return path.is_file().then_some(path);
    }
    let store = PathBuf::from(std::env::var("HOME").ok()?).join(".pie/models");
    for entry in std::fs::read_dir(store).ok()?.flatten() {
        for file in std::fs::read_dir(entry.path()).ok()?.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            if name.contains(SKU) && name.ends_with(".zt") {
                return Some(file.path());
            }
        }
    }
    None
}

fn shellexpand(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => path.to_string(),
    }
}

fn word(query_len: u32, drafts: bool) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false).drafting(drafts)).word()
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

/// Prefill and decode `STEPS` greedy tokens in `slot`, every fire asking for
/// the drafter's context arm or not.
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
fn the_drafters_planes_bind_and_its_context_arm_moves_no_trunk_logit() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!("not asked: no dflash artifact (PIE_DFLASH_ARTIFACT, or one in ~/.pie/models)");
        return;
    };
    let sku = models::sku(SKU).expect("the catalog ships the block-drafter row");
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
        pages: 4 * 512 / 16,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the block drafter's shell loads");
    eprintln!(
        "loaded {SKU} in {:.1}s; drafts advertised: {}, depth {}",
        booted.elapsed().as_secs_f64(),
        shell.drafts(),
        shell.mtp_depth(),
    );
    assert!(
        shell.drafts(),
        "the load carries a block drafter and advertises no draft head"
    );

    let plain = run(&mut shell, 0, false);
    let drafted = run(&mut shell, 1, true);
    for (step, (a, b)) in plain.iter().zip(drafted.iter()).enumerate() {
        finite(a, &format!("plain step {step}"));
        finite(b, &format!("drafting step {step}"));
        assert_eq!(
            a, b,
            "the trunk's logits at step {step} moved when the drafter's context arm ran \
             beside it — the fusion has reached the residual stream"
        );
    }
    let tokens: Vec<u32> = drafted.iter().map(|r| argmax(r)).collect();
    eprintln!("drafting run tokens {tokens:?} — identical to the plain run over {STEPS} decodes");
}
