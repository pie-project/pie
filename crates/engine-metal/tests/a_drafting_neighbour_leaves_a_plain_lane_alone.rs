//! **A PLAIN LANE'S LOGITS DO NOT CHANGE WHEN A DRAFTING LANE SHARES ITS
//! FIRE.** The draft arm runs over the drafting rows only (the `drafts` fact
//! splits every node it touches), so a lane that asked for nothing must read
//! the same trunk logits whether the neighbours beside it drafted or not —
//! bit for bit, at the SAME fire width.
//!
//! The width is held because it is its own variable: on the two-bit routed
//! path a five-row fire and a two-row fire take different kernel tilings and
//! part by up to ~1.4 logits at the readout (measured here, four plain lanes
//! beside one against one plain lane beside one — no head anywhere). That is
//! the numerics floor a crowd gate has to read against; what this file
//! asserts is that a head in the crowd adds nothing to it.
//!
//! `PIE_DSV4_MTP_ARTIFACT` names the artifact; unset, `/tmp/warmstream/
//! dsv4-mini-mtp.zt`, skipping by name when absent.

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "dsv4-flash-mtp-u4g64-u2g64-mxfp4-kv-bf16";
const PROMPT_A: &[u32] = &[0, 671, 6102, 294, 8760, 344, 270, 4593, 294];
const PROMPT_B: &[u32] = &[0, 1357, 14, 982, 295, 811, 671];
const STEPS: usize = 10;

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

fn load(artifact: &PathBuf) -> Shell {
    let sku = models::sku(SKU).expect("the catalog ships the drafting mini row");
    let trace = (sku.trace)(Platform::Metal);
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Metal)
        .unwrap_or_else(|why| panic!("the artifact holds every plane of {SKU}: {why}"));
    drop(source);
    Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: artifact,
        budget: Budget::new(14, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 14,
        pages: (14) * (512) / (16),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the drafting shell loads")
}

/// Decode `STEPS` greedy tokens in lane A (plain) with, beside it in every
/// fire, lane B either plain or drafting. Returns lane A's logits rows.
fn plain_beside(shell: &mut Shell, slot_a: u32, slot_b: u32, b_drafts: bool) -> Vec<Vec<f32>> {
    shell.open(slot_a).expect("slot a opens");
    shell.open(slot_b).expect("slot b opens");
    let got = shell
        .fire(&[
            Lane { slot: slot_a, word: word(PROMPT_A.len() as u32, false), tokens: PROMPT_A },
            Lane { slot: slot_b, word: word(PROMPT_B.len() as u32, b_drafts), tokens: PROMPT_B },
        ])
        .expect("the prefill fires");
    let mut rows_a = vec![got[0].clone()];
    let mut last_b = got[1].clone();
    for step in 0..STEPS {
        let fed_a = [argmax(rows_a.last().expect("a row"))];
        let fed_b = [argmax(&last_b)];
        let got = shell
            .fire(&[
                Lane { slot: slot_a, word: word(1, false), tokens: &fed_a },
                Lane { slot: slot_b, word: word(1, b_drafts), tokens: &fed_b },
            ])
            .unwrap_or_else(|why| panic!("decode step {step} (b drafts={b_drafts}) fires: {why}"));
        rows_a.push(got[0].clone());
        last_b = got[1].clone();
    }
    rows_a
}

#[test]
fn a_plain_lane_reads_the_same_logits_beside_a_drafting_one() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!("not asked: no dsv4 mtp artifact");
        return;
    };
    let mut shell = load(&artifact);
    assert!(shell.drafts());

    // Two rows a fire: one neighbour, plain or drafting.
    let beside_plain = plain_beside(&mut shell, 0, 1, false);
    let beside_drafting = plain_beside(&mut shell, 2, 3, true);
    compare("one drafting lane (against one plain lane)", &beside_plain, &beside_drafting);
    // Five rows a fire: the runtime's window of four drafting one-token
    // lanes, against four plain ones — the same shape with no head in it.
    let beside_crowd = plain_beside_window(&mut shell, 4, &[5, 6, 7, 8], false);
    let beside_window = plain_beside_window(&mut shell, 9, &[10, 11, 12, 13], true);
    let floor = widest_gap(&beside_plain, &beside_crowd);
    eprintln!("the width floor, four plain lanes beside one against one beside one: {floor:.4}");
    compare("a four-lane drafting window (against four plain lanes)", &beside_crowd, &beside_window);
}

fn widest_gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()))
        .fold(0f32, f32::max)
}

/// Lane A plain, decoding as in [`plain_beside`], beside `slots.len()`
/// drafting lanes that each carry one token per fire (the runtime's window).
fn plain_beside_window(shell: &mut Shell, slot_a: u32, slots: &[u32], drafting: bool) -> Vec<Vec<f32>> {
    shell.open(slot_a).expect("slot a opens");
    for &slot in slots {
        shell.open(slot).expect("a window slot opens");
    }
    let mut lanes = vec![Lane { slot: slot_a, word: word(PROMPT_A.len() as u32, false), tokens: PROMPT_A }];
    for &slot in slots {
        lanes.push(Lane { slot, word: word(PROMPT_B.len() as u32, drafting), tokens: PROMPT_B });
    }
    let got = shell.fire(&lanes).expect("the prefill fires");
    let mut rows_a = vec![got[0].clone()];
    let mut fed: Vec<[u32; 1]> = got.iter().map(|row| [argmax(row)]).collect();
    for step in 0..STEPS {
        let mut lanes = vec![Lane { slot: slot_a, word: word(1, false), tokens: &fed[0] }];
        for (at, &slot) in slots.iter().enumerate() {
            lanes.push(Lane { slot, word: word(1, drafting), tokens: &fed[at + 1] });
        }
        let got = shell
            .fire(&lanes)
            .unwrap_or_else(|why| panic!("decode step {step} beside a window fires: {why}"));
        rows_a.push(got[0].clone());
        fed = got.iter().map(|row| [argmax(row)]).collect();
    }
    rows_a
}

fn compare(what: &str, control: &[Vec<f32>], beside: &[Vec<f32>]) {
    let mut first = None;
    for (step, (a, b)) in control.iter().zip(beside.iter()).enumerate() {
        assert_eq!(a.len(), b.len());
        let diffs = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        let max = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        if diffs > 0 && first.is_none() {
            first = Some(step);
        }
        eprintln!(
            "{what}, step {step}: {diffs} differing logits, max |Δ| {max:.4}, argmax {} vs {}",
            argmax(a),
            argmax(b)
        );
    }
    assert!(
        first.is_none(),
        "lane A's trunk logits changed at step {} beside {what}",
        first.unwrap()
    );
}
