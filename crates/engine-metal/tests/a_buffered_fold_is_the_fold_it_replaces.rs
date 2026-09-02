//! **THE RS DEVICE HALF, GATED ON THE EQUIVALENCES IT EXISTS TO KEEP** — the
//! Metal port of `engine-cuda/tests/a_buffered_fold_is_the_fold_it_replaces`,
//! over the qwen4 micro fixture (`tests/fixtures/qwen4-micro`: `Model::flash_micro`'s
//! seeded checkpoint — four layers, GDN recurrences and a PLE hasher, bf16), plus the read path the CUDA plane
//! refuses. Compared byte for byte on the recurrent banks where a fold is the
//! claim, and on the logits where the window's answer is.
//!
//! 1. a buffered scatter folds nothing;
//! 2. `Buffer` then `FoldBuffered(k)` over the same k tokens equals one
//!    `Fold` of them;
//! 3. `commit_len` is live: a fold truncated at `j < k` leaves the banks a
//!    fold of the first `j` tokens leaves — and not the banks of a fold of
//!    all `k`;
//! 4. the mixed row — one fire that buffers the window and lands the fold
//!    boundary at row `j` — leaves the banks a `Fold` of `j` tokens leaves,
//!    and answers the window's last row what a fire over the window from a
//!    state folded through `j` answers;
//! 5. **the read path**: a window buffered unfolded, then a second fire that
//!    REPLAYS it (`replay = k`, `fold = k`) while buffering its own rows,
//!    leaves the banks a `Fold` of the first window leaves and answers the
//!    second window's logits within the bf16 floor of a fire over it after a
//!    plain fold of the first — one speculative round, as
//!    `rs-speculative-decoding` fires it;
//! 6. a lane folding in the forward beside a lane buffering answers exactly
//!    what it answers alone.
//!
//! ```text
//! cargo test -p engine-metal --release --test a_buffered_fold_is_the_fold_it_replaces -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request};

/// Tokens in the speculative window; more than the PLE n-gram and the conv
/// history, so the state after it depends on rows inside it.
const WINDOW: usize = 6;

const PAGE: u32 = 16;

fn fixture() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_QWEN4_FIXTURE") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-micro");
    path.is_dir().then_some(path)
}

fn word(query_len: u32) -> u64 {
    models::qwen_4::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn ready() -> Option<Shell> {
    if !engine_metal::device::present() {
        eprintln!("skipping the buffered fold gate: no Metal device");
        return None;
    }
    let Some(fixture) = fixture() else {
        eprintln!("skipping the buffered fold gate: no qwen4 fixture (crates/engine-metal/tests/fixtures/qwen4-micro or PIE_QWEN4_FIXTURE)");
        return None;
    };
    let micro = models::qwen_4::model::Model::flash_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen4-micro", &micro, Platform::Metal);
    let source = ztensor_compat::index(&fixture.join("model.safetensors")).expect("the fixture opens");
    let contract = micro
        .import(&source, Platform::Metal)
        .expect("the micro text's import fits the fixture it was generated for");
    drop(source);
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &fixture,
        budget: Budget::new(4, 64),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: 128,
        slots: 8,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the micro shell loads");
    eprintln!(
        "recurrent slot: {} bytes; buffered page slab: {:.1} KiB; serves rs verbs: {}",
        shell.state_slot_bytes(),
        shell.buffer_bytes() as f64 / 1024.0,
        shell.serves_rs_verbs()
    );
    assert!(shell.state_slot_bytes() > 0, "a GDN plan has recurrent state");
    assert!(shell.buffer_bytes() > 0, "a GDN plan reserved no buffered page slab, so nothing below can be true");
    assert!(shell.serves_rs_verbs());
    Some(shell)
}

/// Arbitrary ids inside the fixture's 256-token vocabulary.
fn window(seed: u32) -> Vec<u32> {
    (0..WINDOW as u32).map(|at| (seed + at * 37) % 256).collect()
}

fn pages() -> Vec<u32> {
    // the slab's first page slot; a window fits one page
    vec![0]
}

fn buffer(at: u32, fold: u32, replay: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: pages(),
        at,
        fold: FoldLen::Host(fold),
        replay,
    }
}

fn fold_buffered(bound: u32, len: u32) -> RsVerb {
    fold_buffered_at(0, bound, len)
}

/// A replay whose buffer begins at token `at` — where a round's own window
/// sits after the prefix it replayed was folded off the front.
fn fold_buffered_at(at: u32, bound: u32, len: u32) -> RsVerb {
    RsVerb::FoldBuffered {
        pages: pages(),
        at,
        bound,
        len: FoldLen::Host(len),
    }
}

fn seated<'a>(slot: u32, tokens: &'a [u32], rs: &'a RsVerb, reset: RsReset) -> Seated<'a> {
    Seated {
        rs,
        rs_reset: reset,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn argmax(logits: &[f32]) -> usize {
    (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap_or(0)
}

/// Logits differ across the two paths only by the bf16 rounding of a state
/// written back and re-read; the micro parity gate's own pin is 0.15 against
/// the reference.
const LOGIT_FLOOR: f32 = 0.05;

#[test]
fn a_buffered_fold_is_the_fold_it_replaces() {
    let Some(mut shell) = ready() else {
        return;
    };
    let fold = RsVerb::Fold;
    let tokens = window(11);
    let k = WINDOW as u32;

    // ── reference: one ordinary fire, folding every token in the forward
    shell.open(0).expect("slot 0 opens");
    let zeroed = shell.state_bytes(0).expect("slot 0 reads back");
    assert!(zeroed.iter().all(|byte| *byte == 0), "an opened slot's banks are zero");
    let reference = shell
        .fire_seated(&[seated(0, &tokens, &fold, RsReset::Fresh)])
        .expect("the folding fire runs");
    let folded = shell.state_bytes(0).expect("slot 0 reads back");
    assert_ne!(folded, zeroed, "a fold over {WINDOW} tokens left the banks untouched");

    // ── claim 1: a buffered scatter folds nothing
    shell.open(1).expect("slot 1 opens");
    let buffered = shell
        .fire_seated(&[seated(1, &tokens, &buffer(0, 0, 0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    assert_eq!(shell.state_bytes(1).expect("slot 1"), zeroed, "a `RsVerb::Buffer` fire perturbed the folded state");
    // ...and computed the window from the state it should see: the logits
    // of the buffered fire are the folding fire's (same state to start from,
    // same rows), within the floor.
    let d = spread(&reference[0], &buffered[0]);
    eprintln!("claim 1: buffered window vs folding window, last-row logit spread {d:.4}");
    assert!(d <= LOGIT_FLOOR, "a buffered fire answers other logits than the fold it defers: {d}");

    // ── claim 2: the replay is the fold
    shell
        .fire_seated(&[seated(1, &tokens, &fold_buffered(k, k), RsReset::Held)])
        .expect("the fold-buffered fire runs");
    let replayed = shell.state_bytes(1).expect("slot 1");
    assert_eq!(replayed.len(), folded.len());
    assert_eq!(replayed, folded, "`Buffer` then `FoldBuffered(k)` did not leave the banks `Fold` leaves");

    // ── claim 3: commit_len is live — a fold of the first j tokens
    let j = 3u32;
    shell.open(2).expect("slot 2 opens");
    shell
        .fire_seated(&[seated(2, &tokens[..j as usize], &fold, RsReset::Fresh)])
        .expect("the short folding fire runs");
    let folded_j = shell.state_bytes(2).expect("slot 2");
    assert_ne!(folded_j, folded, "folding {j} tokens and folding {k} leave the same banks, so claim 3 can't tell them apart");
    shell.open(3).expect("slot 3 opens");
    shell
        .fire_seated(&[seated(3, &tokens, &buffer(0, 0, 0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    shell
        .fire_seated(&[seated(3, &tokens, &fold_buffered(k, j), RsReset::Held)])
        .expect("the truncated fold runs");
    assert_eq!(shell.state_bytes(3).expect("slot 3"), folded_j, "a replay truncated at {j} did not leave the banks a fold of {j} leaves");

    // ── claim 4: the mixed row — buffer the window, fold through row j
    shell.open(4).expect("slot 4 opens");
    let mixed = shell
        .fire_seated(&[seated(4, &tokens, &buffer(0, j, 0), RsReset::Fresh)])
        .expect("the mixed fire runs");
    assert_eq!(shell.state_bytes(4).expect("slot 4"), folded_j, "the mixed row did not land the fold at row {j}");
    let d = spread(&reference[0], &mixed[0]);
    eprintln!("claim 4: mixed row vs folding window, last-row logit spread {d:.4}");
    assert!(d <= LOGIT_FLOOR, "the mixed row answers other logits than the fold: {d}");

    // ── claim 5: the read path — one speculative round
    //    slot 5: buffer window A unfolded; then fire window B replaying A
    //    (fold = |A|) while buffering B. Banks must equal Fold(A); logits must
    //    equal a fire over B after a plain Fold(A) (slot 6).
    let second = window(101);
    shell.open(5).expect("slot 5 opens");
    shell
        .fire_seated(&[seated(5, &tokens, &buffer(0, 0, 0), RsReset::Fresh)])
        .expect("window A buffers");
    let round = shell
        .fire_seated(&[seated(5, &second, &buffer(k, k, k), RsReset::Held)])
        .expect("window B replays A and buffers itself");
    assert_eq!(shell.state_bytes(5).expect("slot 5"), folded, "replaying A ahead of B did not leave the banks Fold(A) leaves");
    shell.open(6).expect("slot 6 opens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds plainly");
    let plain = shell
        .fire_seated(&[seated(6, &second, &fold, RsReset::Held)])
        .expect("B folds after it");
    let d = spread(&plain[0], &round[0]);
    eprintln!(
        "claim 5: B after replay(A) vs B after fold(A), last-row logit spread {d:.4}; argmax {} vs {}",
        argmax(&round[0]),
        argmax(&plain[0])
    );
    assert!(d <= LOGIT_FLOOR, "the read path answers other logits than a plain fold of the prefix: {d}");
    //    and the round's own buffer (B, scattered at `at = k`) then folds to
    //    what Fold(A) then Fold(B) left
    shell
        .fire_seated(&[seated(5, &second, &fold_buffered_at(k, k, k), RsReset::Held)])
        .expect("B's buffer folds");
    assert_eq!(
        shell.state_bytes(5).expect("slot 5"),
        shell.state_bytes(6).expect("slot 6"),
        "folding the round's buffer did not leave the banks Fold(A) then Fold(B) leaves"
    );

    // ── claim 6: a folding lane beside a buffering lane answers what it answers alone
    let third = window(203);
    shell.open(7).expect("slot 7 opens");
    let alone = shell
        .fire_seated(&[seated(7, &third, &fold, RsReset::Fresh)])
        .expect("the lone fold runs");
    let alone_state = shell.state_bytes(7).expect("slot 7");
    shell.open(7).expect("slot 7 reopens");
    shell.open(1).expect("slot 1 reopens");
    let beside = shell
        .fire_seated(&[
            seated(7, &third, &fold, RsReset::Fresh),
            seated(1, &tokens, &buffer(0, 0, 0), RsReset::Fresh),
        ])
        .expect("the mixed fire runs");
    assert_eq!(shell.state_bytes(7).expect("slot 7"), alone_state, "a folding lane's banks changed for having a buffering peer");
    assert_eq!(shell.state_bytes(1).expect("slot 1"), zeroed, "the buffering peer folded");
    let d = spread(&alone[0], &beside[0]);
    eprintln!("claim 6: folding lane beside a buffering peer, logit spread {d:.4}");
    assert!(d <= LOGIT_FLOOR, "a folding lane answers differently beside a buffering peer: {d}");
}
