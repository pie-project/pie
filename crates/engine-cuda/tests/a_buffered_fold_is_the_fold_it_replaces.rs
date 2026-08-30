//! **The RS device half, gated on the one equivalence it exists to keep**
//! (alto design §6, survey §9; wave F3).
//!
//! The programming model's whole claim is that a speculative window can be
//! folded LATER without being folded WRONG:
//!
//! ```text
//! fold k tokens in the forward          ==   buffer k tokens,
//!   (write_state, one fire)                  then fold-buffered k
//!                                            (two fires, in-proj replayed)
//! ```
//!
//! Byte for byte, on the recurrent banks themselves — not on the logits,
//! because a fold-buffered replay computes no output anybody reads and a
//! logit diff would be measuring the attention layers riding beside it.
//!
//! Six claims, in the order they can fail:
//!
//! 1. **a buffered scatter folds nothing.** After a `RsVerb::Buffer` fire the
//!    slot's banks are exactly what they were, which is what makes a rejected
//!    draft pure host bookkeeping ("no folded state was ever perturbed").
//! 2. **the replay is the fold.** `Buffer(k)` then `FoldBuffered(k)` leaves the
//!    same bytes as one `Fold` over the same k tokens. This is the whole
//!    wave: the buffer holds the conv+scan's INPUTS, the gather puts them back
//!    over the in-projection the replay recomputed, and the recurrence cannot
//!    tell the difference.
//! 3. **`commit_len` is live.** A replay truncated at fewer tokens leaves
//!    DIFFERENT bytes — so the seat the shell fills is one the kernels read,
//!    rather than an argument nobody looks at.
//! 4. **a truncated fold is EXACT against a shorter buffer** (wave F3b, and
//!    the claim the kernel's coupling made impossible). A replay of twenty
//!    buffered tokens truncated at four and a replay BOUNDED at four leave
//!    the same bytes. The two run different launches over different row
//!    counts — one binds `commit_len` and the other binds nothing — and the
//!    old fla scan read `commit_len != nullptr` as `single_round` as well,
//!    a different bf16 rounding of the decay (`ssm.cuh:1660,1697-1706`), so
//!    the two agreed only to a rounding: 3,115,437 of 10,321,884 state bytes
//!    moved for no reason but the flag. The rounding is its own argument now
//!    (`RecurrentPool::fused_decay`), the shell binds the fold's own policy
//!    everywhere, and the equality is exact.
//! 5. **the mixed row is the two fires it replaces** (wave F3b's 2R interior
//!    split). `Buffer(k tokens, fold = j)` — one fire that writes the whole
//!    window into the buffer AND lands the boundary on row `j` — leaves the
//!    same bytes as `Buffer(k)` followed by `FoldBuffered(j)`. The row is cut
//!    into two segments on one stream: the head `[0, j)` folds, and the tail
//!    `[j, k)` continues from what the head wrote and moves nothing. **Both
//!    halves are pinned**, because either can fail alone — the state is
//!    compared against the two-fire reference, and the fire's OUTPUTS against
//!    a plain fold over the same window, which is what a skipped tail would
//!    lose.
//! 6. **the fold predicate is per lane, and it is device data.** One fire
//!    carrying a folding lane BESIDE a buffering one folds exactly one of
//!    them, and `channel::mask_from_commit` is what decided which: the
//!    predicate bytes it wrote are readable afterwards and say `1` for the
//!    lane pointed at the standing committed word and `0` for the lane pointed
//!    at the standing refused one. A refused PASS reaches the same kernel by
//!    the same array with the instance's own commit word in that slot, which
//!    is why this is the mask's gate and not merely the verb's.
//!
//! # Gating
//!
//! Skips at RUN time, like [`serve_smoke`](../serve_smoke.rs), saying which of
//! the machine and the checkpoint was missing — an `#[ignore]`d test on the
//! one box that could run it is a test nobody runs.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test a_buffered_fold_is_the_fold_it_replaces -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_cuda::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this gate serves: a GDN/attention hybrid, which is the
/// family the whole recurrent vocabulary exists for.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Tokens in the speculative window. Twenty against a sixteen-token page is
/// deliberate: the buffer's addressing is page-major, so a window that fits in
/// one page would never exercise the page-crossing arithmetic that the scatter
/// and the gather have to agree about.
const WINDOW: usize = 20;

/// The kv page size, which is also the buffer's page size — dev's rule, so
/// that a buffer page and a kv page are one number.
const PAGE: u32 = 16;

/// How many buffer page slots `WINDOW` tokens need.
const PAGES: u64 = WINDOW.div_ceil(PAGE as usize) as u64;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The lane word the model's own `Classify` computes — runtime-side work, done
/// here because this test IS the runtime for the length of one fire.
fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// A loaded shell, or `None` and a sentence saying what was missing.
fn ready(what: &str) -> Option<Shell> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: 512,
        // One recurrent slot per arm below: the reference fold, the replay,
        // the truncated replay, the shorter-bounded replay, the mixed row,
        // and the mixed row's output reference.
        slots: 8,
        ordinal: 0,
        // **THE EAGER PATH, AND THE SHELL WOULD HAVE INSISTED ANYWAY.** A fire
        // that moves buffered bytes is not graph-replayable (design §6: the
        // plain fold is the only shape that is), so `enqueue` forces the eager
        // walk for one whatever the mode — this states the same thing where a
        // reader can see it.
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    eprintln!(
        "{what}: buffered-activation pool is {:.2} MiB",
        shell.buffer_bytes() as f64 / (1 << 20) as f64,
    );
    assert!(
        shell.buffer_bytes() > 0,
        "a GDN plan reserved no buffered-activation pool, so nothing below can be true"
    );
    Some(shell)
}

/// The window this gate folds — arbitrary ids, because what is being compared
/// is two ways of folding the SAME activations and not what the model says
/// about them.
fn window() -> Vec<u32> {
    (0..WINDOW as u32).map(|at| 1000 + at * 37).collect()
}

fn buffer(at: u32) -> RsVerb {
    // `Host(0)` is the pure scatter: every row into the buffer, the folded
    // state untouched.
    mixed(at, 0)
}

/// **The mixed row** (wave F3b): the same scatter, landing the durable state
/// on row `fold` of the window it is writing.
fn mixed(at: u32, fold: u32) -> RsVerb {
    RsVerb::Buffer {
        // The run is a LIST of physical page slots (wave F3-tail); this gate's
        // buffer happens to be the first `PAGES` of the pool, in order.
        pages: (0..PAGES as u32).collect(),
        at,
        fold: FoldLen::Host(fold),
    }
}

fn fold_buffered(len: u32) -> RsVerb {
    bounded_fold(WINDOW as u32, len)
}

/// A replay whose BOUND is stated too — the shorter buffer claim 4 compares
/// a truncated fold against. `bound` is the lane's row count by contract, so
/// the two arms differ in rows as well as in the seat.
fn bounded_fold(bound: u32, len: u32) -> RsVerb {
    RsVerb::FoldBuffered {
        pages: (0..PAGES as u32).collect(),
        // This gate's buffer begins at buffer token zero: no fold has landed
        // mid-page ahead of it, so the head and the origin are both zero.
        at: 0,
        bound,
        len: FoldLen::Host(len),
    }
}

fn seated<'a>(slot: u32, tokens: &'a [u32], rs: RsVerb, reset: RsReset) -> Seated<'a> {
    Seated {
        lane: Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        },
        rs,
        rs_reset: reset,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

#[test]
fn a_buffered_fold_is_the_fold_it_replaces() {
    let Some(mut shell) = ready("the buffered fold") else {
        return;
    };
    let tokens = window();

    // ── The reference: one ordinary fire, folding every token in the forward.
    shell.open(0).expect("slot 0 opens");
    let zeroed = shell.state_bytes(0).expect("slot 0 reads back");
    assert!(
        zeroed.iter().all(|byte| *byte == 0),
        "an opened slot's recurrent banks are not zero, so nothing below is a comparison"
    );
    let reference = shell
        .fire_seated(&[seated(0, &tokens, RsVerb::Fold, RsReset::Fresh)])
        .expect("the folding fire runs");
    let folded = shell.state_bytes(0).expect("slot 0 reads back");
    assert_ne!(
        folded, zeroed,
        "a fold over {WINDOW} tokens left the banks untouched, so the reference is empty"
    );

    // ── Claim 1: a buffered scatter folds NOTHING.
    shell.open(1).expect("slot 1 opens");
    shell
        .fire_seated(&[seated(1, &tokens, buffer(0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1 reads back"),
        zeroed,
        "a `RsVerb::Buffer` fire perturbed the folded state — the whole point of the \
         buffer is that a rejected draft is pure host bookkeeping"
    );

    // ── Claim 2: the replay IS the fold.
    shell
        .fire_seated(&[seated(1, &tokens, fold_buffered(WINDOW as u32), RsReset::Held)])
        .expect("the fold-buffered fire runs");
    let replayed = shell.state_bytes(1).expect("slot 1 reads back");
    assert_eq!(
        replayed.len(),
        folded.len(),
        "the two slots hold different amounts of state, which is a pool bug"
    );
    let differing = folded
        .iter()
        .zip(&replayed)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "buffer-then-fold-buffered left {differing} of {} state bytes different from the \
         fold it replaces",
        folded.len(),
    );

    // ── Claim 3: `commit_len` is a seat the kernels read.
    shell.open(2).expect("slot 2 opens");
    shell
        .fire_seated(&[seated(2, &tokens, buffer(0), RsReset::Fresh)])
        .expect("the second buffering fire runs");
    shell
        .fire_seated(&[seated(2, &tokens, fold_buffered(4), RsReset::Held)])
        .expect("the truncated fold-buffered fire runs");
    let truncated = shell.state_bytes(2).expect("slot 2 reads back");
    assert_ne!(
        truncated, zeroed,
        "a fold truncated at 4 tokens folded nothing at all"
    );
    assert_ne!(
        truncated, replayed,
        "a fold truncated at 4 tokens left the same bytes as one over all {WINDOW}, so \
         `commit_len` reached the launch and was ignored"
    );

    // ── Claim 4: the truncated fold is EXACT against a shorter buffer.
    //
    //    The same four buffered tokens, folded two ways over the SAME bytes:
    //    slot 2 replayed twenty and stopped at four (`commit_len` bound),
    //    slot 3 replays four and stops at its own end (no seat at all). The
    //    buffer is untouched between the two, so the activations the gather
    //    lays down are the same bytes — which is what makes this a test of
    //    the ARITHMETIC and of nothing else.
    //
    //    While the fla scan read `commit_len != nullptr` as a rounding this
    //    equality could not hold: the bound arm folded the decay into the
    //    update and the unbound one rounded the decayed state to bf16 first.
    shell.open(3).expect("slot 3 opens");
    shell
        .fire_seated(&[seated(
            3,
            &tokens[..4],
            bounded_fold(4, 4),
            RsReset::Fresh,
        )])
        .expect("the shorter-bounded fold-buffered fire runs");
    let shorter = shell.state_bytes(3).expect("slot 3 reads back");
    assert_ne!(
        shorter, zeroed,
        "a replay bounded at 4 tokens folded nothing at all"
    );
    let differing = truncated
        .iter()
        .zip(&shorter)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "a fold of {WINDOW} buffered tokens truncated at 4 left {differing} of {} state \
         bytes different from a fold of the same 4 over a buffer bounded there — the \
         length seat is still changing the arithmetic and not only the count",
        truncated.len(),
    );

    // ── Claim 5: the mixed row IS the two fires it replaces.
    //
    //    One fire: scatter the whole window into the buffer AND land the
    //    boundary on row 4. The row's fold boundary is strictly interior, so
    //    the recurrent arms run twice on one stream — head `[0, 4)` folding,
    //    tail `[4, {WINDOW})` continuing from what the head wrote.
    shell.open(4).expect("slot 4 opens");
    let mixed_out = shell
        .fire_seated(&[seated(4, &tokens, mixed(0, 4), RsReset::Fresh)])
        .expect("the mixed fire runs");
    let mixed_state = shell.state_bytes(4).expect("slot 4 reads back");
    let differing = truncated
        .iter()
        .zip(&mixed_state)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "one fire that buffers {WINDOW} tokens and folds 4 of them left {differing} of {} \
         state bytes different from buffering them and folding 4 in two fires",
        truncated.len(),
    );

    // ── And the TAIL RAN. A split that dropped its second launch would leave
    //    the same folded state and every output past the boundary unwritten,
    //    so the state comparison alone cannot see it. The whole row is one
    //    continuous scan — the head from the pre-fire state, the tail from
    //    the state the head wrote — so the fire's outputs must be the plain
    //    fold's outputs over the same window.
    let mixed_row = mixed_out.first().expect("the mixed fire reads a row back");
    let plain_row = reference.first().expect("the folding fire reads a row back");
    assert_eq!(
        mixed_row.len(),
        plain_row.len(),
        "the two fires read back different row widths"
    );
    assert!(
        !plain_row.is_empty(),
        "the reference fire read no logits back, so the tail cannot be pinned"
    );
    let worst = mixed_row
        .iter()
        .zip(plain_row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3,
        "the mixed fire's own outputs differ from the plain fold's by {worst} — the tail \
         segment `[4, {WINDOW})` did not run, or did not continue from the boundary"
    );
}

/// **Claim 6**: the fold predicate is per lane and the device writes it.
///
/// One fire, two lanes, two verbs: the folding lane's slot advances and the
/// buffering lane's does not, in a fire where BOTH ran through the same
/// launches over the same window. Nothing about the launch distinguishes them
/// — same conv, same scan, same `write_state` — so the only thing that can
/// have separated the two is the byte `channel::mask_from_commit` wrote for
/// each, which is read back and asserted directly.
///
/// **AND THAT KERNEL IS THE PASS PREDICATE, NOT A VERB SWITCH.** The array it
/// scatters holds one commit-word ADDRESS per lane: the standing "committed"
/// word for a lane with no guest, the standing "refused" word for a lane whose
/// verb is a scatter, and the attached instance's OWN pass commit word for a
/// lane that carries a prologue — the word `channel::pull_validate` clears
/// when a prediction turns out stale. So a refused pass and a buffered scatter
/// reach the fold through the same array, the same kernel and the same byte,
/// which is what makes this the mask's gate.
#[test]
fn one_fire_folds_the_lane_that_committed_and_not_the_lane_that_buffered() {
    let Some(mut shell) = ready("the per-lane fold predicate") else {
        return;
    };
    let tokens = window();

    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let zeroed = shell.state_bytes(1).expect("slot 1 reads back");

    shell
        .fire_seated(&[
            seated(0, &tokens, RsVerb::Fold, RsReset::Fresh),
            seated(1, &tokens, buffer(0), RsReset::Fresh),
        ])
        .expect("the mixed fire runs");

    let predicate = shell.fold_predicate(2).expect("the predicate reads back");
    assert_eq!(
        predicate,
        vec![1, 0],
        "the fold predicate the device wrote is {predicate:?}, and this fire's lanes are \
         one folding lane followed by one buffering lane"
    );
    assert_ne!(
        shell.state_bytes(0).expect("slot 0 reads back"),
        zeroed,
        "the folding lane of a mixed fire did not fold"
    );
    assert_eq!(
        shell.state_bytes(1).expect("slot 1 reads back"),
        zeroed,
        "the buffering lane of a mixed fire folded anyway, so the predicate did not reach \
         the scan"
    );
}
