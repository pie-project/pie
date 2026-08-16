//! What varies between two consecutive decode steps, and what reusing the
//! answer costs.
//!
//! The question this file was written to settle: a decode step runs the same
//! plan every token -- same routines, same order, same shapes, same bindings,
//! same grids -- so how much of the 1.15 ms of planning and 0.35 ms of
//! recording per step is re-deriving a fact that did not change?
//!
//! Answering it needs a DIFF of two consecutive fires, field by field, and
//! the diff lives in the driver rather than here: [`driver_vulkan::replay`]
//! plans every fire in full under [`Plans::verify`] and compares it against
//! the one before, counting differences by the four places one could live --
//! a push constant (baked into the command buffer by `vkCmdPushConstants`,
//! and fatal to reuse), a descriptor range (referenced, so rewritable
//! between submits), a workgroup count (baked by `vkCmdDispatch`), and the
//! scalar block (a device buffer the host rewrites before every submit, so
//! free to vary).
//!
//! `where_a_decode_steps_host_milliseconds_go` in `tests/hostprof.rs` says
//! where the milliseconds are; this file says whether they have to be spent.
//!
//! # How to run it
//!
//! ```text
//! cargo test -p driver-vulkan --features native --release --test replay \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! One thread: these tests each open a shell holding a checkpoint's worth of
//! weights, and two at once is both slower and a different memory shape than
//! anything a server does.
//!
//! # Why it duplicates `hostprof.rs`'s fixture
//!
//! An integration test cannot import another one. The checkpoint load, the
//! text and the shell are copied for the same reason that file copies them
//! from `device.rs`.

#![cfg(feature = "native")]

use std::collections::BTreeMap;

/// The prompt `device.rs` and `hostprof.rs` use: six tokens that repeat, so a
/// decode after `n` copies has `6n` of history.
const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

/// The one model this file measures. See `hostprof.rs` for why one.
const MODEL_ID: &str = "qwen3-0.6b";

/// `model.embed_tokens.weight`'s packed shape, which is how a snapshot says
/// which model it is.
const EMBED: &[i64] = &[151_936, 128];

fn facts() -> model::shared::llama_like::forward::facts::LlamaLikeFacts {
    model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b()
}

fn snapshots() -> Vec<String> {
    match std::env::var("PIE_CHECKPOINT") {
        Ok(v) => v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Every weight name this model's decode plan binds.
fn names_a_decode_binds() -> Vec<String> {
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_ir::trace::FireClass;

    let text = llama_like_metal(
        &facts(),
        &LlamaLikeMetalFacts {
            add_bias: true,
            ..LlamaLikeMetalFacts::synthetic()
        },
        FireClass::Decode,
    );
    let low = lower(
        &text,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    let names: std::collections::BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) if !n.starts_with("scale.") => Some(n.clone()),
            _ => None,
        })
        .collect();
    names.into_iter().collect()
}

/// The checkpoint's bytes under the names a plan binds, or `None` to skip.
fn load_weights() -> Option<BTreeMap<String, Vec<u8>>> {
    for dir in snapshots() {
        let path = std::path::Path::new(&dir);
        let Ok(meta) = model_loader::checkpoint::read::parse_checkpoint_metadata(path) else {
            continue;
        };
        let hidden = meta
            .tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        if hidden != EMBED {
            continue;
        }
        let row = model::catalog::find(MODEL_ID)?;
        let config = std::fs::read_to_string(path.join("config.json")).ok()?;
        let encoding = model::encoding::Encoding::from_config_json(&config).ok()?;
        let target = model_loader::plan::StorageTarget::for_backend(
            model_loader::types::BackendKind::Vulkan,
            0,
            1,
        );
        let (plan, _) = model::boot::compile_load_plan_for(
            path,
            &meta,
            &target,
            row,
            &encoding,
            model::boot::Binding::MLX_IN_PLACE,
        )
        .ok()?;
        let storage = match model_loader::executor::Execution::new(&plan, path).run() {
            Ok(storage) => storage,
            Err(e) => {
                eprintln!("the loader would not execute the plan: {e}");
                return None;
            }
        };
        let naming = driver_vulkan::names::Naming::mlx();
        let mut out = BTreeMap::new();
        for traced in names_a_decode_binds() {
            let bytes = naming
                .spellings(&traced)
                .iter()
                .find_map(|s| storage.tensors.get(s.as_str()))
                .unwrap_or_else(|| panic!("`{traced}` resolves to nothing the loader produced"));
            out.insert(traced, bytes.clone());
        }
        return Some(out);
    }
    eprintln!("no snapshot PIE_CHECKPOINT names is the 4-bit `{MODEL_ID}` this file states");
    None
}

/// The checkpoint, read once for the process.
fn weights() -> Option<&'static BTreeMap<String, Vec<u8>>> {
    static HELD: std::sync::OnceLock<Option<&'static BTreeMap<String, Vec<u8>>>> =
        std::sync::OnceLock::new();
    *HELD.get_or_init(|| load_weights().map(|m| &*Box::leak(Box::new(m))))
}

/// A shell serving the model, with its weights held.
fn shelled(real: &'static BTreeMap<String, Vec<u8>>, pages: u32) -> driver_vulkan::shell::Shell {
    use driver_vulkan::shell::{Deployment, Shell, Text};
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_ir::trace::FireClass;

    let facts = facts();
    let metal = LlamaLikeMetalFacts {
        add_bias: true,
        ..LlamaLikeMetalFacts::synthetic()
    };
    let text = Text {
        decode: llama_like_metal(&facts, &metal, FireClass::Decode),
        prefill: llama_like_metal(&facts, &metal, FireClass::Prefill),
        geometry: driver_vulkan::dispatch::Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: 0,
            experts_per_token: 0,
            ..Default::default()
        },
        layers: facts.layers as u16,
    };

    let mut shell = Shell::open(
        text,
        Deployment {
            pages,
            ..Deployment::default()
        },
    )
    .unwrap_or_else(|e| panic!("the shell: {e}"));
    for (name, bytes) in real {
        shell.hold(name, bytes).expect("a weight");
    }
    shell
}

/// One step of one conversation.
fn step(shell: &mut driver_vulkan::shell::Shell, tokens: Vec<u32>) -> driver_vulkan::turns::Step {
    shell
        .step(&[driver_vulkan::turns::Turn { who: 1, tokens }])
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Which token the distribution's largest entry names.
fn argmax(step: &driver_vulkan::turns::Step) -> u32 {
    let row = step
        .logits
        .row(step.readout_of[0])
        .expect("the readout row is in the logits");
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// **Nothing a decode step records varies between one token and the next.**
///
/// The evidence for the whole of [`driver_vulkan::replay`]. Fifty decode
/// steps, every one of them planned in full from the lowering, and every one
/// compared against its predecessor across all six recorded fields of all 452
/// rectangles. The assertion is that the four fields that are BAKED into a
/// command buffer -- the entrypoint, the descriptor ranges, the push
/// constants and the workgroup counts -- are identical every time.
///
/// Fifty steps rather than two, and 128 pages of an eight-row page size, so
/// the run crosses several page boundaries: a page boundary is the one event
/// in a steady-state decode that reallocates a driver table, and if a grid or
/// a bound range were going to follow the history's length that is where it
/// would show.
#[test]
fn nothing_a_decode_records_differs_from_the_step_before_it() {
    if !kernels_vulkan::embedded() {
        eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
        return;
    }
    let Some(real) = weights() else {
        return;
    };
    const STEPS: usize = 50;

    let mut shell = shelled(real, 128);
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..4 {
        prompt.extend_from_slice(&PERIOD);
    }
    step(&mut shell, prompt);
    // Warm: the first decode after a prefill builds every pipeline and the
    // lowering cache is cold until a decode of this row count is asked for
    // once. A diff against a fire that was the first of its kind would be
    // reporting the warm-up rather than the steady state.
    for _ in 0..3 {
        step(&mut shell, vec![PERIOD[0]]);
    }
    // Nothing is reused while this is on: every fire is planned from the
    // lowering, so what is compared is two independent derivations.
    shell.plans().verify(true);
    let replayed_warming = shell.plans().replays();
    for _ in 0..STEPS {
        step(&mut shell, vec![PERIOD[0]]);
    }
    let diff = shell.plans().diff().clone();
    let replays = shell.plans().replays() - replayed_warming;

    eprintln!("\n=== {STEPS} decode steps, each planned in full and diffed ===");
    eprintln!("  {:<38} {:>8}", "fires compared", diff.compared);
    eprintln!(
        "  {:<38} {:>8}",
        "  of those, the key changed", diff.rekeyed
    );
    eprintln!("  {:<38} {:>8}", "differing rectangle counts", diff.shape);
    eprintln!("  {:<38} {:>8}", "rectangles: entrypoint", diff.symbol);
    eprintln!(
        "  {:<38} {:>8}",
        "rectangles: descriptor ranges", diff.buffers
    );
    eprintln!("  {:<38} {:>8}", "  individual ranges", diff.bindings);
    eprintln!("  {:<38} {:>8}", "rectangles: write masks", diff.writes);
    eprintln!("  {:<38} {:>8}", "rectangles: PUSH CONSTANTS", diff.push);
    eprintln!(
        "  {:<38} {:>8}",
        "rectangles: workgroup counts", diff.groups
    );
    eprintln!("  {:<38} {:>8}", "fires: scalar block bytes", diff.block);
    for w in &diff.witnesses {
        eprintln!("    witness: {w}");
    }

    assert_eq!(
        diff.compared, STEPS as u64,
        "every fire but the first should have been compared against the one \
         before it; a missing comparison means a fire was reused rather than \
         planned, which is what `verify` exists to prevent"
    );
    assert_eq!(replays, 0, "verification must not reuse anything");
    assert!(
        diff.rekeyed < diff.compared,
        "every comparison crossed a key change, so nothing was actually \
         checked; the key is varying per step"
    );
    assert!(
        diff.quiet(),
        "two decode steps the key called interchangeable recorded different \
         commands: {:?}",
        diff.witnesses
    );
}

/// **A reusing shell and a re-planning shell decode the same tokens.**
///
/// The correctness half. Two shells in one process over one checkpoint, fed
/// the same prompt and then decoded greedily for thirty steps each, one with
/// the plan cache on and one with it turned off -- and the argmax must agree
/// token for token.
///
/// Argmax rather than the raw logits: this is a matmul chain in half
/// precision on a card that is free to reassociate, and the claim being
/// tested is that the same commands ran, not that floating point is
/// deterministic. A single differing command in a 452-rectangle fire moves
/// the answer, not the last bit of it.
#[test]
fn a_reused_fire_decodes_the_same_tokens_as_a_replanned_one() {
    if !kernels_vulkan::embedded() {
        eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
        return;
    }
    let Some(real) = weights() else {
        return;
    };
    const STEPS: usize = 30;

    let sampled = |reuse: bool| -> Vec<u32> {
        let mut shell = shelled(real, 128);
        if !reuse {
            shell.plans().disable();
        }
        let mut prompt: Vec<u32> = Vec::new();
        for _ in 0..4 {
            prompt.extend_from_slice(&PERIOD);
        }
        let mut out = Vec::with_capacity(STEPS);
        let mut next = argmax(&step(&mut shell, prompt));
        for _ in 0..STEPS {
            out.push(next);
            next = argmax(&step(&mut shell, vec![next]));
        }
        if reuse {
            eprintln!(
                "  reusing: {} replays, {} re-records, {} full plans over {} steps",
                shell.plans().replays(),
                shell.plans().records(),
                shell.plans().planned(),
                STEPS + 1
            );
            assert!(
                shell.plans().replays() > 0,
                "the cache never hit; either the key varies per step or the \
                 recording does not survive one"
            );
        }
        out
    };

    let with = sampled(true);
    let without = sampled(false);
    assert_eq!(
        with, without,
        "reusing a recorded fire changed what the model said"
    );
}

/// **A steady decode reuses its recording; a prefill invalidates it.**
///
/// What the cache actually does, as counters. Every decode after the first of
/// its shape is one `vkQueueSubmit` of a command buffer somebody else
/// recorded; a prefill in between is a different lowering and a different
/// arena and must go all the way back to planning.
///
/// The tolerance on the decode side is not slack: a step whose table
/// reallocated -- which is what crossing a KV page boundary does -- has a
/// device allocation count the key does not match, so it re-plans by design.
/// With eight rows to a page, fifty steps cross six or seven boundaries.
#[test]
fn a_steady_decode_reuses_its_recording_and_a_prefill_does_not() {
    if !kernels_vulkan::embedded() {
        eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
        return;
    }
    let Some(real) = weights() else {
        return;
    };
    const STEPS: usize = 50;

    let mut shell = shelled(real, 128);
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..4 {
        prompt.extend_from_slice(&PERIOD);
    }
    step(&mut shell, prompt);
    for _ in 0..3 {
        step(&mut shell, vec![PERIOD[0]]);
    }
    let before = (
        shell.plans().replays(),
        shell.plans().records(),
        shell.plans().planned(),
    );
    for _ in 0..STEPS {
        step(&mut shell, vec![PERIOD[0]]);
    }
    let replays = shell.plans().replays() - before.0;
    let records = shell.plans().records() - before.1;
    let planned = shell.plans().planned() - before.2;
    eprintln!(
        "\n{STEPS} steady decode steps: {replays} replays, {records} re-records, \
         {planned} full plans"
    );
    assert_eq!(
        replays + records + planned,
        STEPS as u64,
        "every step is one of the three outcomes"
    );
    assert!(
        replays >= (STEPS as u64) * 3 / 4,
        "a steady decode should replay nearly every step; it replayed {replays} \
         of {STEPS}"
    );

    // A second conversation's prefill: a different lowering, so the held fire
    // must not be handed to it.
    let planned_before = shell.plans().planned();
    shell
        .step(&[driver_vulkan::turns::Turn {
            who: 2,
            tokens: PERIOD.to_vec(),
        }])
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        shell.plans().planned() > planned_before,
        "a prefill reused a decode's recording"
    );
}

/// **A conversation that crosses a split boundary replays the new grid.**
///
/// The one hazard the flash decode adds to this cache. `attn::decode_splits`
/// makes the decode's `vkCmdDispatch` z extent a function of the history, so
/// unlike everything else a decode records, ONE recorded number now follows
/// the sequence's length. It moves at powers of two -- 24 tokens of history
/// split four ways, 64 split eight, 128 sixteen, 256 thirty-two -- and a key
/// blind to it would keep replaying the grid from before the crossing. That
/// is not a crash and not a drift: it is half the history silently
/// unattended, at exactly one token out of a hundred and thirty.
///
/// So: a hundred and thirty greedy steps from a 24-token prompt, which walks
/// 24 -> 154 and crosses THREE boundaries, decoded twice -- once with the
/// plan cache on and once with it off -- and the two must say the same words.
/// `crate::binding::FireNumber::KvHistoryBucket` is folded into
/// `turns::state_of`, so each crossing misses the key and re-plans; the
/// counter below is what says the crossings happened rather than being
/// rounded away.
#[test]
fn a_history_crossing_a_split_boundary_decodes_what_a_replanned_one_does() {
    if !kernels_vulkan::embedded() {
        eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
        return;
    }
    let Some(real) = weights() else {
        return;
    };
    // 24 + 130 = 154 tokens of history: past 32, past 64 and past 128.
    const STEPS: usize = 130;

    // The splits those buckets ask for, so a reader can see the grid move.
    // Sixteen query heads and one row is qwen3-0.6b's decode.
    let splits: Vec<i32> = [32, 64, 128, 256]
        .iter()
        .map(|b| kernels_vulkan::attn::decode_splits(*b, 16, 1))
        .collect();
    eprintln!("\nsplits at buckets 32/64/128/256: {splits:?}");
    assert!(
        splits.windows(2).all(|w| w[0] != w[1]),
        "this test is only a test if the grid actually changes: {splits:?}"
    );

    let sampled = |reuse: bool| -> (Vec<u32>, u64) {
        let mut shell = shelled(real, 128);
        if !reuse {
            shell.plans().disable();
        }
        let mut prompt: Vec<u32> = Vec::new();
        for _ in 0..4 {
            prompt.extend_from_slice(&PERIOD);
        }
        let mut out = Vec::with_capacity(STEPS);
        let mut next = argmax(&step(&mut shell, prompt));
        for _ in 0..STEPS {
            out.push(next);
            next = argmax(&step(&mut shell, vec![next]));
        }
        (out, shell.plans().replays())
    };

    let (with, replays) = sampled(true);
    let (without, _) = sampled(false);
    eprintln!("  {replays} replays over {STEPS} steps across three crossings");
    assert!(
        replays > 0,
        "nothing was replayed, so this proves nothing about replaying"
    );
    assert_eq!(
        with, without,
        "a conversation that crossed a split boundary said something else \
         when its fires were replayed; the recorded grid went stale"
    );
}
