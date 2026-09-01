//! **The elastic supply, gated on the three things it claims** (alto design
//! §8, wave C; articles 4, 7 and 8).
//!
//! The reservation model this replaced had one virtue — it could not be wrong
//! about what it held, because it held everything. An elastic pool trades
//! that for three claims, and each of them fails silently if it is not
//! checked:
//!
//! 1. **committed is demand, not ceiling.** A load whose budget ceiling is
//!    deliberately absurd — tens of gigabytes of kv address space — puts
//!    physical pages behind only what a fire actually addresses. Checked twice
//!    over: against the pool's own accounting, and against `cudaMemGetInfo`,
//!    because a counter that moved and a card that did not would be the whole
//!    bug.
//! 2. **a refusal costs nothing.** A frame whose demand the budget will not
//!    cover comes back `Exhausted` with both numbers in it and NOTHING
//!    mapped — which is what makes re-submitting the identical frame the
//!    right response (article 4). The check is that the committed figure is
//!    the same afterwards and that a fitting frame still goes through.
//! 3. **addresses do not move.** Grow, trim, grow again: every arena's base
//!    is the number it was at load. This is article 7, and it is what lets a
//!    `cudaGraphExec_t` recorded before the growth still be correct after it
//!    — the whole reason the pools are virtual ranges rather than a bigger
//!    `cudaMalloc`.
//!
//! # How claim 2 is made to happen on purpose
//!
//! `Exhausted` and `Impossible` are the same sentence at load, when the soft
//! budget and the hard ceiling are both "what the card had free". They come
//! apart exactly when something ELSE takes memory afterwards: the soft budget
//! is recalibrated against a fresh `cudaMemGetInfo` and drops, the hard
//! ceiling does not. So this gate takes the memory itself, with a plain
//! `Buffer`, and then asks for a frame that would have fitted at load. That
//! is not a contrivance — it is the situation the pair of numbers exists for,
//! and the only way to reach it deterministically.
//!
//! # Gating
//!
//! Skips at RUN time, like [`serve_smoke`](../serve_smoke.rs), saying which of
//! the machine and the checkpoint was missing.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test the_pools_commit_what_a_frame_asked_for -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use engine::fire::RsReset;
use engine_cuda::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this gate serves. Any SKU with kv rows would do; this one
/// also carries recurrent state, so both arena kinds are exercised.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Tokens per fire. One page's worth, so the watermark a fire moves is small
/// and legible: a lane with no page table of its own addresses page
/// `base(slot)`, and one with a table addresses exactly the page it names.
const PAGE: u32 = 16;

/// **The absurd ceiling.** `slots * (context / PAGE)` pages of kv address
/// space — tens of gigabytes on any real plan — against which a one-page fire
/// should commit essentially nothing. Address space is free; that is the
/// point being made.
const CONTEXT: u32 = 32768;
/// Sequence seats, and the second half of the ceiling.
const SLOTS: u32 = 32;

/// The page the pointer-stability gate grows to. Thousands of pages up the
/// space, because an arena grows and trims in whole map units: a watermark
/// inside the first unit would trim back to itself and prove nothing.
const GROWN_PAGE: u32 = 8000;

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

/// The lane word the model's own `Classify` computes — runtime-side work,
/// done here because this test IS the runtime for the length of one fire.
fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

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
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
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
        context: CONTEXT,
        slots: SLOTS,
        ordinal: 0,
        // The eager path: this gate is about bytes, and a captured replay
        // would only add a variable that has nothing to do with them.
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    Some(shell)
}

fn tokens() -> Vec<u32> {
    (0..PAGE).map(|at| 1000 + at * 37).collect()
}

/// A lane on the shell's own paging — its pages are `base(slot)` onward.
fn seated<'a>(slot: u32, tokens: &'a [u32]) -> Seated<'a> {
    Seated {
        rs_reset: RsReset::Fresh,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

/// A lane carrying the RUNTIME's page ids — one page, the one it names. The
/// watermark this makes is `page + 1`, which is the whole of why a demand is
/// a watermark and not a count.
fn tabled<'a>(slot: u32, tokens: &'a [u32], pages: &'a [u32]) -> Seated<'a> {
    Seated {
        pages,
        held: Some(0),
        rs_reset: RsReset::Fresh,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

/// **Claim 1**: what a load holds is what a fire asked for.
#[test]
fn the_pools_commit_what_a_frame_asked_for() {
    let Some(mut shell) = ready("the elastic commit") else {
        return;
    };
    let ceiling = shell.footprint().2;
    let (at_load, _, page_bytes, budget_pages) = shell.elastic();
    eprintln!(
        "kv+state ceiling {:.1} MiB, committed at load {:.1} MiB, \
         elastic page {} B, budget {} pages ({:.1} MiB)",
        mib(ceiling),
        mib(at_load),
        page_bytes,
        budget_pages,
        mib(budget_pages * page_bytes),
    );
    assert!(
        ceiling > (4u64 << 30),
        "this gate wants an absurd ceiling to be absurd; {:.1} MiB is not it",
        mib(ceiling),
    );
    assert!(
        page_bytes > 0 && budget_pages > 0,
        "the pools are virtual now and the capability numbers have to say so \
         (`PoolFacts::elastic_page_bytes` / `elastic_budget_pages`)"
    );

    // ── A fire that addresses one page of one slot.
    let free_before = engine_cuda::device::free_bytes().expect("the card answers cudaMemGetInfo");
    shell.open(0).expect("slot 0 opens");
    let ids = tokens();
    let logits = shell
        .fire_seated(&[seated(0, &ids)])
        .expect("the one-page fire runs");
    // **AND THE BYTES UNDER THE ADDRESSES ARE THE RIGHT BYTES.** A kv row's
    // planes are separate arenas now, and a key handle pointed at a value
    // plane would fault nothing — it would answer with numbers. A finite,
    // non-constant logit row is the cheapest observation that says the
    // attention actually read what the append wrote.
    let row = logits.first().expect("the fire answers for its one lane");
    assert!(
        !row.is_empty() && row.iter().all(|value| value.is_finite()),
        "the fire's logits are empty or not finite, so the pools handed the \
         launches something that is not the cache"
    );
    assert!(
        row.iter().any(|value| *value != row[0]),
        "every logit is the same number, which is what a cache read out of \
         the wrong plane looks like"
    );
    let (committed, high_water, _, _) = shell.elastic();
    let free_after = engine_cuda::device::free_bytes().expect("the card answers cudaMemGetInfo");

    eprintln!(
        "after one page: committed {:.1} MiB, high water {:.1} MiB, \
         card free went {:.1} -> {:.1} MiB",
        mib(committed),
        mib(high_water),
        mib(free_before),
        mib(free_after),
    );
    assert!(
        committed > 0,
        "a fire wrote kv and recurrent state and the pools committed nothing, \
         so the launches wrote into unmapped address space"
    );
    assert!(
        committed * 10 < ceiling,
        "a one-page fire committed {:.1} MiB of a {:.1} MiB ceiling — that is \
         the reservation model wearing a virtual costume",
        mib(committed),
        mib(ceiling),
    );
    assert_eq!(
        high_water, committed,
        "nothing has been trimmed yet, so the high water IS what is committed"
    );

    // **THE COUNTER AND THE CARD AGREE.** The pool's accounting says it
    // mapped `committed - at_load` new bytes; `cudaMemGetInfo` has to have
    // seen the same memory leave. A generous band, because the fire also ran
    // kernels and the driver keeps its own scratch — but an order of
    // magnitude off would mean the arenas moved a counter and nothing else.
    let grew = committed.saturating_sub(at_load);
    let took = free_before.saturating_sub(free_after);
    assert!(
        took >= grew / 2,
        "the pools claim {:.1} MiB of new mappings and the card only lost \
         {:.1} MiB — the commit did not reach the device",
        mib(grew),
        mib(took),
    );
}

/// **Claim 2**: a refused frame costs nothing, and the frame after it fits.
#[test]
fn a_refused_frame_maps_nothing_and_the_next_one_fits() {
    let Some(mut shell) = ready("the refusal") else {
        return;
    };
    let ceiling = shell.footprint().2;
    let (_, _, page_bytes, budget_pages) = shell.elastic();
    let hard_bytes = budget_pages.saturating_mul(page_bytes);

    // Settle the pools at a known point first.
    shell.open(0).expect("slot 0 opens");
    let ids = tokens();
    shell
        .fire_seated(&[seated(0, &ids)])
        .expect("the warm-up fire runs");
    let (before, _, _, _) = shell.elastic();

    // The frame to refuse: a page id high enough that backing it costs real
    // memory, but low enough that it is inside both the pool's own ceiling
    // and the hard budget the load was opened against — so the refusal is
    // `Exhausted` (come back) and not `Impossible` (never).
    let per_page = (ceiling / u64::from(SLOTS * (CONTEXT / PAGE))).max(1);
    let want_bytes = (ceiling / 4).min(hard_bytes / 2);
    let high_page = u32::try_from(want_bytes / per_page).unwrap_or(u32::MAX);
    if want_bytes < (512 << 20) || high_page == 0 {
        eprintln!(
            "skipping the refusal: a target of {:.1} MiB is too small to be \
             refused deterministically on this card",
            mib(want_bytes),
        );
        return;
    }
    eprintln!(
        "refusal target: page {high_page} => about {:.1} MiB, against a hard \
         ceiling of {:.1} MiB",
        mib(want_bytes),
        mib(hard_bytes),
    );

    // **TAKE THE MEMORY.** Everything the card has left, so the recalibrated
    // soft budget drops to nothing while the hard ceiling stays where it was
    // at load. This is the situation the two numbers exist to tell apart.
    let free = engine_cuda::device::free_bytes().expect("the card answers cudaMemGetInfo");
    let mut eaten = None;
    for take in [free.saturating_sub(64 << 20), free / 2, free / 4] {
        if take == 0 {
            continue;
        }
        if let Ok(buffer) = engine_cuda::device::Buffer::zeroed(take as usize) {
            eaten = Some(buffer);
            break;
        }
    }
    let Some(eaten) = eaten else {
        eprintln!("skipping the refusal: could not take the card's free memory");
        return;
    };

    let table = [high_page];
    let refusal = shell
        .fire_seated(&[tabled(1, &ids, &table)])
        .expect_err("a frame past the budget is refused");
    let engine_cuda::Fault::OutOfMemory { need, have } = refusal else {
        panic!(
            "a frame the budget will not cover must come back as an exhaustion \
             with both numbers in it, not as {refusal}"
        );
    };
    eprintln!(
        "refused: wanted {:.1} MiB, budget {:.1} MiB",
        mib(need),
        mib(have),
    );
    assert!(
        need > have,
        "an exhaustion whose ask is inside its budget is not an exhaustion"
    );

    // **ZERO SIDE EFFECTS** (article 4). Not one page mapped, not one page
    // unmapped, by the frame that was told no.
    let (after_refusal, _, _, _) = shell.elastic();
    assert_eq!(
        after_refusal, before,
        "a refused frame moved {:.1} MiB of physical mappings — article 4's \
         whole promise is that it moved none",
        mib(after_refusal.abs_diff(before)),
    );

    // And the frame after it, which fits, still goes through.
    drop(eaten);
    shell.open(2).expect("slot 2 opens");
    shell
        .fire_seated(&[seated(2, &ids)])
        .expect("a fitting frame after a refused one still runs");
    let (after_fit, _, _, _) = shell.elastic();
    assert!(
        after_fit >= before,
        "the fitting frame committed less than was already mapped, which is \
         not a thing commit can do"
    );
}

/// **Claim 3**: grow, trim, grow — and every base is where it was.
#[test]
fn a_grow_a_trim_and_a_grow_land_on_the_same_addresses() {
    let Some(mut shell) = ready("pointer stability") else {
        return;
    };
    let at_load = shell.pool_bases();
    assert!(
        !at_load.is_empty() && at_load.iter().all(|base| *base != 0),
        "every arena answers a base before a byte is mapped — that is what \
         makes it addressable at bake time"
    );

    let ids = tokens();
    // ── Grow: a fire high enough up the page space to need MANY map units.
    //    An arena grows and trims in whole map units, so a watermark inside
    //    the first one would be trimmed back to itself and the claim would be
    //    vacuous — this one is thousands of pages up.
    shell.open(0).expect("slot 0 opens");
    let table = [GROWN_PAGE];
    shell
        .fire_seated(&[tabled(0, &ids, &table)])
        .expect("the growing fire runs");
    let grown = shell.elastic().0;
    let after_grow = shell.pool_bases();
    assert_eq!(
        after_grow, at_load,
        "growing the pools moved an arena's base, so every address a captured \
         graph recorded is now wrong (article 7)"
    );

    // ── Trim: hand back everything above one page and one slot. The device
    //    is idle — nothing has been submitted since the fire settled — which
    //    is the condition the unmap is allowed under.
    shell.trim(engine::frame::Demand {
        kv_pages: 1,
        state_slots: 1,
        workspace: 0,
    });
    let trimmed = shell.elastic().0;
    let after_trim = shell.pool_bases();
    eprintln!(
        "grown {:.1} MiB -> trimmed {:.1} MiB",
        mib(grown),
        mib(trimmed),
    );
    assert_eq!(
        after_trim, at_load,
        "trimming the pools moved an arena's base"
    );
    assert!(
        trimmed < grown,
        "a trim from a {GROWN_PAGE}-page watermark to a 1-page one gave \
         nothing back ({:.1} MiB either side), so the tail unmap is not \
         reaching the driver",
        mib(grown),
    );

    // ── And grow again, onto the same addresses and back to the same size.
    shell
        .fire_seated(&[tabled(0, &ids, &table)])
        .expect("the second growing fire runs");
    let regrown = shell.elastic().0;
    assert_eq!(
        shell.pool_bases(),
        at_load,
        "re-growing after a trim moved an arena's base — the cached handle \
         went back somewhere else, which is the one thing a virtual range is \
         supposed to make impossible"
    );
    assert_eq!(
        regrown, grown,
        "the same watermark committed a different number of bytes the second \
         time round"
    );
    let (_, high_water, _, _) = shell.elastic();
    assert!(
        high_water >= grown,
        "the high water forgot the peak it was asked to remember"
    );
}
