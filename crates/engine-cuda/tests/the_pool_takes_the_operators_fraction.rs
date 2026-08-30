//! **THE FRACTION'S LAST HOP, ON A CARD** (alto streaming §3 item 5,
//! `.wiki/alto/next.md` B1 and B2).
//!
//! [`the_operators_fraction_sizes_the_pool`] gates every hop that does not
//! need a device: the boot document reaches `Knobs`, the arithmetic is what
//! streaming §3 item 5 says it is, and the unified accounting sentence refuses
//! the deployment the card does not hold. What it cannot gate is the last hop
//! — `Shell::load` -> `Pools::reserve` -> `PhysicalPool::open` -> a budget in
//! logical pages — because that one reads a card.
//!
//! This is that hop, as an A/B on one machine:
//!
//! ```text
//! load at 1.0    the pool takes what the card had free, less the floor
//! load at 0.45   the pool takes strictly less, and the sentence closes:
//!                pool + weights + floor <= card x 0.45
//! ```
//!
//! **THE A/B IS THE GATE AND NOT A NICETY.** A single load says only that the
//! pool got a number; two loads that differ ONLY in the fraction say the
//! number is a function of it. That is the difference between this test and
//! the four waves during which the key was declared, defaulted, validated,
//! schema'd — and read by nothing.
//!
//! # Gating
//!
//! Skips at RUN time, saying which of the machine and the checkpoint was
//! missing, like every other gate in this directory.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test the_pool_takes_the_operators_fraction -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use engine_cuda::device::elastic::safety_floor_bytes;
use engine_cuda::{Boot, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

/// The shipping dense SKU, as every gate in this directory uses it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Tokens per kv page.
const PAGE: u32 = 16;

/// A deliberately large kv reservation — the pool's ADDRESS SPACE is meant to
/// dwarf what any fraction of the card will back, so that what the budget says
/// is a statement about the fraction and never about the reservation.
const CONTEXT: u32 = 8192;
/// Sequence seats.
const SLOTS: u32 = 8;

/// The stated fraction. Well under one, so the two arms are far apart on any
/// card, and well over the point where a 0.8B model's weights plus one
/// sequence stop fitting.
const FRACTION: f64 = 0.45;

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

/// One boot at one stated fraction. Everything else is held fixed, which is
/// the whole of the experiment.
fn load(what: &str, utilization: f64) -> Option<Shell> {
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

    Some(
        Shell::load(Boot {
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
            // Eager: this gate is about bytes, and a captured replay would add
            // a variable that has nothing to do with them.
            graphs: engine_cuda::Graphs::Off,
            // **THE ONE WORD THAT MOVES BETWEEN THE ARMS.**
            knobs: engine_cuda::Knobs {
                gpu_mem_utilization: utilization,
                ..engine_cuda::Knobs::default()
            },
            program_cache_dir: None,
            runahead: engine::runahead::Runahead::F1,
            weight_cache_dir: None,
        })
        .expect("the shell loads"),
    )
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 30) as f64
}

/// **The pool's budget, in bytes** — `elastic_budget_pages x
/// elastic_page_bytes`, off the same accessor `PoolFacts` publishes to the
/// runtime. There is no probe feature to add: article 8 already made the
/// engine the owner of this number and `Shell::elastic` already answers it.
fn pool_budget(shell: &Shell) -> u64 {
    let (_committed, _high_water, page_bytes, budget_pages) = shell.elastic();
    budget_pages * page_bytes
}

#[test]
fn the_pool_takes_the_operators_fraction() {
    let Some(whole) = load("the whole-card arm", 1.0) else {
        return;
    };
    let uncapped = pool_budget(&whole);
    let open = whole.accounting();
    let card = open.card;
    let weights = open.weights;
    let floor = safety_floor_bytes(card);
    // **WHAT ELSE IS ON THE CARD, MEASURED PER ARM.** Everything this load
    // allocates after the pool opens is the same in both arms and cancels
    // below; what does NOT cancel is another process arriving or leaving
    // between them, and this is how much of it there was.
    let residual_whole = card - engine_cuda::device::free_bytes().expect("a bound device");
    eprintln!(
        "card {:.2} GiB, weight tier {:.2} GiB, floor {:.0} MiB, resident {:.2} GiB",
        gib(card),
        gib(weights),
        floor as f64 / (1 << 20) as f64,
        gib(residual_whole)
    );
    eprintln!(
        "  gpu_mem_utilization = 1.00 -> pool budget {:.2} GiB (predicted {:.2} GiB)",
        gib(uncapped),
        gib(open.pool)
    );
    // At `1.0` the fraction is not in the arithmetic at all, so the pool takes
    // what the card had free less the floor — bounded above by the sentence's
    // own prediction, which charges only THIS load's weights. The difference
    // between the two is the CUDA context, the arena and any other tenant, and
    // it is measured rather than asserted about.
    assert_eq!(open.ceiling, card, "1.0 of the card is the card");
    assert!(
        uncapped <= open.pool,
        "the pool cannot exceed what the sentence promised it: {uncapped} > {}",
        open.pool
    );
    drop(whole);

    let Some(capped) = load("the stated-fraction arm", FRACTION) else {
        return;
    };
    let asked = pool_budget(&capped);
    let accounting = capped.accounting();
    let residual_capped = card - engine_cuda::device::free_bytes().expect("a bound device");
    eprintln!(
        "  gpu_mem_utilization = {FRACTION:.2} -> pool budget {:.2} GiB (predicted {:.2} GiB, \
         resident {:.2} GiB)",
        gib(asked),
        gib(accounting.pool),
        gib(residual_capped)
    );

    // **THE SENTENCE, ON A CARD.** Weight tier + elastic pool + safety floor,
    // inside the operator's fraction of the device. This is the claim B2 asked
    // to be written down, checked against the numbers the shell actually took
    // rather than against the numbers it predicted.
    assert_eq!(accounting.ceiling, (card as f64 * FRACTION) as u64);
    assert!(
        asked + weights + floor <= accounting.ceiling,
        "weight tier {weights} + pool {asked} + floor {floor} must fit inside {} — the \
         operator's {FRACTION} of a {card}-byte card",
        accounting.ceiling
    );
    // And the prediction is an upper bound here too, for the same reason.
    assert!(
        asked <= accounting.pool,
        "the pool cannot exceed what the sentence promised it: {asked} > {}",
        accounting.pool
    );

    // **THE A/B, AND IT IS AN EQUALITY.** Both arms subtract the same
    // occupancy — the same weights, the same arena, the same context — so what
    // is left between them is the part of the card the fraction withheld, plus
    // whatever a NEIGHBOUR did between the two loads:
    //
    // ```text
    // uncapped = 1.00 × card - used_a - floor
    // asked    = 0.45 × card - used_b - floor
    // ----------------------------------------
    // uncapped - asked = (card - ceiling) - (used_a - used_b)
    // ```
    //
    // and `used_a - used_b` is the residual drift, measured above. Everything
    // this load itself allocates is in both terms and cancels, so the equality
    // holds on a busy box as well as a quiet one — which is the whole claim of
    // B1 in one subtraction.
    assert!(
        asked < uncapped,
        "a stated fraction takes strictly less than the whole card: {asked} against {uncapped}"
    );
    let withheld = uncapped - asked;
    let drift = residual_whole.abs_diff(residual_capped);
    let expected = card - accounting.ceiling;
    eprintln!(
        "  withheld {:.2} GiB against the fraction's {:.2} GiB (neighbour drift {:.2} GiB)",
        gib(withheld),
        gib(expected),
        gib(drift)
    );
    assert!(
        withheld.abs_diff(expected) <= drift + (1 << 28),
        "the difference between the arms IS the withheld fraction of the card, once a \
         neighbour's drift of {drift} is allowed for: {withheld} against {expected}"
    );
}
