//! **THE OPERATOR'S `gpu_mem_utilization` REACHES A SHELL** (alto streaming
//! §3 item 5, `.wiki/alto/next.md` B1 and B2).
//!
//! The key was declared in `worker::config`, defaulted to `0.90`, validated in
//! `(0.0, 1.0]` and put in the config schema — and `grep` found no reader in
//! any shell. `PhysicalPool::open` took `cudaMemGetInfo().free` less a safety
//! floor, which is ~100% of whatever the card had left: on the L40S this
//! workspace serves from, 34.4 GB where an operator who wrote `0.90` asked for
//! 29.6.
//!
//! This file is the route, gated at every hop that does not need a device.
//! The fraction rides `Knobs::gpu_mem_utilization` on the typed `DeviceBoot`
//! now — a field cannot fail to arrive the way the old boot document's key
//! did — so the hop this file once made about parsing is about VALIDATION:
//!
//! ```text
//! worker::config            declares, defaults, validates   (its own tests)
//! DeviceBoot.knobs          the fraction arrives typed      (the compiler)
//! engine_cuda::open         refuses an illegal fraction     HERE
//! elastic::budget_bytes     turns it into the pool's bytes  HERE
//! store::Accounting         states the whole sum, ahead     HERE
//! Shell::load               refuses / opens the pool        the GPU smoke
//! ```
//!
//! # Why the arithmetic is a `pub fn` and not only a constructor
//!
//! [`PhysicalPool::open`](engine_cuda::device::elastic::PhysicalPool::open)
//! needs a card. The SUM does not, and the sum is the whole of what the
//! fraction changed — so it is
//! [`budget_bytes`](engine_cuda::device::elastic::budget_bytes), a pure
//! function of three numbers, and this file checks it against the real numbers
//! from the box the gap was measured on rather than against a device that may
//! not be here.
//!
//! ```text
//! cargo test -p engine-cuda --test the_operators_fraction_sizes_the_pool
//! ```

use engine_cuda::device::elastic::{budget_bytes, safety_floor_bytes};
use engine_cuda::store::Accounting;
use engine_cuda::{DeviceBoot, Knobs};

/// The card the gap was measured on: an L40S, 48,305,799,168 bytes.
const CARD: u64 = 48_305_799_168;

/// What an uncapped gpt-oss load leaves on it — streaming §3 item 5's own
/// number, and the reason the fraction matters at all.
const WEIGHTS: u64 = 13_761_281_792;

/// A boot with the operator's fraction stated, everything else the default.
fn boot_with(fraction: f64) -> DeviceBoot {
    DeviceBoot {
        knobs: Knobs {
            gpu_mem_utilization: fraction,
            ..Knobs::default()
        },
        ..DeviceBoot::default()
    }
}

/// A contract lookup for a gate that never loads: `open` must not need one.
fn no_contract() -> engine_cuda::ContractFor {
    |_, _| Err("this gate opens a boot and never loads a model".to_string())
}

/// **HOP 2/3: the fraction the boot states is the fraction the shell holds,
/// and absence is the config's own default.**
///
/// The parsing half of this hop died with the boot document — a struct field
/// cannot fail to arrive the way `weight_cache_dir` once did — so what is
/// left to state is the DEFAULT: `Knobs::default()` must mean what the absent
/// config key has always meant.
#[test]
fn the_boot_carries_the_fraction_and_absence_is_the_configs_default() {
    engine_cuda::open(boot_with(0.75), no_contract()).expect("a fraction in range opens");

    // ABSENT IS THE CONFIG'S OWN DEFAULT, not the whole card: `0.90` is what
    // `worker::config` has meant by an absent key since before the palo
    // rewrite, so the shell's absence and the operator's absence are one
    // number.
    assert!(
        (Knobs::default().gpu_mem_utilization - engine_cuda::DEFAULT_GPU_MEM_UTILIZATION).abs()
            < f64::EPSILON,
    );
    assert!(
        (engine_cuda::DEFAULT_GPU_MEM_UTILIZATION - 0.90).abs() < f64::EPSILON,
        "and that default is 0.90, which is what the worker's config says"
    );
    assert!(
        (DeviceBoot::default().knobs.gpu_mem_utilization - 0.90).abs() < f64::EPSILON,
        "a boot that states nothing is the same answer"
    );

    // The whole card is a legal statement, and it has to be: it is the
    // arithmetic the pool had before the fraction reached it, so an operator
    // must be able to ask for it back by name.
    engine_cuda::open(boot_with(1.0), no_contract()).expect("the whole card opens");
}

/// **A fraction outside `(0.0, 1.0]` refuses AT BOOT, by the knob's name.**
///
/// Clamping would be the silent-wrongness answer: `1.7` would serve as `1.0`
/// and `0.0` would serve as a pool of nothing, and in neither case would the
/// operator learn that the number they wrote is not the number in force.
/// The refusal moved from the document reader to [`engine_cuda::open`] when
/// the boot became a struct; it is the one semantic check the reader did
/// that the type system cannot.
#[test]
fn an_out_of_range_fraction_refuses_at_boot_by_the_knobs_name() {
    for fraction in [0.0, 1.5, -0.25, f64::NAN, f64::INFINITY] {
        let refusal = engine_cuda::open(boot_with(fraction), no_contract())
            .err()
            .unwrap_or_else(|| panic!("`{fraction}` is not a deployment"));
        assert!(
            refusal.contains("gpu_mem_utilization"),
            "the refusal names the knob; got: {refusal}"
        );
        assert!(
            refusal.contains(&format!("{fraction}")) || fraction.is_nan(),
            "and the value it was given; got: {refusal}"
        );
    }
}

/// **HOP 4a: `utilization = 1.0` IS THE OLD ARITHMETIC, BYTE FOR BYTE.**
///
/// The A/B arm. A change to a memory budget that cannot be turned off is a
/// change nobody can bisect, so the pre-fraction pool is not a deleted branch
/// — it is the top of this function's range, and `free - floor` is what it
/// answers there.
#[test]
fn the_whole_card_is_the_arithmetic_the_pool_had_before() {
    let floor = safety_floor_bytes(CARD);
    for free in [CARD, CARD - WEIGHTS, 1 << 30, floor + 1] {
        assert_eq!(
            budget_bytes(free, CARD, 1.0),
            free - floor,
            "at 1.0 the fraction is not in the arithmetic at all"
        );
    }
}

/// **HOP 4b: the fraction is of the CARD, and charges what is already on it.**
///
/// `gpu_mem_utilization` is the operator's ceiling over pie's WHOLE footprint,
/// weights included — which is what the key's own doc in `worker::config` says
/// — so the fraction multiplies `total` and the weight store, already
/// allocated by the time the pool opens, is subtracted from the result rather
/// than left outside it.
#[test]
fn the_fraction_is_of_the_card_and_charges_what_is_already_on_it() {
    let floor = safety_floor_bytes(CARD);
    assert_eq!(
        floor,
        128 * 1024 * 1024,
        "min(128 MiB, card/10) on this card"
    );

    // The measured situation: an uncapped gpt-oss load, then the pool opens.
    let free = CARD - WEIGHTS;

    // What it took before the fraction reached it — streaming §3 item 5's
    // "34.4 GB".
    let uncapped = budget_bytes(free, CARD, 1.0);
    assert_eq!(uncapped, 34_410_299_648);

    // What an operator who wrote `0.90` asked for. §3 item 5 computes
    // `0.90 x 48,305,799,168 - 13,761,281,792 = 29.7 GB` and leaves the safety
    // floor out of its arithmetic; the pool holds the floor back as it always
    // has, so the number is that one less 128 MiB.
    let asked = budget_bytes(free, CARD, 0.90);
    assert_eq!(asked, 29_579_719_731);
    assert_eq!(asked + WEIGHTS + floor, (CARD as f64 * 0.90) as u64);
    assert!(
        uncapped - asked > 4 * (1 << 30),
        "the gap this wave closes is nearly five gigabytes: {uncapped} vs {asked}"
    );

    // A fraction under what is ALREADY on the card is zero, not a wrap. It is
    // reachable — an operator can write `0.10` under a weight store that is a
    // third of the card — and the honest answer is a pool of nothing, which
    // `Accounting` then refuses by name.
    assert_eq!(budget_bytes(free, CARD, 0.10), 0);
}

/// **B2: THE UNIFIED ACCOUNTING SENTENCE** — *weight tiers + elastic pool +
/// safety floor = the card* — and the one refusal it makes possible.
///
/// The two accountings have always summed correctly and by ORDER: the weight
/// store is a `cudaMalloc`, the pool opens afterwards against what is left. So
/// the sum was right and unwritten, and nothing could refuse AHEAD of it. This
/// is the sum written down.
#[test]
fn the_card_does_not_hold_a_deployment_whose_weights_leave_no_context() {
    let floor = safety_floor_bytes(CARD);

    // A deployment that fits: the measured gpt-oss weight tier at `0.90`, and
    // a one-slot minimum of two gigabytes.
    let roomy = Accounting::of(CARD, 0.90, WEIGHTS, 2 << 30);
    assert_eq!(roomy.card, CARD);
    assert_eq!(roomy.weights, WEIGHTS);
    assert_eq!(roomy.floor, floor);
    assert_eq!(roomy.pool, 29_579_719_731);
    // The sentence: the four terms are the card's fraction, exactly.
    assert_eq!(
        roomy.pool + roomy.weights + roomy.floor,
        roomy.ceiling,
        "weight tier + elastic pool + safety floor = the operator's share of the card"
    );
    roomy.admit().expect("29.6 GB holds a 2 GiB sequence");

    // **THE REFUSAL.** The same card and the same fraction, under a weight
    // tier that leaves the pool below one sequence at the declared context.
    // Before this sum existed the load went through, the pool opened tiny, and
    // the deployment discovered its `max_context` was unreachable as an
    // unrelated `Exhausted` on some later fire.
    let tight = Accounting::of(CARD, 0.90, 42 << 30, 4 << 30);
    let refusal = tight
        .admit()
        .expect_err("a pool under one slot at the declared context is not a deployment")
        .to_string();
    for term in [
        &CARD.to_string(),
        &tight.ceiling.to_string(),
        &tight.weights.to_string(),
        &tight.floor.to_string(),
        &tight.pool.to_string(),
        &tight.minimum.to_string(),
    ] {
        assert!(
            refusal.contains(term.as_str()),
            "the refusal spells every term of the sentence; {term} missing from: {refusal}"
        );
    }
    assert!(
        refusal.contains("gpu_mem_utilization") && refusal.contains("device_weight_budget"),
        "and it names the two keys that change the answer: {refusal}"
    );

    // **AND IT IS THE FRACTION THAT DECIDES, NOT ONLY THE WEIGHTS.** The same
    // load the whole card holds is refused at a fraction that does not — which
    // is what makes this a statement about the operator's declaration rather
    // than about the hardware.
    let demand = 32 << 30;
    Accounting::of(CARD, 1.0, WEIGHTS, demand)
        .admit()
        .expect("the whole card holds it");
    assert!(
        Accounting::of(CARD, 0.90, WEIGHTS, demand).admit().is_err(),
        "nine tenths of the same card does not"
    );
}
