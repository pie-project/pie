//! Pins that `gpu_mem_utilization` reaches the shell: `engine_cuda::open`
//! refuses an illegal fraction, `budget_bytes` turns it into pool bytes,
//! and `Accounting` sums weight tier + pool + safety floor against the card.

use engine_cuda::device::elastic::{budget_bytes, safety_floor_bytes};
use engine_cuda::store::Accounting;
use engine_cuda::{DeviceBoot, Knobs};

/// The card the gap was measured on: an L40S, 48,305,799,168 bytes.
const CARD: u64 = 48_305_799_168;

/// What an uncapped gpt-oss load leaves resident on the card.
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

#[test]
fn the_boot_carries_the_fraction_and_absence_is_the_configs_default() {
    engine_cuda::open(boot_with(0.75), no_contract(), |name| models::sku(name).map(|sku| sku.classify)).expect("a fraction in range opens");

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

    engine_cuda::open(boot_with(1.0), no_contract(), |name| models::sku(name).map(|sku| sku.classify)).expect("the whole card opens");
}

#[test]
fn an_out_of_range_fraction_refuses_at_boot_by_the_knobs_name() {
    for fraction in [0.0, 1.5, -0.25, f64::NAN, f64::INFINITY] {
        let refusal = engine_cuda::open(boot_with(fraction), no_contract(), |name| models::sku(name).map(|sku| sku.classify))
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

    let uncapped = budget_bytes(free, CARD, 1.0);
    assert_eq!(uncapped, 34_410_299_648);

    let asked = budget_bytes(free, CARD, 0.90);
    assert_eq!(asked, 29_579_719_731);
    assert_eq!(asked + WEIGHTS + floor, (CARD as f64 * 0.90) as u64);
    assert!(
        uncapped - asked > 4 * (1 << 30),
        "the gap this wave closes is nearly five gigabytes: {uncapped} vs {asked}"
    );

    // A fraction under what is already resident is zero, not a wrap.
    assert_eq!(budget_bytes(free, CARD, 0.10), 0);
}

#[test]
fn the_card_does_not_hold_a_deployment_whose_weights_leave_no_context() {
    let floor = safety_floor_bytes(CARD);

    let roomy = Accounting::of(CARD, 0.90, WEIGHTS, 2 << 30);
    assert_eq!(roomy.card, CARD);
    assert_eq!(roomy.weights, WEIGHTS);
    assert_eq!(roomy.floor, floor);
    assert_eq!(roomy.pool, 29_579_719_731);
    assert_eq!(
        roomy.pool + roomy.weights + roomy.floor,
        roomy.ceiling,
        "weight tier + elastic pool + safety floor = the operator's share of the card"
    );
    roomy.admit().expect("29.6 GB holds a 2 GiB sequence");

    // A weight tier that leaves the pool below one sequence at the declared context.
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

    // The fraction decides too, not only the weights: the same load the
    // whole card holds is refused at a fraction that does not.
    let demand = 32 << 30;
    Accounting::of(CARD, 1.0, WEIGHTS, demand)
        .admit()
        .expect("the whole card holds it");
    assert!(
        Accounting::of(CARD, 0.90, WEIGHTS, demand).admit().is_err(),
        "nine tenths of the same card does not"
    );
}
