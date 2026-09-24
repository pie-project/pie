use engine_cuda::device::elastic::{budget_bytes, safety_floor_bytes};
use engine_cuda::store::{Accounting, pages_within, program_scratch_reserve};
use engine_cuda::{DeviceBoot, Knobs};

const CARD: u64 = 48_305_799_168;

const WEIGHTS: u64 = 13_761_281_792;

fn boot_with(fraction: f64) -> DeviceBoot {
    DeviceBoot {
        knobs: Knobs {
            gpu_mem_utilization: fraction,
            ..Knobs::default()
        },
        ..DeviceBoot::default()
    }
}

fn no_contract() -> engine_cuda::ContractFor {
    |_, _| Err("this gate opens a boot and never loads a model".to_string())
}

#[test]
fn the_operators_fraction_sizes_the_pool_every_case() {
    the_boot_carries_the_fraction_and_absence_is_the_configs_default();
    an_out_of_range_fraction_refuses_at_boot_by_the_knobs_name();
    the_whole_card_is_the_arithmetic_the_pool_had_before();
    the_fraction_is_of_the_card_and_charges_what_is_already_on_it();
    the_card_does_not_hold_a_deployment_whose_weights_leave_no_context();
    the_declared_pool_is_sized_to_what_the_card_hands_out();
    the_pool_leaves_room_for_the_programs_guests_register();
}

fn the_pool_leaves_room_for_the_programs_guests_register() {
    // pie-evals nightly 35971363958 on an RTX 5090: gpt-oss-20b at 256 lanes,
    // the card handing out 9765 MiB for the cache rows once the load was
    // resident. text-completion-bench's sampling epilogue holds three f32 rows
    // of the 201088-wide out seam per lane, so its scratch at 256 lanes was
    // 617840640 bytes, allocated at the first fire past static admission —
    // and the pool, fitted to the room, had left 549 MiB.
    const PLANES: u64 = 48;
    const PLANE_PAGE: u64 = 16 * 1024;
    const MAP_UNIT: u64 = 32 << 20;
    const ROOM: u64 = 9765 << 20;
    const LANES: u32 = 256;
    const VOCAB: u64 = 201_088;
    const BENCH_SCRATCH: u64 = 617_840_640;
    let declared_at = |pages: u64| PLANES * (pages * PLANE_PAGE).div_ceil(MAP_UNIT) * MAP_UNIT;
    let asked = 65536;

    let blind = pages_within(asked, ROOM, declared_at);
    assert_eq!(
        blind, 12288,
        "the fit without a program reserve, as the nightly saw it"
    );
    assert!(
        ROOM - declared_at(blind) < BENCH_SCRATCH,
        "and what it left ({}) is under the program's scratch, the c64/c256 refusal",
        ROOM - declared_at(blind)
    );

    let reserve = program_scratch_reserve(LANES, VOCAB * 4);
    assert!(
        reserve >= BENCH_SCRATCH,
        "the reserve at {LANES} lanes covers the program: {reserve} vs {BENCH_SCRATCH}"
    );
    assert_eq!(
        reserve,
        256 * 804_352 * 4,
        "four out-seam rows a lane, at 256 lanes"
    );

    let fit = pages_within(asked, ROOM - reserve, declared_at);
    assert!(
        fit < blind && fit >= 4096,
        "the pool yields, and still seats sequences: {fit}"
    );
    assert!(
        ROOM - declared_at(fit) >= BENCH_SCRATCH,
        "what the fitted pool leaves holds the program's scratch"
    );

    assert_eq!(
        program_scratch_reserve(300, VOCAB * 4),
        program_scratch_reserve(512, VOCAB * 4),
        "the shell grows a program's scratch by doublings, so the reserve rounds the lanes up"
    );
    assert_eq!(
        program_scratch_reserve(LANES, 0),
        0,
        "a plan with no out seam registers no sampler"
    );
    assert_eq!(
        program_scratch_reserve(u32::MAX, u64::MAX),
        eta_exec::SCRATCH_MAX_BYTES,
        "and never past what the program contract lets one program take"
    );
}

fn the_declared_pool_is_sized_to_what_the_card_hands_out() {
    // gpt-oss-20b at 16 tokens a page: 24 layers of a key and a value plane,
    // each 512 wide in bf16, so 16 KiB of every page lands in each of 48
    // planes and each plane is backed in 32 MiB map units. The config's
    // 256-slot default at a 4096 context asks 65536 pages, the 49152 MiB
    // pool issue #630 saw declared on a 32 GB card.
    const PLANES: u64 = 48;
    const PLANE_PAGE: u64 = 16 * 1024;
    const MAP_UNIT: u64 = 32 << 20;
    let declared_at = |pages: u64| PLANES * (pages * PLANE_PAGE).div_ceil(MAP_UNIT) * MAP_UNIT;
    let asked = 65536;
    assert_eq!(declared_at(asked), 49152 << 20);

    let room = 13_826_523_136;
    let fit = pages_within(asked, room, declared_at);
    assert_eq!(
        fit, 16384,
        "256 MiB of every plane, the last unit that fits"
    );
    assert!(declared_at(fit) <= room, "what is declared is backed");
    assert!(
        declared_at(fit + 1) > room,
        "and nothing the card could back is left on the table"
    );
    assert_eq!(
        declared_at(fit),
        12288 << 20,
        "the step function lands under the room, not at it"
    );

    assert_eq!(
        pages_within(asked, 49152 << 20, declared_at),
        asked,
        "a pool that fits is the pool that was asked"
    );
    assert_eq!(
        pages_within(asked, u64::MAX, declared_at),
        asked,
        "and never more than asked"
    );
    assert_eq!(
        pages_within(asked, PLANES * MAP_UNIT - 1, declared_at),
        0,
        "under one map unit a plane, not one page fits, which the caller refuses"
    );
    assert_eq!(pages_within(0, room, declared_at), 0);
}

fn the_boot_carries_the_fraction_and_absence_is_the_configs_default() {
    engine_cuda::open(boot_with(0.75), no_contract(), |name| {
        models::sku(name).map(|sku| sku.classify)
    })
    .expect("a fraction in range opens");

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

    engine_cuda::open(boot_with(1.0), no_contract(), |name| {
        models::sku(name).map(|sku| sku.classify)
    })
    .expect("the whole card opens");
}

fn an_out_of_range_fraction_refuses_at_boot_by_the_knobs_name() {
    for fraction in [0.0, 1.5, -0.25, f64::NAN, f64::INFINITY] {
        let refusal = engine_cuda::open(boot_with(fraction), no_contract(), |name| {
            models::sku(name).map(|sku| sku.classify)
        })
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

fn the_fraction_is_of_the_card_and_charges_what_is_already_on_it() {
    let floor = safety_floor_bytes(CARD);
    assert_eq!(
        floor,
        128 * 1024 * 1024,
        "min(128 MiB, card/10) on this card"
    );

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

    assert_eq!(budget_bytes(free, CARD, 0.10), 0);
}

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

    let demand = 32 << 30;
    Accounting::of(CARD, 1.0, WEIGHTS, demand)
        .admit()
        .expect("the whole card holds it");
    assert!(
        Accounting::of(CARD, 0.90, WEIGHTS, demand).admit().is_err(),
        "nine tenths of the same card does not"
    );
}
