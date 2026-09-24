use engine_cuda::device::elastic::map_unit_for;
use engine_cuda::device::elastic::{budget_bytes, safety_floor_bytes};
use engine_cuda::store::{
    Accounting, BODIES_FLOOR_BYTES, bodies_allowance, decoded_weight_reserve, pages_within,
    program_scratch_reserve, tokens_within,
};
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
    the_reservations_scale_to_a_twenty_four_gigabyte_card();
    the_decoded_weight_tiles_are_held_out_of_the_pool();
}

fn the_decoded_weight_tiles_are_held_out_of_the_pool() {
    // pie-evals nightly 35998007139 on an L40S: gemma-4-31b 4-bit fitted its
    // pool to 6456 MiB with 1024 MiB held for the guests, then the first
    // prefill's `linear.matmul` died in `cudaMalloc`: an affine plane is
    // decoded to bf16 before the dense gemm, into a scratch the pool never
    // held out. The widest such plane is the lm_head, vocab 262144 by hidden
    // 5376; the mlp planes are 21504 by 5376.
    const VOCAB: u64 = 262_144;
    const HIDDEN: u64 = 5376;
    const INTERMEDIATE: u64 = 21_504;
    const POOL_ROOM: u64 = 6456 << 20;
    const ONE_SLOT_4K: u64 = 3520 << 20;
    const GRAIN: u64 = 8 << 20;

    let lm_head = VOCAB * HIDDEN * 2;
    let mlp = INTERMEDIATE * HIDDEN * 2;
    assert_eq!(lm_head, 2_818_572_288);
    assert_eq!(mlp, 231_211_008);

    let tile = decoded_weight_reserve(lm_head, 1);
    assert_eq!(
        tile,
        lm_head.next_multiple_of(GRAIN),
        "one tile a stream, in the scratch grain"
    );
    assert_eq!(
        decoded_weight_reserve(lm_head, 2),
        2 * tile,
        "a side stream holds its own"
    );
    assert_eq!(
        decoded_weight_reserve(0, 4),
        0,
        "a dense load decodes nothing"
    );
    assert!(
        POOL_ROOM - tile >= ONE_SLOT_4K,
        "held out of the fitted pool, the L40S still seats a 4k sequence: {} MiB left",
        (POOL_ROOM - tile) >> 20
    );
    assert!(
        60 * 7 * mlp > 48 << 30,
        "one tile a region, as it was, is more than the card: {} MiB",
        (60 * 7 * mlp) >> 20
    );
}

fn the_reservations_scale_to_a_twenty_four_gigabyte_card() {
    // Issue #662: an RTX 4090 (24564 MiB) at gpu_mem_utilization 0.9 with
    // gemma-4-26b-a4b 4-bit, 13591 MiB of weights, 60 kv planes of 3755 bytes a
    // token in bf16 (1760 MiB a sequence at 8192), an 8192 envelope and 64 lanes. The fixed reservations —
    // 4 GiB of bodies_mem, an arena and attention workspaces sized for 8192
    // tokens (5230 MiB together), a first slot in 32 MiB map units — were
    // refused with "leaves 0 bytes for the cache rows after 4294967296 held".
    const CARD: u64 = 25_757_220_864;
    const WEIGHTS: u64 = 13591 << 20;
    const PLANES: u64 = 60;
    const PLANE_TOKEN: u64 = 3755;
    const GRANULARITY: u64 = 2 << 20;
    const HANDLE: u64 = 32 << 20;
    const LANES: u32 = 64;
    const ASKED_TOKENS: u32 = 8192;
    const WORKING_A_TOKEN: u64 = 669_440;
    const PROGRAMS: u64 = 64 * 4 * (262_144 * 4);

    let floor = safety_floor_bytes(CARD);
    let after_weights = budget_bytes(CARD, CARD, 0.90) - WEIGHTS;
    assert_eq!(after_weights, (CARD as f64 * 0.90) as u64 - floor - WEIGHTS);
    assert!(after_weights > 8 << 30 && after_weights < 9 << 30);

    let working = |tokens: u32| u64::from(tokens) * WORKING_A_TOKEN;
    assert!(
        working(ASKED_TOKENS) > after_weights / 2,
        "the 8192 working set takes over half"
    );
    let tokens = tokens_within(ASKED_TOKENS, LANES.max(1024), after_weights / 2, working);
    assert_eq!(tokens, 4096, "one halving fits it in half the room");
    assert_eq!(
        tokens_within(ASKED_TOKENS, 1024, u64::MAX, working),
        ASKED_TOKENS,
        "a roomy card serves what was asked"
    );
    assert_eq!(
        tokens_within(ASKED_TOKENS, 1024, 0, working),
        1024,
        "and the halving stops at the floor rather than refusing"
    );

    let plane = |context: u64| context * PLANE_TOKEN;
    let coarse = |context: u64| PLANES * plane(context).div_ceil(HANDLE) * HANDLE;
    let fine = |context: u64| {
        let unit = map_unit_for(plane(context), GRANULARITY, HANDLE);
        PLANES * plane(context).div_ceil(unit) * unit
    };
    assert_eq!(
        coarse(8192),
        2_013_265_920,
        "the 1920 MiB first slot the issue names"
    );
    assert_eq!(
        fine(8192),
        1800 << 20,
        "in the 2 MiB unit a 28 MiB plane earns"
    );
    assert_eq!(
        map_unit_for(4 << 20, GRANULARITY, HANDLE),
        GRANULARITY,
        "gpt-oss at 4096: a 4 MiB plane maps in 2 MiB units, 192 MiB a slot, not 1536"
    );
    assert_eq!(
        map_unit_for(16 << 30, GRANULARITY, HANDLE),
        HANDLE,
        "a huge plane keeps the handle"
    );

    let room = after_weights - working(tokens) - PROGRAMS;
    let fixed = room.saturating_sub(4 << 30);
    assert!(
        fixed < coarse(8192),
        "4 GiB of bodies_mem leaves under one coarse slot: {fixed}"
    );

    let bodies = bodies_allowance(4 << 30, room - fine(8192));
    assert!(
        bodies < 4 << 30 && bodies >= BODIES_FLOOR_BYTES,
        "held to a quarter: {bodies}"
    );
    assert_eq!(bodies, (room - fine(8192)) / 4);
    assert_eq!(
        bodies_allowance(4 << 30, 40 << 30),
        4 << 30,
        "a roomy card keeps the config"
    );
    assert_eq!(
        bodies_allowance(4 << 30, 1 << 30),
        BODIES_FLOOR_BYTES,
        "the floor holds"
    );
    assert_eq!(
        bodies_allowance(4 << 30, 300 << 20),
        300 << 20,
        "and yields to the sequence"
    );
    assert_eq!(
        bodies_allowance(256 << 20, 40 << 30),
        256 << 20,
        "a small config is kept"
    );

    let seats = |context: u64, slot: u64| (room - bodies) / slot;
    assert!(
        seats(8192, fine(8192)) >= 1,
        "the 8k envelope seats a sequence"
    );
    assert!(
        seats(4096, fine(4096)) >= 3,
        "and a 4k one seats a few: {}",
        seats(4096, fine(4096))
    );
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
