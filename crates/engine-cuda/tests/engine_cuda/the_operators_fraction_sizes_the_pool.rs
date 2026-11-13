use engine::Allowance;
use engine::fit::pages_within;
use engine_cuda::device::elastic::map_unit_for;
use engine_cuda::device::elastic::{budget_bytes, safety_floor_bytes};
use engine_cuda::store::kv::Paging;
use engine_cuda::store::{
    Accounting, BODIES_FLOOR_BYTES, PROGRAM_ROWS_A_LANE, bodies_allowance, decoded_weight_reserve,
    holds_within, one_slot_bytes, program_scratch_reserve, tokens_within, window_fire_bytes,
    window_of,
};

const CARD: u64 = 48_305_799_168;

const WEIGHTS: u64 = 13_761_281_792;

#[test]
fn the_operators_fraction_sizes_the_pool_every_case() {
    the_share_is_of_the_card_and_charges_what_is_already_on_it();
    a_stated_budget_caps_what_the_load_takes();
    the_card_does_not_hold_a_deployment_whose_weights_leave_no_context();
    the_pool_leaves_room_for_the_programs_guests_register();
    the_reservations_scale_to_a_twenty_four_gigabyte_card();
    the_decoded_weight_tiles_are_held_out_of_the_pool();
    the_bodies_keep_their_floor_and_the_tile_is_taken_before_any_capture();
    a_twenty_seven_b_serves_on_a_twenty_four_gigabyte_card_at_the_asked_lanes();
    a_sliding_row_holds_its_window_and_not_the_context();
}

fn a_twenty_seven_b_serves_on_a_twenty_four_gigabyte_card_at_the_asked_lanes() {
    // Issue #679: an RTX 4090 (25250627584 bytes, 426901504 in use) at
    // device_memory_share 0.9 with qwen3.6-27b u4g64, 15300601856 of
    // weights, a 4096 envelope and 256 lanes: the load put down an
    // 838467584-byte arena and 2150089216 of inputs, took the lm_head's
    // tile (vocab 248320 by hidden 5120 in bf16) and found no room for one
    // sequence.
    const CARD: u64 = 25_250_627_584;
    const BEFORE: u64 = 426_901_504;
    const WEIGHTS: u64 = 15_300_601_856;
    const ARENA: u64 = 838_467_584;
    const INPUTS: u64 = 2_150_089_216;
    const LANES: u32 = 256;
    const TOKENS: u32 = 4096;
    const VOCAB: u64 = 248_320;
    const HIDDEN: u64 = 5120;
    const INTERMEDIATE: u64 = 17_408;
    // One sequence: 16 full-attention layers' kv at 4096, and the 48
    // linear-attention layers' recurrent state for each of a lane's two slots.
    const SEQUENCE: u64 = 16 * 2 * 4096 * 4 * 256 * 2 + 2 * (48 * 48 * 128 * 128 * 4);
    // A lane's share of the inputs (two prefill workspaces padding a 64-row
    // tile) and of the arena (one vocab-wide bf16 readout row).
    const INPUTS_A_LANE: u64 = 2 * 24 * 64 * 256 * 4;
    const ARENA_A_LANE: u64 = VOCAB * 2;

    // The program reserve the load held then: six out-seam rows a lane (since
    // #1265 the pool holds one and a sampler's derived rows fly narrower), kept
    // here so the boot stays the one the issue saw.
    let issue_reserve =
        |lanes: u32| program_scratch_reserve(lanes, VOCAB * 4) * 6 / PROGRAM_ROWS_A_LANE;
    let live = budget_bytes(CARD - BEFORE, CARD, Allowance::WHOLE) - WEIGHTS;
    let lm_head = decoded_weight_reserve(VOCAB * HIDDEN * 2, 1);
    let mlp = decoded_weight_reserve(INTERMEDIATE * HIDDEN * 2, 1);
    let working = |lanes: u32, tile: u64| {
        let fewer = u64::from(LANES - lanes);
        (ARENA - fewer * ARENA_A_LANE)
            + (INPUTS - fewer * INPUTS_A_LANE)
            + tile
            + issue_reserve(lanes)
            + u64::from(lanes) * u64::from(TOKENS) * 8
    };
    assert!(
        live.saturating_sub(working(LANES, lm_head)) < SEQUENCE,
        "the boot the issue saw: at 256 lanes with the lm_head tile, what is left ({}) is \
         under one sequence ({SEQUENCE})",
        live.saturating_sub(working(LANES, lm_head))
    );

    let room = live - SEQUENCE - BODIES_FLOOR_BYTES;
    assert!(
        tokens_within(LANES, 1, room, |lanes| working(lanes, lm_head)) < LANES,
        "with the tile kept, the lanes would halve to seat one sequence"
    );
    assert_eq!(
        tokens_within(LANES, 1, room, |lanes| working(lanes, mlp)),
        LANES,
        "with the tile held to the widest plane fired at token rows, the asked lanes fit"
    );
}

fn a_sliding_row_holds_its_window_and_not_the_context() {
    // #699: gemma-4-31b's 50 sliding rows (16 kv heads of 256, a window of
    // 1024) were sized to the context beside its 10 global rows (4 of 512),
    // 901120 bytes a token: 31373393920 for one sequence at 34816 cells.
    const GLOBAL: u64 = 10 * 2 * 4 * 512 * 2;
    const SLIDING: u64 = 50 * 2 * 16 * 256 * 2;
    let trace = poem_compiler::catalog::deployment("gemma4-31b-u4g64-kv-bf16")
        .expect("the catalog ships gemma-4-31b")
        .trace(poem::Platform::Cuda);
    let window = window_of(&trace).expect("every sliding read looks through the window");
    assert_eq!(window, Some(1024));
    let at = |context: u32| {
        Paging::of(16, context, 1, 1)
            .expect("a page size of 16 is a paging")
            .windowed(window, 2048)
    };
    let one = |context: u32| one_slot_bytes(&trace, at(context)).expect("the rows size");
    // The window straddles 65 pages and one fire's 2048 rows add 129 more.
    assert_eq!(one(34_816), GLOBAL * 34_816 + SLIDING * 16 * (65 + 129));
    assert_eq!(one(6144), GLOBAL * 6144 + SLIDING * 16 * (65 + 129));
    assert_eq!(
        window_fire_bytes(&trace, at(34_816)).expect("the rows size"),
        SLIDING * 16 * 129,
        "and what one fire's rows add is what `max_forward_tokens` is fitted against"
    );
    assert_eq!(
        one(1024),
        (GLOBAL + SLIDING) * 1024,
        "a window that spans the context is no window"
    );
    assert_eq!(
        at(34_816).window_pages_at(8 * 2176),
        1 + 65 + 129 + 7 * 2176 / 2,
        "past one sequence, a windowed page for every two kv pages"
    );
}

fn the_bodies_keep_their_floor_and_the_tile_is_taken_before_any_capture() {
    // pie-evals nightly 36021487786 on an L40S: gemma-4-31b 4-bit at a 6144
    // envelope. Past one 5280 MiB sequence the card had 6581 MiB; holding
    // the programs (1024 MiB at 256 lanes) and a reserve for the decoded
    // tiles ate it all, bodies_mem was held to 0, nothing armed, and the
    // first fire — a capture — found the `decoded_weight` scratch at 0 bytes
    // and needed 176160768.
    const SPARE: u64 = 6581 << 20;
    const PROGRAMS: u64 = 1024 << 20;
    const CONFIGURED: u64 = 4 << 30;
    const MLP_PLANE: u64 = 21_504 * 5376 * 2;
    const FIRST_CAPTURE: u64 = 176_160_768;

    let (bodies, programs) = holds_within(CONFIGURED, PROGRAMS, SPARE);
    assert!(bodies >= BODIES_FLOOR_BYTES, "the floor holds: {bodies}");
    assert_eq!(programs, PROGRAMS, "and the programs still fit beside it");
    assert!(
        bodies + programs < SPARE,
        "a second sequence's worth is left: {}",
        SPARE - bodies - programs
    );

    let (bodies, programs) = holds_within(CONFIGURED, PROGRAMS, 700 << 20);
    assert_eq!(
        bodies, BODIES_FLOOR_BYTES,
        "on a short card the floor comes first"
    );
    assert_eq!(
        programs,
        (700 << 20) - BODIES_FLOOR_BYTES,
        "and the programs take what is left"
    );
    assert_eq!(
        holds_within(CONFIGURED, PROGRAMS, 0),
        (0, 0),
        "but never past one sequence"
    );
    assert_eq!(
        holds_within(256 << 20, PROGRAMS, SPARE).0,
        256 << 20,
        "a configured allowance under the floor is the floor"
    );

    // The tile a stream is warmed with at load covers the widest plane its
    // regions decode, so the capture's want is under what was taken.
    let tile = decoded_weight_reserve(MLP_PLANE, 1);
    assert!(
        tile >= FIRST_CAPTURE,
        "the mlp stream's tile holds the capture: {tile}"
    );
    assert!(
        (MLP_PLANE..MLP_PLANE + (8 << 20)).contains(&tile),
        "in the scratch grain"
    );
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
    // Issue #662: an RTX 4090 (24564 MiB) at device_memory_share 0.9 with
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
    let after_weights = budget_bytes(CARD, CARD, Allowance::WHOLE) - WEIGHTS;
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
        (BODIES_FLOOR_BYTES..4 << 30).contains(&bodies),
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

    let seats = |slot: u64| (room - bodies) / slot;
    assert!(seats(fine(8192)) >= 1, "the 8k envelope seats a sequence");
    assert!(
        seats(fine(4096)) >= 3,
        "and a 4k one seats a few: {}",
        seats(fine(4096))
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

    // The reserve holds the logits row a program reads, one a lane. The
    // rows a sampler derives from it (three here, six at top-p) are not held
    // back from the pool: a wave whose scratch does not fit flies again
    // narrower (`program::fire`), so the pool keeps the room for sequences.
    let reserve = program_scratch_reserve(LANES, VOCAB * 4);
    assert_eq!(
        reserve,
        256 * 804_352 * PROGRAM_ROWS_A_LANE,
        "one out-seam row a lane at 256 lanes (201088 f32, to the scratch alignment)"
    );

    let fit = pages_within(asked, ROOM - reserve, declared_at);
    assert_eq!(
        fit, blind,
        "the room the nightly's fit left (549 MiB) already holds every lane's logits row, \
         so the pool keeps its size"
    );
    assert!(
        ROOM - declared_at(fit) >= reserve,
        "what the fitted pool leaves holds every lane's logits row"
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

fn the_share_is_of_the_card_and_charges_what_is_already_on_it() {
    let floor = safety_floor_bytes(CARD);
    assert_eq!(
        floor,
        128 * 1024 * 1024,
        "min(128 MiB, card/10) on this card"
    );

    let free = CARD - WEIGHTS;
    let asked = budget_bytes(free, CARD, Allowance::WHOLE);
    assert_eq!(asked, 29_579_719_731);
    assert_eq!(
        asked + WEIGHTS + floor,
        (CARD as f64 * engine::load::DEVICE_SHARE) as u64
    );
    assert_eq!(budget_bytes(1 << 30, CARD, Allowance::WHOLE), 0);
}

fn a_stated_budget_caps_what_the_load_takes() {
    let floor = safety_floor_bytes(CARD);
    let free = CARD - WEIGHTS;
    let budget = |bytes: u64| Allowance {
        budget: Some(bytes),
    };
    assert_eq!(
        budget_bytes(free, CARD, budget(8 << 30)),
        (8 << 30) - floor,
        "under the share, the budget is what the load takes"
    );
    assert_eq!(
        budget_bytes(free, CARD, budget(u64::MAX)),
        budget_bytes(free, CARD, Allowance::WHOLE),
        "past it, the share stands"
    );
}

fn the_card_does_not_hold_a_deployment_whose_weights_leave_no_context() {
    let floor = safety_floor_bytes(CARD);

    let roomy = Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, 2 << 30);
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

    // Weights that fit the share, leaving the pool 2 GiB under a 4 GiB
    // sequence.
    let share = (CARD as f64 * engine::load::DEVICE_SHARE) as u64;
    let tight = Accounting::of(CARD, Allowance::WHOLE, share - floor - (2 << 30), 4 << 30);
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
        refusal.contains("budget_device") && refusal.contains("quantized"),
        "and it names what changes the answer: {refusal}"
    );

    // Dense weights past the share: no pool would make room for them.
    let over = Accounting::of(CARD, Allowance::WHOLE, 42 << 30, 4 << 30);
    let refusal = over
        .admit()
        .expect_err("weights past the share are not a deployment")
        .to_string();
    assert!(
        refusal.contains(&(42u64 << 30).to_string()) && refusal.contains(&over.ceiling.to_string()),
        "the refusal names the weights' bytes and the share's: {refusal}"
    );
    assert!(
        refusal.contains("quantized") && refusal.contains("tensor parallelism"),
        "and what a model that big needs: {refusal}"
    );

    let demand = 32 << 30;
    Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, demand / 2)
        .admit()
        .expect("the card holds half of it");
    assert!(
        Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, demand)
            .admit()
            .is_err(),
        "the share of the same card does not hold it all"
    );
    assert!(
        Accounting::of(
            CARD,
            Allowance {
                budget: Some(WEIGHTS + (8 << 30))
            },
            WEIGHTS,
            demand / 2
        )
        .admit()
        .is_err(),
        "nor does a budget under it"
    );
}
