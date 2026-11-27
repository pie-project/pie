use engine::Allowance;
use engine_cuda::device::elastic::map_unit_for;
use engine_cuda::device::elastic::{
    budget_bytes, ceiling_of, mapped_on_demand, room_under, safety_floor_bytes,
};
use engine_cuda::store::kv::Paging;
use engine_cuda::store::{
    Accounting, MEASURING, decoded_weight_reserve, holds, one_slot_bytes, tokens_within,
    window_fire_bytes, window_of,
};

const CARD: u64 = 48_305_799_168;

const WEIGHTS: u64 = 13_761_281_792;

/// The nine tenths of a card of `card` bytes, `before` of them held, that
/// the boots the issues below report took of it (`device_memory_share
/// 0.9`, when a load set a share of the card aside): as the budget that
/// states it now.
fn nine_tenths(card: u64, before: u64) -> Allowance {
    Allowance {
        budget: Some((card as f64 * 0.90) as u64 - before),
    }
}

#[test]
fn the_operators_fraction_sizes_the_pool_every_case() {
    // The floor is the mapped pool's: the cards these cases record served
    // on it. A flat range (ROCm) takes what the card has left.
    if mapped_on_demand() {
        the_load_takes_the_card_and_charges_what_is_already_on_it();
        a_stated_budget_caps_what_the_load_takes();
        a_stated_budget_counts_the_weights_the_load_took();
        the_card_does_not_hold_a_deployment_whose_weights_leave_no_context();
        the_reservations_scale_to_a_twenty_four_gigabyte_card();
        a_twenty_seven_b_serves_on_a_twenty_four_gigabyte_card_at_the_asked_lanes();
    } else {
        a_flat_range_takes_what_the_card_has_left();
    }
    the_decoded_weight_tiles_are_held_out_of_the_pool();
    the_bodies_are_held_ahead_of_the_rows_and_the_tile_before_any_capture();
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
    let issue_reserve = |lanes: u32| {
        (u64::from(lanes.next_power_of_two()) * (VOCAB * 4).next_multiple_of(256))
            .min(eta_exec::SCRATCH_MAX_BYTES)
            * 6
    };
    let live = budget_bytes(CARD - BEFORE, CARD, nine_tenths(CARD, BEFORE)) - WEIGHTS;
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

    let room = live - SEQUENCE;
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
    // Held between fires, as the staging ring and the declared-context fit
    // size it: the window's 65 pages and no fire's past them.
    let held = |context: u32| {
        let paging = Paging::of(16, context, 1, 1)
            .expect("a page size of 16 is a paging")
            .held(window);
        one_slot_bytes(&trace, paging).expect("the rows size")
    };
    assert_eq!(held(34_816), GLOBAL * 34_816 + SLIDING * 16 * 65);
    assert_eq!(one(34_816) - held(34_816), SLIDING * 16 * 129);
    assert_eq!(
        held(1024),
        one(1024),
        "a window that spans the context is no window"
    );
    assert_eq!(
        at(34_816).window_pages_at(8 * 2176),
        1 + 65 + 129 + 7 * 2176 / 2,
        "past one sequence, a windowed page for every two kv pages"
    );
}

fn the_bodies_are_held_ahead_of_the_rows_and_the_tile_before_any_capture() {
    // pie-evals nightly 36021487786 on an L40S: gemma-4-31b 4-bit at a 6144
    // envelope. Past one 5280 MiB sequence the card had 6581 MiB; holding
    // the programs (1024 MiB at 256 lanes) and a reserve for the decoded
    // tiles ate it all, the bodies were held 0, nothing armed, and the
    // first fire — a capture — found the `decoded_weight` scratch at 0 bytes
    // and needed 176160768.
    const SEQUENCE: u64 = 5280 << 20;
    const SPARE: u64 = 6581 << 20;
    const PROGRAMS: u64 = 1024 << 20;
    const MLP_PLANE: u64 = 21_504 * 5376 * 2;
    const FIRST_CAPTURE: u64 = 176_160_768;
    let live = SEQUENCE + SPARE;

    assert_eq!(
        holds(SEQUENCE, live, MEASURING, PROGRAMS),
        Ok((SPARE - PROGRAMS, PROGRAMS)),
        "bodies still to arm are held everything past the sequence and the programs"
    );
    assert_eq!(
        holds(SEQUENCE, live, MEASURING, SPARE + 1),
        Ok((0, SPARE)),
        "and programs past the card leave them nothing: arming refuses the load by name"
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

    let floor = safety_floor_bytes();
    let after_weights = budget_bytes(CARD, CARD, nine_tenths(CARD, 0)) - WEIGHTS;
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

    // The bodies are held ahead of the cache rows: still to arm, all one
    // sequence leaves; measured, what they took, and the rows fill the rest.
    let claim = fine(8192);
    let (bodies, _) = holds(claim, room, MEASURING, 0).expect("one sequence fits");
    assert_eq!(
        bodies,
        room - claim,
        "the claim before arming is that one sequence"
    );
    assert_eq!(
        holds(claim, room, 1 << 30, 0),
        Ok((1 << 30, 0)),
        "a measured figure is held as it is"
    );
    assert_eq!(
        holds(claim, room, room, 0),
        Err(room - claim),
        "and one past what the sequence leaves is refused"
    );
}

fn a_flat_range_takes_what_the_card_has_left() {
    assert_eq!(safety_floor_bytes(), 0, "a flat range holds nothing back");
    let free = CARD - WEIGHTS;
    assert_eq!(
        budget_bytes(free, CARD, Allowance::WHOLE),
        free,
        "all the card has left, no share of it"
    );
    assert_eq!(
        ceiling_of(CARD, WEIGHTS, Allowance::WHOLE),
        CARD,
        "the card itself"
    );
    let budget = Allowance {
        budget: Some(8 << 30),
    };
    assert_eq!(
        budget_bytes(free, CARD, budget),
        8 << 30,
        "a stated budget still caps it"
    );
    let accounting = Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, 2 << 30);
    assert_eq!(accounting.pool, CARD - WEIGHTS);
    assert_eq!(accounting.floor, 0);
}

fn the_load_takes_the_card_and_charges_what_is_already_on_it() {
    let floor = safety_floor_bytes();
    assert_eq!(floor, 64 << 20, "the driver's 64 MiB, on every card");

    let free = CARD - WEIGHTS;
    let asked = budget_bytes(free, CARD, Allowance::WHOLE);
    assert_eq!(asked, 34_477_408_512);
    assert_eq!(
        asked + WEIGHTS + floor,
        CARD,
        "no share of the card is set aside: what is on it, the floor and the load's room are all \
         of it"
    );
    assert_eq!(
        budget_bytes(1 << 30, CARD, Allowance::WHOLE),
        (1 << 30) - floor
    );
    assert_eq!(budget_bytes(floor, CARD, Allowance::WHOLE), 0);
}

fn a_stated_budget_caps_what_the_load_takes() {
    let floor = safety_floor_bytes();
    let free = CARD - WEIGHTS;
    let budget = |bytes: u64| Allowance {
        budget: Some(bytes),
    };
    assert_eq!(
        budget_bytes(free, CARD, budget(8 << 30)),
        (8 << 30) - floor,
        "under the card, the budget is what the load takes"
    );
    assert_eq!(
        budget_bytes(free, CARD, budget(u64::MAX)),
        budget_bytes(free, CARD, Allowance::WHOLE),
        "past it, the card stands"
    );
}

fn a_stated_budget_counts_the_weights_the_load_took() {
    const BEFORE: u64 = 3 << 30;
    const BUDGET: u64 = 20 << 30;
    let floor = safety_floor_bytes();
    let stated = Allowance {
        budget: Some(BUDGET),
    };
    // The ceiling is resolved once, on the card as the load found it.
    let ceiling = ceiling_of(CARD, BEFORE, stated);
    assert_eq!(ceiling, BEFORE + BUDGET);
    assert_eq!(
        room_under(ceiling, CARD - BEFORE, CARD),
        budget_bytes(CARD - BEFORE, CARD, stated),
        "before the load takes anything, the room is the budget's"
    );
    // The pool opens with the weights resident: they are inside the budget.
    let free = CARD - BEFORE - WEIGHTS;
    assert_eq!(
        room_under(ceiling, free, CARD),
        BUDGET - WEIGHTS - floor,
        "the pool has what the budget leaves past the weights"
    );
    assert_eq!(
        budget_bytes(free, CARD, stated),
        BUDGET - floor,
        "resolved with the weights resident, the budget would be counted past them"
    );
    assert_eq!(
        room_under(ceiling, CARD - BEFORE - BUDGET, CARD),
        0,
        "a load holding its budget has no room"
    );
    // What the process took before its loads (its context) is on the card
    // when the ceiling is resolved and is not held before: the budget
    // counts it.
    const CONTEXT: u64 = 800 << 20;
    let held = BEFORE + CONTEXT;
    let ceiling = ceiling_of(CARD, held - CONTEXT, stated);
    assert_eq!(
        room_under(ceiling, CARD - held - WEIGHTS, CARD),
        BUDGET - CONTEXT - WEIGHTS - floor,
        "the pool has what the budget leaves past the context and the weights"
    );
    assert_eq!(
        ceiling_of(
            CARD,
            BEFORE,
            Allowance {
                budget: Some(u64::MAX)
            }
        ),
        CARD,
        "a budget past the card leaves the card"
    );

    // With no budget the ceiling is the card whenever it is resolved, and
    // every reading under it is the room the allowance states then.
    for held in [0, BEFORE, BEFORE + WEIGHTS, CARD] {
        assert_eq!(ceiling_of(CARD, held, Allowance::WHOLE), CARD);
    }
    let whole = ceiling_of(CARD, BEFORE, Allowance::WHOLE);
    for free in [CARD, CARD - BEFORE, free, 1 << 30, 0] {
        assert_eq!(
            room_under(whole, free, CARD),
            budget_bytes(free, CARD, Allowance::WHOLE)
        );
    }
}

fn the_card_does_not_hold_a_deployment_whose_weights_leave_no_context() {
    let floor = safety_floor_bytes();

    let roomy = Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, 2 << 30);
    assert_eq!(roomy.card, CARD);
    assert_eq!(roomy.weights, WEIGHTS);
    assert_eq!(roomy.floor, floor);
    assert_eq!(roomy.pool, 34_477_408_512);
    assert_eq!(roomy.ceiling, CARD, "no budget stated: the card");
    assert_eq!(
        roomy.pool + roomy.weights + roomy.floor,
        roomy.ceiling,
        "weight tier + elastic pool + safety floor = the card"
    );
    roomy.admit().expect("34.4 GB holds a 2 GiB sequence");

    // Weights that fit the card, leaving the pool 2 GiB under a 4 GiB
    // sequence.
    let tight = Accounting::of(CARD, Allowance::WHOLE, CARD - floor - (2 << 30), 4 << 30);
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

    // Dense weights past the card: no pool would make room for them.
    let over = Accounting::of(CARD, Allowance::WHOLE, 46 << 30, 4 << 30);
    let refusal = over
        .admit()
        .expect_err("weights past the card are not a deployment")
        .to_string();
    assert!(
        refusal.contains(&(46u64 << 30).to_string()) && refusal.contains(&over.ceiling.to_string()),
        "the refusal names the weights' bytes and the card's: {refusal}"
    );
    assert!(
        refusal.contains("quantized") && refusal.contains("tensor parallelism"),
        "and what a model that big needs: {refusal}"
    );

    let demand = 36 << 30;
    Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, demand / 2)
        .admit()
        .expect("the card holds half of it");
    assert!(
        Accounting::of(CARD, Allowance::WHOLE, WEIGHTS, demand)
            .admit()
            .is_err(),
        "the same card does not hold it all"
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
