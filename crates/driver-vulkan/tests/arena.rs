//! Can a Vulkan driver bind the arena a real carve assigns?
//!
//! Every other test in this crate asks about a module or a dispatch. This one
//! asks about the OFFSETS, and it is the precondition the whole arena plan
//! rests on: something upstream places each activation at a byte offset it
//! chooses, `driver-metal` binds those offsets with `setBuffer:offset:` and
//! Apple silicon asks four bytes of alignment for it -- so nothing upstream has
//! ever had a reason to place them more carefully than that.
//!
//! Vulkan asks more. A storage descriptor's offset must be a multiple of
//! `minStorageBufferOffsetAlignment`, and an offset that is not simply cannot
//! be bound: there is no slow path and no fallback, the descriptor is invalid.
//! If the compiler placed activations at, say, sixteen-byte boundaries, this
//! backend could not use the arena on a device that asks for 256 and would need
//! a copy per operand -- which is a different driver, not a slower one.
//!
//! So it is worth knowing rather than assuming, and it is worth knowing
//! against the SPECIFICATION rather than against the card in this machine.
//!
//! # THE PRODUCER CHANGED AND SO DID THE ANSWER
//!
//! This file used to ask the question of `model_compiler::lower`, over the
//! plans `model::shared::llama_like` traced, and the answer was **exactly
//! enough with nothing to spare**: that allocator rounded every placement to
//! 256, which is also the largest `minStorageBufferOffsetAlignment` a
//! conformant Vulkan device may report, so every offset every text produced was
//! bindable on every device and the margin was ZERO.
//!
//! It read like a designed agreement and it was not one. That allocator's own
//! comment said why it picked the number: *"a decode body runs inside a
//! capture, so the same plan must land the same value at the same address on
//! every fire"* -- a Metal capture-replay requirement that happened to coincide
//! with a Vulkan descriptor limit, in a crate that had never heard of either.
//!
//! Which is the entire reason this file exists. **A coincidence nobody wrote
//! down is a coincidence somebody may reasonably undo, and it has been.**
//! `lower` is deleted; what carves an activation arena now is
//! `model_compiler::program::carve`, and its own comment states its number:
//! *"sizes are rounded to 16 so that a freed hole is 16-aligned too, which is
//! what keeps every offset aligned with no separate padding pass."*
//!
//! **IT WAS SIXTEEN AND THIS FILE SAID SO AS AN ASSERTION**, made over a
//! program the real `bound` built, precisely so that the day the carve
//! changed the file would say so rather than go quietly stale.
//!
//! **THAT DAY CAME.** `carve` rounds to 256 now — `model_compiler::program::
//! BIND_ALIGN` — and the guarantee `lower` used to give is back. It cost
//! almost nothing, which is why it was worth doing rather than documenting:
//! measured over every catalog row that compiles, the row pitch is IDENTICAL
//! at 16 and at 256 for every SKU but one, and gpt-oss pays 128 bytes on a
//! 407,936-byte row. A slot count in the hundreds and sizes already far above
//! 256 is why — the rounding almost never has anything to round.
//!
//! So the two tests below state the STRONGER claim now, which is the rewrite
//! their own failure messages asked for.
//!
//! # What that does and does not cost, today
//!
//! It costs nothing on any card whose `minStorageBufferOffsetAlignment` divides
//! 16, which is every card this repository has been run on -- and that number
//! is not assumed here either:
//! `the_facts_the_engine_is_given_are_the_ones_this_driver_keeps` in
//! `tests/device.rs` reads the LOCAL one off the device and holds it against
//! what a descriptor built at it can and cannot address, both ways.
//!
//! What it costs is the DEPLOYMENT GUARANTEE. "This driver binds any program on
//! any conformant device" was true and is not, and the failure mode on a strict
//! device is not slow: it is `vkUpdateDescriptorSets` refusing every arena
//! operand of every fire. The number to watch is the one this file asserts,
//! because it is the only place the two halves -- what a compiler chose and
//! what a descriptor needs -- are compared at all.
//!
//! The half that is still safe is the BANKS, and it is safe on purpose rather
//! than by coincidence: [`driver_vulkan::baker::BANK_ALIGN`] is 256 and its own
//! documentation gives the Vulkan reason. [`every_bank_offset_clears_the_strictest_alignment`]
//! is the check that the two halves of this driver's memory did not drift apart.
//!
//! GPU-free on purpose. The question is about numbers a compiler produced, and
//! a check that needed a device would not run in the builds that change them.

use std::collections::BTreeSet;

use driver_vulkan::baker::{BANK_ALIGN, arena_of};
use model_compiler::program::{Dt, Program, Slot};
use model_ir::plan::{Cond, Op, Param, Plan, Seam, Shard, ValueDef};

/// The strictest `minStorageBufferOffsetAlignment` a conformant Vulkan device
/// may report.
///
/// The specification's required-limits table caps it here, so an offset that
/// is a multiple of 256 is bindable on EVERY Vulkan device, and one that is
/// not is bindable on some and not others. Checking against the local card's
/// 16 would pass a plan that fails on hardware nobody in this repository owns,
/// which is the failure this constant exists to make impossible.
const STRICTEST_ALIGNMENT: u64 = 256;

/// What `model_compiler::program::carve` rounds a reservation to.
///
/// Transcribed from `carve`'s own `BIND_ALIGN` rather than imported, because
/// that constant is private. Transcribing it is the point: if the two ever
/// disagree, the tests below are what notice.
const CARVE_ALIGNMENT: u64 = 256;

/// The width that made the original measurement honest.
///
/// The first version of this file, taken over one text, reported a comfortable
/// 2048 and would have let a change to 128 look harmless. It was `gpt_oss_20b`
/// that gave the real answer: **a 2880-wide row of 2 bytes is 5760, which is
/// not a multiple of 256**, so the next operand lands wherever the allocator
/// insists on and nothing more. The number is kept here for exactly that
/// reason -- a fixture whose rectangles all happen to be 256-aligned measures
/// the fixture and not the allocator.
const HIDDEN: u64 = 2880;

/// A vocabulary, for the embedding table the tower is seeded on.
const VOCAB: u64 = 1024;

/// The head width the norms state. Divides [`HIDDEN`] -- 2880 / 64 = 45 -- so
/// the per-head norm is a statement the plan can make.
const HEAD_DIM: u64 = 64;

/// One statement, in the columns a `Plan` states them.
fn stmt(
    kernel: &str,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    weights: Vec<&str>,
    params: Vec<u64>,
) -> Op {
    Op {
        kernel: kernel.to_string(),
        inputs,
        outputs,
        weights: weights.into_iter().map(str::to_string).collect(),
        params,
        cache: None,
        layer: Some(0),
        cond: Cond::Always,
    }
}

/// A tower the real compiler can SIZE on this plane.
///
/// A result is sizable only if its width rule does not read an operand's
/// rectangle, which across the whole floor is true of exactly four points --
/// `layout.embed` (an embedding table's axis) and the three `gemm.*` (a
/// weight's). `kernels-vulkan` claims `layout.embed`, so `bound` runs the real
/// width walk and the real carve over these five statements. `driver-wgpu`
/// claims none of the four and its twin of this file cannot exist.
///
/// THE DAY THIS STOPS BINDING, the claim went, and every assertion below is
/// measuring a fixture instead of a compiler.
fn seeded() -> Plan {
    Plan {
        name: "an-arena-fixture".into(),
        plane: model_ir::kernels::Backend::Vulkan,
        facts: vec!["qo_one".into()],
        params: vec![
            Param {
                name: "norm.weight".into(),
                shape: vec![HIDDEN],
                shard: Shard::Replicated,
                repr: "dense".into(),
            },
            Param {
                name: "embed.table".into(),
                shape: vec![VOCAB, HIDDEN],
                shard: Shard::Replicated,
                repr: "dense".into(),
            },
        ],
        caches: Vec::new(),
        values: vec![
            ValueDef::Runtime("token_ids".into()),
            ValueDef::Stmt(0),
            ValueDef::Stmt(1),
            ValueDef::Stmt(2),
            ValueDef::Stmt(3),
            ValueDef::Stmt(4),
        ],
        ops: vec![
            stmt(
                "layout.embed",
                vec![0],
                vec![1],
                vec!["embed.table"],
                vec![VOCAB],
            ),
            stmt(
                "norm.rmsnorm_no_scale",
                vec![1],
                vec![2],
                vec![],
                vec![HEAD_DIM, f32::to_bits(1e-6).into()],
            ),
            stmt(
                "norm.rmsnorm_per_head",
                vec![2],
                vec![3],
                vec!["norm.weight"],
                vec![HEAD_DIM, f32::to_bits(1e-6).into()],
            ),
            stmt("norm.residual_add", vec![3, 2], vec![4], vec![], vec![]),
            stmt(
                "attention.logit_softcap",
                vec![4],
                vec![5],
                vec![],
                vec![f32::to_bits(30.0).into()],
            ),
        ],
        seams: vec![Seam {
            seam: model_ir::seam::OUT.name.to_string(),
            values: vec![5],
            layer: None,
        }],
    }
}

/// The one lane [`seeded`] binds, or the reason it did not.
fn carved() -> Program {
    let plan = seeded();
    let lanes = model_compiler::program::bound(&plan);
    assert_eq!(lanes.len(), 1, "nothing here is conditional, so one lane");
    match lanes.into_iter().next().expect("one lane") {
        Ok(program) => program,
        Err(why) => panic!(
            "`layout.embed` should seed this tower and the compiler refused: {why}. \
             Every assertion in this file measures the carve through this program, so \
             a refusal here is not a failure of the arena -- it is the claim going."
        ),
    }
}

/// Every arena offset a carved program assigns, and the rectangle at it.
fn offsets(program: &Program) -> Vec<(u64, u64)> {
    program
        .slots
        .iter()
        .filter_map(|slot| match slot {
            Slot::Arena { offset, .. } => Some((*offset, slot.bytes())),
            Slot::Runtime(_) | Slot::Alias(_) | Slot::Absent => None,
        })
        .collect()
}

/// THE MEASUREMENT THIS FILE EXISTS FOR: the carve rounds to sixteen, and a
/// conformant Vulkan device may ask for two hundred and fifty-six.
///
/// Both halves are asserted, and the second is asserted as a NEGATIVE -- there
/// is at least one offset a strict device could not bind. A test that only
/// checked "every offset is 16-aligned" would pass just as happily on a carve
/// that had gone back to 256, and would then be silent about the thing this
/// file was written to notice.
///
/// The witness is [`HIDDEN`]: a 2880-wide bf16 row is 5760 bytes, `5760 % 256`
/// is 128, so the second rectangle in the arena already sits somewhere a device
/// reporting 256 cannot address.
#[test]
fn every_offset_the_carve_chose_is_bindable_on_any_conformant_device() {
    let program = carved();
    let placed = offsets(&program);
    assert!(
        placed.len() >= 2,
        "a tower of five statements should carve at least two rectangles: {placed:?}",
    );

    for (at, bytes) in &placed {
        assert_eq!(
            at % CARVE_ALIGNMENT,
            0,
            "the carve places a {bytes}-byte rectangle at {at}, which is not a \
             multiple of the {CARVE_ALIGNMENT} it documents",
        );
    }

    let unbindable: Vec<u64> = placed
        .iter()
        .map(|(at, _)| *at)
        .filter(|at| at % STRICTEST_ALIGNMENT != 0)
        .collect();
    assert!(
        unbindable.is_empty(),
        "these offsets are not multiples of {STRICTEST_ALIGNMENT} and so bind on \
         some conformant Vulkan devices and not others: {unbindable:?} out of \
         {placed:?}. This is the guarantee `model_compiler::lower` gave, that \
         `carve` dropped to sixteen, and that `program::BIND_ALIGN` restored -- \
         a failure here means it was dropped again.",
    );
}

/// The bytes a descriptor is actually built at are `offset * rows`, and for a
/// decode that is the offset itself.
///
/// The multiplication is the one thing that could rescue the alignment by
/// accident: a slot's byte base is its offset times the fire's row count, so a
/// prefill of sixteen rows lands every 16-aligned offset on 256. A DECODE IS
/// ONE ROW. So the worst case is not a corner of the arena, it is the ordinary
/// token, which is the reason this cannot be waved off as theoretical.
#[test]
fn a_decode_binds_the_offset_the_carve_chose_with_nothing_multiplied_in() {
    let program = carved();
    let rows = 1u64;
    let bad = offsets(&program)
        .into_iter()
        .filter(|(at, _)| !(at * rows).is_multiple_of(STRICTEST_ALIGNMENT))
        .count();
    assert_eq!(
        bad, 0,
        "a decode binds `offset * 1`, so these offsets reach the descriptor \
         exactly as the carve chose them and {bad} of them do not clear \
         {STRICTEST_ALIGNMENT}",
    );

    // THE ROW COUNT USED TO MATTER AND NO LONGER DOES, which is the whole
    // point of raising the carve. `offset * rows` is a scaling, so while the
    // carve rounded to 16 a prefill at 16 rows bought the difference back by
    // arithmetic and a decode at one row did not -- an alignment guarantee
    // that held for the wide fire and failed for the narrow one, which is the
    // worse half to lose. Asserted across the row counts that used to
    // separate, so that a carve dropped back to 16 fails HERE too and not
    // only in the test above.
    for rows in [1u64, 2, 3, 16] {
        let bad = offsets(&program)
            .into_iter()
            .filter(|(at, _)| !(at * rows).is_multiple_of(STRICTEST_ALIGNMENT))
            .count();
        assert_eq!(
            bad, 0,
            "at {rows} row(s), {bad} offset(s) do not clear {STRICTEST_ALIGNMENT}",
        );
    }
}

/// The pitch sits ON the floor no layout can beat, and no two live values
/// share a byte.
///
/// `carve` reuses bytes between values whose lives do not overlap, which is
/// what makes the arena smaller than the sum of its rectangles -- and it is
/// also the one thing that can go wrong SILENTLY. A reused slab does not fault
/// when it is wrong: the addresses stay inside the block, every launch
/// succeeds, and the only thing that catches it is arithmetic. So both halves
/// are asserted through `model_compiler`'s own two answers rather than
/// re-derived here, because a second derivation is a second chance to agree
/// with itself and disagree with the carve.
///
/// It belongs in THIS file and not only in the compiler's, because an offset
/// that is right and a descriptor that can be built at it are two claims and
/// this crate owes the second one over the first.
#[test]
fn the_pitch_is_the_busiest_instant_and_nothing_live_shares_a_byte() {
    let plan = seeded();
    let program = carved();
    assert_eq!(
        program.row_pitch,
        model_compiler::program::live_bound(&plan, &program),
        "the pitch should sit on the floor `live_bound` computes",
    );
    assert!(
        model_compiler::program::clashes(&plan, &program).is_empty(),
        "two values live at one step share a byte, which is the arena's whole \
         invariant and the one mistake nothing else catches",
    );

    // And the reuse is REAL rather than incidental: the pitch is strictly less
    // than the sum, which is what says two of these five rectangles are sharing.
    let placed = offsets(&program);
    let sum: u64 = placed.iter().map(|(_, b)| b).sum();
    assert!(
        program.row_pitch < sum,
        "the pitch ({}) is the sum of every rectangle ({sum}), so this program \
         reuses nothing and the fixture is no longer measuring the carve",
        program.row_pitch,
    );
    let distinct: BTreeSet<u64> = placed.iter().map(|(at, _)| *at).collect();
    assert!(
        distinct.len() < placed.len(),
        "five rectangles at five distinct offsets: {placed:?}",
    );
}

/// A rectangle's element width is not four, and this crate must not assume it.
///
/// The finding is `driver-metal`'s and it is transcribed rather than inherited:
/// the read-out is bf16 because `affine_qmv_fast` writes bf16, and a text's
/// declared dtype does not change what a kernel does -- a reader that assumed
/// f32 got a vocabulary exactly half zeros, which looks like a dead half of a
/// tensor and is really two elements read as one.
///
/// What is pinned here is the SEED's rectangle, because that is the one the
/// compiler derived rather than propagated: `layout.embed`'s width rule is
/// `[fire, table.axis(1)]`, so the result's width comes off the PARAMETER
/// TABLE and its element width off the plan's `repr` column. Every rectangle
/// downstream of it inherits both.
#[test]
fn the_seed_takes_its_rectangle_from_the_table_and_its_width_from_the_dtype() {
    let program = carved();
    let seed = program
        .slots
        .get(1)
        .expect("value 1 is `layout.embed`'s result");
    assert!(
        matches!(
            seed,
            Slot::Arena { width, dtype: Dt::Bf16, .. } if *width == HIDDEN
        ),
        "the seed's rectangle should be the embedding table's row of {HIDDEN} \
         bf16 elements: {seed:?}",
    );
    assert_eq!(
        seed.bytes(),
        HIDDEN * 2,
        "and its bytes should follow the dtype rather than a four this crate assumed",
    );
    assert_eq!(
        seed.bytes() % STRICTEST_ALIGNMENT,
        128,
        "5760 bytes is the witness this file is built on; if this number moved, \
         `HIDDEN` moved with it and the alignment claim above is measuring \
         something else",
    );
}

/// The other half of this driver's memory is 256-aligned, and on purpose.
///
/// `arena_of` packs the weight banks, and [`BANK_ALIGN`] is 256 with the Vulkan
/// reason written beside it. This is the check that the two halves did not
/// drift apart: activations at 16 and banks at 256 is a deliberate asymmetry
/// only for as long as somebody can point at both numbers.
#[test]
fn every_bank_offset_clears_the_strictest_alignment() {
    assert_eq!(
        BANK_ALIGN, STRICTEST_ALIGNMENT,
        "the bank alignment is what makes a `Const` operand bindable on every \
         conformant device; if it is not 256 that guarantee is gone too",
    );
    // Sizes chosen to be awkward on purpose: 1 byte, the 5760-byte row that
    // broke the original measurement, and one byte past a 256 boundary.
    let produced: Vec<(String, model::produce::HostTensor)> = [1usize, 5760, 257]
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            (
                format!("bank.{i}"),
                model::produce::HostTensor::new(
                    [n as u64],
                    model::produce::Dtype::U8,
                    vec![0u8; n],
                ),
            )
        })
        .collect();
    let (offsets, total) = arena_of(&produced);
    for (at, (name, _)) in offsets.iter().zip(&produced) {
        assert_eq!(
            at % STRICTEST_ALIGNMENT,
            0,
            "`{name}` is placed at {at}, which no strict device could bind",
        );
    }
    assert_eq!(
        total % STRICTEST_ALIGNMENT,
        0,
        "the packed total should end on a boundary too, so the next arena starts on one",
    );
}

// ── WHAT STOOD HERE ────────────────────────────────────────────────────
//
// Eight tests, all of them `lower(&model::shared::llama_like::..., rows)` over
// a real text, and all eight are deleted with the two crates they read. Named
// because each made a claim this driver still owes, and the list is what the
// next person should port rather than reinvent:
//
// * `every_arena_offset_a_real_lowering_assigns_is_bindable` -- the file's
//   subject, over six real texts in both fire classes.
//   `the_carve_aligns_to_sixteen_and_the_specification_asks_256` is its
//   successor and it reports the OPPOSITE answer.
// * `every_symbol_a_real_text_launches_has_a_module` -- every symbol a plan
//   named resolved in `kernels_vulkan::MODULES`. `tests/rules.rs` asks the
//   same question of the module table directly, and
//   `tests/the_walk_is_the_program.rs`'s
//   `every_artifact_the_walk_named_is_one_the_spirv_tree_stamps` asks it of a
//   walk.
// * `what_the_plan_states_and_what_the_module_binds_account_for_each_other` --
//   a launch's operand count against the module's declared bindings, hole by
//   hole. `spirv::Declared::bindings` and `device::Pipelines::get`'s maximum
//   still carry the rule; nothing checks it over a whole program.
// * `the_binder_this_crate_ships_resolves_every_operand_of_every_real_launch`
//   -- `binding::bind` over 14,324 operands, which is where the "zero bytes to
//   spare" figure in `binding.rs`'s header came from.
// * `an_arena_one_byte_short_of_what_the_plan_placed_refuses_what_runs_off_it`
//   -- the arena bound was checked before `Bound::within`, so the refusal named
//   the ARENA rather than the buffer. `Arena::bytes` still exists and still
//   means that; nothing raises the refusal.
// * `every_launchs_scalars_land_where_its_module_reads_them` -- `params` against
//   `spirv::Declared::push_offsets`, per launch. `binding::params_from` is the
//   surviving half and `tests/device.rs` holds it against a real module.
// * `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
//   -- the whole plan through `hold`, counting refusals by name. The walk's
//   equivalent is `baker::resolve::check`, which
//   `tests/the_walk_is_the_program.rs` runs.
// * `every_pool_number_reaches_the_shader_through_the_arm_that_names_it` -- a
//   `FireNumber` traced from `Resolve` to the packed word. `FireNumber` and
//   `Resolve` both survive; the arm that carried it does not.
