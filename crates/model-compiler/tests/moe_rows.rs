//! The routed leg's rows algebra, pinned against qwen35-a3b's real text.
//!
//! `gemma_widths.rs` measures the half of a rectangle the walk always
//! answered: how wide a row is. This measures the half W6 added -- how MANY
//! rows a value has over the fire's own count -- because that half is where a
//! mixture of experts stops being a matmul with a different bank.
//!
//! The claim under test is one sentence: a router picks `top_k` experts per
//! token, `moe.matmul_select` runs one matmul per PICK, and every value
//! between that fan-out and `moe.weighted_sum`'s fold rides `fire_rows *
//! top_k` rows. Get the factor wrong and nothing type-errors -- the arena is
//! `u64` bytes and the row count is a `u32` -- so a program eight times too
//! small binds silently and faults on a device no CI machine has.
//!
//! a3b's `a3()` numbers, and the assertions are stated in them: hidden 2048,
//! 256 experts at top_k 8, `moe_intermediate` 512 (so a `[E, 2I, H]` gate/up
//! bank is 1024 wide) and a `[E, H, I]` down bank 2048 wide.

use model_compiler::program::{Dt, Program, Rows, Slot};
use model_dsl::Plane;
use model_ir::plan::{Op, Plan, ValueId};

const SKU: &str = "qwen35-a3b-bf16-kv-bf16";

const HIDDEN: u64 = 2048;
const TOP_K: u32 = 8;
/// `moe_intermediate_size`, and the gate/up bank's out axis is twice it.
const INTER: u64 = 512;

fn traced() -> Plan {
    let row = model::trace_of(SKU).unwrap_or_else(|| panic!("`{SKU}` is not a catalog row"));
    row(Plane::Cuda)
}

/// The rectangle a lane gives `value`, chasing a merge to the arm that
/// survived on it.
fn rect(program: &Program, value: ValueId) -> Option<(Rows, u64, Dt)> {
    match &program.slots[value as usize] {
        Slot::Arena {
            rows, width, dtype, ..
        } => Some((*rows, *width, *dtype)),
        Slot::Alias(through) => rect(program, *through),
        Slot::Runtime(_) | Slot::Absent => None,
    }
}

/// Every statement of `point` this lane runs.
fn stating<'p>(plan: &'p Plan, program: &Program, point: &str) -> Vec<&'p Op> {
    program
        .steps
        .iter()
        .map(|step| &plan.ops[step.op as usize])
        .filter(|op| op.kernel == point)
        .collect()
}

/// THE GATE W6 CLOSED. a3b refused on `moe.topk_softmax -> UNSIZED` before
/// the rows algebra landed, and the refusal was the walk's own backlog rather
/// than any plane's: every point of its routed leg resolves on cuda.
#[test]
fn every_lane_of_a3b_binds() {
    let plan = traced();
    match model_compiler::program::programs(&plan) {
        Ok(lanes) => {
            assert_eq!(lanes.len(), 2, "a3b states one fact and two behaviors");
            for lane in &lanes {
                assert_eq!(lane.row_pitch % 16, 0, "an unaligned row pitch");
                assert_eq!(
                    lane.slots.len(),
                    plan.values.len(),
                    "a slot column that is not the value column"
                );
            }
        }
        Err(refusals) => {
            let told: Vec<String> = refusals.iter().map(ToString::to_string).collect();
            panic!("a3b refused: {}", told.join(" | "));
        }
    }
}

/// A router's two results are the only place `top_k` is written down in a
/// bound program, and everything routed reads the factor off `routes`.
#[test]
fn a_router_states_two_top_k_wide_columns() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("a3b binds");
    for lane in &lanes {
        let routers = stating(&plan, lane, "moe.topk_softmax");
        assert!(!routers.is_empty(), "a3b routes on every moe layer");
        for op in routers {
            assert_eq!(
                rect(lane, op.outputs[0]),
                Some((Rows::Fire, u64::from(TOP_K), Dt::I32)),
                "`routes` names the chosen experts, one i32 per pick"
            );
            assert_eq!(
                rect(lane, op.outputs[1]),
                Some((Rows::Fire, u64::from(TOP_K), Dt::F32)),
                "`weights` says how much each pick counts, on f32"
            );
        }
    }
}

/// THE FAN-OUT, AND THAT IT HAPPENS ONCE. a3b states `matmul_select` twice
/// per routed layer -- the gate/up leg reads a per-TOKEN row and the down leg
/// reads an already-ROUTED one -- and both results are one row per route. A
/// rule that multiplied its operand's factor would have said `top_k * top_k`
/// on the second, which is the mistake this test exists for.
#[test]
fn the_fan_out_is_one_row_per_route_and_happens_once() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("a3b binds");
    let mut widths = Vec::new();

    for lane in &lanes {
        let selects = stating(&plan, lane, "moe.matmul_select");
        assert!(!selects.is_empty(), "a3b states two selects per moe layer");
        for op in selects {
            let (rows, width, dtype) = rect(lane, op.outputs[0]).expect("a bound routed row");
            assert_eq!(
                rows,
                Rows::FireTimes(TOP_K),
                "a select's result is one row per ROUTE, however its operand rides"
            );
            assert_eq!(dtype, Dt::Bf16, "an expert bank dequantizes into the row");
            widths.push(width);

            // The factor is READ off `routes`, never restated: this
            // statement carries no `top_k` param at all.
            assert!(op.params.is_empty(), "`moe.matmul_select` states no scalar");
            assert_eq!(
                rect(lane, op.inputs[1]).map(|(_, w, d)| (w, d)),
                Some((u64::from(TOP_K), Dt::I32)),
                "`routes` is operand 1 and its width IS top_k"
            );
        }
    }

    widths.sort_unstable();
    widths.dedup();
    assert_eq!(
        widths,
        vec![2 * INTER, HIDDEN],
        "the two legs are the two banks' `[E, N, K]` out axes and no others"
    );
}

/// The activation between the two legs keeps the fan-out and takes the ONE
/// stated intermediate for its width. `mlp.swiglu` has no MoE special case;
/// it rides its operand's rows, which is what makes the compact rectangle the
/// honest carrier of the routed leg.
#[test]
fn the_routed_activation_rides_its_operand_and_the_stated_intermediate() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("a3b binds");
    let mut routed = 0;

    for lane in &lanes {
        for op in stating(&plan, lane, "mlp.swiglu") {
            let over = rect(lane, op.inputs[0]).expect("a bound packed row");
            let out = rect(lane, op.outputs[0]).expect("a bound activation");
            assert_eq!(out.0, over.0, "swiglu never changes its operand's rows");
            if over.0 == Rows::FireTimes(TOP_K) {
                assert_eq!(over.1, 2 * INTER, "the routed `[gate | up]` row");
                assert_eq!(out.1, INTER, "and the ONE stated intermediate out of it");
                routed += 1;
            }
        }
    }
    assert!(routed > 0, "a3b activates inside its routed leg");
}

/// The closing bracket: `weighted_sum` is the only point in the table that
/// NARROWS the row factor, and the shared expert that lands beside it is per
/// token throughout.
#[test]
fn weighted_sum_folds_the_fan_out_back_to_one_row_per_token() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("a3b binds");
    let mut folded = 0;

    for lane in &lanes {
        for op in stating(&plan, lane, "moe.weighted_sum") {
            let routed = rect(lane, op.inputs[0]).expect("a bound routed rectangle");
            assert_eq!(routed.0, Rows::FireTimes(TOP_K), "it folds a fan-out");
            assert_eq!(
                rect(lane, op.outputs[0]),
                Some((Rows::Fire, HIDDEN, Dt::Bf16)),
                "and hands back one hidden row per token"
            );
            folded += 1;
        }
        for op in stating(&plan, lane, "moe.sigmoid_gate_add") {
            assert_eq!(
                rect(lane, op.outputs[0]),
                Some((Rows::Fire, HIDDEN, Dt::Bf16)),
                "the shared expert joins a row that is already folded"
            );
        }
    }
    assert!(folded > 0, "a3b folds on every routed layer");
}

/// THE ARENA PAYS FOR THE FAN-OUT ONCE. `row_pitch` is bytes per FIRE row, so
/// a routed slot's `top_k` sub-rows sit contiguous inside its own column --
/// which is the legacy staging's `[tokens, top_k, width]` rectangle to the
/// byte. Measured rather than asserted about: the pitch must cover every
/// routed slot's WHOLE footprint, fan-out included.
///
/// WHAT THIS NO LONGER ASSERTS is that two slots never overlap. They do, on
/// purpose: values whose lives do not touch share bytes, which is what took
/// a3b's pitch from 7.67 MB to 489 KB. The disjointness that still holds is
/// the liveness-aware one, and `arena_liveness.rs` is where it is checked --
/// here the question is only whether the routed factor was paid for.
#[test]
fn the_row_pitch_carries_every_routed_row() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("a3b binds");

    for lane in &lanes {
        let routed: Vec<(u64, u64)> = lane
            .slots
            .iter()
            .filter_map(|s| match s {
                Slot::Arena {
                    offset,
                    rows: rows @ Rows::FireTimes(_),
                    width,
                    dtype,
                } => Some((*offset, rows.factor() * width * dtype.size())),
                _ => None,
            })
            .collect();
        assert!(!routed.is_empty(), "a3b mints routed slots");
        for (at, bytes) in &routed {
            assert!(
                at + bytes <= lane.row_pitch,
                "the row pitch does not cover a routed slot's fan-out"
            );
        }
        // And the fan-out is really paid for: the widest routed column is
        // `top_k` sub-rows of its own width and not one.
        let widest = routed
            .iter()
            .map(|(_, bytes)| *bytes)
            .max()
            .expect("a routed slot");
        assert!(
            widest >= u64::from(TOP_K) * INTER * 2,
            "a routed column narrower than one route's row times `top_k`"
        );
    }
}
