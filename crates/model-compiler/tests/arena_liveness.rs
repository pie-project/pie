//! The arena's reuse invariant, replayed over every shipping row.
//!
//! A REUSED SLAB DOES NOT FAULT WHEN IT IS WRONG. Two values that share bytes
//! they should not have shared still address inside the block, every launch
//! still succeeds, and the only thing that changes is a number a checkpoint's
//! logits would have to catch on a device no CI machine has. So the layout is
//! guarded here instead, by replaying each lane's own liveness and asking the
//! one question the executors depend on: does any pair of values this lane
//! holds live at the same step share a byte?
//!
//! The second half is the measurement that justified the pass at all. The
//! floor no layout can beat is the arena's busiest instant --
//! `live_bound` -- and every lane of every catalogue row lands exactly on it,
//! which is what turned gemma4-31b's 21.8 MiB per row into 1 MiB and
//! qwen35-d0.8b's 2.45 MiB into 487 KiB. A failure of the equality below is a
//! FRAGMENTATION report and not a correctness one: the greedy carve left a
//! hole some future text's interval graph put there.

use model_compiler::program::{Program, Slot};
use model_dsl::Plane;
use model_ir::plan::Plan;

fn traced(sku: &str) -> Plan {
    let row = model::trace_of(sku).unwrap_or_else(|| panic!("`{sku}` is not a catalog row"));
    row(Plane::Cuda)
}

/// Every lane of every shipping row, bound. A row that refuses is a refusal
/// this file reports rather than skips.
/// Every catalog row whose lanes BIND, with the plan and programs each one
/// compiled to.
///
/// A ROW THAT REFUSES IS SKIPPED, NOT A PANIC. This helper used to unwrap,
/// and every test that used it died on `dsv4-base-bf16-kv-bf16-tp2` — whose
/// lanes refuse `dist.all_reduce`, a point no plane claims. That is a
/// truthful refusal about a backlog, not a compiler defect, and six tests
/// about ARENA LIVENESS reporting it said nothing about arena liveness.
///
/// The set that refuses is not swallowed: [`the_rows_that_refuse_are_named`]
/// asserts it, so a row that starts refusing is as loud as it was here and
/// says which point it is short of.
fn catalogue() -> Vec<(&'static str, Plan, Vec<Program>)> {
    model::catalog()
        .into_iter()
        .filter_map(|(sku, _)| {
            let plan = traced(sku);
            let lanes = model_compiler::program::programs(&plan).ok()?;
            Some((sku, plan, lanes))
        })
        .collect()
}

/// The catalog rows whose lanes do not bind, and the point each is short of.
///
/// This is the assertion [`catalogue`]'s skip would otherwise hide. It is a
/// list rather than a count because WHICH row refuses says which point is
/// missing, and here the list says something a count could not: EVERY ROW
/// THAT REFUSES IS A `-tp2` ROW, and every one of them is short of the same
/// point. `dist.all_reduce` is claimed by no plane, a two-way row cannot
/// resolve without it, and a one-way row never asks. So this is one backlog
/// row wearing six names.
///
/// The panic this replaced showed only the FIRST of the six, because the
/// helper unwrapped and died there — which is how a six-row gap read as a
/// one-row gap for as long as anyone looked at it.
#[test]
fn the_rows_that_refuse_are_named() {
    let mut refused: Vec<&str> = model::catalog()
        .into_iter()
        .filter(|(sku, _)| model_compiler::program::programs(&traced(sku)).is_err())
        .map(|(sku, _)| sku)
        .collect();
    refused.sort_unstable();
    assert_eq!(
        refused,
        [
            "dsv4-base-bf16-kv-bf16-tp2",
            "gemma4-31b-bf16-kv-bf16-tp2",
            "glm5-a12b-bf16-bf16-kv-bf16-tp2",
            "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
            "kimik3-bf16-mxfp4-kv-bf16-tp2",
            "qwen35-a3b-bf16-kv-bf16-tp2",
        ],
        "the rows that refuse have moved. A row GAINED is a point that stopped \
         being claimed; a row LOST is a backlog row that closed, and that edit \
         is the record of it."
    );
    let told = model_compiler::program::programs(&traced("dsv4-base-bf16-kv-bf16-tp2"))
        .expect_err("it refuses");
    assert!(
        told.iter()
            .any(|r| r.to_string().contains("dist.all_reduce")),
        "the refusal names the point it is short of: {told:?}"
    );
}

/// THE INVARIANT. No two values a lane holds live at one step share a byte.
#[test]
fn no_two_live_values_share_a_byte() {
    for (sku, plan, lanes) in catalogue() {
        for (at, lane) in lanes.iter().enumerate() {
            let clashes = model_compiler::program::clashes(&plan, lane);
            assert!(
                clashes.is_empty(),
                "`{sku}` lane {at}: {} pair(s) of simultaneously live values share bytes, \
                 first {:?}",
                clashes.len(),
                clashes.first()
            );
        }
    }
}

/// AND NOTHING LEAVES THE BLOCK. Both executors read a slot at
/// `offset * rows` for `bytes * rows`, and the bound that keeps that inside
/// `row_pitch * rows` is this one, per row.
#[test]
fn every_rectangle_fits_the_pitch() {
    for (sku, _, lanes) in catalogue() {
        for (at, lane) in lanes.iter().enumerate() {
            let mut arena = 0;
            for (id, slot) in lane.slots.iter().enumerate() {
                let Slot::Arena { offset, .. } = slot else {
                    continue;
                };
                arena += 1;
                assert_eq!(
                    offset % 256,
                    0,
                    "`{sku}` lane {at}: value {id} is not bindable on a device that \
                     demands 256 — see `program::BIND_ALIGN`"
                );
                assert!(
                    offset + slot.bytes() <= lane.row_pitch,
                    "`{sku}` lane {at}: value {id} runs past the row pitch"
                );
            }
            assert!(arena > 0, "`{sku}` lane {at} allocates nothing");
            assert_eq!(
                lane.row_pitch % 256,
                0,
                "`{sku}` lane {at}: unaligned pitch"
            );
        }
    }
}

/// A STATEMENT NEVER READS ITS OWN RESULT'S BYTES -- the `InOut` decision,
/// stated where it can be checked.
///
/// The walk gives an `InOut` point's operand and result distinct slabs, and
/// both executors lean on that: they stage the operand INTO the result's
/// rectangle with a device-to-device copy before the launch
/// (`driver-cuda/src/baker/fire.rs`'s `inout`), which is only a copy if the
/// two do not overlap. Sharing them instead would be a claim about the
/// kernel's indexing -- may it read a lane of its input after writing that
/// lane of its output -- that no plan states.
///
/// Checked for every statement and not just the declared-`InOut` ones,
/// because it is the same rule: a launch reads its operands while it writes
/// its results.
#[test]
fn no_statement_writes_the_bytes_it_is_reading() {
    for (sku, plan, lanes) in catalogue() {
        for (at, lane) in lanes.iter().enumerate() {
            for step in &lane.steps {
                let op = &plan.ops[step.op as usize];
                for out in &op.outputs {
                    let Some((o_at, o_bytes)) = span_of(lane, *out) else {
                        continue;
                    };
                    for input in &op.inputs {
                        let Some((i_at, i_bytes)) = span_of(lane, *input) else {
                            continue;
                        };
                        assert!(
                            o_at >= i_at + i_bytes || i_at >= o_at + o_bytes,
                            "`{sku}` lane {at}: `{}` reads value {input} out of the bytes it \
                             writes value {out} into",
                            op.kernel
                        );
                    }
                }
            }
        }
    }
}

/// Where a value's bytes are, chasing a merge to the arm that survived. A
/// merge shares BY CONSTRUCTION -- it is the arm -- so a statement that reads
/// one and writes it back is not what this file is looking for.
fn span_of(lane: &Program, value: u32) -> Option<(u64, u64)> {
    match &lane.slots[value as usize] {
        slot @ Slot::Arena { offset, .. } => {
            let bytes = slot.bytes();
            (bytes > 0).then_some((*offset, bytes))
        }
        Slot::Alias(through) => span_of(lane, *through),
        Slot::Runtime(_) | Slot::Absent => None,
    }
}

/// THE MEASUREMENT. Every lane sits on the floor no layout can beat.
#[test]
fn the_pitch_is_the_busiest_instant() {
    for (sku, plan, lanes) in catalogue() {
        for (at, lane) in lanes.iter().enumerate() {
            let bound = model_compiler::program::live_bound(&plan, lane);
            assert_eq!(
                lane.row_pitch,
                bound,
                "`{sku}` lane {at}: the carve left {} bytes of hole over the busiest \
                 instant -- a fragmentation report, not a wrong answer",
                lane.row_pitch.saturating_sub(bound)
            );
        }
    }
}

/// THE SAME PLAN LAYS OUT THE SAME WAY. A program is cached, compared and
/// fired by its offsets, so the carve's order is a total one over sizes,
/// birth steps and value ids -- never a hash's.
#[test]
fn the_layout_is_deterministic() {
    for (sku, _) in model::catalog() {
        // A row that refuses is skipped for [`catalogue`]'s reason, and
        // [`the_rows_that_refuse_are_named`] is what keeps the skip honest.
        let (Ok(once), Ok(twice)) = (
            model_compiler::program::programs(&traced(sku)),
            model_compiler::program::programs(&traced(sku)),
        ) else {
            continue;
        };
        for (at, (a, b)) in once.iter().zip(&twice).enumerate() {
            assert_eq!(a.row_pitch, b.row_pitch, "`{sku}` lane {at}: two pitches");
            assert_eq!(a.slots, b.slots, "`{sku}` lane {at}: two layouts");
        }
    }
}

/// THE `out` SEAM OUTLIVES THE WALK, and nothing else does.
///
/// The driver repitches `fire.rect(baked.out)` into the logits buffer after
/// the last step has been issued, and `baker-smoke` reads the same rectangle
/// the same way. So the value that seam names is live at a time no statement
/// occupies -- which is a fact about the READER and has to be written down in
/// the walk, because nothing in the plan says it.
#[test]
fn the_logits_live_past_the_last_step() {
    for (sku, plan, lanes) in catalogue() {
        let out = plan
            .seams
            .iter()
            .find(|s| s.seam == model_ir::seam::OUT.name)
            .and_then(|s| s.values.first().copied())
            .unwrap_or_else(|| panic!("`{sku}` states no `out` seam"));
        for (at, lane) in lanes.iter().enumerate() {
            let spans = model_compiler::program::spans(&plan, lane);
            let root = root_of(lane, out);
            let span = spans[root as usize]
                .unwrap_or_else(|| panic!("`{sku}` lane {at}: the `out` seam has no span"));
            assert_eq!(
                span.last,
                lane.steps.len() as u32,
                "`{sku}` lane {at}: the logits are dropped before the delivery reads them"
            );
        }
    }
}

fn root_of(lane: &Program, mut value: u32) -> u32 {
    while let Slot::Alias(through) = lane.slots[value as usize] {
        value = through;
    }
    value
}
