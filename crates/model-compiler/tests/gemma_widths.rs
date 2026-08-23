//! What the shape walk says gemma-31b measures, pinned against the text.
//!
//! The width table in `program.rs` is a claim about six model texts: that a
//! matmul's row is its bank's out axis, that an attention hands back the `q`
//! it was given, that a packed cut is the halves its params state. Nothing in
//! the type system checks that claim -- the plan carries `Vec<u64>` params and
//! `Vec<u64>` shapes, and every arithmetic mistake in the table is a NUMBER
//! that binds silently and computes the wrong thing on a device nobody in CI
//! has.
//!
//! So it is checked here, against the real catalogue row, at the numbers
//! `gemma_4/model.rs` states: hidden 5376, vocab 262_144, 32 query heads over
//! a 256-wide sliding head and a 512-wide global one. A fixture plan would
//! have proved only that the table agrees with itself.

use model_compiler::program::{Dt, Program, Slot};
use model_dsl::Plane;
use model_ir::plan::{Plan, ValueDef, ValueId};

const SKU: &str = "gemma4-31b-bf16-kv-bf16";

/// gemma-31b's `b31()` dims, which every assertion below is stated in.
const HIDDEN: u64 = 5376;
const VOCAB: u64 = 262_144;
const Q_HEADS: u64 = 32;
/// The sliding layers' head, and the global layers' — five of every six
/// layers slide at 256 and the sixth reads globally at 512.
const SLIDING_HEAD: u64 = 256;
const GLOBAL_HEAD: u64 = 512;

fn traced() -> Plan {
    let row = model::trace_of(SKU).unwrap_or_else(|| panic!("`{SKU}` is not a catalog row"));
    row(Plane::Cuda)
}

/// The rectangle a lane gives `value`, chasing a merge to the arm that
/// survived on it.
fn rect(program: &Program, value: ValueId) -> Option<(u64, Dt)> {
    match &program.slots[value as usize] {
        Slot::Arena { width, dtype, .. } => Some((*width, *dtype)),
        Slot::Alias(through) => rect(program, *through),
        Slot::Runtime(_) | Slot::Absent => None,
    }
}

/// Every statement of `point` this lane runs, as `(op index, the op)`.
fn stating<'p>(plan: &'p Plan, program: &Program, point: &str) -> Vec<&'p model_ir::plan::Op> {
    program
        .steps
        .iter()
        .map(|step| &plan.ops[step.op as usize])
        .filter(|op| op.kernel == point)
        .collect()
}

#[test]
fn every_lane_of_gemma_31b_binds() {
    let plan = traced();
    match model_compiler::program::programs(&plan) {
        Ok(lanes) => {
            assert_eq!(lanes.len(), 3, "gemma states two facts and three behaviors");
            for lane in &lanes {
                assert!(lane.row_pitch > 0, "a lane that allocates nothing");
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
            panic!("gemma-31b refused: {}", told.join(" | "));
        }
    }
}

#[test]
fn the_embedding_is_hidden_wide_and_the_head_is_vocab_wide() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("gemma-31b binds");
    for lane in &lanes {
        let embed = stating(&plan, lane, "layout.embed");
        assert_eq!(embed.len(), 1, "gemma states one embedding");
        assert_eq!(
            rect(lane, embed[0].outputs[0]),
            Some((HIDDEN, Dt::Bf16)),
            "the embedding is the table's OWN width -- its operand is token ids"
        );

        let head = stating(&plan, lane, "gemm.lm_head");
        assert_eq!(head.len(), 1, "gemma states one head");
        assert_eq!(
            rect(lane, head[0].outputs[0]),
            Some((VOCAB, Dt::Bf16)),
            "the head's row is the vocabulary: a bank is `[out, in]`"
        );
    }
}

#[test]
fn an_attention_hands_back_the_query_it_was_given() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("gemma-31b binds");
    let mut widths = Vec::new();
    for lane in &lanes {
        for point in ["attention.decode", "attention.prefill", "attention.masked"] {
            for op in stating(&plan, lane, point) {
                let q = rect(lane, op.inputs[0]).expect("a bound query");
                let o = rect(lane, op.outputs[0]).expect("a bound reading");
                assert_eq!(o, q, "`{point}` sized its reading off something else");
                widths.push(q.0);
            }
        }
    }
    widths.sort_unstable();
    widths.dedup();
    assert_eq!(
        widths,
        vec![Q_HEADS * SLIDING_HEAD, Q_HEADS * GLOBAL_HEAD],
        "gemma reads at two head widths and no others"
    );
}

/// The two ways gemma reaches a rotated query -- the tier-2 fusion on the
/// decode lane and the `split_qkv` cut everywhere else -- must agree about how
/// wide one is. They are sized by different rules (a packed row less its two
/// kv planes, against a stated param), which is exactly why this is checked.
#[test]
fn the_fused_query_is_the_split_query() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("gemma-31b binds");
    let sliding_q = Q_HEADS * SLIDING_HEAD;
    let sliding_kv = 16 * SLIDING_HEAD;

    let mut fused = 0;
    let mut split = 0;
    for lane in &lanes {
        for op in stating(&plan, lane, "cuda::qkv_fused_qknorm_rope_vnorm_write") {
            assert_eq!(
                rect(lane, op.outputs[0]),
                Some((sliding_q, Dt::Bf16)),
                "the fusion writes kv straight to the pages and returns q alone"
            );
            fused += 1;
        }
        for op in stating(&plan, lane, "layout.split_qkv") {
            let (q, k, v) = (op.outputs[0], op.outputs[1], op.outputs[2]);
            let q = rect(lane, q).expect("a bound q");
            assert_eq!(rect(lane, k), rect(lane, v), "k and v are one width");
            if q.0 == sliding_q {
                assert_eq!(rect(lane, k), Some((sliding_kv, Dt::Bf16)));
            }
            split += 1;
        }
    }
    assert!(fused > 0 && split > 0, "both paths are in the plan");
}

/// A merge allocates nothing: the value IS the arm that survived, and the
/// arms this lane never runs are `Absent`.
#[test]
fn a_merge_aliases_its_surviving_arm() {
    let plan = traced();
    let lanes = model_compiler::program::programs(&plan).expect("gemma-31b binds");
    let mut aliased = 0;

    for lane in &lanes {
        for (id, def) in plan.values.iter().enumerate() {
            let ValueDef::Merge(arms) = def else { continue };
            let Slot::Alias(through) = lane.slots[id] else {
                panic!("no arm of a merge survives on a lane that reaches it");
            };
            assert!(
                matches!(lane.slots[through as usize], Slot::Arena { .. }),
                "an alias that points at no rectangle"
            );
            let live = arms
                .iter()
                .filter(|(arm, _)| !matches!(lane.slots[*arm as usize], Slot::Absent))
                .count();
            assert_eq!(live, 1, "a lane on which two arms of one merge survive");
            aliased += 1;
        }
    }
    assert!(aliased > 0, "gemma splits and merges every layer");
}
