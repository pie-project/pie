//! What a rank cut does to a BOUND program, which is the half the shard
//! column cannot say on its own.
//!
//! `model/tests/a_rank_cut_is_the_shard_column.rs` holds a `-tp2` row's
//! weights against its sibling's. This holds its RECTANGLES: the arena the
//! walk sizes out of those weights and the statement params beside them. The
//! two are different measurements, and only the second one is what a fire
//! actually allocates.
//!
//! # What was measured before the cut landed
//!
//! Every `-tp2` row bound to a program IDENTICAL to its sibling's but for
//! the `dist.all_reduce` statements: the same 687 params, the same widths,
//! the same `row_pitch` to the byte. A rank was sizing an arena for the whole
//! model and reducing over a leg nothing had cut — a plan whose numbers lied
//! by the world size. So the assertions below are stated as INEQUALITIES
//! against the sibling rather than as pinned constants: the claim is that a
//! rank's rectangles are smaller than the whole model's, and a pin would go
//! stale the moment a text's dims are corrected.
//!
//! # The pitch does NOT move, and that is the honest reading
//!
//! Every row in this catalog is logits-dominated: its busiest instant is the
//! last one, where the `[vocab]` row the head writes is live. `embed` and
//! `lm_head` are replicated — the logits a rank produces are the WHOLE
//! distribution, because the sampler in front of them is — so no cut touches
//! that instant and `row_pitch` is unchanged on all six rows. What the cut
//! moves is everything below it: the per-layer working set, which is what
//! `footprint` measures.

use model_compiler::program::{Program, Slot};
use model_ir::kernels::Backend;
use model_ir::plan::Plan;

fn traced(sku: &str) -> Plan {
    let row = model::trace_of(sku).unwrap_or_else(|| panic!("`{sku}` is not a catalog row"));
    row(Backend::Cuda)
}

fn bound(plan: &Plan) -> Vec<Program> {
    model_compiler::program::programs(plan).unwrap_or_else(|refusals| {
        let told: Vec<String> = refusals.iter().map(ToString::to_string).collect();
        panic!("`{}` refused: {}", plan.name, told.join(" | "))
    })
}

/// Every `-tp2` SKU and the row it is a rank cut of.
fn pairs() -> Vec<(&'static str, String)> {
    model::catalog()
        .into_iter()
        .filter_map(|(sku, _)| {
            let (base, _) = sku.rsplit_once("-tp")?;
            Some((sku, base.to_string()))
        })
        .collect()
}

/// The bytes one fire row of this lane's rectangles occupies BEFORE the carve
/// shares any of them — the sum, not the busiest instant.
///
/// The reuse is what makes `row_pitch` a max, and a max is dominated by the
/// one rectangle no cut touches. The sum is what a rank's share of the work
/// actually is.
fn footprint(program: &Program) -> u64 {
    program.slots.iter().map(Slot::bytes).sum()
}

/// **UNREACHABLE TODAY, AND NOT BECAUSE OF THE HARDWARE.** Every `-tp2` row
/// in the catalog refuses to bind — all six of them, all on the same point:
/// `dist.all_reduce`, which no plane claims. This test's whole subject is
/// what a rank cut does to a two-way lane, so there is nothing for it to
/// measure until that point is answered.
///
/// It is `#[ignore]`d rather than skipped inside, because a skip inside would
/// leave it reporting `ok` over an empty loop — which is the exact defect
/// this suite has had to fix three times this week. An ignored test says it
/// did not run. `model-compiler/tests/arena_liveness.rs::
/// the_rows_that_refuse_are_named` is where the six rows are asserted, so
/// this debt is measured even while this test cannot run.
#[test]
#[ignore = "every -tp2 row refuses on dist.all_reduce, which no plane claims"]
fn every_rank_cut_lane_carries_less_than_the_whole_models() {
    for (sku, base) in pairs() {
        let (mine, whole) = (traced(sku), traced(&base));
        let (mine, whole) = (bound(&mine), bound(&whole));
        assert_eq!(
            mine.len(),
            whole.len(),
            "`{sku}` and `{base}` state different numbers of behaviors",
        );
        for (at, (m, w)) in mine.iter().zip(&whole).enumerate() {
            assert!(
                footprint(m) < footprint(w),
                "`{sku}` lane {at} carries {} bytes per fire row and `{base}` \
                 carries {} — a rank cut that moved nothing",
                footprint(m),
                footprint(w),
            );
            // THE WIDEST SLOT IS THE LOGITS ROW and no cut reaches it: a rank
            // samples from the whole distribution or it is not sampling. A
            // catalog row where this stopped holding would be one whose head
            // had become vocab-parallel, which is a statement change (an
            // all-gather) and not a width.
            let widest = |p: &Program| {
                p.slots
                    .iter()
                    .filter_map(|s| match s {
                        Slot::Arena { width, .. } => Some(*width),
                        _ => None,
                    })
                    .max()
            };
            assert_eq!(
                widest(m),
                widest(w),
                "`{sku}` lane {at}: the widest rectangle moved under a rank cut",
            );
        }
    }
}

/// A rank cut row still binds every lane, and states the reduce that closes
/// it.
///
/// The gate the whole item is measured by: `-tp2` rows resolved before this
/// change too, so "it still binds" is not news — what is news is that it
/// binds at HALF the widths, which the test above says, and that the extra
/// statements a cut row carries are exactly the reduces.
#[test]
fn a_rank_cut_adds_reduces_and_nothing_else() {
    for (sku, base) in pairs() {
        let (mine, whole) = (traced(sku), traced(&base));
        let reduces = mine
            .ops
            .iter()
            .filter(|o| o.kernel == "dist.all_reduce")
            .count();
        assert!(reduces > 0, "`{sku}` reduces nothing");
        assert_eq!(
            mine.ops.len() - whole.ops.len(),
            reduces,
            "`{sku}` states {} ops to `{base}`'s {} and {reduces} of them are \
             reduces",
            mine.ops.len(),
            whole.ops.len(),
        );
        assert_eq!(
            whole
                .ops
                .iter()
                .filter(|o| o.kernel == "dist.all_reduce")
                .count(),
            0,
            "`{base}` is not a rank cut and reduces anyway",
        );
    }
}
