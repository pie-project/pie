//! The capture-unit partition (multimodal §1). Which regions belong to which
//! exec, DERIVED from the dimension algebra and never declared.
//!
//! # The rule, in one sentence
//!
//! **A region's unit is the row axis of the rows it writes.** A node's outputs
//! carry leading `Dim`s; `Dim::axis` reads a `RowAxis` off each; a region is a
//! maximal run of nodes that agree about phase, about class mask AND about
//! that axis; and the units are the axes, in the order their first capture
//! region stands in the region table. The fire launches one exec per unit,
//! chained on one stream.
//!
//! WHY DERIVED AND NOT DECLARED. Because it is already written down. A model
//! text that states a vision tower states patch-shaped values because that is
//! what the tower computes over, and a second declaration — "this block is a
//! tower" — would be a fact free to disagree with the shapes. It is the same
//! argument `Phase` makes: the rule is READ OFF THE TYPE and not off a list of
//! op names, so a tower somebody adds next year is partitioned by the shapes
//! it could not have avoided writing.
//!
//! # Why the axis joins the coalescing key
//!
//! `region::coalesce` already refuses to put two nodes in one run when they
//! disagree about phase or about which classes run them, for the same reason
//! in both cases: a region is the unit the descriptor carries ONE row count
//! for. An axis is a third such reason and the strongest — the row counts are
//! not merely different numbers, they are numbers from different window
//! tables. So the axis joins the key rather than being reconciled afterwards.
//!
//! **AND A `Const`-SHAPED NODE JOINS EITHER SIDE.** A node whose outputs are
//! all fixed blocks is not fire-aligned, has no axis to disagree about, and
//! constrains nothing; it takes the axis of the run it lands in. That is what
//! makes this pass free on a plan with one axis: every node is `Tokens` or
//! `None`, no key ever differs, and the region table is the one P2 always
//! built — byte for byte, which is the G4 invariant.
//!
//! # What it refuses
//!
//! [`Error::UnitsInterleave`]. Two capture units are two execs, and an exec is
//! a contiguous stretch of the record script — the tower's capture, then the
//! trunk's. A plan whose axes alternate down the region table does not name
//! two execs; it names four, or six, and the launch order that would fix it is
//! a scheduling decision this pass has no dependence graph to make. So it is
//! named at the door. The shape that works — and the shape every declared
//! tower has — is the tower stated before the trunk that reads its output,
//! which is program order already.
//!
//! # The fold
//!
//! [`fold_refused`] is the escape hatch multimodal §5.3 adopts, said out loud:
//! the graph-fold plane is structurally one graph per bucket, so a multi-unit
//! bucket has no correct fold and serves the KEYED path for the life of the
//! load. That costs nothing structural and it defers the "6 + 6, not 6 × 6"
//! property, which is a property of per-unit keys. The flag rides in the
//! artifact so the engine refuses BY NAME rather than folding something it
//! cannot fold.

use model_ir::{Operands, RowAxis, Trace, Ty, ValueId};

use crate::compiled::{Phase, Region};
use crate::error::Error;

/// The row axis a node writes, or `None` for a node whose outputs are all
/// fixed blocks or host-owned structs.
///
/// OUTPUTS AND NOT INPUTS, and the handoff is why. `layout::scatter_rows`
/// reads the tower's patch-shaped output and writes TOKEN rows — the embed
/// merge, the one node that touches both axes. It belongs to the token unit,
/// because the descriptor row count its launch reads is a token count; asking
/// the question of its inputs would put it in the tower and give it the wrong
/// window. A node writes into exactly one row space, which is what makes this
/// answer single-valued.
#[must_use]
pub(crate) fn node_axis(trace: &Trace, node: &model_ir::Node, outs: &mut Vec<ValueId>) -> Option<RowAxis> {
    outs.clear();
    node.op.outputs(outs);
    let mut found = None;
    for value in outs.iter() {
        let Some(decl) = trace.values.get(value.0 as usize) else {
            continue;
        };
        let Ty::Tensor { shape, .. } = &decl.ty else {
            continue;
        };
        let Some(axis) = shape.first().and_then(|dim| dim.axis()) else {
            continue;
        };
        // A node writing two axes at once is not a thing the IR can state —
        // one op, one window — and if one ever were, the conservative reading
        // is the primary axis, which is the one the descriptor's row count
        // already means. `partition` below is what would then refuse it.
        found = Some(match found {
            None => axis,
            Some(held) if held == axis => held,
            Some(_) => RowAxis::PRIMARY,
        });
    }
    found
}

/// Every axis this plan states, whether or not any region ended up on it.
///
/// Read off the VALUES rather than off the regions, because a plan may declare
/// a patch-shaped runtime input and compute nothing from it — and a budget
/// that sizes no patch ceiling has to be refused for that too, before the
/// carve reserves a rectangle at zero.
#[must_use]
pub(crate) fn axes_stated(trace: &Trace) -> Vec<RowAxis> {
    let mut axes = Vec::new();
    for decl in &trace.values {
        let Ty::Tensor { shape, .. } = &decl.ty else {
            continue;
        };
        for dim in shape {
            if let Some(axis) = dim.axis() {
                if !axes.contains(&axis) {
                    axes.push(axis);
                }
            }
        }
    }
    axes.sort_unstable();
    axes
}

/// The axis each region runs on — the first one any of its nodes states, which
/// by the coalescing key is the only one any of them states.
#[must_use]
pub(crate) fn axes_of(trace: &Trace, regions: &[Region]) -> Vec<Option<RowAxis>> {
    let mut outs = Vec::new();
    regions
        .iter()
        .map(|region| {
            region
                .nodes
                .clone()
                .filter_map(|at| trace.nodes.get(at as usize))
                .find_map(|node| node_axis(trace, node, &mut outs))
        })
        .collect()
}

/// The axis of each unit in exec order, and the unit of each region.
///
/// # Errors
///
/// [`Error::UnitsInterleave`] when a unit's capture regions are not one
/// contiguous stretch of the record script.
pub(crate) fn partition(
    regions: &[Region],
    axes: &[Option<RowAxis>],
) -> Result<(Vec<RowAxis>, Vec<u32>), Error> {
    // The units, in the order their first CAPTURE region stands. Prepare
    // regions are host work — `hoist` has already put all of them in front of
    // every capture region — so they name no exec and take the unit of the
    // capture half they precede, which is unit 0 by construction.
    let mut units: Vec<RowAxis> = Vec::new();
    for (r, region) in regions.iter().enumerate() {
        if region.phase != Phase::Capture {
            continue;
        }
        let axis = axes.get(r).copied().flatten().unwrap_or(RowAxis::PRIMARY);
        if !units.contains(&axis) {
            units.push(axis);
        }
    }
    if units.is_empty() {
        units.push(RowAxis::PRIMARY);
    }

    // **WHERE THE PRIMARY AXIS'S UNIT STANDS**, which is not unit 0 on a plan
    // whose first capture region is a tower's. Read once: it is what an
    // axis-less PREPARE region takes, and the paragraph below is why.
    let primary = units
        .iter()
        .position(|held| *held == RowAxis::PRIMARY)
        .unwrap_or(0) as u32;

    // A region with no axis of its own — all-`Const` outputs, or a prepare
    // region's struct — belongs to whichever unit is open where it stands,
    // and to unit 0 before any capture region has opened one. That is the
    // reading that keeps a one-axis plan's every region on unit 0.
    //
    // **EXCEPT THAT A PREPARE REGION NAMES NO EXEC, AND "THE UNIT OPEN WHERE
    // IT STANDS" MEANS NOTHING FOR ONE.** `hoist` puts every prepare region in
    // front of every capture region, so `open` is always its initial `0` when
    // one is stamped — unit 0 by POSITION, not by meaning. On a one-axis plan
    // unit 0 IS the primary axis and the reading is accidentally right; on a
    // plan whose first capture unit is the TOWER it is `RowAxis::Patches`, and
    // the stamp is not inert: `engine_cuda::window::Windows::of` and
    // `model_exec::fire::walk` both read a region's unit to decide WHICH
    // WINDOW TABLE its rows come out of. A hoisted `attention.plan_prefill`
    // then gets cut at the patch table, its `indptr_host` is left empty
    // because that vector is the token rectangle's, and the schedule builder
    // refuses with "the host qo_indptr holds 0 entries for a batch of 1".
    //
    // The truth is that a plan builder's rows are the rows the schedule it
    // builds covers, and every one of them in this IR reads per-LANE geometry
    // on the primary axis. So an axis-less prepare region takes the primary
    // unit, and a prepare region that DOES state an axis still takes its own —
    // the `Some` arm above is unchanged, so a patch-axis prepare region a
    // later tower states is partitioned by its shapes like everything else.
    //
    // **AND A ONE-AXIS PLAN CANNOT MOVE**, which is the G4 invariant restated:
    // `units` is then `[Tokens]`, `primary` is 0, `open` is 0, and the two
    // arms answer the same number for every region.
    let mut open = 0u32;
    let mut units_of: Vec<u32> = Vec::with_capacity(regions.len());
    for (r, region) in regions.iter().enumerate() {
        let unit = match axes.get(r).copied().flatten() {
            Some(axis) => units.iter().position(|held| *held == axis).unwrap_or(0) as u32,
            None if region.phase != Phase::Capture => primary,
            None => open,
        };
        if region.phase == Phase::Capture {
            open = unit;
        }
        units_of.push(unit);
    }

    // TWO EXECS, NOT FOUR. Each unit's capture regions have to be one run of
    // the script, because an exec is recorded front to back and a unit that
    // resumes after another one has run is a second exec of the same unit.
    let mut seen: Vec<u32> = Vec::new();
    let mut previous: Option<u32> = None;
    for (r, region) in regions.iter().enumerate() {
        if region.phase != Phase::Capture {
            continue;
        }
        let unit = units_of[r];
        if previous == Some(unit) {
            continue;
        }
        if seen.contains(&unit) {
            return Err(Error::UnitsInterleave {
                axis: units[unit as usize],
                unit,
                nodes: region.nodes.clone(),
            });
        }
        seen.push(unit);
        previous = Some(unit);
    }

    Ok((units, units_of))
}

/// Does this artifact's fold have to stand down?
///
/// **YES FOR EVERY MULTI-UNIT PLAN, AND THAT IS THE WHOLE RULE** (multimodal
/// §5.3). The fold plane arms one graph per bucket per key; a fire that
/// launches two execs has two bucket numbers and no single graph to arm, so
/// there is no correct fold to build and the honest answer is the keyed path
/// for the life of the load. A per-unit fold is its own later wave; a
/// fire-level key carrying both bucket numbers would be exactly the `6 × 6`
/// product §1 refuses.
#[must_use]
pub(crate) fn fold_refused(units: &[RowAxis]) -> bool {
    units.len() > 1
}

#[cfg(test)]
mod tests {
    use crate::budget::{Budget, Budgets, DeviceProfile, PatchLadder};
    use crate::compiled::Phase;
    use crate::fixture::{Build, block, fact, patch};
    use crate::{Error, compile, compile_axes};
    use model_ir::{Guard, RowAxis};

    fn budget() -> Budget {
        Budget::new(4, 16)
    }

    fn with_patches() -> Budgets {
        Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 32,
            buckets: vec![8, 16, 32],
            max_images: 4,
        })
    }

    /// The G4 invariant, in miniature and where the pass can be watched: a
    /// plan that states one row space is one capture unit, its regions all on
    /// it, and NOTHING about the artifact moves when the deployment admits a
    /// patch axis the plan never uses.
    #[test]
    fn a_plan_with_one_row_space_is_one_capture_unit_and_pays_the_axis_nothing() {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 8);
        let y = b.op(o, 8, Guard::Always);
        b.out(y);

        let plain = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(plain.units, vec![RowAxis::Tokens]);
        assert!(plain.units_of.iter().all(|&unit| unit == 0));
        assert_eq!(plain.units_of.len(), plain.regions.len());
        assert!(!plain.fold_refused);
        assert_eq!(plain.patches, None);
        assert_eq!(plain.order_for(RowAxis::Patches), None);

        // THE WHOLE ARTIFACT, NOT A FIELD OF IT. A deployment that declares a
        // patch ladder for a text-only plan gets the artifact it always got.
        let admitted =
            compile_axes(&b.trace, &with_patches(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(plain, admitted);
    }

    /// A `Const`-shaped node has no axis to disagree about, so it neither
    /// splits a run nor opens a unit — which is the clause that keeps the
    /// region table of a one-axis plan exactly the one P2 always built.
    #[test]
    fn a_fixed_block_joins_the_run_it_lands_in_and_opens_no_unit() {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always);
        let fixed = b.shaped(q, block(4, 8), Guard::Always);
        let y = b.op(fixed, 8, Guard::Always);
        b.out(y);

        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(compiled.units, vec![RowAxis::Tokens]);
        assert_eq!(
            compiled
                .regions
                .iter()
                .filter(|r| r.phase == Phase::Capture)
                .count(),
            1,
            "three nodes, one mask, one phase, one axis — one region",
        );
    }

    /// The tower shape: patch rows first, then the token trunk that reads
    /// them. Two units, in exec order, and the fold stands down by name.
    #[test]
    fn a_tower_and_a_trunk_are_two_units_in_exec_order() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let deeper = b.shaped(tower, patch(8), Guard::Always);
        // The embed merge: reads patch rows, WRITES token rows. It belongs to
        // the trunk, because the row count its launch reads is a token count.
        let merged = b.op(deeper, 8, Guard::Always);
        let y = b.op(merged, 8, Guard::Always);
        b.out(y);

        let compiled =
            compile_axes(&b.trace, &with_patches(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(compiled.units, vec![RowAxis::Patches, RowAxis::Tokens]);
        assert!(compiled.fold_refused, "two units have no single graph to arm");

        // The tower's two nodes are one region on unit 0; the trunk's two are
        // one region on unit 1.
        let capture: Vec<(u32, std::ops::Range<u32>)> = compiled
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.phase == Phase::Capture)
            .map(|(r, region)| (compiled.unit_of(r), region.nodes.clone()))
            .collect();
        assert_eq!(capture, vec![(0, 0..2), (1, 2..4)]);
        assert_eq!(compiled.unit_script(0), Some(0..1));
        assert_eq!(compiled.unit_script(1), Some(1..2));

        // A second seriation, over the same classes, with its own answers.
        assert!(compiled.patches.is_some());
        assert_eq!(compiled.order_for(RowAxis::Patches), Some(&compiled.order));
        assert!(compiled.fallback_for(RowAxis::Patches).is_some());
    }

    /// The axis is what splits them, not the mask and not the phase: the same
    /// guard, the same phase, two row spaces, two regions.
    #[test]
    fn one_mask_and_one_phase_still_break_at_the_axis() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let y = b.op(tower, 8, Guard::Always);
        b.out(y);

        let compiled =
            compile_axes(&b.trace, &with_patches(), &DeviceProfile::default()).expect("bakes");
        let capture: Vec<_> = compiled
            .regions
            .iter()
            .filter(|r| r.phase == Phase::Capture)
            .collect();
        assert_eq!(capture.len(), 2);
        assert_eq!(capture[0].mask, capture[1].mask, "the mask did NOT split them");
    }

    /// A plan that states a patch row against a budget that sizes none is a
    /// load that does not happen — not a tower carved at zero rows.
    #[test]
    fn a_patch_row_against_no_patch_ceiling_is_refused_by_name() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let y = b.op(tower, 8, Guard::Always);
        b.out(y);

        let refusal = compile(&b.trace, &budget(), &DeviceProfile::default())
            .expect_err("no ceiling, no load");
        assert_eq!(
            refusal,
            Error::Unsized {
                axis: RowAxis::Patches
            }
        );
        assert!(refusal.to_string().contains("patches"));
    }

    /// A unit that resumes after another has run is not one exec, and the
    /// refusal says which unit and where.
    #[test]
    fn a_unit_that_resumes_after_another_is_refused_rather_than_recorded_twice() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let trunk = b.op(tower, 8, Guard::Always);
        // A SECOND stretch of patch rows, after the trunk has already run.
        let again = b.shaped(trunk, patch(8), Guard::Always);
        let y = b.op(again, 8, Guard::Always);
        b.out(y);

        let refusal = compile_axes(&b.trace, &with_patches(), &DeviceProfile::default())
            .expect_err("two stretches of one axis are two execs of one name");
        assert_eq!(
            refusal,
            Error::UnitsInterleave {
                axis: RowAxis::Patches,
                unit: 0,
                nodes: 2..3,
            }
        );
        assert!(refusal.to_string().contains("one contiguous stretch"));
    }

    /// The patch ladder is checked in the token ladder's vocabulary, and it is
    /// its own vector: a rung past the patch ceiling is a refusal even where
    /// it would have been legal against `max_tokens`.
    #[test]
    fn the_patch_ladder_is_its_own_ladder_and_is_refused_on_its_own_terms() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.out(y);

        let profile = DeviceProfile::default();
        let ceiling = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 0,
            buckets: Vec::new(),
            max_images: 4,
        });
        assert!(matches!(
            compile_axes(&b.trace, &ceiling, &profile),
            Err(Error::Budget { .. })
        ));

        let unsorted = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 32,
            buckets: vec![8, 8],
            max_images: 4,
        });
        assert!(matches!(
            compile_axes(&b.trace, &unsorted, &profile),
            Err(Error::Budget { .. })
        ));

        // 12 is under `max_tokens` and over `max_patches`, which is the whole
        // point of the ladders being two.
        let past = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 8,
            buckets: vec![4, 12],
            max_images: 4,
        });
        assert!(matches!(
            compile_axes(&b.trace, &past, &profile),
            Err(Error::Budget { .. })
        ));
    }

    /// The carve reserves a patch column at the PATCH ceiling and a token
    /// column at the token one — two symbols, two numbers, and neither
    /// derived from the other.
    #[test]
    fn a_patch_column_is_reserved_at_the_patch_ceiling() {
        use crate::arena::{Placement, RowExpr};

        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let y = b.op(tower, 8, Guard::Always);
        b.out(y);

        let compiled =
            compile_axes(&b.trace, &with_patches(), &DeviceProfile::default()).expect("bakes");
        let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[tower.0 as usize]
        else {
            panic!("the tower's output is a rectangle of the arena")
        };
        assert_eq!(*rows, RowExpr::Patches);
        // 32 patches x 8 elements x 2 bytes.
        assert_eq!(*bytes, 32 * 8 * 2);

        let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[y.0 as usize] else {
            panic!("the trunk's output is a rectangle of the arena")
        };
        assert_eq!(*rows, RowExpr::Tokens);
        assert_eq!(*bytes, 16 * 8 * 2);

        // And a patch rectangle never shares a column with a token one: the
        // co-tenancy rule demands equal `rows`, and these are two symbols.
        assert!(!compiled.arena.co_tenants(tower, y));
        assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
    }
}
