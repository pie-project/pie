//! **G4, ON THE REAL CATALOG.** Every SKU that existed before the second row
//! axis did still bakes to exactly ONE capture unit, and to an artifact the
//! axis cannot be shown to have touched.
//!
//! # What "bit-identical" is asked as, and why that is the strongest form
//!
//! A test cannot compare against a compiler that is no longer in the tree. So
//! the invariant is asked in the one form that IS decidable here and is
//! strictly stronger than a stored golden would be:
//!
//! - **the whole artifact is a function of the axes the PLAN states, not of
//!   the axes the DEPLOYMENT admits.** Bake each SKU twice — once through
//!   [`compile`], the pre-campaign door, and once through [`compile_axes`]
//!   against budgets that admit a full patch ladder — and the two
//!   `CompiledModel`s must be `==`. Every field is in that comparison: the
//!   class table, the region table, both class orders, both fallback tables,
//!   the arena's placements and spans and byte count, the concurrency
//!   relation, the stream plan, the unit table. A single byte moved by the
//!   axis existing shows up here.
//! - **and the answer is one unit, on the primary axis, with the fold armed.**
//!   `units == [Tokens]`, every region on unit 0, `fold_refused == false`,
//!   `patches == None`. That is the M1 claim `graph_replay.rs`'s
//!   one-exec-per-fire assumption is currently allowed to rest on, and the
//!   line that has to be amended the day a two-unit SKU lands.
//!
//! A golden file would pin the artifact of ONE compiler build against ONE
//! recorded past. This pins it against the alternative that actually threatens
//! it — a compiler that has grown a second axis — on every SKU and every
//! platform, and it keeps doing so after the numbers legitimately move.
//!
//! # And the second axis is not vacuous
//!
//! The last test bakes a hand-built two-axis plan through the same door, so a
//! green run of the file above cannot be "the patch path is never reached".
//!
//! SILENT ON PURPOSE, like its catalog siblings: the numbers ride in the
//! assert messages.

use model_compiler::{
    Budget, Budgets, DeviceProfile, PatchLadder, Phase, RowAxis, compile, compile_axes,
};
use model_dsl::Platform;
use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Operation, Param, ParamSource, Seam, Trace, Ty,
    ValueDecl, ValueId,
};

/// Every platform a plan can be traced at — a model text may emit a different
/// op per platform, so the region table is not the same graph on each.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// A budget the catalog can actually seat: each plan asked for its own bank
/// capacity, as the two live catalog files already do.
fn budgets_for(trace: &Trace) -> Budget {
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

/// The same deployment, having also declared a patch axis it will never use on
/// a text-only model. Rungs at whole images from the floor
/// (`PATCH_LATTICE_FLOOR`) up, which is what a tower-serving deployment states.
fn also_admitting_patches(trace: &Trace) -> Budgets {
    Budgets::of(budgets_for(trace)).with_patches(PatchLadder {
        max_patches: 4096,
        buckets: vec![64, 128, 256, 512, 1024, 2048, 4096],
        max_images: 16,
    })
}

/// G4, first half: one capture unit, and the fold is not asked to stand down.
#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform; minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn every_pre_campaign_sku_is_exactly_one_capture_unit() {
    let mut wrong: Vec<String> = Vec::new();
    let mut baked = 0usize;

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default())
            else {
                continue; // `every_sku_carves_an_arena` is what says so.
            };
            baked += 1;

            if compiled.units != vec![RowAxis::Tokens] {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} capture units {:?} — a text-only \
                     SKU states one row space and therefore one exec",
                    compiled.units.len(),
                    compiled.units,
                ));
            }
            if compiled.units_of.len() != compiled.regions.len() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} unit stamps for {} regions",
                    compiled.units_of.len(),
                    compiled.regions.len(),
                ));
            }
            if let Some(stray) = compiled.units_of.iter().position(|&unit| unit != 0) {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: region {stray} landed on unit {} with \
                     only one unit declared",
                    compiled.units_of[stray],
                ));
            }
            if compiled.fold_refused {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the fold stood down on a one-unit \
                     artifact — `fold_refused` is the multi-unit escape hatch and \
                     nothing else",
                ));
            }
            if compiled.patches.is_some() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: a patch-axis plan on a text-only SKU",
                ));
            }
            // The whole capture half is one exec, front to back.
            let capture = compiled
                .regions
                .iter()
                .filter(|region| region.phase == Phase::Capture)
                .count();
            match compiled.unit_script(0) {
                Some(script) if script.end - script.start == capture as u32 => {}
                script => wrong.push(format!(
                    "`{sku}` as {platform:?}: unit 0's script is {script:?} over \
                     {capture} capture regions",
                )),
            }
        }
    }

    assert!(baked > 0, "the sweep baked nothing, so it asserted nothing");
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// G4, second half: the artifact is a function of the axes the PLAN states.
///
/// The strong form. A deployment that declares a patch ladder for a model with
/// no tower gets the artifact it always got — every field, not a summary of
/// them — which is what makes the second axis free rather than cheap.
#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform TWICE; minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn admitting_a_patch_axis_does_not_move_one_byte_of_a_text_only_artifact() {
    let mut moved: Vec<String> = Vec::new();
    let mut compared = 0usize;

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let profile = DeviceProfile::default();
            let Ok(before) = compile(&trace, &budgets_for(&trace), &profile) else {
                continue;
            };
            let after = match compile_axes(&trace, &also_admitting_patches(&trace), &profile) {
                Ok(after) => after,
                Err(refusal) => {
                    moved.push(format!(
                        "`{sku}` as {platform:?}: admitting a patch axis refused a \
                         load that baked without one: {}",
                        refusal.say(&trace),
                    ));
                    continue;
                }
            };
            compared += 1;

            if before == after {
                continue;
            }
            // Name the first field that differs, because "the artifacts differ"
            // is a failure a reader cannot act on.
            let mut which: Vec<&str> = Vec::new();
            if before.classes != after.classes {
                which.push("classes");
            }
            if before.regions != after.regions {
                which.push("regions");
            }
            if before.order != after.order {
                which.push("order");
            }
            if before.fallback != after.fallback {
                which.push("fallback");
            }
            if before.arena != after.arena {
                which.push("arena");
            }
            if before.concurrency != after.concurrency {
                which.push("concurrency");
            }
            if before.streams != after.streams {
                which.push("streams");
            }
            if before.units != after.units || before.units_of != after.units_of {
                which.push("units");
            }
            if before.patches != after.patches {
                which.push("patches");
            }
            if before.fold_refused != after.fold_refused {
                which.push("fold_refused");
            }
            moved.push(format!(
                "`{sku}` as {platform:?}: admitting a patch axis moved {} — \
                 arena {} -> {} bytes",
                which.join(", "),
                before.arena.bytes,
                after.arena.bytes,
            ));
        }
    }

    assert!(compared > 0, "the sweep compared nothing");
    assert!(moved.is_empty(), "\n{}\n", moved.join("\n"));
}

/// The one thing the two sweeps above cannot say: that the patch path exists.
///
/// A hand-built tower — patch-shaped rows, then a token trunk that reads them
/// — through the same door, so a green file is never "the second axis is
/// unreachable".
#[test]
fn a_plan_that_states_patch_rows_bakes_two_units_and_stands_the_fold_down() {
    let trace = tower_and_trunk();
    let budgets = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
        max_patches: 64,
        buckets: vec![64],
        max_images: 1,
    });

    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");
    assert_eq!(compiled.units, vec![RowAxis::Patches, RowAxis::Tokens]);
    assert!(compiled.fold_refused);
    assert!(compiled.patches.is_some());
    assert!(compiled.order_for(RowAxis::Patches).is_some());
    assert!(compiled.unit_script(0).is_some());
    assert!(compiled.unit_script(1).is_some());
    assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());

    // And the same plan against budgets that size no patch ceiling is a
    // refusal with the axis in it, not a tower carved at zero rows.
    let refusal = compile(&trace, &Budget::new(4, 16), &DeviceProfile::default())
        .expect_err("no patch ceiling, no load");
    assert!(refusal.to_string().contains("patches"), "{refusal}");
}

/// Two patch-shaped ops, then two token-shaped ones — the tower/trunk shape in
/// four nodes, stated in `Def` and `Ty` because the authoring surface has no
/// tower vocabulary yet (that is M3).
fn tower_and_trunk() -> Trace {
    let mut values: Vec<ValueDecl> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();

    let push = |values: &mut Vec<ValueDecl>, def, ty| {
        values.push(ValueDecl { def, ty });
        ValueId((values.len() - 1) as u32)
    };
    let patch = |width: u64| Ty::Tensor {
        shape: vec![Dim::Patches, Dim::Const(width)],
        dtype: Dtype::Bf16,
    };
    let token = |width: u64| Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(width)],
        dtype: Dtype::Bf16,
    };

    let pixels = push(
        &mut values,
        Def::Input(model_ir::RuntimeInput::Patches),
        patch(8),
    );
    let mut chain = pixels;
    for (at, ty) in [patch(8), patch(8), token(8), token(8)].into_iter().enumerate() {
        let y = push(&mut values, Def::Op(at as u32), ty);
        nodes.push(Node {
            op: Operation::Elementwise(model_ir::Elementwise::RmsnormNoScale {
                x: chain,
                head_dim: 1,
                eps: 1e-6,
                y,
            }),
            guard: Guard::Always,
            layer: None,
        });
        chain = y;
    }

    Trace {
        name: "tower-and-trunk".to_string(),
        platform: model_ir::Platform::Cuda,
        params: Vec::<Param>::new(),
        caches: vec![CacheRow::State {
            name: "state".to_string(),
            slab: vec![1],
        }],
        values,
        nodes,
        seams: vec![Seam {
            seam: "out".to_string(),
            values: vec![chain],
            layer: None,
        }],
    }
}
