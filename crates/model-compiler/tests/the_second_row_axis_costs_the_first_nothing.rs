//! Every SKU that existed before the second row axis still bakes to exactly
//! one capture unit, and to an artifact the axis cannot be shown to have
//! touched: each SKU is baked through both [`compile`] and [`compile_axes`]
//! (with a full patch ladder admitted) and the two `CompiledModel`s must be
//! `==`, field for field. The last test bakes a hand-built two-axis plan
//! through the same door, so a green run above can't mean the patch path is
//! unreachable.

use model_compiler::{
    Budget, Budgets, DeviceProfile, PatchLadder, RowAxis, compile, compile_axes,
};
mod common;
use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Operation, Param, Seam, Trace, Ty,
    ValueDecl, ValueId,
};

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
            dtype: Dtype::Bf16,
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
