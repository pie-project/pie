//! The metal shell's half of the second seriation, with no device in the
//! room: which window a region's launches are cut at (`Windows::of`, pure
//! arithmetic), which rectangle a node reading across both axes is handed,
//! and how big the arena sizes a patch column. A tower region (capture unit
//! `RowAxis::Patches`) is cut at the patch table, not the token table; a
//! trunk region carries the patch interval beside its own since the embed
//! merge reads a patch rectangle from a token-unit node; a patch region gets
//! no rebased qo boundaries; a text-only fire gets zero patch windows; and
//! the arena sizes a patch rectangle at the composition's own patch rows
//! (`FireRows::text_only` used to silently compute zero rows instead of
//! faulting). No launch here — that's a device gate.

use engine_metal::window::{Copies, Windows};
use model_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, PatchLadder, RowAxis, compile_axes,
};
use model_exec::fire::{FireDescriptor, Lane, compose_axes};
use model_exec::store::arena::rect;
use model_ir::ops::Elementwise;
use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Platform, RuntimeInput, Seam, Trace, Ty, ValueDecl,
    ValueId,
};

const WIDTH: u64 = 8;

fn act() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(WIDTH)],
        dtype: Dtype::Bf16,
    }
}

fn patch() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Patches, Dim::Const(WIDTH)],
        dtype: Dtype::Bf16,
    }
}

struct Build {
    trace: Trace,
}

impl Build {
    fn new() -> Build {
        Build {
            trace: Trace {
                name: "hand-built tower".to_string(),
                platform: Platform::Metal,
                params: Vec::new(),
                caches: vec![CacheRow::State {
                    name: "state".to_string(),
                    slab: vec![1],
                    dtype: Dtype::Bf16,
                }],
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
                drafter: None,
            },
        }
    }

    fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

    /// A generic shaped op: the claim is about operand shapes, not computation.
    fn op(&mut self, x: ValueId, ty: Ty, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), ty);
        self.trace.nodes.push(Node {
            op: Elementwise::RmsnormNoScale {
                x,
                head_dim: 1,
                eps: 1e-6,
                y,
            }
            .into(),
            guard,
            layer: None,
        });
        y
    }
}

/// A tower, then a trunk that reads it. Two capture units, split on one fact
/// so that the class tables have something to say.
fn tower_and_trunk() -> Trace {
    let mut b = Build::new();
    let pixels = b.value(Def::Input(RuntimeInput::Patches), patch());
    let tokens = b.value(Def::Input(RuntimeInput::Tokens), act());

    let tower = b.op(pixels, patch(), Guard::Always);
    let deeper = b.op(tower, patch(), Guard::Always);
    // The embed merge: patch rows in, token rows out.
    let merged = b.op(deeper, act(), Guard::Always);
    let seeded = b.op(tokens, act(), Guard::Always);
    let d = b.op(merged, act(), Guard::Fact(0));
    let p = b.op(seeded, act(), Guard::not(Guard::Fact(0)));
    let o = b.value(
        Def::Merge(vec![(d, Guard::Fact(0)), (p, Guard::not(Guard::Fact(0)))]),
        act(),
    );
    let y = b.op(o, act(), Guard::Always);
    b.trace.seams.push(Seam {
        seam: "out".to_string(),
        values: vec![y],
        layer: None,
    });
    b.trace
}

fn budgets() -> Budgets {
    Budgets::of(Budget::new(8, 64)).with_patches(PatchLadder {
        max_patches: 256,
        buckets: vec![64, 128, 256],
        max_images: 4,
    })
}

fn baked() -> (Trace, CompiledModel) {
    let trace = tower_and_trunk();
    let compiled =
        compile_axes(&trace, &budgets(), &DeviceProfile::default()).expect("the tower bakes");
    assert_eq!(
        compiled.units,
        vec![RowAxis::Patches, RowAxis::Tokens],
        "the tower stands before the trunk that reads it",
    );
    (trace, compiled)
}

fn indptr(rows: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in rows {
        out.push(out[out.len() - 1] + n as i32);
    }
    out
}

/// Which regions are on which axis, as the shell reads it.
fn axis_of(compiled: &CompiledModel, region: usize) -> RowAxis {
    compiled.units[compiled.unit_of(region) as usize]
}

/// The tower is cut at the patch table and the trunk at the token one.
#[test]
fn each_region_is_cut_at_its_own_axis_s_window() {
    let (trace, compiled) = baked();
    let budgets = budgets();
    let lanes = [
        Lane::with_images(1, 5, 2, 128),
        Lane::new(0, 3),
        Lane::with_images(1, 4, 1, 64),
    ];
    let fire = compose_axes(&compiled, &budgets, &lanes).expect("the mixed fire composes");
    assert_eq!(fire.rows(), 12);
    assert_eq!(fire.patch_rows(), 192);
    assert_eq!(fire.images(), 3);

    let windows = Windows::of(
        &trace,
        &compiled,
        fire.classes(),
        fire.patch_classes(),
        &indptr(&[5, 3, 4]),
        Copies::off(),
        &[],
    &[],
    )
    .expect("every region seats a window");

    let mut towers = 0;
    let mut trunks = 0;
    // Does some token region see the whole patch rectangle (the embed merge's read)?
    let mut merge_saw_the_tower = false;
    for (at, region) in compiled.template().iter().enumerate() {
        let window = windows.at(at as u32, 0);
        match axis_of(&compiled, at) {
            RowAxis::Patches => {
                towers += 1;
                assert_eq!(
                    window.span.rows,
                    fire.patch_classes().rows_of(&region.mask),
                    "a tower region's launch runs over PATCH rows",
                );
                assert_eq!(
                    window.span.lanes,
                    fire.patch_classes().lanes_of(&region.mask),
                    "and its lane count is images",
                );
                assert_eq!(window.span, window.patch, "one axis, one window");
                // No rebased qo boundaries: the patch axis's bounds vector is
                // RuntimeInput::PatchSegments, which no window carries.
                assert!(
                    window.indptr_host.is_empty(),
                    "region {at} is a tower and was handed token qo boundaries",
                );
            }
            RowAxis::Tokens => {
                trunks += 1;
                assert_eq!(
                    window.span.rows,
                    fire.classes().rows_of(&region.mask),
                    "a trunk region's launch runs over token rows",
                );
                // The patch interval rides along, cut at this region's own classes.
                assert_eq!(
                    window.patch.rows,
                    fire.patch_classes().rows_of(&region.mask),
                );
                merge_saw_the_tower |=
                    window.patch.rows == fire.patch_rows() && window.span.rows == fire.rows();
            }
        }
    }
    assert!(towers > 0 && trunks > 0, "{towers} tower, {trunks} trunk");
    assert!(
        merge_saw_the_tower,
        "no token region was handed the whole patch rectangle, so the embed merge \
         would read somebody else's rows",
    );

    // The class that carries no image has token rows and no patch rows.
    let text_class = compiled.classes.class_of(0).expect("word 0 is a class");
    assert!(fire.classes().as_slice()[text_class].rows > 0);
    assert_eq!(fire.patch_classes().as_slice()[text_class].rows, 0);
}

/// A fire whose lanes carry no image gets the token windows it always had,
/// and a patch window of nothing: the tower's rectangles are all
/// `Dim::Patches`, so an axis-empty fire has zero of them.
#[test]
fn a_fire_with_no_image_gets_the_token_windows_it_always_had() {
    let (trace, compiled) = baked();
    let budgets = budgets();
    let words: [(u64, u32); 3] = [(1, 5), (0, 3), (1, 4)];

    let bare: Vec<Lane> = words.iter().map(|&(w, r)| Lane::new(w, r)).collect();
    let carried: Vec<Lane> = words
        .iter()
        .map(|&(w, r)| {
            if w == 1 {
                Lane::with_images(w, r, 1, 64)
            } else {
                Lane::new(w, r)
            }
        })
        .collect();

    let plain = compose_axes(&compiled, &budgets, &bare).expect("composes");
    let mixed = compose_axes(&compiled, &budgets, &carried).expect("composes");
    let boundaries = indptr(&[5, 3, 4]);

    let of = |fire: &model_exec::fire::Composition| {
        Windows::of(
            &trace,
            &compiled,
            fire.classes(),
            fire.patch_classes(),
            &boundaries,
            Copies::off(),
            &[],
        &[],
        )
        .expect("every region seats a window")
    };
    let without = of(&plain);
    let with = of(&mixed);

    for at in 0..compiled.template().len() {
        let a = without.at(at as u32, 0);
        let b = with.at(at as u32, 0);
        if axis_of(&compiled, at) == RowAxis::Tokens {
            assert_eq!(a.span, b.span, "region {at}'s token window moved");
            assert_eq!(a.indptr_host, b.indptr_host);
        }
        // The imageless fire's patch window is the zero window.
        assert_eq!(a.patch.rows, 0, "region {at} found patch rows in a text fire");
    }
    assert_eq!(plain.patch_rows(), 0);
    assert_eq!(mixed.patch_rows(), 128);
}

/// The arena sizes a patch column at the fire's patch rows. Previously
/// `crate::arena::carve` used `FireRows::text_only`, sizing every
/// `Dim::Patches` rectangle at zero rows without faulting.
#[test]
fn a_patch_rectangle_is_carved_at_the_compositions_own_patch_rows() {
    let (trace, compiled) = baked();
    let budgets = budgets();
    let fire = compose_axes(
        &compiled,
        &budgets,
        &[Lane::with_images(1, 5, 2, 128), Lane::new(0, 3)],
    )
    .expect("composes");

    // A patch-shaped value the arena carves: a tower op's own output, not the input.
    let pixels = trace
        .values
        .iter()
        .position(|decl| {
            matches!(decl.def, Def::Op(_))
                && matches!(&decl.ty, Ty::Tensor { shape, .. }
                    if shape.first() == Some(&Dim::Patches))
        })
        .map(|at| ValueId(at as u32))
        .expect("the hand-built tower computes patch rows");

    let honest = rect(
        &compiled.arena,
        pixels,
        model_compiler::FireRows {
            tokens: u64::from(fire.rows()),
            lanes: u64::from(fire.lane_count()),
            patches: u64::from(fire.patch_rows()),
            images: u64::from(fire.images()),
        },
    )
    .expect("the patch input is carved");
    assert_eq!(
        honest.rows,
        fire.patch_rows(),
        "the patch column is as tall as the fire's patch rows",
    );
    assert_eq!(honest.width as u64, WIDTH);

    // The failure the fix removes: the token-only reading answers a rectangle with no rows.
    let text_only = rect(
        &compiled.arena,
        pixels,
        model_compiler::FireRows::text_only(u64::from(fire.rows()), u64::from(fire.lane_count())),
    )
    .expect("it is carved either way — which is the point");
    assert_eq!(
        text_only.rows, 0,
        "a token-only carve sizes a tower's input at nothing and does not fault",
    );
}

/// The descriptor a device reads carries both tables, and the tower's window
/// survives the trip through the bytes.
#[test]
fn the_table_a_device_reads_carries_both_seriations() {
    let (_, compiled) = baked();
    let budgets = budgets();
    let fire = compose_axes(
        &compiled,
        &budgets,
        &[Lane::with_images(1, 5, 2, 128), Lane::new(0, 3)],
    )
    .expect("composes");

    let packed = FireDescriptor::unpack(&FireDescriptor::of(&fire).pack()).expect("unpacks");
    assert_eq!(packed.rows, fire.rows());
    assert_eq!(packed.patch_rows, fire.patch_rows());
    assert_eq!(packed.images, fire.images());
    assert_eq!(packed.patch_bucket, fire.patch_bucket());
    assert!(packed.has_patches());
    for region in compiled.template() {
        assert_eq!(
            packed.patch_rows_of(&region.mask),
            fire.patch_classes().rows_of(&region.mask),
            "the trip through the bytes changed a patch window",
        );
    }
}
