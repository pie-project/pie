//! **THE SHELL'S HALF OF THE SECOND SERIATION**, with no device in the room:
//! which window a region's launches are cut at, and which rectangle a node
//! that reads ACROSS the two axes is handed.
//!
//! # What this pins, and why it needs no GPU
//!
//! `Windows::of` is arithmetic — a compiled artifact, a fire's two class
//! tables and a boundary vector in, one window per region per run out. Every
//! claim below is about that arithmetic, so it runs on any box:
//!
//! * a TOWER region — one whose capture unit is `RowAxis::Patches` — is cut at
//!   the PATCH table: its rows are patch rows and its lanes are IMAGES. A
//!   shell that cut it at the token table would hand the tower one class's
//!   token interval of somebody else's rows (multimodal §5.1);
//! * a TRUNK region carries the patch interval BESIDE its own, because the
//!   embed merge (`layout.scatter_rows`) is a token-unit node that reads a
//!   patch rectangle — the one node in a tower plan that touches both axes;
//! * a text-only fire of the same artifact gets zero patch windows and its
//!   token windows are the ones it always had, which is gate (a) at the
//!   window table;
//! * and the descriptor round-trips both tables, so the answer a device reads
//!   is the answer the walk reads.
//!
//! What is NOT here is a launch. `attention.dense` firing on real patch rows
//! is the text wave's gate, and it needs a model text that declares a tower.

use engine_cuda::window::{Copies, Windows};
use model_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, PatchLadder, RowAxis, compile_axes,
};
use model_exec::fire::{FireDescriptor, Lane, compose_axes};
use model_ir::ops::Elementwise;

/// A slot table generously above every hand-built fire below — the tests ask
/// about window semantics, not the carve, so the ceiling only has to hold.
/// The last three are what one GATHERED payload is bounded by (rows, kv
/// spaces, pages), which `Slots` owns since the tail acquired a stride.
fn test_slots() -> engine_cuda::window::Slots {
    engine_cuda::window::Slots::new(8, 512, 8, 1, 4096, 4, 4096)
}
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
                platform: Platform::Cuda,
                params: Vec::new(),
                caches: vec![CacheRow::State {
                    name: "state".to_string(),
                    slab: vec![1],
                    dtype: Dtype::Bf16,
                }],
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
            },
        }
    }

    fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

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

/// **THE TOWER IS CUT AT THE PATCH TABLE AND THE TRUNK AT THE TOKEN ONE.**
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
        model_ir::PerAxis::new([fire.classes(), fire.patch_classes()]),
        &indptr(&[5, 3, 4]),
        Copies::off(),
        test_slots(),
    )
    .expect("every region seats a window");

    let mut towers = 0;
    let mut trunks = 0;
    // Does some TOKEN region see the whole patch rectangle? The embed merge
    // is `Guard::Always`, so its mask holds every class and its patch window
    // is the fire's whole one — which is the rectangle `layout.scatter_rows`
    // reads, and the thing a shell carrying only one window pair could not
    // have handed it.
    let mut merge_saw_the_tower = false;
    for (at, region) in compiled.template().iter().enumerate() {
        let window = windows.at(at as u32, 0);
        match axis_of(&compiled, at) {
            RowAxis::Patches => {
                towers += 1;
                assert_eq!(
                    window.span().rows,
                    fire.patch_classes().rows_of(&region.mask),
                    "a tower region's launch runs over PATCH rows",
                );
                assert_eq!(
                    window.span().lanes,
                    fire.patch_classes().lanes_of(&region.mask),
                    "and its lane count is images",
                );
                assert_eq!(window.span(), window.on(RowAxis::Patches), "one axis, one window");
            }
            RowAxis::Tokens => {
                trunks += 1;
                assert_eq!(
                    window.span().rows,
                    fire.classes().rows_of(&region.mask),
                    "a trunk region's launch runs over token rows",
                );
                // The patch interval rides along, cut at THIS region's own
                // classes — a class with token rows and no image contributes
                // none, which is the invariant break.
                assert_eq!(
                    window.on(RowAxis::Patches).rows,
                    fire.patch_classes().rows_of(&region.mask),
                );
                merge_saw_the_tower |= window.on(RowAxis::Patches).rows == fire.patch_rows()
                    && window.span().rows == fire.rows();
            }
        }
    }
    assert!(towers > 0 && trunks > 0, "{towers} tower, {trunks} trunk");
    assert!(
        merge_saw_the_tower,
        "no token region was handed the whole patch rectangle, so the embed merge \
         would read somebody else's rows",
    );

    // AND THE BREAK ITSELF, at the window table: the class that carries no
    // image has token rows and no patch rows in the same fire.
    let text_class = compiled.classes.class_of(0).expect("word 0 is a class");
    assert!(fire.classes().as_slice()[text_class].rows > 0);
    assert_eq!(fire.patch_classes().as_slice()[text_class].rows, 0);
}

/// **GATE (a), AT THE WINDOW TABLE.** A fire of the same artifact whose lanes
/// carry no image gets the token windows it would have had before the axis
/// existed, and a patch window of nothing.
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
            model_ir::PerAxis::new([fire.classes(), fire.patch_classes()]),
            &boundaries,
            Copies::off(),
            test_slots(),
        )
        .expect("every region seats a window")
    };
    let without = of(&plain);
    let with = of(&mixed);

    for at in 0..compiled.template().len() {
        let a = without.at(at as u32, 0);
        let b = with.at(at as u32, 0);
        if axis_of(&compiled, at) == RowAxis::Tokens {
            assert_eq!(a.span(), b.span(), "region {at}'s token window moved");
            assert_eq!(a.indptr_host, b.indptr_host);
        }
        // And the imageless fire's patch window is the zero window, which is
        // what an axis-empty fire has: the tower's launches read zero rows and
        // return, so the unit's exec runs nothing.
        assert_eq!(a.on(RowAxis::Patches).rows, 0, "region {at} found patch rows in a text fire");
    }
    assert_eq!(plain.patch_rows(), 0);
    assert_eq!(mixed.patch_rows(), 128);
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
