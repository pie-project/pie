use model_compiler::{Budget, Budgets, DeviceProfile, PatchLadder, compile_axes};
use model_exec::fire::{Fault, Lane, compose_axes};
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
                drafter: None,
            },
        }
    }

    fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

    fn input(&mut self, which: RuntimeInput, ty: Ty) -> ValueId {
        self.value(Def::Input(which), ty)
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

    fn merge(&mut self, arms: &[(ValueId, Guard)]) -> ValueId {
        self.value(Def::Merge(arms.to_vec()), act())
    }

    fn out(&mut self, v: ValueId) {
        self.trace.seams.push(Seam {
            seam: "out".to_string(),
            values: vec![v],
            layer: None,
        });
    }
}

fn tower_and_trunk() -> Trace {
    let mut b = Build::new();
    let pixels = b.input(RuntimeInput::Patches, patch());
    let tokens = b.input(RuntimeInput::Tokens, act());

    let tower = b.op(pixels, patch(), Guard::Always);
    let deeper = b.op(tower, patch(), Guard::Always);
    let merged = b.op(deeper, act(), Guard::Always);
    let seeded = b.op(tokens, act(), Guard::Always);
    let d = b.op(merged, act(), Guard::Fact(0));
    let p = b.op(seeded, act(), Guard::not(Guard::Fact(0)));
    let o = b.merge(&[(d, Guard::Fact(0)), (p, Guard::not(Guard::Fact(0)))]);
    let y = b.op(o, act(), Guard::Always);
    b.out(y);
    b.trace
}

fn tokens_budget() -> Budget {
    Budget::new(8, 64)
}

fn budgets() -> Budgets {
    Budgets::of(tokens_budget()).with_patches(PatchLadder {
        max_patches: 256,
        buckets: vec![64, 128, 256],
        max_images: 4,
    })
}

#[test]
fn the_patch_axis_seriates_its_own_lanes_every_case() {
    a_class_with_rows_and_no_images_has_a_token_window_and_no_patch_window();
    a_fire_past_the_patch_ceilings_is_refused_by_name();
}

fn a_class_with_rows_and_no_images_has_a_token_window_and_no_patch_window() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    let lanes = [
        Lane::with_images(1, 5, 2, 128),
        Lane::new(0, 3),
        Lane::with_images(1, 4, 1, 64),
        Lane::new(0, 1),
        Lane::new(0, 1),
    ];
    let fire = compose_axes(&compiled, &budgets, &lanes).expect("the mixed fire composes");

    assert_eq!(fire.rows(), 14);
    assert_eq!(fire.patch_rows(), 192);
    assert_eq!(fire.images(), 3);

    let with_images = compiled.classes.class_of(1).expect("word 1 is a class");
    let text_class = compiled.classes.class_of(0).expect("word 0 is a class");
    assert_ne!(with_images, text_class);

    let tokens = fire.classes();
    let patches = fire.patch_classes();

    assert_eq!(tokens.class(with_images).rows, 9);
    assert_eq!(tokens.class(with_images).lanes, 2);
    assert_eq!(tokens.class(text_class).rows, 5);
    assert_eq!(tokens.class(text_class).lanes, 3);

    assert_eq!(patches.class(with_images).rows, 192);
    assert_eq!(patches.class(with_images).lanes, 3, "images, not lanes");
    assert_eq!(patches.class(text_class).rows, 0);
    assert_eq!(patches.class(text_class).lanes, 0);
    assert_eq!(
        fire.patch_classes()
            .present_in_order()
            .collect::<Vec<u32>>(),
        vec![with_images as u32],
        "only the class with images is present on the patch axis",
    );
}

fn a_fire_past_the_patch_ceilings_is_refused_by_name() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    let rows = compose_axes(
        &compiled,
        &budgets,
        &[
            Lane::with_images(1, 1, 2, 200),
            Lane::with_images(1, 1, 2, 200),
        ],
    )
    .expect_err("400 patch rows past a 256 ceiling");
    assert_eq!(
        rows,
        Fault::TooManyPatches {
            patches: 400,
            max: 256,
        }
        .into(),
    );
    assert!(
        rows.to_string()
            .contains("every tower column was cut at 256")
    );

    let images = compose_axes(
        &compiled,
        &budgets,
        &[Lane::with_images(1, 1, 3, 3), Lane::with_images(1, 1, 3, 3)],
    )
    .expect_err("six images past a ceiling of four");
    assert_eq!(images, Fault::TooManyImages { images: 6, max: 4 }.into());

    let short = Budgets::of(tokens_budget()).with_patches(PatchLadder {
        max_patches: 256,
        buckets: vec![64],
        max_images: 4,
    });
    let compiled = compile_axes(&trace, &short, &DeviceProfile::default()).expect("bakes");
    let rung = compose_axes(&compiled, &short, &[Lane::with_images(1, 1, 2, 128)])
        .expect_err("128 patch rows over a ladder that stops at 64");
    assert_eq!(
        rung,
        Fault::NoPatchBucket {
            patches: 128,
            top: 64,
        }
        .into(),
    );
    assert!(rung.to_string().contains("no tower exec to launch it in"));
}
