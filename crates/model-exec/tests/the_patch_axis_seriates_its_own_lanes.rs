//! **THE SECOND SERIATION**: what `compose_axes` answers about a fire that
//! carries images, and what it refuses (multimodal §5.1 and M-1e).
//!
//! # The finding this file exists to pin
//!
//! `compose`'s token pass is built on one invariant — ROWS AND LANES BREAK AT
//! THE SAME PLACES — and it is stated out loud in `WindowTable::spans_into`:
//! "a class with rows has lanes (a lane contributes at least one row) and a
//! class with lanes has rows, so the two prefix sums have their gaps at
//! exactly the same classes". That sentence is FALSE about patches. A lane of
//! a class may carry zero images or three, so a class can hold half the
//! fire's token rows and none of its patch rows; a patch window derived from
//! a token one would hand the tower somebody else's rectangle.
//!
//! So the patch axis is seriated on its own terms, in the same vocabulary:
//! its rows are patch rows, ITS LANES ARE IMAGES, its order is the artifact's
//! own patch `ClassOrder`, and its ladder is its own. What is asserted here:
//!
//! * the token half of a fire that carries images is BIT-IDENTICAL to the
//!   same fire without them — the gate (c) claim, at the composition;
//! * a class with token rows and no images gets a zero patch window while
//!   keeping its token one, which is the invariant break, present;
//! * images are the patch axis's lane count, prefix-summed like lanes are,
//!   and every lane record carries its place in BOTH orders;
//! * the patch ladder rounds on its own rungs;
//! * the three M-1e refusals fire BY NAME.

use model_compiler::{Budget, Budgets, DeviceProfile, PatchLadder, compile, compile_axes};
use model_exec::fire::{Fault, Lane, compose, compose_axes};
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

/// A tower and a trunk, split on one fact. Two classes, two capture units.
///
/// The tower stands FIRST, which is program order for a text that states a
/// vision encoder before the trunk that reads its output — and it is also the
/// only order `model_compiler::unit` admits, because a unit's capture regions
/// have to be one contiguous stretch of the record script.
fn tower_and_trunk() -> Trace {
    let mut b = Build::new();
    let pixels = b.input(RuntimeInput::Patches, patch());
    let tokens = b.input(RuntimeInput::Tokens, act());

    let tower = b.op(pixels, patch(), Guard::Always);
    let deeper = b.op(tower, patch(), Guard::Always);
    // The embed merge: reads patch rows, WRITES token rows. It belongs to the
    // trunk's unit, which is what `unit::node_axis` asking the OUTPUTS buys.
    let merged = b.op(deeper, act(), Guard::Always);
    let seeded = b.op(tokens, act(), Guard::Always);
    let d = b.op(merged, act(), Guard::Fact(0));
    let p = b.op(seeded, act(), Guard::not(Guard::Fact(0)));
    let o = b.merge(&[(d, Guard::Fact(0)), (p, Guard::not(Guard::Fact(0)))]);
    let y = b.op(o, act(), Guard::Always);
    b.out(y);
    b.trace
}

/// The same plan with NO patch row anywhere — a text-only artifact, which is
/// what refusal (ii) is about.
fn text_only() -> Trace {
    let mut b = Build::new();
    let tokens = b.input(RuntimeInput::Tokens, act());
    let seeded = b.op(tokens, act(), Guard::Always);
    let d = b.op(seeded, act(), Guard::Fact(0));
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

/// **THE INVARIANT BREAK, PRESENT AND MEASURED.**
///
/// One fire, two classes, and only one of them carries images. The token
/// windows say both classes have rows and lanes; the patch windows say one of
/// them has patch rows and images and the other has neither. Those two tables
/// are what a single merged prefix sum could not have produced.
#[test]
fn a_class_with_rows_and_no_images_has_a_token_window_and_no_patch_window() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled = compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    // Two lanes in class-of-word-1 carrying images, three in class-of-word-0
    // carrying none.
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

    // Both classes carry token rows and lanes.
    assert_eq!(tokens.class(with_images).rows, 9);
    assert_eq!(tokens.class(with_images).lanes, 2);
    assert_eq!(tokens.class(text_class).rows, 5);
    assert_eq!(tokens.class(text_class).lanes, 3);

    // Exactly one of them carries patch rows and images — the break.
    assert_eq!(patches.class(with_images).rows, 192);
    assert_eq!(patches.class(with_images).lanes, 3, "images, not lanes");
    assert_eq!(patches.class(text_class).rows, 0);
    assert_eq!(patches.class(text_class).lanes, 0);
    assert_eq!(
        fire.patch_present(),
        &[with_images as u32],
        "only the class with images is present on the patch axis",
    );
}

/// **GATE (c) AT THE COMPOSITION.** The token half of a fire with images in
/// it is the token half of the same fire without them, field for field. No
/// image moves a text lane's rows, its lane index, its class or its window.
#[test]
fn images_do_not_move_one_token_row() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    let words: [(u64, u32); 5] = [(1, 5), (0, 3), (1, 4), (0, 1), (0, 1)];
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

    let plain = compose_axes(&compiled, &budgets, &bare).expect("the text-only fire composes");
    let mixed = compose_axes(&compiled, &budgets, &carried).expect("the mixed fire composes");

    assert_eq!(plain.rows(), mixed.rows());
    assert_eq!(plain.bucket(), mixed.bucket());
    assert_eq!(plain.classes(), mixed.classes());
    assert_eq!(plain.present(), mixed.present());
    for (a, b) in plain.lanes().iter().zip(mixed.lanes()) {
        assert_eq!(a.source, b.source);
        assert_eq!(a.word, b.word);
        assert_eq!(a.class, b.class);
        assert_eq!(a.row_offset, b.row_offset);
        assert_eq!(a.rows, b.rows);
    }
    // And the text-only fire really is text-only, so the claim is not vacuous.
    assert_eq!(plain.patch_rows(), 0);
    assert_eq!(mixed.patch_rows(), 128);
}

/// Every lane record carries its place in BOTH orders, and the two prefix
/// sums are each other's independent: images accumulate over the lanes that
/// have them, in submission order inside a class, and skip the ones that do
/// not.
#[test]
fn a_lane_record_carries_its_place_in_both_seriations() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    let lanes = [
        Lane::with_images(1, 2, 1, 64),
        Lane::new(0, 3),
        Lane::with_images(1, 2, 2, 128),
    ];
    let fire = compose_axes(&compiled, &budgets, &lanes).expect("composes");

    let by_source = |source: u32| {
        *fire
            .lanes()
            .iter()
            .find(|row| row.source == source)
            .expect("every submitted lane is placed")
    };

    let first = by_source(0);
    let text = by_source(1);
    let second = by_source(2);

    assert_eq!((first.patch_offset, first.patches), (0, 64));
    assert_eq!((first.image_offset, first.images), (0, 1));
    assert_eq!((second.patch_offset, second.patches), (64, 128));
    assert_eq!((second.image_offset, second.images), (1, 2));
    // The text lane occupies no patch rows and no images, and its offsets are
    // the zero window rather than a stale neighbour's.
    assert_eq!((text.patch_offset, text.patches), (0, 0));
    assert_eq!((text.image_offset, text.images), (0, 0));
    // The patch offsets partition the patch rectangle exactly.
    assert_eq!(
        first.patches + second.patches,
        fire.patch_rows(),
        "the lane runs tile the patch window with no gap and no overlap",
    );
}

/// The patch ladder rounds on ITS OWN rungs, and a fire with no image rounds
/// to nothing at all — which is what "an axis-empty fire does not launch that
/// unit's exec" is, arithmetically.
#[test]
fn the_patch_rung_is_the_patch_ladders_and_an_axis_empty_fire_rounds_to_zero() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    let one = compose_axes(&compiled, &budgets, &[Lane::with_images(1, 2, 1, 64)]).expect("composes");
    assert_eq!(one.patch_rows(), 64);
    assert_eq!(one.patch_bucket(), 64, "64 is a rung of the patch ladder");
    // The token bucket is the token ladder's, and this budget lists none — so
    // it is the row count itself. Two ladders, two answers, neither derived
    // from the other.
    assert_eq!(one.bucket(), 2);

    let two = compose_axes(&compiled, &budgets, &[Lane::with_images(1, 2, 2, 65)]).expect("composes");
    assert_eq!(two.patch_bucket(), 128, "65 rounds up to the next rung");

    let none = compose_axes(&compiled, &budgets, &[Lane::new(0, 4)]).expect("composes");
    assert_eq!(none.patch_rows(), 0);
    assert_eq!(none.patch_bucket(), 0, "no patch rows, no tower exec");
}

/// **REFUSAL (i), HOST HALF**: a media submission whose geometry disagrees
/// with its patch payload.
#[test]
fn a_geometry_that_disagrees_with_its_payload_is_refused_by_name() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    // Images declared, no patch rows shipped.
    let hollow = compose_axes(&compiled, &budgets, &[Lane::with_images(1, 2, 2, 0)])
        .expect_err("two images of no patch rows is not a submission");
    assert_eq!(
        hollow,
        Fault::PatchGeometry {
            lane: 0,
            images: 2,
            patches: 0,
        }
        .into(),
    );
    assert!(hollow.to_string().contains("geometry and its payload disagree"));

    // Patch rows shipped, no image declared — the same disagreement, the
    // other way, and the one that would leave `attention.dense` with rows and
    // no segment to put them in.
    let orphan = compose_axes(&compiled, &budgets, &[Lane::with_images(1, 2, 0, 64)])
        .expect_err("patch rows belonging to no image are not a submission");
    assert_eq!(
        orphan,
        Fault::PatchGeometry {
            lane: 0,
            images: 0,
            patches: 64,
        }
        .into(),
    );
}

/// **REFUSAL (ii)**: patches against a text with no patch axis.
#[test]
fn images_against_a_text_with_no_tower_are_refused_by_name() {
    let trace = text_only();
    let profile = DeviceProfile::default();

    // Through the one-axis door, which admits no patch ladder at all.
    let plain = compile(&trace, &tokens_budget(), &profile).expect("a text-only plan bakes");
    let refusal = compose(&plain, &tokens_budget(), &[Lane::with_images(0, 2, 1, 64)])
        .expect_err("a text has no tower");
    assert_eq!(refusal, Fault::Towerless { lane: 0 }.into());
    assert!(refusal.to_string().contains("declares no patch axis"));

    // And through the two-axis door with a ladder declared: the DEPLOYMENT
    // admitting a patch axis does not give this artifact one, because the
    // axis is a property of the plan and the artifact is bit-identical either
    // way (the G4 invariant).
    let budgets = budgets();
    let admitted = compile_axes(&trace, &budgets, &profile).expect("bakes");
    assert_eq!(admitted.patches, None);
    assert_eq!(
        compose_axes(&admitted, &budgets, &[Lane::with_images(0, 2, 1, 64)])
            .expect_err("a ladder is not a tower"),
        Fault::Towerless { lane: 0 }.into(),
    );
}

/// **REFUSAL (iii)**: over-ceiling patch counts, on the patch ladder's own
/// terms — three of them, because a ladder has three ways to be exceeded.
#[test]
fn a_fire_past_the_patch_ceilings_is_refused_by_name() {
    let trace = tower_and_trunk();
    let budgets = budgets();
    let compiled =
        compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("the tower bakes");

    // Past `max_patches`, at a token count the token ceiling admits happily —
    // which is the whole point of the ladders being two.
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
    assert!(rows.to_string().contains("every tower column was cut at 256"));

    // Past `max_images`, at a patch count the patch ceiling admits.
    let images = compose_axes(
        &compiled,
        &budgets,
        &[
            Lane::with_images(1, 1, 3, 3),
            Lane::with_images(1, 1, 3, 3),
        ],
    )
    .expect_err("six images past a ceiling of four");
    assert_eq!(images, Fault::TooManyImages { images: 6, max: 4 }.into());

    // Past the top RUNG, under the ceiling: a fire with no exec to launch.
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
