use model_compiler::{Budget, Budgets, DeviceProfile, VoxelLadder, compile_axes};
use model_exec::fire::{Fault, FireDescriptor, Lane, compose_axes};
use model_ir::ops::Elementwise;
use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Platform, RowAxis, RuntimeInput, Seam, Trace, Ty,
    ValueDecl, ValueId,
};

const WIDTH: u64 = 8;

fn act() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(WIDTH)],
        dtype: Dtype::Bf16,
    }
}

fn voxel(rows: Dim) -> Ty {
    Ty::Tensor {
        shape: vec![rows, Dim::Const(WIDTH)],
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
                name: "hand-built decoder".to_string(),
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

fn trunk_and_decoder() -> Trace {
    let mut b = Build::new();
    let tokens = b.value(Def::Input(RuntimeInput::Tokens), act());
    let port = b.value(
        Def::Input(RuntimeInput::Voxels {
            port: 0,
            channels: WIDTH as u32,
        }),
        voxel(Dim::Voxels),
    );
    let seeded = b.op(tokens, act(), Guard::Always);
    let y = b.op(seeded, act(), Guard::Always);
    b.trace.seams.push(Seam {
        seam: "out".to_string(),
        values: vec![y],
        layer: None,
    });
    let h = b.op(port, voxel(Dim::Voxels), Guard::Fact(0));
    let pixels = b.op(h, voxel(Dim::VoxelsTimes(4)), Guard::Fact(0));
    b.trace.seams.push(Seam {
        seam: "pixels".to_string(),
        values: vec![pixels],
        layer: None,
    });
    b.trace
}

fn tokens_budget() -> Budget {
    Budget::new(8, 64)
}

fn budgets() -> Budgets {
    Budgets::of(tokens_budget()).with_voxels(VoxelLadder {
        max_voxels: 512,
        buckets: vec![128, 256, 512],
        max_clips: 4,
    })
}

#[test]
fn the_voxel_axis_seriates_its_own_clips_every_case() {
    a_class_with_rows_and_no_clips_has_a_token_window_and_no_voxel_window();
    a_fire_past_the_voxel_ceilings_is_refused_by_name();
    abi_three_packs_a_voxel_trailer_exactly_when_the_fire_carries_clips();
}

fn a_class_with_rows_and_no_clips_has_a_token_window_and_no_voxel_window() {
    let trace = trunk_and_decoder();
    let budgets = budgets();
    let compiled = compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("bakes");
    assert_eq!(compiled.units, vec![RowAxis::Tokens, RowAxis::Voxels]);

    let lanes = [
        Lane::with_clips(1, 5, 2, 200),
        Lane::new(0, 3),
        Lane::with_clips(1, 4, 1, 64),
        Lane::new(0, 1),
    ];
    let fire = compose_axes(&compiled, &budgets, &lanes).expect("the mixed fire composes");
    assert_eq!(fire.rows(), 13);
    assert_eq!(fire.voxel_rows(), 264);
    assert_eq!(fire.clips(), 3);
    assert_eq!(
        fire.voxel_bucket(),
        512,
        "the voxel ladder rounds on its own rungs"
    );
    assert_eq!(fire.patch_rows(), 0, "no patch axis was stated");

    let plain = [
        Lane::new(1, 5),
        Lane::new(0, 3),
        Lane::new(1, 4),
        Lane::new(0, 1),
    ];
    let bare = compose_axes(&compiled, &budgets, &plain).expect("composes");
    assert_eq!(fire.classes(), bare.classes());
    assert_eq!(fire.bucket(), bare.bucket());

    let with_clips = compiled.classes.class_of(1).expect("word 1 is a class");
    let text_class = compiled.classes.class_of(0).expect("word 0 is a class");
    let tokens = fire.classes();
    let voxels = fire.voxel_classes();
    assert_eq!(tokens.class(with_clips).rows, 9);
    assert_eq!(tokens.class(text_class).rows, 4);
    assert_eq!(voxels.class(with_clips).rows, 264);
    assert_eq!(voxels.class(with_clips).lanes, 3, "clips, not lanes");
    assert_eq!(voxels.class(text_class).rows, 0);
    assert_eq!(voxels.class(text_class).lanes, 0);

    let placed: Vec<(u32, u32, u32, u32, u32)> = fire
        .lanes()
        .iter()
        .map(|row| {
            (
                row.source,
                row.voxel_offset,
                row.voxels,
                row.clip_offset,
                row.clips,
            )
        })
        .collect();
    assert!(placed.contains(&(0, 0, 200, 0, 2)));
    assert!(placed.contains(&(2, 200, 64, 2, 1)));
    assert!(placed.contains(&(1, 0, 0, 0, 0)));
}

fn a_fire_past_the_voxel_ceilings_is_refused_by_name() {
    let trace = trunk_and_decoder();
    let budgets = budgets();
    let compiled = compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("bakes");

    let rows = compose_axes(
        &compiled,
        &budgets,
        &[
            Lane::with_clips(1, 1, 1, 300),
            Lane::with_clips(1, 1, 1, 300),
        ],
    )
    .expect_err("600 voxel rows past a 512 ceiling");
    assert_eq!(
        rows,
        Fault::TooManyVoxels {
            voxels: 600,
            max: 512
        }
        .into()
    );
    assert!(rows.to_string().contains("every VAE column was cut at 512"));

    let clips = compose_axes(
        &compiled,
        &budgets,
        &[Lane::with_clips(1, 1, 3, 3), Lane::with_clips(1, 1, 3, 3)],
    )
    .expect_err("six clips past a ceiling of four");
    assert_eq!(clips, Fault::TooManyClips { clips: 6, max: 4 }.into());

    let short = Budgets::of(tokens_budget()).with_voxels(VoxelLadder {
        max_voxels: 512,
        buckets: vec![64],
        max_clips: 4,
    });
    let compiled = compile_axes(&trace, &short, &DeviceProfile::default()).expect("bakes");
    let rung = compose_axes(&compiled, &short, &[Lane::with_clips(1, 1, 1, 128)])
        .expect_err("128 voxel rows over a ladder that stops at 64");
    assert_eq!(
        rung,
        Fault::NoVoxelBucket {
            voxels: 128,
            top: 64
        }
        .into()
    );

    let geometry = compose_axes(&compiled, &short, &[Lane::with_clips(1, 1, 1, 0)])
        .expect_err("a clip is at least one voxel");
    assert_eq!(
        geometry,
        Fault::ClipGeometry {
            lane: 0,
            clips: 1,
            voxels: 0
        }
        .into()
    );
}

fn abi_three_packs_a_voxel_trailer_exactly_when_the_fire_carries_clips() {
    let trace = trunk_and_decoder();
    let budgets = budgets();
    let compiled = compile_axes(&trace, &budgets, &DeviceProfile::default()).expect("bakes");

    let bare =
        compose_axes(&compiled, &budgets, &[Lane::new(1, 2), Lane::new(0, 1)]).expect("composes");
    let bare = FireDescriptor::of(&bare);
    assert!(!bare.has_voxels());
    let bytes = bare.pack();
    assert_eq!(&bytes[4..8], &3u32.to_le_bytes(), "ABI 3");
    assert_eq!(bytes.len() as u64, bare.bytes());
    assert_eq!(FireDescriptor::unpack(&bytes), Ok(bare.clone()));
    assert_eq!(
        bytes.len() as u64,
        model_exec::fire::HEADER_BYTES
            + model_exec::fire::CLASS_BYTES * 2
            + model_exec::fire::LANE_BYTES * 2
    );

    let with = compose_axes(
        &compiled,
        &budgets,
        &[Lane::with_clips(1, 2, 2, 100), Lane::new(0, 1)],
    )
    .expect("composes");
    let with = FireDescriptor::of(&with);
    assert!(with.has_voxels());
    assert_eq!(with.voxel_rows, 100);
    assert_eq!(with.clips, 2);
    let bytes = with.pack();
    assert_eq!(
        bytes.len() as u64,
        bare.bytes() + model_exec::fire::CLASS_BYTES * 2 + model_exec::fire::VOXEL_LANE_BYTES * 2,
        "the voxel trailer: one window per class, one record per lane"
    );
    assert_eq!(
        &bytes[32..36],
        &100u32.to_le_bytes(),
        "voxel_rows at word 8"
    );
    let back = FireDescriptor::unpack(&bytes).expect("round trips");
    assert_eq!(back, with);
    assert_eq!(
        back.table(RowAxis::Voxels)
            .rows_of(&compiled.classes.node_mask[2]),
        100
    );

    let mut wrong = bytes;
    wrong[32..36].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        FireDescriptor::unpack(&wrong),
        Err(model_exec::Error::Fire(Fault::DescriptorVoxelRows {
            counted: 100,
            header: 99
        }))
    ));
}
