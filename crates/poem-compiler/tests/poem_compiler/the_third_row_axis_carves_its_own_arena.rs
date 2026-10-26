use poem::ops::conv::{self, ConvShape};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Weight, ops, trace_hybrid};
use poem_compiler::{
    Budget, Budgets, DeviceProfile, Error, Placement, RowAxis, RowExpr, VoxelLadder, compile,
    compile_axes,
};
use poem_ir::{Operands, Trace};

const C: u32 = 8;

fn conv(name: &str, c_in: u32, c_out: u32) -> Weight {
    Weight::sym(name, [u64::from(c_out), u64::from(c_in) * 27], Dtype::Bf16)
        .conv_taps_major(c_in, 27)
}

fn plane(name: &str, c: u32) -> Weight {
    Weight::sym(name, [u64::from(c)], Dtype::F32)
}

struct Encoder;

impl ForwardHybrid for Encoder {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) {
        let g = inputs.grid();
        let x = inputs.input("voxels", Dtype::Bf16, poem_ir::InputAxis::Voxel, C, None);
        let (h, g1) = conv::conv3d(&x, &g, &conv("conv", C, C), None, ConvShape::same3(), None);
        let h = conv::group_norm(
            &h,
            &g1,
            2,
            &plane("norm.w", C),
            &plane("norm.b", C),
            1e-6,
            true,
        );
        let tg = inputs.token_grid([1, 2, 2]);
        let tokens = conv::patchify(&h, &g1, [1, 2, 2], &tg);
        let w = Weight::sym("proj", [64, u64::from(C) * 4], Dtype::Bf16);
        let logits = ops::linear::matmul(&tokens, &w);
        poem::output("logits", &logits, None);
    }
}

struct Decoder;

impl ForwardHybrid for Decoder {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) {
        let latents = inputs.input("latents", Dtype::Bf16, poem_ir::InputAxis::Row, C * 4, None);
        let w = Weight::sym("proj", [u64::from(C) * 4, u64::from(C) * 4], Dtype::Bf16);
        let tokens = ops::linear::matmul(&latents, &w);
        let g = inputs.grid();
        let tg = inputs.token_grid([1, 2, 2]);
        let x = conv::unpatchify(&tokens, &tg, [1, 2, 2], &g);
        let (h, g1) = conv::conv3d(&x, &g, &conv("conv1", C, C), None, ConvShape::same3(), None);
        let (h, g2) = conv::upsample_nearest(&h, &g1, [1, 2, 2], false);
        let (h, g3) = conv::conv3d(
            &h,
            &g2,
            &conv("conv2", C, 12),
            None,
            ConvShape::same3(),
            None,
        );
        let (y, g4) = conv::pixel_shuffle(&h, &g3, [1, 2, 2]);
        poem::output("pixels", &y, Some(&g4));
    }
}

fn budgets() -> Budgets {
    Budgets::of(Budget::new(4, 64)).with_voxels(VoxelLadder {
        max_voxels: 256,
        buckets: vec![64, 256],
        max_clips: 4,
    })
}

fn unit_axes(trace: &Trace, compiled: &poem_compiler::CompiledModel) -> Vec<(String, RowAxis)> {
    compiled
        .template()
        .iter()
        .enumerate()
        .filter(|(_, region)| region.phase == poem_compiler::Phase::Capture)
        .flat_map(|(at, region)| {
            let axis = compiled.axis_of(at);
            region
                .nodes
                .clone()
                .map(move |node| (trace.nodes[node as usize].op.name().to_string(), axis))
        })
        .collect()
}

#[test]
fn the_third_row_axis_carves_its_own_arena_every_case() {
    an_encoder_runs_its_voxel_unit_first_and_the_patchify_opens_the_token_one();
    a_decoder_carves_its_pixels_at_sixteen_voxel_ceilings_and_shares_no_column();
    a_voxel_plan_against_no_voxel_ceiling_is_refused_by_name();
}

fn an_encoder_runs_its_voxel_unit_first_and_the_patchify_opens_the_token_one() {
    let trace = trace_hybrid("encoder", &Encoder, Platform::Cuda);
    let compiled = compile_axes(&trace, &budgets(), &DeviceProfile::default()).expect("bakes");
    assert_eq!(compiled.units, vec![RowAxis::Voxels, RowAxis::Tokens]);
    assert!(compiled.fold_refused, "two units, no single graph to fold");
    assert!(
        compiled.voxels.is_some(),
        "the voxel axis has its own seriation"
    );
    assert!(compiled.order_for(RowAxis::Voxels).is_some());
    assert!(compiled.patches.is_none(), "and no patch axis was stated");

    let placed = unit_axes(&trace, &compiled);
    let boundary = placed
        .iter()
        .position(|(name, _)| name == "conv.patchify")
        .expect("the patchify is a capture node");
    assert!(
        placed[..boundary]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Voxels),
        "everything before the patchify is the voxel unit's: {placed:?}"
    );
    assert!(
        placed[boundary..]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Tokens),
        "the patchify writes token rows and opens the token unit: {placed:?}"
    );
    assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
}

fn a_decoder_carves_its_pixels_at_sixteen_voxel_ceilings_and_shares_no_column() {
    let trace = trace_hybrid("decoder", &Decoder, Platform::Cuda);
    let compiled = compile_axes(&trace, &budgets(), &DeviceProfile::default()).expect("bakes");
    assert_eq!(compiled.units, vec![RowAxis::Tokens, RowAxis::Voxels]);

    let placed = unit_axes(&trace, &compiled);
    let boundary = placed
        .iter()
        .position(|(name, _)| name == "conv.unpatchify")
        .expect("the unpatchify is a capture node");
    assert!(
        placed[..boundary]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Tokens)
    );
    assert!(
        placed[boundary..]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Voxels),
        "the unpatchify writes voxel rows and opens the voxel unit: {placed:?}"
    );

    let pixels = trace
        .outputs
        .iter()
        .find(|s| s.name == "pixels")
        .expect("planted");
    let (plane, grid) = (
        pixels.value,
        pixels.grid.expect("the pixels name their grid"),
    );
    let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[plane.0 as usize] else {
        panic!("the pixel plane is a rectangle of the arena")
    };
    assert_eq!(*rows, RowExpr::VoxelsTimes(16));
    assert_eq!(*bytes, 16 * 256 * 3 * 2);
    let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[grid.0 as usize] else {
        panic!("the grid is a rectangle of the arena")
    };
    assert_eq!(*rows, RowExpr::Clips);
    assert_eq!(*bytes, 4 * 4 * 4, "four clips of four i32");

    let token_values: Vec<_> = (0..trace.values.len() as u32)
        .map(poem_ir::ValueId)
        .filter(|id| {
            matches!(&compiled.arena.placements[id.0 as usize], Placement::Arena { rows, .. }
                if rows.axis() == Some(RowAxis::Tokens))
        })
        .collect();
    assert!(!token_values.is_empty());
    for token in token_values {
        assert!(!compiled.arena.co_tenants(plane, token));
        assert!(!compiled.arena.co_tenants(grid, token));
    }
    assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
}

fn a_voxel_plan_against_no_voxel_ceiling_is_refused_by_name() {
    let trace = trace_hybrid("decoder", &Decoder, Platform::Cuda);
    let refusal = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
        .expect_err("no voxel ceiling, no load");
    assert_eq!(
        refusal,
        Error::Unsized {
            axis: RowAxis::Voxels
        }
    );
    assert!(refusal.to_string().contains("voxels"), "{refusal}");

    let past = Budgets::of(Budget::new(4, 64)).with_voxels(VoxelLadder {
        max_voxels: 32,
        buckets: vec![16, 48],
        max_clips: 2,
    });
    assert!(matches!(
        compile_axes(&trace, &past, &DeviceProfile::default()),
        Err(Error::Budget { .. })
    ));
}
