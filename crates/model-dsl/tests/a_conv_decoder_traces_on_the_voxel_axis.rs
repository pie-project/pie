use model_dsl::ops::spatial::{self, Conv};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, seam,
    trace_hybrid,
};
use model_ir::{Dim, Operands, ParamLayout, Ty};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

const C_IN: u32 = 8;
const C_MID: u32 = 16;
const C_OUT: u32 = 12;

struct Decoder {
    conv1: Weight,
    b1: Weight,
    gn_w: Weight,
    gn_b: Weight,
    conv2: Weight,
    b2: Weight,
}

impl Decoder {
    fn new() -> Decoder {
        Decoder {
            conv1: Weight::sym(
                "conv1",
                [u64::from(C_MID), u64::from(C_IN) * 27],
                Dtype::Bf16,
            )
            .conv_taps_major(C_IN, 27),
            b1: Weight::sym("conv1.bias", [u64::from(C_MID)], Dtype::F32),
            gn_w: Weight::sym("norm.weight", [u64::from(C_MID)], Dtype::F32),
            gn_b: Weight::sym("norm.bias", [u64::from(C_MID)], Dtype::F32),
            conv2: Weight::sym(
                "conv2",
                [u64::from(C_OUT), u64::from(C_MID) * 27],
                Dtype::Bf16,
            )
            .conv_taps_major(C_MID, 27),
            b2: Weight::sym("conv2.bias", [u64::from(C_OUT)], Dtype::F32),
        }
    }
}

impl ForwardHybrid for Decoder {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let g = inputs.grid();
        let x = inputs.voxels(0, C_IN, Dtype::Bf16);
        let (h, g1) = spatial::conv3d(&x, &g, &self.conv1, Some(&self.b1), Conv::same3(), None);
        assert_eq!(h.rows(), Dim::Voxels, "a convolution keeps its rows");
        assert_eq!(h.width(), u64::from(C_MID));
        let h = spatial::group_norm(&h, &g1, 4, &self.gn_w, &self.gn_b, 1e-6, true);
        let (h, g2) = spatial::upsample_nearest(&h, &g1, [1, 2, 2], false);
        assert_eq!(
            h.rows(),
            Dim::VoxelsTimes(4),
            "an upsample grows rows by its volume"
        );
        let (h, g3) = spatial::conv3d(&h, &g2, &self.conv2, Some(&self.b2), Conv::same3(), None);
        let (y, g4) = spatial::pixel_shuffle(&h, &g3, [1, 2, 2]);
        assert_eq!(
            y.rows(),
            Dim::VoxelsTimes(16),
            "a shuffle grows rows by its block"
        );
        assert_eq!(y.width(), 3, "and divides the width by it");
        seam::at(seam::PIXELS, &[&y, &g4]);
        y
    }
}

#[test]
fn a_conv_decoder_traces_on_the_voxel_axis_every_case() {
    the_decoder_traces_and_its_grids_are_values();
    the_patchify_pair_crosses_the_axis_and_back();
}

fn the_decoder_traces_and_its_grids_are_values() {
    let trace = trace_hybrid("conv-decoder", &Decoder::new(), Platform::Cuda);
    model_ir::check(&trace).expect("the validator accepts the decoder");

    let names: Vec<&str> = trace.nodes.iter().map(|node| node.op.name()).collect();
    assert_eq!(
        names,
        vec![
            "spatial.grid",
            "spatial.conv3d",
            "spatial.group_norm",
            "spatial.grid",
            "spatial.upsample_nearest",
            "spatial.grid",
            "spatial.conv3d",
            "spatial.grid",
            "spatial.pixel_shuffle",
        ],
        "one grid node ahead of every op that changes the box, none for the norm"
    );

    for param in &trace.params {
        let want = if param.name == "conv1" {
            ParamLayout::ConvTapsMajor {
                c_in: C_IN,
                taps: 27,
            }
        } else if param.name == "conv2" {
            ParamLayout::ConvTapsMajor {
                c_in: C_MID,
                taps: 27,
            }
        } else {
            ParamLayout::Natural
        };
        assert_eq!(param.layout, want, "`{}`", param.name);
    }

    assert!(!trace.seams.iter().any(|seam| seam.seam == "out"));
    let pixels = trace
        .seams
        .iter()
        .find(|seam| seam.seam == "pixels")
        .expect("the pixels seam is planted");
    assert_eq!(pixels.values.len(), 2, "the plane and its grid");
    let plane = &trace.values[pixels.values[0].0 as usize].ty;
    let grid = &trace.values[pixels.values[1].0 as usize].ty;
    assert_eq!(
        *plane,
        Ty::Tensor {
            shape: vec![Dim::VoxelsTimes(16), Dim::Const(3)],
            dtype: Dtype::Bf16
        }
    );
    assert_eq!(
        *grid,
        Ty::Tensor {
            shape: vec![Dim::Clips, Dim::Const(4)],
            dtype: Dtype::I32
        }
    );
}

struct Patchifier;

impl ForwardHybrid for Patchifier {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let g = inputs.grid();
        let x = inputs.voxels(0, C_IN, Dtype::Bf16);
        let tg = inputs.token_grid([1, 2, 2]);
        let tokens = spatial::patchify(&x, &g, [1, 2, 2], &tg);
        assert_eq!(
            tokens.rows(),
            Dim::Tokens,
            "patchify lands on the token axis"
        );
        assert_eq!(tokens.width(), u64::from(C_IN) * 4);
        let back = spatial::unpatchify(&tokens, &tg, [1, 2, 2], &g);
        assert_eq!(back.rows(), Dim::Voxels, "and unpatchify comes back");
        assert_eq!(back.width(), u64::from(C_IN));
        seam::at(seam::PIXELS, &[&back, &g]);
        back
    }
}

fn the_patchify_pair_crosses_the_axis_and_back() {
    let trace = trace_hybrid("patchify", &Patchifier, Platform::Cuda);
    model_ir::check(&trace).expect("the validator accepts the pair");
    let names: Vec<&str> = trace.nodes.iter().map(|node| node.op.name()).collect();
    assert_eq!(names, vec!["spatial.patchify", "spatial.unpatchify"]);
    assert!(trace.values.iter().any(|decl| matches!(
        decl.def,
        model_ir::Def::Input(model_ir::RuntimeInput::TokenGrid { p: [1, 2, 2] })
    )));
}

struct NaturalWeight;

impl ForwardHybrid for NaturalWeight {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let g = inputs.grid();
        let x = inputs.voxels(0, C_IN, Dtype::Bf16);
        let w = Weight::sym(
            "conv",
            [u64::from(C_MID), u64::from(C_IN) * 27],
            Dtype::Bf16,
        );
        spatial::conv3d(&x, &g, &w, None, Conv::same3(), None).0
    }
}

#[test]
#[should_panic(expected = "conv_taps_major")]
fn a_conv_weight_declared_natural_is_refused_at_trace_time() {
    let _ = trace_hybrid("natural", &NaturalWeight, Platform::Cuda);
}
