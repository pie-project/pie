use crate::z_image_dims::{self as model, vae};
use poem::{Def, Dim, Dtype, Operation, Platform, Trace, Ty};
use poem_ir::{Attn, Conv, GridRule, InputAxis, Output, ParamLayout};

const TURBO: &str = "z-image-turbo-bf16-kv-bf16";
const MINI: &str = "z-image-mini-bf16-kv-bf16";

fn row(deployment: &str) -> &'static poem_compiler::catalog::Deployment {
    poem_compiler::catalog::deployment(deployment)
        .unwrap_or_else(|| panic!("this build ships no `{deployment}`"))
}

fn trace(deployment: &str) -> Trace {
    row(deployment).trace(Platform::Cuda)
}

fn readings(deployment: &str) -> Vec<String> {
    trace(deployment)
        .facts
        .declared("reading")
        .map(|s| s.values.clone())
        .unwrap_or_default()
}

#[test]
fn the_z_image_vae_bakes_every_case() {
    the_flagship_declares_the_two_vae_readings_and_the_miniature_neither();
    the_trace_reads_its_voxel_inputs_and_plants_pixels_twice();
    the_shapes_are_the_flux_vaes();
}

fn the_flagship_declares_the_two_vae_readings_and_the_miniature_neither() {
    assert_eq!(
        readings(TURBO),
        vec!["text", "denoise", "vae.decode", "vae.encode"]
    );
    assert!(
        !readings(MINI).iter().any(|r| r.starts_with("vae.")),
        "the miniature has no VAE"
    );
    assert!(
        !trace(MINI).values.iter().any(|v| matches!(
            v.def,
            Def::Input(poem::RuntimeInput::Guest {
                axis: InputAxis::Voxel,
                id,
                ..
            }) if trace(MINI).inputs[id as usize].name != "latents"
        )),
        "and reads no voxel input but the denoiser's latent"
    );
}

fn the_trace_reads_its_voxel_inputs_and_plants_pixels_twice() {
    // The VAE's two, and the denoiser's VAE-native latent.
    let plan = trace(TURBO);
    let mut named: Vec<(&str, u32, Dtype)> = plan
        .values
        .iter()
        .filter_map(|v| match &v.def {
            Def::Input(poem::RuntimeInput::Guest {
                id,
                axis: InputAxis::Voxel,
                width,
                dtype,
            }) => Some((plan.inputs[*id as usize].name.as_str(), *width, *dtype)),
            _ => None,
        })
        .collect();
    named.sort_unstable_by_key(|(name, ..)| *name);
    assert_eq!(
        named,
        vec![
            ("latents", model::CHANNELS, Dtype::Bf16),
            ("latents", model::CHANNELS, Dtype::Bf16),
            ("pixels", vae::RGB, Dtype::Bf16)
        ]
    );
    let pixels: Vec<&Output> = plan.outputs.iter().filter(|s| s.name == "pixels").collect();
    assert_eq!(pixels.len(), 2, "one pixels seam per VAE reading");
    let mut planes: Vec<Ty> = pixels
        .iter()
        .map(|s| {
            assert_eq!(
                plan.values[s.grid.expect("the pixels name their grid").0 as usize].ty,
                Ty::Tensor {
                    shape: vec![Dim::Clips, Dim::Const(4)],
                    dtype: Dtype::I32
                }
            );
            plan.values[s.value.0 as usize].ty.clone()
        })
        .collect();
    planes.sort_by_key(|ty| format!("{ty:?}"));
    let plane = |rows: Dim, width: u64| Ty::Tensor {
        shape: vec![rows, Dim::Const(width)],
        dtype: Dtype::Bf16,
    };
    let mut want = vec![
        plane(Dim::VoxelsScaled(64, 1), u64::from(vae::RGB)),
        plane(Dim::VoxelsScaled(1, 64), u64::from(model::CHANNELS)),
    ];
    want.sort_by_key(|ty| format!("{ty:?}"));
    assert_eq!(planes, want);
    assert!(
        plan.outputs.iter().filter(|s| s.name == "velocity").count() == 1
            && !plan.outputs.iter().any(|s| s.name == "logits"),
        "the token readings' seams are untouched"
    );
}

fn the_shapes_are_the_flux_vaes() {
    let plan = trace(TURBO);
    let (mut attentions, mut upsamples, mut strided, mut convs, mut norms) = (0, 0, 0, 0, 0);
    let mut silu_off = 0;
    for node in &plan.nodes {
        match &node.op {
            Operation::Attn(Attn::Voxel { sm_scale, q, .. }) => {
                attentions += 1;
                let Ty::Tensor { shape, dtype } = &plan.values[q.0 as usize].ty else {
                    panic!("a query is a tensor")
                };
                assert_eq!((shape[1], *dtype), (Dim::Const(512), Dtype::Bf16));
                assert!(matches!(shape[0], Dim::Voxels | Dim::VoxelsScaled(1, 64)));
                assert!((sm_scale - 512f32.sqrt().recip()).abs() < 1e-9);
            }
            Operation::Conv(Conv::Upsample {
                factor,
                keep_first_frame,
                ..
            }) => {
                upsamples += 1;
                assert_eq!((*factor, *keep_first_frame), ([1, 2, 2], false));
            }
            Operation::Conv(Conv::Conv3d {
                x,
                w,
                k,
                stride,
                pad,
                pad_back,
                causal_t,
                cache,
                ..
            }) => {
                convs += 1;
                assert!(!causal_t && cache.is_none() && k[0] == 1 && stride[0] == 1);
                let c_in = match &plan.values[x.0 as usize].ty {
                    Ty::Tensor { shape, .. } => match shape[1] {
                        Dim::Const(c) => c as u32,
                        other => panic!("{other:?}"),
                    },
                    other => panic!("{other:?}"),
                };
                let Def::Weight(at) = plan.values[w.0 as usize].def else {
                    panic!("a conv reads a weight")
                };
                let param = &plan.params[at as usize];
                assert_eq!(
                    param.layout,
                    ParamLayout::ConvTapsMajor {
                        c_in,
                        taps: k[1] * k[2]
                    },
                    "`{}`",
                    param.name
                );
                if stride[1] == 2 {
                    strided += 1;
                    assert_eq!((*k, *pad, *pad_back), ([1, 3, 3], [0; 3], [0, 1, 1]));
                } else {
                    assert_eq!(pad, pad_back);
                    assert!(matches!((k[1], pad[1]), (3, 1) | (1, 0)));
                }
            }
            Operation::Conv(Conv::GroupNorm {
                groups, eps, silu, ..
            }) => {
                norms += 1;
                assert_eq!((*groups, *eps), (vae::GN_GROUPS, vae::GN_EPS));
                if !silu {
                    silu_off += 1;
                }
            }
            Operation::Conv(Conv::Grid {
                rule: GridRule::Conv { pad, pad_back, .. },
                ..
            }) => {
                assert!(pad == pad_back || *pad_back == [0, 1, 1]);
            }
            _ => {}
        }
    }
    assert_eq!(attentions, 2, "one mid-block attention per reading");
    assert_eq!(upsamples, 3);
    assert_eq!(strided, 3);
    assert_eq!(convs, 62);
    assert_eq!(norms, 52);
    assert_eq!(silu_off, 2, "only the attention's norm has no SiLU");
}
