use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{Expr, Partition, TensorType};

use crate::flux_2_dims::{self as model, vae};
use poem::{Def, Dim, Dtype, Operation, Platform, Request, Trace, Ty, seam};
use poem_ir::{Attn, Conv, GridRule, InputAxis, ParamLayout, Seam};

const KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";
const MINI: &str = "flux2-mini-bf16-kv-bf16";

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
fn the_flux_2_vae_bakes_every_case() {
    the_flagship_declares_the_two_vae_readings_and_the_miniature_neither();
    the_trace_reads_two_voxel_ports_and_plants_pixels_twice();
    the_shapes_are_the_flux2_autoencoders();
    each_vae_lane_has_a_class_of_its_own();
    the_import_reads_every_vae_tensor_of_the_real_snapshot_once();
}

fn the_flagship_declares_the_two_vae_readings_and_the_miniature_neither() {
    assert_eq!(
        readings(KLEIN),
        ["text", "denoise", "vae.decode", "vae.encode"].map(String::from)
    );
    let facts = row(KLEIN).generative().expect("facts");
    let latent = facts.latent.as_ref().expect("a latent space");
    assert_eq!(
        (latent.channels, latent.spatial_compression),
        (model::VAE_CHANNELS, model::VAE_COMPRESSION),
        "the public latent is the VAE's own 32 channels at /8"
    );
    assert_eq!(
        model::VAE_CHANNELS * model::PACK * model::PACK,
        model::IN_CHANNELS,
        "the denoiser's 128-wide rows are the VAE's 32 channels packed 2x2"
    );

    assert!(
        !readings(MINI).iter().any(|r| r.starts_with("vae.")),
        "the miniature has no VAE"
    );
    let mini = trace(MINI);
    let voxel_ports: Vec<&str> = mini
        .values
        .iter()
        .filter_map(|v| match &v.def {
            Def::Input(poem::RuntimeInput::Guest {
                id,
                axis: InputAxis::Voxel,
                ..
            }) => Some(mini.inputs[*id as usize].name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        voxel_ports,
        vec!["latents"],
        "and reads no voxel port but the denoiser's latent"
    );
}

fn the_trace_reads_two_voxel_ports_and_plants_pixels_twice() {
    let plan = trace(KLEIN);
    let mut ports: Vec<(&str, u32, Dtype)> = plan
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
    ports.sort_unstable_by_key(|(name, ..)| *name);
    assert_eq!(
        ports,
        vec![
            ("latent", model::VAE_CHANNELS, Dtype::Bf16),
            ("latents", model::VAE_CHANNELS, Dtype::Bf16),
            ("pixels", vae::RGB, Dtype::Bf16)
        ]
    );
    let pixels: Vec<&Seam> = plan
        .seams
        .iter()
        .filter(|s| s.seam == seam::PIXELS.name)
        .collect();
    assert_eq!(pixels.len(), 2, "one pixels seam per VAE reading");
    let mut planes: Vec<Ty> = pixels
        .iter()
        .map(|s| {
            assert_eq!(s.values.len(), 2, "the plane and its grid");
            assert_eq!(
                plan.values[s.values[1].0 as usize].ty,
                Ty::Tensor {
                    shape: vec![Dim::Clips, Dim::Const(4)],
                    dtype: Dtype::I32
                }
            );
            plan.values[s.values[0].0 as usize].ty.clone()
        })
        .collect();
    planes.sort_by_key(|ty| format!("{ty:?}"));
    let plane = |rows: Dim, width: u64| Ty::Tensor {
        shape: vec![rows, Dim::Const(width)],
        dtype: Dtype::Bf16,
    };
    let mut want = vec![
        plane(Dim::VoxelsTimes(256), u64::from(vae::RGB)),
        plane(Dim::VoxelsTimes(4), u64::from(model::VAE_CHANNELS)),
    ];
    want.sort_by_key(|ty| format!("{ty:?}"));
    assert_eq!(planes, want);
    assert_eq!(
        plan.seams
            .iter()
            .filter(|s| s.seam == seam::VELOCITY.name)
            .count(),
        1,
        "the denoise reading's seam is untouched"
    );
}

fn the_shapes_are_the_flux2_autoencoders() {
    let plan = trace(KLEIN);
    let (mut attentions, mut upsamples, mut copies, mut strided, mut convs, mut norms) =
        (0, 0, 0, 0, 0, 0);
    let (mut shuffles, mut unshuffles, mut silu_off) = (0, 0, 0);
    for node in &plan.nodes {
        match &node.op {
            Operation::Attn(Attn::Voxel { sm_scale, q, .. }) => {
                attentions += 1;
                let Ty::Tensor { shape, dtype } = &plan.values[q.0 as usize].ty else {
                    panic!("a query is a tensor")
                };
                assert_eq!(*dtype, Dtype::Bf16);
                assert_eq!(shape[1], Dim::Const(512));
                assert!(matches!(shape[0], Dim::Voxels | Dim::VoxelsTimes(4)));
                assert!((sm_scale - 512f32.sqrt().recip()).abs() < 1e-9);
            }
            Operation::Conv(Conv::Upsample {
                factor,
                keep_first_frame,
                ..
            }) => {
                assert!(!keep_first_frame);
                match *factor {
                    [1, 1, 1] => copies += 1,
                    [1, 2, 2] => upsamples += 1,
                    other => panic!("no upsample of {other:?} belongs to this VAE"),
                }
            }
            Operation::Conv(Conv::PixelShuffle { r, .. }) => {
                shuffles += 1;
                assert_eq!(*r, [1, model::PACK, model::PACK]);
            }
            Operation::Conv(Conv::PixelUnshuffle { r, .. }) => {
                unshuffles += 1;
                assert_eq!(*r, [1, model::PACK, model::PACK]);
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
    assert_eq!(copies, 0, "the decode's pack is its fresh copy of the port");
    assert_eq!(strided, 3);
    assert_eq!(
        (shuffles, unshuffles),
        (2, 2),
        "each reading packs 2x2 for the batch norm and unpacks after"
    );
    assert_eq!(convs, 64);
    assert_eq!(norms, 52);
    assert_eq!(silu_off, 2, "only the attention's norm has no SiLU");
}

fn each_vae_lane_has_a_class_of_its_own() {
    let plan = trace(KLEIN);
    let classes = poem::resolve_classes(&plan).expect("every merge resolves");
    let class_of = |name: &str, stream: &str| {
        let request = Request::new(4, false)
            .switched("stream", stream)
            .switched("reading", name);
        let word = plan.facts.word(&request);
        classes
            .class_of(word & classes.mask)
            .unwrap_or_else(|| panic!("a {name} lane has no class"))
    };
    let seen: BTreeSet<usize> = [
        class_of("text", "text"),
        class_of("denoise", "text"),
        class_of("denoise", "image"),
        class_of("denoise", "reference"),
        class_of("vae.decode", "image"),
        class_of("vae.encode", "image"),
    ]
    .into_iter()
    .collect();
    assert_eq!(seen.len(), 6, "six lanes, six classes");
}

fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

fn snapshot() -> Option<PathBuf> {
    let snapshots = hub().join("models--black-forest-labs--FLUX.2-klein-4B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("vae/config.json").is_file())
}

struct Types<'a>(&'a ztensor::Source);

impl CheckpointTypes for Types<'_> {
    fn tensor_type(&self, name: &str) -> Option<TensorType> {
        let tensor = self.0.get(name)?;
        let encoding = checkpoint::file::encoding_of(&tensor).ok()?;
        Some(TensorType {
            shape: tensor.shape().iter().map(|&n| n as i64).collect(),
            encoding,
        })
    }
}

fn the_import_reads_every_vae_tensor_of_the_real_snapshot_once() {
    let Some(root) = snapshot() else {
        eprintln!(
            "skipping: no black-forest-labs/FLUX.2-klein-4B snapshot in the HuggingFace cache"
        );
        return;
    };
    let src = checkpoint::file::diffusers::open(&root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let mut contract = poem_compiler::catalog::deployment("flux2-klein-4b-bf16-kv-bf16")
        .expect("the klein row")
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the pipeline does not read this snapshot: {why}"));
    contract.tensors.retain(|t| t.name.starts_with("vae."));

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &contract.tensors {
        for source in tensor.expr.sources() {
            *counts.entry(source.to_string()).or_default() += 1;
        }
    }
    let index: BTreeSet<String> = src
        .names()
        .filter(|n| n.starts_with("vae."))
        .map(str::to_string)
        .collect();
    assert_eq!(index.len(), 251);
    let want: BTreeSet<String> = index
        .iter()
        .filter(|n| n.as_str() != "vae.bn.num_batches_tracked")
        .cloned()
        .collect();
    let read: BTreeSet<String> = counts.keys().cloned().collect();
    assert_eq!(
        read, want,
        "every `vae.` tensor but the BatchNorm's step counter, and only those"
    );
    assert!(
        counts.values().all(|c| *c == 1),
        "each exactly once (`bn.running_var` included: `√(var+eps)` and \
         `(var+eps)^-1/2` share one internal sum)"
    );

    let types = Types(&src);
    let mut resolver = Resolver::new(&types, Partition::WHOLE);
    let mut kernels = 0;
    for tensor in &contract.tensors {
        if tensor.name.starts_with("vae.") && tensor.name.ends_with("conv_in") {
            kernels += 1;
        }
        if let Expr::Transmute { .. } = &tensor.expr {
            assert!(tensor.name.starts_with("vae."), "`{}`", tensor.name);
        }
        if tensor.name == "vae.quant" || tensor.name == "vae.quant.bias.head" {
            assert!(
                format!("{:?}", tensor.expr).contains("Slice"),
                "`{}` is sliced down to the mean's rows",
                tensor.name
            );
        }
        if tensor.name.starts_with("vae.quant") {
            assert_eq!(
                tensor.shape.as_ref().map(|s| s[0]),
                Some(i64::from(model::VAE_CHANNELS)),
                "`{}`",
                tensor.name
            );
        }
        let ty = resolver
            .infer(&tensor.expr, &tensor.name)
            .unwrap_or_else(|why| panic!("`{}` does not type: {why}", tensor.name));
        if let Some(shape) = &tensor.shape {
            assert_eq!(&ty.shape, shape, "`{}`", tensor.name);
        }
        assert_eq!(ty.encoding, tensor.encoding, "`{}`", tensor.name);
        resolver.publish(&tensor.name, ty);
    }
    assert_eq!(kernels, 2, "one conv_in per side");
    let transmuted = contract
        .tensors
        .iter()
        .filter(|t| matches!(&t.expr, Expr::Transmute { .. }))
        .count();
    assert_eq!(
        transmuted, 63,
        "every conv kernel but the sliced `quant_conv`, and nothing else"
    );
}
