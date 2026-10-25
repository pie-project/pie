use poem_ir::{NormAxis, NormWeight};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::ltx_2_dims as model;
use poem::{Def, Dtype, Operation, Platform, RuntimeInput, Trace};
use poem_ir::{Conv, GridRule, InputAxis, TimePad};

const FLAGSHIP: &str = "ltx25-bf16-kv-bf16";
const MINI: &str = "ltx25-mini-bf16-kv-bf16";

fn row(deployment: &str) -> &'static poem_compiler::catalog::Deployment {
    poem_compiler::catalog::deployment(deployment)
        .unwrap_or_else(|| panic!("this build ships no `{deployment}`"))
}

fn trace(deployment: &str) -> Trace {
    row(deployment).trace(Platform::Cuda)
}

#[test]
fn the_ltx_2_vae_bakes_every_case() {
    the_flagship_declares_the_decode_reading_and_the_miniature_does_not();
    the_shapes_are_the_ltx_decoders();
    the_import_reads_every_decoder_tensor_of_the_real_snapshot_once();
}

fn the_flagship_declares_the_decode_reading_and_the_miniature_does_not() {
    let readings = |deployment: &str| {
        trace(deployment)
            .facts
            .declared("reading")
            .map(|s| s.values.clone())
            .unwrap_or_default()
    };
    assert_eq!(
        readings(FLAGSHIP).last().map(String::as_str),
        Some("vae.decode"),
        "the flagship declares `vae.decode`, the last value"
    );
    assert!(
        readings(MINI).iter().all(|r| r != "vae.decode"),
        "the miniature's checkpoint carries no VAE"
    );
    for (deployment, want) in [
        (FLAGSHIP, vec!["latent", "latents"]),
        (MINI, vec!["latents"]),
    ] {
        let plan = trace(deployment);
        let voxels: BTreeSet<&str> = plan
            .values
            .iter()
            .filter_map(|decl| match decl.def {
                Def::Input(RuntimeInput::Guest {
                    id,
                    axis: InputAxis::Voxel,
                    ..
                }) => Some(plan.inputs[id as usize].name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            voxels,
            want.into_iter().collect(),
            "{deployment}: the denoiser's latents on the voxel axis, and the VAE's input iff \
             the row carries the VAE"
        );
    }
}

fn the_shapes_are_the_ltx_decoders() {
    let plan = trace(FLAGSHIP);
    assert!(
        plan.caches.is_empty(),
        "a non-causal decoder holds nothing between fires"
    );

    let mut convs = 0usize;
    let mut shuffles: Vec<([u32; 3], u32)> = Vec::new();
    let mut rms_eps: BTreeSet<String> = BTreeSet::new();
    for node in &plan.nodes {
        match &node.op {
            Operation::Conv(Conv::Conv3d {
                k,
                stride,
                pad,
                pad_back,
                causal_t,
                time_pad,
                cache,
                ..
            }) => {
                convs += 1;
                assert_eq!(
                    (*k, *stride, *pad, *pad_back),
                    ([3; 3], [1; 3], [1; 3], [1; 3])
                );
                assert!(!causal_t, "the decoder is non-causal");
                assert_eq!(
                    *time_pad,
                    TimePad::Replicate,
                    "the clip's own end frames pad time"
                );
                assert!(cache.is_none(), "no frame cache on a one-fire decoder");
            }
            Operation::Conv(Conv::PixelShuffle { r, trim_t, .. }) => {
                shuffles.push((*r, *trim_t));
            }
            Operation::Conv(Conv::Grid { rule, .. }) => match rule {
                GridRule::Conv { .. } | GridRule::Shuffle { .. } => {}
                other => panic!("a grid rule this decoder never states: {other:?}"),
            },
            Operation::Conv(Conv::Patchify { .. } | Conv::Unpatchify { .. } | Conv::Cells { .. }) => {}
            Operation::Conv(other) => {
                panic!("a spatial member this decoder never states: {other:?}")
            }
            Operation::Elemwise(poem::Elemwise::Rmsnorm {
                eps,
                axis: NormAxis::Head(_),
                weight: NormWeight::None,
                ..
            }) => {
                rms_eps.insert(format!("{eps:e}"));
            }
            _ => {}
        }
    }
    let resnets: u32 = model::VAE_MID_RESNETS + model::VAE_UP_RESNETS.iter().sum::<u32>();
    assert_eq!(convs, 2 + 4 + 2 * resnets as usize, "41 convolutions");
    assert_eq!(
        shuffles,
        vec![
            ([2, 2, 2], 1),
            ([2, 2, 2], 1),
            ([2, 1, 1], 1),
            ([1, 2, 2], 0),
            ([1, model::VAE_PATCH, model::VAE_PATCH], 0),
        ],
        "four upsamplers, the temporal ones trimming one frame, then the un-patchify"
    );
    assert!(
        rms_eps.contains(&format!("{:e}", model::VAE_EPS)),
        "`PerChannelRMSNorm` at 1e-8: {rms_eps:?}"
    );
    for param in &plan.params {
        if param.name.starts_with("vae.") && param.name.ends_with(".bias") {
            assert_eq!(
                param.dtype,
                Dtype::F32,
                "{}: a conv bias is f32",
                param.name
            );
        }
    }
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
    let snapshots = hub().join("models--Lightricks--LTX-2.5-Diffusers/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.join("vae/config.json").is_file()
                && path
                    .join("vae/diffusion_pytorch_model.safetensors")
                    .is_file()
        })
}

fn the_import_reads_every_decoder_tensor_of_the_real_snapshot_once() {
    let Some(root) = snapshot() else {
        eprintln!(
            "skipping: no Lightricks/LTX-2.5-Diffusers snapshot with a vae/ in the HuggingFace cache"
        );
        return;
    };
    let src = checkpoint::file::diffusers::open(&root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let contract = import_vae(&src);

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &contract.tensors {
        for source in tensor.expr.sources() {
            *counts.entry(source.to_string()).or_default() += 1;
        }
    }
    let index: BTreeSet<String> = src
        .names()
        .filter(|n| n.starts_with("vae.decoder.") || n.starts_with("vae.latents_"))
        .map(str::to_string)
        .collect();
    assert_eq!(
        index.len(),
        84 + 2,
        "84 decoder tensors and the two buffers"
    );
    let read: BTreeSet<String> = counts.keys().cloned().collect();
    assert_eq!(
        read, index,
        "every decoder tensor and buffer, and nothing else"
    );
    assert!(counts.values().all(|c| *c == 1), "each exactly once");
    assert!(
        !read.iter().any(|n| n.starts_with("vae.encoder.")),
        "the encoder is not traced and not read"
    );
    let stated: Vec<&str> = contract
        .tensors
        .iter()
        .filter(|t| t.expr.sources().is_empty() && t.expr.outputs().is_empty())
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(stated, vec!["vae.zero"]);
}

/// The VAE's reads alone, by the package's own functions.
const VAE_FORMATS: &str = r#"
def formats(m):
    return [format("vae", read = lambda reads: read_vae(reads, m.vae))]
"#;

fn import_vae(src: &ztensor::Source) -> checkpoint::contract::ModelContract {
    let package = poem_compiler::catalog::replacing("ltx25", "formats", VAE_FORMATS)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"));
    package
        .import("ltx25", &flagship(), src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the VAE does not read this snapshot: {why}"))
}

fn flagship() -> poem::star::Deploy {
    poem::star::Deploy {
        weights: vec![Dtype::Bf16.into()],
        kv: Dtype::Bf16,
        tp: 1,
        ep: 1,
        parts: vec![],
        drafter: None,
        yarn: None,
        tap: None,
    }
}
