use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use model_dsl::{Def, Dtype, Operation, Platform, RuntimeInput, Trace};
use model_ir::{GridRule, Spatial, TimePad};
use models::ltx_2::forward::VAE_DECODE;
use models::ltx_2::model::{self, Model};
use models::{PortKind, ReadoutKind};

const FLAGSHIP: &str = "ltx25-bf16-kv-bf16";
const MINI: &str = "ltx25-mini-bf16-kv-bf16";

fn row(sku: &str) -> &'static models::Sku {
    models::sku(sku).unwrap_or_else(|| panic!("this build ships no `{sku}`"))
}

fn trace(sku: &str) -> Trace {
    (row(sku).trace)(Platform::Cuda)
}

#[test]
fn the_ltx_2_vae_bakes_every_case() {
    the_flagship_declares_the_decode_reading_and_the_miniature_does_not();
    the_shapes_are_the_ltx_decoders();
    the_import_reads_every_decoder_tensor_of_the_real_snapshot_once();
}

fn the_flagship_declares_the_decode_reading_and_the_miniature_does_not() {
    let facts = row(FLAGSHIP).generative.as_ref().expect("facts");
    let decode = facts
        .readings
        .iter()
        .find(|r| r.name == "vae.decode")
        .expect("the flagship declares `vae.decode`");
    assert_eq!(decode.index, VAE_DECODE);
    assert_eq!(
        usize::from(decode.index),
        facts.readings.len() - 1,
        "the last code"
    );
    assert!(!decode.has_kv && !decode.takes_tokens);
    assert_eq!(decode.streams, vec![model_dsl::Stream::Video]);
    assert_eq!(decode.ports.len(), 1);
    let (index, port) = decode.port("latent").expect("the latent port");
    assert_eq!(
        (index, port.kind, port.width),
        (model::port::VOXELS, PortKind::Voxels, model::VAE_Z)
    );
    assert!(decode.positions.is_none(), "a VAE tile takes no positions");
    assert_eq!(decode.readout, ReadoutKind::Pixels);
    assert_eq!(decode.readout_width, model::VAE_RGB);

    let mini = row(MINI).generative.as_ref().expect("facts");
    assert!(
        mini.readings.iter().all(|r| r.name != "vae.decode"),
        "the miniature's checkpoint carries no VAE"
    );
    for (sku, want) in [(FLAGSHIP, 1), (MINI, 0)] {
        let voxels = trace(sku)
            .values
            .iter()
            .filter(|decl| matches!(decl.def, Def::Input(RuntimeInput::Voxels { .. })))
            .count();
        assert_eq!(
            voxels, want,
            "{sku}: the voxel port iff the row carries the VAE"
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
            Operation::Spatial(Spatial::Conv3d {
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
            Operation::Spatial(Spatial::PixelShuffle { r, trim_t, .. }) => {
                shuffles.push((*r, *trim_t));
            }
            Operation::Spatial(Spatial::Grid { rule, .. }) => match rule {
                GridRule::Conv { .. } | GridRule::Shuffle { .. } => {}
                other => panic!("a grid rule this decoder never states: {other:?}"),
            },
            Operation::Spatial(other) => {
                panic!("a spatial member this decoder never states: {other:?}")
            }
            Operation::Elementwise(model_dsl::Elementwise::RmsnormNoScale { eps, .. }) => {
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
    let contract = Model::ltx_2_5(Dtype::Bf16, 1)
        .import_vae(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the VAE does not read this snapshot: {why}"));

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
