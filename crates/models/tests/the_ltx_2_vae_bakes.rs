use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use model::VAE_DECODE;
use models::{PortKind, ReadoutKind};
use poem::{Def, Dtype, Operation, Platform, RuntimeInput, Trace};
use poem_ir::{GridRule, Spatial, TimePad};

const FLAGSHIP: &str = "ltx25-bf16-kv-bf16";
const MINI: &str = "ltx25-mini-bf16-kv-bf16";

fn row(sku: &str) -> &'static models::Deployment {
    models::deployment(sku).unwrap_or_else(|| panic!("this build ships no `{sku}`"))
}

fn trace(sku: &str) -> Trace {
    row(sku).trace(Platform::Cuda)
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
    assert_eq!(decode.streams, vec![poem::Stream::Video]);
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
            Operation::Elementwise(poem::Elementwise::RmsnormNoScale { eps, .. }) => {
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
    return [format("vae", read = lambda reads: vae(reads, m.vae))]
"#;

fn import_vae(src: &ztensor::Source) -> checkpoint::contract::ModelContract {
    let package = models::star::replacing("ltx25", "formats.star", "formats", VAE_FORMATS)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"));
    package
        .import("ltx25", &flagship(), src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the VAE does not read this snapshot: {why}"))
}

fn flagship() -> poem::star::Deploy {
    poem::star::Deploy {
        weights: vec![Dtype::Bf16],
        kv: Dtype::Bf16,
        tp: 1,
        parts: vec![],
        drafter: None,
    }
}

/// The widths and constants ltx_2's package declares.
#[allow(dead_code)]
mod model {
    pub const PATCH_T: u32 = 1;
    pub const PATCH_H: u32 = 1;
    pub const PATCH_W: u32 = 1;

    pub const VAE_SPATIAL_COMPRESSION: u32 = 32;
    pub const VAE_TEMPORAL_COMPRESSION: u32 = 8;
    pub const VAE_Z: u32 = 128;

    pub const VAE_RGB: u32 = 3;
    pub const VAE_PATCH: u32 = 4;
    pub const VAE_EPS: f32 = 1e-8;
    pub const VAE_DECODER_DIMS: [u32; 5] = [1024, 512, 512, 256, 128];
    pub const VAE_MID_RESNETS: u32 = 2;
    pub const VAE_UP_RESNETS: [u32; 4] = [2, 4, 6, 4];
    pub const VAE_UP_STRIDES: [[u32; 3]; 4] = [[2, 2, 2], [2, 2, 2], [2, 1, 1], [1, 2, 2]];

    pub const T_FREQ_DIM: u32 = 256;
    pub const T_MAX_PERIOD: f32 = 10_000.0;
    pub const T_FLIP_SIN_COS: bool = true;
    pub const T_SCALE: f32 = 1.0;

    pub const NORM_EPS: f32 = 1e-6;

    pub const ROPE_THETA: f32 = 10_000.0;
    pub const ROPE_MAX_POS: [f32; 3] = [20.0, 2048.0, 2048.0];
    pub const AUDIO_ROPE_MAX_POS: f32 = 20.0;
    pub const CROSS_ROPE_MAX_POS: f32 = 20.0;
    pub const ROPE_AXES: u8 = 3;
    pub const AUDIO_ROPE_AXES: u8 = 1;

    pub const VIDEO_SCALE: [f32; 3] = [8.0, 32.0, 32.0];
    pub const AUDIO_SCALE: f32 = 4.0;
    pub const CAUSAL_OFFSET: f32 = 1.0;
    pub const AUDIO_SAMPLING_RATE: f32 = 16_000.0;
    pub const AUDIO_HOP: f32 = 160.0;

    pub const GATE_SCALE: f32 = 2.0;

    pub const MOD_SLICES: u32 = 9;
    pub const AV_SS_SLICES: u32 = 4;
    pub const AV_GATE_SLICES: u32 = 1;
    pub const PROMPT_SLICES: u32 = 2;
    pub const HEAD_SLICES: u32 = 2;

    pub const AV_GATE_TIMESTEP_SCALE: f32 = 1.0;

    pub const TRAIN_STEPS: u32 = 1000;
    pub const DISTILLED_SIGMAS: [f32; 8] = [
        1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875,
    ];
    pub const STAGE2_SIGMAS: [f32; 3] = [0.909_375, 0.725, 0.421_875];

    pub const TEXT_LAYERS: u32 = 49;
    pub const TEXT_LEN: u32 = 1024;
    pub const CONN_REGISTERS: u32 = 128;
    pub const CONN_ROPE_BASE: f32 = 4096.0;
    pub const CONN_FF_MULT: u32 = 4;

    pub mod port {
        pub const LATENTS: u8 = 0;
        pub const CONTEXT: u8 = 0;
        pub const AUDIO_CONTEXT: u8 = 1;
        pub const TEXT: u8 = 1;
        pub const TIMESTEP: u8 = 0;
        pub const POSITIONS: u8 = 0;
        pub const TIME_POSITIONS: u8 = 1;
        pub const VOXELS: u8 = 0;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Dims {
        pub layers: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub audio_heads: u32,
        pub audio_head_dim: u32,
        pub channels: u32,
        pub cross_dim: u32,
        pub audio_cross_dim: u32,
        pub ff_mult: u32,
        pub caption: u32,
        pub conn_layers: u32,
    }

    impl Dims {
        #[must_use]
        pub const fn ltx_2_5() -> Dims {
            Dims {
                layers: 48,
                heads: 32,
                head_dim: 128,
                audio_heads: 32,
                audio_head_dim: 64,
                channels: 128,
                cross_dim: 4096,
                audio_cross_dim: 2048,
                ff_mult: 4,
                caption: 3840,
                conn_layers: 8,
            }
        }

        #[must_use]
        pub const fn mini() -> Dims {
            Dims {
                layers: 2,
                heads: 2,
                head_dim: 128,
                audio_heads: 2,
                audio_head_dim: 64,
                channels: 128,
                cross_dim: 256,
                audio_cross_dim: 128,
                ff_mult: 4,
                caption: 16,
                conn_layers: 1,
            }
        }

        #[must_use]
        pub const fn dim(&self) -> u32 {
            self.heads * self.head_dim
        }

        #[must_use]
        pub const fn audio_dim(&self) -> u32 {
            self.audio_heads * self.audio_head_dim
        }

        #[must_use]
        pub const fn av_inner(&self) -> u32 {
            self.audio_heads * self.audio_head_dim
        }

        #[must_use]
        pub const fn text_in(&self) -> u32 {
            self.caption * TEXT_LAYERS
        }

        #[must_use]
        pub fn sm_scale(&self) -> f32 {
            (self.head_dim as f32).sqrt().recip()
        }

        #[must_use]
        pub fn audio_sm_scale(&self) -> f32 {
            (self.audio_head_dim as f32).sqrt().recip()
        }

        #[must_use]
        pub const fn rope_dims(&self) -> [u32; 4] {
            let f = self.dim() / (2 * ROPE_AXES as u32);
            [2 * f, 2 * f, 2 * f, 0]
        }

        #[must_use]
        pub const fn audio_rope_dims(&self) -> [u32; 4] {
            [self.audio_dim(), 0, 0, 0]
        }

        #[must_use]
        pub const fn av_rope_dims(&self) -> [u32; 4] {
            [self.av_inner(), 0, 0, 0]
        }
    }

    #[must_use]
    pub const fn rope_pad(dim: u32, axes: u8) -> u32 {
        let axes = axes as u32;
        dim / 2 - axes * (dim / (2 * axes))
    }

    pub const DENOISE: u8 = 0;
    pub const REFINE_VIDEO: u8 = 1;
    pub const REFINE_AUDIO: u8 = 2;
    pub const VAE_DECODE: u8 = 3;
}
