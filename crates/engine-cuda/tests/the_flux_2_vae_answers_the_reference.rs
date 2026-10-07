#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Recording, Shell};
use models::flux_2::forward::Facts;
use models::flux_2::model::{IN_CHANNELS, Model};
use models::flux_2::vae;
use poem_compiler::{Budget, VoxelLadder};
use poem_dsl::{Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Value, trace_hybrid};

struct OneArm {
    model: Model,
    decode: bool,
}

impl ForwardHybrid for OneArm {
    type Facts = Facts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<Facts>) -> Value {
        let v = self
            .model
            .vae
            .as_ref()
            .expect("the flagship carries the VAE");
        if self.decode {
            vae::decode(&inputs, v)
        } else {
            vae::encode(&inputs, v)
        }
    }
}

fn artifact() -> Option<PathBuf> {
    let path = std::env::var_os("PIE_IMAGEGEN_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen"))
        .join("flux2-klein-4b.zt");
    path.is_file().then_some(path)
}

fn golden() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let dir = root.join("flux2/flux2_vae");
    dir.join("shapes.json").is_file().then_some(dir)
}

fn f32s(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| {
            let bits = v.to_bits();
            let rounding = 0x7fff + ((bits >> 16) & 1);
            (((bits + rounding) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

struct Score {
    cos: f64,
    max_abs: f64,
    mean_abs: f64,
}

fn score(got: &[f32], want: &[f32]) -> Score {
    assert_eq!(got.len(), want.len(), "one value per reference value");
    let (mut dot, mut gg, mut ww, mut max_abs, mut sum_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (f64::from(*g), f64::from(*w));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        let err = (g - w).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }
    Score {
        cos: dot / (gg.sqrt() * ww.sqrt()).max(1e-30),
        max_abs,
        mean_abs: sum_abs / want.len() as f64,
    }
}

fn fire(
    root: &PathBuf,
    decode: bool,
    max_voxels: u32,
    clip: [u32; 3],
    payload: &[f32],
) -> (Vec<f32>, Vec<[u32; 3]>, f64, f64) {
    let model = Model::klein_4b(Dtype::Bf16);
    let arm = OneArm { model, decode };
    let trace = trace_hybrid(
        if decode {
            "flux2-vae-decode"
        } else {
            "flux2-vae-encode"
        },
        &arm,
        Platform::Cuda,
    );
    let src = ztensor::Source::open(root).unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the artifact does not hold this arm's planes: {why}"));
    drop(src);
    let contract = &contract;
    let word =
        Facts::of(&poem_dsl::Request::new(1, false).on_stream(poem_dsl::Stream::Image)).word();
    let started = Instant::now();
    let mut shell = Shell::load(Boot {
        classify: |request| Facts::of(request).word(),
        trace,
        contract,
        checkpoint: root,
        budget: Budget::new(2, 16),
        patches: None,
        voxels: Some(VoxelLadder::new(max_voxels, 2)),
        profile: None,
        page_size: 16,
        context: 64,
        slots: 2,
        pages: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: Knobs {
            recording: Recording::Off,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: true,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| {
        panic!(
            "the VAE {} does not load: {why}",
            if decode { "decoder" } else { "encoder" }
        )
    });
    let load_s = started.elapsed().as_secs_f64();
    shell.open(0).expect("slot 0 opens");
    let tokens = [0u32];
    let lanes = [Seated::of(Lane {
        slot: 0,
        word,
        tokens: &tokens,
    })];
    let bytes = bf16_bytes(payload);
    let clips = [Clips {
        lane: 0,
        clips: &[clip],
        payload: &bytes,
    }];
    let _ = shell.fire_voxels(&lanes, &clips).expect("the first fire");
    let started = Instant::now();
    let mut answered = shell.fire_voxels(&lanes, &clips).expect("the second fire");
    let fire_s = started.elapsed().as_secs_f64();
    assert_eq!(answered.len(), 1);
    let (values, boxes) = answered.remove(0);
    drop(shell);
    (values, boxes, load_s, fire_s)
}

#[test]
fn the_vae_decodes_and_encodes_the_golden_clip() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the VAE parity gate: no CUDA device");
        return;
    }
    let Some(root) = artifact() else {
        eprintln!(
            "skipping the VAE parity gate: no flux2-klein-4b.zt (run `pie model import` first)"
        );
        return;
    };
    let Some(gold) = golden() else {
        eprintln!("skipping the VAE parity gate: no flux2_golden.py --vae dump");
        return;
    };
    let latent = f32s(&gold.join("latent.f32"));
    let pixels = f32s(&gold.join("pixels.f32"));
    let mean = f32s(&gold.join("mean.f32"));
    let square = |len: usize, channels: usize| -> [u32; 3] {
        let side = ((len / channels) as f64).sqrt().round() as u32;
        assert_eq!(
            side as usize * side as usize * channels,
            len,
            "a square still"
        );
        [1, side, side]
    };
    let width = IN_CHANNELS as usize;
    let latent_box = square(latent.len(), width);
    let pixel_box = square(pixels.len(), 3);
    let mean_box = square(mean.len(), width);
    let voxels = |b: [u32; 3]| (b[0] * b[1] * b[2]) as usize;
    assert_eq!(latent.len(), voxels(latent_box) * width);
    assert_eq!(pixels.len(), voxels(pixel_box) * 3);
    assert_eq!(mean.len(), voxels(mean_box) * width);
    assert_eq!(
        pixel_box[1],
        latent_box[1] * models::flux_2::model::TOKEN_COMPRESSION,
        "one token is 16 pixels a side"
    );

    let (got, boxes, load_s, fire_s) = fire(
        &root,
        true,
        voxels(latent_box) as u32 + 8,
        latent_box,
        &latent,
    );
    assert_eq!(boxes, vec![pixel_box], "the clip comes back at 16x");
    let s = score(&got, &pixels);
    {
        let side = pixel_box[1] as usize;
        let mut worst: Vec<(f32, usize, usize, usize)> = got
            .iter()
            .zip(&pixels)
            .enumerate()
            .map(|(i, (g, w))| ((g - w).abs(), (i / 3) / side, (i / 3) % side, i % 3))
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let over = worst.iter().filter(|w| w.0 > 0.05).count();
        eprintln!(
            "decode: {over} of {} values past 0.05; worst (err, y, x, c): {:?}",
            got.len(),
            &worst[..8]
        );
        assert!(
            over * 10_000 <= got.len(),
            "{over} values past 0.05 is more than one in ten thousand"
        );
    }
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "decode {latent_box:?} -> {pixel_box:?}: load {load_s:.1} s, fire {fire_s:.3} s, cos {:.6}, \
         max |err| {:.4}, mean |err| {:.5}, range [{lo:.3}, {hi:.3}]",
        s.cos, s.max_abs, s.mean_abs
    );
    assert!(
        s.cos >= 0.999 && s.mean_abs <= 0.005 && s.max_abs <= 0.2,
        "the decode drifts from the reference: cos {}, mean |err| {}, max |err| {}",
        s.cos,
        s.mean_abs,
        s.max_abs
    );

    let (got, boxes, load_s, fire_s) = fire(
        &root,
        false,
        voxels(pixel_box) as u32 + 8,
        pixel_box,
        &pixels,
    );
    assert_eq!(boxes, vec![mean_box], "the clip comes back at 1/16");
    let s = score(&got, &mean);
    {
        let side = mean_box[1] as usize;
        let mut worst: Vec<(f32, usize, usize, usize)> = got
            .iter()
            .zip(&mean)
            .enumerate()
            .map(|(i, (g, w))| {
                (
                    (g - w).abs(),
                    (i / width) / side,
                    (i / width) % side,
                    i % width,
                )
            })
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let over = worst.iter().filter(|w| w.0 > 0.1).count();
        let edge = worst
            .iter()
            .filter(|w| w.0 > 0.1)
            .filter(|w| w.1 == 0 || w.2 == 0 || w.1 + 1 == side || w.2 + 1 == side)
            .count();
        let (lo, hi) = mean
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        eprintln!(
            "encode: {over} of {} values past 0.1 ({edge} on the border); reference range \
             [{lo:.3}, {hi:.3}]; worst (err, y, x, c): {:?}",
            got.len(),
            &worst[..8]
        );
    }
    eprintln!(
        "encode {pixel_box:?} -> {mean_box:?}: load {load_s:.1} s, fire {fire_s:.3} s, cos {:.6}, \
         max |err| {:.4}, mean |err| {:.5}",
        s.cos, s.max_abs, s.mean_abs
    );
    assert!(
        s.cos >= 0.9995 && s.mean_abs <= 0.02 && s.max_abs <= 0.5,
        "the encode drifts from the reference: cos {}, mean |err| {}, max |err| {}",
        s.cos,
        s.mean_abs,
        s.max_abs
    );
}
