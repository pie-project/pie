#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

pub mod common_vae;

use common_vae::{artifact, bf16_bytes, f32s, golden, one_arm, score};
use engine_cuda::Lane;
use engine_cuda::serve::{Clips, Seated};
use poem::Platform;

const ROW: &str = "flux2-klein-4b-bf16-kv-bf16";
const IN_CHANNELS: u32 = 128;
const TOKEN_COMPRESSION: u32 = 16;

fn fire(
    root: &Path,
    decode: bool,
    max_voxels: u32,
    clip: [u32; 3],
    payload: &[f32],
) -> (Vec<f32>, Vec<[u32; 3]>, f64, f64) {
    let row = poem_compiler::catalog::deployment(ROW).expect("the flagship row");
    let trace = one_arm("flux2-klein-4b", decode)
        .trace(
            &row.model.id,
            &row.deploy,
            if decode {
                "flux2-vae-decode"
            } else {
                "flux2-vae-encode"
            },
            Platform::Cuda,
        )
        .unwrap_or_else(|why| panic!("{why:#}"));
    let src = ztensor::Source::open(root).unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let contract = poem::import::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the artifact does not hold this arm's planes: {why}"));
    drop(src);
    let word = trace
        .facts
        .word(&poem::Request::new(1, false).on_stream(poem::Stream::Image));
    let started = Instant::now();
    let mut shell = common_vae::load(trace, &contract, root, max_voxels);
    let load_s = started.elapsed().as_secs_f64();
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
#[ignore = "needs a CUDA device, flux2-klein-4b.zt under PIE_IMAGEGEN_ARTIFACTS and the flux2_golden.py --vae dump under PIE_IMAGEGEN_GOLDEN"]
fn the_vae_decodes_and_encodes_the_golden_clip() {
    assert!(engine_cuda::device::present(), "no CUDA device");
    let root = artifact("flux2-klein-4b.zt");
    let gold = golden("flux2/flux2_vae");
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
        latent_box[1] * TOKEN_COMPRESSION,
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
