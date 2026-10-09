#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

pub mod common_vae;

use common_vae::{
    Score, bf16_bytes, boxed, deploy, f32s, golden, keep_what_reads, score, shapes, snapshot,
};
use engine_cuda::Lane;
use engine_cuda::serve::{Clips, Seated};
use poem::{Platform, Request, Stream, Trace};

const FLAGSHIP: &str = "ltx25-bf16-kv-bf16";
const VAE_RGB: u32 = 3;
const VAE_Z: u32 = 128;

/// The VAE's decode reading alone, by the package's own functions.
const VAE_ONLY: &str = r#"
def forward(m, inputs):
    return inputs.reading("vae.decode", lambda rows: vae_decode(rows, m.vae))
"#;

fn vae_only() -> Trace {
    models::star::replacing("ltx25", "forward.poem", "forward", VAE_ONLY)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"))
        .trace(
            "ltx25",
            &deploy(FLAGSHIP),
            "ltx25-vae-decode",
            Platform::Cuda,
        )
        .unwrap_or_else(|why| panic!("the VAE-only plan: {why:#}"))
}

/// The VAE's reads alone, by the package's own functions.
const VAE_FORMATS: &str = r#"
def formats(m):
    return [format("vae", read = lambda reads: vae(reads, m.vae))]
"#;

fn import_vae(src: &ztensor::Source) -> checkpoint::contract::ModelContract {
    let package = models::star::replacing("ltx25", "formats.poem", "formats", VAE_FORMATS)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"));
    package
        .import("ltx25", &deploy(FLAGSHIP), src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the VAE does not read this snapshot: {why}"))
}

fn fire(
    root: &Path,
    max_voxels: u32,
    clip: [u32; 3],
    payload: &[f32],
) -> (Vec<f32>, [u32; 3], f64, f64) {
    let src = checkpoint::file::diffusers::open(root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let mut contract = import_vae(&src);
    drop(src);
    let trace = vae_only();
    keep_what_reads(&mut contract, &trace);
    let word = trace.facts.word(
        &Request::new(1, false)
            .on_stream(Stream::Video)
            .in_reading("vae.decode"),
    );
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
    assert_eq!(boxes.len(), 1, "one clip in, one clip out");
    drop(shell);
    (values, boxes[0], load_s, fire_s)
}

#[test]
#[ignore = "needs a CUDA device, a Lightricks/LTX-2.5-Diffusers snapshot with a vae/ and the ltx2_golden.py --vae dump under PIE_IMAGEGEN_GOLDEN"]
fn the_decoder_answers_the_reference_in_one_fire() {
    assert!(engine_cuda::device::present(), "no CUDA device");
    let root = snapshot(
        "Lightricks/LTX-2.5-Diffusers",
        &["vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
    );
    let name = std::env::var("PIE_LTX2_VAE_GOLDEN").unwrap_or_else(|_| "ltx2_vae".to_string());
    let gold = golden(&format!("ltx25/{name}"));
    let shapes = shapes(&gold);
    let (latent_box, latent_c) = boxed(&shapes, "latent");
    let (pixel_box, pixel_c) = boxed(&shapes, "pixels");
    assert_eq!(latent_c, VAE_Z as usize);
    assert_eq!(pixel_c, VAE_RGB as usize);
    let latent = f32s(&gold.join("latent.f32"));
    let pixels = f32s(&gold.join("pixels.f32"));
    let [frames, hp, wp] = pixel_box;
    let [t_lat, hl, wl] = latent_box;
    assert_eq!(
        frames,
        8 * t_lat - 7,
        "an LTX clip is 8T - 7 frames; the golden says otherwise"
    );
    assert_eq!((hp, wp), (32 * hl, 32 * wl));
    let voxels = (t_lat * hl * wl) as usize;
    let out_plane = (hp * wp) as usize;
    assert_eq!(latent.len(), voxels * latent_c);
    assert_eq!(pixels.len(), out_plane * frames as usize * pixel_c);

    let (got, out_box, load_s, fire_s) = fire(&root, voxels as u32 + 8, latent_box, &latent);
    eprintln!(
        "ltx vae: load {load_s:.1} s, fire {fire_s:.3} s, {t_lat}x{hl}x{wl} latent -> {}x{}x{} pixels",
        out_box[0], out_box[1], out_box[2]
    );
    assert_eq!(
        out_box, pixel_box,
        "{t_lat} latent frames land {frames} frames of {hp}x{wp}"
    );
    assert_eq!(got.len(), pixels.len());

    let whole = score(&got, &pixels);
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "decode {t_lat}x{hl}x{wl} -> {frames}x{hp}x{wp}: cos {:.6}, mean |err| {:.5}, \
         max |err| {:.4}, range [{lo:.3}, {hi:.3}]",
        whole.cos, whole.mean_abs, whole.max_abs
    );
    let per_frame: Vec<Score> = (0..frames as usize)
        .map(|f| {
            let at = f * out_plane * pixel_c;
            score(
                &got[at..at + out_plane * pixel_c],
                &pixels[at..at + out_plane * pixel_c],
            )
        })
        .collect();
    for (f, s) in per_frame.iter().enumerate() {
        eprintln!(
            "  frame {f:2}: cos {:.6}, mean |err| {:.5}, max |err| {:.4}",
            s.cos, s.mean_abs, s.max_abs
        );
    }
    for (f, s) in per_frame.iter().enumerate() {
        assert!(
            s.cos >= 0.9999 && s.mean_abs <= 0.005,
            "frame {f} drifts from the reference: cos {}, mean |err| {} — an end frame alone \
             is the replicate time padding or the anchor drop; a middle frame is a conv or \
             a shuffle",
            s.cos,
            s.mean_abs
        );
    }
    assert!(
        whole.cos >= 0.9999 && whole.mean_abs <= 0.005,
        "the clip drifts from the reference: cos {}, mean |err| {}",
        whole.cos,
        whole.mean_abs
    );
}
