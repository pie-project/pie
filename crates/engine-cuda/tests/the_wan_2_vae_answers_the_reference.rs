#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

pub mod common_vae;

use common_vae::{Score, artifact, bf16_bytes, boxed, deploy, f32s, golden, score, shapes};
use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Lane, Shell};
use poem::{Platform, Request, Stream, Trace};

/// The VAE's decode readings alone, by the package's own functions.
const VAE_ONLY: &str = r#"
def forward(m, inputs):
    inputs.reading("vae.decode.head", lambda rows: vae_decode(rows, m.vae, True))
    return inputs.reading("vae.decode", lambda rows: vae_decode(rows, m.vae, False))
"#;

fn vae_only() -> Trace {
    poem_compiler::catalog::replacing("wan22-ti2v-5b", "forward.poem", "forward", VAE_ONLY)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"))
        .trace(
            "wan22-ti2v-5b",
            &deploy("wan22-ti2v-5b-bf16-kv-bf16"),
            "wan22-vae-decode",
            Platform::Cuda,
        )
        .unwrap_or_else(|why| panic!("the VAE-only plan: {why:#}"))
}

struct Decoder {
    shell: Shell,
}

impl Decoder {
    fn frame(&mut self, first: bool, clip: [u32; 3], payload: &[f32]) -> (Vec<f32>, [u32; 3]) {
        let reading = if first {
            "vae.decode.head"
        } else {
            "vae.decode"
        };
        let tokens = [0u32];
        let lanes = [Seated::of(Lane {
            slot: 0,
            word: self.shell.trace().facts.word(
                &Request::new(1, false)
                    .on_stream(Stream::Video)
                    .in_reading(reading),
            ),
            tokens: &tokens,
        })];
        let bytes = bf16_bytes(payload);
        let clips = [Clips {
            lane: 0,
            clips: &[clip],
            payload: &bytes,
        }];
        let mut answered = self
            .shell
            .fire_voxels(&lanes, &clips)
            .expect("the decode fire");
        assert_eq!(answered.len(), 1);
        let (values, boxes) = answered.remove(0);
        assert_eq!(boxes.len(), 1, "one clip in, one clip out");
        (values, boxes[0])
    }

    fn rewind(&mut self) {
        self.shell.open(0).expect("slot 0 opens");
    }
}

fn load(artifact: &Path, max_voxels: u32) -> (Decoder, f64) {
    let trace = vae_only();
    let src = ztensor::Source::open(artifact)
        .unwrap_or_else(|why| panic!("{}: {why}", artifact.display()));
    let contract = poem::import::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| {
            panic!(
                "{} does not hold every plane of the VAE plan: {why}",
                artifact.display()
            )
        });
    drop(src);
    let started = Instant::now();
    let shell = common_vae::load(trace, &contract, artifact, max_voxels);
    (Decoder { shell }, started.elapsed().as_secs_f64())
}

#[test]
#[ignore = "needs a CUDA device, wan22-ti2v-5b.zt under PIE_IMAGEGEN_ARTIFACTS and the wan22_golden.py --vae dump under PIE_IMAGEGEN_GOLDEN"]
fn the_decoder_answers_the_reference_frame_by_frame() {
    assert!(engine_cuda::device::present(), "no CUDA device");
    let root = artifact("wan22-ti2v-5b.zt");
    let gold = golden("wan22/wan22_vae");
    let shapes = shapes(&gold);
    let (latent_box, latent_c) = boxed(&shapes, "latent");
    let (pixel_box, pixel_c) = boxed(&shapes, "pixels");
    let latent = f32s(&gold.join("latent.f32"));
    let pixels = f32s(&gold.join("pixels.f32"));
    let [frames, hp, wp] = pixel_box;
    let [t_lat, hl, wl] = latent_box;
    assert_eq!(
        frames,
        4 * t_lat - 3,
        "a Wan clip is 4T - 3 frames; the golden says otherwise"
    );
    let plane = (hl * wl) as usize;
    let out_plane = (hp * wp) as usize;
    assert_eq!(latent.len(), plane * t_lat as usize * latent_c);
    assert_eq!(pixels.len(), out_plane * frames as usize * pixel_c);

    let (mut vae, load_s) = load(&root, plane as u32 + 8);
    eprintln!("wan vae: load {load_s:.1} s, {t_lat} latent frames of {hl}x{wl}");

    let mut got: Vec<f32> = Vec::with_capacity(pixels.len());
    let mut per_chunk: Vec<(u32, Score)> = Vec::new();
    let mut at = 0usize;
    for k in 0..t_lat {
        let first = k == 0;
        let rows = &latent[(k as usize) * plane * latent_c..(k as usize + 1) * plane * latent_c];
        let started = Instant::now();
        let (out, out_box) = vae.frame(first, [1, hl, wl], rows);
        let fire_s = started.elapsed().as_secs_f64();
        let want_frames = if first { 1 } else { 4 };
        assert_eq!(
            out_box,
            [want_frames, hp, wp],
            "latent frame {k} lands {want_frames} output frames"
        );
        let len = want_frames as usize * out_plane * pixel_c;
        assert_eq!(out.len(), len);
        let s = score(&out, &pixels[at..at + len]);
        eprintln!(
            "  frame {k} ({} arm, {fire_s:.3} s): cos {:.6}, mean |err| {:.5}, max |err| {:.4}",
            if first { "head" } else { "later" },
            s.cos,
            s.mean_abs,
            s.max_abs
        );
        per_chunk.push((k, s));
        got.extend_from_slice(&out);
        at += len;
    }
    assert_eq!(at, pixels.len(), "the fires cover the reference's frames");

    let whole = score(&got, &pixels);
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "decode {t_lat}x{hl}x{wl} -> {frames}x{hp}x{wp}: cos {:.6}, mean |err| {:.5}, \
         max |err| {:.4}, range [{lo:.3}, {hi:.3}]",
        whole.cos, whole.mean_abs, whole.max_abs
    );
    for (k, s) in &per_chunk {
        assert!(
            s.cos >= 0.999 && s.mean_abs <= 0.02,
            "latent frame {k} drifts from the reference: cos {}, mean |err| {} — a chunk \
             that alone is wrong is a frame cache that did not carry",
            s.cos,
            s.mean_abs
        );
    }
    assert!(
        whole.cos >= 0.999 && whole.mean_abs <= 0.02,
        "the clip drifts from the reference: cos {}, mean |err| {}",
        whole.cos,
        whole.mean_abs
    );

    assert!(t_lat >= 2, "the caches can only be claimed past frame 0");
    let k = (t_lat - 1) as usize;
    let rows = &latent[k * plane * latent_c..(k + 1) * plane * latent_c];
    vae.rewind();
    let (cold, _) = vae.frame(false, [1, hl, wl], rows);
    let warm_at = 1 + 4 * (k - 1);
    let warm = &pixels[warm_at * out_plane * pixel_c..(warm_at + 4) * out_plane * pixel_c];
    let cacheless = score(&cold, warm);
    eprintln!(
        "cacheless frame {k}: cos {:.6}, mean |err| {:.5} (the cached fire read {:.6})",
        cacheless.cos, cacheless.mean_abs, per_chunk[k].1.cos
    );
    assert!(
        cacheless.mean_abs > 0.02,
        "zeroing the frame caches moved the pixels by mean |err| {}, inside the gate's own \
         tolerance: the `CacheRow::State` slabs are not reaching the causal convolutions and \
         this gate is measuring a cacheless decoder",
        cacheless.mean_abs
    );

    vae.rewind();
    let (_, later_on_zero) = vae.frame(false, [1, hl, wl], &latent[..plane * latent_c]);
    assert_eq!(
        later_on_zero,
        [4, hp, wp],
        "the later-frames arm lands four frames whatever it is fed; only the head arm lands one"
    );
}
