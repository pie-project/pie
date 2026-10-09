#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

pub mod common_vae;

use common_vae::{Score, artifact, bf16_bytes, boxed, deploy, f32s, golden, score, shapes};
use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Lane, Shell};
use poem::{Platform, Request, Stream, Trace};

/// The VAE's encode readings alone, by the package's own functions.
const VAE_ONLY: &str = r#"
def forward(m, inputs):
    inputs.reading("vae.encode.head", lambda rows: vae_encode(rows, m.vae.enc, True))
    return inputs.reading("vae.encode", lambda rows: vae_encode(rows, m.vae.enc, False))
"#;

fn vae_only() -> Trace {
    models::star::replacing("wan22-ti2v-5b", "forward.star", "forward", VAE_ONLY)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"))
        .trace(
            "wan22-ti2v-5b",
            &deploy("wan22-ti2v-5b-bf16-kv-bf16"),
            "wan22-vae-encode",
            Platform::Cuda,
        )
        .unwrap_or_else(|why| panic!("the VAE-only plan: {why:#}"))
}

struct Encoder {
    shell: Shell,
}

impl Encoder {
    fn chunk(&mut self, first: bool, clip: [u32; 3], payload: &[f32]) -> (Vec<f32>, [u32; 3]) {
        let reading = if first {
            "vae.encode.head"
        } else {
            "vae.encode"
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
            .expect("the encode fire");
        assert_eq!(answered.len(), 1);
        let (values, boxes) = answered.remove(0);
        assert_eq!(boxes.len(), 1, "one clip in, one clip out");
        (values, boxes[0])
    }

    fn rewind(&mut self) {
        self.shell.open(0).expect("slot 0 opens");
    }
}

fn load(artifact: &Path, max_voxels: u32) -> (Encoder, f64) {
    let trace = vae_only();
    let src = ztensor::Source::open(artifact)
        .unwrap_or_else(|why| panic!("{}: {why}", artifact.display()));
    let contract = poem::import::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| {
            panic!(
                "{} does not hold every plane of the VAE encoder plan: {why}",
                artifact.display()
            )
        });
    drop(src);
    let started = Instant::now();
    let shell = common_vae::load(trace, &contract, artifact, max_voxels);
    (Encoder { shell }, started.elapsed().as_secs_f64())
}

#[test]
#[ignore = "needs a CUDA device, wan22-ti2v-5b.zt under PIE_IMAGEGEN_ARTIFACTS and the wan22_golden.py --vae encode dump under PIE_IMAGEGEN_GOLDEN"]
fn the_encoder_answers_the_reference_chunk_by_chunk() {
    assert!(engine_cuda::device::present(), "no CUDA device");
    let root = artifact("wan22-ti2v-5b.zt");
    let gold = golden("wan22/wan22_vae_encode");
    let shapes = shapes(&gold);
    let (pixel_box, pixel_c) = boxed(&shapes, "pixels");
    let (latent_box, latent_c) = boxed(&shapes, "latent");
    let pixels = f32s(&gold.join("pixels.f32"));
    let latent = f32s(&gold.join("latent.f32"));
    let raw_mean = f32s(&gold.join("mean.f32"));
    let [frames, hp, wp] = pixel_box;
    let [t_lat, hl, wl] = latent_box;
    assert_eq!(
        frames,
        4 * t_lat - 3,
        "a Wan clip is 4T - 3 frames; the golden says otherwise"
    );
    let in_plane = (hp * wp) as usize;
    let out_plane = (hl * wl) as usize;
    assert_eq!(pixels.len(), in_plane * frames as usize * pixel_c);
    assert_eq!(latent.len(), out_plane * t_lat as usize * latent_c);
    assert_eq!(raw_mean.len(), latent.len());
    let chunks: Vec<u32> = shapes["chunks"]
        .as_array()
        .expect("the chunk boundaries")
        .iter()
        .map(|v| v.as_u64().expect("a frame index") as u32)
        .collect();
    assert_eq!(chunks.len(), t_lat as usize + 1);

    let widest = chunks
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize * in_plane)
        .max()
        .expect("at least one chunk");
    let (mut vae, load_s) = load(&root, widest as u32 + 8);
    eprintln!("wan vae encode: load {load_s:.1} s, {frames} pixel frames of {hp}x{wp}");

    let mut got: Vec<f32> = Vec::with_capacity(latent.len());
    let mut per_chunk: Vec<(u32, Score)> = Vec::new();
    for k in 0..t_lat as usize {
        let first = k == 0;
        let (from, to) = (chunks[k] as usize, chunks[k + 1] as usize);
        let rows = &pixels[from * in_plane * pixel_c..to * in_plane * pixel_c];
        let started = Instant::now();
        let (out, out_box) = vae.chunk(first, [(to - from) as u32, hp, wp], rows);
        let fire_s = started.elapsed().as_secs_f64();
        assert_eq!(
            out_box,
            [1, hl, wl],
            "chunk {k} ({} pixel frames) lands ONE latent frame",
            to - from
        );
        let len = out_plane * latent_c;
        assert_eq!(out.len(), len);
        let s = score(&out, &latent[k * len..(k + 1) * len]);
        eprintln!(
            "  chunk {k} ({} arm, {} frames, {fire_s:.3} s): cos {:.6}, mean |err| {:.5}, \
             max |err| {:.4}",
            if first { "head" } else { "later" },
            to - from,
            s.cos,
            s.mean_abs,
            s.max_abs
        );
        per_chunk.push((k as u32, s));
        got.extend_from_slice(&out);
    }
    assert_eq!(
        got.len(),
        latent.len(),
        "the fires cover the golden's frames"
    );

    let whole = score(&got, &latent);
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "encode {frames}x{hp}x{wp} -> {t_lat}x{hl}x{wl}: cos {:.6}, mean |err| {:.5}, \
         max |err| {:.4}, range [{lo:.3}, {hi:.3}]",
        whole.cos, whole.mean_abs, whole.max_abs
    );
    for (k, s) in &per_chunk {
        assert!(
            s.cos >= 0.999 && s.mean_abs <= 0.05,
            "chunk {k} drifts from the reference: cos {}, mean |err| {} — a chunk that alone \
             is wrong is a frame cache that did not carry",
            s.cos,
            s.mean_abs
        );
    }
    assert!(
        whole.cos >= 0.999 && whole.mean_abs <= 0.05,
        "the clip drifts from the reference: cos {}, mean |err| {}",
        whole.cos,
        whole.mean_abs
    );

    let unnormalised = score(&got, &raw_mean);
    eprintln!(
        "against the RAW posterior mean: cos {:.6}, mean |err| {:.5}",
        unnormalised.cos, unnormalised.mean_abs
    );
    assert!(
        unnormalised.cos < whole.cos - 1e-3,
        "the arm's rows fit the raw posterior mean as well as the normalised latent \
         (cos {} against {}): `vae.encode` is meant to answer the DENOISER's space, and a \
         guest that hands this to `denoise` would be handing it the wrong numbers",
        unnormalised.cos,
        whole.cos
    );

    assert!(t_lat >= 2, "the caches can only be claimed past chunk 0");
    let k = (t_lat - 1) as usize;
    let (from, to) = (chunks[k] as usize, chunks[k + 1] as usize);
    let rows = &pixels[from * in_plane * pixel_c..to * in_plane * pixel_c];
    vae.rewind();
    let (cold, _) = vae.chunk(false, [(to - from) as u32, hp, wp], rows);
    let len = out_plane * latent_c;
    let cacheless = score(&cold, &latent[k * len..(k + 1) * len]);
    eprintln!(
        "cacheless chunk {k}: cos {:.6}, mean |err| {:.5} (the cached fire read {:.6})",
        cacheless.cos, cacheless.mean_abs, per_chunk[k].1.cos
    );
    assert!(
        cacheless.mean_abs > 0.05,
        "zeroing the frame caches moved the latent by mean |err| {}, inside the gate's own \
         tolerance: the `CacheRow::State` slabs are not reaching the causal convolutions and \
         this gate is measuring a cacheless encoder",
        cacheless.mean_abs
    );
}
