#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Recording, Shell};
use poem::{Dtype, Platform, Request, Stream, Trace};
use poem_compiler::{Budget, VoxelLadder};

/// The VAE's decode readings alone, by the package's own functions.
const VAE_ONLY: &str = r#"
def forward(m, inputs):
    inputs.reading("vae.decode.head", lambda rows: vae_decode(rows, m.vae, True))
    return inputs.reading("vae.decode", lambda rows: vae_decode(rows, m.vae, False))
"#;

fn vae_only() -> Trace {
    let package = models::star::with_forward("wan22-ti2v-5b", VAE_ONLY)
        .unwrap_or_else(|why| panic!("the VAE-only package: {why}"));
    let deploy = poem::star::Deploy {
        weights: vec![Dtype::Bf16],
        kv: Dtype::Bf16,
        tp: 1,
        parts: vec![],
        drafter: None,
    };
    package
        .trace("wan22-ti2v-5b", &deploy, "wan22-vae-decode", Platform::Cuda)
        .unwrap_or_else(|why| panic!("the VAE-only plan: {why:#}"))
}

fn artifact() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen"));
    let file = root.join("wan22-ti2v-5b.zt");
    file.is_file().then_some(file)
}

fn golden() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let dir = root.join("wan22/wan22_vae");
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
        mean_abs: sum_abs / want.len().max(1) as f64,
    }
}

fn boxed(shapes: &serde_json::Value, key: &str) -> ([u32; 3], usize) {
    let at = &shapes[key];
    let get = |name: &str| at[name].as_u64().expect("a box extent") as u32;
    (
        [get("t"), get("h"), get("w")],
        at["channels"].as_u64().expect("channels") as usize,
    )
}

fn word(facts: &poem_ir::Facts, reading: &str) -> u64 {
    facts.word(
        &Request::new(1, false)
            .on_stream(Stream::Video)
            .in_reading(reading),
    )
}

struct Decoder {
    shell: Shell,
    facts: poem_ir::Facts,
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
            word: word(&self.facts, reading),
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

fn load(artifact: &PathBuf, max_voxels: u32) -> (Decoder, f64) {
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
    let mut shell = Shell::load(Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
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
        deferred_tier: true,
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("the VAE decoder does not load: {why}"));
    let load_s = started.elapsed().as_secs_f64();
    shell.open(0).expect("slot 0 opens");
    (
        Decoder {
            shell,
            facts: trace.facts,
        },
        load_s,
    )
}

#[test]
fn the_decoder_answers_the_reference_frame_by_frame() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the VAE parity gate: no CUDA device");
        return;
    }
    let Some(root) = artifact() else {
        eprintln!("skipping the VAE parity gate: no wan22-ti2v-5b.zt artifact");
        return;
    };
    let Some(gold) = golden() else {
        eprintln!("skipping the VAE parity gate: no wan22_golden.py --vae dump");
        return;
    };
    let shapes: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("shapes.json")).expect("shapes.json"))
            .expect("shapes.json parses");
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
