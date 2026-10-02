#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Recording, Shell};
use model_compiler::{Budget, VoxelLadder};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Stream, Value,
    trace_hybrid,
};
use models::ltx_2::forward::{Facts, VAE_DECODE, vae_decode};
use models::ltx_2::model::{Model, VAE_RGB, VAE_Z};

struct VaeOnly {
    model: Model,
}

impl ForwardHybrid for VaeOnly {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        self.model.caches()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let vae = self
            .model
            .vae
            .as_ref()
            .expect("the flagship carries the VAE");
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        vae_decode(&arms[usize::from(VAE_DECODE)], vae)
    }
}

fn hub() -> PathBuf {
    if let Some(hub) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(hub);
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
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

fn golden() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let name = std::env::var("PIE_LTX2_VAE_GOLDEN").unwrap_or_else(|_| "ltx2_vae".to_string());
    let dir = root.join("ltx25").join(name);
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

fn word() -> u64 {
    Facts::of(
        &Request::new(1, false)
            .on_stream(Stream::Video)
            .in_reading(VAE_DECODE),
    )
    .word()
}

fn fire(
    root: &PathBuf,
    max_voxels: u32,
    clip: [u32; 3],
    payload: &[f32],
) -> (Vec<f32>, [u32; 3], f64, f64) {
    let model = Model::ltx_2_5(Dtype::Bf16, 1);
    let src = checkpoint::file::diffusers::open(root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let mut contract = model
        .import_vae(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the VAE does not read the snapshot: {why}"));
    drop(src);
    let arm = VaeOnly { model };
    let trace = trace_hybrid("ltx25-vae-decode", &arm, Platform::Cuda);
    let mut keep: std::collections::BTreeSet<String> =
        trace.params.iter().map(|p| p.name.clone()).collect();
    loop {
        let more: Vec<String> = contract
            .tensors
            .iter()
            .filter(|t| keep.contains(&t.name))
            .flat_map(|t| t.expr.outputs().into_iter().map(str::to_string))
            .filter(|name: &String| !keep.contains(name))
            .collect();
        if more.is_empty() {
            break;
        }
        keep.extend(more);
    }
    contract.tensors.retain(|t| keep.contains(&t.name));
    let started = Instant::now();
    let mut shell = Shell::load(Boot {
        classify: |request| Facts::of(request).word(),
        trace,
        contract: &contract,
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
    let tokens = [0u32];
    let lanes = [Seated::of(Lane {
        slot: 0,
        word: word(),
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
fn the_decoder_answers_the_reference_in_one_fire() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the VAE parity gate: no CUDA device");
        return;
    }
    let Some(root) = snapshot() else {
        eprintln!(
            "skipping the VAE parity gate: no Lightricks/LTX-2.5-Diffusers snapshot with a vae/"
        );
        return;
    };
    let Some(gold) = golden() else {
        eprintln!("skipping the VAE parity gate: no ltx2_golden.py --vae dump");
        return;
    };
    let shapes: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("shapes.json")).expect("shapes.json"))
            .expect("shapes.json parses");
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
