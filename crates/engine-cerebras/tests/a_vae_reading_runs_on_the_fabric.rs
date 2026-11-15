//! flux2-mini's VAE decode on the wafer (GENERIC.md M1i): its `vae.decode` reading compiled by the
//! grid compiler (`kernels_cerebras::stream::grid`: passes over one unit, every convolution a
//! gather and a matvec, every group norm a reduction over its input's rows), from the trace
//! alone, the static arrays from the checkpoint and the entries from the engine.
//!
//! - On the host, the compile's own program (`grid::emulate`: its passes, each weight PE's tile
//!   from the compile's spans, its gathers' taps) is engine-cpu's decode within the bf16
//!   tolerance, and diffusers' `AutoencoderKLFlux2` (`mk_tiny_flux2_pipe.py --vae`) as engine-cpu
//!   is.
//! - `flux_2_minis_vae_on_the_fabric` (ignored) runs it on the simulator (or a system):
//!   compiled and linked into `READING_OUT/vae`, the pixels written to `READING_OUT/pixels.f32`
//!   and held to engine-cpu's and diffusers'. `READING_CLIP=t,h,w` decodes a smaller latent of
//!   drawn values (against engine-cpu alone).

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;

use engine_cpu::Forward;
use engine_cpu::forward::{StreamLane, StreamStep, Values};
use kernels_cerebras::stream::grid;
use poem_ir::ValueId;

/// diffusers' decode of the pipeline's latent (`mk_tiny_flux2_pipe.py --vae`): 64 x 64 pixels.
const VAE_GOLDEN: &[u8] = include_bytes!("flux_2_mini_vae16_pixels.f32");
const PIPE_LATENT: &[u8] = include_bytes!("flux_2_mini_pipe16_latent2.f32");

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

fn rel_cos(a: &[f32], b: &[f32]) -> (f64, f64) {
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum::<f64>()
    };
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    (
        (dot(&d, &d) / dot(b, b)).sqrt(),
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt(),
    )
}

/// The PSNR of two images in [-1, 1] (clamped), as 8-bit images' peak.
fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let (x, y) = (f64::from(x.clamp(-1.0, 1.0)), f64::from(y.clamp(-1.0, 1.0)));
            ((x - y) / 2.0).powi(2)
        })
        .sum::<f64>()
        / a.len() as f64;
    -10.0 * mse.log10()
}

/// engine-cpu's forward over the checkpoint as the engine lands it.
fn forward(trace: &poem_ir::Trace, ckpt: &Path) -> Forward {
    common::checkpoint(trace, ckpt);
    let contract = common::contract_for(trace, ckpt).expect("the contract");
    let landed = engine_cerebras::prepare::land(trace, &contract, ckpt).expect("it lands");
    Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        let p =
            <engine_cerebras::prepare::Landed as kernels_cerebras::tiles::Params>::get(&landed, n)
                .unwrap_or_else(|| panic!("{n} landed"));
        let b = p.bytes;
        if b.len() == 4 * c {
            f32s(b)
        } else {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
                .collect()
        }
    })
    .expect("flux2-mini loads from its checkpoint")
}

/// The decode's step: one lane of the clip, its latent.
fn step(word: u64, clip: [u32; 3], latents: &[f32]) -> StreamStep<'static> {
    StreamStep {
        lanes: vec![StreamLane {
            word,
            rows: 0,
            clip: Some(clip),
        }],
        inputs: vec![("latents", latents.to_vec())],
        tokens: Vec::new(),
    }
}

fn floats(c: &engine_cpu::forward::Captured) -> Vec<f32> {
    match &c.values {
        Values::F32(v) => v.clone(),
        Values::I32(v) => v.iter().map(|&x| x as f32).collect(),
    }
}

/// The clip and latent a run decodes: `READING_CLIP` (drawn values), else the pipeline's.
fn latent() -> ([u32; 3], Vec<f32>, bool) {
    match std::env::var("READING_CLIP") {
        Ok(c) => {
            let v: Vec<u32> = c.split(',').filter_map(|x| x.parse().ok()).collect();
            let clip = [v[0], v[1], v[2]];
            let n = (clip[0] * clip[1] * clip[2]) as usize * common::CHANNELS;
            (clip, common::units("vae.latents", n), false)
        }
        Err(_) => ([1, 8, 8], f32s(PIPE_LATENT), true),
    }
}

/// The program's decode on the host (`grid::emulate`) against engine-cpu's and diffusers'.
#[test]
fn flux_2_minis_vae_program_is_the_reference() {
    let trace = common::trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let mut f = forward(&trace, &tmp.path().join("ckpt"));
    let clip = [1, 8, 8];
    let g = grid::program(&trace, "vae.decode", clip, true).unwrap_or_else(|e| panic!("{e}"));
    let lat = f32s(PIPE_LATENT);
    let st = step(g.word, clip, &lat);
    let want: Vec<ValueId> = g.outs.clone();
    let reference: Vec<Vec<f32>> = f
        .stream_values(&st, &[], &want)
        .expect("engine-cpu decodes")
        .iter()
        .map(floats)
        .collect();
    let cache: std::cell::RefCell<BTreeMap<u32, grid::Rows>> =
        std::cell::RefCell::new(BTreeMap::new());
    let f = std::cell::RefCell::new(f);
    let got = grid::emulate(&g, &|v| {
        if let Some(x) = cache.borrow().get(&v) {
            return Ok(x.clone());
        }
        let c = f
            .borrow_mut()
            .stream_values(&st, &[], &[ValueId(v)])?
            .remove(0);
        let r = (c.rows, c.width, Rc::new(floats(&c)));
        cache.borrow_mut().insert(v, r.clone());
        Ok(r)
    })
    .unwrap_or_else(|e| panic!("{e}"));
    let px = &got[&want[0].0];
    let (rel, cos) = rel_cos(px, &reference[0]);
    let gold = f32s(VAE_GOLDEN);
    let (drel, dcos) = rel_cos(px, &gold);
    eprintln!(
        "the grid program on the host vs engine-cpu: rel L2 {rel:.3e}, cosine {cos:.8}, PSNR {:.1} dB; \
         vs diffusers: rel L2 {drel:.3e}, cosine {dcos:.8}, PSNR {:.1} dB",
        psnr(px, &reference[0]),
        psnr(px, &gold)
    );
    assert!(rel <= 1e-2 && cos >= 0.9999, "rel L2 {rel}, cosine {cos}");
    assert!(psnr(px, &gold) >= 40.0);
}

/// The decode on the simulator (or `READING_CMADDR`'s system): the program compiled and
/// linked into `READING_OUT/vae`, run through `Reading` as the engine does.
#[test]
#[ignore = "runs on the fabric simulator; set READING_OUT"]
fn flux_2_minis_vae_on_the_fabric() {
    use engine_cerebras::reading::{Fabric, Programs, Reading};
    let out = std::path::PathBuf::from(std::env::var("READING_OUT").expect("set READING_OUT"));
    let trace = common::trace();
    let mut f = forward(&trace, &out.join("ckpt"));
    let (clip, lat, pipe) = latent();
    // `READING_COMBINE`: co-resident with the text and denoise readings (16 rows a class), one
    // program, the decode launched by its reading's id
    let combine = std::env::var("READING_COMBINE").is_ok();
    let dir = out.join(if combine { "vae.decode" } else { "vae" });
    let t0 = std::time::Instant::now();
    let upto = std::env::var("READING_PASSES")
        .ok()
        .and_then(|n| n.parse().ok());
    // `READING_WEIGHT_WORDS`: a weight PE's words for tiles and stores, fewer than it holds (the
    // decode in weight groups, a launch each, as a reading too large for one is)
    let budget = std::env::var("READING_WEIGHT_WORDS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(grid::WEIGHT_WORDS);
    let g = grid::compile_within(&trace, "vae.decode", clip, true, upto, budget, &dir)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "compiled: {} passes over {} x {} roots, {} x {} PEs, weight groups {:?} ({:.1} s)",
        g.passes.len(),
        g.p,
        g.p,
        g.stream.w,
        g.stream.h,
        g.groups,
        t0.elapsed().as_secs_f64()
    );
    let cfg = engine_cerebras::Config::default();
    let (root, (w, h)) = if combine {
        let mut names = Vec::new();
        for r in ["text", "denoise"] {
            let o = kernels_cerebras::stream::Options {
                reading: r.into(),
                rows: 16,
                round16: true,
                upto: None,
                taps: Vec::new(),
            };
            kernels_cerebras::stream::compile_reading(&trace, &o, &out.join(r))
                .unwrap_or_else(|e| panic!("{r}: {e}"));
            names.push(r.to_string());
        }
        names.push("vae.decode".into());
        let placed = kernels_cerebras::stream::coresident::combine(&out, &names)
            .unwrap_or_else(|e| panic!("{e}"));
        eprintln!("co-resident: {placed:?}");
        let fab = std::fs::read_to_string(out.join("fabric.txt")).expect("fabric.txt");
        let v: Vec<u32> = fab
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        (out.clone(), (v[0], v[1]))
    } else {
        (dir.clone(), (g.stream.w, g.stream.h))
    };
    let link = root.join("out");
    if !link.join("out.json").is_file() {
        let ops = dir.join("ops");
        std::fs::create_dir_all(&ops).expect("ops");
        for (n, src) in kernels_cerebras::sources::OPS
            .iter()
            .chain(kernels_cerebras::sources::STREAM_OPS)
        {
            std::fs::write(ops.join(n), src).expect("an op");
        }
        let cslc = kernels_cerebras::toolchain::Cslc::find(&cfg.toolchain).expect("cslc");
        let (fabric_dims, fabric_offsets) = kernels_cerebras::toolchain::memcpy_fabric(w, h);
        let t0 = std::time::Instant::now();
        cslc.compile(&kernels_cerebras::toolchain::Compile {
            arch: kernels_cerebras::toolchain::Arch::Wse3,
            layout: root.join("layout.csl"),
            fabric_dims,
            fabric_offsets,
            params: Vec::new(),
            memcpy: true,
            channels: 1,
            import_paths: Vec::new(),
            out: link.clone(),
            bf16: true,
        })
        .expect("cslc links the program");
        eprintln!("linked ({:.1} s)", t0.elapsed().as_secs_f64());
    }
    let reading = Reading::open(&dir).expect("the reading");
    let st = step(g.word, clip, &lat);
    let statics = reading.statics(&mut f, &st).expect("the static arrays");
    let mut host = Forward::host_for(
        trace.clone(),
        &|n: &str, c: usize| common::plane(&trace, n, c),
        &reading.entry_values(),
    )
    .expect("the host forward");
    let entries = reading.entries(&mut host, &st).expect("the entries");
    let platform = match std::env::var("READING_CMADDR") {
        Ok(cmaddr) => engine_cerebras::sdk::Platform::System { cmaddr },
        Err(_) => engine_cerebras::sdk::Platform::Simulator(engine_cerebras::sdk::Simulator {
            num_threads: 8,
            ..engine_cerebras::sdk::Simulator::new(engine_cerebras::sdk::Target::Wse3)
        }),
    };
    let mut fab = Programs {
        sdk_lib: cfg.toolchain.sdk_lib(),
        platform,
        rt: None,
    };
    fab.switch(&link).expect("the program loads");
    reading
        .write_statics(&mut fab, &statics)
        .expect("the static arrays written");
    let t0 = std::time::Instant::now();
    let (got, cycles) = reading
        .launch_with(&mut fab, &st, &entries, &statics)
        .expect("the decode runs");
    eprintln!(
        "the decode: {cycles:?} cycles, {:.0} s",
        t0.elapsed().as_secs_f64()
    );
    let back = reading.rows_back(&got).expect("the rows back");
    if upto.is_some() {
        // a part of the decode: its last value against the compile's own program on the host
        let cache: std::cell::RefCell<BTreeMap<u32, grid::Rows>> =
            std::cell::RefCell::new(BTreeMap::new());
        let f = std::cell::RefCell::new(f);
        let want = grid::emulate(&g, &|v| {
            if let Some(x) = cache.borrow().get(&v) {
                return Ok(x.clone());
            }
            let c = f
                .borrow_mut()
                .stream_values(&st, &[], &[ValueId(v)])?
                .remove(0);
            let r = (c.rows, c.width, Rc::new(floats(&c)));
            cache.borrow_mut().insert(v, r.clone());
            Ok(r)
        })
        .expect("the program emulated");
        let got = floats(&back[0]);
        let want = &want[&g.outs[0].0];
        let (rel, cos) = rel_cos(&got, want);
        println!(
            "READING_VAE pass {:?}'s value %{} vs the host's program: rel L2 {rel:.3e}, cosine {cos:.8}",
            upto, g.outs[0].0
        );
        // (the attention's 16-bit probabilities, amplified by the norms after it: a few keys
        // of a small clip average none of their roundings)
        assert!(rel <= 3e-2, "rel L2 {rel}");
        return;
    }
    let px = reading
        .finish(&mut host, &trace, &st, &back, &["pixels"])
        .expect("the pixels")
        .remove(0);
    let bytes: Vec<u8> = px.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(out.join("pixels.f32"), bytes).expect("pixels written");
    let reference = floats(
        &f.stream_values(&st, &[], &reading.out_values())
            .expect("engine-cpu decodes")[0],
    );
    let (rel, cos) = rel_cos(&px, &reference);
    println!(
        "READING_VAE vs engine-cpu: rel L2 {rel:.3e}, cosine {cos:.8}, PSNR {:.1} dB",
        psnr(&px, &reference)
    );
    if pipe {
        let gold = f32s(VAE_GOLDEN);
        let (rel, cos) = rel_cos(&px, &gold);
        println!(
            "READING_VAE vs diffusers: rel L2 {rel:.3e}, cosine {cos:.8}, PSNR {:.1} dB",
            psnr(&px, &gold)
        );
    }
    assert!(rel <= 2e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
}

/// The program's host side (its entries, statics and rows back) on the host fabric, as the
/// engine serves it: engine-cpu's decode, the rows read back laid out as the program's.
#[test]
fn flux_2_minis_vae_reading_serves_on_the_host_fabric() {
    use engine_cerebras::reading::{HostFabric, Reading};
    let trace = common::trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let mut f = forward(&trace, &tmp.path().join("ckpt"));
    let dir = tmp.path().join("vae");
    let clip = [1, 8, 8];
    grid::compile(&trace, "vae.decode", clip, true, &dir).unwrap_or_else(|e| panic!("{e}"));
    let reading = Reading::open(&dir).expect("the reading");
    let lat = f32s(PIPE_LATENT);
    let st = step(reading.classes[0].word, clip, &lat);
    let mut host = Forward::host_for(
        trace.clone(),
        &|n: &str, c: usize| common::plane(&trace, n, c),
        &reading.entry_values(),
    )
    .expect("the host forward");
    let entries = reading.entries(&mut host, &st).expect("the entries");
    let mut fab = HostFabric::new(forward(&trace, &tmp.path().join("ckpt2")));
    let (got, _) = reading
        .launch_on(&mut fab, &st, &entries)
        .expect("the decode runs");
    let back = reading.rows_back(&got).expect("the rows back");
    let px = reading
        .finish(&mut host, &trace, &st, &back, &["pixels"])
        .expect("the pixels")
        .remove(0);
    let reference = floats(
        &f.stream_values(&st, &[], &reading.out_values())
            .expect("engine-cpu decodes")[0],
    );
    assert_eq!(px.len(), reference.len());
    let (rel, _) = rel_cos(&px, &reference);
    assert!(rel == 0.0, "rel L2 {rel}");
    let gold = f32s(VAE_GOLDEN);
    eprintln!("host fabric vs diffusers: PSNR {:.1} dB", psnr(&px, &gold));
}
