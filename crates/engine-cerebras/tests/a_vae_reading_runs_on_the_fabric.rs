//! flux2-mini's VAE decode on the wafer: its `vae.decode` reading compiled by the grid compiler
//! (`kernels_cerebras::compile::stream::grid`) from the trace alone. On the host, the compile's own
//! program (`grid::emulate`) is engine-cpu's decode within the bf16 tolerance;
//! `flux_2_minis_vae_on_the_fabric` (ignored) runs it on a fabric (`READING_OUT`, `READING_CLIP`).

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;

use engine_cpu::Forward;
use engine_cpu::forward::{StreamLane, StreamStep, Values};
use kernels_cerebras::compile::stream::{grid, tiled};
use poem_ir::ValueId;

/// diffusers' decode of the pipeline's latent: 64 x 64 pixels.
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
            <engine_cerebras::prepare::Landed as kernels_cerebras::compile::tiles::Params>::get(
                &landed, n,
            )
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

/// The tiled program whose op table streams in with its weights and whose buffers are packed
/// tightly decodes bit for bit as the one holding its table.
#[test]
fn a_tiled_program_streaming_its_op_table_and_packed_tightly_decodes_alike() {
    let trace = common::trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let f = forward(&trace, &tmp.path().join("ckpt"));
    let clip = [1, 8, 8];
    let held = tiled::program(&trace, "vae.decode", clip, true, Some((2, 2)), None)
        .unwrap_or_else(|e| panic!("{e}"));
    let streamed = tiled::program_with(
        &trace,
        "vae.decode",
        clip,
        true,
        (2, 2),
        tiled::Chunks {
            stream_ops: true,
            tight: true,
            ..tiled::Chunks::default()
        },
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(held.op_words.is_empty());
    assert_eq!(streamed.op_words.len(), streamed.ops.len());
    assert!(streamed.stream_words > held.stream_words);
    assert!(streamed.arena <= held.arena && streamed.hx <= held.hx);
    let lat = f32s(PIPE_LATENT);
    let st = step(held.word, clip, &lat);
    let f = std::cell::RefCell::new(f);
    let get = |v: u32| -> Result<grid::Rows, String> {
        let c = f
            .borrow_mut()
            .stream_values(&st, &[], &[ValueId(v)])?
            .remove(0);
        Ok((c.rows, c.width, Rc::new(floats(&c))))
    };
    let a = tiled::emulate(&held, &get).unwrap_or_else(|e| panic!("{e}"));
    let b = tiled::emulate(&streamed, &get).unwrap_or_else(|e| panic!("{e}"));
    let o = held.outs[0].0;
    assert_eq!(a[&o], b[&o], "the streamed table's decode differs");
}

/// The tiled program's decode on the host (`tiled::emulate`) against engine-cpu's and
/// diffusers'.
#[test]
fn flux_2_minis_tiled_vae_program_is_the_reference() {
    let trace = common::trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let mut f = forward(&trace, &tmp.path().join("ckpt"));
    let clip = [1, 8, 8];
    let g = tiled::program(&trace, "vae.decode", clip, true, Some((2, 2)), None)
        .unwrap_or_else(|e| panic!("{e}"));
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
    let got = tiled::emulate(&g, &|v| {
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
        "the tiled program on the host vs engine-cpu: rel L2 {rel:.3e}, cosine {cos:.8}, PSNR {:.1} dB; \
         vs diffusers: rel L2 {drel:.3e}, cosine {dcos:.8}, PSNR {:.1} dB",
        psnr(px, &reference[0]),
        psnr(px, &gold)
    );
    assert!(rel <= 1e-2 && cos >= 0.9999, "rel L2 {rel}, cosine {cos}");
    assert!(psnr(px, &gold) >= 40.0);
}

/// The program's host side on the host fabric, as the engine serves it: engine-cpu's decode.
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

/// The tiled program's host side on the host fabric, as engine-cpu's decode.
#[test]
fn flux_2_minis_tiled_vae_reading_serves_on_the_host_fabric() {
    use engine_cerebras::reading::{HostFabric, Reading};
    let trace = common::trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let mut f = forward(&trace, &tmp.path().join("ckpt"));
    let dir = tmp.path().join("vae");
    let clip = [1, 8, 8];
    tiled::compile(
        &trace,
        "vae.decode",
        clip,
        true,
        Some((2, 2)),
        None,
        None,
        &dir,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let reading = Reading::open(&dir).expect("the reading");
    assert!(reading.d2h.iter().any(|b| b.w * b.h > 1));
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
}
