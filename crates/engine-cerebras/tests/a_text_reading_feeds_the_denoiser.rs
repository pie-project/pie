//! flux2-mini's text encoder through the cerebras engine: its `text` reading compiled on load
//! from the trace alone, the guest's text pass through `CerebrasEngine::submit`, its rows the
//! context the denoise loop takes. On the host fabric the rows are engine-cpu's bit for bit and
//! within the bf16 tolerance of transformers'; `flux_2_minis_text_on_the_fabric` (ignored) runs
//! the pass on a fabric (`READING_CACHE`, `READING_OUT`).

mod common;

use std::path::Path;

use common::{MINI, rows};
use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::fire::{Attachment, FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use engine_cpu::Forward;
use eta_dsl::prelude::*;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use poem_ir::Request;

/// transformers' rows of the 16-token prompt below.
const TEXT_GOLDEN: &[u8] = include_bytes!("flux_2_mini_text16.f32");

/// "a red fox sleeping" in the chat template, no thinking: 16 tokens of Qwen3's vocabulary.
const PROMPT: [u32; 16] = [
    151_644, 872, 198, 64, 2518, 38835, 21127, 151_645, 198, 151_644, 77091, 198, 151_667, 271,
    151_668, 271,
];

const HIDDEN: usize = 192;

/// diffusers' latent after the prompt's 2-step loop at 16 + 16 rows.
const PIPE_GOLDEN: &[u8] = include_bytes!("flux_2_mini_pipe16_latent2.f32");

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
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

/// The engine, its programs compiled on load into `cache` at `rows()` rows a class.
fn engine_at(cache: &Path, fabric: Option<&str>) -> engine_cerebras::CerebrasEngine {
    let mut cfg = engine_cerebras::Config {
        cache_dir: Some(cache.to_path_buf()),
        num_threads: Some(8),
        // a system's programs are linked against its whole wafer (`prepare::stream_programs`)
        cmaddr: std::env::var("READING_CMADDR").ok(),
        ..engine_cerebras::Config::default()
    };
    cfg.wafer.stream_rows = Some(rows() as u32);
    cfg.debug = Some(engine_cerebras::config::Debug {
        stream_fabric: fabric.map(str::to_string),
        ..engine_cerebras::config::Debug::default()
    });
    engine_cerebras::CerebrasEngine::new(cfg, common::contract_for)
}

fn load(engine: &mut engine_cerebras::CerebrasEngine, ckpt: &Path) -> engine::caps::Capabilities {
    let grid = common::grid();
    engine
        .load(LoadRequest {
            trace: common::trace(),
            checkpoint: Checkpoint::Path(ckpt.to_path_buf()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 2 * rows() as u32,
                buckets: Vec::new(),
                max_adapters: 0,
                page_size: 16,
                max_context: 4 * rows() as u32,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: Some(grid[1] * grid[2]),
                max_clips: Some(1),
            },
            residency: Residency::default(),
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("flux2-mini loads, its stream programs compiled on load")
        .caps
}

/// The guest's text pass over `tokens`: an epilogue that puts the `hidden` rows read out in a
/// channel. Returns them.
fn text_pass(engine: &mut engine_cerebras::CerebrasEngine, tokens: &[u32]) -> Vec<f32> {
    let trace = common::trace();
    let n = tokens.len() as u32;
    let out = Channel::new([n * HIDDEN as u32], eta_dsl::dtype::f32);
    let o = out.clone();
    let decl = OutputDecl {
        name: "hidden".into(),
        axis: Axis::Row,
        width: HIDDEN as u32,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    };
    let mut b = Builder::new(32, 16);
    b.outputs(vec![vec![decl.clone()]]);
    b.stage(Stage::Epilogue, move || {
        let h = intrinsics::output_rows(0, "hidden", n);
        o.put(reshape(&h, [n * HIDDEN as u32]));
    });
    let traced = b.build().expect("the text pass traces");
    let container = traced.container().clone();
    let bound = eta_ir::validate::bind(container.clone(), vec![vec![decl]], EngineCaps::default())
        .expect("the text pass binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: bound.hash,
            launch,
            ..Default::default()
        })
        .expect("the text pass registers");
    let order = traced.channel_order().to_vec();
    let mut channels = Vec::new();
    for (i, decl) in container.channels.iter().enumerate() {
        let id = ChannelId(200 + i as u64 + 100 * u64::from(n));
        engine
            .register_channel(&ChannelRegistration {
                id,
                shape: decl.shape.dims().to_vec(),
                dtype: decl.dtype,
                host_role: decl.host_role,
                seeded: decl.seeded,
                extern_dir: None,
                capacity: decl.capacity.max(1),
                extern_name: Vec::new(),
            })
            .expect("a channel registers");
        channels.push(id);
    }
    let instance = engine
        .bind_instance(&InstanceBinding {
            program,
            channels,
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            device_resolved: Default::default(),
            extents: engine::program::BindExtents {
                row_count: n,
                token_count: n,
                sampled_rows: n,
                query_len: n,
                ..engine::program::BindExtents::default()
            },
        })
        .expect("the text pass binds")
        .id;
    let word = trace
        .facts
        .word(&Request::new(n, false).switched("reading", "text"));
    let frame = FrameSubmission::of(Step {
        lanes: vec![Lane {
            slot: 0,
            word,
            tokens: tokens.to_vec(),
            readout: Readout::Rows((0..n).collect()),
            group: Some(0),
            kv_less: true,
            ..Lane::default()
        }],
        attachments: vec![Attachment {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }],
        media: Vec::new(),
        voxels: Vec::new(),
    });
    let mut ticket = engine.submit(&frame).expect("the text pass runs");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let at = order
        .iter()
        .position(|&g| g == out.gid())
        .expect("the out channel") as u32;
    f32s(
        &engine
            .take_channel(instance, at)
            .expect("the channel reads")
            .expect("the epilogue put the rows"),
    )
}

/// engine-cpu's text rows of `tokens`, from the drawn weights.
fn host_text(tokens: &[u32]) -> Vec<f32> {
    let trace = common::trace();
    let mut f = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let word = trace
        .facts
        .word(&Request::new(tokens.len() as u32, false).switched("reading", "text"));
    let step = engine_cpu::forward::StreamStep {
        lanes: vec![engine_cpu::forward::StreamLane {
            word,
            rows: tokens.len(),
            clip: None,
        }],
        inputs: Vec::new(),
        tokens: tokens.to_vec(),
    };
    match f
        .stream(&step)
        .expect("the text step runs")
        .into_iter()
        .find(|o| o.name == "hidden")
    {
        Some(engine_cpu::forward::Output {
            values: engine_cpu::forward::Values::F32(v),
            ..
        }) => v,
        _ => panic!("no hidden rows"),
    }
}

/// The text pass on the host fabric, compiled on load: engine-cpu's rows bit for bit,
/// transformers' within the bf16 tolerance; a second load compiles nothing.
#[test]
fn flux_2_minis_text_runs_through_the_engine() {
    if rows() != 16 {
        eprintln!("the prompt is 16 tokens: READING_ROWS=16 runs this");
        return;
    }
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (cache, ckpt) = (tmp.path().join("cache"), tmp.path().join("ckpt"));
    common::checkpoint(&common::trace(), &ckpt);
    let mut engine = engine_at(&cache, Some("host"));
    let _ = load(&mut engine, &ckpt);
    let got = text_pass(&mut engine, &PROMPT);
    let want = host_text(&PROMPT);
    assert_eq!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "the text rows"
    );
    let (rel, cos) = rel_cos(&got, &f32s(TEXT_GOLDEN));
    eprintln!("flux2-mini's text rows vs transformers: rel L2 {rel:.3e}, cosine {cos:.8}");
    assert!(rel <= 2e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
    // a 15-token prompt: the program's 16th row is padding
    let short = &PROMPT[..15];
    let got = text_pass(&mut engine, short);
    assert_eq!(got.len(), 15 * HIDDEN);
    assert_eq!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        host_text(short)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "a short prompt's rows"
    );
    drop(engine);
    let programs = std::fs::read_dir(cache.join("wafer"))
        .expect("the cache")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("s-"))
        .count();
    assert_eq!(programs, 1, "one entry of stream programs");
    let mut again = engine_at(&cache, Some("host"));
    let _ = load(&mut again, &ckpt);
    let after = std::fs::read_dir(cache.join("wafer"))
        .expect("the cache")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("s-"))
        .count();
    assert_eq!(after, 1, "the second load finds the programs");
}

/// The host's own denoise loop over the text rows `context` (engine-cpu's whole step a
/// velocity), each step's latent.
fn host_loop(steps: u32, context: &[f32]) -> Vec<Vec<f32>> {
    let trace = common::trace();
    let mut forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let image_rows = common::grid()[1] * common::grid()[2] / 4;
    let d = engine_cerebras::reading::Denoise::of(MINI, steps, Some(image_rows))
        .expect("a flow schedule");
    let mut inputs = common::inputs();
    let mut ctx = context.to_vec();
    ctx.resize((rows() + image_rows as usize) * HIDDEN, 0.0);
    for (n, v) in &mut inputs {
        if n == "context" {
            v.clone_from(&ctx);
        }
    }
    let word = |stream: &str| {
        trace.facts.word(
            &Request::new(rows() as u32, false)
                .switched("reading", "denoise")
                .switched("stream", stream),
        )
    };
    let (_, vs) = d
        .run(&mut inputs, 2, |_, ins| {
            let step = engine_cpu::forward::StreamStep {
                lanes: vec![
                    engine_cpu::forward::StreamLane {
                        word: word("context"),
                        rows: rows(),
                        clip: None,
                    },
                    engine_cpu::forward::StreamLane {
                        word: word("latents"),
                        rows: image_rows as usize,
                        clip: Some(common::grid()),
                    },
                ],
                inputs: ins.iter().map(|(n, v)| (n.as_str(), v.clone())).collect(),
                tokens: Vec::new(),
            };
            Ok(
                match forward
                    .stream(&step)?
                    .into_iter()
                    .find(|o| o.name == "velocity")
                {
                    Some(engine_cpu::forward::Output {
                        values: engine_cpu::forward::Values::F32(v),
                        ..
                    }) => v,
                    _ => return Err("no velocity".into()),
                },
            )
        })
        .expect("the host's loop runs");
    let mut x = common::latents();
    let mut out = Vec::new();
    for (i, v) in vs.iter().enumerate() {
        let dt = d.sigmas[i + 1] - d.sigmas[i];
        for (x, v) in x.iter_mut().zip(v) {
            *x += dt * v;
        }
        out.push(x.clone());
    }
    out
}

/// The text pass's rows as the denoise loop's context on the host fabric: the host's own loop
/// bit for bit, and diffusers' pipeline within the bf16 tolerance.
#[test]
fn the_text_rows_feed_the_denoise_loop() {
    if rows() != 16 {
        eprintln!("the prompt is 16 tokens: READING_ROWS=16 runs this");
        return;
    }
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (cache, ckpt) = (tmp.path().join("cache"), tmp.path().join("ckpt"));
    common::checkpoint(&common::trace(), &ckpt);
    let mut engine = engine_at(&cache, Some("host"));
    let caps = load(&mut engine, &ckpt);
    let ctx = text_pass(&mut engine, &PROMPT);
    let ran = common::guest::loop_on(&mut engine, &caps, 2, &ctx);
    let want = host_loop(2, &host_text(&PROMPT));
    for (i, (got, want)) in ran.latents.iter().zip(&want).enumerate() {
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the latent after step {i}"
        );
    }
    let (rel, cos) = rel_cos(ran.latents.last().expect("a latent"), &f32s(PIPE_GOLDEN));
    eprintln!(
        "prompt -> text -> 2 denoise steps vs diffusers' pipeline: rel L2 {rel:.3e}, cosine {cos:.8}"
    );
    assert!(rel <= 2e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
}

/// The text's own 16 rows as the context lane of a 64-row class: the engine pads the lane with
/// zero rows, and the loop lands on the host's own over the padded rows, bit for bit.
#[test]
fn a_short_context_lane_pads_to_its_class() {
    if rows() != 64 {
        eprintln!(
            "the programs' classes are 64 rows here: READING_ROWS=64 (the default) runs this"
        );
        return;
    }
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (cache, ckpt) = (tmp.path().join("cache"), tmp.path().join("ckpt"));
    common::checkpoint(&common::trace(), &ckpt);
    let mut engine = engine_at(&cache, Some("host"));
    let caps = load(&mut engine, &ckpt);
    let ctx = text_pass(&mut engine, &PROMPT);
    assert_eq!(ctx.len(), PROMPT.len() * HIDDEN, "the text pass's own rows");
    let ran = common::guest::loop_on_rows(&mut engine, &caps, 2, &ctx, PROMPT.len() as u32);
    let want = host_loop(2, &host_text(&PROMPT));
    for (i, (got, want)) in ran.latents.iter().zip(&want).enumerate() {
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the latent after step {i}"
        );
    }
}

/// diffusers' decode of the pipeline's latent above: 64 x 64 pixels, in [-1, 1].
const VAE_GOLDEN: &[u8] = include_bytes!("flux_2_mini_vae16_pixels.f32");
const PIPE_LATENT: &[u8] = include_bytes!("flux_2_mini_pipe16_latent2.f32");

/// The `vae.decode` reading on engine-cpu's forward over the landed checkpoint: diffusers'
/// `AutoencoderKLFlux2` decode of the same latent, within the bf16 tolerance.
#[test]
fn flux_2_minis_vae_decode_answers_diffusers() {
    let trace = common::trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let ckpt = tmp.path().join("ckpt");
    common::checkpoint(&trace, &ckpt);
    let contract = common::contract_for(&trace, &ckpt).expect("the contract");
    let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt).expect("it lands");
    let mut f = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        let p =
            <engine_cerebras::prepare::Landed as kernels_cerebras::compile::tiles::Params>::get(
                &landed, n,
            )
            .unwrap_or_else(|| panic!("{n} landed"));
        let b = p.bytes;
        if b.len() == 4 * c {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        } else {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
                .collect()
        }
    })
    .expect("flux2-mini loads from its checkpoint");
    let word = trace.facts.word(
        &Request::new(0, false)
            .switched("reading", "vae.decode")
            .switched("stream", "latents"),
    );
    let step = engine_cpu::forward::StreamStep {
        lanes: vec![engine_cpu::forward::StreamLane {
            word,
            rows: 0,
            clip: Some([1, 8, 8]),
        }],
        inputs: vec![("latents", f32s(PIPE_LATENT))],
        tokens: Vec::new(),
    };
    let px = match f
        .stream(&step)
        .expect("the decode runs")
        .into_iter()
        .find(|o| o.name == "pixels")
    {
        Some(engine_cpu::forward::Output {
            values: engine_cpu::forward::Values::F32(v),
            ..
        }) => v,
        _ => panic!("no pixels"),
    };
    let want = f32s(VAE_GOLDEN);
    assert_eq!(px.len(), want.len());
    let (rel, cos) = rel_cos(&px, &want);
    let mse: f64 = px
        .iter()
        .zip(&want)
        .map(|(a, b)| {
            let (a, b) = (f64::from(a.clamp(-1.0, 1.0)), f64::from(b.clamp(-1.0, 1.0)));
            ((a - b) / 2.0).powi(2)
        })
        .sum::<f64>()
        / px.len() as f64;
    let psnr = -10.0 * mse.log10();
    eprintln!(
        "vae.decode on the host vs diffusers: rel L2 {rel:.3e}, cosine {cos:.8}, PSNR {psnr:.1} dB"
    );
    assert!(rel <= 2e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
}
