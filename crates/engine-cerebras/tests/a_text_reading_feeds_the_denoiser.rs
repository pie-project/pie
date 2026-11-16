//! flux2-mini's text encoder through the cerebras engine (GENERIC.md M1h): its `text` reading
//! (two Qwen3 layers over a prompt's tokens, causal) compiled by the stream compiler on load, as
//! the denoise reading is, from the trace alone, into the engine's cache; the guest's text pass
//! (the prompt's ids, the rows read out) through `CerebrasEngine::submit`, and its rows the
//! context the denoise loop takes.
//!
//! - On the host fabric (`stream_fabric = "host"`), the text rows are engine-cpu's forward of
//!   the same tokens bit for bit, a short prompt's too (the causal program's rows past the
//!   prompt are padding no earlier row reads), and within the bf16 tolerance of transformers'
//!   `Qwen3ForCausalLM` (`mk_tiny_flux2_pipe.py --text`); the denoise loop over them lands on
//!   the host's own loop over the same context bit for bit.
//! - `flux_2_minis_text_on_the_fabric` (ignored) runs the text pass on the simulator or a
//!   system: the programs compiled and linked on load (`READING_CACHE`), the rows written to
//!   `READING_OUT/text.f32` and held to transformers'.

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

/// transformers' rows of the 16-token prompt below (`mk_tiny_flux2_pipe.py --text`).
const TEXT_GOLDEN: &[u8] = include_bytes!("flux_2_mini_text16.f32");

/// "a red fox sleeping" in the chat template, no thinking (the text-to-media inferlet's
/// `first_user` + `cue`): 16 tokens of Qwen3's vocabulary.
const PROMPT: [u32; 16] = [
    151_644, 872, 198, 64, 2518, 38835, 21127, 151_645, 198, 151_644, 77091, 198, 151_667, 271,
    151_668, 271,
];

const HIDDEN: usize = 192;

/// diffusers' latent after the prompt's 2-step loop at 16 + 16 rows (`mk_tiny_flux2_pipe.py
/// --pipe "a red fox sleeping" draw 8,8 <sigmas>`).
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
        num_threads: 8,
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

/// The text pass on the host fabric, compiled on load: engine-cpu's rows bit for bit (a short
/// prompt padded by the causal program, too), transformers' within the bf16 tolerance; a
/// second load compiles nothing (the cache holds the programs).
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

/// The text pass on the fabric (the simulator, or the system at `READING_CMADDR`): the
/// programs compiled and linked on load into `READING_CACHE`, the checkpoint written to
/// `READING_OUT/ckpt`, the rows to `READING_OUT/text.f32`, held to transformers'.
#[test]
#[ignore = "runs on the fabric simulator; set READING_CACHE, READING_OUT, READING_ROWS=16"]
fn flux_2_minis_text_on_the_fabric() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let (cache, out) = (
        std::path::PathBuf::from(var("READING_CACHE")),
        std::path::PathBuf::from(var("READING_OUT")),
    );
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .try_init();
    let ckpt = out.join("ckpt");
    common::checkpoint(&common::trace(), &ckpt);
    let mut engine = engine_at(&cache, None);
    let t0 = std::time::Instant::now();
    let caps = load(&mut engine, &ckpt);
    println!(
        "READING_TEXT the load: {:.1} s (the programs compiled or found, opened, arrays written)",
        t0.elapsed().as_secs_f64()
    );
    // the simulator holds one program a process: `READING_CONTEXT` (a text pass's rows a run
    // before wrote) runs the denoise loop alone
    let got = if let Ok(ctx) = std::env::var("READING_CONTEXT") {
        f32s(&std::fs::read(ctx).expect("the text rows"))
    } else {
        let t0 = std::time::Instant::now();
        let got = text_pass(&mut engine, &PROMPT);
        println!(
            "READING_TEXT the text pass: {:.1} ms (its program opened at load)",
            t0.elapsed().as_secs_f64() * 1e3
        );
        std::fs::write(
            out.join("text.f32"),
            got.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("the rows");
        got
    };
    let (rel, cos) = rel_cos(&got, &host_text(&PROMPT));
    println!("READING_TEXT vs engine-cpu: rel L2 {rel:.3e}, cosine {cos:.8}");
    let (rel, cos) = rel_cos(&got, &f32s(TEXT_GOLDEN));
    println!("READING_TEXT vs transformers: rel L2 {rel:.3e}, cosine {cos:.8}");
    assert!(rel <= 2e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
    if std::env::var("READING_STEPS").is_ok_and(|s| s != "0") {
        let steps: u32 = std::env::var("READING_STEPS")
            .unwrap()
            .parse()
            .expect("steps");
        let t0 = std::time::Instant::now();
        let ran = common::guest::loop_on(&mut engine, &caps, steps, &got);
        println!(
            "READING_TEXT the {steps}-step loop: {:.1} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
        let host = host_loop(steps, &host_text(&PROMPT));
        for (i, x) in ran.latents.iter().enumerate() {
            std::fs::write(
                out.join(format!("latent{}.f32", i + 1)),
                common::guest::bytes(x),
            )
            .expect("a latent");
            let (rel, cos) = rel_cos(x, &host[i]);
            println!(
                "READING_TEXT latent after {} vs engine-cpu: rel L2 {rel:.3e}, cosine {cos:.8}",
                i + 1
            );
        }
        if steps == 2 {
            let (rel, cos) = rel_cos(ran.latents.last().expect("a latent"), &f32s(PIPE_GOLDEN));
            println!(
                "READING_TEXT latent after 2 vs diffusers' pipeline: rel L2 {rel:.3e}, cosine {cos:.8}"
            );
        }
    }
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

/// The text pass's rows as the denoise loop's context, both readings compiled on load, on the
/// host fabric: the loop lands on the host's own loop over engine-cpu's text rows bit for bit,
/// and within the bf16 tolerance of diffusers' pipeline (`mk_tiny_flux2_pipe.py --pipe`: the
/// prompt through transformers' Qwen3, the loop over the same sigmas from the same latent).
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

/// diffusers' decode of the pipeline's latent above (`mk_tiny_flux2_pipe.py --vae`): 64 x 64
/// pixels, in [-1, 1].
const VAE_GOLDEN: &[u8] = include_bytes!("flux_2_mini_vae16_pixels.f32");
const PIPE_LATENT: &[u8] = include_bytes!("flux_2_mini_pipe16_latent2.f32");

/// The `vae.decode` reading on engine-cpu's forward over the checkpoint as the engine lands it
/// (its convolutions, group norms, mid-block attention and upsamplings the host's reference):
/// diffusers' `AutoencoderKLFlux2` decode of the same latent, within the bf16 tolerance.
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
            <engine_cerebras::prepare::Landed as kernels_cerebras::tiles::Params>::get(&landed, n)
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
