//! flux2-mini's denoise reading run on the host as one stream step (a 64-row text lane and a
//! 16 x 16 latent at sigma 0.5, guidance 4): its velocity is diffusers'
//! `Flux2Transformer2DModel`'s on the same weights and inputs (`flux_2_mini_velocity.f32`), within
//! the bf16 tolerance (relative L2 1e-2, cosine 0.999). Weights and inputs are drawn by name,
//! here and in the golden's generator alike.

use engine_cpu::Forward;
use engine_cpu::forward::{Output, StreamLane, StreamStep, Values};
use poem_ir::{Operands, Platform, Request, Trace};

const MINI: &str = "flux2-mini-bf16-kv-bf16";
const GOLDEN: &[u8] = include_bytes!("flux_2_mini_velocity.f32");
const TEXT: usize = 64;
const H: u32 = 16;
const W: u32 = 16;
const CHANNELS: usize = 32;
const CONTEXT: usize = 192;

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `count` values in [-1/2, 1/2), each exact in f32: what the golden's generator draws for `name`.
fn units(name: &str, count: usize) -> Vec<f32> {
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 40) as i64 - (1 << 23)) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn trace() -> Trace {
    poem_compiler::catalog::repository()
        .parse(MINI)
        .expect("the catalog holds flux2-mini")
        .trace(Platform::Cerebras)
}

fn velocity(outs: &[Output]) -> Vec<f32> {
    match outs.iter().find(|o| o.name == "velocity") {
        Some(Output {
            values: Values::F32(v),
            rows,
            width,
            ..
        }) => {
            assert_eq!((*rows, *width), ((H * W) as usize, CHANNELS));
            v.clone()
        }
        _ => panic!("the step reads out no velocity"),
    }
}

/// flux2-mini loaded with the weights the golden's generator draws, and the step it runs.
fn setup(trace: &Trace) -> (Forward, StreamStep<'static>) {
    let shapes: Vec<(String, Vec<u64>)> = (trace.params.iter())
        .map(|p| (p.name.clone(), p.shape.clone()))
        .collect();
    let plane = |name: &str, count: usize| -> Vec<f32> {
        let u = units(name, count);
        if name.ends_with("norm") {
            return u.iter().map(|u| 1.0 + 0.5 * u).collect();
        }
        let fan_in = (shapes.iter())
            .find(|(n, _)| n == name)
            .and_then(|(_, s)| s.last().copied())
            .unwrap_or(1) as f32;
        let s = (12.0 / fan_in).sqrt();
        u.iter().map(|u| u * s).collect()
    };
    let word = |stream: &str| {
        let r = Request::new(TEXT as u32, false)
            .switched("reading", "denoise")
            .switched("stream", stream);
        trace.facts.word(&r)
    };
    let (text, image) = (word("context"), word("latents"));
    let image_rows = (H / 2 * W / 2) as usize;
    let s = 12f32.sqrt();
    let mut context: Vec<f32> = units("input.context", TEXT * CONTEXT)
        .iter()
        .map(|u| bf16(u * s))
        .collect();
    context.resize((TEXT + image_rows) * CONTEXT, 0.0);
    let latents: Vec<f32> = units("input.latents", (H * W) as usize * CHANNELS)
        .iter()
        .map(|u| bf16(u * s))
        .collect();
    let step = StreamStep {
        lanes: vec![
            StreamLane {
                word: text,
                rows: TEXT,
                clip: None,
            },
            StreamLane {
                word: image,
                rows: image_rows,
                clip: Some([1, H, W]),
            },
        ],
        inputs: vec![
            ("timestep", vec![500.0; 2]),
            ("guidance", vec![4.0; 2]),
            ("context", context),
            ("latents", latents),
        ],
        tokens: Vec::new(),
    };
    let forward = Forward::new(trace.clone(), None, &plane).expect("flux2-mini loads");
    (forward, step)
}

/// The relative L2 distance and cosine of `trace`'s velocity from diffusers'.
fn against_diffusers(trace: &Trace) -> (f64, f64) {
    let (mut forward, step) = setup(trace);
    let got = velocity(&forward.stream(&step).expect("the denoise step runs"));
    let want: Vec<f32> = GOLDEN
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "every voxel is the image lane's"
    );
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum::<f64>();
    let diff: Vec<f32> = got.iter().zip(&want).map(|(a, b)| a - b).collect();
    let rel = (dot(&diff, &diff) / dot(&want, &want)).sqrt();
    let cos = dot(&got, &want) / (dot(&got, &got) * dot(&want, &want)).sqrt();
    (rel, cos)
}

#[test]
fn flux_2_minis_denoise_answers_diffusers() {
    let mut trace = trace();
    let (rel, cos) = against_diffusers(&trace);
    eprintln!("flux2-mini denoise vs diffusers: rel L2 {rel:.3e}, cosine {cos:.6}");
    assert!(rel <= 1e-2, "relative L2 {rel}");
    assert!(cos >= 0.999, "cosine {cos}");
    for decl in &mut trace.values {
        if let poem_ir::Ty::Tensor { dtype, .. } = &mut decl.ty
            && *dtype == poem_ir::Dtype::Bf16
        {
            *dtype = poem_ir::Dtype::F32;
        }
    }
    let (rel, cos) = against_diffusers(&trace);
    eprintln!("every activation f32: rel L2 {rel:.3e}, cosine {cos:.8}");
    assert!(rel <= 1e-4, "relative L2 {rel} with every activation f32");
}

/// A node's values as the step held them: the exit projection's output is the velocity's
/// patches.
#[test]
fn a_stream_step_captures_a_nodes_values() {
    let trace = trace();
    let (mut forward, step) = setup(&trace);
    let at = |name: &str, nth: usize| {
        (trace.nodes.iter().enumerate())
            .filter(|(_, n)| n.op.name() == name)
            .nth(nth)
            .map(|(i, _)| i)
            .expect("flux2-mini runs the op")
    };
    // the exit projection: the matmul whose rows the unpatchify reads
    let unpatch = at("conv.unpatchify", 0);
    let last = match &trace.nodes[unpatch].op {
        poem_ir::Operation::Conv(poem_ir::Conv::Unpatchify { x, .. }) => {
            match trace.values[x.0 as usize].def {
                poem_ir::Def::Op(n) => n as usize,
                _ => panic!("the patches are a node's"),
            }
        }
        _ => panic!("an unpatchify"),
    };
    assert_eq!(trace.nodes[last].op.name(), "linear.matmul");
    let (outs, caps) = forward
        .stream_capture(&step, &[last, unpatch])
        .expect("the denoise step runs");
    let held = |node: usize, input: bool, k: usize| {
        caps.iter()
            .filter(|c| c.node == node && c.input == input)
            .nth(k)
            .expect("a captured value")
    };
    let (proj, patches) = (held(last, false, 0), held(unpatch, true, 0));
    // (bit for bit: the text rows the image lane's projection keeps from are NaN)
    let bits = |v: &Values| match v {
        Values::F32(v) => v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>(),
        Values::I32(_) => panic!("a float value"),
    };
    assert_eq!(proj.value, patches.value);
    assert_eq!(bits(&proj.values), bits(&patches.values));
    assert_eq!((proj.rows, proj.width), (2 * TEXT, 128));
    let unpatched = held(unpatch, false, 0);
    assert_eq!(bits(&Values::F32(velocity(&outs))), bits(&unpatched.values));
}
