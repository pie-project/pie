//! Suite inferlets' step programs, traced here as their crates trace them,
//! on the device against the host interpreter over several fires: the
//! sampler shapes the op-by-op battery only meets one op at a time — a
//! generated top-p cut over a trained model's vocabulary, eta truncation's
//! entropy-relaxed floor, guidance over a pair of logits rows, and beam
//! search at one beam and at four, whose state rides channels fire to fire.
//!
//! Discrete outputs (tokens, parents, masks) must be the interpreter's
//! exactly; so must every float the program computes without `exp` or
//! `log`. A float built on them is held to a few ulps of its scale. The
//! logits are fixed per seed, so a boundary either agrees or never does.

use engine::{ChannelId, InstanceId, ProgramRegistration};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::Plane;
use engine_cuda::program::compile::Disk;
use engine_cuda::program::launch::INTRINSIC_STORAGE_F32;
use engine_cuda::program::session::{Fired, Launched};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_compiler::plan::compile_bound;
use eta_dsl::Traced;
use eta_dsl::prelude::*;
use eta_exec::Extents;
use eta_ir::registry::{GeometryClass, Port};
use eta_ir::types::Dtype;
use eta_ir::validate::BoundTrace;

const QWEN_VOCAB: u32 = 151_936;
const LANES: &[u32] = &[1, 8];
const STEPS: u32 = 3;

#[derive(Clone, Copy)]
enum Check {
    Exact,
    /// Within this many ulps of the larger of one and the expected value.
    Scaled(u32),
}

struct Suite {
    name: String,
    traced: Traced,
    rows: u32,
    vocab: u32,
    /// Seeded channels by name, each lane's key folded into `rng`-named ones.
    seeds: Vec<(&'static str, Value)>,
    outputs: Vec<(&'static str, Dtype, Check)>,
}

fn leak(channel: Channel) -> &'static Channel {
    Box::leak(Box::new(channel))
}

/// A builder over `rows` logits rows: the embed indptr port says how many.
fn builder(vocab: u32, rows: u32) -> Builder<'static> {
    let indptr = leak(Channel::from((0u32..=rows).collect::<Vec<u32>>()).named("embed_indptr"));
    let mut b = Builder::new(vocab, 16);
    b.outputs(vec![vec![crate::bound::logits_decl(vocab)]]);
    b.bind_port(Port::EmbedIndptr, indptr);
    b
}

fn port_seeds(rows: u32) -> Vec<(&'static str, Value)> {
    vec![("embed_indptr", Value::U32((0..=rows).collect()))]
}

/// `sampling-primitives`: the greedy token, the log-softmax, its entropy,
/// and a top-p keep-mask no sampler tail claims — the generated cut.
fn sampling_primitives() -> Suite {
    const TOP_P: f32 = 0.9;
    let vocab = QWEN_VOCAB;
    let token_out = leak(Channel::new([1], dtype::i32).capacity(4).named("token"));
    let entropy_out = leak(Channel::new([1], dtype::f32).capacity(4).named("entropy"));
    let probs_out = leak(
        Channel::new([vocab], dtype::f32)
            .capacity(4)
            .named("probabilities"),
    );
    let keep_out = leak(
        Channel::new([vocab], dtype::i32)
            .capacity(4)
            .named("nucleus_keep"),
    );
    let mut b = builder(vocab, 1);
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::output(0, "logits");
        let logprobs = log_softmax(&logits);
        let probabilities = exp(&logprobs);
        let entropy = reshape(entropy_from_logprobs(&probabilities, &logprobs), [1]);
        let token = reshape(reduce_argmax(&logits), [1]);
        let keep = pivot_threshold(&probabilities, cummass_le(TOP_P));
        keep_out.put(reshape(cast(&keep, dtype::i32), [vocab]));
        token_out.put(&token);
        entropy_out.put(&entropy);
        probs_out.put(&probabilities);
    });
    for channel in [token_out, entropy_out, probs_out, keep_out] {
        channel.note_host_take();
    }
    Suite {
        name: "sampling-primitives".into(),
        traced: b.build().expect("sampling-primitives traces"),
        rows: 1,
        vocab,
        seeds: port_seeds(1),
        outputs: vec![
            ("token", Dtype::I32, Check::Exact),
            ("entropy", Dtype::F32, Check::Scaled(64)),
            ("probabilities", Dtype::F32, Check::Scaled(8)),
            ("nucleus_keep", Dtype::I32, Check::Exact),
        ],
    }
}

/// `eta-epsilon-sampling`'s truncation step, in either mode: a probability
/// floor (relaxed by the entropy, for eta), the argmax OR-ed back in, and a
/// Gumbel-max draw over what survives; the key advances fire to fire.
fn truncation(eta: bool) -> Suite {
    let vocab = 32_000u32;
    let (epsilon, temperature) = (0.0003f32, 0.8f32);
    let rng = leak(Channel::from([0x3f17u32 ^ 0x5bd1, 0]).named("rng"));
    let tok_out = leak(Channel::new([1], dtype::i32).capacity(4).named("tok_out"));
    let kept_out = leak(Channel::new([1], dtype::f32).capacity(4).named("kept_out"));
    let mass_out = leak(Channel::new([1], dtype::f32).capacity(4).named("mass_out"));
    let mut b = builder(vocab, 1);
    b.stage(Stage::Epilogue, move || {
        let r = rng.take();
        let scaled = intrinsics::output(0, "logits") / temperature;
        let logprobs = log_softmax(&scaled);
        let probs = exp(&logprobs);
        let threshold = if eta {
            let h = entropy_from_logprobs(&probs, &logprobs);
            min_elem(epsilon, epsilon.sqrt() * exp(-&h))
        } else {
            Tensor::constant(epsilon)
        };
        let base_mask = pivot_threshold(&probs, prob_ge(threshold));
        let argmax_ind = ge(&probs, broadcast(reduce_max(&probs), [vocab]));
        let keep = or(base_mask, argmax_ind);
        let zeros = broadcast(0.0f32, [vocab]);
        let kept_mass = reshape(reduce_sum(select(&keep, &probs, &zeros)), [1]);
        let kept = reshape(reduce_sum(cast(&keep, dtype::f32)), [1]);
        let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
        let token = gumbel_max(select(&keep, &scaled, &neg_inf), &r);
        tok_out.put(&token);
        kept_out.put(&kept);
        mass_out.put(&kept_mass);
        rng.put(&r + iota(2));
    });
    for channel in [tok_out, kept_out, mass_out] {
        channel.note_host_take();
    }
    let mut seeds = port_seeds(1);
    seeds.push(("rng", Value::U32(vec![0x3f17 ^ 0x5bd1, 0])));
    Suite {
        name: format!(
            "eta-epsilon-sampling ({})",
            if eta { "eta" } else { "epsilon" }
        ),
        traced: b.build().expect("the truncation step traces"),
        rows: 1,
        vocab,
        seeds,
        outputs: vec![
            ("tok_out", Dtype::I32, Check::Exact),
            ("kept_out", Dtype::F32, Check::Exact),
            ("mass_out", Dtype::F32, Check::Scaled(64)),
        ],
    }
}

/// `classifier-free-guidance` (`guided`) and `context-aware-decoding`: two
/// logits rows, the unconditioned and the conditioned, mixed in log space.
fn guidance(guided: bool) -> Suite {
    let vocab = 32_000u32;
    let weight = if guided { 1.5f32 } else { 0.5 };
    let out = leak(Channel::new([3], dtype::f32).capacity(4).named("out"));
    let next = leak(Channel::new([2], dtype::i32).capacity(4).named("next"));
    let mut b = builder(vocab, 2);
    b.stage(Stage::Epilogue, move || {
        let logits = reshape(intrinsics::output(0, "logits"), [2, vocab]);
        let first = reshape(gather(&logits, iota(1)), [vocab]);
        let second = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
        let (cond, uncond) = (log_softmax(&second), log_softmax(&first));
        let mixed = if guided {
            log_softmax(&uncond + (&cond - &uncond) * weight)
        } else {
            log_softmax(&cond * (1.0 + weight) - &uncond * weight)
        };
        let token = reduce_argmax(&mixed);
        let shift = cast(ne(&token, reduce_argmax(&cond)), dtype::i32);
        let kl = reshape(reduce_sum(exp(&mixed) * (&mixed - &cond)), [1]);
        let token = reshape(cast(&token, dtype::i32), [1]);
        next.put(broadcast(&token, [2]));
        let k = iota(3);
        out.put(select(
            eq(&k, 0u32),
            broadcast(cast(&token, dtype::f32), [3]),
            select(
                eq(&k, 1u32),
                broadcast(cast(reshape(shift, [1]), dtype::f32), [3]),
                broadcast(&kl, [3]),
            ),
        ));
    });
    out.note_host_take();
    next.note_host_take();
    Suite {
        name: if guided {
            "classifier-free-guidance".into()
        } else {
            "context-aware-decoding".into()
        },
        traced: b.build().expect("the guided step traces"),
        rows: 2,
        vocab,
        seeds: port_seeds(2),
        outputs: vec![
            ("next", Dtype::I32, Check::Exact),
            ("out", Dtype::F32, Check::Scaled(64)),
        ],
    }
}

/// `beam-search`'s step at `beams` wide: each beam's score plus its
/// log-softmax, the best `beams` of all, their parents' masks inherited
/// along axis 0 and their new positions marked.
fn beam(beams: u32) -> Suite {
    let vocab = 4_096u32;
    let pool_len = 96 * beams + 1;
    let mut initial = vec![f32::NEG_INFINITY; beams as usize];
    initial[0] = 0.0;
    let init_mask: Vec<bool> = (0..beams)
        .flat_map(|_| (0..pool_len).map(|p| p == 0))
        .collect();
    let scores = leak(Channel::from(initial.clone()).named("scores"));
    // What the next forward pass would consume (its tokens, slots and
    // lengths) the host takes here instead.
    let toks = leak(
        Channel::new([beams], dtype::i32)
            .capacity(4)
            .named("beam_toks"),
    );
    let mask = leak(Channel::from_shaped([beams, pool_len], init_mask.clone()).named("mask"));
    let slots = leak(Channel::new([beams], dtype::u32).capacity(4).named("slots"));
    let pos = leak(Channel::from(vec![0u32; beams as usize]).named("pos"));
    let fill = leak(Channel::from([1u32]).named("fill"));
    let kv_len = leak(
        Channel::new([beams], dtype::u32)
            .capacity(4)
            .named("kv_len"),
    );
    let out = leak(Channel::new([beams], dtype::i32).capacity(4).named("out"));
    let out_par = leak(
        Channel::new([beams], dtype::u32)
            .capacity(4)
            .named("out_par"),
    );
    let out_scr = leak(
        Channel::new([beams], dtype::f32)
            .capacity(4)
            .named("out_scr"),
    );
    let out_mask = leak(
        Channel::new([beams, pool_len], dtype::bool)
            .capacity(4)
            .named("out_mask"),
    );
    let (b, v) = (beams, vocab);
    let mut builder = builder(vocab, beams);
    builder.stage(Stage::Epilogue, move || {
        let logits = reshape(intrinsics::output(0, "logits"), [b, v]);
        let cand = broadcast(reshape(scores.take(), [b, 1]), [b, v]) + log_softmax(&logits);
        let (s, i) = top_k(reshape(cand, [b * v]), b);
        let parent = &i / v;
        let tok_i = cast(&i % v, dtype::i32);
        scores.put(&s);
        toks.put(&tok_i);
        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        let base = fill.take();
        let wpos = broadcast(reshape(&base, [1]), [b]) + iota(b);
        let inherited = gather(mask.take(), &parent);
        let col = broadcast(reshape(iota(pool_len), [1, pool_len]), [b, pool_len]);
        let wpos_b = broadcast(reshape(&wpos, [b, 1]), [b, pool_len]);
        let marked = or(inherited, eq(col, wpos_b));
        mask.put(&marked);
        out_mask.put(&marked);
        let filled = &base + b;
        slots.put(&wpos);
        kv_len.put(broadcast(reshape(&filled, [1]), [b]));
        pos.put(pos.take() + 1u32);
        fill.put(&filled);
    });
    for channel in [out, out_par, out_scr, out_mask, toks, slots, kv_len] {
        channel.note_host_take();
    }
    let mut seeds = port_seeds(beams);
    seeds.extend([
        ("scores", Value::F32(initial)),
        ("mask", Value::Bool(init_mask)),
        ("pos", Value::U32(vec![0; beams as usize])),
        ("fill", Value::U32(vec![1])),
    ]);
    Suite {
        name: format!("beam-search ({beams} beam(s))"),
        traced: builder.build().expect("the beam step traces"),
        rows: beams,
        vocab,
        seeds,
        outputs: vec![
            ("out", Dtype::I32, Check::Exact),
            ("out_par", Dtype::U32, Check::Exact),
            ("out_scr", Dtype::F32, Check::Scaled(16)),
            ("out_mask", Dtype::Bool, Check::Exact),
            ("beam_toks", Dtype::I32, Check::Exact),
            ("slots", Dtype::U32, Check::Exact),
            ("kv_len", Dtype::U32, Check::Exact),
        ],
    }
}

struct Noise(u64);

impl Noise {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u.ln()).sqrt() * (core::f32::consts::TAU * v).cos()
    }
}

/// A trained model's spread of logits: gaussian, a few tokens far ahead.
fn logits(rows: u32, vocab: u32, seed: u64) -> Vec<f32> {
    let mut noise = Noise(seed | 1);
    let mut plane: Vec<f32> = (0..rows * vocab).map(|_| noise.gaussian() * 3.0).collect();
    for r in 0..rows as usize {
        for k in 0..4usize {
            let at = (r * 7_919 + k * 104_729) % vocab as usize;
            plane[r * vocab as usize + at] += 6.0 - k as f32;
        }
    }
    plane
}

fn wire_bytes(value: &Value) -> Vec<u8> {
    match value {
        Value::Bool(bits) => {
            let mut bytes = vec![0u8; bits.len().div_ceil(8)];
            for (i, &bit) in bits.iter().enumerate() {
                bytes[i / 8] |= u8::from(bit) << (i % 8);
            }
            bytes
        }
        other => other.to_le_bytes(),
    }
}

fn wire_value(bytes: &[u8], like: &Value) -> Value {
    let words = || bytes.as_chunks::<4>().0.iter();
    match like {
        Value::F32(_) => Value::F32(words().map(|c| f32::from_le_bytes(*c)).collect()),
        Value::I32(_) => Value::I32(words().map(|c| i32::from_le_bytes(*c)).collect()),
        Value::U32(_) => Value::U32(words().map(|c| u32::from_le_bytes(*c)).collect()),
        Value::Bool(v) => Value::Bool(
            (0..v.len())
                .map(|i| bytes[i / 8] >> (i % 8) & 1 == 1)
                .collect(),
        ),
    }
}

fn agree(got: &Value, want: &Value, check: Check) -> Result<(), String> {
    if let (Value::F32(g), Value::F32(w), Check::Scaled(n)) = (got, want, check) {
        let bad = g.iter().zip(w).position(|(&x, &y)| {
            !(x.is_nan() && y.is_nan())
                && x != y
                && (x - y).abs() > n as f32 * f32::EPSILON * y.abs().max(1.0)
        });
        return match bad {
            None if g.len() == w.len() => Ok(()),
            None => Err(format!("{} elements against {}", g.len(), w.len())),
            Some(i) => Err(format!(
                "[{i}]: device {:e} vs interpreter {:e}",
                g[i], w[i]
            )),
        };
    }
    if got == want {
        return Ok(());
    }
    let differ = match (got, want) {
        (Value::F32(g), Value::F32(w)) => g
            .iter()
            .zip(w)
            .position(|(x, y)| x.to_bits() != y.to_bits())
            .map(|i| format!("[{i}]: device {:e} vs interpreter {:e}", g[i], w[i])),
        (Value::I32(g), Value::I32(w)) => g
            .iter()
            .zip(w)
            .position(|(x, y)| x != y)
            .map(|i| format!("[{i}]: device {} vs interpreter {}", g[i], w[i])),
        (Value::U32(g), Value::U32(w)) => g
            .iter()
            .zip(w)
            .position(|(x, y)| x != y)
            .map(|i| format!("[{i}]: device {} vs interpreter {}", g[i], w[i])),
        (Value::Bool(g), Value::Bool(w)) => {
            let wrong = g.iter().zip(w).filter(|(x, y)| x != y).count();
            g.iter().zip(w).position(|(x, y)| x != y).map(|i| {
                format!(
                    "[{i}]: device {} vs interpreter {}; {wrong} differ",
                    g[i], w[i]
                )
            })
        }
        _ => None,
    };
    Err(differ.unwrap_or_else(|| "the values differ in kind or length".into()))
}

struct Lane {
    host: Instance,
    device: InstanceId,
    seed: u64,
}

fn run(context: &Context, suite: &Suite) -> Vec<String> {
    let names = suite.traced.channel_names();
    let at = |name: &str| -> u32 {
        names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("{}: no channel {name}", suite.name)) as u32
    };
    let bound: BoundTrace = crate::bound::logits(suite.traced.container().clone(), suite.vocab)
        .unwrap_or_else(|e| panic!("{}: binds: {e:?}", suite.name));
    let stages = compile_bound(&bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let registration = ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };
    let mut plane = Plane::new(Disk::disabled());
    let program = match plane.register(context, &registration) {
        Ok(program) => program,
        Err(e) => return vec![format!("{}: does not compile: {e:?}", suite.name)],
    };
    let count = bound.container.channels.len();
    let seeds: Vec<(u32, Value)> = suite
        .seeds
        .iter()
        .map(|(name, value)| (at(name), value.clone()))
        .collect();
    let mut failures = Vec::new();
    for &lanes in LANES {
        let mut fleet: Vec<Lane> = (0..lanes)
            .map(|lane| {
                let wire: Vec<(u32, Vec<u8>)> =
                    seeds.iter().map(|(c, v)| (*c, wire_bytes(v))).collect();
                let ids: Vec<ChannelId> = (1..=count as u64)
                    .map(|c| ChannelId(c + 1000 * u64::from(lane)))
                    .collect();
                let device = plane
                    .bind(
                        program,
                        &wire,
                        Extents {
                            row_count: suite.rows,
                            sampled_rows: suite.rows,
                            query_len: suite.rows,
                            ..Extents::default()
                        },
                        GeometryClass::Host,
                        eta_ir::registry::PortMask::NONE,
                        false,
                        &vec![None; count],
                        &ids,
                        context.stream(),
                    )
                    .unwrap_or_else(|e| panic!("{}: the instance binds: {e:?}", suite.name));
                Lane {
                    host: Instance::new(&bound, &seeds)
                        .unwrap_or_else(|e| panic!("{}: instantiates: {e:?}", suite.name)),
                    device,
                    seed: 0x5eed_0000 + u64::from(lane) * 7_919,
                }
            })
            .collect();
        'steps: for step in 0..STEPS {
            let mut planes = Vec::new();
            let mut plane_values = Vec::new();
            for lane in &fleet {
                let values = logits(suite.rows, suite.vocab, lane.seed ^ u64::from(step) << 40);
                let mut buffer = Buffer::zeroed(values.len() * 4).expect("a logits plane");
                let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                buffer.write(0, &bytes).expect("the plane uploads");
                plane
                    .bind_output(
                        lane.device,
                        "logits",
                        0,
                        buffer.ptr(),
                        INTRINSIC_STORAGE_F32,
                        suite.vocab,
                        suite.vocab,
                        0,
                    )
                    .expect("the logits bind");
                planes.push(buffer);
                plane_values.push(values);
            }
            for lane in &fleet {
                let launched = plane
                    .stage(lane.device, Stage::Epilogue)
                    .unwrap_or_else(|e| panic!("{}: stages: {e:?}", suite.name));
                assert!(
                    matches!(launched, Launched::Airborne),
                    "{}: refused: {launched:?}",
                    suite.name
                );
            }
            plane.fly(context).expect("the wave flies");
            plane.land(context).expect("the wave lands");
            context.synchronize().expect("the device synchronizes");
            for (index, (lane, values)) in fleet.iter_mut().zip(plane_values).enumerate() {
                let fired = plane
                    .settle_launched(lane.device)
                    .expect("the lane settles");
                if !matches!(fired, Fired::Committed) {
                    failures.push(format!(
                        "{}: step {step}, lane {index} of {lanes}: the device did not commit: {fired:?}",
                        suite.name
                    ));
                    break 'steps;
                }
                let stepped = lane
                    .host
                    .step(
                        &bound,
                        &PassInputs {
                            outputs: vec![("logits".to_string(), 0, Value::F32(values))],
                        },
                        &mut NoKernels,
                    )
                    .expect("the interpreter steps");
                assert!(
                    stepped.committed,
                    "{}: the reference did not commit",
                    suite.name
                );
                let session = plane.instance_mut(lane.device).expect("the instance");
                for &(name, _, check) in &suite.outputs {
                    let want = lane
                        .host
                        .host_take(&bound, at(name))
                        .expect("a reference output");
                    let bytes = session
                        .take(at(name))
                        .expect("a channel reads")
                        .expect("the channel holds a cell");
                    let got = wire_value(&bytes, &want);
                    if let Err(why) = agree(&got, &want, check) {
                        failures.push(format!(
                            "{} {name}: step {step}, lane {index} of {lanes}: {why}",
                            suite.name
                        ));
                    }
                }
            }
            drop(planes);
            if !failures.is_empty() {
                break;
            }
        }
        for lane in fleet {
            plane.close_instance(lane.device).expect("the lane closes");
        }
    }
    plane.close_program(program).expect("the program closes");
    failures
}

#[test]
fn every_suite_step_agrees_with_the_host_interpreter() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a CUDA context");
    let suites = [
        sampling_primitives(),
        truncation(true),
        truncation(false),
        guidance(true),
        guidance(false),
        beam(1),
        beam(4),
    ];
    let failures: Vec<String> = suites
        .iter()
        .flat_map(|suite| run(&context, suite))
        .collect();
    assert!(
        failures.is_empty(),
        "{} disagreement(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
