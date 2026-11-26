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
    /// bf16 values within this many bf16 ulps of the expected value: what a
    /// value built on `exp` or `log` keeps where the device's and the host's
    /// f32 differ by an ulp at a rounding boundary.
    Bf16Ulps(u32),
}

struct Suite {
    name: String,
    traced: Traced,
    rows: u32,
    vocab: u32,
    /// Seeded channels by name, each lane's key folded into `rng`-named ones.
    seeds: Vec<(&'static str, Value)>,
    outputs: Vec<(&'static str, Dtype, Check)>,
    /// Each named channel's id ([`named`]).
    names: Vec<(&'static str, u64)>,
}

fn leak(channel: Channel) -> &'static Channel {
    Box::leak(Box::new(channel))
}

thread_local! {
    /// The channels the suite being built names, with their ids.
    static NAMES: std::cell::RefCell<Vec<(&'static str, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `channel`, leaked, under `name` in the suite being built.
fn named(name: &'static str, channel: Channel) -> &'static Channel {
    NAMES.with(|names| names.borrow_mut().push((name, channel.gid())));
    leak(channel)
}

/// The names the suite being built gave, taken.
fn take_names() -> Vec<(&'static str, u64)> {
    NAMES.with(|names| names.take())
}

/// A builder over `rows` logits rows: the embed indptr port says how many.
fn builder(vocab: u32, rows: u32) -> Builder<'static> {
    let indptr = named(
        "embed_indptr",
        Channel::from((0u32..=rows).collect::<Vec<u32>>()),
    );
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
    let token_out = named("token", Channel::new([1], dtype::i32).capacity(4));
    let entropy_out = named("entropy", Channel::new([1], dtype::f32).capacity(4));
    let probs_out = named(
        "probabilities",
        Channel::new([vocab], dtype::f32).capacity(4),
    );
    let keep_out = named(
        "nucleus_keep",
        Channel::new([vocab], dtype::i32).capacity(4),
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
        names: take_names(),
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
    let rng = named("rng", Channel::from([0x3f17u32 ^ 0x5bd1, 0]));
    let tok_out = named("tok_out", Channel::new([1], dtype::i32).capacity(4));
    let kept_out = named("kept_out", Channel::new([1], dtype::f32).capacity(4));
    let mass_out = named("mass_out", Channel::new([1], dtype::f32).capacity(4));
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
        names: take_names(),
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
    let out = named("out", Channel::new([3], dtype::f32).capacity(4));
    let next = named("next", Channel::new([2], dtype::i32).capacity(4));
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
        names: take_names(),
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
    let scores = named("scores", Channel::from(initial.clone()));
    // What the next forward pass would consume (its tokens, slots and
    // lengths) the host takes here instead.
    let toks = named("beam_toks", Channel::new([beams], dtype::i32).capacity(4));
    let mask = named(
        "mask",
        Channel::from_shaped([beams, pool_len], init_mask.clone()),
    );
    let slots = named("slots", Channel::new([beams], dtype::u32).capacity(4));
    let pos = named("pos", Channel::from(vec![0u32; beams as usize]));
    let fill = named("fill", Channel::from([1u32]));
    let kv_len = named("kv_len", Channel::new([beams], dtype::u32).capacity(4));
    let out = named("out", Channel::new([beams], dtype::i32).capacity(4));
    let out_par = named("out_par", Channel::new([beams], dtype::u32).capacity(4));
    let out_scr = named("out_scr", Channel::new([beams], dtype::f32).capacity(4));
    let out_mask = named(
        "out_mask",
        Channel::new([beams, pool_len], dtype::bool).capacity(4),
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
        names: take_names(),
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
        Value::Bf16(_) => Value::Bf16(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect(),
        ),
    }
}

/// The size of one bf16 ulp at `x`.
fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.to_bits() >> 23) & 0xff) as i32;
    f32::from_bits(((exponent - 7).max(1) as u32) << 23)
}

fn agree(got: &Value, want: &Value, check: Check) -> Result<(), String> {
    if let (Value::Bf16(g), Value::Bf16(w), Check::Bf16Ulps(n)) = (got, want, check) {
        let wide = |b: u16| eta_ir::types::bf16_to_f32(b);
        let bad = g.iter().zip(w).position(|(&x, &y)| {
            let (x, y) = (wide(x), wide(y));
            !(x.is_nan() && y.is_nan()) && x != y && (x - y).abs() > n as f32 * bf16_ulp(y)
        });
        return match bad {
            None if g.len() == w.len() => Ok(()),
            None => Err(format!("{} elements against {}", g.len(), w.len())),
            Some(i) => Err(format!(
                "[{i}]: device {:e} vs interpreter {:e}",
                wide(g[i]),
                wide(w[i])
            )),
        };
    }
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
        (Value::Bf16(g), Value::Bf16(w)) => g
            .iter()
            .zip(w)
            .position(|(x, y)| x != y)
            .map(|i| format!("[{i}]: device {:#06x} vs interpreter {:#06x}", g[i], w[i])),
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

/// `suite` bound over its logits, and its registration as the host emits it.
fn registration(suite: &Suite) -> (BoundTrace, ProgramRegistration) {
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
    (bound, registration)
}

fn run(context: &Context, suite: &Suite) -> Vec<String> {
    let at = |name: &str| -> u32 {
        let gid = suite
            .names
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("{}: no channel {name}", suite.name))
            .1;
        suite
            .traced
            .channel_order()
            .iter()
            .position(|&g| g == gid)
            .unwrap_or_else(|| panic!("{}: channel {name} is not in the trace", suite.name))
            as u32
    };
    let (bound, registration) = registration(suite);
    let mut plane = crate::guest::alone_plane();
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
                        crate::guest::alone_slots(&plane, program),
                        &crate::guest::alone_span,
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
                        suite.rows,
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

/// `channel` under `name` in the suite being built, kept owned.
fn noted(name: &'static str, channel: Channel) -> Channel {
    NAMES.with(|names| names.borrow_mut().push((name, channel.gid())));
    channel
}

/// bf16 guest values over `rows` rows of an odd vocabulary: a cast in,
/// elementwise ops mixing an f32 constant, the bf16 softmax and
/// log-softmax, reductions, a scan over rows of odd length, a gather of
/// rows, a scatter-add hitting one cell twice, a transpose, a top-k and an
/// argmax, each landed at its own dtype. The values are an odd number of
/// elements, so a value's last word holds padding and, past one row, a
/// row's elements start mid-word.
fn bf16_battery(rows: u32) -> Suite {
    let vocab = 1_001u32;
    let n = rows * vocab;
    let bf =
        |name: &'static str, len: u32| named(name, Channel::new([len], dtype::bf16).capacity(4));
    let narrow_out = bf("narrow", n);
    let scaled_out = bf("scaled", n);
    let exp_out = bf("exp", n);
    let softmax_out = bf("softmax", n);
    let log_softmax_out = bf("log_softmax", n);
    let cumsum_out = bf("cumsum", n);
    let gathered_out = bf("gathered", n);
    let scattered_out = bf("scattered", n);
    let transposed_out = bf("transposed", n);
    let selected_out = bf("selected", n);
    let top_out = bf("top", rows * 5);
    let top_ids_out = named("top_ids", Channel::new([rows * 5], dtype::u32).capacity(4));
    let stats_out = named("stats", Channel::new([3 * rows], dtype::f32).capacity(4));
    let argmax_out = named("argmax", Channel::new([rows], dtype::i32).capacity(4));
    let mut b = builder(vocab, rows);
    b.stage(Stage::Epilogue, move || {
        let logits = reshape(intrinsics::output(0, "logits"), [rows, vocab]);
        let x = cast(&logits, dtype::bf16);
        let flat = |t: &Tensor| reshape(t, [n]);
        narrow_out.put(flat(&x));
        let scaled = &x * 0.5f32 + 1.0f32;
        scaled_out.put(flat(&scaled));
        exp_out.put(flat(&exp(&scaled)));
        softmax_out.put(flat(&softmax(&x)));
        log_softmax_out.put(flat(&log_softmax(&x)));
        let odd = reshape(&x, [7 * rows, 143]);
        cumsum_out.put(flat(&cumsum(&odd)));
        let reversed = (iota(7 * rows) * 0u32 + (7 * rows - 1)) - iota(7 * rows);
        gathered_out.put(flat(&gather(&odd, &reversed)));
        let cells = reshape(&x, [n]);
        let hit = min_elem(iota(3), 1u32);
        scattered_out.put(scatter_add(&cells, &hit, 1.0f32));
        transposed_out.put(flat(&transpose(&odd)));
        selected_out.put(flat(&select(gt(&x, 0.0f32), &x, f32::NEG_INFINITY)));
        let (top, ids) = top_k(&x, 5);
        top_out.put(reshape(&top, [rows * 5]));
        top_ids_out.put(reshape(&ids, [rows * 5]));
        let stats =
            [reduce_sum(&x), reduce_max(&x), reduce_sum(exp(&x))].map(|s| reshape(s, [rows, 1]));
        let k = broadcast(reshape(iota(3), [1, 3]), [rows, 3]);
        let pick = |i: u32| broadcast(&stats[i as usize], [rows, 3]);
        stats_out.put(reshape(
            select(
                eq(&k, 0u32),
                pick(0),
                select(eq(&k, 1u32), pick(1), pick(2)),
            ),
            [3 * rows],
        ));
        argmax_out.put(reshape(cast(reduce_argmax(&scaled), dtype::i32), [rows]));
    });
    for channel in [
        narrow_out,
        scaled_out,
        exp_out,
        softmax_out,
        log_softmax_out,
        cumsum_out,
        gathered_out,
        scattered_out,
        transposed_out,
        selected_out,
        top_out,
        top_ids_out,
        stats_out,
        argmax_out,
    ] {
        channel.note_host_take();
    }
    Suite {
        names: take_names(),
        name: format!("bf16 battery ({rows} row(s) of {vocab})"),
        traced: b.build().expect("the bf16 battery traces"),
        rows,
        vocab,
        seeds: port_seeds(rows),
        outputs: vec![
            ("narrow", Dtype::Bf16, Check::Exact),
            ("scaled", Dtype::Bf16, Check::Exact),
            ("exp", Dtype::Bf16, Check::Bf16Ulps(1)),
            ("softmax", Dtype::Bf16, Check::Bf16Ulps(1)),
            ("log_softmax", Dtype::Bf16, Check::Bf16Ulps(1)),
            ("cumsum", Dtype::Bf16, Check::Exact),
            ("gathered", Dtype::Bf16, Check::Exact),
            ("scattered", Dtype::Bf16, Check::Exact),
            ("transposed", Dtype::Bf16, Check::Exact),
            ("selected", Dtype::Bf16, Check::Exact),
            ("top", Dtype::Bf16, Check::Exact),
            ("top_ids", Dtype::U32, Check::Exact),
            ("stats", Dtype::F32, Check::Scaled(64)),
            ("argmax", Dtype::I32, Check::Exact),
        ],
    }
}

/// The SDK's standard sampler's decode fire of `shape`
/// (`eta_dsl::sampler::decode`) over Qwen3's vocabulary, its knobs in
/// channels, as the built-in inferlets build it: its vocabulary-sized
/// values bf16.
fn standard(name: &str, shape: &eta_dsl::sampler::Shape) -> Suite {
    use eta_dsl::sampler::{self, Knobs, Slots};
    let vocab = QWEN_VOCAB;
    let mut seeds = port_seeds(1);
    let mut knob = |label: &'static str, on: bool, value: Value| {
        on.then(|| {
            let channel = match &value {
                Value::U32(v) => Channel::from(v.clone()),
                Value::F32(v) => Channel::from(v.clone()),
                _ => unreachable!("a knob is f32 or u32"),
            };
            seeds.push((label, value));
            named(label, channel)
        })
    };
    // The temperature rides as its reciprocal, a knob like the rest.
    let inverse = knob(
        "inverse_temperature",
        shape.temperature,
        Value::F32(vec![sampler::inverse_temperature(0.7)]),
    );
    let top_p = knob("top_p", shape.top_p, Value::F32(vec![0.9]));
    let top_k = knob("top_k", shape.top_k, Value::U32(vec![40]));
    let min_p = knob("min_p", shape.min_p, Value::F32(vec![0.05]));
    let repetition = knob("repetition", shape.repetition, Value::F32(vec![1.3]));
    let frequency = knob("frequency", shape.frequency, Value::F32(vec![0.4]));
    let presence = knob("presence", shape.presence, Value::F32(vec![0.6]));
    let penalized = shape.repetition || shape.frequency || shape.presence;
    let hist = penalized.then(|| {
        // A few tokens seen, some of them several times.
        let counts: Vec<f32> = (0..vocab)
            .map(|t| [0.0, 0.0, 0.0, 1.0, 0.0, 3.0][t as usize % 6])
            .collect();
        let present: Vec<f32> = (0..vocab)
            .map(|t| if t % 11 == 0 { 1.0 } else { 0.0 })
            .collect();
        seeds.push(("counts", Value::F32(counts.clone())));
        seeds.push(("present", Value::F32(present.clone())));
        (
            noted("counts", Channel::from(counts)),
            noted("present", Channel::from(present)),
        )
    });
    let logprobs = shape.logprobs.map(|top| {
        let values = noted("logprobs", Channel::new([1 + top], dtype::f32).capacity(4));
        values.note_host_take();
        let ids = (top > 0).then(|| {
            let ids = noted("logprob_ids", Channel::new([top], dtype::i32).capacity(4));
            ids.note_host_take();
            ids
        });
        (values, ids)
    });
    let rng_seed = [0x9e37u32, 1];
    seeds.push(("rng", Value::U32(rng_seed.to_vec())));
    let slots: &'static Slots = Box::leak(Box::new(Slots {
        rng: noted("rng", Channel::from(rng_seed)),
        hist,
        mask: None,
        logprobs,
    }));
    let tok_in = named("tok_in", Channel::new([1], dtype::i32).capacity(4));
    let out = named("out", Channel::new([1], dtype::i32).capacity(4));
    out.note_host_take();
    tok_in.note_host_take();
    let logprob_top = shape.logprobs;
    let mut b = builder(vocab, 1);
    b.stage(Stage::Epilogue, move || {
        let t = |c: Option<&'static Channel>| c.map(|c| reshape(c.read(), []));
        let knobs = Knobs {
            temperature: t(inverse),
            top_p: t(top_p),
            top_k: t(top_k),
            min_p: t(min_p),
            repetition: t(repetition),
            frequency: t(frequency),
            presence: t(presence),
            logprobs: logprob_top,
        };
        sampler::decode(vocab, &knobs, slots, tok_in, out);
    });
    let mut outputs = vec![("out", Dtype::I32, Check::Exact)];
    if let Some(top) = shape.logprobs {
        outputs.push(("logprobs", Dtype::F32, Check::Scaled(1 << 16)));
        if top > 0 {
            outputs.push(("logprob_ids", Dtype::I32, Check::Exact));
        }
    }
    Suite {
        names: take_names(),
        name: format!("standard {name}"),
        traced: b.build().expect("the standard sampler traces"),
        rows: 1,
        vocab,
        seeds,
        outputs,
    }
}

/// `beam-search`'s bf16 step at `beams` wide, as the example runs it: the
/// candidates bf16, each beam's score relative to the best, the absolute
/// scores carried in f32.
fn beam_bf16(beams: u32) -> Suite {
    let vocab = QWEN_VOCAB;
    let mut initial = vec![f32::NEG_INFINITY; beams as usize];
    initial[0] = 0.0;
    let scores = named("scores", Channel::from(initial.clone()));
    let out = named("out", Channel::new([beams], dtype::i32).capacity(4));
    let out_par = named("out_par", Channel::new([beams], dtype::u32).capacity(4));
    let out_scr = named("out_scr", Channel::new([beams], dtype::f32).capacity(4));
    let greedy = named("greedy", Channel::new([beams], dtype::i32).capacity(4));
    let (b, v) = (beams, vocab);
    let mut builder = builder(vocab, beams);
    builder.stage(Stage::Epilogue, move || {
        let logits = cast(
            reshape(intrinsics::output(0, "logits"), [b, v]),
            dtype::bf16,
        );
        let carried = scores.take();
        let best = reduce_max(&carried);
        let relative = cast(&carried - &best, dtype::bf16);
        let cand = broadcast(reshape(relative, [b, 1]), [b, v]) + log_softmax(&logits);
        let (s, i) = top_k(reshape(cand, [b * v]), b);
        let s = cast(&s, dtype::f32) + &best;
        scores.put(&s);
        out.put(cast(&i % v, dtype::i32));
        out_par.put(&i / v);
        out_scr.put(&s);
        greedy.put(reshape(cast(reduce_argmax(&logits), dtype::i32), [b]));
    });
    for channel in [out, out_par, out_scr, greedy] {
        channel.note_host_take();
    }
    let mut seeds = port_seeds(beams);
    seeds.push(("scores", Value::F32(initial)));
    Suite {
        names: take_names(),
        name: format!("beam-search bf16 ({beams} beam(s))"),
        traced: builder.build().expect("the bf16 beam step traces"),
        rows: beams,
        vocab,
        seeds,
        outputs: vec![
            ("out", Dtype::I32, Check::Exact),
            ("out_par", Dtype::U32, Check::Exact),
            ("out_scr", Dtype::F32, Check::Scaled(1 << 16)),
            ("greedy", Dtype::I32, Check::Exact),
        ],
    }
}

/// `gumbel_argmax` over a bf16 row at a temperature, over `rows` rows of
/// Qwen3's vocabulary: every lane kept, or a top-k (40) and top-p (0.9) keep
/// mask of the row's softmax; the key advances fire to fire. Each perturbed
/// lane is made in f32 in the reduction, as the interpreter makes it.
fn bf16_draw(rows: u32, truncated: bool) -> Suite {
    let vocab = QWEN_VOCAB;
    let temperature = 0.7f32;
    let rng = named("rng", Channel::from([0x51edu32, 3]));
    let tok_out = named("tok_out", Channel::new([rows], dtype::i32).capacity(4));
    let mut b = builder(vocab, rows);
    b.stage(Stage::Epilogue, move || {
        let r = rng.take();
        let logits = cast(
            reshape(intrinsics::output(0, "logits"), [rows, vocab]),
            dtype::bf16,
        );
        let token = if truncated {
            let probs = softmax(&logits / temperature);
            let keep = and(
                pivot_threshold(&probs, rank_le(40u32)),
                pivot_threshold(&probs, cummass_le(0.9f32)),
            );
            gumbel_argmax(&logits, keep, temperature, &r)
        } else {
            gumbel_argmax(&logits, true, temperature, &r)
        };
        tok_out.put(reshape(cast(token, dtype::i32), [rows]));
        rng.put(&r + iota(2));
    });
    tok_out.note_host_take();
    let mut seeds = port_seeds(rows);
    seeds.push(("rng", Value::U32(vec![0x51ed, 3])));
    Suite {
        names: take_names(),
        name: format!(
            "bf16 gumbel_argmax ({rows} row(s), {})",
            if truncated {
                "top-k + top-p"
            } else {
                "every lane"
            }
        ),
        traced: b.build().expect("the bf16 draw traces"),
        rows,
        vocab,
        seeds,
        outputs: vec![("tok_out", Dtype::I32, Check::Exact)],
    }
}

/// A bare argmax, and a log-softmax's sum and max, over `rows` bf16 rows of
/// Qwen3's vocabulary in one region: past `LANE_SAMPLER_BLOCKS` rows the
/// launch does not split them, and each row's region still runs every phase
/// its reductions fold in.
fn split_rows(rows: u32) -> Suite {
    let vocab = QWEN_VOCAB;
    let greedy = named("greedy", Channel::new([rows], dtype::i32).capacity(4));
    let scored = named("scored", Channel::new([rows], dtype::i32).capacity(4));
    let mut builder = builder(vocab, rows);
    builder.stage(Stage::Epilogue, move || {
        let logits = cast(
            reshape(intrinsics::output(0, "logits"), [rows, vocab]),
            dtype::bf16,
        );
        greedy.put(reshape(cast(reduce_argmax(&logits), dtype::i32), [rows]));
        scored.put(reshape(
            cast(reduce_argmax(log_softmax(&logits)), dtype::i32),
            [rows],
        ));
    });
    for channel in [greedy, scored] {
        channel.note_host_take();
    }
    Suite {
        names: take_names(),
        name: format!("split rows ({rows} row(s))"),
        traced: builder.build().expect("the split rows trace"),
        rows,
        vocab,
        seeds: port_seeds(rows),
        outputs: vec![
            ("greedy", Dtype::I32, Check::Exact),
            ("scored", Dtype::I32, Check::Exact),
        ],
    }
}

/// Every suite whose vocabulary-sized values are bf16.
fn bf16_suites() -> Vec<Suite> {
    let penalized = eta_dsl::sampler::Shape {
        temperature: true,
        top_p: true,
        min_p: true,
        repetition: true,
        frequency: true,
        presence: true,
        ..eta_dsl::sampler::Shape::GREEDY
    };
    let mut suites = vec![
        bf16_battery(1),
        bf16_battery(3),
        bf16_draw(1, false),
        bf16_draw(1, true),
        bf16_draw(3, false),
        bf16_draw(3, true),
    ];
    suites.extend(
        eta_dsl::sampler::standard()
            .iter()
            .map(|(name, shape)| standard(name, shape)),
    );
    suites.push(standard("penalized min-p + top-p", &penalized));
    // Seventeen rows: more than a split row's launch splits
    // (`LANE_SAMPLER_BLOCKS`), each row on one block through every phase.
    suites.extend([
        beam_bf16(1),
        beam_bf16(4),
        beam_bf16(17),
        split_rows(16),
        split_rows(17),
    ]);
    suites
}

/// The bf16 suites on the device against the host interpreter, which
/// computes in f32 and rounds each bf16 store once: every value either
/// computes without `exp` or `log` is the interpreter's bit for bit.
#[test]
fn every_bf16_suite_step_agrees_with_the_host_interpreter() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a CUDA context");
    let failures: Vec<String> = bf16_suites()
        .iter()
        .flat_map(|suite| {
            let failed = run(&context, suite);
            eprintln!("{}: {} disagreement(s)", suite.name, failed.len());
            failed
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} disagreement(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
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

/// A program's registration compiles off the thread that fires
/// (`Programs::prepare`): a lane of the sampling primitives fires on
/// while the beam step's regions compile cold on another thread, and the
/// beam step registers once they have.
#[test]
fn a_cold_registration_compiles_while_a_lane_fires() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a CUDA context");
    let serving = sampling_primitives();
    let (bound, serving_registration) = registration(&serving);
    let mut plane = crate::guest::alone_plane();
    let program = plane
        .register(&context, &serving_registration)
        .expect("the sampling primitives compile");
    let count = bound.container.channels.len();
    let index = |name: &str| {
        let gid = serving
            .names
            .iter()
            .find(|(n, _)| *n == name)
            .expect("named")
            .1;
        let order = serving.traced.channel_order();
        order.iter().position(|&g| g == gid).expect("in the trace") as u32
    };
    let seeds: Vec<(u32, Vec<u8>)> = serving
        .seeds
        .iter()
        .map(|(name, value)| (index(name), wire_bytes(value)))
        .collect();
    let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
    let lane = plane
        .bind(
            program,
            &seeds,
            Extents {
                row_count: 1,
                sampled_rows: 1,
                query_len: 1,
                ..Extents::default()
            },
            GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            false,
            &vec![None; count],
            &ids,
            crate::guest::alone_slots(&plane, program),
            &crate::guest::alone_span,
            context.stream(),
        )
        .expect("the lane binds");
    let values = logits(1, serving.vocab, 7);
    let mut buffer = Buffer::zeroed(values.len() * 4).expect("a logits plane");
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    buffer.write(0, &bytes).expect("the plane uploads");
    let fire = |plane: &mut engine_cuda::ProgramPlane| {
        plane
            .bind_output(
                lane,
                "logits",
                0,
                buffer.ptr(),
                INTRINSIC_STORAGE_F32,
                serving.vocab,
                serving.vocab,
                0,
                1,
            )
            .expect("the logits bind");
        let launched = plane.stage(lane, Stage::Epilogue).expect("the lane stages");
        assert!(
            matches!(launched, Launched::Airborne),
            "refused: {launched:?}"
        );
        plane.fly(&context).expect("the wave flies");
        plane.land(&context).expect("the wave lands");
        context.synchronize().expect("the device synchronizes");
        let fired = plane.settle_launched(lane).expect("the lane settles");
        assert!(matches!(fired, Fired::Committed), "{fired:?}");
        let session = plane.instance_mut(lane).expect("the instance");
        for &(name, _, _) in &serving.outputs {
            session.take(index(name)).expect("a channel reads");
        }
    };
    fire(&mut plane);
    let compiled = plane.stats().compilations;

    let (_, cold) = registration(&beam(4));
    let mut pending = plane.prepare(&context, &cold).expect("the beam step plans");
    let began = std::time::Instant::now();
    let compile = std::thread::spawn(move || {
        pending.build();
        (pending, began.elapsed())
    });
    let (mut fires, mut longest, mut last) = (0u32, std::time::Duration::ZERO, began);
    while !compile.is_finished() {
        fire(&mut plane);
        fires += 1;
        longest = longest.max(last.elapsed());
        last = std::time::Instant::now();
    }
    let (pending, built) = compile.join().expect("the compile finishes");
    let finishing = std::time::Instant::now();
    let beam = plane.finish(pending).expect("the beam step registers");
    let finished = finishing.elapsed();
    eprintln!(
        "beam step: compiled in {built:?} off the lane, finished in {finished:?}; \
         the lane fired {fires} times meanwhile, at most {longest:?} apart"
    );
    assert!(plane.program(beam).is_some(), "the beam step is registered");
    assert!(
        plane.stats().compilations > compiled,
        "the beam step compiled cold: {:?}",
        plane.stats()
    );
    assert!(
        fires >= 3 && longest < built / 2,
        "the lane stalled behind the compile: {fires} fires in {built:?}, at most {longest:?} apart"
    );
    fire(&mut plane);
}
