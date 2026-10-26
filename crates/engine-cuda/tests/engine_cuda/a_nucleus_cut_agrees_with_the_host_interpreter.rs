//! `PivotThreshold(CummassLe)` — the top-p cut — on the device against the
//! host interpreter, bit for bit: the reference walks a row in descending
//! order and sums its mass sequentially in f32, so the kept set is decided by
//! that exact rounding, and the device must land on the same boundary.

use engine::ProgramRegistration;
use engine::{ChannelId, InstanceId};
use engine_cuda::device::Context;
use engine_cuda::program::Plane;
use engine_cuda::program::compile::Disk;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype, Predicate, Shape};
use eta_ir::validate::BoundTrace;

const INPUT: u32 = 0;
const THRESHOLD: u32 = 1;
const KEEP: u32 = 2;

const QWEN_VOCAB: u32 = 151_936;

#[derive(Clone, Copy)]
enum Cut {
    Mass,
    Rank,
}

fn program(rows: u32, len: u32, cut: Cut) -> TraceContainer {
    let plane = Shape::matrix(rows, len);
    let predicate = match cut {
        Cut::Mass => Predicate::CummassLe(1),
        Cut::Rank => Predicate::RankLe(1),
    };
    let threshold_dtype = match cut {
        Cut::Rank => Dtype::I32,
        _ => Dtype::F32,
    };
    let ops = vec![
        Op::ChanRead(INPUT),
        Op::ChanRead(THRESHOLD),
        Op::PivotThreshold {
            input: 0,
            predicate,
        },
        Op::ChanPut {
            chan: KEEP,
            value: 2,
        },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            ChannelDecl {
                shape: plane,
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity: 1,
                host_role: HostRole::Writer,
                seeded: true,
            },
            ChannelDecl {
                shape: Shape::vector(rows),
                dtype: ChanDType::Concrete(threshold_dtype),
                capacity: 1,
                host_role: HostRole::Writer,
                seeded: true,
            },
            ChannelDecl {
                shape: plane,
                dtype: ChanDType::Concrete(Dtype::Bool),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

struct Rig {
    plane: Plane,
    context: Context,
    instance: InstanceId,
}

impl Rig {
    fn new(bound: &BoundTrace, seeds: &[(u32, Value)], rows: u32) -> Rig {
        let context = Context::bind(0, None).expect("a device context");
        let stages = compile_bound(bound);
        let launch = eta_compiler::codegen::launch::build(bound, &stages);
        let backend =
            Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
        let registration = ProgramRegistration {
            program_hash: 0x70_9c07 ^ u64::from(rows),
            emitted_kernels: emit_program(backend, &stages, bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        };
        let mut plane = Plane::new(Disk::disabled());
        let program = plane
            .register(&context, &registration)
            .expect("the program compiles for the device");
        let extents = Extents {
            row_count: rows,
            sampled_rows: rows,
            query_len: rows,
            ..Extents::default()
        };
        let wire_seeds: Vec<(u32, Vec<u8>)> =
            seeds.iter().map(|(c, v)| (*c, v.to_le_bytes())).collect();
        let count = bound.container.channels.len();
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        let instance = plane
            .bind(
                program,
                &wire_seeds,
                extents,
                GeometryClass::Host,
                eta_ir::registry::PortMask::NONE,
                false,
                &vec![None; count],
                &ids,
                context.stream(),
            )
            .expect("the instance binds");
        Rig {
            plane,
            context,
            instance,
        }
    }

    fn fire(&mut self) -> Vec<u8> {
        let fired = self
            .plane
            .fire(&self.context, self.instance)
            .expect("the fire runs");
        assert!(
            matches!(fired, Fired::Committed),
            "the device fire did not commit: {fired:?}"
        );
        self.plane
            .instance_mut(self.instance)
            .expect("the instance")
            .take(KEEP)
            .expect("a channel reads")
            .expect("the channel holds a cell")
    }
}

fn host_keep(bound: &BoundTrace, seeds: &[(u32, Value)]) -> Vec<bool> {
    let mut reference = Instance::new(bound, seeds).expect("the interpreter instantiates");
    let stepped = reference
        .step(bound, &PassInputs::default(), &mut NoKernels)
        .expect("the interpreter steps");
    assert!(stepped.committed, "the reference did not commit");
    match reference.host_take(bound, KEEP).unwrap() {
        Value::Bool(v) => v,
        other => panic!("expected bool, got {other:?}"),
    }
}

fn wire_bools(bytes: &[u8], count: usize) -> Vec<bool> {
    (0..count)
        .map(|i| bytes[i / 8] >> (i % 8) & 1 == 1)
        .collect()
}

struct Noise(u64);

impl Noise {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn below(&mut self, n: u32) -> u32 {
        ((self.unit() as f64 * f64::from(n)) as u32).min(n - 1)
    }
}

/// A softmax over `scale`-wide uniform logits, as the device's own softmax
/// would leave it: f32 exponentials over an f32 sum.
fn softmax(noise: &mut Noise, len: u32, scale: f32) -> Vec<f32> {
    let logits: Vec<f32> = (0..len).map(|_| (noise.unit() - 0.5) * scale).collect();
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| (l - peak).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&x| x / sum).collect()
}

/// A softmax over gaussian logits of deviation `scale`: at 4 the 0.9 nucleus
/// of a Qwen-wide row is some 800 tokens, at 3 some 5000 — the spread of a
/// trained model's next-token distribution.
fn gaussian(noise: &mut Noise, len: u32, scale: f32) -> Vec<f32> {
    let logits: Vec<f32> = (0..len)
        .map(|_| {
            let (u, v) = (noise.unit().max(1e-12), noise.unit());
            (-2.0 * u.ln()).sqrt() * (core::f32::consts::TAU * v).cos() * scale
        })
        .collect();
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| (l - peak).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&x| x / sum).collect()
}

fn peaked(noise: &mut Noise, len: u32) -> Vec<f32> {
    gaussian(noise, len, 4.0)
}

fn spiked(noise: &mut Noise, len: u32) -> Vec<f32> {
    let logits: Vec<f32> = (0..len)
        .map(|i| {
            let u = noise.unit();
            let base = -12.0 * u.powf(0.35);
            if i % 4099 == 17 { base + 14.0 } else { base }
        })
        .collect();
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| (l - peak).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&x| x / sum).collect()
}

/// Probabilities rounded to bf16: long runs of equal values, which the cut
/// must take in index order.
fn bf16_rounded(row: &[f32]) -> Vec<f32> {
    row.iter()
        .map(|&x| f32::from_bits((x.to_bits() + 0x8000) & 0xffff_0000))
        .collect()
}

/// Small dyadic masses `n * 2^-28`: the sequential sum lands exactly halfway
/// between two floats again and again, so ties-to-even decides the boundary.
fn dyadic(noise: &mut Noise, len: u32) -> Vec<f32> {
    (0..len)
        .map(|_| (1 + noise.below(4096)) as f32 * (2.0f32).powi(-28))
        .collect()
}

fn compare(rows: u32, len: u32, data: Vec<f32>, thresholds: Vec<f32>, what: &[&str]) {
    compare_cut(Cut::Mass, rows, len, data, &thresholds, what);
}

fn compare_rank(rows: u32, len: u32, data: Vec<f32>, ks: &[i32], what: &[&str]) {
    let as_f32: Vec<f32> = ks.iter().map(|&k| k as f32).collect();
    compare_cut(Cut::Rank, rows, len, data, &as_f32, what);
}

fn compare_cut(cut: Cut, rows: u32, len: u32, data: Vec<f32>, thresholds: &[f32], what: &[&str]) {
    let bound = crate::bound::logits(program(rows, len, cut), len).expect("the cut binds");
    let threshold = match cut {
        Cut::Rank => Value::I32(thresholds.iter().map(|&k| k as i32).collect()),
        _ => Value::F32(thresholds.to_vec()),
    };
    let seeds = vec![(INPUT, Value::F32(data.clone())), (THRESHOLD, threshold)];
    let want = host_keep(&bound, &seeds);
    let mut rig = Rig::new(&bound, &seeds, rows);
    let got = wire_bools(&rig.fire(), (rows * len) as usize);
    for row in 0..rows as usize {
        let span = row * len as usize..(row + 1) * len as usize;
        let (w, g) = (&want[span.clone()], &got[span]);
        let kept_w = w.iter().filter(|&&b| b).count();
        let kept_g = g.iter().filter(|&&b| b).count();
        let first = w.iter().zip(g).position(|(a, b)| a != b);
        assert!(
            first.is_none(),
            "row {row} ({}; threshold {}): the device keeps {kept_g}, the interpreter {kept_w}; \
             first disagreement at column {}",
            what[row],
            thresholds[row],
            first.unwrap_or(0)
        );
        eprintln!(
            "row {row} ({}): {kept_w} of {len} kept, as the interpreter",
            what[row]
        );
    }
}

/// The cut's plan rests on f32 bucket sums that device atomics add in no set
/// order; its verdicts must not: the same rows, fired over and over, keep
/// the same elements bit for bit (and the interpreter's).
fn a_cut_keeps_the_same_elements_every_fire() {
    // A Qwen-next vocabulary: rows as a random-weight model's and a trained
    // one's next-token distributions spread, and bf16 runs of equal values.
    let len = 248_320;
    let mut noise = Noise(0x0dd5_eed5_c0ff_ee11);
    let rows: Vec<(&str, Vec<f32>, f32)> = vec![
        ("flat", softmax(&mut noise, len, 2.0), 0.9),
        ("mini-like", gaussian(&mut noise, len, 1.2), 0.9),
        ("trained", gaussian(&mut noise, len, 3.5), 0.9),
        ("peaked", peaked(&mut noise, len), 0.95),
        (
            "bf16 ties",
            bf16_rounded(&gaussian(&mut noise, len, 2.0)),
            0.5,
        ),
    ];
    let count = rows.len() as u32;
    let thresholds: Vec<f32> = rows.iter().map(|r| r.2).collect();
    let data: Vec<f32> = rows.into_iter().flat_map(|r| r.1).collect();
    let bound = crate::bound::logits(program(count, len, Cut::Mass), len).expect("the cut binds");
    let seeds = vec![
        (INPUT, Value::F32(data)),
        (THRESHOLD, Value::F32(thresholds)),
    ];
    let want = host_keep(&bound, &seeds);
    let mut rig = Rig::new(&bound, &seeds, count);
    let first = rig.fire();
    assert!(
        wire_bools(&first, (count * len) as usize) == want,
        "the first fire's verdicts are not the interpreter's"
    );
    for fire in 1..32 {
        assert!(
            rig.fire() == first,
            "fire {fire} kept other elements than the first"
        );
    }
}

#[test]
fn a_nucleus_cut_agrees_with_the_host_interpreter_every_case() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    a_vocabulary_wide_cut_matches_bit_for_bit();
    a_cut_keeps_the_same_elements_every_fire();
    the_edges_of_the_cut_match();
    a_row_outside_the_mass_contract_still_matches();
    a_top_k_cut_matches();
}

fn a_top_k_cut_matches() {
    let len = QWEN_VOCAB;
    let mut noise = Noise(0x0bad_cafe_f00d_d00d);
    let flat = softmax(&mut noise, len, 2.0);
    let peak = peaked(&mut noise, len);
    let ties = bf16_rounded(&softmax(&mut noise, len, 6.0));
    let equal = vec![1.0f32 / len as f32; len as usize];
    let mut with_nan = softmax(&mut noise, len, 4.0);
    for i in (3..len as usize).step_by(7919) {
        with_nan[i] = f32::NAN;
    }
    let signed: Vec<f32> = (0..len).map(|_| noise.unit() - 0.5).collect();
    let rows: Vec<(&str, Vec<f32>, i32)> = vec![
        ("flat, k 40", flat.clone(), 40),
        ("peaked, k 40", peak.clone(), 40),
        ("bf16 ties, k 1000", ties.clone(), 1000),
        ("bf16 ties, k 1", ties, 1),
        ("all equal, k 5", equal, 5),
        ("nan sprinkled, k 50", with_nan.clone(), 50),
        ("nan sprinkled, k all", with_nan, len as i32),
        ("signed, k 3", signed, 3),
        ("flat, k 0", flat.clone(), 0),
        ("flat, k -2", flat.clone(), -2),
        ("flat, k past the row", flat, len as i32 + 9),
        ("peaked, k 70000", peak, 70_000),
    ];
    let count = rows.len() as u32;
    let what: Vec<&str> = rows.iter().map(|r| r.0).collect();
    let ks: Vec<i32> = rows.iter().map(|r| r.2).collect();
    let data: Vec<f32> = rows.into_iter().flat_map(|r| r.1).collect();
    compare_rank(count, len, data, &ks, &what);
}

fn a_vocabulary_wide_cut_matches_bit_for_bit() {
    let len = QWEN_VOCAB;
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    let flat = softmax(&mut noise, len, 2.0);
    let peak = peaked(&mut noise, len);
    let ties = bf16_rounded(&softmax(&mut noise, len, 6.0));
    let equal = vec![1.0f32 / len as f32; len as usize];
    let halves = dyadic(&mut noise, len);
    let half_units = vec![1023.0f32 * (2.0f32).powi(-30); len as usize];
    let mixed_halves: Vec<f32> = half_units
        .iter()
        .map(|&x| {
            if noise.below(3) == 0 {
                x * (1.0 + noise.unit())
            } else {
                x
            }
        })
        .collect();
    let mut with_nan = softmax(&mut noise, len, 4.0);
    for i in (5..len as usize).step_by(9973) {
        with_nan[i] = f32::NAN;
    }
    let broad = gaussian(&mut noise, len, 3.0);
    let spike = spiked(&mut noise, len);
    let rows: Vec<(&str, Vec<f32>, f32)> = vec![
        ("flat", flat.clone(), 0.9),
        ("peaked", peak.clone(), 0.9),
        ("broad", broad, 0.9),
        ("spiked", spike, 0.9),
        ("bf16 ties", ties.clone(), 0.5),
        ("all equal", equal, 0.9),
        ("dyadic halves", halves, 0.9),
        // 1023 * 2^-30 is half a unit off a whole once the sum passes 2^-6:
        // the walk crosses its binades inside a run of ties.
        ("a run of half units", half_units.clone(), 0.1),
        ("half units among others", mixed_halves, 0.15),
        ("nan sprinkled", with_nan.clone(), 0.95),
        ("nan sprinkled, whole mass", with_nan, 2.0),
        ("flat, whole mass", flat, 1.0),
    ];
    let count = rows.len() as u32;
    let what: Vec<&str> = rows.iter().map(|r| r.0).collect();
    let thresholds: Vec<f32> = rows.iter().map(|r| r.2).collect();
    let data: Vec<f32> = rows.into_iter().flat_map(|r| r.1).collect();
    compare(count, len, data, thresholds, &what);
}

fn the_edges_of_the_cut_match() {
    let len = 5000;
    let mut noise = Noise(0x0123_4567_89ab_cdef);
    let flat = softmax(&mut noise, len, 3.0);
    let peak = peaked(&mut noise, len);
    let mut zeros = softmax(&mut noise, len, 3.0);
    for (i, x) in zeros.iter_mut().enumerate() {
        if i % 3 == 0 {
            *x = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
    }
    let boundary = flat.clone();
    let exact: f32 = {
        let mut order: Vec<usize> = (0..len as usize).collect();
        order.sort_by(|&a, &b| {
            boundary[b]
                .partial_cmp(&boundary[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        let mut s = 0.0f32;
        for &i in order.iter().take(1234) {
            s += boundary[i];
        }
        s
    };
    let rows: Vec<(&str, Vec<f32>, f32)> = vec![
        ("cutoff 0", flat.clone(), 0.0),
        ("cutoff -1", flat.clone(), -1.0),
        ("cutoff NaN", flat.clone(), f32::NAN),
        ("cutoff tiny", flat.clone(), 1e-30),
        ("cutoff 1", flat.clone(), 1.0),
        ("cutoff inf", flat.clone(), f32::INFINITY),
        ("cutoff at an exact prefix sum", boundary.clone(), exact),
        (
            "cutoff one ulp past a prefix sum",
            boundary,
            f32::from_bits(exact.to_bits() + 1),
        ),
        ("peaked 0.5", peak.clone(), 0.5),
        ("peaked 0.999", peak, 0.999),
        ("signed zeros", zeros, 0.97),
        ("all zero", vec![0.0; len as usize], 0.5),
    ];
    let count = rows.len() as u32;
    let what: Vec<&str> = rows.iter().map(|r| r.0).collect();
    let thresholds: Vec<f32> = rows.iter().map(|r| r.2).collect();
    let data: Vec<f32> = rows.into_iter().flat_map(|r| r.1).collect();
    compare(count, len, data, thresholds, &what);
    compare(1, 1, vec![0.25], vec![0.5], &["a single column"]);
}

fn a_row_outside_the_mass_contract_still_matches() {
    let len = 3000;
    let mut noise = Noise(0x5555_aaaa_1234_9876);
    let signed: Vec<f32> = (0..len).map(|_| noise.unit() - 0.4).collect();
    let mut infinite = softmax(&mut noise, len, 3.0);
    infinite[17] = f32::INFINITY;
    infinite[1000] = f32::NEG_INFINITY;
    let huge: Vec<f32> = (0..len).map(|_| noise.unit() * 1e30).collect();
    let rows: Vec<(&str, Vec<f32>, f32)> = vec![
        ("signed values", signed, 0.3),
        ("infinities", infinite, 0.5),
        ("huge values", huge, 1e33),
    ];
    let count = rows.len() as u32;
    let what: Vec<&str> = rows.iter().map(|r| r.0).collect();
    let thresholds: Vec<f32> = rows.iter().map(|r| r.2).collect();
    let data: Vec<f32> = rows.into_iter().flat_map(|r| r.1).collect();
    compare(count, len, data, thresholds, &what);
}
