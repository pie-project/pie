//! A whole model with a sampler attached to every lane: each decode step's
//! guest pass runs on the device, all lanes of one program as one batched
//! executable reading the fire's readout where it lies, and samples what
//! the host interpreter samples from the same logits. Prints the host time
//! the guest pass costs both ways.
//!
//! Asked for with `PIE_XLA_SNAPSHOT` + `PIE_XLA_SKU` (see
//! `a_model_speaks_on_the_device`); `PIE_XLA_SAMPLER_LANES` sets the batch
//! (default 8), `PIE_XLA_SAMPLER_STEPS` the decode steps (default 12),
//! `PIE_XLA_SAMPLER_TOP_K` adds a top-k cut (slow on the host side),
//! `PIE_XLA_SAMPLER_PENALTIES=0` drops the vocabulary-wide histogram.

mod common;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use engine::channel::ChannelSeed;
use engine::program::{InstanceBinding, ProgramRegistration};
use engine_xla::program::Plane;
use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use eta_compiler::plan::compile_bound;
use eta_exec::Value;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape};
use poem::{Platform, Request};
use poem_compiler::Budget;

const RNG: u32 = 0;
const COUNTS: u32 = 1;
const OUT: u32 = 2;
const GREEDY: u32 = 3;

struct Program {
    ops: Vec<Op>,
    next: u32,
}

impl Program {
    fn op(&mut self, op: Op) -> u32 {
        let id = self.next;
        self.next += op.result_count();
        self.ops.push(op);
        id
    }
}

/// The compat sampler's shape: penalties over a carried histogram,
/// temperature, top-k, top-p, a keyed Gumbel draw; the greedy token beside.
fn sampler(vocab: u32, top_k: bool, penalties: bool) -> TraceContainer {
    let v = Shape::vector(vocab);
    let mut p = Program {
        ops: Vec::new(),
        next: 0,
    };
    let r = p.op(Op::ChanTake(RNG));
    let raw = p.op(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(1, vocab),
        dtype: Dtype::F32,
    });
    let mut logits = p.op(Op::Reshape {
        value: raw,
        shape: v,
    });
    let counts = penalties.then(|| p.op(Op::ChanTake(COUNTS)));
    if let Some(counts) = counts {
        let zero = p.op(Op::Const(Literal::F32(0.0)));
        let zb = p.op(Op::Broadcast {
            value: zero,
            shape: v,
        });
        let seen = p.op(Op::Gt(counts, zb));
        let rp = p.op(Op::Const(Literal::F32(1.1)));
        let rpb = p.op(Op::Broadcast {
            value: rp,
            shape: v,
        });
        let positive = p.op(Op::Gt(logits, zb));
        let shrunk = p.op(Op::Div(logits, rpb));
        let grown = p.op(Op::Mul(logits, rpb));
        let rep = p.op(Op::Select {
            cond: positive,
            a: shrunk,
            b: grown,
        });
        let penalized = p.op(Op::Select {
            cond: seen,
            a: rep,
            b: logits,
        });
        let fp = p.op(Op::Const(Literal::F32(0.3)));
        let fpb = p.op(Op::Broadcast {
            value: fp,
            shape: v,
        });
        let freq = p.op(Op::Mul(fpb, counts));
        logits = p.op(Op::Sub(penalized, freq));
    }
    let greedy = p.op(Op::ReduceArgmax(logits));
    let temp = p.op(Op::Const(Literal::F32(0.8)));
    let scaled = p.op(Op::Div(logits, temp));
    let m = p.op(Op::ReduceMax(scaled));
    let mb = p.op(Op::Broadcast { value: m, shape: v });
    let c = p.op(Op::Sub(scaled, mb));
    let e = p.op(Op::Exp(c));
    let s = p.op(Op::ReduceSum(e));
    let sb = p.op(Op::Broadcast { value: s, shape: v });
    let probs = p.op(Op::Div(e, sb));
    let top_p = p.op(Op::Const(Literal::F32(0.9)));
    let mut keep = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::CummassLe(top_p),
    });
    if top_k {
        // The interpreter's rank_le counts, per lane, the lanes above it:
        // quadratic in the vocabulary, so the host reference is asked for.
        let k = p.op(Op::Const(Literal::U32(50)));
        let keep_k = p.op(Op::PivotThreshold {
            input: probs,
            predicate: Predicate::RankLe(k),
        });
        keep = p.op(Op::And(keep_k, keep));
    }
    let ninf = p.op(Op::Const(Literal::F32(f32::NEG_INFINITY)));
    let truncated = p.op(Op::Select {
        cond: keep,
        a: scaled,
        b: ninf,
    });
    let noise = p.op(Op::RngKeyed {
        state: r,
        shape: v,
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(truncated, noise));
    let token = p.op(Op::ReduceArgmax(perturbed));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let greedy1 = p.op(Op::Reshape {
        value: greedy,
        shape: Shape::vector(1),
    });
    let step = p.op(Op::Iota { len: 2 });
    let r_next = p.op(Op::Add(r, step));
    p.op(Op::ChanPut {
        chan: OUT,
        value: token1,
    });
    p.op(Op::ChanPut {
        chan: GREEDY,
        value: greedy1,
    });
    p.op(Op::ChanPut {
        chan: RNG,
        value: r_next,
    });
    if let Some(counts) = counts {
        let one = p.op(Op::Const(Literal::F32(1.0)));
        let next = p.op(Op::ScatterAdd {
            base: counts,
            idx: token,
            vals: one,
        });
        p.op(Op::ChanPut {
            chan: COUNTS,
            value: next,
        });
    }
    let chan =
        |shape: Shape, dtype: Dtype, role: HostRole, seeded: bool, capacity: u32| ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity,
            host_role: role,
            seeded,
        };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            chan(Shape::vector(2), Dtype::U32, HostRole::None, true, 1),
            chan(v, Dtype::F32, HostRole::None, true, 1),
            chan(Shape::vector(1), Dtype::I32, HostRole::Reader, false, 2),
            chan(Shape::vector(1), Dtype::I32, HostRole::Reader, false, 2),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: p.ops,
        }],
        externs: Vec::new(),
    }
}

fn wire(v: &Value) -> Vec<u8> {
    let mut bytes = vec![0u8; eta_exec::wire_cell_bytes(v.dtype(), v.len())];
    eta_exec::encode_wire(v, &mut bytes);
    bytes
}

fn token(plane: &mut Plane, instance: u64, chan: u32) -> i32 {
    let bytes = plane
        .take(instance, chan)
        .expect("the channel reads")
        .expect("the pass put a token");
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
fn a_batch_samples_on_the_device_as_the_interpreter_does() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_SNAPSHOT + PIE_XLA_SKU or PIE_XLA_ARTIFACT");
        return;
    };
    let lanes: u32 = std::env::var("PIE_XLA_SAMPLER_LANES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let steps: usize = std::env::var("PIE_XLA_SAMPLER_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let sku = m.sku;
    let facts = sku.trace(models::Platform::Xla).facts;
    let word = |query_len: u32| facts.word(&Request::new(query_len, false));
    let context = 512;
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace: sku.trace(Platform::Xla),
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(lanes, 1024),
        page_size: 16,
        context,
        slots: lanes,
        pages: lanes * context / 16,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    let vocab = shell.out_width();

    // One program, registered with both planes; an instance per lane.
    let profile = ModelProfile {
        vocab,
        ..ModelProfile::dummy()
    };
    let bound = eta_ir::validate::bind(
        sampler(
            vocab,
            std::env::var_os("PIE_XLA_SAMPLER_TOP_K").is_some(),
            std::env::var("PIE_XLA_SAMPLER_PENALTIES").map_or(true, |v| v != "0"),
        ),
        profile,
    )
    .expect("the sampler binds");
    let launch = eta_compiler::codegen::launch::build(&bound, &compile_bound(&bound));
    let registration = ProgramRegistration {
        launch,
        ..Default::default()
    };
    let mut device_plane = Plane::default();
    let mut host_plane = Plane::default();
    let on_device = device_plane.register(&registration).expect("registers");
    let on_host = host_plane.register(&registration).expect("registers");
    assert!(
        device_plane.device_form(on_device).0,
        "{:?}",
        device_plane.device_form(on_device).1
    );
    let mut ids: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
    for lane in 0..lanes {
        let seeds = vec![
            ChannelSeed {
                channel: RNG,
                bytes: wire(&Value::U32(vec![0x51ed ^ lane, 1])),
            },
            ChannelSeed {
                channel: COUNTS,
                bytes: wire(&Value::F32(vec![0.0; vocab as usize])),
            },
        ];
        let bind = |program| InstanceBinding {
            program,
            channels: (0..4).map(|c| u64::from(lane * 16 + c + 1)).collect(),
            seeds: seeds.clone(),
            geometry: GeometryClass::Host,
            extents: Default::default(),
        };
        let d = device_plane.bind(&bind(on_device)).expect("binds").id;
        let h = host_plane.bind(&bind(on_host)).expect("binds").id;
        ids.insert(lane, (d, h));
    }

    // Prefill every lane with its own prompt.
    let tokenizer = common::tokenizer(&m);
    let prompts = [
        "The capital of France is",
        "Once upon a time",
        "The quick brown fox",
        "In the beginning",
        "My favourite food is",
        "The meaning of life is",
        "Water boils at",
        "The best way to learn",
    ];
    let mut fed: Vec<u32> = Vec::new();
    for lane in 0..lanes {
        let prompt = tokenizer.encode(prompts[lane as usize % prompts.len()]);
        shell.open(lane).expect("a slot opens");
        let rows = shell
            .fire(&[Lane {
                slot: lane,
                word: word(prompt.len() as u32),
                tokens: &prompt,
            }])
            .expect("the prefill fires");
        let row = &rows[0];
        let best = (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap_or(0);
        fed.push(best as u32);
    }

    let (mut fire_t, mut device_t, mut read_t, mut host_t) = (
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    let (mut agreed, mut greedy_agreed, mut total) = (0usize, 0usize, 0usize);
    let mut texts: Vec<Vec<u32>> = vec![Vec::new(); lanes as usize];
    for step in 0..=steps {
        let toks: Vec<[u32; 1]> = fed.iter().map(|&t| [t]).collect();
        let seated: Vec<Seated<'_>> = toks
            .iter()
            .enumerate()
            .map(|(slot, t)| {
                Seated::of(Lane {
                    slot: slot as u32,
                    word: word(1),
                    tokens: t,
                })
            })
            .collect();
        let t0 = Instant::now();
        let fired = shell.fire_kept(&seated).expect("a decode fires");
        let kept = fired.kept.expect("the fire kept its readout");
        let t1 = Instant::now();
        let attached: Vec<(u64, u32)> = (0..lanes).map(|l| (ids[&l].0, l)).collect();
        let ran = device_plane
            .fire_guests(shell.device(), &attached, Some(&kept), vocab, 0)
            .expect("the guest passes run");
        assert!(ran.iter().all(|&r| r), "every lane ran on the device");
        let device_tokens: Vec<(i32, i32)> = (0..lanes)
            .map(|l| {
                (
                    token(&mut device_plane, ids[&l].0, OUT),
                    token(&mut device_plane, ids[&l].0, GREEDY),
                )
            })
            .collect();
        let t2 = Instant::now();
        // The old path: every lane's rows to the host, then the interpreter.
        let rows: Vec<Vec<f32>> = (0..lanes as usize)
            .map(|l| kept.lane_rows(l).expect("rows"))
            .collect();
        let t3 = Instant::now();
        let host_tokens: Vec<(i32, i32)> = (0..lanes)
            .map(|l| {
                let h = ids[&l].1;
                host_plane
                    .fire_interpreted(h, Some(&rows[l as usize]), 1, vocab, None, 0)
                    .expect("the interpreter steps");
                (
                    token(&mut host_plane, h, OUT),
                    token(&mut host_plane, h, GREEDY),
                )
            })
            .collect();
        let t4 = Instant::now();
        if step > 0 {
            // Step 0 compiles.
            fire_t += t1 - t0;
            device_t += t2 - t1;
            read_t += t3 - t2;
            host_t += t4 - t3;
        }
        for (d, h) in device_tokens.iter().zip(&host_tokens) {
            total += 1;
            agreed += usize::from(d.0 == h.0);
            greedy_agreed += usize::from(d.1 == h.1);
        }
        for (l, (d, h)) in device_tokens.iter().zip(&host_tokens).enumerate() {
            if d.0 != h.0 {
                eprintln!(
                    "step {step} lane {l}: device sampled {} and the interpreter {}",
                    d.0, h.0
                );
            }
        }
        fed = device_tokens.iter().map(|&(t, _)| t as u32).collect();
        for (l, &t) in fed.iter().enumerate() {
            texts[l].push(t);
        }
    }
    for (l, text) in texts.iter().enumerate().take(4) {
        eprintln!("lane {l}: {:?}", tokenizer.decode(text, false));
    }
    let per = |d: Duration| d.as_secs_f64() * 1e3 / steps as f64;
    eprintln!(
        "{lanes} lanes x {steps} steps (vocab {vocab}): fire {:.2} ms/step; guest pass on the device \
         {:.2} ms/step; the interpreter's path: readout download {:.2} ms + interpreter {:.2} ms \
         = {:.2} ms/step",
        per(fire_t),
        per(device_t),
        per(read_t),
        per(host_t),
        per(read_t + host_t)
    );
    let (runs, carried) = device_plane.stage_runs();
    eprintln!(
        "device stage runs {runs} carrying {carried} lanes; samples agreed {agreed}/{total}, greedy {greedy_agreed}/{total}"
    );
    assert_eq!(greedy_agreed, total, "the greedy token is exact");
    // The Gumbel draw rounds differently on the device; a flip needs two
    // perturbed logits within an ulp of each other.
    assert!(
        agreed * 100 >= total * 98,
        "sampled tokens agree ({agreed}/{total})"
    );
}
