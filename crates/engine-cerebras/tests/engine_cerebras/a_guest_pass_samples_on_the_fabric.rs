//! A sampler attached to several lanes steps through the plane the engine
//! drives: each decode step's guest pass runs on the fabric simulator, one
//! lane per PE, reading the fire's kept readout, and samples what the host
//! interpreter samples from the same logits (the greedy token exactly; the
//! Gumbel draw rounds differently on the device, so a sampled token may
//! flip when two perturbed logits lie within an ulp).

use std::collections::BTreeMap;

use engine::channel::{ChannelId, ChannelSeed};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration};
use engine_cerebras::device::{Buffer, Device, Platform};
use engine_cerebras::program::Plane;
use engine_cerebras::readout::{Kept, Seat};
use engine_cerebras::sdk::Target;
use eta_compiler::plan::compile_bound;
use eta_exec::Value;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape};

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
fn sampler(vocab: u32) -> TraceContainer {
    let v = Shape::vector(vocab);
    let mut p = Program {
        ops: Vec::new(),
        next: 0,
    };
    let r = p.op(Op::ChanTake(RNG));
    let raw = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, vocab),
        dtype: Dtype::F32,
    });
    let logits = p.op(Op::Reshape {
        value: raw,
        shape: v,
    });
    let counts = p.op(Op::ChanTake(COUNTS));
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
    let logits = p.op(Op::Sub(penalized, freq));
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
    let keep_p = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::CummassLe(top_p),
    });
    let k = p.op(Op::Const(Literal::U32(12)));
    let keep_k = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::RankLe(k),
    });
    let keep = p.op(Op::And(keep_k, keep_p));
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
    let chan =
        |shape: Shape, dtype: Dtype, role: HostRole, seeded: bool, capacity: u32| ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity,
            host_role: role,
            seeded,
        };
    TraceContainer {
        names: vec!["logits".to_string()],
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
        inputs: Vec::new(),
    }
}

fn wire(v: &Value) -> Vec<u8> {
    let mut bytes = vec![0u8; eta_exec::wire_cell_bytes(v.dtype(), v.len())];
    eta_exec::encode_wire(v, &mut bytes);
    bytes
}

fn token(plane: &mut Plane, instance: InstanceId, chan: u32) -> i32 {
    let bytes = plane
        .take(instance, chan)
        .expect("the channel reads")
        .expect("the pass put a token");
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn logits_plane(rows: u32, vocab: u32, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut plane = Vec::with_capacity((rows * vocab) as usize);
    for row in 0..rows as usize {
        for c in 0..vocab as usize {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            let mut x = -4.0 + 8.0 * u + (c % 7) as f32 * 0.01;
            if c == (row * 97 + 5 + seed as usize) % vocab as usize {
                x = 9.0 + row as f32;
            }
            plane.push(x);
        }
    }
    plane
}

#[test]
fn a_guest_pass_samples_on_the_fabric_as_the_interpreter_does() {
    let device = match Device::open(Platform::Simulator {
        target: Target::Wse3,
    }) {
        Ok(device) => device,
        Err(e) => {
            eprintln!("no simulator ({e}): skipping");
            return;
        }
    };
    let vocab = 64u32;
    let lanes = 3u32;
    let steps = 4usize;
    let bound = bind_logits(sampler(vocab), vocab).expect("the sampler binds");
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
    let mut ids: BTreeMap<u32, (InstanceId, InstanceId)> = BTreeMap::new();
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
            channels: (0..4)
                .map(|c| ChannelId(u64::from(lane * 16 + c + 1)))
                .collect(),
            seeds: seeds.clone(),
            geometry: GeometryClass::Host,
            extents: Default::default(),
        };
        let d = device_plane.bind(&bind(on_device)).expect("binds").id;
        let h = host_plane.bind(&bind(on_host)).expect("binds").id;
        ids.insert(lane, (d, h));
    }
    let (mut agreed, mut greedy_agreed, mut total) = (0usize, 0usize, 0usize);
    for step in 0..steps {
        let plane = logits_plane(lanes, vocab, 100 + step as u64);
        let words: Vec<u32> = plane.iter().map(|x| x.to_bits()).collect();
        let buffer = Buffer::new(dtype::Dtype::F32, lanes, vocab, words).expect("a plane");
        let kept = Kept {
            planes: vec![engine_cerebras::readout::Plane {
                name: "logits".to_string(),
                axis: poem_ir::InputAxis::Row,
                readout: true,
                buffer,
                rows: lanes,
                width: vocab,
                layers: 0,
                dtype: dtype::Dtype::F32,
            }],
            seats: (0..lanes)
                .map(|l| vec![Some(Seat { first: l, count: 1 })])
                .collect(),
            readback: (0..lanes).map(|l| Some((0, vec![l]))).collect(),
            pixels: (String::new(), Vec::new()),
        };
        let attached: Vec<(InstanceId, u32)> = (0..lanes).map(|l| (ids[&l].0, l)).collect();
        let ran = device_plane
            .fire_guests(
                &device,
                &attached,
                Some(&kept),
                eta_ir::registry::Stage::Epilogue,
            )
            .expect("the guest passes run");
        assert!(
            ran.iter().all(|&r| r),
            "every lane ran on the fabric: {ran:?}"
        );
        for l in 0..lanes {
            let (d, h) = ids[&l];
            let row = kept.lane_value(l as usize, "logits").expect("rows");
            host_plane
                .fire_interpreted(
                    h,
                    &[("logits".to_string(), 0, row.expect("the lane reads logits"))],
                    eta_ir::registry::Stage::Epilogue,
                )
                .expect("the interpreter steps");
            let (dt, dg) = (
                token(&mut device_plane, d, OUT),
                token(&mut device_plane, d, GREEDY),
            );
            let (ht, hg) = (
                token(&mut host_plane, h, OUT),
                token(&mut host_plane, h, GREEDY),
            );
            total += 1;
            agreed += usize::from(dt == ht);
            greedy_agreed += usize::from(dg == hg);
            if dt != ht {
                eprintln!("step {step} lane {l}: fabric sampled {dt} and the interpreter {ht}");
            }
        }
    }
    let (runs, carried) = device_plane.stage_runs();
    eprintln!(
        "fabric stage runs {runs} carrying {carried} lane(s); samples agreed {agreed}/{total}, greedy {greedy_agreed}/{total}"
    );
    // The lanes step in lockstep: one program per stage per step carries
    // every lane.
    assert_eq!(runs, steps as u64, "one stage run per step");
    assert_eq!(
        carried,
        u64::from(lanes) * steps as u64,
        "every lane carried every step"
    );
    assert_eq!(greedy_agreed, total, "the greedy token is exact");
    assert!(
        agreed * 10 >= total * 9,
        "sampled tokens agree ({agreed}/{total})"
    );
}

/// Binds `container` as a one-lane pass reading `vocab`-wide logits.
fn bind_logits(
    container: eta_ir::container::TraceContainer,
    vocab: u32,
) -> Result<eta_ir::validate::BoundTrace, eta_ir::validate::ValidateError> {
    eta_ir::validate::bind(
        container,
        vec![vec![eta_ir::registry::OutputDecl {
            name: "logits".to_string(),
            axis: eta_ir::registry::Axis::Row,
            width: vocab,
            dtype: eta_ir::types::Dtype::F32,
            layers: 0,
            readout: true,
        }]],
        eta_ir::registry::EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
}
