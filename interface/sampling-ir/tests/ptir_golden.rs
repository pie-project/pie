//! Golden vectors (thrust-3 P0.4): canonical container bytes + identity hash
//! + validator verdict + tier-0 reference results, checked into
//! `tests/golden-ptir/*.txt`. **This is the conformance suite for every
//! backend** — charlie's CUDA tiers and delta's SDK diff against these files:
//! the SDK must emit byte-identical containers (same hex, same hash) and any
//! executor must reproduce the step lines exactly.
//!
//! Regenerate (bless) with:
//! `PTIR_REGEN=1 cargo test -p pie-sampling-ir --features eval --test ptir_golden`
#![cfg(feature = "eval")]

use pie_sampling_ir::eval::Value;
use pie_sampling_ir::ptir::container::{
    encode, ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram,
    TraceContainer,
};
use pie_sampling_ir::ptir::registry::Port;
use pie_sampling_ir::ptir::container_hash;
use pie_sampling_ir::ptir::interp::{Instance, NoKernels, PassInputs};
use pie_sampling_ir::ptir::op::{IntrinsicId, Op};
use pie_sampling_ir::ptir::registry::{ModelProfile, Stage};
use pie_sampling_ir::ptir::validate::{bind, BoundTrace};
use pie_sampling_ir::types::{DType, Literal, Shape};
use std::fmt::Write as _;

#[path = "common/traces.rs"]
mod traces;
use traces::*;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct Report(String);

impl Report {
    fn new(name: &str, c: &TraceContainer) -> Report {
        let bytes = encode(c);
        let mut s = String::new();
        writeln!(s, "name: {name}").unwrap();
        writeln!(s, "hash: 0x{:016x}", container_hash(&bytes)).unwrap();
        writeln!(s, "container: {}", hex(&bytes)).unwrap();
        Report(s)
    }
    fn verdict(mut self, r: &Result<BoundTrace, pie_sampling_ir::ptir::validate::ValidateError>) -> Report {
        match r {
            Ok(b) => {
                writeln!(self.0, "verdict: OK").unwrap();
                // The PTIB typed sidecar (PTIR-CONTAINER.md §7): per-value
                // (shape, dtype) + readiness + channel classes — what a
                // backend consumes instead of re-inferring (option B). The
                // readiness/class lines below restate it human-readably.
                writeln!(
                    self.0,
                    "sidecar: {}",
                    hex(&pie_sampling_ir::ptir::sidecar::encode_bound(b))
                )
                .unwrap();
                for e in &b.readiness {
                    writeln!(
                        self.0,
                        "readiness: chan={} phase=0x{:02x} dir={:?}",
                        e.chan,
                        e.phase.tag(),
                        e.dir
                    )
                    .unwrap();
                }
                for (i, cl) in b.classes.iter().enumerate() {
                    writeln!(self.0, "class: chan={i} {cl:?}").unwrap();
                }
            }
            Err(e) => writeln!(self.0, "verdict: ERR {e:?}").unwrap(),
        }
        self
    }
    fn line(mut self, l: &str) -> Report {
        writeln!(self.0, "{l}").unwrap();
        self
    }
}

/// Compare (or bless) one case's report against its golden file.
fn check(name: &str, report: Report) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden-ptir");
    let path = format!("{dir}/{name}.txt");
    if std::env::var("PTIR_REGEN").is_ok() {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&path, &report.0).unwrap();
        return;
    }
    let on_disk = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("{path} missing — bless with PTIR_REGEN=1"));
    assert_eq!(
        on_disk, report.0,
        "golden mismatch for {name} — if intentional, bless with PTIR_REGEN=1"
    );
}

fn step_line(n: usize, inst: &mut Instance, b: &BoundTrace, inputs: &PassInputs) -> String {
    // The `inputs` line makes each golden self-contained: a backend runner
    // replays the pass from the file alone (no transcription from this code).
    let fed = format!("inputs {n}: {inputs:?}");
    match inst.step(b, inputs, &mut NoKernels) {
        Ok(r) => format!(
            "{fed}\nstep {n}: committed={} missed={:?} sinks={}",
            r.committed,
            r.missed.map(|(c, p)| (c, p.tag())),
            r.sinks.len()
        ),
        Err(e) => format!("{fed}\nstep {n}: error {e:?}"),
    }
}

/// Emit the per-instance seed values (D2 data — NOT in the container/hash,
/// but a conformance runner needs them to reproduce the steps).
fn seed_lines(mut rep: Report, seeds: &[(u32, Value)]) -> Report {
    for (c, v) in seeds {
        rep = rep.line(&format!("seed chan={c} = {v:?}"));
    }
    rep
}

/// A host_put a runner must replay before the next step.
fn put_line(inst: &mut Instance, b: &BoundTrace, chan: u32, v: Value) -> String {
    let l = format!("host_put chan={chan} = {v:?}");
    inst.host_put(b, chan, v).unwrap();
    l
}

fn take_line(inst: &mut Instance, b: &BoundTrace, chan: u32) -> String {
    match inst.host_take(b, chan) {
        Ok(v) => format!("take chan={chan} = {v:?}"),
        Err(e) => format!("take chan={chan} = ERR {e:?}"),
    }
}

// ── positive cases ─────────────────────────────────────────────────────────

#[test]
fn golden_greedy_argmax() {
    // Minimal epilogue: tok/out <- argmax(logits).
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(1, 8),
        dtype: DType::F32,
    });
    let flat = b.p(Op::Reshape { value: lg, shape: Shape::vector(8) });
    let t = b.p(Op::ReduceArgmax(flat));
    let t1 = b.p(Op::Reshape { value: t, shape: Shape::vector(1) });
    b.p(Op::ChanPut { chan: 0, value: t1 });
    b.p(Op::ChanPut { chan: 1, value: t1 });
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: vec![PortBinding { port: Port::EmbedTokens, source: PortSource::Channel(0) }],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = 8;
    let bound = bind(c.clone(), profile).unwrap();
    let mut rep = Report::new("greedy_argmax", &c).verdict(&Ok(bound.clone()));
    let seeds = [(0u32, i32s(&[1]))];
    rep = seed_lines(rep, &seeds);
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let inputs = PassInputs { logits: Some(f32s(&[0., 1., 9., 2., 0., 0., 0., 3.])), ..Default::default() };
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1));
    let inputs = PassInputs { logits: Some(f32s(&[7., 1., 0., 2., 0., 0., 0., 3.])), ..Default::default() };
    rep = rep.line(&step_line(1, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1));
    check("greedy_argmax", rep);
}

#[test]
fn golden_section3_masked_gumbel() {
    let c = section3_trace();
    let bound = bind(c.clone(), ModelProfile::dummy()).unwrap();
    let mut rep = Report::new("section3_masked_gumbel", &c).verdict(&Ok(bound.clone()));
    let seeds = [(0u32, i32s(&[1])), (3u32, u32s(&[1])), (4u32, u32s(&[1234, 0]))];
    rep = seed_lines(rep, &seeds);
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let inputs = PassInputs { logits: Some(flat_logits(7, 100.0)), ..Default::default() };
    rep = rep.line(&put_line(&mut inst, &bound, 2, allow_all()));
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1));
    // Late mask: dummy-run, no commit.
    rep = rep.line(&step_line(1, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1));
    rep = rep.line(&put_line(&mut inst, &bound, 2, allow_only(&[3])));
    rep = rep.line(&step_line(2, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1));
    check("section3_masked_gumbel", rep);
}

#[test]
fn golden_beam_epilogue() {
    let c = beam_trace();
    let bound = bind(c.clone(), beam_profile()).unwrap();
    let mut rep = Report::new("beam_epilogue", &c).verdict(&Ok(bound.clone()));
    let seeds: Vec<(u32, Value)> = vec![
        (0, u32s(&[5, 6, 0, 5, 6, 0])),
        (1, u32s(&[4, 2, 0, 4, 2, 0])),
        (2, u32s(&[6, 6])),
        (3, {
            let mut m = vec![false; (BB * P * PAGE) as usize];
            for lane in 0..BB as usize {
                for j in 0..P as usize {
                    for o in 0..[4, 2, 0][j] {
                        m[lane * (P * PAGE) as usize + j * PAGE as usize + o] = true;
                    }
                }
            }
            Value::Bool(m)
        }),
        (4, u32s(&[6, 6])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[6, 6])),
        (7, u32s(&[2, 2])),
        (8, u32s(&[6, 6])),
        (9, u32s(&[2, 2])),
        (10, i32s(&[1, 2])),
        (11, f32s(&[0.0, 0.0])),
    ];
    rep = seed_lines(rep, &seeds);
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let mut logits = vec![0.0f32; (BB * V) as usize];
    logits[3] = 8.0;
    logits[(V + 5) as usize] = 7.0;
    let inputs = PassInputs { logits: Some(Value::F32(logits)), ..Default::default() };
    // Step 0: no fresh grant → miss.
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&put_line(&mut inst, &bound, 12, u32s(&[7, 8])));
    rep = rep.line(&step_line(1, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 13));
    rep = rep.line(&take_line(&mut inst, &bound, 14));
    rep = rep.line(&take_line(&mut inst, &bound, 15));
    check("beam_epilogue", rep);
}

#[test]
fn golden_counter_pingpong() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::U32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::U32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::Const(Literal::U32(1)),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
                Op::ChanPut { chan: 1, value: 2 },
            ],
        }],
    };
    let bound = bind(c.clone(), ModelProfile::dummy()).unwrap();
    let mut rep = Report::new("counter_pingpong", &c).verdict(&Ok(bound.clone()));
    let seeds = [(0u32, u32s(&[10]))];
    rep = seed_lines(rep, &seeds);
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let inputs = PassInputs::default();
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&step_line(1, &mut inst, &bound, &inputs)); // out full: this
    rep = rep.line(&take_line(&mut inst, &bound, 1)); //             one commits (cap-1 ring)
    rep = rep.line(&step_line(2, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1));
    rep = rep.line(&take_line(&mut inst, &bound, 1)); // WouldBlock
    check("counter_pingpong", rep);
}

// ── negative cases (verdict-only) ──────────────────────────────────────────

fn neg_report(name: &str, c: TraceContainer, profile: ModelProfile) {
    let verdict = bind(c.clone(), profile);
    let rep = Report::new(name, &c).verdict(&verdict);
    assert!(verdict.is_err(), "{name} must fail validation");
    check(name, rep);
}

fn onechan(host_role: HostRole) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(4),
        dtype: ChanDType::Concrete(DType::F32),
        capacity: 1,
        host_role,
        seeded: true,
    }
}

#[test]
fn golden_neg_spsc_second_producer() {
    // Host writes chan 0; the epilogue also puts → SPSC bind error.
    let c = TraceContainer {
        names: vec![],
        channels: vec![{
            let mut ch = onechan(HostRole::Writer);
            ch.seeded = false;
            ch
        }],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Const(Literal::F32(0.0)),
                Op::Broadcast { value: 0, shape: Shape::vector(4) },
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
    };
    neg_report("neg_spsc_second_producer", c, ModelProfile::dummy());
}

#[test]
fn golden_neg_sink_at_epilogue() {
    let c = TraceContainer {
        names: vec!["lora".into()],
        channels: vec![onechan(HostRole::None)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::ChanRead(0), Op::SinkCall { name: 0, args: vec![0] }],
        }],
    };
    neg_report("neg_sink_at_epilogue", c, ModelProfile::dummy());
}

#[test]
fn golden_neg_t10_nonreplayable() {
    let c = TraceContainer {
        names: vec!["gpu_load".into()],
        channels: vec![onechan(HostRole::None)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::KernelCall { name: 0, args: vec![], shape: Shape::vector(1), dtype: DType::F32 },
                Op::ChanTake(0),
                Op::Broadcast { value: 0, shape: Shape::vector(4) },
                Op::Add(1, 2),
                Op::ChanPut { chan: 0, value: 3 },
            ],
        }],
    };
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(pie_sampling_ir::ptir::registry::KernelInfo {
        name: "gpu_load".into(),
        sink_scope: None,
        replayable: false,
    });
    neg_report("neg_t10_nonreplayable", c, profile);
}

#[test]
fn golden_neg_intrinsic_wrong_stage() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![onechan(HostRole::None)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(1, 32),
                    dtype: DType::F32,
                },
                Op::ChanTake(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
    };
    neg_report("neg_intrinsic_wrong_stage", c, ModelProfile::dummy());
}

#[test]
fn golden_neg_model_gated_missing() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![onechan(HostRole::Reader)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::MtpLogits,
                    shape: Shape::matrix(4, 32),
                    dtype: DType::F32,
                },
                Op::ReduceSum(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
    };
    let mut profile = ModelProfile::dummy();
    profile.has_mtp_logits = false;
    // note: put shape [4] vs chan [4] — fine; the gate fires first anyway.
    neg_report("neg_model_gated_missing", c, profile);
}

#[test]
fn golden_neg_body_type_error() {
    // and() on numerics — a body dtype error with a stable op index.
    let c = TraceContainer {
        names: vec![],
        channels: vec![onechan(HostRole::None)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::And(0, 0),
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
    };
    neg_report("neg_body_type_error", c, ModelProfile::dummy());
}

#[test]
fn golden_matrix_mask_apply_packed() {
    // Per-row packed-mask semantics (pinned after the §6.1 matrix gap): ONE
    // packed mask broadcasts across rows, bit index = column (j % n) — never
    // the flat element index. Mask 0b00101000 allows columns 3 and 5 only.
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(2, 8),
        dtype: DType::F32,
    });
    let m = b.p(Op::Const(Literal::U32(0b0010_1000)));
    let m1 = b.p(Op::Reshape { value: m, shape: Shape::vector(1) });
    let masked = b.p(Op::MaskApply { logits: lg, mask: m1 });
    let t = b.p(Op::ReduceArgmax(masked)); // [2] i32, per row
    b.p(Op::ChanPut { chan: 0, value: t });
    let c = TraceContainer {
        names: vec![],
        channels: vec![ChannelDecl {
            shape: Shape::vector(2),
            dtype: ChanDType::Concrete(DType::I32),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: vec![],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = 8;
    let bound = bind(c.clone(), profile).unwrap();
    let mut rep = Report::new("matrix_mask_apply_packed", &c).verdict(&Ok(bound.clone()));
    let mut inst = Instance::new(&bound, &[]).unwrap();
    // Row 0's raw argmax (col 2) is MASKED -> falls to col 5 (2.0 > col 3's
    // 1.0). Row 1's raw argmax (col 7) is masked -> falls to col 3.
    let logits = vec![
        0., 0., 9., 1., 0., 2., 0., 0., // row 0 -> 5
        0., 0., 0., 4., 0., 3., 0., 9., // row 1 -> 3
    ];
    let inputs = PassInputs { logits: Some(f32s(&logits)), ..Default::default() };
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 0));
    check("matrix_mask_apply_packed", rep);
}
