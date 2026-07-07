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
    step_line_k(n, inst, b, inputs, &mut NoKernels, fed)
}

/// step_line with an explicit kernel host (Quest's `envelope_dot` etc.).
fn step_line_k(
    n: usize,
    inst: &mut Instance,
    b: &BoundTrace,
    inputs: &PassInputs,
    host: &mut dyn pie_sampling_ir::ptir::interp::KernelHost,
    fed: String,
) -> String {
    match inst.step(b, inputs, host) {
        Ok(r) => {
            let mut out = format!(
                "{fed}\nstep {n}: committed={} missed={:?} sinks={}",
                r.committed,
                r.missed.map(|(c, p)| (c, p.tag())),
                r.sinks.len()
            );
            for sk in &r.sinks {
                out.push_str(&format!(
                    "\nsink {}: stage={:?} layer={} args={:?}",
                    sk.name, sk.stage, sk.layer, sk.args
                ));
            }
            out
        }
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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
    externs: Vec::new(),
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

#[test]
fn golden_matrix_select_mask() {
    // The §6.1 per-POSITION select-mask shape at k=4 (the cuda_mtpverify OOB
    // repro, pinned): dselect(allow[k,v], logits[k,v], broadcast(-inf ->
    // [k,v])) -> per-row argmax. EVERY matrix operand materializes k FULL
    // rows (k*v, row-major). Non-degenerate: each row's raw argmax is masked
    // out, so undersized rows / bf16-vs-f32 indexing errors cannot pass.
    // Cross-backend gate: charlie's CUDA JIT + mac-master's Metal both match
    // these step lines bit-exact.
    let (k, v) = (4u32, 8u32);
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(k, v),
        dtype: DType::F32,
    });
    let allow = b.p(Op::ChanTake(0)); // bool [k,v] per-position grammar mask
    let ninf = b.p(Op::Const(Literal::F32(f32::NEG_INFINITY)));
    let nb = b.p(Op::Broadcast { value: ninf, shape: Shape::matrix(k, v) });
    let masked = b.p(Op::Select { cond: allow, a: lg, b: nb });
    let t = b.p(Op::ReduceArgmax(masked)); // [k] i32 per row
    b.p(Op::ChanPut { chan: 1, value: t });
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            ChannelDecl {
                shape: Shape::matrix(k, v),
                dtype: ChanDType::Concrete(DType::Bool),
                capacity: 1,
                host_role: HostRole::Writer,
                seeded: false,
            },
            ChannelDecl {
                shape: Shape::vector(k),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: vec![],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
    externs: Vec::new(),
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = v;
    let bound = bind(c.clone(), profile).unwrap();
    let mut rep = Report::new("matrix_select_mask", &c).verdict(&Ok(bound.clone()));
    let mut inst = Instance::new(&bound, &[]).unwrap();
    // logits[r]: raw max 9.0 at col r (MASKED); allowed col (r+2)%v = 1.0.
    let mut logits = vec![0.0f32; (k * v) as usize];
    let mut allow_bits = vec![false; (k * v) as usize];
    for r in 0..k as usize {
        logits[r * v as usize + r] = 9.0;
        let a = (r + 2) % v as usize;
        logits[r * v as usize + a] = 1.0;
        allow_bits[r * v as usize + a] = true;
    }
    let inputs = PassInputs { logits: Some(f32s(&logits)), ..Default::default() };
    rep = rep.line(&{
        let l = format!("host_put chan=0 = {:?}", Value::Bool(allow_bits.clone()));
        inst.host_put(&bound, 0, Value::Bool(allow_bits)).unwrap();
        l
    });
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 1)); // expect I32([2,3,4,5])
    check("matrix_select_mask", rep);
}

#[test]
fn golden_mtp_verify_tail() {
    // The full MTP match-verify DAG at K=3, V=8 (overview §6.1 steps 1-6),
    // hand-checkable — the cross-backend anchor for the accept-prefix logic
    // (eq → cumprod → select with the -1 sentinel) on top of the
    // matrix_select_mask shape. Non-degenerate: WITHOUT the mask row 2 would
    // match the draft (raw argmax = 6 = d3) — the mask forbids 6 and forces
    // picked[2]=2, so the run stops at n_acc=2. Hand check:
    //   prev_drafts=[3,5,6]; picked=[3,5,2,4] -> hit=[T,T,F] -> n_acc=2
    //   commit lanes 0..=2, lane 3 = -1  =>  out = I32([3, 5, 2, -1])
    //   drafts = argmax(mtp_logits) = [1, 4, 0] (also the new prev_drafts)
    let (k, v) = (3u32, 8u32); // K drafts; K+1 = 4 verify rows
    let kp1 = k + 1;
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(kp1, v),
        dtype: DType::F32,
    });
    let mtp = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::MtpLogits,
        shape: Shape::matrix(k, v),
        dtype: DType::F32,
    });
    let m = b.p(Op::ChanTake(1)); // allow [K+1, v] bool (host-fed)
    let ninf = b.p(Op::Const(Literal::F32(f32::NEG_INFINITY)));
    let nb = b.p(Op::Broadcast { value: ninf, shape: Shape::matrix(kp1, v) });
    let masked = b.p(Op::Select { cond: m, a: lg, b: nb });
    let picked = b.p(Op::ReduceArgmax(masked)); // [K+1] i32
    let d = b.p(Op::ChanTake(0)); // prev_drafts [K] i32
    let vl = b.p(Op::Iota { len: k }); // VERIFY_LANES = [0,1,2]
    let head = b.p(Op::Gather { src: picked, idx: vl }); // picked[0..K]
    let hit = b.p(Op::Eq(head, d)); // [K] bool
    let one = b.p(Op::Const(Literal::F32(1.0)));
    let zero = b.p(Op::Const(Literal::F32(0.0)));
    let ones = b.p(Op::Broadcast { value: one, shape: Shape::vector(k) });
    let zeros = b.p(Op::Broadcast { value: zero, shape: Shape::vector(k) });
    let runf = b.p(Op::Select { cond: hit, a: ones, b: zeros }); // [K] {1,0}
    let run = b.p(Op::CumProd(runf));
    let naccf = b.p(Op::ReduceSum(run)); // scalar f32 = leading-hit count
    let nacc = b.p(Op::Cast { value: naccf, dtype: DType::U32 });
    let naccb = b.p(Op::Broadcast { value: nacc, shape: Shape::vector(kp1) });
    let lanes = b.p(Op::Iota { len: kp1 });
    let keep = b.p(Op::Ge(naccb, lanes)); // keep[i] = i <= n_acc
    let neg1 = b.p(Op::Const(Literal::I32(-1)));
    let commit = b.p(Op::Select { cond: keep, a: picked, b: neg1 }); // [K+1]
    b.p(Op::ChanPut { chan: 2, value: commit });
    let drf = b.p(Op::ReduceArgmax(mtp)); // [K] fresh drafts (greedy proposals)
    b.p(Op::ChanPut { chan: 0, value: drf }); // prev_drafts ping-pong
    b.p(Op::ChanPut { chan: 3, value: drf }); // publish for the host grammar walk
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(k),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            }, // 0 prev_drafts
            ChannelDecl {
                shape: Shape::matrix(kp1, v),
                dtype: ChanDType::Concrete(DType::Bool),
                capacity: 1,
                host_role: HostRole::Writer,
                seeded: false,
            }, // 1 allow
            ChannelDecl {
                shape: Shape::vector(kp1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            }, // 2 out (committed window, -1 sentinel)
            ChannelDecl {
                shape: Shape::vector(k),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            }, // 3 draft_out
        ],
        ports: vec![],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
    externs: Vec::new(),
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = v; // has_mtp_logits = true in dummy()
    let bound = bind(c.clone(), profile).unwrap();
    let mut rep = Report::new("mtp_verify_tail", &c).verdict(&Ok(bound.clone()));
    let seeds = [(0u32, i32s(&[3, 5, 6]))];
    rep = seed_lines(rep, &seeds);
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    // logits rows (raw argmax): r0->3 (=d1 hit), r1->5 (=d2 hit),
    // r2->6 (WOULD match d3 — but the mask forbids 6, allows {2} -> picked 2,
    // miss), r3->4 (bonus row; unused since n_acc=2 -> bonus is picked[2]).
    let mut logits = vec![0.0f32; (kp1 * v) as usize];
    logits[0 * 8 + 3] = 9.0;
    logits[8 + 5] = 9.0;
    logits[2 * 8 + 6] = 9.0; // masked out
    logits[2 * 8 + 2] = 1.0; // the allowed survivor
    logits[3 * 8 + 4] = 9.0;
    // mask: rows 0,1,3 allow-all; row 2 allows only col 2.
    let mut allow = vec![true; (kp1 * v) as usize];
    for col in 0..v as usize {
        allow[2 * 8 + col] = col == 2;
    }
    // mtp rows argmax -> [1, 4, 0]
    let mut mtpv = vec![0.0f32; (k * v) as usize];
    mtpv[1] = 7.0;
    mtpv[8 + 4] = 7.0;
    mtpv[2 * 8] = 7.0;
    let inputs = PassInputs {
        logits: Some(f32s(&logits)),
        mtp_logits: Some(f32s(&mtpv)),
        ..Default::default()
    };
    rep = rep.line(&put_line(&mut inst, &bound, 1, Value::Bool(allow)));
    rep = rep.line(&step_line(0, &mut inst, &bound, &inputs));
    rep = rep.line(&take_line(&mut inst, &bound, 2)); // I32([3, 5, 2, -1])
    rep = rep.line(&take_line(&mut inst, &bound, 3)); // I32([1, 4, 0])
    check("mtp_verify_tail", rep);
}

// ═══════════════════════════════════════════════════════════════════════════
// Capstone: Pentathlon+1 — one MCTS iteration composing all SIX techniques
// (quest + mcts + beam + constrained + speculative + contrastive) through the
// tier-0 oracle. docs/ptir/pentathlon-design.md §2; zero new ops.
// ═══════════════════════════════════════════════════════════════════════════

const PV: u32 = 8; // vocab
const PB: u32 = 2; // beam width per expansion
const PK: u32 = 3; // MTP draft width (K+1 = 4 verify rows)
const PPAGES: u32 = 4; // P_MAX (quest page-mask width)

/// Deterministic Quest kernel host: `envelope_dot(query)` -> [P_MAX] scores,
/// varying per call so layer-0/layer-1 selections differ visibly.
struct QuestKernels {
    calls: u32,
}
impl pie_sampling_ir::ptir::interp::KernelHost for QuestKernels {
    fn kernel(
        &mut self,
        name: &str,
        args: &[Value],
        result: pie_sampling_ir::types::ValueType,
    ) -> Result<Value, String> {
        if name != "envelope_dot" {
            return Err(format!("no such kernel: {name}"));
        }
        // score[p] = |q[0]| + p + calls  — deterministic, call-varying.
        let q0 = match &args[0] {
            Value::F32(q) => q[0].abs(),
            _ => 0.0,
        };
        let n = result.shape.numel() as usize;
        let c = self.calls as f32;
        self.calls += 1;
        Ok(Value::F32((0..n).map(|p| q0 + p as f32 + c).collect()))
    }
}

fn pentathlon_profile() -> ModelProfile {
    let mut p = ModelProfile::dummy(); // num_layers = 2, mtp+value gated ON
    p.vocab = PV;
    p.kernels.push(pie_sampling_ir::ptir::registry::KernelInfo {
        name: "envelope_dot".into(),
        sink_scope: None,
        replayable: true,
    });
    p
}

/// The Quest tap (design §2.1): per layer, envelope scores -> top-`budget`
/// page mask -> the attn_page_mask sink. `budget_ch` is a [1] u32 channel.
fn quest_tap(budget_ch: u32) -> StageProgram {
    let mut b = B::new();
    let q = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Query,
        shape: Shape::vector(PPAGES),
        dtype: DType::F32,
    });
    let scores = b.p(Op::KernelCall {
        name: 0, // "envelope_dot"
        args: vec![q],
        shape: Shape::vector(PPAGES),
        dtype: DType::F32,
    });
    let bud = b.p(Op::ChanRead(budget_ch)); // [1] u32 (peek — a knob)
    let buds = b.p(Op::Reshape { value: bud, shape: Shape::SCALAR });
    let pm = b.p(Op::PivotThreshold {
        input: scores,
        predicate: pie_sampling_ir::types::Predicate::RankLe(buds),
    });
    b.p(Op::SinkCall { name: 1, args: vec![pm] }); // "attn_page_mask"
    StageProgram { stage: Stage::OnAttnProj, ops: b.ops }
}

/// Contrastive pick (design §2.3, ORDER PINNED BY THIS GOLDEN): grammar folds
/// into the expert distribution BEFORE the plausibility max — i.e. the
/// α-filter runs within the CONSTRAINED support. Computing plausibility
/// against the unmasked max annihilates every legal token whenever the
/// grammar masks the expert's peak (the composition-order bug this golden
/// caught in the first cut). Returns the scored [rows, V] value id.
fn contrastive_score(b: &mut B, lse_in: u32, am_take: u32, gmask: u32, rows: u32) -> u32 {
    let sh = Shape::matrix(rows, PV);
    let lse = expand::log_softmax(&mut b.ops, lse_in, sh);
    let lsa = expand::log_softmax(&mut b.ops, am_take, sh);
    let lam = b.p(Op::Const(Literal::F32(0.5))); // λ
    let pen = b.p(Op::Mul(lsa, lam));
    let cd = b.p(Op::Sub(lse, pen));
    let ninf = b.p(Op::Const(Literal::F32(f32::NEG_INFINITY)));
    let lse_g = b.p(Op::Select { cond: gmask, a: lse, b: ninf }); // grammar FIRST
    let mx = b.p(Op::ReduceMax(lse_g)); // [rows] max over the LEGAL set
    let mx1 = b.p(Op::Reshape { value: mx, shape: Shape::matrix(rows, 1) });
    let mxb = b.p(Op::Broadcast { value: mx1, shape: sh });
    let la = b.p(Op::Const(Literal::F32(-1.0))); // log α
    let thr = b.p(Op::Add(mxb, la));
    let plaus = b.p(Op::Ge(lse_g, thr)); // -inf rows auto-excluded (grammar ∧ α)
    b.p(Op::Select { cond: plaus, a: cd, b: ninf })
}
use pie_sampling_ir::ptir::expand;

/// design §2.1 + §2.3: quest tap + contrastive beam expansion + leaf value.
/// Channels: 0 am [1,V] f32 W · 1 gmask [1,V] bool W · 2 budget [1] u32 seed
/// · 3 prio [B] f32 R · 4 pids [B] u32 R · 5 val [1] f32 R.
fn expand_trace() -> TraceContainer {
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(1, PV),
        dtype: DType::F32,
    });
    let am = b.p(Op::ChanTake(0));
    let gm = b.p(Op::ChanTake(1));
    let scored = contrastive_score(&mut b, lg, am, gm, 1);
    let pr = b.p(Op::TopK { input: scored, k: PB }); // pr, pr+1=ids  [1,B]
    let prv = b.p(Op::Reshape { value: pr, shape: Shape::vector(PB) });
    let idv = b.p(Op::Reshape { value: pr + 1, shape: Shape::vector(PB) });
    b.p(Op::ChanPut { chan: 3, value: prv });
    b.p(Op::ChanPut { chan: 4, value: idv });
    let vh = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::ValueHead,
        shape: Shape::vector(1),
        dtype: DType::F32,
    });
    b.p(Op::ChanPut { chan: 5, value: vh });
    let ch = |shape, dtype, host_role, seeded| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    };
    TraceContainer {
        names: vec!["envelope_dot".into(), "attn_page_mask".into()],
        channels: vec![
            ch(Shape::matrix(1, PV), DType::F32, HostRole::Writer, false), // 0 am
            ch(Shape::matrix(1, PV), DType::Bool, HostRole::Writer, false), // 1 gmask
            ch(Shape::vector(1), DType::U32, HostRole::None, true),        // 2 budget
            ch(Shape::vector(PB), DType::F32, HostRole::Reader, false),    // 3 prio
            ch(Shape::vector(PB), DType::U32, HostRole::Reader, false),    // 4 pids
            ch(Shape::vector(1), DType::F32, HostRole::Reader, false),     // 5 val
        ],
        ports: vec![],
        stages: vec![quest_tap(2), StageProgram { stage: Stage::Epilogue, ops: b.ops }],
        externs: Vec::new(),
    }
}

/// design §2.2 + §2.3: quest tap + contrastive pick over the [K+1] window +
/// the mtp_verify accept tail + value tap.
/// Channels: 0 prev_drafts [K] i32 seed · 1 gmask [K+1,V] bool W ·
/// 2 am [K+1,V] f32 W · 3 out [K+1] i32 R · 4 draft_out [K] i32 R ·
/// 5 val [K+1] f32 R · 6 budget [1] u32 seed.
fn rollout_trace() -> TraceContainer {
    let kp1 = PK + 1;
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(kp1, PV),
        dtype: DType::F32,
    });
    let mtp = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::MtpLogits,
        shape: Shape::matrix(PK, PV),
        dtype: DType::F32,
    });
    let am = b.p(Op::ChanTake(2));
    let gm = b.p(Op::ChanTake(1));
    let scored = contrastive_score(&mut b, lg, am, gm, kp1);
    let picked = b.p(Op::ReduceArgmax(scored)); // [K+1] i32 — contrastive pick
    // verify tail (== golden_mtp_verify_tail)
    let d = b.p(Op::ChanTake(0));
    let vl = b.p(Op::Iota { len: PK });
    let head = b.p(Op::Gather { src: picked, idx: vl });
    let hit = b.p(Op::Eq(head, d));
    let one = b.p(Op::Const(Literal::F32(1.0)));
    let zero = b.p(Op::Const(Literal::F32(0.0)));
    let ones = b.p(Op::Broadcast { value: one, shape: Shape::vector(PK) });
    let zeros = b.p(Op::Broadcast { value: zero, shape: Shape::vector(PK) });
    let runf = b.p(Op::Select { cond: hit, a: ones, b: zeros });
    let run = b.p(Op::CumProd(runf));
    let naccf = b.p(Op::ReduceSum(run));
    let nacc = b.p(Op::Cast { value: naccf, dtype: DType::U32 });
    let naccb = b.p(Op::Broadcast { value: nacc, shape: Shape::vector(kp1) });
    let lanes = b.p(Op::Iota { len: kp1 });
    let keep = b.p(Op::Ge(naccb, lanes));
    let neg1 = b.p(Op::Const(Literal::I32(-1)));
    let commit = b.p(Op::Select { cond: keep, a: picked, b: neg1 });
    b.p(Op::ChanPut { chan: 3, value: commit });
    let drf = b.p(Op::ReduceArgmax(mtp));
    b.p(Op::ChanPut { chan: 0, value: drf });
    b.p(Op::ChanPut { chan: 4, value: drf });
    let vh = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::ValueHead,
        shape: Shape::vector(kp1),
        dtype: DType::F32,
    });
    b.p(Op::ChanPut { chan: 5, value: vh });
    let ch = |shape, dtype, host_role, seeded| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    };
    TraceContainer {
        names: vec!["envelope_dot".into(), "attn_page_mask".into()],
        channels: vec![
            ch(Shape::vector(PK), DType::I32, HostRole::None, true), // 0 prev_drafts
            ch(Shape::matrix(kp1, PV), DType::Bool, HostRole::Writer, false), // 1 gmask
            ch(Shape::matrix(kp1, PV), DType::F32, HostRole::Writer, false), // 2 am
            ch(Shape::vector(kp1), DType::I32, HostRole::Reader, false), // 3 out
            ch(Shape::vector(PK), DType::I32, HostRole::Reader, false), // 4 draft_out
            ch(Shape::vector(kp1), DType::F32, HostRole::Reader, false), // 5 val
            ch(Shape::vector(1), DType::U32, HostRole::None, true),   // 6 budget
        ],
        ports: vec![],
        stages: vec![quest_tap(6), StageProgram { stage: Stage::Epilogue, ops: b.ops }],
        externs: Vec::new(),
    }
}

#[test]
fn golden_pentathlon_iter() {
    // ONE MCTS iteration at R=2 (host process; the tree/PUCT is ordinary host
    // code — here, the scripted select/expand/backprop below): two EXPAND
    // passes (leaves A, B) then one ROLLOUT step per chosen child. All six
    // techniques fire: quest (per-layer envelope_dot -> page-mask sink),
    // beam (top_k B=2), grammar (per-position masks), speculative (MTP
    // verify tail), contrastive (expert − λ·amateur + plausibility), MCTS
    // (this script + value_head taps).
    let profile = pentathlon_profile();
    let exp_c = expand_trace();
    let rol_c = rollout_trace();
    let exp_b = bind(exp_c.clone(), profile.clone()).expect("expand binds");
    let rol_b = bind(rol_c.clone(), profile).expect("rollout binds");

    let mut rep = Report::new("pentathlon_expand", &exp_c).verdict(&Ok(exp_b.clone()));
    rep.0.push_str(&Report::new("pentathlon_rollout", &rol_c).verdict(&Ok(rol_b.clone())).0);
    let mut kh = QuestKernels { calls: 0 };

    // ── phase 1: EXPAND leaves A and B ──────────────────────────────────
    // Leaf A: expert peaks on {2, 5}; amateur ALSO loves 2 (contrastive
    // demotes it) -> beam picks 5 first, 2 second. Leaf B: grammar only
    // allows {1, 3}; expert peaks 6 (masked) -> beam = {3, 1}.
    for (label, elogits, amlogits, allow, q, vh) in [
        (
            "A",
            vec![0., 0., 6.0, 0., 0., 5.9, 0., 0.],
            vec![0., 0., 9.0, 0., 0., 0., 0., 0.],
            vec![true; PV as usize],
            2.0f32,
            0.25f32,
        ),
        (
            "B",
            vec![0., 5.0, 0., 5.5, 0., 0., 9.0, 0.],
            vec![0.0; PV as usize],
            {
                let mut a = vec![false; PV as usize];
                a[1] = true;
                a[3] = true;
                a
            },
            -1.0f32,
            0.75f32,
        ),
    ] {
        rep = rep.line(&format!("== expand leaf {label}"));
        let seeds = [(2u32, u32s(&[2]))]; // quest budget = top-2 pages
        rep = seed_lines(rep, &seeds);
        let mut inst = Instance::new(&exp_b, &seeds).unwrap();
        rep = rep.line(&put_line(&mut inst, &exp_b, 0, f32s(&amlogits)));
        rep = rep.line(&put_line(&mut inst, &exp_b, 1, Value::Bool(allow)));
        let inputs = PassInputs {
            logits: Some(f32s(&elogits)),
            value_head: Some(f32s(&[vh])),
            query: vec![f32s(&[q, 0., 0., 0.]), f32s(&[q + 1.0, 0., 0., 0.])],
            ..Default::default()
        };
        let fed = format!("inputs {label}: {inputs:?}");
        rep = rep.line(&step_line_k(0, &mut inst, &exp_b, &inputs, &mut kh, fed));
        rep = rep.line(&take_line(&mut inst, &exp_b, 3)); // prio
        rep = rep.line(&take_line(&mut inst, &exp_b, 4)); // pids: A=[5,2] B=[3,1]
        rep = rep.line(&take_line(&mut inst, &exp_b, 5)); // leaf value
    }

    // ── phase 2: ROLLOUT one MTP step per chosen child ──────────────────
    // Child A5: drafts [5,2,7]; contrastive+grammar picks rows -> [5,2,4,..]
    // row2's raw pick (7) is masked -> forced 4 -> miss at i=2 -> n_acc=2,
    // out=[5,2,4,-1]. Child B3: drafts [1,1,1]; row0 picks 1 (hit), row1
    // amateur demotes 1 -> picks 3 -> miss -> n_acc=1, out=[1,3,-1,-1].
    for (label, drafts, elogits_rows, am_rows, allow_rows, mtp_rows, vh) in [
        (
            "A5",
            [5i32, 2, 7],
            [[(5usize, 9.0f32), (0, 0.0)], [(2, 9.0), (0, 0.0)], [(7, 9.0), (0, 0.0)], [(0, 9.0), (0, 0.0)]],
            [(0usize, 0.0f32); 4],
            // row2 forbids 7, allows 4; other rows allow-all
            Some((2usize, 4usize)),
            [(6usize, 8.0f32), (0, 8.0), (2, 8.0)],
            [0.9f32, 0.8, 0.7, 0.6],
        ),
        (
            "B3",
            [1i32, 1, 1],
            // row1: TWO plausible expert tokens (1 at 9.0, 3 at 8.5, inside
            // the α-window); the amateur loves 1 -> cd reranks 3 above 1.
            [[(1usize, 9.0f32), (0, 0.0)], [(1, 9.0), (3, 8.5)], [(1, 9.0), (0, 0.0)], [(1, 9.0), (0, 0.0)]],
            [(0, 0.0), (1, 18.0), (0, 0.0), (0, 0.0)],
            None,
            [(5usize, 8.0f32), (5, 8.0), (5, 8.0)],
            [0.1, 0.2, 0.3, 0.4],
        ),
    ] {
        let kp1 = (PK + 1) as usize;
        rep = rep.line(&format!("== rollout child {label}"));
        let seeds = [(0u32, i32s(&drafts)), (6u32, u32s(&[2]))];
        rep = seed_lines(rep, &seeds);
        let mut inst = Instance::new(&rol_b, &seeds).unwrap();
        let mut allow = vec![true; kp1 * PV as usize];
        if let Some((row, only)) = allow_rows {
            for c in 0..PV as usize {
                allow[row * PV as usize + c] = c == only;
            }
        }
        let mut am = vec![0.0f32; kp1 * PV as usize];
        for (r, (c, x)) in am_rows.iter().enumerate() {
            am[r * PV as usize + c] = *x;
        }
        // row3 of elogits_rows fills the bonus row too (index 3).
        let mut el = vec![0.0f32; kp1 * PV as usize];
        for (r, row) in elogits_rows.iter().enumerate() {
            for (c, x) in row {
                el[r * PV as usize + c] = *x;
            }
        }
        let mut mtpl = vec![0.0f32; PK as usize * PV as usize];
        for (r, (c, x)) in mtp_rows.iter().enumerate() {
            mtpl[r * PV as usize + c] = *x;
        }
        rep = rep.line(&put_line(&mut inst, &rol_b, 1, Value::Bool(allow)));
        rep = rep.line(&put_line(&mut inst, &rol_b, 2, f32s(&am)));
        let inputs = PassInputs {
            logits: Some(f32s(&el)),
            mtp_logits: Some(f32s(&mtpl)),
            value_head: Some(f32s(&vh)),
            query: vec![f32s(&[1., 0., 0., 0.]), f32s(&[2., 0., 0., 0.])],
            ..Default::default()
        };
        let fed = format!("inputs {label}: {inputs:?}");
        rep = rep.line(&step_line_k(0, &mut inst, &rol_b, &inputs, &mut kh, fed));
        rep = rep.line(&take_line(&mut inst, &rol_b, 3)); // out (committed + -1)
        rep = rep.line(&take_line(&mut inst, &rol_b, 4)); // next drafts
        rep = rep.line(&take_line(&mut inst, &rol_b, 5)); // values -> backprop
    }
    check("pentathlon_iter", rep);
}

#[test]
fn golden_dfa_ingraph() {
    // The G1 demonstrator (design §5): for a REGULAR grammar the mask walk
    // moves IN-GRAPH with existing ops — allow/next tables live in seeded
    // device-private channels (read-only), the DFA state is a [1] u32
    // ping-pong channel, and the per-step mask row is `gather(allow, state)`.
    // ZERO host-fed channels: the grammar edge (the §3 "one host-coupled
    // edge") is deleted; the host only harvests `out`.
    //
    // DFA (S=3, V=8): state0 allows {1,2}; state1 allows {3}; state2 allows
    // {0}. next: 0-[1|2]->1, 1-[3]->2, 2-[0]->2. Logits favor a FORBIDDEN
    // token every step, so the walk visibly forces: expect out = 2, 3, 0 and
    // the state path 0->1->2->2.
    let (sn, v) = (3u32, 8u32);
    let mut b = B::new();
    let lg = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(1, v),
        dtype: DType::F32,
    });
    let allow = b.p(Op::ChanRead(0)); // [S, V] bool (read-only table)
    let next = b.p(Op::ChanRead(1)); // [S*V] u32 (flat next-state table)
    let st = b.p(Op::ChanTake(2)); // [1] u32
    let row = b.p(Op::Gather { src: allow, idx: st }); // [1, V] bool — THE MASK
    let ninf = b.p(Op::Const(Literal::F32(f32::NEG_INFINITY)));
    let masked = b.p(Op::Select { cond: row, a: lg, b: ninf });
    let picked = b.p(Op::ReduceArgmax(masked)); // [1] i32
    b.p(Op::ChanPut { chan: 3, value: picked });
    // state' = next[state*V + picked]  — the in-graph walk
    let vc = b.p(Op::Const(Literal::U32(v)));
    let base = b.p(Op::Mul(st, vc));
    let pu = b.p(Op::Cast { value: picked, dtype: DType::U32 });
    let idx = b.p(Op::Add(base, pu));
    let ns = b.p(Op::Gather { src: next, idx });
    b.p(Op::ChanPut { chan: 2, value: ns });
    let ch = |shape, dtype, host_role, seeded| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    };
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            ch(Shape::matrix(sn, v), DType::Bool, HostRole::None, true), // 0 allow (needs drain-refill? read-only: read never consumes ✓)
            ch(Shape::vector(sn * v), DType::U32, HostRole::None, true), // 1 next
            ch(Shape::vector(1), DType::U32, HostRole::None, true),      // 2 state
            ch(Shape::vector(1), DType::I32, HostRole::Reader, false),   // 3 out
        ],
        ports: vec![],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
    externs: Vec::new(),
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = v;
    let bound = bind(c.clone(), profile).unwrap();
    let mut rep = Report::new("dfa_ingraph", &c).verdict(&Ok(bound.clone()));

    // Tables as instance seeds (per-instance data; a trace-level large-const
    // form is a container-v1.1 nicety — seeded read-only channels work today).
    let mut allow_t = vec![false; (sn * v) as usize];
    allow_t[1] = true; // s0: {1, 2}
    allow_t[2] = true;
    allow_t[(v + 3) as usize] = true; // s1: {3}
    allow_t[(2 * v) as usize] = true; // s2: {0}
    let mut next_t = vec![0u32; (sn * v) as usize];
    next_t[1] = 1; // 0 -[1]-> 1
    next_t[2] = 1; // 0 -[2]-> 1
    next_t[(v + 3) as usize] = 2; // 1 -[3]-> 2
    next_t[(2 * v) as usize] = 2; // 2 -[0]-> 2
    let seeds = [
        (0u32, Value::Bool(allow_t)),
        (1u32, u32s(&next_t)),
        (2u32, u32s(&[0])),
    ];
    rep = seed_lines(rep, &seeds);
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    // Every step the raw logits favor a FORBIDDEN token; the in-graph mask
    // must force the legal pick. (Step 0 also has legal 2 > legal 1.)
    for (n, fav, second) in [(0usize, 5usize, 2usize), (1, 6, 3), (2, 7, 0)] {
        let mut l = vec![0.0f32; v as usize];
        l[fav] = 9.0;
        l[second] = 1.0;
        let inputs = PassInputs { logits: Some(f32s(&l)), ..Default::default() };
        rep = rep.line(&step_line(n, &mut inst, &bound, &inputs));
        rep = rep.line(&take_line(&mut inst, &bound, 3)); // 2, 3, 0
    }
    check("dfa_ingraph", rep);
}

// ═══════════════════════════════════════════════════════════════════════════
// v1.1 extern channels — the FAITHFUL multi-model contrastive path (G6):
// a real second-model (amateur) instance EXPORTS its logits through an
// extern channel; the expert instance IMPORTS them for the contrastive pick.
// Cross-instance SPSC + back-pressure + the cross-pipeline readiness miss,
// all through the tier-0 oracle.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn golden_extern_contrastive() {
    use pie_sampling_ir::ptir::container::{ExternDecl, ExternDir};
    use pie_sampling_ir::ptir::interp::ExternChannel;
    let v = 8u32;

    // ── amateur: publish this model's logits into the exported channel ──
    let am_trace = {
        let mut b = B::new();
        let lg = b.p(Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, v),
            dtype: DType::F32,
        });
        b.p(Op::ChanPut { chan: 0, value: lg });
        TraceContainer {
            names: vec!["cd_amateur_logits".to_string()],
            channels: vec![ChannelDecl {
                shape: Shape::matrix(1, v),
                dtype: ChanDType::Concrete(DType::F32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: false,
            }],
            ports: vec![],
            stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
            externs: vec![ExternDecl { name: 0, dir: ExternDir::Export, chan: 0 }],
        }
    };

    // ── expert: contrastive pick over the IMPORTED amateur logits ───────
    let ex_trace = {
        let mut b = B::new();
        let lg = b.p(Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, v),
            dtype: DType::F32,
        });
        let am = b.p(Op::ChanTake(0)); // IMPORT — the amateur's real logits
        let gm = b.p(Op::ChanTake(1)); // grammar mask (host)
        let scored = contrastive_score(&mut b, lg, am, gm, 1);
        let picked = b.p(Op::ReduceArgmax(scored)); // [1] i32
        b.p(Op::ChanPut { chan: 2, value: picked });
        TraceContainer {
            names: vec!["cd_amateur_logits".to_string()],
            channels: vec![
                ChannelDecl {
                    shape: Shape::matrix(1, v),
                    dtype: ChanDType::Concrete(DType::F32),
                    capacity: 1,
                    host_role: HostRole::None,
                    seeded: false,
                }, // 0 am (import)
                ChannelDecl {
                    shape: Shape::matrix(1, v),
                    dtype: ChanDType::Concrete(DType::Bool),
                    capacity: 1,
                    host_role: HostRole::Writer,
                    seeded: false,
                }, // 1 gmask
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(DType::I32),
                    capacity: 1,
                    host_role: HostRole::Reader,
                    seeded: false,
                }, // 2 out
            ],
            ports: vec![],
            stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
            externs: vec![ExternDecl { name: 0, dir: ExternDir::Import, chan: 0 }],
        }
    };

    let mut profile = ModelProfile::dummy();
    profile.vocab = v;
    let am_b = bind(am_trace.clone(), profile.clone()).expect("amateur binds");
    let ex_b = bind(ex_trace.clone(), profile).expect("expert binds");
    let mut rep = Report::new("extern_amateur", &am_trace).verdict(&Ok(am_b.clone()));
    rep.0.push_str(&Report::new("extern_expert", &ex_trace).verdict(&Ok(ex_b.clone())).0);

    // The broker: ONE shared ring, handed to both instances (§1: SPSC
    // constrains endpoints, not clocks).
    let ring = ExternChannel::for_decl(&am_trace.channels[0]);
    let mut am = pie_sampling_ir::ptir::interp::Instance::new_with_externs(
        &am_b,
        &[],
        &[(0, ring.clone())],
    )
    .unwrap();
    let mut ex = pie_sampling_ir::ptir::interp::Instance::new_with_externs(
        &ex_b,
        &[],
        &[(0, ring)],
    )
    .unwrap();

    // Expert logits peak at 2 (9.0) and 5 (8.5), both in the α-window; the
    // AMATEUR (a genuinely different model) loves 2 -> contrastive demotes
    // it -> the expert must pick 5. Amateur alone would pick 2; expert
    // alone would pick 2. Only the REAL two-model composition yields 5.
    let mut exl = vec![0.0f32; v as usize];
    exl[2] = 9.0;
    exl[5] = 8.5;
    let mut aml = vec![0.0f32; v as usize];
    aml[2] = 9.0;
    let ex_in = PassInputs { logits: Some(f32s(&exl)), ..Default::default() };
    let am_in = PassInputs { logits: Some(f32s(&aml)), ..Default::default() };

    // 1. Expert fires FIRST: the extern import is empty — the cross-pipeline
    //    readiness miss (the amateur's clock hasn't produced yet).
    rep = rep.line(&put_line(&mut ex, &ex_b, 1, Value::Bool(vec![true; v as usize])));
    rep = rep.line(&{
        let fed = format!("inputs expert-early: {ex_in:?}");
        step_line_k(0, &mut ex, &ex_b, &ex_in, &mut NoKernels, fed)
    });
    // 2. Amateur fires: publishes its logits into the shared ring.
    rep = rep.line(&{
        let fed = format!("inputs amateur: {am_in:?}");
        step_line_k(1, &mut am, &am_b, &am_in, &mut NoKernels, fed)
    });
    // 3. Amateur fires AGAIN immediately: back-pressure across instances —
    //    the ring (cap 1) is still full, the expert hasn't consumed.
    rep = rep.line(&{
        let fed = format!("inputs amateur-again: {am_in:?}");
        step_line_k(2, &mut am, &am_b, &am_in, &mut NoKernels, fed)
    });
    // 4. Expert resubmits: takes the amateur logits, contrastive-picks 5.
    rep = rep.line(&{
        let fed = format!("inputs expert: {ex_in:?}");
        step_line_k(3, &mut ex, &ex_b, &ex_in, &mut NoKernels, fed)
    });
    rep = rep.line(&take_line(&mut ex, &ex_b, 2)); // I32([5])
    // 5. Amateur's resubmission now commits (ring drained by the expert).
    rep = rep.line(&{
        let fed = format!("inputs amateur-refill: {am_in:?}");
        step_line_k(4, &mut am, &am_b, &am_in, &mut NoKernels, fed)
    });
    check("extern_contrastive", rep);
}

#[test]
fn extern_v2_round_trip_and_v1_hashes_stable() {
    use pie_sampling_ir::ptir::container::{decode, encode, ExternDecl, ExternDir};
    // A v1 container (no externs) encodes version 1 — byte layout untouched.
    let c1 = TraceContainer {
        names: vec![],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(DType::U32),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::Const(pie_sampling_ir::types::Literal::U32(1)),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
            ],
        }],
        externs: vec![],
    };
    let b1 = encode(&c1);
    assert_eq!(u16::from_le_bytes([b1[4], b1[5]]), 1, "no externs => wire v1");
    assert_eq!(decode(&b1).unwrap(), c1);
    // With an extern: version 2, round-trips, and the hash differs (a
    // different trace IS a different identity).
    let mut c2 = c1.clone();
    c2.names = vec!["x".to_string()];
    c2.channels.push(ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(DType::U32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: false,
    });
    c2.externs = vec![ExternDecl { name: 0, dir: ExternDir::Import, chan: 1 }];
    let b2 = encode(&c2);
    assert_eq!(u16::from_le_bytes([b2[4], b2[5]]), 2, "externs => wire v2");
    assert_eq!(decode(&b2).unwrap(), c2);
    assert_ne!(container_hash(&b1), container_hash(&b2));
}

#[test]
fn extern_direction_violations_rejected() {
    use pie_sampling_ir::ptir::container::{ExternDecl, ExternDir};
    use pie_sampling_ir::ptir::validate::ValidateError;
    // A stage PUT on an IMPORT channel = second producer across the pair.
    let mk = |dir: ExternDir, ops: Vec<Op>| TraceContainer {
        names: vec!["x".to_string()],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::F32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: false,
            },
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::F32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: vec![],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops }],
        externs: vec![ExternDecl { name: 0, dir, chan: 0 }],
    };
    let put_on_import = mk(
        ExternDir::Import,
        vec![
            Op::Const(pie_sampling_ir::types::Literal::F32(1.0)),
            Op::Broadcast { value: 0, shape: Shape::vector(1) },
            Op::ChanPut { chan: 0, value: 1 },
        ],
    );
    assert!(matches!(
        bind(put_on_import, ModelProfile::dummy()),
        Err(ValidateError::ExternDirViolation { chan: 0, .. })
    ));
    let read_own_export = mk(
        ExternDir::Export,
        vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
    );
    assert!(matches!(
        bind(read_own_export, ModelProfile::dummy()),
        Err(ValidateError::ExternDirViolation { chan: 0, .. })
    ));
}
