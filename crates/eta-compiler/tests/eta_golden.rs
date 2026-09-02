//! Golden vectors: canonical container bytes, identity hash, validator
//! verdict and tier-0 reference results, checked into `golden/*.txt` — the
//! conformance suite every backend diffs against.
//!
//! Regenerate (bless) with:
//! `PTIR_REGEN=1 cargo test -p eta-compiler --test eta_golden`

use std::fmt::Write as _;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer, encode,
};
use eta_ir::container_hash;
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::{BoundTrace, bind};

#[path = "common/traces.rs"]
mod traces;

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
    fn verdict(mut self, r: &Result<BoundTrace, eta_ir::validate::ValidateError>) -> Report {
        match r {
            Ok(b) => {
                writeln!(self.0, "verdict: OK").unwrap();
                // Per-value (shape, dtype), readiness, and channel classes:
                // what a backend is handed instead of re-inferring.
                for stage in eta_compiler::plan::compile_bound(b) {
                    let metrics = stage.metrics();
                    writeln!(
                        self.0,
                        "plan: stage={} signature=0x{:016x} source_ops={} normalized_ops={} singleton_regions={} fused_regions={} library_regions={} static_scratch_bytes={} direct_sink_bytes={}",
                        stage.normalized.stage.name(),
                        stage.signature.hash,
                        metrics.source_ops,
                        metrics.normalized_ops,
                        metrics.singleton_regions,
                        metrics.fused_regions,
                        metrics.library_regions,
                        metrics.static_scratch_bytes,
                        metrics.direct_channel_sink_bytes,
                    )
                    .unwrap();
                }
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
}

/// Compare (or bless) one case's report against its golden file.
fn check(name: &str, report: Report) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
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

fn neg_report(name: &str, c: TraceContainer, profile: ModelProfile) {
    let verdict = bind(c.clone(), profile);
    let rep = Report::new(name, &c).verdict(&verdict);
    assert!(verdict.is_err(), "{name} must fail validation");
    check(name, rep);
}

fn onechan(host_role: HostRole) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(4),
        dtype: ChanDType::Concrete(Dtype::F32),
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
                Op::Broadcast {
                    value: 0,
                    shape: Shape::vector(4),
                },
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
            ops: vec![
                Op::ChanRead(0),
                Op::SinkCall {
                    name: 0,
                    args: vec![0],
                },
            ],
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
                Op::KernelCall {
                    name: 0,
                    args: vec![],
                    shape: Shape::vector(1),
                    dtype: Dtype::F32,
                },
                Op::ChanTake(0),
                Op::Broadcast {
                    value: 0,
                    shape: Shape::vector(4),
                },
                Op::Add(1, 2),
                Op::ChanPut { chan: 0, value: 3 },
            ],
        }],
        externs: Vec::new(),
    };
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(eta_ir::registry::KernelInfo {
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
                    dtype: Dtype::F32,
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
                    dtype: Dtype::F32,
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

// One MCTS iteration composing quest, beam, grammar, speculative and
// contrastive techniques through the tier-0 interpreter, from existing ops.

// v1.1 extern channels: a real second-model (amateur) instance exports its
// logits through an extern channel; the expert instance imports them for
// the contrastive pick, exercising cross-instance SPSC, back-pressure and
// the cross-pipeline readiness miss.

#[test]
fn extern_v2_round_trip_and_v1_hashes_stable() {
    use eta_ir::container::{ExternDecl, ExternDir, decode, encode};
    // A v1 container (no externs) encodes version 1 — byte layout untouched.
    let c1 = TraceContainer {
        names: vec![],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(Dtype::U32),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::Const(eta_ir::types::Literal::U32(1)),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
            ],
        }],
        externs: vec![],
    };
    let b1 = encode(&c1);
    assert_eq!(
        u16::from_le_bytes([b1[4], b1[5]]),
        1,
        "no externs => wire v1"
    );
    assert_eq!(decode(&b1).unwrap(), c1);
    // With an extern: version 2, round-trips, and the hash differs (a
    // different trace IS a different identity).
    let mut c2 = c1.clone();
    c2.names = vec!["x".to_string()];
    c2.channels.push(ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(Dtype::U32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: false,
    });
    c2.externs = vec![ExternDecl {
        name: 0,
        dir: ExternDir::Import,
        chan: 1,
    }];
    let b2 = encode(&c2);
    assert_eq!(u16::from_le_bytes([b2[4], b2[5]]), 2, "externs => wire v2");
    assert_eq!(decode(&b2).unwrap(), c2);
    assert_ne!(container_hash(&b1), container_hash(&b2));
}

#[test]
fn extern_direction_violations_rejected() {
    use eta_ir::container::{ExternDecl, ExternDir};
    use eta_ir::validate::ValidateError;
    // A stage PUT on an IMPORT channel = second producer across the pair.
    let mk = |dir: ExternDir, ops: Vec<Op>| TraceContainer {
        names: vec!["x".to_string()],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: false,
            },
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: vec![ExternDecl {
            name: 0,
            dir,
            chan: 0,
        }],
    };
    let put_on_import = mk(
        ExternDir::Import,
        vec![
            Op::Const(eta_ir::types::Literal::F32(1.0)),
            Op::Broadcast {
                value: 0,
                shape: Shape::vector(1),
            },
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
