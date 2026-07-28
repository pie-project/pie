// Included by several test binaries via `#[path]`; each uses a different
// subset, so "never used" here means "unused by *this* binary".
#![allow(dead_code)]

//! The shared corpus the Metal MSL emitters are exercised over.
//!
//! The plan-taking emitters are driven from **real plans**: every golden in
//! `compiler/tests/golden/` is decoded from its `container:` hex, bound, and
//! planned by `pie_plan::compile_bound`.
//!
//! These cases were originally pinned against a C++ oracle that ran the
//! drivers' own emitters over the same plans (`compiler/tests/oracle/`, since
//! deleted with those emitters). What survives is `golden-{msl,cuda}/`: 2,838
//! cases that now serve as the Rust emitters' regression net. Bless with
//! `PTIR_REGEN=1` only after reading `ptir-refactor.md` §3.2.
//!
//! (The goldens' own `sidecar:` blobs are not used as the plan source: three
//! of them are stale on this branch relative to today's `pie-plan`.)

use pie_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use pie_ir::op::{IntrinsicId, Op};
use pie_ir::registry::{KernelInfo, ModelProfile, SinkScope, Stage};
use pie_ir::types::{DType, RngKind, Shape};
use pie_ir::validate::bind;
use pie_plan::{CompiledStage, compile_bound, debug_stage_plan};

/// Golden names in the order the corpus enumerates them.
pub const GOLDEN_NAMES: &[&str] = &[
    "beam_epilogue",
    "counter_pingpong",
    "dfa_ingraph",
    "extern_contrastive",
    "greedy_argmax",
    "matrix_mask_apply_packed",
    "matrix_select_mask",
    "mtp_verify_tail",
    "neg_body_type_error",
    "neg_intrinsic_wrong_stage",
    "neg_model_gated_missing",
    "neg_sink_at_epilogue",
    "neg_spsc_second_producer",
    "neg_t10_nonreplayable",
    "nucleus_sample",
    "pentathlon_iter",
    "pivot_predicates_multistage",
    "section3_masked_gumbel",
    "structured_masks",
];

pub fn golden_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/golden").into()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|index| u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).unwrap())
        .collect()
}

/// The `container:` hex line of a golden.
pub fn golden_container(name: &str) -> TraceContainer {
    let path = format!("{}/{name}.txt", golden_dir());
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path} missing"));
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("container: "))
        .unwrap_or_else(|| panic!("{path} has no container line"));
    pie_ir::container::decode(&unhex(line)).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

/// The bind-time profile each golden was authored against (mirrors the
/// `ptir_golden.rs` case that produced it — the goldens do not carry it).
pub fn golden_profile(name: &str) -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    match name {
        // `ModelProfile::dummy()` verbatim (vocab 32).
        "counter_pingpong"
        | "neg_body_type_error"
        | "neg_intrinsic_wrong_stage"
        | "neg_sink_at_epilogue"
        | "neg_spsc_second_producer"
        | "section3_masked_gumbel"
        | "structured_masks" => {}
        // Vendored-only trace (no compiler golden): its `Logits` is `[1, 4]`,
        // so it binds at vocab 4 and nowhere else. It did not "drift" -- the
        // profile was being guessed at 8 by the catch-all below.
        "staged_dispatch" => profile.vocab = 4,
        "beam_epilogue" => {
            profile.vocab = 8;
            profile.page_size = 4;
        }
        "neg_model_gated_missing" => profile.has_mtp_logits = false,
        "neg_t10_nonreplayable" => profile.kernels.push(KernelInfo {
            name: "envelope_dot".into(),
            sink_scope: None,
            replayable: false,
        }),
        "pentathlon_iter" => {
            profile.vocab = 8;
            profile.kernels.push(KernelInfo {
                name: "envelope_dot".into(),
                sink_scope: None,
                replayable: true,
            });
        }
        _ => profile.vocab = 8,
    }
    profile
}

fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

fn epilogue(channels: Vec<ChannelDecl>, ops: Vec<Op>) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

/// Library ops and intrinsics no golden reaches: `sort_desc`, `matmul`, the
/// `mtp_drafts` intrinsic (its own emission path in the grouped emitter), and
/// a second-party `sink_call` (the Metal sink-boundary check).
pub fn synthetic_traces() -> Vec<(&'static str, TraceContainer, ModelProfile)> {
    let mut small = ModelProfile::dummy();
    small.vocab = 8;
    let mut with_sink = small.clone();
    with_sink.kernels.push(KernelInfo {
        name: "lora".into(),
        sink_scope: Some(SinkScope::PassWide),
        replayable: true,
    });
    vec![
        (
            "synthetic_sort_desc",
            epilogue(
                vec![
                    chan(Shape::vector(8), DType::F32, HostRole::None, true),
                    chan(Shape::vector(8), DType::F32, HostRole::Reader, false),
                    chan(Shape::vector(8), DType::U32, HostRole::Reader, false),
                ],
                vec![
                    Op::ChanRead(0),
                    Op::SortDesc(0),
                    Op::ChanPut { chan: 1, value: 1 },
                    Op::ChanPut { chan: 2, value: 2 },
                ],
            ),
            small.clone(),
        ),
        (
            "synthetic_matmul",
            epilogue(
                vec![
                    chan(Shape::matrix(2, 3), DType::F32, HostRole::None, true),
                    chan(Shape::matrix(3, 4), DType::F32, HostRole::None, true),
                    chan(Shape::matrix(2, 4), DType::F32, HostRole::Reader, false),
                ],
                vec![
                    Op::ChanRead(0),
                    Op::ChanRead(1),
                    Op::MatMul(0, 1),
                    Op::ChanPut { chan: 2, value: 2 },
                ],
            ),
            small.clone(),
        ),
        (
            "synthetic_mtp_drafts",
            epilogue(
                vec![chan(Shape::vector(3), DType::I32, HostRole::Reader, false)],
                vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::MtpDrafts,
                        shape: Shape::vector(3),
                        dtype: DType::I32,
                    },
                    Op::IntrinsicVal {
                        intr: IntrinsicId::MtpLogits,
                        shape: Shape::matrix(3, 8),
                        dtype: DType::F32,
                    },
                    Op::ReduceArgmax(1),
                    Op::Cast {
                        value: 2,
                        dtype: DType::I32,
                    },
                    Op::Add(0, 3),
                    Op::ChanPut { chan: 0, value: 4 },
                ],
            ),
            small,
        ),
        (
            "synthetic_sink_call",
            TraceContainer {
                names: vec!["lora".into()],
                channels: vec![chan(Shape::vector(4), DType::F32, HostRole::None, true)],
                ports: Vec::new(),
                stages: vec![StageProgram {
                    stage: Stage::Prologue,
                    ops: vec![
                        Op::ChanRead(0),
                        Op::SinkCall {
                            name: 0,
                            args: vec![0],
                        },
                    ],
                }],
                externs: Vec::new(),
            },
            with_sink,
        ),
    ]
}

/// One corpus entry: a stage plan and where it came from.
pub struct CorpusStage {
    pub golden: String,
    pub stage_index: usize,
    pub stage_tag: u8,
    pub plan: CompiledStage,
    /// The plan rendered the way the runtime engine renders it
    /// (`pie_plan::debug_stage_plan`). Pinned instead of an encoding so a
    /// planning change lands in a golden as a readable diff.
    pub debug: String,
}

impl CorpusStage {
    /// The stable case id both dumps print.
    pub fn id(&self) -> String {
        format!("{}#{}", self.golden, self.stage_index)
    }
}

/// Compile every golden that binds, then the synthetic fill-in traces; goldens
/// whose bind fails contribute no stages (the `neg_*` cases).
pub fn corpus_stages() -> Vec<CorpusStage> {
    let mut stages = Vec::new();
    let mut push = |name: &str, container: TraceContainer, profile: ModelProfile| {
        let Ok(bound) = bind(container, profile) else {
            return;
        };
        for (stage_index, plan) in compile_bound(&bound).into_iter().enumerate() {
            let debug = debug_stage_plan(&plan);
            stages.push(CorpusStage {
                golden: name.into(),
                stage_index,
                stage_tag: plan.normalized.stage as u8,
                plan,
                debug,
            });
        }
    };
    for name in GOLDEN_NAMES {
        push(name, golden_container(name), golden_profile(name));
    }
    for (name, container, profile) in synthetic_traces() {
        push(name, container, profile);
    }
    stages
}

/// The op tag byte at `node` — `stage.ops[node].op.tag`.
pub fn op_tag(stage: &CompiledStage, node: u32) -> u8 {
    pie_codegen::metal::OpView::of(&stage.normalized.ops[node as usize]).tag
}

/// Whether the region is a library region — the C++ `region.library` bit.
pub fn is_library(region: &pie_plan::Region) -> bool {
    matches!(region.kind, pie_plan::RegionKind::Library(_))
}

/// The C++ `region.library_op` byte. Generated regions encode `0`, which
/// happens to collide with `PTIR_LIBRARY_NUCLEUS_SAMPLE`.
pub fn library_op_byte(region: &pie_plan::Region) -> u8 {
    match region.kind {
        pie_plan::RegionKind::Library(op) => op as u8,
        _ => 0,
    }
}

/// The `region: ...` line the oracle prints for each case.
pub fn region_shape(region: &pie_plan::Region) -> String {
    format!(
        "library={} library_op={} schedule={} nodes={} inputs={} outputs={} sinks={}",
        is_library(region) as u8,
        library_op_byte(region),
        region.schedule as u8,
        region.nodes.len(),
        region.inputs.len(),
        region.outputs.len(),
        region.sinks.len()
    )
}

/// A bound trace to hand `emit_program`, which needs one for the program-wide
/// Metal effect kernels.
///
/// `corpus_stages` deliberately flattens stages across every golden, so there
/// is no single program the flattened list belongs to. The coverage tests only
/// care that each kernel *family* is emitted, so any bound trace serves; this
/// returns the first golden's.
pub fn corpus_bound() -> pie_ir::validate::BoundTrace {
    let name = GOLDEN_NAMES[0];
    bind(golden_container(name), golden_profile(name)).expect("first golden binds")
}

fn staged(stage: Stage, channels: Vec<ChannelDecl>, ops: Vec<Op>) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram { stage, ops }],
        externs: Vec::new(),
    }
}

/// Traces that exist only to reach what the goldens and `synthetic_traces` do
/// not: 17 of the 55 ops in `OP_TABLE`, three of the eight intrinsics, the
/// `HierarchicalRow` schedule, and `Stage::OnAttn`.
///
/// Kept apart from [`corpus_stages`] on purpose. That corpus is the input to
/// `golden-{msl,cuda}/`, which are dumps of a C++ oracle that no longer exists;
/// adding cases there would change the case list of files whose expected column
/// can never be re-derived. These are pinned separately, against this
/// compiler's own output, and their goldens say so.
pub fn extended_traces() -> Vec<(&'static str, TraceContainer, ModelProfile)> {
    let mut small = ModelProfile::dummy();
    small.vocab = 8;
    let mut attn = small.clone();
    attn.has_attn_score = true;
    let mut identity_kernel = small.clone();
    identity_kernel.kernels.push(KernelInfo {
        name: "metal.identity".into(),
        sink_scope: None,
        replayable: true,
    });
    let mut discard_sink = small.clone();
    discard_sink.kernels.push(KernelInfo {
        name: "metal.discard".into(),
        sink_scope: Some(SinkScope::PassWide),
        replayable: true,
    });

    let f32v =
        |n: u32, role: HostRole, seeded: bool| chan(Shape::vector(n), DType::F32, role, seeded);
    let read = |n: u32| f32v(n, HostRole::None, true);
    let write = |n: u32| f32v(n, HostRole::Reader, false);
    let write_scalar = || {
        chan(
            Shape::new(&[]).unwrap(),
            DType::F32,
            HostRole::Reader,
            false,
        )
    };

    vec![
        (
            "extended_unary",
            staged(
                Stage::Epilogue,
                vec![read(4), write(4)],
                vec![
                    Op::ChanRead(0),
                    Op::Neg(0),
                    Op::Recip(1),
                    Op::Abs(2),
                    Op::Sign(3),
                    Op::ChanPut { chan: 1, value: 4 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_minmax",
            staged(
                Stage::Epilogue,
                vec![read(4), read(4), write(4), write_scalar()],
                vec![
                    Op::ChanRead(0),
                    Op::ChanRead(1),
                    Op::MaxElem(0, 1),
                    Op::MinElem(0, 1),
                    Op::ReduceMin(2),
                    Op::ChanPut { chan: 2, value: 3 },
                    Op::ChanPut { chan: 3, value: 4 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_predicates",
            staged(
                Stage::Epilogue,
                vec![
                    read(4),
                    read(4),
                    chan(Shape::vector(4), DType::Bool, HostRole::Reader, false),
                ],
                vec![
                    Op::ChanRead(0),
                    Op::ChanRead(1),
                    Op::Gt(0, 1),
                    Op::Ne(0, 1),
                    Op::Le(0, 1),
                    Op::Or(2, 3),
                    Op::Not(5),
                    Op::And(6, 4),
                    Op::ChanPut { chan: 2, value: 7 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_transpose_cumsum",
            staged(
                Stage::Epilogue,
                vec![
                    chan(Shape::matrix(2, 3), DType::F32, HostRole::None, true),
                    chan(Shape::matrix(3, 2), DType::F32, HostRole::Reader, false),
                    write(4),
                    read(4),
                ],
                vec![
                    Op::ChanRead(0),
                    Op::Transpose(0),
                    Op::ChanPut { chan: 1, value: 1 },
                    Op::ChanRead(3),
                    Op::CumSum(2),
                    Op::ChanPut { chan: 2, value: 3 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_gather_scatter",
            staged(
                Stage::Epilogue,
                vec![
                    chan(Shape::matrix(3, 4), DType::F32, HostRole::None, true),
                    chan(Shape::vector(3), DType::U32, HostRole::None, true),
                    read(4),
                    write(3),
                    write(4),
                ],
                vec![
                    Op::ChanRead(0),
                    Op::ChanRead(1),
                    Op::ChanRead(2),
                    Op::GatherRow { src: 0, idx: 1 },
                    Op::ChanPut { chan: 3, value: 3 },
                    Op::ScatterAdd {
                        base: 2,
                        idx: 1,
                        vals: 3,
                    },
                    Op::ChanPut { chan: 4, value: 4 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_rng_stream",
            staged(
                Stage::Epilogue,
                vec![write(4)],
                vec![
                    Op::Rng {
                        stream: 0,
                        shape: Shape::vector(4),
                        kind: RngKind::Uniform,
                    },
                    Op::ChanPut { chan: 0, value: 0 },
                ],
            ),
            small.clone(),
        ),
        (
            // Reduction over a static extent past the 32 768 the planner uses
            // to pick `HierarchicalRow`; no other corpus stage reaches it.
            "extended_hierarchical",
            staged(
                Stage::Epilogue,
                vec![read(65_536), write_scalar()],
                vec![
                    Op::ChanRead(0),
                    Op::ReduceSum(0),
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_hidden",
            staged(
                Stage::Epilogue,
                vec![write(1)],
                vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Hidden,
                        shape: Shape::matrix(1, 8),
                        dtype: DType::F32,
                    },
                    Op::ReduceSum(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            ),
            small.clone(),
        ),
        (
            "extended_on_attn",
            staged(
                Stage::OnAttn,
                vec![
                    chan(
                        Shape::new(&[]).unwrap(),
                        DType::U32,
                        HostRole::Reader,
                        false,
                    ),
                    write(4),
                ],
                vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Layer,
                        shape: Shape::new(&[]).unwrap(),
                        dtype: DType::U32,
                    },
                    Op::IntrinsicVal {
                        intr: IntrinsicId::AttnScore,
                        shape: Shape::vector(4),
                        dtype: DType::F32,
                    },
                    Op::ChanPut { chan: 0, value: 0 },
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            ),
            attn,
        ),
        (
            // `validate_singleton_plan` accepts exactly one kernel_call --
            // `metal.identity`, whose one argument has the result's type -- and
            // exactly one sink_call, `metal.discard`. Both are magic strings
            // matched against the container's name table and spelled in one
            // place each; nothing else in the corpus carries a name that
            // reaches an *accepted* plan, so both accept arms were unreached
            // and the name lookup behind them untested.
            "extended_metal_identity",
            TraceContainer {
                names: vec!["metal.identity".into()],
                channels: vec![read(4), write(4)],
                ports: Vec::new(),
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops: vec![
                        Op::ChanRead(0),
                        Op::KernelCall {
                            name: 0,
                            args: vec![0],
                            shape: Shape::vector(4),
                            dtype: DType::F32,
                        },
                        Op::ChanPut { chan: 1, value: 1 },
                    ],
                }],
                externs: Vec::new(),
            },
            identity_kernel,
        ),
        (
            "extended_metal_discard",
            TraceContainer {
                names: vec!["metal.discard".into()],
                channels: vec![read(4)],
                ports: Vec::new(),
                stages: vec![StageProgram {
                    // Prologue: a PassWide sink's effect is consumed for the
                    // whole pass, which is where bind will take one.
                    stage: Stage::Prologue,
                    ops: vec![
                        Op::ChanRead(0),
                        Op::SinkCall {
                            name: 0,
                            args: vec![0],
                        },
                    ],
                }],
                externs: Vec::new(),
            },
            discard_sink,
        ),
    ]
}

/// [`extended_traces`] compiled, in declaration order.
pub fn extended_stages() -> Vec<CorpusStage> {
    let mut stages = Vec::new();
    for (name, container, profile) in extended_traces() {
        let bound = bind(container, profile)
            .unwrap_or_else(|error| panic!("extended trace `{name}` must bind: {error:?}"));
        for (stage_index, plan) in compile_bound(&bound).into_iter().enumerate() {
            let debug = debug_stage_plan(&plan);
            stages.push(CorpusStage {
                golden: name.into(),
                stage_index,
                stage_tag: plan.normalized.stage as u8,
                plan,
                debug,
            });
        }
    }
    stages
}
