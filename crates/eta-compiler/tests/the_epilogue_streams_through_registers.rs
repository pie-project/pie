use eta_compiler::codegen::cuda::fused::emit_fused_region;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::RngKind;
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::bind;

const ROWS: u32 = 256;
const VOCAB: u32 = 4096;

fn scalar_writer() -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 1,
        host_role: HostRole::Writer,
        seeded: true,
    }
}

fn rows_out(dtype: Dtype) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(ROWS),
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

fn subject() -> TraceContainer {
    let ops = vec![
        Op::ChanRead(0),
        Op::Reshape {
            value: 0,
            shape: Shape::new(&[]).unwrap(),
        },
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(ROWS, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Div(2, 1),
        Op::ReduceMax(3),
        Op::Broadcast {
            value: 4,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Sub(3, 5),
        Op::Exp(6),
        Op::ReduceSum(7),
        Op::Broadcast {
            value: 8,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Div(7, 9),
        Op::Log(10),
        Op::Mul(10, 11),
        Op::ReduceSum(12),
        Op::Neg(13),
        Op::ReduceArgmax(3),
        Op::Rng {
            stream: 7,
            shape: Shape::matrix(ROWS, VOCAB),
            kind: RngKind::Gumbel,
        },
        Op::Add(3, 16),
        Op::ReduceArgmax(17),
        Op::Const(Literal::F32(0.1)),
        Op::Lt(14, 19),
        Op::ChanPut { chan: 1, value: 15 },
        Op::ChanPut { chan: 2, value: 18 },
        Op::ChanPut { chan: 3, value: 20 },
        Op::ChanPut { chan: 4, value: 14 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            scalar_writer(),
            rows_out(Dtype::I32),
            rows_out(Dtype::I32),
            rows_out(Dtype::Bool),
            rows_out(Dtype::F32),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

fn profile() -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    profile.vocab = VOCAB;
    profile
}

#[test]
fn the_epilogue_streams_through_registers_every_case() {
    the_row_parallel_epilogue_fuses_into_streams();
    a_reshaped_row_vector_broadcasts_through_the_stream();
    a_top_k_of_the_scaled_logits_reads_the_plane_and_stores_nothing();
}

fn the_row_parallel_epilogue_fuses_into_streams() {
    let bound = bind(subject(), profile()).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        if region.row_value.is_none() {
            continue;
        }
        let name = format!("k{index}");
        let emitted = emit_fused_region(&name, stage, region).expect("cuda emits");
        let body = emitted
            .split("descriptors = ptir_rowdesc;")
            .nth(1)
            .expect("a row-parallel kernel has the row-view preamble");
        source.push_str(body);
    }
    assert!(!source.is_empty(), "no row-parallel region");
    let streams = source.matches("// stream of ").count();
    assert!(
        (2..=5).contains(&streams),
        "{streams} streams for a chain of a dozen elementwise ops; the emitter fell back to a loop per op or fused across a reduction"
    );
    let helper_loops = source.matches("ptir_parallel_elementwise(").count();
    assert!(
        helper_loops <= 2,
        "{helper_loops} elementwise ops still run as their own loop over the row"
    );
    assert!(
        !source.contains("ptir_parallel_broadcast("),
        "a scalar's broadcast still materialises the row"
    );
    assert!(
        !source.contains("ptir_parallel_intrinsic("),
        "the logits are read only by the stream and must not be copied into scratch first"
    );
    assert!(
        !source.contains("offsets[2]"),
        "the logits value (2) never touches scratch"
    );
    for spent in [3u32, 6, 11, 12] {
        assert!(
            !source.contains(&format!("offsets[{spent}]")),
            "value {spent} is spent inside its stream and must not touch scratch"
        );
    }
}

fn a_reshaped_row_vector_broadcasts_through_the_stream() {
    let ops = vec![
        Op::ChanRead(0),
        Op::Reshape {
            value: 0,
            shape: Shape::new(&[]).unwrap(),
        },
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(ROWS, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Div(2, 1),
        Op::ReduceMax(3),
        Op::Reshape {
            value: 4,
            shape: Shape::matrix(ROWS, 1),
        },
        Op::Broadcast {
            value: 5,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Sub(3, 6),
        Op::Exp(7),
        Op::ReduceSum(8),
        Op::ChanPut { chan: 1, value: 9 },
    ];
    let container = TraceContainer {
        names: Vec::new(),
        channels: vec![scalar_writer(), rows_out(Dtype::F32)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    };
    let bound = bind(container, profile()).expect("binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    assert!(
        !stage
            .normalized
            .ops
            .iter()
            .any(|op| matches!(op, Op::Reshape { shape, .. } if shape.rank() == 2)),
        "the [rows, 1] reshape survived normalization: {:?}",
        stage.normalized.ops
    );
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        if region.row_value.is_none() {
            continue;
        }
        let emitted = emit_fused_region(&format!("k{index}"), stage, region).expect("cuda emits");
        source.push_str(
            emitted
                .split("descriptors = ptir_rowdesc;")
                .nth(1)
                .unwrap_or(""),
        );
    }
    assert!(
        !source.contains("ptir_parallel_broadcast("),
        "the row max's broadcast still materialises the row"
    );
}

fn a_top_k_of_the_scaled_logits_reads_the_plane_and_stores_nothing() {
    let ops = vec![
        Op::ChanRead(0),
        Op::Reshape {
            value: 0,
            shape: Shape::new(&[]).unwrap(),
        },
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(ROWS, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Div(2, 1),
        Op::ReduceMax(3),
        Op::Reshape {
            value: 4,
            shape: Shape::matrix(ROWS, 1),
        },
        Op::Broadcast {
            value: 5,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Sub(3, 6),
        Op::Exp(7),
        Op::ReduceSum(8),
        Op::TopK { input: 3, k: 8 },
        Op::ChanPut { chan: 1, value: 9 },
        Op::ChanPut { chan: 2, value: 11 },
    ];
    let taps_out = ChannelDecl {
        shape: Shape::matrix(ROWS, 8),
        dtype: ChanDType::Concrete(Dtype::U32),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    };
    let container = TraceContainer {
        names: Vec::new(),
        channels: vec![scalar_writer(), rows_out(Dtype::F32), taps_out],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    };
    let bound = bind(container, profile()).expect("binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let mut generated = String::new();
    let mut order = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        let name = format!("k{index}");
        if eta_compiler::codegen::cuda::order::is_order_region(stage, region) {
            order.push_str(
                &eta_compiler::codegen::cuda::order::emit_order_region(&name, stage, region)
                    .expect("order emits"),
            );
        } else if region.row_value.is_some() {
            let emitted = emit_fused_region(&name, stage, region).expect("cuda emits");
            generated.push_str(
                emitted
                    .split("descriptors = ptir_rowdesc;")
                    .nth(1)
                    .unwrap_or(""),
            );
        }
    }
    assert!(
        !generated.contains("offsets[3]"),
        "`scaled` is stored although its only readers recompute it or rank the plane"
    );
    assert!(
        order.contains("kDirectIntrinsic = 0u") || order.contains("kDirectIntrinsic = 1u"),
        "the top_k does not rank the intrinsic plane directly:\n{}",
        order
            .lines()
            .filter(|l| l.contains("kDirect"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        order.contains("kDirectDivisor = 1u"),
        "the divisor is not the value the divide names (the reshaped temperature, 1)"
    );
    let one_element = |value: &u32| {
        stage.normalized.value_types[*value as usize]
            .dims
            .iter()
            .all(|dim| matches!(dim, eta_compiler::plan::Dimension::Static(1)))
    };
    let ranks = stage
        .fused
        .regions
        .iter()
        .find(|region| {
            matches!(
                region.kind,
                eta_compiler::plan::RegionKind::Library(eta_compiler::plan::LibraryOp::TopK)
            )
        })
        .expect("the top_k has a region");
    let divisor = ranks
        .inputs
        .iter()
        .find(|value| one_element(value))
        .unwrap_or_else(|| {
            panic!(
                "the top_k region names no one-element input: {:?}",
                ranks.inputs
            )
        });
    assert!(
        stage
            .fused
            .regions
            .iter()
            .any(|region| !core::ptr::eq(region, ranks) && region.outputs.contains(divisor)),
        "no region names the divisor {divisor} an output"
    );
}
