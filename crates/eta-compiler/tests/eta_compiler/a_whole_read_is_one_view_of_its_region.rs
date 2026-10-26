use crate::msl_corpus::Served;
use eta_compiler::codegen::cuda::emit_fused_region;
use eta_compiler::plan::{CompiledStage, compile_bound};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, RngKind, Shape};

/// Two rows: a keyed draw's `[key, counter]` state is shaped like a row
/// vector of them, one element a row.
const ROWS: u32 = 2;
const WIDTH: u32 = 16;

fn chan(shape: Shape, dtype: Dtype, host_role: HostRole, seeded: bool) -> ChannelDecl {
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
        inputs: Vec::new(),
    }
}

/// The state is read whole by the draw and, by rows, by an op on the row
/// geometry beside it (`state + state`).
fn read_both_ways() -> TraceContainer {
    let row = Shape::matrix(ROWS, WIDTH);
    epilogue(
        vec![
            chan(row, Dtype::F32, HostRole::None, true),
            chan(Shape::vector(ROWS), Dtype::U32, HostRole::None, true),
            chan(Shape::vector(ROWS), Dtype::F32, HostRole::Reader, false),
            chan(Shape::vector(ROWS), Dtype::U32, HostRole::Reader, false),
        ],
        vec![
            Op::ChanRead(0),
            Op::ChanRead(1),
            Op::RngKeyed {
                state: 1,
                shape: row,
                kind: RngKind::Uniform,
            },
            Op::Add(0, 2),
            Op::ReduceMax(3),
            Op::Add(1, 1),
            Op::ChanPut { chan: 2, value: 4 },
            Op::ChanPut { chan: 3, value: 5 },
        ],
    )
}

/// The state is computed by rows (a row's argmax) and then read whole by
/// the draw.
fn computed_by_rows() -> TraceContainer {
    let row = Shape::matrix(ROWS, WIDTH);
    epilogue(
        vec![
            chan(row, Dtype::F32, HostRole::None, true),
            chan(Shape::vector(ROWS), Dtype::F32, HostRole::Reader, false),
        ],
        vec![
            Op::ChanRead(0),
            Op::ReduceArgmax(0),
            Op::Cast {
                value: 1,
                dtype: Dtype::U32,
            },
            Op::RngKeyed {
                state: 2,
                shape: row,
                kind: RngKind::Uniform,
            },
            Op::Add(0, 3),
            Op::ReduceMax(4),
            Op::ChanPut { chan: 1, value: 5 },
        ],
    )
}

fn row_kinds(source: &str) -> Vec<String> {
    let head = source
        .find("ptir_rowkind[")
        .expect("a row-parallel region declares its row kinds");
    let open = head + source[head..].find('{').expect("the kinds' initializer");
    let close = open
        + source[open..]
            .find('}')
            .expect("the kinds' initializer ends");
    source[open + 1..close]
        .split(',')
        .map(|kind| kind.trim().to_string())
        .collect()
}

/// Every value an op of a region reads whole is whole in the region: no
/// other op of it reads the value by rows or writes it, and the emitted
/// region never narrows it to a row.
fn assert_whole_reads_are_whole(name: &str, stage: &CompiledStage) {
    let ops = &stage.normalized.ops;
    let mut producer = Vec::new();
    for (node, op) in ops.iter().enumerate() {
        producer.extend(core::iter::repeat_n(node, op.result_count() as usize));
    }
    let mut checked = 0;
    for (draw, op) in ops.iter().enumerate() {
        let Op::RngKeyed { state, .. } = *op else {
            continue;
        };
        let region = stage
            .fused
            .regions
            .iter()
            .find(|region| region.nodes.iter().any(|n| n.get() as usize == draw))
            .expect("the draw is in a region");
        let nodes: Vec<usize> = region.nodes.iter().map(|n| n.get() as usize).collect();
        assert!(
            !nodes.contains(&producer[state as usize]),
            "{name}: the region that reads the state whole also computes it"
        );
        for &other in &nodes {
            assert!(
                other == draw || !ops[other].operands().contains(&state),
                "{name}: node {other} reads by rows the state node {draw} reads whole, \
                 in one region"
            );
        }
        let source = emit_fused_region("whole", stage, region).expect("the CUDA emitter serves it");
        assert_eq!(
            row_kinds(&source)[state as usize],
            "0u",
            "{name}: the region narrows the state its draw reads whole"
        );
        checked += 1;
    }
    assert!(checked > 0, "{name}: the subject draws");
}

#[test]
fn a_whole_read_is_one_view_of_its_region() {
    for (name, container) in [
        ("read both ways", read_both_ways()),
        ("computed by rows", computed_by_rows()),
    ] {
        let bound = Served::of(WIDTH)
            .bind(container)
            .expect("the subject binds");
        let stages = compile_bound(&bound);
        assert_whole_reads_are_whole(name, stages.first().expect("one stage"));
    }
}
