use crate::msl_corpus::Served;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Shape};

fn channel(len: u32, dtype: Dtype, host_role: HostRole) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(len),
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded: false,
    }
}

/// `scatter_add(base, idx, vals)` of `dtype` values, put to the host.
fn subject(dtype: Dtype) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            channel(4, dtype, HostRole::Writer),
            channel(8, Dtype::U32, HostRole::Writer),
            channel(8, dtype, HostRole::Writer),
            channel(4, dtype, HostRole::Reader),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::ChanTake(2),
                Op::ScatterAdd {
                    base: 0,
                    idx: 1,
                    vals: 2,
                },
                Op::ChanPut { chan: 3, value: 3 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The calls a kernel body makes (the runtime text defines both functions).
const PARALLEL: &str = "    ptir_parallel_scatter_add_int(";
const SERIAL: &str = "if (threadIdx.x == 0u) ptir_scatter_updates(";

/// Every region of `subject(dtype)` as the CUDA and HIP emitters emit it.
fn emitted(dtype: Dtype) -> String {
    let bound = Served::of(16).bind(subject(dtype)).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        let name = format!("k{index}");
        source.push_str(
            &eta_compiler::codegen::cuda::emit_region(&name, stage, region).expect("cuda emits"),
        );
        source.push_str(
            &eta_compiler::codegen::hip::emit_region(&name, stage, region).expect("hip emits"),
        );
    }
    source
}

#[test]
fn an_integer_scatter_add_runs_on_every_thread_every_case() {
    an_integer_scatter_add_adds_on_every_thread();
    a_float_scatter_add_keeps_its_order();
}

/// An integer scatter-add adds with atomics across the block (integer sums
/// are exact in any order), never on one thread.
fn an_integer_scatter_add_adds_on_every_thread() {
    for dtype in [Dtype::U32, Dtype::I32] {
        let source = emitted(dtype);
        assert!(
            source.contains(PARALLEL),
            "a {dtype:?} scatter-add does not run on every thread"
        );
        assert!(
            !source.contains(SERIAL),
            "a {dtype:?} scatter-add still runs on one thread"
        );
    }
}

/// A float scatter-add keeps its serial order (atomic float sums would round
/// by arrival order).
fn a_float_scatter_add_keeps_its_order() {
    let source = emitted(Dtype::F32);
    assert!(source.contains(SERIAL));
    assert!(!source.contains(PARALLEL));
}
