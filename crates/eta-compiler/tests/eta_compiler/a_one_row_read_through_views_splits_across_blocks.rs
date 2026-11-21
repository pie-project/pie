//! A one-row region that reads the model's row through views (squeezed to
//! a one-row fire's `[v]`, then reshaped back to `[1, v]`) and a channel's
//! scalar cell runs across its lane's blocks, as a region whose rows are
//! the output itself does: each block reads its stretch of the model's row,
//! which none writes, and the region's reductions meet in its row's work.

use crate::msl_corpus::Served;
use eta_compiler::codegen::cuda::emit_fused_region;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{Op, tags};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Shape};

const VOCAB: u32 = 65_536;

fn cell(len: u32, dtype: Dtype, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(len),
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded,
    }
}

/// A width-1 beam step's candidates: its carried score relative to the
/// best, broadcast over the row, plus the row's log-softmax; its top one.
fn subject() -> TraceContainer {
    let row = Shape::matrix(1, VOCAB);
    let ops = vec![
        Op::ChanTake(0),
        Op::Output {
            index: 0,
            lane: 0,
            shape: row,
            dtype: Dtype::F32,
        },
        Op::Reshape { value: 1, shape: Shape::vector(VOCAB) },
        Op::Reshape { value: 2, shape: row },
        Op::ReduceMax(0),
        Op::Broadcast { value: 4, shape: Shape::vector(1) },
        Op::Sub(0, 5),
        Op::Broadcast { value: 6, shape: row },
        Op::ReduceMax(3),
        Op::Broadcast { value: 8, shape: row },
        Op::Sub(3, 9),
        Op::Exp(10),
        Op::ReduceSum(11),
        Op::Log(12),
        Op::Broadcast { value: 13, shape: row },
        Op::Sub(10, 14),
        Op::Add(7, 15),
        Op::Reshape { value: 16, shape: Shape::vector(VOCAB) },
        Op::TopK { input: 17, k: 1 },
        Op::ChanPut { chan: 1, value: 18 },
    ];
    TraceContainer {
        names: vec!["logits".into()],
        channels: vec![
            cell(1, Dtype::F32, HostRole::Writer, true),
            cell(1, Dtype::F32, HostRole::Reader, false),
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

#[test]
fn a_one_row_read_through_views_splits_across_blocks() {
    let bound = Served::of(VOCAB).bind(subject()).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let package = eta_compiler::codegen::launch::build(&bound, &stages);
    let plan = package.plans.first().expect("one plan");
    let reads_row = plan
        .fused
        .iter()
        .find(|region| {
            region
                .nodes
                .iter()
                .any(|&node| plan.ops[node as usize].tag == tags::OUTPUT)
        })
        .expect("a region reads the model's row");
    assert!(
        reads_row.splits_rows,
        "the region reading the model's row through its views runs on one block"
    );
    for (index, region) in stage.fused.regions.iter().enumerate() {
        if region.kind != eta_compiler::plan::RegionKind::Generated {
            continue;
        }
        emit_fused_region(&format!("k{index}"), stage, region).expect("cuda emits");
    }
}
