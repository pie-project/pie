use engine_cuda::program::scratch_offsets;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op, tags};
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::bind;

const ROWS: u32 = 256;
const VOCAB: u32 = 4096;

fn softmax_epilogue() -> TraceContainer {
    let ops = vec![
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(ROWS, VOCAB),
            dtype: Dtype::F32,
        },
        Op::ReduceMax(0),
        Op::Reshape {
            value: 1,
            shape: Shape::matrix(ROWS, 1),
        },
        Op::Broadcast {
            value: 2,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Sub(0, 3),
        Op::Exp(4),
        Op::ReduceSum(5),
        Op::Reshape {
            value: 6,
            shape: Shape::matrix(ROWS, 1),
        },
        Op::Broadcast {
            value: 7,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Div(5, 8),
        Op::ChanPut { chan: 0, value: 9 },
    ];
    let probs_out = ChannelDecl {
        shape: Shape::matrix(ROWS, VOCAB),
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![probs_out],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

fn package() -> LaunchPackage {
    let profile = ModelProfile {
        vocab: VOCAB,
        ..ModelProfile::dummy()
    };
    let bound = bind(softmax_epilogue(), profile).expect("the softmax epilogue binds");
    let stages = compile_bound(&bound);
    eta_compiler::codegen::launch::build(&bound, &stages)
}

#[test]
fn the_row_max_keeps_its_slot_until_the_region_ends() {
    let package = package();
    let plan = package
        .plans
        .iter()
        .find(|plan| plan.ops.iter().any(|op| op.tag == tags::REDUCE_SUM))
        .expect("the epilogue's plan");
    let mut base = 0u32;
    let mut max = None;
    let mut sum = None;
    for op in &plan.ops {
        if op.tag == tags::REDUCE_MAX {
            max = Some(base);
        }
        if op.tag == tags::REDUCE_SUM {
            sum = Some(base);
        }
        base += u32::from(op.result_count);
    }
    let (max, sum) = (max.expect("a row max"), sum.expect("a row sum"));
    let offsets = scratch_offsets(plan, Extents::default()).expect("the lane's layout");
    assert_ne!(
        offsets[max as usize], offsets[sum as usize],
        "the row max (value {max}) and the row sum (value {sum}) share a slot: the \
         softmax's last pass still reads the max to recompute exp(x - m) after the \
         pass that stores the sum, so the max must live until the region ends"
    );
}
