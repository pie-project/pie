use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, RopeForm, Value, ops,
    seam, trace_hybrid,
};
use model_ir::{Def, Dim, Elementwise, Operands, Operation, RuntimeInput, Ty};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

struct OneRope {
    axes: u8,
    dims: [u32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
}

const HEADS: u32 = 2;

impl ForwardHybrid for OneRope {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let q = inputs.latents(0, HEADS * self.head_dim, Dtype::Bf16);
        let positions = inputs.axis_positions(0, self.axes);
        let q = ops::elemwise::rope_axes(
            &q,
            &positions,
            self.dims,
            [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            self.form,
            self.rotary_dim,
            self.head_dim,
        );
        seam::at(seam::VELOCITY, &[&q]);
        q
    }
}

fn refusal(rope: OneRope) -> String {
    match std::panic::catch_unwind(|| trace_hybrid("rope", &rope, Platform::Cuda)) {
        Ok(_) => panic!("the rope traced"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default(),
    }
}

#[test]
fn the_axis_rope_states_its_axes_once_every_case() {
    three_axes_over_a_whole_head_trace_in_place();
    a_partial_rotation_leaves_the_tail_of_the_head_alone();
    a_rope_whose_numbers_disagree_is_refused_by_name();
}

fn three_axes_over_a_whole_head_trace_in_place() {
    let trace = trace_hybrid(
        "flux",
        &OneRope {
            axes: 3,
            dims: [16, 56, 56, 0],
            form: RopeForm::Interleaved,
            rotary_dim: 128,
            head_dim: 128,
        },
        Platform::Cuda,
    );
    let node = trace
        .nodes
        .iter()
        .find(|node| {
            matches!(
                node.op,
                Operation::Elementwise(Elementwise::RopeAxes { .. })
            )
        })
        .expect("one rope");
    let Operation::Elementwise(Elementwise::RopeAxes {
        x,
        positions,
        dims,
        form,
        x_out,
        ..
    }) = &node.op
    else {
        unreachable!()
    };
    assert_eq!(*dims, [16, 56, 56, 0]);
    assert_eq!(*form, RopeForm::Interleaved);
    let want = Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(u64::from(HEADS * 128))],
        dtype: Dtype::Bf16,
    };
    assert_eq!(trace.values[x_out.0 as usize].ty, want);
    let mut pairs = Vec::new();
    node.op.aliases(&mut pairs);
    assert_eq!(pairs, vec![(*x_out, *x)], "rotated in place");
    assert_eq!(
        trace.values[positions.0 as usize].def,
        Def::Input(RuntimeInput::AxisPositions { port: 0, axes: 3 })
    );
    assert_eq!(
        trace.values[positions.0 as usize].ty,
        Ty::Tensor {
            shape: vec![Dim::Tokens, Dim::Const(3)],
            dtype: Dtype::F32
        }
    );
}

fn a_partial_rotation_leaves_the_tail_of_the_head_alone() {
    let trace = trace_hybrid(
        "minimax",
        &OneRope {
            axes: 3,
            dims: [32, 32, 32, 0],
            form: RopeForm::Neox,
            rotary_dim: 96,
            head_dim: 128,
        },
        Platform::Cuda,
    );
    assert!(trace.nodes.iter().any(|node| matches!(
        node.op,
        Operation::Elementwise(Elementwise::RopeAxes {
            rotary_dim: 96,
            head_dim: 128,
            ..
        })
    )));
}

fn a_rope_whose_numbers_disagree_is_refused_by_name() {
    let message = refusal(OneRope {
        axes: 2,
        dims: [16, 56, 56, 0],
        form: RopeForm::Split,
        rotary_dim: 128,
        head_dim: 128,
    });
    assert!(
        message.contains("3 axes of dims want 3 position columns, not 2"),
        "{message}"
    );

    let message = refusal(OneRope {
        axes: 3,
        dims: [16, 56, 56, 0],
        form: RopeForm::Split,
        rotary_dim: 96,
        head_dim: 128,
    });
    assert!(message.contains("do not sum to rotary_dim 96"), "{message}");

    let message = refusal(OneRope {
        axes: 2,
        dims: [64, 64, 0, 0],
        form: RopeForm::Split,
        rotary_dim: 128,
        head_dim: 64,
    });
    assert!(
        message.contains("rotary_dim 128 within head_dim 64"),
        "{message}"
    );
}
