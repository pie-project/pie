use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, RopeForm, Value, ops, seam, trace_hybrid,
};
use poem_ir::{Def, Dim, Elemwise, InputAxis, Operands, Operation, RuntimeInput, Ty};

struct OneRope {
    axes: u8,
    dims: [u32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
}

const HEADS: u32 = 2;

impl ForwardHybrid for OneRope {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let q = inputs.input(
            "latents",
            Dtype::Bf16,
            poem_ir::InputAxis::Row,
            HEADS * self.head_dim,
            None,
        );
        let positions = inputs.input(
            "positions",
            Dtype::F32,
            poem_ir::InputAxis::Row,
            u32::from(self.axes),
            None,
        );
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
        .find(|node| matches!(node.op, Operation::Elemwise(Elemwise::RopeAxes { .. })))
        .expect("one rope");
    let Operation::Elemwise(Elemwise::RopeAxes {
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
        Def::Input(RuntimeInput::Guest {
            id: trace
                .inputs
                .iter()
                .position(|i| i.name == "positions")
                .expect("a positions input") as u32,
            axis: InputAxis::Row,
            width: 3,
            dtype: Dtype::F32,
        })
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
        Operation::Elemwise(Elemwise::RopeAxes {
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
