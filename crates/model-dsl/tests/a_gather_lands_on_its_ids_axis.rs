use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::{Dim, Layout, Operation, Trace, Ty};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

#[derive(Clone, Copy)]
enum Axis {
    Tokens,
    Patches,
}

#[derive(Clone, Copy)]
enum Gather {
    Plain,
    Weighted,
}

struct OneGather {
    axis: Axis,
    gather: Gather,
}

const HIDDEN: u64 = 64;

const VOCAB: u32 = 2304;

const TAPS: u32 = 4;

impl ForwardHybrid for OneGather {
    type Facts = NoFacts;

    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let table = Weight::sym("pos_embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16);
        let taps = match self.gather {
            Gather::Plain => 1,
            Gather::Weighted => TAPS,
        };
        let ids = match self.axis {
            Axis::Tokens => inputs.tokens(),
            Axis::Patches => inputs.patch_embed_rows(taps),
        };
        match self.gather {
            Gather::Plain => ops::layout::embed(&ids, &table, VOCAB),
            Gather::Weighted => {
                let weights = inputs.patch_embed_weights(taps);
                ops::layout::embed_weighted(&ids, &weights, &table, VOCAB)
            }
        }
    }
}

fn gathered(axis: Axis, gather: Gather) -> Ty {
    let trace: Trace = trace_hybrid("one_gather", &OneGather { axis, gather }, Platform::Cuda);
    let out = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Layout(Layout::Embed { y, .. } | Layout::EmbedWeighted { y, .. }) => {
                Some(*y)
            }
            _ => None,
        })
        .expect("the trace holds the gather it was written for");
    trace.values[out.0 as usize].ty.clone()
}

fn leading(ty: &Ty) -> Dim {
    match ty {
        Ty::Tensor { shape, .. } => *shape.first().expect("a gather answers a rectangle"),
        Ty::Struct(_) => panic!("a gather does not answer a plan payload"),
    }
}

#[test]
fn a_gather_lands_on_its_ids_axis_every_case() {
    the_plain_gather_follows_its_ids_onto_either_axis();
    the_interpolating_gather_follows_its_ids_too();
    the_axis_is_not_how_a_text_picks_the_op();
}

fn the_plain_gather_follows_its_ids_onto_either_axis() {
    let tokens = gathered(Axis::Tokens, Gather::Plain);
    assert_eq!(
        leading(&tokens),
        Dim::Tokens,
        "a token-axis gather answered {tokens:?}, and every text before the towers reads it \
         as a token rectangle"
    );

    let patches = gathered(Axis::Patches, Gather::Plain);
    assert_eq!(
        leading(&patches),
        Dim::Patches,
        "a patch-axis gather answered {patches:?}. A `Dim::Tokens` here is the bug §9.2's \
         design rests on not existing: the rectangle would be cut at the token window, sized \
         by max_tokens, and run in the trunk's capture unit"
    );

    assert_eq!(
        tokens,
        Ty::Tensor {
            shape: vec![Dim::Tokens, Dim::Const(HIDDEN)],
            dtype: Dtype::Bf16,
        },
        "the gather's width and element are the table's, on both axes"
    );
}

fn the_interpolating_gather_follows_its_ids_too() {
    assert_eq!(
        leading(&gathered(Axis::Tokens, Gather::Weighted)),
        Dim::Tokens
    );
    let patches = gathered(Axis::Patches, Gather::Weighted);
    assert_eq!(
        leading(&patches),
        Dim::Patches,
        "the interpolating gather answered {patches:?} on the patch axis"
    );
}

fn the_axis_is_not_how_a_text_picks_the_op() {
    for axis in [Axis::Tokens, Axis::Patches] {
        let trace = trace_hybrid(
            "one_gather",
            &OneGather {
                axis,
                gather: Gather::Plain,
            },
            Platform::Cuda,
        );
        let gathers = trace
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Operation::Layout(Layout::Embed { .. })))
            .count();
        assert_eq!(
            gathers, 1,
            "a gather on either axis is one `layout.embed` node and not an axis-specific arm"
        );
    }
}
