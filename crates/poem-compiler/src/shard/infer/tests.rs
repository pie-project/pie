use poem_ir::Operation;

/// Every op the inference's tables name is an op: a row under a name no op
/// carries never applies.
#[test]
fn every_op_the_inference_names_is_an_op() {
    let named = super::UNPACKS
        .iter()
        .chain(super::AFFINE.iter())
        .chain(super::PER_HEAD.iter().map(|(op, _)| op));
    let stale: Vec<&&str> = named.filter(|name| !Operation::names_an_op(name)).collect();
    assert!(stale.is_empty(), "named by no op: {stale:?}");
}

/// A one-token MLP: `x` (`hidden` wide) through `up` (`inner` wide), an
/// activation and `down` (`out` wide), added to a residual every rank holds
/// whole; and the indices of `up` and `down`.
fn mlp(hidden: u64, inner: u64, out: u64) -> (poem_ir::Trace, usize, usize) {
    use crate::fixture::{Build, act};
    use poem_ir::{
        ActFn, BinaryOp, Def, Dim, Dtype, Elemwise, Guard, Linear, Node, Param, Shard, Ty,
    };

    let weight = |b: &mut Build, name: &str, rows: u64, cols: u64| {
        let p = b.trace.params.len();
        b.trace.params.push(Param {
            name: name.to_string(),
            shape: vec![rows, cols],
            shard: Shard::Inferred,
            dtype: Dtype::Bf16,
            source: Default::default(),
            layout: Default::default(),
            placement: Default::default(),
            experts: None,
        });
        let ty = Ty::Tensor {
            shape: vec![Dim::Const(rows), Dim::Const(cols)],
            dtype: Dtype::Bf16,
        };
        (p, b.value(Def::Weight(p as u32), ty))
    };
    let push = |b: &mut Build, width: u64, op: &dyn Fn(poem_ir::ValueId) -> poem_ir::Operation| {
        let y = b.value(Def::Op(b.trace.nodes.len() as u32), act(width));
        b.trace.nodes.push(Node {
            op: op(y),
            guard: Guard::Always,
            layer: None,
        });
        y
    };
    let product = |b: &mut Build, act, w, width| {
        push(b, width, &|y| {
            Linear::Matmul {
                act,
                w,
                input_scales: None,
                scales: None,
                quant: None,
                logits: false,
                y,
            }
            .into()
        })
    };
    let silu = |b: &mut Build, x, width| {
        push(b, width, &|x_out| {
            Elemwise::Act {
                s: None,
                x,
                act: ActFn::Silu,
                scale: None,
                x_out,
            }
            .into()
        })
    };
    let mut b = Build::new();
    let x = b.input(hidden);
    let (up, up_w) = weight(&mut b, "up", inner, hidden);
    let (down, down_w) = weight(&mut b, "down", out, inner);
    let h = product(&mut b, x, up_w, inner);
    let a = silu(&mut b, h, inner);
    let y = product(&mut b, a, down_w, out);
    let residual = b.input(out);
    push(&mut b, out, &|sum| {
        Elemwise::Binary {
            op: BinaryOp::Add,
            x: y,
            y: residual,
            z: sum,
        }
        .into()
    });
    (b.trace, up, down)
}

/// A chain is cut where it saves what its collectives cost: a low-rank pair
/// on a wide row stays whole, its reduce not worth what it divides; a square
/// one is cut by its columns, then its rows; and a light product whose input
/// arrives cut follows the cut that made it, weighed with it.
#[test]
fn a_chain_is_cut_only_where_it_outweighs_its_collectives() {
    let cut = |shard: &poem_ir::Shard, axis: u32| matches!(shard, poem_ir::Shard::Cut { axis: a, .. } if *a == axis);
    let (light, up, down) = mlp(4096, 64, 4096);
    let shards = super::infer(&light);
    assert_eq!(
        shards[up],
        poem_ir::Shard::Replicated,
        "a low-rank pair is whole"
    );
    assert_eq!(shards[down], poem_ir::Shard::Replicated);

    let (heavy, up, down) = mlp(4096, 4096, 4096);
    let shards = super::infer(&heavy);
    assert!(cut(&shards[up], 0), "{:?}", shards[up]);
    assert!(cut(&shards[down], 1), "{:?}", shards[down]);

    let (follows, up, down) = mlp(4096, 4096, 8);
    let shards = super::infer(&follows);
    assert!(cut(&shards[up], 0), "{:?}", shards[up]);
    assert!(cut(&shards[down], 1), "a light product read cut follows");
}
