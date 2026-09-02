//! Peepholes over a traced forward: two adjacent nodes become the one
//! launch that lands both. Every value and every reader survives, so a
//! fused trace checks and compiles as the traced one did.

use crate::ops::{Elementwise, Operation};
use crate::trace::{Node, Trace};
use crate::value::Def;

/// `residual_add` followed by the `rmsnorm` that reads its result, under
/// the same guard, becomes `residual_add_rmsnorm`. The pair may straddle a
/// layer boundary (a block's last fold and the next block's first norm);
/// the fused node keeps the norm's layer, since that is the weight it
/// reads.
#[must_use]
pub fn residual_norm(mut trace: Trace) -> Trace {
    let mut nodes = Vec::with_capacity(trace.nodes.len());
    // Where each traced node lands: a value's `Def::Op` names its node by
    // index, and a fused pair's second node lands on the first's.
    let mut landed = Vec::with_capacity(trace.nodes.len());
    let mut rest = trace.nodes.into_iter().peekable();
    while let Some(node) = rest.next() {
        let at = nodes.len() as u32;
        landed.push(at);
        match rest.peek().and_then(|next| pair(&node, next)) {
            Some(fused) => {
                rest.next();
                landed.push(at);
                nodes.push(fused);
            }
            None => nodes.push(node),
        }
    }
    for value in &mut trace.values {
        if let Def::Op(node) = &mut value.def {
            *node = landed[*node as usize];
        }
    }
    trace.nodes = nodes;
    trace
}

fn pair(add: &Node, norm: &Node) -> Option<Node> {
    if add.guard != norm.guard {
        return None;
    }
    let Operation::Elementwise(Elementwise::ResidualAdd { x, y, y_out }) = &add.op else {
        return None;
    };
    let (normed, weight, plus_one, eps, out) = match &norm.op {
        Operation::Elementwise(Elementwise::Rmsnorm { x, weight, eps, y }) => {
            (*x, *weight, false, *eps, *y)
        }
        Operation::Elementwise(Elementwise::RmsnormPlusOne { x, weight, eps, y }) => {
            (*x, *weight, true, *eps, *y)
        }
        _ => return None,
    };
    if normed != *y_out {
        return None;
    }
    Some(Node {
        op: Operation::Elementwise(Elementwise::ResidualAddRmsnorm {
            x: *x,
            y: *y,
            y_out: *y_out,
            weight,
            plus_one,
            eps,
            out,
        }),
        guard: add.guard.clone(),
        layer: norm.layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard::Guard;
    use crate::value::ValueId;

    fn node(op: Elementwise, layer: Option<u32>) -> Node {
        Node {
            op: Operation::Elementwise(op),
            guard: Guard::Always,
            layer,
        }
    }

    fn add(y_out: u32) -> Elementwise {
        Elementwise::ResidualAdd {
            x: ValueId(1),
            y: ValueId(2),
            y_out: ValueId(y_out),
        }
    }

    fn norm(x: u32) -> Elementwise {
        Elementwise::RmsnormPlusOne {
            x: ValueId(x),
            weight: ValueId(4),
            eps: 1e-6,
            y: ValueId(5),
        }
    }

    fn trace_of(nodes: Vec<Node>) -> Trace {
        Trace {
            name: String::new(),
            platform: crate::trace::Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
            nodes,
            seams: Vec::new(),
        }
    }

    #[test]
    fn the_add_and_the_norm_that_reads_it_become_one_node() {
        let fused = residual_norm(trace_of(vec![node(add(3), Some(0)), node(norm(3), Some(0))]));
        assert_eq!(fused.nodes.len(), 1);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::ResidualAddRmsnorm {
                y_out: ValueId(3),
                out: ValueId(5),
                plus_one: true,
                ..
            })
        ));
    }

    #[test]
    fn a_value_defined_past_the_pair_still_names_its_node() {
        use crate::value::{Ty, ValueDecl};
        let mut trace =
            trace_of(vec![node(add(3), Some(0)), node(norm(3), Some(0)), node(add(7), Some(1))]);
        let decl = |node: u32| ValueDecl {
            def: Def::Op(node),
            ty: Ty::Tensor {
                shape: Vec::new(),
                dtype: dtype::Dtype::Bf16,
            },
        };
        trace.values = vec![decl(1), decl(2)];
        let fused = residual_norm(trace);
        assert_eq!(fused.nodes.len(), 2);
        assert!(matches!(fused.values[0].def, Def::Op(0)));
        assert!(matches!(fused.values[1].def, Def::Op(1)));
    }

    #[test]
    fn a_norm_of_something_else_stays_apart_and_a_layer_boundary_does_not() {
        let other = residual_norm(trace_of(vec![node(add(3), Some(0)), node(norm(9), Some(0))]));
        assert_eq!(other.nodes.len(), 2);
        let layer = residual_norm(trace_of(vec![node(add(3), Some(0)), node(norm(3), Some(1))]));
        assert_eq!(layer.nodes.len(), 1);
        assert_eq!(layer.nodes[0].layer, Some(1));
    }
}
