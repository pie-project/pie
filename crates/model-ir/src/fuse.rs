use crate::operands::Operands;
use crate::ops::elemwise::{NormKind, PostNorm};
use crate::ops::{Attention, Elementwise, Layout, Linear, Operation};
use crate::trace::{Node, Trace};
use crate::value::{Def, ValueDecl, ValueId};

#[must_use]
pub fn residual_norm(mut trace: Trace) -> Trace {
    let mut nodes = Vec::with_capacity(trace.nodes.len());
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

#[must_use]
pub fn residual_chains(mut trace: Trace) -> Trace {
    let mut nodes = Vec::with_capacity(trace.nodes.len());
    let mut landed = Vec::with_capacity(trace.nodes.len());
    let old = trace.nodes;
    let values = &trace.values;
    let mut i = 0usize;
    while i < old.len() {
        if let Some((fused, took)) = norm_chain(&old[i..], values) {
            let at = nodes.len() as u32;
            landed.extend(std::iter::repeat_n(at, took));
            nodes.push(fused);
            i += took;
            continue;
        }
        if let Some((fused, before, took)) = embed_chain(&old[i..]) {
            let ahead = before.is_some();
            if let Some(before) = before {
                nodes.push(before);
            }
            let at = nodes.len() as u32;
            for k in 0..took {
                landed.push(if ahead && k == 2 { at - 1 } else { at });
            }
            nodes.push(fused);
            i += took;
            continue;
        }
        landed.push(nodes.len() as u32);
        nodes.push(old[i].clone());
        i += 1;
    }
    for value in &mut trace.values {
        if let Def::Op(node) = &mut value.def {
            *node = landed[*node as usize];
        }
    }
    trace.nodes = nodes;
    trace
}

fn norm_chain(rest: &[Node], values: &[crate::value::ValueDecl]) -> Option<(Node, usize)> {
    const WIDEST: u64 = 256 * 32;
    let [first, second, ..] = rest else {
        return None;
    };
    let Operation::Elementwise(Elementwise::Rmsnorm {
        x,
        weight,
        eps,
        y: t,
    }) = &first.op
    else {
        return None;
    };
    if first.guard != second.guard {
        return None;
    }
    let width = values.get(t.0 as usize).and_then(|decl| match &decl.ty {
        crate::value::Ty::Tensor { shape, .. } => match shape.last() {
            Some(crate::value::Dim::Const(width)) => Some(*width),
            _ => None,
        },
        crate::value::Ty::Struct(_) => None,
    });
    if !width.is_some_and(|width| width <= WIDEST) {
        return None;
    }
    let (x, weight, eps, t) = (*x, *weight, *eps, *t);
    match &second.op {
        Operation::Elementwise(Elementwise::ResidualAddRmsnorm {
            x: folded,
            y,
            y_out,
            weight: post_weight,
            plus_one,
            eps: post_eps,
            out,
        }) if *folded == t => Some((
            Node {
                op: Operation::Elementwise(Elementwise::RmsnormResidualAdd {
                    x,
                    weight,
                    eps,
                    t,
                    y: *y,
                    y_out: *y_out,
                    scale: None,
                    post: Some(PostNorm {
                        weight: *post_weight,
                        plus_one: *plus_one,
                        eps: *post_eps,
                        out: *out,
                    }),
                }),
                guard: second.guard.clone(),
                layer: second.layer,
            },
            2,
        )),
        Operation::Elementwise(Elementwise::ResidualAdd {
            x: folded,
            y,
            y_out,
        }) if *folded == t => {
            let (y, y_out) = (*y, *y_out);
            let mut took = 2;
            let mut layer = second.layer;
            let mut row = y_out;
            let scale = match rest.get(took) {
                Some(node) if node.guard == second.guard => match &node.op {
                    Operation::Elementwise(Elementwise::Scale {
                        s,
                        x: scaled_x,
                        x_out,
                    }) if *scaled_x == row => {
                        took += 1;
                        layer = node.layer;
                        row = *x_out;
                        Some((*s, *x_out))
                    }
                    _ => None,
                },
                _ => None,
            };
            let post = match rest.get(took) {
                Some(node) if node.guard == second.guard => match &node.op {
                    Operation::Elementwise(Elementwise::Rmsnorm {
                        x: normed,
                        weight,
                        eps,
                        y,
                    }) if *normed == row => {
                        took += 1;
                        layer = node.layer;
                        Some(PostNorm {
                            weight: *weight,
                            plus_one: false,
                            eps: *eps,
                            out: *y,
                        })
                    }
                    Operation::Elementwise(Elementwise::RmsnormPlusOne {
                        x: normed,
                        weight,
                        eps,
                        y,
                    }) if *normed == row => {
                        took += 1;
                        layer = node.layer;
                        Some(PostNorm {
                            weight: *weight,
                            plus_one: true,
                            eps: *eps,
                            out: *y,
                        })
                    }
                    _ => None,
                },
                _ => None,
            };
            Some((
                Node {
                    op: Operation::Elementwise(Elementwise::RmsnormResidualAdd {
                        x,
                        weight,
                        eps,
                        t,
                        y,
                        y_out,
                        scale,
                        post,
                    }),
                    guard: second.guard.clone(),
                    layer,
                },
                took,
            ))
        }
        _ => None,
    }
}

fn embed_chain(rest: &[Node]) -> Option<(Node, Option<Node>, usize)> {
    let (embed, scale_e, fold, scale_y, between) = match rest {
        [embed, scale_e, between, fold, scale_y, ..]
            if matches!(between.op, Operation::Layout(Layout::Select { .. })) =>
        {
            (embed, scale_e, fold, scale_y, Some(between))
        }
        [embed, scale_e, fold, scale_y, ..] => (embed, scale_e, fold, scale_y, None),
        _ => return None,
    };
    let Operation::Layout(Layout::Embed {
        ids,
        table,
        vocab,
        y: e,
    }) = &embed.op
    else {
        return None;
    };
    let Operation::Elementwise(Elementwise::MulScalar {
        s: embed_scale,
        x: scaled_x,
        x_out: e_scaled,
    }) = &scale_e.op
    else {
        return None;
    };
    let Operation::Elementwise(Elementwise::ResidualAdd {
        x: folded,
        y,
        y_out,
    }) = &fold.op
    else {
        return None;
    };
    let Operation::Elementwise(Elementwise::MulScalar {
        s: out_scale,
        x: scaled_y,
        x_out: y_scaled,
    }) = &scale_y.op
    else {
        return None;
    };
    let same_guard = [scale_e, fold, scale_y]
        .iter()
        .chain(between.iter())
        .all(|node| node.guard == embed.guard);
    if !same_guard || *scaled_x != *e || *folded != *e_scaled || *scaled_y != *y_out {
        return None;
    }
    if let Some(between) = between {
        let Operation::Layout(Layout::Select { y: sliced, .. }) = &between.op else {
            return None;
        };
        if *sliced != *y {
            return None;
        }
    }
    Some((
        Node {
            op: Operation::Elementwise(Elementwise::EmbedScaleAdd {
                ids: *ids,
                table: *table,
                vocab: *vocab,
                e: *e,
                embed_scale: *embed_scale,
                e_scaled: *e_scaled,
                y: *y,
                y_out: *y_out,
                out_scale: *out_scale,
                y_scaled: *y_scaled,
            }),
            guard: embed.guard.clone(),
            layer: scale_y.layer,
        },
        between.cloned(),
        if between.is_some() { 5 } else { 4 },
    ))
}

#[must_use]
pub fn gemm_epilogues(trace: Trace) -> Trace {
    fold_pairs(trace, epilogue_pair)
}

#[must_use]
pub fn q_norm_rope(trace: Trace) -> Trace {
    fold_pairs(trace, q_pair)
}

fn q_pair(nodes: &[Node], i: usize, values: &[ValueDecl]) -> Option<Node> {
    let (first, second) = (&nodes[i], &nodes[i + 1]);
    if first.guard != second.guard {
        return None;
    }
    let (
        Operation::Elementwise(Elementwise::RmsnormPerHead {
            x,
            weight,
            head_dim,
            eps,
            y,
        }),
        Operation::Elementwise(Elementwise::RopePartialQ {
            q,
            positions,
            rotary_dim,
            head_dim: rope_head_dim,
            theta,
            q_out,
        }),
    ) = (&first.op, &second.op)
    else {
        return None;
    };
    if q != y || rope_head_dim != head_dim || read_elsewhere(nodes, values, *y, i) {
        return None;
    }
    Some(Node {
        op: Operation::Elementwise(Elementwise::RmsnormRopePartialQ {
            x: *x,
            weight: *weight,
            head_dim: *head_dim,
            eps: *eps,
            positions: *positions,
            rotary_dim: *rotary_dim,
            theta: *theta,
            y: *y,
            q_out: *q_out,
        }),
        guard: first.guard.clone(),
        layer: second.layer.or(first.layer),
    })
}

#[must_use]
pub fn embed_select(trace: Trace) -> Trace {
    fold_pairs(trace, embed_select_pair)
}

fn embed_select_pair(nodes: &[Node], i: usize, values: &[ValueDecl]) -> Option<Node> {
    let (first, second) = (&nodes[i], &nodes[i + 1]);
    if first.guard != second.guard {
        return None;
    }
    let (
        Operation::Layout(Layout::Select {
            table: stacked,
            layer,
            width,
            y: copied,
        }),
        Operation::Elementwise(Elementwise::EmbedScaleAdd {
            ids,
            table,
            vocab,
            e,
            embed_scale,
            e_scaled,
            y,
            y_out,
            out_scale,
            y_scaled,
        }),
    ) = (&first.op, &second.op)
    else {
        return None;
    };
    if y != copied || read_elsewhere(nodes, values, *copied, i) {
        return None;
    }
    Some(Node {
        op: Operation::Elementwise(Elementwise::EmbedScaleAddSelect {
            ids: *ids,
            table: *table,
            vocab: *vocab,
            e: *e,
            embed_scale: *embed_scale,
            e_scaled: *e_scaled,
            stacked: *stacked,
            layer: *layer,
            width: *width,
            y_out: *y_out,
            out_scale: *out_scale,
            y_scaled: *y_scaled,
        }),
        guard: first.guard.clone(),
        layer: second.layer.or(first.layer),
    })
}

fn fold_pairs(mut trace: Trace, pair: fn(&[Node], usize, &[ValueDecl]) -> Option<Node>) -> Trace {
    let old = trace.nodes;
    let mut nodes = Vec::with_capacity(old.len());
    let mut landed = Vec::with_capacity(old.len());
    let mut i = 0usize;
    while i < old.len() {
        if i + 1 < old.len()
            && let Some(fused) = pair(&old, i, &trace.values)
        {
            let at = nodes.len() as u32;
            landed.extend([at, at]);
            nodes.push(fused);
            i += 2;
            continue;
        }
        landed.push(nodes.len() as u32);
        nodes.push(old[i].clone());
        i += 1;
    }
    for value in &mut trace.values {
        if let Def::Op(node) = &mut value.def {
            *node = landed[*node as usize];
        }
    }
    trace.nodes = nodes;
    trace
}

#[must_use]
pub fn modulation(mut trace: Trace) -> Trace {
    let mut nodes = Vec::with_capacity(trace.nodes.len());
    let mut landed = Vec::with_capacity(trace.nodes.len());
    let old = trace.nodes;
    let mut i = 0usize;
    while i < old.len() {
        if let Some((fused, took)) = gated_chain(&old[i..]).or_else(|| norm_modulate(&old[i..])) {
            let at = nodes.len() as u32;
            landed.extend(std::iter::repeat_n(at, took));
            nodes.push(fused);
            i += took;
            continue;
        }
        landed.push(nodes.len() as u32);
        nodes.push(old[i].clone());
        i += 1;
    }
    for value in &mut trace.values {
        if let Def::Op(node) = &mut value.def {
            *node = landed[*node as usize];
        }
    }
    trace.nodes = nodes;
    trace
}

fn scale_free_norm(node: &Node) -> Option<(ValueId, NormKind, ValueId)> {
    match &node.op {
        Operation::Elementwise(Elementwise::LayernormNoScale { x, eps, y }) => {
            Some((*x, NormKind::Layernorm { eps: *eps }, *y))
        }
        Operation::Elementwise(Elementwise::RmsnormNoScale {
            x,
            head_dim,
            eps,
            y,
        }) => Some((
            *x,
            NormKind::Rmsnorm {
                head_dim: *head_dim,
                eps: *eps,
            },
            *y,
        )),
        _ => None,
    }
}

fn norm_modulate(rest: &[Node]) -> Option<(Node, usize)> {
    let [first, second, ..] = rest else {
        return None;
    };
    if first.guard != second.guard {
        return None;
    }
    let (x, norm, normed) = scale_free_norm(first)?;
    let Operation::Elementwise(Elementwise::Modulate {
        x: modulated,
        m,
        lane_of_row,
        form,
        y,
    }) = &second.op
    else {
        return None;
    };
    if *modulated != normed {
        return None;
    }
    Some((
        Node {
            op: Operation::Elementwise(Elementwise::NormModulate {
                x,
                norm,
                normed,
                m: *m,
                lane_of_row: *lane_of_row,
                form: *form,
                y: *y,
            }),
            guard: second.guard.clone(),
            layer: second.layer.or(first.layer),
        },
        2,
    ))
}

fn gated_chain(rest: &[Node]) -> Option<(Node, usize)> {
    let [fold, norm_node, modulate, ..] = rest else {
        return None;
    };
    if fold.guard != norm_node.guard || fold.guard != modulate.guard {
        return None;
    }
    let Operation::Elementwise(Elementwise::GatedResidualAdd {
        r,
        g,
        y,
        lane_of_row,
        r_out,
    }) = &fold.op
    else {
        return None;
    };
    let (normed_x, norm, normed) = scale_free_norm(norm_node)?;
    if normed_x != *r_out {
        return None;
    }
    let Operation::Elementwise(Elementwise::Modulate {
        x: modulated,
        m,
        lane_of_row: modulate_lanes,
        form,
        y: out,
    }) = &modulate.op
    else {
        return None;
    };
    if *modulated != normed || modulate_lanes != lane_of_row {
        return None;
    }
    Some((
        Node {
            op: Operation::Elementwise(Elementwise::GatedResidualNormModulate {
                r: *r,
                g: *g,
                y: *y,
                lane_of_row: *lane_of_row,
                r_out: *r_out,
                norm,
                normed,
                m: *m,
                form: *form,
                out: *out,
            }),
            guard: modulate.guard.clone(),
            layer: modulate.layer.or(norm_node.layer).or(fold.layer),
        },
        3,
    ))
}

fn epilogue_pair(nodes: &[Node], i: usize, values: &[ValueDecl]) -> Option<Node> {
    let (first, second) = (&nodes[i], &nodes[i + 1]);
    if first.guard != second.guard {
        return None;
    }
    match (&first.op, &second.op) {
        (
            Operation::Linear(Linear::Matmul { act, w, y: packed }),
            Operation::Linear(Linear::MlpGegluTanhPacked {
                packed: read,
                intermediate,
                y,
            }),
        ) if read == packed && !read_elsewhere(nodes, values, *packed, i) => Some(Node {
            op: Operation::Linear(Linear::MatmulGeglu {
                act: *act,
                w: *w,
                intermediate: *intermediate,
                packed: *packed,
                y: *y,
            }),
            guard: first.guard.clone(),
            layer: second.layer.or(first.layer),
        }),
        (
            Operation::Linear(Linear::LmHead { act, w, y }),
            Operation::Attention(Attention::LogitSoftcap { x, cap, x_out }),
        ) if x == y && !read_elsewhere(nodes, values, *y, i) => Some(Node {
            op: Operation::Linear(Linear::LmHeadSoftcap {
                act: *act,
                w: *w,
                cap: *cap,
                y: *y,
                y_out: *x_out,
            }),
            guard: first.guard.clone(),
            layer: second.layer.or(first.layer),
        }),
        _ => None,
    }
}

fn read_elsewhere(nodes: &[Node], values: &[ValueDecl], value: ValueId, i: usize) -> bool {
    let mut ins = Vec::new();
    for (j, node) in nodes.iter().enumerate() {
        if j == i || j == i + 1 {
            continue;
        }
        ins.clear();
        node.op.inputs(&mut ins);
        if ins.contains(&value) {
            return true;
        }
    }
    values.iter().any(
        |decl| matches!(&decl.def, Def::Merge(arms) if arms.iter().any(|(arm, _)| *arm == value)),
    )
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
            drafter: None,
        }
    }

    #[test]
    fn fuse_every_case() {
        the_add_and_the_norm_that_reads_it_become_one_node();
        a_value_defined_past_the_pair_still_names_its_node();
        a_row_wider_than_the_launch_seats_stays_as_traced();
        norm_add_scale_norm_becomes_one_node_and_keeps_every_value();
        a_norm_before_the_pair_residual_norm_wrote_joins_it();
        a_norm_whose_fold_reads_something_else_stays_apart();
        the_per_layer_input_gather_and_its_fold_become_one_node();
        a_select_between_the_scale_and_the_fold_moves_ahead_of_the_gather();
        a_norm_of_something_else_stays_apart_and_a_layer_boundary_does_not();
        a_matmul_and_the_geglu_over_it_fold_unless_the_packed_output_has_another_reader();
        a_per_head_norm_and_the_q_rope_over_it_fold();
        a_select_and_the_embed_fold_over_its_copy_fold_unless_the_copy_is_read_again();
        an_lm_head_and_the_softcap_over_it_fold_and_keep_the_alias();
        a_scale_free_norm_and_the_modulate_over_it_become_one_node();
        the_gated_fold_its_norm_and_the_modulate_become_one_two_output_node();
        a_gate_per_lane_under_a_modulate_per_token_stays_apart();
    }

    fn the_add_and_the_norm_that_reads_it_become_one_node() {
        let fused = residual_norm(trace_of(vec![
            node(add(3), Some(0)),
            node(norm(3), Some(0)),
        ]));
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

    fn a_value_defined_past_the_pair_still_names_its_node() {
        use crate::value::{Ty, ValueDecl};
        let mut trace = trace_of(vec![
            node(add(3), Some(0)),
            node(norm(3), Some(0)),
            node(add(7), Some(1)),
        ]);
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

    fn rms(x: u32, w: u32, y: u32) -> Elementwise {
        Elementwise::Rmsnorm {
            x: ValueId(x),
            weight: ValueId(w),
            eps: 1e-6,
            y: ValueId(y),
        }
    }

    fn rows(n: u32, width: u64) -> Vec<crate::value::ValueDecl> {
        use crate::value::{Dim, Ty, ValueDecl};
        (0..n)
            .map(|_| ValueDecl {
                def: Def::Op(0),
                ty: Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(width)],
                    dtype: dtype::Dtype::Bf16,
                },
            })
            .collect()
    }

    fn a_row_wider_than_the_launch_seats_stays_as_traced() {
        let mut trace = trace_of(vec![node(rms(10, 1, 11), Some(0)), node(add(3), Some(0))]);
        trace.values = rows(16, 16384);
        let mut chain = trace.clone();
        chain.nodes[1] = node(
            Elementwise::ResidualAdd {
                x: ValueId(11),
                y: ValueId(12),
                y_out: ValueId(13),
            },
            Some(0),
        );
        assert_eq!(residual_chains(chain).nodes.len(), 2);
    }

    fn norm_add_scale_norm_becomes_one_node_and_keeps_every_value() {
        let mut trace = trace_of(vec![
            node(rms(10, 1, 11), Some(3)),
            node(
                Elementwise::ResidualAdd {
                    x: ValueId(11),
                    y: ValueId(12),
                    y_out: ValueId(13),
                },
                Some(3),
            ),
            node(
                Elementwise::Scale {
                    s: ValueId(20),
                    x: ValueId(13),
                    x_out: ValueId(14),
                },
                Some(3),
            ),
            node(rms(14, 2, 15), Some(4)),
        ]);
        trace.values = rows(21, 2560);
        let fused = residual_chains(trace);
        assert_eq!(fused.nodes.len(), 1);
        assert_eq!(
            fused.nodes[0].layer,
            Some(4),
            "the last weight read names the layer"
        );
        let Operation::Elementwise(Elementwise::RmsnormResidualAdd {
            t,
            y_out,
            scale,
            post,
            ..
        }) = &fused.nodes[0].op
        else {
            panic!("the chain did not fuse: {:?}", fused.nodes[0].op);
        };
        assert_eq!((*t, *y_out), (ValueId(11), ValueId(13)));
        assert_eq!(*scale, Some((ValueId(20), ValueId(14))));
        assert_eq!(
            post.as_ref().map(|p| (p.out, p.plus_one)),
            Some((ValueId(15), false))
        );
        use crate::operands::Operands;
        let mut outs = Vec::new();
        fused.nodes[0].op.outputs(&mut outs);
        assert_eq!(
            outs,
            vec![ValueId(11), ValueId(13), ValueId(14), ValueId(15)]
        );
    }

    fn a_norm_before_the_pair_residual_norm_wrote_joins_it() {
        let mut trace = residual_norm(trace_of(vec![
            node(rms(10, 1, 1), Some(0)),
            node(add(3), Some(0)),
            node(norm(3), Some(0)),
        ]));
        trace.values = rows(12, 2560);
        assert_eq!(trace.nodes.len(), 2);
        let fused = residual_chains(trace);
        assert_eq!(fused.nodes.len(), 1);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::RmsnormResidualAdd {
                t: ValueId(1),
                scale: None,
                post: Some(PostNorm {
                    plus_one: true,
                    out: ValueId(5),
                    ..
                }),
                ..
            })
        ));
    }

    fn a_norm_whose_fold_reads_something_else_stays_apart() {
        let mut trace = trace_of(vec![node(rms(10, 1, 11), Some(0)), node(add(3), Some(0))]);
        trace.values = rows(12, 2560);
        assert_eq!(residual_chains(trace).nodes.len(), 2);
    }

    fn the_per_layer_input_gather_and_its_fold_become_one_node() {
        let trace = trace_of(vec![
            Node {
                op: Operation::Layout(Layout::Embed {
                    ids: ValueId(0),
                    table: ValueId(1),
                    vocab: 8,
                    y: ValueId(2),
                }),
                guard: Guard::Always,
                layer: Some(0),
            },
            node(
                Elementwise::MulScalar {
                    s: 16.0,
                    x: ValueId(2),
                    x_out: ValueId(3),
                },
                Some(0),
            ),
            node(
                Elementwise::ResidualAdd {
                    x: ValueId(3),
                    y: ValueId(4),
                    y_out: ValueId(5),
                },
                Some(0),
            ),
            node(
                Elementwise::MulScalar {
                    s: 0.5,
                    x: ValueId(5),
                    x_out: ValueId(6),
                },
                Some(0),
            ),
            node(add(9), Some(1)),
        ]);
        let fused = residual_chains(trace);
        assert_eq!(fused.nodes.len(), 2);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::EmbedScaleAdd {
                e: ValueId(2),
                e_scaled: ValueId(3),
                y_out: ValueId(5),
                y_scaled: ValueId(6),
                ..
            })
        ));
    }

    fn a_select_between_the_scale_and_the_fold_moves_ahead_of_the_gather() {
        use crate::value::{Ty, ValueDecl};
        let mut trace = trace_of(vec![
            Node {
                op: Operation::Layout(Layout::Embed {
                    ids: ValueId(0),
                    table: ValueId(1),
                    vocab: 8,
                    y: ValueId(2),
                }),
                guard: Guard::Always,
                layer: Some(0),
            },
            node(
                Elementwise::MulScalar {
                    s: 16.0,
                    x: ValueId(2),
                    x_out: ValueId(3),
                },
                Some(0),
            ),
            Node {
                op: Operation::Layout(Layout::Select {
                    table: ValueId(7),
                    layer: 0,
                    width: 4,
                    y: ValueId(4),
                }),
                guard: Guard::Always,
                layer: Some(0),
            },
            node(
                Elementwise::ResidualAdd {
                    x: ValueId(3),
                    y: ValueId(4),
                    y_out: ValueId(5),
                },
                Some(0),
            ),
            node(
                Elementwise::MulScalar {
                    s: 0.5,
                    x: ValueId(5),
                    x_out: ValueId(6),
                },
                Some(0),
            ),
        ]);
        let decl = |node: u32| ValueDecl {
            def: Def::Op(node),
            ty: Ty::Tensor {
                shape: Vec::new(),
                dtype: dtype::Dtype::Bf16,
            },
        };
        trace.values = vec![
            decl(0),
            decl(0),
            decl(0),
            decl(1),
            decl(2),
            decl(3),
            decl(4),
        ];
        let fused = residual_chains(trace);
        assert_eq!(fused.nodes.len(), 2);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Layout(Layout::Select { .. })
        ));
        assert!(matches!(
            fused.nodes[1].op,
            Operation::Elementwise(Elementwise::EmbedScaleAdd { .. })
        ));
        assert!(
            matches!(fused.values[4].def, Def::Op(0)),
            "the select's value still names it"
        );
        assert!(matches!(fused.values[6].def, Def::Op(1)));
        assert!(matches!(fused.values[2].def, Def::Op(1)));
    }

    fn a_norm_of_something_else_stays_apart_and_a_layer_boundary_does_not() {
        let other = residual_norm(trace_of(vec![
            node(add(3), Some(0)),
            node(norm(9), Some(0)),
        ]));
        assert_eq!(other.nodes.len(), 2);
        let layer = residual_norm(trace_of(vec![
            node(add(3), Some(0)),
            node(norm(3), Some(1)),
        ]));
        assert_eq!(layer.nodes.len(), 1);
        assert_eq!(layer.nodes[0].layer, Some(1));
    }
    fn matmul(act: u32, w: u32, y: u32) -> Node {
        Node {
            op: Operation::Linear(Linear::Matmul {
                act: ValueId(act),
                w: ValueId(w),
                y: ValueId(y),
            }),
            guard: Guard::Always,
            layer: Some(3),
        }
    }

    fn geglu(packed: u32, y: u32) -> Node {
        Node {
            op: Operation::Linear(Linear::MlpGegluTanhPacked {
                packed: ValueId(packed),
                intermediate: 8,
                y: ValueId(y),
            }),
            guard: Guard::Always,
            layer: None,
        }
    }

    fn a_matmul_and_the_geglu_over_it_fold_unless_the_packed_output_has_another_reader() {
        let fused = gemm_epilogues(trace_of(vec![matmul(0, 1, 2), geglu(2, 3)]));
        assert_eq!(fused.nodes.len(), 1);
        let Operation::Linear(Linear::MatmulGeglu {
            act,
            w,
            intermediate,
            packed,
            y,
        }) = &fused.nodes[0].op
        else {
            panic!("the pair fused into {:?}", fused.nodes[0].op);
        };
        assert_eq!(
            (*act, *w, *intermediate, *packed, *y),
            (ValueId(0), ValueId(1), 8, ValueId(2), ValueId(3))
        );
        assert_eq!(
            fused.nodes[0].layer,
            Some(3),
            "the fused node keeps the weight's layer"
        );

        let mut outs = Vec::new();
        fused.nodes[0].op.outputs(&mut outs);
        assert_eq!(
            outs,
            vec![ValueId(2), ValueId(3)],
            "the packed value is still the node's"
        );

        let apart = gemm_epilogues(trace_of(vec![matmul(0, 1, 2), geglu(2, 3), geglu(2, 4)]));
        assert_eq!(
            apart.nodes.len(),
            3,
            "a second reader of the packed output keeps the matmul"
        );
    }

    fn a_per_head_norm_and_the_q_rope_over_it_fold() {
        let norm = |y: u32| Node {
            op: Operation::Elementwise(Elementwise::RmsnormPerHead {
                x: ValueId(0),
                weight: ValueId(1),
                head_dim: 256,
                eps: 1e-6,
                y: ValueId(y),
            }),
            guard: Guard::Always,
            layer: Some(7),
        };
        let rope = |head_dim: u32| Node {
            op: Operation::Elementwise(Elementwise::RopePartialQ {
                q: ValueId(2),
                positions: ValueId(3),
                rotary_dim: 128,
                head_dim,
                theta: 1e6,
                q_out: ValueId(4),
            }),
            guard: Guard::Always,
            layer: None,
        };
        let fused = q_norm_rope(trace_of(vec![norm(2), rope(256)]));
        assert_eq!(fused.nodes.len(), 1);
        let mut aliases = Vec::new();
        fused.nodes[0].op.aliases(&mut aliases);
        assert_eq!(aliases, vec![(ValueId(4), ValueId(2))]);
        assert_eq!(fused.nodes[0].layer, Some(7));
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::RmsnormRopePartialQ {
                rotary_dim: 128,
                ..
            })
        ));
        assert_eq!(
            q_norm_rope(trace_of(vec![norm(2), rope(128)])).nodes.len(),
            2
        );
    }

    fn a_select_and_the_embed_fold_over_its_copy_fold_unless_the_copy_is_read_again() {
        let select = Node {
            op: Operation::Layout(Layout::Select {
                table: ValueId(10),
                layer: 3,
                width: 256,
                y: ValueId(11),
            }),
            guard: Guard::Always,
            layer: None,
        };
        let fold = |y: u32| Node {
            op: Operation::Elementwise(Elementwise::EmbedScaleAdd {
                ids: ValueId(0),
                table: ValueId(1),
                vocab: 64,
                e: ValueId(2),
                embed_scale: 2.0,
                e_scaled: ValueId(3),
                y: ValueId(y),
                y_out: ValueId(12),
                out_scale: 0.5,
                y_scaled: ValueId(13),
            }),
            guard: Guard::Always,
            layer: Some(3),
        };
        let fused = embed_select(trace_of(vec![select.clone(), fold(11)]));
        assert_eq!(fused.nodes.len(), 1);
        let Operation::Elementwise(Elementwise::EmbedScaleAddSelect {
            stacked,
            layer,
            width,
            y_out,
            ..
        }) = &fused.nodes[0].op
        else {
            panic!("the pair fused into {:?}", fused.nodes[0].op);
        };
        assert_eq!(
            (*stacked, *layer, *width, *y_out),
            (ValueId(10), 3, 256, ValueId(12))
        );
        let mut aliases = Vec::new();
        fused.nodes[0].op.aliases(&mut aliases);
        assert!(aliases.is_empty(), "the folded row is a fresh output");

        let reader = Node {
            op: Operation::Elementwise(Elementwise::MulScalar {
                s: 1.0,
                x: ValueId(11),
                x_out: ValueId(14),
            }),
            guard: Guard::Always,
            layer: None,
        };
        assert_eq!(
            embed_select(trace_of(vec![select, fold(11), reader]))
                .nodes
                .len(),
            3
        );
    }

    fn an_lm_head_and_the_softcap_over_it_fold_and_keep_the_alias() {
        let head = Node {
            op: Operation::Linear(Linear::LmHead {
                act: ValueId(0),
                w: ValueId(1),
                y: ValueId(2),
            }),
            guard: Guard::Always,
            layer: None,
        };
        let softcap = Node {
            op: Operation::Attention(Attention::LogitSoftcap {
                x: ValueId(2),
                cap: 30.0,
                x_out: ValueId(3),
            }),
            guard: Guard::Always,
            layer: None,
        };
        let fused = gemm_epilogues(trace_of(vec![head, softcap]));
        assert_eq!(fused.nodes.len(), 1);
        let mut aliases = Vec::new();
        fused.nodes[0].op.aliases(&mut aliases);
        assert_eq!(aliases, vec![(ValueId(3), ValueId(2))]);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Linear(Linear::LmHeadSoftcap { cap, .. }) if (cap - 30.0).abs() < f32::EPSILON
        ));
    }

    use crate::ops::elemwise::ModulateForm;

    fn gated_add(r: u32, g: u32, y: u32, lanes: Option<u32>, r_out: u32) -> Elementwise {
        Elementwise::GatedResidualAdd {
            r: ValueId(r),
            g: ValueId(g),
            y: ValueId(y),
            lane_of_row: lanes.map(ValueId),
            r_out: ValueId(r_out),
        }
    }

    fn ln(x: u32, y: u32) -> Elementwise {
        Elementwise::LayernormNoScale {
            x: ValueId(x),
            eps: 1e-6,
            y: ValueId(y),
        }
    }

    fn modulate(x: u32, m: u32, lanes: Option<u32>, y: u32) -> Elementwise {
        Elementwise::Modulate {
            x: ValueId(x),
            m: ValueId(m),
            lane_of_row: lanes.map(ValueId),
            form: ModulateForm::ScaleShift,
            y: ValueId(y),
        }
    }

    fn a_scale_free_norm_and_the_modulate_over_it_become_one_node() {
        let fused = modulation(trace_of(vec![
            node(ln(1, 2), Some(0)),
            node(modulate(2, 3, Some(4), 5), Some(0)),
        ]));
        assert_eq!(fused.nodes.len(), 1);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::NormModulate {
                x: ValueId(1),
                normed: ValueId(2),
                m: ValueId(3),
                lane_of_row: Some(ValueId(4)),
                y: ValueId(5),
                norm: NormKind::Layernorm { .. },
                ..
            })
        ));
        let rms = Elementwise::RmsnormNoScale {
            x: ValueId(1),
            head_dim: 8,
            eps: 1e-6,
            y: ValueId(2),
        };
        let fused = modulation(trace_of(vec![
            node(rms.clone(), Some(0)),
            node(modulate(2, 3, None, 5), Some(0)),
        ]));
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::NormModulate {
                norm: NormKind::Rmsnorm { head_dim: 8, .. },
                lane_of_row: None,
                ..
            })
        ));
        let apart = modulation(trace_of(vec![
            node(rms, Some(0)),
            node(modulate(9, 3, None, 5), Some(0)),
        ]));
        assert_eq!(apart.nodes.len(), 2);
    }

    fn the_gated_fold_its_norm_and_the_modulate_become_one_two_output_node() {
        let mut trace = trace_of(vec![
            node(gated_add(1, 2, 3, Some(4), 5), Some(0)),
            node(ln(5, 6), Some(0)),
            node(modulate(6, 7, Some(4), 8), Some(0)),
            node(add(9), Some(1)),
        ]);
        trace.values = rows(12, 64);
        for (value, at) in [(5u32, 0u32), (6, 1), (8, 2), (9, 3)] {
            trace.values[value as usize].def = Def::Op(at);
        }
        let fused = modulation(trace);
        assert_eq!(
            fused.nodes.len(),
            2,
            "three nodes became one, the fourth stood"
        );
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::GatedResidualNormModulate {
                r: ValueId(1),
                g: ValueId(2),
                y: ValueId(3),
                lane_of_row: Some(ValueId(4)),
                r_out: ValueId(5),
                normed: ValueId(6),
                m: ValueId(7),
                out: ValueId(8),
                ..
            })
        ));
        let mut outs = Vec::new();
        fused.nodes[0].op.outputs(&mut outs);
        assert_eq!(outs, vec![ValueId(5), ValueId(6), ValueId(8)]);
        let mut pairs = Vec::new();
        fused.nodes[0].op.aliases(&mut pairs);
        assert_eq!(
            pairs,
            vec![(ValueId(5), ValueId(1))],
            "only the fold is in place"
        );
        for value in [5usize, 6, 8] {
            assert!(matches!(fused.values[value].def, Def::Op(0)));
        }
        assert!(matches!(fused.values[9].def, Def::Op(1)));
    }

    fn a_gate_per_lane_under_a_modulate_per_token_stays_apart() {
        let fused = modulation(trace_of(vec![
            node(gated_add(1, 2, 3, Some(4), 5), Some(0)),
            node(ln(5, 6), Some(0)),
            node(modulate(6, 7, None, 8), Some(0)),
        ]));
        assert_eq!(fused.nodes.len(), 2);
        assert!(matches!(
            fused.nodes[0].op,
            Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
        ));
        assert!(matches!(
            fused.nodes[1].op,
            Operation::Elementwise(Elementwise::NormModulate { .. })
        ));
    }
}
