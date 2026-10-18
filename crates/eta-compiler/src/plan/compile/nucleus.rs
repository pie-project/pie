use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use eta_ir::op::Op;
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId};

use super::normalize::{NodeIndex, NormalizedStage};
use super::region::{LibraryOp, StageIndex};
use super::symbolic::{Dimension, symbolic_dims_match_expected, symbolic_shape_matches_static};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LibraryMatch {
    pub(crate) library: LibraryOp,
    pub(crate) nodes: Vec<NodeIndex>,
    pub(crate) inputs: Vec<ValueId>,
    pub(crate) outputs: Vec<ValueId>,
}
pub(crate) fn recognize_library_dataflows(
    stage: &NormalizedStage,
    index: &StageIndex,
) -> Vec<LibraryMatch> {
    let mut claimed = BTreeSet::new();
    let mut matches = Vec::new();
    for final_node in (0..stage.ops.len() as u32).map(NodeIndex) {
        let Some(candidate) = match_nucleus_dataflow(stage, final_node, index) else {
            continue;
        };
        if candidate.nodes.iter().any(|node| claimed.contains(node)) {
            continue;
        }
        claimed.extend(candidate.nodes.iter().copied());
        matches.push(candidate);
    }
    // A top-p or top-k cut outside a sampler chain, over f32 rows a
    // vocabulary wide: the lane cut (its row on several blocks).
    for node in (0..stage.ops.len() as u32).map(NodeIndex) {
        if claimed.contains(&node) {
            continue;
        }
        let Some(Op::PivotThreshold { input, predicate }) = stage.ops.get(node.index()) else {
            continue;
        };
        let threshold = match *predicate {
            Predicate::CummassLe(value) | Predicate::RankLe(value) => value,
            Predicate::ProbGe(_) => continue,
        };
        let Some(input_type) = stage.value_types.get(*input as usize) else {
            continue;
        };
        let wide = matches!(
            input_type.dims.last(),
            Some(Dimension::Static(width)) if *width >= LANE_CUT_WIDTH
        );
        if input_type.dtype != Dtype::F32 || !wide || !(1..=2).contains(&input_type.rank()) {
            continue;
        }
        let Some(output) = index.base(node) else {
            continue;
        };
        claimed.insert(node);
        matches.push(LibraryMatch {
            library: LibraryOp::LaneCut,
            nodes: vec![node],
            inputs: vec![*input, threshold],
            outputs: vec![output],
        });
    }
    matches
}

/// The narrowest row a lane cut takes: below it one block walks the row as
/// fast as a lane's blocks meet.
const LANE_CUT_WIDTH: u32 = 1024;

fn match_nucleus_dataflow(
    stage: &NormalizedStage,
    final_node: NodeIndex,
    index: &StageIndex,
) -> Option<LibraryMatch> {
    let Op::ReduceArgmax(perturbed) = stage.ops.get(final_node.index())? else {
        return None;
    };
    let add_node = index.producer(*perturbed)?;
    let Op::Add(left, right) = stage.ops.get(add_node.index())? else {
        return None;
    };

    let (perturbed, left, right) = (*perturbed, *left, *right);
    match_nucleus_add_order(stage, final_node, perturbed, left, right, index)
        .or_else(|| match_nucleus_add_order(stage, final_node, perturbed, right, left, index))
}

struct Softmax {
    maximum_node: NodeIndex,
    maximum_broadcast_node: NodeIndex,
    centered_node: NodeIndex,
    exponential_node: NodeIndex,
    sum_node: NodeIndex,
    sum_broadcast_node: NodeIndex,
    div_node: NodeIndex,
    maximum: ValueId,
    maximum_broadcast: ValueId,
    centered: ValueId,
    exponentials: ValueId,
    sum: ValueId,
    sum_broadcast: ValueId,
    shape: Shape,
}

fn match_softmax(
    stage: &NormalizedStage,
    probabilities: ValueId,
    logits: ValueId,
    index: &StageIndex,
) -> Option<Softmax> {
    let div_node = index.producer(probabilities)?;
    let Op::Div(exponentials, sum_broadcast) = stage.ops.get(div_node.index())? else {
        return None;
    };
    let (exponentials, sum_broadcast) = (*exponentials, *sum_broadcast);

    let exponential_node = index.producer(exponentials)?;
    let Op::Exp(centered) = stage.ops.get(exponential_node.index())? else {
        return None;
    };
    let centered = *centered;

    let centered_node = index.producer(centered)?;
    let Op::Sub(centered_logits, maximum_broadcast) = stage.ops.get(centered_node.index())? else {
        return None;
    };
    if *centered_logits != logits {
        return None;
    }
    let maximum_broadcast = *maximum_broadcast;

    let maximum_broadcast_node = index.producer(maximum_broadcast)?;
    let Op::Broadcast {
        value: maximum,
        shape,
    } = stage.ops.get(maximum_broadcast_node.index())?
    else {
        return None;
    };
    let (maximum, shape) = (*maximum, *shape);

    let maximum_node = index.producer(maximum)?;
    let Op::ReduceMax(maximum_logits) = stage.ops.get(maximum_node.index())? else {
        return None;
    };
    if *maximum_logits != logits {
        return None;
    }

    let sum_broadcast_node = index.producer(sum_broadcast)?;
    let Op::Broadcast {
        value: sum,
        shape: sum_shape,
    } = stage.ops.get(sum_broadcast_node.index())?
    else {
        return None;
    };
    let (sum, sum_shape) = (*sum, *sum_shape);

    let sum_node = index.producer(sum)?;
    let Op::ReduceSum(sum_exponentials) = stage.ops.get(sum_node.index())? else {
        return None;
    };
    if *sum_exponentials != exponentials || shape != sum_shape {
        return None;
    }

    Some(Softmax {
        maximum_node,
        maximum_broadcast_node,
        centered_node,
        exponential_node,
        sum_node,
        sum_broadcast_node,
        div_node,
        maximum,
        maximum_broadcast,
        centered,
        exponentials,
        sum,
        sum_broadcast,
        shape,
    })
}

struct Chain {
    softmax: Softmax,
    pivot_node: NodeIndex,
    negative_infinity_node: NodeIndex,
    select_node: NodeIndex,
    rng_node: NodeIndex,
    add_node: NodeIndex,
    final_node: NodeIndex,
    logits: ValueId,
    probabilities: ValueId,
    keep: ValueId,
    negative_infinity: ValueId,
    masked: ValueId,
    noise: ValueId,
    perturbed: ValueId,
    token: ValueId,
    top_p: ValueId,
    rng_state: ValueId,
    rng_shape: Shape,
    /// The cut is top-k (`RankLe`), not top-p (`CummassLe`).
    rank: bool,
}

impl Chain {
    fn nodes(&self) -> [NodeIndex; 13] {
        let s = &self.softmax;
        [
            s.maximum_node,
            s.maximum_broadcast_node,
            s.centered_node,
            s.exponential_node,
            s.sum_node,
            s.sum_broadcast_node,
            s.div_node,
            self.pivot_node,
            self.negative_infinity_node,
            self.select_node,
            self.rng_node,
            self.add_node,
            self.final_node,
        ]
    }

    fn expected_consumers(&self) -> [(ValueId, Vec<NodeIndex>); 12] {
        let s = &self.softmax;
        [
            (s.maximum, vec![s.maximum_broadcast_node]),
            (s.maximum_broadcast, vec![s.centered_node]),
            (s.centered, vec![s.exponential_node]),
            (s.exponentials, vec![s.sum_node, s.div_node]),
            (s.sum, vec![s.sum_broadcast_node]),
            (s.sum_broadcast, vec![s.div_node]),
            (self.probabilities, vec![self.pivot_node]),
            (self.keep, vec![self.select_node]),
            (self.negative_infinity, vec![self.select_node]),
            (self.masked, vec![self.add_node]),
            (self.noise, vec![self.add_node]),
            (self.perturbed, vec![self.final_node]),
        ]
    }
}

fn match_chain(
    stage: &NormalizedStage,
    final_node: NodeIndex,
    perturbed: ValueId,
    masked: ValueId,
    noise: ValueId,
    index: &StageIndex,
) -> Option<Chain> {
    let add_node = index.producer(perturbed)?;

    let select_node = index.producer(masked)?;
    let Op::Select {
        cond: keep,
        a: logits,
        b: negative_infinity,
    } = stage.ops.get(select_node.index())?
    else {
        return None;
    };
    let (keep, logits, negative_infinity) = (*keep, *logits, *negative_infinity);

    let negative_infinity_node = index.producer(negative_infinity)?;
    let Op::Const(Literal::F32(value)) = stage.ops.get(negative_infinity_node.index())? else {
        return None;
    };
    if value.to_bits() != f32::NEG_INFINITY.to_bits() {
        return None;
    }

    let rng_node = index.producer(noise)?;
    let Op::RngKeyed {
        state: rng_state,
        shape: rng_shape,
        kind: RngKind::Gumbel,
    } = stage.ops.get(rng_node.index())?
    else {
        return None;
    };
    let (rng_state, rng_shape) = (*rng_state, *rng_shape);

    let pivot_node = index.producer(keep)?;
    let Op::PivotThreshold {
        input: probabilities,
        predicate,
    } = stage.ops.get(pivot_node.index())?
    else {
        return None;
    };
    let (top_p, rank) = match *predicate {
        Predicate::CummassLe(top_p) => (top_p, false),
        Predicate::RankLe(k) => (k, true),
        Predicate::ProbGe(_) => return None,
    };
    let probabilities = *probabilities;

    let softmax = match_softmax(stage, probabilities, logits, index)?;
    if softmax.shape != rng_shape {
        return None;
    }

    Some(Chain {
        softmax,
        pivot_node,
        negative_infinity_node,
        select_node,
        rng_node,
        add_node,
        final_node,
        logits,
        probabilities,
        keep,
        negative_infinity,
        masked,
        noise,
        perturbed,
        token: index.base(final_node)?,
        top_p,
        rng_state,
        rng_shape,
        rank,
    })
}

struct LibraryInputs {
    values: Vec<ValueId>,
    scaled: Option<(ValueId, ValueId)>,
}

fn nucleus_library_inputs(
    stage: &NormalizedStage,
    chain: &Chain,
    index: &StageIndex,
) -> Option<LibraryInputs> {
    let plain = LibraryInputs {
        values: vec![chain.logits, chain.top_p, chain.rng_state],
        scaled: None,
    };
    let Some(scale_node) = index.producer(chain.logits) else {
        return Some(plain);
    };
    let Some(Op::Div(raw_logits, divisor)) = stage.ops.get(scale_node.index()) else {
        return Some(plain);
    };
    let mut actual = index.consumers(chain.logits)?.to_vec();
    actual.sort_unstable();
    let mut expected = vec![
        chain.softmax.maximum_node,
        chain.softmax.centered_node,
        chain.select_node,
    ];
    expected.sort_unstable();
    if actual != expected {
        return Some(plain);
    }
    let mut library_logits = *raw_logits;
    if let Some(reshape_node) = index.producer(*raw_logits)
        && let Some(Op::Reshape { value, .. }) = stage.ops.get(reshape_node.index())
    {
        library_logits = *value;
    }
    Some(LibraryInputs {
        values: vec![
            library_logits,
            *divisor,
            chain.logits,
            chain.top_p,
            chain.rng_state,
        ],
        scaled: Some((library_logits, *divisor)),
    })
}

fn chain_is_exclusive(
    chain: &Chain,
    library_inputs: &[ValueId],
    index: &StageIndex,
) -> Option<Vec<NodeIndex>> {
    let nodes = chain.nodes();
    let mut ordered_nodes = nodes.to_vec();
    ordered_nodes.sort_unstable();
    ordered_nodes.dedup();
    if ordered_nodes.len() != nodes.len() {
        return None;
    }
    let node_set: BTreeSet<NodeIndex> = ordered_nodes.iter().copied().collect();
    if library_inputs.iter().copied().any(|input| {
        index
            .producer(input)
            .is_some_and(|node| node_set.contains(&node))
    }) {
        return None;
    }
    for (value, mut expected) in chain.expected_consumers() {
        let mut actual = index.consumers(value)?.to_vec();
        actual.sort_unstable();
        expected.sort_unstable();
        if actual != expected {
            return None;
        }
    }
    if index
        .consumers(chain.token)?
        .iter()
        .all(|consumer| node_set.contains(consumer))
    {
        return None;
    }
    Some(ordered_nodes)
}

fn chain_types_agree(
    stage: &NormalizedStage,
    chain: &Chain,
    scaled_input: Option<(ValueId, ValueId)>,
) -> Option<()> {
    let value_type = |value: ValueId| stage.value_types.get(value as usize);
    let logits_type = value_type(chain.logits)?;
    if logits_type.dtype != Dtype::F32
        || !(1..=2).contains(&logits_type.rank())
        || !symbolic_shape_matches_static(logits_type, chain.softmax.shape)
    {
        return None;
    }
    let row_dims = &logits_type.dims[..logits_type.dims.len() - 1];
    if let Some((raw_logits, divisor)) = scaled_input {
        let raw_type = value_type(raw_logits)?;
        if raw_type.dtype != Dtype::F32
            || !(1..=2).contains(&raw_type.rank())
            || raw_type.dims.last() != logits_type.dims.last()
        {
            return None;
        }
        let divisor_type = value_type(divisor)?;
        if divisor_type.dtype != Dtype::F32
            || (!divisor_type.is_scalar() && divisor_type.dims.as_slice() != row_dims)
        {
            return None;
        }
    }
    let top_p_type = value_type(chain.top_p)?;
    let threshold_ok = if chain.rank {
        matches!(top_p_type.dtype, Dtype::I32 | Dtype::U32)
    } else {
        top_p_type.dtype == Dtype::F32
    };
    if !threshold_ok
        || (!top_p_type.is_scalar()
            && !symbolic_dims_match_expected(
                &top_p_type.dims,
                row_dims,
                &chain.softmax.shape.dims()[..chain.softmax.shape.rank() - 1],
            ))
    {
        return None;
    }
    let rng_state_type = value_type(chain.rng_state)?;
    if rng_state_type.dtype != Dtype::U32
        || rng_state_type.dims.as_slice() != [Dimension::Static(2)]
    {
        return None;
    }
    let token_type = value_type(chain.token)?;
    if token_type.dtype != Dtype::I32 || token_type.dims.as_slice() != row_dims {
        return None;
    }

    for value in [
        chain.softmax.maximum_broadcast,
        chain.softmax.centered,
        chain.softmax.exponentials,
        chain.softmax.sum_broadcast,
        chain.probabilities,
        chain.masked,
        chain.perturbed,
    ] {
        if value_type(value)? != logits_type {
            return None;
        }
    }
    let noise_type = value_type(chain.noise)?;
    if noise_type.dtype != Dtype::F32 || !symbolic_shape_matches_static(noise_type, chain.rng_shape)
    {
        return None;
    }
    for value in [chain.softmax.maximum, chain.softmax.sum] {
        let ty = value_type(value)?;
        if ty.dtype != Dtype::F32 || ty.dims.as_slice() != row_dims {
            return None;
        }
    }
    let keep_type = value_type(chain.keep)?;
    if keep_type.dtype != Dtype::Bool || keep_type.dims != logits_type.dims {
        return None;
    }
    let negative_infinity_type = value_type(chain.negative_infinity)?;
    if negative_infinity_type.dtype != Dtype::F32 || !negative_infinity_type.dims.is_empty() {
        return None;
    }
    Some(())
}

fn match_nucleus_add_order(
    stage: &NormalizedStage,
    final_node: NodeIndex,
    perturbed: ValueId,
    masked: ValueId,
    noise: ValueId,
    index: &StageIndex,
) -> Option<LibraryMatch> {
    let chain = match_chain(stage, final_node, perturbed, masked, noise, index)?;
    let inputs = nucleus_library_inputs(stage, &chain, index)?;
    let nodes = chain_is_exclusive(&chain, &inputs.values, index)?;
    chain_types_agree(stage, &chain, inputs.scaled)?;

    Some(LibraryMatch {
        library: if chain.rank {
            LibraryOp::RankSample
        } else {
            LibraryOp::NucleusSample
        },
        nodes,
        inputs: inputs.values,
        outputs: vec![chain.token],
    })
}
