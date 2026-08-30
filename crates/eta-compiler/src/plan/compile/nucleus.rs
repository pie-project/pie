//! Recognizing the nucleus-sampling dataflow.
//!
//! One question: does this run of normalized ops spell the `softmax -> top-p
//! mask -> Gumbel noise -> argmax` pipeline that a backend can hand to a
//! single library call? Everything here exists to answer it, and the answer
//! is a [`LibraryMatch`]; [`super::region`] decides what to do with one.
//!
//! The match is deliberately narrow. Four independent conditions have to hold
//! -- the ops are the right ops, the library's inputs are recoverable, the
//! chain is used by nothing outside itself, and the types line up -- and each
//! is a separate function that returns `None` rather than one predicate that
//! returns a reason. Refusing to match is always correct: the ops then go
//! down the generated path and produce the same numbers more slowly. Matching
//! wrongly is not, so every condition is checked, never inferred.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use eta_ir::op::Op;
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId};

use super::normalize::{NodeIndex, NormalizedStage};
use super::region::{LibraryOp, StageIndex};
use super::symbolic::{Dimension, symbolic_dims_match_expected, symbolic_shape_matches_static};

/// One recognized library dataflow: which library, which nodes it consumes,
/// and the values crossing its boundary.
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
    matches
}

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

/// The seven ops [`eta_ir::expand::softmax`] emits, recovered by walking back
/// from the probabilities they produce.
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
    /// The shape both broadcasts were given, which they have to agree on.
    shape: Shape,
}

/// Invert [`eta_ir::expand::softmax`], checking that both reductions read the
/// same `logits` the caller already found.
///
/// Only the `ReduceMax` identity has a mutation of its own
/// (`NucleusMutation::ForeignMaximum`). The `ReduceSum` one and the two
/// declared-shape comparisons cannot fail alone: the sum is already named as a
/// consumer of the exponentials by [`Chain::expected_consumers`], and two ops
/// in one chain that declare different shapes have different types, which
/// `eta_ir::infer` rejects before this runs. They are here so that reading
/// this function tells you it matched a softmax.
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

/// The chain [`eta_ir::expand::nucleus_sample`] emits, read backwards.
///
/// Recovering the chain and judging it are separate jobs, kept separate.
/// [`match_chain`] does the recovery and answers nothing about whether the
/// match may be taken; three named predicates do that, so each can be read and
/// tested on its own rather than as an unmarked phase of one long function.
/// The three questions asked of a `Chain`: which values enter the library call
/// ([`nucleus_library_inputs`]), whether the chain is the library's alone to
/// take ([`chain_is_exclusive`]), and whether the types line up
/// ([`chain_types_agree`]).
struct Chain {
    softmax: Softmax,
    pivot_node: NodeIndex,
    negative_infinity_node: NodeIndex,
    select_node: NodeIndex,
    rng_node: NodeIndex,
    add_node: NodeIndex,
    final_node: NodeIndex,
    /// What the chain reduces, centers and selects from. A temperature divide
    /// feeding it is folded in by [`nucleus_library_inputs`], so this is not
    /// necessarily what the library ends up reading.
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
    /// The shape the gumbel draw was given, which the softmax has to match.
    rng_shape: Shape,
}

impl Chain {
    /// Every node the library call would consume, in emission order.
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

    /// Each intermediate and the nodes the chain expects to read it.
    ///
    /// This is the chain's fan-out, and it is what makes the match exclusive
    /// rather than merely present: an intermediate read by anything outside is
    /// still live after the library call replaces the ops that produced it.
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

/// Walk back from an `argmax(masked + noise)` over the shape
/// [`eta_ir::expand::nucleus_sample`] emits.
///
/// The order mirrors that function read bottom-up: `mask_apply`, then
/// `gumbel`, then the pivot that decides the kept set, then the softmax the
/// pivot ranks. Nothing here looks at types or at who else reads what; this
/// only answers "are these the right ops, wired the right way".
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
        predicate: Predicate::CummassLe(top_p),
    } = stage.ops.get(pivot_node.index())?
    else {
        return None;
    };
    let (probabilities, top_p) = (*probabilities, *top_p);

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
    })
}

/// What the library call reads.
struct LibraryInputs {
    /// The region's inputs, in the order the kernel takes them.
    values: Vec<ValueId>,
    /// The `(logits, divisor)` of a temperature divide the kernel absorbed.
    scaled: Option<(ValueId, ValueId)>,
}

/// The values the library call reads, and the temperature divide it absorbs.
///
/// A `logits / t` feeding the chain is folded in when the chain is the only
/// thing that reads the scaled result, because the library kernel divides on
/// the way in; the scaled value stays an input so the region still names what
/// it replaced. A `reshape` between the two is seen through. When the divide
/// is shared the fold is dropped rather than the match: the divide is then a
/// real op that has to survive.
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

/// Whether the chain is the library call's alone to take, and the nodes it
/// would take.
///
/// Three ways it is not: a node appears twice, so the "chain" folded back on
/// itself; an input is produced inside, so replacing the nodes would delete
/// its own argument; or an intermediate is read from outside, so it outlives
/// the ops that made it.
///
/// The third subsumes the second. Twelve of the thirteen chain nodes define a
/// value the table names, and the thirteenth defines the result, which nothing
/// earlier can read -- so an input produced inside the chain always shows up as
/// a consumer the table did not expect. The explicit test stays because it is
/// the cheaper statement of a property the table only implies.
///
/// The result also has to escape, for the opposite reason: a chain nothing
/// reads is dead code, and a library call is not how to delete it. That one
/// cannot currently fire -- normalization deletes the chain first, which
/// `a_sampler_nobody_reads_is_deleted_before_it_is_matched` pins -- and it
/// stays only because it is what this function is for.
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

/// Whether every value in the chain has the type the library kernel assumes.
///
/// The kernel takes f32 logits over rows, one `u32[2]` rng state, an f32
/// `top_p` that is either scalar or one per row, and writes one i32 per row.
/// Everything in between is either logits-shaped, row-shaped or scalar.
///
/// No test reaches a rejection here, and the two attempts recorded in
/// `nucleus_lookalikes_remain_generic`'s history -- a `top_p` with the wrong
/// row count, and rank-3 logits -- were both rejected by `eta_ir::infer`
/// first: `Op::PivotThreshold` already requires rank 1 or 2 and a scalar or
/// per-row threshold, and the elementwise ops already force the chain to one
/// shape. This is a second opinion on a bound stage, not the first one.
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
    if top_p_type.dtype != Dtype::F32
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

/// Match one operand order of the `masked + noise` add.
///
/// Four independent questions, asked in the only order that works: what the
/// ops are, what the library would read, whether the chain is exclusively
/// its own, and whether the types agree.
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
        library: LibraryOp::NucleusSample,
        nodes,
        inputs: inputs.values,
        outputs: vec![chain.token],
    })
}
