//! Stage normalization: DCE, broadcast/CSE cleanup, constant folding, and
//! dense renumbering, producing the [`NormalizedStage`] that gets signed,
//! partitioned and encoded.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use eta_ir::container::PortSource;
use eta_ir::op::{ChannelIndex, Op};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, ValueId, ValueType};
use eta_ir::validate::BoundTrace;

use super::fold::{canonicalize_commutative, cse_candidate, cse_key, fold_scalar, simplify_alias};
use super::signature::signature_ports;
use super::symbolic::{Dimension, SymbolicType, symbolic_result_type};

/// What a normalized value is *about*, coarsely. Unread by any emitter or
/// engine branch, but hashed into [`StageSignature`](super::signature::StageSignature),
/// so it affects the emitted kernel's name and the engine's cache key.
/// `PageDescriptor` and `EffectToken` are never produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueDomain {
    /// A rank-0 value, or one whose every dimension is `1`.
    Scalar = 0,
    /// A per-row tensor — the fallthrough when nothing else applies, so
    /// reductions land here too.
    PerRow = 1,
    /// A tensor whose trailing dimension is the model's vocabulary width.
    Vocabulary = 2,
    /// Device-materialized indices, i.e. the result of [`Op::Iota`].
    GeneratedIndex = 3,
    /// A boolean (`Dtype::Bool`) mask.
    Mask = 4,
    /// A KV-page descriptor. Reserved: `value_domain` never returns it.
    PageDescriptor = 5,
    /// The result of a library op — [`Op::TopK`], [`Op::SortDesc`],
    /// [`Op::MatMul`], or [`Op::KernelCall`].
    LibraryResult = 6,
    /// An effect token. Reserved: `value_domain` never returns it.
    EffectToken = 7,
}

/// A position in a stage's op list. Deliberately not a [`ValueId`]: both are
/// dense `u32` starting at zero, so mixing them up type-checks but names the
/// wrong ops.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeIndex(pub u32);

impl NodeIndex {
    /// The wire form. Explicit because the wire cannot tell the two spaces
    /// apart either.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// As a slice index into the stage's op list.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A channel's position in one stage's own channel table. Deliberately not a
/// [`ChannelIndex`] (that numbers the container's channel declarations): both
/// are dense `u32` from zero, so they silently diverge past the first channel.
/// Only `localize_stage` converts between the two.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ChannelSlot(pub u32);

impl ChannelSlot {
    /// The bare number, for the wire and the generated sources — neither can
    /// tell the two spaces apart either.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// As a slice index into the stage's channel bindings.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A normalized stage with local channel/name numbering.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedStage {
    /// The stage this body belongs to.
    pub stage: Stage,
    /// ETA op count before normalization, retained for the `source_ops`
    /// metric now that [`ops`](Self::ops) has been thinned.
    pub source_op_count: u32,
    /// The normalized op list, densely renumbered.
    pub ops: Vec<Op>,
    /// Symbolic type of each SSA value, indexed by [`ValueId`].
    pub value_types: Vec<SymbolicType>,
    /// [`ValueDomain`] of each SSA value, parallel to
    /// [`value_types`](Self::value_types).
    pub value_domains: Vec<ValueDomain>,
    /// Original ETA op positions represented by each normalized op.
    pub source_ops: Vec<Vec<u32>>,
    /// Local channel slot -> program-global dense channel index.
    pub channel_bindings: Vec<u32>,
    /// Local name slot -> canonical second-party name.
    pub names: Vec<String>,
}

pub(crate) fn normalize_stage(bound: &BoundTrace, stage_index: usize) -> NormalizedStage {
    let stage_program = &bound.container.stages[stage_index];
    let original_types = &bound.stage_types[stage_index];
    let (result_bases, producer) = result_layout(&stage_program.ops);
    let keep = live_ops(stage_program, &result_bases, &producer);
    let redundant = redundant_select_broadcasts(stage_program, original_types, &result_bases);

    let mut value_map = vec![u32::MAX; original_types.len()];
    let mut normalized_ops: Vec<Op> = Vec::new();
    let mut normalized_types: Vec<SymbolicType> = Vec::new();
    let mut normalized_domains: Vec<ValueDomain> = Vec::new();
    let mut source_ops: Vec<Vec<u32>> = Vec::new();
    let mut literals: Vec<Option<Literal>> = Vec::new();
    let mut cse: BTreeMap<Vec<u8>, (u32, u32)> = BTreeMap::new();

    for (op_index, original_op) in stage_program.ops.iter().enumerate() {
        let base = result_bases[op_index] as usize;
        let result_count = original_op.result_count() as usize;
        if !keep[op_index] {
            continue;
        }
        // A scalar broadcast feeding only `Op::Select` is a no-op: `Select`
        // already broadcasts its operands. Alias it away instead of keeping it.
        if redundant[op_index]
            && let Op::Broadcast { value, .. } = original_op
        {
            value_map[base] = value_map[*value as usize];
            continue;
        }

        let mut op = original_op.clone();
        op.map_operands(|value| {
            let mapped = value_map[value as usize];
            debug_assert_ne!(mapped, u32::MAX, "live op references removed value");
            mapped
        });

        let result_types: Vec<SymbolicType> = (0..result_count)
            .map(|result| {
                symbolic_result_type(
                    bound,
                    original_op,
                    original_types[base + result],
                    &op,
                    original_types,
                    &normalized_types,
                )
            })
            .collect();

        if result_count == 1 {
            if let Some(alias) = simplify_alias(&op, &result_types[0], &literals) {
                value_map[base] = alias;
                continue;
            }
            if let Some(literal) = fold_scalar(&op, &literals) {
                op = Op::Const(literal);
            }
        }

        canonicalize_commutative(&mut op, result_types.first());

        let cse_key = if cse_candidate(&op) {
            Some(cse_key(&op, &result_types))
        } else {
            None
        };
        if let Some(key) = cse_key.as_ref()
            && let Some(&(existing_base, existing_op)) = cse.get(key)
        {
            for result in 0..result_count {
                value_map[base + result] = existing_base + result as u32;
            }
            source_ops[existing_op as usize].push(op_index as u32);
            continue;
        }

        let new_base = normalized_types.len() as u32;
        let normalized_op_index = normalized_ops.len() as u32;
        for (result, symbolic_type) in result_types.into_iter().enumerate() {
            value_map[base + result] = new_base + result as u32;
            normalized_domains.push(value_domain(bound.profile.vocab, &op, &symbolic_type));
            let literal = match &op {
                Op::Const(literal) => Some(*literal),
                _ => None,
            };
            literals.push(literal);
            normalized_types.push(symbolic_type);
        }
        normalized_ops.push(op);
        source_ops.push(vec![op_index as u32]);
        if let Some(key) = cse_key {
            cse.insert(key, (new_base, normalized_op_index));
        }
    }

    NormalizedStage {
        stage: stage_program.stage,
        source_op_count: stage_program.ops.len() as u32,
        ops: normalized_ops,
        value_types: normalized_types,
        value_domains: normalized_domains,
        source_ops,
        channel_bindings: Vec::new(),
        names: Vec::new(),
    }
}

pub(crate) fn result_layout(ops: &[Op]) -> (Vec<ValueId>, Vec<NodeIndex>) {
    let mut bases = Vec::with_capacity(ops.len());
    let mut producer = Vec::new();
    let mut next = 0u32;
    for (op_index, op) in ops.iter().enumerate() {
        bases.push(next);
        for _ in 0..op.result_count() {
            producer.push(NodeIndex(op_index as u32));
            next += 1;
        }
    }
    (bases, producer)
}

/// A `Broadcast` of a scalar whose every consumer is an `Op::Select` operand
/// (not the condition) is redundant, since `Select` already broadcasts its
/// operands.
pub(crate) fn redundant_select_broadcasts(
    stage_program: &eta_ir::container::StageProgram,
    original_types: &[ValueType],
    result_bases: &[ValueId],
) -> Vec<bool> {
    let mut redundant = vec![false; stage_program.ops.len()];
    let mut consumers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); original_types.len()];
    for (node, op) in stage_program.ops.iter().enumerate() {
        for (slot, operand) in op.operands().into_iter().enumerate() {
            if let Some(entry) = consumers.get_mut(operand as usize) {
                entry.push((node, slot));
            }
        }
    }
    for (node, op) in stage_program.ops.iter().enumerate() {
        let Op::Broadcast { value, .. } = op else {
            continue;
        };
        let Some(source_type) = original_types.get(*value as usize) else {
            continue;
        };
        if source_type.shape.rank() != 0 {
            continue;
        }
        let result = result_bases[node] as usize;
        let Some(uses) = consumers.get(result) else {
            continue;
        };
        // `Op::Select` operands are `[cond, a, b]`; slot 0 keeps its own shape.
        if !uses.is_empty()
            && uses.iter().all(|(consumer, slot)| {
                matches!(stage_program.ops[*consumer], Op::Select { .. }) && *slot != 0
            })
        {
            redundant[node] = true;
        }
    }
    redundant
}

pub(crate) fn live_ops(
    stage_program: &eta_ir::container::StageProgram,
    result_bases: &[ValueId],
    producer: &[NodeIndex],
) -> Vec<bool> {
    let mut keep = vec![false; stage_program.ops.len()];
    let mut values = Vec::new();
    for (op_index, op) in stage_program.ops.iter().enumerate() {
        // DCE roots at the effectful ops.
        if op.is_effectful() {
            keep[op_index] = true;
            values.extend(op.operands());
        }
    }
    while let Some(value) = values.pop() {
        let op_index = producer[value as usize].index();
        if !keep[op_index] {
            keep[op_index] = true;
            values.extend(stage_program.ops[op_index].operands());
        }
    }

    // A kept multi-result producer keeps all of its positional results.
    debug_assert_eq!(
        result_bases.last().copied().unwrap_or(0)
            + stage_program.ops.last().map(Op::result_count).unwrap_or(0),
        producer.len() as u32
    );
    keep
}

/// Classify one value.
pub(crate) fn value_domain(vocab: u32, op: &Op, value_type: &SymbolicType) -> ValueDomain {
    if value_type.is_scalar() {
        return ValueDomain::Scalar;
    }
    if matches!(op, Op::Iota { .. }) {
        return ValueDomain::GeneratedIndex;
    }
    if value_type.dtype == Dtype::Bool {
        return ValueDomain::Mask;
    }
    if value_type
        .dims
        .last()
        .is_some_and(|dimension| *dimension == Dimension::Static(vocab))
    {
        return ValueDomain::Vocabulary;
    }
    if matches!(
        op,
        Op::TopK { .. } | Op::SortDesc(_) | Op::MatMul(_, _) | Op::KernelCall { .. }
    ) {
        return ValueDomain::LibraryResult;
    }
    ValueDomain::PerRow
}

pub(crate) fn localize_stage(bound: &BoundTrace, stage: &mut NormalizedStage) {
    let mut channels = Vec::new();
    let mut names = Vec::new();
    for op in &mut stage.ops {
        // Not `else if`: an op could carry both.
        if let Some(channel) = op.channel_mut() {
            *channel = local_channel(&mut channels, *channel).get();
        }
        if let Some(name) = op.name_index_mut() {
            *name = local_name(&bound.container.names, &mut names, *name);
        }
    }
    for port in signature_ports(bound, stage.stage) {
        if let PortSource::Channel(channel) = port.source {
            local_channel(&mut channels, channel);
        }
    }

    stage.channel_bindings = channels;
    stage.names = names;
}

/// The slot `global` occupies in this stage's channel table, binding it if the
/// stage has not touched it yet.
pub(crate) fn local_channel(channels: &mut Vec<ChannelIndex>, global: ChannelIndex) -> ChannelSlot {
    if let Some(local) = channels.iter().position(|channel| *channel == global) {
        ChannelSlot(local as u32)
    } else {
        channels.push(global);
        ChannelSlot((channels.len() - 1) as u32)
    }
}

pub(crate) fn local_name(global_names: &[String], names: &mut Vec<String>, global: u16) -> u16 {
    let name = &global_names[global as usize];
    if let Some(local) = names.iter().position(|candidate| candidate == name) {
        local as u16
    } else {
        names.push(name.clone());
        (names.len() - 1) as u16
    }
}

#[cfg(test)]
mod value_domain_tests {
    use super::*;
    use crate::plan::compile::signature::stage_signature;
    

    // Pins that the signature still hashes value_domains, since that hash is
    // the emitted kernel's entry-point name and the engine's cache key.
    #[test]
    fn the_signature_still_depends_on_value_domains() {
        let mut stage = NormalizedStage {
            stage: Stage::Epilogue,
            source_op_count: 1,
            ops: alloc::vec![Op::Iota { len: 4 }],
            value_types: alloc::vec![SymbolicType {
                dtype: Dtype::U32,
                dims: alloc::vec![Dimension::Static(4)],
            }],
            value_domains: alloc::vec![ValueDomain::GeneratedIndex],
            source_ops: alloc::vec![alloc::vec![0]],
            channel_bindings: alloc::vec![],
            names: alloc::vec![],
        };
        let bound = super::super::tests::program(0, 1);
        let before = stage_signature(&bound, &stage).hash;
        stage.value_domains[0] = ValueDomain::PerRow;
        assert_ne!(
            before,
            stage_signature(&bound, &stage).hash,
            "the signature stopped hashing value_domains, so the field now \
             affects nothing at all and should be deleted rather than kept as \
             an unread enum -- but deleting it renames every emitted kernel"
        );
    }

    // Reductions fall through to PerRow.
    #[test]
    fn reductions_are_per_row_by_falling_through() {
        let per_row = SymbolicType {
            dtype: Dtype::F32,
            dims: alloc::vec![Dimension::Static(3), Dimension::Static(5)],
        };
        let bound = super::super::tests::program(0, 1);
        for op in [
            Op::ReduceSum(0),
            Op::ReduceMax(0),
            Op::ReduceMin(0),
            Op::Add(0, 0),
        ] {
            assert_eq!(
                value_domain(bound.profile.vocab, &op, &per_row),
                ValueDomain::PerRow,
                "{op:?}"
            );
        }
    }
}
