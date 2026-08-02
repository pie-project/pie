//! When may a reshape be elided and its consumers pointed at its source?
//!
//! Both fused emitters want the same trick. A reshape that only relabels a
//! value's shape still costs a whole copy through device memory at runtime, and
//! in the sampler's graph that copy is vocabulary-wide. Eliding it means
//! rewriting every consumer of the result to read the source's bytes instead.
//!
//! One condition is a correctness rule for both backends: the source must
//! [cover](covers) the result. Both runtimes copy the *result's* element count
//! out of the source — `ptir_parallel_copy(a0, o0, out.len, dtype)` on CUDA,
//! `m1_copy_typed(src, dst, dst_desc.len, dtype)` on Metal — so a result longer
//! than its source reads off the end of the source. `metal::fused` had this
//! rule and `cuda::fused` did not; it lives here now so that "is this reshape a
//! view" has one answer rather than one per backend.
//!
//! The other condition is a boundary question, and the two backends legitimately
//! answer it differently — see [`escaping_values`]. Once a reshape is elided,
//! [`AliasTable`] is what carries that decision to every consumer.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::plan::{Dimension, Region, SymbolicType};

/// Which value's bytes each elided reshape's consumers should read instead.
///
/// Both fused emitters build one of these while deciding elisions, then resolve
/// every operand through it before emitting. Sharing the type is what keeps
/// "elided" meaning the same thing on both backends; the emitters still decide
/// *which* reshapes to elide themselves, because [`escaping_values`] documents
/// a boundary they answer differently.
///
/// [`Self::resolve`] walks, rather than reading a single entry, because
/// nothing here constrains the order elisions are recorded in: recording
/// `b -> c` after `a -> b` leaves `a` one hop short. The walk is bounded by the
/// number of entries, which is a termination proof rather than a guess — a
/// chain that took more steps than there are entries would have to revisit one,
/// and a revisited entry is a cycle. The bound is reached routinely, by every
/// chain that is as long as the table is large, and reaching it is correct:
/// after that many hops the walk has followed each entry at most once, so the
/// value it holds cannot be a key unless the aliases form a cycle. SSA makes
/// cycles unreachable; the `debug_assert!` after the loop is what a future
/// caller who breaks that assumption trips, so the failure is a panic in a
/// debug build rather than a silently wrong offset in a release one.
#[derive(Debug, Default, Clone)]
pub struct AliasTable {
    of: BTreeMap<u32, u32>,
}

impl AliasTable {
    /// An empty table: every value holds its own bytes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `result` was elided and its consumers should read `source`.
    pub fn elide(&mut self, result: u32, source: u32) {
        let source = self.resolve(source);
        self.of.insert(result, source);
    }

    /// The value that actually holds `value`'s bytes — `value` itself unless it
    /// was elided.
    pub fn resolve(&self, mut value: u32) -> u32 {
        for _ in 0..self.of.len() {
            match self.of.get(&value) {
                Some(&source) => value = source,
                None => return value,
            }
        }
        debug_assert!(
            !self.of.contains_key(&value),
            "alias chain from {value} outlived the table; the aliases form a cycle"
        );
        value
    }

    /// Whether `value` was elided, and so is not written by any emitted node.
    pub fn is_elided(&self, value: u32) -> bool {
        self.of.contains_key(&value)
    }
}

/// The values `metal::fused` refuses to elide: this region's outputs and sinks.
///
/// Outputs are a correctness rule everywhere — a value consumed outside the
/// region is read through its own offset by a kernel that knows nothing of this
/// one's aliases, and an elided node never writes that offset.
///
/// Sinks are not. A sink is a `chan_put` *inside* the region
/// (`plan::compile::region`), and both emitters alias-resolve every operand
/// they emit, including that one. Excluding sinks costs Metal a copy rather
/// than buying it a guarantee, so `cuda::fused` deliberately does not do it and
/// asks `region.outputs` directly.
pub fn escaping_values(region: &Region) -> BTreeSet<u32> {
    region
        .outputs
        .iter()
        .copied()
        .chain(region.sinks.iter().map(|sink| sink.value))
        .collect()
}

/// Same dtype, and `result` is no longer than `source` under every binding.
///
/// What the runtime actually does for a reshape is
/// `m1_copy_typed(src, dst, dst_desc.len, dtype)` — it takes the *result's*
/// element count from offset 0 of the source. So the result is a prefix view of
/// the source whenever the source is at least as long, and pointing consumers
/// at the source reproduces it byte for byte: they read through the result's
/// own descriptor, so they still read exactly `dst.len`.
///
/// Comparing lengths must survive symbolic extents, because the one graph that
/// needs this is the sampler, whose reshape is `[SampledRows, vocab] ->
/// [vocab]`. Splitting a shape into its static product and the multiset of its
/// symbolic ids is enough: if the result's symbolic ids are a sub-multiset of
/// the source's and its static product is no larger, the result is no longer
/// than the source under every binding. (An earlier version demanded fully
/// static extents on both sides and so never fired at all.)
///
/// The reverse reshape — `[vocab] -> [SampledRows, vocab]`, equally legal in
/// the IR, where `numel` is compared before symbolic lowering — is exactly the
/// case this rejects.
pub fn covers(value_types: &[SymbolicType], source: u32, result: u32) -> bool {
    let (
        Some((src_dtype, src_static, src_symbolic)),
        Some((dst_dtype, dst_static, mut dst_symbolic)),
    ) = (
        footprint(value_types, source),
        footprint(value_types, result),
    )
    else {
        return false;
    };
    if src_dtype != dst_dtype || dst_static > src_static {
        return false;
    }
    let mut remaining = src_symbolic;
    dst_symbolic.retain(|id| match remaining.iter().position(|kept| kept == id) {
        Some(at) => {
            remaining.remove(at);
            false
        }
        None => true,
    });
    dst_symbolic.is_empty()
}

/// A value's dtype, its static extent product, and its symbolic ids sorted.
fn footprint(value_types: &[SymbolicType], value: u32) -> Option<(u8, u64, Vec<u8>)> {
    let ty = value_types.get(value as usize)?;
    let mut statics: u64 = 1;
    let mut symbolic: Vec<u8> = Vec::new();
    for dim in &ty.dims {
        match dim {
            Dimension::Static(extent) => statics *= u64::from(*extent),
            Dimension::Symbolic(id) => symbolic.push(*id as u8),
        }
    }
    symbolic.sort_unstable();
    Some((ty.dtype as u8, statics, symbolic))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use tensor_ir::types::DType;
    use crate::plan::SymbolicExtent;

    fn ty(dtype: DType, dims: &[Dimension]) -> SymbolicType {
        SymbolicType {
            dtype,
            dims: dims.to_vec(),
        }
    }

    #[test]
    fn an_untouched_value_holds_its_own_bytes() {
        let mut table = AliasTable::new();
        assert_eq!(table.resolve(7), 7);
        assert!(!table.is_elided(7));
        table.elide(3, 1);
        assert_eq!(table.resolve(7), 7);
    }

    #[test]
    fn eliding_points_consumers_at_the_source() {
        let mut table = AliasTable::new();
        table.elide(4, 2);
        assert_eq!(table.resolve(4), 2);
        assert!(table.is_elided(4));
        assert!(!table.is_elided(2));
    }

    #[test]
    fn a_chain_resolves_whichever_order_it_was_recorded_in() {
        // Recorded source-first, the second `elide` already sees a root.
        let mut forward = AliasTable::new();
        forward.elide(2, 1);
        forward.elide(3, 2);

        // Recorded result-first, `3 -> 2` is one hop short until the walk runs.
        // Both emitters visit region nodes in an order this module does not
        // constrain, so the two must agree.
        let mut backward = AliasTable::new();
        backward.elide(3, 2);
        backward.elide(2, 1);

        assert_eq!(forward.resolve(3), 1);
        assert_eq!(backward.resolve(3), 1);
    }

    /// Index 0 is the sampler's source, 1 its result, 2 the reverse reshape's
    /// result, 3 the same shape at another dtype, 4 a larger static row.
    fn sampler_types() -> Vec<SymbolicType> {
        let rows = Dimension::Symbolic(SymbolicExtent::SampledRows);
        vec![
            ty(DType::F32, &[rows, Dimension::Static(128)]),
            ty(DType::F32, &[Dimension::Static(128)]),
            ty(DType::F32, &[rows, Dimension::Static(128)]),
            ty(DType::I32, &[Dimension::Static(128)]),
            ty(DType::F32, &[Dimension::Static(256)]),
        ]
    }

    #[test]
    fn the_samplers_reshape_is_still_a_view() {
        // `[SampledRows, vocab] -> [vocab]`. If this stopped holding, every
        // fused sampler region would go back to a vocabulary-wide device copy.
        assert!(covers(&sampler_types(), 0, 1));
    }

    #[test]
    fn a_reshape_that_grows_is_not_a_view() {
        // `[vocab] -> [SampledRows, vocab]`, which `Op::Reshape`'s `numel`
        // check admits because it is applied before symbolic lowering. Aliasing
        // it points a `SampledRows`-row read at a one-row buffer.
        assert!(!covers(&sampler_types(), 1, 2));
    }

    #[test]
    fn a_reshape_cannot_change_dtype_or_outgrow_a_static_source() {
        assert!(!covers(&sampler_types(), 1, 3), "f32 source, i32 result");
        assert!(!covers(&sampler_types(), 1, 4), "128 source, 256 result");
        assert!(!covers(&sampler_types(), 0, 9), "value id out of range");
    }
}
