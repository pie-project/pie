//! When may a reshape be elided and its consumers pointed at its source? A
//! reshape that only relabels shape still costs a device-memory copy, so
//! eliding it rewrites every consumer to read the source's bytes instead.
//! [`covers`] is the correctness rule shared by both backends; [`escaping_values`]
//! is a boundary the two backends answer differently. [`AliasTable`] carries
//! the elision decision to every consumer.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use eta_ir::types::to_wire;

use crate::plan::{Dimension, Region, SymbolicType};

/// Which value's bytes each elided reshape's consumers should read instead.
/// [`Self::resolve`] walks rather than reading one entry, since nothing
/// constrains recording order (`b -> c` after `a -> b` needs the walk to
/// reach `a`); it's bounded by table size as a termination proof, and the
/// trailing `debug_assert!` catches a cycle that SSA should make unreachable.
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
/// Outputs are a correctness rule everywhere (an elided node never writes the
/// offset an outside consumer reads). Sinks are a Metal-only precaution;
/// `cuda::fused` alias-resolves them instead and asks `region.outputs` directly.
pub fn escaping_values(region: &Region) -> BTreeSet<u32> {
    region
        .outputs
        .iter()
        .copied()
        .chain(region.sinks.iter().map(|sink| sink.value))
        .collect()
}

/// Same dtype, and `result` is no longer than `source` under every binding.
/// Must survive symbolic extents (the sampler's `[SampledRows, vocab] ->
/// [vocab]`): a shape's static product and multiset of symbolic ids decide
/// it, since the result's symbolic ids must be a sub-multiset of the
/// source's with no larger static product.
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
    Some((to_wire(ty.dtype)?, statics, symbolic))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    

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

}
