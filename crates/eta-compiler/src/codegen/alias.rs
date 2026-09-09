use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use eta_ir::types::to_wire;

use crate::plan::{Dimension, Region, SymbolicType};

#[derive(Debug, Default, Clone)]
pub struct AliasTable {
    of: BTreeMap<u32, u32>,
}

impl AliasTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn elide(&mut self, result: u32, source: u32) {
        let source = self.resolve(source);
        self.of.insert(result, source);
    }

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

    pub fn is_elided(&self, value: u32) -> bool {
        self.of.contains_key(&value)
    }
}

pub fn escaping_values(region: &Region) -> BTreeSet<u32> {
    region
        .outputs
        .iter()
        .copied()
        .chain(region.sinks.iter().map(|sink| sink.value))
        .collect()
}

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
        let mut forward = AliasTable::new();
        forward.elide(2, 1);
        forward.elide(3, 2);

        let mut backward = AliasTable::new();
        backward.elide(3, 2);
        backward.elide(2, 1);

        assert_eq!(forward.resolve(3), 1);
        assert_eq!(backward.resolve(3), 1);
    }
}
