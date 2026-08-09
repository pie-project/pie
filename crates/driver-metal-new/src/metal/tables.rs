//! Argument tables, kept by ordinal so a step encodes without allocating.
//!
//! # Why a cache rather than a table per dispatch
//!
//! An MTL4 argument table holds GPU ADDRESSES, and an address outlives the
//! encoder it was bound in. So the binding work of a step is not per-step
//! work at all: the executor walks its graph once, binds every dispatch's
//! addresses into a table of its own, and every token after that only points
//! the encoder at a table that already exists.
//!
//! That is what makes the encode cost of a step flat in the number of tokens,
//! and it is the reason the command buffer of token *n+1* is byte-identical
//! to token *n*'s. Creating a table inside the encode loop would give that up
//! for no gain: `newArgumentTableWithDescriptor:` is a device allocation, and
//! there are a few hundred dispatches in one forward pass.
//!
//! # The key is the flat ordinal, and nothing else
//!
//! Not `(kind, layer)`. Within one layer the same kernel recurs -- two RMS
//! norms, two residual adds -- so a `(kind, layer)` key collides, and a
//! collision here is not a miss, it is one dispatch reading the other's
//! addresses. The executor's DAG walk already numbers dispatches 0..n in a
//! stable order; that number is the key.
//!
//! # Lookup on the encode path never creates
//!
//! [`Tables::get`] returns what is there. A step that asks for an ordinal
//! nobody bound is a bug in the graph walk, and the interesting part is what
//! Metal does with it: nothing. `setArgumentTable:` is simply not called, the
//! PREVIOUS dispatch's table stays bound, and the kernel runs to completion
//! over another dispatch's buffers. Every check stays green. So the miss is
//! an error here, raised before the dispatch is encoded.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use super::context::Context;
use super::encoder::ArgumentTable;
use super::heap::Slot;
use crate::error::{Error, Result};

/// The most buffer bindings one dispatch can have.
///
/// MSL's own limit: `[[buffer(0)]]` through `[[buffer(30)]]`. A table wider
/// than this could be created, but nothing could address the extra entries.
pub const MAX_BINDINGS: usize = 31;

/// Argument tables kept by dispatch ordinal.
///
/// Build the tables during setup with [`Tables::bind`]; read them on the
/// encode path with [`Tables::get`], which never allocates.
#[derive(Debug, Default)]
pub struct Tables {
    tables: BTreeMap<u32, ArgumentTable>,
    /// Which `(ordinal, index)` pairs have had an address written to them.
    ///
    /// Metal cannot be asked. A table reports neither which of its entries
    /// were set nor what they hold, and an entry that was never written reads
    /// as address zero -- which a kernel dereferences like any other.
    bound: BTreeMap<u32, u32>,
    /// What was written, kept beside the mask rather than in place of it.
    ///
    /// The mask alone answers "was this wired up", which is what a coverage
    /// test over the DAG asks. It cannot answer "is it wired up to the right
    /// thing", and that is the failure that actually happens: two ordinals
    /// that should share a residual buffer are each bound to something, both
    /// report bound, and the model produces plausible garbage. Zero is not
    /// usable as the "never written" sentinel here because zero is also what
    /// an unwritten Metal entry reads as -- hence a mask AND a value.
    addresses: BTreeMap<u32, [u64; MAX_BINDINGS]>,
}

impl Tables {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind `slot` at `index` in the table for `ordinal`, creating it if new.
    ///
    /// This is the setup path, and the only one that allocates.
    pub fn bind(
        &mut self,
        context: &Context,
        ordinal: u32,
        index: usize,
        slot: &Slot<'_>,
    ) -> Result<()> {
        self.bind_address(context, ordinal, index, slot.gpu_address())
    }

    /// Bind a raw GPU address at `index` in the table for `ordinal`.
    pub fn bind_address(
        &mut self,
        context: &Context,
        ordinal: u32,
        index: usize,
        address: u64,
    ) -> Result<()> {
        if index >= MAX_BINDINGS {
            return Err(Error::Create {
                what: "argument table binding",
                message: format!(
                    "ordinal {ordinal} index {index}: MSL allows at most {MAX_BINDINGS} \
                     buffer bindings, [[buffer(0..{})]]",
                    MAX_BINDINGS - 1
                ),
            });
        }
        let table = match self.tables.entry(ordinal) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(ArgumentTable::new(context, MAX_BINDINGS)?),
        };
        table.bind_address(index, address)?;
        // `index` is below MAX_BINDINGS, which is 31, so the shift is in range.
        *self.bound.entry(ordinal).or_default() |= 1u32 << index;
        self.addresses
            .entry(ordinal)
            .or_insert_with(|| [0; MAX_BINDINGS])[index] = address;
        Ok(())
    }

    /// The table for `ordinal`, if one was built.
    ///
    /// Never creates one. See the module docs for what a silent miss costs.
    #[must_use]
    pub fn get(&self, ordinal: u32) -> Option<&ArgumentTable> {
        self.tables.get(&ordinal)
    }

    /// The table for `ordinal`, as an error if it was never built.
    pub fn expect(&self, ordinal: u32) -> Result<&ArgumentTable> {
        self.get(ordinal).ok_or_else(|| Error::Create {
            what: "argument table",
            message: format!(
                "nothing was bound for ordinal {ordinal}; the dispatch would run over the \
                 previous one's addresses"
            ),
        })
    }

    /// The address written at `index` of `ordinal`, or `None` if none was.
    ///
    /// `None` and `Some(0)` are different answers and the distinction is the
    /// reason this returns an `Option`: an entry nobody bound reads as zero
    /// from Metal, so a bare `u64` would report "never wired" and "wired to
    /// a null address" identically.
    #[must_use]
    pub fn address(&self, ordinal: u32, index: usize) -> Option<u64> {
        if !self.is_bound(ordinal, index) {
            return None;
        }
        self.addresses.get(&ordinal).map(|slots| slots[index])
    }

    /// Whether `index` of `ordinal` has had an address written to it.
    #[must_use]
    pub fn is_bound(&self, ordinal: u32, index: usize) -> bool {
        if index >= MAX_BINDINGS {
            return false;
        }
        self.bound
            .get(&ordinal)
            .is_some_and(|mask| mask & (1u32 << index) != 0)
    }

    /// How many bindings `ordinal` has.
    #[must_use]
    pub fn binding_count(&self, ordinal: u32) -> u32 {
        self.bound.get(&ordinal).map_or(0, |m| m.count_ones())
    }

    /// Drop the table for `ordinal`.
    ///
    /// Its addresses go with it, which is the point: a graph that is rebuilt
    /// must not inherit a stale binding from the one before.
    pub fn forget(&mut self, ordinal: u32) -> bool {
        self.addresses.remove(&ordinal);
        self.bound.remove(&ordinal);
        self.tables.remove(&ordinal).is_some()
    }

    /// How many tables are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// Whether no table has been built.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// The ordinals that have a table, in order.
    pub fn ordinals(&self) -> impl Iterator<Item = u32> + '_ {
        self.tables.keys().copied()
    }
}
