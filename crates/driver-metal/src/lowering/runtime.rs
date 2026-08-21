//! The plan's runtime STREAMS, answered by name.
//!
//! The no-ask contract's third channel has two halves on this driver. The
//! resident objects (`kv_cache`, `recurrent_state`, `attention_mask`) cross
//! as `Arg::Raised` and are built into views by [`super::views`]. The
//! per-fire streams (`positions`, `token_ids`, ...) are TENSORS: the text
//! mints each as a named runtime value (`ForwardPlan::runtime`), the
//! lowering leaves it `Buffers::NAMED`, and it reaches a launch as
//! `Arg::Named { value }` — the same door a seam value arrives through.
//!
//! [`Streams`] is the translation that door needs: value id → the fire
//! table this driver stages the stream in. [`super::resolve::Store`]'s
//! `Resolver::named` consults it FIRST and falls back to the seam map, so
//! seam values keep the answer they always had and a runtime stream binds
//! the fire's own staged table.
//!
//! Derived from the `ForwardPlan` and cached beside its lowering
//! ([`super::cached::Lowerings`]), because this driver drops the plan once
//! the lowering exists and value ids are the plan's own numbering.

use std::collections::BTreeMap;

use model_ir::trace::{ForwardPlan, ValueId};

use super::executor::FireTable;

/// Which fire table answers each runtime stream value of ONE plan.
#[derive(Clone, Debug, Default)]
pub struct Streams {
    by_value: BTreeMap<ValueId, FireTable>,
}

impl Streams {
    /// The map for one plan, from its runtime table.
    ///
    /// Names with no table on this driver (`qo_indptr`, `row_valid`,
    /// `first_token`) are left out rather than guessed: a launch that names
    /// one refuses `UnknownNamed` at bind, by value id, instead of reading a
    /// stand-in fluently.
    #[must_use]
    pub fn of(plan: &ForwardPlan) -> Self {
        let mut by_value = BTreeMap::new();
        for binding in &plan.runtime {
            if let Some(which) = table_of(&binding.name) {
                by_value.insert(binding.value, which);
            }
        }
        Self { by_value }
    }

    /// The fire table answering `value`, if it is one of the plan's streams.
    #[must_use]
    pub fn table_of(&self, value: ValueId) -> Option<FireTable> {
        self.by_value.get(&value).copied()
    }

    /// Whether this plan mints no streams at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_value.is_empty()
    }
}

/// The fire table a runtime NAME lands in on this driver.
///
/// The names are `kernels::runtime`'s; the tables are the ones
/// `crate::bind::tables::stage` fills every fire.
fn table_of(name: &str) -> Option<FireTable> {
    Some(match name {
        "positions" => FireTable::Positions,
        "token_ids" => FireTable::TokenIds,
        "request_of_token" => FireTable::RequestOfToken,
        "sampling_indices" => FireTable::SamplingIndices,
        // Tier-2 on the vocabulary, tier-1 on this driver: the rope table is
        // staged every fire and the rope routines take it as an operand.
        "rope_frequencies" => FireTable::RopeFrequencies,
        _ => return None,
    })
}
