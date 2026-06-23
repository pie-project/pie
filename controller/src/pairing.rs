//! Pairing / topology axis.
//!
//! One of three orthogonal axes. This module owns *how nodes are wired to each
//! other*: it matches one prefill node `A` with one decode node `B` to form a
//! pair `A↔B`, and later dissolves it (the "step out"). It knows nothing about
//! the **backend** axis (no driver here) and treats the **role** axis as an
//! input it validates against, never something it owns.
//!
//! A node may belong to at most one live pair. Forming a pair requires the role
//! layout to be exactly one prefill + one decode; that check is the one place
//! the role axis and the topology axis meet, and it is explicit.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{ControllerError, Result};
use crate::role::RoleTable;
use pie_schema::{Role, WorkerId};

/// Opaque handle to a prefill↔decode pairing, minted by the controller.
///
/// Lives here with the pairing logic. Kept (per the minimal-start spec) for the
/// future PD-disaggregation path; the current trivial same-node `pair` does not
/// mint these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PairId(pub u64);

impl std::fmt::Display for PairId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pair#{}", self.0)
    }
}

/// A live prefill↔decode pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pair {
    pub id: PairId,
    pub prefill: WorkerId,
    pub decode: WorkerId,
}

/// Tracks the set of live pairs and which node belongs to which pair.
#[derive(Debug, Default)]
pub struct PairingTable {
    pairs: HashMap<PairId, Pair>,
    /// Reverse index: node → the pair it currently belongs to.
    membership: HashMap<WorkerId, PairId>,
    next_id: u64,
}

impl PairingTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Form a pair `prefill ↔ decode`, validating roles against `roles`.
    ///
    /// Fails if either node is unknown to `roles`, if the role layout is not
    /// exactly one prefill + one decode, or if either node is already paired.
    pub fn pair(
        &mut self,
        prefill: WorkerId,
        decode: WorkerId,
        roles: &RoleTable,
    ) -> Result<PairId> {
        let prefill_role = roles
            .role_of(prefill)
            .ok_or(ControllerError::UnknownWorker(prefill))?;
        let decode_role = roles
            .role_of(decode)
            .ok_or(ControllerError::UnknownWorker(decode))?;

        if prefill_role != Role::Prefill || decode_role != Role::Decode {
            return Err(ControllerError::RolePairMismatch {
                prefill,
                prefill_role,
                decode,
                decode_role,
            });
        }
        if self.membership.contains_key(&prefill) {
            return Err(ControllerError::AlreadyPaired(prefill));
        }
        if self.membership.contains_key(&decode) {
            return Err(ControllerError::AlreadyPaired(decode));
        }

        let id = PairId(self.next_id);
        self.next_id += 1;
        self.pairs.insert(
            id,
            Pair {
                id,
                prefill,
                decode,
            },
        );
        self.membership.insert(prefill, id);
        self.membership.insert(decode, id);
        Ok(id)
    }

    /// Dissolve a pair — the "step out". Returns the pair that was removed so
    /// the caller can notify its members.
    pub fn step_out(&mut self, id: PairId) -> Result<Pair> {
        let pair = self
            .pairs
            .remove(&id)
            .ok_or(ControllerError::UnknownPair(id))?;
        self.membership.remove(&pair.prefill);
        self.membership.remove(&pair.decode);
        Ok(pair)
    }

    /// The pair a node currently belongs to, if any.
    pub fn pair_of(&self, node: WorkerId) -> Option<Pair> {
        self.membership
            .get(&node)
            .and_then(|id| self.pairs.get(id))
            .copied()
    }

    /// All live pairs.
    pub fn pairs(&self) -> impl Iterator<Item = &Pair> {
        self.pairs.values()
    }

    /// Number of live pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether there are no live pairs.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles() -> RoleTable {
        let mut r = RoleTable::new();
        r.assign(WorkerId(1), Role::Prefill);
        r.assign(WorkerId(2), Role::Decode);
        r
    }

    #[test]
    fn pair_then_step_out() {
        let roles = roles();
        let mut t = PairingTable::new();

        let id = t.pair(WorkerId(1), WorkerId(2), &roles).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.pair_of(WorkerId(1)).unwrap().id, id);

        let removed = t.step_out(id).unwrap();
        assert_eq!(removed.prefill, WorkerId(1));
        assert!(t.is_empty());
        assert_eq!(t.pair_of(WorkerId(1)), None);
    }

    #[test]
    fn rejects_role_mismatch() {
        let mut roles = roles();
        roles.assign(WorkerId(2), Role::Prefill); // both prefill now
        let mut t = PairingTable::new();
        let err = t.pair(WorkerId(1), WorkerId(2), &roles).unwrap_err();
        assert!(matches!(err, ControllerError::RolePairMismatch { .. }));
    }

    #[test]
    fn rejects_unknown_node() {
        let roles = roles();
        let mut t = PairingTable::new();
        let err = t.pair(WorkerId(1), WorkerId(42), &roles).unwrap_err();
        assert!(matches!(err, ControllerError::UnknownWorker(WorkerId(42))));
    }

    #[test]
    fn rejects_double_pairing() {
        let mut roles = roles();
        roles.assign(WorkerId(3), Role::Decode);
        let mut t = PairingTable::new();
        t.pair(WorkerId(1), WorkerId(2), &roles).unwrap();
        let err = t.pair(WorkerId(1), WorkerId(3), &roles).unwrap_err();
        assert!(matches!(err, ControllerError::AlreadyPaired(WorkerId(1))));
    }

    #[test]
    fn step_out_unknown_fails() {
        let mut t = PairingTable::new();
        assert!(matches!(
            t.step_out(PairId(7)).unwrap_err(),
            ControllerError::UnknownPair(PairId(7))
        ));
    }
}
