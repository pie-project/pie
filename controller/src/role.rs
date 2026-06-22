//! Role-assignment axis.
//!
//! One of three orthogonal axes. This module owns *which node serves which
//! stage* — prefill, decode, or encode — and nothing else. It deliberately
//! knows nothing about:
//!
//! - the **backend** axis (cuda / portable / dummy): the controller is control
//!   plane only and never sees a driver, and
//! - the **topology** axis (on-device vs distributed, and prefill↔decode
//!   pairing): that lives in [`crate::pairing`].
//!
//! Keeping the axes in separate modules is the whole point — a role is a
//! property of one node, a pairing is a relation between two, and a backend is
//! invisible here.

use std::collections::HashMap;

use pie_schema::{Role, WorkerId};

/// Maps each known node to its assigned role.
///
/// Assignment is last-writer-wins: re-assigning a node simply overwrites its
/// role. Pairing validity (does this role layout permit a pair?) is enforced
/// elsewhere, in [`crate::pairing`].
#[derive(Debug, Default, Clone)]
pub struct RoleTable {
    roles: HashMap<WorkerId, Role>,
}

impl RoleTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign (or re-assign) `node` to `role`.
    pub fn assign(&mut self, node: WorkerId, role: Role) {
        self.roles.insert(node, role);
    }

    /// Drop any assignment for `node` (e.g. when it leaves the cluster).
    pub fn forget(&mut self, node: WorkerId) {
        self.roles.remove(&node);
    }

    /// The role currently assigned to `node`, if any.
    pub fn role_of(&self, node: WorkerId) -> Option<Role> {
        self.roles.get(&node).copied()
    }

    /// Every node currently assigned `role`.
    pub fn nodes_with(&self, role: Role) -> Vec<WorkerId> {
        self.roles
            .iter()
            .filter_map(|(&node, &r)| (r == role).then_some(node))
            .collect()
    }

    /// Number of assigned nodes.
    pub fn len(&self) -> usize {
        self.roles.len()
    }

    /// Whether no node has a role yet.
    pub fn is_empty(&self) -> bool {
        self.roles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_and_query() {
        let mut t = RoleTable::new();
        t.assign(WorkerId(1), Role::Prefill);
        t.assign(WorkerId(2), Role::Decode);
        t.assign(WorkerId(3), Role::Prefill);

        assert_eq!(t.role_of(WorkerId(1)), Some(Role::Prefill));
        assert_eq!(t.role_of(WorkerId(99)), None);

        let mut prefills = t.nodes_with(Role::Prefill);
        prefills.sort();
        assert_eq!(prefills, vec![WorkerId(1), WorkerId(3)]);
    }

    #[test]
    fn reassign_overwrites() {
        let mut t = RoleTable::new();
        t.assign(WorkerId(1), Role::Prefill);
        t.assign(WorkerId(1), Role::Encode);
        assert_eq!(t.role_of(WorkerId(1)), Some(Role::Encode));
        assert!(t.nodes_with(Role::Prefill).is_empty());
    }

    #[test]
    fn forget_removes() {
        let mut t = RoleTable::new();
        t.assign(WorkerId(1), Role::Decode);
        t.forget(WorkerId(1));
        assert_eq!(t.role_of(WorkerId(1)), None);
        assert!(t.is_empty());
    }
}
