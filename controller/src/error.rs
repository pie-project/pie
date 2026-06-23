//! Crate error type.

use crate::pairing::PairId;
use pie_schema::{Role, WorkerId};

/// Everything the coordination API can refuse to do.
///
/// Coordination errors are about *cluster state* (unknown worker, no eligible
/// route target, pairing conflicts) or — in the distributed deployment — the
/// control-RPC transport. The controller never observes data-plane failures.
#[derive(Debug, thiserror::Error)]
pub enum ControllerError {
    /// Referenced a worker that was never registered (or already removed).
    #[error("unknown worker {0}")]
    UnknownWorker(WorkerId),

    /// `route`/`pair` found no live worker to place work on.
    #[error("no eligible worker to route to")]
    NoEligibleWorker,

    /// Control-RPC transport or codec failure (distributed deployment only).
    /// Carries a rendered description — the embedded in-proc controller never
    /// produces this.
    #[error("control-rpc transport: {0}")]
    Transport(String),

    /// The remote controller returned an error response over the control-RPC
    /// (distributed deployment only) — a coordination error that occurred on the
    /// controller side, relayed back as text.
    #[error("controller rejected request: {0}")]
    Remote(String),

    // --- pairing.rs (kept per the minimal-start spec, not extended) ---
    /// Referenced a pair that does not exist (or was already stepped out).
    #[error("unknown pair {0}")]
    UnknownPair(PairId),

    /// Tried to pair two workers where the role layout is wrong — pairing matches
    /// exactly one [`Role::Prefill`] worker with one [`Role::Decode`] worker.
    #[error(
        "cannot pair {prefill} ({prefill_role}) with {decode} ({decode_role}): expected one prefill and one decode"
    )]
    RolePairMismatch {
        prefill: WorkerId,
        prefill_role: Role,
        decode: WorkerId,
        decode_role: Role,
    },

    /// A worker is already a member of another live pair.
    #[error("worker {0} is already paired")]
    AlreadyPaired(WorkerId),
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, ControllerError>;
