//! `Error` — what an engine says when it will not do the thing.
//!
//! # The `PIE_STATUS_*` graveyard
//!
//! This enum replaces a block of `i32` constants that outlived the C ABI they
//! were the return type of:
//!
//! ```text
//! PIE_STATUS_OK               =  0   ->  Ok(_)
//! PIE_STATUS_INVALID_ARGUMENT = -1   ->  Error::Invalid
//! PIE_STATUS_BAD_ABI_VERSION  = -2   ->  (nothing — see below)
//! PIE_STATUS_UNSUPPORTED      = -3   ->  Error::Unsupported
//! PIE_STATUS_CLOSED           = -4   ->  Error::Closed
//! PIE_STATUS_DRIVER_ERROR     = -5   ->  Error::Device
//! PIE_STATUS_EXHAUSTED        = -6   ->  Error::Exhausted
//! PIE_STATUS_IMPOSSIBLE       = -7   ->  Error::Impossible
//! ```
//!
//! Every engine in this workspace is Rust and is called in-process through a
//! `&mut dyn Engine`. Nothing marshals, so a status code bought nothing and
//! cost the two things a code always costs: the caller had to keep a table to
//! read it, and a code with no message attached had to be paired with a
//! separate string that nothing made it agree with. `Err(Error::…)`
//! carries both halves in one value and `match` checks the table.
//!
//! **`BAD_ABI_VERSION` has no successor, on purpose.** It guarded
//! `PIE_DRIVER_ABI_VERSION`, a `u32` stamped into `DeviceFacts` and compared
//! on every load — an ABI version on an in-process Rust call between two
//! crates that Cargo compiles together, which cannot disagree. Where two
//! *processes* really do face each other, the version is the transport's
//! (decision 19: "remote is a property, not an encoding") and it is checked
//! where the bytes are, not here.
//!
//! # Two of these are scheduling answers, not failures
//!
//! [`Error::Exhausted`] and [`Error::Impossible`] are what the old
//! `FrameLaunchOutcome::{Exhausted, Impossible}` said, and they are the reason
//! the runtime matched on the status ladder at all. A fire that does not fit
//! **right now** is `Exhausted` — retry it behind something that frees pages.
//! A fire that will never fit is `Impossible` — refuse the request. Keeping
//! them as error variants rather than as a third `Ok` shape is deliberate:
//! there is no ticket to hand back, and a caller that ignores the distinction
//! gets a loud failure rather than a silently dropped submission.

use std::fmt;

/// What an engine answers when a verb does not complete.
///
/// One enum for every verb, because the shells answer a `Result` and the
/// runtime's dispatch loop wants one `match`. Variants carry a message where a
/// human is the audience and a number where the caller can act on it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// This engine does not serve this verb at all.
    ///
    /// The one failure the runtime has always matched on rather than logged —
    /// a Metal shell has no `copy_kv`, and a caller that asked is expected to
    /// route around it rather than fail the request. Both halves are
    /// `&'static str` so the value is `Copy`-cheap to build on the default
    /// method bodies in [`Engine`](crate::Engine).
    #[error("the {engine} engine does not serve `{verb}`")]
    Unsupported {
        /// The verb, spelled as the trait method.
        verb: &'static str,
        /// The engine kind that refused it, as [`Engine::kind`](crate::Engine::kind).
        engine: &'static str,
    },

    /// The submission is malformed: a CSR that decreases, a lane naming a slot
    /// the pools do not have, a shape that does not multiply out.
    ///
    /// The caller built something the contract does not describe. It is not a
    /// device condition and retrying it verbatim cannot help.
    #[error("invalid submission: {0}")]
    Invalid(String),

    /// A handle the submission names is closed, or was never open.
    #[error("{what} {id} is closed")]
    Closed {
        /// What kind of handle — `"instance"`, `"channel"`, `"program"`.
        what: &'static str,
        /// Its id.
        id: u64,
    },

    /// The device cannot fit this fire **now**, but a later one could.
    ///
    /// A scheduling answer (see the module header). `wanted`/`available` are
    /// in whatever unit the exhausted resource counts in — pages, slots,
    /// adapter banks — and `resource` names it, because a bare "exhausted" is
    /// a log line and these three are a decision.
    #[error("{resource} exhausted: wanted {wanted}, {available} available")]
    Exhausted {
        /// Which pool ran out.
        resource: &'static str,
        /// How much the submission asked for.
        wanted: u64,
        /// How much there was.
        available: u64,
    },

    /// The device can never fit this fire: it is past a ceiling this load was
    /// baked against, and no amount of freeing helps.
    #[error("impossible submission: {0}")]
    Impossible(String),

    /// The load failed: a checkpoint that does not fit the plan, a plan these
    /// budgets do not admit, a device that would not bind.
    #[error("load failed: {0}")]
    Load(String),

    /// A guest program would not compile, adopt, or bind.
    #[error("program: {0}")]
    Program(String),

    /// The device itself failed — an allocation, a launch, a transfer, a
    /// synchronize. The message is the backend's, verbatim.
    #[error("device: {0}")]
    Device(String),

    /// The engine is gone: the remote hung up, the process died, the shell was
    /// torn down under an outstanding ticket.
    #[error("engine disconnected: {0}")]
    Disconnected(String),
}

impl Error {
    /// The refusal a default trait-method body answers with.
    #[must_use]
    pub const fn unsupported(engine: &'static str, verb: &'static str) -> Error {
        Error::Unsupported { verb, engine }
    }

    /// Build an [`Error::Invalid`] from anything that renders.
    pub fn invalid(why: impl fmt::Display) -> Error {
        Error::Invalid(why.to_string())
    }

    /// Build an [`Error::Device`] from anything that renders.
    pub fn device(why: impl fmt::Display) -> Error {
        Error::Device(why.to_string())
    }

    /// Build an [`Error::Program`] from anything that renders.
    pub fn program(why: impl fmt::Display) -> Error {
        Error::Program(why.to_string())
    }

    /// Build an [`Error::Load`] from anything that renders.
    pub fn load(why: impl fmt::Display) -> Error {
        Error::Load(why.to_string())
    }

    /// True for the two variants that are scheduling answers rather than
    /// failures — see the module header.
    ///
    /// A caller that retries on [`Error::Exhausted`] and drops the
    /// request on [`Error::Impossible`] is reading this correctly; one
    /// that treats either as a device fault will either spin or give up on
    /// work that would have run.
    #[must_use]
    pub const fn is_scheduling(&self) -> bool {
        matches!(
            self,
            Error::Exhausted { .. } | Error::Impossible(_)
        )
    }

    /// **True for the one refusal the runtime answers by submitting the same
    /// frame again** (alto design §1 article 4).
    ///
    /// Article 4 splits [`is_scheduling`](Error::is_scheduling) in two, and
    /// the split is the point of the article. Admission is atomic: a frame
    /// that does not fit is refused **before any stream work**, with zero side
    /// effects, so the identical frame is a legal thing to submit again the
    /// moment something frees what it wanted. That is [`Error::Exhausted`],
    /// and it is what the engine lane's retry-in-place loop turns on.
    ///
    /// [`Error::Impossible`] is scheduling too — it is an answer about
    /// capacity rather than a fault — but it is permanent for this deployment
    /// and retrying it is a spin. Everything else is a fault to surface.
    ///
    /// **RETRY IS NOT A LAUNCH OUTCOME.** This predicate reads an ADMISSION
    /// refusal. A device gate that answered "retry" after the commit would be
    /// a contract violation, because everything such a gate could refuse was
    /// proved impossible at submit.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Error::Exhausted { .. })
    }
}

/// What every [`Engine`](crate::Engine) verb answers.
pub type Result<T> = std::result::Result<T, Error>;
