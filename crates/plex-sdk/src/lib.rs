//! The PLEX 1.1 guest SDK (`docs/plex-1.1.md`).
//!
//! A policy is a type with a `react`, exported with [`export!`]:
//!
//! ```ignore
//! use plex::prelude::*;
//!
//! #[derive(Default)]
//! struct Fifo;
//!
//! impl Policy for Fifo {
//!     fn react(&mut self, _events: &[Event], _now: Mark) {
//!         let rows: Vec<_> = snapshot::get_processes().iter().map(|p| (p.id.clone(), -(p.arrival as i64))).collect();
//!         schedule::priority::set_list(&rows).ok();
//!     }
//! }
//!
//! plex::export!(Fifo);
//! ```
//!
//! The policy's component imports only the interfaces it calls, and an
//! import is the authority to write: build for `wasm32-unknown-unknown` and
//! wrap with `wasm-tools component new`.

#![no_std]
extern crate alloc;

#[doc(hidden)]
pub mod bindings {
    wit_bindgen::generate!({
        path: ["../plex-host/wit", "../plex-host/wit-ext"],
        inline: "
            package plex:sdk;
            world policy {
                include plex:policy/worker@1.1.0;
                include plex:policy/cluster@1.1.0;
                include pie:plex-ext/cluster@0.1.0;
            }
        ",
        world: "plex:sdk/policy",
        features: ["placement-helpers"],
        pub_export_macro: true,
        export_macro_name: "__export_policy",
        default_bindings_module: "plex::bindings",
        additional_derives: [PartialEq],
        generate_all,
    });
}

pub use bindings::plex::core::types::{
    By, Change, Dispatch, ErrorCode, Event, Eviction, Exit, Mark, Outcome, Pause, Prefix,
    ProcessId, ProcessState, Retention, SessionId, Subject, Tier, Value, WorkerId, WorkingSetId,
};
pub use bindings::plex::core::{alarm, hints, notes, snapshot, terminate};

/// The worker's tables: when its processes run, what KV it keeps.
pub mod schedule {
    pub use crate::bindings::plex::schedule::{hold, priority};
}
pub mod retention {
    pub use crate::bindings::plex::retention::{priority, tiers};
}
/// The cluster's tables: where processes run, where KV is copied.
pub mod placement {
    pub use crate::bindings::plex::placement::{helpers, workers};
}
pub mod replication {
    pub use crate::bindings::plex::replication::workers;
}

/// pie's extension of PLEX 1.1 (`pie:plex-ext`, not part of the 1.1
/// contract): a cluster policy's cross-tenant share grants. A policy that
/// imports it loads only on pie.
pub mod ext {
    pub use crate::bindings::pie::plex_ext::share_groups;
}

/// A length of time, in nanoseconds, as marks are.
pub type Duration = u64;
pub const MICROSECOND: Duration = 1_000;
pub const MILLISECOND: Duration = 1_000_000;
pub const SECOND: Duration = 1_000_000_000;

/// A policy overrides `react`, or the handlers it needs and `plan`: the
/// default `react` hands each event to its handler, then calls `plan`
/// once, where the policy writes its rows from the snapshot.
#[allow(unused_variables)]
pub trait Policy {
    /// React to what happened. The snapshot is the truth; `events` say
    /// what changed since the last call.
    fn react(&mut self, events: &[Event], now: Mark) {
        for event in events {
            match event {
                Event::Arrived(process) => self.on_arrived(process, now),
                Event::Dispatched(dispatch) => self.on_dispatched(dispatch, now),
                Event::Started(process) => self.on_started(process, now),
                Event::Paused(pause) => self.on_paused(pause, now),
                Event::Exited(exit) => self.on_exited(exit, now),
                Event::Closed(session) => self.on_closed(session, now),
                Event::Retained(retention) => self.on_retained(retention, now),
                Event::Evicted(eviction) => self.on_evicted(eviction, now),
                Event::WorkerChanged(worker) => self.on_worker_changed(worker, now),
                Event::HintChanged(change) => self.on_hint_changed(change, now),
                Event::NoteChanged(change) => self.on_note_changed(change, now),
                Event::Refused => self.on_refused(now),
                Event::Alarm => self.on_alarm(now),
            }
        }
        self.plan(now);
    }

    fn on_arrived(&mut self, process: &ProcessId, now: Mark) {}
    fn on_dispatched(&mut self, dispatch: &Dispatch, now: Mark) {}
    fn on_started(&mut self, process: &ProcessId, now: Mark) {}
    fn on_paused(&mut self, pause: &Pause, now: Mark) {}
    fn on_exited(&mut self, exit: &Exit, now: Mark) {}
    fn on_closed(&mut self, session: &SessionId, now: Mark) {}
    fn on_retained(&mut self, retention: &Retention, now: Mark) {}
    fn on_evicted(&mut self, eviction: &Eviction, now: Mark) {}
    fn on_worker_changed(&mut self, worker: &WorkerId, now: Mark) {}
    fn on_hint_changed(&mut self, change: &Change, now: Mark) {}
    fn on_note_changed(&mut self, change: &Change, now: Mark) {}
    fn on_refused(&mut self, now: Mark) {}
    fn on_alarm(&mut self, now: Mark) {}

    /// Called once per call, after the handlers.
    fn plan(&mut self, now: Mark) {}
}

impl Prefix {
    /// All of a working set.
    pub fn all(working_set: impl Into<WorkingSetId>) -> Prefix {
        Prefix {
            working_set: working_set.into(),
            tokens: None,
        }
    }

    /// Its first `tokens` tokens, from the start of the prompt.
    pub fn upto(working_set: impl Into<WorkingSetId>, tokens: u32) -> Prefix {
        Prefix {
            working_set: working_set.into(),
            tokens: Some(tokens),
        }
    }
}

impl Value {
    pub fn boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// A float, or an integer as one.
    pub fn float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Value::Text(t) => Some(t),
            _ => None,
        }
    }
}

/// Reads of a hint or a note as one type, none if absent or another type.
pub trait Reading {
    fn boolean(&self) -> Option<bool>;
    fn int(&self) -> Option<i64>;
    fn float(&self) -> Option<f64>;
    fn text(&self) -> Option<&str>;
}

impl Reading for Option<Value> {
    fn boolean(&self) -> Option<bool> {
        self.as_ref()?.boolean()
    }
    fn int(&self) -> Option<i64> {
        self.as_ref()?.int()
    }
    fn float(&self) -> Option<f64> {
        self.as_ref()?.float()
    }
    fn text(&self) -> Option<&str> {
        self.as_ref()?.text()
    }
}

/// A process's hint.
pub fn hint(process: &str, name: &str) -> Option<Value> {
    hints::get(&Subject::Process(process.into()), name)
}

/// A session's hint.
pub fn session_hint(session: &str, name: &str) -> Option<Value> {
    hints::get(&Subject::Session(session.into()), name)
}

/// How many of a working set's first tokens `worker` holds, in any tier:
/// its own copy there, or the deepest ancestor's. For a pending process
/// (its working set has its id) these are the prompt tokens that worker
/// would not compute again.
pub fn cached_on(working_set: &str, worker: &str) -> u32 {
    let mut best = 0;
    let mut next = snapshot::get_working_set(working_set);
    // A tree is finite; the bound guards a host that reports a cycle.
    for _ in 0..64 {
        let Some(set) = next else { break };
        best = best.max(set.on(worker).min(set.tokens));
        next = set.parent.as_deref().and_then(snapshot::get_working_set);
    }
    best
}

/// A process's hint, else its session's.
pub fn hint_or_session(process: &snapshot::Process, name: &str) -> Option<Value> {
    hint(&process.id, name).or_else(|| {
        process
            .session
            .as_deref()
            .and_then(|s| session_hint(s, name))
    })
}

impl snapshot::Process {
    pub fn live(&self) -> bool {
        self.state != ProcessState::Exited
    }

    /// Not yet started: pending at the gateway or waiting at its worker.
    pub fn queued(&self) -> bool {
        matches!(self.state, ProcessState::Pending | ProcessState::Waiting)
    }

    pub fn running(&self) -> bool {
        self.state == ProcessState::Running
    }

    /// Admitted and not yet exited: running, or paused after it ran. A
    /// queued process has arrived but not been admitted, and an engine
    /// reads a context kept for it (from its device or a lower tier) only
    /// on admission: a policy that holds KV for a session's next turn
    /// should let it go once that turn has started, not once it arrives.
    pub fn started(&self) -> bool {
        self.live() && !self.queued()
    }

    pub fn tenant(&self) -> &str {
        self.tenant.as_deref().unwrap_or("default")
    }
}

impl snapshot::Worker {
    pub fn room(&self, tier: Tier) -> Option<&snapshot::Room> {
        self.tiers.iter().find(|r| r.tier == tier)
    }

    /// The device's capacity and use, in tokens.
    pub fn device(&self) -> Option<&snapshot::Room> {
        self.room(Tier::Device)
    }
}

impl snapshot::WorkingSet {
    /// Retained: cached with no live owner.
    pub fn retained(&self) -> bool {
        self.owner.is_none()
    }

    /// How many of its first tokens `worker` holds, in any tier.
    pub fn on(&self, worker: &str) -> u32 {
        self.copies
            .iter()
            .filter(|c| c.worker == worker)
            .map(|c| c.tokens)
            .max()
            .unwrap_or(0)
    }
}

pub mod prelude {
    pub use crate::{
        By, Change, Dispatch, Duration, ErrorCode, Event, Eviction, Exit, MICROSECOND, MILLISECOND,
        Mark, Outcome, Pause, Policy, Prefix, ProcessId, ProcessState, Reading, Retention, SECOND,
        SessionId, Subject, Tier, Value, WorkerId, WorkingSetId,
    };
    pub use crate::{
        alarm, cached_on, hint, hint_or_session, hints, notes, placement, replication, retention,
        schedule, session_hint, snapshot, terminate,
    };
    pub use alloc::string::{String, ToString};
    pub use alloc::vec::Vec;
}

#[doc(hidden)]
pub use bindings::exports::plex::policy::policy::Guest as __Guest;

/// Exports a policy: one instance of the type, made with `Default`, takes
/// every call.
#[macro_export]
macro_rules! export {
    ($policy:ty) => {
        struct __PlexExport;

        impl $crate::__Guest for __PlexExport {
            fn react(events: ::std::vec::Vec<$crate::Event>, now: $crate::Mark) {
                ::std::thread_local! {
                    static POLICY: ::std::cell::RefCell<$policy> = ::std::cell::RefCell::new(<$policy as ::std::default::Default>::default());
                }
                POLICY.with(|policy| $crate::Policy::react(&mut *policy.borrow_mut(), &events, now));
            }
        }

        $crate::bindings::__export_policy!(__PlexExport with_types_in $crate::bindings);
    };
}
