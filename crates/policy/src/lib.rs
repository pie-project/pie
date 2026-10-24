//! Writing a PLEX serving policy for pie (`plex:policy@1.0.0`).
//!
//! Implement [`worker::Policy`] or [`cluster::Policy`] on a `Default` type and
//! export it with [`worker!`] or [`cluster!`]; `policy.toml` beside `Cargo.toml`
//! is embedded as the manifest. A host restarts an instance that traps or
//! overruns its deadline, without its state.

/// One binding for both worlds: a component imports only what its code calls.
#[doc(hidden)]
pub mod bindings {
    wit_bindgen::generate!({
        path: "wit",
        inline: "
            package plex:support;

            world support {
                include plex:policy/worker@1.0.0;
                include plex:policy/cluster@1.0.0;
            }
        ",
        world: "plex:support/support",
        pub_export_macro: true,
        export_macro_name: "__export",
        generate_all,
    });
}

pub use wit_bindgen;

pub use bindings::plex::policy::reads::capability;
pub use bindings::plex::policy::types::{
    Accept, By, Change, Dispatch, Edge, Exit, Offload, OffloadEvent, OffloadMode, Outcome,
    Placement, Replicate, Residency, Retain, Scale, Site, Subject, SubjectKind, Tier,
    TransferClass, Transition, Value,
};
pub use bindings::plex::policy::writes::{set_alarm, terminate};

pub type Id = String;

pub fn candidates(kind: SubjectKind) -> Vec<Id> {
    bindings::plex::policy::reads::candidates(kind)
}

/// A host-measured value of `of`, if it has one.
pub fn attr(of: &Subject, name: &str) -> Lookup {
    Lookup(bindings::plex::policy::reads::attr(of, name))
}

/// A value supplied from outside the host, if any.
pub fn prop(of: &Subject, name: &str) -> Lookup {
    Lookup(bindings::plex::policy::reads::prop(of, name))
}

impl Retain {
    pub fn all(working_set: impl Into<Id>, key: i64) -> Self {
        Self {
            working_set: working_set.into(),
            key,
            upto: None,
        }
    }

    /// `key` on the first `tokens` tokens only.
    pub fn upto(working_set: impl Into<Id>, key: i64, tokens: u32) -> Self {
        Self {
            working_set: working_set.into(),
            key,
            upto: Some(tokens),
        }
    }
}

/// What a read found, read as the type the caller expects.
#[derive(Debug, Clone)]
pub struct Lookup(pub Option<Value>);

impl Lookup {
    pub fn value(self) -> Option<Value> {
        self.0
    }

    pub fn is_some(&self) -> bool {
        self.0.is_some()
    }

    pub fn int(&self) -> Option<i64> {
        self.0.as_ref()?.int()
    }

    pub fn float(&self) -> Option<f64> {
        self.0.as_ref()?.float()
    }

    pub fn boolean(&self) -> Option<bool> {
        self.0.as_ref()?.boolean()
    }

    pub fn text(&self) -> Option<&str> {
        self.0.as_ref()?.text()
    }

    pub fn json(&self) -> Option<&str> {
        self.0.as_ref()?.json()
    }
}

impl Value {
    pub fn int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// An int reads as a float too.
    pub fn float(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Value::Text(v) => Some(v),
            _ => None,
        }
    }

    /// A prop that was a JSON array or object, as its JSON text.
    pub fn json(&self) -> Option<&str> {
        match self {
            Value::Json(v) => Some(v),
            _ => None,
        }
    }
}

/// The worker world: one instance inside each worker. When a worker's passes
/// run, and where its state sits within it.
pub mod worker {
    use crate::{By, Change, Edge, Exit, Id, OffloadEvent};

    pub use crate::bindings::plex::policy::worker_writes::{
        discard, install_offload, install_retention, install_schedule, publish,
    };

    /// One handler per edge kind, then [`Policy::plan`]; every method defaults
    /// to nothing.
    pub trait Policy: Default + 'static {
        fn on_arrived(&mut self, _process: &Id) {}
        fn on_started(&mut self, _process: &Id) {}
        fn on_paused(&mut self, _process: &Id) {}
        fn on_exited(&mut self, _exit: &Exit) {}
        fn on_retained(&mut self, _working_set: &Id) {}
        fn on_discarded(&mut self, _working_set: &Id, _by: By) {}
        fn on_offloaded(&mut self, _offload: &OffloadEvent) {}
        fn on_offload_failed(&mut self, _process: &Id) {}
        fn on_prop_changed(&mut self, _change: &Change) {}
        fn on_closed(&mut self, _session: &Id) {}
        fn on_timer(&mut self) {}

        /// Once per `react`, after every edge: where a policy installs its
        /// tables.
        fn plan(&mut self, _now_ms: u64) {}
    }

    #[doc(hidden)]
    pub fn react<P: Policy>(policy: &mut P, edges: Vec<Edge>, now_ms: u64) {
        for edge in &edges {
            match edge {
                Edge::Arrived(process) => policy.on_arrived(process),
                Edge::Started(process) => policy.on_started(process),
                Edge::Paused(process) => policy.on_paused(process),
                Edge::Exited(exit) => policy.on_exited(exit),
                Edge::Retained(working_set) => policy.on_retained(working_set),
                Edge::Discarded(t) => policy.on_discarded(&t.subject, t.by),
                Edge::Offloaded(offload) => policy.on_offloaded(offload),
                Edge::OffloadFailed(process) => policy.on_offload_failed(process),
                Edge::PropChanged(change) => policy.on_prop_changed(change),
                Edge::Closed(session) => policy.on_closed(session),
                Edge::Timer => policy.on_timer(),
                // Cluster edges: a worker host does not send them.
                Edge::Dispatched(_)
                | Edge::WorkerRemoved(_)
                | Edge::WorkerDraining(_)
                | Edge::Migrated(_)
                | Edge::MigrationFailed(_)
                | Edge::Copied(_)
                | Edge::Dropped(_)
                | Edge::TierPressure(_)
                | Edge::LinkChanged(_)
                | Edge::PlacementChanged(_)
                | Edge::ReplicationChanged(_)
                | Edge::RetentionChanged(_)
                | Edge::Published(_) => {}
            }
        }
        policy.plan(now_ms);
    }
}

/// The cluster world: one instance in the gateway. Where programs and their
/// state live.
pub mod cluster {
    use crate::{Change, Dispatch, Edge, Exit, Id, Residency, Subject};

    pub use crate::bindings::plex::policy::cluster_writes::{
        install_placement, install_scale, install_schedule, publish,
    };
    pub use crate::bindings::plex::policy::state_writes::{
        discard, install_acceptance, install_replication, install_retention,
        install_transfer_shares,
    };

    /// One handler per edge kind, then [`Policy::plan`].
    pub trait Policy: Default + 'static {
        fn on_arrived(&mut self, _process: &Id) {}
        fn on_dispatched(&mut self, _dispatch: &Dispatch) {}
        fn on_migrated(&mut self, _dispatch: &Dispatch) {}
        fn on_migration_failed(&mut self, _process: &Id) {}
        fn on_exited(&mut self, _exit: &Exit) {}
        fn on_prop_changed(&mut self, _change: &Change) {}
        fn on_closed(&mut self, _session: &Id) {}
        fn on_worker_removed(&mut self, _worker: &Id) {}
        fn on_worker_draining(&mut self, _worker: &Id) {}
        fn on_copied(&mut self, _residency: &Residency) {}
        fn on_dropped(&mut self, _residency: &Residency) {}
        fn on_tier_pressure(&mut self, _worker: &Id) {}
        fn on_link_changed(&mut self, _link: &Id) {}
        /// Another program changed this process's committed placement (a
        /// call, or its death), since this program's last call (attrs
        /// `placement/*`).
        fn on_placement_changed(&mut self, _process: &Id) {}
        /// Another program changed this group's committed replication (a
        /// call, or its death), since this program's last call (attr
        /// `replication/sites`).
        fn on_replication_changed(&mut self, _group: &Id) {}
        /// Another program changed this group's committed retention key (a
        /// call, or its death), since this program's last call (attr
        /// `retention/key`).
        fn on_retention_changed(&mut self, _group: &Id) {}
        /// Another program published or withdrew a value about this subject
        /// (a call, or its death), since this program's last call (attrs
        /// `published/<program>/<name>`).
        fn on_published(&mut self, _subject: &Subject) {}
        fn on_timer(&mut self) {}

        fn plan(&mut self, _now_ms: u64) {}
    }

    #[doc(hidden)]
    pub fn react<P: Policy>(policy: &mut P, edges: Vec<Edge>, now_ms: u64) {
        for edge in &edges {
            match edge {
                Edge::Arrived(process) => policy.on_arrived(process),
                Edge::Dispatched(dispatch) => policy.on_dispatched(dispatch),
                Edge::Migrated(dispatch) => policy.on_migrated(dispatch),
                Edge::MigrationFailed(process) => policy.on_migration_failed(process),
                Edge::Exited(exit) => policy.on_exited(exit),
                Edge::PropChanged(change) => policy.on_prop_changed(change),
                Edge::Closed(session) => policy.on_closed(session),
                Edge::WorkerRemoved(worker) => policy.on_worker_removed(worker),
                Edge::WorkerDraining(worker) => policy.on_worker_draining(worker),
                Edge::Copied(residency) => policy.on_copied(residency),
                Edge::Dropped(residency) => policy.on_dropped(residency),
                Edge::TierPressure(worker) => policy.on_tier_pressure(worker),
                Edge::LinkChanged(link) => policy.on_link_changed(link),
                Edge::PlacementChanged(process) => policy.on_placement_changed(process),
                Edge::ReplicationChanged(group) => policy.on_replication_changed(group),
                Edge::RetentionChanged(group) => policy.on_retention_changed(group),
                Edge::Published(subject) => policy.on_published(subject),
                Edge::Timer => policy.on_timer(),
                // Worker edges: a cluster host does not send them.
                Edge::Started(_)
                | Edge::Paused(_)
                | Edge::Retained(_)
                | Edge::Discarded(_)
                | Edge::Offloaded(_)
                | Edge::OffloadFailed(_) => {}
            }
        }
        policy.plan(now_ms);
    }
}

/// Exports `$policy`, a [`worker::Policy`], as a worker-world component.
#[macro_export]
macro_rules! worker {
    ($policy:ty) => {
        $crate::__policy!($policy, worker);
    };
}

/// Exports `$policy`, a [`cluster::Policy`], as a cluster-world component.
#[macro_export]
macro_rules! cluster {
    ($policy:ty) => {
        $crate::__policy!($policy, cluster);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __policy {
    ($policy:ty, $world:ident) => {
        #[used]
        #[unsafe(link_section = "plex.manifest")]
        static __PLEX_MANIFEST: [u8; include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/policy.toml"
        ))
        .len()] = *include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/policy.toml"));

        struct __PlexPolicy;

        // Not a thread-local: under WASI 0.3 each call is a task of its own,
        // and a task's thread-locals start afresh.
        static __PLEX_STATE: ::std::sync::Mutex<::std::option::Option<$policy>> =
            ::std::sync::Mutex::new(::std::option::Option::None);

        impl $crate::bindings::exports::plex::policy::policy::Guest for __PlexPolicy {
            fn react(edges: ::std::vec::Vec<$crate::Edge>, now_ms: u64) {
                let mut state = __PLEX_STATE
                    .lock()
                    .expect("plex state lock poisoned");
                let policy = state.get_or_insert_with(::std::default::Default::default);
                $crate::$world::react(policy, edges, now_ms);
            }
        }

        $crate::bindings::__export!(__PlexPolicy with_types_in $crate::bindings);
    };
}
