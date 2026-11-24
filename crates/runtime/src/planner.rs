#[macro_use]
mod trace;

mod acquire;
mod departure;
mod diagnostics;
mod error;
mod eviction;
mod exec;
mod global;
mod grant;
mod holdings;
mod in_flight;
mod inner;
mod lock;
mod plan;
mod pool;
mod proc;
mod rank;
mod reclaim;
mod report;
mod residency_planner;
mod short;
mod starvation;
mod stats;
mod to_end;
mod victim;
mod wait;

#[cfg(test)]
mod pressure_tests;
#[cfg(test)]
mod seat_tests;
#[cfg(test)]
mod service_order_tests;
#[cfg(test)]
mod sim_tests;
#[cfg(test)]
mod starvation_race_tests;

pub use acquire::{Acquired, BindRoomWait};
pub use departure::{Departure, DepartureWatch, follow_restart};
pub use diagnostics::{PlannerDiagnostics, PlannerQueueEntry};
pub use error::{ADMISSION_REFUSED, PlannerError, StarveCause, TIERS_FULL};
pub(crate) use global::reset;
pub use global::{
    assign, engine_of, init_planner, persist_run, place, planner, planner_for, planner_of,
    planners, lead_result, rekey_holders, release, to_end_facts, wake,
};
pub use grant::{AllocationGrant, Demand};
pub use in_flight::FireInFlight;
pub use pool::{ChunkMap, ChunkStats, PoolPort, RegistryPool, RiseRoom};
pub use residency_planner::ResidencyPlanner;
pub use short::WeightShed;
pub use stats::PlannerStats;
pub use trace::trace_enabled;

pub type ProcessId = uuid::Uuid;

// The planner's maps and sets by process: under test they iterate in pid
// order, so a seeded simulation's victim picks, ties and every order read
// from them are the same on each run ([`sim_tests`]); in service they hash.
#[cfg(test)]
pub(crate) type PidMap<V> = std::collections::BTreeMap<ProcessId, V>;
#[cfg(not(test))]
pub(crate) type PidMap<V> = std::collections::HashMap<ProcessId, V>;
#[cfg(test)]
pub(crate) type PidSet = std::collections::BTreeSet<ProcessId>;
#[cfg(not(test))]
pub(crate) type PidSet = std::collections::HashSet<ProcessId>;
