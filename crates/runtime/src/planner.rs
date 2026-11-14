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
mod service_order_tests;
#[cfg(test)]
mod starvation_race_tests;

pub use acquire::Acquired;
pub use departure::{Departure, DepartureWatch};
pub use diagnostics::{PlannerDiagnostics, PlannerQueueEntry};
pub use error::{PlannerError, StarveCause};
pub use global::{
    assign, engine_of, init_planner, persist_run, place, planner, planner_for, planner_of,
    planners, rekey_holders, release, wake,
};
pub use grant::{AllocationGrant, Demand};
pub use in_flight::FireInFlight;
pub use pool::{ChunkStats, PoolPort, RegistryPool};
pub use residency_planner::ResidencyPlanner;
pub use short::WeightShed;
pub use stats::PlannerStats;
pub use trace::trace_enabled;

pub type ProcessId = uuid::Uuid;
