#[macro_use]
mod trace;

mod acquire;
mod diagnostics;
mod error;
mod eviction;
mod exec;
mod global;
mod grant;
mod holdings;
mod inner;
mod lock;
mod plan;
mod pool;
mod proc;
mod rank;
mod reclaim;
mod report;
mod residency_planner;
mod starvation;
mod stats;
mod victim;
mod wait;

#[cfg(test)]
mod service_order_tests;
#[cfg(test)]
mod starvation_race_tests;

pub use acquire::Acquired;
pub use diagnostics::{PlannerDiagnostics, PlannerQueueEntry};
pub use error::{PlannerError, StarveCause};
pub use global::{init_planner, planner, planner_for, wake};
pub use grant::{AllocationGrant, Demand};
pub use pool::{PoolPort, RegistryPool};
pub use residency_planner::ResidencyPlanner;
pub use stats::PlannerStats;
pub use trace::trace_enabled;

pub type ProcessId = uuid::Uuid;
