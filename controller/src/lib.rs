mod coordinator;
mod error;
mod health;
mod pairing;
mod role;
mod serve;
mod service;

pub use coordinator::ControllerConfig;
pub use error::{ControllerError, Result};
pub use health::HealthChecker;
pub use pairing::{Pair, PairId, PairingTable};
pub use role::RoleTable;
pub use serve::{ProcessConfig, run_as_process};
pub use service::{ControlApi, ControlApiClient};

pub use pie_schema::{
    HealthStatus, LoadState, Placement, RequestId, RequestMeta, Role, WorkerId, WorkerInfo,
};
