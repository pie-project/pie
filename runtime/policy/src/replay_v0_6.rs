use std::sync::Arc;

use pie_plex::Document;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AttachmentRegistryV0_6, HostSupportV0_6, InMemoryPolicyStateBackendV0_6, PlexRuntimeV0_6,
    PolicyEngine, PolicyEngineConfig, ProtocolLimitsV0_6, QueryHandler, RejectingQueryHandler,
    StateMetricsV0_6,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayReportV0_6 {
    pub outcomes: Vec<Document>,
    pub state_metrics: StateMetricsReportV0_6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateMetricsReportV0_6 {
    pub loads: u64,
    pub auto_joined_groups: u64,
    pub commit_attempts: u64,
    pub commits: u64,
    pub shared_conflicts: u64,
    pub group_conflicts: u64,
    pub request_conflicts: u64,
}

impl From<StateMetricsV0_6> for StateMetricsReportV0_6 {
    fn from(metrics: StateMetricsV0_6) -> Self {
        Self {
            loads: metrics.loads,
            auto_joined_groups: metrics.auto_joined_groups,
            commit_attempts: metrics.commit_attempts,
            commits: metrics.commits,
            shared_conflicts: metrics.shared_conflicts,
            group_conflicts: metrics.group_conflicts,
            request_conflicts: metrics.request_conflicts,
        }
    }
}

#[derive(Clone)]
pub struct ReplayRunnerV0_6 {
    package: Arc<Vec<u8>>,
    query_handler: Arc<dyn QueryHandler>,
    support: HostSupportV0_6,
    protocol_limits: ProtocolLimitsV0_6,
}

impl ReplayRunnerV0_6 {
    pub fn new(package: Vec<u8>, support: HostSupportV0_6) -> Self {
        Self {
            package: Arc::new(package),
            query_handler: Arc::new(RejectingQueryHandler),
            support,
            protocol_limits: ProtocolLimitsV0_6::default(),
        }
    }

    pub fn with_query_handler(mut self, query_handler: Arc<dyn QueryHandler>) -> Self {
        self.query_handler = query_handler;
        self
    }

    pub fn with_protocol_limits(mut self, protocol_limits: ProtocolLimitsV0_6) -> Self {
        self.protocol_limits = protocol_limits;
        self
    }

    pub fn run(&self, events: &[Document]) -> Result<ReplayReportV0_6, ReplayErrorV0_6> {
        let runtime = self.runtime()?;
        let outcomes = events
            .iter()
            .cloned()
            .map(|event| runtime.invoke(event).map_err(ReplayErrorV0_6::Invoke))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ReplayReportV0_6 {
            outcomes,
            state_metrics: runtime.metrics().into(),
        })
    }

    pub fn verify_deterministic(
        &self,
        events: &[Document],
    ) -> Result<ReplayReportV0_6, ReplayErrorV0_6> {
        let first = self.run(events)?;
        let second = self.run(events)?;
        if first != second {
            return Err(ReplayErrorV0_6::Diverged { first, second });
        }
        Ok(first)
    }

    fn runtime(&self) -> Result<PlexRuntimeV0_6, ReplayErrorV0_6> {
        let engine = PolicyEngine::new(PolicyEngineConfig::deterministic_replay())
            .map_err(|error| ReplayErrorV0_6::Setup(error.to_string()))?;
        let registry = AttachmentRegistryV0_6::new(engine, self.support.clone());
        registry
            .attach(&self.package)
            .map_err(|error| ReplayErrorV0_6::Setup(error.to_string()))?;
        PlexRuntimeV0_6::with_parts(
            registry,
            Arc::new(InMemoryPolicyStateBackendV0_6::default()),
            self.query_handler.clone(),
            self.protocol_limits,
        )
        .map_err(|error| ReplayErrorV0_6::Setup(error.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum ReplayErrorV0_6 {
    #[error("failed to create replay runtime: {0}")]
    Setup(String),
    #[error("replay invocation failed")]
    Invoke(#[source] crate::PlexErrorV0_6),
    #[error("deterministic replay diverged")]
    Diverged {
        first: ReplayReportV0_6,
        second: ReplayReportV0_6,
    },
}
