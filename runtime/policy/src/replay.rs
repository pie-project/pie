use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pie_plex::Document;
use pie_plex::v0_5::Operation;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AttachmentRegistry, Invocation, InvocationFailureKind, LifecycleHost, PlacementOutcome,
    PolicyStateBackend, PreparedPolicyResult, QueryHandler, RejectingQueryHandler,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum ReplayCommand {
    Attach {
        package: String,
    },
    Replace {
        package: String,
    },
    DetachOperation {
        operation: Operation,
    },
    DetachPackage {
        package: String,
    },
    CreateRequest {
        logical_request_id: String,
        body: Document,
        metadata: Document,
    },
    ContinueRequest {
        logical_request_id: String,
        body: Document,
        metadata: Document,
    },
    Invoke {
        operation: Operation,
        context: Document,
    },
    RouteAdmit {
        logical_request_id: String,
        cause: String,
        candidates: Vec<Document>,
        context: Document,
    },
    FeedbackRemove {
        context: Document,
        terminal_logical_ids: Vec<String>,
    },
    ReadShared,
    ReplaceShared {
        shared: Document,
    },
    MergeRequestFacts {
        logical_request_id: String,
        facts: Document,
    },
    RecordEnactedPlacement {
        logical_request_id: String,
        target_id: String,
    },
    ReadRequest {
        logical_request_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTrace {
    pub commands: Vec<ReplayCommand>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "kebab-case")]
pub enum ReplayOutcome {
    AttachmentPublished {
        generation: u64,
    },
    AttachmentDetached {
        generation: u64,
    },
    SharedState {
        shared: Document,
    },
    RequestState {
        request: Document,
    },
    Invocation {
        operation: Operation,
        response: PreparedPolicyResult,
        selection: Option<Document>,
    },
    Placement {
        placement: PlacementOutcome,
    },
    Unavailable {
        operation: Operation,
    },
    FallbackRequired {
        operation: Operation,
        kind: InvocationFailureKind,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub outcomes: Vec<ReplayOutcome>,
}

pub struct ReplayRunner {
    registry: AttachmentRegistry,
    lifecycle: LifecycleHost,
    packages: BTreeMap<String, Vec<u8>>,
}

impl ReplayRunner {
    pub fn new(
        registry: AttachmentRegistry,
        backend: Arc<dyn PolicyStateBackend>,
        packages: BTreeMap<String, Vec<u8>>,
        max_defer_retries: u32,
    ) -> Result<Self, ReplaySetupError> {
        Self::with_host(
            registry,
            backend,
            packages,
            Arc::new(RejectingQueryHandler),
            BTreeSet::new(),
            max_defer_retries,
        )
    }

    pub fn with_host(
        registry: AttachmentRegistry,
        backend: Arc<dyn PolicyStateBackend>,
        packages: BTreeMap<String, Vec<u8>>,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: BTreeSet<String>,
        max_defer_retries: u32,
    ) -> Result<Self, ReplaySetupError> {
        if registry.uses_realtime_epochs() {
            return Err(ReplaySetupError::RealtimeEpochs);
        }
        let lifecycle = LifecycleHost::with_host(
            registry.clone(),
            backend,
            query_handler,
            supported_actions,
            max_defer_retries,
        );
        Ok(Self {
            registry,
            lifecycle,
            packages,
        })
    }

    pub fn run(&self, trace: &ReplayTrace) -> Result<ReplayReport, ReplayError> {
        let mut outcomes = Vec::with_capacity(trace.commands.len());
        for (index, command) in trace.commands.iter().enumerate() {
            outcomes.push(
                self.execute(command)
                    .map_err(|message| ReplayError { index, message })?,
            );
        }
        Ok(ReplayReport { outcomes })
    }

    pub fn verify(
        &self,
        trace: &ReplayTrace,
        expected: &ReplayReport,
    ) -> Result<ReplayReport, ReplayDivergence> {
        let actual = self.run(trace).map_err(|error| ReplayDivergence {
            index: error.index,
            operation: trace.commands.get(error.index).and_then(command_operation),
            expected: expected.outcomes.get(error.index).cloned().map(Box::new),
            actual: None,
            detail: error.message,
        })?;
        let length = expected.outcomes.len().max(actual.outcomes.len());
        for index in 0..length {
            let expected_outcome = expected.outcomes.get(index);
            let actual_outcome = actual.outcomes.get(index);
            if expected_outcome != actual_outcome {
                return Err(ReplayDivergence {
                    index,
                    operation: trace.commands.get(index).and_then(command_operation),
                    expected: expected_outcome.cloned().map(Box::new),
                    actual: actual_outcome.cloned().map(Box::new),
                    detail: "first replay outcome mismatch".into(),
                });
            }
        }
        Ok(actual)
    }

    fn execute(&self, command: &ReplayCommand) -> Result<ReplayOutcome, String> {
        match command {
            ReplayCommand::Attach { package } => self
                .registry
                .attach(self.package(package)?)
                .map(|generation| ReplayOutcome::AttachmentPublished { generation })
                .map_err(|error| error.to_string()),
            ReplayCommand::Replace { package } => self
                .registry
                .replace(self.package(package)?)
                .map(|generation| ReplayOutcome::AttachmentPublished { generation })
                .map_err(|error| error.to_string()),
            ReplayCommand::DetachOperation { operation } => self
                .registry
                .detach_operation(*operation)
                .map(|generation| ReplayOutcome::AttachmentDetached { generation })
                .map_err(|error| error.to_string()),
            ReplayCommand::DetachPackage { package } => self
                .registry
                .detach_package(package)
                .map(|generation| ReplayOutcome::AttachmentDetached { generation })
                .map_err(|error| error.to_string()),
            ReplayCommand::CreateRequest {
                logical_request_id,
                body,
                metadata,
            } => self
                .lifecycle
                .create_request(logical_request_id, body.clone(), metadata.clone())
                .map(|request| ReplayOutcome::RequestState { request })
                .map_err(|error| error.to_string()),
            ReplayCommand::ContinueRequest {
                logical_request_id,
                body,
                metadata,
            } => self
                .lifecycle
                .continue_request(logical_request_id, body.clone(), metadata.clone())
                .map(|request| ReplayOutcome::RequestState { request })
                .map_err(|error| error.to_string()),
            ReplayCommand::Invoke { operation, context } => Ok(
                match self.lifecycle.invoke_and_apply(*operation, context.clone()) {
                    Invocation::Success(response) => ReplayOutcome::Invocation {
                        operation: *operation,
                        selection: selection(*operation, context, &response.result)?,
                        response,
                    },
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: *operation,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: *operation,
                        kind: failure.kind,
                    },
                },
            ),
            ReplayCommand::RouteAdmit {
                logical_request_id,
                cause,
                candidates,
                context,
            } => Ok(
                match self.lifecycle.route_and_admit(
                    logical_request_id,
                    cause,
                    candidates.clone(),
                    context.clone(),
                ) {
                    Invocation::Success(placement) => ReplayOutcome::Placement { placement },
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Route,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Route,
                        kind: failure.kind,
                    },
                },
            ),
            ReplayCommand::FeedbackRemove {
                context,
                terminal_logical_ids,
            } => Ok(
                match self
                    .lifecycle
                    .feedback_and_remove(context.clone(), terminal_logical_ids)
                {
                    Invocation::Success(response) => ReplayOutcome::Invocation {
                        operation: Operation::Feedback,
                        response,
                        selection: None,
                    },
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Feedback,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Feedback,
                        kind: failure.kind,
                    },
                },
            ),
            ReplayCommand::ReadShared => self
                .lifecycle
                .backend()
                .read_shared()
                .map(|shared| ReplayOutcome::SharedState { shared })
                .map_err(|error| error.to_string()),
            ReplayCommand::ReplaceShared { shared } => {
                self.lifecycle
                    .replace_shared(shared.clone())
                    .map_err(|error| error.to_string())?;
                self.lifecycle
                    .backend()
                    .read_shared()
                    .map(|shared| ReplayOutcome::SharedState { shared })
                    .map_err(|error| error.to_string())
            }
            ReplayCommand::MergeRequestFacts {
                logical_request_id,
                facts,
            } => {
                self.lifecycle
                    .merge_request_facts(logical_request_id, facts.clone())
                    .map_err(|error| error.to_string())?;
                self.request_outcome(logical_request_id)
            }
            ReplayCommand::RecordEnactedPlacement {
                logical_request_id,
                target_id,
            } => {
                self.lifecycle
                    .record_enacted_placement(logical_request_id, target_id)
                    .map_err(|error| error.to_string())?;
                self.request_outcome(logical_request_id)
            }
            ReplayCommand::ReadRequest { logical_request_id } => {
                self.request_outcome(logical_request_id)
            }
        }
    }

    fn request_outcome(&self, logical_request_id: &str) -> Result<ReplayOutcome, String> {
        self.lifecycle
            .backend()
            .read_request(logical_request_id)
            .map(|request| ReplayOutcome::RequestState { request })
            .map_err(|error| error.to_string())
    }

    fn package(&self, name: &str) -> Result<&[u8], String> {
        self.packages
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("replay package {name:?} is not registered"))
    }
}

fn selection(
    operation: Operation,
    context: &Document,
    result: &Document,
) -> Result<Option<Document>, String> {
    let selection = match operation {
        Operation::Route => Some(serde_json::json!({
            "order": pie_plex::v0_5::rank_route(
                result,
                context["candidates"].as_array().expect("validated").len()
            ).map_err(|error| error.to_string())?
        })),
        Operation::Admit => Some(serde_json::json!({
            "decision": pie_plex::v0_5::validate_admit(result)
                .map_err(|error| error.to_string())?
        })),
        Operation::Schedule => Some(
            serde_json::to_value(
                pie_plex::v0_5::select_schedule(context, result)
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
        ),
        Operation::Evict => Some(
            serde_json::to_value(
                pie_plex::v0_5::select_evictions(context, result)
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
        ),
        Operation::Feedback => None,
    };
    Ok(selection)
}

fn command_operation(command: &ReplayCommand) -> Option<Operation> {
    match command {
        ReplayCommand::Invoke { operation, .. } => Some(*operation),
        ReplayCommand::RouteAdmit { .. } => Some(Operation::Route),
        ReplayCommand::FeedbackRemove { .. } => Some(Operation::Feedback),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReplaySetupError {
    #[error("replay requires PolicyEngineConfig::deterministic_replay()")]
    RealtimeEpochs,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("replay command {index} failed: {message}")]
pub struct ReplayError {
    pub index: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Error)]
#[error("replay diverged at command {index}: {detail}")]
pub struct ReplayDivergence {
    pub index: usize,
    pub operation: Option<Operation>,
    pub expected: Option<Box<ReplayOutcome>>,
    pub actual: Option<Box<ReplayOutcome>>,
    pub detail: String,
}
