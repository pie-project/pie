use std::collections::BTreeMap;

use pie_plex::{Document, Operation};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    AttachmentRegistry, CanonicalRequestStore, Invocation, InvocationFailureKind, JsonResponse,
    LifecycleHost, PlacementOutcome,
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
        input: Document,
    },
    RouteAdmit {
        logical_request_id: String,
        cause: String,
        candidates: Vec<Document>,
        context: Document,
    },
    FeedbackRemove {
        input: Document,
        terminal_logical_ids: Vec<String>,
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
    RequestStored {
        request: Document,
    },
    Invocation {
        operation: Operation,
        response: JsonResponse,
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
        requests: CanonicalRequestStore,
        packages: BTreeMap<String, Vec<u8>>,
        max_defer_retries: u32,
    ) -> Result<Self, ReplaySetupError> {
        if registry.uses_realtime_epochs() {
            return Err(ReplaySetupError::RealtimeEpochs);
        }
        let lifecycle = LifecycleHost::new(registry.clone(), requests, max_defer_retries);
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
                .requests()
                .create(logical_request_id, body.clone(), metadata.clone())
                .map(|request| ReplayOutcome::RequestStored { request })
                .map_err(|error| error.to_string()),
            ReplayCommand::ContinueRequest {
                logical_request_id,
                body,
                metadata,
            } => self
                .lifecycle
                .requests()
                .continuation(logical_request_id, body.clone(), metadata.clone())
                .map(|request| ReplayOutcome::RequestStored { request })
                .map_err(|error| error.to_string()),
            ReplayCommand::Invoke { operation, input } => {
                let mut input = input.clone();
                hydrate_requests(*operation, &mut input, self.lifecycle.requests())?;
                Ok(match self.lifecycle.invoke_and_apply(*operation, input) {
                    Invocation::Success(response) => ReplayOutcome::Invocation {
                        operation: *operation,
                        selection: selection(*operation, &response)
                            .map_err(|error| error.to_string())?,
                        response,
                    },
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: *operation,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: *operation,
                        kind: failure.kind,
                    },
                })
            }
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
                input,
                terminal_logical_ids,
            } => {
                let mut input = input.clone();
                hydrate_requests(Operation::Feedback, &mut input, self.lifecycle.requests())?;
                Ok(
                    match self
                        .lifecycle
                        .feedback_and_remove(input, terminal_logical_ids)
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
                )
            }
            ReplayCommand::ReadRequest { logical_request_id } => self
                .lifecycle
                .requests()
                .get(logical_request_id)
                .map(|request| ReplayOutcome::RequestStored { request })
                .map_err(|error| error.to_string()),
        }
    }

    fn package(&self, name: &str) -> Result<&[u8], String> {
        self.packages
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("replay package {name:?} is not registered"))
    }
}

fn hydrate_requests(
    operation: Operation,
    input: &mut Value,
    store: &CanonicalRequestStore,
) -> Result<(), String> {
    match operation {
        Operation::Route | Operation::Admit => {
            hydrate(input.get_mut("request"), store)?;
        }
        Operation::Schedule => {
            for candidate in input
                .get_mut("runnable")
                .and_then(Value::as_array_mut)
                .ok_or("missing runnable array")?
            {
                hydrate(candidate.get_mut("request"), store)?;
            }
        }
        Operation::Evict => {
            for unit in input
                .get_mut("resident")
                .and_then(Value::as_array_mut)
                .ok_or("missing resident array")?
            {
                if unit
                    .get("request")
                    .is_some_and(|request| !request.is_null())
                {
                    hydrate(unit.get_mut("request"), store)?;
                }
            }
        }
        Operation::Feedback => {
            for record in input
                .get_mut("records")
                .and_then(Value::as_array_mut)
                .ok_or("missing records array")?
            {
                hydrate(record.get_mut("request"), store)?;
            }
        }
    }
    Ok(())
}

fn hydrate(request: Option<&mut Value>, store: &CanonicalRequestStore) -> Result<(), String> {
    let request = request.ok_or("missing request")?;
    let logical_id = request
        .pointer("/identity/logical_request_id")
        .and_then(Value::as_str)
        .ok_or("request placeholder has no logical_request_id")?;
    *request = store.get(logical_id).map_err(|error| error.to_string())?;
    Ok(())
}

fn selection(operation: Operation, response: &JsonResponse) -> Result<Option<Document>, String> {
    let selection = match operation {
        Operation::Route => Some(serde_json::json!({
            "order": pie_plex::rank_route(
                &response.result,
                response.input["candidates"].as_array().expect("validated").len()
            ).map_err(|error| error.to_string())?
        })),
        Operation::Admit => Some(serde_json::json!({
            "decision": pie_plex::validate_admit(&response.result)
                .map_err(|error| error.to_string())?
        })),
        Operation::Schedule => Some(
            serde_json::to_value(
                pie_plex::select_schedule(&response.input, &response.result)
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
        ),
        Operation::Evict => Some(
            serde_json::to_value(
                pie_plex::select_evictions(&response.input, &response.result)
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
