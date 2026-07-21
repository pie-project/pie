use std::collections::BTreeMap;
use std::sync::Arc;

use pie_plex::{
    AdmissionInput, AdmissionOutput, DenseScores, EvictionInput, FeedbackAcknowledgement,
    FeedbackBatch, MapHandle, MapKey, Operation, PlacementInput, Revision, ScheduleInput,
    SelectedEviction, SelectedService, ServicePlan, TypedValue,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AttachmentRegistry, Clock, Invocation, InvocationFailureKind, ManualClock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Enactment {
    Commit,
    Abort,
}

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
    Admit {
        input: AdmissionInput,
        enactment: Enactment,
    },
    Route {
        input: PlacementInput,
        enactment: Enactment,
    },
    Schedule {
        input: ScheduleInput,
        enactment: Enactment,
    },
    Evict {
        input: EvictionInput,
        enactment: Enactment,
    },
    Feedback {
        input: FeedbackBatch,
    },
    PublishExternal {
        package: String,
        handle: MapHandle,
        entries: Vec<(MapKey, TypedValue)>,
    },
    ReadMap {
        package: String,
        handle: MapHandle,
        key: MapKey,
    },
    AdvanceClock {
        millis: u64,
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
    AdmissionCommitted {
        output: AdmissionOutput,
        revision: Option<Revision>,
        attempts: u32,
    },
    AdmissionAborted {
        output: AdmissionOutput,
        attempts: u32,
    },
    RouteCommitted {
        output: DenseScores,
        order: Vec<u32>,
        revision: Option<Revision>,
        attempts: u32,
    },
    RouteAborted {
        output: DenseScores,
        order: Vec<u32>,
        attempts: u32,
    },
    ScheduleCommitted {
        decision: ServicePlan,
        selected: Vec<SelectedService>,
        revision: Option<Revision>,
        attempts: u32,
    },
    ScheduleAborted {
        decision: ServicePlan,
        selected: Vec<SelectedService>,
        attempts: u32,
    },
    EvictionCommitted {
        output: DenseScores,
        selected: Vec<SelectedEviction>,
        revision: Option<Revision>,
        attempts: u32,
    },
    EvictionAborted {
        output: DenseScores,
        selected: Vec<SelectedEviction>,
        attempts: u32,
    },
    Feedback {
        acknowledgement: FeedbackAcknowledgement,
    },
    Unavailable {
        operation: Operation,
    },
    FallbackRequired {
        operation: Operation,
        kind: InvocationFailureKind,
    },
    ExternalPublished {
        revision: Revision,
    },
    MapValue {
        value: Option<TypedValue>,
    },
    ClockAdvanced {
        now_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub outcomes: Vec<ReplayOutcome>,
}

pub struct ReplayRunner {
    registry: AttachmentRegistry,
    clock: Arc<ManualClock>,
    packages: BTreeMap<String, Vec<u8>>,
}

impl ReplayRunner {
    pub fn new(
        registry: AttachmentRegistry,
        clock: Arc<ManualClock>,
        packages: BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, ReplaySetupError> {
        if registry.uses_realtime_epochs() {
            return Err(ReplaySetupError::RealtimeEpochs);
        }
        let registry_clock: Arc<dyn Clock> = clock.clone();
        if !registry.uses_clock(&registry_clock) {
            return Err(ReplaySetupError::ClockMismatch);
        }
        Ok(Self {
            registry,
            clock,
            packages,
        })
    }

    pub fn run(&self, trace: &ReplayTrace) -> Result<ReplayReport, ReplayError> {
        let mut outcomes = Vec::with_capacity(trace.commands.len());
        for (index, command) in trace.commands.iter().enumerate() {
            let outcome = self
                .execute(command)
                .map_err(|message| ReplayError { index, message })?;
            outcomes.push(outcome);
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
            decision_element: None,
            map_operation: None,
            expected_revision: expected
                .outcomes
                .get(error.index)
                .and_then(outcome_revision),
            actual_revision: None,
            expected: expected.outcomes.get(error.index).cloned().map(Box::new),
            actual: None,
            detail: error.message,
        })?;
        let length = expected.outcomes.len().max(actual.outcomes.len());
        for index in 0..length {
            let expected_outcome = expected.outcomes.get(index);
            let actual_outcome = actual.outcomes.get(index);
            if expected_outcome != actual_outcome {
                let (decision_element, map_operation, detail) =
                    mismatch_detail(expected_outcome, actual_outcome);
                return Err(ReplayDivergence {
                    index,
                    operation: trace.commands.get(index).and_then(command_operation),
                    decision_element,
                    map_operation,
                    expected_revision: expected_outcome.and_then(outcome_revision),
                    actual_revision: actual_outcome.and_then(outcome_revision),
                    expected: expected_outcome.cloned().map(Box::new),
                    actual: actual_outcome.cloned().map(Box::new),
                    detail,
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
            ReplayCommand::Admit { input, enactment } => {
                let snapshot = self
                    .registry
                    .snapshot()
                    .map_err(|error| error.to_string())?;
                Ok(match snapshot.admit(input.clone()) {
                    Invocation::Success(prepared) => {
                        let attempts = prepared.attempts();
                        match enactment {
                            Enactment::Commit => {
                                let (output, commit) = prepared.commit();
                                ReplayOutcome::AdmissionCommitted {
                                    output,
                                    revision: commit.revision,
                                    attempts,
                                }
                            }
                            Enactment::Abort => ReplayOutcome::AdmissionAborted {
                                output: prepared.abort(),
                                attempts,
                            },
                        }
                    }
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Admit,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Admit,
                        kind: failure.kind,
                    },
                })
            }
            ReplayCommand::Route { input, enactment } => {
                let snapshot = self
                    .registry
                    .snapshot()
                    .map_err(|error| error.to_string())?;
                Ok(match snapshot.route(input.clone()) {
                    Invocation::Success(prepared) => {
                        let attempts = prepared.attempts();
                        let order = pie_plex::rank_placements(input, prepared.decision())
                            .map_err(|error| error.to_string())?;
                        match enactment {
                            Enactment::Commit => {
                                let (output, commit) = prepared.commit();
                                ReplayOutcome::RouteCommitted {
                                    output,
                                    order,
                                    revision: commit.revision,
                                    attempts,
                                }
                            }
                            Enactment::Abort => ReplayOutcome::RouteAborted {
                                output: prepared.abort(),
                                order,
                                attempts,
                            },
                        }
                    }
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Route,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Route,
                        kind: failure.kind,
                    },
                })
            }
            ReplayCommand::Schedule { input, enactment } => {
                let snapshot = self
                    .registry
                    .snapshot()
                    .map_err(|error| error.to_string())?;
                Ok(match snapshot.schedule(input.clone()) {
                    Invocation::Success(prepared) => {
                        let attempts = prepared.attempts();
                        let selected = pie_plex::select_service(input, prepared.decision())
                            .map_err(|error| error.to_string())?;
                        match enactment {
                            Enactment::Commit => {
                                let (decision, commit) = prepared.commit();
                                ReplayOutcome::ScheduleCommitted {
                                    decision,
                                    selected,
                                    revision: commit.revision,
                                    attempts,
                                }
                            }
                            Enactment::Abort => ReplayOutcome::ScheduleAborted {
                                decision: prepared.abort(),
                                selected,
                                attempts,
                            },
                        }
                    }
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Schedule,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Schedule,
                        kind: failure.kind,
                    },
                })
            }
            ReplayCommand::Evict { input, enactment } => {
                let snapshot = self
                    .registry
                    .snapshot()
                    .map_err(|error| error.to_string())?;
                Ok(match snapshot.evict(input.clone()) {
                    Invocation::Success(prepared) => {
                        let attempts = prepared.attempts();
                        let selected = pie_plex::select_evictions(input, prepared.decision())
                            .map_err(|error| error.to_string())?;
                        match enactment {
                            Enactment::Commit => {
                                let (output, commit) = prepared.commit();
                                ReplayOutcome::EvictionCommitted {
                                    output,
                                    selected,
                                    revision: commit.revision,
                                    attempts,
                                }
                            }
                            Enactment::Abort => ReplayOutcome::EvictionAborted {
                                output: prepared.abort(),
                                selected,
                                attempts,
                            },
                        }
                    }
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Evict,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Evict,
                        kind: failure.kind,
                    },
                })
            }
            ReplayCommand::Feedback { input } => {
                let snapshot = self
                    .registry
                    .snapshot()
                    .map_err(|error| error.to_string())?;
                Ok(match snapshot.feedback(input.clone()) {
                    Invocation::Success(acknowledgement) => {
                        ReplayOutcome::Feedback { acknowledgement }
                    }
                    Invocation::Unavailable => ReplayOutcome::Unavailable {
                        operation: Operation::Feedback,
                    },
                    Invocation::FallbackRequired(failure) => ReplayOutcome::FallbackRequired {
                        operation: Operation::Feedback,
                        kind: failure.kind,
                    },
                })
            }
            ReplayCommand::PublishExternal {
                package,
                handle,
                entries,
            } => self
                .registry
                .publish_external(package, *handle, entries.clone())
                .map(|revision| ReplayOutcome::ExternalPublished { revision })
                .map_err(|error| error.to_string()),
            ReplayCommand::ReadMap {
                package,
                handle,
                key,
            } => self
                .registry
                .map_store(package)
                .and_then(|store| store.read(*handle, key).map_err(crate::RegistryError::Map))
                .map(|value| ReplayOutcome::MapValue { value })
                .map_err(|error| error.to_string()),
            ReplayCommand::AdvanceClock { millis } => {
                self.clock.advance(*millis);
                Ok(ReplayOutcome::ClockAdvanced {
                    now_ms: Clock::now_ms(self.clock.as_ref()),
                })
            }
        }
    }

    fn package(&self, name: &str) -> Result<&[u8], String> {
        self.packages
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("replay package {name:?} is not registered"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReplaySetupError {
    #[error("replay requires PolicyEngineConfig::deterministic_replay()")]
    RealtimeEpochs,
    #[error("replay clock is not the clock attached to the policy registry")]
    ClockMismatch,
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
    pub decision_element: Option<usize>,
    pub map_operation: Option<usize>,
    pub expected_revision: Option<Revision>,
    pub actual_revision: Option<Revision>,
    pub expected: Option<Box<ReplayOutcome>>,
    pub actual: Option<Box<ReplayOutcome>>,
    pub detail: String,
}

fn command_operation(command: &ReplayCommand) -> Option<Operation> {
    match command {
        ReplayCommand::Admit { .. } => Some(Operation::Admit),
        ReplayCommand::Route { .. } => Some(Operation::Route),
        ReplayCommand::Schedule { .. } => Some(Operation::Schedule),
        ReplayCommand::Evict { .. } => Some(Operation::Evict),
        ReplayCommand::Feedback { .. } => Some(Operation::Feedback),
        _ => None,
    }
}

fn outcome_revision(outcome: &ReplayOutcome) -> Option<Revision> {
    match outcome {
        ReplayOutcome::AdmissionCommitted { revision, .. }
        | ReplayOutcome::RouteCommitted { revision, .. }
        | ReplayOutcome::ScheduleCommitted { revision, .. }
        | ReplayOutcome::EvictionCommitted { revision, .. } => *revision,
        ReplayOutcome::Feedback {
            acknowledgement:
                FeedbackAcknowledgement::Committed(ack) | FeedbackAcknowledgement::Duplicate(ack),
        } => Some(ack.revision),
        ReplayOutcome::ExternalPublished { revision } => Some(*revision),
        _ => None,
    }
}

fn mismatch_detail(
    expected: Option<&ReplayOutcome>,
    actual: Option<&ReplayOutcome>,
) -> (Option<usize>, Option<usize>, String) {
    let decision_element = match (expected, actual) {
        (
            Some(
                ReplayOutcome::RouteCommitted {
                    output: expected, ..
                }
                | ReplayOutcome::RouteAborted {
                    output: expected, ..
                }
                | ReplayOutcome::EvictionCommitted {
                    output: expected, ..
                }
                | ReplayOutcome::EvictionAborted {
                    output: expected, ..
                },
            ),
            Some(
                ReplayOutcome::RouteCommitted { output: actual, .. }
                | ReplayOutcome::RouteAborted { output: actual, .. }
                | ReplayOutcome::EvictionCommitted { output: actual, .. }
                | ReplayOutcome::EvictionAborted { output: actual, .. },
            ),
        ) => first_difference(&expected.scores, &actual.scores),
        (
            Some(
                ReplayOutcome::ScheduleCommitted {
                    decision: expected, ..
                }
                | ReplayOutcome::ScheduleAborted {
                    decision: expected, ..
                },
            ),
            Some(
                ReplayOutcome::ScheduleCommitted {
                    decision: actual, ..
                }
                | ReplayOutcome::ScheduleAborted {
                    decision: actual, ..
                },
            ),
        ) => first_difference(&expected.decisions, &actual.decisions),
        _ => None,
    };
    let expected_mutations = expected.and_then(outcome_mutations);
    let actual_mutations = actual.and_then(outcome_mutations);
    let map_operation = match (expected_mutations, actual_mutations) {
        (Some(expected), Some(actual)) => first_difference(expected, actual),
        _ => None,
    };
    let detail = if decision_element.is_some() {
        "first decision element mismatch"
    } else if map_operation.is_some() {
        "first staged map operation mismatch"
    } else {
        "first replay outcome mismatch"
    };
    (decision_element, map_operation, detail.into())
}

fn outcome_mutations(outcome: &ReplayOutcome) -> Option<&[pie_plex::MapMutation]> {
    match outcome {
        ReplayOutcome::AdmissionCommitted { output, .. }
        | ReplayOutcome::AdmissionAborted { output, .. } => Some(&output.mutations),
        ReplayOutcome::RouteCommitted { output, .. }
        | ReplayOutcome::RouteAborted { output, .. }
        | ReplayOutcome::EvictionCommitted { output, .. }
        | ReplayOutcome::EvictionAborted { output, .. } => Some(&output.mutations),
        ReplayOutcome::ScheduleCommitted { decision, .. }
        | ReplayOutcome::ScheduleAborted { decision, .. } => Some(&decision.mutations),
        _ => None,
    }
}

fn first_difference<T: PartialEq>(expected: &[T], actual: &[T]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

#[cfg(test)]
mod tests {
    use pie_plex::{ServiceDecision, ServicePlan};

    use super::*;

    #[test]
    fn diagnostics_locate_first_decision_difference() {
        let outcome = |score| ReplayOutcome::ScheduleAborted {
            decision: ServicePlan {
                decisions: vec![ServiceDecision {
                    score,
                    token_budget: None,
                }],
                mutations: Vec::new(),
            },
            selected: Vec::new(),
            attempts: 1,
        };
        let expected = outcome(1.0);
        let actual = outcome(2.0);
        let (decision, mutation, detail) = mismatch_detail(Some(&expected), Some(&actual));
        assert_eq!(decision, Some(0));
        assert_eq!(mutation, None);
        assert_eq!(detail, "first decision element mismatch");
    }
}
