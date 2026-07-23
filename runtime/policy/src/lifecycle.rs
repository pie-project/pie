use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use pie_plex::Document;
use pie_plex::v0_5::{AdmissionDecision, Operation, rank_route, validate_admit};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::host::{QueryHandler, RejectingQueryHandler};
use crate::protocol::validate_context;
use crate::state_store::{FeedbackCommit, PolicyStateBackend, StateBackendError, StateSnapshot};
use crate::{
    AttachmentRegistry, Invocation, InvocationFailure, InvocationFailureKind, PreparedPolicyResult,
};

#[derive(Clone)]
pub struct LifecycleHost {
    registry: AttachmentRegistry,
    backend: Arc<dyn PolicyStateBackend>,
    query_handler: Arc<dyn QueryHandler>,
    supported_actions: Arc<BTreeSet<String>>,
    stateful_invocation: Arc<Mutex<()>>,
    max_defer_retries: u32,
}

pub(crate) struct AppliedPolicyResult {
    pub prepared: PreparedPolicyResult,
    pub state_snapshot: Option<StateSnapshot>,
}

impl LifecycleHost {
    pub fn new(
        registry: AttachmentRegistry,
        backend: Arc<dyn PolicyStateBackend>,
        max_defer_retries: u32,
    ) -> Self {
        Self::with_host(
            registry,
            backend,
            Arc::new(RejectingQueryHandler),
            BTreeSet::new(),
            max_defer_retries,
        )
    }

    pub fn with_host(
        registry: AttachmentRegistry,
        backend: Arc<dyn PolicyStateBackend>,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: BTreeSet<String>,
        max_defer_retries: u32,
    ) -> Self {
        Self {
            registry,
            backend,
            query_handler,
            supported_actions: Arc::new(supported_actions),
            stateful_invocation: Arc::new(Mutex::new(())),
            max_defer_retries,
        }
    }

    pub fn backend(&self) -> &Arc<dyn PolicyStateBackend> {
        &self.backend
    }

    pub fn create_request(
        &self,
        logical_request_id: impl Into<String>,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateBackendError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.backend
            .create_request(logical_request_id.into(), body, metadata)
    }

    pub fn continue_request(
        &self,
        logical_request_id: &str,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateBackendError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.backend
            .continue_request(logical_request_id, body, metadata)
    }

    pub fn replace_shared(&self, shared: Document) -> Result<(), StateBackendError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.backend.replace_shared(shared)
    }

    pub fn merge_request_facts(
        &self,
        logical_request_id: &str,
        facts: Document,
    ) -> Result<(), StateBackendError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.backend.merge_request_facts(logical_request_id, facts)
    }

    pub fn record_enacted_placement(
        &self,
        logical_request_id: &str,
        target_id: impl Into<String>,
    ) -> Result<(), StateBackendError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.backend
            .record_enacted_placement(logical_request_id, target_id.into())
    }

    pub fn invoke_and_apply(
        &self,
        operation: Operation,
        context: Document,
    ) -> Invocation<PreparedPolicyResult> {
        let _guard = self.stateful_invocation.lock().unwrap();
        discard_state_snapshot(self.invoke_and_apply_locked_with_state_snapshot(
            operation,
            context,
            &[],
            false,
        ))
    }

    pub(crate) fn invoke_and_apply_with_state_snapshot(
        &self,
        operation: Operation,
        context: Document,
    ) -> Invocation<AppliedPolicyResult> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.invoke_and_apply_locked_with_state_snapshot(operation, context, &[], true)
    }

    fn invoke_and_apply_locked(
        &self,
        operation: Operation,
        context: Document,
    ) -> Invocation<PreparedPolicyResult> {
        discard_state_snapshot(self.invoke_and_apply_locked_with_state_snapshot(
            operation,
            context,
            &[],
            false,
        ))
    }

    fn invoke_and_apply_locked_with_removals(
        &self,
        operation: Operation,
        context: Document,
        terminal_logical_ids: &[String],
    ) -> Invocation<PreparedPolicyResult> {
        discard_state_snapshot(self.invoke_and_apply_locked_with_state_snapshot(
            operation,
            context,
            terminal_logical_ids,
            false,
        ))
    }

    fn invoke_and_apply_locked_with_state_snapshot(
        &self,
        operation: Operation,
        context: Document,
        terminal_logical_ids: &[String],
        backend_load_errors: bool,
    ) -> Invocation<AppliedPolicyResult> {
        let request_ids = match validate_context(operation, &context) {
            Ok(request_ids) => request_ids,
            Err(error) => return invalid_input(error.to_string()),
        };
        let feedback_delivery = if operation == Operation::Feedback {
            let delivery_id = context["delivery_id"]
                .as_str()
                .expect("feedback context was validated");
            match self.backend.feedback_result(delivery_id) {
                Ok(Some(result)) => {
                    return duplicate_feedback(result);
                }
                Ok(None) => Some(delivery_id.to_owned()),
                Err(error) => return backend_failure(error),
            }
        } else {
            None
        };

        let state = match self.backend.load(&request_ids) {
            Ok(state) => state,
            Err(error) => {
                if let Some(delivery_id) = feedback_delivery.as_deref() {
                    match self.backend.feedback_result(delivery_id) {
                        Ok(Some(result)) => return duplicate_feedback(result),
                        Ok(None) => {}
                        Err(recheck_error) => return backend_failure(recheck_error),
                    }
                }
                return if backend_load_errors {
                    backend_failure(error)
                } else {
                    invalid_input(error.to_string())
                };
            }
        };
        let snapshot = match self.registry.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::Instantiation,
                    error.to_string(),
                ));
            }
        };
        let prepared = match snapshot.invoke(
            operation,
            context,
            state.clone(),
            self.query_handler.clone(),
            self.supported_actions.clone(),
        ) {
            Invocation::Success(prepared) => prepared,
            Invocation::Unavailable => return Invocation::Unavailable,
            Invocation::FallbackRequired(failure) => {
                return Invocation::FallbackRequired(failure);
            }
        };
        let feedback = feedback_delivery.map(|delivery_id| FeedbackCommit {
            delivery_id,
            result: prepared.result.clone(),
            maximum_deliveries: self.registry.max_feedback_deliveries(),
        });
        if let Err(error) = self.backend.commit(
            &state,
            &prepared.state_updates,
            feedback.as_ref(),
            terminal_logical_ids,
        ) {
            if let StateBackendError::DuplicateFeedback(delivery_id) = &error
                && let Ok(Some(result)) = self.backend.feedback_result(delivery_id)
            {
                return duplicate_feedback(result);
            }
            return backend_failure(error);
        }

        Invocation::Success(AppliedPolicyResult {
            prepared,
            state_snapshot: Some(state),
        })
    }

    pub fn route_and_admit(
        &self,
        logical_request_id: &str,
        cause: &str,
        candidates: Vec<Document>,
        context: Document,
    ) -> Invocation<PlacementOutcome> {
        let _guard = self.stateful_invocation.lock().unwrap();
        if let Err(error) = self.backend.read_request(logical_request_id) {
            return invalid_input(error.to_string());
        }
        let route_context = json!({
            "cause": cause,
            "request_id": logical_request_id,
            "candidates": candidates.clone(),
            "context": context.clone()
        });
        let route = match self.invoke_and_apply_locked(Operation::Route, route_context) {
            Invocation::Success(response) => response,
            Invocation::Unavailable => return Invocation::Unavailable,
            Invocation::FallbackRequired(failure) => {
                return Invocation::FallbackRequired(failure);
            }
        };
        let order = match rank_route(&route.result, candidates.len()) {
            Ok(order) => order,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
        };
        for index in order {
            let target = candidates[index].clone();
            let target_id = target["id"]
                .as_str()
                .expect("validated target ID")
                .to_owned();
            for defer_attempt in 0..=self.max_defer_retries {
                let admit_context = json!({
                    "cause": cause,
                    "request_id": logical_request_id,
                    "target": target.clone(),
                    "context": context.clone()
                });
                let admit = match self.invoke_and_apply_locked(Operation::Admit, admit_context) {
                    Invocation::Success(response) => response,
                    Invocation::Unavailable => return Invocation::Unavailable,
                    Invocation::FallbackRequired(failure) => {
                        return Invocation::FallbackRequired(failure);
                    }
                };
                let decision = match validate_admit(&admit.result) {
                    Ok(decision) => decision,
                    Err(error) => {
                        return Invocation::FallbackRequired(InvocationFailure::new(
                            InvocationFailureKind::InvalidOutput,
                            error.to_string(),
                        ));
                    }
                };
                match decision {
                    AdmissionDecision::Accept => {
                        let request_map = match self.backend.read_request(logical_request_id) {
                            Ok(request) => request,
                            Err(error) => return backend_failure(error),
                        };
                        return Invocation::Success(PlacementOutcome::Accepted {
                            target_id,
                            request_map,
                            defer_attempts: defer_attempt,
                        });
                    }
                    AdmissionDecision::Reject => break,
                    AdmissionDecision::Defer if defer_attempt < self.max_defer_retries => {}
                    AdmissionDecision::Defer => {
                        let request_map = match self.backend.read_request(logical_request_id) {
                            Ok(request) => request,
                            Err(error) => return backend_failure(error),
                        };
                        return Invocation::Success(PlacementOutcome::Deferred {
                            target_id,
                            request_map,
                            attempts: defer_attempt + 1,
                        });
                    }
                }
            }
        }
        Invocation::Success(PlacementOutcome::Rejected)
    }

    pub fn feedback_and_remove(
        &self,
        context: Document,
        terminal_logical_ids: &[String],
    ) -> Invocation<PreparedPolicyResult> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.invoke_and_apply_locked_with_removals(
            Operation::Feedback,
            context,
            terminal_logical_ids,
        )
    }

    pub(crate) fn feedback_and_remove_with_state_snapshot(
        &self,
        context: Document,
        terminal_logical_ids: &[String],
    ) -> Invocation<AppliedPolicyResult> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.invoke_and_apply_locked_with_state_snapshot(
            Operation::Feedback,
            context,
            terminal_logical_ids,
            true,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum PlacementOutcome {
    Accepted {
        target_id: String,
        request_map: Document,
        defer_attempts: u32,
    },
    Deferred {
        target_id: String,
        request_map: Document,
        attempts: u32,
    },
    Rejected,
}

fn invalid_input<T>(message: String) -> Invocation<T> {
    Invocation::FallbackRequired(InvocationFailure::new(
        InvocationFailureKind::InvalidInput,
        message,
    ))
}

fn duplicate_feedback(result: Document) -> Invocation<AppliedPolicyResult> {
    Invocation::Success(AppliedPolicyResult {
        prepared: PreparedPolicyResult::duplicate_feedback(result),
        state_snapshot: None,
    })
}

fn discard_state_snapshot(
    invocation: Invocation<AppliedPolicyResult>,
) -> Invocation<PreparedPolicyResult> {
    match invocation {
        Invocation::Success(applied) => Invocation::Success(applied.prepared),
        Invocation::Unavailable => Invocation::Unavailable,
        Invocation::FallbackRequired(failure) => Invocation::FallbackRequired(failure),
    }
}

fn backend_failure<T>(error: StateBackendError) -> Invocation<T> {
    let kind = match error {
        StateBackendError::RevisionConflict(_) => InvocationFailureKind::StateConflict,
        _ => InvocationFailureKind::BackendFailure,
    };
    Invocation::FallbackRequired(InvocationFailure::new(kind, error.to_string()))
}
