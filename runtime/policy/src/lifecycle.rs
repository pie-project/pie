use std::sync::{Arc, Mutex};

use pie_plex::{AdmissionDecision, Document, Operation, rank_route, validate_admit};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::protocol::referenced_request_ids;
use crate::{
    AttachmentRegistry, Invocation, InvocationFailure, InvocationFailureKind, JsonResponse,
    PolicyStateStore, StateStoreError,
};

#[derive(Clone)]
pub struct LifecycleHost {
    registry: AttachmentRegistry,
    state: PolicyStateStore,
    stateful_invocation: Arc<Mutex<()>>,
    max_defer_retries: u32,
}

impl LifecycleHost {
    pub fn new(
        registry: AttachmentRegistry,
        state: PolicyStateStore,
        max_defer_retries: u32,
    ) -> Self {
        Self {
            registry,
            state,
            stateful_invocation: Arc::new(Mutex::new(())),
            max_defer_retries,
        }
    }

    pub fn state(&self) -> &PolicyStateStore {
        &self.state
    }

    pub fn create_request(
        &self,
        logical_request_id: impl Into<String>,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateStoreError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state
            .create_request(logical_request_id, body, metadata)
    }

    pub fn continue_request(
        &self,
        logical_request_id: &str,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateStoreError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state
            .continue_request(logical_request_id, body, metadata)
    }

    pub fn reset_global(&self) {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state.reset_global();
    }

    pub fn replace_global_facts(&self, facts: Document) -> Result<(), StateStoreError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state.replace_global_facts(facts)
    }

    pub fn merge_global_facts(&self, facts: Document) -> Result<(), StateStoreError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state.merge_global_facts(facts)
    }

    pub fn merge_request_facts(
        &self,
        logical_request_id: &str,
        facts: Document,
    ) -> Result<(), StateStoreError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state.merge_request_facts(logical_request_id, facts)
    }

    pub fn record_enacted_placement(
        &self,
        logical_request_id: &str,
        target_id: impl Into<String>,
    ) -> Result<(), StateStoreError> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.state
            .record_enacted_placement(logical_request_id, target_id)
    }

    pub fn invoke_and_apply(
        &self,
        operation: Operation,
        input: Document,
    ) -> Invocation<JsonResponse> {
        let _guard = self.stateful_invocation.lock().unwrap();
        self.invoke_and_apply_locked(operation, input)
    }

    fn invoke_and_apply_locked(
        &self,
        operation: Operation,
        input: Document,
    ) -> Invocation<JsonResponse> {
        self.invoke_and_apply_locked_with_removals(operation, input, &[])
    }

    fn invoke_and_apply_locked_with_removals(
        &self,
        operation: Operation,
        input: Document,
        terminal_logical_ids: &[String],
    ) -> Invocation<JsonResponse> {
        if input.get("global").is_some() || input.get("requests").is_some() {
            return invalid_input(
                "operation input must not provide global or requests state".into(),
            );
        }
        let feedback_delivery = if operation == Operation::Feedback {
            if let Err(error) = referenced_request_ids(operation, &input) {
                return invalid_input(error.to_string());
            }
            let delivery_id = input["delivery_id"]
                .as_str()
                .expect("feedback context was validated");
            if let Some(result) = self.state.feedback_result(delivery_id) {
                return Invocation::Success(JsonResponse::duplicate_feedback(input, result));
            }
            Some(delivery_id.to_owned())
        } else {
            None
        };

        let hydrated = match self.state.hydrate(operation, input) {
            Ok(input) => input,
            Err(error) => return invalid_input(error.to_string()),
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
        match snapshot.invoke(operation, hydrated.clone()) {
            Invocation::Success(response) => {
                let feedback = feedback_delivery.as_deref().map(|delivery_id| {
                    (
                        delivery_id,
                        &response.result,
                        self.registry.max_feedback_deliveries(),
                    )
                });
                if let Err(error) = self.state.commit_policy_mutation(
                    &hydrated,
                    &response.input,
                    feedback,
                    terminal_logical_ids,
                ) {
                    return commit_failure(error);
                }
                Invocation::Success(response)
            }
            Invocation::Unavailable => Invocation::Unavailable,
            Invocation::FallbackRequired(failure) => Invocation::FallbackRequired(failure),
        }
    }

    pub fn route_and_admit(
        &self,
        logical_request_id: &str,
        cause: &str,
        candidates: Vec<Document>,
        context: Document,
    ) -> Invocation<PlacementOutcome> {
        let _guard = self.stateful_invocation.lock().unwrap();
        if let Err(error) = self.state.read_request(logical_request_id) {
            return invalid_input(error.to_string());
        }
        let route_input = json!({
            "cause": cause,
            "request_id": logical_request_id,
            "candidates": candidates,
            "context": context
        });
        let route = match self.invoke_and_apply_locked(Operation::Route, route_input) {
            Invocation::Success(response) => response,
            Invocation::Unavailable => return Invocation::Unavailable,
            Invocation::FallbackRequired(failure) => {
                return Invocation::FallbackRequired(failure);
            }
        };
        let order = match rank_route(
            &route.result,
            route.input["candidates"]
                .as_array()
                .expect("validated route input")
                .len(),
        ) {
            Ok(order) => order,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
        };
        let candidates = route.input["candidates"]
            .as_array()
            .expect("validated route input");
        for index in order {
            let target = candidates[index].clone();
            let target_id = target["id"]
                .as_str()
                .expect("validated target ID")
                .to_owned();
            for defer_attempt in 0..=self.max_defer_retries {
                let admit_input = json!({
                    "cause": cause,
                    "request_id": logical_request_id,
                    "target": target,
                    "context": route.input["context"].clone()
                });
                let admit = match self.invoke_and_apply_locked(Operation::Admit, admit_input) {
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
                        return Invocation::Success(PlacementOutcome::Accepted {
                            target_id,
                            request_map: admit.input["requests"][logical_request_id].clone(),
                            defer_attempts: defer_attempt,
                        });
                    }
                    AdmissionDecision::Reject => break,
                    AdmissionDecision::Defer if defer_attempt < self.max_defer_retries => {}
                    AdmissionDecision::Defer => {
                        return Invocation::Success(PlacementOutcome::Deferred {
                            target_id,
                            request_map: admit.input["requests"][logical_request_id].clone(),
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
        input: Document,
        terminal_logical_ids: &[String],
    ) -> Invocation<JsonResponse> {
        let _guard = self.stateful_invocation.lock().unwrap();
        if input["delivery_id"]
            .as_str()
            .is_some_and(|delivery_id| self.state.feedback_result(delivery_id).is_some())
        {
            return self.invoke_and_apply_locked(Operation::Feedback, input);
        }
        self.invoke_and_apply_locked_with_removals(Operation::Feedback, input, terminal_logical_ids)
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

fn commit_failure<T>(error: StateStoreError) -> Invocation<T> {
    let kind = if matches!(error, StateStoreError::FeedbackLedgerFull(_)) {
        InvocationFailureKind::HostSaturated
    } else {
        InvocationFailureKind::InvalidOutput
    };
    Invocation::FallbackRequired(InvocationFailure::new(kind, error.to_string()))
}
