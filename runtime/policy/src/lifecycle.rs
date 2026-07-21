use pie_plex::{AdmissionDecision, Document, Operation, rank_route, validate_admit};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    AttachmentRegistry, CanonicalRequestStore, Invocation, InvocationFailure,
    InvocationFailureKind, JsonResponse,
};

#[derive(Clone)]
pub struct LifecycleHost {
    registry: AttachmentRegistry,
    requests: CanonicalRequestStore,
    max_defer_retries: u32,
}

impl LifecycleHost {
    pub fn new(
        registry: AttachmentRegistry,
        requests: CanonicalRequestStore,
        max_defer_retries: u32,
    ) -> Self {
        Self {
            registry,
            requests,
            max_defer_retries,
        }
    }

    pub fn requests(&self) -> &CanonicalRequestStore {
        &self.requests
    }

    pub fn invoke_and_apply(
        &self,
        operation: Operation,
        input: Document,
    ) -> Invocation<JsonResponse> {
        let snapshot = match self.registry.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::Instantiation,
                    error.to_string(),
                ));
            }
        };
        match snapshot.invoke(operation, input) {
            Invocation::Success(response) => {
                if !(operation == Operation::Feedback && response.duplicate_feedback)
                    && let Err(error) = self.requests.apply_operation(operation, &response.input)
                {
                    return Invocation::FallbackRequired(InvocationFailure::new(
                        InvocationFailureKind::InvalidOutput,
                        error.to_string(),
                    ));
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
        let request = match self.requests.get(logical_request_id) {
            Ok(request) => request,
            Err(error) => return invalid_input(error.to_string()),
        };
        let route_input = json!({
            "cause": cause,
            "request": request,
            "candidates": candidates,
            "context": context
        });
        let route = match self.invoke_and_apply(Operation::Route, route_input) {
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
                let request = match self.requests.get(logical_request_id) {
                    Ok(request) => request,
                    Err(error) => return invalid_input(error.to_string()),
                };
                let admit_input = json!({
                    "cause": cause,
                    "request": request,
                    "target": target,
                    "context": route.input["context"].clone()
                });
                let admit = match self.invoke_and_apply(Operation::Admit, admit_input) {
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
                            request: admit.input["request"].clone(),
                            defer_attempts: defer_attempt,
                        });
                    }
                    AdmissionDecision::Reject => break,
                    AdmissionDecision::Defer if defer_attempt < self.max_defer_retries => {}
                    AdmissionDecision::Defer => {
                        return Invocation::Success(PlacementOutcome::Deferred {
                            target_id,
                            request: admit.input["request"].clone(),
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
        match self.invoke_and_apply(Operation::Feedback, input) {
            Invocation::Success(response) => {
                for logical_id in terminal_logical_ids {
                    if let Err(error) = self.requests.remove(logical_id) {
                        return Invocation::FallbackRequired(InvocationFailure::new(
                            InvocationFailureKind::InvalidOutput,
                            error.to_string(),
                        ));
                    }
                }
                Invocation::Success(response)
            }
            other => other,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum PlacementOutcome {
    Accepted {
        target_id: String,
        request: Document,
        defer_attempts: u32,
    },
    Deferred {
        target_id: String,
        request: Document,
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
