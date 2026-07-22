use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use pie_plex::{AdmissionDecision, Document, Operation};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::context::is_versioned_method;
use crate::protocol::validate_context;
use crate::{
    AttachmentRegistry, InMemoryPolicyStateBackend, Invocation, InvocationFailureKind,
    LifecycleHost, PolicyEngine, PolicyEngineConfig, PolicyStateBackend, QueryHandler,
    RejectingQueryHandler, StateBackendError,
};

pub const ENGINE_API_VERSION: &str = "pie.plex.engine@1";

#[derive(Clone)]
pub struct PlexRuntime {
    lifecycle: LifecycleHost,
    backend: Arc<dyn PolicyStateBackend>,
    supported_actions: BTreeSet<String>,
    invocation_gate: Arc<Mutex<()>>,
}

impl PlexRuntime {
    pub fn with_parts(
        registry: AttachmentRegistry,
        backend: Arc<dyn PolicyStateBackend>,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: BTreeSet<String>,
        max_defer_retries: u32,
    ) -> Result<Self, PlexError> {
        if let Some(method) = supported_actions
            .iter()
            .find(|method| !is_versioned_method(method))
        {
            return Err(PlexError::Runtime(format!(
                "supported action {method:?} is not a versioned method name"
            )));
        }
        let lifecycle = LifecycleHost::with_host(
            registry,
            backend.clone(),
            query_handler,
            supported_actions.clone(),
            max_defer_retries,
        );
        Ok(Self {
            lifecycle,
            backend,
            supported_actions,
            invocation_gate: Arc::new(Mutex::new(())),
        })
    }

    pub fn from_package_bytes(
        package: &[u8],
        query_handler: Option<Arc<dyn QueryHandler>>,
        supported_actions: BTreeSet<String>,
    ) -> Result<Self, PlexError> {
        let engine = PolicyEngine::new(PolicyEngineConfig::default())
            .map_err(|error| PlexError::Runtime(error.to_string()))?;
        let registry = AttachmentRegistry::new(engine);
        registry
            .attach(package)
            .map_err(|error| PlexError::PolicyPackage(error.to_string()))?;
        Self::with_parts(
            registry,
            Arc::new(InMemoryPolicyStateBackend::default()),
            query_handler.unwrap_or_else(|| Arc::new(RejectingQueryHandler)),
            supported_actions,
            2,
        )
    }

    pub fn invoke(&self, event: Document) -> Result<Document, PlexError> {
        let _guard = self.invocation_gate.lock().unwrap();
        let event = EngineEvent::parse(event)?;
        let operation = event.operation;
        let request_events = parse_request_events(&event.request_events, operation)?;
        let terminal_requests = request_events
            .iter()
            .filter_map(|event| match event {
                RequestEvent::Finish { request_id } => Some(request_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        let context = prepare_hook_context(event.context, operation, &self.supported_actions)?;
        let request_ids = validate_context(operation, &context)
            .map_err(|error| PlexError::InvalidEvent(error.to_string()))?;
        for request_id in &terminal_requests {
            if !request_ids.contains(request_id) {
                return Err(PlexError::InvalidEvent(format!(
                    "finish request {request_id} is not referenced by feedback"
                )));
            }
        }
        let feedback_delivery_id = (operation == Operation::Feedback).then(|| {
            context["delivery_id"]
                .as_str()
                .expect("feedback context was validated")
        });
        let mut duplicate_feedback = self.feedback_is_committed(feedback_delivery_id)?;
        if !duplicate_feedback
            && let Err(error) = validate_request_event_state(&request_events, self.backend.as_ref())
        {
            if self.feedback_is_committed(feedback_delivery_id)? {
                duplicate_feedback = true;
            } else {
                return Err(error);
            }
        }

        if !duplicate_feedback {
            for request_event in request_events
                .iter()
                .filter(|event| !matches!(event, RequestEvent::Finish { .. }))
            {
                if let Err(error) = self.apply_request_event(request_event) {
                    if self.feedback_is_committed(feedback_delivery_id)? {
                        break;
                    }
                    return Err(error);
                }
            }
        }

        let invocation = if operation == Operation::Feedback && !terminal_requests.is_empty() {
            self.lifecycle
                .feedback_and_remove_with_state_snapshot(context.clone(), &terminal_requests)
        } else {
            self.lifecycle
                .invoke_and_apply_with_state_snapshot(operation, context.clone())
        };

        match invocation {
            Invocation::Success(applied) => {
                let decision = normalize_decision(operation, &context, &applied.prepared.result)
                    .map_err(PlexError::Runtime)?;
                let request_fields =
                    applied
                        .state_snapshot
                        .as_ref()
                        .map_or_else(Map::new, |snapshot| {
                            changed_request_fields(
                                &snapshot.requests,
                                &applied.prepared.state_updates.requests,
                            )
                        });
                Ok(json!({
                    "status": "success",
                    "decision": decision,
                    "request_fields": request_fields,
                    "actions": applied.prepared.actions,
                }))
            }
            Invocation::Unavailable => {
                self.remove_terminal_requests(&terminal_requests)?;
                Ok(json!({"status": "unavailable"}))
            }
            Invocation::FallbackRequired(failure) => {
                self.remove_terminal_requests(&terminal_requests)?;
                if failure.kind == InvocationFailureKind::BackendFailure {
                    return Err(PlexError::Backend(failure.message));
                }
                Ok(json!({
                    "status": "fallback",
                    "failure": {
                        "kind": failure.kind,
                        "message": failure.message,
                    }
                }))
            }
        }
    }

    pub fn invoke_json(&self, event_json: &str) -> Result<String, PlexError> {
        let event = serde_json::from_str(event_json)
            .map_err(|error| PlexError::InvalidEvent(format!("invalid event JSON: {error}")))?;
        serde_json::to_string(&self.invoke(event)?)
            .map_err(|error| PlexError::Runtime(format!("failed to serialize outcome: {error}")))
    }

    pub fn backend(&self) -> &Arc<dyn PolicyStateBackend> {
        &self.backend
    }

    fn feedback_is_committed(&self, delivery_id: Option<&str>) -> Result<bool, PlexError> {
        match delivery_id {
            Some(delivery_id) => Ok(self.backend.feedback_result(delivery_id)?.is_some()),
            None => Ok(false),
        }
    }

    fn apply_request_event(&self, event: &RequestEvent) -> Result<(), PlexError> {
        match event {
            RequestEvent::Create {
                request_id,
                facts,
                fields,
            } => {
                self.lifecycle.create_request(
                    request_id,
                    fields["body"].clone(),
                    fields["metadata"].clone(),
                )?;
                self.backend
                    .replace_request_fields(request_id, fields.clone())?;
                self.lifecycle
                    .merge_request_facts(request_id, facts.clone())?;
            }
            RequestEvent::Continue {
                request_id,
                facts,
                fields,
            } => {
                let current = self.backend.read_request(request_id)?;
                let expected_generation = current["facts"]["generation_id"]
                    .as_u64()
                    .and_then(|generation| generation.checked_add(1))
                    .ok_or_else(|| {
                        PlexError::InvalidEvent(format!(
                            "request {request_id} has no next generation"
                        ))
                    })?;
                if facts["generation_id"].as_u64() != Some(expected_generation) {
                    return Err(PlexError::InvalidEvent(format!(
                        "continuation generation for {request_id} must be {expected_generation}"
                    )));
                }
                let continued = self.lifecycle.continue_request(
                    request_id,
                    fields["body"].clone(),
                    fields["metadata"].clone(),
                )?;
                let mut canonical_fields = continued["fields"].clone();
                for (key, value) in fields.as_object().expect("validated fields") {
                    if !matches!(key.as_str(), "body" | "metadata") {
                        canonical_fields[key] = value.clone();
                    }
                }
                self.backend
                    .replace_request_fields(request_id, canonical_fields)?;
                self.lifecycle
                    .merge_request_facts(request_id, facts.clone())?;
            }
            RequestEvent::MergeFacts { request_id, facts } => {
                self.lifecycle
                    .merge_request_facts(request_id, facts.clone())?;
            }
            RequestEvent::Finish { .. } => {}
        }
        Ok(())
    }

    fn remove_terminal_requests(&self, request_ids: &[String]) -> Result<(), PlexError> {
        for request_id in request_ids {
            match self.backend.remove_request(request_id) {
                Ok(_) | Err(StateBackendError::NotFound(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

fn validate_request_event_state(
    events: &[RequestEvent],
    backend: &dyn PolicyStateBackend,
) -> Result<(), PlexError> {
    let mut generations = BTreeMap::<String, Option<u64>>::new();
    for event in events {
        let request_id = match event {
            RequestEvent::Create { request_id, .. }
            | RequestEvent::Continue { request_id, .. }
            | RequestEvent::MergeFacts { request_id, .. }
            | RequestEvent::Finish { request_id } => request_id,
        };
        if !generations.contains_key(request_id) {
            let generation = match backend.read_request(request_id) {
                Ok(request) => request["facts"]["generation_id"].as_u64(),
                Err(StateBackendError::NotFound(_)) => None,
                Err(error) => return Err(error.into()),
            };
            generations.insert(request_id.clone(), generation);
        }
        let current = generations[request_id];
        match event {
            RequestEvent::Create { .. } => {
                if current.is_some() {
                    return Err(PlexError::InvalidEvent(format!(
                        "create request {request_id} already exists"
                    )));
                }
                generations.insert(request_id.clone(), Some(0));
            }
            RequestEvent::Continue { facts, .. } => {
                let expected = current
                    .and_then(|generation| generation.checked_add(1))
                    .ok_or_else(|| {
                        PlexError::InvalidEvent(format!(
                            "continue request {request_id} has no current generation"
                        ))
                    })?;
                if facts["generation_id"].as_u64() != Some(expected) {
                    return Err(PlexError::InvalidEvent(format!(
                        "continuation generation for {request_id} must be {expected}"
                    )));
                }
                generations.insert(request_id.clone(), Some(expected));
            }
            RequestEvent::MergeFacts { facts, .. } => {
                let current = current.ok_or_else(|| {
                    PlexError::InvalidEvent(format!(
                        "merge-facts request {request_id} does not exist"
                    ))
                })?;
                if let Some(value) = facts.get("generation_id") {
                    let generation = value.as_u64().ok_or_else(|| {
                        PlexError::InvalidEvent(format!(
                            "merge-facts generation for {request_id} must be unsigned"
                        ))
                    })?;
                    if generation != current {
                        return Err(PlexError::InvalidEvent(format!(
                            "merge-facts cannot change generation for {request_id}"
                        )));
                    }
                }
            }
            RequestEvent::Finish { .. } => {
                if current.is_none() {
                    return Err(PlexError::InvalidEvent(format!(
                        "finish request {request_id} does not exist"
                    )));
                }
                generations.insert(request_id.clone(), None);
            }
        }
    }
    Ok(())
}

struct EngineEvent {
    operation: Operation,
    context: Document,
    request_events: Vec<Document>,
}

impl EngineEvent {
    fn parse(event: Document) -> Result<Self, PlexError> {
        let event = event
            .as_object()
            .ok_or_else(|| PlexError::InvalidEvent("engine event must be an object".into()))?;
        require_exact_keys(
            event,
            &["api_version", "context", "hook", "request_events"],
            "engine event",
        )?;
        if event["api_version"].as_str() != Some(ENGINE_API_VERSION) {
            return Err(PlexError::InvalidEvent(format!(
                "api_version must be {ENGINE_API_VERSION}"
            )));
        }
        let operation = match event["hook"].as_str() {
            Some("route") => Operation::Route,
            Some("admit") => Operation::Admit,
            Some("schedule") => Operation::Schedule,
            Some("evict") => Operation::Evict,
            Some("feedback") => Operation::Feedback,
            _ => {
                return Err(PlexError::InvalidEvent(
                    "hook must be route, admit, schedule, evict, or feedback".into(),
                ));
            }
        };
        let context = event["context"]
            .as_object()
            .ok_or_else(|| PlexError::InvalidEvent("context must be an object".into()))?;
        let request_events = event["request_events"]
            .as_array()
            .ok_or_else(|| PlexError::InvalidEvent("request_events must be an array".into()))?;
        Ok(Self {
            operation,
            context: Value::Object(context.clone()),
            request_events: request_events.clone(),
        })
    }
}

enum RequestEvent {
    Create {
        request_id: String,
        facts: Document,
        fields: Document,
    },
    Continue {
        request_id: String,
        facts: Document,
        fields: Document,
    },
    MergeFacts {
        request_id: String,
        facts: Document,
    },
    Finish {
        request_id: String,
    },
}

fn parse_request_events(
    events: &[Document],
    operation: Operation,
) -> Result<Vec<RequestEvent>, PlexError> {
    let mut parsed = Vec::with_capacity(events.len());
    let mut finished = BTreeSet::new();
    for event in events {
        let object = event
            .as_object()
            .ok_or_else(|| PlexError::InvalidEvent("request event must be an object".into()))?;
        let op = object
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| PlexError::InvalidEvent("request event op must be a string".into()))?;
        let request_id = object
            .get("request_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                PlexError::InvalidEvent("request event request_id must be non-empty".into())
            })?
            .to_owned();
        match op {
            "create" | "continue" => {
                require_exact_keys(object, &["facts", "fields", "op", "request_id"], op)?;
                let facts = require_document_object(object.get("facts"), "request event facts")?;
                let fields = require_document_object(object.get("fields"), "request event fields")?;
                require_request_fields(&fields)?;
                if facts
                    .get("logical_request_id")
                    .is_some_and(|id| id.as_str() != Some(request_id.as_str()))
                {
                    return Err(PlexError::InvalidEvent(format!(
                        "{op} facts.logical_request_id does not match {request_id}"
                    )));
                }
                let generation = facts["generation_id"].as_u64().ok_or_else(|| {
                    PlexError::InvalidEvent(format!("{op} requires facts.generation_id"))
                })?;
                if op == "create" && generation != 0 {
                    return Err(PlexError::InvalidEvent(
                        "create generation_id must be 0".into(),
                    ));
                }
                if op == "create" {
                    parsed.push(RequestEvent::Create {
                        request_id,
                        facts,
                        fields,
                    });
                } else {
                    parsed.push(RequestEvent::Continue {
                        request_id,
                        facts,
                        fields,
                    });
                }
            }
            "merge-facts" => {
                require_exact_keys(object, &["facts", "op", "request_id"], op)?;
                let facts = require_document_object(object.get("facts"), "request event facts")?;
                if facts
                    .get("logical_request_id")
                    .is_some_and(|id| id.as_str() != Some(request_id.as_str()))
                {
                    return Err(PlexError::InvalidEvent(format!(
                        "merge-facts logical_request_id does not match {request_id}"
                    )));
                }
                parsed.push(RequestEvent::MergeFacts { request_id, facts });
            }
            "finish" => {
                require_exact_keys(object, &["op", "request_id"], op)?;
                if operation != Operation::Feedback {
                    return Err(PlexError::InvalidEvent(
                        "finish request events are valid only with feedback".into(),
                    ));
                }
                if !finished.insert(request_id.clone()) {
                    return Err(PlexError::InvalidEvent(format!(
                        "duplicate finish event for {request_id}"
                    )));
                }
                parsed.push(RequestEvent::Finish { request_id });
            }
            _ => {
                return Err(PlexError::InvalidEvent(format!(
                    "unsupported request event operation {op}"
                )));
            }
        }
    }
    Ok(parsed)
}

fn prepare_hook_context(
    mut context: Document,
    operation: Operation,
    supported_actions: &BTreeSet<String>,
) -> Result<Document, PlexError> {
    let object = context
        .as_object_mut()
        .ok_or_else(|| PlexError::InvalidEvent("hook context must be an object".into()))?;
    if operation != Operation::Feedback && !object.contains_key("cause") {
        object.insert(
            "cause".into(),
            json!(match operation {
                Operation::Route | Operation::Admit => "engine-event",
                Operation::Schedule => "service-step",
                Operation::Evict => "allocation-deficit",
                Operation::Feedback => unreachable!(),
            }),
        );
    }
    let host_context = object.entry("context").or_insert_with(|| json!({}));
    let host_context = host_context
        .as_object_mut()
        .ok_or_else(|| PlexError::InvalidEvent("context.context must be an object".into()))?;
    let capabilities = host_context
        .entry("capabilities")
        .or_insert_with(|| json!({}));
    let capabilities = capabilities.as_object_mut().ok_or_else(|| {
        PlexError::InvalidEvent("context.context.capabilities must be an object".into())
    })?;
    capabilities.insert(
        "actions".into(),
        json!(supported_actions.iter().collect::<Vec<_>>()),
    );
    Ok(context)
}

fn changed_request_fields(
    original: &BTreeMap<String, Document>,
    updates: &BTreeMap<String, crate::RequestStateUpdate>,
) -> Map<String, Value> {
    updates
        .iter()
        .filter(|(request_id, update)| original[*request_id]["fields"] != update.fields)
        .map(|(request_id, update)| (request_id.clone(), update.fields.clone()))
        .collect()
}

fn normalize_decision(
    operation: Operation,
    context: &Document,
    result: &Document,
) -> Result<Document, String> {
    match operation {
        Operation::Route => Ok(json!({
            "order": pie_plex::rank_route(
                result,
                context["candidates"]
                    .as_array()
                    .expect("validated route context")
                    .len(),
            )
            .map_err(|error| error.to_string())?
        })),
        Operation::Admit => {
            let decision = pie_plex::validate_admit(result).map_err(|error| error.to_string())?;
            Ok(json!({
                "decision": match decision {
                    AdmissionDecision::Accept => "accept",
                    AdmissionDecision::Defer => "defer",
                    AdmissionDecision::Reject => "reject",
                }
            }))
        }
        Operation::Schedule => Ok(json!({
            "selected": pie_plex::select_schedule(context, result)
                .map_err(|error| error.to_string())?
        })),
        Operation::Evict => Ok(json!({
            "selected": pie_plex::select_evictions(context, result)
                .map_err(|error| error.to_string())?
        })),
        Operation::Feedback => Ok(json!({})),
    }
}

fn require_request_fields(fields: &Document) -> Result<(), PlexError> {
    for key in ["body", "metadata"] {
        if !fields.get(key).is_some_and(Value::is_object) {
            return Err(PlexError::InvalidEvent(format!(
                "request fields.{key} must be an object"
            )));
        }
    }
    Ok(())
}

fn require_document_object(value: Option<&Value>, name: &str) -> Result<Document, PlexError> {
    value
        .and_then(Value::as_object)
        .cloned()
        .map(Value::Object)
        .ok_or_else(|| PlexError::InvalidEvent(format!("{name} must be an object")))
}

fn require_exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    name: &str,
) -> Result<(), PlexError> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(PlexError::InvalidEvent(format!(
            "{name} contains unexpected or missing keys"
        )));
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum PlexError {
    #[error("invalid PLEX engine event: {0}")]
    InvalidEvent(String),
    #[error("PLEX state backend failed: {0}")]
    Backend(String),
    #[error("invalid PLEX policy package: {0}")]
    PolicyPackage(String),
    #[error("PLEX runtime failed: {0}")]
    Runtime(String),
}

impl From<StateBackendError> for PlexError {
    fn from(error: StateBackendError) -> Self {
        Self::Backend(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_engine_keys_and_versions() {
        assert!(matches!(
            EngineEvent::parse(json!({
                "api_version": ENGINE_API_VERSION,
                "hook": "route",
                "context": {},
                "request_events": [],
                "extra": true
            })),
            Err(PlexError::InvalidEvent(_))
        ));
        assert!(matches!(
            EngineEvent::parse(json!({
                "api_version": "wrong",
                "hook": "route",
                "context": {},
                "request_events": []
            })),
            Err(PlexError::InvalidEvent(_))
        ));
    }

    #[test]
    fn normalizes_decisions_once() {
        assert_eq!(
            normalize_decision(
                Operation::Route,
                &json!({"candidates": [{}, {}, {}]}),
                &json!({"scores": [1.0, 3.0, 3.0]}),
            )
            .unwrap(),
            json!({"order": [1, 2, 0]})
        );
    }
}
