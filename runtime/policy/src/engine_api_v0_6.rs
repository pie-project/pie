use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pie_plex::Document;
use pie_plex::v0_6::{
    FeedbackContext, FeedbackSubject, GroupStatus, MechanicId, Operation, OutcomeKind,
    RequestStatus,
};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

use crate::{
    AttachmentRegistryV0_6, CacheEpisodeTrackerV0_6, FeedbackCommitV0_6, HostSupportV0_6,
    InMemoryPolicyStateBackendV0_6, Invocation, InvocationFailureKind, LifecycleEventV0_6,
    LifecycleHostV0_6, NormalizedPlanV0_6, OperationContextV0_6, OpportunityTrackerV0_6,
    PolicyEngine, PolicyEngineConfig, PolicyStateBackendV0_6, ProtocolLimitsV0_6, QueryHandler,
    RegistryErrorV0_6, RejectingQueryHandler, StagedAction, StateBackendErrorV0_6,
    StateMetricsV0_6, TerminalCleanupV0_6, WorkingSetV0_6, snapshot_ref_v0_6, working_set_v0_6,
};

pub const ENGINE_API_VERSION_V0_6: &str = "pie.plex.engine@2";

#[derive(Clone)]
pub struct PlexRuntimeV0_6 {
    registry: AttachmentRegistryV0_6,
    backend: Arc<dyn PolicyStateBackendV0_6>,
    lifecycle: LifecycleHostV0_6,
    query_handler: Arc<dyn QueryHandler>,
    protocol_limits: ProtocolLimitsV0_6,
    opportunities: Arc<OpportunityTrackerV0_6>,
    cache_episodes: Arc<CacheEpisodeTrackerV0_6>,
    actions: Arc<Mutex<ActionLedgerV0_6>>,
    invocation_gate: Arc<Mutex<()>>,
}

impl PlexRuntimeV0_6 {
    pub fn with_parts(
        registry: AttachmentRegistryV0_6,
        backend: Arc<dyn PolicyStateBackendV0_6>,
        query_handler: Arc<dyn QueryHandler>,
        protocol_limits: ProtocolLimitsV0_6,
    ) -> Result<Self, PlexErrorV0_6> {
        let lifecycle = LifecycleHostV0_6::new(backend.clone());
        Ok(Self {
            registry,
            backend,
            lifecycle,
            query_handler,
            protocol_limits,
            opportunities: Arc::new(OpportunityTrackerV0_6::new(
                protocol_limits.max_tracked_opportunities,
            )?),
            cache_episodes: Arc::new(CacheEpisodeTrackerV0_6::new(
                protocol_limits.max_cache_episodes,
            )?),
            actions: Arc::new(Mutex::new(ActionLedgerV0_6 {
                maximum: protocol_limits.max_action_records,
                records: BTreeMap::new(),
            })),
            invocation_gate: Arc::new(Mutex::new(())),
        })
    }

    pub fn from_package_bytes(
        package: &[u8],
        query_handler: Option<Arc<dyn QueryHandler>>,
        support: HostSupportV0_6,
    ) -> Result<Self, PlexErrorV0_6> {
        let engine = PolicyEngine::new(PolicyEngineConfig::default())
            .map_err(|error| PlexErrorV0_6::Runtime(error.to_string()))?;
        let registry = AttachmentRegistryV0_6::new(engine, support);
        registry
            .attach(package)
            .map_err(|error| PlexErrorV0_6::PolicyPackage(error.to_string()))?;
        Self::with_parts(
            registry,
            Arc::new(InMemoryPolicyStateBackendV0_6::default()),
            query_handler.unwrap_or_else(|| Arc::new(RejectingQueryHandler)),
            ProtocolLimitsV0_6::default(),
        )
    }

    pub fn invoke(&self, event: Document) -> Result<Document, PlexErrorV0_6> {
        let _guard = self.invocation_gate.lock().unwrap();
        let event = EngineEventV0_6::parse(event)?;
        let mut context = event.context()?;

        if let OperationContextV0_6::Feedback(feedback) = &context
            && let Some(result) = self.backend.feedback_result(&feedback.delivery_id)?
        {
            return Ok(result);
        }

        for lifecycle_event in event.lifecycle {
            self.lifecycle.apply(lifecycle_event)?;
        }

        let registry = self.registry.snapshot()?;
        let mut working_set = working_set_v0_6(&context, self.protocol_limits)?;
        add_cleanup_to_working_set(&mut working_set, &event.cleanup);
        let snapshot = self.backend.load(&working_set)?;
        bind_host_metadata(&mut context, &snapshot, &registry)?;

        if let Some(opportunity_id) = opportunity_id(&context) {
            self.opportunities.begin(&context)?;
            if let OperationContextV0_6::Cache(cache) = &context {
                self.cache_episodes.observe(cache)?;
            }
            debug_assert!(!opportunity_id.as_str().is_empty());
        }
        let action_updates = self.actions.lock().unwrap().validate_feedback(&context)?;
        validate_cleanup_context(&context, &event.cleanup, &action_updates)?;

        let invocation = registry.invoke(
            context.clone(),
            snapshot.clone(),
            self.query_handler.clone(),
            self.protocol_limits,
        );
        match invocation {
            Invocation::Success(prepared) => {
                let correlation_id = correlation_id(&context);
                self.actions
                    .lock()
                    .unwrap()
                    .ensure_capacity(&correlation_id, &prepared.actions)?;
                let generation = registry.generation();
                let outcome = success_outcome(
                    generation,
                    &context,
                    &prepared.normalized_plan,
                    &prepared.state_update,
                    &prepared.actions,
                    false,
                )?;
                let feedback = match &context {
                    OperationContextV0_6::Feedback(feedback) => Some(FeedbackCommitV0_6 {
                        delivery_id: feedback.delivery_id.clone(),
                        result: success_outcome(
                            generation,
                            &context,
                            &prepared.normalized_plan,
                            &prepared.state_update,
                            &[],
                            true,
                        )?,
                        maximum_deliveries: self.registry.max_feedback_deliveries(),
                    }),
                    _ => None,
                };
                match self.backend.commit(
                    &snapshot,
                    &prepared.state_update,
                    feedback.as_ref(),
                    &event.cleanup,
                ) {
                    Ok(()) => {}
                    Err(StateBackendErrorV0_6::DuplicateFeedback(delivery_id)) => {
                        return self.backend.feedback_result(&delivery_id)?.ok_or_else(|| {
                            PlexErrorV0_6::Backend(
                                "duplicate feedback has no recorded result".into(),
                            )
                        });
                    }
                    Err(StateBackendErrorV0_6::RevisionConflict { .. }) => {
                        return Ok(fallback_outcome(
                            InvocationFailureKind::StateConflict,
                            "policy state changed during invocation",
                        ));
                    }
                    Err(error) => return Err(error.into()),
                }
                if let Some(opportunity_id) = opportunity_id(&context) {
                    self.opportunities.complete(opportunity_id)?;
                }
                {
                    let mut actions = self.actions.lock().unwrap();
                    actions.apply_feedback(&action_updates);
                    actions.record(&correlation_id, &prepared.actions);
                }
                if event.complete_cache_episode
                    && let OperationContextV0_6::Cache(cache) = &context
                    && let Some(episode) = &cache.episode
                {
                    self.cache_episodes.complete(episode.episode_id.as_str())?;
                }
                Ok(outcome)
            }
            Invocation::Unavailable => {
                complete_non_retryable(&context, &self.opportunities, &self.cache_episodes)?;
                Ok(json!({
                    "status": "unavailable",
                    "operation": context.operation(),
                    "attachment_generation": registry.generation(),
                }))
            }
            Invocation::FallbackRequired(failure) => {
                if failure.kind != InvocationFailureKind::StateConflict {
                    complete_non_retryable(&context, &self.opportunities, &self.cache_episodes)?;
                }
                Ok(fallback_outcome(failure.kind, failure.message))
            }
        }
    }

    pub fn invoke_json(&self, event_json: &str) -> Result<String, PlexErrorV0_6> {
        let event = serde_json::from_str(event_json)
            .map_err(|error| PlexErrorV0_6::InvalidEvent(format!("invalid event JSON: {error}")))?;
        serde_json::to_string(&self.invoke(event)?).map_err(|error| {
            PlexErrorV0_6::Runtime(format!("failed to serialize outcome: {error}"))
        })
    }

    pub fn backend(&self) -> &Arc<dyn PolicyStateBackendV0_6> {
        &self.backend
    }

    pub fn metrics(&self) -> StateMetricsV0_6 {
        self.backend.metrics()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEngineEventV0_6 {
    api_version: String,
    operation: Operation,
    context: Value,
    #[serde(default)]
    lifecycle: Vec<LifecycleEventV0_6>,
    #[serde(default)]
    cleanup: TerminalCleanupV0_6,
    #[serde(default)]
    complete_cache_episode: bool,
}

struct EngineEventV0_6 {
    operation: Operation,
    context: Value,
    lifecycle: Vec<LifecycleEventV0_6>,
    cleanup: TerminalCleanupV0_6,
    complete_cache_episode: bool,
}

impl EngineEventV0_6 {
    fn parse(event: Document) -> Result<Self, PlexErrorV0_6> {
        let event: RawEngineEventV0_6 = serde_json::from_value(event)
            .map_err(|error| PlexErrorV0_6::InvalidEvent(error.to_string()))?;
        if event.api_version != ENGINE_API_VERSION_V0_6 {
            return Err(PlexErrorV0_6::InvalidEvent(format!(
                "unsupported engine API version {:?}; expected {}",
                event.api_version, ENGINE_API_VERSION_V0_6
            )));
        }
        if event.operation != Operation::Feedback
            && (!event.cleanup.requests.is_empty() || !event.cleanup.groups.is_empty())
        {
            return Err(PlexErrorV0_6::InvalidEvent(
                "terminal cleanup is allowed only with feedback".into(),
            ));
        }
        if event.operation != Operation::Cache && event.complete_cache_episode {
            return Err(PlexErrorV0_6::InvalidEvent(
                "complete_cache_episode is valid only for cache".into(),
            ));
        }
        Ok(Self {
            operation: event.operation,
            context: event.context,
            lifecycle: event.lifecycle,
            cleanup: event.cleanup,
            complete_cache_episode: event.complete_cache_episode,
        })
    }

    fn context(&self) -> Result<OperationContextV0_6, PlexErrorV0_6> {
        let context = match self.operation {
            Operation::Admit => OperationContextV0_6::Admit(
                serde_json::from_value(self.context.clone())
                    .map_err(|error| PlexErrorV0_6::InvalidEvent(error.to_string()))?,
            ),
            Operation::Route => OperationContextV0_6::Route(
                serde_json::from_value(self.context.clone())
                    .map_err(|error| PlexErrorV0_6::InvalidEvent(error.to_string()))?,
            ),
            Operation::Schedule => OperationContextV0_6::Schedule(
                serde_json::from_value(self.context.clone())
                    .map_err(|error| PlexErrorV0_6::InvalidEvent(error.to_string()))?,
            ),
            Operation::Cache => OperationContextV0_6::Cache(
                serde_json::from_value(self.context.clone())
                    .map_err(|error| PlexErrorV0_6::InvalidEvent(error.to_string()))?,
            ),
            Operation::Feedback => OperationContextV0_6::Feedback(
                serde_json::from_value(self.context.clone())
                    .map_err(|error| PlexErrorV0_6::InvalidEvent(error.to_string()))?,
            ),
        };
        Ok(context)
    }
}

fn bind_host_metadata(
    context: &mut OperationContextV0_6,
    snapshot: &crate::StateSnapshotV0_6,
    registry: &crate::AttachmentSnapshotV0_6,
) -> Result<(), PlexErrorV0_6> {
    let snapshot_ref = snapshot_ref_v0_6(snapshot)?;
    let mechanics = registry
        .negotiated_mechanics(context.operation())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect::<Vec<MechanicId>>();
    match context {
        OperationContextV0_6::Admit(context) => {
            context.meta.snapshot = snapshot_ref;
            context.meta.mechanics = mechanics;
        }
        OperationContextV0_6::Route(context) => {
            context.meta.snapshot = snapshot_ref;
            context.meta.mechanics = mechanics;
        }
        OperationContextV0_6::Schedule(context) => {
            context.meta.snapshot = snapshot_ref;
            context.meta.mechanics = mechanics;
        }
        OperationContextV0_6::Cache(context) => {
            context.meta.snapshot = snapshot_ref;
            context.meta.mechanics = mechanics;
        }
        OperationContextV0_6::Feedback(_) => {}
    }
    Ok(())
}

fn add_cleanup_to_working_set(working_set: &mut WorkingSetV0_6, cleanup: &TerminalCleanupV0_6) {
    working_set.request_ids.extend(
        cleanup
            .requests
            .iter()
            .map(|request| request.request_id.clone()),
    );
    working_set
        .group_ids
        .extend(cleanup.groups.iter().map(|group| group.group_id.clone()));
}

fn validate_cleanup_context(
    context: &OperationContextV0_6,
    cleanup: &TerminalCleanupV0_6,
    action_updates: &[ActionFeedbackUpdateV0_6],
) -> Result<(), PlexErrorV0_6> {
    if cleanup.requests.is_empty() && cleanup.groups.is_empty() {
        return Ok(());
    }
    let OperationContextV0_6::Feedback(feedback) = context else {
        return Err(PlexErrorV0_6::InvalidEvent(
            "terminal cleanup requires feedback".into(),
        ));
    };
    for terminal in &cleanup.requests {
        let expected = match terminal.status {
            RequestStatus::Completed => OutcomeKind::Completed,
            RequestStatus::Failed => OutcomeKind::Failed,
            RequestStatus::Cancelled => OutcomeKind::Cancelled,
            RequestStatus::Expired => OutcomeKind::Expired,
            _ => {
                return Err(PlexErrorV0_6::InvalidEvent(format!(
                    "request cleanup status {:?} is not feedback-terminal",
                    terminal.status
                )));
            }
        };
        if !has_feedback_outcome(
            feedback,
            &FeedbackSubject::Request(terminal.request_id.clone()),
            expected,
        ) {
            return Err(PlexErrorV0_6::InvalidEvent(format!(
                "request cleanup {:?} has no matching feedback record",
                terminal.request_id
            )));
        }
        if terminal.status == RequestStatus::Cancelled
            && !action_updates.iter().any(|update| {
                update.outcome == ActionTerminalStatusV0_6::Succeeded
                    && update.record.method == "pie.request.cancel@1"
                    && update.record.args["request_id"].as_str()
                        == Some(terminal.request_id.as_str())
            })
        {
            return Err(PlexErrorV0_6::InvalidEvent(format!(
                "cancelled request {:?} has no successful cancellation action",
                terminal.request_id
            )));
        }
    }
    for terminal in &cleanup.groups {
        let expected = match terminal.status {
            GroupStatus::Closed => OutcomeKind::Completed,
            GroupStatus::Cancelled => OutcomeKind::Cancelled,
            GroupStatus::Expired => OutcomeKind::Expired,
            GroupStatus::Open => unreachable!("cleanup parser rejects open groups"),
        };
        if !has_feedback_outcome(
            feedback,
            &FeedbackSubject::WorkGroup(terminal.group_id.clone()),
            expected,
        ) {
            return Err(PlexErrorV0_6::InvalidEvent(format!(
                "group cleanup {:?} has no matching feedback record",
                terminal.group_id
            )));
        }
        if terminal.status == GroupStatus::Cancelled
            && !action_updates.iter().any(|update| {
                update.outcome == ActionTerminalStatusV0_6::Succeeded
                    && update.record.method == "pie.group.cancel@1"
                    && update.record.args["group_id"].as_str() == Some(terminal.group_id.as_str())
            })
        {
            return Err(PlexErrorV0_6::InvalidEvent(format!(
                "cancelled group {:?} has no successful cancellation action",
                terminal.group_id
            )));
        }
    }
    Ok(())
}

fn correlation_id(context: &OperationContextV0_6) -> String {
    match context {
        OperationContextV0_6::Admit(context) => context.meta.opportunity_id.0.clone(),
        OperationContextV0_6::Route(context) => context.meta.opportunity_id.0.clone(),
        OperationContextV0_6::Schedule(context) => context.meta.opportunity_id.0.clone(),
        OperationContextV0_6::Cache(context) => context.meta.opportunity_id.0.clone(),
        OperationContextV0_6::Feedback(context) => format!(
            "feedback:{}",
            blake3::hash(context.delivery_id.as_str().as_bytes()).to_hex()
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionTerminalStatusV0_6 {
    Succeeded,
    Failed,
    AlreadyTerminal,
    Expired,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq)]
struct ActionRecordV0_6 {
    method: String,
    args: Document,
    idempotency_key: String,
    terminal: Option<ActionTerminalStatusV0_6>,
}

#[derive(Debug, Clone)]
struct ActionFeedbackUpdateV0_6 {
    key: (String, u64),
    record: ActionRecordV0_6,
    outcome: ActionTerminalStatusV0_6,
}

struct ActionLedgerV0_6 {
    maximum: usize,
    records: BTreeMap<(String, u64), ActionRecordV0_6>,
}

impl ActionLedgerV0_6 {
    fn ensure_capacity(
        &self,
        correlation_id: &str,
        actions: &[StagedAction],
    ) -> Result<(), PlexErrorV0_6> {
        let additional = actions
            .iter()
            .filter(|action| {
                !self
                    .records
                    .contains_key(&(correlation_id.to_owned(), action.id))
            })
            .count();
        if self.records.len().saturating_add(additional) > self.maximum {
            return Err(PlexErrorV0_6::Runtime(format!(
                "action ledger reached its limit of {}",
                self.maximum
            )));
        }
        Ok(())
    }

    fn record(&mut self, correlation_id: &str, actions: &[StagedAction]) {
        for action in actions {
            let idempotency_key = action.args["idempotency_key"]
                .as_str()
                .unwrap_or("")
                .to_owned();
            let previous = self.records.insert(
                (correlation_id.to_owned(), action.id),
                ActionRecordV0_6 {
                    method: action.method.clone(),
                    args: action.args.clone(),
                    idempotency_key,
                    terminal: None,
                },
            );
            debug_assert!(previous.is_none());
        }
    }

    fn validate_feedback(
        &self,
        context: &OperationContextV0_6,
    ) -> Result<Vec<ActionFeedbackUpdateV0_6>, PlexErrorV0_6> {
        let OperationContextV0_6::Feedback(feedback) = context else {
            return Ok(Vec::new());
        };
        let mut updates = Vec::new();
        for record in &feedback.records {
            let FeedbackSubject::Action(action_id) = &record.subject else {
                continue;
            };
            let facts = record
                .facts
                .as_object()
                .expect("feedback facts were validated");
            if let Some(field) = facts.keys().find(|field| {
                !matches!(
                    field.as_str(),
                    "opportunity_id" | "method" | "idempotency_key" | "status" | "details"
                )
            }) {
                return Err(PlexErrorV0_6::InvalidEvent(format!(
                    "action feedback contains unsupported field {field:?}"
                )));
            }
            if let Some(details) = facts.get("details")
                && !details.is_object()
            {
                return Err(PlexErrorV0_6::InvalidEvent(
                    "action feedback details must be a JSON object".into(),
                ));
            }
            let correlation_id = record.facts["opportunity_id"].as_str().ok_or_else(|| {
                PlexErrorV0_6::InvalidEvent("action feedback requires facts.opportunity_id".into())
            })?;
            let key = (correlation_id.to_owned(), action_id.0);
            let action = self.records.get(&key).ok_or_else(|| {
                PlexErrorV0_6::InvalidEvent(format!(
                    "action feedback references unknown action {correlation_id}/{}",
                    action_id.0
                ))
            })?;
            if action.terminal.is_some() {
                return Err(PlexErrorV0_6::InvalidEvent(format!(
                    "action {correlation_id}/{} is already terminal",
                    action_id.0
                )));
            }
            if record.facts["method"].as_str() != Some(action.method.as_str())
                || record.facts["idempotency_key"].as_str() != Some(action.idempotency_key.as_str())
            {
                return Err(PlexErrorV0_6::InvalidEvent(format!(
                    "action feedback correlation mismatch for {correlation_id}/{}",
                    action_id.0
                )));
            }
            let status = record.facts["status"].as_str().ok_or_else(|| {
                PlexErrorV0_6::InvalidEvent("action feedback requires facts.status".into())
            })?;
            let outcome = match (record.outcome, status) {
                (OutcomeKind::ActionSucceeded, "succeeded") => ActionTerminalStatusV0_6::Succeeded,
                (OutcomeKind::ActionFailed, "failed") => ActionTerminalStatusV0_6::Failed,
                (OutcomeKind::ActionFailed, "already-terminal") => {
                    ActionTerminalStatusV0_6::AlreadyTerminal
                }
                (OutcomeKind::ActionFailed, "expired") => ActionTerminalStatusV0_6::Expired,
                (OutcomeKind::ActionFailed, "unsupported") => ActionTerminalStatusV0_6::Unsupported,
                _ => {
                    return Err(PlexErrorV0_6::InvalidEvent(format!(
                        "action feedback outcome/status mismatch: {:?}/{status}",
                        record.outcome
                    )));
                }
            };
            updates.push(ActionFeedbackUpdateV0_6 {
                key,
                record: action.clone(),
                outcome,
            });
        }
        Ok(updates)
    }

    fn apply_feedback(&mut self, updates: &[ActionFeedbackUpdateV0_6]) {
        for update in updates {
            let action = self
                .records
                .get_mut(&update.key)
                .expect("validated action feedback key");
            action.terminal = Some(update.outcome);
        }
    }
}

fn has_feedback_outcome(
    feedback: &FeedbackContext,
    subject: &FeedbackSubject,
    outcome: OutcomeKind,
) -> bool {
    feedback
        .records
        .iter()
        .any(|record| &record.subject == subject && record.outcome == outcome)
}

fn opportunity_id(context: &OperationContextV0_6) -> Option<&pie_plex::v0_6::OpportunityId> {
    match context {
        OperationContextV0_6::Admit(context) => Some(&context.meta.opportunity_id),
        OperationContextV0_6::Route(context) => Some(&context.meta.opportunity_id),
        OperationContextV0_6::Schedule(context) => Some(&context.meta.opportunity_id),
        OperationContextV0_6::Cache(context) => Some(&context.meta.opportunity_id),
        OperationContextV0_6::Feedback(_) => None,
    }
}

fn success_outcome(
    generation: u64,
    context: &OperationContextV0_6,
    plan: &NormalizedPlanV0_6,
    state_update: &pie_plex::v0_6::StateUpdate,
    actions: &[crate::StagedAction],
    duplicate_feedback: bool,
) -> Result<Document, PlexErrorV0_6> {
    serde_json::to_value(json!({
        "status": "success",
        "operation": context.operation(),
        "plan": plan,
        "state_update": state_update,
        "actions": actions,
        "duplicate_feedback": duplicate_feedback,
        "attachment_generation": generation,
    }))
    .map_err(|error| PlexErrorV0_6::Runtime(error.to_string()))
}

fn fallback_outcome(kind: InvocationFailureKind, message: impl Into<String>) -> Document {
    json!({
        "status": "fallback",
        "failure": {
            "kind": kind,
            "message": message.into(),
        }
    })
}

fn complete_non_retryable(
    context: &OperationContextV0_6,
    opportunities: &OpportunityTrackerV0_6,
    cache_episodes: &CacheEpisodeTrackerV0_6,
) -> Result<(), PlexErrorV0_6> {
    if let Some(opportunity_id) = opportunity_id(context) {
        opportunities.complete(opportunity_id)?;
    }
    if let OperationContextV0_6::Cache(cache) = context
        && let Some(episode) = &cache.episode
    {
        cache_episodes.complete(episode.episode_id.as_str())?;
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum PlexErrorV0_6 {
    #[error("invalid PLEX engine event: {0}")]
    InvalidEvent(String),
    #[error("PLEX state backend failed: {0}")]
    Backend(String),
    #[error("PLEX policy package failed: {0}")]
    PolicyPackage(String),
    #[error("PLEX runtime failed: {0}")]
    Runtime(String),
}

impl From<StateBackendErrorV0_6> for PlexErrorV0_6 {
    fn from(error: StateBackendErrorV0_6) -> Self {
        Self::Backend(error.to_string())
    }
}

impl From<RegistryErrorV0_6> for PlexErrorV0_6 {
    fn from(error: RegistryErrorV0_6) -> Self {
        Self::PolicyPackage(error.to_string())
    }
}

impl From<crate::ProtocolErrorV0_6> for PlexErrorV0_6 {
    fn from(error: crate::ProtocolErrorV0_6) -> Self {
        Self::InvalidEvent(error.to_string())
    }
}
