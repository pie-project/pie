use std::sync::Arc;

use pie_plex::Document;
use pie_plex::v0_6::{
    GroupId, GroupLimits, GroupState, GroupStatus, PrincipalId, RequestId, RequestState,
    RequestStatus,
};
use serde::{Deserialize, Serialize};

use crate::{PolicyStateBackendV0_6, StateBackendErrorV0_6};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "kebab-case", deny_unknown_fields)]
pub enum LifecycleEventV0_6 {
    CreateGroup {
        group_id: GroupId,
        principal_id: PrincipalId,
        limits: GroupLimits,
        facts: Document,
    },
    CloseGroup {
        group_id: GroupId,
    },
    CancelGroup {
        group_id: GroupId,
    },
    ExpireGroup {
        group_id: GroupId,
    },
    CreateRequest {
        request_id: RequestId,
        principal_id: PrincipalId,
        group_id: Option<GroupId>,
        fields: Document,
        facts: Document,
    },
    ContinueRequest {
        request_id: RequestId,
        fields: Document,
        facts: Document,
    },
    AdmitRequest {
        request_id: RequestId,
    },
    ActivateRequest {
        request_id: RequestId,
    },
    PauseRequest {
        request_id: RequestId,
    },
    MergeGroupFacts {
        group_id: GroupId,
        facts: Document,
    },
    MergeRequestFacts {
        request_id: RequestId,
        facts: Document,
    },
    ReplaceRequestFields {
        request_id: RequestId,
        fields: Document,
    },
    ReplaceShared {
        shared: Document,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum LifecycleOutcomeV0_6 {
    Group(GroupState),
    Request(RequestState),
    SharedUpdated,
}

#[derive(Clone)]
pub struct LifecycleHostV0_6 {
    backend: Arc<dyn PolicyStateBackendV0_6>,
}

impl LifecycleHostV0_6 {
    pub fn new(backend: Arc<dyn PolicyStateBackendV0_6>) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &Arc<dyn PolicyStateBackendV0_6> {
        &self.backend
    }

    pub fn apply(
        &self,
        event: LifecycleEventV0_6,
    ) -> Result<LifecycleOutcomeV0_6, StateBackendErrorV0_6> {
        match event {
            LifecycleEventV0_6::CreateGroup {
                group_id,
                principal_id,
                limits,
                facts,
            } => self
                .backend
                .create_group(group_id, principal_id, limits, facts)
                .map(LifecycleOutcomeV0_6::Group),
            LifecycleEventV0_6::CloseGroup { group_id } => self
                .backend
                .transition_group(&group_id, GroupStatus::Closed)
                .map(LifecycleOutcomeV0_6::Group),
            LifecycleEventV0_6::CancelGroup { group_id } => self
                .backend
                .transition_group(&group_id, GroupStatus::Cancelled)
                .map(LifecycleOutcomeV0_6::Group),
            LifecycleEventV0_6::ExpireGroup { group_id } => self
                .backend
                .transition_group(&group_id, GroupStatus::Expired)
                .map(LifecycleOutcomeV0_6::Group),
            LifecycleEventV0_6::CreateRequest {
                request_id,
                principal_id,
                group_id,
                fields,
                facts,
            } => self
                .backend
                .create_request(request_id, principal_id, group_id, fields, facts)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::ContinueRequest {
                request_id,
                fields,
                facts,
            } => self
                .backend
                .continue_request(&request_id, fields, facts)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::AdmitRequest { request_id } => self
                .backend
                .transition_request(&request_id, RequestStatus::Admitted)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::ActivateRequest { request_id } => self
                .backend
                .transition_request(&request_id, RequestStatus::Active)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::PauseRequest { request_id } => self
                .backend
                .transition_request(&request_id, RequestStatus::Paused)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::MergeGroupFacts { group_id, facts } => self
                .backend
                .merge_group_facts(&group_id, facts)
                .map(LifecycleOutcomeV0_6::Group),
            LifecycleEventV0_6::MergeRequestFacts { request_id, facts } => self
                .backend
                .merge_request_facts(&request_id, facts)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::ReplaceRequestFields { request_id, fields } => self
                .backend
                .replace_request_fields(&request_id, fields)
                .map(LifecycleOutcomeV0_6::Request),
            LifecycleEventV0_6::ReplaceShared { shared } => {
                self.backend.replace_shared(shared)?;
                Ok(LifecycleOutcomeV0_6::SharedUpdated)
            }
        }
    }

    pub fn apply_all(
        &self,
        events: impl IntoIterator<Item = LifecycleEventV0_6>,
    ) -> Result<Vec<LifecycleOutcomeV0_6>, StateBackendErrorV0_6> {
        events.into_iter().map(|event| self.apply(event)).collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{InMemoryPolicyStateBackendV0_6, WorkingSetV0_6};

    use super::*;

    fn trace() -> Vec<LifecycleEventV0_6> {
        vec![
            LifecycleEventV0_6::CreateGroup {
                group_id: "G".into(),
                principal_id: "tenant".into(),
                limits: GroupLimits {
                    max_members: 2,
                    max_scratch_bytes: 1024,
                },
                facts: json!({"deadline_ms": 1000}),
            },
            LifecycleEventV0_6::CreateRequest {
                request_id: "A".into(),
                principal_id: "tenant".into(),
                group_id: Some("G".into()),
                fields: json!({"body": {}, "metadata": {}}),
                facts: json!({}),
            },
            LifecycleEventV0_6::AdmitRequest {
                request_id: "A".into(),
            },
            LifecycleEventV0_6::ActivateRequest {
                request_id: "A".into(),
            },
            LifecycleEventV0_6::PauseRequest {
                request_id: "A".into(),
            },
            LifecycleEventV0_6::ContinueRequest {
                request_id: "A".into(),
                fields: json!({"body": {"prompt": "next"}, "metadata": {}}),
                facts: json!({"generation_id": 1}),
            },
            LifecycleEventV0_6::ActivateRequest {
                request_id: "A".into(),
            },
        ]
    }

    #[test]
    fn lifecycle_trace_is_deterministic_and_replayable() {
        let run = || {
            let backend = InMemoryPolicyStateBackendV0_6::default();
            let host = LifecycleHostV0_6::new(Arc::new(backend.clone()));
            host.apply_all(trace()).unwrap();
            backend
                .load(&WorkingSetV0_6::default().with_request("A"))
                .unwrap()
                .state
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn unknown_event_fields_are_rejected() {
        assert!(
            serde_json::from_value::<LifecycleEventV0_6>(json!({
                "event": "create-request",
                "request_id": "A",
                "principal_id": "tenant",
                "group_id": null,
                "fields": {},
                "facts": {},
                "forged": true
            }))
            .is_err()
        );
    }
}
