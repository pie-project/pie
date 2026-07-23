use serde::{Deserialize, Serialize};

use crate::Document;

use super::MechanicId;

macro_rules! opaque_string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

opaque_string_id!(OpportunityId);
opaque_string_id!(SnapshotId);
opaque_string_id!(RequestId);
opaque_string_id!(GroupId);
opaque_string_id!(PrincipalId);
opaque_string_id!(TargetId);
opaque_string_id!(CacheObjectId);
opaque_string_id!(DeliveryId);
opaque_string_id!(EpisodeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ActionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Operation {
    Admit,
    Route,
    Schedule,
    Cache,
    Feedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GroupStatus {
    Open,
    Closed,
    Cancelled,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RequestStatus {
    Pending,
    Admitted,
    Active,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Expired,
    Rejected,
}

impl RequestStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Expired | Self::Rejected
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRef {
    pub id: SnapshotId,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionMeta {
    pub opportunity_id: OpportunityId,
    pub snapshot: SnapshotRef,
    pub attempt: u32,
    pub mechanics: Vec<MechanicId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestRef {
    pub request_id: RequestId,
    pub generation_id: u64,
    pub group_id: Option<GroupId>,
    pub principal_id: PrincipalId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupLimits {
    pub max_members: u32,
    pub max_scratch_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupState {
    pub group_id: GroupId,
    pub principal_id: PrincipalId,
    pub status: GroupStatus,
    pub limits: GroupLimits,
    pub member_count: u32,
    pub facts: Document,
    pub scratch: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestState {
    pub request: RequestRef,
    pub status: RequestStatus,
    pub facts: Document,
    pub fields: Document,
    pub scratch: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyState {
    pub shared: Document,
    pub groups: Vec<GroupState>,
    pub requests: Vec<RequestState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupStateUpdate {
    pub group_id: GroupId,
    pub scratch: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestStateUpdate {
    pub request_id: RequestId,
    pub fields: Option<Document>,
    pub scratch: Option<Document>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateUpdate {
    pub shared: Option<Document>,
    pub groups: Vec<GroupStateUpdate>,
    pub requests: Vec<RequestStateUpdate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAmount {
    pub name: String,
    pub unit: String,
    pub amount: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimit {
    pub name: String,
    pub unit: String,
    pub maximum: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdmitCause {
    Arrival,
    Retry,
    CapacityChanged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionCandidate {
    pub request: RequestRef,
    pub demand: Vec<ResourceAmount>,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionCapacity {
    pub max_accepted: u32,
    pub limits: Vec<ResourceLimit>,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmitContext {
    pub meta: DecisionMeta,
    pub cause: AdmitCause,
    pub candidates: Vec<AdmissionCandidate>,
    pub capacity: AdmissionCapacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdmissionDecision {
    Accept,
    Defer,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmitPlan {
    pub decisions: Vec<AdmissionDecision>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteCause {
    Admission,
    Retry,
    Rebalance,
    TargetChanged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteRequest {
    pub request: RequestRef,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteTarget {
    pub target_id: TargetId,
    pub max_assignments: u32,
    pub capacity: Vec<ResourceLimit>,
    pub revision: u64,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteEdge {
    pub request_index: u32,
    pub target_index: u32,
    pub demand: Vec<ResourceAmount>,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteContext {
    pub meta: DecisionMeta,
    pub cause: RouteCause,
    pub requests: Vec<RouteRequest>,
    pub targets: Vec<RouteTarget>,
    pub feasible_edges: Vec<RouteEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", content = "edge_index", rename_all = "kebab-case")]
pub enum RouteDecision {
    Assign(u32),
    Defer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutePlan {
    pub decisions: Vec<RouteDecision>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScheduleCause {
    Arrival,
    Completion,
    CapacityChanged,
    Timer,
    Feedback,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleCandidate {
    pub request: RequestRef,
    pub max_token_budget: u32,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleCapacity {
    pub max_selections: u32,
    pub max_requests: u32,
    pub max_total_tokens: u64,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleContext {
    pub meta: DecisionMeta,
    pub cause: ScheduleCause,
    pub runnable: Vec<ScheduleCandidate>,
    pub capacity: ScheduleCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleSelection {
    pub requests: Vec<u32>,
    pub token_budgets: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulePlan {
    pub selections: Vec<ScheduleSelection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheCause {
    Insertion,
    Pressure,
    Expiry,
    DependencyProgress,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "kebab-case")]
pub enum Beneficiary {
    Request(RequestId),
    Group(GroupId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheObject {
    pub object_id: CacheObjectId,
    pub size_bytes: u64,
    pub beneficiaries: Vec<Beneficiary>,
    pub beneficiary_count: u32,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentCacheObject {
    pub object: CacheObject,
    pub reclaimable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheCapacity {
    pub max_bytes: u64,
    pub fixed_bytes: u64,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheEpisode {
    pub episode_id: EpisodeId,
    pub iteration: u32,
    pub max_iterations: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheContext {
    pub meta: DecisionMeta,
    pub cause: CacheCause,
    pub resident: Vec<ResidentCacheObject>,
    pub prospective: Vec<CacheObject>,
    pub capacity: CacheCapacity,
    pub episode: Option<CacheEpisode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheAdmission {
    Cache,
    Bypass,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePlan {
    pub admissions: Vec<CacheAdmission>,
    pub reclaim: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteAssignmentSubject {
    pub opportunity_id: OpportunityId,
    pub request_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleSelectionSubject {
    pub opportunity_id: OpportunityId,
    pub selection_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum FeedbackSubject {
    Request(RequestId),
    WorkGroup(GroupId),
    CacheObject(CacheObjectId),
    RouteAssignment(RouteAssignmentSubject),
    ScheduleSelection(ScheduleSelectionSubject),
    Action(ActionId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutcomeKind {
    Progress,
    Completed,
    Failed,
    Cancelled,
    Expired,
    ActionSucceeded,
    ActionFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackRecord {
    pub subject: FeedbackSubject,
    pub outcome: OutcomeKind,
    pub facts: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackContext {
    pub delivery_id: DeliveryId,
    pub records: Vec<FeedbackRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyError {
    pub code: String,
    pub message: String,
    pub details: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmitInvocation {
    pub context: AdmitContext,
    pub state: PolicyState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmitOutput {
    pub plan: AdmitPlan,
    pub state_update: StateUpdate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteInvocation {
    pub context: RouteContext,
    pub state: PolicyState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteOutput {
    pub plan: RoutePlan,
    pub state_update: StateUpdate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleInvocation {
    pub context: ScheduleContext,
    pub state: PolicyState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleOutput {
    pub plan: SchedulePlan,
    pub state_update: StateUpdate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheInvocation {
    pub context: CacheContext,
    pub state: PolicyState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheOutput {
    pub plan: CachePlan,
    pub state_update: StateUpdate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackInvocation {
    pub context: FeedbackContext,
    pub state: PolicyState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackOutput {
    pub state_update: StateUpdate,
}
