use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{
    CapabilityHandle, DeliveryId, EventHandle, FactHandle, GenerationId, LogicalRequestId,
    MapHandle, MetadataHandle,
};
use crate::manifest::PolicyLimits;
use crate::map::{MapMutation, Revision};
use crate::record::{RecordBatch, RecordValidationError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Operation {
    Admit,
    Route,
    Schedule,
    Evict,
    Feedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementCause {
    GenerationArrival,
    StageTransition,
    Continuation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScheduleCause {
    Enqueue,
    ServiceStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvictionCause {
    AllocationDeficit,
    MemoryWatermark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TerminalOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "operation", content = "cause", rename_all = "kebab-case")]
pub enum InvocationCause {
    Admit,
    Route(PlacementCause),
    Schedule(ScheduleCause),
    Evict(EvictionCause),
    Feedback,
}

/// Attachment resolution by manifest declaration ordinal.
///
/// Required `None` entries reject attachment. Optional `None` entries are
/// passed to the guest explicitly; they never cause a nearby capability or
/// field to be substituted.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LinkSet {
    pub facts: Vec<Option<FactHandle>>,
    pub metadata: Vec<Option<MetadataHandle>>,
    pub maps: Vec<Option<MapHandle>>,
    pub events: Vec<Option<EventHandle>>,
    pub capabilities: Vec<Option<CapabilityHandle>>,
}

impl LinkSet {
    fn charged_bytes(&self) -> Result<usize, OperationInputError> {
        let mut total = 0usize;
        for len in [
            self.facts.len(),
            self.metadata.len(),
            self.maps.len(),
            self.events.len(),
            self.capabilities.len(),
        ] {
            total = checked_input_add(total, 4)?;
            total = checked_input_add(total, checked_input_mul(len, 5)?)?;
        }
        Ok(total)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestContext {
    pub logical_request_id: LogicalRequestId,
    pub generation_id: Option<GenerationId>,
    /// One-row request-scoped fact and metadata batch.
    pub fields: RecordBatch,
}

impl RequestContext {
    fn validate(&self) -> Result<(), OperationInputError> {
        if self.fields.rows != 1 {
            return Err(OperationInputError::RequestRows(self.fields.rows));
        }
        self.fields
            .validate(usize::MAX)
            .map_err(OperationInputError::Records)
    }

    fn charged_bytes(&self) -> Result<usize, OperationInputError> {
        // Logical ID + generation option tag/payload + field batch.
        let generation = 1 + usize::from(self.generation_id.is_some()) * 8;
        checked_input_add(
            checked_input_add(16, generation)?,
            self.fields
                .charged_bytes()
                .map_err(OperationInputError::Records)?,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmissionInput {
    pub links: LinkSet,
    pub request: RequestContext,
}

impl AdmissionInput {
    pub fn validate(&self, limits: &PolicyLimits) -> Result<(), OperationInputError> {
        self.request.validate()?;
        let charged =
            checked_input_add(self.links.charged_bytes()?, self.request.charged_bytes()?)?;
        enforce_input_limit(charged, limits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdmissionDecision {
    Accept,
    Defer,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmissionOutput {
    pub decision: AdmissionDecision,
    pub mutations: Vec<MapMutation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSet<T> {
    pub candidates: Vec<T>,
    /// Candidate-aligned fact and metadata columns.
    pub fields: RecordBatch,
}

impl<T> CandidateSet<T> {
    pub fn validate(&self) -> Result<(), OperationInputError> {
        if usize::try_from(self.fields.rows).ok() != Some(self.candidates.len()) {
            return Err(OperationInputError::CandidateRows {
                candidates: self.candidates.len(),
                rows: self.fields.rows,
            });
        }
        self.fields
            .validate(usize::MAX)
            .map_err(OperationInputError::Records)
    }
}

impl<T: CandidateCharge> CandidateSet<T> {
    fn charged_bytes(&self) -> Result<usize, OperationInputError> {
        let mut total = 4usize;
        for candidate in &self.candidates {
            total = checked_input_add(total, candidate.charged_bytes())?;
        }
        checked_input_add(
            total,
            self.fields
                .charged_bytes()
                .map_err(OperationInputError::Records)?,
        )
    }
}

pub trait CandidateCharge {
    fn charged_bytes(&self) -> usize;
}

/// Placement identity remains host-local. Policy-visible placement properties
/// are supplied through the aligned candidate columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementCandidate;

impl CandidateCharge for PlacementCandidate {
    fn charged_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementInput {
    pub links: LinkSet,
    pub cause: PlacementCause,
    pub request: RequestContext,
    pub placements: CandidateSet<PlacementCandidate>,
}

impl PlacementInput {
    pub fn validate(&self, limits: &PolicyLimits) -> Result<(), OperationInputError> {
        self.request.validate()?;
        self.placements.validate()?;
        let mut charged = self.links.charged_bytes()?;
        charged = checked_input_add(charged, 1)?;
        charged = checked_input_add(charged, self.request.charged_bytes()?)?;
        charged = checked_input_add(charged, self.placements.charged_bytes()?)?;
        enforce_input_limit(charged, limits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceCandidate {
    pub logical_request_id: LogicalRequestId,
    pub generation_id: GenerationId,
    /// Adapter-enforced upper bound for this candidate in one opportunity.
    pub max_token_budget: u32,
}

impl CandidateCharge for ServiceCandidate {
    fn charged_bytes(&self) -> usize {
        28
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceCapacity {
    pub max_selected: u32,
    pub max_total_tokens: u32,
    pub max_token_budget: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduleInput {
    pub links: LinkSet,
    pub cause: ScheduleCause,
    pub runnable: CandidateSet<ServiceCandidate>,
    pub capacity: ServiceCapacity,
}

impl ScheduleInput {
    pub fn validate(&self, limits: &PolicyLimits) -> Result<(), OperationInputError> {
        self.runnable.validate()?;
        let mut charged = self.links.charged_bytes()?;
        charged = checked_input_add(charged, 1)?;
        charged = checked_input_add(charged, self.runnable.charged_bytes()?)?;
        charged = checked_input_add(charged, 12)?;
        enforce_input_limit(charged, limits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentUnit {
    pub size_bytes: u64,
    pub logical_request_id: Option<LogicalRequestId>,
    pub generation_id: Option<GenerationId>,
}

impl CandidateCharge for ResidentUnit {
    fn charged_bytes(&self) -> usize {
        8 + 1
            + usize::from(self.logical_request_id.is_some()) * 16
            + 1
            + usize::from(self.generation_id.is_some()) * 8
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvictionInput {
    pub links: LinkSet,
    pub cause: EvictionCause,
    pub bytes_needed: u64,
    pub resident: CandidateSet<ResidentUnit>,
}

impl EvictionInput {
    pub fn validate(&self, limits: &PolicyLimits) -> Result<(), OperationInputError> {
        self.resident.validate()?;
        let mut charged = self.links.charged_bytes()?;
        charged = checked_input_add(charged, 9)?;
        charged = checked_input_add(charged, self.resident.charged_bytes()?)?;
        enforce_input_limit(charged, limits)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackBatch {
    pub links: LinkSet,
    pub delivery_id: DeliveryId,
    pub events: Vec<EventHandle>,
    pub subjects: Vec<FeedbackSubject>,
    pub records: RecordBatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackSubject {
    pub logical_request_id: LogicalRequestId,
    pub generation_id: Option<GenerationId>,
    pub terminal_outcome: Option<TerminalOutcome>,
}

impl FeedbackBatch {
    pub fn validate(&self, limits: &PolicyLimits) -> Result<(), OperationInputError> {
        if usize::try_from(self.records.rows).ok() != Some(self.events.len()) {
            return Err(OperationInputError::FeedbackRows {
                events: self.events.len(),
                rows: self.records.rows,
            });
        }
        if self.subjects.len() != self.events.len() {
            return Err(OperationInputError::FeedbackSubjects {
                events: self.events.len(),
                subjects: self.subjects.len(),
            });
        }
        if self.events.len() > limits.feedback_records as usize {
            return Err(OperationInputError::TooManyFeedbackRecords {
                actual: self.events.len(),
                maximum: limits.feedback_records as usize,
            });
        }
        self.records
            .validate(usize::MAX)
            .map_err(OperationInputError::Records)?;

        let mut charged = self.links.charged_bytes()?;
        charged = checked_input_add(charged, 20)?;
        charged = checked_input_add(charged, checked_input_mul(self.events.len(), 4)?)?;
        charged = checked_input_add(charged, checked_input_mul(self.subjects.len(), 27)?)?;
        charged = checked_input_add(
            charged,
            self.records
                .charged_bytes()
                .map_err(OperationInputError::Records)?,
        )?;
        enforce_input_limit(charged, limits)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseScores {
    pub scores: Vec<f64>,
    pub mutations: Vec<MapMutation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ServiceDecision {
    pub score: f64,
    /// `None` requests the host maximum; `Some(0)` requests no service.
    pub token_budget: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServicePlan {
    pub decisions: Vec<ServiceDecision>,
    pub mutations: Vec<MapMutation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackOutput {
    pub mutations: Vec<MapMutation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackAck {
    pub delivery_id: DeliveryId,
    pub revision: Revision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "acknowledgement", rename_all = "kebab-case")]
pub enum FeedbackAcknowledgement {
    Committed(FeedbackAck),
    Duplicate(FeedbackAck),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedService {
    pub candidate_index: u32,
    pub token_budget: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedEviction {
    pub candidate_index: u32,
    pub size_bytes: u64,
}

pub fn validate_admission_output(
    output: &AdmissionOutput,
    limits: &PolicyLimits,
) -> Result<(), DecisionValidationError> {
    validate_effects(&output.mutations, 1, limits)
}

pub fn validate_dense_scores(
    output: &DenseScores,
    candidate_count: usize,
    limits: &PolicyLimits,
) -> Result<(), DecisionValidationError> {
    if output.scores.len() != candidate_count {
        return Err(DecisionValidationError::DenseLength {
            expected: candidate_count,
            actual: output.scores.len(),
        });
    }
    if let Some((index, _)) = output
        .scores
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(DecisionValidationError::NonFiniteScore { index });
    }
    let score_bytes = checked_output_mul(output.scores.len(), 8)?;
    validate_effects(
        &output.mutations,
        checked_output_add(4, score_bytes)?,
        limits,
    )
}

pub fn validate_service_plan(
    output: &ServicePlan,
    input: &ScheduleInput,
    token_budget_capability: bool,
    limits: &PolicyLimits,
) -> Result<(), DecisionValidationError> {
    let candidate_count = input.runnable.candidates.len();
    if output.decisions.len() != candidate_count {
        return Err(DecisionValidationError::DenseLength {
            expected: candidate_count,
            actual: output.decisions.len(),
        });
    }

    for (index, (decision, candidate)) in output
        .decisions
        .iter()
        .zip(&input.runnable.candidates)
        .enumerate()
    {
        if !decision.score.is_finite() {
            return Err(DecisionValidationError::NonFiniteScore { index });
        }
        if let Some(token_budget) = decision.token_budget {
            if !token_budget_capability {
                return Err(DecisionValidationError::TokenBudgetUnsupported { index });
            }
            let maximum = candidate
                .max_token_budget
                .min(input.capacity.max_token_budget);
            if token_budget > maximum {
                return Err(DecisionValidationError::TokenBudgetTooLarge {
                    index,
                    actual: token_budget,
                    maximum,
                });
            }
        }
    }

    // score + option tag + optional u32, charged conservatively for every row.
    let decisions = checked_output_mul(output.decisions.len(), 13)?;
    validate_effects(&output.mutations, checked_output_add(4, decisions)?, limits)
}

pub fn validate_feedback_output(
    output: &FeedbackOutput,
    limits: &PolicyLimits,
) -> Result<(), DecisionValidationError> {
    validate_effects(&output.mutations, 0, limits)
}

/// Stable descending-score greedy fill.
///
/// Ties retain input order. `None` uses the candidate/host maximum, `Some(0)`
/// skips service, and the final candidate may be clamped to remaining total
/// capacity because a token budget is an upper bound rather than an exact grant.
pub fn select_service(
    input: &ScheduleInput,
    output: &ServicePlan,
) -> Result<Vec<SelectedService>, DecisionValidationError> {
    if output.decisions.len() != input.runnable.candidates.len() {
        return Err(DecisionValidationError::DenseLength {
            expected: input.runnable.candidates.len(),
            actual: output.decisions.len(),
        });
    }
    if let Some((index, _)) = output
        .decisions
        .iter()
        .enumerate()
        .find(|(_, decision)| !decision.score.is_finite())
    {
        return Err(DecisionValidationError::NonFiniteScore { index });
    }

    let mut order: Vec<usize> = (0..output.decisions.len()).collect();
    order.sort_by(|left, right| {
        output.decisions[*right]
            .score
            .partial_cmp(&output.decisions[*left].score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });

    let max_selected = input.capacity.max_selected as usize;
    let mut remaining = input.capacity.max_total_tokens;
    let mut selected = Vec::with_capacity(max_selected.min(order.len()));
    for index in order {
        if selected.len() == max_selected || remaining == 0 {
            break;
        }
        let candidate = input.runnable.candidates[index];
        let requested = output.decisions[index].token_budget.unwrap_or_else(|| {
            candidate
                .max_token_budget
                .min(input.capacity.max_token_budget)
        });
        if requested == 0 {
            continue;
        }
        let granted = requested
            .min(candidate.max_token_budget)
            .min(input.capacity.max_token_budget)
            .min(remaining);
        if granted == 0 {
            continue;
        }
        selected.push(SelectedService {
            candidate_index: u32::try_from(index)
                .map_err(|_| DecisionValidationError::CandidateIndexOverflow)?,
            token_budget: granted,
        });
        remaining -= granted;
    }
    Ok(selected)
}

/// Stable descending placement order. Equal scores preserve adapter input order.
pub fn rank_placements(
    input: &PlacementInput,
    output: &DenseScores,
) -> Result<Vec<u32>, DecisionValidationError> {
    rank_scores(output, input.placements.candidates.len(), true)
}

/// Stable low-retention-first fill until `bytes_needed` is satisfied.
pub fn select_evictions(
    input: &EvictionInput,
    output: &DenseScores,
) -> Result<Vec<SelectedEviction>, DecisionValidationError> {
    let order = rank_scores(output, input.resident.candidates.len(), false)?;
    let mut freed = 0u64;
    let mut selected = Vec::new();
    for candidate_index in order {
        if freed >= input.bytes_needed {
            break;
        }
        let index = usize::try_from(candidate_index)
            .map_err(|_| DecisionValidationError::CandidateIndexOverflow)?;
        let size_bytes = input.resident.candidates[index].size_bytes;
        freed = freed.saturating_add(size_bytes);
        selected.push(SelectedEviction {
            candidate_index,
            size_bytes,
        });
    }
    Ok(selected)
}

fn rank_scores(
    output: &DenseScores,
    candidate_count: usize,
    descending: bool,
) -> Result<Vec<u32>, DecisionValidationError> {
    if output.scores.len() != candidate_count {
        return Err(DecisionValidationError::DenseLength {
            expected: candidate_count,
            actual: output.scores.len(),
        });
    }
    if let Some((index, _)) = output
        .scores
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(DecisionValidationError::NonFiniteScore { index });
    }
    let mut order: Vec<usize> = (0..candidate_count).collect();
    order.sort_by(|left, right| {
        let ordering = if descending {
            output.scores[*right].partial_cmp(&output.scores[*left])
        } else {
            output.scores[*left].partial_cmp(&output.scores[*right])
        };
        ordering
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order
        .into_iter()
        .map(|index| {
            u32::try_from(index).map_err(|_| DecisionValidationError::CandidateIndexOverflow)
        })
        .collect()
}

fn validate_effects(
    mutations: &[MapMutation],
    fixed_bytes: usize,
    limits: &PolicyLimits,
) -> Result<(), DecisionValidationError> {
    if mutations.len() > limits.staged_mutations as usize {
        return Err(DecisionValidationError::TooManyMutations {
            actual: mutations.len(),
            maximum: limits.staged_mutations as usize,
        });
    }
    let mut charged = checked_output_add(fixed_bytes, 4)?;
    for mutation in mutations {
        charged = checked_output_add(
            charged,
            mutation
                .charged_bytes()
                .map_err(|_| DecisionValidationError::OutputSizeOverflow)?,
        )?;
    }
    if u64::try_from(charged).unwrap_or(u64::MAX) > limits.output_bytes {
        return Err(DecisionValidationError::OutputTooLarge {
            actual: charged,
            maximum: limits.output_bytes,
        });
    }
    Ok(())
}

fn checked_input_add(lhs: usize, rhs: usize) -> Result<usize, OperationInputError> {
    lhs.checked_add(rhs)
        .ok_or(OperationInputError::InputSizeOverflow)
}

fn checked_input_mul(lhs: usize, rhs: usize) -> Result<usize, OperationInputError> {
    lhs.checked_mul(rhs)
        .ok_or(OperationInputError::InputSizeOverflow)
}

fn checked_output_add(lhs: usize, rhs: usize) -> Result<usize, DecisionValidationError> {
    lhs.checked_add(rhs)
        .ok_or(DecisionValidationError::OutputSizeOverflow)
}

fn checked_output_mul(lhs: usize, rhs: usize) -> Result<usize, DecisionValidationError> {
    lhs.checked_mul(rhs)
        .ok_or(DecisionValidationError::OutputSizeOverflow)
}

fn enforce_input_limit(actual: usize, limits: &PolicyLimits) -> Result<(), OperationInputError> {
    if u64::try_from(actual).unwrap_or(u64::MAX) > limits.input_bytes {
        Err(OperationInputError::InputTooLarge {
            actual,
            maximum: limits.input_bytes,
        })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OperationInputError {
    #[error("request field batch must contain one row, got {0}")]
    RequestRows(u32),
    #[error("candidate list contains {candidates} entries but field batch has {rows} rows")]
    CandidateRows { candidates: usize, rows: u32 },
    #[error("feedback contains {events} event handles but field batch has {rows} rows")]
    FeedbackRows { events: usize, rows: u32 },
    #[error("feedback contains {events} event handles but {subjects} core subjects")]
    FeedbackSubjects { events: usize, subjects: usize },
    #[error("feedback contains {actual} records; maximum is {maximum}")]
    TooManyFeedbackRecords { actual: usize, maximum: usize },
    #[error("invocation input charge overflowed usize")]
    InputSizeOverflow,
    #[error("invocation input charge is {actual} bytes; maximum is {maximum}")]
    InputTooLarge { actual: usize, maximum: u64 },
    #[error(transparent)]
    Records(RecordValidationError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DecisionValidationError {
    #[error("dense result contains {actual} entries; expected {expected}")]
    DenseLength { expected: usize, actual: usize },
    #[error("score at candidate index {index} is not finite")]
    NonFiniteScore { index: usize },
    #[error("candidate index {index} returned a token budget without the required capability")]
    TokenBudgetUnsupported { index: usize },
    #[error("candidate index {index} returned token budget {actual}; maximum is {maximum}")]
    TokenBudgetTooLarge {
        index: usize,
        actual: u32,
        maximum: u32,
    },
    #[error("result staged {actual} mutations; maximum is {maximum}")]
    TooManyMutations { actual: usize, maximum: usize },
    #[error("invocation output charge overflowed usize")]
    OutputSizeOverflow,
    #[error("invocation output charge is {actual} bytes; maximum is {maximum}")]
    OutputTooLarge { actual: usize, maximum: u64 },
    #[error("candidate index does not fit in u32")]
    CandidateIndexOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> PolicyLimits {
        PolicyLimits {
            memory_bytes: 1 << 20,
            fuel: 100_000,
            deadline_ms: 10,
            input_bytes: 1 << 16,
            output_bytes: 1 << 16,
            map_calls: 64,
            map_bytes: 1 << 14,
            staged_mutations: 16,
            feedback_records: 64,
            telemetry_records: 0,
            telemetry_bytes: 0,
        }
    }

    fn schedule_input() -> ScheduleInput {
        ScheduleInput {
            links: LinkSet::default(),
            cause: ScheduleCause::ServiceStep,
            runnable: CandidateSet {
                candidates: vec![
                    ServiceCandidate {
                        logical_request_id: LogicalRequestId::new([1; 16]),
                        generation_id: GenerationId::new(1),
                        max_token_budget: 8,
                    },
                    ServiceCandidate {
                        logical_request_id: LogicalRequestId::new([2; 16]),
                        generation_id: GenerationId::new(2),
                        max_token_budget: 4,
                    },
                    ServiceCandidate {
                        logical_request_id: LogicalRequestId::new([3; 16]),
                        generation_id: GenerationId::new(3),
                        max_token_budget: 8,
                    },
                ],
                fields: RecordBatch::empty(3),
            },
            capacity: ServiceCapacity {
                max_selected: 2,
                max_total_tokens: 7,
                max_token_budget: 6,
            },
        }
    }

    #[test]
    fn validates_dense_scores() {
        validate_dense_scores(
            &DenseScores {
                scores: vec![2.0, 1.0],
                mutations: Vec::new(),
            },
            2,
            &limits(),
        )
        .unwrap();
        assert!(matches!(
            validate_dense_scores(
                &DenseScores {
                    scores: vec![1.0],
                    mutations: Vec::new(),
                },
                2,
                &limits(),
            ),
            Err(DecisionValidationError::DenseLength { .. })
        ));
        assert_eq!(
            validate_dense_scores(
                &DenseScores {
                    scores: vec![f64::NAN],
                    mutations: Vec::new(),
                },
                1,
                &limits(),
            ),
            Err(DecisionValidationError::NonFiniteScore { index: 0 })
        );
    }

    #[test]
    fn validates_token_budget_capability_and_bounds() {
        let input = schedule_input();
        let plan = ServicePlan {
            decisions: vec![
                ServiceDecision {
                    score: 2.0,
                    token_budget: Some(6),
                },
                ServiceDecision {
                    score: 1.0,
                    token_budget: Some(4),
                },
                ServiceDecision {
                    score: 0.0,
                    token_budget: None,
                },
            ],
            mutations: Vec::new(),
        };
        validate_service_plan(&plan, &input, true, &limits()).unwrap();
        assert_eq!(
            validate_service_plan(&plan, &input, false, &limits()),
            Err(DecisionValidationError::TokenBudgetUnsupported { index: 0 })
        );

        let too_large = ServicePlan {
            decisions: vec![
                ServiceDecision {
                    score: 2.0,
                    token_budget: Some(7),
                },
                ServiceDecision {
                    score: 1.0,
                    token_budget: None,
                },
                ServiceDecision {
                    score: 0.0,
                    token_budget: None,
                },
            ],
            mutations: Vec::new(),
        };
        assert_eq!(
            validate_service_plan(&too_large, &input, true, &limits()),
            Err(DecisionValidationError::TokenBudgetTooLarge {
                index: 0,
                actual: 7,
                maximum: 6,
            })
        );
    }

    #[test]
    fn stable_fill_clamps_to_total_capacity() {
        let input = schedule_input();
        let plan = ServicePlan {
            decisions: vec![
                ServiceDecision {
                    score: 10.0,
                    token_budget: Some(6),
                },
                ServiceDecision {
                    score: 10.0,
                    token_budget: None,
                },
                ServiceDecision {
                    score: 9.0,
                    token_budget: Some(0),
                },
            ],
            mutations: Vec::new(),
        };
        assert_eq!(
            select_service(&input, &plan).unwrap(),
            vec![
                SelectedService {
                    candidate_index: 0,
                    token_budget: 6,
                },
                SelectedService {
                    candidate_index: 1,
                    token_budget: 1,
                },
            ]
        );
    }

    #[test]
    fn positive_and_negative_zero_tie_by_input_order() {
        let mut input = schedule_input();
        input.capacity.max_selected = 2;
        input.capacity.max_total_tokens = 2;
        let plan = ServicePlan {
            decisions: vec![
                ServiceDecision {
                    score: -0.0,
                    token_budget: Some(1),
                },
                ServiceDecision {
                    score: 0.0,
                    token_budget: Some(1),
                },
                ServiceDecision {
                    score: -1.0,
                    token_budget: Some(1),
                },
            ],
            mutations: Vec::new(),
        };
        assert_eq!(
            select_service(&input, &plan).unwrap(),
            vec![
                SelectedService {
                    candidate_index: 0,
                    token_budget: 1,
                },
                SelectedService {
                    candidate_index: 1,
                    token_budget: 1,
                },
            ]
        );
    }

    #[test]
    fn placement_ranking_is_descending_and_stable() {
        let input = PlacementInput {
            links: LinkSet::default(),
            cause: PlacementCause::GenerationArrival,
            request: RequestContext {
                logical_request_id: LogicalRequestId::new([1; 16]),
                generation_id: Some(GenerationId::new(1)),
                fields: RecordBatch::empty(1),
            },
            placements: CandidateSet {
                candidates: vec![PlacementCandidate, PlacementCandidate, PlacementCandidate],
                fields: RecordBatch::empty(3),
            },
        };
        let output = DenseScores {
            scores: vec![1.0, 3.0, 3.0],
            mutations: Vec::new(),
        };
        assert_eq!(rank_placements(&input, &output).unwrap(), vec![1, 2, 0]);
    }

    #[test]
    fn eviction_fill_is_low_retention_first() {
        let input = EvictionInput {
            links: LinkSet::default(),
            cause: EvictionCause::AllocationDeficit,
            bytes_needed: 6,
            resident: CandidateSet {
                candidates: vec![
                    ResidentUnit {
                        size_bytes: 4,
                        logical_request_id: None,
                        generation_id: None,
                    },
                    ResidentUnit {
                        size_bytes: 3,
                        logical_request_id: None,
                        generation_id: None,
                    },
                    ResidentUnit {
                        size_bytes: 8,
                        logical_request_id: None,
                        generation_id: None,
                    },
                ],
                fields: RecordBatch::empty(3),
            },
        };
        let output = DenseScores {
            scores: vec![2.0, 1.0, 3.0],
            mutations: Vec::new(),
        };
        assert_eq!(
            select_evictions(&input, &output).unwrap(),
            vec![
                SelectedEviction {
                    candidate_index: 1,
                    size_bytes: 3,
                },
                SelectedEviction {
                    candidate_index: 0,
                    size_bytes: 4,
                },
            ]
        );
    }

    #[test]
    fn complete_input_charge_enforces_limits_without_columns() {
        let input = schedule_input();
        let mut tiny = limits();
        tiny.input_bytes = 1;
        assert!(matches!(
            input.validate(&tiny),
            Err(OperationInputError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn candidate_columns_must_align() {
        let candidates = CandidateSet {
            candidates: vec![PlacementCandidate, PlacementCandidate],
            fields: RecordBatch::empty(1),
        };
        assert_eq!(
            candidates.validate(),
            Err(OperationInputError::CandidateRows {
                candidates: 2,
                rows: 1,
            })
        );
    }

    #[test]
    fn feedback_count_limit_is_enforced() {
        let batch = FeedbackBatch {
            links: LinkSet::default(),
            delivery_id: DeliveryId::new([1; 16]),
            events: vec![EventHandle::new(0), EventHandle::new(0)],
            subjects: vec![
                FeedbackSubject {
                    logical_request_id: LogicalRequestId::new([1; 16]),
                    generation_id: None,
                    terminal_outcome: None,
                };
                2
            ],
            records: RecordBatch::empty(2),
        };
        let mut limits = limits();
        limits.feedback_records = 1;
        assert_eq!(
            batch.validate(&limits),
            Err(OperationInputError::TooManyFeedbackRecords {
                actual: 2,
                maximum: 1,
            })
        );
    }
}
