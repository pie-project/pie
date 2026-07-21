use std::sync::Arc;
use std::time::{Duration, Instant};

use pie_plex::{
    AdmissionInput, AdmissionOutput, DecisionValidationError, DenseScores, EvictionInput,
    FeedbackAcknowledgement, FeedbackBatch, FieldLocation, Manifest, MapMutation, Operation,
    PlacementInput, ScheduleInput, ServicePlan,
};
use wasmtime::Store;
use wasmtime::component::{Component, HasSelf, Linker};

use crate::bindings::exports::pie::plex::policy::Guest as PolicyGuest;
use crate::bindings::pie::plex::types::PolicyError;
use crate::bindings::{PlexPolicy, PlexPolicyPre};
use crate::context::InvocationContext;
use crate::convert::{self, ConversionError};
use crate::engine::{PolicyEngine, PolicyEngineConfig};
use crate::error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
use crate::link::{AttachmentResolution, CapabilityCatalog, link_manifest};
use crate::maps::{
    Clock, CommitResult, DedupLimits, FeedbackStart, InvocationTransaction, MapStore, PrepareError,
    PreparedTransaction, SystemClock,
};
use crate::package_format::{PackageLimits, PolicyPackage};
use crate::telemetry::{TelemetryBuffer, TelemetryRecord};

#[derive(Clone)]
pub struct AttachedPolicy {
    inner: Arc<AttachedPolicyInner>,
}

struct AttachedPolicyInner {
    engine: PolicyEngine,
    manifest: Manifest,
    linked: AttachmentResolution,
    maps: MapStore,
    policy_identity: String,
    telemetry: TelemetryBuffer,
    pre: PlexPolicyPre<InvocationContext>,
}

pub struct PreparedDecision<T> {
    decision: T,
    effects: PreparedTransaction,
    attempts: u32,
    metrics: InvocationMetrics,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InvocationMetrics {
    pub snapshot_ns: u64,
    pub instantiate_ns: u64,
    pub execute_ns: u64,
    pub validate_ns: u64,
    pub prepare_ns: u64,
}

impl<T: std::fmt::Debug> std::fmt::Debug for PreparedDecision<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedDecision")
            .field("decision", &self.decision)
            .field("effects", &"<prepared>")
            .field("attempts", &self.attempts)
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl<T> PreparedDecision<T> {
    pub fn decision(&self) -> &T {
        &self.decision
    }

    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    pub fn metrics(&self) -> InvocationMetrics {
        self.metrics
    }

    pub fn commit(self) -> (T, CommitResult) {
        let result = self.effects.commit();
        (self.decision, result)
    }

    pub fn abort(self) -> T {
        self.effects.abort();
        self.decision
    }
}

impl AttachedPolicy {
    pub fn compile_package(
        engine: PolicyEngine,
        package_bytes: &[u8],
        catalog: &CapabilityCatalog,
    ) -> Result<Self, AttachmentError> {
        let config = engine.config();
        let package = PolicyPackage::decode(
            package_bytes,
            PackageLimits {
                max_package_bytes: config.max_package_bytes,
                max_manifest_bytes: config.max_manifest_bytes,
                max_component_bytes: config.max_component_bytes,
            },
        )
        .map_err(AttachmentError::Package)?;
        let (manifest, component) = package.into_parts();
        Self::compile(engine, &component, manifest, catalog)
    }

    pub fn compile(
        engine: PolicyEngine,
        component_bytes: &[u8],
        manifest: Manifest,
        catalog: &CapabilityCatalog,
    ) -> Result<Self, AttachmentError> {
        Self::compile_with_clock(
            engine,
            component_bytes,
            manifest,
            catalog,
            Arc::new(SystemClock::new()),
            DedupLimits::default(),
        )
    }

    pub fn compile_with_clock(
        engine: PolicyEngine,
        component_bytes: &[u8],
        manifest: Manifest,
        catalog: &CapabilityCatalog,
        clock: Arc<dyn Clock>,
        dedup_limits: DedupLimits,
    ) -> Result<Self, AttachmentError> {
        manifest.validate()?;
        validate_host_limits(engine.config(), &manifest)?;
        let linked = link_manifest(&manifest, catalog).map_err(AttachmentError::Resolve)?;
        let maps = MapStore::new(
            linked
                .maps()
                .iter()
                .map(|(handle, declaration)| (*handle, declaration.clone())),
            clock,
            dedup_limits,
        )
        .map_err(AttachmentError::Maps)?;
        if component_bytes.len() > engine.config().max_component_bytes {
            return Err(AttachmentError::ComponentTooLarge {
                actual: component_bytes.len(),
                maximum: engine.config().max_component_bytes,
            });
        }

        let component = Component::new(engine.raw(), component_bytes)
            .map_err(|error| AttachmentError::Compile(error.to_string()))?;
        verify_component_surface(engine.raw(), &component)?;
        let mut linker = Linker::<InvocationContext>::new(engine.raw());
        type Host = HasSelf<InvocationContext>;
        PlexPolicy::add_to_linker::<InvocationContext, Host>(&mut linker, |context| context)
            .map_err(|error| AttachmentError::Link(error.to_string()))?;
        let instance_pre = linker
            .instantiate_pre(&component)
            .map_err(|error| AttachmentError::Link(error.to_string()))?;
        let pre = PlexPolicyPre::new(instance_pre)
            .map_err(|error| AttachmentError::Link(error.to_string()))?;

        let policy_identity = manifest.package_name.clone();
        let telemetry = TelemetryBuffer::new(engine.config().telemetry_buffer_records);
        probe_instantiation(&engine, &manifest, &maps, &telemetry, &pre)?;
        telemetry.drain();
        Ok(Self {
            inner: Arc::new(AttachedPolicyInner {
                engine,
                manifest,
                linked,
                maps,
                policy_identity,
                telemetry,
                pre,
            }),
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.inner.manifest
    }

    pub fn resolution(&self) -> &AttachmentResolution {
        &self.inner.linked
    }

    pub fn maps(&self) -> &MapStore {
        &self.inner.maps
    }

    pub fn drain_telemetry(&self) -> Vec<TelemetryRecord> {
        self.inner.telemetry.drain()
    }

    pub(crate) fn transfer_state_from(
        &self,
        source: &AttachedPolicy,
    ) -> Result<(), crate::StateTransferError> {
        self.inner.maps.transfer_from(
            &source.inner.maps,
            &source.inner.policy_identity,
            &self.inner.policy_identity,
        )
    }

    pub fn admit(
        &self,
        mut input: AdmissionInput,
    ) -> Invocation<PreparedDecision<AdmissionOutput>> {
        if !self.inner.manifest.operations.contains(&Operation::Admit) {
            return Invocation::Unavailable;
        }
        input.links = self.inner.linked.links().clone();
        if let Err(failure) = self.validate_admission_input(&input) {
            return Invocation::FallbackRequired(failure);
        }
        let wit_input = convert::admission_input(&input);
        self.invoke_decision(
            &wit_input,
            |policy, store, input| policy.call_admit(store, input),
            convert::admission_output,
            |output| pie_plex::validate_admission_output(output, &self.inner.manifest.limits),
            |output| &output.mutations,
        )
    }

    pub fn route(&self, mut input: PlacementInput) -> Invocation<PreparedDecision<DenseScores>> {
        if !self.inner.manifest.operations.contains(&Operation::Route) {
            return Invocation::Unavailable;
        }
        input.links = self.inner.linked.links().clone();
        if let Err(failure) = self.validate_placement_input(&input) {
            return Invocation::FallbackRequired(failure);
        }
        let candidate_count = input.placements.candidates.len();
        let wit_input = convert::placement_input(&input);
        self.invoke_decision(
            &wit_input,
            |policy, store, input| policy.call_route(store, input),
            convert::dense_output,
            |output| {
                pie_plex::validate_dense_scores(
                    output,
                    candidate_count,
                    &self.inner.manifest.limits,
                )
            },
            |output| &output.mutations,
        )
    }

    pub fn schedule(&self, mut input: ScheduleInput) -> Invocation<PreparedDecision<ServicePlan>> {
        if !self
            .inner
            .manifest
            .operations
            .contains(&Operation::Schedule)
        {
            return Invocation::Unavailable;
        }
        input.links = self.inner.linked.links().clone();
        if let Err(failure) = self.validate_schedule_input(&input) {
            return Invocation::FallbackRequired(failure);
        }
        let wit_input = convert::schedule_input(&input);
        self.invoke_decision(
            &wit_input,
            |policy, store, input| policy.call_schedule(store, input),
            convert::service_plan,
            |output| {
                pie_plex::validate_service_plan(
                    output,
                    &input,
                    self.inner
                        .linked
                        .has_capability("pie.schedule.token-budget@1"),
                    &self.inner.manifest.limits,
                )
            },
            |output| &output.mutations,
        )
    }

    pub fn evict(&self, mut input: EvictionInput) -> Invocation<PreparedDecision<DenseScores>> {
        if !self.inner.manifest.operations.contains(&Operation::Evict) {
            return Invocation::Unavailable;
        }
        input.links = self.inner.linked.links().clone();
        if let Err(failure) = self.validate_eviction_input(&input) {
            return Invocation::FallbackRequired(failure);
        }
        let candidate_count = input.resident.candidates.len();
        let wit_input = convert::eviction_input(&input);
        self.invoke_decision(
            &wit_input,
            |policy, store, input| policy.call_evict(store, input),
            convert::dense_output,
            |output| {
                pie_plex::validate_dense_scores(
                    output,
                    candidate_count,
                    &self.inner.manifest.limits,
                )
            },
            |output| &output.mutations,
        )
    }

    fn invoke_decision<WInput, WOutput, Output, Call, Convert, Validate, Effects>(
        &self,
        input: &WInput,
        call: Call,
        convert: Convert,
        validate: Validate,
        effects: Effects,
    ) -> Invocation<PreparedDecision<Output>>
    where
        Call: Fn(
            &PolicyGuest,
            &mut Store<InvocationContext>,
            &WInput,
        ) -> wasmtime::Result<Result<WOutput, PolicyError>>,
        Convert: Fn(WOutput) -> Result<Output, ConversionError>,
        Validate: Fn(&Output) -> Result<(), DecisionValidationError>,
        Effects: Fn(&Output) -> &[MapMutation],
    {
        let started = Instant::now();
        let retries = self.inner.engine.config().max_conflict_retries;
        for attempt in 0..=retries {
            let Some(deadline_ms) = self.remaining_deadline_ms(started) else {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::DeadlineExceeded,
                    "policy retry deadline elapsed",
                ));
            };
            match self.invoke_decision_once(
                input,
                deadline_ms,
                &call,
                &convert,
                &validate,
                &effects,
            ) {
                Ok(mut prepared) => {
                    prepared.attempts = attempt + 1;
                    return Invocation::Success(prepared);
                }
                Err(failure)
                    if failure.kind == InvocationFailureKind::TransactionConflict
                        && attempt < retries =>
                {
                    continue;
                }
                Err(failure) => return Invocation::FallbackRequired(failure),
            }
        }
        unreachable!("bounded retry loop always returns")
    }

    fn invoke_decision_once<WInput, WOutput, Output, Call, Convert, Validate, Effects>(
        &self,
        input: &WInput,
        deadline_ms: u64,
        call: &Call,
        convert: &Convert,
        validate: &Validate,
        effects: &Effects,
    ) -> Result<PreparedDecision<Output>, InvocationFailure>
    where
        Call: Fn(
            &PolicyGuest,
            &mut Store<InvocationContext>,
            &WInput,
        ) -> wasmtime::Result<Result<WOutput, PolicyError>>,
        Convert: Fn(WOutput) -> Result<Output, ConversionError>,
        Validate: Fn(&Output) -> Result<(), DecisionValidationError>,
        Effects: Fn(&Output) -> &[MapMutation],
    {
        let snapshot_started = Instant::now();
        let transaction = self.inner.maps.begin().map_err(|error| {
            InvocationFailure::new(InvocationFailureKind::Instantiation, error.to_string())
        })?;
        let snapshot_ns = duration_ns(snapshot_started.elapsed());
        let instantiate_started = Instant::now();
        let (mut store, policy) = self.instantiate_attempt(transaction, deadline_ms)?;
        let instantiate_ns = duration_ns(instantiate_started.elapsed());
        let execute_started = Instant::now();
        let result = call(policy.pie_plex_policy(), &mut store, input);
        let execute_ns = duration_ns(execute_started.elapsed());
        if let Some(failure) = store.data_mut().take_failure() {
            return Err(failure);
        }
        let wire_output = match result {
            Ok(Ok(output)) => output,
            Ok(Err(_)) => {
                return Err(InvocationFailure::new(
                    InvocationFailureKind::PolicyFallback,
                    "policy requested native fallback",
                ));
            }
            Err(error) => {
                return Err(classify_wasmtime(InvocationFailureKind::Trap, error));
            }
        };
        let validate_started = Instant::now();
        let output = convert(wire_output).map_err(|error| {
            InvocationFailure::new(InvocationFailureKind::InvalidOutput, error.to_string())
        })?;
        validate(&output).map_err(InvocationFailure::output)?;
        let validate_ns = duration_ns(validate_started.elapsed());
        let prepare_started = Instant::now();
        let mut transaction = store.data_mut().take_transaction();
        transaction
            .stage_all(effects(&output).iter().cloned())
            .map_err(|error| {
                InvocationFailure::new(InvocationFailureKind::InvalidOutput, error.to_string())
            })?;
        let prepared = transaction.prepare().map_err(prepare_failure)?;
        let prepare_ns = duration_ns(prepare_started.elapsed());
        Ok(PreparedDecision {
            decision: output,
            effects: prepared,
            attempts: 1,
            metrics: InvocationMetrics {
                snapshot_ns,
                instantiate_ns,
                execute_ns,
                validate_ns,
                prepare_ns,
            },
        })
    }

    fn validate_admission_input(&self, input: &AdmissionInput) -> Result<(), InvocationFailure> {
        input
            .validate(&self.inner.manifest.limits)
            .map_err(InvocationFailure::input)?;
        self.validate_records(
            &input.request.fields,
            Operation::Admit,
            FieldLocation::Request,
        )
    }

    fn validate_placement_input(&self, input: &PlacementInput) -> Result<(), InvocationFailure> {
        input
            .validate(&self.inner.manifest.limits)
            .map_err(InvocationFailure::input)?;
        self.validate_records(
            &input.request.fields,
            Operation::Route,
            FieldLocation::Request,
        )?;
        self.validate_records(
            &input.placements.fields,
            Operation::Route,
            FieldLocation::Candidate,
        )
    }

    fn validate_schedule_input(&self, input: &ScheduleInput) -> Result<(), InvocationFailure> {
        input
            .validate(&self.inner.manifest.limits)
            .map_err(InvocationFailure::input)?;
        self.validate_records(
            &input.runnable.fields,
            Operation::Schedule,
            FieldLocation::Candidate,
        )
    }

    fn validate_eviction_input(&self, input: &EvictionInput) -> Result<(), InvocationFailure> {
        input
            .validate(&self.inner.manifest.limits)
            .map_err(InvocationFailure::input)?;
        self.validate_records(
            &input.resident.fields,
            Operation::Evict,
            FieldLocation::Candidate,
        )
    }

    fn validate_records(
        &self,
        records: &pie_plex::RecordBatch,
        operation: Operation,
        location: FieldLocation,
    ) -> Result<(), InvocationFailure> {
        let maximum = usize::try_from(self.inner.manifest.limits.input_bytes).unwrap_or(usize::MAX);
        records
            .validate_against(
                self.inner.linked.record_schema(operation, location),
                maximum,
            )
            .map_err(|error| {
                InvocationFailure::input(pie_plex::OperationInputError::Records(error))
            })
    }

    fn instantiate_attempt(
        &self,
        transaction: InvocationTransaction,
        deadline_ms: u64,
    ) -> Result<(Store<InvocationContext>, PlexPolicy), InvocationFailure> {
        let permit = self.inner.engine.try_acquire().ok_or_else(|| {
            InvocationFailure::new(
                InvocationFailureKind::HostSaturated,
                "policy engine has no free invocation slot",
            )
        })?;
        let memory_bytes =
            usize::try_from(self.inner.manifest.limits.memory_bytes).unwrap_or(usize::MAX);
        let mut store = InvocationContext::store(
            self.inner.engine.raw(),
            memory_bytes,
            transaction,
            &self.inner.manifest.limits,
            self.inner.telemetry.clone(),
            permit,
        );
        store
            .set_fuel(self.inner.manifest.limits.fuel)
            .map_err(|error| {
                InvocationFailure::new(InvocationFailureKind::Instantiation, error.to_string())
            })?;
        store.set_epoch_deadline(self.inner.engine.deadline_ticks(deadline_ms));
        store.epoch_deadline_trap();
        let policy = self
            .inner
            .pre
            .instantiate(&mut store)
            .map_err(|error| classify_wasmtime(InvocationFailureKind::Instantiation, error))?;
        Ok((store, policy))
    }

    fn remaining_deadline_ms(&self, started: Instant) -> Option<u64> {
        if !self.inner.engine.uses_realtime_epochs() {
            return Some(self.inner.manifest.limits.deadline_ms);
        }
        let total = Duration::from_millis(self.inner.manifest.limits.deadline_ms);
        let remaining = total.checked_sub(started.elapsed())?;
        let nanos = remaining.as_nanos();
        let millis = nanos.saturating_add(999_999) / 1_000_000;
        Some(u64::try_from(millis.max(1)).unwrap_or(u64::MAX))
    }

    pub fn feedback(&self, mut input: FeedbackBatch) -> Invocation<FeedbackAcknowledgement> {
        if !self
            .inner
            .manifest
            .operations
            .contains(&Operation::Feedback)
        {
            return Invocation::Unavailable;
        }
        input.links = self.inner.linked.links().clone();
        if input.events.iter().any(|event| {
            !input
                .links
                .events
                .iter()
                .flatten()
                .any(|linked| linked == event)
        }) {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                "feedback references an unlinked event handle",
            ));
        }
        if let Err(error) = input.validate(&self.inner.manifest.limits) {
            return Invocation::FallbackRequired(InvocationFailure::input(error));
        }
        if let Err(failure) =
            self.validate_records(&input.records, Operation::Feedback, FieldLocation::Feedback)
        {
            return Invocation::FallbackRequired(failure);
        }

        let wit_input = convert::feedback_input(&input);
        let started = Instant::now();
        let retries = self.inner.engine.config().max_conflict_retries;
        for attempt in 0..=retries {
            let Some(deadline_ms) = self.remaining_deadline_ms(started) else {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::DeadlineExceeded,
                    "feedback retry deadline elapsed",
                ));
            };
            match self.feedback_once(&input, &wit_input, deadline_ms) {
                Ok(acknowledgement) => return Invocation::Success(acknowledgement),
                Err(failure)
                    if failure.kind == InvocationFailureKind::TransactionConflict
                        && attempt < retries =>
                {
                    continue;
                }
                Err(failure) => return Invocation::FallbackRequired(failure),
            }
        }
        unreachable!("bounded retry loop always returns")
    }

    fn feedback_once(
        &self,
        input: &FeedbackBatch,
        wit_input: &crate::bindings::pie::plex::types::FeedbackInput,
        deadline_ms: u64,
    ) -> Result<FeedbackAcknowledgement, InvocationFailure> {
        let transaction = match self
            .inner
            .maps
            .begin_feedback(&self.inner.policy_identity, input.delivery_id)
            .map_err(|error| {
                InvocationFailure::new(InvocationFailureKind::Instantiation, error.to_string())
            })? {
            FeedbackStart::Duplicate(acknowledgement) => return Ok(acknowledgement),
            FeedbackStart::New(transaction) => transaction,
        };
        let (mut store, policy) = self.instantiate_attempt(transaction, deadline_ms)?;
        let result = policy
            .pie_plex_policy()
            .call_feedback(&mut store, wit_input);
        if let Some(failure) = store.data_mut().take_failure() {
            return Err(failure);
        }
        let wire_output = match result {
            Ok(Ok(output)) => output,
            Ok(Err(_)) => {
                return Err(InvocationFailure::new(
                    InvocationFailureKind::PolicyFallback,
                    "policy requested native fallback",
                ));
            }
            Err(error) => {
                return Err(classify_wasmtime(InvocationFailureKind::Trap, error));
            }
        };
        let output = convert::feedback_output(wire_output).map_err(|error| {
            InvocationFailure::new(InvocationFailureKind::InvalidOutput, error.to_string())
        })?;
        pie_plex::validate_feedback_output(&output, &self.inner.manifest.limits)
            .map_err(InvocationFailure::output)?;
        let mut transaction = store.data_mut().take_transaction();
        transaction.stage_all(output.mutations).map_err(|error| {
            InvocationFailure::new(InvocationFailureKind::InvalidOutput, error.to_string())
        })?;
        let prepared = match transaction.prepare() {
            Ok(prepared) => prepared,
            Err(PrepareError::Duplicate(acknowledgement)) => return Ok(acknowledgement),
            Err(error) => return Err(prepare_failure(error)),
        };
        Ok(prepared
            .commit()
            .feedback
            .expect("feedback commit always returns an acknowledgement"))
    }
}

fn classify_wasmtime(default: InvocationFailureKind, error: wasmtime::Error) -> InvocationFailure {
    let kind = match error.downcast_ref::<wasmtime::Trap>() {
        Some(wasmtime::Trap::OutOfFuel) => InvocationFailureKind::FuelExhausted,
        Some(wasmtime::Trap::Interrupt) => InvocationFailureKind::DeadlineExceeded,
        _ => default,
    };
    InvocationFailure::new(kind, error.to_string())
}

fn prepare_failure(error: PrepareError) -> InvocationFailure {
    let kind = if matches!(error, PrepareError::Conflict) {
        InvocationFailureKind::TransactionConflict
    } else {
        InvocationFailureKind::InvalidOutput
    };
    InvocationFailure::new(kind, error.to_string())
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn probe_instantiation(
    engine: &PolicyEngine,
    manifest: &Manifest,
    maps: &MapStore,
    telemetry: &TelemetryBuffer,
    pre: &PlexPolicyPre<InvocationContext>,
) -> Result<(), AttachmentError> {
    let transaction = maps.begin().map_err(AttachmentError::Maps)?;
    let permit = engine
        .try_acquire()
        .ok_or(AttachmentError::EngineSaturated)?;
    let memory_bytes = usize::try_from(manifest.limits.memory_bytes).unwrap_or(usize::MAX);
    let mut store = InvocationContext::store(
        engine.raw(),
        memory_bytes,
        transaction,
        &manifest.limits,
        telemetry.clone(),
        permit,
    );
    store
        .set_fuel(manifest.limits.fuel)
        .map_err(|error| AttachmentError::Instantiate(error.to_string()))?;
    store.set_epoch_deadline(engine.deadline_ticks(manifest.limits.deadline_ms));
    store.epoch_deadline_trap();
    pre.instantiate(&mut store)
        .map_err(|error| AttachmentError::Instantiate(error.to_string()))?;
    Ok(())
}

fn verify_component_surface(
    engine: &wasmtime::Engine,
    component: &Component,
) -> Result<(), AttachmentError> {
    const TYPES: &str = "pie:plex/types@0.1.0";
    const MAPS: &str = "pie:plex/maps@0.1.0";
    const TELEMETRY: &str = "pie:plex/telemetry@0.1.0";
    const POLICY: &str = "pie:plex/policy@0.1.0";

    let component_type = component.component_type();
    for (name, _) in component_type.imports(engine) {
        if name != TYPES && name != MAPS && name != TELEMETRY {
            return Err(AttachmentError::UnsupportedImport(name.to_owned()));
        }
    }
    let mut policy_exported = false;
    for (name, _) in component_type.exports(engine) {
        if name == POLICY {
            policy_exported = true;
        } else {
            return Err(AttachmentError::UnsupportedExport(name.to_owned()));
        }
    }
    if !policy_exported {
        return Err(AttachmentError::MissingPolicyExport);
    }
    Ok(())
}

fn validate_host_limits(
    host: &PolicyEngineConfig,
    manifest: &Manifest,
) -> Result<(), AttachmentError> {
    let limits = &manifest.limits;
    check_limit(
        "memory_bytes",
        limits.memory_bytes,
        host.max_memory_bytes as u64,
    )?;
    check_limit("fuel", limits.fuel, host.max_fuel)?;
    check_limit("deadline_ms", limits.deadline_ms, host.max_deadline_ms)?;
    check_limit("input_bytes", limits.input_bytes, host.max_input_bytes)?;
    check_limit("output_bytes", limits.output_bytes, host.max_output_bytes)?;
    check_limit(
        "map_calls",
        u64::from(limits.map_calls),
        u64::from(host.max_map_calls),
    )?;
    check_limit("map_bytes", limits.map_bytes, host.max_map_bytes)?;
    check_limit(
        "map_count",
        u64::try_from(manifest.maps.len()).unwrap_or(u64::MAX),
        u64::try_from(host.max_maps).unwrap_or(u64::MAX),
    )?;
    let mut map_entries = 0u64;
    let mut map_storage_bytes = 0u64;
    for declaration in &manifest.maps {
        let entries = u64::from(declaration.schema.max_entries);
        map_entries = map_entries.saturating_add(entries);
        let entry_bytes = u64::from(declaration.schema.max_key_bytes)
            .saturating_add(u64::from(declaration.schema.max_value_bytes))
            .saturating_add(256);
        map_storage_bytes = map_storage_bytes.saturating_add(entries.saturating_mul(entry_bytes));
    }
    check_limit("map_entries", map_entries, host.max_map_entries)?;
    // One active root plus one prepared or retained snapshot root per map.
    check_limit(
        "map_storage_bytes",
        map_storage_bytes.saturating_mul(2),
        host.max_map_storage_bytes,
    )?;
    check_limit(
        "staged_mutations",
        u64::from(limits.staged_mutations),
        u64::from(host.max_staged_mutations),
    )?;
    check_limit(
        "feedback_records",
        u64::from(limits.feedback_records),
        u64::from(host.max_feedback_records),
    )?;
    check_limit(
        "telemetry_records",
        u64::from(limits.telemetry_records),
        u64::from(host.max_telemetry_records),
    )?;
    check_limit(
        "telemetry_bytes",
        limits.telemetry_bytes,
        host.max_telemetry_bytes,
    )
}

fn check_limit(field: &'static str, requested: u64, maximum: u64) -> Result<(), AttachmentError> {
    if requested > maximum {
        Err(AttachmentError::HostLimit {
            field,
            requested,
            maximum,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_component_imports_outside_plex() {
        let engine = PolicyEngine::new(PolicyEngineConfig::deterministic_replay()).unwrap();
        let component = Component::new(
            engine.raw(),
            r#"(component
                (type $f (func))
                (import "wasi:evil/run@0.1.0" (func (type $f)))
            )"#,
        )
        .unwrap();
        assert!(matches!(
            verify_component_surface(engine.raw(), &component),
            Err(AttachmentError::UnsupportedImport(name))
                if name == "wasi:evil/run@0.1.0"
        ));
    }

    #[test]
    fn rejects_component_without_policy_export() {
        let engine = PolicyEngine::new(PolicyEngineConfig::deterministic_replay()).unwrap();
        let component = Component::new(engine.raw(), "(component)").unwrap();
        assert!(matches!(
            verify_component_surface(engine.raw(), &component),
            Err(AttachmentError::MissingPolicyExport)
        ));
    }
}
