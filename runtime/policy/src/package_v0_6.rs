use std::collections::BTreeSet;
use std::sync::Arc;

use pie_plex::v0_6::{Manifest, PolicyError, StateUpdate};
use serde::{Deserialize, Serialize};
use wasmtime::Store;
use wasmtime::component::{Component, HasSelf, Linker};

use crate::bindings_v0_6::exports::pie::plex::policy::Guest as PolicyGuest;
use crate::bindings_v0_6::{PlexPolicy, PlexPolicyPre};
use crate::context::{InvocationContext, InvocationContextConfig};
use crate::engine::{PolicyEngine, PolicyEngineConfig};
use crate::error::{Invocation, InvocationFailure, InvocationFailureKind};
use crate::host::{QueryHandler, RejectingQueryHandler, StagedAction};
use crate::package_format::PackageLimits;
use crate::package_format_v0_6::{PackageErrorV0_6, PolicyPackageV0_6};
use crate::protocol_v0_6::{
    NormalizedPlanV0_6, OperationContextV0_6, OperationPlanV0_6, ProtocolLimitsV0_6,
    validate_output_v0_6, validate_snapshot_context_v0_6,
};
use crate::state_store_v0_6::StateSnapshotV0_6;
use crate::wire_v0_6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedPolicyResultV0_6 {
    pub plan: OperationPlanV0_6,
    pub normalized_plan: NormalizedPlanV0_6,
    pub state_update: StateUpdate,
    pub actions: Vec<StagedAction>,
    pub duplicate_feedback: bool,
}

#[derive(Clone)]
pub struct AttachedPolicyV0_6 {
    inner: Arc<AttachedPolicyInnerV0_6>,
}

struct AttachedPolicyInnerV0_6 {
    engine: PolicyEngine,
    manifest: Manifest,
    pre: PlexPolicyPre<InvocationContext>,
}

impl AttachedPolicyV0_6 {
    pub fn compile_package(
        engine: PolicyEngine,
        package_bytes: &[u8],
    ) -> Result<Self, AttachmentErrorV0_6> {
        let config = engine.config();
        let package = PolicyPackageV0_6::decode(
            package_bytes,
            PackageLimits {
                max_package_bytes: config.max_package_bytes,
                max_manifest_bytes: config.max_manifest_bytes,
                max_component_bytes: config.max_component_bytes,
            },
        )
        .map_err(AttachmentErrorV0_6::Package)?;
        let (manifest, component) = package.into_parts();
        Self::compile(engine, &component, manifest)
    }

    pub fn compile(
        engine: PolicyEngine,
        component_bytes: &[u8],
        manifest: Manifest,
    ) -> Result<Self, AttachmentErrorV0_6> {
        manifest.validate()?;
        validate_host_limits(engine.config(), &manifest)?;
        let component = Component::new(engine.raw(), component_bytes)
            .map_err(|error| AttachmentErrorV0_6::Compile(error.to_string()))?;
        verify_component_surface(engine.raw(), &component)?;
        let mut linker = Linker::<InvocationContext>::new(engine.raw());
        PlexPolicy::add_to_linker::<InvocationContext, HasSelf<InvocationContext>>(
            &mut linker,
            |context| context,
        )
        .map_err(|error| AttachmentErrorV0_6::Link(error.to_string()))?;
        let instance_pre = linker
            .instantiate_pre(&component)
            .map_err(|error| AttachmentErrorV0_6::Link(error.to_string()))?;
        let pre = PlexPolicyPre::new(instance_pre)
            .map_err(|error| AttachmentErrorV0_6::Link(error.to_string()))?;
        probe_instantiation(&engine, &manifest, &pre)?;
        Ok(Self {
            inner: Arc::new(AttachedPolicyInnerV0_6 {
                engine,
                manifest,
                pre,
            }),
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.inner.manifest
    }

    pub(crate) fn invoke(
        &self,
        context: OperationContextV0_6,
        state: StateSnapshotV0_6,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: Arc<BTreeSet<String>>,
        protocol_limits: ProtocolLimitsV0_6,
    ) -> Invocation<PreparedPolicyResultV0_6> {
        let operation = context.operation();
        if !self.inner.manifest.implements.contains(&operation) {
            return Invocation::Unavailable;
        }
        if let Err(error) = validate_snapshot_context_v0_6(&context, &state, protocol_limits) {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                error.to_string(),
            ));
        }
        self.invoke_owned(
            context,
            state,
            query_handler,
            supported_actions,
            protocol_limits,
        )
    }

    fn invoke_owned(
        &self,
        context: OperationContextV0_6,
        state: StateSnapshotV0_6,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: Arc<BTreeSet<String>>,
        protocol_limits: ProtocolLimitsV0_6,
    ) -> Invocation<PreparedPolicyResultV0_6> {
        let input_bytes = match serialized_len(&(&context, &state.state)) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidInput,
                    error,
                ));
            }
        };
        if input_bytes > self.inner.manifest.limits.input_bytes {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                "policy invocation exceeds the package byte limit",
            ));
        }

        let _permit = match self.inner.engine.try_acquire() {
            Some(permit) => permit,
            None => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::HostSaturated,
                    "policy engine has no free invocation slot",
                ));
            }
        };
        let (mut store, policy) = match self.instantiate(query_handler, supported_actions) {
            Ok(value) => value,
            Err(failure) => return Invocation::FallbackRequired(failure),
        };
        let guest = policy.pie_plex_policy();
        let (plan, state_update) = match call_operation(&context, &state.state, guest, &mut store) {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                if let Some(failure) = store.data_mut().take_reported_failure(&error.message) {
                    return Invocation::FallbackRequired(failure);
                }
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::PolicyFallback,
                    format!("{}: {}", error.code, error.message),
                ));
            }
            Err(CallErrorV0_6::Wire(error)) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
            Err(CallErrorV0_6::Trap(error)) => {
                return Invocation::FallbackRequired(classify_wasmtime(
                    InvocationFailureKind::Trap,
                    error,
                ));
            }
        };

        let output_bytes = match serialized_len(&(&plan, &state_update)) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error,
                ));
            }
        };
        if output_bytes > self.inner.manifest.limits.output_bytes {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidOutput,
                "policy output exceeds the package byte limit",
            ));
        }
        let normalized_plan = match validate_output_v0_6(
            &context,
            &plan,
            &state.state,
            &state_update,
            protocol_limits,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
        };
        let actions = match store.data_mut().finish() {
            Ok(actions) => actions,
            Err(failure) => return Invocation::FallbackRequired(failure),
        };
        Invocation::Success(PreparedPolicyResultV0_6 {
            plan,
            normalized_plan,
            state_update,
            actions,
            duplicate_feedback: false,
        })
    }

    fn instantiate(
        &self,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: Arc<BTreeSet<String>>,
    ) -> Result<(Store<InvocationContext>, PlexPolicy), InvocationFailure> {
        let memory_bytes =
            usize::try_from(self.inner.manifest.limits.memory_bytes).unwrap_or(usize::MAX);
        let mut store = InvocationContext::store(
            self.inner.engine.raw(),
            InvocationContextConfig {
                memory_bytes,
                query_handler,
                supported_actions,
                max_host_calls: self
                    .inner
                    .manifest
                    .limits
                    .host_calls
                    .min(self.inner.engine.config().max_host_calls),
                max_host_call_bytes: self
                    .inner
                    .manifest
                    .limits
                    .host_call_bytes
                    .min(self.inner.engine.config().max_host_call_bytes),
            },
        );
        store
            .set_fuel(self.inner.manifest.limits.fuel)
            .map_err(|error| {
                InvocationFailure::new(InvocationFailureKind::Instantiation, error.to_string())
            })?;
        store.set_epoch_deadline(
            self.inner
                .engine
                .deadline_ticks(self.inner.manifest.limits.deadline_ms),
        );
        store.epoch_deadline_trap();
        let policy = self
            .inner
            .pre
            .instantiate(&mut store)
            .map_err(|error| classify_wasmtime(InvocationFailureKind::Instantiation, error))?;
        Ok((store, policy))
    }
}

enum CallErrorV0_6 {
    Wire(crate::wire_v0_6::WireErrorV0_6),
    Trap(wasmtime::Error),
}

fn call_operation(
    context: &OperationContextV0_6,
    state: &pie_plex::v0_6::PolicyState,
    policy: &PolicyGuest,
    store: &mut Store<InvocationContext>,
) -> Result<Result<(OperationPlanV0_6, StateUpdate), PolicyError>, CallErrorV0_6> {
    match context {
        OperationContextV0_6::Admit(context) => {
            let input = wire_v0_6::admit_invocation(context, state).map_err(CallErrorV0_6::Wire)?;
            match policy
                .call_admit(store, &input)
                .map_err(CallErrorV0_6::Trap)?
            {
                Ok(output) => {
                    let (plan, update) =
                        wire_v0_6::admit_output(output).map_err(CallErrorV0_6::Wire)?;
                    Ok(Ok((OperationPlanV0_6::Admit(plan), update)))
                }
                Err(error) => Ok(Err(
                    wire_v0_6::policy_error_from_wire(error).map_err(CallErrorV0_6::Wire)?
                )),
            }
        }
        OperationContextV0_6::Route(context) => {
            let input = wire_v0_6::route_invocation(context, state).map_err(CallErrorV0_6::Wire)?;
            match policy
                .call_route(store, &input)
                .map_err(CallErrorV0_6::Trap)?
            {
                Ok(output) => {
                    let (plan, update) =
                        wire_v0_6::route_output(output).map_err(CallErrorV0_6::Wire)?;
                    Ok(Ok((OperationPlanV0_6::Route(plan), update)))
                }
                Err(error) => Ok(Err(
                    wire_v0_6::policy_error_from_wire(error).map_err(CallErrorV0_6::Wire)?
                )),
            }
        }
        OperationContextV0_6::Schedule(context) => {
            let input =
                wire_v0_6::schedule_invocation(context, state).map_err(CallErrorV0_6::Wire)?;
            match policy
                .call_schedule(store, &input)
                .map_err(CallErrorV0_6::Trap)?
            {
                Ok(output) => {
                    let (plan, update) =
                        wire_v0_6::schedule_output(output).map_err(CallErrorV0_6::Wire)?;
                    Ok(Ok((OperationPlanV0_6::Schedule(plan), update)))
                }
                Err(error) => Ok(Err(
                    wire_v0_6::policy_error_from_wire(error).map_err(CallErrorV0_6::Wire)?
                )),
            }
        }
        OperationContextV0_6::Cache(context) => {
            let input = wire_v0_6::cache_invocation(context, state).map_err(CallErrorV0_6::Wire)?;
            match policy
                .call_cache(store, &input)
                .map_err(CallErrorV0_6::Trap)?
            {
                Ok(output) => {
                    let (plan, update) =
                        wire_v0_6::cache_output(output).map_err(CallErrorV0_6::Wire)?;
                    Ok(Ok((OperationPlanV0_6::Cache(plan), update)))
                }
                Err(error) => Ok(Err(
                    wire_v0_6::policy_error_from_wire(error).map_err(CallErrorV0_6::Wire)?
                )),
            }
        }
        OperationContextV0_6::Feedback(context) => {
            let input =
                wire_v0_6::feedback_invocation(context, state).map_err(CallErrorV0_6::Wire)?;
            match policy
                .call_feedback(store, &input)
                .map_err(CallErrorV0_6::Trap)?
            {
                Ok(output) => {
                    let update = wire_v0_6::feedback_output(output).map_err(CallErrorV0_6::Wire)?;
                    Ok(Ok((OperationPlanV0_6::Feedback, update)))
                }
                Err(error) => Ok(Err(
                    wire_v0_6::policy_error_from_wire(error).map_err(CallErrorV0_6::Wire)?
                )),
            }
        }
    }
}

fn probe_instantiation(
    engine: &PolicyEngine,
    manifest: &Manifest,
    pre: &PlexPolicyPre<InvocationContext>,
) -> Result<(), AttachmentErrorV0_6> {
    let _permit = engine
        .try_acquire()
        .ok_or(AttachmentErrorV0_6::EngineSaturated)?;
    let memory_bytes = usize::try_from(manifest.limits.memory_bytes).unwrap_or(usize::MAX);
    let mut store = InvocationContext::store(
        engine.raw(),
        InvocationContextConfig {
            memory_bytes,
            query_handler: Arc::new(RejectingQueryHandler),
            supported_actions: Arc::new(BTreeSet::new()),
            max_host_calls: manifest
                .limits
                .host_calls
                .min(engine.config().max_host_calls),
            max_host_call_bytes: manifest
                .limits
                .host_call_bytes
                .min(engine.config().max_host_call_bytes),
        },
    );
    store
        .set_fuel(manifest.limits.fuel)
        .map_err(|error| AttachmentErrorV0_6::Instantiate(error.to_string()))?;
    store.set_epoch_deadline(engine.deadline_ticks(manifest.limits.deadline_ms));
    store.epoch_deadline_trap();
    pre.instantiate(&mut store)
        .map_err(|error| AttachmentErrorV0_6::Instantiate(error.to_string()))?;
    Ok(())
}

fn verify_component_surface(
    engine: &wasmtime::Engine,
    component: &Component,
) -> Result<(), AttachmentErrorV0_6> {
    const HOST: &str = "pie:plex/host@0.6.0";
    const POLICY: &str = "pie:plex/policy@0.6.0";
    let component_type = component.component_type();
    let imports = component_type
        .imports(engine)
        .map(|(name, _)| name.to_owned())
        .collect::<BTreeSet<_>>();
    let expected_imports = BTreeSet::from([HOST.to_owned()]);
    if imports != expected_imports {
        if let Some(unsupported) = imports.difference(&expected_imports).next() {
            return Err(AttachmentErrorV0_6::UnsupportedImport(unsupported.clone()));
        }
        return Err(AttachmentErrorV0_6::MissingRequiredImport(HOST.into()));
    }
    let exports = component_type
        .exports(engine)
        .map(|(name, _)| name.to_owned())
        .collect::<BTreeSet<_>>();
    if exports != BTreeSet::from([POLICY.to_owned()]) {
        if let Some(unsupported) = exports.iter().find(|name| name.as_str() != POLICY) {
            return Err(AttachmentErrorV0_6::UnsupportedExport(unsupported.clone()));
        }
        return Err(AttachmentErrorV0_6::MissingPolicyExport);
    }
    Ok(())
}

fn validate_host_limits(
    host: &PolicyEngineConfig,
    manifest: &Manifest,
) -> Result<(), AttachmentErrorV0_6> {
    for (field, requested, maximum) in [
        (
            "memory_bytes",
            manifest.limits.memory_bytes,
            host.max_memory_bytes as u64,
        ),
        ("fuel", manifest.limits.fuel, host.max_fuel),
        (
            "deadline_ms",
            manifest.limits.deadline_ms,
            host.max_deadline_ms,
        ),
        (
            "input_bytes",
            manifest.limits.input_bytes,
            host.max_input_bytes,
        ),
        (
            "output_bytes",
            manifest.limits.output_bytes,
            host.max_output_bytes,
        ),
        (
            "host_calls",
            u64::from(manifest.limits.host_calls),
            u64::from(host.max_host_calls),
        ),
        (
            "host_call_bytes",
            manifest.limits.host_call_bytes,
            host.max_host_call_bytes,
        ),
    ] {
        if requested > maximum {
            return Err(AttachmentErrorV0_6::HostLimit {
                field,
                requested,
                maximum,
            });
        }
    }
    Ok(())
}

fn serialized_len(value: &impl Serialize) -> Result<u64, String> {
    serde_json::to_vec(value)
        .map(|value| value.len() as u64)
        .map_err(|error| format!("failed to serialize typed policy data: {error}"))
}

fn classify_wasmtime(default: InvocationFailureKind, error: wasmtime::Error) -> InvocationFailure {
    let kind = match error.downcast_ref::<wasmtime::Trap>() {
        Some(wasmtime::Trap::OutOfFuel) => InvocationFailureKind::FuelExhausted,
        Some(wasmtime::Trap::Interrupt) => InvocationFailureKind::DeadlineExceeded,
        _ => default,
    };
    InvocationFailure::new(kind, error.to_string())
}

#[derive(Debug, thiserror::Error)]
pub enum AttachmentErrorV0_6 {
    #[error(transparent)]
    Manifest(#[from] pie_plex::v0_6::ManifestValidationError),
    #[error("manifest limit {field} requests {requested}; host maximum is {maximum}")]
    HostLimit {
        field: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("failed to compile policy component: {0}")]
    Compile(String),
    #[error("failed to link policy component: {0}")]
    Link(String),
    #[error("failed to instantiate policy component within declared limits: {0}")]
    Instantiate(String),
    #[error("policy engine has no free invocation slot")]
    EngineSaturated,
    #[error("policy package is invalid")]
    Package(#[source] PackageErrorV0_6),
    #[error("PLEX policy component imports unsupported interface {0}")]
    UnsupportedImport(String),
    #[error("policy component does not import required interface {0}")]
    MissingRequiredImport(String),
    #[error("policy component exports unsupported interface {0}")]
    UnsupportedExport(String),
    #[error("policy component does not export pie:plex/policy@0.6.0")]
    MissingPolicyExport,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_v0_5_component_surface() {
        let engine = PolicyEngine::new(PolicyEngineConfig::deterministic_replay()).unwrap();
        let component = Component::new(
            engine.raw(),
            r#"(component
                (type $f (func))
                (import "pie:plex/host@0.5.0" (func (type $f)))
            )"#,
        )
        .unwrap();
        assert!(matches!(
            verify_component_surface(engine.raw(), &component),
            Err(AttachmentErrorV0_6::UnsupportedImport(_))
        ));
    }
}
