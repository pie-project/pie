use std::collections::BTreeSet;
use std::sync::Arc;

use pie_plex::Document;
use pie_plex::v0_5::{Manifest, Operation};
use serde::{Deserialize, Serialize};
use wasmtime::Store;
use wasmtime::component::{Component, HasSelf, Linker};

use crate::bindings::exports::pie::plex::policy::{
    Guest as PolicyGuest, Invocation as WitInvocation,
};
use crate::bindings::{PlexPolicy, PlexPolicyPre};
use crate::context::{InvocationContext, InvocationContextConfig};
use crate::engine::{PolicyEngine, PolicyEngineConfig};
use crate::error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
use crate::host::{QueryHandler, RejectingQueryHandler, StagedAction};
use crate::package_format::{PackageLimits, PolicyPackage};
use crate::protocol::{parse_result, parse_state_updates, validate_context};
use crate::state_store::{StateSnapshot, StateUpdates};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedPolicyResult {
    pub result: Document,
    pub state_updates: StateUpdates,
    pub actions: Vec<StagedAction>,
    pub duplicate_feedback: bool,
}

impl PreparedPolicyResult {
    pub(crate) fn duplicate_feedback(result: Document) -> Self {
        Self {
            result,
            state_updates: StateUpdates::default(),
            actions: Vec::new(),
            duplicate_feedback: true,
        }
    }
}

#[derive(Clone)]
pub struct AttachedPolicy {
    inner: Arc<AttachedPolicyInner>,
}

struct AttachedPolicyInner {
    engine: PolicyEngine,
    manifest: Manifest,
    pre: PlexPolicyPre<InvocationContext>,
}

impl AttachedPolicy {
    pub fn compile_package(
        engine: PolicyEngine,
        package_bytes: &[u8],
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
        Self::compile(engine, &component, manifest)
    }

    pub fn compile(
        engine: PolicyEngine,
        component_bytes: &[u8],
        manifest: Manifest,
    ) -> Result<Self, AttachmentError> {
        manifest.validate()?;
        validate_host_limits(engine.config(), &manifest)?;
        let component = Component::new(engine.raw(), component_bytes)
            .map_err(|error| AttachmentError::Compile(error.to_string()))?;
        verify_component_surface(engine.raw(), &component)?;
        let mut linker = Linker::<InvocationContext>::new(engine.raw());
        PlexPolicy::add_to_linker::<InvocationContext, HasSelf<InvocationContext>>(
            &mut linker,
            |context| context,
        )
        .map_err(|error| AttachmentError::Link(error.to_string()))?;
        let instance_pre = linker
            .instantiate_pre(&component)
            .map_err(|error| AttachmentError::Link(error.to_string()))?;
        let pre = PlexPolicyPre::new(instance_pre)
            .map_err(|error| AttachmentError::Link(error.to_string()))?;
        probe_instantiation(&engine, &manifest, &pre)?;
        Ok(Self {
            inner: Arc::new(AttachedPolicyInner {
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
        operation: Operation,
        context: Document,
        state: StateSnapshot,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: Arc<BTreeSet<String>>,
    ) -> Invocation<PreparedPolicyResult> {
        if !self.inner.manifest.operations.contains(&operation) {
            return Invocation::Unavailable;
        }
        let referenced = match validate_context(operation, &context) {
            Ok(referenced) => referenced,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidInput,
                    error.to_string(),
                ));
            }
        };
        let exposed = state.requests.keys().cloned().collect::<BTreeSet<_>>();
        if referenced != exposed {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                format!(
                    "referenced request IDs {referenced:?} do not match prepared state {exposed:?}"
                ),
            ));
        }
        self.invoke_owned(operation, context, state, query_handler, supported_actions)
    }

    fn invoke_owned(
        &self,
        operation: Operation,
        context: Document,
        state: StateSnapshot,
        query_handler: Arc<dyn QueryHandler>,
        supported_actions: Arc<BTreeSet<String>>,
    ) -> Invocation<PreparedPolicyResult> {
        let context_json = match serde_json::to_string(&context) {
            Ok(context_json) => context_json,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidInput,
                    format!("failed to serialize policy context: {error}"),
                ));
            }
        };
        let state_json = match serde_json::to_string(&state.document()) {
            Ok(state_json) => state_json,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidInput,
                    format!("failed to serialize policy state: {error}"),
                ));
            }
        };
        if context_json.len().saturating_add(state_json.len()) as u64
            > self.inner.manifest.limits.input_bytes
        {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                "policy invocation exceeds the package byte limit",
            ));
        }
        // Declared before the Store so pooled resources are returned before
        // the invocation slot becomes available to another thread.
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
        let input = WitInvocation {
            context_json,
            state_json,
        };
        let output = match call_operation(operation, policy.pie_plex_policy(), &mut store, &input) {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                if let Some(failure) = store.data_mut().take_reported_failure(&error) {
                    return Invocation::FallbackRequired(failure);
                }
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::PolicyFallback,
                    error,
                ));
            }
            Err(error) => {
                return Invocation::FallbackRequired(classify_wasmtime(
                    InvocationFailureKind::Trap,
                    error,
                ));
            }
        };
        if output
            .result_json
            .len()
            .saturating_add(output.state_update_json.len()) as u64
            > self.inner.manifest.limits.output_bytes
        {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidOutput,
                "policy output exceeds the package byte limit",
            ));
        }
        let result = match parse_result(&output.result_json) {
            Ok(result) => result,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
        };
        let state_updates = match parse_state_updates(&output.state_update_json, &state) {
            Ok(updates) => updates,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
        };
        if let Err(error) = validate_result(operation, &context, &result) {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidOutput,
                error,
            ));
        }
        let actions = match store.data_mut().finish() {
            Ok(actions) => actions,
            Err(failure) => return Invocation::FallbackRequired(failure),
        };
        Invocation::Success(PreparedPolicyResult {
            result,
            state_updates,
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
                authorization: None,
                max_host_calls: self.inner.engine.config().max_host_calls,
                max_host_call_bytes: self.inner.engine.config().max_host_call_bytes,
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

fn call_operation(
    operation: Operation,
    policy: &PolicyGuest,
    store: &mut Store<InvocationContext>,
    input: &WitInvocation,
) -> wasmtime::Result<Result<crate::bindings::exports::pie::plex::policy::PolicyOutput, String>> {
    match operation {
        Operation::Route => policy.call_route(store, input),
        Operation::Admit => policy.call_admit(store, input),
        Operation::Schedule => policy.call_schedule(store, input),
        Operation::Evict => policy.call_evict(store, input),
        Operation::Feedback => policy.call_feedback(store, input),
    }
}

pub(crate) fn validate_result(
    operation: Operation,
    context: &Document,
    result: &Document,
) -> Result<(), String> {
    let validation = match operation {
        Operation::Route => pie_plex::v0_5::rank_route(
            result,
            context["candidates"]
                .as_array()
                .expect("validated context")
                .len(),
        )
        .map(|_| ()),
        Operation::Admit => pie_plex::v0_5::validate_admit(result).map(|_| ()),
        Operation::Schedule => pie_plex::v0_5::select_schedule(context, result).map(|_| ()),
        Operation::Evict => pie_plex::v0_5::select_evictions(context, result).map(|_| ()),
        Operation::Feedback => Ok(()),
    };
    validation.map_err(|error| error.to_string())
}

fn probe_instantiation(
    engine: &PolicyEngine,
    manifest: &Manifest,
    pre: &PlexPolicyPre<InvocationContext>,
) -> Result<(), AttachmentError> {
    let _permit = engine
        .try_acquire()
        .ok_or(AttachmentError::EngineSaturated)?;
    let memory_bytes = usize::try_from(manifest.limits.memory_bytes).unwrap_or(usize::MAX);
    let mut store = InvocationContext::store(
        engine.raw(),
        InvocationContextConfig {
            memory_bytes,
            query_handler: Arc::new(RejectingQueryHandler),
            supported_actions: Arc::new(BTreeSet::new()),
            authorization: None,
            max_host_calls: engine.config().max_host_calls,
            max_host_call_bytes: engine.config().max_host_call_bytes,
        },
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
    const HOST: &str = "pie:plex/host@0.5.0";
    const POLICY: &str = "pie:plex/policy@0.5.0";
    let component_type = component.component_type();
    let imports = component_type
        .imports(engine)
        .map(|(name, _)| name.to_owned())
        .collect::<BTreeSet<_>>();
    let expected_imports = BTreeSet::from([HOST.to_owned()]);
    if imports != expected_imports {
        if let Some(unsupported) = imports.difference(&expected_imports).next() {
            return Err(AttachmentError::UnsupportedImport(unsupported.clone()));
        }
        return Err(AttachmentError::MissingRequiredImport(HOST.into()));
    }
    let exports = component_type
        .exports(engine)
        .map(|(name, _)| name.to_owned())
        .collect::<BTreeSet<_>>();
    if exports != BTreeSet::from([POLICY.to_owned()]) {
        if let Some(unsupported) = exports.iter().find(|name| name.as_str() != POLICY) {
            return Err(AttachmentError::UnsupportedExport(unsupported.clone()));
        }
        return Err(AttachmentError::MissingPolicyExport);
    }
    Ok(())
}

fn validate_host_limits(
    host: &PolicyEngineConfig,
    manifest: &Manifest,
) -> Result<(), AttachmentError> {
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
    ] {
        if requested > maximum {
            return Err(AttachmentError::HostLimit {
                field,
                requested,
                maximum,
            });
        }
    }
    Ok(())
}

fn classify_wasmtime(default: InvocationFailureKind, error: wasmtime::Error) -> InvocationFailure {
    let kind = match error.downcast_ref::<wasmtime::Trap>() {
        Some(wasmtime::Trap::OutOfFuel) => InvocationFailureKind::FuelExhausted,
        Some(wasmtime::Trap::Interrupt) => InvocationFailureKind::DeadlineExceeded,
        _ => default,
    };
    InvocationFailure::new(kind, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unapproved_component_import() {
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
            Err(AttachmentError::UnsupportedImport(_))
        ));
    }
}
