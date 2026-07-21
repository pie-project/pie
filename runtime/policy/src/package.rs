use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use pie_plex::{Document, Manifest, Operation};
use wasmtime::Store;
use wasmtime::component::{Component, Linker};

use crate::bindings::exports::pie::plex::policy::Guest as PolicyGuest;
use crate::bindings::{PlexPolicy, PlexPolicyPre};
use crate::context::InvocationContext;
use crate::engine::{PolicyEngine, PolicyEngineConfig};
use crate::error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
use crate::package_format::{PackageLimits, PolicyPackage};
use crate::protocol::{JsonResponse, parse_response, validate_input};

#[derive(Clone)]
pub struct AttachedPolicy {
    inner: Arc<AttachedPolicyInner>,
}

struct AttachedPolicyInner {
    engine: PolicyEngine,
    manifest: Manifest,
    pre: PlexPolicyPre<InvocationContext>,
    dedup: Mutex<DedupLedger>,
}

#[derive(Default)]
struct DedupLedger {
    committed: BTreeMap<String, Document>,
    in_flight: BTreeSet<String>,
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
        let linker = Linker::<InvocationContext>::new(engine.raw());
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
                dedup: Mutex::new(DedupLedger::default()),
            }),
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.inner.manifest
    }

    pub fn route(&self, input: Document) -> Invocation<JsonResponse> {
        self.invoke(Operation::Route, input, |policy, store, input| {
            policy.call_route(store, input)
        })
    }

    pub fn admit(&self, input: Document) -> Invocation<JsonResponse> {
        self.invoke(Operation::Admit, input, |policy, store, input| {
            policy.call_admit(store, input)
        })
    }

    pub fn schedule(&self, input: Document) -> Invocation<JsonResponse> {
        self.invoke(Operation::Schedule, input, |policy, store, input| {
            policy.call_schedule(store, input)
        })
    }

    pub fn evict(&self, input: Document) -> Invocation<JsonResponse> {
        self.invoke(Operation::Evict, input, |policy, store, input| {
            policy.call_evict(store, input)
        })
    }

    pub fn feedback(&self, input: Document) -> Invocation<JsonResponse> {
        if !self
            .inner
            .manifest
            .operations
            .contains(&Operation::Feedback)
        {
            return Invocation::Unavailable;
        }
        if let Err(error) = validate_input(Operation::Feedback, &input) {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                error.to_string(),
            ));
        }
        let delivery_id = input["delivery_id"]
            .as_str()
            .expect("feedback input was validated")
            .to_owned();
        {
            let mut dedup = self.inner.dedup.lock().unwrap();
            if let Some(result) = dedup.committed.get(&delivery_id) {
                return Invocation::Success(JsonResponse::unchanged_feedback(
                    input,
                    result.clone(),
                ));
            }
            if dedup.in_flight.contains(&delivery_id) {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::HostSaturated,
                    "feedback delivery is already in flight",
                ));
            }
            if dedup.committed.len() + dedup.in_flight.len()
                >= self.inner.engine.config().max_feedback_deliveries
            {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::HostSaturated,
                    "feedback deduplication ledger is full",
                ));
            }
            dedup.in_flight.insert(delivery_id.clone());
        }

        let invocation = self.invoke_owned(Operation::Feedback, input, |policy, store, input| {
            policy.call_feedback(store, input)
        });
        let mut dedup = self.inner.dedup.lock().unwrap();
        dedup.in_flight.remove(&delivery_id);
        if let Invocation::Success(response) = &invocation {
            dedup.committed.insert(delivery_id, response.result.clone());
        }
        invocation
    }

    pub(crate) fn transfer_dedup_from(&self, source: &AttachedPolicy) {
        let source = source.inner.dedup.lock().unwrap();
        let mut target = self.inner.dedup.lock().unwrap();
        target.committed = source.committed.clone();
    }

    fn invoke<Call>(
        &self,
        operation: Operation,
        input: Document,
        call: Call,
    ) -> Invocation<JsonResponse>
    where
        Call: Fn(
            &PolicyGuest,
            &mut Store<InvocationContext>,
            &str,
        ) -> wasmtime::Result<Result<String, String>>,
    {
        if !self.inner.manifest.operations.contains(&operation) {
            return Invocation::Unavailable;
        }
        if let Err(error) = validate_input(operation, &input) {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                error.to_string(),
            ));
        }
        self.invoke_owned(operation, input, call)
    }

    fn invoke_owned<Call>(
        &self,
        operation: Operation,
        input: Document,
        call: Call,
    ) -> Invocation<JsonResponse>
    where
        Call: Fn(
            &PolicyGuest,
            &mut Store<InvocationContext>,
            &str,
        ) -> wasmtime::Result<Result<String, String>>,
    {
        let input_json = match serde_json::to_string(&input) {
            Ok(input_json) => input_json,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidInput,
                    format!("failed to serialize policy input: {error}"),
                ));
            }
        };
        if input_json.len() as u64 > self.inner.manifest.limits.input_bytes {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                "policy input exceeds the package byte limit",
            ));
        }
        let (mut store, policy) = match self.instantiate() {
            Ok(value) => value,
            Err(failure) => return Invocation::FallbackRequired(failure),
        };
        let output = match call(policy.pie_plex_policy(), &mut store, &input_json) {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
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
        if output.len() as u64 > self.inner.manifest.limits.output_bytes {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidOutput,
                "policy output exceeds the package byte limit",
            ));
        }
        let response = match parse_response(&output, &input, operation) {
            Ok(response) => response,
            Err(error) => {
                return Invocation::FallbackRequired(InvocationFailure::new(
                    InvocationFailureKind::InvalidOutput,
                    error.to_string(),
                ));
            }
        };
        if let Err(error) = validate_result(operation, &response.input, &response.result) {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidOutput,
                error,
            ));
        }
        Invocation::Success(response)
    }

    fn instantiate(&self) -> Result<(Store<InvocationContext>, PlexPolicy), InvocationFailure> {
        let permit = self.inner.engine.try_acquire().ok_or_else(|| {
            InvocationFailure::new(
                InvocationFailureKind::HostSaturated,
                "policy engine has no free invocation slot",
            )
        })?;
        let memory_bytes =
            usize::try_from(self.inner.manifest.limits.memory_bytes).unwrap_or(usize::MAX);
        let mut store = InvocationContext::store(self.inner.engine.raw(), memory_bytes, permit);
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

fn validate_result(
    operation: Operation,
    input: &Document,
    result: &Document,
) -> Result<(), String> {
    let validation = match operation {
        Operation::Route => pie_plex::rank_route(
            result,
            input["candidates"]
                .as_array()
                .expect("validated input")
                .len(),
        )
        .map(|_| ()),
        Operation::Admit => pie_plex::validate_admit(result).map(|_| ()),
        Operation::Schedule => pie_plex::select_schedule(input, result).map(|_| ()),
        Operation::Evict => pie_plex::select_evictions(input, result).map(|_| ()),
        Operation::Feedback => Ok(()),
    };
    validation.map_err(|error| error.to_string())
}

fn probe_instantiation(
    engine: &PolicyEngine,
    manifest: &Manifest,
    pre: &PlexPolicyPre<InvocationContext>,
) -> Result<(), AttachmentError> {
    let permit = engine
        .try_acquire()
        .ok_or(AttachmentError::EngineSaturated)?;
    let memory_bytes = usize::try_from(manifest.limits.memory_bytes).unwrap_or(usize::MAX);
    let mut store = InvocationContext::store(engine.raw(), memory_bytes, permit);
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
    const POLICY: &str = "pie:plex/policy@0.2.0";
    let component_type = component.component_type();
    if let Some((name, _)) = component_type.imports(engine).next() {
        return Err(AttachmentError::UnsupportedImport(name.to_owned()));
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
    fn rejects_any_component_import() {
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
