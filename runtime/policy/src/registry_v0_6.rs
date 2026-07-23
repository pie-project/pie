use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Condvar, Mutex};

use pie_plex::v0_6::{
    Manifest, MechanicId, MechanicKind, Operation, SchemaKind, standard_mechanic,
};
use thiserror::Error;

use crate::OperationContextV0_6;
use crate::host::QueryHandler;
use crate::{
    AttachedPolicyV0_6, AttachmentErrorV0_6, Invocation, InvocationFailure, InvocationFailureKind,
    PolicyEngine, PreparedPolicyResultV0_6, ProtocolLimitsV0_6, StateSnapshotV0_6,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SchemaKeyV0_6 {
    pub kind: SchemaKind,
    pub id: String,
}

#[derive(Debug, Clone, Default)]
pub struct HostSupportV0_6 {
    pub mechanics: BTreeSet<MechanicId>,
    pub schemas: BTreeSet<SchemaKeyV0_6>,
    pub action_methods: BTreeSet<String>,
}

impl HostSupportV0_6 {
    pub fn with_standard_ids(
        mechanics: impl IntoIterator<Item = String>,
    ) -> Result<Self, RegistryErrorV0_6> {
        Self::with_standard_mechanics(mechanics.into_iter().map(MechanicId))
    }

    pub fn with_standard_mechanics(
        mechanics: impl IntoIterator<Item = MechanicId>,
    ) -> Result<Self, RegistryErrorV0_6> {
        let mechanics = mechanics.into_iter().collect::<BTreeSet<_>>();
        let action_methods = standard_mechanic_action_methods(&mechanics)?;
        Ok(Self {
            mechanics,
            schemas: BTreeSet::new(),
            action_methods,
        })
    }
}

#[derive(Clone)]
pub struct AttachmentRegistryV0_6 {
    engine: PolicyEngine,
    support: Arc<HostSupportV0_6>,
    inner: Arc<RegistryInnerV0_6>,
}

struct RegistryInnerV0_6 {
    state: Mutex<RegistryStateV0_6>,
    changed: Condvar,
}

struct RegistryStateV0_6 {
    active: Arc<AttachmentSetV0_6>,
    updating: bool,
    snapshots: u64,
}

#[derive(Clone, Default)]
struct AttachmentSetV0_6 {
    generation: u64,
    owners: BTreeMap<Operation, AttachedRecordV0_6>,
    packages: BTreeMap<String, AttachedRecordV0_6>,
}

#[derive(Clone)]
struct AttachedRecordV0_6 {
    policy: AttachedPolicyV0_6,
    negotiated_mechanics: BTreeSet<MechanicId>,
    action_methods: BTreeSet<String>,
}

impl AttachmentRegistryV0_6 {
    pub fn new(engine: PolicyEngine, support: HostSupportV0_6) -> Self {
        Self {
            engine,
            support: Arc::new(support),
            inner: Arc::new(RegistryInnerV0_6 {
                state: Mutex::new(RegistryStateV0_6 {
                    active: Arc::new(AttachmentSetV0_6::default()),
                    updating: false,
                    snapshots: 0,
                }),
                changed: Condvar::new(),
            }),
        }
    }

    pub fn prepare(&self, package: &[u8]) -> Result<AttachedPolicyV0_6, RegistryErrorV0_6> {
        let policy = AttachedPolicyV0_6::compile_package(self.engine.clone(), package)
            .map_err(RegistryErrorV0_6::Prepare)?;
        validate_requirements(policy.manifest(), &self.support)?;
        Ok(policy)
    }

    pub fn attach(&self, package: &[u8]) -> Result<u64, RegistryErrorV0_6> {
        let prepared = self.prepare(package)?;
        self.attach_prepared(prepared)
    }

    pub fn attach_prepared(&self, policy: AttachedPolicyV0_6) -> Result<u64, RegistryErrorV0_6> {
        validate_requirements(policy.manifest(), &self.support)?;
        let negotiated_mechanics = negotiated_mechanics(policy.manifest(), &self.support);
        let record = AttachedRecordV0_6 {
            action_methods: standard_mechanic_action_methods(&negotiated_mechanics)?,
            negotiated_mechanics,
            policy,
        };
        let mut state = self.lock_stable();
        let package_name = record.policy.manifest().package_name.clone();
        if state.active.packages.contains_key(&package_name) {
            return Err(RegistryErrorV0_6::PackageAlreadyAttached(package_name));
        }
        ensure_operations_available(&state.active, &record, None)?;
        let generation = next_generation(state.active.generation)?;
        let mut next = (*state.active).clone();
        for operation in &record.policy.manifest().implements {
            next.owners.insert(*operation, record.clone());
        }
        next.packages.insert(package_name, record);
        next.generation = generation;
        state.active = Arc::new(next);
        self.inner.changed.notify_all();
        Ok(generation)
    }

    pub fn replace(&self, package: &[u8]) -> Result<u64, RegistryErrorV0_6> {
        let policy = self.prepare(package)?;
        let negotiated_mechanics = negotiated_mechanics(policy.manifest(), &self.support);
        let replacement = AttachedRecordV0_6 {
            action_methods: standard_mechanic_action_methods(&negotiated_mechanics)?,
            negotiated_mechanics,
            policy,
        };
        let package_name = replacement.policy.manifest().package_name.clone();
        let mut state = self.inner.state.lock().unwrap();
        while state.updating {
            state = self.inner.changed.wait(state).unwrap();
        }
        state.updating = true;
        while state.snapshots != 0 {
            state = self.inner.changed.wait(state).unwrap();
        }
        let result = (|| {
            state
                .active
                .packages
                .get(&package_name)
                .ok_or_else(|| RegistryErrorV0_6::PackageNotAttached(package_name.clone()))?;
            ensure_operations_available(&state.active, &replacement, Some(&package_name))?;
            let generation = next_generation(state.active.generation)?;
            let mut next = (*state.active).clone();
            next.owners
                .retain(|_, owner| owner.policy.manifest().package_name != package_name);
            for operation in &replacement.policy.manifest().implements {
                next.owners.insert(*operation, replacement.clone());
            }
            next.packages.insert(package_name, replacement);
            next.generation = generation;
            state.active = Arc::new(next);
            Ok(generation)
        })();
        state.updating = false;
        self.inner.changed.notify_all();
        result
    }

    pub fn detach_operation(&self, operation: Operation) -> Result<u64, RegistryErrorV0_6> {
        let mut state = self.lock_stable();
        let owner = state
            .active
            .owners
            .get(&operation)
            .cloned()
            .ok_or(RegistryErrorV0_6::OperationNotAttached(operation))?;
        let generation = next_generation(state.active.generation)?;
        let mut next = (*state.active).clone();
        next.owners.remove(&operation);
        if !next.owners.values().any(|candidate| {
            candidate.policy.manifest().package_name == owner.policy.manifest().package_name
        }) {
            next.packages.remove(&owner.policy.manifest().package_name);
        }
        next.generation = generation;
        state.active = Arc::new(next);
        Ok(generation)
    }

    pub fn detach_package(&self, package_name: &str) -> Result<u64, RegistryErrorV0_6> {
        let mut state = self.lock_stable();
        if !state.active.packages.contains_key(package_name) {
            return Err(RegistryErrorV0_6::PackageNotAttached(
                package_name.to_owned(),
            ));
        }
        let generation = next_generation(state.active.generation)?;
        let mut next = (*state.active).clone();
        next.owners
            .retain(|_, owner| owner.policy.manifest().package_name != package_name);
        next.packages.remove(package_name);
        next.generation = generation;
        state.active = Arc::new(next);
        Ok(generation)
    }

    pub fn snapshot(&self) -> Result<AttachmentSnapshotV0_6, RegistryErrorV0_6> {
        let mut state = self.lock_stable();
        state.snapshots = state
            .snapshots
            .checked_add(1)
            .ok_or(RegistryErrorV0_6::SnapshotCounterExhausted)?;
        Ok(AttachmentSnapshotV0_6 {
            set: state.active.clone(),
            _lease: Arc::new(SnapshotLeaseV0_6 {
                registry: self.inner.clone(),
            }),
        })
    }

    pub(crate) fn max_feedback_deliveries(&self) -> usize {
        self.engine.config().max_feedback_deliveries
    }

    fn lock_stable(&self) -> std::sync::MutexGuard<'_, RegistryStateV0_6> {
        let mut state = self.inner.state.lock().unwrap();
        while state.updating {
            state = self.inner.changed.wait(state).unwrap();
        }
        state
    }
}

fn validate_requirements(
    manifest: &Manifest,
    support: &HostSupportV0_6,
) -> Result<(), RegistryErrorV0_6> {
    if let Some(mechanic) = manifest
        .requires
        .iter()
        .find(|mechanic| !support.mechanics.contains(*mechanic))
    {
        return Err(RegistryErrorV0_6::MissingRequiredMechanic(mechanic.clone()));
    }
    if let Some(schema) = manifest.schemas.iter().find(|schema| {
        schema.required
            && !support.schemas.contains(&SchemaKeyV0_6 {
                kind: schema.kind,
                id: schema.id.clone(),
            })
    }) {
        return Err(RegistryErrorV0_6::MissingRequiredSchema {
            kind: schema.kind,
            id: schema.id.clone(),
        });
    }
    Ok(())
}

fn negotiated_mechanics(manifest: &Manifest, support: &HostSupportV0_6) -> BTreeSet<MechanicId> {
    manifest
        .requires
        .iter()
        .chain(
            manifest
                .optional
                .iter()
                .filter(|mechanic| support.mechanics.contains(*mechanic)),
        )
        .cloned()
        .collect()
}

fn ensure_operations_available(
    active: &AttachmentSetV0_6,
    candidate: &AttachedRecordV0_6,
    replacing: Option<&str>,
) -> Result<(), RegistryErrorV0_6> {
    for operation in &candidate.policy.manifest().implements {
        if let Some(owner) = active.owners.get(operation)
            && replacing.is_none_or(|package| owner.policy.manifest().package_name != package)
        {
            return Err(RegistryErrorV0_6::OperationAlreadyOwned {
                operation: *operation,
                package: owner.policy.manifest().package_name.clone(),
            });
        }
    }
    Ok(())
}

fn next_generation(current: u64) -> Result<u64, RegistryErrorV0_6> {
    current
        .checked_add(1)
        .ok_or(RegistryErrorV0_6::GenerationExhausted)
}

struct SnapshotLeaseV0_6 {
    registry: Arc<RegistryInnerV0_6>,
}

impl Drop for SnapshotLeaseV0_6 {
    fn drop(&mut self) {
        let mut state = self.registry.state.lock().unwrap();
        state.snapshots = state.snapshots.saturating_sub(1);
        self.registry.changed.notify_all();
    }
}

#[derive(Clone)]
pub struct AttachmentSnapshotV0_6 {
    set: Arc<AttachmentSetV0_6>,
    _lease: Arc<SnapshotLeaseV0_6>,
}

impl AttachmentSnapshotV0_6 {
    pub fn generation(&self) -> u64 {
        self.set.generation
    }

    pub fn owner(&self, operation: Operation) -> Option<&str> {
        self.set
            .owners
            .get(&operation)
            .map(|record| record.policy.manifest().package_name.as_str())
    }

    pub fn negotiated_mechanics(&self, operation: Operation) -> Option<&BTreeSet<MechanicId>> {
        self.set
            .owners
            .get(&operation)
            .map(|record| &record.negotiated_mechanics)
    }

    pub fn invoke(
        &self,
        context: OperationContextV0_6,
        state: StateSnapshotV0_6,
        query_handler: Arc<dyn QueryHandler>,
        protocol_limits: ProtocolLimitsV0_6,
    ) -> Invocation<PreparedPolicyResultV0_6> {
        let operation = context.operation();
        let Some(record) = self.set.owners.get(&operation) else {
            return Invocation::Unavailable;
        };
        if let Some(actual) = context_mechanics(&context)
            && actual != record.negotiated_mechanics
        {
            return Invocation::FallbackRequired(InvocationFailure::new(
                InvocationFailureKind::InvalidInput,
                format!(
                    "context mechanics {actual:?} do not match negotiated mechanics {:?}",
                    record.negotiated_mechanics
                ),
            ));
        }
        record.policy.invoke(
            context,
            state,
            query_handler,
            Arc::new(record.action_methods.clone()),
            Arc::new(record.negotiated_mechanics.clone()),
            protocol_limits,
        )
    }
}

fn context_mechanics(context: &OperationContextV0_6) -> Option<BTreeSet<MechanicId>> {
    match context {
        OperationContextV0_6::Admit(context) => {
            Some(context.meta.mechanics.iter().cloned().collect())
        }

        OperationContextV0_6::Route(context) => {
            Some(context.meta.mechanics.iter().cloned().collect())
        }
        OperationContextV0_6::Schedule(context) => {
            Some(context.meta.mechanics.iter().cloned().collect())
        }
        OperationContextV0_6::Cache(context) => {
            Some(context.meta.mechanics.iter().cloned().collect())
        }
        OperationContextV0_6::Feedback(_) => None,
    }
}

fn standard_mechanic_action_methods(
    mechanics: &BTreeSet<MechanicId>,
) -> Result<BTreeSet<String>, RegistryErrorV0_6> {
    let mut methods = BTreeSet::new();
    for mechanic in mechanics {
        let standard = standard_mechanic(mechanic.as_str()).ok_or_else(|| {
            RegistryErrorV0_6::StandardMechanic(format!(
                "unknown standard mechanic {}",
                mechanic.as_str()
            ))
        })?;
        if standard.kind == MechanicKind::Action {
            let method = standard.method.ok_or_else(|| {
                RegistryErrorV0_6::StandardMechanic(format!(
                    "action mechanic {} has no method",
                    mechanic.as_str()
                ))
            })?;
            methods.insert(method.to_owned());
        }
    }
    Ok(methods)
}

#[derive(Debug, Error)]
pub enum RegistryErrorV0_6 {
    #[error("failed to prepare policy attachment")]
    Prepare(#[source] AttachmentErrorV0_6),
    #[error("package {0} is already attached")]
    PackageAlreadyAttached(String),
    #[error("package {0} is not attached")]
    PackageNotAttached(String),
    #[error("operation {0:?} is not attached")]
    OperationNotAttached(Operation),
    #[error("operation {operation:?} is already owned by package {package}")]
    OperationAlreadyOwned {
        operation: Operation,
        package: String,
    },
    #[error("required mechanic {0:?} is unavailable")]
    MissingRequiredMechanic(MechanicId),
    #[error("required schema {kind:?}/{id:?} is unavailable")]
    MissingRequiredSchema { kind: SchemaKind, id: String },
    #[error("standard mechanic registry error: {0}")]
    StandardMechanic(String),
    #[error("attachment generation counter exhausted")]
    GenerationExhausted,
    #[error("attachment snapshot counter exhausted")]
    SnapshotCounterExhausted,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pie_plex::v0_6::{ContractVersion, PolicyLimits, SchemaKind, SchemaRequirement};

    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            contract: ContractVersion::V0_6,
            package_name: "test".into(),
            package_version: "0.6.0".into(),
            implements: BTreeSet::from([Operation::Schedule]),
            requires: BTreeSet::from(["request.cancel@1".into()]),
            optional: BTreeSet::from(["cache.prefetch@1".into()]),
            schemas: BTreeSet::<SchemaRequirement>::new(),
            limits: PolicyLimits {
                memory_bytes: 1,
                fuel: 1,
                deadline_ms: 1,
                input_bytes: 1,
                output_bytes: 1,
                host_calls: 1,
                host_call_bytes: 1,
            },
        }
    }

    #[test]
    fn required_and_optional_mechanics_are_negotiated_explicitly() {
        let manifest = manifest();
        let missing = HostSupportV0_6::default();
        assert!(matches!(
            validate_requirements(&manifest, &missing),
            Err(RegistryErrorV0_6::MissingRequiredMechanic(_))
        ));

        let support = HostSupportV0_6::with_standard_mechanics([
            MechanicId::from("request.cancel@1"),
            MechanicId::from("cache.prefetch@1"),
        ])
        .unwrap();
        validate_requirements(&manifest, &support).unwrap();
        assert_eq!(
            negotiated_mechanics(&manifest, &support),
            BTreeSet::from([
                MechanicId::from("request.cancel@1"),
                MechanicId::from("cache.prefetch@1")
            ])
        );

        let mut schema_manifest = manifest;
        schema_manifest.schemas.insert(SchemaRequirement {
            kind: SchemaKind::ActionInput,
            id: "pie.request.cancel@1".into(),
            required: true,
        });
        assert!(matches!(
            validate_requirements(&schema_manifest, &support),
            Err(RegistryErrorV0_6::MissingRequiredSchema { .. })
        ));
        let mut schema_support = support;
        schema_support.schemas.insert(SchemaKeyV0_6 {
            kind: SchemaKind::ActionInput,
            id: "pie.request.cancel@1".into(),
        });
        validate_requirements(&schema_manifest, &schema_support).unwrap();
    }
}
