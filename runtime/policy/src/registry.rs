use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};

use pie_plex::{Document, Operation};
use thiserror::Error;

use crate::{AttachedPolicy, AttachmentError, Invocation, JsonResponse, PolicyEngine};

#[derive(Clone)]
pub struct AttachmentRegistry {
    engine: PolicyEngine,
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    state: Mutex<RegistryState>,
    changed: Condvar,
}

struct RegistryState {
    active: Arc<AttachmentSet>,
    updating: bool,
    snapshots: u64,
}

#[derive(Clone, Default)]
struct AttachmentSet {
    generation: u64,
    owners: BTreeMap<Operation, AttachedPolicy>,
    packages: BTreeMap<String, AttachedPolicy>,
}

impl AttachmentRegistry {
    pub fn new(engine: PolicyEngine) -> Self {
        Self {
            engine,
            inner: Arc::new(RegistryInner {
                state: Mutex::new(RegistryState {
                    active: Arc::new(AttachmentSet::default()),
                    updating: false,
                    snapshots: 0,
                }),
                changed: Condvar::new(),
            }),
        }
    }

    pub fn prepare(&self, package: &[u8]) -> Result<AttachedPolicy, RegistryError> {
        AttachedPolicy::compile_package(self.engine.clone(), package)
            .map_err(RegistryError::Prepare)
    }

    pub fn attach(&self, package: &[u8]) -> Result<u64, RegistryError> {
        let prepared = self.prepare(package)?;
        self.attach_prepared(prepared)
    }

    pub fn attach_prepared(&self, policy: AttachedPolicy) -> Result<u64, RegistryError> {
        let mut state = self.lock_stable();
        let package_name = policy.manifest().package_name.clone();
        if state.active.packages.contains_key(&package_name) {
            return Err(RegistryError::PackageAlreadyAttached(package_name));
        }
        ensure_operations_available(&state.active, &policy, None)?;
        let generation = next_generation(state.active.generation)?;
        let mut next = (*state.active).clone();
        for operation in &policy.manifest().operations {
            next.owners.insert(*operation, policy.clone());
        }
        next.packages.insert(package_name, policy);
        next.generation = generation;
        state.active = Arc::new(next);
        self.inner.changed.notify_all();
        Ok(generation)
    }

    pub fn replace(&self, package: &[u8]) -> Result<u64, RegistryError> {
        let replacement = self.prepare(package)?;
        let package_name = replacement.manifest().package_name.clone();
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
                .ok_or_else(|| RegistryError::PackageNotAttached(package_name.clone()))?;
            ensure_operations_available(&state.active, &replacement, Some(&package_name))?;

            let generation = next_generation(state.active.generation)?;
            let mut next = (*state.active).clone();
            next.owners
                .retain(|_, owner| owner.manifest().package_name != package_name);
            for operation in &replacement.manifest().operations {
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

    pub fn detach_operation(&self, operation: Operation) -> Result<u64, RegistryError> {
        let mut state = self.lock_stable();
        let owner = state
            .active
            .owners
            .get(&operation)
            .cloned()
            .ok_or(RegistryError::OperationNotAttached(operation))?;
        let generation = next_generation(state.active.generation)?;
        let mut next = (*state.active).clone();
        next.owners.remove(&operation);
        if !next
            .owners
            .values()
            .any(|candidate| candidate.manifest().package_name == owner.manifest().package_name)
        {
            next.packages.remove(&owner.manifest().package_name);
        }
        next.generation = generation;
        state.active = Arc::new(next);
        Ok(generation)
    }

    pub fn detach_package(&self, package_name: &str) -> Result<u64, RegistryError> {
        let mut state = self.lock_stable();
        if !state.active.packages.contains_key(package_name) {
            return Err(RegistryError::PackageNotAttached(package_name.to_owned()));
        }
        let generation = next_generation(state.active.generation)?;
        let mut next = (*state.active).clone();
        next.owners
            .retain(|_, owner| owner.manifest().package_name != package_name);
        next.packages.remove(package_name);
        next.generation = generation;
        state.active = Arc::new(next);
        Ok(generation)
    }

    pub fn snapshot(&self) -> Result<AttachmentSnapshot, RegistryError> {
        let mut state = self.lock_stable();
        state.snapshots = state
            .snapshots
            .checked_add(1)
            .ok_or(RegistryError::SnapshotCounterExhausted)?;
        Ok(AttachmentSnapshot {
            set: state.active.clone(),
            _lease: Arc::new(SnapshotLease {
                registry: self.inner.clone(),
            }),
        })
    }

    pub(crate) fn uses_realtime_epochs(&self) -> bool {
        self.engine.uses_realtime_epochs()
    }

    pub(crate) fn max_feedback_deliveries(&self) -> usize {
        self.engine.config().max_feedback_deliveries
    }

    fn lock_stable(&self) -> std::sync::MutexGuard<'_, RegistryState> {
        let mut state = self.inner.state.lock().unwrap();
        while state.updating {
            state = self.inner.changed.wait(state).unwrap();
        }
        state
    }
}

fn ensure_operations_available(
    active: &AttachmentSet,
    candidate: &AttachedPolicy,
    replacing: Option<&str>,
) -> Result<(), RegistryError> {
    for operation in &candidate.manifest().operations {
        if let Some(owner) = active.owners.get(operation)
            && replacing.is_none_or(|package| owner.manifest().package_name != package)
        {
            return Err(RegistryError::OperationAlreadyOwned {
                operation: *operation,
                package: owner.manifest().package_name.clone(),
            });
        }
    }
    Ok(())
}

fn next_generation(current: u64) -> Result<u64, RegistryError> {
    current
        .checked_add(1)
        .ok_or(RegistryError::GenerationExhausted)
}

struct SnapshotLease {
    registry: Arc<RegistryInner>,
}

impl Drop for SnapshotLease {
    fn drop(&mut self) {
        let mut state = self.registry.state.lock().unwrap();
        state.snapshots = state.snapshots.saturating_sub(1);
        self.registry.changed.notify_all();
    }
}

#[derive(Clone)]
pub struct AttachmentSnapshot {
    set: Arc<AttachmentSet>,
    _lease: Arc<SnapshotLease>,
}

impl AttachmentSnapshot {
    pub fn generation(&self) -> u64 {
        self.set.generation
    }

    pub fn owner(&self, operation: Operation) -> Option<&str> {
        self.set
            .owners
            .get(&operation)
            .map(|policy| policy.manifest().package_name.as_str())
    }

    pub fn invoke(&self, operation: Operation, input: Document) -> Invocation<JsonResponse> {
        let Some(policy) = self.set.owners.get(&operation) else {
            return Invocation::Unavailable;
        };
        match operation {
            Operation::Route => policy.route(input),
            Operation::Admit => policy.admit(input),
            Operation::Schedule => policy.schedule(input),
            Operation::Evict => policy.evict(input),
            Operation::Feedback => policy.feedback(input),
        }
    }
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed to prepare policy attachment")]
    Prepare(#[source] AttachmentError),
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
    #[error("attachment generation counter exhausted")]
    GenerationExhausted,
    #[error("attachment snapshot counter exhausted")]
    SnapshotCounterExhausted,
}
