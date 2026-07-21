use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pie_plex::{
    DeliveryId, FeedbackAck, FeedbackAcknowledgement, MapClass, MapDeclaration, MapHandle, MapKey,
    MapMutation, MapMutationValidationError, MapPersistence, Revision, Symbol, TypedValue,
};
use thiserror::Error;

const MAX_POLICY_IDENTITY_BYTES: usize = 128;

pub trait Clock: Send + Sync + 'static {
    fn now_ms(&self) -> u64;
}

pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

#[derive(Default)]
pub struct ManualClock {
    now_ms: AtomicU64,
}

impl ManualClock {
    pub fn new(now_ms: u64) -> Self {
        Self {
            now_ms: AtomicU64::new(now_ms),
        }
    }

    pub fn set(&self, now_ms: u64) {
        self.now_ms.store(now_ms, Ordering::Release);
    }

    pub fn advance(&self, delta_ms: u64) {
        self.now_ms.fetch_add(delta_ms, Ordering::AcqRel);
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.now_ms.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DedupLimits {
    pub max_entries: usize,
}

impl Default for DedupLimits {
    fn default() -> Self {
        Self { max_entries: 4096 }
    }
}

#[derive(Clone)]
pub struct MapStore {
    inner: Arc<MapStoreInner>,
}

struct MapStoreInner {
    clock: Arc<dyn Clock>,
    dedup_limits: DedupLimits,
    state: Mutex<StoreState>,
}

struct StoreState {
    maps: BTreeMap<MapHandle, StoredMap>,
    next_revision: u64,
    next_transaction: u64,
    snapshot_revisions: BTreeMap<(MapHandle, Revision), usize>,
    fences: BTreeMap<EntryKey, u64>,
    map_fences: BTreeMap<MapHandle, u64>,
    reserved_entries: BTreeMap<MapHandle, usize>,
    dedup: BTreeMap<DedupKey, DedupEntry>,
    dedup_fences: BTreeMap<DedupKey, u64>,
}

#[derive(Clone)]
struct StoredMap {
    declaration: MapDeclaration,
    revision: Revision,
    entries: Arc<BTreeMap<MapKey, StoredEntry>>,
}

#[derive(Clone)]
struct StoredEntry {
    value: TypedValue,
    revision: Revision,
    expires_at_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EntryKey {
    map: MapHandle,
    key: MapKey,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DedupKey {
    policy: String,
    delivery: DeliveryId,
}

#[derive(Clone, Copy)]
struct DedupEntry {
    ack: FeedbackAck,
}

impl MapStore {
    pub fn new(
        declarations: impl IntoIterator<Item = (MapHandle, MapDeclaration)>,
        clock: Arc<dyn Clock>,
        dedup_limits: DedupLimits,
    ) -> Result<Self, MapStoreError> {
        if dedup_limits.max_entries == 0 {
            return Err(MapStoreError::InvalidDedupLimits);
        }
        let mut maps = BTreeMap::new();
        for (handle, declaration) in declarations {
            if maps
                .insert(
                    handle,
                    StoredMap {
                        declaration,
                        revision: Revision::ZERO,
                        entries: Arc::new(BTreeMap::new()),
                    },
                )
                .is_some()
            {
                return Err(MapStoreError::DuplicateMap(handle));
            }
        }
        Ok(Self {
            inner: Arc::new(MapStoreInner {
                clock,
                dedup_limits,
                state: Mutex::new(StoreState {
                    maps,
                    next_revision: 0,
                    next_transaction: 0,
                    snapshot_revisions: BTreeMap::new(),
                    fences: BTreeMap::new(),
                    map_fences: BTreeMap::new(),
                    reserved_entries: BTreeMap::new(),
                    dedup: BTreeMap::new(),
                    dedup_fences: BTreeMap::new(),
                }),
            }),
        })
    }

    pub fn begin(&self) -> Result<InvocationTransaction, MapStoreError> {
        self.begin_inner(None)
    }

    pub fn uses_clock(&self, clock: &Arc<dyn Clock>) -> bool {
        Arc::ptr_eq(&self.inner.clock, clock)
    }

    pub fn begin_feedback(
        &self,
        policy: impl Into<String>,
        delivery: DeliveryId,
    ) -> Result<FeedbackStart, MapStoreError> {
        let policy = policy.into();
        if policy.is_empty() {
            return Err(MapStoreError::EmptyPolicyIdentity);
        }
        if policy.len() > MAX_POLICY_IDENTITY_BYTES {
            return Err(MapStoreError::PolicyIdentityTooLong {
                actual: policy.len(),
                maximum: MAX_POLICY_IDENTITY_BYTES,
            });
        }
        let key = DedupKey { policy, delivery };
        let state = self.inner.state.lock().unwrap();
        if let Some(entry) = state.dedup.get(&key) {
            return Ok(FeedbackStart::Duplicate(
                FeedbackAcknowledgement::Duplicate(entry.ack),
            ));
        }
        if state.dedup.len().saturating_add(state.dedup_fences.len())
            >= self.inner.dedup_limits.max_entries
        {
            return Err(MapStoreError::DedupCapacity {
                maximum: self.inner.dedup_limits.max_entries,
            });
        }
        drop(state);
        self.begin_inner(Some(key)).map(FeedbackStart::New)
    }

    fn begin_inner(
        &self,
        feedback: Option<DedupKey>,
    ) -> Result<InvocationTransaction, MapStoreError> {
        let now = self.inner.clock.now_ms();
        let mut state = self.inner.state.lock().unwrap();
        purge_expired(&mut state, now)?;
        let transaction = state
            .next_transaction
            .checked_add(1)
            .ok_or(MapStoreError::TransactionExhausted)?;
        state.next_transaction = transaction;
        let snapshot = state
            .maps
            .iter()
            .map(|(handle, map)| {
                (
                    *handle,
                    SnapshotMap {
                        declaration: map.declaration.clone(),
                        revision: map.revision,
                        entries: map.entries.clone(),
                    },
                )
            })
            .collect();
        let snapshots: Vec<_> = state
            .maps
            .iter()
            .map(|(handle, map)| (*handle, map.revision))
            .collect();
        for snapshot in &snapshots {
            *state.snapshot_revisions.entry(*snapshot).or_insert(0) += 1;
        }
        Ok(InvocationTransaction {
            store: self.clone(),
            transaction,
            snapshot,
            observed: BTreeMap::new(),
            mutations: Vec::new(),
            feedback,
            snapshot_time_ms: now,
            lease: Some(ActiveLease {
                store: self.clone(),
                snapshots,
            }),
        })
    }

    pub fn publish_external(
        &self,
        handle: MapHandle,
        entries: impl IntoIterator<Item = (MapKey, TypedValue)>,
    ) -> Result<Revision, MapStoreError> {
        let declaration = {
            let state = self.inner.state.lock().unwrap();
            state
                .maps
                .get(&handle)
                .ok_or(MapStoreError::UnknownMap(handle))?
                .declaration
                .clone()
        };
        if !matches!(declaration.class, MapClass::External { .. }) {
            return Err(MapStoreError::NotExternal(handle));
        }

        let mut replacement = BTreeMap::new();
        for (key, value) in entries {
            validate_entry(&declaration, &key, &value)?;
            if replacement.insert(key, value).is_some() {
                return Err(MapStoreError::DuplicateExternalKey(handle));
            }
            if replacement.len() > declaration.schema.max_entries as usize {
                return Err(MapStoreError::Capacity {
                    handle,
                    maximum: declaration.schema.max_entries as usize,
                });
            }
        }

        let now = self.inner.clock.now_ms();
        let mut state = self.inner.state.lock().unwrap();
        purge_expired(&mut state, now)?;
        let current_revision = state
            .maps
            .get(&handle)
            .ok_or(MapStoreError::UnknownMap(handle))?
            .revision;
        if state.map_fences.contains_key(&handle)
            || has_older_snapshot(&state, handle, current_revision)
        {
            return Err(MapStoreError::Busy(handle));
        }
        let revision = next_revision(&mut state)?;
        let map = state.maps.get_mut(&handle).expect("map validated above");
        map.revision = revision;
        map.entries = Arc::new(
            replacement
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        StoredEntry {
                            value,
                            revision,
                            expires_at_ms: None,
                        },
                    )
                })
                .collect(),
        );
        Ok(revision)
    }

    pub fn read(
        &self,
        handle: MapHandle,
        key: &MapKey,
    ) -> Result<Option<TypedValue>, MapStoreError> {
        let now = self.inner.clock.now_ms();
        let mut state = self.inner.state.lock().unwrap();
        purge_expired(&mut state, now)?;
        let map = state
            .maps
            .get(&handle)
            .ok_or(MapStoreError::UnknownMap(handle))?;
        Ok(map
            .entries
            .get(key)
            .filter(|entry| is_live(entry, now))
            .map(|entry| entry.value.clone()))
    }

    pub(crate) fn transfer_from(
        &self,
        source: &MapStore,
        source_identity: &str,
        target_identity: &str,
    ) -> Result<(), StateTransferError> {
        let transfer = source.export_transfer(source_identity)?;
        self.import_transfer(transfer, target_identity)
    }

    fn export_transfer(&self, policy_identity: &str) -> Result<StoreTransfer, StateTransferError> {
        let now = self.inner.clock.now_ms();
        let mut state = self.inner.state.lock().unwrap();
        purge_expired(&mut state, now)?;
        if !state.snapshot_revisions.is_empty()
            || !state.map_fences.is_empty()
            || !state.dedup_fences.is_empty()
        {
            return Err(StateTransferError::Busy);
        }
        let maps = state
            .maps
            .values()
            .filter_map(|map| {
                let transferable = matches!(
                    map.declaration.class,
                    MapClass::External { .. }
                        | MapClass::PolicyOwned {
                            persistence: MapPersistence::Pinned
                        }
                );
                transferable.then(|| {
                    let entries = map
                        .entries
                        .iter()
                        .filter(|(_, entry)| is_live(entry, now))
                        .map(|(key, entry)| TransferEntry {
                            key: key.clone(),
                            value: entry.value.clone(),
                            revision: entry.revision,
                            ttl_ms: entry
                                .expires_at_ms
                                .map(|expires_at| expires_at.saturating_sub(now)),
                        })
                        .collect();
                    (
                        map.declaration.name.clone(),
                        TransferMap {
                            declaration: map.declaration.clone(),
                            revision: map.revision,
                            entries,
                        },
                    )
                })
            })
            .collect();
        let dedup = state
            .dedup
            .iter()
            .filter(|(key, _)| key.policy == policy_identity)
            .map(|(key, entry)| (key.delivery, entry.ack))
            .collect();
        Ok(StoreTransfer {
            maps,
            dedup,
            next_revision: state.next_revision,
        })
    }

    fn import_transfer(
        &self,
        transfer: StoreTransfer,
        target_identity: &str,
    ) -> Result<(), StateTransferError> {
        let now = self.inner.clock.now_ms();
        let mut state = self.inner.state.lock().unwrap();
        if state.maps.values().any(|map| !map.entries.is_empty())
            || !state.dedup.is_empty()
            || !state.snapshot_revisions.is_empty()
            || !state.fences.is_empty()
            || !state.map_fences.is_empty()
            || !state.reserved_entries.is_empty()
            || !state.dedup_fences.is_empty()
        {
            return Err(StateTransferError::TargetNotEmpty);
        }

        for transfer_map in transfer.maps.values().filter(|map| {
            matches!(
                map.declaration.class,
                MapClass::PolicyOwned {
                    persistence: MapPersistence::Pinned
                }
            )
        }) {
            let Some(target) = state
                .maps
                .values()
                .find(|map| map.declaration.name == transfer_map.declaration.name)
            else {
                return Err(StateTransferError::MissingPinnedMap(
                    transfer_map.declaration.name.clone(),
                ));
            };
            if target.declaration != transfer_map.declaration {
                return Err(StateTransferError::IncompatibleMap(
                    transfer_map.declaration.name.clone(),
                ));
            }
        }

        let target_by_name: BTreeMap<_, _> = state
            .maps
            .iter()
            .map(|(handle, map)| (map.declaration.name.clone(), *handle))
            .collect();
        let transferable: Vec<_> = transfer
            .maps
            .into_iter()
            .filter_map(|(name, transfer_map)| {
                target_by_name
                    .get(&name)
                    .copied()
                    .map(|handle| (handle, transfer_map))
            })
            .collect();
        for (handle, transfer_map) in &transferable {
            let target = state
                .maps
                .get(handle)
                .expect("target handle came from the target map set");
            if target.declaration != transfer_map.declaration {
                return Err(StateTransferError::IncompatibleMap(
                    transfer_map.declaration.name.clone(),
                ));
            }
            if transfer_map.entries.len() > target.declaration.schema.max_entries as usize {
                return Err(StateTransferError::Capacity(
                    transfer_map.declaration.name.clone(),
                ));
            }
        }
        if transfer.dedup.len() > self.inner.dedup_limits.max_entries {
            return Err(StateTransferError::DedupCapacity {
                actual: transfer.dedup.len(),
                maximum: self.inner.dedup_limits.max_entries,
            });
        }

        state.next_revision = transfer.next_revision;
        for (handle, transfer_map) in transferable {
            let entries = transfer_map
                .entries
                .into_iter()
                .map(|entry| {
                    (
                        entry.key,
                        StoredEntry {
                            value: entry.value,
                            revision: entry.revision,
                            expires_at_ms: entry.ttl_ms.map(|ttl| now.saturating_add(ttl)),
                        },
                    )
                })
                .collect();
            let target = state
                .maps
                .get_mut(&handle)
                .expect("target handle was validated");
            target.revision = transfer_map.revision;
            target.entries = Arc::new(entries);
        }
        for (delivery, ack) in transfer.dedup {
            state.dedup.insert(
                DedupKey {
                    policy: target_identity.to_owned(),
                    delivery,
                },
                DedupEntry { ack },
            );
        }
        Ok(())
    }
}

struct StoreTransfer {
    maps: BTreeMap<Symbol, TransferMap>,
    dedup: Vec<(DeliveryId, FeedbackAck)>,
    next_revision: u64,
}

struct TransferMap {
    declaration: MapDeclaration,
    revision: Revision,
    entries: Vec<TransferEntry>,
}

struct TransferEntry {
    key: MapKey,
    value: TypedValue,
    revision: Revision,
    ttl_ms: Option<u64>,
}

pub enum FeedbackStart {
    Duplicate(FeedbackAcknowledgement),
    New(InvocationTransaction),
}

struct SnapshotMap {
    declaration: MapDeclaration,
    revision: Revision,
    entries: Arc<BTreeMap<MapKey, StoredEntry>>,
}

#[derive(Clone, Copy)]
enum ObservedRevision {
    Present(Revision),
    Absent(Revision),
}

pub struct InvocationTransaction {
    store: MapStore,
    transaction: u64,
    snapshot: BTreeMap<MapHandle, SnapshotMap>,
    observed: BTreeMap<EntryKey, ObservedRevision>,
    mutations: Vec<MapMutation>,
    feedback: Option<DedupKey>,
    snapshot_time_ms: u64,
    lease: Option<ActiveLease>,
}

struct ActiveLease {
    store: MapStore,
    snapshots: Vec<(MapHandle, Revision)>,
}

impl Drop for ActiveLease {
    fn drop(&mut self) {
        let mut state = self.store.inner.state.lock().unwrap();
        for snapshot in &self.snapshots {
            let remove = if let Some(count) = state.snapshot_revisions.get_mut(snapshot) {
                *count = count.saturating_sub(1);
                *count == 0
            } else {
                false
            };
            if remove {
                state.snapshot_revisions.remove(snapshot);
            }
        }
    }
}

impl InvocationTransaction {
    pub fn get(
        &mut self,
        handle: MapHandle,
        key: &MapKey,
    ) -> Result<Option<TypedValue>, MapAccessError> {
        self.get_bounded(handle, key, u64::MAX)
    }

    pub fn get_bounded(
        &mut self,
        handle: MapHandle,
        key: &MapKey,
        max_value_bytes: u64,
    ) -> Result<Option<TypedValue>, MapAccessError> {
        let map = self
            .snapshot
            .get(&handle)
            .ok_or(MapAccessError::UnknownMap(handle))?;
        validate_key(&map.declaration, key)?;
        let entry_key = EntryKey {
            map: handle,
            key: key.clone(),
        };
        let value = map
            .entries
            .get(key)
            .filter(|entry| is_live(entry, self.snapshot_time_ms));
        self.observed.entry(entry_key).or_insert_with(|| {
            value.map_or(ObservedRevision::Absent(map.revision), |entry| {
                ObservedRevision::Present(entry.revision)
            })
        });
        if let Some(entry) = value {
            let actual = u64::try_from(entry.value.payload_len()).unwrap_or(u64::MAX);
            if actual > max_value_bytes {
                return Err(MapAccessError::InvocationByteLimit {
                    actual,
                    maximum: max_value_bytes,
                });
            }
        }
        Ok(value.map(|entry| entry.value.clone()))
    }

    pub fn stage(&mut self, mutation: MapMutation) -> Result<(), MapAccessError> {
        let handle = mutation.map();
        let map = self
            .snapshot
            .get(&handle)
            .ok_or(MapAccessError::UnknownMap(handle))?;
        mutation
            .validate_against(&map.declaration)
            .map_err(MapAccessError::Mutation)?;
        let entry_key = EntryKey {
            map: handle,
            key: mutation.key().clone(),
        };
        self.observed.entry(entry_key).or_insert_with(|| {
            map.entries
                .get(mutation.key())
                .filter(|entry| is_live(entry, self.snapshot_time_ms))
                .map_or(ObservedRevision::Absent(map.revision), |entry| {
                    ObservedRevision::Present(entry.revision)
                })
        });
        self.mutations.push(mutation);
        Ok(())
    }

    pub fn stage_all(
        &mut self,
        mutations: impl IntoIterator<Item = MapMutation>,
    ) -> Result<(), MapAccessError> {
        for mutation in mutations {
            self.stage(mutation)?;
        }
        Ok(())
    }

    pub fn prepare(mut self) -> Result<PreparedTransaction, PrepareError> {
        let now = self.store.inner.clock.now_ms();
        let mut state = self.store.inner.state.lock().unwrap();
        purge_expired(&mut state, now).map_err(PrepareError::Store)?;

        if let Some(feedback) = &self.feedback {
            if let Some(entry) = state.dedup.get(feedback) {
                return Err(PrepareError::Duplicate(FeedbackAcknowledgement::Duplicate(
                    entry.ack,
                )));
            }
            if state.dedup_fences.contains_key(feedback) {
                return Err(PrepareError::Conflict);
            }
            if state.dedup.len().saturating_add(state.dedup_fences.len())
                >= self.store.inner.dedup_limits.max_entries
            {
                return Err(PrepareError::Store(MapStoreError::DedupCapacity {
                    maximum: self.store.inner.dedup_limits.max_entries,
                }));
            }
        }

        for (key, observed) in &self.observed {
            if state
                .map_fences
                .get(&key.map)
                .is_some_and(|transaction| *transaction != self.transaction)
            {
                return Err(PrepareError::Conflict);
            }
            if state
                .fences
                .get(key)
                .is_some_and(|transaction| *transaction != self.transaction)
            {
                return Err(PrepareError::Conflict);
            }
            let map = state
                .maps
                .get(&key.map)
                .ok_or(PrepareError::Store(MapStoreError::UnknownMap(key.map)))?;
            let current = map
                .entries
                .get(&key.key)
                .filter(|entry| is_live(entry, now));
            let matches = match observed {
                ObservedRevision::Present(revision) => {
                    current.is_some_and(|entry| entry.revision == *revision)
                }
                ObservedRevision::Absent(revision) => {
                    current.is_none() && map.revision == *revision
                }
            };
            if !matches {
                return Err(PrepareError::Conflict);
            }
        }

        let write_keys: BTreeSet<_> = self
            .mutations
            .iter()
            .map(|mutation| EntryKey {
                map: mutation.map(),
                key: mutation.key().clone(),
            })
            .collect();
        let write_maps: BTreeSet<_> = write_keys.iter().map(|key| key.map).collect();
        for handle in &write_maps {
            let current = state
                .maps
                .get(handle)
                .ok_or(PrepareError::Store(MapStoreError::UnknownMap(*handle)))?
                .revision;
            if state
                .snapshot_revisions
                .iter()
                .any(|((snapshot_handle, revision), count)| {
                    snapshot_handle == handle && *revision != current && *count > 0
                })
            {
                return Err(PrepareError::Conflict);
            }
        }
        if write_maps
            .iter()
            .any(|handle| state.map_fences.contains_key(handle))
        {
            return Err(PrepareError::Conflict);
        }
        if write_keys.iter().any(|key| state.fences.contains_key(key)) {
            return Err(PrepareError::Conflict);
        }

        let changes = simulate_changes(&state, &self.mutations, now)?;
        let mut net_growth = BTreeMap::<MapHandle, isize>::new();
        for (key, change) in &changes {
            let map = state
                .maps
                .get(&key.map)
                .ok_or(PrepareError::Store(MapStoreError::UnknownMap(key.map)))?;
            let before = isize::from(
                map.entries
                    .get(&key.key)
                    .is_some_and(|entry| is_live(entry, now)),
            );
            let after = isize::from(change.value.is_some());
            *net_growth.entry(key.map).or_insert(0) += after - before;
        }
        let reservations: BTreeMap<_, _> = net_growth
            .into_iter()
            .filter_map(|(handle, growth)| {
                (growth > 0).then_some((
                    handle,
                    usize::try_from(growth).expect("positive isize fits usize"),
                ))
            })
            .collect();
        for (handle, count) in &reservations {
            let map = state
                .maps
                .get(handle)
                .ok_or(PrepareError::Store(MapStoreError::UnknownMap(*handle)))?;
            let reserved = state.reserved_entries.get(handle).copied().unwrap_or(0);
            let live_entries = map
                .entries
                .values()
                .filter(|entry| is_live(entry, now))
                .count();
            let projected = live_entries
                .checked_add(reserved)
                .and_then(|value| value.checked_add(*count))
                .ok_or(PrepareError::Capacity {
                    handle: *handle,
                    maximum: map.declaration.schema.max_entries as usize,
                })?;
            if projected > map.declaration.schema.max_entries as usize {
                return Err(PrepareError::Capacity {
                    handle: *handle,
                    maximum: map.declaration.schema.max_entries as usize,
                });
            }
        }

        let revision = if !changes.is_empty() || self.feedback.is_some() {
            Some(next_revision(&mut state).map_err(PrepareError::Store)?)
        } else {
            None
        };
        let mut roots = BTreeMap::new();
        for handle in &write_maps {
            let map = state
                .maps
                .get(handle)
                .ok_or(PrepareError::Store(MapStoreError::UnknownMap(*handle)))?;
            let mut root = (*map.entries).clone();
            root.retain(|_, entry| is_live(entry, now));
            for (key, change) in changes.iter().filter(|(key, _)| key.map == *handle) {
                match &change.value {
                    Some(value) => {
                        root.insert(
                            key.key.clone(),
                            StoredEntry {
                                value: value.clone(),
                                revision: revision.expect("map changes reserve a revision"),
                                expires_at_ms: None,
                            },
                        );
                    }
                    None => {
                        root.remove(&key.key);
                    }
                }
            }
            if root.len() > map.declaration.schema.max_entries as usize {
                return Err(PrepareError::Capacity {
                    handle: *handle,
                    maximum: map.declaration.schema.max_entries as usize,
                });
            }
            roots.insert(*handle, Arc::new(root));
        }
        for key in &write_keys {
            state.fences.insert(key.clone(), self.transaction);
        }
        for handle in &write_maps {
            state.map_fences.insert(*handle, self.transaction);
        }
        for (handle, count) in &reservations {
            *state.reserved_entries.entry(*handle).or_insert(0) += count;
        }
        if let Some(feedback) = &self.feedback {
            state
                .dedup_fences
                .insert(feedback.clone(), self.transaction);
        }
        drop(state);
        let store = self.store.clone();
        let lease = self
            .lease
            .take()
            .expect("active lease transfers to prepare");
        Ok(PreparedTransaction {
            store,
            transaction: self.transaction,
            changes,
            write_keys,
            reservations,
            feedback: self.feedback,
            revision,
            roots,
            write_maps,
            _lease: lease,
            finished: false,
        })
    }
}

struct PreparedChange {
    value: Option<TypedValue>,
    ttl_ms: Option<u64>,
}

fn simulate_changes(
    state: &StoreState,
    mutations: &[MapMutation],
    now_ms: u64,
) -> Result<BTreeMap<EntryKey, PreparedChange>, PrepareError> {
    let mut changes: BTreeMap<EntryKey, PreparedChange> = BTreeMap::new();
    for mutation in mutations {
        let handle = mutation.map();
        let map = state
            .maps
            .get(&handle)
            .ok_or(PrepareError::Store(MapStoreError::UnknownMap(handle)))?;
        mutation
            .validate_against(&map.declaration)
            .map_err(PrepareError::Mutation)?;
        let key = EntryKey {
            map: handle,
            key: mutation.key().clone(),
        };
        let current = match changes.get(&key) {
            Some(change) => change.value.clone(),
            None => map
                .entries
                .get(&key.key)
                .filter(|entry| is_live(entry, now_ms))
                .map(|entry| entry.value.clone()),
        };
        let ttl_ms = mutation.ttl_ms().or(map.declaration.schema.default_ttl_ms);
        let change = match mutation {
            MapMutation::Upsert { value, .. } => PreparedChange {
                value: Some(value.clone()),
                ttl_ms,
            },
            MapMutation::AddI64 { delta, .. } => {
                let value = match current {
                    Some(TypedValue::I64(value)) => value
                        .checked_add(*delta)
                        .ok_or(PrepareError::NumericOverflow)?,
                    None => *delta,
                    Some(_) => return Err(PrepareError::StoreInvariant),
                };
                PreparedChange {
                    value: Some(TypedValue::I64(value)),
                    ttl_ms,
                }
            }
            MapMutation::AddU64 { delta, .. } => {
                let value = match current {
                    Some(TypedValue::U64(value)) => value
                        .checked_add(*delta)
                        .ok_or(PrepareError::NumericOverflow)?,
                    None => *delta,
                    Some(_) => return Err(PrepareError::StoreInvariant),
                };
                PreparedChange {
                    value: Some(TypedValue::U64(value)),
                    ttl_ms,
                }
            }
            MapMutation::Delete { .. } => PreparedChange {
                value: None,
                ttl_ms: None,
            },
        };
        changes.insert(key, change);
    }
    Ok(changes)
}

#[must_use = "prepared effects must be committed or aborted after enactment"]
pub struct PreparedTransaction {
    store: MapStore,
    transaction: u64,
    changes: BTreeMap<EntryKey, PreparedChange>,
    write_keys: BTreeSet<EntryKey>,
    reservations: BTreeMap<MapHandle, usize>,
    feedback: Option<DedupKey>,
    revision: Option<Revision>,
    roots: BTreeMap<MapHandle, Arc<BTreeMap<MapKey, StoredEntry>>>,
    write_maps: BTreeSet<MapHandle>,
    _lease: ActiveLease,
    finished: bool,
}

impl PreparedTransaction {
    pub fn commit(mut self) -> CommitResult {
        let mut state = self.store.inner.state.lock().unwrap();
        for key in &self.write_keys {
            assert_eq!(
                state.fences.get(key),
                Some(&self.transaction),
                "prepared PLEX transaction lost its write fence"
            );
        }
        if let Some(feedback) = &self.feedback {
            assert_eq!(
                state.dedup_fences.get(feedback),
                Some(&self.transaction),
                "prepared PLEX feedback transaction lost its dedup fence"
            );
        }

        let now = self.store.inner.clock.now_ms();
        let revision = self.revision;
        let mut roots = std::mem::take(&mut self.roots);
        for (key, change) in &self.changes {
            if change.value.is_some() {
                let root = roots
                    .get_mut(&key.map)
                    .expect("prepared root exists for every changed map");
                let entry = Arc::get_mut(root)
                    .expect("prepared root is private until publication")
                    .get_mut(&key.key)
                    .expect("prepared live value exists");
                entry.expires_at_ms = change.ttl_ms.map(|ttl| now.saturating_add(ttl));
            }
        }
        if let Some(revision) = revision {
            for (handle, root) in roots {
                let map = state
                    .maps
                    .get_mut(&handle)
                    .expect("prepared map remains attached");
                map.entries = root;
                if self.write_maps.contains(&handle) {
                    map.revision = revision;
                }
            }
        }

        let feedback = self.feedback.as_ref().map(|key| {
            let ack = FeedbackAck {
                delivery_id: key.delivery,
                revision: revision.expect("feedback always allocates a revision"),
            };
            state.dedup.insert(key.clone(), DedupEntry { ack });
            FeedbackAcknowledgement::Committed(ack)
        });

        release_prepared(&mut state, &self);
        self.finished = true;
        CommitResult { revision, feedback }
    }

    pub fn abort(mut self) {
        let mut state = self.store.inner.state.lock().unwrap();
        release_prepared(&mut state, &self);
        self.finished = true;
    }
}

impl Drop for PreparedTransaction {
    fn drop(&mut self) {
        if !self.finished {
            let mut state = self.store.inner.state.lock().unwrap();
            release_prepared(&mut state, self);
            self.finished = true;
        }
    }
}

fn release_prepared(state: &mut StoreState, prepared: &PreparedTransaction) {
    for key in &prepared.write_keys {
        if state.fences.get(key) == Some(&prepared.transaction) {
            state.fences.remove(key);
        }
    }
    for handle in &prepared.write_maps {
        if state.map_fences.get(handle) == Some(&prepared.transaction) {
            state.map_fences.remove(handle);
        }
    }
    for (handle, count) in &prepared.reservations {
        let remove = if let Some(reserved) = state.reserved_entries.get_mut(handle) {
            *reserved = reserved.saturating_sub(*count);
            *reserved == 0
        } else {
            false
        };
        if remove {
            state.reserved_entries.remove(handle);
        }
    }
    if let Some(feedback) = &prepared.feedback
        && state.dedup_fences.get(feedback) == Some(&prepared.transaction)
    {
        state.dedup_fences.remove(feedback);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitResult {
    pub revision: Option<Revision>,
    pub feedback: Option<FeedbackAcknowledgement>,
}

fn purge_expired(state: &mut StoreState, now_ms: u64) -> Result<(), MapStoreError> {
    let expired: BTreeMap<_, Vec<_>> = state
        .maps
        .iter()
        .filter_map(|(handle, map)| {
            if state.map_fences.contains_key(handle)
                || has_older_snapshot(state, *handle, map.revision)
            {
                return None;
            }
            let keys: Vec<_> = map
                .entries
                .iter()
                .filter(|(key, entry)| {
                    entry.expires_at_ms.is_some_and(|expiry| expiry <= now_ms)
                        && !state.fences.contains_key(&EntryKey {
                            map: *handle,
                            key: (*key).clone(),
                        })
                })
                .map(|(key, _)| key.clone())
                .collect();
            (!keys.is_empty()).then_some((*handle, keys))
        })
        .collect();
    if expired.is_empty() {
        return Ok(());
    }
    let revision = next_revision(state)?;
    for (handle, keys) in expired {
        let map = state.maps.get_mut(&handle).expect("map came from state");
        for key in keys {
            Arc::make_mut(&mut map.entries).remove(&key);
        }
        map.revision = revision;
    }
    Ok(())
}

fn has_older_snapshot(state: &StoreState, handle: MapHandle, current: Revision) -> bool {
    state
        .snapshot_revisions
        .iter()
        .any(|((snapshot_handle, revision), count)| {
            *snapshot_handle == handle && *revision != current && *count > 0
        })
}

fn is_live(entry: &StoredEntry, now_ms: u64) -> bool {
    entry
        .expires_at_ms
        .is_none_or(|expires_at_ms| expires_at_ms > now_ms)
}

fn next_revision(state: &mut StoreState) -> Result<Revision, MapStoreError> {
    state.next_revision = state
        .next_revision
        .checked_add(1)
        .ok_or(MapStoreError::RevisionExhausted)?;
    Ok(Revision::new(state.next_revision))
}

fn validate_key(declaration: &MapDeclaration, key: &MapKey) -> Result<(), MapAccessError> {
    if key.key_type() != declaration.schema.key_type {
        return Err(MapAccessError::KeyType);
    }
    if key.payload_len() > declaration.schema.max_key_bytes as usize {
        return Err(MapAccessError::KeyTooLarge);
    }
    Ok(())
}

fn validate_entry(
    declaration: &MapDeclaration,
    key: &MapKey,
    value: &TypedValue,
) -> Result<(), MapStoreError> {
    validate_key(declaration, key).map_err(MapStoreError::Access)?;
    value.validate().map_err(|_| MapStoreError::InvalidValue)?;
    if value.value_type() != declaration.schema.value_type {
        return Err(MapStoreError::ValueType);
    }
    if value.payload_len() > declaration.schema.max_value_bytes as usize {
        return Err(MapStoreError::ValueTooLarge);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MapStoreError {
    #[error("deduplication bounds must be non-zero")]
    InvalidDedupLimits,
    #[error("duplicate map handle {0:?}")]
    DuplicateMap(MapHandle),
    #[error("unknown map handle {0:?}")]
    UnknownMap(MapHandle),
    #[error("map handle {0:?} is not external")]
    NotExternal(MapHandle),
    #[error("map handle {0:?} has a prepared writer or retained older snapshot")]
    Busy(MapHandle),
    #[error("external map {0:?} contains a duplicate key")]
    DuplicateExternalKey(MapHandle),
    #[error("map {handle:?} exceeds capacity {maximum}")]
    Capacity { handle: MapHandle, maximum: usize },
    #[error("revision counter exhausted")]
    RevisionExhausted,
    #[error("transaction counter exhausted")]
    TransactionExhausted,
    #[error("feedback policy identity is empty")]
    EmptyPolicyIdentity,
    #[error("feedback policy identity contains {actual} bytes; maximum is {maximum}")]
    PolicyIdentityTooLong { actual: usize, maximum: usize },
    #[error("feedback deduplication ledger is full; maximum is {maximum}")]
    DedupCapacity { maximum: usize },
    #[error(transparent)]
    Access(#[from] MapAccessError),
    #[error("map value contains a non-finite float")]
    InvalidValue,
    #[error("map value type does not match its schema")]
    ValueType,
    #[error("map value exceeds its schema byte limit")]
    ValueTooLarge,
}

#[derive(Debug, Error)]
pub enum StateTransferError {
    #[error("source policy still has prepared map or feedback transactions")]
    Busy,
    #[error("replacement target map store is not empty")]
    TargetNotEmpty,
    #[error("replacement does not declare pinned map {0}")]
    MissingPinnedMap(Symbol),
    #[error("replacement map {0} has an incompatible declaration")]
    IncompatibleMap(Symbol),
    #[error("replacement map {0} exceeds its declared capacity")]
    Capacity(Symbol),
    #[error("replacement dedup ledger contains {actual} entries; maximum is {maximum}")]
    DedupCapacity { actual: usize, maximum: usize },
    #[error(transparent)]
    Store(#[from] MapStoreError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MapAccessError {
    #[error("unknown map handle {0:?}")]
    UnknownMap(MapHandle),
    #[error("map key type does not match its schema")]
    KeyType,
    #[error("map key exceeds its schema byte limit")]
    KeyTooLarge,
    #[error("map value contains {actual} bytes; invocation has {maximum} bytes remaining")]
    InvocationByteLimit { actual: u64, maximum: u64 },
    #[error(transparent)]
    Mutation(#[from] MapMutationValidationError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PrepareError {
    #[error("transaction conflicts with a newer value or prepared write")]
    Conflict,
    #[error("feedback delivery was already committed")]
    Duplicate(FeedbackAcknowledgement),
    #[error("map {handle:?} cannot reserve another entry; capacity is {maximum}")]
    Capacity { handle: MapHandle, maximum: usize },
    #[error("numeric add overflowed")]
    NumericOverflow,
    #[error("stored map value violates its declaration")]
    StoreInvariant,
    #[error(transparent)]
    Mutation(#[from] MapMutationValidationError),
    #[error(transparent)]
    Store(#[from] MapStoreError),
}

#[cfg(test)]
mod tests {
    use pie_plex::{
        DependencyRequirement, MapKeyType, MapPersistence, MapSchema, Symbol, ValueType,
    };

    use super::*;

    fn policy_map(handle: u32, max_entries: u32, ttl: Option<u64>) -> (MapHandle, MapDeclaration) {
        (
            MapHandle::new(handle),
            MapDeclaration {
                name: Symbol::new(format!("policy.map-{handle}@1")).unwrap(),
                class: MapClass::PolicyOwned {
                    persistence: MapPersistence::Attachment,
                },
                schema: MapSchema {
                    key_type: MapKeyType::Bytes,
                    value_type: ValueType::U64,
                    max_entries,
                    max_key_bytes: 16,
                    max_value_bytes: 8,
                    default_ttl_ms: ttl,
                    max_ttl_ms: ttl,
                },
            },
        )
    }

    fn external_map(handle: u32) -> (MapHandle, MapDeclaration) {
        (
            MapHandle::new(handle),
            MapDeclaration {
                name: Symbol::new(format!("operator.map-{handle}@1")).unwrap(),
                class: MapClass::External {
                    requirement: DependencyRequirement::Required,
                },
                schema: MapSchema {
                    key_type: MapKeyType::Bytes,
                    value_type: ValueType::U64,
                    max_entries: 8,
                    max_key_bytes: 16,
                    max_value_bytes: 8,
                    default_ttl_ms: None,
                    max_ttl_ms: None,
                },
            },
        )
    }

    fn store(
        maps: impl IntoIterator<Item = (MapHandle, MapDeclaration)>,
    ) -> (MapStore, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::default());
        (
            MapStore::new(maps, clock.clone(), DedupLimits::default()).unwrap(),
            clock,
        )
    }

    fn key(value: u8) -> MapKey {
        MapKey::Bytes(vec![value])
    }

    fn upsert(handle: u32, key_value: u8, value: u64) -> MapMutation {
        MapMutation::Upsert {
            map: MapHandle::new(handle),
            key: key(key_value),
            value: TypedValue::U64(value),
            ttl_ms: None,
        }
    }

    #[test]
    fn commit_and_abort_are_enactment_coupled() {
        let (store, _) = store([policy_map(0, 4, None)]);

        let mut aborted = store.begin().unwrap();
        aborted.stage(upsert(0, 1, 10)).unwrap();
        aborted.prepare().unwrap().abort();
        assert_eq!(store.read(MapHandle::new(0), &key(1)).unwrap(), None);

        let mut committed = store.begin().unwrap();
        committed.stage(upsert(0, 1, 10)).unwrap();
        let result = committed.prepare().unwrap().commit();
        assert_eq!(result.revision, Some(Revision::new(2)));
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(10))
        );
    }

    #[test]
    fn stale_read_and_prepared_write_conflict() {
        let (store, _) = store([policy_map(0, 4, None)]);
        let mut first = store.begin().unwrap();
        let mut second = store.begin().unwrap();
        assert_eq!(first.get(MapHandle::new(0), &key(1)).unwrap(), None);
        assert_eq!(second.get(MapHandle::new(0), &key(1)).unwrap(), None);
        first.stage(upsert(0, 1, 1)).unwrap();
        second.stage(upsert(0, 1, 2)).unwrap();

        let prepared = first.prepare().unwrap();
        assert!(matches!(second.prepare(), Err(PrepareError::Conflict)));
        prepared.commit();
    }

    #[test]
    fn observed_foreign_fence_prevents_write_skew() {
        let (store, _) = store([policy_map(0, 4, None)]);
        let mut seed = store.begin().unwrap();
        seed.stage(upsert(0, 1, 0)).unwrap();
        seed.stage(upsert(0, 2, 0)).unwrap();
        seed.prepare().unwrap().commit();

        let mut first = store.begin().unwrap();
        let mut second = store.begin().unwrap();
        first.get(MapHandle::new(0), &key(1)).unwrap();
        first.stage(upsert(0, 2, 1)).unwrap();
        second.get(MapHandle::new(0), &key(2)).unwrap();
        second.stage(upsert(0, 1, 1)).unwrap();

        let prepared = first.prepare().unwrap();
        assert!(matches!(second.prepare(), Err(PrepareError::Conflict)));
        prepared.abort();
    }

    #[test]
    fn old_snapshot_blocks_a_second_map_version() {
        let (store, _) = store([policy_map(0, 4, None)]);
        let old_snapshot = store.begin().unwrap();

        let mut first_writer = store.begin().unwrap();
        first_writer.stage(upsert(0, 1, 1)).unwrap();
        first_writer.prepare().unwrap().commit();

        let mut blocked_writer = store.begin().unwrap();
        blocked_writer.stage(upsert(0, 2, 2)).unwrap();
        assert!(matches!(
            blocked_writer.prepare(),
            Err(PrepareError::Conflict)
        ));

        drop(old_snapshot);
        let mut retry = store.begin().unwrap();
        retry.stage(upsert(0, 2, 2)).unwrap();
        retry.prepare().unwrap().commit();
    }

    #[test]
    fn disjoint_prepared_writes_conflict_within_one_map() {
        let (store, _) = store([policy_map(0, 1, None)]);
        let mut first = store.begin().unwrap();
        let mut second = store.begin().unwrap();
        first.stage(upsert(0, 1, 1)).unwrap();
        second.stage(upsert(0, 2, 2)).unwrap();
        let prepared = first.prepare().unwrap();
        assert!(matches!(second.prepare(), Err(PrepareError::Conflict)));
        prepared.abort();
    }

    #[test]
    fn external_revision_change_conflicts_with_snapshot() {
        let (store, _) = store([external_map(0), policy_map(1, 4, None)]);
        store
            .publish_external(MapHandle::new(0), [(key(1), TypedValue::U64(1))])
            .unwrap();
        let mut transaction = store.begin().unwrap();
        assert_eq!(
            transaction.get(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(1))
        );
        transaction.stage(upsert(1, 1, 1)).unwrap();
        store
            .publish_external(MapHandle::new(0), [(key(1), TypedValue::U64(2))])
            .unwrap();
        assert!(matches!(transaction.prepare(), Err(PrepareError::Conflict)));
    }

    #[test]
    fn external_publication_waits_for_older_snapshot_to_drain() {
        let (store, _) = store([external_map(0)]);
        store
            .publish_external(MapHandle::new(0), [(key(1), TypedValue::U64(1))])
            .unwrap();
        let old_snapshot = store.begin().unwrap();
        store
            .publish_external(MapHandle::new(0), [(key(1), TypedValue::U64(2))])
            .unwrap();
        assert_eq!(
            store.publish_external(MapHandle::new(0), [(key(1), TypedValue::U64(3))]),
            Err(MapStoreError::Busy(MapHandle::new(0)))
        );
        drop(old_snapshot);
        store
            .publish_external(MapHandle::new(0), [(key(1), TypedValue::U64(3))])
            .unwrap();
    }

    #[test]
    fn ttl_expires_at_commit_relative_time() {
        let (store, clock) = store([policy_map(0, 4, Some(10))]);
        let mut transaction = store.begin().unwrap();
        transaction.stage(upsert(0, 1, 1)).unwrap();
        let prepared = transaction.prepare().unwrap();
        clock.advance(5);
        prepared.commit();
        clock.advance(9);
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(1))
        );
        clock.advance(1);
        assert_eq!(store.read(MapHandle::new(0), &key(1)).unwrap(), None);
    }

    #[test]
    fn expiry_defers_while_key_is_fenced() {
        let (store, clock) = store([policy_map(0, 1, Some(10))]);
        let mut seed = store.begin().unwrap();
        seed.stage(upsert(0, 1, 1)).unwrap();
        seed.prepare().unwrap().commit();

        clock.set(9);
        let mut update = store.begin().unwrap();
        update.get(MapHandle::new(0), &key(1)).unwrap();
        update.stage(upsert(0, 1, 2)).unwrap();
        let prepared = update.prepare().unwrap();

        clock.set(10);
        let mut competing = store.begin().unwrap();
        competing.stage(upsert(0, 2, 3)).unwrap();
        assert!(matches!(competing.prepare(), Err(PrepareError::Conflict)));
        prepared.commit();
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(2))
        );
    }

    #[test]
    fn delete_then_add_restarts_from_zero() {
        let (store, _) = store([policy_map(0, 1, None)]);
        let mut seed = store.begin().unwrap();
        seed.stage(upsert(0, 1, 10)).unwrap();
        seed.prepare().unwrap().commit();

        let mut transaction = store.begin().unwrap();
        transaction
            .stage(MapMutation::Delete {
                map: MapHandle::new(0),
                key: key(1),
            })
            .unwrap();
        transaction
            .stage(MapMutation::AddU64 {
                map: MapHandle::new(0),
                key: key(1),
                delta: 1,
                ttl_ms: None,
            })
            .unwrap();
        transaction.prepare().unwrap().commit();
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(1))
        );
    }

    #[test]
    fn atomic_delete_and_insert_uses_net_capacity() {
        let (store, _) = store([policy_map(0, 1, None)]);
        let mut seed = store.begin().unwrap();
        seed.stage(upsert(0, 1, 1)).unwrap();
        seed.prepare().unwrap().commit();

        let mut transaction = store.begin().unwrap();
        transaction
            .stage(MapMutation::Delete {
                map: MapHandle::new(0),
                key: key(1),
            })
            .unwrap();
        transaction.stage(upsert(0, 2, 2)).unwrap();
        transaction.prepare().unwrap().commit();
        assert_eq!(store.read(MapHandle::new(0), &key(1)).unwrap(), None);
        assert_eq!(
            store.read(MapHandle::new(0), &key(2)).unwrap(),
            Some(TypedValue::U64(2))
        );
    }

    #[test]
    fn commit_remains_infallible_at_clock_limit() {
        let (store, clock) = store([policy_map(0, 1, Some(10))]);
        clock.set(u64::MAX - 1);
        let mut transaction = store.begin().unwrap();
        transaction.stage(upsert(0, 1, 1)).unwrap();
        let result = transaction.prepare().unwrap().commit();
        assert!(result.revision.is_some());
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(1))
        );
        clock.set(u64::MAX);
        assert_eq!(store.read(MapHandle::new(0), &key(1)).unwrap(), None);
    }

    #[test]
    fn feedback_commit_is_deduplicated_with_map_effects() {
        let (store, _) = store([policy_map(0, 4, None)]);
        let delivery = DeliveryId::new([7; 16]);
        let FeedbackStart::New(mut transaction) =
            store.begin_feedback("policy@1", delivery).unwrap()
        else {
            panic!("first delivery must be new");
        };
        transaction.stage(upsert(0, 1, 5)).unwrap();
        let result = transaction.prepare().unwrap().commit();
        assert!(matches!(
            result.feedback,
            Some(FeedbackAcknowledgement::Committed(_))
        ));

        let FeedbackStart::Duplicate(FeedbackAcknowledgement::Duplicate(duplicate)) =
            store.begin_feedback("policy@1", delivery).unwrap()
        else {
            panic!("replay must be deduplicated");
        };
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(5))
        );
        assert_eq!(duplicate.delivery_id, delivery);
    }

    #[test]
    fn full_dedup_ledger_preserves_existing_acknowledgements() {
        let clock = Arc::new(ManualClock::default());
        let store = MapStore::new(
            [policy_map(0, 4, None)],
            clock,
            DedupLimits { max_entries: 1 },
        )
        .unwrap();
        let first = DeliveryId::new([1; 16]);
        let FeedbackStart::New(transaction) = store.begin_feedback("policy@1", first).unwrap()
        else {
            panic!("first delivery must be new");
        };
        transaction.prepare().unwrap().commit();

        assert!(matches!(
            store.begin_feedback("policy@1", DeliveryId::new([2; 16])),
            Err(MapStoreError::DedupCapacity { maximum: 1 })
        ));
        assert!(matches!(
            store.begin_feedback("policy@1", first).unwrap(),
            FeedbackStart::Duplicate(FeedbackAcknowledgement::Duplicate(_))
        ));
    }

    #[test]
    fn concurrent_feedback_prepares_reserve_dedup_capacity() {
        let clock = Arc::new(ManualClock::default());
        let store = MapStore::new(
            [policy_map(0, 4, None)],
            clock,
            DedupLimits { max_entries: 1 },
        )
        .unwrap();
        let FeedbackStart::New(first) = store
            .begin_feedback("policy@1", DeliveryId::new([1; 16]))
            .unwrap()
        else {
            panic!("first delivery must be new");
        };
        let FeedbackStart::New(second) = store
            .begin_feedback("policy@1", DeliveryId::new([2; 16]))
            .unwrap()
        else {
            panic!("second delivery begins before either reserves");
        };

        let prepared = first.prepare().unwrap();
        assert!(matches!(
            second.prepare(),
            Err(PrepareError::Store(MapStoreError::DedupCapacity {
                maximum: 1
            }))
        ));
        prepared.commit();
    }

    #[test]
    fn feedback_policy_identity_is_bounded() {
        let (store, _) = store([policy_map(0, 4, None)]);
        assert!(matches!(
            store.begin_feedback(
                "x".repeat(MAX_POLICY_IDENTITY_BYTES + 1),
                DeliveryId::new([1; 16])
            ),
            Err(MapStoreError::PolicyIdentityTooLong { .. })
        ));
    }

    #[test]
    fn numeric_adds_apply_in_staged_order_and_detect_overflow() {
        let (store, _) = store([policy_map(0, 4, None)]);
        let mut transaction = store.begin().unwrap();
        transaction
            .stage(MapMutation::AddU64 {
                map: MapHandle::new(0),
                key: key(1),
                delta: 2,
                ttl_ms: None,
            })
            .unwrap();
        transaction
            .stage(MapMutation::AddU64 {
                map: MapHandle::new(0),
                key: key(1),
                delta: 3,
                ttl_ms: None,
            })
            .unwrap();
        transaction.prepare().unwrap().commit();
        assert_eq!(
            store.read(MapHandle::new(0), &key(1)).unwrap(),
            Some(TypedValue::U64(5))
        );

        let mut overflow = store.begin().unwrap();
        overflow
            .stage(MapMutation::AddU64 {
                map: MapHandle::new(0),
                key: key(1),
                delta: u64::MAX,
                ttl_ms: None,
            })
            .unwrap();
        assert!(matches!(
            overflow.prepare(),
            Err(PrepareError::NumericOverflow)
        ));
    }

    #[test]
    fn external_iterator_can_reenter_store_without_deadlock() {
        let (store, _) = store([external_map(0)]);
        let reader = store.clone();
        let entries = std::iter::once_with(move || {
            assert_eq!(reader.read(MapHandle::new(0), &key(9)).unwrap(), None);
            (key(1), TypedValue::U64(1))
        });
        store.publish_external(MapHandle::new(0), entries).unwrap();
    }
}
