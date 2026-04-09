use crate::instance::InstanceId;
use crate::runtime::{self, TerminationCause};
use crate::service::ServiceCommand;
use crate::telemetry;
use crate::utils::IdPool;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::time::Instant;
use thiserror::Error;
pub type ResourceId = u32;
pub type ResourceTypeId = u32;
pub type GroupId = usize;

// ---- Reference-counted resource pool ----

/// A resource pool that tracks per-page reference counts.
///
/// Pages are free when ref_count == 0 (in the inner IdPool's free list).
/// Pages are in use when ref_count > 0 (held by allocated instances and/or exports).
/// A page returns to the free list only when its ref count reaches zero.
#[derive(Debug)]
struct RefCountedPool {
    inner: IdPool<u32>,
    /// ref_count for each page. Only pages with ref_count > 0 are tracked.
    ref_counts: HashMap<u32, u32>,
}

impl RefCountedPool {
    fn new(capacity: u32) -> Self {
        Self {
            inner: IdPool::new(capacity),
            ref_counts: HashMap::new(),
        }
    }

    fn set_capacity(&mut self, capacity: u32) -> anyhow::Result<()> {
        self.inner.set_capacity(capacity)
    }

    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    fn available(&self) -> usize {
        self.inner.available()
    }

    /// Acquire a single page from the free list, setting ref_count = 1.
    fn acquire(&mut self) -> anyhow::Result<ResourceId> {
        let id = self.inner.acquire()?;
        self.ref_counts.insert(id, 1);
        Ok(id)
    }

    /// Acquire multiple pages, each with ref_count = 1.
    fn acquire_many(&mut self, count: usize) -> anyhow::Result<Vec<ResourceId>> {
        let ids = self.inner.acquire_many(count)?;
        for &id in &ids {
            self.ref_counts.insert(id, 1);
        }
        Ok(ids)
    }

    /// Increment the reference count for a page.
    /// Panics if the page is not currently tracked (ref_count == 0).
    fn inc_ref(&mut self, id: ResourceId) {
        let count = self
            .ref_counts
            .get_mut(&id)
            .expect("inc_ref on page with ref_count == 0");
        *count += 1;
    }

    /// Decrement the reference count for a page.
    /// Returns `true` if the page was freed (ref_count reached 0).
    /// Panics if the page is not currently tracked.
    fn dec_ref(&mut self, id: ResourceId) -> bool {
        let count = self
            .ref_counts
            .get_mut(&id)
            .expect("dec_ref on page with ref_count == 0");
        *count -= 1;
        if *count == 0 {
            self.ref_counts.remove(&id);
            self.inner.release(id).expect("failed to release page back to pool");
            true
        } else {
            false
        }
    }

    /// Get the current reference count for a page. Returns 0 if not tracked.
    fn ref_count(&self, id: ResourceId) -> u32 {
        self.ref_counts.get(&id).copied().unwrap_or(0)
    }
}

pub static KV_PAGE_TYPE_ID: ResourceTypeId = 0;
pub static EMBED_TYPE_ID: ResourceTypeId = 1;
pub static ADAPTER_TYPE_ID: ResourceTypeId = 2;

// ---- Custom ResourceError enum ----
#[derive(Debug, Error)]
pub enum ResourceError {
    #[error("Resource pool for type {type_id:?} in group {group_id:?} does not exist")]
    PoolNotFound {
        type_id: ResourceTypeId,
        group_id: GroupId,
    },

    #[error("Out of memory for resource type {type_id:?} in group {group_id:?}")]
    OutOfMemory {
        type_id: ResourceTypeId,
        group_id: GroupId,
    },

    #[error("Instance {inst_id:?} has no allocated resources of type {type_id:?}")]
    InstanceNotAllocated {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
    },

    #[error("Instance {inst_id:?} is not assigned to any device group")]
    InstanceGroupNotFound { inst_id: InstanceId },

    #[error("Pointer {ptr:?} is not allocated to instance {inst_id:?}")]
    PointerNotAllocated {
        ptr: ResourceId,
        inst_id: InstanceId,
    },

    #[error("Exported resource with name '{name}' already exists")]
    ExportNameExists { name: String },

    #[error("Exported resource with name '{name}' not found")]
    ExportNotFound { name: String },

    #[error("OOM unrecoverable: {0}")]
    OomUnrecoverable(String),

    #[error("IdPool error: {0:?}")]
    IdPoolError(String),
}

/// Manages the state of all resources, instances, and exports across multiple device groups.
#[derive(Debug)]
pub struct ResourceManager {
    /// Pools are sharded by GroupId. Uses ref counting for safe shared ownership.
    res_pool: HashMap<(GroupId, ResourceTypeId), RefCountedPool>,
    /// Exports are global (name-based) but point to resources in a specific group.
    /// Value is (GroupId, Vec<ResourceId>)
    res_exported: HashMap<ResourceTypeId, HashMap<String, (GroupId, Vec<ResourceId>)>>,
    /// Allocated resources per instance.
    res_allocated: HashMap<(ResourceTypeId, InstanceId), HashSet<ResourceId>>,
    /// Map instance to its assigned device group.
    instance_groups: HashMap<InstanceId, GroupId>,
    inst_start_time: HashMap<InstanceId, Instant>,
    /// Round-robin counter for distributing instances across groups
    next_group_rr: std::cell::Cell<usize>,
    /// Total number of groups for round-robin
    num_groups: usize,
}

impl ResourceManager {
    pub fn new(resources: HashMap<ResourceTypeId, u32>, num_groups: usize) -> Self {
        let mut res_pool = HashMap::new();
        // Create independent pools for each group
        for group_id in 0..num_groups {
            for (res_id, capacity) in &resources {
                res_pool.insert((group_id, *res_id), RefCountedPool::new(*capacity));
            }
        }

        Self {
            res_pool,
            res_exported: HashMap::new(),
            res_allocated: HashMap::new(),
            instance_groups: HashMap::new(),
            inst_start_time: HashMap::new(),
            next_group_rr: std::cell::Cell::new(0),
            num_groups,
        }
    }

    /// Assign an instance to a specific device group.
    /// Must be called before allocating resources for the instance.
    pub fn assign_group(&mut self, inst_id: InstanceId, group_id: GroupId) {
        self.instance_groups.insert(inst_id, group_id);
    }

    pub fn get_group(&self, inst_id: &InstanceId) -> Option<GroupId> {
        self.instance_groups.get(inst_id).copied()
    }

    /// A new combined allocation method that handles the OOM logic internally.
    pub fn allocate_with_oom(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
    ) -> Result<Vec<ResourceId>, ResourceError> {
        // Round-robin assignment across groups if not already assigned
        let group_id = *self
            .instance_groups
            .entry(inst_id)
            .or_insert_with(|| {
                let group = self.next_group_rr.get();
                self.next_group_rr.set((group + 1) % self.num_groups);
                group
            });

        let available = self.available(group_id, type_id)?;

        if available < count {
            tracing::debug!(
                target: "resource.oom",
                group_id = group_id,
                type_id = type_id,
                requested = count,
                available = available,
                "OOM triggered, starting victim selection"
            );
            // Not enough memory, trigger the OOM killer.
            let terminated = self.oom_kill(group_id, type_id, count, inst_id)?;
            // Dispatch termination notifications for killed instances
            for (victim_id, tier) in &terminated {
                if let Some(m) = telemetry::metrics() {
                    m.kv_pages_oom_kills.add(1, &[]);
                }
                let msg = match tier {
                    1 => "Terminated by OOM killer (tier 1: non-shared pages)",
                    2 => "Terminated by OOM killer (tier 2: shared export cascade)",
                    _ => "Terminated by OOM killer",
                };
                runtime::Command::TerminateInstance {
                    inst_id: *victim_id,
                    notification_to_client: Some(TerminationCause::OutOfResources(msg.to_string())),
                }
                .dispatch();
            }
        }

        // A successful oom_kill guarantees enough space.
        let result = self.allocate(inst_id, type_id, count)?;

        // Log allocation metrics
        let new_available = self.available(group_id, type_id).unwrap_or(0);
        tracing::trace!(
            target: "resource.metrics",
            group_id = group_id,
            type_id = type_id,
            allocated = count,
            available_after = new_available,
            inst_id = ?inst_id,
            "Resource allocation"
        );

        // Record OTel metrics for KV pages (type_id 0)
        if type_id == KV_PAGE_TYPE_ID {
            if let Some(m) = telemetry::metrics() {
                let pool = self.res_pool.get(&(group_id, type_id));
                if let Some(pool) = pool {
                    let capacity = pool.capacity() as usize;
                    let available = pool.available();
                    // We might want to tag metrics with group_id in the future
                    m.kv_pages_allocated.record((capacity - available) as u64, &[]);
                    m.kv_pages_available.record(available as u64, &[]);
                }
            }
        }

        Ok(result)
    }

    fn available(&self, group_id: GroupId, type_id: ResourceTypeId) -> Result<usize, ResourceError> {
        let pool = self
            .res_pool
            .get(&(group_id, type_id))
            .ok_or(ResourceError::PoolNotFound { type_id, group_id })?;
        Ok(pool.available())
    }

    /// Get the number of KV pages allocated to a specific instance.
    pub fn get_kv_pages_count(&self, inst_id: InstanceId) -> u32 {
        self.res_allocated
            .get(&(KV_PAGE_TYPE_ID, inst_id))
            .map(|set| set.len() as u32)
            .unwrap_or(0)
    }


    fn allocate(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
    ) -> Result<Vec<ResourceId>, ResourceError> {
        let group_id = self
            .instance_groups
            .get(&inst_id)
            .copied()
            .ok_or(ResourceError::InstanceGroupNotFound { inst_id })?;

        let pool = self
            .res_pool
            .get_mut(&(group_id, type_id))
            .ok_or(ResourceError::PoolNotFound { type_id, group_id })?;

        if pool.available() < count {
            return Err(ResourceError::OutOfMemory { type_id, group_id });
        }

        let allocated = pool.acquire_many(count).unwrap();
        self.inst_start_time
            .entry(inst_id)
            .or_insert_with(Instant::now);
        self.res_allocated
            .entry((type_id, inst_id))
            .or_default()
            .extend(&allocated);

        Ok(allocated)
    }

    pub fn deallocate(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
    ) -> Result<(), ResourceError> {
        let group_id = self
            .instance_groups
            .get(&inst_id)
            .copied()
            .ok_or(ResourceError::InstanceGroupNotFound { inst_id })?;

        let allocated = self
            .res_allocated
            .get_mut(&(type_id, inst_id))
            .ok_or(ResourceError::InstanceNotAllocated { inst_id, type_id })?;

        let pool = self
            .res_pool
            .get_mut(&(group_id, type_id))
            .ok_or(ResourceError::PoolNotFound { type_id, group_id })?;

        for ptr in ptrs {
            if allocated.remove(&ptr) {
                pool.dec_ref(ptr);
            }
        }

        // Prune empty entries to prevent unbounded HashMap growth
        // from per-request WASM instances that allocate then free all pages.
        if allocated.is_empty() {
            self.res_allocated.remove(&(type_id, inst_id));
            // If this instance has no more allocations of any type, clean up tracking
            let has_any = self.res_allocated.keys().any(|(_, id)| *id == inst_id);
            if !has_any {
                self.instance_groups.remove(&inst_id);
                self.inst_start_time.remove(&inst_id);
            }
        }

        Ok(())
    }

    /// OOM killer: reclaim pages by terminating instances and releasing exports.
    /// Returns list of (victim_id, tier) pairs for the caller to dispatch terminations.
    /// Tier 1: non-shared pages. Tier 2: exports + cascade.
    pub(crate) fn oom_kill(
        &mut self,
        group_id: GroupId,
        type_id: ResourceTypeId,
        size: usize,
        inst_id_to_exclude: InstanceId,
    ) -> Result<Vec<(InstanceId, u8)>, ResourceError> {
        let requester_start_time = self
            .inst_start_time
            .get(&inst_id_to_exclude)
            .copied()
            .ok_or_else(|| {
                ResourceError::OomUnrecoverable(
                    "Requesting instance has no start time.".to_string(),
                )
            })?;

        let mut terminated = Vec::new();

        // Tier 1: Kill instances whose pages are all ref_count == 1 (not shared
        // with any export). These pages free immediately on cleanup.
        loop {
            if self.available(group_id, type_id)? >= size {
                return Ok(terminated);
            }

            let victim_id = self.find_non_shared_victim(
                group_id, type_id, inst_id_to_exclude, requester_start_time,
            );

            if let Some(victim_id) = victim_id {
                tracing::warn!(
                    target: "resource.oom",
                    victim_id = ?victim_id,
                    group_id = group_id,
                    type_id = type_id,
                    tier = 1,
                    "OOM killer terminating instance (non-shared pages)"
                );

                self.cleanup(victim_id)?;
                terminated.push((victim_id, 1));
            } else {
                break; // No more non-shared victims
            }
        }

        // Tier 2: Kill exports and their importing instances to reclaim shared pages.
        loop {
            if self.available(group_id, type_id)? >= size {
                return Ok(terminated);
            }

            let export_info = self.find_largest_export(group_id, type_id);

            if let Some((export_name, export_pages)) = export_info {
                tracing::warn!(
                    target: "resource.oom",
                    export_name = %export_name,
                    group_id = group_id,
                    type_id = type_id,
                    page_count = export_pages.len(),
                    tier = 2,
                    "OOM killer releasing export and terminating holders"
                );

                // Find all instances holding refs to these pages
                let holders = self.find_page_holders(type_id, &export_pages, inst_id_to_exclude);

                // Terminate all holders first (dec_ref their allocated pages)
                for holder_id in &holders {
                    self.cleanup(*holder_id)?;
                    terminated.push((*holder_id, 2));
                }

                // Release the export (dec_ref remaining refs)
                self.release_exported(type_id, export_name)?;
            } else {
                // Last resort: kill any remaining instance in the group
                let victim_id = self
                    .inst_start_time
                    .iter()
                    .filter(|(id, time)| {
                        **id != inst_id_to_exclude
                            && self.instance_groups.get(id) == Some(&group_id)
                            && **time > requester_start_time
                    })
                    .max_by_key(|(_, time)| **time)
                    .map(|(id, _)| *id);

                if let Some(victim_id) = victim_id {
                    self.cleanup(victim_id)?;
                    terminated.push((victim_id, 3));
                } else {
                    return Err(ResourceError::OomUnrecoverable(
                        "Not enough memory after terminating all instances and exports.".to_string(),
                    ));
                }
            }
        }
    }

    /// Find a victim instance whose pages ALL have ref_count == 1 (not shared).
    /// Among candidates, pick the newest (most recently started) one.
    fn find_non_shared_victim(
        &self,
        group_id: GroupId,
        type_id: ResourceTypeId,
        exclude: InstanceId,
        requester_start_time: Instant,
    ) -> Option<InstanceId> {
        let pool = self.res_pool.get(&(group_id, type_id))?;

        self.inst_start_time
            .iter()
            .filter(|(id, time)| {
                **id != exclude
                    && self.instance_groups.get(id) == Some(&group_id)
                    && **time > requester_start_time
            })
            .filter(|(id, _)| {
                // Check if ALL pages for this instance have ref_count == 1
                if let Some(pages) = self.res_allocated.get(&(type_id, **id)) {
                    !pages.is_empty() && pages.iter().all(|ptr| pool.ref_count(*ptr) == 1)
                } else {
                    false // No pages allocated, not useful as a victim
                }
            })
            .max_by_key(|(_, time)| **time)
            .map(|(id, _)| *id)
    }

    /// Find the largest export in a given group, returning (name, pages).
    fn find_largest_export(
        &self,
        group_id: GroupId,
        type_id: ResourceTypeId,
    ) -> Option<(String, Vec<ResourceId>)> {
        self.res_exported
            .get(&type_id)?
            .iter()
            .filter(|(_, (gid, _))| *gid == group_id)
            .max_by_key(|(_, (_, pages))| pages.len())
            .map(|(name, (_, pages))| (name.clone(), pages.clone()))
    }

    /// Find all instances that have any of the given pages in their allocated set.
    fn find_page_holders(
        &self,
        type_id: ResourceTypeId,
        pages: &[ResourceId],
        exclude: InstanceId,
    ) -> Vec<InstanceId> {
        let page_set: HashSet<&ResourceId> = pages.iter().collect();
        self.res_allocated
            .iter()
            .filter(|((ty, inst_id), allocated)| {
                *ty == type_id
                    && *inst_id != exclude
                    && allocated.iter().any(|ptr| page_set.contains(ptr))
            })
            .map(|((_, inst_id), _)| *inst_id)
            .collect()
    }

    pub fn cleanup(&mut self, inst_id: InstanceId) -> Result<(), ResourceError> {
        let _ = self.cleanup_with_freed_kv(inst_id)?;
        Ok(())
    }

    /// Like cleanup() but returns the KV page IDs that were actually freed
    /// (ref_count reached 0). Pages still referenced by exports are NOT included.
    pub fn cleanup_with_freed_kv(&mut self, inst_id: InstanceId) -> Result<Vec<ResourceId>, ResourceError> {
        // If instance was never assigned a group, just clean up start time and return
        let group_id = match self.instance_groups.get(&inst_id) {
            Some(g) => *g,
            None => {
                self.inst_start_time.remove(&inst_id);
                return Ok(Vec::new());
            }
        };

        let mut to_release_by_type: HashMap<ResourceTypeId, Vec<ResourceId>> = HashMap::new();
        self.res_allocated.retain(|(ty, id), ptrs| {
            if *id == inst_id {
                to_release_by_type
                    .entry(*ty)
                    .or_default()
                    .extend(ptrs.iter());
                false
            } else {
                true
            }
        });

        // Dec-ref each page; only collect KV pages that actually freed (ref reached 0)
        let mut freed_kv = Vec::new();
        for (ty, ptrs) in to_release_by_type {
            let pool = self
                .res_pool
                .get_mut(&(group_id, ty))
                .ok_or(ResourceError::PoolNotFound { type_id: ty, group_id })?;
            for ptr in ptrs {
                let actually_freed = pool.dec_ref(ptr);
                if ty == KV_PAGE_TYPE_ID && actually_freed {
                    freed_kv.push(ptr);
                }
            }
        }
        self.inst_start_time.remove(&inst_id);
        self.instance_groups.remove(&inst_id);
        Ok(freed_kv)
    }

    // --- export, import, release_exported, and get_all_exported methods ---
    // These are moved here from Model with minimal changes, now returning ResourceError
    pub fn export(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        name: String,
    ) -> Result<(), ResourceError> {
        let group_id = self
            .instance_groups
            .get(&inst_id)
            .copied()
            .ok_or(ResourceError::InstanceGroupNotFound { inst_id })?;

        let allocated = self
            .res_allocated
            .get_mut(&(type_id, inst_id))
            .ok_or(ResourceError::InstanceNotAllocated { inst_id, type_id })?;

        for ptr in &ptrs {
            if !allocated.contains(ptr) {
                return Err(ResourceError::PointerNotAllocated { ptr: *ptr, inst_id });
            }
        }

        let type_exports = self.res_exported.entry(type_id).or_default();
        match type_exports.entry(name) {
            Entry::Occupied(entry) => Err(ResourceError::ExportNameExists {
                name: entry.key().clone(),
            }),
            Entry::Vacant(entry) => {
                ptrs.iter().for_each(|ptr| {
                    allocated.remove(ptr);
                });
                entry.insert((group_id, ptrs));
                Ok(())
            }
        }
    }

    pub fn import(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> Result<(GroupId, Vec<ResourceId>), ResourceError> {
        // Non-consuming import: the export entry stays so other instances
        // can also import the same resource (read-only shared).  Pages are
        // freed only when the export is explicitly released (e.g. session
        // eviction) or when the resource pool needs to reclaim memory.
        self.res_exported
            .get(&type_id)
            .and_then(|exports| exports.get(&name))
            .cloned()
            .ok_or(ResourceError::ExportNotFound { name })
    }

    /// Register imported pages in the importing instance's allocation table.
    /// Increments ref_count for each page (+1, since the export already holds a ref).
    /// Also assigns the instance to the exporter's resource group.
    pub fn register_imported(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        group_id: GroupId,
        ptrs: &[ResourceId],
    ) {
        self.instance_groups.entry(inst_id).or_insert(group_id);

        let pool = self
            .res_pool
            .get_mut(&(group_id, type_id))
            .expect("pool must exist for register_imported");

        let allocated = self
            .res_allocated
            .entry((type_id, inst_id))
            .or_default();
        for &ptr in ptrs {
            if allocated.insert(ptr) {
                // Only inc_ref if we actually added the page (not already present)
                pool.inc_ref(ptr);
            }
        }
    }

    pub fn release_exported(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> Result<(), ResourceError> {
        let type_exports = self
            .res_exported
            .get_mut(&type_id)
            .ok_or(ResourceError::PoolNotFound { type_id, group_id: 0 })?;

        if let Some((group_id, ptrs_to_release)) = type_exports.remove(&name) {
            let pool = self
                .res_pool
                .get_mut(&(group_id, type_id))
                .ok_or(ResourceError::PoolNotFound { type_id, group_id })?;
            for ptr in ptrs_to_release {
                pool.dec_ref(ptr);
            }
            Ok(())
        } else {
            Err(ResourceError::ExportNotFound { name })
        }
    }

    pub fn get_all_exported(&self, type_id: ResourceTypeId) -> Vec<(String, Vec<ResourceId>)> {
        self.res_exported
            .get(&type_id)
            .map(|exports| {
                exports
                    .iter()
                    .map(|(name, (_, ptrs))| (name.clone(), ptrs.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Appends detailed statistics about the resource manager's state to a given HashMap.
    pub fn append_stats_to(&self, stats: &mut HashMap<String, String>) {
        // Report on each resource pool
        for ((group_id, type_id), pool) in &self.res_pool {
            let capacity = pool.capacity() as usize;
            let available = pool.available();
            let used = capacity - available;

            stats.insert(
                format!("resource.g{}.{}.capacity", group_id, type_id),
                capacity.to_string(),
            );
            stats.insert(
                format!("resource.g{}.{}.available", group_id, type_id),
                available.to_string(),
            );
            stats.insert(format!("resource.g{}.{}.used", group_id, type_id), used.to_string());
        }

        // Report on active instances
        stats.insert(
            "instances.active_count".to_string(),
            self.inst_start_time.len().to_string(),
        );

        // Report per-instance KV pages
        for ((type_id, inst_id), ptrs) in &self.res_allocated {
            if *type_id == KV_PAGE_TYPE_ID {
                stats.insert(
                    format!("instance.{:?}.kv_pages", inst_id),
                    ptrs.len().to_string(),
                );
            }
        }

        // Report on exported resources

        for (type_id, exports) in &self.res_exported {
            stats.insert(
                format!("resource.{}.exported_count", type_id),
                exports.len().to_string(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    const TY: ResourceTypeId = KV_PAGE_TYPE_ID;

    fn new_inst() -> InstanceId {
        Uuid::new_v4()
    }

    fn make_mgr(capacity: u32) -> ResourceManager {
        let mut resources = HashMap::new();
        resources.insert(TY, capacity);
        ResourceManager::new(resources, 1)
    }

    // ---- RefCountedPool unit tests ----

    #[test]
    fn pool_acquire_sets_ref_to_one() {
        let mut pool = RefCountedPool::new(10);
        let id = pool.acquire().unwrap();
        assert_eq!(pool.ref_count(id), 1);
        assert_eq!(pool.available(), 9);
    }

    #[test]
    fn pool_dec_ref_frees_at_zero() {
        let mut pool = RefCountedPool::new(10);
        let id = pool.acquire().unwrap();
        assert!(pool.dec_ref(id)); // ref 1 -> 0, freed
        assert_eq!(pool.ref_count(id), 0);
        assert_eq!(pool.available(), 10);
    }

    #[test]
    fn pool_inc_dec_ref_lifecycle() {
        let mut pool = RefCountedPool::new(10);
        let id = pool.acquire().unwrap();
        pool.inc_ref(id); // ref = 2
        assert_eq!(pool.ref_count(id), 2);
        assert!(!pool.dec_ref(id)); // ref = 1, NOT freed
        assert_eq!(pool.ref_count(id), 1);
        assert_eq!(pool.available(), 9); // still in use
        assert!(pool.dec_ref(id)); // ref = 0, freed
        assert_eq!(pool.available(), 10);
    }

    #[test]
    #[should_panic(expected = "inc_ref on page with ref_count == 0")]
    fn pool_inc_ref_panics_on_free_page() {
        let mut pool = RefCountedPool::new(10);
        pool.inc_ref(0); // page 0 was never acquired
    }

    #[test]
    #[should_panic(expected = "dec_ref on page with ref_count == 0")]
    fn pool_dec_ref_panics_on_free_page() {
        let mut pool = RefCountedPool::new(10);
        let id = pool.acquire().unwrap();
        pool.dec_ref(id); // ref -> 0, freed
        pool.dec_ref(id); // should panic
    }

    // ---- ResourceManager integration tests ----

    #[test]
    fn allocate_export_import_cleanup_export_survives() {
        let mut mgr = make_mgr(100);
        let inst1 = new_inst();
        let inst2 = new_inst();

        // Turn 1: allocate + export
        let pages = mgr.allocate_with_oom(inst1, TY, 10).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 90);

        mgr.export(inst1, TY, pages.clone(), "session-kv".into()).unwrap();
        // Pages transferred from allocated to exported, ref stays 1
        assert_eq!(mgr.available(0, TY).unwrap(), 90);

        // Turn 1 ends: cleanup inst1 (no pages in allocated, nothing to free)
        mgr.cleanup(inst1).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 90);

        // Turn 2: import
        let (group_id, imported) = mgr.import(TY, "session-kv".into()).unwrap();
        assert_eq!(imported.len(), 10);
        mgr.register_imported(inst2, TY, group_id, &imported);
        // Now ref = 2 for each page (export + allocated)
        assert_eq!(mgr.available(0, TY).unwrap(), 90);

        // Turn 2 crashes: cleanup inst2 without re-export
        let freed = mgr.cleanup_with_freed_kv(inst2).unwrap();
        assert!(freed.is_empty()); // Pages NOT freed — export still holds ref
        assert_eq!(mgr.available(0, TY).unwrap(), 90);

        // Export still valid
        let (_, still_there) = mgr.import(TY, "session-kv".into()).unwrap();
        assert_eq!(still_there.len(), 10);

        // Now release the export — pages freed
        mgr.release_exported(TY, "session-kv".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 100);
    }

    #[test]
    fn allocate_export_import_release_then_cleanup_frees() {
        let mut mgr = make_mgr(100);
        let inst1 = new_inst();
        let inst2 = new_inst();

        let pages = mgr.allocate_with_oom(inst1, TY, 5).unwrap();
        mgr.export(inst1, TY, pages.clone(), "ex1".into()).unwrap();
        mgr.cleanup(inst1).unwrap();

        let (gid, imported) = mgr.import(TY, "ex1".into()).unwrap();
        mgr.register_imported(inst2, TY, gid, &imported);
        // ref = 2

        // Release export first
        mgr.release_exported(TY, "ex1".into()).unwrap();
        // ref = 1 (still in allocated[inst2])
        assert_eq!(mgr.available(0, TY).unwrap(), 95);

        // Now cleanup inst2 — ref reaches 0, pages freed
        let freed = mgr.cleanup_with_freed_kv(inst2).unwrap();
        assert_eq!(freed.len(), 5);
        assert_eq!(mgr.available(0, TY).unwrap(), 100);
    }

    #[test]
    fn re_export_after_import_keeps_correct_refs() {
        let mut mgr = make_mgr(100);
        let inst1 = new_inst();
        let inst2 = new_inst();

        // Turn 1: allocate + export
        let pages1 = mgr.allocate_with_oom(inst1, TY, 10).unwrap();
        mgr.export(inst1, TY, pages1.clone(), "t1".into()).unwrap();
        mgr.cleanup(inst1).unwrap();

        // Turn 2: import + allocate new + export all
        let (gid, imported) = mgr.import(TY, "t1".into()).unwrap();
        mgr.register_imported(inst2, TY, gid, &imported);
        let new_pages = mgr.allocate_with_oom(inst2, TY, 5).unwrap();
        // imported pages ref=2, new pages ref=1

        let mut all_pages = imported.clone();
        all_pages.extend(&new_pages);
        mgr.export(inst2, TY, all_pages, "t2".into()).unwrap();
        // export transfers from allocated → exported (no ref change)
        // imported pages: still ref=2 (t1 export + t2 export)
        // new pages: ref=1 (t2 export only)
        mgr.cleanup(inst2).unwrap(); // allocated is empty

        // Release t1: imported pages ref=2→1
        mgr.release_exported(TY, "t1".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 85); // 15 pages still in t2

        // Release t2: all pages freed
        mgr.release_exported(TY, "t2".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 100);
    }

    #[test]
    fn double_free_impossible() {
        // Same page in both allocated and exported — cleanup + release_exported
        // should free it exactly once.
        let mut mgr = make_mgr(100);
        let inst1 = new_inst();
        let inst2 = new_inst();

        let pages = mgr.allocate_with_oom(inst1, TY, 5).unwrap();
        mgr.export(inst1, TY, pages.clone(), "ex".into()).unwrap();
        mgr.cleanup(inst1).unwrap();

        let (gid, imported) = mgr.import(TY, "ex".into()).unwrap();
        mgr.register_imported(inst2, TY, gid, &imported);
        // ref = 2 for each page

        // Both cleanup and release_exported on the same pages
        mgr.cleanup(inst2).unwrap(); // ref -> 1
        mgr.release_exported(TY, "ex".into()).unwrap(); // ref -> 0
        assert_eq!(mgr.available(0, TY).unwrap(), 100);
    }

    #[test]
    fn deallocate_partial_with_shared_pages() {
        let mut mgr = make_mgr(100);
        let inst1 = new_inst();
        let inst2 = new_inst();

        let pages = mgr.allocate_with_oom(inst1, TY, 10).unwrap();
        mgr.export(inst1, TY, pages[..5].to_vec(), "ex".into()).unwrap();
        mgr.cleanup(inst1).unwrap();

        let (gid, imported) = mgr.import(TY, "ex".into()).unwrap();
        mgr.register_imported(inst2, TY, gid, &imported);
        // imported 5 pages ref=2, inst1's remaining 5 already freed in cleanup

        // Deallocate the imported pages from inst2
        mgr.deallocate(inst2, TY, imported.clone()).unwrap();
        // ref -> 1 (export still holds)
        assert_eq!(mgr.available(0, TY).unwrap(), 95);

        mgr.release_exported(TY, "ex".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 100);
    }

    #[test]
    fn session_kv_multi_turn_lifecycle() {
        // Simulates 3 turns of session KV reuse
        let mut mgr = make_mgr(1000);

        // Turn 1
        let t1 = new_inst();
        let p1 = mgr.allocate_with_oom(t1, TY, 100).unwrap();
        mgr.export(t1, TY, p1, "session".into()).unwrap();
        mgr.cleanup(t1).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 900);

        // Turn 2: import + allocate more + re-export under new name
        let t2 = new_inst();
        let (gid, imp2) = mgr.import(TY, "session".into()).unwrap();
        mgr.register_imported(t2, TY, gid, &imp2);
        let new2 = mgr.allocate_with_oom(t2, TY, 50).unwrap();
        let mut all2 = imp2;
        all2.extend(&new2);
        mgr.export(t2, TY, all2, "session2".into()).unwrap();
        mgr.cleanup(t2).unwrap();
        // Release old export
        mgr.release_exported(TY, "session".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 850);

        // Turn 3: import session2
        let t3 = new_inst();
        let (gid, imp3) = mgr.import(TY, "session2".into()).unwrap();
        assert_eq!(imp3.len(), 150);
        mgr.register_imported(t3, TY, gid, &imp3);
        let new3 = mgr.allocate_with_oom(t3, TY, 30).unwrap();
        let mut all3 = imp3;
        all3.extend(&new3);
        mgr.export(t3, TY, all3, "session3".into()).unwrap();
        mgr.cleanup(t3).unwrap();
        mgr.release_exported(TY, "session2".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 820); // 180 pages in session3

        // Final cleanup
        mgr.release_exported(TY, "session3".into()).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 1000);
    }

    // ---- RefCountedPool basic tests ----

    #[test]
    fn pool_acquire_many() {
        let mut pool = RefCountedPool::new(5);
        let ids = pool.acquire_many(3).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(pool.available(), 2);
        for &id in &ids {
            assert_eq!(pool.ref_count(id), 1);
        }
    }

    #[test]
    fn pool_capacity_and_set_capacity() {
        let mut pool = RefCountedPool::new(10);
        assert_eq!(pool.capacity(), 10);
        pool.set_capacity(20).unwrap();
        assert_eq!(pool.capacity(), 20);
        assert_eq!(pool.available(), 20);
    }

    #[test]
    fn oom_tier1_kills_non_shared_first() {
        let mut mgr = make_mgr(20); // small pool
        let requester = new_inst();
        let old = new_inst();
        let victim = new_inst();

        // requester registers first (oldest — protected by OOM killer)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _p_req = mgr.allocate_with_oom(requester, TY, 1).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 19);

        // old allocates 8 pages (newer than requester but older than victim)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _p_old = mgr.allocate_with_oom(old, TY, 8).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 11);

        // victim allocates 8 pages (newest — first to be killed)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _p_victim = mgr.allocate_with_oom(victim, TY, 8).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 3);

        // requester needs 10 pages — only 3 available, OOM kills victim (newest, non-shared)
        let terminated = mgr.oom_kill(0, TY, 10, requester).unwrap();

        assert_eq!(terminated.len(), 1);
        assert_eq!(terminated[0].0, victim);
        assert_eq!(terminated[0].1, 1); // tier 1
        assert!(mgr.available(0, TY).unwrap() >= 10);
    }

    #[test]
    fn oom_tier2_kills_export_and_holders() {
        let mut mgr = make_mgr(20);
        let exporter = new_inst();
        let importer = new_inst();
        let requester = new_inst();

        // exporter allocates 10 pages, exports them
        std::thread::sleep(std::time::Duration::from_millis(1));
        let pages = mgr.allocate_with_oom(exporter, TY, 10).unwrap();
        mgr.export(exporter, TY, pages.clone(), "session-kv".into()).unwrap();
        mgr.cleanup(exporter).unwrap(); // exporter instance ends
        assert_eq!(mgr.available(0, TY).unwrap(), 10);

        // importer imports the pages (ref goes to 2)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (gid, imported) = mgr.import(TY, "session-kv".into()).unwrap();
        mgr.register_imported(importer, TY, gid, &imported);
        assert_eq!(mgr.available(0, TY).unwrap(), 10);

        // importer also allocates 5 more pages
        let _more = mgr.allocate_with_oom(importer, TY, 5).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 5);

        // requester allocates 1 page first, then needs 15 total — only 4 left
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _p_req = mgr.allocate_with_oom(requester, TY, 1).unwrap(); // 4 remaining
        let terminated = mgr.oom_kill(0, TY, 15, requester).unwrap();

        // importer killed (tier 2), export released
        assert!(terminated.iter().any(|(id, _)| *id == importer));
        assert!(terminated.iter().any(|(_, tier)| *tier == 2));
        assert!(mgr.available(0, TY).unwrap() >= 15);

        // Export should be gone
        assert!(mgr.import(TY, "session-kv".into()).is_err());
    }

    #[test]
    fn oom_tier1_before_tier2() {
        let mut mgr = make_mgr(30);
        let requester = new_inst();
        let exporter = new_inst();
        let importer = new_inst();
        let non_shared = new_inst();

        // requester registers first (oldest — protected)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _p_req = mgr.allocate_with_oom(requester, TY, 1).unwrap();

        // exporter allocates 10, exports
        std::thread::sleep(std::time::Duration::from_millis(1));
        let pages = mgr.allocate_with_oom(exporter, TY, 10).unwrap();
        mgr.export(exporter, TY, pages.clone(), "export-1".into()).unwrap();
        mgr.cleanup(exporter).unwrap();

        // importer imports (ref=2) + allocates 5 own
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (gid, imported) = mgr.import(TY, "export-1".into()).unwrap();
        mgr.register_imported(importer, TY, gid, &imported);
        let _own = mgr.allocate_with_oom(importer, TY, 5).unwrap();
        // available: 30 - 1(req) - 10(exported) - 5(importer own) = 14

        // non_shared allocates 10 (all ref=1, newest — first OOM candidate)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _ns = mgr.allocate_with_oom(non_shared, TY, 10).unwrap();
        assert_eq!(mgr.available(0, TY).unwrap(), 4);

        // requester needs 12 — tier 1 should kill non_shared first (10 pages freed → 14 available)
        let terminated = mgr.oom_kill(0, TY, 12, requester).unwrap();

        // non_shared killed (tier 1, newest), NOT importer (shared pages)
        assert_eq!(terminated.len(), 1);
        assert_eq!(terminated[0].0, non_shared);
        assert_eq!(terminated[0].1, 1); // tier 1, NOT tier 2
        assert!(mgr.available(0, TY).unwrap() >= 12);

        // Export should still exist (tier 2 not needed)
        assert!(mgr.import(TY, "export-1".into()).is_ok());
    }
}
