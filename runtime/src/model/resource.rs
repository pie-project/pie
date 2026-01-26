use crate::instance::InstanceId;
use crate::runtime::{self, TerminationCause};
use crate::actor::ServiceCommand;
use crate::telemetry;
use crate::utils::IdPool;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::time::Instant;
use thiserror::Error;
pub type ResourceId = u32;
pub type ResourceTypeId = u32;
pub type GroupId = usize;

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
    /// Pools are sharded by GroupId.
    res_pool: HashMap<(GroupId, ResourceTypeId), IdPool<u32>>,
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
                res_pool.insert((group_id, *res_id), IdPool::new(*capacity));
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
                // eprintln!("[DEBUG] Round-robin assigning instance {:?} to group {}", inst_id, group);
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
            self.oom_kill(group_id, type_id, count, inst_id)?;
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
                pool.release(ptr).unwrap();
            }
        }

        Ok(())
    }

    fn oom_kill(
        &mut self,
        group_id: GroupId,
        type_id: ResourceTypeId,
        size: usize,
        inst_id_to_exclude: InstanceId,
    ) -> Result<(), ResourceError> {
        let requester_start_time = self
            .inst_start_time
            .get(&inst_id_to_exclude)
            .copied()
            .ok_or_else(|| {
                ResourceError::OomUnrecoverable(
                    "Requesting instance has no start time.".to_string(),
                )
            })?;

        loop {
            if self.available(group_id, type_id)? >= size {
                break;
            }

            // Find victim ONLY in the same group
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
                tracing::warn!(
                    target: "resource.oom",
                    victim_id = ?victim_id,
                    group_id = group_id,
                    type_id = type_id,
                    "OOM killer terminating instance"
                );

                // Record OOM kill metric
                if let Some(m) = telemetry::metrics() {
                    m.kv_pages_oom_kills.add(1, &[]);
                }

                self.cleanup(victim_id)?;
                runtime::Command::TerminateInstance {
                    inst_id: victim_id,
                    notification_to_client: Some(TerminationCause::OutOfResources(
                        "Terminated by OOM killer for an older instance".to_string(),
                    )),
                }
                .dispatch();
            } else {
                return Err(ResourceError::OomUnrecoverable(
                    "Not enough memory after terminating all newer instances.".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn cleanup(&mut self, inst_id: InstanceId) -> Result<(), ResourceError> {
        // If instance was never assigned a group, just clean up start time and return
        let group_id = match self.instance_groups.get(&inst_id) {
            Some(g) => *g,
            None => {
                self.inst_start_time.remove(&inst_id);
                return Ok(());
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

        for (ty, ptrs) in to_release_by_type {
            let pool = self
                .res_pool
                .get_mut(&(group_id, ty))
                .ok_or(ResourceError::PoolNotFound { type_id: ty, group_id })?;
            for ptr in ptrs {
                pool.release(ptr).unwrap();
            }
        }
        self.inst_start_time.remove(&inst_id);
        self.instance_groups.remove(&inst_id);
        Ok(())
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
    ) -> Result<Vec<ResourceId>, ResourceError> {
        self.res_exported
            .get(&type_id)
            .and_then(|exports| exports.get(&name))
            .map(|(_, ptrs)| ptrs.clone())
            .ok_or(ResourceError::ExportNotFound { name })
    }

    /// Import with group info (useful for ensuring we don't mix resources across groups,
    /// though currently exports are global and transfer ownership...)
    /// Actually, if we import, we might need to know which group it belongs to so the
    /// importing instance can be verified to be in the same group?
    /// For now, we assume exports act as a bridge or are only valid within same group.
    /// TODO: Enforce group safety if needed.
    pub fn import_with_group(
        &self,
        type_id: ResourceTypeId,
        name: &str,
    ) -> Result<(GroupId, Vec<ResourceId>), ResourceError> {
        self.res_exported
            .get(&type_id)
            .and_then(|exports| exports.get(name))
            .cloned()
            .ok_or(ResourceError::ExportNotFound { name: name.to_string() })
    }

    pub fn release_exported(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> Result<(), ResourceError> {
        let type_exports = self
            .res_exported
            .get_mut(&type_id)
            .ok_or(ResourceError::PoolNotFound { type_id, group_id: 0 })?; // Error type slightly misused here, but ok

        if let Some((group_id, ptrs_to_release)) = type_exports.remove(&name) {
            let pool = self
                .res_pool
                .get_mut(&(group_id, type_id))
                .ok_or(ResourceError::PoolNotFound { type_id, group_id })?;
            for ptr in ptrs_to_release {
                pool.release(ptr).unwrap();
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
