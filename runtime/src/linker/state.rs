//! Instance state for WASM component execution.
//!
//! This module provides the runtime state for each WASM instance,
//! including WASI context and dynamic linking support.

use std::collections::HashMap;
use std::path::PathBuf;
use wasmtime::component::{ResourceAny, ResourceTable};
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

use super::output::LogStream;

use crate::context::{self, ContextId};
use crate::process::ProcessId;

pub struct InstanceState {
    // Wasm states
    id: ProcessId,
    username: String,

    // WASI states
    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,

    /// Per-instance scratch directory, deleted on Drop.
    scratch_dir: PathBuf,

    // Dynamic linking support for proxy resources
    /// Maps host rep → guest ResourceAny for dynamic linking
    dynamic_resource_map: HashMap<u32, ResourceAny>,
    /// Maps guest ResourceAny → host rep (for identity preservation)
    guest_resource_map: Vec<(ResourceAny, u32)>,
    /// Counter for allocating unique host reps
    next_dynamic_rep: u32,

    /// Anonymous contexts owned by this instance, auto-destroyed on Drop.
    owned_contexts: Vec<(usize, ContextId)>,
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.scratch_dir);
        // Auto-destroy all anonymous contexts owned by this instance
        for (model_idx, context_id) in self.owned_contexts.drain(..) {
            tokio::spawn(async move {
                let _ = context::destroy(model_idx, context_id, 0, true).await;
            });
        }
    }
}

impl WasiView for InstanceState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi_ctx,
            table: &mut self.resource_table,
        }
    }
}

impl WasiHttpView for InstanceState {
    fn ctx(&mut self) -> &mut WasiHttpCtx {
        &mut self.http_ctx
    }

    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
}

impl InstanceState {
    pub fn new(
        id: ProcessId,
        username: String,
        capture_outputs: bool,
        allow_filesystem: bool,
    ) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_network(); // TODO: Replace with socket_addr_check later.

        if capture_outputs {
            builder.stdout(LogStream::new_stdout(id));
            builder.stderr(LogStream::new_stderr(id));
        }

        // Cross-platform temp dir: /tmp on Linux, %TEMP% on Windows, etc.
        let scratch_dir = std::env::temp_dir()
            .join("pie")
            .join(id.to_string());

        if allow_filesystem {
            std::fs::create_dir_all(&scratch_dir)
                .expect("failed to create scratch dir");

            builder.preopened_dir(
                &scratch_dir,
                "/scratch",
                DirPerms::all(),
                FilePerms::all(),
            ).expect("failed to preopen scratch dir");
        }

        InstanceState {
            id,
            username,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            scratch_dir,
            // Dynamic linking support
            dynamic_resource_map: HashMap::new(),
            guest_resource_map: Vec::new(),
            next_dynamic_rep: 1,
            // Context lifecycle tracking
            owned_contexts: Vec::new(),
        }
    }

    pub fn id(&self) -> ProcessId {
        self.id
    }

    pub fn get_username(&self) -> String {
        self.username.clone()
    }

    // ========================================================================
    // Dynamic Linking Support Methods
    // ========================================================================

    /// Allocates a new host rep for dynamic resource mapping.
    pub fn alloc_dynamic_rep(&mut self) -> u32 {
        let rep = self.next_dynamic_rep;
        self.next_dynamic_rep = self.next_dynamic_rep.checked_add(1).unwrap();
        rep
    }

    /// Gets the guest ResourceAny for a given host rep.
    pub fn get_dynamic_resource(&self, rep: u32) -> Option<ResourceAny> {
        self.dynamic_resource_map.get(&rep).copied()
    }

    /// Gets the host rep for a given guest ResourceAny (for identity preservation).
    pub fn rep_for_guest_resource(&self, resource: ResourceAny) -> Option<u32> {
        self.guest_resource_map
            .iter()
            .find(|(r, _)| *r == resource)
            .map(|(_, rep)| *rep)
    }

    /// Inserts a mapping between host rep and guest ResourceAny.
    pub fn insert_dynamic_resource_mapping(&mut self, rep: u32, resource: ResourceAny) {
        self.dynamic_resource_map.insert(rep, resource);
        // Only insert the reverse mapping if not already present
        if self.rep_for_guest_resource(resource).is_none() {
            self.guest_resource_map.push((resource, rep));
        }
    }

    /// Removes the mapping for a host rep and returns the guest ResourceAny.
    pub fn remove_dynamic_resource_mapping(&mut self, rep: u32) -> Option<ResourceAny> {
        if let Some(resource) = self.dynamic_resource_map.remove(&rep) {
            self.guest_resource_map.retain(|(r, _)| *r != resource);
            Some(resource)
        } else {
            None
        }
    }

    // ========================================================================
    // Context Lifecycle Tracking
    // ========================================================================

    /// Track an anonymous context for auto-cleanup on instance drop.
    pub fn track_context(&mut self, model_idx: usize, context_id: ContextId) {
        self.owned_contexts.push((model_idx, context_id));
    }

    /// Stop tracking a context (e.g. after save or explicit destroy).
    /// Returns true if the context was tracked (i.e., was anonymous).
    pub fn untrack_context(&mut self, model_idx: usize, context_id: ContextId) -> bool {
        let before = self.owned_contexts.len();
        self.owned_contexts.retain(|&(m, c)| !(m == model_idx && c == context_id));
        self.owned_contexts.len() < before
    }
}
