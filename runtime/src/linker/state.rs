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
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.scratch_dir);
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
}
