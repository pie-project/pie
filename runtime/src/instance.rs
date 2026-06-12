//! Instance state for WASM component execution.
//!
//! Per-instance runtime state attached to every wasmtime `Store`: WASI
//! context, filesystem/Python preopens, and dynamic-linking resource maps.

mod output;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmtime::component::{ResourceAny, ResourceTable};
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

use self::output::LogStream;

use crate::context;
use crate::linker::InstancePolicy;
use crate::metadata_store::MetadataOwner;
use crate::process::ProcessId;

/// Where an instance's stdout/stderr are routed.
pub enum OutputMode {
    /// Discard outputs (wasmtime's default sink). Used for snapshot init,
    /// where guest output is noise.
    Discard,
    /// Route to the per-process actor channel, drained by an attached client.
    Process,
    /// Route to pie-server's `tracing` log, tagged with `program`. Used by the
    /// daemon request path, which serves HTTP directly and has no client to
    /// attach — without this, guest stderr falls through to the default sink.
    Server { program: String },
}

pub struct InstanceState {
    // Wasm states
    id: ProcessId,
    username: String,
    program: String,

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
        // Unregister the process: destroy all contexts and remove process entries.
        context::unregister_process(self.id);
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
    pub async fn new(
        id: ProcessId,
        username: String,
        program: String,
        output: OutputMode,
        policy: &InstancePolicy,
        token_budget: Option<usize>,
        py_runtime_dir: Option<&Path>,
    ) -> anyhow::Result<Self> {
        // Register the process with all model context managers. Fails if the
        // admission gate (Σ endowment ≤ capacity × overbook) refuses.
        // Partial failures are rolled back inside register_process, so Drop
        // does not need to care about double-unregistration.
        context::register_process(id, token_budget).await?;

        let mut builder = WasiCtx::builder();

        // Network capability. `inherit_network` exposes the host network;
        // `socket_addr_check` filters per-connect/per-bind. Skipping
        // `inherit_network` denies all socket operations entirely.
        if policy.network.allow {
            builder.inherit_network();
            if !policy.network.is_unrestricted() {
                let net = policy.network.clone();
                builder.socket_addr_check(move |addr, _use| {
                    let ok = net.check(&addr);
                    Box::pin(async move { ok })
                });
            }
        }

        match output {
            OutputMode::Discard => {}
            OutputMode::Process => {
                builder.stdout(LogStream::new_stdout(id));
                builder.stderr(LogStream::new_stderr(id));
            }
            OutputMode::Server { program } => {
                let program: Arc<str> = Arc::from(program);
                builder.stdout(LogStream::new_server_stdout(program.clone()));
                builder.stderr(LogStream::new_server_stderr(program));
            }
        }

        let scratch_dir = policy.fs.base_dir.join(id.to_string());

        if policy.fs.allow {
            std::fs::create_dir_all(&scratch_dir)
                .expect("failed to create scratch dir");

            builder.preopened_dir(
                &scratch_dir,
                "/scratch",
                DirPerms::all(),
                FilePerms::all(),
            ).expect("failed to preopen scratch dir");
        }

        // Set up Python runtime environment if py-runtime directory is available.
        // Layout: py-runtime/runtime/{python,bundled}, py-runtime/site-packages
        if let Some(dir) = py_runtime_dir {
            let runtime_dir = dir.join("runtime");
            let site_packages_dir = dir.join("site-packages");

            const PYTHON_PATH: &str = "/python:/0:/bundled";

            builder
                .env("PYTHONHOME", "/python")
                .env("PYTHONPATH", PYTHON_PATH)
                .env("PYTHONUNBUFFERED", "1");

            builder
                .preopened_dir(
                    runtime_dir.join("python"),
                    "python",
                    DirPerms::READ,
                    FilePerms::READ,
                )
                .expect("failed to preopen python dir");

            builder
                .preopened_dir(
                    runtime_dir.join("bundled"),
                    "bundled",
                    DirPerms::READ,
                    FilePerms::READ,
                )
                .expect("failed to preopen bundled dir");

            builder
                .preopened_dir(site_packages_dir, "0", DirPerms::READ, FilePerms::READ)
                .expect("failed to preopen site-packages dir");
        }

        Ok(InstanceState {
            id,
            username,
            program,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            scratch_dir,
            // Dynamic linking support
            dynamic_resource_map: HashMap::new(),
            guest_resource_map: Vec::new(),
            next_dynamic_rep: 1,
        })
    }

    pub fn id(&self) -> ProcessId {
        self.id
    }

    pub fn get_username(&self) -> String {
        self.username.clone()
    }

    pub fn metadata_owner(&self) -> anyhow::Result<MetadataOwner> {
        MetadataOwner::new(self.username.clone(), self.program.clone())
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
