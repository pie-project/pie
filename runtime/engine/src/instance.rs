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
use wasmtime_wasi_http::WasiHttpCtx;
use wasmtime_wasi_http::p2::{WasiHttpCtxView, WasiHttpView};
use wasmtime_wasi_http::p3::{
    WasiHttpCtxView as P3WasiHttpCtxView, WasiHttpHooks, WasiHttpView as P3WasiHttpView,
};

use self::output::LogStream;

use crate::linker::InstancePolicy;
use crate::process::ProcessId;

/// Where an instance's stdout/stderr are routed.
pub enum OutputMode {
    /// Discard outputs (wasmtime's default sink). Used for snapshot init,
    /// where guest output is noise.
    Discard,
    /// Route to the per-process actor channel, drained by an attached client.
    Stream,
    /// Route to pie-worker's `tracing` log, tagged with `program`, when no
    /// client session is attached.
    Log { program: String },
}

pub struct InstanceState {
    // Wasm states
    id: ProcessId,
    username: String,

    // WASI states
    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,
    /// wasi:http@0.3 host hooks. Enforces the instance's network policy at the
    /// one host-side choke point (`is_supported_scheme`), the p3 analog of the
    /// old `pie:core/http.fetch` `network_allowed()` gate.
    http_hooks: PieHttpHooks,

    /// Whether outbound network is permitted (gates `pie:core/http.fetch`,
    /// parity with the wasi:http linker which is only wired when allowed).
    network_allowed: bool,

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
        // (Process/context unregister removed — Phase 5; working sets drop with
        // the instance's resource table.)
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
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            ctx: &mut self.http_ctx,
            table: &mut self.resource_table,
            hooks: Default::default(),
        }
    }
}

/// wasi:http@0.3 hooks carrying the instance's network policy. When the
/// network is disabled every scheme is reported unsupported, so
/// `wasi:http/handler#handle` fails each outgoing request with a protocol
/// error and the guest's `client.send` returns `Err` — instantiation still
/// succeeds (parity with the old host-side `network_allowed()` gate; the p2
/// path drops the link entirely instead).
pub struct PieHttpHooks {
    network_allowed: bool,
}

impl WasiHttpHooks for PieHttpHooks {
    fn is_supported_scheme(&mut self, scheme: &http::uri::Scheme) -> bool {
        self.network_allowed
            && (*scheme == http::uri::Scheme::HTTP || *scheme == http::uri::Scheme::HTTPS)
    }
}

impl P3WasiHttpView for InstanceState {
    fn http(&mut self) -> P3WasiHttpCtxView<'_> {
        P3WasiHttpCtxView {
            ctx: &mut self.http_ctx,
            table: &mut self.resource_table,
            hooks: &mut self.http_hooks,
        }
    }
}

impl InstanceState {
    pub async fn new(
        id: ProcessId,
        username: String,
        output: OutputMode,
        policy: &InstancePolicy,
        py_runtime_dir: Option<&Path>,
    ) -> anyhow::Result<Self> {
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
            OutputMode::Stream => {
                builder.stdout(LogStream::new_stdout(id));
                builder.stderr(LogStream::new_stderr(id));
            }
            OutputMode::Log { program } => {
                let program: Arc<str> = Arc::from(program);
                builder.stdout(LogStream::new_server_stdout(program.clone()));
                builder.stderr(LogStream::new_server_stderr(program));
            }
        }

        let scratch_dir = policy.fs.base_dir.join(id.to_string());

        if policy.fs.allow {
            std::fs::create_dir_all(&scratch_dir).expect("failed to create scratch dir");

            builder
                .preopened_dir(&scratch_dir, "/scratch", DirPerms::all(), FilePerms::all())
                .expect("failed to preopen scratch dir");
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
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            http_hooks: PieHttpHooks {
                network_allowed: policy.network.allow,
            },
            network_allowed: policy.network.allow,
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

    /// Whether outbound network is permitted for this inferlet.
    pub fn network_allowed(&self) -> bool {
        self.network_allowed
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
