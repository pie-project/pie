use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::ordered::OrderedPermit;
use wasmtime::component::ResourceTable;
#[cfg(not(target_arch = "wasm32"))]
use wasmtime_wasi::FsPerms;
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};
#[cfg(not(target_arch = "wasm32"))]
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpCtxView, WasiHttpHooks, WasiHttpView};

use super::ProcessId;
use super::output::LogStream;
use super::residency::ProcessResidency;
use crate::inferlet::program::Script;
use crate::inferlet::sandbox::InstancePolicy;
use crate::store::kv::page_table::WorkingSetId;
use crate::store::rs::RsWorkingSetId;

pub enum OutputMode {
    Discard,
    Stream,
    Log { program: String },
}

pub struct ProcessCtx {
    id: ProcessId,
    username: String,

    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    #[cfg(not(target_arch = "wasm32"))]
    http_ctx: WasiHttpCtx,
    #[cfg(not(target_arch = "wasm32"))]
    http_hooks: PieHttpHooks,

    scratch_dir: Option<PathBuf>,

    /// Set when the component is a language component and this is the
    /// script it runs.
    script: Option<Arc<Script>>,

    residency: Arc<Mutex<ProcessResidency>>,
    /// Handed to each KV working set it makes: see [`crate::planner::Departure`].
    departure: Option<Arc<crate::planner::Departure>>,
    prewarm_permit: Option<OrderedPermit>,
    props: client_api::Props,
    files: Arc<client_api::Files>,
    bind_admitted: bool,
    execution_seated: bool,
    execution_admitted: bool,
    admission_wait_us: u64,
    residency_flag: Option<Arc<AtomicBool>>,
    limits: wasmtime::StoreLimits,
}

impl Drop for ProcessCtx {
    fn drop(&mut self) {
        let seated = std::mem::take(&mut self.execution_seated);
        // A paused process gave its permits back already.
        let mut seat = super::take_permits(self.id);
        let bind_permit = seat.as_mut().and_then(|seat| seat.bind.take());
        self.execution_admitted = false;
        self.bind_admitted = false;
        let terminate_fences =
            seated.then(|| crate::scheduler::worker::post_process_terminate_fenced(self.id));
        if seat.as_ref().is_some_and(|seat| seat.exec.is_some()) {
            crate::scheduler::worker::notify_execution_slot_released(self.id);
        }
        drop(seat);
        let resources = std::mem::replace(&mut self.resource_table, ResourceTable::new());
        super::teardown::defer_resource_teardown(
            self.id,
            resources,
            self.residency.clone(),
            terminate_fences,
            bind_permit,
            std::mem::take(&mut self.scratch_dir),
        );
    }
}

impl WasiView for ProcessCtx {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi_ctx,
            table: &mut self.resource_table,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct PieHttpHooks {
    network_allowed: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl WasiHttpHooks for PieHttpHooks {
    fn is_supported_scheme(&mut self, scheme: &http::uri::Scheme) -> bool {
        self.network_allowed
            && (*scheme == http::uri::Scheme::HTTP || *scheme == http::uri::Scheme::HTTPS)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WasiHttpView for ProcessCtx {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            ctx: &mut self.http_ctx,
            table: &mut self.resource_table,
            hooks: &mut self.http_hooks,
        }
    }
}

impl ProcessCtx {
    pub fn script(&self) -> Option<&Arc<Script>> {
        self.script.as_ref()
    }

    pub async fn new(
        id: ProcessId,
        username: String,
        output: OutputMode,
        policy: &InstancePolicy,
        script: Option<Arc<Script>>,
    ) -> anyhow::Result<Self> {
        #[cfg(target_arch = "wasm32")]
        let wasi_ctx = {
            let mut builder = WasiCtx::builder();
            match output {
                OutputMode::Discard => {}
                OutputMode::Stream => {
                    let (out, err) = (LogStream::new_stdout(id), LogStream::new_stderr(id));
                    builder = builder
                        .stdout(move |bytes| out.write_bytes(bytes))
                        .stderr(move |bytes| err.write_bytes(bytes));
                }
                OutputMode::Log { program } => {
                    let program: Arc<str> = Arc::from(program);
                    let (out, err) = (
                        LogStream::new_server_stdout(program.clone()),
                        LogStream::new_server_stderr(program),
                    );
                    builder = builder
                        .stdout(move |bytes| out.write_bytes(bytes))
                        .stderr(move |bytes| err.write_bytes(bytes));
                }
            }
            builder.build()
        };
        #[cfg(target_arch = "wasm32")]
        let scratch_dir: Option<PathBuf> = None;

        #[cfg(not(target_arch = "wasm32"))]
        let mut builder = WasiCtx::builder();

        #[cfg(not(target_arch = "wasm32"))]
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

        #[cfg(not(target_arch = "wasm32"))]
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

        #[cfg(not(target_arch = "wasm32"))]
        let scratch_dir = if policy.fs.allow {
            let scratch_dir = policy.fs.base_dir.join(id.to_string());
            std::fs::create_dir_all(&scratch_dir).expect("failed to create scratch dir");

            builder
                .preopened_dir(&scratch_dir, "/scratch", FsPerms::ReadWrite)
                .expect("failed to preopen scratch dir");
            Some(scratch_dir)
        } else {
            None
        };

        #[cfg(not(target_arch = "wasm32"))]
        let wasi_ctx = builder.build();

        Ok(ProcessCtx {
            id,
            username,
            wasi_ctx,
            resource_table: ResourceTable::new(),
            #[cfg(not(target_arch = "wasm32"))]
            http_ctx: WasiHttpCtx::new(),
            #[cfg(not(target_arch = "wasm32"))]
            http_hooks: PieHttpHooks {
                network_allowed: policy.network.allow,
            },
            scratch_dir,
            script,
            residency: {
                let residency = Arc::new(Mutex::new(ProcessResidency::default()));
                super::residency::register_residency(id, Arc::downgrade(&residency));
                residency
            },
            departure: crate::planner::planner_of(id).and_then(|planner| planner.departure(id)),
            prewarm_permit: None,
            props: client_api::Props::new(),
            files: Arc::default(),
            bind_admitted: false,
            execution_seated: false,
            execution_admitted: false,
            admission_wait_us: 0,
            residency_flag: None,
            limits: wasmtime::StoreLimitsBuilder::new()
                .memory_size(policy.memory_limit)
                .build(),
        })
    }

    pub(crate) fn limits(&mut self) -> &mut wasmtime::StoreLimits {
        &mut self.limits
    }

    pub fn id(&self) -> ProcessId {
        self.id
    }

    pub(crate) fn is_resident_fast(&mut self) -> bool {
        if self.residency_flag.is_none() {
            let Some(planner) = crate::planner::planner_of(self.id) else {
                return true;
            };
            self.residency_flag = planner.residency_flag(self.id);
        }
        match &self.residency_flag {
            Some(flag) => flag.load(Ordering::Acquire),
            None => true,
        }
    }

    pub(crate) fn install_prewarm_permit(&mut self, permit: Option<OrderedPermit>) {
        self.prewarm_permit = permit;
    }

    pub(crate) fn props(&self) -> &client_api::Props {
        &self.props
    }

    pub(crate) fn file(&self, name: &str) -> Option<Vec<u8>> {
        self.files.get(name).map(|bytes| bytes.to_vec())
    }

    pub(crate) fn set_files(&mut self, files: Arc<client_api::Files>) {
        self.files = files;
    }

    pub(crate) fn set_props(&mut self, props: client_api::Props) {
        let tokens = client_api::props::persist_tokens(&props);
        let owner = crate::policy::schedule::plex_id(self.id());
        for stores in crate::store::registry::all_for_model(0) {
            crate::store::registry::with_kv_lock(&stores.kv, "persist-hint", |kv| {
                kv.set_persist_hint(owner, tokens);
            });
        }
        self.props = props;
    }

    pub(crate) fn release_prewarm_permit(&mut self) {
        self.prewarm_permit = None;
    }

    pub(crate) fn execution_admitted(&self) -> bool {
        self.execution_admitted
    }

    pub(crate) fn bind_admitted(&self) -> bool {
        self.bind_admitted
    }

    pub(crate) fn admit_bind(&mut self) {
        self.bind_admitted = true;
        self.prewarm_permit = None;
    }

    pub(crate) fn admit_execution(&mut self, seated: bool, wait_us: u64) {
        self.execution_seated = seated;
        self.execution_admitted = true;
        self.admission_wait_us = wait_us;
        self.prewarm_permit = None;
    }

    pub(crate) fn admission_wait_us(&self) -> u64 {
        self.admission_wait_us
    }

    pub fn get_username(&self) -> String {
        self.username.clone()
    }

    pub(crate) fn residency_pipelines(&self) -> Vec<crate::pipeline::fire::PendingFires> {
        self.residency.lock().unwrap().pipelines()
    }

    pub(crate) fn register_kv_working_set(&self, ws: &crate::store::kv::working_set::KvWorkingSet) {
        self.residency
            .lock()
            .unwrap()
            .kv_working_sets
            .insert((ws.model, ws.engine, ws.id), ws.suspend_handle());
        if let Some(departure) = &self.departure {
            ws.hold_departure(departure.clone());
        }
        let stores = crate::store::registry::get(ws.model, ws.engine);
        let owned = crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.set_owner(ws.id, self.id())
        });
        if let Err(error) = owned {
            tracing::warn!(pid = %self.id(), %error, "a working set has no owner");
        }
    }

    pub(crate) fn unregister_kv_working_set(
        &self,
        model: usize,
        engine: crate::engine::EngineId,
        id: WorkingSetId,
    ) {
        self.residency
            .lock()
            .unwrap()
            .kv_working_sets
            .remove(&(model, engine, id));
    }

    pub(crate) fn register_rs_working_set(
        &self,
        model: usize,
        engine: crate::engine::EngineId,
        id: RsWorkingSetId,
    ) {
        self.residency
            .lock()
            .unwrap()
            .rs_working_sets
            .insert((model, engine, id));
    }

    pub(crate) fn unregister_rs_working_set(
        &self,
        model: usize,
        engine: crate::engine::EngineId,
        id: RsWorkingSetId,
    ) {
        self.residency
            .lock()
            .unwrap()
            .rs_working_sets
            .remove(&(model, engine, id));
    }

    pub(crate) fn register_pipeline(
        &self,
        scope: &crate::store::PipelineScope,
        fires: &crate::pipeline::fire::PendingFires,
    ) {
        let mut residency = self.residency.lock().unwrap();
        residency
            .pipelines
            .retain(|pipeline| pipeline.fires.strong_count() > 0);
        residency
            .pipelines
            .push(super::residency::ResidentPipeline {
                scope: scope.clone(),
                fires: Arc::downgrade(fires),
            });
    }
}
