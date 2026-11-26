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
use super::output::{LogStream, StderrTail};
use super::residency::ProcessResidency;
use crate::inferlet::program::Script;
use crate::inferlet::sandbox::InstancePolicy;
use crate::store::kv::page_table::WorkingSetId;
use crate::store::rs::RsWorkingSetId;

pub enum OutputMode {
    Discard,
    Stream,
}

pub struct ProcessCtx {
    id: ProcessId,

    wasi_ctx: WasiCtx,
    /// What the guest last wrote to stderr: said beside a trap.
    stderr_tail: StderrTail,
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
    /// What this launch was lent, and what it has given back.
    handles: Option<crate::handle::Bag>,
    bind_admitted: bool,
    execution_seated: bool,
    execution_admitted: bool,
    admission_wait_us: u64,
    residency_flag: Option<Arc<AtomicBool>>,
    limits: wasmtime::StoreLimits,
    /// Prompt tokens this process's fires found cached and mapped in.
    cached_tokens: u64,
    /// Whom its working sets share as (`working-set.share`): set from its
    /// arrival before it runs.
    keying: Option<Keying>,
    /// This context's count among those a shutdown waits to see torn down
    /// ([`super::teardown::all_torn_down`]).
    live: Option<super::teardown::Live>,
}

/// How a process's working sets key what they share.
#[derive(Debug, Clone)]
pub(crate) enum Keying {
    /// Under the scope a set asks, as its authenticated tenant and session.
    Asker(crate::policy::share::Asker),
    /// A pass a peer offloaded here: under the asking set's domain.
    OnBehalf(crate::store::kv::hash::Hash256),
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
            self.handles.take(),
            self.live.take(),
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
    /// The process whose guest sends: it waits on the outside until the
    /// response comes back ([`crate::inferlet::process::away_while`]).
    pid: ProcessId,
}

#[cfg(not(target_arch = "wasm32"))]
impl WasiHttpHooks for PieHttpHooks {
    fn is_supported_scheme(&mut self, scheme: &http::uri::Scheme) -> bool {
        self.network_allowed
            && (*scheme == http::uri::Scheme::HTTP || *scheme == http::uri::Scheme::HTTPS)
    }

    /// The default send, its response awaited as outside input: the
    /// process's lanes leave the frame while the guest waits on it.
    fn send_request(
        &mut self,
        request: http::Request<wasmtime_wasi_http::WasiBody>,
        options: Option<wasmtime_wasi_http::RequestOptions>,
        fut: Box<dyn Future<Output = Result<(), wasmtime_wasi_http::Error>> + Send>,
    ) -> Box<
        dyn Future<
                Output = Result<
                    (
                        http::Response<wasmtime_wasi_http::WasiBody>,
                        Box<dyn Future<Output = Result<(), wasmtime_wasi_http::Error>> + Send>,
                    ),
                    wasmtime_wasi_http::Error,
                >,
            > + Send,
    > {
        _ = fut;
        let pid = self.pid;
        Box::new(async move {
            use http_body_util::BodyExt;
            let (response, io) = crate::inferlet::process::away_while(
                pid,
                wasmtime_wasi_http::default_send_request(request, options),
            )
            .await?;
            Ok((
                response.map(BodyExt::boxed_unsync),
                Box::new(io) as Box<dyn Future<Output = _> + Send>,
            ))
        })
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
        output: OutputMode,
        policy: &InstancePolicy,
        script: Option<Arc<Script>>,
    ) -> anyhow::Result<Self> {
        let stderr_tail = StderrTail::default();
        let streamed = matches!(output, OutputMode::Stream);
        let err = LogStream::new_stderr(id, stderr_tail.clone(), streamed);

        #[cfg(target_arch = "wasm32")]
        let wasi_ctx = {
            let mut builder = WasiCtx::builder();
            if streamed {
                let out = LogStream::new_stdout(id);
                builder = builder.stdout(move |bytes| out.write_bytes(bytes));
            }
            builder.stderr(move |bytes| err.write_bytes(bytes)).build()
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
        {
            if streamed {
                builder.stdout(LogStream::new_stdout(id));
            }
            builder.stderr(err);
        }

        #[cfg(not(target_arch = "wasm32"))]
        let scratch_dir = if policy.fs.allow {
            let scratch_dir = policy.fs.base_dir.join(id.to_string());
            std::fs::create_dir_all(&scratch_dir).map_err(|why| {
                anyhow::anyhow!("scratch dir {}: not created: {why}", scratch_dir.display())
            })?;
            builder
                .preopened_dir(&scratch_dir, "/scratch", FsPerms::ReadWrite)
                .map_err(|why| {
                    anyhow::anyhow!("scratch dir {}: not opened: {why}", scratch_dir.display())
                })?;
            Some(scratch_dir)
        } else {
            None
        };

        #[cfg(not(target_arch = "wasm32"))]
        let wasi_ctx = builder.build();

        Ok(ProcessCtx {
            id,
            wasi_ctx,
            stderr_tail,
            resource_table: ResourceTable::new(),
            #[cfg(not(target_arch = "wasm32"))]
            http_ctx: WasiHttpCtx::new(),
            #[cfg(not(target_arch = "wasm32"))]
            http_hooks: PieHttpHooks {
                network_allowed: policy.network.allow,
                pid: id,
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
            handles: None,
            bind_admitted: false,
            execution_seated: false,
            execution_admitted: false,
            admission_wait_us: 0,
            residency_flag: None,
            cached_tokens: 0,
            keying: None,
            live: Some(super::teardown::Live::new()),
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

    /// What the guest last wrote to stderr; `None` if nothing.
    pub(crate) fn stderr_said(&self) -> Option<String> {
        self.stderr_tail.said()
    }

    pub(crate) fn tokens_cached(&self) -> u64 {
        self.cached_tokens
    }

    pub(crate) fn add_cached_tokens(&mut self, rows: u64) {
        self.cached_tokens = self.cached_tokens.saturating_add(rows);
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

    pub(crate) fn set_handles(&mut self, handles: crate::handle::Bag) {
        self.handles = Some(handles);
    }

    pub(crate) fn handles(&mut self) -> Option<&mut crate::handle::Bag> {
        self.handles.as_mut()
    }

    pub(crate) fn set_files(&mut self, files: Arc<client_api::Files>) {
        self.files = files;
    }

    pub(crate) fn set_keying(&mut self, keying: Keying) {
        self.keying = Some(keying);
    }

    pub(crate) fn keying(&self) -> Option<&Keying> {
        self.keying.as_ref()
    }

    pub(crate) fn set_props(&mut self, props: client_api::Props) {
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

    pub(crate) fn residency_pipelines(&self) -> Vec<crate::pipeline::fire::PendingFires> {
        self.residency.lock().unwrap().pipelines()
    }

    pub(crate) fn register_kv_working_set(&self, ws: &crate::store::kv::working_set::KvWorkingSet) {
        match crate::planner::planner_for(ws.model, ws.engine) {
            Some(planner) => planner.register_kv_working_set(self.id(), &self.residency, ws),
            None => {
                self.residency
                    .lock()
                    .unwrap()
                    .kv_working_sets
                    .insert((ws.model, ws.engine, ws.id), ws.suspend_handle());
            }
        }
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

    pub(crate) fn register_rs_working_set(&self, ws: &crate::store::rs::working_set::RsWorkingSet) {
        self.residency
            .lock()
            .unwrap()
            .rs_working_sets
            .insert((ws.model, ws.engine, ws.id));
        if let Some(departure) = &self.departure {
            ws.hold_departure(departure.clone());
        }
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
