//! Executor verb server and shared FIFO driver actor.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{Context, Result};
use futures::StreamExt;
use pie_driver_abi::{
    ExecutorRequest, ExecutorResponse, ExecutorRpc, ExecutorRpcRequest, ExecutorRpcResponse,
    HelloRequest, HelloResponse, InlineKvPayload, MemoryDomain, ModelIdentity,
    PIE_TERMINAL_OUTCOME_FAILED, PIE_TERMINAL_OUTCOME_PENDING, PIE_TERMINAL_OUTCOME_SUCCESS,
    PieTerminalCell, PushKv, REMOTE_WIRE_VERSION, RemoteBindInstance, RemoteBindResponse,
    RemoteChannelBinding, RemoteError, RemoteErrorKind, RemoteLaunch, RemotePeerConn,
    RemoteRegisterChannel, RemoteTerminal, RemoteTransferKind, ScratchGrant, TerminalCellState,
};
use pie_engine::driver::{
    BoundInstance, ChannelValue, DriverBackend, InstanceBindingPlan, LaunchSubmission,
};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::{mpsc, oneshot};

use crate::translate::ModelDrivers;

type ClientId = u64;

#[cfg(feature = "driver-cuda")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaMemset(dst: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
}

#[cfg(feature = "driver-cuda")]
struct CudaDeviceGuard(i32);

#[cfg(feature = "driver-cuda")]
impl CudaDeviceGuard {
    fn select(device: u32) -> Result<Self> {
        let mut previous = 0;
        let status = unsafe { cudaGetDevice(&mut previous) };
        anyhow::ensure!(status == 0, "cudaGetDevice failed with status {status}");
        if previous != device as i32 {
            let status = unsafe { cudaSetDevice(device as i32) };
            anyhow::ensure!(status == 0, "cudaSetDevice failed with status {status}");
        }
        Ok(Self(previous))
    }
}

#[cfg(feature = "driver-cuda")]
impl Drop for CudaDeviceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaSetDevice(self.0);
        }
    }
}

fn append_region_bytes(
    domain: MemoryDomain,
    src: u64,
    len: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    match domain {
        MemoryDomain::HostPinned => {
            let bytes = unsafe { std::slice::from_raw_parts(src as *const u8, len) };
            out.extend_from_slice(bytes);
            Ok(())
        }

        MemoryDomain::CudaDevice(_device) => {
            #[cfg(feature = "driver-cuda")]
            {
                let _guard = CudaDeviceGuard::select(_device)?;
                let offset = out.len();
                out.resize(offset + len, 0);
                let status = unsafe {
                    cudaMemcpy(
                        out[offset..].as_mut_ptr() as *mut std::ffi::c_void,
                        src as *const std::ffi::c_void,
                        len,
                        4,
                    )
                };
                anyhow::ensure!(status == 0, "cudaMemcpy D2H failed with status {status}");
                Ok(())
            }
            #[cfg(not(feature = "driver-cuda"))]
            {
                Err(anyhow::anyhow!(
                    "CUDA KV export requires feature \"driver-cuda\""
                ))
            }
        }
        other => Err(anyhow::anyhow!(
            "inline KV export does not support {other:?}"
        )),
    }
}

fn zero_grant(handle: &pie_driver_abi::KvHandle, grant: ScratchGrant) -> Result<()> {
    for region in &handle.regions {
        let offset = (grant.base_page as u64)
            .checked_mul(region.page_stride)
            .context("scratch grant byte offset overflow")?;
        let bytes = (grant.num_pages as u64)
            .checked_mul(region.page_stride)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .context("scratch grant byte length overflow")?;
        anyhow::ensure!(
            region.page_stride > 0
                && offset
                    .checked_add(bytes as u64)
                    .is_some_and(|end| end <= region.len),
            "scratch grant exceeds exported KV region"
        );
        let base = region.base + offset;
        match region.domain {
            MemoryDomain::HostPinned => unsafe {
                std::ptr::write_bytes(base as *mut u8, 0, bytes);
            },
            MemoryDomain::CudaDevice(_device) => {
                #[cfg(feature = "driver-cuda")]
                {
                    let _guard = CudaDeviceGuard::select(_device)?;
                    let status = unsafe { cudaMemset(base as *mut std::ffi::c_void, 0, bytes) };
                    anyhow::ensure!(
                        status == 0,
                        "cudaMemset scratch grant failed with status {status}"
                    );
                }
                #[cfg(not(feature = "driver-cuda"))]
                {
                    anyhow::bail!("CUDA scratch zeroing requires feature \"driver-cuda\"");
                }
            }
            other => anyhow::bail!("scratch zeroing does not support {other:?}"),
        }
    }
    Ok(())
}

pub(crate) async fn connect(addr: &str) -> Result<pie_driver_abi::ExecutorRpcClient> {
    let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
    let transport = tcp::connect(tcp_addr, Bincode::default)
        .await
        .with_context(|| format!("dialing executor at {addr}"))?;
    Ok(pie_driver_abi::ExecutorRpcClient::new(tarpc::client::Config::default(), transport).spawn())
}

#[derive(Default)]
pub(crate) struct ExecutorStats {
    inflight: AtomicU32,
    leased_pages: AtomicU32,
}

#[cfg(feature = "nixl")]
#[derive(Clone)]
struct ExecutorNixl {
    engine: Arc<pie_transport::NixlEngine>,
    local: pie_transport::RegisteredHandle,
    metadata: Vec<u8>,
}

impl ExecutorStats {
    pub(crate) fn inflight(&self) -> u32 {
        self.inflight.load(Ordering::Relaxed)
    }

    pub(crate) fn kv_pressure_bucket(&self, total_pages: u32) -> u8 {
        if total_pages == 0 {
            return 0;
        }
        let used = self.leased_pages.load(Ordering::Relaxed) as u64;
        ((used.saturating_mul(u8::MAX as u64) / total_pages as u64).min(u8::MAX as u64)) as u8
    }
}

pub(crate) struct ExecutorServer {
    endpoint: String,
    accept_task: tokio::task::JoinHandle<()>,
    actor_task: tokio::task::JoinHandle<()>,
    core: ExecutorCoreHandle,
    stats: Arc<ExecutorStats>,
    total_pages: u32,
}

impl ExecutorServer {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) async fn bind(
        addr: &str,
        drivers: ModelDrivers,
        model: ModelIdentity,
        max_clients: usize,
    ) -> Result<Self> {
        Self::bind_with_transfer(
            addr,
            drivers,
            model,
            max_clients,
            crate::config::OffloadTransfer::Inline,
        )
        .await
    }

    pub(crate) async fn bind_with_transfer(
        addr: &str,
        mut drivers: ModelDrivers,
        model: ModelIdentity,
        max_clients: usize,
        transfer: crate::config::OffloadTransfer,
    ) -> Result<Self> {
        anyhow::ensure!(
            drivers.groups.len() == 1,
            "executor mode currently requires exactly one driver group, got {}",
            drivers.groups.len()
        );
        let group = drivers.groups.pop().expect("one executor driver group");
        let kv_handle = group.backend.export_kv_handle();
        if model.component == pie_driver_abi::ModelComponent::Encode {
            anyhow::ensure!(
                group.caps.supports_media_encode,
                "encode executor backend does not advertise media encoding"
            );
        } else {
            anyhow::ensure!(
                kv_handle.is_some(),
                "prefill executor backend does not export a KV layout"
            );
            anyhow::ensure!(
                max_clients <= group.caps.total_pages as usize,
                "executor max_clients {max_clients} exceeds {} KV pages",
                group.caps.total_pages
            );
        }

        let stats = Arc::new(ExecutorStats::default());
        #[cfg(feature = "nixl")]
        let nixl = match kv_handle.as_ref() {
            Some(kv_handle) => build_executor_nixl(transfer, &model, kv_handle)?,
            None => None,
        };
        #[cfg(not(feature = "nixl"))]
        let nixl = {
            anyhow::ensure!(
                model.component == pie_driver_abi::ModelComponent::Encode
                    || transfer != crate::config::OffloadTransfer::Nixl,
                "offload.transfer=nixl requires feature \"nixl\""
            );
            None
        };
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        anyhow::ensure!(
            !tcp_addr.starts_with("unix:"),
            "executor verb server currently requires a TCP address"
        );
        let incoming = tcp::listen(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("binding executor verb server at {addr}"))?;
        let endpoint = incoming.local_addr().to_string();

        let (tx, rx) = mpsc::channel(1024);
        let core = ExecutorCoreHandle { tx };
        let actor = ExecutorActor::new(
            group.backend,
            group.caps.clone(),
            model,
            kv_handle,
            max_clients,
            Arc::clone(&stats),
            nixl,
        );
        let actor_task = tokio::spawn(actor.run(rx, core.clone()));
        let next_client_id = Arc::new(AtomicU64::new(1));
        let accept_core = core.clone();
        let accept_task = tokio::spawn(
            incoming
                .filter_map(|connection| async move { connection.ok() })
                .for_each_concurrent(None, move |transport| {
                    let core = accept_core.clone();
                    let client_id = next_client_id.fetch_add(1, Ordering::Relaxed);
                    async move {
                        if core.connect(client_id).await.is_err() {
                            return;
                        }

                        let server = RpcServer {
                            core: core.clone(),
                            client_id,
                        };
                        BaseChannel::with_defaults(transport)
                            .execute(server.serve())
                            .for_each_concurrent(None, |request| async move {
                                tokio::spawn(request);
                            })
                            .await;
                        core.disconnect(client_id).await;
                    }
                }),
        );

        Ok(Self {
            endpoint,
            accept_task,
            actor_task,
            core,
            stats,
            total_pages: group.caps.total_pages,
        })
    }

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(crate) fn stats(&self) -> Arc<ExecutorStats> {
        Arc::clone(&self.stats)
    }

    pub(crate) fn total_pages(&self) -> u32 {
        self.total_pages
    }

    pub(crate) async fn shutdown(self) {
        self.accept_task.abort();
        let _ = self.accept_task.await;
        self.core.shutdown().await;
        let _ = self.actor_task.await;
    }
}

#[cfg(feature = "nixl")]
fn build_executor_nixl(
    transfer: crate::config::OffloadTransfer,
    model: &ModelIdentity,
    kv_handle: &pie_driver_abi::KvHandle,
) -> Result<Option<ExecutorNixl>> {
    use pie_transport::Engine;

    if transfer == crate::config::OffloadTransfer::Inline {
        return Ok(None);
    }
    let suffix = model.hash[..6]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let result = (|| {
        let engine = Arc::new(pie_transport::NixlEngine::new(&format!(
            "pie-executor-{}-{suffix}",
            std::process::id()
        ))?);
        let local = engine.register(pie_transport::WorkerId(0), kv_handle.clone())?;
        let metadata = engine.local_metadata()?;
        Ok::<_, pie_transport::TransportError>(ExecutorNixl {
            engine,
            local,
            metadata,
        })
    })();
    match (transfer, result) {
        (_, Ok(nixl)) => Ok(Some(nixl)),
        (crate::config::OffloadTransfer::Nixl, Err(error)) => {
            Err(anyhow::anyhow!("initializing executor NIXL: {error}"))
        }
        (crate::config::OffloadTransfer::Auto, Err(error)) => {
            tracing::warn!(%error, "NIXL unavailable; executor using inline KV transfer");
            Ok(None)
        }
        (crate::config::OffloadTransfer::Inline, _) => unreachable!(),
    }
}

#[derive(Clone)]
struct RpcServer {
    core: ExecutorCoreHandle,
    client_id: ClientId,
}

impl ExecutorRpc for RpcServer {
    async fn execute(
        self,
        _: tarpc::context::Context,
        request: ExecutorRequest,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        self.core.execute(self.client_id, request).await
    }
}

#[derive(Clone)]
struct ExecutorCoreHandle {
    tx: mpsc::Sender<Command>,
}

impl ExecutorCoreHandle {
    async fn connect(&self, client_id: ClientId) -> Result<()> {
        self.tx
            .send(Command::Connect { client_id })
            .await
            .context("executor actor stopped")
    }

    async fn execute(
        &self,
        client_id: ClientId,
        request: ExecutorRequest,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Execute {
                client_id,
                request,
                reply,
            })
            .await
            .map_err(|_| disconnected("executor actor stopped"))?;
        rx.await
            .map_err(|_| disconnected("executor request was cancelled"))?
    }

    async fn disconnect(&self, client_id: ClientId) {
        let _ = self.tx.send(Command::Disconnect { client_id }).await;
    }

    async fn shutdown(&self) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::Shutdown { reply }).await.is_ok() {
            let _ = rx.await;
        }
    }
}

enum Command {
    Connect {
        client_id: ClientId,
    },
    Execute {
        client_id: ClientId,
        request: ExecutorRequest,
        reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
    },
    Retired {
        client_id: ClientId,
    },
    Disconnect {
        client_id: ClientId,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

struct QueuedLaunch {
    client_id: ClientId,
    request: RemoteLaunch,
    reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
}

struct QueuedEncode {
    client_id: ClientId,
    request: pie_driver_abi::RemoteEncode,
    reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
}

struct EncodePartition {
    client_id: ClientId,
    image_start: usize,
    image_count: usize,
    max_rows: usize,
    anchors: Vec<u32>,
    reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
}

struct LeasedInstance {
    local_id: u64,
    bound: BoundInstance,
}

struct LeasedChannel {
    local_id: u64,
    registered: pie_engine::driver::RegisteredChannel,
}

#[derive(Default)]
struct ClientState {
    hello: bool,
    client_nonce: u64,
    lease_slot: Option<usize>,
    grant: Option<ScratchGrant>,
    transfer: Option<RemoteTransferKind>,
    local_instance_ids: HashSet<u64>,
    instances: HashMap<u64, LeasedInstance>,
    local_channel_ids: HashSet<u64>,
    channels: HashMap<u64, LeasedChannel>,
    inflight: u32,
    disconnecting: bool,
}

struct ProgramRecord {
    registration: pie_driver_abi::ProgramRegistration,
    program_id: u64,
}

struct ExecutorActor {
    backend: DriverBackend,
    capabilities: pie_driver_abi::DriverCapabilities,
    model: ModelIdentity,
    kv_handle: Option<pie_driver_abi::KvHandle>,
    max_clients: usize,
    lease_slots: Vec<bool>,
    clients: HashMap<ClientId, ClientState>,
    programs: HashMap<u64, ProgramRecord>,
    next_channel_id: u64,
    stats: Arc<ExecutorStats>,
    #[cfg(feature = "nixl")]
    nixl: Option<ExecutorNixl>,
}

impl ExecutorActor {
    fn new(
        backend: DriverBackend,
        capabilities: pie_driver_abi::DriverCapabilities,
        model: ModelIdentity,
        kv_handle: Option<pie_driver_abi::KvHandle>,
        max_clients: usize,
        stats: Arc<ExecutorStats>,
        #[cfg(feature = "nixl")] nixl: Option<ExecutorNixl>,
        #[cfg(not(feature = "nixl"))] _nixl: Option<()>,
    ) -> Self {
        Self {
            backend,
            capabilities,
            model,
            kv_handle,
            max_clients,
            lease_slots: vec![false; max_clients],
            clients: HashMap::new(),
            programs: HashMap::new(),
            next_channel_id: 1,
            stats,
            #[cfg(feature = "nixl")]
            nixl,
        }
    }

    async fn run(mut self, mut rx: mpsc::Receiver<Command>, handle: ExecutorCoreHandle) {
        let mut backlog = VecDeque::new();
        loop {
            let command = match backlog.pop_front() {
                Some(command) => command,
                None => match rx.recv().await {
                    Some(command) => command,
                    None => break,
                },
            };
            match command {
                Command::Connect { client_id } => {
                    self.clients.entry(client_id).or_default();
                }
                Command::Execute {
                    client_id,
                    request: ExecutorRequest::Launch(request),
                    reply,
                } => {
                    if let Err(error) = self
                        .validate_client_ready(client_id)
                        .and_then(|()| self.validate_launch(client_id, &request))
                    {
                        let _ = reply.send(Err(error));
                        continue;
                    }
                    let mut launches = vec![QueuedLaunch {
                        client_id,
                        request,
                        reply,
                    }];
                    while let Ok(next) = rx.try_recv() {
                        match next {
                            Command::Execute {
                                client_id,
                                request: ExecutorRequest::Launch(request),
                                reply,
                            } if self.validate_client_ready(client_id).is_ok()
                                && self.validate_launch(client_id, &request).is_ok()
                                && self.can_coalesce(&launches, &request) =>
                            {
                                launches.push(QueuedLaunch {
                                    client_id,
                                    request,
                                    reply,
                                });
                            }
                            other => {
                                backlog.push_back(other);
                                break;
                            }
                        }
                    }
                    self.launch_group(launches, handle.clone());
                }
                Command::Execute {
                    client_id,
                    request: ExecutorRequest::Encode(request),
                    reply,
                } => {
                    if let Err(error) = self.validate_client_ready(client_id) {
                        let _ = reply.send(Err(error));
                        continue;
                    }
                    let mut encodes = vec![QueuedEncode {
                        client_id,
                        request,
                        reply,
                    }];
                    while let Ok(next) = rx.try_recv() {
                        match next {
                            Command::Execute {
                                client_id,
                                request: ExecutorRequest::Encode(request),
                                reply,
                            } if self.validate_client_ready(client_id).is_ok()
                                && self.can_coalesce_encode(&encodes, &request) =>
                            {
                                encodes.push(QueuedEncode {
                                    client_id,
                                    request,
                                    reply,
                                });
                            }
                            other => {
                                backlog.push_back(other);
                                break;
                            }
                        }
                    }
                    self.encode_group(encodes, handle.clone());
                }
                Command::Execute {
                    client_id,
                    request,
                    reply,
                } => self.execute(client_id, request, reply, handle.clone()),
                Command::Retired { client_id } => self.retired(client_id),
                Command::Disconnect { client_id } => self.disconnect(client_id),
                Command::Shutdown { reply } => {
                    let clients = self.clients.keys().copied().collect::<Vec<_>>();
                    for client_id in clients {
                        self.cleanup_client(client_id);
                    }
                    let _ = reply.send(());
                    break;
                }
            }
        }
    }

    fn execute(
        &mut self,
        client_id: ClientId,
        request: ExecutorRequest,
        reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
        handle: ExecutorCoreHandle,
    ) {
        if let Err(error) = self.validate_client_connection(client_id) {
            let _ = reply.send(Err(error));
            return;
        }

        match request {
            ExecutorRequest::Hello(request) => {
                let _ = reply.send(self.hello(client_id, request));
            }
            ExecutorRequest::LoadedModel => {
                let _ = reply.send(Ok(ExecutorResponse::LoadedModel(true)));
            }
            _other if !self.clients[&client_id].hello => {
                let _ = reply.send(Err(invalid("Hello must be the first executor request")));
            }
            ExecutorRequest::RegisterProgram(registration) => {
                let result = self.register_program(registration);
                let _ = reply.send(result);
            }
            ExecutorRequest::RegisterChannel(request) => {
                let result = self.register_channel(client_id, request);
                let _ = reply.send(result);
            }
            ExecutorRequest::BindInstance(request) => {
                let result = self.bind_instance(client_id, request);
                let _ = reply.send(result);
            }
            ExecutorRequest::Launch(request) => {
                self.launch(client_id, request, reply, handle);
            }
            ExecutorRequest::CopyKv(plan) => {
                self.copy_kv(client_id, plan, reply, handle);
            }
            ExecutorRequest::Encode(request) => {
                self.encode_group(
                    vec![QueuedEncode {
                        client_id,
                        request,
                        reply,
                    }],
                    handle,
                );
            }
            ExecutorRequest::PushKv(request) => {
                self.push_kv(client_id, request, reply, handle);
            }
            ExecutorRequest::CloseInstance(instance_id) => {
                let result = self.close_instance(client_id, instance_id);
                let _ = reply.send(result);
            }
            ExecutorRequest::CloseChannel(channel_id) => {
                let result = self.close_channel(client_id, channel_id);
                let _ = reply.send(result);
            }
        }
    }

    fn hello(
        &mut self,
        client_id: ClientId,
        request: HelloRequest,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        if self.clients[&client_id].hello {
            return Err(invalid("Hello was already completed"));
        }
        if self.clients.iter().any(|(&other_id, client)| {
            other_id != client_id
                && client.hello
                && !client.disconnecting
                && client.client_nonce == request.client_nonce
        }) {
            return Err(invalid(format!(
                "client nonce {} is already leased",
                request.client_nonce
            )));
        }
        if request.wire_version != REMOTE_WIRE_VERSION {
            return Err(incompatible(format!(
                "wire version {} != {}",
                request.wire_version, REMOTE_WIRE_VERSION
            )));
        }
        if request.model != self.model {
            return Err(incompatible("model identity mismatch"));
        }
        if self.model.component == pie_driver_abi::ModelComponent::Encode {
            if !self.capabilities.supports_media_encode {
                return Err(unsupported(
                    "encode executor backend does not support media encoding",
                ));
            }
            let client = self.clients.get_mut(&client_id).expect("client exists");
            client.hello = true;
            client.client_nonce = request.client_nonce;
            client.transfer = Some(RemoteTransferKind::Inline);
            return Ok(ExecutorResponse::Hello(HelloResponse {
                wire_version: REMOTE_WIRE_VERSION,
                model: self.model.clone(),
                kv_layout: request.kv_layout,
                capabilities: self.capabilities.clone(),
                grant: ScratchGrant {
                    base_page: 0,
                    num_pages: 0,
                },
                peer_conn: RemotePeerConn {
                    kind: RemoteTransferKind::Inline,
                    handle: None,
                    metadata: Vec::new(),
                },
            }));
        }
        let kv_handle = self
            .kv_handle
            .as_ref()
            .ok_or_else(|| incompatible("prefill executor has no KV export"))?;
        if !request.kv_layout.compatible_with(&kv_handle.layout) {
            return Err(incompatible("KV layout mismatch"));
        }
        #[cfg(feature = "nixl")]
        let mut selected_transfer = RemoteTransferKind::Inline;
        #[cfg(not(feature = "nixl"))]
        let selected_transfer = RemoteTransferKind::Inline;
        #[cfg(feature = "nixl")]
        if let (Some(nixl), Some(peer)) = (self.nixl.as_ref(), request.peer_conn.as_ref())
            && peer.kind == RemoteTransferKind::Nixl
        {
            use pie_transport::Engine;

            let handle = peer
                .handle
                .clone()
                .ok_or_else(|| invalid("NIXL peer connection is missing its KV handle"))?;
            nixl.engine
                .connect(&pie_transport::PeerConn {
                    worker: pie_transport::WorkerId(request.client_nonce),
                    handle,
                    metadata: peer.metadata.clone(),
                })
                .map_err(driver_error)?;
            selected_transfer = RemoteTransferKind::Nixl;
        }
        let Some(slot) = self.lease_slots.iter().position(|used| !used) else {
            return Err(RemoteError::new(
                RemoteErrorKind::ResourceExhausted,
                format!("executor max_clients {} reached", self.max_clients),
            ));
        };
        let pages_per_client = self.capabilities.total_pages / self.max_clients as u32;
        if pages_per_client == 0 {
            return Err(RemoteError::new(
                RemoteErrorKind::ResourceExhausted,
                "executor has no pages available per client",
            ));
        }
        let grant = ScratchGrant {
            base_page: slot as u32 * pages_per_client,
            num_pages: pages_per_client,
        };
        zero_grant(kv_handle, grant).map_err(driver_error)?;
        self.lease_slots[slot] = true;
        let client = self.clients.get_mut(&client_id).expect("client exists");
        client.hello = true;
        client.client_nonce = request.client_nonce;
        client.lease_slot = Some(slot);
        client.grant = Some(grant);
        client.transfer = Some(selected_transfer);
        self.stats
            .leased_pages
            .fetch_add(grant.num_pages, Ordering::Relaxed);
        Ok(ExecutorResponse::Hello(HelloResponse {
            wire_version: REMOTE_WIRE_VERSION,
            model: self.model.clone(),
            kv_layout: kv_handle.layout.clone(),
            capabilities: self.capabilities.clone(),
            grant,
            peer_conn: RemotePeerConn {
                kind: selected_transfer,
                handle: Some(kv_handle.clone()),
                metadata: {
                    #[cfg(feature = "nixl")]
                    {
                        if selected_transfer == RemoteTransferKind::Nixl {
                            self.nixl
                                .as_ref()
                                .map(|nixl| nixl.metadata.clone())
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        }
                    }
                    #[cfg(not(feature = "nixl"))]
                    {
                        Vec::new()
                    }
                },
            },
        }))
    }

    fn register_program(
        &mut self,
        registration: pie_driver_abi::ProgramRegistration,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        if let Some(existing) = self.programs.get(&registration.program_hash) {
            if existing.registration != registration {
                return Err(invalid(format!(
                    "program hash {} was reused with different bytes",
                    registration.program_hash
                )));
            }
            return Ok(ExecutorResponse::ProgramRegistered(existing.program_id));
        }
        let program_id = self
            .backend
            .register_program(&registration)
            .map_err(driver_error)?;
        self.programs.insert(
            registration.program_hash,
            ProgramRecord {
                registration,
                program_id,
            },
        );
        Ok(ExecutorResponse::ProgramRegistered(program_id))
    }

    fn bind_instance(
        &mut self,
        client_id: ClientId,
        request: RemoteBindInstance,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        if request.local_instance_id == 0 {
            return Err(invalid("local instance id must be nonzero"));
        }
        if self.clients[&client_id]
            .local_instance_ids
            .contains(&request.local_instance_id)
        {
            return Err(invalid(format!(
                "local instance {} is already bound",
                request.local_instance_id
            )));
        }
        let client = &self.clients[&client_id];
        if request
            .channel_ids
            .iter()
            .chain(request.seed_values.iter().map(|value| &value.channel_id))
            .any(|channel| !client.channels.contains_key(channel))
        {
            return Err(invalid(
                "instance binding references a channel outside its client lease",
            ));
        }
        let pacing_wait_id = pie_engine::driver::waker::WakerTable::global().alloc();
        let plan = InstanceBindingPlan {
            driver_id: 0,
            program_id: request.program_id,
            requested_instance_id: 0,
            pacing_wait_id,
            channel_ids: request.channel_ids,
            seed_values: request
                .seed_values
                .into_iter()
                .map(|value| ChannelValue {
                    channel: value.channel_id,
                    bytes: value.bytes,
                })
                .collect(),
        };
        let bound = match self.backend.bind_instance(&plan) {
            Ok(bound) => bound,
            Err(error) => {
                pie_engine::driver::waker::WakerTable::global().free(pacing_wait_id);
                return Err(driver_error(error));
            }
        };
        let executor_instance_id = bound.instance_id;
        let client = self.clients.get_mut(&client_id).expect("client exists");
        if client.instances.contains_key(&executor_instance_id) {
            let _ = self.backend.close_instance(executor_instance_id);
            bound.close_wait_slots();
            return Err(RemoteError::new(
                RemoteErrorKind::Internal,
                format!("executor minted duplicate instance {executor_instance_id}"),
            ));
        }
        client.local_instance_ids.insert(request.local_instance_id);
        client.instances.insert(
            executor_instance_id,
            LeasedInstance {
                local_id: request.local_instance_id,
                bound,
            },
        );
        Ok(ExecutorResponse::InstanceBound(RemoteBindResponse {
            local_instance_id: request.local_instance_id,
            executor_instance_id,
        }))
    }

    fn register_channel(
        &mut self,
        client_id: ClientId,
        request: RemoteRegisterChannel,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        if request.local_channel_id == 0 {
            return Err(invalid("local channel id must be nonzero"));
        }
        if self.clients[&client_id]
            .local_channel_ids
            .contains(&request.local_channel_id)
        {
            return Err(invalid(format!(
                "local channel {} is already registered",
                request.local_channel_id
            )));
        }
        let local_channel_id = request.local_channel_id;
        let executor_channel_id = self.next_channel_id;
        self.next_channel_id = self.next_channel_id.checked_add(1).ok_or_else(|| {
            RemoteError::new(
                RemoteErrorKind::ResourceExhausted,
                "channel id space exhausted",
            )
        })?;
        let table = pie_engine::driver::waker::WakerTable::global();
        let reader_wait_id = table.alloc();
        let writer_wait_id = table.alloc();
        let plan = pie_driver_abi::ChannelRegistrationPlan {
            driver_id: 0,
            channel_id: executor_channel_id,
            shape: request.shape,
            dtype: request.dtype,
            host_role: request.host_role,
            seeded: request.seeded,
            extern_dir: request.extern_dir,
            capacity: request.capacity,
            reader_wait_id,
            writer_wait_id,
            extern_name: request.extern_name,
        };
        let registered = match self.backend.register_channel(&plan) {
            Ok(registered) => registered,
            Err(error) => {
                table.free(reader_wait_id);
                table.free(writer_wait_id);
                return Err(driver_error(error));
            }
        };
        let client = self.clients.get_mut(&client_id).expect("client exists");
        client.local_channel_ids.insert(local_channel_id);
        client.channels.insert(
            executor_channel_id,
            LeasedChannel {
                local_id: local_channel_id,
                registered,
            },
        );
        Ok(ExecutorResponse::ChannelRegistered(RemoteChannelBinding {
            local_channel_id,
            executor_channel_id,
        }))
    }

    fn launch(
        &mut self,
        client_id: ClientId,
        request: RemoteLaunch,
        reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
        handle: ExecutorCoreHandle,
    ) {
        if let Err(error) = self.validate_launch(client_id, &request) {
            let _ = reply.send(Err(error));
            return;
        }
        self.launch_group(
            vec![QueuedLaunch {
                client_id,
                request,
                reply,
            }],
            handle,
        );
    }

    fn can_coalesce(&self, current: &[QueuedLaunch], candidate: &RemoteLaunch) -> bool {
        if candidate.plan.has_user_mask
            || !candidate.plan.kv_len_device.is_empty()
            || current.iter().any(|launch| {
                launch.request.plan.has_user_mask || !launch.request.plan.kv_len_device.is_empty()
            })
        {
            return false;
        }
        let mut instances = current
            .iter()
            .flat_map(|launch| launch.request.instance_ids.iter().copied())
            .collect::<HashSet<_>>();
        if candidate
            .instance_ids
            .iter()
            .any(|instance| !instances.insert(*instance))
        {
            return false;
        }
        let requests = instances.len();
        let tokens = current
            .iter()
            .map(|launch| launch.request.plan.token_ids.len())
            .sum::<usize>()
            + candidate.plan.token_ids.len();
        let page_refs = current
            .iter()
            .map(|launch| launch.request.plan.kv_page_indices.len())
            .sum::<usize>()
            + candidate.plan.kv_page_indices.len();
        requests <= self.capabilities.max_forward_requests as usize
            && tokens <= self.capabilities.max_forward_tokens as usize
            && page_refs <= self.capabilities.max_page_refs as usize
    }

    fn can_coalesce_encode(
        &self,
        current: &[QueuedEncode],
        candidate: &pie_driver_abi::RemoteEncode,
    ) -> bool {
        if self.backend.kind() == "dummy"
            || !candidate.plan.embed_rows.is_empty()
            || validate_raw_encode_plan(&candidate.plan, self.capabilities.hidden_size).is_err()
            || current
                .iter()
                .any(|encode| !encode.request.plan.embed_rows.is_empty())
        {
            return false;
        }
        let requests = current
            .iter()
            .map(|encode| encode.request.plan.image_anchor_rows.len())
            .sum::<usize>()
            + candidate.plan.image_anchor_rows.len();
        let tokens = current
            .iter()
            .map(|encode| encode.request.plan.token_ids.len())
            .sum::<usize>()
            + candidate.plan.token_ids.len();
        requests <= self.capabilities.max_forward_requests as usize
            && tokens <= self.capabilities.max_forward_tokens as usize
    }

    fn launch_group(&mut self, launches: Vec<QueuedLaunch>, handle: ExecutorCoreHandle) {
        let mut metadata = Vec::with_capacity(launches.len());
        let mut requests = Vec::with_capacity(launches.len());
        for launch in launches {
            metadata.push((
                launch.client_id,
                launch.request.terminal_count as usize,
                launch.reply,
            ));
            requests.push(launch.request);
        }
        let merged = match merge_remote_launches(requests) {
            Ok(merged) => merged,
            Err(error) => {
                for (_, _, reply) in metadata {
                    let _ = reply.send(Err(invalid(error.to_string())));
                }
                return;
            }
        };
        let mut cells = (0..merged.terminal_count)
            .map(|_| {
                Box::new(PieTerminalCell {
                    outcome: PIE_TERMINAL_OUTCOME_PENDING,
                    reserved0: 0,
                })
            })
            .collect::<Vec<_>>();
        let terminal_cells = cells
            .iter_mut()
            .map(|cell| cell.as_mut() as *mut PieTerminalCell)
            .collect();
        let submission = LaunchSubmission {
            plan: merged.plan,
            instance_ids: merged.instance_ids,
            terminal_cells,
            kv_translation: merged.kv_translation,
            kv_translation_indptr: merged.kv_translation_indptr,
            program_row_indptr: merged.program_row_indptr,
            logical_fire_ids: merged.logical_fire_ids,
            channel_expected_head: merged.channel_expected_head,
            channel_expected_tail: merged.channel_expected_tail,
            channel_ticket_indptr: merged.channel_ticket_indptr,
        };
        let completion = match self.backend.launch(&submission) {
            Ok(completion) => completion,
            Err(error) => {
                let error = driver_error(error);
                for (_, _, reply) in metadata {
                    let _ = reply.send(Err(error.clone()));
                }
                return;
            }
        };
        for (client_id, _, _) in &metadata {
            self.begin_inflight(*client_id);
        }
        tokio::spawn(async move {
            let result = completion.await.map_err(driver_error);
            let states = result.as_ref().ok().map(|()| {
                cells
                    .iter()
                    .map(|cell| read_terminal(cell))
                    .collect::<Vec<_>>()
            });
            let mut offset = 0usize;
            for (client_id, count, reply) in metadata {
                let response = match (&result, &states) {
                    (Ok(()), Some(states)) => {
                        let end = offset + count;
                        let terminal = RemoteTerminal {
                            per_request: states[offset..end].to_vec(),
                        };
                        offset = end;
                        Ok(ExecutorResponse::Terminal(terminal))
                    }
                    (Err(error), _) => Err(error.clone()),
                    _ => unreachable!(),
                };
                let _ = reply.send(response);
                let _ = handle.tx.send(Command::Retired { client_id }).await;
            }
        });
    }

    fn validate_launch(
        &self,
        client_id: ClientId,
        request: &RemoteLaunch,
    ) -> std::result::Result<(), RemoteError> {
        if request.terminal_count as usize != request.logical_fire_ids.len() {
            return Err(invalid(format!(
                "terminal count {} != logical fire count {}",
                request.terminal_count,
                request.logical_fire_ids.len()
            )));
        }
        let rows = request.plan.qo_indptr.len().saturating_sub(1);
        if request.plan.qo_indptr.first().copied() != Some(0)
            || request.plan.qo_indptr.last().copied() != Some(request.plan.token_ids.len() as u32)
            || !request
                .plan
                .qo_indptr
                .windows(2)
                .all(|window| window[0] <= window[1])
        {
            return Err(invalid("launch query-row CSR is malformed"));
        }
        if request.program_row_indptr.len() != request.instance_ids.len() + 1
            || request.program_row_indptr.first().copied() != Some(0)
            || request.program_row_indptr.last().copied() != Some(rows as u32)
            || !request
                .program_row_indptr
                .windows(2)
                .all(|window| window[0] <= window[1])
        {
            return Err(invalid("launch program-row attribution is malformed"));
        }
        if !valid_optional_csr(
            &request.plan.kv_page_indptr,
            rows,
            request.plan.kv_page_indices.len(),
        ) || !valid_optional_csr(&request.plan.mask_indptr, rows, request.plan.masks.len())
            || !valid_optional_csr(
                &request.plan.sampling_indptr,
                rows,
                request.plan.sampling_indices.len(),
            )
            || !valid_optional_csr(
                &request.plan.rs_buffer_slot_indptr,
                rows,
                request.plan.rs_buffer_slot_ids.len(),
            )
        {
            return Err(invalid("launch contains a malformed per-row CSR"));
        }
        if !request.plan.sampling_indptr.is_empty() {
            for row in 0..rows {
                let row_len = request.plan.qo_indptr[row + 1] - request.plan.qo_indptr[row];
                let start = request.plan.sampling_indptr[row] as usize;
                let end = request.plan.sampling_indptr[row + 1] as usize;
                if request.plan.sampling_indices[start..end]
                    .iter()
                    .any(|&index| index >= row_len)
                {
                    return Err(invalid("launch sampling index exceeds its request row"));
                }
            }
        }
        let client = self
            .clients
            .get(&client_id)
            .ok_or_else(|| disconnected("unknown executor connection"))?;
        if request
            .instance_ids
            .iter()
            .any(|instance| !client.instances.contains_key(instance))
        {
            return Err(invalid("launch references an instance outside its lease"));
        }
        let grant = client
            .grant
            .ok_or_else(|| invalid("Hello must precede Launch"))?;
        if request.kv_translation_indptr.len() != request.instance_ids.len() + 1
            || request.kv_translation_indptr.first().copied() != Some(0)
            || request.kv_translation_indptr.last().copied()
                != Some(request.kv_translation.len() as u32)
            || !request
                .kv_translation_indptr
                .windows(2)
                .all(|window| window[0] <= window[1])
        {
            return Err(invalid("launch KV translation CSR is malformed"));
        }
        if request.channel_expected_head.len() != request.channel_expected_tail.len()
            || request.channel_ticket_indptr.len() != request.terminal_count as usize + 1
            || request.channel_ticket_indptr.first().copied() != Some(0)
            || request.channel_ticket_indptr.last().copied()
                != Some(request.channel_expected_head.len() as u32)
            || !request
                .channel_ticket_indptr
                .windows(2)
                .all(|window| window[0] <= window[1])
        {
            return Err(invalid("launch channel-ticket CSR is malformed"));
        }
        if request
            .kv_translation
            .iter()
            .any(|&page| !grant.contains(page))
        {
            return Err(invalid("launch references a KV page outside its grant"));
        }
        let has_passthrough_segment = request
            .kv_translation_indptr
            .windows(2)
            .any(|window| window[0] == window[1]);
        if has_passthrough_segment
            && request
                .plan
                .kv_page_indices
                .iter()
                .any(|&page| !grant.contains(page))
        {
            return Err(invalid(
                "launch pass-through page references escape the client grant",
            ));
        }
        Ok(())
    }

    fn validate_client_connection(
        &self,
        client_id: ClientId,
    ) -> std::result::Result<(), RemoteError> {
        let client = self
            .clients
            .get(&client_id)
            .ok_or_else(|| disconnected("unknown executor connection"))?;
        if client.disconnecting {
            return Err(disconnected("executor connection is closing"));
        }
        Ok(())
    }

    fn validate_client_ready(&self, client_id: ClientId) -> std::result::Result<(), RemoteError> {
        self.validate_client_connection(client_id)?;
        if !self.clients[&client_id].hello {
            return Err(invalid("Hello must be the first executor request"));
        }
        Ok(())
    }

    fn copy_kv(
        &mut self,
        client_id: ClientId,
        plan: pie_driver_abi::KvCopyPlan,
        reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
        handle: ExecutorCoreHandle,
    ) {
        let grant = self.clients[&client_id]
            .grant
            .expect("hello established a grant");
        let out_of_grant = plan
            .src_page_ids
            .iter()
            .chain(plan.dst_page_ids.iter())
            .copied()
            .chain(
                plan.cells
                    .iter()
                    .flat_map(|cell| [cell.src_page_id, cell.dst_page_id]),
            )
            .any(|page| !grant.contains(page));
        if out_of_grant {
            let _ = reply.send(Err(invalid("KV copy references a page outside its grant")));
            return;
        }

        let completion = match self.backend.copy_kv(&plan) {
            Ok(completion) => completion,
            Err(error) => {
                let _ = reply.send(Err(driver_error(error)));
                return;
            }
        };
        self.begin_inflight(client_id);
        tokio::spawn(async move {
            let state = if completion.await.is_ok() {
                TerminalCellState {
                    outcome: PIE_TERMINAL_OUTCOME_SUCCESS,
                    reserved0: 0,
                }
            } else {
                TerminalCellState {
                    outcome: PIE_TERMINAL_OUTCOME_FAILED,
                    reserved0: 0,
                }
            };
            let _ = reply.send(Ok(ExecutorResponse::Terminal(RemoteTerminal {
                per_request: vec![state],
            })));
            let _ = handle.tx.send(Command::Retired { client_id }).await;
        });
    }

    fn encode_group(&mut self, requests: Vec<QueuedEncode>, handle: ExecutorCoreHandle) {
        let mut valid = Vec::with_capacity(requests.len());
        for request in requests {
            let validation = if request.request.plan.embed_rows.is_empty() {
                validate_raw_encode_plan(&request.request.plan, self.capabilities.hidden_size)
            } else {
                validate_embeddings(&request.request.plan)
            };
            if let Err(error) = validation {
                let _ = request.reply.send(Err(error));
            } else {
                valid.push(request);
            }
        }
        if valid.is_empty() {
            return;
        }
        if valid.len() == 1
            || self.backend.kind() == "dummy"
            || valid
                .iter()
                .any(|request| !request.request.plan.embed_rows.is_empty())
        {
            for request in valid {
                self.encode_one(
                    request.client_id,
                    request.request,
                    request.reply,
                    handle.clone(),
                );
            }
            return;
        }

        if !self.capabilities.supports_media_encode || self.capabilities.hidden_size == 0 {
            for request in valid {
                let _ = request.reply.send(Err(unsupported(
                    "this backend does not expose standalone media encoding",
                )));
            }
            return;
        }

        let hidden = self.capabilities.hidden_size as usize;
        let total_pixel_bytes = valid.iter().try_fold(0usize, |total, request| {
            total.checked_add(request.request.plan.image_pixels.len())
        });
        if !total_pixel_bytes.is_some_and(|bytes| u32::try_from(bytes).is_ok()) {
            for request in valid {
                let _ = request
                    .reply
                    .send(Err(invalid("coalesced media payload is too large")));
            }
            return;
        }
        let total_max_rows = valid
            .iter()
            .map(|request| request.request.plan.token_ids.len())
            .sum::<usize>();
        let Some(output_bytes) = total_max_rows
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(2))
        else {
            for request in valid {
                let _ = request
                    .reply
                    .send(Err(invalid("media encode output size overflow")));
            }
            return;
        };

        let mut encode = pie_driver_abi::MediaEncodePlan {
            image_pixel_indptr: vec![0],
            ..Default::default()
        };
        let mut partitions = Vec::with_capacity(valid.len());
        for request in valid {
            let plan = request.request.plan;
            let image_start = encode.image_anchor_rows.len();
            let image_count = plan.image_anchor_rows.len();
            let pixel_base = encode.image_pixels.len() as u32;
            encode.image_grids.extend(plan.image_grids);
            encode.image_pixels.extend(plan.image_pixels);
            encode.image_pixel_indptr.extend(
                plan.image_pixel_indptr
                    .into_iter()
                    .skip(1)
                    .map(|offset| pixel_base + offset),
            );
            encode
                .image_patch_positions
                .extend(plan.image_patch_positions);
            encode
                .image_anchor_rows
                .extend(plan.image_anchor_rows.iter().copied());
            partitions.push(EncodePartition {
                client_id: request.client_id,
                image_start,
                image_count,
                max_rows: plan.token_ids.len(),
                anchors: plan.image_anchor_rows,
                reply: request.reply,
            });
        }
        encode.output_rows = vec![0; output_bytes];
        encode.output_row_indptr = vec![0; encode.image_anchor_rows.len() + 1];

        let completion = match self.backend.encode(&mut encode) {
            Ok(completion) => completion,
            Err(error) => {
                let error = error.to_string();
                for partition in partitions {
                    let _ = partition.reply.send(Err(RemoteError::new(
                        RemoteErrorKind::Driver,
                        error.clone(),
                    )));
                }
                return;
            }
        };
        for partition in &partitions {
            self.begin_inflight(partition.client_id);
        }
        tokio::spawn(async move {
            let completion_error = completion.await.err().map(|error| error.to_string());
            let boundaries_valid = completion_error.is_none()
                && encode.output_row_indptr.first().copied() == Some(0)
                && encode
                    .output_row_indptr
                    .windows(2)
                    .all(|window| window[0] <= window[1])
                && encode
                    .output_row_indptr
                    .last()
                    .copied()
                    .is_some_and(|rows| rows as usize <= total_max_rows);

            for partition in partitions {
                let result = if let Some(error) = &completion_error {
                    Err(RemoteError::new(RemoteErrorKind::Driver, error.clone()))
                } else if !boundaries_valid {
                    Err(invalid(
                        "driver returned malformed embedding row boundaries",
                    ))
                } else {
                    let first = partition.image_start;
                    let last = first + partition.image_count;
                    let row_start = encode.output_row_indptr[first] as usize;
                    let row_end = encode.output_row_indptr[last] as usize;
                    if row_end - row_start > partition.max_rows {
                        Err(invalid(
                            "driver returned more embedding rows than request placeholders",
                        ))
                    } else {
                        let byte_start = row_start * hidden * 2;
                        let byte_end = row_end * hidden * 2;
                        Ok(ExecutorResponse::Embeddings(
                            pie_driver_abi::RemoteEmbeddings {
                                rows: encode.output_rows[byte_start..byte_end].to_vec(),
                                indptr: encode.output_row_indptr[first..=last]
                                    .iter()
                                    .map(|row| (row - row_start as u32) * hidden as u32 * 2)
                                    .collect(),
                                shapes: encode.output_row_indptr[first..=last]
                                    .windows(2)
                                    .flat_map(|window| [window[1] - window[0], hidden as u32])
                                    .collect(),
                                dtypes: vec![2; partition.image_count],
                                anchor_rows: partition.anchors,
                            },
                        ))
                    }
                };
                let _ = partition.reply.send(result);
                let _ = handle
                    .tx
                    .send(Command::Retired {
                        client_id: partition.client_id,
                    })
                    .await;
            }
        });
    }

    fn encode_one(
        &mut self,
        client_id: ClientId,
        request: pie_driver_abi::RemoteEncode,
        reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
        handle: ExecutorCoreHandle,
    ) {
        let plan = request.plan;
        if !plan.embed_rows.is_empty() {
            let response = validate_embeddings(&plan).map(|()| {
                ExecutorResponse::Embeddings(pie_driver_abi::RemoteEmbeddings {
                    rows: plan.embed_rows,
                    indptr: plan.embed_indptr,
                    shapes: plan.embed_shapes,
                    dtypes: plan.embed_dtypes,
                    anchor_rows: plan.embed_anchor_rows,
                })
            });
            let _ = reply.send(response);
            return;
        }
        if self.backend.kind() == "dummy" {
            let rows = plan.token_ids.len().max(1);
            let mut bytes = Vec::with_capacity(rows * std::mem::size_of::<u16>());
            for row in 0..rows {
                let value = plan.token_ids.get(row).copied().unwrap_or_default() as u16;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let _ = reply.send(Ok(ExecutorResponse::Embeddings(
                pie_driver_abi::RemoteEmbeddings {
                    rows: bytes,
                    indptr: vec![0, (rows * std::mem::size_of::<u16>()) as u32],
                    shapes: vec![rows as u32, 1],
                    dtypes: vec![2],
                    anchor_rows: vec![plan.image_anchor_rows.first().copied().unwrap_or(0)],
                },
            )));
            return;
        }
        if !self.capabilities.supports_media_encode
            || self.capabilities.hidden_size == 0
            || plan.image_anchor_rows.is_empty()
            || plan.token_ids.is_empty()
        {
            let _ = reply.send(Err(unsupported(
                "this backend does not expose standalone media encoding",
            )));
            return;
        }
        let hidden = self.capabilities.hidden_size as usize;
        let max_rows = plan.token_ids.len();
        let Some(output_bytes) = max_rows
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(2))
        else {
            let _ = reply.send(Err(invalid("media encode output size overflow")));
            return;
        };
        let anchors = plan.image_anchor_rows.clone();
        let mut encode = pie_driver_abi::MediaEncodePlan {
            image_grids: plan.image_grids,
            image_pixels: plan.image_pixels,
            image_pixel_indptr: plan.image_pixel_indptr,
            image_patch_positions: plan.image_patch_positions,
            image_anchor_rows: anchors.clone(),
            output_rows: vec![0; output_bytes],
            output_row_indptr: vec![0; anchors.len() + 1],
        };
        let completion = match self.backend.encode(&mut encode) {
            Ok(completion) => completion,
            Err(error) => {
                let _ = reply.send(Err(driver_error(error)));
                return;
            }
        };
        self.begin_inflight(client_id);
        tokio::spawn(async move {
            let result = match completion.await {
                Ok(()) => {
                    let valid = encode.output_row_indptr.first().copied() == Some(0)
                        && encode
                            .output_row_indptr
                            .windows(2)
                            .all(|window| window[0] <= window[1])
                        && encode
                            .output_row_indptr
                            .last()
                            .copied()
                            .is_some_and(|rows| rows as usize <= max_rows);
                    if !valid {
                        Err(invalid(
                            "driver returned malformed embedding row boundaries",
                        ))
                    } else {
                        let rows = *encode.output_row_indptr.last().unwrap() as usize;
                        encode.output_rows.truncate(rows * hidden * 2);
                        Ok(ExecutorResponse::Embeddings(
                            pie_driver_abi::RemoteEmbeddings {
                                rows: encode.output_rows,
                                indptr: encode
                                    .output_row_indptr
                                    .iter()
                                    .map(|row| row * (hidden as u32) * 2)
                                    .collect(),
                                shapes: encode
                                    .output_row_indptr
                                    .windows(2)
                                    .flat_map(|window| [window[1] - window[0], hidden as u32])
                                    .collect(),
                                dtypes: vec![2; anchors.len()],
                                anchor_rows: anchors,
                            },
                        ))
                    }
                }
                Err(error) => Err(driver_error(error)),
            };
            let _ = reply.send(result);
            let _ = handle.tx.send(Command::Retired { client_id }).await;
        });
    }

    fn push_kv(
        &mut self,
        client_id: ClientId,
        request: PushKv,
        reply: oneshot::Sender<std::result::Result<ExecutorResponse, RemoteError>>,
        _handle: ExecutorCoreHandle,
    ) {
        if request.src_page_ids.len() != request.dst_page_ids.len() {
            let _ = reply.send(Err(invalid(
                "KV push source/destination page counts differ",
            )));
            return;
        }
        let client = &self.clients[&client_id];
        if request.dst_worker != client.client_nonce {
            let _ = reply.send(Err(invalid(
                "KV push destination worker does not match Hello",
            )));
            return;
        }
        let Some(grant) = client.grant else {
            let _ = reply.send(Err(unsupported(
                "encode-only executors do not expose KV pages",
            )));
            return;
        };
        if request
            .src_page_ids
            .iter()
            .any(|&page| !grant.contains(page))
        {
            let _ = reply.send(Err(invalid(
                "KV push references a source page outside its grant",
            )));
            return;
        }
        #[cfg(feature = "nixl")]
        if client.transfer == Some(RemoteTransferKind::Nixl) {
            use pie_transport::Engine;

            let Some(nixl) = self.nixl.as_ref() else {
                let _ = reply.send(Err(unsupported("NIXL transfer was not initialized")));
                return;
            };
            let transfer = match nixl.engine.send_mapped(
                &nixl.local,
                &pie_transport::PageSet::new(request.src_page_ids),
                &pie_transport::PageSet::new(request.dst_page_ids),
                pie_transport::WorkerId(request.dst_worker),
            ) {
                Ok(transfer) => transfer,
                Err(error) => {
                    let _ = reply.send(Err(driver_error(error)));
                    return;
                }
            };
            let engine = Arc::clone(&nixl.engine);
            self.begin_inflight(client_id);
            tokio::spawn(async move {
                let result = loop {
                    match engine.poll(transfer) {
                        Ok(pie_transport::Completion::Done) => {
                            break Ok(ExecutorResponse::KvPushed);
                        }
                        Ok(pie_transport::Completion::Failed(message)) => {
                            break Err(driver_error(message));
                        }
                        Ok(pie_transport::Completion::Pending) => {
                            tokio::time::sleep(std::time::Duration::from_micros(50)).await;
                        }
                        Err(error) => break Err(driver_error(error)),
                    }
                };
                let _ = reply.send(result);
                let _ = _handle.tx.send(Command::Retired { client_id }).await;
            });
            return;
        }
        let Some(kv_handle) = self.kv_handle.as_ref() else {
            let _ = reply.send(Err(unsupported(
                "executor backend does not expose KV storage",
            )));
            return;
        };
        let result = (|| {
            if kv_handle.regions.is_empty() {
                return Err(unsupported(
                    "inline KV push requires at least one executor region",
                ));
            }
            let page_bytes = kv_handle.page_bytes();
            let total_bytes = page_bytes
                .checked_mul(request.src_page_ids.len() as u64)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| invalid("inline KV payload size overflow"))?;
            let mut bytes = Vec::with_capacity(total_bytes);
            for &page in &request.src_page_ids {
                for region in &kv_handle.regions {
                    let offset = (page as u64)
                        .checked_mul(region.page_stride)
                        .ok_or_else(|| invalid("inline KV source offset overflow"))?;
                    if region.page_stride == 0
                        || !offset
                            .checked_add(region.page_stride)
                            .is_some_and(|end| end <= region.len)
                    {
                        return Err(invalid(format!(
                            "inline KV source page {page} exceeds executor region"
                        )));
                    }
                    append_region_bytes(
                        region.domain,
                        region.base + offset,
                        region.page_stride as usize,
                        &mut bytes,
                    )
                    .map_err(driver_error)?;
                }
            }
            Ok(ExecutorResponse::KvPayload(InlineKvPayload {
                dst_page_ids: request.dst_page_ids,
                page_bytes,
                bytes,
            }))
        })();
        let _ = reply.send(result);
    }

    fn close_instance(
        &mut self,
        client_id: ClientId,
        instance_id: u64,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        let Some(instance) = self
            .clients
            .get_mut(&client_id)
            .expect("client exists")
            .instances
            .remove(&instance_id)
        else {
            return Err(invalid(format!(
                "instance {instance_id} is not owned by this client"
            )));
        };
        self.backend
            .close_instance(instance_id)
            .map_err(driver_error)?;
        instance.bound.close_wait_slots();
        self.clients
            .get_mut(&client_id)
            .expect("client exists")
            .local_instance_ids
            .remove(&instance.local_id);
        Ok(ExecutorResponse::Closed)
    }

    fn close_channel(
        &mut self,
        client_id: ClientId,
        channel_id: u64,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        let Some(channel) = self
            .clients
            .get_mut(&client_id)
            .expect("client exists")
            .channels
            .remove(&channel_id)
        else {
            return Err(invalid(format!(
                "channel {channel_id} is not owned by this client"
            )));
        };
        self.backend
            .close_channel(channel_id)
            .map_err(driver_error)?;
        let table = pie_engine::driver::waker::WakerTable::global();
        for wait_id in [
            channel.registered.reader_wait_id,
            channel.registered.writer_wait_id,
        ] {
            table.free(wait_id);
        }
        self.clients
            .get_mut(&client_id)
            .expect("client exists")
            .local_channel_ids
            .remove(&channel.local_id);
        Ok(ExecutorResponse::Closed)
    }

    fn begin_inflight(&mut self, client_id: ClientId) {
        self.clients
            .get_mut(&client_id)
            .expect("client exists")
            .inflight += 1;
        self.stats.inflight.fetch_add(1, Ordering::Relaxed);
    }

    fn retired(&mut self, client_id: ClientId) {
        self.stats.inflight.fetch_sub(1, Ordering::Relaxed);
        let Some(client) = self.clients.get_mut(&client_id) else {
            return;
        };
        client.inflight = client.inflight.saturating_sub(1);
        if client.disconnecting && client.inflight == 0 {
            self.cleanup_client(client_id);
        }
    }

    fn disconnect(&mut self, client_id: ClientId) {
        let Some(client) = self.clients.get_mut(&client_id) else {
            return;
        };
        client.disconnecting = true;
        if client.inflight == 0 {
            self.cleanup_client(client_id);
        }
    }

    fn cleanup_client(&mut self, client_id: ClientId) {
        let Some(client) = self.clients.remove(&client_id) else {
            return;
        };
        for (instance_id, instance) in client.instances {
            if let Err(error) = self.backend.close_instance(instance_id) {
                tracing::warn!(client_id, instance_id, %error, "closing leased executor instance");
            }
            instance.bound.close_wait_slots();
        }
        for (channel_id, channel) in client.channels {
            if let Err(error) = self.backend.close_channel(channel_id) {
                tracing::warn!(client_id, channel_id, %error, "closing leased executor channel");
            }
            let table = pie_engine::driver::waker::WakerTable::global();
            for wait_id in [
                channel.registered.reader_wait_id,
                channel.registered.writer_wait_id,
            ] {
                table.free(wait_id);
            }
        }
        if let Some(slot) = client.lease_slot {
            self.lease_slots[slot] = false;
        }
        if let Some(grant) = client.grant {
            self.stats
                .leased_pages
                .fetch_sub(grant.num_pages, Ordering::Relaxed);
        }
    }
}

fn append_csr(destination: &mut Vec<u32>, source: &[u32], base: u32, label: &str) -> Result<()> {
    if source.is_empty() {
        return Ok(());
    }
    anyhow::ensure!(source[0] == 0, "{label} must start at zero");
    anyhow::ensure!(
        source.windows(2).all(|window| window[0] <= window[1]),
        "{label} must be monotonic"
    );
    if destination.is_empty() {
        destination.push(0);
    }
    for &value in source.iter().skip(1) {
        destination.push(
            base.checked_add(value)
                .with_context(|| format!("{label} offset overflow"))?,
        );
    }
    Ok(())
}

fn valid_optional_csr(indptr: &[u32], rows: usize, values: usize) -> bool {
    indptr.is_empty()
        || (indptr.len() == rows + 1
            && indptr.first().copied() == Some(0)
            && indptr.last().copied() == Some(values as u32)
            && indptr.windows(2).all(|window| window[0] <= window[1]))
}

fn append_plan(
    destination: &mut pie_driver_abi::LaunchPlan,
    source: pie_driver_abi::LaunchPlan,
) -> Result<()> {
    anyhow::ensure!(
        source.kv_len_device.is_empty(),
        "coalescing device-resident KV lengths is unsupported"
    );
    let token_base = u32::try_from(destination.token_ids.len()).context("token offset overflow")?;
    let page_base =
        u32::try_from(destination.kv_page_indices.len()).context("page offset overflow")?;
    let rs_buffer_base =
        u32::try_from(destination.rs_buffer_slot_ids.len()).context("RS offset overflow")?;
    let mask_base = u32::try_from(destination.masks.len()).context("mask offset overflow")?;
    let sampling_base =
        u32::try_from(destination.sampling_indices.len()).context("sampling offset overflow")?;
    let image_base =
        u32::try_from(destination.image_grids.len() / 3).context("image offset overflow")?;
    let pixel_base =
        u32::try_from(destination.image_pixels.len()).context("pixel offset overflow")?;
    let mrope_base =
        u32::try_from(destination.image_mrope_positions.len()).context("mrope offset overflow")?;
    let audio_feature_base =
        u32::try_from(destination.audio_features.len()).context("audio offset overflow")?;
    let audio_base =
        u32::try_from(destination.audio_anchor_rows.len()).context("audio row overflow")?;
    let embed_byte_base =
        u32::try_from(destination.embed_rows.len()).context("embedding byte offset overflow")?;
    let embed_block_base =
        u32::try_from(destination.embed_dtypes.len()).context("embedding block overflow")?;

    append_csr(
        &mut destination.kv_page_indptr,
        &source.kv_page_indptr,
        page_base,
        "kv_page_indptr",
    )?;
    append_csr(
        &mut destination.qo_indptr,
        &source.qo_indptr,
        token_base,
        "qo_indptr",
    )?;
    append_csr(
        &mut destination.rs_buffer_slot_indptr,
        &source.rs_buffer_slot_indptr,
        rs_buffer_base,
        "rs_buffer_slot_indptr",
    )?;
    append_csr(
        &mut destination.mask_indptr,
        &source.mask_indptr,
        mask_base,
        "mask_indptr",
    )?;
    append_csr(
        &mut destination.sampling_indptr,
        &source.sampling_indptr,
        sampling_base,
        "sampling_indptr",
    )?;
    append_csr(
        &mut destination.image_indptr,
        &source.image_indptr,
        image_base,
        "image_indptr",
    )?;
    append_csr(
        &mut destination.image_pixel_indptr,
        &source.image_pixel_indptr,
        pixel_base,
        "image_pixel_indptr",
    )?;
    append_csr(
        &mut destination.image_mrope_indptr,
        &source.image_mrope_indptr,
        mrope_base,
        "image_mrope_indptr",
    )?;
    append_csr(
        &mut destination.audio_feature_indptr,
        &source.audio_feature_indptr,
        audio_feature_base,
        "audio_feature_indptr",
    )?;
    append_csr(
        &mut destination.audio_indptr,
        &source.audio_indptr,
        audio_base,
        "audio_indptr",
    )?;
    append_csr(
        &mut destination.embed_indptr,
        &source.embed_indptr,
        embed_byte_base,
        "embed_indptr",
    )?;
    append_csr(
        &mut destination.embed_block_indptr,
        &source.embed_block_indptr,
        embed_block_base,
        "embed_block_indptr",
    )?;

    destination.token_ids.extend(source.token_ids);
    destination.position_ids.extend(source.position_ids);
    destination.kv_page_indices.extend(source.kv_page_indices);
    destination
        .kv_last_page_lens
        .extend(source.kv_last_page_lens);
    destination.rs_slot_ids.extend(source.rs_slot_ids);
    destination.rs_slot_flags.extend(source.rs_slot_flags);
    destination.rs_fold_lens.extend(source.rs_fold_lens);
    destination
        .rs_buffer_slot_ids
        .extend(source.rs_buffer_slot_ids);
    destination.masks.extend(source.masks);
    destination
        .sampling_indices
        .extend(source.sampling_indices.into_iter());
    destination.context_ids.extend(source.context_ids);
    destination.single_token_mode &= source.single_token_mode;
    destination.has_user_mask |= source.has_user_mask;

    destination.image_grids.extend(source.image_grids);
    destination
        .image_anchor_positions
        .extend(source.image_anchor_positions);
    destination.image_pixels.extend(source.image_pixels);
    destination
        .image_mrope_positions
        .extend(source.image_mrope_positions);
    destination
        .image_patch_positions
        .extend(source.image_patch_positions);
    destination.image_anchor_rows.extend(
        source
            .image_anchor_rows
            .into_iter()
            .map(|row| token_base + row),
    );
    destination.audio_features.extend(source.audio_features);
    destination.audio_anchor_rows.extend(
        source
            .audio_anchor_rows
            .into_iter()
            .map(|row| token_base + row),
    );
    destination.embed_rows.extend(source.embed_rows);
    destination.embed_shapes.extend(source.embed_shapes);
    destination.embed_dtypes.extend(source.embed_dtypes);
    destination.embed_anchor_rows.extend(
        source
            .embed_anchor_rows
            .into_iter()
            .map(|row| token_base + row),
    );
    destination.kv_len.extend(source.kv_len);
    destination.kv_translation.extend(source.kv_translation);
    destination.kv_translation_version = destination
        .kv_translation_version
        .max(source.kv_translation_version);
    destination
        .channel_expected_head
        .extend(source.channel_expected_head);
    destination
        .channel_expected_tail
        .extend(source.channel_expected_tail);
    Ok(())
}

fn merge_remote_launches(launches: Vec<RemoteLaunch>) -> Result<RemoteLaunch> {
    anyhow::ensure!(!launches.is_empty(), "cannot merge an empty launch set");
    let mut merged = RemoteLaunch {
        plan: pie_driver_abi::LaunchPlan {
            single_token_mode: true,
            ..Default::default()
        },
        instance_ids: Vec::new(),
        terminal_count: 0,
        kv_translation: Vec::new(),
        kv_translation_indptr: vec![0],
        program_row_indptr: vec![0],
        logical_fire_ids: Vec::new(),
        channel_expected_head: Vec::new(),
        channel_expected_tail: Vec::new(),
        channel_ticket_indptr: vec![0],
    };
    for launch in launches {
        anyhow::ensure!(
            launch.kv_translation_indptr.len() == launch.instance_ids.len() + 1,
            "kv_translation_indptr does not partition instances"
        );
        anyhow::ensure!(
            launch.program_row_indptr.len() == launch.instance_ids.len() + 1,
            "program_row_indptr does not partition instances"
        );
        anyhow::ensure!(
            launch.channel_ticket_indptr.len() == launch.terminal_count as usize + 1,
            "channel_ticket_indptr does not partition terminal rows"
        );
        let translation_base =
            u32::try_from(merged.kv_translation.len()).context("translation offset overflow")?;
        let row_base = u32::try_from(merged.plan.qo_indptr.len().saturating_sub(1))
            .context("program row offset overflow")?;
        let ticket_base = u32::try_from(merged.channel_expected_head.len())
            .context("channel ticket offset overflow")?;
        append_csr(
            &mut merged.kv_translation_indptr,
            &launch.kv_translation_indptr,
            translation_base,
            "kv_translation_indptr",
        )?;
        append_csr(
            &mut merged.program_row_indptr,
            &launch.program_row_indptr,
            row_base,
            "program_row_indptr",
        )?;
        append_csr(
            &mut merged.channel_ticket_indptr,
            &launch.channel_ticket_indptr,
            ticket_base,
            "channel_ticket_indptr",
        )?;
        merged.instance_ids.extend(launch.instance_ids);
        merged.terminal_count = merged
            .terminal_count
            .checked_add(launch.terminal_count)
            .context("terminal count overflow")?;
        merged.kv_translation.extend(launch.kv_translation);
        merged.logical_fire_ids.extend(launch.logical_fire_ids);
        merged
            .channel_expected_head
            .extend(launch.channel_expected_head);
        merged
            .channel_expected_tail
            .extend(launch.channel_expected_tail);
        append_plan(&mut merged.plan, launch.plan)?;
    }
    Ok(merged)
}

fn validate_embeddings(plan: &pie_driver_abi::LaunchPlan) -> std::result::Result<(), RemoteError> {
    let blocks = plan.embed_dtypes.len();
    if plan.embed_indptr.len() != blocks + 1
        || plan.embed_shapes.len() != blocks * 2
        || plan.embed_anchor_rows.len() != blocks
        || plan.embed_indptr.first().copied() != Some(0)
        || plan.embed_indptr.last().copied() != Some(plan.embed_rows.len() as u32)
        || !plan
            .embed_indptr
            .windows(2)
            .all(|window| window[0] <= window[1])
    {
        return Err(invalid("precomputed embedding metadata is inconsistent"));
    }

    for block in 0..blocks {
        if plan.embed_dtypes[block] != 2 {
            return Err(invalid("precomputed embeddings currently require bf16"));
        }
        let element_bytes = 2usize;
        let rows = plan.embed_shapes[2 * block] as usize;
        let width = plan.embed_shapes[2 * block + 1] as usize;
        let start = plan.embed_indptr[block] as usize;
        let end = plan.embed_indptr[block + 1] as usize;
        let expected = rows
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| invalid("embedding shape size overflow"))?;
        if rows == 0 || width == 0 || end - start != expected {
            return Err(invalid("embedding byte length does not match its shape"));
        }
    }
    Ok(())
}

fn validate_raw_encode_plan(
    plan: &pie_driver_abi::LaunchPlan,
    hidden_size: u32,
) -> std::result::Result<(), RemoteError> {
    let images = plan.image_anchor_rows.len();
    if hidden_size == 0
        || plan.token_ids.is_empty()
        || images == 0
        || plan.image_grids.len() != images.saturating_mul(3)
        || plan.image_pixel_indptr.len() != images + 1
        || plan.image_pixel_indptr.first().copied() != Some(0)
        || plan.image_pixel_indptr.last().copied() != u32::try_from(plan.image_pixels.len()).ok()
        || plan.image_pixels.is_empty()
        || plan.image_pixels.len() % std::mem::size_of::<f32>() != 0
        || plan.image_patch_positions.is_empty()
        || plan.image_patch_positions.len() % 2 != 0
        || !plan.image_pixel_indptr.windows(2).all(|window| {
            window[0] <= window[1]
                && window[0] % std::mem::size_of::<f32>() as u32 == 0
                && window[1] % std::mem::size_of::<f32>() as u32 == 0
        })
        || plan
            .image_anchor_rows
            .iter()
            .any(|&row| row as usize >= plan.token_ids.len())
        || !plan.audio_features.is_empty()
    {
        return Err(invalid("media encode plan is malformed or unsupported"));
    }
    Ok(())
}

fn read_terminal(cell: &PieTerminalCell) -> TerminalCellState {
    let outcome = unsafe {
        std::sync::atomic::AtomicU32::from_ptr(std::ptr::addr_of!(cell.outcome).cast_mut())
            .load(Ordering::Acquire)
    };
    TerminalCellState {
        outcome,
        reserved0: cell.reserved0,
    }
}

fn invalid(message: impl Into<String>) -> RemoteError {
    RemoteError::new(RemoteErrorKind::InvalidRequest, message)
}

fn incompatible(message: impl Into<String>) -> RemoteError {
    RemoteError::new(RemoteErrorKind::Incompatible, message)
}

fn unsupported(message: impl Into<String>) -> RemoteError {
    RemoteError::new(RemoteErrorKind::Unsupported, message)
}

fn disconnected(message: impl Into<String>) -> RemoteError {
    RemoteError::new(RemoteErrorKind::Disconnected, message)
}

fn driver_error(error: impl std::fmt::Display) -> RemoteError {
    RemoteError::new(RemoteErrorKind::Driver, error.to_string())
}

#[allow(dead_code)]
fn _assert_tarpc_types(_: ExecutorRpcRequest, _: ExecutorRpcResponse) {}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_driver_abi::{
        EncodedMask, ExecutorRpcClient, LaunchPlan, ModelComponent, ProgramRegistration,
    };
    use pie_driver_dummy_lib::DummyDriverOptions;
    use pie_engine::driver::{DriverBackend, DummyDriver};
    use pie_ptir::container::{StageProgram, TraceContainer};
    use pie_ptir::registry::Stage;

    fn fixture(
        max_clients: usize,
        callback_delay_ms: u64,
    ) -> (ModelDrivers, ModelIdentity, pie_driver_abi::KvLayout) {
        fixture_with_log(max_clients, callback_delay_ms, None)
    }

    fn fixture_with_log(
        max_clients: usize,
        callback_delay_ms: u64,
        operation_log: Option<Arc<std::sync::Mutex<Vec<String>>>>,
    ) -> (ModelDrivers, ModelIdentity, pie_driver_abi::KvLayout) {
        let options = DummyDriverOptions {
            total_pages: 32,
            kv_page_size: 16,
            max_forward_tokens: 64,
            max_forward_requests: 8,
            max_page_refs: 128,
            callback_delay_ms,
            operation_log,
            ..DummyDriverOptions::default()
        };
        let backend = DriverBackend::Dummy(DummyDriver::new(options));
        let caps = match &backend {
            DriverBackend::Dummy(driver) => driver.capabilities().clone(),
            _ => unreachable!(),
        };
        let layout = backend.export_kv_handle().expect("dummy exports KV").layout;
        let drivers = ModelDrivers {
            groups: vec![crate::translate::GroupDriver { caps, backend }],
        };
        let model = ModelIdentity {
            hash: [7; 32],
            component: ModelComponent::Full,
        };
        assert!(max_clients > 0);
        (drivers, model, layout)
    }

    async fn rpc(
        client: &ExecutorRpcClient,
        request: ExecutorRequest,
    ) -> std::result::Result<ExecutorResponse, RemoteError> {
        client
            .execute(tarpc::context::current(), request)
            .await
            .expect("executor transport")
    }

    async fn hello(
        client: &ExecutorRpcClient,
        model: &ModelIdentity,
        layout: &pie_driver_abi::KvLayout,
        nonce: u64,
    ) -> HelloResponse {
        let response = rpc(
            client,
            ExecutorRequest::Hello(HelloRequest {
                wire_version: REMOTE_WIRE_VERSION,
                client_nonce: nonce,
                model: model.clone(),
                kv_layout: layout.clone(),
                peer_conn: None,
            }),
        )
        .await
        .expect("hello accepted");
        let ExecutorResponse::Hello(response) = response else {
            panic!("unexpected hello response {response:?}");
        };
        response
    }

    fn empty_program() -> ProgramRegistration {
        let bytes = TraceContainer {
            names: Vec::new(),
            externs: Vec::new(),
            channels: Vec::new(),
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: Vec::new(),
            }],
        }
        .encode();
        ProgramRegistration {
            program_hash: pie_ptir::container_hash(&bytes),
            canonical_bytes: bytes,
            sidecar_bytes: Vec::new(),
        }
    }

    fn launch(instance_id: u64) -> RemoteLaunch {
        RemoteLaunch {
            plan: LaunchPlan {
                token_ids: vec![1],
                position_ids: vec![0],
                kv_page_indptr: vec![0, 0],
                kv_last_page_lens: vec![0],
                qo_indptr: vec![0, 1],
                masks: vec![EncodedMask::new(vec![0, 1], 1)],
                mask_indptr: vec![0, 1],
                sampling_indices: vec![0],
                sampling_indptr: vec![0, 1],
                single_token_mode: true,
                ..LaunchPlan::default()
            },
            instance_ids: vec![instance_id],
            terminal_count: 1,
            kv_translation: Vec::new(),
            kv_translation_indptr: vec![0, 0],
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![1],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_client_rpc_mints_distinct_instances_and_executes() {
        let (drivers, model, layout) = fixture(2, 0);
        let server = ExecutorServer::bind("127.0.0.1:0", drivers, model.clone(), 2)
            .await
            .unwrap();
        let a = connect(server.endpoint()).await.unwrap();
        let b = connect(server.endpoint()).await.unwrap();
        let grant_a = hello(&a, &model, &layout, 1).await.grant;
        let grant_b = hello(&b, &model, &layout, 2).await.grant;
        assert_ne!(grant_a.base_page, grant_b.base_page);
        assert_eq!(grant_a.num_pages, grant_b.num_pages);
        let ExecutorResponse::KvPayload(payload) = rpc(
            &a,
            ExecutorRequest::PushKv(PushKv {
                src_page_ids: vec![grant_a.base_page],
                dst_page_ids: vec![3],
                dst_worker: 1,
            }),
        )
        .await
        .unwrap() else {
            panic!("KV payload response");
        };
        assert_eq!(payload.dst_page_ids, vec![3]);
        assert_eq!(payload.bytes.len() as u64, layout.page_bytes());
        let ExecutorResponse::Embeddings(embeddings) = rpc(
            &a,
            ExecutorRequest::Encode(pie_driver_abi::RemoteEncode {
                plan: pie_driver_abi::LaunchPlan {
                    token_ids: vec![4, 5],
                    image_pixels: vec![1, 2, 3, 4],
                    image_grids: vec![1, 1, 1],
                    image_pixel_indptr: vec![0, 4],
                    image_patch_positions: vec![0, 0],
                    image_anchor_rows: vec![0],
                    ..Default::default()
                },
            }),
        )
        .await
        .unwrap() else {
            panic!("embedding response");
        };
        assert_eq!(embeddings.shapes, vec![2, 1]);
        assert_eq!(embeddings.dtypes, vec![2]);
        assert_eq!(embeddings.rows.len(), 4);

        let registration = empty_program();
        let ExecutorResponse::ProgramRegistered(program_a) =
            rpc(&a, ExecutorRequest::RegisterProgram(registration.clone()))
                .await
                .unwrap()
        else {
            panic!("program response");
        };
        let ExecutorResponse::ProgramRegistered(program_b) =
            rpc(&b, ExecutorRequest::RegisterProgram(registration))
                .await
                .unwrap()
        else {
            panic!("program response");
        };
        assert_eq!(program_a, program_b, "program cache is shared");
        let ExecutorResponse::ChannelRegistered(channel) = rpc(
            &a,
            ExecutorRequest::RegisterChannel(RemoteRegisterChannel {
                local_channel_id: 77,
                shape: vec![1],
                dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                host_role: pie_driver_abi::PIE_CHANNEL_HOST_ROLE_NONE,
                seeded: false,
                extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                capacity: 1,
                extern_name: Vec::new(),
            }),
        )
        .await
        .unwrap() else {
            panic!("channel response");
        };
        let error = rpc(
            &b,
            ExecutorRequest::BindInstance(RemoteBindInstance {
                local_instance_id: 99,
                program_id: program_b,
                channel_ids: vec![channel.executor_channel_id],
                seed_values: Vec::new(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::InvalidRequest);

        let bind = |local_instance_id, program_id| {
            ExecutorRequest::BindInstance(RemoteBindInstance {
                local_instance_id,
                program_id,
                channel_ids: Vec::new(),
                seed_values: Vec::new(),
            })
        };
        let ExecutorResponse::InstanceBound(bound_a) = rpc(&a, bind(1, program_a)).await.unwrap()
        else {
            panic!("bind response");
        };
        let ExecutorResponse::InstanceBound(bound_b) = rpc(&b, bind(1, program_b)).await.unwrap()
        else {
            panic!("bind response");
        };
        assert_ne!(
            bound_a.executor_instance_id, bound_b.executor_instance_id,
            "executor-minted ids must not collide across clients"
        );
        let mut escaped = launch(bound_b.executor_instance_id);
        escaped.plan.kv_page_indices = vec![grant_a.base_page];
        escaped.plan.kv_page_indptr = vec![0, 1];
        let error = rpc(&b, ExecutorRequest::Launch(escaped)).await.unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::InvalidRequest);

        let ExecutorResponse::Terminal(terminal) = rpc(
            &a,
            ExecutorRequest::Launch(launch(bound_a.executor_instance_id)),
        )
        .await
        .unwrap() else {
            panic!("launch response");
        };
        assert_eq!(terminal.per_request.len(), 1);
        assert_eq!(
            terminal.per_request[0].outcome,
            PIE_TERMINAL_OUTCOME_SUCCESS
        );

        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hello_rejects_mismatch_and_enforces_client_limit() {
        let (drivers, model, layout) = fixture(1, 0);
        let server = ExecutorServer::bind("127.0.0.1:0", drivers, model.clone(), 1)
            .await
            .unwrap();
        let bad = connect(server.endpoint()).await.unwrap();
        let error = rpc(&bad, ExecutorRequest::Launch(launch(1)))
            .await
            .unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::InvalidRequest);
        let error = rpc(
            &bad,
            ExecutorRequest::Hello(HelloRequest {
                wire_version: REMOTE_WIRE_VERSION + 1,
                client_nonce: 1,
                model: model.clone(),
                kv_layout: layout.clone(),
                peer_conn: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::Incompatible);

        let a = connect(server.endpoint()).await.unwrap();
        hello(&a, &model, &layout, 2).await;
        let mut malformed = launch(1);
        malformed.program_row_indptr = vec![0, 2];
        let error = rpc(&a, ExecutorRequest::Launch(malformed))
            .await
            .unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::InvalidRequest);
        let b = connect(server.endpoint()).await.unwrap();
        let error = rpc(
            &b,
            ExecutorRequest::Hello(HelloRequest {
                wire_version: REMOTE_WIRE_VERSION,
                client_nonce: 2,
                model: model.clone(),
                kv_layout: layout.clone(),
                peer_conn: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::InvalidRequest);
        let error = rpc(
            &b,
            ExecutorRequest::Hello(HelloRequest {
                wire_version: REMOTE_WIRE_VERSION,
                client_nonce: 3,
                model,
                kv_layout: layout,
                peer_conn: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, RemoteErrorKind::ResourceExhausted);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_backend_bridges_terminal_cells() {
        let (drivers, model, layout) = fixture(1, 0);
        let server = ExecutorServer::bind("127.0.0.1:0", drivers, model.clone(), 1)
            .await
            .unwrap();
        let client = connect(server.endpoint()).await.unwrap();
        let hello = hello(&client, &model, &layout, 1).await;
        let runtime = tokio::runtime::Handle::current();
        let program = empty_program();
        let (done_tx, done_rx) = oneshot::channel();

        std::thread::spawn(move || {
            let mut remote = pie_engine::driver::RemoteDriver::new(
                client,
                runtime.clone(),
                hello.capabilities,
                hello.grant,
            );
            let program_id = remote.register_program(&program).unwrap();
            let pacing_wait_id = pie_engine::driver::waker::WakerTable::global().alloc();
            let bound = remote
                .bind_instance(&InstanceBindingPlan {
                    driver_id: 7,
                    program_id,
                    requested_instance_id: 0,
                    pacing_wait_id,
                    channel_ids: Vec::new(),
                    seed_values: Vec::new(),
                })
                .unwrap();
            let mut terminal = Box::new(PieTerminalCell {
                outcome: PIE_TERMINAL_OUTCOME_PENDING,
                reserved0: 0,
            });
            let wire = launch(bound.instance_id);
            let completion = remote
                .launch(&LaunchSubmission {
                    plan: wire.plan,
                    instance_ids: wire.instance_ids,
                    terminal_cells: vec![terminal.as_mut()],
                    kv_translation: wire.kv_translation,
                    kv_translation_indptr: wire.kv_translation_indptr,
                    program_row_indptr: wire.program_row_indptr,
                    logical_fire_ids: wire.logical_fire_ids,
                    channel_expected_head: wire.channel_expected_head,
                    channel_expected_tail: wire.channel_expected_tail,
                    channel_ticket_indptr: wire.channel_ticket_indptr,
                })
                .unwrap();
            runtime.block_on(completion).unwrap();
            let outcome = read_terminal(&terminal).outcome;
            remote.close_instance(bound.instance_id).unwrap();
            bound.close_wait_slots();
            let _ = done_tx.send(outcome);
        });

        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), done_rx)
                .await
                .unwrap()
                .unwrap(),
            PIE_TERMINAL_OUTCOME_SUCCESS
        );
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn executor_loss_poisoned_remote_completion_without_hang() {
        let (drivers, model, layout) = fixture(1, 500);
        let server = ExecutorServer::bind("127.0.0.1:0", drivers, model.clone(), 1)
            .await
            .unwrap();
        let client = connect(server.endpoint()).await.unwrap();
        let hello = hello(&client, &model, &layout, 1).await;
        let runtime = tokio::runtime::Handle::current();
        let program = empty_program();
        let (accepted_tx, accepted_rx) = oneshot::channel();
        let (done_tx, done_rx) = oneshot::channel();

        std::thread::spawn(move || {
            let mut remote = pie_engine::driver::RemoteDriver::new(
                client,
                runtime.clone(),
                hello.capabilities,
                hello.grant,
            );
            let program_id = remote.register_program(&program).unwrap();
            let pacing_wait_id = pie_engine::driver::waker::WakerTable::global().alloc();
            let bound = remote
                .bind_instance(&InstanceBindingPlan {
                    driver_id: 9,
                    program_id,
                    requested_instance_id: 0,
                    pacing_wait_id,
                    channel_ids: Vec::new(),
                    seed_values: Vec::new(),
                })
                .unwrap();
            let mut terminal = Box::new(PieTerminalCell {
                outcome: PIE_TERMINAL_OUTCOME_PENDING,
                reserved0: 0,
            });
            let wire = launch(bound.instance_id);
            let completion = remote
                .launch(&LaunchSubmission {
                    plan: wire.plan,
                    instance_ids: wire.instance_ids,
                    terminal_cells: vec![terminal.as_mut()],
                    kv_translation: wire.kv_translation,
                    kv_translation_indptr: wire.kv_translation_indptr,
                    program_row_indptr: wire.program_row_indptr,
                    logical_fire_ids: wire.logical_fire_ids,
                    channel_expected_head: wire.channel_expected_head,
                    channel_expected_tail: wire.channel_expected_tail,
                    channel_ticket_indptr: wire.channel_ticket_indptr,
                })
                .unwrap();
            let _ = accepted_tx.send(());
            let result = runtime
                .block_on(completion)
                .map_err(|error| error.to_string());
            bound.close_wait_slots();
            let _ = done_tx.send(result);
        });

        accepted_rx.await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        server.shutdown().await;
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), done_rx)
            .await
            .expect("remote completion must not hang")
            .unwrap();
        assert!(result.is_err(), "executor loss must poison completion");
    }

    #[test]
    fn remote_launch_merge_rebases_csrs_and_rows() {
        let mut first = launch(11);
        first.plan.token_ids = vec![1, 9];
        first.plan.position_ids = vec![0, 1];
        first.plan.qo_indptr = vec![0, 2];
        first.plan.sampling_indices = vec![1];
        let mut second = launch(22);
        second.plan.token_ids = vec![2, 3];
        second.plan.position_ids = vec![4, 5];
        second.plan.qo_indptr = vec![0, 2];
        second.plan.sampling_indices = vec![1];
        second.plan.sampling_indptr = vec![0, 1];
        second.plan.mask_indptr = vec![0, 1];
        second.logical_fire_ids = vec![2];
        let merged = merge_remote_launches(vec![first, second]).unwrap();
        assert_eq!(merged.instance_ids, vec![11, 22]);
        assert_eq!(merged.terminal_count, 2);
        assert_eq!(merged.plan.token_ids, vec![1, 9, 2, 3]);
        assert_eq!(merged.plan.qo_indptr, vec![0, 2, 4]);
        assert_eq!(merged.plan.sampling_indices, vec![1, 1]);
        assert_eq!(merged.program_row_indptr, vec![0, 1, 2]);
        assert_eq!(merged.logical_fire_ids, vec![1, 2]);
    }

    #[test]
    fn grant_zeroing_clears_only_the_leased_range() {
        let mut bytes = vec![0xAAu8; 24];
        let handle = pie_driver_abi::KvHandle {
            regions: vec![pie_driver_abi::KvRegion {
                base: bytes.as_mut_ptr() as u64,
                len: bytes.len() as u64,
                page_stride: 8,
                domain: MemoryDomain::HostPinned,
            }],
            layout: pie_driver_abi::KvLayout {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 1,
                page_size: 1,
                dtype: pie_driver_abi::KvDtype::I8,
                kind: pie_driver_abi::KvLayoutKind::FusedLatent,
                storage_format: "test".to_string(),
                region_page_bytes: vec![8],
            },
        };
        zero_grant(
            &handle,
            ScratchGrant {
                base_page: 1,
                num_pages: 1,
            },
        )
        .unwrap();
        assert_eq!(&bytes[..8], &[0xAA; 8]);
        assert_eq!(&bytes[8..16], &[0; 8]);
        assert_eq!(&bytes[16..], &[0xAA; 8]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dequeue_coalesces_already_queued_cross_client_launches() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (drivers, model, layout) = fixture_with_log(8, 10, Some(Arc::clone(&log)));
        let server = ExecutorServer::bind("127.0.0.1:0", drivers, model.clone(), 8)
            .await
            .unwrap();
        let mut clients = Vec::new();
        for nonce in 1..=8 {
            let client = connect(server.endpoint()).await.unwrap();
            hello(&client, &model, &layout, nonce).await;
            clients.push(client);
        }
        let registration = empty_program();
        let ExecutorResponse::ProgramRegistered(program_id) =
            rpc(&clients[0], ExecutorRequest::RegisterProgram(registration))
                .await
                .unwrap()
        else {
            panic!("program response");
        };
        let mut instances = Vec::new();
        for (index, client) in clients.iter().enumerate() {
            let ExecutorResponse::InstanceBound(binding) = rpc(
                client,
                ExecutorRequest::BindInstance(RemoteBindInstance {
                    local_instance_id: index as u64 + 1,
                    program_id,
                    channel_ids: Vec::new(),
                    seed_values: Vec::new(),
                }),
            )
            .await
            .unwrap() else {
                panic!("bind response");
            };
            instances.push(binding.executor_instance_id);
        }
        log.lock().unwrap().clear();
        let barrier = Arc::new(tokio::sync::Barrier::new(clients.len()));
        let launches = clients
            .into_iter()
            .zip(instances)
            .map(|(client, instance)| {
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    client
                        .execute(
                            tarpc::context::current(),
                            ExecutorRequest::Launch(launch(instance)),
                        )
                        .await
                        .unwrap()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for result in futures::future::join_all(launches).await {
            assert!(matches!(result.unwrap(), ExecutorResponse::Terminal(_)));
        }
        let launch_calls = log
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.as_str() == "launch")
            .count();
        assert!(
            launch_calls < 8,
            "queued cross-client launches were not coalesced: {launch_calls}"
        );
        server.shutdown().await;
    }
}
