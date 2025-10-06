pub mod batching;
pub mod request;
pub mod resource;
pub mod tokenizer;

use crate::instance::InstanceId;
use crate::model::batching::{BatchPolicySelector, BatchScheduler, ForwardPassPolicy};
use crate::model::request::{
    FORWARD_PASS_ID, HANDSHAKE_ID, HandshakeRequest, HandshakeResponse, HeartbeatRequest, Request,
};
use crate::model::resource::{ResourceId, ResourceManager, ResourceTypeId};
use crate::model::tokenizer::BytePairEncoder;
use crate::runtime::trap_exception;
use crate::service::{self, Service, ServiceError};
use anyhow::Result;
use anyhow::bail;
use bytes::Bytes;
use futures::future;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

pub type HandlerId = u32;

pub type CmdQueueId = u32;

static REGISTERED_MODELS: std::sync::LazyLock<boxcar::Vec<(String, usize)>> =
    std::sync::LazyLock::new(boxcar::Vec::new);

pub fn registered_models() -> Vec<String> {
    REGISTERED_MODELS
        .iter()
        .map(|(_, (model_name, _))| model_name.clone())
        .collect()
}

pub fn register_model(model_name: String, service_id: usize) {
    REGISTERED_MODELS.push((model_name, service_id));
}

pub fn model_service_id(model_name: &str) -> Option<usize> {
    REGISTERED_MODELS
        .iter()
        .find(|(_, (name, _))| name == model_name)
        .map(|(_, (_, service_id))| *service_id)
}

pub fn cleanup_instance(inst_id: InstanceId) {
    REGISTERED_MODELS.iter().for_each(|(_, (_, service_id))| {
        Command::Cleanup { inst_id }.dispatch(*service_id).ok();
    })
}

/// Asynchronously collects runtime statistics from all registered models.
pub async fn runtime_stats() -> HashMap<String, String> {
    let mut aggregated_stats = HashMap::new();
    let mut futures = Vec::new();

    // Dispatch requests to all models concurrently.
    for (_, (model_name, service_id)) in REGISTERED_MODELS.iter() {
        let (tx, rx) = oneshot::channel();
        let cmd = Command::GetRuntimeStats { response: tx };

        if cmd.dispatch(*service_id).is_ok() {
            // Store the model name and the future for its response.
            futures.push((model_name.clone(), rx));
        } else {
            // Handle cases where the service is unavailable immediately.
            aggregated_stats.insert(
                format!("{}.error", model_name),
                "failed to dispatch command to service".to_string(),
            );
        }
    }

    // Await all responses in parallel.
    let results = future::join_all(
        futures
            .into_iter()
            .map(async move |(name, rx)| (name, rx.await)),
    )
    .await;

    // Process the results.
    for (model_name, result) in results {
        match result {
            Ok(model_stats) => {
                for (key, value) in model_stats {
                    aggregated_stats.insert(format!("{}.{}", model_name, key), value);
                }
            }
            Err(e) => {
                // The service might have crashed or failed to respond.
                aggregated_stats.insert(
                    format!("{}.error", model_name),
                    format!("failed to receive stats from service: {}", e),
                );
            }
        }
    }

    aggregated_stats
}

pub fn submit_request(
    service_id: usize,
    cmd_queue_id: CmdQueueId,
    priority: u32,
    req: Request,
) -> Result<()> {
    Command::Submit {
        cmd_queue_id,
        priority,
        req,
    }
    .dispatch(service_id)?;
    Ok(())
}

/// Defines the set of operations available for the key-value store.
#[derive(Debug)]
pub enum Command {
    Submit {
        cmd_queue_id: CmdQueueId,
        priority: u32,
        req: Request,
    },
    GetInfo {
        response: oneshot::Sender<ModelInfo>,
    },
    GetRuntimeStats {
        response: oneshot::Sender<HashMap<String, String>>,
    },
    Allocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
        response: oneshot::Sender<Vec<ResourceId>>,
    },
    Deallocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
    },
    Cleanup {
        inst_id: InstanceId,
    },
    GetAllExported {
        type_id: ResourceTypeId,
        response: oneshot::Sender<Vec<(String, Vec<ResourceId>)>>,
    },
    Export {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        name: String,
    },
    Import {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        name: String,
        response: oneshot::Sender<Vec<ResourceId>>,
    },
    ReleaseExported {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        name: String,
    },
}

impl Command {
    pub fn dispatch(self, service_id: usize) -> Result<(), ServiceError> {
        service::dispatch(service_id, self)
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub traits: Vec<String>,
    pub description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub prompt_stop_tokens: Vec<String>,
    pub tokenizer: Arc<BytePairEncoder>,
    pub kv_page_size: u32,
    pub max_batch_tokens: usize,
}

/// An in-memory key-value store service.
#[derive(Debug)]
pub struct Model {
    info: ModelInfo,
    resource_manager: ResourceManager,
    shutdown_tx: broadcast::Sender<()>,
    scheduler_tx: mpsc::UnboundedSender<(CmdQueueId, u32, Request)>,
    scheduling_worker_handle: Option<JoinHandle<()>>,
    backend_worker_handle: Option<JoinHandle<()>>,
}

impl Model {
    pub async fn new(endpoint: &str) -> Result<Self> {
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;

        let handshake_info = Self::handshake(&mut socket).await?;

        let mut batch_triggers = HashMap::new();
        let forward_pass_trigger = Arc::new(AtomicBool::new(true));
        let forward_pass_policy = ForwardPassPolicy::new(
            forward_pass_trigger.clone(),
            handshake_info.max_batch_tokens,
            Duration::ZERO,
        );
        let batch_policy = BatchPolicySelector::new(forward_pass_policy);

        batch_triggers.insert(FORWARD_PASS_ID, forward_pass_trigger);

        let (backend_tx, backend_rx) = mpsc::unbounded_channel();
        let (scheduler_tx, scheduler_rx) = mpsc::unbounded_channel();

        // NEW: Shutdown channel for workers.
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        let backend_worker_handle = tokio::spawn(Self::backend_worker(
            socket,
            backend_rx,
            batch_triggers,
            shutdown_rx,
        ));
        let scheduling_worker_handle = tokio::spawn(Self::scheduling_worker(
            batch_policy,
            scheduler_rx,
            backend_tx,
            shutdown_tx.subscribe(),
        ));

        let tokenizer = Arc::new(BytePairEncoder::new(
            handshake_info.tokenizer_num_vocab,
            handshake_info.tokenizer_merge_table.into_iter().collect(),
            handshake_info.tokenizer_special_tokens,
            &handshake_info.tokenizer_split_regex,
            handshake_info.tokenizer_escape_non_printable,
        ));

        let info = ModelInfo {
            name: handshake_info.model_name,
            traits: handshake_info.model_traits,
            description: handshake_info.model_description,
            prompt_template: handshake_info.prompt_template,
            prompt_template_type: handshake_info.prompt_template_type,
            prompt_stop_tokens: handshake_info.prompt_stop_tokens,
            tokenizer,
            kv_page_size: handshake_info.kv_page_size,
            max_batch_tokens: handshake_info.max_batch_tokens,
        };

        let resource_manager = ResourceManager::new(handshake_info.resources);

        Ok(Model {
            info,
            resource_manager,
            scheduler_tx,
            shutdown_tx,
            scheduling_worker_handle: Some(scheduling_worker_handle),
            backend_worker_handle: Some(backend_worker_handle),
        })
    }

    async fn handshake(socket: &mut DealerSocket) -> Result<HandshakeResponse> {
        let req = Bytes::from(rmp_serde::to_vec_named(&HandshakeRequest {
            version: "0.1.0".to_string(),
        })?);

        Self::send_zmq_message(socket, 0, HANDSHAKE_ID, req).await?;
        let (corr_id, handler_id, mut frames) = Self::recv_zmq_messages(socket).await?;

        if corr_id != 0 {
            bail!("[Error] Invalid correlation ID in handshake response.");
        }

        if handler_id != HANDSHAKE_ID {
            bail!("[Error] Invalid handler ID in handshake response.");
        }

        let handshake_frame = frames
            .pop_front()
            .ok_or(anyhow::format_err!("Missing handshake frame"))?;
        let handshake_info = rmp_serde::from_slice::<HandshakeResponse>(&handshake_frame)?;

        Ok(handshake_info)
    }

    pub async fn shutdown(mut self) -> Result<()> {
        println!("[Info] Shutting down model service...");
        self.shutdown_tx.send(())?;

        if let Some(handle) = self.scheduling_worker_handle.take() {
            handle.await?;
        }
        if let Some(handle) = self.backend_worker_handle.take() {
            handle.await?;
        }
        println!("[Info] Model service shut down gracefully.");
        Ok(())
    }

    async fn scheduling_worker(
        batch_policy: BatchPolicySelector,
        mut req_rx: mpsc::UnboundedReceiver<(CmdQueueId, u32, Request)>,
        backend_tx: mpsc::UnboundedSender<Vec<Request>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut sched = BatchScheduler::new(batch_policy);
        let mut next_poll_duration: Option<Duration> = None;
        const IDLE_TIMEOUT: Duration = Duration::from_secs(1);

        loop {
            let sleep_duration = next_poll_duration.unwrap_or(IDLE_TIMEOUT);

            tokio::select! {
                _ = shutdown_rx.recv() => {
                    println!("[Info] Shutdown signal received, exiting scheduling worker.");
                    break;
                },
                maybe_msg = req_rx.recv() => {
                    if let Some((cmd_queue_id, priority, request )) = maybe_msg {
                        if request.is_eager() {
                            if backend_tx.send(vec![request]).is_err() {
                               eprintln!("[Warn] Backend channel closed, could not send non-batched message.");
                            }
                            continue;
                        }
                        sched.push(cmd_queue_id, priority, request, Instant::now());
                    } else {
                        println!("[Info] Command channel closed, shutting down scheduler handler.");
                        break;
                    }
                },
                _ = tokio::time::sleep(sleep_duration) => {}
            }

            let batches = sched.schedule(Instant::now());
            for batch in batches {
                if batch.first().unwrap().is_sync_req() {
                    if let Request::Synchronize(sender) = batch.into_iter().next().unwrap() {
                        sender.send(()).ok();
                    }
                } else {
                    backend_tx.send(batch).ok();
                }
            }
            next_poll_duration = sched.next_poll_in(Instant::now());
        }
    }

    async fn backend_worker(
        mut socket: DealerSocket,
        mut batch_rx: mpsc::UnboundedReceiver<Vec<Request>>,
        batch_triggers: HashMap<HandlerId, Arc<AtomicBool>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut corr_id: u32 = 0;
        let mut event_table: HashMap<(u32, usize), (Request, Instant)> = HashMap::new();
        const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
        const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
        const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(10);
        let mut heartbeat_interval = tokio::time::interval(HEARTBEAT_INTERVAL);
        let mut heartbeat_pending: Option<Instant> = None;
        // Use a special correlation ID to distinguish heartbeats from regular requests.
        let heartbeat_corr_id = u32::MAX;

        loop {
            let sleep_duration = event_table
                .values()
                .map(|(_, instant)| {
                    instant.saturating_duration_since(Instant::now()) + REQUEST_TIMEOUT
                })
                .min()
                .unwrap_or(REQUEST_TIMEOUT);

            tokio::select! {
                _ = shutdown_rx.recv() => {
                    println!("[Info] Shutdown signal received, exiting backend worker.");
                    break;
                },

                _ = heartbeat_interval.tick() => {
                    if let Some(sent_at) = heartbeat_pending {
                        if sent_at.elapsed() > HEARTBEAT_TIMEOUT {
                            eprintln!("[Warn] backend not responsive");
                        }
                    }

                    let heartbeat_req = Request::Heartbeat(HeartbeatRequest {});
                    let payload = heartbeat_req.serialize_req().unwrap();
                    let res = Self::send_zmq_message(
                        &mut socket,
                        heartbeat_corr_id,
                        heartbeat_req.handler_id(),
                        payload
                    ).await;

                    if let Err(e) = res {
                        eprintln!("[Error] Socket send failed for heartbeat: {:?}", e);
                    } else {
                        heartbeat_pending = Some(Instant::now());
                    }
                },

                maybe_command = batch_rx.recv() => {
                    if let Some(requests) = maybe_command {
                        let current_corr_id = corr_id;
                        let handler_id = requests.first().unwrap().handler_id();
                        let serialized:Vec<Bytes> = requests.iter().map(|request| request.serialize_req().unwrap()).collect();

                        let res = Self::send_zmq_messages(
                            &mut socket,
                            current_corr_id,
                            handler_id,
                            serialized
                        ).await;

                        if let Err(e) = res {
                            eprintln!("[Error] Socket send failed for corr_id {}: {:?}", current_corr_id, e);
                            continue;
                        }

                        for (idx, request) in requests.into_iter().enumerate() {
                            if request.has_response() {
                                event_table.insert((current_corr_id, idx), (request, Instant::now()));
                            }
                        }
                        corr_id = corr_id.wrapping_add(1);
                    } else {
                        println!("[Info] Command channel closed, shutting down backend handler.");
                        break;
                    }
                },
                result = Self::recv_zmq_messages(&mut socket) => {
                    match result {
                        Ok((received_corr_id, received_handler_id, frames)) => {

                            if received_corr_id == heartbeat_corr_id {
                                heartbeat_pending = None;
                                continue; // Skip further processing for heartbeats.
                            }

                            for (idx, payload) in frames.into_iter().enumerate() {
                                let key = (received_corr_id, idx);
                                if let Some((request, _)) = event_table.remove(&key) {
                                    let _ = request.deserialize_resp(payload);
                                }
                            }

                            if let Some(trigger) = batch_triggers.get(&received_handler_id) {
                                trigger.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        },
                        Err(e) => {
                            eprintln!("[Error] Socket receive error: {}. Shutting down.", e);
                            break;
                        }
                    }
                },
                _ = tokio::time::sleep(sleep_duration) => {
                    let now = Instant::now();
                    event_table.retain(|_key, (_, instant)| {
                        if now.duration_since(*instant) > REQUEST_TIMEOUT {
                            eprintln!("[Warn] Request timed out. Dropping sender.");
                            false
                        } else {
                            true
                        }
                    });
                }
            }
        }
    }

    pub fn submit(&self, cmd_queue_id: CmdQueueId, priority: u32, req: Request) {
        // FIX: Log error if the scheduler channel is closed.
        if self
            .scheduler_tx
            .send((cmd_queue_id, priority, req))
            .is_err()
        {
            eprintln!("[Error] Failed to submit command: Scheduler channel is closed.");
        }
    }

    /// Gathers detailed runtime statistics for this specific model instance.
    pub fn runtime_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("model.name".to_string(), self.info.name.clone());

        // Collect stats from the ResourceManager.
        self.resource_manager.append_stats_to(&mut stats);

        stats
    }

    async fn send_zmq_message(
        socket: &mut DealerSocket,
        corr_id: u32,
        handler_id: u32,
        payload: Bytes,
    ) -> Result<()> {
        let mut frames: VecDeque<Bytes> = VecDeque::with_capacity(3);
        frames.push_back(Bytes::copy_from_slice(&corr_id.to_be_bytes()));
        frames.push_back(Bytes::copy_from_slice(&handler_id.to_be_bytes()));
        frames.push_back(payload);
        socket.send(ZmqMessage::try_from(frames).unwrap()).await?;
        Ok(())
    }

    async fn send_zmq_messages(
        socket: &mut DealerSocket,
        corr_id: u32,
        handler_id: u32,
        payloads: Vec<Bytes>,
    ) -> Result<()> {
        let mut frames: VecDeque<Bytes> = VecDeque::with_capacity(3);
        frames.push_back(Bytes::copy_from_slice(&corr_id.to_be_bytes()));
        frames.push_back(Bytes::copy_from_slice(&handler_id.to_be_bytes()));
        frames.extend(payloads);
        socket.send(ZmqMessage::try_from(frames).unwrap()).await?;
        Ok(())
    }

    async fn recv_zmq_messages(socket: &mut DealerSocket) -> Result<(u32, u32, VecDeque<Bytes>)> {
        let zmq_msg = socket.recv().await?;

        let mut frames = zmq_msg.into_vecdeque();
        let corr_id_bytes = frames
            .pop_front()
            .ok_or(anyhow::format_err!("Missing correlation ID frame"))?;
        let handler_id_bytes = frames
            .pop_front()
            .ok_or(anyhow::format_err!("Missing handler ID frame"))?;
        let corr_id_slice = corr_id_bytes.as_ref().try_into()?;
        let handler_id_slice = handler_id_bytes.as_ref().try_into()?;

        let received_corr_id = u32::from_be_bytes(corr_id_slice);
        let received_handler_id = u32::from_be_bytes(handler_id_slice);

        Ok((received_corr_id, received_handler_id, frames))
    }
}

impl Service for Model {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Submit {
                cmd_queue_id,
                priority,
                req,
            } => {
                self.submit(cmd_queue_id, priority, req);
            }

            Command::GetInfo { response } => {
                if response.send(self.info.clone()).is_err() {
                    println!("[Warn] GetInfo response channel closed before sending.");
                }
            }

            Command::GetRuntimeStats { response } => {
                if response.send(self.runtime_stats()).is_err() {
                    println!("[Warn] GetRuntimeStats response channel closed before sending.");
                }
            }

            Command::Allocate {
                inst_id,
                type_id,
                count,
                response,
            } => match self
                .resource_manager
                .allocate_with_oom(inst_id, type_id, count)
            {
                Ok(allocated_ids) => {
                    if response.send(allocated_ids).is_err() {
                        println!("[Warn] Allocate response channel closed before sending.");
                    }
                }
                Err(e) => trap_exception(inst_id, e),
            },
            Command::Deallocate {
                inst_id,
                type_id,
                ptrs,
            } => {
                if let Err(e) = self.resource_manager.deallocate(inst_id, type_id, ptrs) {
                    trap_exception(inst_id, e);
                }
            }
            Command::Cleanup { inst_id } => {
                if let Err(e) = self.resource_manager.cleanup(inst_id) {
                    trap_exception(inst_id, e);
                }
            }
            Command::GetAllExported { type_id, response } => {
                let list = self.resource_manager.get_all_exported(type_id);
                if response.send(list).is_err() {
                    println!("[Warn] GetAllExported response channel closed before sending.");
                }
            }
            Command::Export {
                inst_id,
                type_id,
                ptrs,
                name,
            } => {
                if let Err(e) = self.resource_manager.export(inst_id, type_id, ptrs, name) {
                    trap_exception(inst_id, e);
                }
            }
            Command::Import {
                inst_id,
                type_id,
                name,
                response,
            } => match self.resource_manager.import(type_id, name) {
                Ok(ptrs) => {
                    if response.send(ptrs).is_err() {
                        println!("[Warn] Import response channel closed before sending.");
                    }
                }
                Err(e) => trap_exception(inst_id, e),
            },
            Command::ReleaseExported {
                inst_id,
                type_id,
                name,
            } => {
                if let Err(e) = self.resource_manager.release_exported(type_id, name) {
                    trap_exception(inst_id, e);
                }
            }
        }
    }
}
