pub mod actor;
pub mod batching;
pub mod chat_template;
pub mod ffi_bridge;
pub mod ffi_ipc;
pub mod request;
pub mod resource;
pub mod tokenizer;

use super::model::batching::SharedScheduler;
use super::model::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, FormatChatRequest, FormatChatResponse,
    ForwardPassRequest, ForwardPassResponse, HandshakeRequest, HandshakeResponse, QueryRequest,
    QueryResponse, Request,
};
use super::model::resource::{ResourceId, ResourceManager, ResourceTypeId};
use super::model::tokenizer::BytePairEncoder;
use super::runtime::{self, TerminationCause};
use super::service::ServiceCommand;
use crate::instance::InstanceId;
use anyhow::Result;

use futures::future;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::{self, JoinHandle};

// Re-export SchedulerConfig for public API
pub use batching::SchedulerConfig;

/// Backend for RPC calls to Python via IPC.
/// 
/// Uses cross-process IPC for GIL isolation in the symmetric worker architecture.
#[derive(Clone)]
pub struct RpcBackend {
    client: ffi_bridge::AsyncIpcClient,
}

impl RpcBackend {
    /// Create a new RpcBackend from an IPC client.
    pub fn new_ipc(client: ffi_bridge::AsyncIpcClient) -> Self {
        Self { client }
    }

    /// Call a remote method with typed arguments and return value.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: DeserializeOwned + Send + 'static,
    {
        self.client.call(method, args).await
    }

    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: Duration,
    ) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: DeserializeOwned + Send + 'static,
    {
        self.client.call_with_timeout(method, args, timeout).await
    }

    /// Fire-and-forget notification.
    pub async fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize + Send + Sync + Clone + 'static,
    {
        self.client.notify(method, args).await
    }
}

pub type HandlerId = u32;
pub type CmdQueueId = u32;

static MODEL_DISPATCHER: LazyLock<ModelDispatcher> = LazyLock::new(|| ModelDispatcher {
    models: boxcar::Vec::new(),
});

#[derive(Debug, Error)]
pub enum ModelDispatchError {
    #[error("Invalid model index: {0}")]
    InvalidModelIndex(usize),
}

#[derive(Debug)]
struct ModelDispatcher {
    models: boxcar::Vec<(String, mpsc::UnboundedSender<Command>)>,
}

pub fn install_model(model_name: String, mut model: Model) -> Option<usize> {
    for (_, (existing_name, _)) in MODEL_DISPATCHER.models.iter() {
        if existing_name == &model_name {
            return None;
        }
    }

    let (tx, mut rx) = mpsc::unbounded_channel();
    MODEL_DISPATCHER.models.push((model_name, tx));
    let model_id = MODEL_DISPATCHER.models.count() - 1;

    task::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            model.handle(cmd).await;
        }
    });

    Some(model_id)
}

pub fn registered_models() -> Vec<String> {
    MODEL_DISPATCHER
        .models
        .iter()
        .map(|(_, (name, _))| name.clone())
        .collect()
}

pub fn model_service_id(model_name: &str) -> Option<usize> {
    MODEL_DISPATCHER
        .models
        .iter()
        .find(|(_, (name, _))| name == model_name)
        .map(|(idx, _)| idx)
}

pub fn cleanup_instance(inst_id: InstanceId) {
    for (model_id, _) in MODEL_DISPATCHER.models.iter() {
        Command::Cleanup { inst_id }.dispatch(model_id).ok();
    }
}

pub async fn runtime_stats() -> HashMap<String, String> {
    let mut aggregated_stats = HashMap::new();
    let mut futures = Vec::new();

    for (model_id, (model_name, _)) in MODEL_DISPATCHER.models.iter() {
        let (tx, rx) = oneshot::channel();
        let cmd = Command::GetRuntimeStats { response: tx };

        if cmd.dispatch(model_id).is_ok() {
            futures.push((model_name.clone(), rx));
        } else {
            aggregated_stats.insert(
                format!("{}.error", model_name),
                "failed to dispatch command to service".to_string(),
            );
        }
    }

    let results = future::join_all(
        futures
            .into_iter()
            .map(async move |(name, rx)| (name, rx.await)),
    )
    .await;

    for (model_name, result) in results {
        match result {
            Ok(model_stats) => {
                for (key, value) in model_stats {
                    aggregated_stats.insert(format!("{}.{}", model_name, key), value);
                }
            }
            Err(e) => {
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

fn terminate_instance_with_exception<T>(inst_id: InstanceId, exception: T)
where
    T: ToString,
{
    runtime::Command::TerminateInstance {
        inst_id,
        notification_to_client: Some(TerminationCause::Exception(exception.to_string())),
    }
    .dispatch();
}

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
    // Actor Commands
    ActorGlobalContextRef { username: String, uid: String },
    ActorGlobalContextDestroy { username: String, uid: String },
    ActorGlobalContextExtend { username: String, uid: String, page_ids: Vec<u32>, last_page_len: u32 },
    ActorGlobalContextTrim { username: String, uid: String, len: u32 },
    ActorGlobalContextRead { username: String, uid: String, num_tokens: u32, offset: u32, response: oneshot::Sender<Vec<u32>> },
    ActorAdapterRef { username: String, uid: String },
    ActorAdapterDestroy { username: String, uid: String },
    ActorAdapterBlank { username: String, uid: String, rank: u32, alpha: f32 },
    ActorAdapterLoad { username: String, uid: String, path: String },
    ActorOptimizerRef { username: String, uid: String },
    ActorOptimizerDestroy { username: String, uid: String },
    ActorOptimizerLoad { username: String, uid: String, path: String },
    ActorOptimizerSave { username: String, uid: String, path: String },
    ActorOptimizerInitialize { username: String, uid: String, adapter_uid: String, params: Vec<u8> },
    ActorOptimizerUpdate { username: String, uid: String, params: Vec<u8> },
}

impl Command {
    pub fn dispatch(self, model_id: usize) -> Result<(), ModelDispatchError> {
        let (_, tx) = MODEL_DISPATCHER
            .models
            .get(model_id)
            .ok_or(ModelDispatchError::InvalidModelIndex(model_id))?;
        tx.send(self).unwrap();
        Ok(())
    }
}

/// Number of format_chat requests to validate (both paths) before trusting the fast path.
pub const FORMAT_CHAT_VALIDATION_COUNT: u32 = 1;

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
    /// Raw HF chat template for in-process minijinja rendering.
    pub chat_template: String,
    /// Whether to use the minijinja fast path for format_chat.
    /// Decided once at startup by rendering a synthetic message through both
    /// minijinja and Python RPC and comparing token IDs.
    pub use_minijinja_fast_path: bool,
}

/// Model service for communication with Python backend via FFI.
pub struct Model {
    info: ModelInfo,
    resource_manager: ResourceManager,
    shutdown_tx: broadcast::Sender<()>,
    /// Per-group backends for parallel dispatch
    backends: Vec<RpcBackend>,
    /// Primary backend (backends[0]) for global operations
    primary_backend: RpcBackend,
    /// Batch limits from handshake
    max_batch_tokens: usize,
    max_batch_size: usize,
    /// Channel for forward pass requests (req, response_tx, group_id)
    forward_pass_tx: mpsc::UnboundedSender<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>, usize)>,
    worker_handle: Option<JoinHandle<()>>,
    /// Scheduler configuration
    scheduler_config: SchedulerConfig,
    /// Shared scheduler for metrics tracking
    scheduler: SharedScheduler,
}

impl Model {
    /// Get the model name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Constructor that works with per-group IPC backends.
    pub async fn new_with_backends(backends: Vec<RpcBackend>, scheduler_config: SchedulerConfig) -> Result<Self> {
        let num_groups = backends.len();
        let primary_backend = backends[0].clone();
        
        // Handshake via primary backend
        let handshake_info = Self::handshake(&primary_backend).await?;

        let (forward_pass_tx, forward_pass_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        let max_batch_tokens = handshake_info.max_batch_tokens;
        let max_batch_size = handshake_info.max_batch_size;

        // Create shared scheduler for adaptive batching (MultiGroup)
        // SharedScheduler is alias for Arc<Mutex<MultiGroupScheduler>>
        let scheduler = Arc::new(Mutex::new(super::model::batching::MultiGroupScheduler::new(
            scheduler_config.clone(),
            max_batch_size,
            num_groups,
        )));

        let worker_handle = tokio::spawn(Self::inference_worker(
            backends.clone(),
            forward_pass_rx,
            shutdown_rx,
            max_batch_tokens,
            max_batch_size,
            Arc::clone(&scheduler),
            scheduler_config.max_in_flight_batches,
            num_groups,
        ));

        let tokenizer = Arc::new(BytePairEncoder::new(
            handshake_info.tokenizer_num_vocab,
            handshake_info.tokenizer_merge_table.into_iter().collect(),
            handshake_info.tokenizer_special_tokens,
            &handshake_info.tokenizer_split_regex,
            handshake_info.tokenizer_escape_non_printable,
            handshake_info.tokenizer_sentencepiece_space,
        ));

        let chat_template = handshake_info.chat_template.clone();

        // Validate minijinja fast path at startup: render a synthetic message
        // through both minijinja and Python RPC, compare token IDs.
        let use_minijinja_fast_path = if chat_template.is_empty() {
            false
        } else {
            Self::validate_minijinja_fast_path(
                &chat_template,
                &tokenizer,
                &primary_backend,
            ).await
        };

        if use_minijinja_fast_path {
            eprintln!("[INFO] Chat template minijinja fast path ENABLED (validated at startup)");
        } else if !chat_template.is_empty() {
            eprintln!("[INFO] Chat template minijinja fast path DISABLED; using Python RPC");
        }

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
            chat_template,
            use_minijinja_fast_path,
        };

        let resource_manager = ResourceManager::new(handshake_info.resources, num_groups);

        Ok(Model {
            info,
            resource_manager,
            backends,
            primary_backend,
            max_batch_tokens,
            max_batch_size,
            forward_pass_tx,
            shutdown_tx,
            worker_handle: Some(worker_handle),
            scheduler_config,
            scheduler,
        })
    }

    async fn handshake(backend: &RpcBackend) -> Result<HandshakeResponse> {
        const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
        let req = HandshakeRequest { version: "0.1.0".to_string() };
        let response: HandshakeResponse = backend
            .call_with_timeout("handshake", &req, HANDSHAKE_TIMEOUT)
            .await?;
        Ok(response)
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.shutdown_tx.send(())?;
        if let Some(handle) = self.worker_handle.take() {
            handle.await?;
        }
        Ok(())
    }

    /// Adaptive inference worker with concurrent batch execution.
    ///
    /// Design:
    /// - Uses adaptive scheduler to decide when to fire batches (optimizing for throughput)
    /// - Multiple batches execute concurrently via spawned tasks (pipelining)
    /// - Scheduler tracks arrival rate and latency to make optimal decisions
    /// - Overlaps batch accumulation with GPU inference for maximum utilization
    /// - Uses per-group backends for parallel DP execution
    async fn inference_worker(
        backends: Vec<RpcBackend>,
        mut req_rx: mpsc::UnboundedReceiver<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>, usize)>,
        mut shutdown_rx: broadcast::Receiver<()>,
        max_batch_tokens: usize,
        max_batch_size: usize,
        scheduler: SharedScheduler,
        max_in_flight_batches: usize,
        num_groups: usize,
    ) {
        const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

        // State per group
        let mut batches: Vec<Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>> =
            (0..num_groups).map(|_| Vec::new()).collect();
        let mut group_tokens: Vec<usize> = vec![0; num_groups];
        let mut in_flight_counts: Vec<usize> = vec![0; num_groups];

        // Channel for batch completions (batch_size, tokens, latency, group_id)
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel::<(usize, usize, Duration, usize)>();

        // Channel for multi-step continuations: re-enqueued requests bypass
        // WASM and coalesce, going directly back into the batch queue.
        let (continuation_tx, mut continuation_rx) =
            mpsc::unbounded_channel::<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>, usize)>();


        // PROFILING: Track request rates and batch firing
        let mut prof_requests_received: Vec<usize> = vec![0; num_groups];
        let mut prof_batches_fired: Vec<usize> = vec![0; num_groups];
        let mut prof_last_report = Instant::now();

        // Track whether each group's pending batch contains new (non-continuation) requests.
        // When a batch has ONLY continuations, skip the coalesce window since there's
        // no benefit in waiting for more arrivals — the continuations are already ready.
        let mut group_has_new_requests: Vec<bool> = vec![false; num_groups];

        loop {
            if Self::trace_enabled() {
                let pending: usize = batches.iter().map(|b| b.len()).sum();
                Self::trace("rs.loop_top", prof_batches_fired[0],
                    &format!("inf={} pending={}", in_flight_counts.iter().sum::<usize>(), pending));
            }

            // Process any completed batches (non-blocking) - from any group
            while let Ok((batch_size, tokens_in_batch, latency, group_id)) = completion_rx.try_recv() {
                if group_id < num_groups {
                    if Self::trace_enabled() {
                        Self::trace("rs.complete_enter", prof_batches_fired[group_id],
                            &format!("reqs={}", batch_size));
                    }
                    if in_flight_counts[group_id] > 0 {
                        in_flight_counts[group_id] -= 1;
                    }
                    if Self::sched_tracing_enabled() {
                        eprintln!(
                            "[SCHED-COMPLETE] t={} group={} reqs={} latency={:.1}ms in_flight={}",
                            Self::mono_ns(), group_id, batch_size,
                            latency.as_secs_f64() * 1000.0, in_flight_counts[group_id],
                        );
                    }
                    let mut sched = scheduler.lock().unwrap();
                    sched.on_batch_complete(group_id, batch_size, tokens_in_batch, latency);
                }
            }

            // Drain multi-step continuations into batch queues (bypasses WASM).
            {
                let mut cont_count = 0u32;
                while let Ok((req, tx, group_id)) = continuation_rx.try_recv() {
                    let group_id = std::cmp::min(group_id, num_groups - 1);
                    group_tokens[group_id] += req.input_tokens.len();
                    batches[group_id].push((req, tx));
                    cont_count += 1;
                }
                if cont_count > 0 {
                    if Self::trace_enabled() {
                        Self::trace("rs.cont_drain_done", prof_batches_fired[0],
                            &format!("n={}", cont_count));
                    }
                    if Self::sched_tracing_enabled() {
                        eprintln!("[SCHED-CONT-DRAIN] t={} drained={}", Self::mono_ns(), cont_count);
                    }
                }
            }

            let all_batches_empty = batches.iter().all(|b| b.is_empty());

            if all_batches_empty {
                if Self::trace_enabled() {
                    Self::trace("rs.select_enter", prof_batches_fired[0],
                        &format!("inf={}", in_flight_counts.iter().sum::<usize>()));
                }
                // Wait for any event: new request, completion, or continuation.
                tokio::select! {
                    biased;
                    _ = shutdown_rx.recv() => break,
                    maybe_completion = completion_rx.recv() => {
                        if let Some((batch_size, tokens_in_batch, latency, group_id)) = maybe_completion {
                            if group_id < num_groups {
                                if in_flight_counts[group_id] > 0 {
                                    in_flight_counts[group_id] -= 1;
                                }
                                let mut sched = scheduler.lock().unwrap();
                                sched.on_batch_complete(group_id, batch_size, tokens_in_batch, latency);
                            }
                        }
                        if Self::trace_enabled() {
                            Self::trace("rs.select_wake", prof_batches_fired[0], "src=completion");
                        }
                        continue; // re-check — continuation drain at top of loop
                    }
                    maybe_cont = continuation_rx.recv() => {
                        if let Some((req, tx, group_id)) = maybe_cont {
                            let group_id = std::cmp::min(group_id, num_groups - 1);
                            group_tokens[group_id] += req.input_tokens.len();
                            batches[group_id].push((req, tx));
                        }
                        if Self::trace_enabled() {
                            Self::trace("rs.select_wake", prof_batches_fired[0], "src=continuation");
                        }
                        continue; // skip the req_rx add below
                    }
                    maybe_req = req_rx.recv() => {
                        match maybe_req {
                            Some((req, tx, group_id)) => {
                                let group_id = std::cmp::min(group_id, num_groups - 1);
                                {
                                    let mut sched = scheduler.lock().unwrap();
                                    let arrival_time = req.arrival_time.unwrap_or_else(Instant::now);
                                    sched.on_request_arrival(group_id, arrival_time);
                                }
                                group_tokens[group_id] += req.input_tokens.len();
                                batches[group_id].push((req, tx));
                                group_has_new_requests[group_id] = true;
                                prof_requests_received[group_id] += 1;
                            }
                            None => break,
                        }
                    }
                };
            }

            // Drain new requests (non-blocking). With max_in_flight=1,
            // the GPU step (~30ms) provides a natural coalesce window:
            // all continuations + new requests accumulate while busy.
            {
                while let Ok((req, tx, group_id)) = req_rx.try_recv() {
                    let group_id = std::cmp::min(group_id, num_groups - 1);
                    {
                        let mut sched = scheduler.lock().unwrap();
                        let arrival_time = req.arrival_time.unwrap_or_else(Instant::now);
                        sched.on_request_arrival(group_id, arrival_time);
                    }
                    group_tokens[group_id] += req.input_tokens.len();
                    batches[group_id].push((req, tx));
                    group_has_new_requests[group_id] = true;
                    prof_requests_received[group_id] += 1;
                }
            }

            // Fire non-empty groups immediately — no should_fire heuristic,
            // no coalesce. Only gate: max_in_flight (1 = Python controls GPU).
            let mut fired_any = false;
            for group_id in 0..num_groups {
                let batch_len = batches[group_id].len();
                if batch_len == 0 {
                    continue;
                }

                // Gate: don't exceed max in-flight batches.
                // With max_in_flight=MAX (default), Rust fires immediately.
                // Python's manager drains ALL pending from the side-channel,
                // merges into one GPU batch — achieving continuous batching.
                if in_flight_counts[group_id] >= max_in_flight_batches {
                    if Self::trace_enabled() {
                        Self::trace("rs.fire_blocked", prof_batches_fired[group_id],
                            &format!("inf={} pending={}", in_flight_counts[group_id], batch_len));
                    }
                    continue;
                }

                let total_tok = group_tokens[group_id];

                {
                    // Fire!
                    let batch_to_fire = std::mem::take(&mut batches[group_id]);
                    group_tokens[group_id] = 0;
                    group_has_new_requests[group_id] = false;
                    in_flight_counts[group_id] += 1;
                    fired_any = true;
                    prof_batches_fired[group_id] += 1;

                    if Self::trace_enabled() {
                        Self::trace("rs.fire", prof_batches_fired[group_id],
                            &format!("reqs={} toks={}", batch_len, total_tok));
                    }

                     {
                        let mut sched = scheduler.lock().unwrap();
                        sched.on_batch_fired(group_id);
                    }

                    // Spawn batch execution using group-specific backend
                    let backend_clone = backends[group_id].clone();
                    let completion_tx_clone = completion_tx.clone();
                    let continuation_tx_clone = continuation_tx.clone();
                    let batch_size = batch_len;
                    let tokens_in_batch = total_tok;
                    // Python's _batch_counter is 0-based; prof_batches_fired is post-increment
                    let trace_batch_id = prof_batches_fired[group_id].saturating_sub(1);

                    if Self::sched_tracing_enabled() {
                        eprintln!(
                            "[SCHED-FIRE] t={} group={} reqs={} toks={} in_flight={}",
                            Self::mono_ns(), group_id, batch_size, tokens_in_batch,
                            in_flight_counts[group_id],
                        );
                    }

                    tokio::spawn(async move {
                         let start_time = Instant::now();
                         Self::execute_forward_pass_batch(
                             &backend_clone, batch_to_fire, group_id,
                             REQUEST_TIMEOUT, &continuation_tx_clone,
                             trace_batch_id,
                         ).await;
                         if Self::trace_enabled() {
                             Self::trace("rs.pre_completion", trace_batch_id,
                                 &format!("reqs={}", batch_size));
                         }
                         let latency = start_time.elapsed();
                         completion_tx_clone.send((batch_size, tokens_in_batch, latency, group_id)).ok();
                         if Self::trace_enabled() {
                             Self::trace("rs.post_completion", trace_batch_id,
                                 &format!("reqs={}", batch_size));
                         }
                    });
                }
            }

            // PROFILING: Report every 10 seconds
            if prof_last_report.elapsed().as_secs() >= 10 {
                let total_reqs: usize = prof_requests_received.iter().sum();
                let total_batches: usize = prof_batches_fired.iter().sum();
                eprintln!("[RUST PROFILING] Reqs: {:?} ({}) | Batches: {:?} ({}) | InFlight: {:?}",
                    prof_requests_received, total_reqs,
                    prof_batches_fired, total_batches,
                    in_flight_counts);
                // Reset counters
                for c in &mut prof_requests_received { *c = 0; }
                for c in &mut prof_batches_fired { *c = 0; }
                prof_last_report = Instant::now();
            }

            // Wait logic: select on all event sources to minimize latency.
            if !fired_any {
                 if Self::sched_tracing_enabled() {
                     let any_pending = batches.iter().any(|b| !b.is_empty());
                     eprintln!("[SCHED-WAIT] t={} any_pending={} in_flight={:?}",
                         Self::mono_ns(), any_pending, &in_flight_counts);
                 }
                 // No in-flight limit — always accept new requests.
                 // Python controls GPU timing via side-channel.
                 tokio::select! {
                    biased;
                    _ = shutdown_rx.recv() => break,
                    maybe_completion = completion_rx.recv() => {
                        if let Some((batch_size, tokens_in_batch, latency, group_id)) = maybe_completion {
                            if group_id < num_groups {
                                if in_flight_counts[group_id] > 0 {
                                    in_flight_counts[group_id] -= 1;
                                }
                                if Self::sched_tracing_enabled() {
                                    eprintln!(
                                        "[SCHED-COMPLETE] t={} group={} reqs={} latency={:.1}ms in_flight={}",
                                        Self::mono_ns(), group_id, batch_size,
                                        latency.as_secs_f64() * 1000.0, in_flight_counts[group_id],
                                    );
                                }
                                let mut sched = scheduler.lock().unwrap();
                                sched.on_batch_complete(group_id, batch_size, tokens_in_batch, latency);
                            }
                        }
                    }
                    maybe_cont = continuation_rx.recv() => {
                        if let Some((req, tx, group_id)) = maybe_cont {
                            let group_id = std::cmp::min(group_id, num_groups - 1);
                            group_tokens[group_id] += req.input_tokens.len();
                            batches[group_id].push((req, tx));
                        }
                    }
                    maybe_req = req_rx.recv() => {
                        match maybe_req {
                            Some((req, tx, group_id)) => {
                                let group_id = std::cmp::min(group_id, num_groups - 1);
                                {
                                    let mut sched = scheduler.lock().unwrap();
                                    let arrival_time = req.arrival_time.unwrap_or_else(Instant::now);
                                    sched.on_request_arrival(group_id, arrival_time);
                                }
                                group_tokens[group_id] += req.input_tokens.len();
                                batches[group_id].push((req, tx));
                            }
                            None => break,
                        }
                    }
                 }
            }
        }

        // Shutdown cleanup: Fire remaining batches using group-specific backend
        for group_id in 0..num_groups {
            if !batches[group_id].is_empty() {
                let batch = std::mem::take(&mut batches[group_id]);
                Self::execute_forward_pass_batch(&backends[group_id], batch, group_id, REQUEST_TIMEOUT, &continuation_tx, 0).await;
            }
        }

        // Wait for all in-flight batches (sum of all counts)
        while in_flight_counts.iter().sum::<usize>() > 0 {
             if let Some((_, _, _, group_id)) = completion_rx.recv().await {
                 if group_id < num_groups && in_flight_counts[group_id] > 0 {
                     in_flight_counts[group_id] -= 1;
                 }
             } else {
                 break;
             }
        }
    }

    /// Get CLOCK_MONOTONIC nanoseconds for cross-process trace correlation.
    fn mono_ns() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
    }

    fn sched_tracing_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("PIE_SCHED_TIMING").is_ok())
    }

    fn trace_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("PIE_TRACE").is_ok())
    }

    fn trace(phase: &str, batch_id: usize, extra: &str) {
        eprintln!("[PIE-TRACE] {} {} {} {}", Self::mono_ns(), phase, batch_id, extra);
    }

    #[tracing::instrument(
        name = "rust.fire_batch",
        skip(backend, requests, continuation_tx),
        fields(batch_size = requests.len())
    )]
    async fn execute_forward_pass_batch(
        backend: &RpcBackend,
        requests: Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
        group_id: usize,
        timeout: Duration,
        continuation_tx: &mpsc::UnboundedSender<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>, usize)>,
        trace_batch_id: usize,
    ) {
        let _t_entry = if Self::sched_tracing_enabled() { Some(Self::mono_ns()) } else { None };

        let batch_size = requests.len();
        let mut batch_req = BatchedForwardPassRequest::new();
        batch_req.group_id = Some(group_id);
        for (fp_req, _) in &requests {
            batch_req.add_request(fp_req);
        }
        let batch_tokens = batch_req.total_tokens();

        // Inject trace context
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let current_span = tracing::Span::current();
        let context = current_span.context();
        use opentelemetry::trace::TraceContextExt;
        let otel_span = context.span();
        let span_context = otel_span.span_context();
        if span_context.is_valid() {
            let traceparent = format!(
                "00-{}-{}-{:02x}",
                span_context.trace_id(),
                span_context.span_id(),
                span_context.trace_flags().to_u8()
            );
            batch_req.set_trace_context(traceparent);
        }

        let _t_pre_rpc = if Self::sched_tracing_enabled() { Some(Self::mono_ns()) } else { None };

        let result: Result<BatchedForwardPassResponse, _> = backend
            .call_with_timeout("fire_batch", &batch_req, timeout)
            .await;

        let _t_post_rpc = if Self::sched_tracing_enabled() { Some(Self::mono_ns()) } else { None };
        if Self::trace_enabled() {
            Self::trace("rs.rpc_done", trace_batch_id, &format!("reqs={}", batch_size));
        }

        match result {
            Ok(batch_resp) => {
                // Collect all continuations FIRST, then send them in bulk.
                // This prevents tokio from interleaving the scheduler loop
                // between individual sends, which causes batch fragmentation
                // (scheduler fires reqs=1 before all continuations arrive).
                let mut pending_continuations: Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>, usize)> = Vec::new();

                let mut resp_iter = batch_resp.results.into_iter();
                for (mut fp_req, resp_tx) in requests {
                    let resp = resp_iter.next();

                    if fp_req.max_decode_steps > 1 || !fp_req.multi_step_tokens.is_empty() {
                        // Multi-step: accumulate token(s), re-enqueue or finish.
                        if let Some(ref r) = resp {
                            if r.tokens.len() > 1 {
                                // Stream all tokens at once if Python returned multiple
                                if let Some(ref tx) = fp_req.token_stream_tx {
                                    for &t in &r.tokens {
                                        if tx.send(t).is_err() { break; }
                                    }
                                }
                                fp_req.multi_step_tokens.extend(&r.tokens);
                                fp_req.max_decode_steps = 0;
                            } else if let Some(&token) = r.tokens.first() {
                                // Stream this token immediately to WASM
                                let cancelled = if let Some(ref tx) = fp_req.token_stream_tx {
                                    tx.send(token).is_err() // WASM dropped receiver = cancel
                                } else {
                                    false
                                };

                                fp_req.multi_step_tokens.push(token);
                                fp_req.max_decode_steps -= 1;

                                if !cancelled && fp_req.max_decode_steps > 0 {
                                    fp_req.kv_page_last_len += 1;
                                    if fp_req.kv_page_size > 0
                                        && fp_req.kv_page_last_len > fp_req.kv_page_size
                                    {
                                        fp_req.kv_page_last_len = 1;
                                    }
                                    let next_pos = fp_req.input_token_positions.last()
                                        .map(|&p| p + 1).unwrap_or(0);
                                    let ctx_len: u32 = fp_req.mask.last()
                                        .map(|m| m.iter().copied().sum()).unwrap_or(0);
                                    fp_req.input_tokens = vec![token];
                                    fp_req.input_token_positions = vec![next_pos];
                                    fp_req.mask = vec![vec![ctx_len + 1]];
                                    fp_req.output_token_indices = vec![0];
                                    if Self::sched_tracing_enabled() {
                                        eprintln!("[MS-REQUEUE] t={} step={} remaining={} token={}",
                                            Self::mono_ns(), fp_req.multi_step_tokens.len(),
                                            fp_req.max_decode_steps, token);
                                    }
                                    pending_continuations.push((fp_req, resp_tx, group_id));
                                    continue;
                                }
                            }
                        }
                        // Drop the streaming sender before sending final oneshot.
                        // This closes the mpsc channel, signaling WASM that streaming is done.
                        drop(fp_req.token_stream_tx.take());
                        if let Some(tx) = resp_tx {
                            tx.send(ForwardPassResponse {
                                tokens: fp_req.multi_step_tokens,
                                dists: resp.map(|r| r.dists).unwrap_or_default(),
                            }).ok();
                        }
                    } else {
                        if let Some(tx) = resp_tx {
                            if let Some(r) = resp {
                                tx.send(r).ok();
                            }
                        }
                    }
                }

                if Self::trace_enabled() {
                    Self::trace("rs.dispatch_done", trace_batch_id, &format!("reqs={} conts={}", batch_size, pending_continuations.len()));
                }

                // Send all continuations in bulk — no tokio yield points between sends.
                for cont in pending_continuations {
                    continuation_tx.send(cont).ok();
                }
                if Self::trace_enabled() {
                    Self::trace("rs.conts_sent", trace_batch_id, &format!("reqs={}", batch_size));
                }
            }
            Err(e) => {
                eprintln!("[Error] fire_batch failed: {:?}", e);
            }
        }

        if let (Some(t_entry), Some(t_pre_rpc), Some(t_post_rpc)) = (_t_entry, _t_pre_rpc, _t_post_rpc) {
            let t_exit = Self::mono_ns();
            eprintln!(
                "[SCHED-TRACE] batch_form={:.2}ms rpc={:.1}ms dispatch={:.2}ms total={:.1}ms reqs={} toks={} entry={} exit={}",
                (t_pre_rpc - t_entry) as f64 / 1e6,
                (t_post_rpc - t_pre_rpc) as f64 / 1e6,
                (t_exit - t_post_rpc) as f64 / 1e6,
                (t_exit - t_entry) as f64 / 1e6,
                batch_size, batch_tokens,
                t_entry, t_exit,
            );
        }
    }

    /// Execute eager RPC calls (non-batched)
    async fn execute_query(backend: &RpcBackend, req: QueryRequest) -> Option<QueryResponse> {
        const TIMEOUT: Duration = Duration::from_secs(30);
        backend.call_with_timeout("query", &req, TIMEOUT).await.ok()
    }

    /// Validate minijinja fast path at startup by rendering a synthetic message
    /// through both minijinja (in-process) and Python RPC, comparing token IDs.
    async fn validate_minijinja_fast_path(
        chat_template: &str,
        tokenizer: &Arc<BytePairEncoder>,
        backend: &RpcBackend,
    ) -> bool {
        // Synthetic test message for validation.
        let test_messages_json = r#"{"messages":[{"role":"user","content":"Hello"}]}"#;

        // Step 1: Try minijinja rendering.
        let mj_result = chat_template::render_and_tokenize(
            chat_template,
            test_messages_json,
            None, // no tools
            true, // add_generation_prompt
            tokenizer,
        );
        let mj_tokens = match mj_result {
            Ok(tokens) => tokens,
            Err(reason) => {
                eprintln!("[INFO] minijinja startup validation: render failed: {}", reason);
                return false;
            }
        };

        // Step 2: Send same message through Python RPC.
        let rpc_req = FormatChatRequest {
            messages_json: test_messages_json.to_string(),
            tools_json: None,
            add_generation_prompt: true,
        };
        let rpc_result = Self::execute_format_chat(backend, rpc_req).await;
        let rpc_tokens = match rpc_result {
            Some(resp) => resp.token_ids,
            None => {
                eprintln!("[WARN] minijinja startup validation: Python RPC failed");
                return false;
            }
        };

        // Step 3: Compare.
        if mj_tokens == rpc_tokens {
            eprintln!(
                "[INFO] minijinja startup validation: tokens match ({} tokens)",
                mj_tokens.len()
            );
            true
        } else {
            eprintln!(
                "[WARN] minijinja startup validation: token MISMATCH! \
                 minijinja={} tokens, rpc={} tokens. \
                 First divergence at position {}. Using Python RPC.",
                mj_tokens.len(),
                rpc_tokens.len(),
                mj_tokens.iter().zip(rpc_tokens.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(mj_tokens.len().min(rpc_tokens.len())),
            );
            false
        }
    }

    async fn execute_format_chat(backend: &RpcBackend, req: FormatChatRequest) -> Option<FormatChatResponse> {
        const TIMEOUT: Duration = Duration::from_secs(30);
        backend.call_with_timeout("format_chat", &req, TIMEOUT).await.ok()
    }

    pub fn submit(&self, _cmd_queue_id: CmdQueueId, _priority: u32, req: Request) {
        match req {
            Request::ForwardPass(mut fp_req, resp_tx) => {
                // Capture arrival time before queuing to avoid measurement distortion
                // when requests pile up behind the in-flight limit.
                
                // Lookup group ID from resource manager
                let group_id = if let Some(inst_id) = fp_req.inst_id {
                    self.resource_manager.get_group(&inst_id).unwrap_or(0)
                } else {
                    0
                };

                if self.forward_pass_tx.send((fp_req, resp_tx, group_id)).is_err() {
                    eprintln!("[Error] Forward pass channel closed");
                }
            }
            Request::Query(query_req, resp_tx) => {
                let backend_clone = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Some(resp) = Self::execute_query(&backend_clone, query_req).await {
                        resp_tx.send(resp).ok();
                    }
                });
            }
            Request::EmbedImage(req) => {
                let backend = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("embed_image", &req).await {
                        eprintln!("[Error] embed_image failed: {:?}", e);
                    }
                });
            }
            Request::InitializeAdapter(req) => {
                let backend = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("initialize_adapter", &req).await {
                        eprintln!("[Error] initialize_adapter failed: {:?}", e);
                    }
                });
            }
            Request::UpdateAdapter(req) => {
                let backend = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("update_adapter", &req).await {
                        eprintln!("[Error] update_adapter failed: {:?}", e);
                    }
                });
            }
            Request::UploadAdapter(req) => {
                let backend = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("upload_adapter", &req).await {
                        eprintln!("[Error] upload_adapter failed: {:?}", e);
                    }
                });
            }
            Request::DownloadAdapter(req) => {
                let backend = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("download_adapter", &req).await {
                        eprintln!("[Error] download_adapter failed: {:?}", e);
                    }
                });
            }
            Request::Handshake(_, _) => {
                eprintln!("[Warn] Unexpected handshake request in submit");
            }
            Request::FormatChat(fc_req, resp_tx) => {
                let backend_clone = self.primary_backend.clone();
                tokio::spawn(async move {
                    if let Some(resp) = Self::execute_format_chat(&backend_clone, fc_req).await {
                        resp_tx.send(resp).ok();
                    }
                });
            }
            Request::Synchronize(tx) => {
                tx.send(()).ok();
            }
        }
    }

    pub fn runtime_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("model.name".to_string(), self.info.name.clone());
        self.resource_manager.append_stats_to(&mut stats);
        // Add throughput and latency from scheduler
        if let Ok(sched) = self.scheduler.lock() {
            let (tps, avg_lat) = sched.get_aggregate_metrics();
            stats.insert("model.throughput.tokens_per_second".to_string(), format!("{:.1}", tps));
            stats.insert("model.latency.avg_ms".to_string(), format!("{:.1}", avg_lat));
        }
        stats
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Submit { cmd_queue_id, priority, req } => {
                self.submit(cmd_queue_id, priority, req);
            }
            Command::GetInfo { response } => {
                response.send(self.info.clone()).ok();
            }
            Command::GetRuntimeStats { response } => {
                response.send(self.runtime_stats()).ok();
            }
            Command::Allocate { inst_id, type_id, count, response } => {
                match self.resource_manager.allocate_with_oom(inst_id, type_id, count) {
                    Ok(allocated_ids) => { response.send(allocated_ids).ok(); }
                    Err(e) => terminate_instance_with_exception(inst_id, e),
                }
            }
            Command::Deallocate { inst_id, type_id, ptrs } => {
                if let Err(e) = self.resource_manager.deallocate(inst_id, type_id, ptrs) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            Command::Cleanup { inst_id } => {
                if let Err(e) = self.resource_manager.cleanup(inst_id) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            Command::GetAllExported { type_id, response } => {
                response.send(self.resource_manager.get_all_exported(type_id)).ok();
            }
            Command::Export { inst_id, type_id, ptrs, name } => {
                if let Err(e) = self.resource_manager.export(inst_id, type_id, ptrs, name) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            Command::Import { inst_id, type_id, name, response } => {
                match self.resource_manager.import(type_id, name) {
                    Ok((group_id, ptrs)) => {
                        // Register imported pointers in the importing instance's
                        // allocation table so they can be re-exported later.
                        // This also increments the ref count for each page (+1
                        // since the export already holds a ref).
                        self.resource_manager.register_imported(inst_id, type_id, group_id, &ptrs);
                        response.send(ptrs).ok();
                    }
                    Err(e) => terminate_instance_with_exception(inst_id, e),
                }
            }
            Command::ReleaseExported { inst_id, type_id, name } => {
                if let Err(e) = self.resource_manager.release_exported(type_id, name) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            // Actor Commands (stubs)
            Command::ActorGlobalContextRef { .. } |
            Command::ActorGlobalContextDestroy { .. } |
            Command::ActorGlobalContextExtend { .. } |
            Command::ActorGlobalContextTrim { .. } |
            Command::ActorAdapterRef { .. } |
            Command::ActorAdapterDestroy { .. } |
            Command::ActorAdapterBlank { .. } |
            Command::ActorAdapterLoad { .. } |
            Command::ActorOptimizerRef { .. } |
            Command::ActorOptimizerDestroy { .. } |
            Command::ActorOptimizerLoad { .. } |
            Command::ActorOptimizerSave { .. } |
            Command::ActorOptimizerInitialize { .. } |
            Command::ActorOptimizerUpdate { .. } => {}
            Command::ActorGlobalContextRead { response, .. } => {
                response.send(vec![]).ok();
            }
        }
    }
}
