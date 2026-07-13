use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::ffi::c_void;
#[cfg(test)]
use std::sync::Condvar;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use anyhow::{Result, anyhow, bail, ensure};
use pie_driver_abi::{
    DeviceFacts, DriverCapabilities, ModelLoadDesc, PIE_CHANNEL_DTYPE_ACT, PIE_CHANNEL_DTYPE_BOOL,
    PIE_CHANNEL_DTYPE_F32, PIE_CHANNEL_DTYPE_I32, PIE_CHANNEL_DTYPE_U32, PIE_CHANNEL_EXTERN_NONE,
    PIE_TERMINAL_OUTCOME_FAILED, PIE_TERMINAL_OUTCOME_RETRY, PIE_TERMINAL_OUTCOME_SUCCESS,
    PieBytes, PieChannelDesc, PieChannelEndpointBinding, PieChannelValueDescSlice, PieCompletion,
    PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc, PieLaunchDesc, PiePoolResizeDesc,
    PieProgramDesc, PieRuntimeCallbacks, PieStateCopyDesc, PieTerminalCell,
    PieTerminalCellPtrSlice, PieU32Slice, PieU64Slice, validate_channel_desc, validate_completion,
    validate_instance_desc, validate_kv_copy_desc, validate_launch_desc, validate_pool_resize_desc,
    validate_program_desc, validate_state_copy_desc,
};
use pie_ptir::container::{self, ExternDir, HostRole};
use pie_ptir::interp::{
    ExternChannel, HostError, Instance as InterpInstance, NoKernels, PassInputs, StepError, Value,
};
use pie_ptir::op::{IntrinsicId, Op};
use pie_ptir::registry::{KernelInfo, ModelProfile};
use pie_ptir::types::{DType, ValueType};
use pie_ptir::validate::BoundTrace;

/// One accepted `launch`'s forward geometry, copied out of the descriptor into
/// owned vectors (safe to hold past the call) for test probes.
#[derive(Debug, Clone, Default)]
pub struct LaunchObservation {
    pub token_ids: Vec<u32>,
    pub qo_indptr: Vec<u32>,
    pub kv_page_indices: Vec<u32>,
    pub kv_page_indptr: Vec<u32>,
    pub kv_last_page_lens: Vec<u32>,
}

/// Synchronous observer invoked on the launch path for every accepted launch
/// (test probes: page-assignment assertions, injected per-fire latency — a
/// sleeping observer keeps the fire outstanding, forcing real overlap).
#[derive(Clone)]
pub struct LaunchObserver(pub Arc<dyn Fn(&LaunchObservation) + Send + Sync>);

impl std::fmt::Debug for LaunchObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LaunchObserver")
    }
}

#[derive(Debug, Clone)]
pub struct DummyDriverOptions {
    pub total_pages: u32,
    pub kv_page_size: u32,
    pub swap_pool_size: u32,
    pub vocab_size: u32,
    pub max_model_len: u32,
    pub arch_name: String,
    pub activation_dtype: String,
    pub snapshot_dir: String,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
    pub max_page_refs: u32,
    pub has_mtp_logits: bool,
    pub has_mtp_drafts: bool,
    pub has_value_head: bool,
    pub callback_delay_ms: u64,
    pub reject_launches: bool,
    pub reject_launches_remaining: u32,
    pub fail_launches_after_accept: bool,
    pub retry_launches_remaining: u32,
    pub operation_log: Option<Arc<Mutex<Vec<String>>>>,
    pub launch_observer: Option<LaunchObserver>,
}

impl Default for DummyDriverOptions {
    fn default() -> Self {
        Self {
            total_pages: 4096,
            kv_page_size: 16,
            swap_pool_size: 0,
            vocab_size: 32_000,
            max_model_len: 8192,
            arch_name: "dummy".to_string(),
            activation_dtype: "f32".to_string(),
            snapshot_dir: String::new(),
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: 65_536,
            has_mtp_logits: true,
            has_mtp_drafts: true,
            has_value_head: true,
            callback_delay_ms: 0,
            reject_launches: false,
            reject_launches_remaining: 0,
            fail_launches_after_accept: false,
            retry_launches_remaining: 0,
            operation_log: None,
            launch_observer: None,
        }
    }
}

#[derive(Clone, Debug)]
struct DummyProgram {
    hash: u64,
    bound: BoundTrace,
    intrinsics: ProgramIntrinsics,
}

#[derive(Clone, Debug, Default)]
struct ProgramIntrinsics {
    logits: Option<ValueType>,
    mtp_logits: Option<ValueType>,
    hidden: Option<ValueType>,
    query: Option<ValueType>,
    value_head: Option<ValueType>,
    mtp_drafts: Option<ValueType>,
}

#[derive(Debug, Clone)]
struct DummyChannel {
    global_id: u64,
    host_role: HostRole,
    ty: ValueType,
    endpoint: Arc<Mutex<DummyEndpoint>>,
}

#[derive(Debug)]
struct DummyEndpoint {
    channel_id: u64,
    shape: Vec<u32>,
    dtype: u8,
    host_role: u8,
    seeded: bool,
    capacity: u32,
    reader_wait_id: u64,
    writer_wait_id: u64,
    extern_name: Option<String>,
    mirror: Box<[u8]>,
    words: Box<[AtomicU64]>,
    attachments: HashMap<u64, Option<ExternDir>>,
    shared: Option<ExternChannel>,
    /// Host-writer ring cursor. `pulled` counts host-published ring entries
    /// already staged into the interpreter; actual head/tail words remain
    /// commit-predicated and ticket-authoritative.
    pulled: u64,
    seed_credit: bool,
}

#[derive(Debug)]
struct BoundInstanceState {
    program: Arc<DummyProgram>,
    instance_id: u64,
    inner: Mutex<BoundInstanceInner>,
}

#[derive(Debug)]
struct BoundInstanceInner {
    interp: InterpInstance,
    channels: Vec<DummyChannel>,
    next_pacing_epoch: u64,
    closed: bool,
}

#[derive(Default)]
struct DummyState {
    programs: HashMap<u64, Arc<DummyProgram>>,
    channels: HashMap<u64, Arc<Mutex<DummyEndpoint>>>,
    instances: HashMap<u64, Arc<BoundInstanceState>>,
}

#[derive(Clone, Debug)]
struct OwnedValueDesc {
    channel_id: u64,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ChannelTicket {
    expected_head: Option<u64>,
    expected_tail: Option<u64>,
    require_input: bool,
}

#[derive(Debug)]
struct LaunchInstanceWork {
    instance: Arc<BoundInstanceState>,
    pacing_epoch: u64,
    tickets: Vec<ChannelTicket>,
    terminal_cell: *mut PieTerminalCell,
}

unsafe impl Send for LaunchInstanceWork {}
unsafe impl Sync for LaunchInstanceWork {}

struct LaunchInstanceResult {
    outcome: u32,
    notifications: Vec<(u64, u64)>,
}

#[derive(Clone)]
struct SendableRuntimeCallbacks {
    ctx: usize,
    notify: pie_driver_abi::PieRuntimeNotifyFn,
    operation_log: Option<Arc<Mutex<Vec<String>>>>,
}

unsafe impl Send for SendableRuntimeCallbacks {}
unsafe impl Sync for SendableRuntimeCallbacks {}

enum PreparedCallback {
    Inline,
    Worker(mpsc::Sender<()>),
}

pub struct DummyDriver {
    device_facts: DeviceFacts,
    capabilities: DriverCapabilities,
    load_storage: Option<pie_load_planner::host_executor::HostStorage>,
    state: Arc<Mutex<DummyState>>,
    next_program_id: AtomicU64,
    next_instance_id: AtomicU64,
    reject_launches: bool,
    reject_launches_remaining: u32,
    fail_launches_after_accept: bool,
    retry_launches_remaining: u32,
    callback_delay_ms: u64,
    runtime: SendableRuntimeCallbacks,
    callback_workers: Vec<std::thread::JoinHandle<()>>,
    operation_log: Option<Arc<Mutex<Vec<String>>>>,
    launch_observer: Option<LaunchObserver>,
}

impl DummyDriver {
    pub fn new() -> Self {
        Self::with_options(DummyDriverOptions::default())
    }

    pub fn with_options(options: DummyDriverOptions) -> Self {
        Self::with_runtime(options, PieRuntimeCallbacks::default())
    }

    pub fn with_runtime(options: DummyDriverOptions, runtime: PieRuntimeCallbacks) -> Self {
        let state = Arc::new(Mutex::new(DummyState::default()));
        let operation_log = options.operation_log.clone();
        let runtime = SendableRuntimeCallbacks {
            ctx: runtime.ctx as usize,
            notify: runtime.notify,
            operation_log: operation_log.clone(),
        };
        Self {
            device_facts: DeviceFacts {
                abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
                backend: "dummy".to_string(),
                unified_memory: true,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: std::mem::align_of::<usize>() as u32,
                storage_max_tile_bytes: 64 * 1024 * 1024,
                storage_tile_map_mask: pie_load_planner::load_plan::HOST_TILE_MAP_MASK,
                page_size: 1,
            },
            capabilities: DriverCapabilities {
                abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
                total_pages: options.total_pages,
                kv_page_size: options.kv_page_size,
                swap_pool_size: options.swap_pool_size,
                kv_copy_domain_mask: if options.swap_pool_size > 0 {
                    pie_driver_abi::KV_COPY_DEVICE_TO_DEVICE
                        | pie_driver_abi::KV_COPY_DEVICE_TO_HOST
                        | pie_driver_abi::KV_COPY_HOST_TO_DEVICE
                        | pie_driver_abi::KV_COPY_HOST_TO_HOST
                } else {
                    pie_driver_abi::KV_COPY_DEVICE_TO_DEVICE
                },
                rs_cache_required: false,
                rs_cache_slots: 0,
                rs_cache_slot_bytes: 0,
                has_mtp_logits: options.has_mtp_logits,
                has_mtp_drafts: options.has_mtp_drafts,
                has_value_head: options.has_value_head,
                max_forward_tokens: options.max_forward_tokens,
                max_forward_requests: options.max_forward_requests,
                max_page_refs: options.max_page_refs,
                arch_name: options.arch_name,
                vocab_size: options.vocab_size,
                max_model_len: options.max_model_len,
                activation_dtype: options.activation_dtype,
                snapshot_dir: options.snapshot_dir,
            },
            load_storage: None,
            state,
            next_program_id: AtomicU64::new(1),
            next_instance_id: AtomicU64::new(1),
            reject_launches: options.reject_launches,
            reject_launches_remaining: options.reject_launches_remaining,
            fail_launches_after_accept: options.fail_launches_after_accept,
            retry_launches_remaining: options.retry_launches_remaining,
            callback_delay_ms: options.callback_delay_ms,
            runtime,
            callback_workers: Vec::new(),
            operation_log,
            launch_observer: options.launch_observer,
        }
    }

    fn record_op(&self, name: &str) {
        if let Some(log) = &self.operation_log {
            log.lock().unwrap().push(name.to_string());
        }
    }

    fn prepare_callback(&mut self, completion: PieCompletion) -> Result<PreparedCallback> {
        self.reap_callback_workers();
        if self.callback_delay_ms == 0 {
            return Ok(PreparedCallback::Inline);
        }
        let (tx, rx) = mpsc::channel();
        let runtime = self.runtime.clone();
        let callback_delay_ms = self.callback_delay_ms;
        let worker = std::thread::Builder::new()
            .name("pie-dummy-callback".to_string())
            .spawn(move || {
                if rx.recv().is_ok() {
                    notify_completion(
                        &runtime,
                        completion.wait_id,
                        completion.target_epoch,
                        callback_delay_ms,
                    );
                }
            })
            .map_err(|err| anyhow!("spawn dummy callback worker: {err}"))?;
        self.callback_workers.push(worker);
        Ok(PreparedCallback::Worker(tx))
    }

    fn publish_callback(&self, callback: PreparedCallback, completion: PieCompletion) {
        match callback {
            PreparedCallback::Inline => notify_completion(
                &self.runtime,
                completion.wait_id,
                completion.target_epoch,
                0,
            ),
            PreparedCallback::Worker(trigger) if trigger.send(()).is_err() => {
                notify_completion(
                    &self.runtime,
                    completion.wait_id,
                    completion.target_epoch,
                    self.callback_delay_ms,
                );
            }
            PreparedCallback::Worker(_) => {}
        }
    }

    fn reap_callback_workers(&mut self) {
        let mut live = Vec::with_capacity(self.callback_workers.len());
        for worker in self.callback_workers.drain(..) {
            if worker.is_finished() {
                let _ = worker.join();
            } else {
                live.push(worker);
            }
        }
        self.callback_workers = live;
    }

    pub fn capabilities(&self) -> &DriverCapabilities {
        &self.capabilities
    }

    pub fn device_facts(&self) -> &DeviceFacts {
        &self.device_facts
    }

    pub fn load_model(&mut self, desc: &ModelLoadDesc) -> Result<DriverCapabilities> {
        ensure!(self.load_storage.is_none(), "dummy model is already loaded");
        ensure!(
            desc.compiler_version == pie_load_planner::load_plan::compiler_version(),
            "dummy compiler version mismatch"
        );
        self.record_op("load_model");
        let storage = pie_load_planner::host_executor::execute_serialized_plan(
            &desc.load_plan_bytes,
            &desc.snapshot_dir,
        )
        .map_err(|err| anyhow!("dummy LoadPlan execution failed: {err}"))?;
        self.capabilities.snapshot_dir = desc.snapshot_dir.display().to_string();
        self.load_storage = Some(storage);
        Ok(self.capabilities.clone())
    }

    pub fn register_program(&mut self, desc: &PieProgramDesc) -> Result<u64> {
        validate_program_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("register_program");
        let canonical_bytes = copy_bytes(desc.canonical_bytes, "program.canonical_bytes")?;
        let _sidecar = copy_bytes(desc.sidecar_bytes, "program.sidecar_bytes")?;
        ensure!(
            !canonical_bytes.is_empty(),
            "program registration requires canonical PTIR bytes"
        );
        let hash = pie_ptir::container_hash(&canonical_bytes);
        if desc.program_hash != 0 {
            ensure!(
                desc.program_hash == hash,
                "program hash mismatch: descriptor={} actual={hash}",
                desc.program_hash
            );
        }
        let container = container::decode(&canonical_bytes)
            .map_err(|err| anyhow!("program decode failed: {err}"))?;
        let bound = pie_ptir::validate::bind(container, self.model_profile()).map_err(|err| {
            anyhow!(
                "program bind failed: {err} (profile: vocab={}, page_size={})",
                self.model_profile().vocab,
                self.model_profile().page_size
            )
        })?;
        let program = Arc::new(DummyProgram {
            hash,
            intrinsics: collect_intrinsics(&bound),
            bound,
        });
        let program_id = self.next_program_id.fetch_add(1, Ordering::Relaxed);
        self.state
            .lock()
            .unwrap()
            .programs
            .insert(program_id, program);
        Ok(program_id)
    }

    pub fn register_channel(&mut self, desc: &PieChannelDesc) -> Result<PieChannelEndpointBinding> {
        unsafe { validate_channel_desc(desc) }.map_err(|err| anyhow!(err))?;
        self.record_op("register_channel");
        let shape = copy_u32_slice(desc.shape, "channel.shape")?;
        let extern_name = copy_bytes(desc.extern_name, "channel.extern_name")?;
        let cell_bytes = channel_wire_bytes(&shape, desc.dtype)?;
        let mirror_len = cell_bytes
            .checked_mul(desc.capacity as usize + 1)
            .ok_or_else(|| anyhow!("channel mirror size overflow"))?;
        let endpoint = DummyEndpoint {
            channel_id: desc.channel_id,
            shape: shape.clone(),
            dtype: desc.dtype,
            host_role: desc.host_role,
            seeded: desc.seeded != 0,
            capacity: desc.capacity,
            reader_wait_id: desc.reader_wait_id,
            writer_wait_id: desc.writer_wait_id,
            extern_name: if extern_name.is_empty() {
                None
            } else {
                Some(
                    String::from_utf8(extern_name)
                        .map_err(|_| anyhow!("channel extern name must be UTF-8"))?,
                )
            },
            mirror: vec![0; mirror_len].into_boxed_slice(),
            words: (0..4)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            pulled: 0,
            seed_credit: desc.seeded != 0
                && desc.host_role == pie_driver_abi::PIE_CHANNEL_HOST_ROLE_WRITER,
            attachments: HashMap::new(),
            shared: if desc.extern_dir == PIE_CHANNEL_EXTERN_NONE {
                None
            } else {
                let dtype = channel_program_dtype(desc.dtype)?;
                let shape = pie_ptir::types::Shape::new(&shape)
                    .ok_or_else(|| anyhow!("channel shape rank is unsupported"))?;
                Some(ExternChannel::new(
                    ValueType::new(shape, dtype),
                    desc.capacity,
                ))
            },
        };
        let binding = endpoint_binding(&endpoint);
        let mut state = self.state.lock().unwrap();
        ensure!(
            !state.channels.contains_key(&desc.channel_id),
            "channel {} is already registered",
            desc.channel_id
        );
        state
            .channels
            .insert(desc.channel_id, Arc::new(Mutex::new(endpoint)));
        Ok(binding)
    }

    pub fn bind_instance(&mut self, desc: &PieInstanceDesc) -> Result<PieInstanceBinding> {
        unsafe { validate_instance_desc(desc) }.map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("bind_instance");
        ensure!(
            desc.pacing_wait_id != 0,
            "bind requires a nonzero pacing wait id"
        );
        let channel_ids = copy_u64_slice(desc.channel_ids, "instance.channel_ids")?;
        ensure!(
            channel_ids.iter().copied().collect::<HashSet<_>>().len() == channel_ids.len(),
            "bind channel ids must be unique"
        );
        let seed_values = copy_value_descs(desc.seed_values, "instance.seed_values")?;

        let mut state = self.state.lock().unwrap();
        let program = state
            .programs
            .get(&desc.program_id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown program {}", desc.program_id))?;
        let instance_id = if desc.requested_instance_id == 0 {
            self.next_instance_id.fetch_add(1, Ordering::Relaxed)
        } else {
            desc.requested_instance_id
        };
        ensure!(
            !state.instances.contains_key(&instance_id),
            "instance {instance_id} already bound"
        );
        let endpoints = channel_ids
            .iter()
            .map(|channel_id| {
                state
                    .channels
                    .get(channel_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("channel {channel_id} is not registered"))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut externs = Vec::new();
        for (dense, (decl, endpoint)) in program
            .bound
            .container
            .channels
            .iter()
            .zip(endpoints.iter())
            .enumerate()
        {
            let endpoint = endpoint.lock().unwrap();
            ensure_endpoint_matches_program(&endpoint, &program, dense, decl, instance_id)?;
            if program
                .bound
                .container
                .externs
                .iter()
                .any(|binding| binding.chan == dense as u32)
            {
                externs.push((
                    dense as u32,
                    endpoint
                        .shared
                        .clone()
                        .ok_or_else(|| anyhow!("extern channel has no shared ring"))?,
                ));
            }
        }

        ensure!(
            channel_ids.len() == program.bound.container.channels.len(),
            "bind channel count mismatch: got {} expected {}",
            channel_ids.len(),
            program.bound.container.channels.len()
        );

        let mut dense_by_global = HashMap::with_capacity(channel_ids.len());
        for (dense, &global_id) in channel_ids.iter().enumerate() {
            dense_by_global.insert(global_id, dense);
        }
        let seeds = seed_values
            .iter()
            .map(|seed| {
                let dense = dense_by_global
                    .get(&seed.channel_id)
                    .copied()
                    .ok_or_else(|| {
                        anyhow!("seed references unknown channel {}", seed.channel_id)
                    })?;
                let ty = program.bound.channel_types[dense];
                let value = decode_native_value(&seed.bytes, ty)
                    .map_err(|err| anyhow!("seed channel {}: {err}", seed.channel_id))?;
                Ok((dense as u32, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let interp = InterpInstance::new_with_externs(&program.bound, &seeds, &externs)
            .map_err(|err| anyhow!("instance bind failed: {err:?}"))?;
        let channels = program
            .bound
            .container
            .channels
            .iter()
            .zip(channel_ids.iter().copied())
            .zip(endpoints.iter().cloned())
            .enumerate()
            .map(|(dense, ((decl, global_id), endpoint))| DummyChannel {
                global_id,
                host_role: decl.host_role,
                ty: program.bound.channel_types[dense],
                endpoint,
            })
            .collect::<Vec<_>>();
        let instance = Arc::new(BoundInstanceState {
            program: Arc::clone(&program),
            instance_id,
            inner: Mutex::new(BoundInstanceInner {
                interp,
                channels,
                next_pacing_epoch: 1,
                closed: false,
            }),
        });
        for (dense, value) in &seeds {
            if program.bound.container.channels[*dense as usize].host_role != HostRole::Reader {
                continue;
            }
            let bytes = encode_wire_value(value, program.bound.channel_types[*dense as usize])?;
            publish_endpoint_value(
                &endpoints[*dense as usize],
                channel_ids[*dense as usize],
                &bytes,
            )?;
        }
        for (dense, _) in &seeds {
            let role = program.bound.container.channels[*dense as usize].host_role;
            let mut endpoint = endpoints[*dense as usize].lock().unwrap();
            endpoint.words[1].store(1, Ordering::Release);
            if role == HostRole::Writer {
                endpoint.pulled = 1;
            }
        }
        for (dense, endpoint) in endpoints.iter().enumerate() {
            let extern_dir = program
                .bound
                .container
                .externs
                .iter()
                .find(|binding| binding.chan == dense as u32)
                .map(|binding| binding.dir);
            endpoint
                .lock()
                .unwrap()
                .attachments
                .insert(instance_id, extern_dir);
        }
        let binding = PieInstanceBinding {
            instance_id,
            ..PieInstanceBinding::default()
        };
        state.instances.insert(instance_id, instance);
        Ok(binding)
    }

    pub fn launch(&mut self, desc: &PieLaunchDesc, completion: PieCompletion) -> Result<()> {
        unsafe { validate_launch_desc(desc) }.map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("launch");
        // Shape trace for tests that assert on launch geometry (e.g. the
        // prefix-cache trim). Extra entry — existing "launch" filters keep
        // matching.
        self.record_op(&format!(
            "launch-shape tokens={} programs={}",
            desc.token_ids.len, desc.instance_ids.len
        ));
        let instance_ids = copy_u64_slice(desc.instance_ids, "launch.instance_ids")?;
        let terminal_cells = copy_terminal_cell_ptrs(desc.terminal_cells)?;
        let ticket_heads =
            copy_u64_slice(desc.channel_expected_head, "launch.channel_expected_head")?;
        let ticket_tails =
            copy_u64_slice(desc.channel_expected_tail, "launch.channel_expected_tail")?;
        let ticket_indptr =
            copy_u32_slice(desc.channel_ticket_indptr, "launch.channel_ticket_indptr")?;
        ensure!(
            !instance_ids.is_empty(),
            "launch requires at least one bound instance"
        );
        ensure_completion(completion)?;
        validate_launch_shape(desc, instance_ids.len())?;
        if self.reject_launches {
            bail!("launch rejected by dummy test option");
        }
        if self.reject_launches_remaining != 0 {
            self.reject_launches_remaining -= 1;
            bail!("launch rejected by dummy test option");
        }
        ensure_unique_launch_members(&instance_ids, &terminal_cells)?;

        // Test probe: surface the accepted launch's forward geometry, on the
        // launch path (a sleeping observer keeps this fire outstanding).
        if let Some(observer) = &self.launch_observer {
            let observation = LaunchObservation {
                token_ids: copy_u32_slice(desc.token_ids, "launch.token_ids")?,
                qo_indptr: copy_u32_slice(desc.qo_indptr, "launch.qo_indptr")?,
                kv_page_indices: copy_u32_slice(desc.kv_page_indices, "launch.kv_page_indices")?,
                kv_page_indptr: copy_u32_slice(desc.kv_page_indptr, "launch.kv_page_indptr")?,
                kv_last_page_lens: copy_u32_slice(
                    desc.kv_last_page_lens,
                    "launch.kv_last_page_lens",
                )?,
            };
            (observer.0)(&observation);
        }

        let state = self.state.lock().unwrap();
        let instances = instance_ids
            .iter()
            .map(|instance_id| {
                state
                    .instances
                    .get(instance_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("unknown instance {instance_id}"))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(state);

        if self.retry_launches_remaining != 0 {
            self.retry_launches_remaining -= 1;
            let callback = self.prepare_callback(completion)?;
            for (instance, terminal_cell) in instances.iter().zip(&terminal_cells) {
                instance.inner.lock().unwrap().next_pacing_epoch += 1;
                publish_terminal(*terminal_cell, PIE_TERMINAL_OUTCOME_RETRY);
            }
            self.publish_callback(callback, completion);
            return Ok(());
        }

        let mut prepared = Vec::with_capacity(instances.len());
        for (slot, instance) in instances.iter().enumerate() {
            let inner = instance.inner.lock().unwrap();
            ensure!(!inner.closed, "instance {} is closed", instance.instance_id);
            let tickets = launch_tickets(
                &inner,
                &instance.program,
                slot,
                &ticket_heads,
                &ticket_tails,
                &ticket_indptr,
            )?;
            let pacing_epoch = inner.next_pacing_epoch;
            drop(inner);
            prepared.push(LaunchInstanceWork {
                instance: Arc::clone(instance),
                pacing_epoch,
                tickets,
                terminal_cell: terminal_cells[slot],
            });
        }

        let callback = self.prepare_callback(completion)?;
        for instance in &prepared {
            let mut inner = instance.instance.inner.lock().unwrap();
            debug_assert_eq!(inner.next_pacing_epoch, instance.pacing_epoch);
            inner.next_pacing_epoch += 1;
        }
        let results = prepared
            .iter()
            .map(|instance| {
                if self.fail_launches_after_accept {
                    fail_launch_instance(instance, "forced launch failure")
                } else {
                    process_launch_instance(instance)
                }
            })
            .collect::<Vec<_>>();
        for (instance, result) in prepared.iter().zip(results.iter()) {
            publish_terminal(instance.terminal_cell, result.outcome);
        }
        for result in &results {
            for &(wait_id, epoch) in &result.notifications {
                notify_runtime(&self.runtime, wait_id, epoch);
            }
        }
        self.publish_callback(callback, completion);
        Ok(())
    }

    pub fn copy_kv(&mut self, desc: &PieKvCopyDesc, completion: PieCompletion) -> Result<()> {
        validate_kv_copy_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("copy_kv");
        ensure_completion_mode(completion, true)?;
        let _src_page_ids = copy_u32_slice(desc.src_page_ids, "copy_kv.src_page_ids")?;
        let _dst_page_ids = copy_u32_slice(desc.dst_page_ids, "copy_kv.dst_page_ids")?;
        let cells = copy_kv_cells(desc)?;
        if !cells.is_empty() {
            ensure!(
                desc.src_page_ids.len == 0 && desc.dst_page_ids.len == 0,
                "copy_kv cells and whole-page lists are mutually exclusive"
            );
        }
        self.complete_noop(completion)
    }

    pub fn copy_state(&mut self, desc: &PieStateCopyDesc, completion: PieCompletion) -> Result<()> {
        validate_state_copy_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("copy_state");
        ensure_completion_mode(completion, true)?;
        let ranges = copy_state_ranges(desc)?;
        if !ranges.is_empty() {
            ensure!(
                ranges
                    .iter()
                    .all(|range| range.token_count > 0 || range.src_slot_id != range.dst_slot_id),
                "copy_state requires non-empty token counts or distinct src/dst slots"
            );
        }
        self.complete_noop(completion)
    }

    pub fn resize_pool(
        &mut self,
        desc: &PiePoolResizeDesc,
        completion: PieCompletion,
    ) -> Result<()> {
        validate_pool_resize_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("resize_pool");
        ensure_completion_mode(completion, true)?;
        let _maps = copy_pool_ranges(desc.map_ranges, "resize_pool.map_ranges")?;
        let _unmaps = copy_pool_ranges(desc.unmap_ranges, "resize_pool.unmap_ranges")?;
        self.complete_noop(completion)
    }

    pub fn close_instance(&mut self, instance_id: u64) -> Result<()> {
        self.record_op("close_instance");
        let instance = {
            let state = self.state.lock().unwrap();
            state
                .instances
                .get(&instance_id)
                .cloned()
                .ok_or_else(|| anyhow!("unknown instance {instance_id}"))?
        };
        let mut inner = instance.inner.lock().unwrap();
        inner.closed = true;
        let endpoints = inner
            .channels
            .iter()
            .map(|channel| Arc::clone(&channel.endpoint))
            .collect::<Vec<_>>();
        drop(inner);
        self.state.lock().unwrap().instances.remove(&instance_id);
        for endpoint in endpoints {
            endpoint.lock().unwrap().attachments.remove(&instance_id);
        }
        Ok(())
    }

    pub fn close_channel(&mut self, channel_id: u64) -> Result<()> {
        self.record_op("close_channel");
        let mut state = self.state.lock().unwrap();
        let endpoint = state
            .channels
            .get(&channel_id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown channel {channel_id}"))?;
        let endpoint = endpoint.lock().unwrap();
        ensure!(
            endpoint.attachments.is_empty(),
            "channel {channel_id} still has instance attachments"
        );
        endpoint.words[3].store(1, Ordering::Release);
        drop(endpoint);
        state.channels.remove(&channel_id);
        Ok(())
    }

    fn complete_noop(&mut self, completion: PieCompletion) -> Result<()> {
        let callback = self.prepare_callback(completion)?;
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        self.publish_callback(callback, completion);
        Ok(())
    }

    fn model_profile(&self) -> ModelProfile {
        ModelProfile {
            vocab: self.capabilities.vocab_size,
            page_size: self.capabilities.kv_page_size,
            num_layers: 2,
            activation: DType::F32,
            has_mtp_logits: self.capabilities.has_mtp_logits,
            has_mtp_drafts: self.capabilities.has_mtp_drafts,
            has_value_head: self.capabilities.has_value_head,
            kernels: vec![KernelInfo {
                name: "boom".to_string(),
                sink_scope: None,
                replayable: true,
            }],
        }
    }
}

impl Default for DummyDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DummyDriver {
    fn drop(&mut self) {
        for worker in self.callback_workers.drain(..) {
            let _ = worker.join();
        }
        self.record_op("destroy");
    }
}

enum InstanceStepResult {
    Committed(Vec<(u64, u64)>),
    Retry,
}

fn process_launch_instance(instance: &LaunchInstanceWork) -> LaunchInstanceResult {
    let mut notify_waits = Vec::new();
    let outcome = {
        let mut inner = instance.instance.inner.lock().unwrap();
        if !tickets_ready(&inner, &instance.tickets) {
            return LaunchInstanceResult {
                outcome: PIE_TERMINAL_OUTCOME_RETRY,
                notifications: Vec::new(),
            };
        }
        for dense in 0..inner.channels.len() {
            let channel = inner.channels[dense].clone();
            if channel.host_role == HostRole::Writer
                && let Err(err) =
                    pull_writer_ring(&mut inner, &instance.instance.program, dense, &channel)
            {
                poison_instance(&mut inner, &mut notify_waits, &err.to_string());
                return LaunchInstanceResult {
                    outcome: PIE_TERMINAL_OUTCOME_FAILED,
                    notifications: notify_waits,
                };
            }
        }
        let outcome = match run_instance_step(
            &mut inner,
            &instance.instance.program,
            instance.instance.instance_id,
            instance.pacing_epoch,
            &instance.tickets,
        ) {
            Ok(InstanceStepResult::Committed(reader_epochs)) => {
                notify_waits.extend(reader_epochs);
                PIE_TERMINAL_OUTCOME_SUCCESS
            }
            Ok(InstanceStepResult::Retry) => PIE_TERMINAL_OUTCOME_RETRY,
            Err(err) => {
                eprintln!(
                    "[pie-driver-dummy] instance {} launch failed: {err:#}",
                    instance.instance.instance_id
                );
                poison_instance(&mut inner, &mut notify_waits, &err.to_string());
                PIE_TERMINAL_OUTCOME_FAILED
            }
        };
        outcome
    };
    LaunchInstanceResult {
        outcome,
        notifications: notify_waits,
    }
}

fn fail_launch_instance(instance: &LaunchInstanceWork, reason: &str) -> LaunchInstanceResult {
    let mut notify_waits = Vec::new();
    {
        let mut inner = instance.instance.inner.lock().unwrap();
        poison_instance(&mut inner, &mut notify_waits, reason);
    }
    LaunchInstanceResult {
        outcome: PIE_TERMINAL_OUTCOME_FAILED,
        notifications: notify_waits,
    }
}

fn publish_terminal(cell: *mut PieTerminalCell, outcome: u32) {
    if cell.is_null() {
        return;
    }
    {
        unsafe {
            (*cell).reserved0 = 0;
            AtomicU32::from_ptr(cell.cast::<u32>()).store(outcome, Ordering::Release);
        }
    }
}

/// Whether one fire of `bound` consumes (takes) dense channel `dense` — via a
/// stage `ChanTake` or a consuming descriptor port. Take is register-like
/// within a pass (the ring index bumps at most once per fire), so this is the
/// per-fire consume count.
fn fire_takes_channel(bound: &BoundTrace, dense: u32) -> bool {
    let stage_take = bound.container.stages.iter().any(|stage| {
        stage
            .ops
            .iter()
            .any(|op| matches!(op, Op::ChanTake(c) if *c == dense))
    });
    let port_take = bound.container.ports.iter().any(|binding| {
        binding.port.consumes()
            && matches!(binding.source, container::PortSource::Channel(c) if c == dense)
    });
    stage_take || port_take
}

fn fire_puts_channel(bound: &BoundTrace, dense: u32) -> bool {
    bound.container.stages.iter().any(|stage| {
        stage
            .ops
            .iter()
            .any(|op| matches!(op, Op::ChanPut { chan, .. } if *chan == dense))
    })
}

fn fire_requires_channel_input(bound: &BoundTrace, dense: u32) -> bool {
    if bound
        .container
        .ports
        .iter()
        .any(|binding| matches!(binding.source, container::PortSource::Channel(c) if c == dense))
    {
        return true;
    }
    for stage in &bound.container.stages {
        for op in &stage.ops {
            match *op {
                Op::ChanPut { chan, .. } if chan == dense => return false,
                Op::ChanTake(chan) | Op::ChanRead(chan) if chan == dense => return true,
                _ => {}
            }
        }
    }
    false
}

fn launch_tickets(
    inner: &BoundInstanceInner,
    program: &DummyProgram,
    program_index: usize,
    heads: &[u64],
    tails: &[u64],
    indptr: &[u32],
) -> Result<Vec<ChannelTicket>> {
    if !heads.is_empty() || !tails.is_empty() {
        ensure!(
            indptr.len() > program_index + 1,
            "channel tickets require a per-program CSR"
        );
        let lo = indptr[program_index] as usize;
        let hi = indptr[program_index + 1] as usize;
        ensure!(
            hi - lo == inner.channels.len(),
            "channel ticket segment length {} does not match instance channel count {}",
            hi - lo,
            inner.channels.len()
        );
        return Ok(heads[lo..hi]
            .iter()
            .zip(&tails[lo..hi])
            .enumerate()
            .map(|(dense, (&head, &tail))| ChannelTicket {
                expected_head: (head != u64::MAX).then_some(head),
                expected_tail: (tail != u64::MAX).then_some(tail),
                require_input: fire_requires_channel_input(&program.bound, dense as u32),
            })
            .collect());
    }

    Ok(inner
        .channels
        .iter()
        .enumerate()
        .map(|(dense, channel)| {
            let endpoint = channel.endpoint.lock().unwrap();
            let consume = fire_takes_channel(&program.bound, dense as u32);
            let publish = fire_puts_channel(&program.bound, dense as u32);
            ChannelTicket {
                expected_head: consume.then(|| endpoint.words[0].load(Ordering::Acquire)),
                expected_tail: publish.then(|| endpoint.words[1].load(Ordering::Acquire)),
                require_input: fire_requires_channel_input(&program.bound, dense as u32),
            }
        })
        .collect())
}

fn tickets_ready(inner: &BoundInstanceInner, tickets: &[ChannelTicket]) -> bool {
    inner.channels.iter().zip(tickets).all(|(channel, ticket)| {
        let endpoint = channel.endpoint.lock().unwrap();
        let head = endpoint.words[0].load(Ordering::Acquire);
        let tail = endpoint.words[1].load(Ordering::Acquire);
        let consume_ready = ticket.expected_head.is_none_or(|expected| head == expected)
            && (!ticket.require_input || tail > head);
        let publish_ready = ticket.expected_tail.is_none_or(|expected| {
            let same_fire_consume = u64::from(ticket.expected_head.is_some());
            tail == expected
                && tail.saturating_sub(head) < u64::from(endpoint.capacity) + same_fire_consume
        });
        consume_ready && publish_ready
    })
}

/// §4.3 driver pull: move host-published writer-ring entries (mirror cells up
/// to the release-published tail word) into the interp's channel queue,
/// advancing the endpoint's `pulled` cursor. Interp backpressure leaves the
/// remainder in the ring for a later fire.
fn pull_writer_ring(
    inner: &mut BoundInstanceInner,
    program: &DummyProgram,
    dense: usize,
    channel: &DummyChannel,
) -> Result<()> {
    loop {
        let (sequence, wire) = {
            let endpoint = channel.endpoint.lock().unwrap();
            let tail = endpoint.words[1].load(Ordering::Acquire);
            if endpoint.pulled >= tail {
                return Ok(());
            }
            let cell = channel_wire_bytes(&endpoint.shape, endpoint.dtype)?;
            let cap1 = u64::from(endpoint.capacity) + 1;
            let offset = (endpoint.pulled % cap1) as usize * cell;
            (
                endpoint.pulled,
                endpoint.mirror[offset..offset + cell].to_vec(),
            )
        };
        let value = decode_wire_value(&wire, channel.ty)
            .map_err(|err| anyhow!("writer ring channel {}: {err}", channel.global_id))?;
        match inner.interp.host_put(&program.bound, dense as u32, value) {
            Ok(()) => {
                channel.endpoint.lock().unwrap().pulled = sequence + 1;
            }
            Err(HostError::WouldBlock) => return Ok(()),
            Err(err) => bail!(
                "writer ring pull failed for channel {}: {err:?}",
                channel.global_id
            ),
        }
    }
}

fn run_instance_step(
    inner: &mut BoundInstanceInner,
    program: &DummyProgram,
    instance_id: u64,
    pacing_epoch: u64,
    tickets: &[ChannelTicket],
) -> Result<InstanceStepResult> {
    let mut notify_waits = Vec::new();
    let pass_inputs = build_pass_inputs(
        &program.intrinsics,
        pacing_epoch,
        instance_id,
        program.hash,
        program.bound.profile.vocab,
    );
    let report = inner
        .interp
        .step(&program.bound, &pass_inputs, &mut NoKernels)
        .map_err(|err| anyhow!("interp step failed: {}", format_step_error(&err)))?;
    if !report.committed {
        return Ok(InstanceStepResult::Retry);
    }

    for (dense, ticket) in tickets.iter().enumerate() {
        let channel = &inner.channels[dense];
        let mut endpoint = channel.endpoint.lock().unwrap();
        if fire_takes_channel(&program.bound, dense as u32) && endpoint.seed_credit {
            endpoint.seed_credit = false;
        }
        if let Some(expected) = ticket.expected_head {
            let next = expected + 1;
            endpoint.words[0].store(next, Ordering::Release);
            if channel.host_role == HostRole::Writer {
                notify_waits.push((endpoint.writer_wait_id, next));
            }
        }
        if let Some(expected) = ticket.expected_tail
            && channel.host_role == HostRole::None
        {
            endpoint.words[1].store(expected + 1, Ordering::Release);
        }
    }

    for dense in 0..inner.channels.len() {
        let channel = inner.channels[dense].clone();
        if channel.host_role != HostRole::Reader {
            continue;
        }
        loop {
            match inner.interp.host_take(&program.bound, dense as u32) {
                Ok(value) => {
                    let bytes = encode_wire_value(&value, channel.ty).map_err(|err| {
                        anyhow!("channel {} encode failed: {err}", channel.global_id)
                    })?;
                    let publication = publish_reader_value(inner, dense, &bytes)?;
                    notify_waits.push(publication);
                }
                Err(HostError::WouldBlock) => break,
                Err(err) => bail!(
                    "host take failed for channel {}: {err:?}",
                    channel.global_id
                ),
            }
        }
    }
    Ok(InstanceStepResult::Committed(notify_waits))
}

fn poison_instance(
    inner: &mut BoundInstanceInner,
    notify_waiters: &mut Vec<(u64, u64)>,
    _reason: &str,
) {
    inner.interp.poison();
    for channel in &inner.channels {
        let endpoint = channel.endpoint.lock().unwrap();
        let tail = endpoint.words[1].load(Ordering::Acquire);
        let poison_epoch = tail.saturating_add(1).max(1);
        endpoint.words[2].store(poison_epoch, Ordering::Release);
        let wait_id = match channel.host_role {
            HostRole::Reader => endpoint.reader_wait_id,
            HostRole::Writer => endpoint.writer_wait_id,
            HostRole::None => 0,
        };
        notify_waiters.push((wait_id, poison_epoch));
    }
}

fn publish_reader_value(
    inner: &mut BoundInstanceInner,
    dense: usize,
    bytes: &[u8],
) -> Result<(u64, u64)> {
    let channel = &inner.channels[dense];
    let tail = publish_endpoint_value(&channel.endpoint, channel.global_id, bytes)?;
    let wait_id = channel.endpoint.lock().unwrap().reader_wait_id;
    Ok((wait_id, tail))
}

fn publish_endpoint_value(
    endpoint: &Arc<Mutex<DummyEndpoint>>,
    channel_id: u64,
    bytes: &[u8],
) -> Result<u64> {
    let mut endpoint = endpoint.lock().unwrap();
    let cell_bytes = channel_wire_bytes(&endpoint.shape, endpoint.dtype)?;
    ensure!(
        bytes.len() == cell_bytes,
        "channel {} wrote {} bytes, expected {}",
        channel_id,
        bytes.len(),
        cell_bytes
    );
    let tail = endpoint.words[1].load(Ordering::Acquire);
    let head = endpoint.words[0].load(Ordering::Acquire);
    ensure!(
        tail.saturating_sub(head) < endpoint.capacity as u64,
        "channel {} has no reserved output capacity",
        channel_id
    );
    let ring_cells = endpoint.capacity as usize + 1;
    let slot = (tail as usize % ring_cells) * cell_bytes;
    endpoint.mirror[slot..slot + cell_bytes].copy_from_slice(bytes);
    let next_tail = tail + 1;
    endpoint.words[1].store(next_tail, Ordering::Release);
    endpoint.words[2].store(0, Ordering::Release);
    Ok(next_tail)
}

fn notify_completion(
    runtime: &SendableRuntimeCallbacks,
    wait_id: u64,
    epoch: u64,
    callback_delay_ms: u64,
) {
    if wait_id == 0 || epoch == 0 {
        return;
    }
    if callback_delay_ms != 0 {
        std::thread::sleep(std::time::Duration::from_millis(callback_delay_ms));
    }
    if let Some(log) = &runtime.operation_log {
        log.lock().unwrap().push("callback".to_string());
    }
    notify_runtime(runtime, wait_id, epoch);
}

fn notify_runtime(runtime: &SendableRuntimeCallbacks, wait_id: u64, epoch: u64) {
    if wait_id == 0 || epoch == 0 {
        return;
    }
    if let Some(notify) = runtime.notify {
        let _ = std::panic::catch_unwind(|| unsafe {
            notify(runtime.ctx as *mut std::ffi::c_void, wait_id, epoch)
        });
    }
}

fn channel_program_dtype(dtype: u8) -> Result<DType> {
    match dtype {
        PIE_CHANNEL_DTYPE_F32 | PIE_CHANNEL_DTYPE_ACT => Ok(DType::F32),
        PIE_CHANNEL_DTYPE_I32 => Ok(DType::I32),
        PIE_CHANNEL_DTYPE_U32 => Ok(DType::U32),
        PIE_CHANNEL_DTYPE_BOOL => Ok(DType::Bool),
        _ => bail!("unsupported channel dtype {dtype}"),
    }
}

fn channel_wire_bytes(shape: &[u32], dtype: u8) -> Result<usize> {
    let numel = shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim as usize)
            .ok_or_else(|| anyhow!("channel shape size overflow"))
    })?;
    Ok(if dtype == PIE_CHANNEL_DTYPE_BOOL {
        numel.div_ceil(8)
    } else {
        numel
            .checked_mul(4)
            .ok_or_else(|| anyhow!("channel byte size overflow"))?
    })
}

fn endpoint_binding(endpoint: &DummyEndpoint) -> PieChannelEndpointBinding {
    PieChannelEndpointBinding {
        channel_id: endpoint.channel_id,
        mirror_base: endpoint.mirror.as_ptr() as u64,
        word_base: endpoint.words.as_ptr() as u64,
        mirror_bytes: endpoint.mirror.len() as u64,
        word_bytes: (endpoint.words.len() * std::mem::size_of::<u64>()) as u64,
        cell_bytes: channel_wire_bytes(&endpoint.shape, endpoint.dtype)
            .expect("validated endpoint geometry") as u32,
        capacity: endpoint.capacity,
        head_word_index: 0,
        tail_word_index: 1,
        poison_word_index: 2,
        closed_word_index: 3,
    }
}

fn ensure_endpoint_matches_program(
    endpoint: &DummyEndpoint,
    program: &DummyProgram,
    dense: usize,
    decl: &pie_ptir::container::ChannelDecl,
    instance_id: u64,
) -> Result<()> {
    ensure!(
        endpoint.shape == decl.shape.dims()
            && endpoint.dtype == decl.dtype.tag()
            && endpoint.host_role == decl.host_role as u8
            && endpoint.seeded == decl.seeded
            && endpoint.capacity == decl.capacity,
        "channel {} contract conflicts with program declaration",
        endpoint.channel_id
    );
    let extern_decl = program
        .bound
        .container
        .externs
        .iter()
        .find(|binding| binding.chan == dense as u32);
    match extern_decl {
        None => {
            ensure!(
                endpoint.extern_name.is_none() && endpoint.attachments.is_empty(),
                "private channel {} is already attached",
                endpoint.channel_id
            );
        }
        Some(extern_decl) => {
            let name = &program.bound.container.names[extern_decl.name as usize];
            ensure!(
                endpoint.extern_name.as_deref() == Some(name.as_str()),
                "channel {} extern name conflicts with program",
                endpoint.channel_id
            );
            ensure!(
                !endpoint
                    .attachments
                    .values()
                    .any(|dir| *dir == Some(extern_decl.dir)),
                "channel {} extern {:?} endpoint is already attached",
                endpoint.channel_id,
                extern_decl.dir
            );
        }
    }
    ensure!(
        !endpoint.attachments.contains_key(&instance_id),
        "channel {} already attached to instance {instance_id}",
        endpoint.channel_id
    );
    Ok(())
}

fn collect_intrinsics(bound: &BoundTrace) -> ProgramIntrinsics {
    let mut out = ProgramIntrinsics::default();
    for stage in &bound.container.stages {
        for op in &stage.ops {
            let Op::IntrinsicVal { intr, shape, dtype } = op else {
                continue;
            };
            let target = match intr {
                IntrinsicId::Logits => &mut out.logits,
                IntrinsicId::MtpLogits => &mut out.mtp_logits,
                IntrinsicId::Hidden => &mut out.hidden,
                IntrinsicId::Query => &mut out.query,
                IntrinsicId::ValueHead => &mut out.value_head,
                IntrinsicId::MtpDrafts => &mut out.mtp_drafts,
                IntrinsicId::Layer => continue,
            };
            target.get_or_insert(ValueType::new(*shape, *dtype));
        }
    }
    out
}

fn build_pass_inputs(
    intrinsics: &ProgramIntrinsics,
    launch_epoch: u64,
    instance_id: u64,
    program_hash: u64,
    vocab: u32,
) -> PassInputs {
    let base = launch_epoch ^ instance_id ^ program_hash;
    let logits = intrinsics
        .logits
        .map(|ty| deterministic_logits(ty, base, vocab));
    let mtp_logits = intrinsics
        .mtp_logits
        .map(|ty| deterministic_logits(ty, base.wrapping_add(17), vocab));
    let hidden = intrinsics
        .hidden
        .map(|ty| deterministic_value(ty, base.wrapping_add(31)));
    let value_head = intrinsics
        .value_head
        .map(|ty| deterministic_value(ty, base.wrapping_add(47)));
    let mtp_drafts = intrinsics
        .mtp_drafts
        .map(|ty| deterministic_drafts(ty, base, vocab.max(2)));
    let query = intrinsics
        .query
        .map(|ty| {
            (0..2)
                .map(|layer| deterministic_value(ty, base.wrapping_add(layer as u64 * 13)))
                .collect()
        })
        .unwrap_or_default();
    PassInputs {
        logits,
        mtp_logits,
        mtp_drafts,
        hidden,
        value_head,
        query,
    }
}

fn deterministic_logits(ty: ValueType, base: u64, vocab: u32) -> Value {
    let rows = ty.shape.rows() as usize;
    let cols = ty.shape.last_len().unwrap_or(vocab.max(1)) as usize;
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let favored = ((base as usize) + row) % cols.max(1);
        for col in 0..cols {
            let value = if col == favored {
                5.0
            } else if col == (favored + 1) % cols.max(1) {
                2.0
            } else {
                -((col as f32 + 1.0) / cols.max(1) as f32)
            };
            values.push(value);
        }
    }
    Value::F32(values)
}

fn deterministic_value(ty: ValueType, base: u64) -> Value {
    let len = ty.shape.numel().max(1) as usize;
    match ty.dtype {
        DType::F32 => Value::F32(
            (0..len)
                .map(|idx| base as f32 * 0.01 + idx as f32 * 0.25)
                .collect(),
        ),
        DType::I32 => Value::I32(
            (0..len)
                .map(|idx| (base as i32).wrapping_add(idx as i32))
                .collect(),
        ),
        DType::U32 => Value::U32((0..len).map(|idx| base as u32 + idx as u32).collect()),
        DType::Bool => Value::Bool(
            (0..len)
                .map(|idx| ((base as usize + idx) & 1) == 0)
                .collect(),
        ),
    }
}

fn deterministic_drafts(ty: ValueType, base: u64, vocab: u32) -> Value {
    let len = ty.shape.numel().max(1) as usize;
    let mut drafts = Vec::with_capacity(len);
    for idx in 0..len {
        drafts.push(((base as u32 + idx as u32) % vocab.max(2)) as i32);
    }
    Value::I32(drafts)
}

fn format_step_error(err: &StepError) -> String {
    match err {
        StepError::Poisoned => "poisoned".to_string(),
        StepError::KernelFault { name, message } => format!("kernel {name} fault: {message}"),
        StepError::MissingIntrinsic(intr) => format!("missing intrinsic {}", intr.name()),
        StepError::Fault(message) => message.clone(),
    }
}

fn ensure_abi(abi_version: u32) -> Result<()> {
    ensure!(
        abi_version == pie_driver_abi::PIE_DRIVER_ABI_VERSION,
        "unexpected ABI version {abi_version}"
    );
    Ok(())
}

fn ensure_completion(completion: PieCompletion) -> Result<()> {
    ensure_completion_mode(completion, false)
}

fn ensure_completion_mode(completion: PieCompletion, require_terminal_cell: bool) -> Result<()> {
    validate_completion(completion, require_terminal_cell).map_err(|err| anyhow!(err.message()))?;
    if require_terminal_cell {
        ensure_terminal_cell_pending(completion.terminal_cell)?;
    } else {
        ensure!(
            completion.terminal_cell.is_null(),
            "launch completion terminal_cell must be null"
        );
    }
    Ok(())
}

fn copy_terminal_cell_ptrs(slice: PieTerminalCellPtrSlice) -> Result<Vec<*mut PieTerminalCell>> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec())
}

fn ensure_unique_launch_members(
    instance_ids: &[u64],
    terminal_cells: &[*mut PieTerminalCell],
) -> Result<()> {
    let mut unique_instances = HashSet::with_capacity(instance_ids.len());
    let mut unique_cells = HashSet::with_capacity(terminal_cells.len());
    for (&instance_id, &terminal_cell) in instance_ids.iter().zip(terminal_cells) {
        ensure!(
            unique_instances.insert(instance_id),
            "launch contains duplicate instance {instance_id}"
        );
        ensure!(
            unique_cells.insert(terminal_cell as usize),
            "launch terminal cells must be distinct"
        );
        ensure_terminal_cell_pending(terminal_cell)?;
    }
    Ok(())
}

fn ensure_terminal_cell_pending(cell: *mut PieTerminalCell) -> Result<()> {
    let outcome = unsafe { AtomicU32::from_ptr(cell.cast::<u32>()).load(Ordering::Acquire) };
    ensure!(
        outcome == pie_driver_abi::PIE_TERMINAL_OUTCOME_PENDING,
        "terminal cell must be Pending before acceptance"
    );
    ensure!(
        unsafe { (*cell).reserved0 } == 0,
        "terminal cell reserved0 must be zero"
    );
    Ok(())
}

fn validate_launch_shape(desc: &PieLaunchDesc, _instance_count: usize) -> Result<()> {
    let token_ids = copy_u32_slice(desc.token_ids, "launch.token_ids")?;
    let kv_page_indices = copy_u32_slice(desc.kv_page_indices, "launch.kv_page_indices")?;
    let kv_page_indptr = copy_u32_slice(desc.kv_page_indptr, "launch.kv_page_indptr")?;
    let kv_last_page_lens = copy_u32_slice(desc.kv_last_page_lens, "launch.kv_last_page_lens")?;
    let qo_indptr = copy_u32_slice(desc.qo_indptr, "launch.qo_indptr")?;
    let sampling_indices = copy_u32_slice(desc.sampling_indices, "launch.sampling_indices")?;
    let sampling_indptr = copy_u32_slice(desc.sampling_indptr, "launch.sampling_indptr")?;
    let rs_slot_ids = copy_u32_slice(desc.rs_slot_ids, "launch.rs_slot_ids")?;
    let rs_slot_flags = copy_u8_slice(desc.rs_slot_flags, "launch.rs_slot_flags")?;
    let rs_fold_lens = copy_u32_slice(desc.rs_fold_lens, "launch.rs_fold_lens")?;
    let rs_buffer_slot_ids = copy_u32_slice(desc.rs_buffer_slot_ids, "launch.rs_buffer_slot_ids")?;
    let rs_buffer_slot_indptr =
        copy_u32_slice(desc.rs_buffer_slot_indptr, "launch.rs_buffer_slot_indptr")?;
    let image_indptr = copy_u32_slice(desc.image_indptr, "launch.image_indptr")?;
    let image_pixel_indptr = copy_u32_slice(desc.image_pixel_indptr, "launch.image_pixel_indptr")?;
    let audio_feature_indptr =
        copy_u32_slice(desc.audio_feature_indptr, "launch.audio_feature_indptr")?;
    let audio_indptr = copy_u32_slice(desc.audio_indptr, "launch.audio_indptr")?;
    let _image_pixels = copy_bytes(desc.image_pixels, "launch.image_pixels")?;
    let _audio_features = copy_bytes(desc.audio_features, "launch.audio_features")?;
    let _kv_len = copy_u32_slice(desc.kv_len, "launch.kv_len")?;
    let _kv_len_device = copy_u64_slice(desc.kv_len_device, "launch.kv_len_device")?;
    let mask_request = copy_u32_slice(desc.masks.request_indptr, "launch.masks.request_indptr")?;
    let mask_word = copy_u32_slice(desc.masks.word_indptr, "launch.masks.word_indptr")?;
    let mask_words = copy_u32_slice(desc.masks.words, "launch.masks.words")?;
    let row_count = qo_indptr.len().saturating_sub(1);

    if !kv_page_indptr.is_empty() || !kv_page_indices.is_empty() {
        validate_indptr(
            &kv_page_indptr,
            kv_page_indices.len(),
            row_count,
            "launch.kv_page_indptr",
        )?;
    }
    if !kv_last_page_lens.is_empty() {
        ensure!(
            kv_last_page_lens.len() == row_count,
            "launch.kv_last_page_lens must have one entry per resolved row"
        );
    }
    if !qo_indptr.is_empty() {
        validate_indptr(&qo_indptr, token_ids.len(), row_count, "launch.qo_indptr")?;
        ensure!(
            qo_indptr.windows(2).all(|w| w[0] <= w[1]),
            "launch.qo_indptr must be monotonic"
        );
    }
    if !sampling_indptr.is_empty() || !sampling_indices.is_empty() {
        validate_indptr(
            &sampling_indptr,
            sampling_indices.len(),
            row_count,
            "launch.sampling_indptr",
        )?;
    }
    if !rs_slot_ids.is_empty() || !rs_slot_flags.is_empty() {
        ensure!(
            rs_slot_ids.len() == rs_slot_flags.len(),
            "launch rs slot id/flag vectors must be parallel"
        );
        if !qo_indptr.is_empty() {
            ensure!(
                rs_slot_ids.len() == row_count,
                "launch rs slot vectors must match resolved qo rows"
            );
        }
    }
    if !rs_fold_lens.is_empty() {
        ensure!(
            rs_fold_lens.len() == rs_slot_ids.len(),
            "launch rs fold lengths must match rs slot vectors"
        );
    }
    if !rs_buffer_slot_indptr.is_empty() || !rs_buffer_slot_ids.is_empty() {
        validate_indptr(
            &rs_buffer_slot_indptr,
            rs_buffer_slot_ids.len(),
            row_count,
            "launch.rs_buffer_slot_indptr",
        )?;
    }
    if !mask_request.is_empty() {
        ensure!(
            mask_request.len() == row_count + 1,
            "launch.masks.request_indptr length mismatch"
        );
        ensure!(
            mask_request.windows(2).all(|w| w[0] <= w[1]),
            "launch.masks.request_indptr must be monotonic"
        );
        ensure!(
            mask_request.last().copied().unwrap_or_default() as usize + 1 == mask_word.len(),
            "launch.masks.word_indptr length mismatch"
        );
        validate_flat_indptr(&mask_word, mask_words.len(), "launch.masks.word_indptr")?;
    }
    if !image_indptr.is_empty() {
        ensure!(
            image_indptr.len() == row_count + 1,
            "launch.image_indptr length mismatch"
        );
        ensure!(
            image_indptr.windows(2).all(|w| w[0] <= w[1]),
            "launch.image_indptr must be monotonic"
        );
    }
    if !image_pixel_indptr.is_empty() {
        validate_flat_indptr(
            &image_pixel_indptr,
            desc.image_pixels.len,
            "launch.image_pixel_indptr",
        )?;
    }
    if !audio_feature_indptr.is_empty() {
        validate_flat_indptr(
            &audio_feature_indptr,
            desc.audio_features.len,
            "launch.audio_feature_indptr",
        )?;
    }
    if !audio_indptr.is_empty() {
        ensure!(
            audio_indptr.len() == row_count + 1,
            "launch.audio_indptr length mismatch"
        );
        ensure!(
            audio_indptr.windows(2).all(|w| w[0] <= w[1]),
            "launch.audio_indptr must be monotonic"
        );
    }
    Ok(())
}

fn validate_indptr(
    indptr: &[u32],
    values_len: usize,
    expected_segments: usize,
    name: &str,
) -> Result<()> {
    ensure!(
        indptr.len() == expected_segments + 1,
        "{name} must have {expected_segments}+1 entries"
    );
    validate_flat_indptr(indptr, values_len, name)
}

fn validate_flat_indptr(indptr: &[u32], values_len: usize, name: &str) -> Result<()> {
    ensure!(!indptr.is_empty(), "{name} must start with a zero entry");
    ensure!(indptr[0] == 0, "{name} must start at zero");
    ensure!(
        indptr.windows(2).all(|w| w[0] <= w[1]),
        "{name} must be monotonic"
    );
    ensure!(
        indptr.last().copied().unwrap_or_default() as usize == values_len,
        "{name} last entry must match value count"
    );
    Ok(())
}

fn copy_bytes(bytes: PieBytes, name: &str) -> Result<Vec<u8>> {
    if bytes.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !bytes.ptr.is_null(),
        "{name} pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(bytes.ptr, bytes.len) }.to_vec())
}

fn copy_u8_slice(slice: pie_driver_abi::PieU8Slice, name: &str) -> Result<Vec<u8>> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !slice.ptr.is_null(),
        "{name} pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec())
}

fn copy_u32_slice(slice: PieU32Slice, name: &str) -> Result<Vec<u32>> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !slice.ptr.is_null(),
        "{name} pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec())
}

fn copy_u64_slice(slice: PieU64Slice, name: &str) -> Result<Vec<u64>> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !slice.ptr.is_null(),
        "{name} pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec())
}

fn copy_value_descs(slice: PieChannelValueDescSlice, name: &str) -> Result<Vec<OwnedValueDesc>> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !slice.ptr.is_null(),
        "{name} pointer is null with nonzero length"
    );
    unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }
        .iter()
        .map(|value| {
            Ok(OwnedValueDesc {
                channel_id: value.channel_id,
                bytes: copy_bytes(value.bytes, &format!("{name}[{}].bytes", value.channel_id))?,
            })
        })
        .collect()
}

fn copy_kv_cells(desc: &PieKvCopyDesc) -> Result<Vec<pie_driver_abi::PieKvMoveCell>> {
    if desc.cells.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !desc.cells.ptr.is_null(),
        "copy_kv.cells pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(desc.cells.ptr, desc.cells.len) }.to_vec())
}

fn copy_state_ranges(desc: &PieStateCopyDesc) -> Result<Vec<pie_driver_abi::PieStateCopyRange>> {
    if desc.slot_ranges.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !desc.slot_ranges.ptr.is_null(),
        "copy_state.slot_ranges pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(desc.slot_ranges.ptr, desc.slot_ranges.len) }.to_vec())
}

fn copy_pool_ranges(
    slice: pie_driver_abi::PiePoolRangeSlice,
    name: &str,
) -> Result<Vec<pie_driver_abi::PiePoolRange>> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !slice.ptr.is_null(),
        "{name} pointer is null with nonzero length"
    );
    Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec())
}

fn wire_len(ty: ValueType) -> usize {
    let numel = ty.shape.numel().max(1) as usize;
    match ty.dtype {
        DType::Bool => numel.div_ceil(8),
        _ => numel * container::const_elem_size(ty.dtype),
    }
}

fn native_len(ty: ValueType) -> usize {
    ty.shape.numel().max(1) as usize * container::const_elem_size(ty.dtype)
}

fn decode_native_value(bytes: &[u8], ty: ValueType) -> Result<Value> {
    let expected = native_len(ty);
    ensure!(
        bytes.len() == expected,
        "{} bytes, expected {expected}",
        bytes.len()
    );
    Ok(match ty.dtype {
        DType::F32 => Value::F32(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        DType::I32 => Value::I32(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        DType::U32 => Value::U32(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        DType::Bool => Value::Bool(bytes.iter().map(|&byte| byte != 0).collect()),
    })
}

fn decode_wire_value(bytes: &[u8], ty: ValueType) -> Result<Value> {
    match ty.dtype {
        DType::Bool => {
            ensure!(
                bytes.len() == wire_len(ty),
                "{} bytes, expected {}",
                bytes.len(),
                wire_len(ty)
            );
            let native = unpack_bool(bytes, ty.shape.numel().max(1) as usize);
            Ok(Value::Bool(
                native.into_iter().map(|byte| byte != 0).collect(),
            ))
        }
        _ => decode_native_value(bytes, ty),
    }
}

fn encode_wire_value(value: &Value, ty: ValueType) -> Result<Vec<u8>> {
    ensure!(
        value_matches(value, ty),
        "value does not match channel type {:?}",
        ty
    );
    Ok(match value {
        Value::F32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Value::I32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Value::U32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Value::Bool(values) => pack_bool(
            &values
                .iter()
                .map(|&value| u8::from(value))
                .collect::<Vec<_>>(),
        ),
    })
}

fn value_matches(value: &Value, ty: ValueType) -> bool {
    value.dtype() == ty.dtype && value.len() as u64 == ty.shape.numel().max(1)
}

fn pack_bool(native: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; native.len().div_ceil(8)];
    for (idx, byte) in native.iter().enumerate() {
        if *byte != 0 {
            out[idx / 8] |= 1 << (idx % 8);
        }
    }
    out
}

fn unpack_bool(wire: &[u8], numel: usize) -> Vec<u8> {
    (0..numel)
        .map(|idx| (wire.get(idx / 8).copied().unwrap_or(0) >> (idx % 8)) & 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_driver_abi::{
        PIE_CHANNEL_EXTERN_EXPORT, PIE_CHANNEL_HOST_ROLE_NONE, PIE_TERMINAL_OUTCOME_PENDING,
        PieChannelValueDesc,
    };
    use pie_ptir::container::{
        ChanDType, ChannelDecl, ExternDecl, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::expand;
    use pie_ptir::registry::{Port, Stage};
    use pie_ptir::types::{Literal, Shape};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[derive(Clone, Copy, Default)]
    struct PieChannelWait {
        reader_wait_id: u64,
        writer_wait_id: u64,
    }

    struct TestInstanceBinding {
        instance_id: u64,
        endpoints: Vec<PieChannelEndpointBinding>,
        reader_dense: Vec<usize>,
        mirror_bytes: u64,
        word_count: u32,
        channel_count: u32,
    }

    #[derive(Clone, Copy)]
    struct TestChannelBinding {
        channel_id: u64,
        cell_bytes: u32,
        mirror_offset: u64,
        head_word_index: u32,
        tail_word_index: u32,
        poison_word_index: u32,
        endpoint_index: usize,
    }

    #[derive(Default)]
    struct CallbackState {
        notifications: Mutex<Vec<(u64, u64)>>,
        count: AtomicUsize,
        pair: (Mutex<usize>, Condvar),
    }

    unsafe extern "C" fn notify(ctx: *mut c_void, wait_id: u64, epoch: u64) {
        let state = unsafe { &*(ctx as *const CallbackState) };
        state.notifications.lock().unwrap().push((wait_id, epoch));
        let next = state.count.fetch_add(1, Ordering::SeqCst) + 1;
        let (lock, cv) = &state.pair;
        *lock.lock().unwrap() = next;
        cv.notify_all();
    }

    impl CallbackState {
        fn notifications(&self) -> Vec<(u64, u64)> {
            self.notifications.lock().unwrap().clone()
        }

        fn wait_for_notification(&self, expected: (u64, u64)) {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (lock, cv) = &self.pair;
            let mut seen = lock.lock().unwrap();
            loop {
                if self.notifications.lock().unwrap().contains(&expected) {
                    return;
                }
                let now = Instant::now();
                assert!(
                    now < deadline,
                    "timed out waiting for notification {expected:?}"
                );
                let timeout = deadline.saturating_duration_since(now);
                let (next_seen, wait) = cv.wait_timeout(seen, timeout).unwrap();
                seen = next_seen;
                assert!(
                    !wait.timed_out(),
                    "timed out waiting for notification {expected:?}"
                );
            }
        }

        fn notifications_for(&self, wait_id: u64) -> Vec<(u64, u64)> {
            self.notifications()
                .into_iter()
                .filter(|notice| notice.0 == wait_id)
                .collect()
        }
    }

    fn driver_with_callbacks(callback_delay_ms: u64) -> (DummyDriver, Arc<CallbackState>) {
        let state = Arc::new(CallbackState::default());
        (
            DummyDriver::with_runtime(
                DummyDriverOptions {
                    callback_delay_ms,
                    vocab_size: 8,
                    ..DummyDriverOptions::default()
                },
                PieRuntimeCallbacks {
                    abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
                    reserved0: 0,
                    ctx: Arc::as_ptr(&state) as *mut c_void,
                    notify: Some(notify),
                },
            ),
            state,
        )
    }

    fn pending_terminal_cell() -> PieTerminalCell {
        PieTerminalCell {
            outcome: 0,
            reserved0: 0,
        }
    }

    fn bind_program(
        driver: &mut DummyDriver,
        bytes: Vec<u8>,
        channel_ids: &[u64],
        waits: &[PieChannelWait],
        seeds: &[OwnedValueDesc],
        pacing_wait_id: u64,
    ) -> TestInstanceBinding {
        let container = container::decode(&bytes).unwrap();
        assert_eq!(channel_ids.len(), container.channels.len());
        assert_eq!(waits.len(), container.channels.len());
        let mut endpoints = Vec::with_capacity(channel_ids.len());
        let mut reader_dense = Vec::new();
        for (dense, ((&channel_id, wait), decl)) in channel_ids
            .iter()
            .zip(waits)
            .zip(&container.channels)
            .enumerate()
        {
            let extern_decl = container
                .externs
                .iter()
                .find(|binding| binding.chan == dense as u32);
            let (extern_dir, extern_name) = match extern_decl {
                None => (PIE_CHANNEL_EXTERN_NONE, &[][..]),
                Some(binding) => (
                    match binding.dir {
                        ExternDir::Import => pie_driver_abi::PIE_CHANNEL_EXTERN_IMPORT,
                        ExternDir::Export => pie_driver_abi::PIE_CHANNEL_EXTERN_EXPORT,
                    },
                    container.names[binding.name as usize].as_bytes(),
                ),
            };
            let desc = PieChannelDesc {
                channel_id,
                shape: PieU32Slice {
                    ptr: decl.shape.dims().as_ptr(),
                    len: decl.shape.dims().len(),
                },
                dtype: decl.dtype.tag(),
                host_role: decl.host_role as u8,
                seeded: u8::from(decl.seeded),
                extern_dir,
                capacity: decl.capacity,
                reader_wait_id: wait.reader_wait_id,
                writer_wait_id: wait.writer_wait_id,
                extern_name: PieBytes {
                    ptr: extern_name.as_ptr(),
                    len: extern_name.len(),
                },
                ..PieChannelDesc::default()
            };
            endpoints.push(driver.register_channel(&desc).unwrap());
            if decl.host_role == HostRole::Reader {
                reader_dense.push(dense);
            }
        }
        let program_id = driver
            .register_program(&PieProgramDesc {
                program_hash: pie_ptir::container_hash(&bytes),
                canonical_bytes: PieBytes {
                    ptr: bytes.as_ptr(),
                    len: bytes.len(),
                },
                ..PieProgramDesc::default()
            })
            .unwrap();
        let seed_descs = seeds
            .iter()
            .map(|seed| PieChannelValueDesc {
                channel_id: seed.channel_id,
                bytes: PieBytes {
                    ptr: seed.bytes.as_ptr(),
                    len: seed.bytes.len(),
                },
            })
            .collect::<Vec<_>>();
        let instance = driver
            .bind_instance(&PieInstanceDesc {
                program_id,
                pacing_wait_id,
                channel_ids: PieU64Slice {
                    ptr: channel_ids.as_ptr(),
                    len: channel_ids.len(),
                },
                seed_values: PieChannelValueDescSlice {
                    ptr: seed_descs.as_ptr(),
                    len: seed_descs.len(),
                },
                ..PieInstanceDesc::default()
            })
            .unwrap();
        TestInstanceBinding {
            instance_id: instance.instance_id,
            mirror_bytes: reader_dense
                .iter()
                .map(|&dense| endpoints[dense].mirror_bytes)
                .sum(),
            word_count: (reader_dense.len() * 4) as u32,
            channel_count: reader_dense.len() as u32,
            endpoints,
            reader_dense,
        }
    }

    fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role,
            seeded,
        }
    }

    fn endpoint_contract(
        driver: &mut DummyDriver,
        channel_id: u64,
        extern_dir: u8,
        extern_name: &[u8],
    ) -> PieChannelEndpointBinding {
        let shape = [1u32];
        driver
            .register_channel(&PieChannelDesc {
                channel_id,
                shape: PieU32Slice {
                    ptr: shape.as_ptr(),
                    len: shape.len(),
                },
                dtype: PIE_CHANNEL_DTYPE_U32,
                host_role: PIE_CHANNEL_HOST_ROLE_NONE,
                seeded: 0,
                extern_dir,
                capacity: 1,
                reader_wait_id: channel_id + 1_000,
                writer_wait_id: channel_id + 2_000,
                extern_name: PieBytes {
                    ptr: extern_name.as_ptr(),
                    len: extern_name.len(),
                },
                ..PieChannelDesc::default()
            })
            .unwrap()
    }

    fn register_test_program(driver: &mut DummyDriver, container: TraceContainer) -> u64 {
        let bytes = container.encode();
        driver
            .register_program(&PieProgramDesc {
                program_hash: pie_ptir::container_hash(&bytes),
                canonical_bytes: PieBytes {
                    ptr: bytes.as_ptr(),
                    len: bytes.len(),
                },
                ..PieProgramDesc::default()
            })
            .unwrap()
    }

    fn bind_existing_channels(
        driver: &mut DummyDriver,
        program_id: u64,
        requested_instance_id: u64,
        channel_ids: &[u64],
    ) -> Result<PieInstanceBinding> {
        driver.bind_instance(&PieInstanceDesc {
            program_id,
            requested_instance_id,
            pacing_wait_id: requested_instance_id + 10_000,
            channel_ids: PieU64Slice {
                ptr: channel_ids.as_ptr(),
                len: channel_ids.len(),
            },
            ..PieInstanceDesc::default()
        })
    }

    fn private_container(channel_count: usize) -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: (0..channel_count)
                .map(|_| chan(Shape::vector(1), DType::U32, HostRole::None, false))
                .collect(),
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
        }
    }

    fn extern_container(dir: ExternDir) -> TraceContainer {
        let ops = match dir {
            ExternDir::Export => vec![
                Op::Const(Literal::U32(1)),
                Op::Broadcast {
                    value: 0,
                    shape: Shape::vector(1),
                },
                Op::ChanPut { chan: 0, value: 1 },
            ],
            ExternDir::Import => vec![Op::ChanTake(0)],
        };
        TraceContainer {
            names: vec!["shared".to_string()],
            externs: vec![ExternDecl {
                name: 0,
                dir,
                chan: 0,
            }],
            channels: vec![chan(Shape::vector(1), DType::U32, HostRole::None, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
    }

    fn token_seed(channel_id: u64) -> OwnedValueDesc {
        OwnedValueDesc {
            channel_id,
            bytes: 1i32.to_le_bytes().to_vec(),
        }
    }

    fn suite_container(vocab: u32) -> Vec<u8> {
        let mut ops = Vec::new();
        let logits2 = expand::next_id(&ops);
        ops.push(Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, vocab),
            dtype: DType::F32,
        });
        let logits = expand::next_id(&ops);
        ops.push(Op::Reshape {
            value: logits2,
            shape: Shape::vector(vocab),
        });
        let argmax = expand::next_id(&ops);
        ops.push(Op::ReduceArgmax(logits));
        let tok = expand::next_id(&ops);
        ops.push(Op::Reshape {
            value: argmax,
            shape: Shape::vector(1),
        });
        let log_p = expand::log_softmax(&mut ops, logits, Shape::vector(vocab));
        let probs = expand::next_id(&ops);
        ops.push(Op::Exp(log_p));
        let ent_terms = expand::next_id(&ops);
        ops.push(Op::Mul(probs, log_p));
        let ent = expand::next_id(&ops);
        ops.push(Op::ReduceSum(ent_terms));
        let neg_ent = expand::next_id(&ops);
        ops.push(Op::Neg(ent));
        let ent_vec = expand::next_id(&ops);
        ops.push(Op::Reshape {
            value: neg_ent,
            shape: Shape::vector(1),
        });
        ops.push(Op::ChanPut {
            chan: 1,
            value: tok,
        });
        ops.push(Op::ChanPut {
            chan: 2,
            value: ent_vec,
        });
        ops.push(Op::ChanPut {
            chan: 3,
            value: probs,
        });
        ops.push(Op::ChanPut {
            chan: 4,
            value: log_p,
        });
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32, HostRole::None, true),
                chan(Shape::vector(1), DType::I32, HostRole::Reader, false),
                chan(Shape::vector(1), DType::F32, HostRole::Reader, false),
                chan(Shape::vector(vocab), DType::F32, HostRole::Reader, false),
                chan(Shape::vector(vocab), DType::F32, HostRole::Reader, false),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|w| w.to_le_bytes()).collect(),
                    },
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
        .encode()
    }

    fn entropy_container(vocab: u32) -> Vec<u8> {
        let mut ops = Vec::new();
        let logits2 = expand::next_id(&ops);
        ops.push(Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, vocab),
            dtype: DType::F32,
        });
        let logits = expand::next_id(&ops);
        ops.push(Op::Reshape {
            value: logits2,
            shape: Shape::vector(vocab),
        });
        let log_p = expand::log_softmax(&mut ops, logits, Shape::vector(vocab));
        let probs = expand::next_id(&ops);
        ops.push(Op::Exp(log_p));
        let ent_terms = expand::next_id(&ops);
        ops.push(Op::Mul(probs, log_p));
        let ent = expand::next_id(&ops);
        ops.push(Op::ReduceSum(ent_terms));
        let neg_ent = expand::next_id(&ops);
        ops.push(Op::Neg(ent));
        let ent_vec = expand::next_id(&ops);
        ops.push(Op::Reshape {
            value: neg_ent,
            shape: Shape::vector(1),
        });
        ops.push(Op::ChanPut {
            chan: 1,
            value: ent_vec,
        });
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32, HostRole::None, true),
                chan(Shape::vector(1), DType::F32, HostRole::Reader, false),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|w| w.to_le_bytes()).collect(),
                    },
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
        .encode()
    }

    fn vector_echo_container() -> Vec<u8> {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(3), DType::I32, HostRole::None, true),
                chan(Shape::vector(3), DType::I32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::ChanPut { chan: 0, value: 0 },
                    Op::ChanPut { chan: 1, value: 0 },
                ],
            }],
        }
        .encode()
    }

    fn kernel_fault_container() -> Vec<u8> {
        TraceContainer {
            names: vec!["boom".to_string()],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32, HostRole::None, true),
                chan(Shape::vector(1), DType::F32, HostRole::Reader, false),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|w| w.to_le_bytes()).collect(),
                    },
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::KernelCall {
                        name: 0,
                        args: vec![],
                        shape: Shape::vector(1),
                        dtype: DType::F32,
                    },
                    Op::ChanPut { chan: 1, value: 0 },
                ],
            }],
        }
        .encode()
    }

    fn counter_container() -> Vec<u8> {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::None, true),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Const(Literal::U32(1)),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 0, value: 2 },
                    Op::ChanPut { chan: 1, value: 2 },
                ],
            }],
        }
        .encode()
    }

    fn mixed_role_publish_container() -> Vec<u8> {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::Writer, false),
                chan(Shape::vector(1), DType::U32, HostRole::None, false),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
                chan(Shape::vector(1), DType::U32, HostRole::None, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Const(Literal::U32(1)),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 1, value: 2 },
                    Op::ChanPut { chan: 2, value: 0 },
                    Op::ChanTake(1),
                    Op::ChanPut { chan: 3, value: 3 },
                    Op::ChanPut { chan: 4, value: 3 },
                ],
            }],
        }
        .encode()
    }

    fn mixed_role_fault_container() -> Vec<u8> {
        TraceContainer {
            names: vec!["boom".to_string()],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::Writer, false),
                chan(Shape::vector(1), DType::U32, HostRole::None, false),
                chan(Shape::vector(1), DType::F32, HostRole::Reader, false),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::KernelCall {
                        name: 0,
                        args: vec![],
                        shape: Shape::vector(1),
                        dtype: DType::F32,
                    },
                    Op::ChanPut { chan: 2, value: 0 },
                ],
            }],
        }
        .encode()
    }

    fn read_binding(binding: &TestInstanceBinding, index: usize) -> TestChannelBinding {
        let endpoint_index = binding.reader_dense[index];
        let endpoint = binding.endpoints[endpoint_index];
        let mirror_offset = binding.reader_dense[..index]
            .iter()
            .map(|&dense| binding.endpoints[dense].mirror_bytes)
            .sum();
        TestChannelBinding {
            channel_id: endpoint.channel_id,
            cell_bytes: endpoint.cell_bytes,
            mirror_offset,
            head_word_index: (index * 4) as u32,
            tail_word_index: (index * 4 + 1) as u32,
            poison_word_index: (index * 4 + 2) as u32,
            endpoint_index,
        }
    }

    fn read_bindings(binding: &TestInstanceBinding) -> Vec<TestChannelBinding> {
        (0..binding.channel_count as usize)
            .map(|index| read_binding(binding, index))
            .collect()
    }

    fn read_binding_by_id(binding: &TestInstanceBinding, channel_id: u64) -> TestChannelBinding {
        read_bindings(binding)
            .into_iter()
            .find(|channel| channel.channel_id == channel_id)
            .unwrap_or_else(|| panic!("missing exported binding for channel {channel_id}"))
    }

    fn read_words(binding: &TestInstanceBinding) -> Vec<u64> {
        let mut out = Vec::with_capacity(binding.word_count as usize);
        for &dense in &binding.reader_dense {
            let endpoint = binding.endpoints[dense];
            let words =
                unsafe { std::slice::from_raw_parts(endpoint.word_base as *const AtomicU64, 4) };
            out.extend(words.iter().map(|word| word.load(Ordering::Acquire)));
        }
        out
    }

    /// Direct host put (ABI v2): write the wire bytes into the writer
    /// endpoint's pinned ring cell and release-publish the tail word.
    fn ring_put(binding: &TestInstanceBinding, dense: usize, wire: &[u8]) {
        let endpoint = binding.endpoints[dense];
        let words =
            unsafe { std::slice::from_raw_parts(endpoint.word_base as *const AtomicU64, 4) };
        let tail = words[endpoint.tail_word_index as usize].load(Ordering::Acquire);
        let cap1 = u64::from(endpoint.capacity) + 1;
        let offset = (tail % cap1) as usize * endpoint.cell_bytes as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                wire.as_ptr(),
                (endpoint.mirror_base as *mut u8).add(offset),
                wire.len(),
            );
        }
        words[endpoint.tail_word_index as usize].store(tail + 1, Ordering::Release);
    }

    fn read_cell(
        binding: &TestInstanceBinding,
        channel: TestChannelBinding,
        slot: usize,
    ) -> Vec<u8> {
        let endpoint = binding.endpoints[channel.endpoint_index];
        let offset = slot * channel.cell_bytes as usize;
        unsafe {
            std::slice::from_raw_parts(
                (endpoint.mirror_base as *const u8).add(offset),
                channel.cell_bytes as usize,
            )
        }
        .to_vec()
    }

    #[test]
    fn endpoint_attachments_are_exclusive_atomic_and_close_ordered() {
        let mut driver = DummyDriver::default();

        endpoint_contract(&mut driver, 1, PIE_CHANNEL_EXTERN_NONE, &[]);
        let private_program = register_test_program(&mut driver, private_container(1));
        let private = bind_existing_channels(&mut driver, private_program, 101, &[1]).unwrap();
        assert!(bind_existing_channels(&mut driver, private_program, 102, &[1]).is_err());
        assert!(driver.close_channel(1).is_err());
        driver.close_instance(private.instance_id).unwrap();
        driver.close_channel(1).unwrap();

        endpoint_contract(&mut driver, 2, PIE_CHANNEL_EXTERN_EXPORT, b"shared");
        let export_program =
            register_test_program(&mut driver, extern_container(ExternDir::Export));
        let import_program =
            register_test_program(&mut driver, extern_container(ExternDir::Import));
        let export = bind_existing_channels(&mut driver, export_program, 201, &[2]).unwrap();
        let import = bind_existing_channels(&mut driver, import_program, 202, &[2]).unwrap();
        assert!(bind_existing_channels(&mut driver, export_program, 203, &[2]).is_err());
        assert!(driver.close_channel(2).is_err());
        driver.close_instance(export.instance_id).unwrap();
        driver.close_instance(import.instance_id).unwrap();
        driver.close_channel(2).unwrap();

        endpoint_contract(&mut driver, 3, PIE_CHANNEL_EXTERN_NONE, &[]);
        endpoint_contract(&mut driver, 4, PIE_CHANNEL_EXTERN_NONE, &[]);
        let occupied = bind_existing_channels(&mut driver, private_program, 301, &[4]).unwrap();
        let two_channel_program = register_test_program(&mut driver, private_container(2));
        assert!(bind_existing_channels(&mut driver, two_channel_program, 302, &[3, 4]).is_err());
        let free = bind_existing_channels(&mut driver, private_program, 303, &[3]).unwrap();
        driver.close_instance(free.instance_id).unwrap();
        driver.close_instance(occupied.instance_id).unwrap();
        driver.close_channel(3).unwrap();
        driver.close_channel(4).unwrap();
    }

    #[test]
    fn launch_publishes_outputs_terminals_and_one_batch_callback() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let vocab = 8;
        let channels = [10u64, 11, 12, 13, 14];
        let waits = [
            PieChannelWait {
                reader_wait_id: 100,
                writer_wait_id: 200,
            },
            PieChannelWait {
                reader_wait_id: 101,
                writer_wait_id: 201,
            },
            PieChannelWait {
                reader_wait_id: 102,
                writer_wait_id: 202,
            },
            PieChannelWait {
                reader_wait_id: 103,
                writer_wait_id: 203,
            },
            PieChannelWait {
                reader_wait_id: 104,
                writer_wait_id: 204,
            },
        ];
        let binding = bind_program(
            &mut driver,
            suite_container(vocab),
            &channels,
            &waits,
            &[token_seed(10)],
            301,
        );
        for &dense in &binding.reader_dense {
            let endpoint = binding.endpoints[dense];
            unsafe {
                std::slice::from_raw_parts_mut(
                    endpoint.mirror_base as *mut u8,
                    endpoint.mirror_bytes as usize,
                )
            }
            .fill(0xA5);
        }
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 401,
                    target_epoch: 7,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        callbacks.wait_for_notification((401, 7));
        let notices = callbacks.notifications_for(401);
        assert_eq!(notices, vec![(401, 7)]);
        for reader_wait_id in [101, 102, 103, 104] {
            assert_eq!(
                callbacks.notifications_for(reader_wait_id),
                vec![(reader_wait_id, 1)]
            );
        }
        assert_eq!(terminal_cells[0].outcome, PIE_TERMINAL_OUTCOME_SUCCESS);

        let token = read_binding_by_id(&binding, 11);
        let entropy = read_binding_by_id(&binding, 12);
        let dist = read_binding_by_id(&binding, 13);
        let logprobs = read_binding_by_id(&binding, 14);
        let words = read_words(&binding);
        for channel in [token, entropy, dist, logprobs] {
            assert_eq!(words[channel.tail_word_index as usize], 1);
            assert_eq!(words[channel.poison_word_index as usize], 0);
            assert_eq!(
                read_cell(&binding, channel, 1),
                vec![0xA5; channel.cell_bytes as usize],
                "unused ring slot should retain canary bytes"
            );
        }

        let logits = match deterministic_logits(
            ValueType::new(Shape::matrix(1, vocab), DType::F32),
            1 ^ binding.instance_id ^ pie_ptir::container_hash(&suite_container(vocab)),
            vocab,
        ) {
            Value::F32(values) => values,
            _ => unreachable!(),
        };
        let favored = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i32;
        let token_bytes = read_cell(&binding, token, 0);
        assert_eq!(i32::from_le_bytes(token_bytes.try_into().unwrap()), favored);
        let entropy_bytes = read_cell(&binding, entropy, 0);
        let entropy_value = f32::from_le_bytes(entropy_bytes.try_into().unwrap());
        assert!(entropy_value.is_finite() && entropy_value > 0.0);
        let dist_vals = read_cell(&binding, dist, 0)
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let logprob_vals = read_cell(&binding, logprobs, 0)
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(dist_vals.len(), vocab as usize);
        assert_eq!(logprob_vals.len(), vocab as usize);
        let best = dist_vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i32;
        assert_eq!(best, favored);
    }

    #[test]
    fn lone_entropy_output_publishes_scalar() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let channels = [20u64, 21u64];
        let waits = [
            PieChannelWait {
                reader_wait_id: 120,
                writer_wait_id: 220,
            },
            PieChannelWait {
                reader_wait_id: 121,
                writer_wait_id: 221,
            },
        ];
        let binding = bind_program(
            &mut driver,
            entropy_container(8),
            &channels,
            &waits,
            &[token_seed(20)],
            321,
        );
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 421,
                    target_epoch: 3,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        callbacks.wait_for_notification((421, 3));
        assert_eq!(callbacks.notifications_for(421), vec![(421, 3)]);
        assert_eq!(terminal_cells[0].outcome, PIE_TERMINAL_OUTCOME_SUCCESS);
        let channel = read_binding_by_id(&binding, 21);
        let bytes = read_cell(&binding, channel, 0);
        let entropy = f32::from_le_bytes(bytes.try_into().unwrap());
        assert!(entropy.is_finite() && entropy > 0.0);
    }

    #[test]
    fn batched_launches_keep_instances_distinct_and_support_empty_prefix_vectors() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let waits_a = [
            PieChannelWait {
                reader_wait_id: 131,
                writer_wait_id: 231,
            },
            PieChannelWait {
                reader_wait_id: 132,
                writer_wait_id: 232,
            },
        ];
        let waits_b = [
            PieChannelWait {
                reader_wait_id: 141,
                writer_wait_id: 241,
            },
            PieChannelWait {
                reader_wait_id: 142,
                writer_wait_id: 242,
            },
        ];
        let seed_a = [OwnedValueDesc {
            channel_id: 31,
            bytes: [-1i32, -1, -1]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }];
        let seed_b = [OwnedValueDesc {
            channel_id: 41,
            bytes: [4i32, 5, -1]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }];
        let binding_a = bind_program(
            &mut driver,
            vector_echo_container(),
            &[31, 32],
            &waits_a,
            &seed_a,
            331,
        );
        let binding_b = bind_program(
            &mut driver,
            vector_echo_container(),
            &[41, 42],
            &waits_b,
            &seed_b,
            341,
        );
        let instance_ids = [binding_a.instance_id, binding_b.instance_id];
        let mut terminal_cells = [pending_terminal_cell(), pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 431,
                    target_epoch: 5,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        callbacks.wait_for_notification((431, 5));
        assert_eq!(callbacks.notifications_for(431), vec![(431, 5)]);
        assert!(
            terminal_cells
                .iter()
                .all(|cell| cell.outcome == PIE_TERMINAL_OUTCOME_SUCCESS)
        );
        let chan_a = read_binding_by_id(&binding_a, 32);
        let chan_b = read_binding_by_id(&binding_b, 42);
        let vals_a = read_cell(&binding_a, chan_a, 0)
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let vals_b = read_cell(&binding_b, chan_b, 0)
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(vals_a, vec![-1, -1, -1]);
        assert_eq!(vals_b, vec![4, 5, -1]);
    }

    #[test]
    fn mixed_roles_export_only_readers_in_reader_rank_order_with_compact_extents() {
        let (mut driver, _callbacks) = driver_with_callbacks(0);
        let channel_ids = [90u64, 91, 92, 93, 94];
        let waits = [
            PieChannelWait {
                reader_wait_id: 190,
                writer_wait_id: 290,
            },
            PieChannelWait {
                reader_wait_id: 191,
                writer_wait_id: 291,
            },
            PieChannelWait {
                reader_wait_id: 192,
                writer_wait_id: 292,
            },
            PieChannelWait {
                reader_wait_id: 193,
                writer_wait_id: 293,
            },
            PieChannelWait {
                reader_wait_id: 194,
                writer_wait_id: 294,
            },
        ];
        let binding = bind_program(
            &mut driver,
            mixed_role_publish_container(),
            &channel_ids,
            &waits,
            &[],
            391,
        );
        let bindings = read_bindings(&binding);
        assert_eq!(
            binding.channel_count, 2,
            "only reader channels are exported"
        );
        assert_eq!(
            binding.word_count, 8,
            "only reader channels contribute words"
        );
        assert_eq!(
            binding.mirror_bytes,
            2 * 2 * std::mem::size_of::<u32>() as u64
        );
        assert_eq!(
            bindings
                .iter()
                .map(|channel| channel.channel_id)
                .collect::<Vec<_>>(),
            vec![92, 93],
            "reader bindings stay in reader-rank order"
        );
        assert_eq!(bindings[0].mirror_offset, 0);
        assert_eq!(bindings[0].head_word_index, 0);
        assert_eq!(bindings[0].tail_word_index, 1);
        assert_eq!(bindings[0].poison_word_index, 2);
        assert_eq!(bindings[1].mirror_offset, 8);
        assert_eq!(bindings[1].head_word_index, 4);
        assert_eq!(bindings[1].tail_word_index, 5);
        assert_eq!(bindings[1].poison_word_index, 6);
    }

    #[test]
    fn mixed_roles_publish_only_exported_readers_and_close_still_works() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let channel_ids = [100u64, 101, 102, 103, 104];
        let waits = [
            PieChannelWait {
                reader_wait_id: 200,
                writer_wait_id: 300,
            },
            PieChannelWait {
                reader_wait_id: 201,
                writer_wait_id: 301,
            },
            PieChannelWait {
                reader_wait_id: 202,
                writer_wait_id: 302,
            },
            PieChannelWait {
                reader_wait_id: 203,
                writer_wait_id: 303,
            },
            PieChannelWait {
                reader_wait_id: 204,
                writer_wait_id: 304,
            },
        ];
        let binding = bind_program(
            &mut driver,
            mixed_role_publish_container(),
            &channel_ids,
            &waits,
            &[],
            401,
        );
        // ABI v2: the put is a direct ring write, pulled by the launch.
        ring_put(&binding, 0, &7u32.to_le_bytes());
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 501,
                    target_epoch: 11,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        callbacks.wait_for_notification((501, 11));
        let notices = callbacks.notifications_for(501);
        assert_eq!(notices, vec![(501, 11)]);
        for wait_id in [300, 202, 203] {
            assert_eq!(callbacks.notifications_for(wait_id), vec![(wait_id, 1)]);
        }
        assert_eq!(terminal_cells[0].outcome, PIE_TERMINAL_OUTCOME_SUCCESS);
        let writer_words = unsafe {
            std::slice::from_raw_parts(binding.endpoints[0].word_base as *const AtomicU64, 4)
        };
        assert_eq!(
            writer_words[binding.endpoints[0].head_word_index as usize].load(Ordering::Acquire),
            1,
            "a committed host Writer value must release capacity"
        );

        let reader0 = read_binding_by_id(&binding, 102);
        let reader1 = read_binding_by_id(&binding, 103);
        let words = read_words(&binding);
        assert_eq!(words[reader0.tail_word_index as usize], 1);
        assert_eq!(words[reader1.tail_word_index as usize], 1);
        assert_eq!(words[reader0.poison_word_index as usize], 0);
        assert_eq!(words[reader1.poison_word_index as usize], 0);
        assert_eq!(
            u32::from_le_bytes(read_cell(&binding, reader0, 0).try_into().unwrap()),
            7
        );
        assert_eq!(
            u32::from_le_bytes(read_cell(&binding, reader1, 0).try_into().unwrap()),
            8
        );
        driver.close_instance(binding.instance_id).unwrap();
        let err = driver.close_instance(binding.instance_id).unwrap_err();
        assert!(err.to_string().contains("unknown instance"));
    }

    #[test]
    fn mixed_roles_kernel_fault_poisons_host_endpoints() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let channel_ids = [110u64, 111, 112, 113];
        let waits = [
            PieChannelWait {
                reader_wait_id: 210,
                writer_wait_id: 310,
            },
            PieChannelWait {
                reader_wait_id: 211,
                writer_wait_id: 311,
            },
            PieChannelWait {
                reader_wait_id: 212,
                writer_wait_id: 312,
            },
            PieChannelWait {
                reader_wait_id: 213,
                writer_wait_id: 313,
            },
        ];
        let binding = bind_program(
            &mut driver,
            mixed_role_fault_container(),
            &channel_ids,
            &waits,
            &[],
            411,
        );
        // ABI v2: the put is a direct ring write, pulled by the launch.
        ring_put(&binding, 0, &9u32.to_le_bytes());
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 511,
                    target_epoch: 13,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        callbacks.wait_for_notification((511, 13));
        let notices = callbacks.notifications_for(511);
        assert_eq!(notices, vec![(511, 13)]);
        // The writer ring already published tail 1, so its poison epoch is 2;
        // the untouched readers poison at 1.
        assert_eq!(callbacks.notifications_for(310), vec![(310, 2)]);
        for wait_id in [212, 213] {
            assert_eq!(callbacks.notifications_for(wait_id), vec![(wait_id, 1)]);
        }
        assert_eq!(terminal_cells[0].outcome, PIE_TERMINAL_OUTCOME_FAILED);
        let reader0 = read_binding_by_id(&binding, 112);
        let reader1 = read_binding_by_id(&binding, 113);
        let words = read_words(&binding);
        assert_eq!(words[reader0.tail_word_index as usize], 0);
        assert_eq!(words[reader1.tail_word_index as usize], 0);
        assert_eq!(words[reader0.poison_word_index as usize], 1);
        assert_eq!(words[reader1.poison_word_index as usize], 1);
        driver.close_instance(binding.instance_id).unwrap();
    }

    #[test]
    fn async_kernel_fault_poisons_channels_and_still_notifies() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let waits = [
            PieChannelWait {
                reader_wait_id: 150,
                writer_wait_id: 250,
            },
            PieChannelWait {
                reader_wait_id: 151,
                writer_wait_id: 251,
            },
        ];
        let binding = bind_program(
            &mut driver,
            kernel_fault_container(),
            &[50, 51],
            &waits,
            &[token_seed(50)],
            351,
        );
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 451,
                    target_epoch: 9,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        callbacks.wait_for_notification((451, 9));
        let channel = read_binding_by_id(&binding, 51);
        let words = read_words(&binding);
        assert_eq!(words[channel.tail_word_index as usize], 0);
        assert_eq!(words[channel.poison_word_index as usize], 1);
        let notices = callbacks.notifications_for(451);
        assert_eq!(notices, vec![(451, 9)]);
        assert_eq!(terminal_cells[0].outcome, PIE_TERMINAL_OUTCOME_FAILED);
    }

    #[test]
    fn invalid_descriptors_fail_synchronously() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        assert!(driver.register_program(&PieProgramDesc::default()).is_err());
        let err = driver
            .bind_instance(&PieInstanceDesc {
                program_id: 1,
                pacing_wait_id: 0,
                channel_ids: PieU64Slice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                ..PieInstanceDesc::default()
            })
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        let bytes = counter_container();
        let waits = [
            PieChannelWait {
                reader_wait_id: 161,
                writer_wait_id: 261,
            },
            PieChannelWait {
                reader_wait_id: 162,
                writer_wait_id: 262,
            },
        ];
        let seed = [OwnedValueDesc {
            channel_id: 61,
            bytes: 0u32.to_le_bytes().to_vec(),
        }];
        let binding = bind_program(&mut driver, bytes, &[61, 62], &waits, &seed, 361);
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        // token_ids without matching position_ids is a malformed descriptor.
        let tokens = [1u32];
        let err = driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    token_ids: PieU32Slice {
                        ptr: tokens.as_ptr(),
                        len: tokens.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 461,
                    target_epoch: 1,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        assert!(
            callbacks.notifications().is_empty(),
            "sync validation must not notify"
        );
    }

    #[test]
    fn multi_row_rs_shape_uses_resolved_rows_not_instance_count() {
        let tokens = [10u32, 11];
        let qo_indptr = [0u32, 1, 2];
        let rs_slot_ids = [7u32, 9];
        let rs_slot_flags = [pie_driver_abi::PIE_RS_FLAG_RESET, 0];
        let desc = PieLaunchDesc {
            token_ids: PieU32Slice {
                ptr: tokens.as_ptr(),
                len: tokens.len(),
            },
            qo_indptr: PieU32Slice {
                ptr: qo_indptr.as_ptr(),
                len: qo_indptr.len(),
            },
            rs_slot_ids: PieU32Slice {
                ptr: rs_slot_ids.as_ptr(),
                len: rs_slot_ids.len(),
            },
            rs_slot_flags: pie_driver_abi::PieU8Slice {
                ptr: rs_slot_flags.as_ptr(),
                len: rs_slot_flags.len(),
            },
            ..PieLaunchDesc::default()
        };
        validate_launch_shape(&desc, 1).unwrap();

        let one_slot = [7u32];
        let one_flag = [0u8];
        let wrong_rows = PieLaunchDesc {
            rs_slot_ids: PieU32Slice {
                ptr: one_slot.as_ptr(),
                len: one_slot.len(),
            },
            rs_slot_flags: pie_driver_abi::PieU8Slice {
                ptr: one_flag.as_ptr(),
                len: one_flag.len(),
            },
            ..desc
        };
        assert!(
            validate_launch_shape(&wrong_rows, 1)
                .unwrap_err()
                .to_string()
                .contains("resolved qo rows")
        );
    }

    #[test]
    fn later_invalid_batch_member_rejects_without_mutating_earlier_members() {
        let (mut driver, callbacks) = driver_with_callbacks(0);
        let waits = [
            PieChannelWait {
                reader_wait_id: 163,
                writer_wait_id: 263,
            },
            PieChannelWait {
                reader_wait_id: 164,
                writer_wait_id: 264,
            },
        ];
        let seed = [OwnedValueDesc {
            channel_id: 63,
            bytes: 0u32.to_le_bytes().to_vec(),
        }];
        let binding = bind_program(
            &mut driver,
            counter_container(),
            &[63, 64],
            &waits,
            &seed,
            362,
        );
        let instance_ids = [binding.instance_id, u64::MAX];
        let mut terminal_cells = [pending_terminal_cell(), pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);

        let err = driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 462,
                    target_epoch: 1,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap_err();

        assert!(err.to_string().contains("unknown instance"));
        assert_eq!(
            terminal_cells.map(|cell| cell.outcome),
            [PIE_TERMINAL_OUTCOME_PENDING; 2]
        );
        assert_eq!(
            read_words(&binding)[0],
            0,
            "earlier member must not execute"
        );
        assert!(callbacks.notifications().is_empty());
    }

    #[test]
    fn close_after_terminal_publication_does_not_wait_for_callback() {
        let (mut driver, callbacks) = driver_with_callbacks(50);
        let waits = [
            PieChannelWait {
                reader_wait_id: 171,
                writer_wait_id: 271,
            },
            PieChannelWait {
                reader_wait_id: 172,
                writer_wait_id: 272,
            },
        ];
        let seed = [OwnedValueDesc {
            channel_id: 71,
            bytes: 0u32.to_le_bytes().to_vec(),
        }];
        let binding = bind_program(
            &mut driver,
            counter_container(),
            &[71, 72],
            &waits,
            &seed,
            371,
        );
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 471,
                    target_epoch: 1,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        let started = Instant::now();
        driver.close_instance(binding.instance_id).unwrap();
        assert!(started.elapsed() < Duration::from_millis(40));
        callbacks.wait_for_notification((471, 1));
        assert_eq!(callbacks.notifications_for(471), vec![(471, 1)]);
        let err = driver.close_instance(binding.instance_id).unwrap_err();
        assert!(err.to_string().contains("unknown instance"));
    }

    #[test]
    fn drop_joins_callback_work_before_returning() {
        let (mut driver, callbacks) = driver_with_callbacks(25);
        let waits = [
            PieChannelWait {
                reader_wait_id: 181,
                writer_wait_id: 281,
            },
            PieChannelWait {
                reader_wait_id: 182,
                writer_wait_id: 282,
            },
        ];
        let seed = [OwnedValueDesc {
            channel_id: 81,
            bytes: 0u32.to_le_bytes().to_vec(),
        }];
        let binding = bind_program(
            &mut driver,
            counter_container(),
            &[81, 82],
            &waits,
            &seed,
            381,
        );
        let instance_ids = [binding.instance_id];
        let mut terminal_cells = [pending_terminal_cell()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    terminal_cells: PieTerminalCellPtrSlice {
                        ptr: terminal_ptrs.as_ptr(),
                        len: terminal_ptrs.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 481,
                    target_epoch: 1,
                    terminal_cell: std::ptr::null_mut(),
                },
            )
            .unwrap();
        drop(driver);
        let at_drop = callbacks.count.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(callbacks.count.load(Ordering::SeqCst), at_drop);
        assert_eq!(callbacks.notifications_for(481), vec![(481, 1)]);
    }
}
