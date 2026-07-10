use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};

use anyhow::{Result, anyhow, bail, ensure};
use pie_driver_abi::{
    DriverCapabilities, PieBytes, PieChannelBinding, PieChannelBindingSlice,
    PieChannelValueDescSlice, PieChannelWait, PieChannelWaitSlice, PieCompletion,
    PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc, PieLaunchDesc, PiePoolResizeDesc,
    PieProgramDesc, PieRuntimeCallbacks, PieStateCopyDesc, PieU32Slice, PieU64Slice,
    validate_instance_desc, validate_kv_copy_desc, validate_launch_desc, validate_pool_resize_desc,
    validate_program_desc, validate_state_copy_desc,
};
use pie_ptir::container::{self, HostRole};
use pie_ptir::interp::{
    HostError, Instance as InterpInstance, NoKernels, PassInputs, StepError, Value,
};
use pie_ptir::op::{IntrinsicId, Op};
use pie_ptir::registry::{KernelInfo, ModelProfile};
use pie_ptir::types::{DType, ValueType};
use pie_ptir::validate::BoundTrace;

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
    pub callback_delay_ms: u64,
    pub operation_log: Option<Arc<Mutex<Vec<String>>>>,
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
            callback_delay_ms: 0,
            operation_log: None,
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
    capacity: u32,
    cell_bytes: usize,
    mirror_offset: usize,
    head_word_index: usize,
    tail_word_index: usize,
    poison_word_index: usize,
    waits: PieChannelWait,
}

#[derive(Debug)]
struct BoundInstanceState {
    program: Arc<DummyProgram>,
    instance_id: u64,
    pacing_wait_id: u64,
    inner: Mutex<BoundInstanceInner>,
    cv: Condvar,
}

#[derive(Debug)]
struct BoundInstanceInner {
    interp: InterpInstance,
    channels: Vec<DummyChannel>,
    channel_index_by_id: HashMap<u64, usize>,
    bindings: Box<[PieChannelBinding]>,
    mirror: Box<[u8]>,
    words: Box<[AtomicU64]>,
    next_pacing_epoch: u64,
    in_flight: usize,
    closed: bool,
}

#[derive(Default)]
struct DummyState {
    programs: HashMap<u64, Arc<DummyProgram>>,
    instances: HashMap<u64, Arc<BoundInstanceState>>,
}

#[derive(Clone, Debug)]
struct OwnedValueDesc {
    channel_id: u64,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct PreparedHostPut {
    dense_channel: usize,
    value: Value,
}

#[derive(Debug)]
struct LaunchInstanceWork {
    instance: Arc<BoundInstanceState>,
    pacing_epoch: u64,
    host_puts: Vec<PreparedHostPut>,
}

#[derive(Debug)]
enum WorkItem {
    Launch {
        instances: Vec<LaunchInstanceWork>,
        completion: PieCompletion,
    },
    CopyKv {
        completion: PieCompletion,
    },
    CopyState {
        completion: PieCompletion,
    },
    ResizePool {
        completion: PieCompletion,
    },
}

#[derive(Clone)]
struct SendableRuntimeCallbacks {
    ctx: usize,
    notify: pie_driver_abi::PieRuntimeNotifyFn,
    operation_log: Option<Arc<Mutex<Vec<String>>>>,
}

unsafe impl Send for SendableRuntimeCallbacks {}
unsafe impl Sync for SendableRuntimeCallbacks {}

pub struct DummyDriver {
    capabilities: DriverCapabilities,
    state: Arc<Mutex<DummyState>>,
    next_program_id: AtomicU64,
    next_instance_id: AtomicU64,
    operation_log: Option<Arc<Mutex<Vec<String>>>>,
    work_tx: Option<mpsc::Sender<WorkItem>>,
    worker: Option<std::thread::JoinHandle<()>>,
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
        let (work_tx, work_rx) = mpsc::channel();
        let callback_delay_ms = options.callback_delay_ms;
        let operation_log = options.operation_log.clone();
        let runtime = SendableRuntimeCallbacks {
            ctx: runtime.ctx as usize,
            notify: runtime.notify,
            operation_log: operation_log.clone(),
        };
        let worker = std::thread::Builder::new()
            .name("pie-dummy-driver".to_string())
            .spawn(move || worker_loop(work_rx, runtime, callback_delay_ms))
            .expect("spawn dummy driver worker");
        Self {
            capabilities: DriverCapabilities {
                abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
                total_pages: options.total_pages,
                kv_page_size: options.kv_page_size,
                swap_pool_size: options.swap_pool_size,
                rs_cache_required: false,
                rs_cache_slots: 0,
                rs_cache_slot_bytes: 0,
                max_forward_tokens: options.max_forward_tokens,
                max_forward_requests: options.max_forward_requests,
                max_page_refs: options.max_page_refs,
                arch_name: options.arch_name,
                vocab_size: options.vocab_size,
                max_model_len: options.max_model_len,
                activation_dtype: options.activation_dtype,
                snapshot_dir: options.snapshot_dir,
                storage_backend: String::new(),
                max_tile_bytes: 0,
                preferred_alignment: 0,
                mxfp4_moe_policy: String::new(),
                native_mxfp4_moe: false,
            },
            state,
            next_program_id: AtomicU64::new(1),
            next_instance_id: AtomicU64::new(1),
            operation_log,
            work_tx: Some(work_tx),
            worker: Some(worker),
        }
    }

    fn record_op(&self, name: &str) {
        if let Some(log) = &self.operation_log {
            log.lock().unwrap().push(name.to_string());
        }
    }

    pub fn capabilities(&self) -> &DriverCapabilities {
        &self.capabilities
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
        let bound = pie_ptir::validate::bind(container, self.model_profile())
            .map_err(|err| anyhow!("program bind failed: {err}"))?;
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

    pub fn bind_instance(&mut self, desc: &PieInstanceDesc) -> Result<PieInstanceBinding> {
        unsafe { validate_instance_desc(desc) }.map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("bind_instance");
        ensure!(
            desc.pacing_wait_id != 0,
            "bind requires a nonzero pacing wait id"
        );
        let channel_ids = copy_u64_slice(desc.channel_ids, "instance.channel_ids")?;
        let channel_waits = copy_channel_waits(desc.channel_waits, "instance.channel_waits")?;
        ensure!(
            channel_waits.len() == channel_ids.len(),
            "bind requires one channel wait pair per channel id"
        );
        ensure!(
            channel_ids.iter().copied().collect::<HashSet<_>>().len() == channel_ids.len(),
            "bind channel ids must be unique"
        );
        for (idx, wait) in channel_waits.iter().enumerate() {
            ensure!(
                wait.reader_wait_id != 0 && wait.writer_wait_id != 0,
                "channel wait {idx} must use nonzero wait ids"
            );
        }
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
            ensure!(
                !state.instances.contains_key(&desc.requested_instance_id),
                "instance {} already bound",
                desc.requested_instance_id
            );
            desc.requested_instance_id
        };

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
        let interp = InterpInstance::new(&program.bound, &seeds)
            .map_err(|err| anyhow!("instance bind failed: {err:?}"))?;
        let (channels, bindings, mirror, words) =
            build_channel_layout(&program, &channel_ids, &channel_waits)?;
        let channel_index_by_id = channels
            .iter()
            .enumerate()
            .map(|(idx, channel)| (channel.global_id, idx))
            .collect();
        let instance = Arc::new(BoundInstanceState {
            program,
            instance_id,
            pacing_wait_id: desc.pacing_wait_id,
            inner: Mutex::new(BoundInstanceInner {
                interp,
                channels,
                channel_index_by_id,
                bindings,
                mirror,
                words,
                next_pacing_epoch: 1,
                in_flight: 0,
                closed: false,
            }),
            cv: Condvar::new(),
        });
        let binding = {
            let inner = instance.inner.lock().unwrap();
            PieInstanceBinding {
                instance_id,
                frame_base: 0,
                mirror_base: inner.mirror.as_ptr() as u64,
                word_base: inner.words.as_ptr() as u64,
                channel_count: inner.bindings.len() as u32,
                word_count: inner.words.len() as u32,
                frame_bytes: 0,
                mirror_bytes: inner.mirror.len() as u64,
                word_bytes: (inner.words.len() * std::mem::size_of::<u64>()) as u64,
                channels: PieChannelBindingSlice {
                    ptr: inner.bindings.as_ptr(),
                    len: inner.bindings.len(),
                },
            }
        };
        state.instances.insert(instance_id, instance);
        Ok(binding)
    }

    pub fn launch(&mut self, desc: &PieLaunchDesc, completion: PieCompletion) -> Result<()> {
        unsafe { validate_launch_desc(desc) }.map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("launch");
        let instance_ids = copy_u64_slice(desc.instance_ids, "launch.instance_ids")?;
        ensure!(
            !instance_ids.is_empty(),
            "launch requires at least one bound instance"
        );
        ensure_completion(completion)?;
        validate_launch_shape(desc, instance_ids.len())?;
        let host_put_values =
            copy_value_descs(desc.ptir_host_put_values, "launch.ptir_host_put_values")?;
        let host_put_indptr = copy_u32_slice(desc.host_put_indptr, "launch.host_put_indptr")?;
        validate_indptr(
            &host_put_indptr,
            host_put_values.len(),
            instance_ids.len(),
            "launch.host_put_indptr",
        )?;

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

        let mut prepared = Vec::with_capacity(instances.len());
        for (slot, instance) in instances.iter().enumerate() {
            let range = host_put_indptr[slot] as usize..host_put_indptr[slot + 1] as usize;
            let mut inner = instance.inner.lock().unwrap();
            ensure!(!inner.closed, "instance {} is closed", instance.instance_id);
            let mut puts = Vec::with_capacity(range.len());
            for value_desc in &host_put_values[range] {
                let dense = *inner
                    .channel_index_by_id
                    .get(&value_desc.channel_id)
                    .ok_or_else(|| {
                        anyhow!(
                            "instance {} has no channel {}",
                            instance.instance_id,
                            value_desc.channel_id
                        )
                    })?;
                let channel = &inner.channels[dense];
                ensure!(
                    channel.host_role == HostRole::Writer,
                    "channel {} is not a host-writer channel",
                    value_desc.channel_id
                );
                let value = decode_wire_value(&value_desc.bytes, channel.ty)
                    .map_err(|err| anyhow!("host put channel {}: {err}", value_desc.channel_id))?;
                puts.push(PreparedHostPut {
                    dense_channel: dense,
                    value,
                });
            }
            let pacing_epoch = inner.next_pacing_epoch;
            inner.next_pacing_epoch += 1;
            inner.in_flight += 1;
            drop(inner);
            prepared.push(LaunchInstanceWork {
                instance: Arc::clone(instance),
                pacing_epoch,
                host_puts: puts,
            });
        }

        self.work_tx
            .as_ref()
            .ok_or_else(|| anyhow!("dummy driver worker is not available"))?
            .send(WorkItem::Launch {
                instances: prepared,
                completion,
            })
            .map_err(|_| anyhow!("dummy driver worker has shut down"))
    }

    pub fn copy_kv(&mut self, desc: &PieKvCopyDesc, completion: PieCompletion) -> Result<()> {
        validate_kv_copy_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("copy_kv");
        ensure_completion(completion)?;
        let _src_page_ids = copy_u32_slice(desc.src_page_ids, "copy_kv.src_page_ids")?;
        let _dst_page_ids = copy_u32_slice(desc.dst_page_ids, "copy_kv.dst_page_ids")?;
        let cells = copy_kv_cells(desc)?;
        if !cells.is_empty() {
            ensure!(
                desc.src_page_ids.len == 0 && desc.dst_page_ids.len == 0,
                "copy_kv cells and whole-page lists are mutually exclusive"
            );
        }
        self.enqueue_noop(WorkItem::CopyKv { completion })
    }

    pub fn copy_state(&mut self, desc: &PieStateCopyDesc, completion: PieCompletion) -> Result<()> {
        validate_state_copy_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("copy_state");
        ensure_completion(completion)?;
        let ranges = copy_state_ranges(desc)?;
        if !ranges.is_empty() {
            ensure!(
                ranges
                    .iter()
                    .all(|range| range.token_count > 0 || range.src_slot_id != range.dst_slot_id),
                "copy_state requires non-empty token counts or distinct src/dst slots"
            );
        }
        self.enqueue_noop(WorkItem::CopyState { completion })
    }

    pub fn resize_pool(
        &mut self,
        desc: &PiePoolResizeDesc,
        completion: PieCompletion,
    ) -> Result<()> {
        validate_pool_resize_desc(desc).map_err(|err| anyhow!(err))?;
        ensure_abi(desc.abi_version)?;
        self.record_op("resize_pool");
        ensure_completion(completion)?;
        let _maps = copy_pool_ranges(desc.map_ranges, "resize_pool.map_ranges")?;
        let _unmaps = copy_pool_ranges(desc.unmap_ranges, "resize_pool.unmap_ranges")?;
        self.enqueue_noop(WorkItem::ResizePool { completion })
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
        while inner.in_flight != 0 {
            inner = instance.cv.wait(inner).unwrap();
        }
        inner.closed = true;
        drop(inner);
        self.state.lock().unwrap().instances.remove(&instance_id);
        Ok(())
    }

    fn enqueue_noop(&mut self, item: WorkItem) -> Result<()> {
        self.work_tx
            .as_ref()
            .ok_or_else(|| anyhow!("dummy driver worker is not available"))?
            .send(item)
            .map_err(|_| anyhow!("dummy driver worker has shut down"))
    }

    fn model_profile(&self) -> ModelProfile {
        ModelProfile {
            vocab: self.capabilities.vocab_size,
            page_size: self.capabilities.kv_page_size,
            num_layers: 2,
            activation: DType::F32,
            has_mtp_logits: true,
            has_mtp_drafts: true,
            has_value_head: true,
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
        let _ = self.work_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.record_op("destroy");
    }
}

fn worker_loop(
    rx: mpsc::Receiver<WorkItem>,
    runtime: SendableRuntimeCallbacks,
    callback_delay_ms: u64,
) {
    while let Ok(item) = rx.recv() {
        match item {
            WorkItem::Launch {
                instances,
                completion,
            } => {
                for instance in instances {
                    process_launch_instance(&instance, &runtime, callback_delay_ms);
                }
                notify_completion(
                    &runtime,
                    completion.wait_id,
                    completion.target_epoch,
                    callback_delay_ms,
                );
            }
            WorkItem::CopyKv { completion }
            | WorkItem::CopyState { completion }
            | WorkItem::ResizePool { completion } => {
                notify_completion(
                    &runtime,
                    completion.wait_id,
                    completion.target_epoch,
                    callback_delay_ms,
                );
            }
        }
    }
}

fn process_launch_instance(
    instance: &LaunchInstanceWork,
    runtime: &SendableRuntimeCallbacks,
    callback_delay_ms: u64,
) {
    let mut notify_readers = Vec::new();
    let pacing_wait_id = instance.instance.pacing_wait_id;
    {
        let mut inner = instance.instance.inner.lock().unwrap();
        match run_instance_step(
            &mut inner,
            &instance.instance.program,
            instance.instance.instance_id,
            instance.pacing_epoch,
            &instance.host_puts,
        ) {
            Ok(reader_epochs) => {
                for (wait_id, epoch) in reader_epochs {
                    notify_readers.push((wait_id, epoch));
                }
            }
            Err(err) => {
                poison_instance(&mut inner, &mut notify_readers, &err.to_string());
            }
        }
    }
    for (wait_id, epoch) in notify_readers {
        notify_completion(runtime, wait_id, epoch, callback_delay_ms);
    }
    if pacing_wait_id != 0 {
        notify_completion(
            runtime,
            pacing_wait_id,
            instance.pacing_epoch,
            callback_delay_ms,
        );
    }
    let mut inner = instance.instance.inner.lock().unwrap();
    inner.in_flight = inner.in_flight.saturating_sub(1);
    instance.instance.cv.notify_all();
}

fn run_instance_step(
    inner: &mut BoundInstanceInner,
    program: &DummyProgram,
    instance_id: u64,
    pacing_epoch: u64,
    host_puts: &[PreparedHostPut],
) -> Result<Vec<(u64, u64)>> {
    for host_put in host_puts {
        inner
            .interp
            .host_put(
                &program.bound,
                host_put.dense_channel as u32,
                host_put.value.clone(),
            )
            .map_err(|err| anyhow!("host put failed: {err:?}"))?;
    }
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
        return Ok(Vec::new());
    }

    let mut notified = Vec::new();
    for dense in 0..inner.channels.len() {
        let channel = inner.channels[dense].clone();
        if channel.host_role != HostRole::Reader {
            continue;
        }
        let mut channel_notified = false;
        loop {
            match inner.interp.host_take(&program.bound, dense as u32) {
                Ok(value) => {
                    let bytes = encode_wire_value(&value, channel.ty).map_err(|err| {
                        anyhow!("channel {} encode failed: {err}", channel.global_id)
                    })?;
                    let epoch = publish_reader_value(inner, dense, &bytes)?;
                    if !channel_notified {
                        notified.push((channel.waits.reader_wait_id, epoch));
                        channel_notified = true;
                    } else if let Some((_, last_epoch)) = notified.last_mut() {
                        *last_epoch = epoch;
                    }
                }
                Err(HostError::WouldBlock) => break,
                Err(err) => bail!(
                    "host take failed for channel {}: {err:?}",
                    channel.global_id
                ),
            }
        }
    }
    Ok(notified)
}

fn poison_instance(
    inner: &mut BoundInstanceInner,
    notify_readers: &mut Vec<(u64, u64)>,
    _reason: &str,
) {
    inner.interp.poison();
    for channel in &inner.channels {
        if channel.host_role != HostRole::Reader {
            continue;
        }
        let tail = inner.words[channel.tail_word_index].load(Ordering::Acquire);
        inner.words[channel.poison_word_index].store(1, Ordering::Release);
        notify_readers.push((channel.waits.reader_wait_id, tail.saturating_add(1).max(1)));
    }
}

fn publish_reader_value(inner: &mut BoundInstanceInner, dense: usize, bytes: &[u8]) -> Result<u64> {
    let channel = &inner.channels[dense];
    ensure!(
        bytes.len() == channel.cell_bytes,
        "channel {} wrote {} bytes, expected {}",
        channel.global_id,
        bytes.len(),
        channel.cell_bytes
    );
    let tail = inner.words[channel.tail_word_index].load(Ordering::Acquire);
    let ring_cells = channel.capacity as usize + 1;
    let slot = (tail as usize % ring_cells) * channel.cell_bytes;
    let start = channel.mirror_offset + slot;
    let end = start + channel.cell_bytes;
    inner.mirror[start..end].copy_from_slice(bytes);
    let next_tail = tail + 1;
    inner.words[channel.tail_word_index].store(next_tail, Ordering::Release);
    let head = inner.words[channel.head_word_index].load(Ordering::Acquire);
    if next_tail.saturating_sub(head) > channel.capacity as u64 {
        inner.words[channel.head_word_index]
            .store(next_tail - channel.capacity as u64, Ordering::Release);
    }
    inner.words[channel.poison_word_index].store(0, Ordering::Release);
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
    if let Some(notify) = runtime.notify {
        let _ = std::panic::catch_unwind(|| unsafe {
            notify(runtime.ctx as *mut std::ffi::c_void, wait_id, epoch)
        });
    }
}

fn build_channel_layout(
    program: &Arc<DummyProgram>,
    channel_ids: &[u64],
    channel_waits: &[PieChannelWait],
) -> Result<(
    Vec<DummyChannel>,
    Box<[PieChannelBinding]>,
    Box<[u8]>,
    Box<[AtomicU64]>,
)> {
    let mut channels = Vec::with_capacity(program.bound.container.channels.len());
    let mut bindings = Vec::new();
    let mut mirror_bytes = 0usize;
    let mut reader_rank = 0usize;
    for (dense, decl) in program.bound.container.channels.iter().enumerate() {
        let ty = program.bound.channel_types[dense];
        let cell_bytes = wire_len(ty);
        let (mirror_offset, head_word_index, tail_word_index, poison_word_index) =
            if decl.host_role == HostRole::Reader {
                let mirror_offset = mirror_bytes;
                mirror_bytes += cell_bytes * (decl.capacity as usize + 1);
                let head_word_index = reader_rank * 3;
                let tail_word_index = head_word_index + 1;
                let poison_word_index = head_word_index + 2;
                reader_rank += 1;
                bindings.push(PieChannelBinding {
                    channel_id: channel_ids[dense],
                    cell_bytes: cell_bytes as u32,
                    capacity: decl.capacity,
                    mirror_offset: mirror_offset as u64,
                    head_word_index: head_word_index as u32,
                    tail_word_index: tail_word_index as u32,
                    poison_word_index: poison_word_index as u32,
                    reserved: 0,
                });
                (
                    mirror_offset,
                    head_word_index,
                    tail_word_index,
                    poison_word_index,
                )
            } else {
                (0, 0, 0, 0)
            };
        channels.push(DummyChannel {
            global_id: channel_ids[dense],
            host_role: decl.host_role,
            ty,
            capacity: decl.capacity,
            cell_bytes,
            mirror_offset,
            head_word_index,
            tail_word_index,
            poison_word_index,
            waits: channel_waits[dense],
        });
    }
    let mirror = vec![0xA5u8; mirror_bytes].into_boxed_slice();
    let words = (0..reader_rank * 3)
        .map(|_| AtomicU64::new(0))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok((channels, bindings.into_boxed_slice(), mirror, words))
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
    ensure!(
        completion.wait_id != 0,
        "completion wait_id must be nonzero"
    );
    ensure!(
        completion.target_epoch != 0,
        "completion target_epoch must be nonzero"
    );
    Ok(())
}

fn validate_launch_shape(desc: &PieLaunchDesc, instance_count: usize) -> Result<()> {
    let _token_ids = copy_u32_slice(desc.token_ids, "launch.token_ids")?;
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

    if !kv_page_indptr.is_empty() || !kv_page_indices.is_empty() {
        validate_indptr(
            &kv_page_indptr,
            kv_page_indices.len(),
            instance_count,
            "launch.kv_page_indptr",
        )?;
    }
    if !kv_last_page_lens.is_empty() {
        ensure!(
            kv_last_page_lens.len() == instance_count,
            "launch.kv_last_page_lens must have one entry per instance"
        );
    }
    if !qo_indptr.is_empty() {
        ensure!(
            qo_indptr.len() == instance_count + 1,
            "launch.qo_indptr length mismatch"
        );
        ensure!(
            qo_indptr.windows(2).all(|w| w[0] <= w[1]),
            "launch.qo_indptr must be monotonic"
        );
    }
    if !sampling_indptr.is_empty() || !sampling_indices.is_empty() {
        validate_indptr(
            &sampling_indptr,
            sampling_indices.len(),
            instance_count,
            "launch.sampling_indptr",
        )?;
    }
    if !rs_slot_ids.is_empty() || !rs_slot_flags.is_empty() || !rs_fold_lens.is_empty() {
        ensure!(
            rs_slot_ids.len() == instance_count
                && rs_slot_flags.len() == instance_count
                && rs_fold_lens.len() == instance_count,
            "launch rs slot vectors must be parallel to instance_ids"
        );
    }
    if !rs_buffer_slot_indptr.is_empty() || !rs_buffer_slot_ids.is_empty() {
        validate_indptr(
            &rs_buffer_slot_indptr,
            rs_buffer_slot_ids.len(),
            instance_count,
            "launch.rs_buffer_slot_indptr",
        )?;
    }
    if !mask_request.is_empty() {
        ensure!(
            mask_request.len() == instance_count + 1,
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
            image_indptr.len() == instance_count + 1,
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
            audio_indptr.len() == instance_count + 1,
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

fn copy_channel_waits(slice: PieChannelWaitSlice, name: &str) -> Result<Vec<PieChannelWait>> {
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
    use pie_driver_abi::PieChannelValueDesc;
    use pie_ptir::container::{
        ChanDType, ChannelDecl, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::expand;
    use pie_ptir::registry::{Port, Stage};
    use pie_ptir::types::{Literal, Shape};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

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
        fn wait_for(&self, expected: usize) {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (lock, cv) = &self.pair;
            let mut seen = lock.lock().unwrap();
            while *seen < expected {
                let now = Instant::now();
                assert!(
                    now < deadline,
                    "timed out waiting for {expected} notifications"
                );
                let timeout = deadline.saturating_duration_since(now);
                let (next_seen, wait) = cv.wait_timeout(seen, timeout).unwrap();
                seen = next_seen;
                assert!(
                    !wait.timed_out(),
                    "timed out waiting for {expected} notifications"
                );
            }
        }

        fn notifications(&self) -> Vec<(u64, u64)> {
            self.notifications.lock().unwrap().clone()
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

    fn bind_program(
        driver: &mut DummyDriver,
        bytes: Vec<u8>,
        channel_ids: &[u64],
        waits: &[PieChannelWait],
        seeds: &[OwnedValueDesc],
        pacing_wait_id: u64,
    ) -> PieInstanceBinding {
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
        driver
            .bind_instance(&PieInstanceDesc {
                program_id,
                pacing_wait_id,
                channel_waits: PieChannelWaitSlice {
                    ptr: waits.as_ptr(),
                    len: waits.len(),
                },
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
            .unwrap()
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

    fn read_binding(binding: &PieInstanceBinding, index: usize) -> PieChannelBinding {
        unsafe { *binding.channels.ptr.add(index) }
    }

    fn read_bindings(binding: &PieInstanceBinding) -> Vec<PieChannelBinding> {
        (0..binding.channel_count as usize)
            .map(|index| read_binding(binding, index))
            .collect()
    }

    fn read_binding_by_id(binding: &PieInstanceBinding, channel_id: u64) -> PieChannelBinding {
        read_bindings(binding)
            .into_iter()
            .find(|channel| channel.channel_id == channel_id)
            .unwrap_or_else(|| panic!("missing exported binding for channel {channel_id}"))
    }

    fn read_words(binding: &PieInstanceBinding) -> Vec<u64> {
        unsafe {
            std::slice::from_raw_parts(
                binding.word_base as *const AtomicU64,
                binding.word_count as usize,
            )
        }
        .iter()
        .map(|word| word.load(Ordering::Acquire))
        .collect()
    }

    fn read_cell(binding: &PieInstanceBinding, channel: PieChannelBinding, slot: usize) -> Vec<u8> {
        let offset = channel.mirror_offset as usize + slot * channel.cell_bytes as usize;
        unsafe {
            std::slice::from_raw_parts(
                (binding.mirror_base as *const u8).add(offset),
                channel.cell_bytes as usize,
            )
        }
        .to_vec()
    }

    #[test]
    fn launch_publishes_token_entropy_dist_logprobs_and_notifies_readers_pacing_and_batch() {
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
        let instance_ids = [binding.instance_id];
        let host_put_indptr = [0u32, 0u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 401,
                    target_epoch: 7,
                },
            )
            .unwrap();
        callbacks.wait_for(6);
        let notices = callbacks.notifications();
        assert_eq!(
            notices
                .iter()
                .filter(|&&(id, epoch)| id == 401 && epoch == 7)
                .count(),
            1
        );
        assert!(
            notices.contains(&(301, 1)),
            "missing pacing notification: {notices:?}"
        );
        for wait_id in [101, 102, 103, 104] {
            assert!(
                notices
                    .iter()
                    .any(|&(id, epoch)| id == wait_id && epoch == 1)
            );
        }

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
        let host_put_indptr = [0u32, 0u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 421,
                    target_epoch: 3,
                },
            )
            .unwrap();
        callbacks.wait_for(3);
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
        let host_put_indptr = [0u32, 0u32, 0u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 431,
                    target_epoch: 5,
                },
            )
            .unwrap();
        callbacks.wait_for(5);
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
            binding.word_count, 6,
            "only reader channels contribute words"
        );
        assert_eq!(binding.word_bytes, 6 * std::mem::size_of::<u64>() as u64);
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
        assert_eq!(bindings[1].head_word_index, 3);
        assert_eq!(bindings[1].tail_word_index, 4);
        assert_eq!(bindings[1].poison_word_index, 5);
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
        let host_put_bytes = 7u32.to_le_bytes();
        let host_puts = [PieChannelValueDesc {
            channel_id: 100,
            bytes: PieBytes {
                ptr: host_put_bytes.as_ptr(),
                len: std::mem::size_of::<u32>(),
            },
        }];
        let instance_ids = [binding.instance_id];
        let host_put_indptr = [0u32, 1u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    ptir_host_put_values: PieChannelValueDescSlice {
                        ptr: host_puts.as_ptr(),
                        len: host_puts.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 501,
                    target_epoch: 11,
                },
            )
            .unwrap();
        callbacks.wait_for(4);
        let notices = callbacks.notifications();
        assert!(notices.contains(&(202, 1)));
        assert!(notices.contains(&(203, 1)));
        assert!(notices.contains(&(401, 1)));
        assert!(notices.contains(&(501, 11)));
        for wait_id in [200, 201, 204] {
            assert!(
                !notices.iter().any(|&(id, _)| id == wait_id),
                "non-reader wait {wait_id} must not be notified: {notices:?}"
            );
        }

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
    fn mixed_roles_kernel_fault_poisons_only_exported_readers() {
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
        let host_put_bytes = 9u32.to_le_bytes();
        let host_puts = [PieChannelValueDesc {
            channel_id: 110,
            bytes: PieBytes {
                ptr: host_put_bytes.as_ptr(),
                len: std::mem::size_of::<u32>(),
            },
        }];
        let instance_ids = [binding.instance_id];
        let host_put_indptr = [0u32, 1u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    ptir_host_put_values: PieChannelValueDescSlice {
                        ptr: host_puts.as_ptr(),
                        len: host_puts.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 511,
                    target_epoch: 13,
                },
            )
            .unwrap();
        callbacks.wait_for(4);
        let notices = callbacks.notifications();
        assert!(notices.contains(&(212, 1)));
        assert!(notices.contains(&(213, 1)));
        assert!(notices.contains(&(411, 1)));
        assert!(notices.contains(&(511, 13)));
        for wait_id in [210, 211] {
            assert!(
                !notices.iter().any(|&(id, _)| id == wait_id),
                "non-reader wait {wait_id} must not be notified: {notices:?}"
            );
        }
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
        let host_put_indptr = [0u32, 0u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 451,
                    target_epoch: 9,
                },
            )
            .unwrap();
        callbacks.wait_for(3);
        let channel = read_binding_by_id(&binding, 51);
        let words = read_words(&binding);
        assert_eq!(words[channel.tail_word_index as usize], 0);
        assert_eq!(words[channel.poison_word_index as usize], 1);
        let notices = callbacks.notifications();
        assert!(notices.contains(&(151, 1)));
        assert!(notices.contains(&(351, 1)));
        assert!(notices.contains(&(451, 9)));
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
        let host_put_indptr = [1u32, 1u32];
        let err = driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 461,
                    target_epoch: 1,
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
    fn close_waits_for_inflight_then_rejects_stale_instance_id() {
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
        let host_put_indptr = [0u32, 0u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 471,
                    target_epoch: 1,
                },
            )
            .unwrap();
        let started = Instant::now();
        driver.close_instance(binding.instance_id).unwrap();
        assert!(started.elapsed() >= Duration::from_millis(40));
        callbacks.wait_for(3);
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
        let host_put_indptr = [0u32, 0u32];
        driver
            .launch(
                &PieLaunchDesc {
                    instance_ids: PieU64Slice {
                        ptr: instance_ids.as_ptr(),
                        len: instance_ids.len(),
                    },
                    host_put_indptr: PieU32Slice {
                        ptr: host_put_indptr.as_ptr(),
                        len: host_put_indptr.len(),
                    },
                    ..PieLaunchDesc::default()
                },
                PieCompletion {
                    wait_id: 481,
                    target_epoch: 1,
                },
            )
            .unwrap();
        drop(driver);
        let at_drop = callbacks.count.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(callbacks.count.load(Ordering::SeqCst), at_drop);
        assert_eq!(
            at_drop, 3,
            "drop should wait for the in-flight callback work"
        );
    }
}
