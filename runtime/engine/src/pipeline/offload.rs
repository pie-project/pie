//! Remote executor partner registry, admission, transfer, and prefix adoption.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use pie_driver_abi::{
    ExecutorRequest, ExecutorResponse, InlineKvPayload, MemoryDomain, PushKv, RemoteTransferKind,
};

use crate::pipeline::program::RegisteredProgram;
use crate::store::kv::page_table::WorkingSetId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartnerRole {
    Prefill,
    Encode,
}

pub struct Partner {
    worker_id: u64,
    destination_worker_id: u64,
    driver_id: Option<usize>,
    role: PartnerRole,
    transfer: RemoteTransferKind,
    client: Option<pie_driver_abi::ExecutorRpcClient>,
    max_outstanding: u32,
    outstanding: AtomicU32,
    available: AtomicBool,
    drained: tokio::sync::Notify,
}

impl Partner {
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn driver_id(&self) -> usize {
        self.driver_id
            .expect("only prefill partners have remote driver slots")
    }

    pub fn role(&self) -> PartnerRole {
        self.role
    }

    pub fn transfer_kind(&self) -> RemoteTransferKind {
        self.transfer
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding.load(Ordering::Relaxed)
    }

    pub fn mark_suspect(&self) {
        self.available.store(false, Ordering::Release);
    }

    pub fn mark_available(&self) {
        self.available.store(true, Ordering::Release);
    }

    pub async fn wait_drained(&self) {
        loop {
            if self.outstanding() == 0 {
                return;
            }
            let notified = self.drained.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.outstanding() == 0 {
                return;
            }
            notified.await;
        }
    }

    pub async fn pull_kv(&self, src_page_ids: Vec<u32>, dst_page_ids: Vec<u32>) -> Result<()> {
        ensure!(
            src_page_ids.len() == dst_page_ids.len(),
            "inline KV source/destination page counts differ"
        );
        ensure!(
            matches!(
                self.transfer,
                RemoteTransferKind::Inline | RemoteTransferKind::Nixl
            ),
            "transfer {:?} is not wired yet",
            self.transfer
        );
        let expected_dst_page_ids = dst_page_ids.clone();
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow!("partner has no transfer client"))?;
        let mut context = tarpc::context::current();
        context.deadline = Instant::now() + Duration::from_secs(24 * 60 * 60);
        let response = client
            .execute(
                context,
                ExecutorRequest::PushKv(PushKv {
                    src_page_ids,
                    dst_page_ids,
                    dst_worker: self.destination_worker_id,
                }),
            )
            .await
            .map_err(|error| anyhow!("inline KV transport failed: {error}"))?
            .map_err(|error| anyhow!("inline KV push rejected: {error}"))?;
        match (self.transfer, response) {
            (RemoteTransferKind::Inline, ExecutorResponse::KvPayload(payload)) => {
                ensure!(
                    payload.dst_page_ids == expected_dst_page_ids,
                    "executor changed inline KV destination page ids"
                );
                import_inline_payload(&payload)
            }
            (RemoteTransferKind::Nixl, ExecutorResponse::KvPushed) => Ok(()),
            (_, response) => Err(anyhow!(
                "executor returned unexpected KV push response {response:?}"
            )),
        }
    }

    pub async fn encode(
        &self,
        plan: pie_driver_abi::LaunchPlan,
    ) -> Result<pie_driver_abi::RemoteEmbeddings> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow!("partner has no executor client"))?;
        let mut context = tarpc::context::current();
        context.deadline = Instant::now() + Duration::from_secs(300);
        let response = client
            .execute(
                context,
                ExecutorRequest::Encode(pie_driver_abi::RemoteEncode { plan }),
            )
            .await
            .map_err(|error| anyhow!("encode transport failed: {error}"))?
            .map_err(|error| anyhow!("encode rejected: {error}"))?;
        let ExecutorResponse::Embeddings(embeddings) = response else {
            return Err(anyhow!(
                "executor returned unexpected encode response {response:?}"
            ));
        };
        Ok(embeddings)
    }

    fn try_claim(self: &Arc<Self>) -> Option<PartnerGuard> {
        if !self.available.load(Ordering::Acquire) {
            return None;
        }
        let claimed = self
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.max_outstanding).then_some(current + 1)
            })
            .is_ok();
        if !claimed {
            return None;
        }
        if !self.available.load(Ordering::Acquire) {
            let previous = self.outstanding.fetch_sub(1, Ordering::AcqRel);
            if previous == 1 {
                self.drained.notify_waiters();
            }
            return None;
        }
        Some(PartnerGuard {
            partner: Arc::clone(self),
        })
    }
}

pub struct PartnerGuard {
    partner: Arc<Partner>,
}

impl PartnerGuard {
    pub fn partner(&self) -> &Arc<Partner> {
        &self.partner
    }
}

impl Drop for PartnerGuard {
    fn drop(&mut self) {
        let previous = self.partner.outstanding.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.partner.drained.notify_waiters();
        }
    }
}

type PartnerKey = (PartnerRole, u64);

static PARTNERS: LazyLock<RwLock<HashMap<PartnerKey, Arc<Partner>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub fn register_partner(
    worker_id: u64,
    destination_worker_id: u64,
    driver_id: impl Into<Option<usize>>,
    role: PartnerRole,
    max_outstanding: u32,
    transfer: RemoteTransferKind,
    client: Option<pie_driver_abi::ExecutorRpcClient>,
) -> Arc<Partner> {
    let partner = Arc::new(Partner {
        worker_id,
        destination_worker_id,
        driver_id: driver_id.into(),
        role,
        transfer,
        client,
        max_outstanding: max_outstanding.max(1),
        outstanding: AtomicU32::new(0),
        available: AtomicBool::new(true),
        drained: tokio::sync::Notify::new(),
    });
    if let Some(previous) = PARTNERS
        .write()
        .unwrap()
        .insert((role, worker_id), Arc::clone(&partner))
    {
        previous.mark_suspect();
    }
    partner
}

pub fn remove_partner(worker_id: u64, role: PartnerRole) {
    if let Some(partner) = PARTNERS.write().unwrap().remove(&(role, worker_id)) {
        partner.mark_suspect();
    }
}

pub fn close_driver_surrogates(driver_id: usize) {
    let mut cache = SURROGATES.lock().unwrap();
    let keys = cache
        .keys()
        .filter(|(candidate, _, _)| *candidate == driver_id)
        .copied()
        .collect::<Vec<_>>();
    for key in keys {
        if let Some(surrogate) = cache.remove(&key) {
            let _ = crate::scheduler::close_instance(&surrogate.bound);
        }
    }
}

pub(crate) fn close_home_instance(home_instance_id: u64) {
    let users = {
        let mut states = HOME_STATES.lock().unwrap();
        let state = states.entry(home_instance_id).or_default();
        state.closed = true;
        state.users
    };
    let surrogates = {
        let mut cache = SURROGATES.lock().unwrap();
        let keys = cache
            .keys()
            .filter(|(_, _, candidate)| *candidate == home_instance_id)
            .copied()
            .collect::<Vec<_>>();
        keys.into_iter()
            .filter_map(|key| cache.remove(&key))
            .collect::<Vec<_>>()
    };
    if surrogates.is_empty() {
        if users == 0 {
            HOME_STATES.lock().unwrap().remove(&home_instance_id);
        }
        return;
    }
    let close = move || {
        for surrogate in surrogates {
            while Arc::strong_count(&surrogate) != 1 {
                std::thread::sleep(Duration::from_millis(1));
            }
            let _ = crate::scheduler::close_instance(&surrogate.bound);
        }
        let mut states = HOME_STATES.lock().unwrap();
        if states
            .get(&home_instance_id)
            .is_some_and(|state| state.closed && state.users == 0)
        {
            states.remove(&home_instance_id);
        }
    };
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn_blocking(close);
    } else {
        let _ = std::thread::Builder::new()
            .name(format!("pie-surrogate-close-{home_instance_id}"))
            .spawn(close);
    }
}

pub fn select_partner(role: PartnerRole) -> Option<PartnerGuard> {
    let candidates = PARTNERS
        .read()
        .unwrap()
        .values()
        .filter(|partner| {
            partner.role == role
                && partner.available.load(Ordering::Acquire)
                && partner.outstanding() < partner.max_outstanding
        })
        .cloned()
        .collect::<Vec<_>>();
    match candidates.len() {
        0 => None,
        1 => candidates[0].try_claim(),
        len => {
            let first = (next_random() % len as u64) as usize;
            let mut second = (next_random() % (len as u64 - 1)) as usize;
            if second >= first {
                second += 1;
            }
            let (a, b) = (&candidates[first], &candidates[second]);
            let preferred = if (a.outstanding(), a.driver_id) <= (b.outstanding(), b.driver_id) {
                [Arc::clone(a), Arc::clone(b)]
            } else {
                [Arc::clone(b), Arc::clone(a)]
            };
            let selected = preferred
                .into_iter()
                .find_map(|partner| partner.try_claim());
            selected.or_else(|| candidates.iter().find_map(|partner| partner.try_claim()))
        }
    }
}

fn next_random() -> u64 {
    static STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);
    let mut value = STATE
        .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, Copy)]
struct OffloadSettings {
    enabled: bool,
    prefill_min_suffix_tokens: usize,
    encode_injection: bool,
    encode_hidden_size: u32,
}

static SETTINGS: LazyLock<RwLock<OffloadSettings>> = LazyLock::new(|| {
    RwLock::new(OffloadSettings {
        enabled: false,
        prefill_min_suffix_tokens: 0,
        encode_injection: false,
        encode_hidden_size: 0,
    })
});

static HOME_KV_HANDLE: LazyLock<RwLock<Option<pie_driver_abi::KvHandle>>> =
    LazyLock::new(|| RwLock::new(None));

#[cfg(feature = "driver-cuda")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
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
        ensure!(status == 0, "cudaGetDevice failed with status {status}");
        if previous != device as i32 {
            let status = unsafe { cudaSetDevice(device as i32) };
            ensure!(status == 0, "cudaSetDevice failed with status {status}");
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

fn copy_host_to_region(domain: MemoryDomain, dst: u64, src: &[u8]) -> Result<()> {
    match domain {
        MemoryDomain::HostPinned => unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
            Ok(())
        },
        MemoryDomain::CudaDevice(_device) => {
            #[cfg(feature = "driver-cuda")]
            {
                let _guard = CudaDeviceGuard::select(_device)?;
                let status = unsafe {
                    cudaMemcpy(
                        dst as *mut std::ffi::c_void,
                        src.as_ptr() as *const std::ffi::c_void,
                        src.len(),
                        4,
                    )
                };
                ensure!(status == 0, "cudaMemcpy H2D failed with status {status}");
                Ok(())
            }
            #[cfg(not(feature = "driver-cuda"))]
            {
                Err(anyhow!("CUDA KV import requires feature \"driver-cuda\""))
            }
        }
        other => Err(anyhow!("inline KV import does not support {other:?}")),
    }
}

pub fn configure(enabled: bool, prefill_min_suffix_tokens: usize) {
    let mut settings = SETTINGS.write().unwrap();
    settings.enabled = enabled;
    settings.prefill_min_suffix_tokens = prefill_min_suffix_tokens;
}

pub fn configure_encode_injection(enabled: bool, hidden_size: u32) {
    let mut settings = SETTINGS.write().unwrap();
    settings.encode_injection = enabled;
    settings.encode_hidden_size = if enabled { hidden_size } else { 0 };
}

pub fn set_home_kv_handle(handle: pie_driver_abi::KvHandle) {
    *HOME_KV_HANDLE.write().unwrap() = Some(handle);
}

fn prefill_threshold(kind: RemoteTransferKind) -> Option<usize> {
    let settings = *SETTINGS.read().unwrap();
    if !settings.enabled {
        return None;
    }
    Some(if settings.prefill_min_suffix_tokens != 0 {
        settings.prefill_min_suffix_tokens
    } else if kind == RemoteTransferKind::Nixl {
        512
    } else {
        2048
    })
}

fn import_inline_payload(payload: &InlineKvPayload) -> Result<()> {
    let handle = HOME_KV_HANDLE
        .read()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow!("home KV handle is not registered"))?;
    ensure!(!handle.regions.is_empty(), "home KV handle has no regions");
    let page_bytes = handle.page_bytes();
    ensure!(
        payload.page_bytes == page_bytes,
        "inline KV page size {} != home page size {page_bytes}",
        payload.page_bytes
    );
    let expected = page_bytes
        .checked_mul(payload.dst_page_ids.len() as u64)
        .ok_or_else(|| anyhow!("inline KV payload size overflow"))?;
    ensure!(
        payload.bytes.len() as u64 == expected,
        "inline KV payload has {} bytes, expected {expected}",
        payload.bytes.len()
    );
    for (&page, page_payload) in payload
        .dst_page_ids
        .iter()
        .zip(payload.bytes.chunks_exact(page_bytes as usize))
    {
        let mut payload_offset = 0usize;
        for region in &handle.regions {
            let offset = (page as u64)
                .checked_mul(region.page_stride)
                .ok_or_else(|| anyhow!("inline KV destination offset overflow"))?;
            ensure!(
                region.page_stride > 0
                    && offset
                        .checked_add(region.page_stride)
                        .is_some_and(|end| end <= region.len),
                "inline KV destination page {page} exceeds home region"
            );
            let end = payload_offset + region.page_stride as usize;
            copy_host_to_region(
                region.domain,
                region.base + offset,
                &page_payload[payload_offset..end],
            )?;
            payload_offset = end;
        }
        ensure!(
            payload_offset == page_payload.len(),
            "inline KV region strides do not cover the logical page"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OffloadCounterSnapshot {
    pub no_partner: u64,
    pub below_threshold: u64,
    pub noncanonical: u64,
    pub recurrent_state: u64,
    pub user_mask: u64,
    pub channels: u64,
    pub media: u64,
    pub shape: u64,
    pub remote_failure: u64,
    pub transfer_failure: u64,
    pub adopted: u64,
}

#[derive(Default)]
struct OffloadCounters {
    no_partner: AtomicU64,
    below_threshold: AtomicU64,
    noncanonical: AtomicU64,
    recurrent_state: AtomicU64,
    user_mask: AtomicU64,
    channels: AtomicU64,
    media: AtomicU64,
    shape: AtomicU64,
    remote_failure: AtomicU64,
    transfer_failure: AtomicU64,
    adopted: AtomicU64,
}

static COUNTERS: LazyLock<OffloadCounters> = LazyLock::new(OffloadCounters::default);

pub fn counters() -> OffloadCounterSnapshot {
    let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
    OffloadCounterSnapshot {
        no_partner: load(&COUNTERS.no_partner),
        below_threshold: load(&COUNTERS.below_threshold),
        noncanonical: load(&COUNTERS.noncanonical),
        recurrent_state: load(&COUNTERS.recurrent_state),
        user_mask: load(&COUNTERS.user_mask),
        channels: load(&COUNTERS.channels),
        media: load(&COUNTERS.media),
        shape: load(&COUNTERS.shape),
        remote_failure: load(&COUNTERS.remote_failure),
        transfer_failure: load(&COUNTERS.transfer_failure),
        adopted: load(&COUNTERS.adopted),
    }
}

pub(crate) async fn try_encode(request: &mut pie_driver_abi::LaunchPlan) -> bool {
    let settings = *SETTINGS.read().unwrap();
    if !settings.enabled || !settings.encode_injection || request.image_pixels.is_empty() {
        return false;
    }
    let Some(partner_guard) = select_partner(PartnerRole::Encode) else {
        COUNTERS.no_partner.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let partner = Arc::clone(partner_guard.partner());
    let embeddings = match partner.encode(request.clone()).await {
        Ok(embeddings) => embeddings,
        Err(error) => {
            tracing::debug!(%error, "encode offload failed; using local media path");
            return false;
        }
    };
    let blocks = embeddings.dtypes.len();
    let valid = blocks > 0
        && settings.encode_hidden_size > 0
        && embeddings.indptr.len() == blocks + 1
        && embeddings.shapes.len() == blocks * 2
        && embeddings.anchor_rows.len() == blocks
        && embeddings.indptr.first().copied() == Some(0)
        && embeddings.indptr.last().copied() == Some(embeddings.rows.len() as u32)
        && embeddings
            .indptr
            .windows(2)
            .all(|window| window[0] <= window[1])
        && (0..blocks).all(|block| {
            let rows = embeddings.shapes[2 * block] as usize;
            let width = embeddings.shapes[2 * block + 1];
            let begin = embeddings.indptr[block] as usize;
            let end = embeddings.indptr[block + 1] as usize;
            embeddings.dtypes[block] == 2
                && rows > 0
                && width == settings.encode_hidden_size
                && end >= begin
                && rows
                    .checked_mul(width as usize)
                    .and_then(|elements| elements.checked_mul(2))
                    == Some(end - begin)
                && (embeddings.anchor_rows[block] as usize)
                    .checked_add(rows)
                    .is_some_and(|end_row| end_row <= request.token_ids.len())
        });
    if !valid {
        tracing::debug!("encode partner returned malformed embedding metadata");
        return false;
    }
    request.embed_rows = embeddings.rows;
    request.embed_indptr = embeddings.indptr;
    request.embed_shapes = embeddings.shapes;
    request.embed_dtypes = embeddings.dtypes;
    request.embed_anchor_rows = embeddings.anchor_rows;
    request.embed_block_indptr = vec![0, blocks as u32];
    request.image_indptr.clear();
    request.image_grids.clear();
    request.image_anchor_positions.clear();
    request.image_pixels.clear();
    request.image_pixel_indptr.clear();
    request.image_mrope_positions.clear();
    request.image_mrope_indptr.clear();
    request.image_patch_positions.clear();
    request.image_anchor_rows.clear();
    request.audio_features.clear();
    request.audio_feature_indptr.clear();
    request.audio_anchor_rows.clear();
    request.audio_indptr.clear();
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OffloadAdoption {
    pub token_count: usize,
}

type SurrogateKey = (usize, u64, u64);
struct Surrogate {
    bound: crate::driver::BoundInstance,
    _channels: Vec<Arc<crate::driver::ChannelEndpoint>>,
}

static SURROGATES: LazyLock<Mutex<HashMap<SurrogateKey, Arc<Surrogate>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[derive(Default)]
struct HomeState {
    closed: bool,
    users: usize,
}

static HOME_STATES: LazyLock<Mutex<HashMap<u64, HomeState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_SURROGATE_CHANNEL_ID: AtomicU64 = AtomicU64::new(1);

struct HomeUse {
    instance_id: u64,
}

impl HomeUse {
    fn acquire(instance_id: u64) -> Result<Self> {
        let mut states = HOME_STATES.lock().unwrap();
        let state = states.entry(instance_id).or_default();
        ensure!(!state.closed, "home instance {instance_id} is closed");
        state.users += 1;
        Ok(Self { instance_id })
    }
}

impl Drop for HomeUse {
    fn drop(&mut self) {
        let mut states = HOME_STATES.lock().unwrap();
        let Some(state) = states.get_mut(&self.instance_id) else {
            return;
        };
        state.users = state.users.saturating_sub(1);
        if state.closed && state.users == 0 {
            states.remove(&self.instance_id);
        }
    }
}

fn home_instance_closed(instance_id: u64) -> bool {
    HOME_STATES
        .lock()
        .unwrap()
        .get(&instance_id)
        .is_some_and(|state| state.closed)
}

fn surrogate(
    driver_id: usize,
    program: &Arc<RegisteredProgram>,
    home_instance_id: u64,
) -> Result<Arc<Surrogate>> {
    let key = (driver_id, program.hash, home_instance_id);
    ensure!(
        !home_instance_closed(home_instance_id),
        "home instance {home_instance_id} is closed"
    );
    {
        let cache = SURROGATES.lock().unwrap();
        if let Some(bound) = cache.get(&key) {
            return Ok(Arc::clone(bound));
        }
    }
    let program_id = crate::scheduler::register_program(
        driver_id,
        pie_driver_abi::ProgramRegistration {
            program_hash: program.hash,
            canonical_bytes: program.bytes.clone(),
            sidecar_bytes: program.sidecar.clone(),
        },
    )?;
    let mut channel_ids = Vec::with_capacity(program.bound.container.channels.len());
    let mut channels = Vec::with_capacity(program.bound.container.channels.len());
    for declaration in &program.bound.container.channels {
        let channel_id = NEXT_SURROGATE_CHANNEL_ID.fetch_add(1, Ordering::Relaxed);
        ensure!(
            channel_id != 0,
            "remote surrogate channel id space exhausted"
        );
        let endpoint = crate::scheduler::register_channel(
            driver_id,
            pie_driver_abi::ChannelRegistrationPlan {
                driver_id,
                channel_id,
                shape: declaration.shape.dims().to_vec(),
                dtype: declaration.dtype.tag(),
                host_role: declaration.host_role as u8,
                seeded: declaration.seeded,
                extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                capacity: declaration.capacity,
                reader_wait_id: 0,
                writer_wait_id: 0,
                extern_name: Vec::new(),
            },
        )?;
        channel_ids.push(channel_id);
        channels.push(endpoint);
    }
    let bound = crate::scheduler::bind_instance(driver_id, program_id, 0, channel_ids, Vec::new())?;
    let surrogate = Arc::new(Surrogate {
        bound,
        _channels: channels,
    });
    let states = HOME_STATES.lock().unwrap();
    if states
        .get(&home_instance_id)
        .is_some_and(|state| state.closed)
    {
        drop(states);
        let _ = crate::scheduler::close_instance(&surrogate.bound);
        return Err(anyhow!(
            "home instance {home_instance_id} closed during bind"
        ));
    }
    let mut cache = SURROGATES.lock().unwrap();
    if let Some(existing) = cache.get(&key) {
        let _ = crate::scheduler::close_instance(&surrogate.bound);
        return Ok(Arc::clone(existing));
    }
    cache.insert(key, Arc::clone(&surrogate));
    Ok(surrogate)
}

fn context_extension_program(
    profile: &pie_ptir::registry::ModelProfile,
) -> Result<Arc<RegisteredProgram>> {
    use pie_ptir::container::{StageProgram, TraceContainer};
    use pie_ptir::registry::Stage;

    static PROGRAM: std::sync::OnceLock<Arc<RegisteredProgram>> = std::sync::OnceLock::new();
    if let Some(program) = PROGRAM.get() {
        return Ok(Arc::clone(program));
    }
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
    let program = crate::pipeline::program::register(bytes, profile)
        .map_err(|error| anyhow!("registering context-extension program: {error}"))?;
    let _ = PROGRAM.set(Arc::clone(&program));
    Ok(PROGRAM.get().cloned().unwrap_or(program))
}

fn release_scratch(model_idx: usize, driver_id: usize, ws: WorkingSetId) {
    let stores = crate::store::registry::get(model_idx, driver_id);
    let mut kv = stores.kv.lock().unwrap();
    let epoch = kv.current_epoch();
    kv.release_working_set(ws, epoch);
    kv.retire_idle();
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn try_prefill(
    model_idx: usize,
    home_driver_id: usize,
    home_ws: WorkingSetId,
    page_size: u32,
    request: &mut pie_driver_abi::LaunchPlan,
    canonical_tokens: &[u32],
    program: &Arc<RegisteredProgram>,
    home_instance_id: u64,
    lifetime_guard: Option<crate::store::kv::working_set::KvFireLease>,
) -> Option<OffloadAdoption> {
    let task = tokio::spawn(try_prefill_owned(
        model_idx,
        home_driver_id,
        home_ws,
        page_size,
        request.clone(),
        canonical_tokens.to_vec(),
        Arc::clone(program),
        home_instance_id,
        lifetime_guard,
    ));
    match task.await {
        Ok(Some((adoption, rewritten))) => {
            *request = rewritten;
            Some(adoption)
        }
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, "prefill offload task failed");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn try_prefill_owned(
    model_idx: usize,
    home_driver_id: usize,
    home_ws: WorkingSetId,
    page_size: u32,
    mut request: pie_driver_abi::LaunchPlan,
    canonical_tokens: Vec<u32>,
    program: Arc<RegisteredProgram>,
    home_instance_id: u64,
    _lifetime_guard: Option<crate::store::kv::working_set::KvFireLease>,
) -> Option<(OffloadAdoption, pie_driver_abi::LaunchPlan)> {
    if !SETTINGS.read().unwrap().enabled {
        return None;
    }
    let _home_use = match HomeUse::acquire(home_instance_id) {
        Ok(usage) => usage,
        Err(_) => return None,
    };
    let Some(partner_guard) = select_partner(PartnerRole::Prefill) else {
        COUNTERS.no_partner.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let partner = Arc::clone(partner_guard.partner());
    let threshold = prefill_threshold(partner.transfer_kind()).expect("offload is enabled");
    if canonical_tokens.len() < threshold {
        COUNTERS.below_threshold.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if canonical_tokens != request.token_ids {
        COUNTERS.noncanonical.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !request.rs_slot_ids.is_empty()
        || !request.rs_slot_flags.is_empty()
        || !request.rs_buffer_slot_ids.is_empty()
    {
        COUNTERS.recurrent_state.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if request.has_user_mask || !request.masks.is_empty() {
        COUNTERS.user_mask.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let mutates_context =
        program.bound.container.stages.iter().any(|stage| {
            stage.stage != pie_ptir::registry::Stage::Epilogue && !stage.ops.is_empty()
        });
    if mutates_context {
        COUNTERS.channels.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !request.image_pixels.is_empty()
        || !request.image_grids.is_empty()
        || !request.audio_features.is_empty()
    {
        COUNTERS.media.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let count = request.token_ids.len();
    let single_lane = request.qo_indptr == [0, count as u32]
        && request.position_ids.len() == count
        && request
            .sampling_indices
            .iter()
            .all(|&index| (index as usize) < count);
    if !single_lane || count <= 1 || page_size == 0 {
        COUNTERS.shape.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let first_sample = request
        .sampling_indices
        .iter()
        .copied()
        .min()
        .map(|value| value as usize)
        .unwrap_or(count - 1);
    let offload_rows = (first_sample.min(count - 1) / page_size as usize) * page_size as usize;
    if offload_rows == 0 {
        COUNTERS.shape.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let home_stores = crate::store::registry::get(model_idx, home_driver_id);
    {
        let home = home_stores.kv.lock().unwrap();
        if home.mapped_len(home_ws).ok() != Some(0)
            || home.chain_state(home_ws).ok().flatten().is_some()
        {
            COUNTERS.noncanonical.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    }

    let remote_program = match context_extension_program(&program.bound.profile) {
        Ok(program) => program,
        Err(error) => {
            COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(%error, "prefill context-extension program failed");
            return None;
        }
    };
    let surrogate_driver = partner.driver_id();
    let surrogate_program = Arc::clone(&remote_program);
    let surrogate = match tokio::task::spawn_blocking(move || {
        surrogate(surrogate_driver, &surrogate_program, home_instance_id)
    })
    .await
    {
        Ok(Ok(bound)) => bound,
        Ok(Err(error)) => {
            partner.mark_suspect();
            COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(%error, "prefill offload surrogate setup failed");
            return None;
        }
        Err(error) => {
            partner.mark_suspect();
            COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(%error, "prefill offload surrogate task failed");
            return None;
        }
    };

    let remote_stores = crate::store::registry::get(model_idx, partner.driver_id());
    let scratch_ws = {
        let mut remote = remote_stores.kv.lock().unwrap();
        remote.create_working_set()
    };
    let prefix_tokens = &canonical_tokens[..offload_rows];
    let prepared = {
        let mut remote = remote_stores.kv.lock().unwrap();
        crate::pipeline::fire::kv::prepare(
            &mut remote,
            scratch_ws,
            0,
            prefix_tokens,
            page_size,
            Some(prefix_tokens),
        )
    };
    let (projection, copies, translation, txn) = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            release_scratch(model_idx, partner.driver_id(), scratch_ws);
            COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(%error, "prefill offload scratch preparation failed");
            return None;
        }
    };
    if !copies.0.is_empty() || !copies.1.is_empty() {
        let mut remote = remote_stores.kv.lock().unwrap();
        crate::pipeline::fire::kv::abandon(&mut remote, txn);
        drop(remote);
        release_scratch(model_idx, partner.driver_id(), scratch_ws);
        COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    let mut remote_request = request.clone();
    remote_request.token_ids.truncate(offload_rows);
    remote_request.position_ids.truncate(offload_rows);
    remote_request.qo_indptr = vec![0, offload_rows as u32];
    remote_request.kv_page_indices.clear();
    remote_request.kv_page_indptr.clear();
    remote_request.kv_last_page_lens.clear();
    remote_request.kv_len.clear();
    remote_request.kv_len_device.clear();
    remote_request.kv_translation = translation;
    remote_request.sampling_indices.clear();
    remote_request.sampling_indptr = vec![0, 0];
    remote_request.context_ids.clear();
    remote_request.masks.clear();
    remote_request.mask_indptr = vec![0, 0];
    remote_request.single_token_mode = false;
    remote_request.has_user_mask = false;
    remote_request.channel_expected_head.clear();
    remote_request.channel_expected_tail.clear();

    let completion = surrogate.bound.reserve_completion();
    let submit = crate::scheduler::submit_async(
        remote_request,
        partner.driver_id(),
        surrogate.bound.instance_id,
        (0..projection.physical_page_ids.len() as u32).collect(),
        projection.last_page_len,
        None,
        completion.clone(),
    );
    if let Err(error) = submit {
        let mut remote = remote_stores.kv.lock().unwrap();
        crate::pipeline::fire::kv::abandon(&mut remote, txn);
        drop(remote);
        release_scratch(model_idx, partner.driver_id(), scratch_ws);
        partner.mark_suspect();
        COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(%error, "prefill offload submit failed");
        return None;
    }
    let remote_success = completion.await.is_ok();
    {
        let mut remote = remote_stores.kv.lock().unwrap();
        if let Err(error) = crate::pipeline::fire::kv::finalize(&mut remote, txn, remote_success) {
            tracing::debug!(%error, "prefill offload scratch finalize failed");
        }
    }
    if !remote_success {
        release_scratch(model_idx, partner.driver_id(), scratch_ws);
        partner.mark_suspect();
        COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    let destination_pages = {
        let mut home = home_stores.kv.lock().unwrap();
        home.reserve_device_pages(projection.physical_page_ids.len())
    };
    let Some(destination_pages) = destination_pages else {
        release_scratch(model_idx, partner.driver_id(), scratch_ws);
        COUNTERS.transfer_failure.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let destination_ids = destination_pages
        .iter()
        .map(|page| page.0)
        .collect::<Vec<_>>();
    if let Err(error) = partner
        .pull_kv(projection.physical_page_ids.clone(), destination_ids)
        .await
    {
        if partner.transfer_kind() == RemoteTransferKind::Inline {
            home_stores
                .kv
                .lock()
                .unwrap()
                .release_device_reservation(destination_pages);
            release_scratch(model_idx, partner.driver_id(), scratch_ws);
        } else {
            tracing::warn!(
                partner = partner.worker_id(),
                "quarantining source and destination KV pages after uncertain NIXL failure"
            );
        }
        partner.mark_suspect();
        COUNTERS.transfer_failure.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(%error, "prefill offload transfer failed");
        return None;
    }
    let adopted = home_stores.kv.lock().unwrap().adopt_offloaded_prefix(
        home_ws,
        prefix_tokens,
        destination_pages,
        page_size,
    );
    release_scratch(model_idx, partner.driver_id(), scratch_ws);
    if let Err(error) = adopted {
        COUNTERS.transfer_failure.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(%error, "prefill offload adoption failed");
        return None;
    }

    request.token_ids.drain(..offload_rows);
    request.position_ids.drain(..offload_rows);
    request.qo_indptr = vec![0, (count - offload_rows) as u32];
    for index in &mut request.sampling_indices {
        *index -= offload_rows as u32;
    }
    COUNTERS.adopted.fetch_add(1, Ordering::Relaxed);
    Some((
        OffloadAdoption {
            token_count: offload_rows,
        },
        request,
    ))
}

#[cfg(test)]
pub(crate) fn clear_partners() {
    let partners = std::mem::take(&mut *PARTNERS.write().unwrap());
    for partner in partners.into_values() {
        partner.mark_suspect();
    }
    SURROGATES.lock().unwrap().clear();
    HOME_STATES.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use pie_driver_abi::{
        DriverCapabilities, ExecutorResponse, ExecutorRpc, KvDtype, KvHandle, KvLayout,
        KvLayoutKind, KvRegion, PIE_TERMINAL_OUTCOME_SUCCESS, RemoteBindResponse,
        RemoteChannelBinding, RemoteError, RemoteTerminal, ScratchGrant, TerminalCellState,
    };
    use pie_ptir::container::{StageProgram, TraceContainer};
    use pie_ptir::registry::{ModelProfile, Stage};
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU64;
    use tarpc::server::{BaseChannel, Channel};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn selection_respects_cap_and_releases_on_drop() {
        let _lock = TEST_LOCK.lock().unwrap();
        clear_partners();
        let partner = register_partner(
            1,
            2,
            4,
            PartnerRole::Prefill,
            1,
            RemoteTransferKind::Inline,
            None,
        );
        let guard = select_partner(PartnerRole::Prefill).unwrap();
        assert_eq!(guard.partner().driver_id(), 4);
        assert_eq!(partner.outstanding(), 1);
        assert!(select_partner(PartnerRole::Prefill).is_none());
        drop(guard);
        assert!(select_partner(PartnerRole::Prefill).is_some());
        clear_partners();
    }

    #[test]
    fn suspect_partner_is_not_selected() {
        let _lock = TEST_LOCK.lock().unwrap();
        clear_partners();
        let partner = register_partner(
            1,
            2,
            4,
            PartnerRole::Prefill,
            2,
            RemoteTransferKind::Inline,
            None,
        );
        partner.mark_suspect();
        assert!(select_partner(PartnerRole::Prefill).is_none());
        clear_partners();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn partner_drain_waits_for_active_guard() {
        let _lock = TEST_LOCK.lock().unwrap();
        clear_partners();
        let partner = register_partner(
            3,
            4,
            5,
            PartnerRole::Prefill,
            1,
            RemoteTransferKind::Inline,
            None,
        );
        let guard = select_partner(PartnerRole::Prefill).unwrap();
        partner.mark_suspect();
        assert!(
            tokio::time::timeout(Duration::from_millis(5), partner.wait_drained())
                .await
                .is_err()
        );
        drop(guard);
        tokio::time::timeout(Duration::from_secs(1), partner.wait_drained())
            .await
            .unwrap();
        clear_partners();
    }

    #[test]
    fn closed_home_state_is_removed_after_users_drain() {
        let _lock = TEST_LOCK.lock().unwrap();
        clear_partners();
        let usage = HomeUse::acquire(404).unwrap();
        close_home_instance(404);
        assert!(home_instance_closed(404));
        drop(usage);
        assert!(!HOME_STATES.lock().unwrap().contains_key(&404));
        clear_partners();
    }

    #[derive(Clone)]
    struct StubExecutor {
        next_instance: Arc<AtomicU64>,
        page_bytes: u64,
        fail_launch: bool,
        fail_push: bool,
    }

    impl ExecutorRpc for StubExecutor {
        async fn execute(
            self,
            _: tarpc::context::Context,
            request: ExecutorRequest,
        ) -> std::result::Result<ExecutorResponse, RemoteError> {
            Ok(match request {
                ExecutorRequest::LoadedModel => ExecutorResponse::LoadedModel(true),
                ExecutorRequest::RegisterProgram(_) => ExecutorResponse::ProgramRegistered(1),
                ExecutorRequest::RegisterChannel(channel) => {
                    ExecutorResponse::ChannelRegistered(RemoteChannelBinding {
                        local_channel_id: channel.local_channel_id,
                        executor_channel_id: channel.local_channel_id,
                    })
                }
                ExecutorRequest::BindInstance(binding) => {
                    ExecutorResponse::InstanceBound(RemoteBindResponse {
                        local_instance_id: binding.local_instance_id,
                        executor_instance_id: self.next_instance.fetch_add(1, Ordering::Relaxed),
                    })
                }
                ExecutorRequest::Launch(launch) => {
                    if self.fail_launch {
                        return Err(RemoteError::new(
                            pie_driver_abi::RemoteErrorKind::Driver,
                            "injected launch failure",
                        ));
                    }
                    ExecutorResponse::Terminal(RemoteTerminal {
                        per_request: (0..launch.terminal_count)
                            .map(|_| TerminalCellState {
                                outcome: PIE_TERMINAL_OUTCOME_SUCCESS,
                                reserved0: 0,
                            })
                            .collect(),
                    })
                }
                ExecutorRequest::CopyKv(_) => ExecutorResponse::Terminal(RemoteTerminal {
                    per_request: vec![TerminalCellState {
                        outcome: PIE_TERMINAL_OUTCOME_SUCCESS,
                        reserved0: 0,
                    }],
                }),
                ExecutorRequest::Encode(encode) => {
                    if encode.plan.embed_rows.is_empty() {
                        let rows = encode.plan.token_ids.len().max(1);
                        let mut bytes = Vec::with_capacity(rows * 2);
                        for token in
                            encode.plan.token_ids.iter().copied().chain(
                                std::iter::repeat(0).take(rows - encode.plan.token_ids.len()),
                            )
                        {
                            bytes.extend_from_slice(&(token as u16).to_le_bytes());
                        }
                        ExecutorResponse::Embeddings(pie_driver_abi::RemoteEmbeddings {
                            rows: bytes,
                            indptr: vec![0, (rows * 2) as u32],
                            shapes: vec![rows as u32, 1],
                            dtypes: vec![2],
                            anchor_rows: vec![0],
                        })
                    } else {
                        ExecutorResponse::Embeddings(pie_driver_abi::RemoteEmbeddings {
                            rows: encode.plan.embed_rows,
                            indptr: encode.plan.embed_indptr,
                            shapes: encode.plan.embed_shapes,
                            dtypes: encode.plan.embed_dtypes,
                            anchor_rows: encode.plan.embed_anchor_rows,
                        })
                    }
                }
                ExecutorRequest::PushKv(push) => ExecutorResponse::KvPayload(InlineKvPayload {
                    dst_page_ids: if self.fail_push {
                        return Err(RemoteError::new(
                            pie_driver_abi::RemoteErrorKind::Driver,
                            "injected transfer failure",
                        ));
                    } else {
                        push.dst_page_ids
                    },
                    page_bytes: self.page_bytes,
                    bytes: vec![0; push.src_page_ids.len() * self.page_bytes as usize],
                }),
                ExecutorRequest::CloseInstance(_) | ExecutorRequest::CloseChannel(_) => {
                    ExecutorResponse::Closed
                }
                ExecutorRequest::Hello(_) => unreachable!("test registers the paired client"),
            })
        }
    }

    fn test_caps() -> DriverCapabilities {
        DriverCapabilities {
            abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
            total_pages: 8,
            kv_page_size: 16,
            swap_pool_size: 0,
            kv_copy_domain_mask: pie_driver_abi::KV_COPY_DEVICE_TO_DEVICE,
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            has_mtp_logits: true,
            has_mtp_drafts: true,
            has_value_head: true,
            max_forward_tokens: 128,
            max_forward_requests: 8,
            max_page_refs: 128,
            arch_name: "dummy".to_string(),
            vocab_size: 32,
            max_model_len: 128,
            activation_dtype: "f32".to_string(),
            hidden_size: 1,
            supports_media_encode: false,
            snapshot_dir: String::new(),
            kv_handle: None,
        }
    }

    fn empty_registered_program() -> Arc<RegisteredProgram> {
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
        let mut profile = ModelProfile::dummy();
        profile.page_size = 16;
        let mut registry =
            crate::pipeline::program::Registry::new(std::num::NonZeroUsize::new(4).unwrap());
        registry.register(bytes, &profile).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_prefill_transfers_adopts_and_trims() {
        let _lock = TEST_LOCK.lock().unwrap();
        clear_partners();
        configure(true, 1);

        let layout = KvLayout {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            page_size: 16,
            dtype: KvDtype::I8,
            kind: KvLayoutKind::FusedLatent,
            storage_format: "test-i8".to_string(),
            region_page_bytes: Vec::new(),
        };
        let page_bytes = layout.page_bytes();
        let (client_transport, server_transport) = tarpc::transport::channel::unbounded();
        let server_task = tokio::spawn(
            BaseChannel::with_defaults(server_transport)
                .execute(
                    StubExecutor {
                        next_instance: Arc::new(AtomicU64::new(1)),
                        page_bytes,
                        fail_launch: false,
                        fail_push: false,
                    }
                    .serve(),
                )
                .for_each_concurrent(None, |request| async move {
                    tokio::spawn(request);
                }),
        );
        let new_client = pie_driver_abi::ExecutorRpcClient::new(
            tarpc::client::Config::default(),
            client_transport,
        );
        let client = new_client.client;
        tokio::spawn(new_client.dispatch);
        let remote = crate::driver::RemoteDriver::new(
            client.clone(),
            tokio::runtime::Handle::current(),
            test_caps(),
            ScratchGrant {
                base_page: 0,
                num_pages: 8,
            },
        );
        let remote_driver_id = crate::driver::register_driver_backend(
            crate::driver::DriverSpec {
                num_kv_pages: 8,
                limits: crate::driver::SchedulerLimits {
                    max_forward_requests: 8,
                    max_forward_tokens: 128,
                    max_page_refs: 128,
                },
            },
            crate::driver::DriverBackend::Remote(remote),
        );
        let capacities = vec![8; remote_driver_id + 1];
        let model_idx =
            crate::store::registry::register_model(16, &capacities, &vec![0; capacities.len()]);
        crate::scheduler::spawn_driver(remote_driver_id, 16, 5).unwrap();

        let mut home_bytes = vec![0u8; page_bytes as usize * 8].into_boxed_slice();
        set_home_kv_handle(KvHandle {
            regions: vec![KvRegion {
                base: home_bytes.as_mut_ptr() as u64,
                len: home_bytes.len() as u64,
                page_stride: page_bytes,
                domain: MemoryDomain::HostPinned,
            }],
            layout: layout.clone(),
        });
        register_partner(
            99,
            42,
            remote_driver_id,
            PartnerRole::Prefill,
            4,
            RemoteTransferKind::Inline,
            Some(client),
        );

        let home_ws = {
            let stores = crate::store::registry::get(model_idx, 0);
            let id = stores.kv.lock().unwrap().create_working_set();
            id
        };
        let program = empty_registered_program();
        let tokens = (0..33).collect::<Vec<u32>>();
        let mut request = pie_driver_abi::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (0..33).collect(),
            qo_indptr: vec![0, 33],
            sampling_indices: vec![32],
            sampling_indptr: vec![0, 1],
            ..Default::default()
        };
        let adopted = try_prefill(
            model_idx,
            0,
            home_ws,
            16,
            &mut request,
            &tokens,
            &program,
            7,
            None,
        )
        .await
        .expect("offload adopted");
        assert_eq!(adopted.token_count, 32);
        assert_eq!(request.token_ids, vec![32]);
        assert_eq!(request.position_ids, vec![32]);
        assert_eq!(request.sampling_indices, vec![0]);
        assert_eq!(
            crate::store::registry::get(model_idx, 0)
                .kv
                .lock()
                .unwrap()
                .committed_token_len(home_ws, 16)
                .unwrap(),
            32
        );

        remove_partner(99, PartnerRole::Prefill);
        crate::scheduler::stop_driver(remote_driver_id).unwrap();
        crate::driver::unregister_driver(remote_driver_id).unwrap();
        configure(false, 0);
        clear_partners();
        server_task.abort();
        drop(home_bytes);
    }

    async fn assert_failure_falls_back(fail_launch: bool, fail_push: bool) {
        clear_partners();
        configure(true, 1);
        let layout = KvLayout {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            page_size: 16,
            dtype: KvDtype::I8,
            kind: KvLayoutKind::FusedLatent,
            storage_format: "test-i8".to_string(),
            region_page_bytes: Vec::new(),
        };
        let page_bytes = layout.page_bytes();
        let (client_transport, server_transport) = tarpc::transport::channel::unbounded();
        let server_task = tokio::spawn(
            BaseChannel::with_defaults(server_transport)
                .execute(
                    StubExecutor {
                        next_instance: Arc::new(AtomicU64::new(1)),
                        page_bytes,
                        fail_launch,
                        fail_push,
                    }
                    .serve(),
                )
                .for_each_concurrent(None, |request| async move {
                    tokio::spawn(request);
                }),
        );
        let new_client = pie_driver_abi::ExecutorRpcClient::new(
            tarpc::client::Config::default(),
            client_transport,
        );
        let client = new_client.client;
        tokio::spawn(new_client.dispatch);
        let remote = crate::driver::RemoteDriver::new(
            client.clone(),
            tokio::runtime::Handle::current(),
            test_caps(),
            ScratchGrant {
                base_page: 0,
                num_pages: 8,
            },
        );
        let remote_driver_id = crate::driver::register_driver_backend(
            crate::driver::DriverSpec {
                num_kv_pages: 8,
                limits: crate::driver::SchedulerLimits {
                    max_forward_requests: 8,
                    max_forward_tokens: 128,
                    max_page_refs: 128,
                },
            },
            crate::driver::DriverBackend::Remote(remote),
        );
        let capacities = vec![8; remote_driver_id + 1];
        let model_idx =
            crate::store::registry::register_model(16, &capacities, &vec![0; capacities.len()]);
        crate::scheduler::spawn_driver(remote_driver_id, 16, 5).unwrap();
        let mut home_bytes = vec![0u8; page_bytes as usize * 8].into_boxed_slice();
        set_home_kv_handle(KvHandle {
            regions: vec![KvRegion {
                base: home_bytes.as_mut_ptr() as u64,
                len: home_bytes.len() as u64,
                page_stride: page_bytes,
                domain: MemoryDomain::HostPinned,
            }],
            layout,
        });
        register_partner(
            100 + u64::from(fail_push),
            42,
            remote_driver_id,
            PartnerRole::Prefill,
            4,
            RemoteTransferKind::Inline,
            Some(client),
        );
        let home_stores = crate::store::registry::get(model_idx, 0);
        let home_ws = home_stores.kv.lock().unwrap().create_working_set();
        let program = empty_registered_program();
        let tokens = (0..33).collect::<Vec<u32>>();
        let original = pie_driver_abi::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (0..33).collect(),
            qo_indptr: vec![0, 33],
            sampling_indices: vec![32],
            sampling_indptr: vec![0, 1],
            ..Default::default()
        };
        let mut request = original.clone();
        assert!(
            try_prefill(
                model_idx,
                0,
                home_ws,
                16,
                &mut request,
                &tokens,
                &program,
                8,
                None,
            )
            .await
            .is_none()
        );
        assert_eq!(request, original, "fallback must keep the full local plan");
        let home = home_stores.kv.lock().unwrap();
        assert_eq!(home.mapped_len(home_ws).unwrap(), 0);
        assert_eq!(home.available_pages(), 8);
        drop(home);

        remove_partner(100 + u64::from(fail_push), PartnerRole::Prefill);
        crate::scheduler::stop_driver(remote_driver_id).unwrap();
        crate::driver::unregister_driver(remote_driver_id).unwrap();
        configure(false, 0);
        clear_partners();
        server_task.abort();
        drop(home_bytes);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn launch_and_transfer_failures_preserve_local_fallback() {
        let _lock = TEST_LOCK.lock().unwrap();
        assert_failure_falls_back(true, false).await;
        assert_failure_falls_back(false, true).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn encode_partner_rewrites_media_to_embedding_rows() {
        let _lock = TEST_LOCK.lock().unwrap();
        clear_partners();
        configure(true, 1);
        configure_encode_injection(true, 1);
        let (client_transport, server_transport) = tarpc::transport::channel::unbounded();
        let server_task = tokio::spawn(
            BaseChannel::with_defaults(server_transport)
                .execute(
                    StubExecutor {
                        next_instance: Arc::new(AtomicU64::new(1)),
                        page_bytes: 16,
                        fail_launch: false,
                        fail_push: false,
                    }
                    .serve(),
                )
                .for_each_concurrent(None, |request| async move {
                    tokio::spawn(request);
                }),
        );
        let new_client = pie_driver_abi::ExecutorRpcClient::new(
            tarpc::client::Config::default(),
            client_transport,
        );
        let client = new_client.client;
        tokio::spawn(new_client.dispatch);
        register_partner(
            200,
            42,
            99,
            PartnerRole::Encode,
            4,
            RemoteTransferKind::Inline,
            Some(client),
        );
        let mut plan = pie_driver_abi::LaunchPlan {
            token_ids: vec![7, 8],
            image_pixels: vec![1, 2, 3, 4],
            image_anchor_rows: vec![0],
            ..Default::default()
        };
        assert!(try_encode(&mut plan).await);
        assert!(plan.image_pixels.is_empty());
        assert_eq!(plan.embed_shapes, vec![2, 1]);
        assert_eq!(plan.embed_dtypes, vec![2]);
        assert_eq!(plan.embed_anchor_rows, vec![0]);
        assert_eq!(plan.embed_rows.len(), 4);

        remove_partner(200, PartnerRole::Encode);
        configure_encode_injection(false, 0);
        configure(false, 0);
        clear_partners();
        server_task.abort();
    }
}
