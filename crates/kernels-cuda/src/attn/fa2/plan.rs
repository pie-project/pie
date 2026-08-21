use std::ffi::c_void;
use std::sync::OnceLock;

use super::geometry::Device as FaDevice;
use crate::attn::fa2 as lattice;
use crate::attn::plan::info::{DecodePlanInfo, PrefillPlanInfo, PrefillPlanSm90Info};
use crate::attn::plan::{self, Device, Workspace};

use kernels::Refusal;

use crate::jit::PinnedBytes;

use super::dispatch::PrefillDispatch;

#[must_use]
pub enum Planned {
    Full,
    StaticNonsplit,
    Declined(Decline),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    HeadDim { head_dim: u32 },
    HeadDimTile { head_dim: u32, cta_tile_q: u32 },
    NoRequests,
    Planner(&'static str),
    WorkspaceTooSmall { needed: usize, have: usize },
    ScoreCaptureWindow { window_left: i32 },
    Pin,
}

impl core::fmt::Display for Decline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HeadDim { head_dim } => {
                write!(
                    f,
                    "flashinfer fa2: unsupported head_dim {head_dim}; the lattice holds {{64, 128, 256, 512}}"
                )
            }
            Self::HeadDimTile {
                head_dim,
                cta_tile_q,
            } => write!(
                f,
                "flashinfer fa2 prefill: head_dim {head_dim} with CTA_TILE_Q {cta_tile_q} has no \
                 valid KernelTraits -- `IsInvalid()` (prefill.cuh:221-232) is true for every \
                 NUM_MMA_KV, so no unit exists and none can",
            ),
            Self::NoRequests => write!(f, "flashinfer fa2: empty batch"),
            Self::Pin => write!(
                f,
                "flashinfer fa2: the plan descriptor's page-locked buffer could not be taken, and \
                  a pageable one is not capturable",
            ),
            Self::Planner(which) => {
                write!(f, "flashinfer fa2: the {which} planner refused")
            }
            Self::WorkspaceTooSmall { needed, have } => write!(
                f,
                "flashinfer decode static plan: attention int workspace too small -- \
                 {needed} bytes needed, {have} granted",
            ),
            Self::ScoreCaptureWindow { window_left } => write!(
                f,
                "flashinfer prefill plan: score capture is not available with a sliding window \
                 (window_left {window_left}) -- LogitsMask runs after LogitsTransform, so the \
                 captured scores would include positions the kernel discards",
            ),
        }
    }
}

const HEAD_DIMS: [u32; 4] = [64, 128, 256, 512];

#[must_use]
pub fn head_dim_instantiated(head_dim: u32) -> bool {
    HEAD_DIMS.contains(&head_dim)
}

#[must_use]
pub fn force_split_kv_small_enabled() -> bool {
    truthy("PIE_CUDA_FORCE_SPLIT_KV_SMALL")
}

#[must_use]
pub fn window_split_kv_enabled() -> bool {
    truthy("PIE_CUDA_WINDOW_SPLIT_KV")
}

fn truthy(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

#[derive(Debug)]
pub struct DecodePlanCache {
    pub plan_info: DecodePlanInfo,
    pub int_upload: PinnedBytes,
    pub num_requests: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub page_size: i32,
    pub num_pages_in_batch: i32,
    pub enable_pdl: bool,
    pub full_attention_variant: bool,
    pub hnd_layout: bool,
    pub valid: bool,
    pub page_count_independent: bool,
    pub int_base_bytes: usize,
    pub static_nonsplit_num_requests: i32,
    pub static_request_indices: Vec<i32>,
    pub static_kv_tile_indices: Vec<i32>,
    pub static_o_indptr: Vec<i32>,
    pub indptr_h_buf: Vec<i32>,
    /// The int workspace this plan was planned against — the carve the
    /// driver stamps when it raises the cache. Was `keys::AttnWorkspaceInt`.
    pub int_workspace: *mut core::ffi::c_void,
    /// Its float half. Was `keys::AttnWorkspaceFloat`.
    pub float_workspace: *mut core::ffi::c_void,
}

// NOT DERIVED: the two carve pointers have no `Default` of their own, and
// null is exactly what an unraised cache should say there.
impl Default for DecodePlanCache {
    fn default() -> Self {
        Self {
            plan_info: DecodePlanInfo::default(),
            int_upload: PinnedBytes::default(),
            num_requests: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            page_size: 0,
            num_pages_in_batch: 0,
            enable_pdl: false,
            full_attention_variant: false,
            hnd_layout: false,
            valid: false,
            page_count_independent: false,
            int_base_bytes: 0,
            static_nonsplit_num_requests: 0,
            static_request_indices: Vec::new(),
            static_kv_tile_indices: Vec::new(),
            static_o_indptr: Vec::new(),
            indptr_h_buf: Vec::new(),
            int_workspace: core::ptr::null_mut(),
            float_workspace: core::ptr::null_mut(),
        }
    }
}

impl DecodePlanCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_int_base(&mut self, bytes: usize) {
        self.int_base_bytes = bytes;
    }

    #[must_use]
    pub fn can_use_static_nonsplit(
        num_requests: i32,
        cc_major: i32,
        enable_cuda_graph: bool,
    ) -> bool {
        !enable_cuda_graph
            && !force_split_kv_small_enabled()
            && cc_major >= 8
            && num_requests > 0
            && num_requests <= 512
    }

    fn refresh_static_vectors(&mut self, num_requests: i32) {
        if self.static_nonsplit_num_requests == num_requests {
            return;
        }
        self.static_nonsplit_num_requests = num_requests;
        let n = num_requests.max(0) as usize;
        self.static_request_indices.clear();
        self.static_request_indices.extend((0..n).map(|r| r as i32));
        self.static_kv_tile_indices.clear();
        self.static_kv_tile_indices.resize(n, 0);
        self.static_o_indptr.clear();
        self.static_o_indptr.extend((0..=n).map(|r| r as i32));
    }
}

#[derive(Debug)]
pub struct PrefillPlanCache {
    pub plan_info: PrefillPlanInfo,
    pub sm90_plan: Option<PrefillPlanSm90Info>,
    pub int_upload: PinnedBytes,
    pub total_tokens: i32,
    pub num_requests: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub page_size: i32,
    pub window_left: i32,
    pub full_attention_variant: bool,
    pub causal_mask: bool,
    pub hnd_layout: bool,
    pub use_sm90: bool,
    pub enable_pdl: bool,
    pub valid: bool,
    pub graph_capturable: bool,
    pub qo_h_buf: Vec<i32>,
    pub kv_h_buf: Vec<i32>,
    pub cta_tile_q: u32,
    pub int_base_bytes: usize,
    /// The int workspace this plan was planned against — the carve the
    /// driver stamps when it raises the cache. Was
    /// `keys::AttnPrefillWorkspaceInt` on the planned path and
    /// `keys::AttnWorkspaceInt` on the planless one.
    pub int_workspace: *mut core::ffi::c_void,
    /// Its float half. Was `keys::AttnPrefillWorkspaceFloat` /
    /// `keys::AttnWorkspaceFloat`.
    pub float_workspace: *mut core::ffi::c_void,
    /// The int carve's size, which the planless prefill plans against. Was
    /// `keys::AttnWorkspaceIntBytes`.
    pub int_workspace_bytes: usize,
    /// The float carve's size. Was `keys::AttnWorkspaceFloatBytes`.
    pub float_workspace_bytes: usize,
}

// NOT DERIVED: the two carve pointers have no `Default` of their own, and
// null is exactly what an unraised cache should say there.
impl Default for PrefillPlanCache {
    fn default() -> Self {
        Self {
            plan_info: PrefillPlanInfo::default(),
            sm90_plan: None,
            int_upload: PinnedBytes::default(),
            total_tokens: 0,
            num_requests: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            page_size: 0,
            window_left: 0,
            full_attention_variant: false,
            causal_mask: false,
            hnd_layout: false,
            use_sm90: false,
            enable_pdl: false,
            valid: false,
            graph_capturable: false,
            qo_h_buf: Vec::new(),
            kv_h_buf: Vec::new(),
            cta_tile_q: 0,
            int_base_bytes: 0,
            int_workspace: core::ptr::null_mut(),
            float_workspace: core::ptr::null_mut(),
            int_workspace_bytes: 0,
            float_workspace_bytes: 0,
        }
    }
}

impl PrefillPlanCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_int_base(&mut self, bytes: usize) {
        self.int_base_bytes = bytes;
    }
}

fn facts() -> (Device, FaDevice) {
    static FACTS: OnceLock<(Device, FaDevice)> = OnceLock::new();
    *FACTS.get_or_init(|| {
        let Some((num_sm, cc_major, smem_sm, smem_block)) = queried() else {
            return (Device::new(148, 8), FaDevice::L40S);
        };
        (
            Device::new(num_sm.max(1), cc_major.cast_signed()),
            FaDevice {
                cc_major,
                max_smem_per_sm: smem_sm,
                max_smem_per_block_optin: smem_block,
            },
        )
    })
}

#[cfg(feature = "_cuda")]
fn queried() -> Option<(u32, u32, u32, u32)> {
    use crate::jit::device;

    Some((
        device::multiprocessors().ok()?,
        device::compute_capability_major()?,
        device::max_shared_memory_per_sm().ok()?,
        device::max_shared_memory_per_block_optin().ok()?,
    ))
}

#[cfg(not(feature = "_cuda"))]
const fn queried() -> Option<(u32, u32, u32, u32)> {
    None
}

#[must_use]
pub fn plan_device() -> Device {
    facts().0
}

#[must_use]
pub fn fa_device() -> FaDevice {
    facts().1
}

#[must_use]
pub fn decode_max_grid_size(head_dim: i32, num_q_heads: i32, num_kv_heads: i32) -> u32 {
    let (device, fa) = facts();
    let floor = device.num_sm.max(1);
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return floor;
    }
    let group = if num_kv_heads > 0 {
        (num_q_heads / num_kv_heads).max(1)
    } else {
        1
    };
    match lattice::decode_blocks_per_sm(head_dim as u32, group as u32, fa) {
        Some(per_sm) => per_sm.max(1).saturating_mul(floor),
        None => floor,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plan_static_nonsplit_decode(
    cache: &mut DecodePlanCache,
    kv_page_indptr_h: &[u32],
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    workspace: Workspace,
    cc_major: i32,
    enable_cuda_graph: bool,
    full_attention_variant: bool,
    hnd_layout: bool,
) -> Planned {
    fn put_i32s(dst: &mut [u8], at: i64, src: &[i32]) {
        let at = at.max(0) as usize;
        for (i, v) in src.iter().enumerate() {
            let lo = at + i * 4;
            dst[lo..lo + 4].copy_from_slice(&v.to_ne_bytes());
        }
    }

    fn carve(cursor: &mut usize, bytes: usize, alignment: usize) -> i64 {
        *cursor = cursor.next_multiple_of(alignment);
        let offset = *cursor;
        *cursor += bytes;
        offset as i64
    }

    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }
    cache.refresh_static_vectors(num_requests);

    let n = num_requests as usize;

    const ID: usize = 4;
    let mut cursor = 0usize;
    let request_indices_offset = carve(&mut cursor, ID * n, 16);
    let kv_tile_indices_offset = carve(&mut cursor, ID * n, 16);
    let o_indptr_offset = carve(&mut cursor, ID * (n + 1), 16);
    let kv_chunk_size_ptr_offset = carve(&mut cursor, ID, 1);

    let needed = cursor.saturating_add(cache.int_base_bytes);
    if needed > workspace.int_bytes {
        return Planned::Declined(Decline::WorkspaceTooSmall {
            needed,
            have: workspace.int_bytes,
        });
    }

    let mut staging = vec![0u8; cursor];
    put_i32s(
        &mut staging,
        request_indices_offset,
        &cache.static_request_indices,
    );
    put_i32s(
        &mut staging,
        kv_tile_indices_offset,
        &cache.static_kv_tile_indices,
    );
    put_i32s(&mut staging, o_indptr_offset, &cache.static_o_indptr);

    put_i32s(&mut staging, kv_chunk_size_ptr_offset, &[page_size]);
    if cache.int_upload.fill(&staging).is_err() {
        return Planned::Declined(Decline::Pin);
    }

    cache.plan_info = DecodePlanInfo {
        enable_cuda_graph,
        split_kv: false,
        padded_batch_size: i64::from(num_requests),
        request_indices_offset,
        kv_tile_indices_offset,
        o_indptr_offset,
        kv_chunk_size_ptr_offset,
        v_offset: 0,
        s_offset: 0,
        block_valid_mask_offset: 0,
    };

    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.num_pages_in_batch = kv_page_indptr_h
        .get(num_requests as usize)
        .copied()
        .unwrap_or(0) as i32;

    cache.enable_pdl = cc_major >= 9;
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout = hnd_layout;
    cache.page_count_independent = true;
    cache.valid = true;

    Planned::StaticNonsplit
}

#[allow(clippy::too_many_arguments)]
pub fn plan_decode(
    cache: &mut DecodePlanCache,
    kv_page_indptr_h: &[u32],
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    workspace: Workspace,
    device: &Device,
    max_grid_size: u32,
    enable_cuda_graph: bool,
    full_attention_variant: bool,
    hnd_layout: bool,
    window_left: i32,
) -> Planned {
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return Planned::Declined(Decline::HeadDim {
            head_dim: head_dim.max(0) as u32,
        });
    }
    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }

    let windowed_split = window_split_kv_enabled() && window_left >= 0;
    if !windowed_split
        && DecodePlanCache::can_use_static_nonsplit(
            num_requests,
            device.cc_major,
            enable_cuda_graph,
        )
    {
        return plan_static_nonsplit_decode(
            cache,
            kv_page_indptr_h,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            workspace,
            device.cc_major,
            enable_cuda_graph,
            full_attention_variant,
            hnd_layout,
        );
    }

    cache.indptr_h_buf.clear();
    cache.indptr_h_buf.extend(
        kv_page_indptr_h
            .iter()
            .take(num_requests as usize + 1)
            .map(|&v| v as i32),
    );

    let gqa_group_size = if num_kv_heads > 0 {
        num_q_heads / num_kv_heads
    } else {
        0
    };
    let req = plan::decode::Request {
        kv_indptr: &cache.indptr_h_buf,
        batch_size: num_requests as u32,
        num_qo_heads: num_q_heads as u32,
        gqa_group_size: gqa_group_size as u32,
        page_size: page_size as u32,
        head_dim: head_dim as u32,
        enable_cuda_graph,
    };

    let planned = match plan::decode::plan(&req, max_grid_size, workspace) {
        Ok(p) => p,
        Err(_) => return Planned::Declined(Decline::Planner("decode")),
    };

    cache.plan_info = planned.info;
    if cache.int_upload.fill(&planned.int_upload).is_err() {
        return Planned::Declined(Decline::Pin);
    }
    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.num_pages_in_batch = kv_page_indptr_h
        .get(num_requests as usize)
        .copied()
        .unwrap_or(0) as i32;
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout = hnd_layout;

    cache.enable_pdl = device.cc_major >= 9;

    cache.page_count_independent = false;
    cache.valid = true;

    Planned::Full
}

#[allow(clippy::too_many_arguments)]
pub fn plan_prefill(
    cache: &mut PrefillPlanCache,
    qo_indptr_h: &[u32],
    kv_page_indptr_h: &[u32],
    total_tokens: i32,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    workspace: Workspace,
    device: &Device,
    enable_cuda_graph: bool,
    window_left: i32,
    full_attention_variant: bool,
    hnd_layout: bool,
    causal_mask: bool,
    custom_mask: bool,
    wants_prefill_score: bool,
) -> Planned {
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return Planned::Declined(Decline::HeadDim {
            head_dim: head_dim.max(0) as u32,
        });
    }
    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }

    let full_attention_variant = if wants_prefill_score {
        if window_left >= 0 {
            return Planned::Declined(Decline::ScoreCaptureWindow { window_left });
        }
        true
    } else {
        full_attention_variant
    };

    cache.qo_h_buf.clear();
    cache.qo_h_buf.extend(
        qo_indptr_h
            .iter()
            .take(num_requests as usize + 1)
            .map(|&v| v as i32),
    );
    cache.kv_h_buf.clear();
    cache.kv_h_buf.extend(
        kv_page_indptr_h
            .iter()
            .take(num_requests as usize + 1)
            .map(|&v| v as i32),
    );

    let req = plan::prefill::Request {
        qo_indptr: &cache.qo_h_buf,
        kv_indptr: &cache.kv_h_buf,
        total_num_rows: total_tokens.max(0) as u32,
        batch_size: num_requests as u32,
        num_qo_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim_qk: head_dim as u32,
        head_dim_vo: head_dim as u32,
        page_size: page_size as u32,
        enable_cuda_graph,
        sizeof_dtype_o: 2,
        window_left,
        fixed_split_size: -1,
        disable_split_kv: !head_dim_instantiated(head_dim.max(0) as u32),
        num_colocated_ctas: 0,
    };

    let (planned, capturable) = match plan::prefill::plan(&req, device, workspace) {
        Ok(p) => (p, enable_cuda_graph),
        Err(_) if enable_cuda_graph => {
            let req = plan::prefill::Request {
                enable_cuda_graph: false,
                ..req
            };
            match plan::prefill::plan(&req, device, workspace) {
                Ok(p) => (p, false),
                Err(_) => return Planned::Declined(Decline::Planner("prefill")),
            }
        }
        Err(_) => return Planned::Declined(Decline::Planner("prefill")),
    };

    let cta_tile_q = u32::try_from(planned.info.cta_tile_q).unwrap_or(0);

    if head_dim as u32 >= 256 && cta_tile_q == 128 {
        return Planned::Declined(Decline::HeadDimTile {
            head_dim: head_dim as u32,
            cta_tile_q,
        });
    }

    cache.plan_info = planned.info;
    if cache.int_upload.fill(&planned.int_upload).is_err() {
        return Planned::Declined(Decline::Pin);
    }
    cache.total_tokens = total_tokens;
    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.window_left = window_left;
    cache.full_attention_variant = full_attention_variant;
    cache.causal_mask = causal_mask;
    cache.hnd_layout = hnd_layout;
    cache.cta_tile_q = cta_tile_q;

    cache.use_sm90 = false;
    cache.sm90_plan = None;

    cache.graph_capturable = capturable;

    let _ = custom_mask;

    cache.enable_pdl = device.cc_major >= 9;
    cache.valid = true;

    Planned::Full
}

/// # Safety
///
/// `int_buffer + int_base_bytes` must address at least `bytes.len()` bytes
/// of device memory, and `stream` must be a live stream in the current
/// context. Nothing here can check either: the destination arrives as a
/// `u64` precisely because it is not a pointer this side may dereference.
pub unsafe fn upload_int_plan(
    bytes: &[u8],
    int_buffer: u64,
    int_base_bytes: usize,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    if bytes.is_empty() {
        return Ok(());
    }
    let dst = (int_buffer as usize).saturating_add(int_base_bytes) as *mut c_void;
    #[cfg(feature = "_cuda")]
    {
        unsafe { crate::jit::device::upload(dst, bytes, stream) }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (dst, stream);
        Err(Refusal::Device {
            why: "this build selected no CUDA runtime",
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlanUpload<'a> {
    pub bytes: &'a [u8],
    pub int_buffer: u64,
    pub int_base_bytes: usize,
}

/// # Safety
///
/// [`upload_int_plan`]'s, plus `dispatch` must hold device pointers that
/// are still mapped and still describe the geometry the plan was measured
/// against. `stream` must be live for both the upload and the launch.
pub unsafe fn fire_prefill<P: lattice::PrefillBlock>(
    dispatch: &mut PrefillDispatch<P>,
    upload: PlanUpload<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    unsafe {
        upload_int_plan(
            upload.bytes,
            upload.int_buffer,
            upload.int_base_bytes,
            stream,
        )?;
    }

    let ctx = unsafe { crate::jit::Ctx::on(stream) };
    lattice::prefill(&ctx, dispatch.at, &dispatch.params)
}
