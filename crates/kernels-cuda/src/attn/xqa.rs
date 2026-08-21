
use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::by_value;
use core::ffi::c_void;

use crate::jit::{Ctx, Launch, Root};
use crate::jit::abi::MaybeConst;
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::raises::Struct;
use kernels::routine::{Const, In, Out};
use crate::views::KvCache;

pub use crate::jit::abi::DevicePtr;
use kernels::Ty;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct KvCacheList {

    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub page_list: DevicePtr,
    pub seq_len_list: DevicePtr,
    pub max_pages_per_seq: u32,
}

by_value! {
    KvCacheList as "KVCacheList<true>",
    tag = KvCacheLayerView,
    probe = "nvrtc-probes/xqa_kvcachelist.py",
    size = 40, align = 8,
    {
        k_cache           @ 0  as "kCacheVLLM",
        v_cache           @ 8  as "vCacheVLLM",
        page_list         @ 16 as "kvCachePageList",
        seq_len_list      @ 24 as "seqLenList",
        max_pages_per_seq @ 32 as "maxNbPagesPerSeq",
    }
}

impl KvCacheList {

    #[must_use]
    pub const fn paged(
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        page_list: DevicePtr,
        seq_len_list: DevicePtr,
        max_pages_per_seq: u32,
    ) -> Self {
        Self { k_cache, v_cache, page_list, seq_len_list, max_pages_per_seq }
    }
}

pub static LAYOUTS: &[crate::jit::Layout] = &[<KvCacheList as crate::jit::ByValue>::LAYOUT];

pub enum XqaIoHead {}

impl crate::jit::Abi for *const XqaIoHead {
    const CPP: &'static str = "const IOHead*";
    const TY: Ty = Ty::Buf;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Ptr(*self as *mut c_void)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::Ptr(p) => Ok(p.cast::<XqaIoHead>().cast_const()),
            _ => Err(kernels::Refusal::Kind { at, want: Ty::Buf }),
        }
    }
}

impl kernels::Elem for XqaIoHead {

    type Read = *const XqaIoHead;
    type Write = *mut XqaIoHead;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {

        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {

        unsafe { write.add(elems) }
    }

    const CPP_CONST: &'static str = "const IOHead*";
    const CPP_MUT: &'static str = "OutputHead*";
    const TY_CONST: Ty = Ty::Buf;
    const TY_MUT: Ty = Ty::BufMut;
}

crate::arg_via_abi!(*const XqaIoHead);

impl crate::jit::Abi for *mut XqaIoHead {
    const CPP: &'static str = "OutputHead*";
    const TY: Ty = Ty::BufMut;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Ptr(self.cast::<c_void>())
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::Ptr(p) => Ok(p.cast::<XqaIoHead>()),
            _ => Err(kernels::Refusal::Kind { at, want: Ty::BufMut }),
        }
    }
}

crate::arg_via_abi!(*mut XqaIoHead);

pub struct XqaVariant {

    pub unit: &'static str,
    pub options: &'static [&'static str],
    pub entry: &'static str,
    pub from: &'static str,
    pub because: &'static str,
}

pub const XQA_COMMON_OPTIONS: &[&str] = &[
    "-DGENERATE_CUBIN=1",
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DUSE_INPUT_KV=0",
    "-DUSE_CUSTOM_BARRIER=1",
    "-DINPUT_FP16=0",
    "-DDTYPE=pie::bf16",
    "-DCACHE_ELEM_ENUM=0",
    "-DHEAD_ELEMS=128",
    "-DSLIDING_WINDOW=0",
    "-DLOW_PREC_OUTPUT=0",
    "-DSPEC_DEC=0",
    "-DMLA_WRAPPER=0",
    "--device-as-default-execution-space",
];

pub const XQA_LATTICE: [XqaVariant; 6] = [
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa2_p32",
        options: &[
            "-DHEAD_GRP_SIZE=2",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa2_bf16_p32_h128",
        from: "attn/attention_xqa_gqa2.cu:24-33",
        because: "head_group_size=2, used by small Qwen GQA models such as \
                  Qwen3-0.6B and Qwen3-1.7B",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa2_p16",
        options: &[
            "-DHEAD_GRP_SIZE=2",
            "-DTOKENS_PER_PAGE=16",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p16_h128",
        ],
        entry: "kernel_mha_xqa_gqa2_bf16_p16_h128",
        from: "attn/attention_xqa_gqa2_p16.cu:24-33",
        because: "the same head group at a 16-token page. Dead code TODAY and \
                  kept anyway: `xqa_decode_page_bucket` never returns 16 \
                  because `xqa_gqa2_page16_enabled()` returns false, so the \
                  only way to reach this member is to flip that — which is \
                  what it is for. Deleting it would make flipping the flag a \
                  port rather than a flag.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa4_p32",
        options: &[
            "-DHEAD_GRP_SIZE=4",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa4_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa4_bf16_p32_h128",
        from: "attn/attention_xqa_gqa4.cu:24-33",
        because: "head_group_size=4, used by medium Qwen GQA models such as Qwen3-4B and Qwen3-8B",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa5_p32",
        options: &[
            "-DHEAD_GRP_SIZE=5",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa5_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa5_bf16_p32_h128",
        from: "attn/attention_xqa.cu:65-74",
        because: "head_group_size=5, the ratio Llama-3.1-8B-shaped models use \
                  (32 query heads over 8 KV heads is 4; 40 over 8 is 5). It \
                  lives in the family's dispatch head rather than a sibling \
                  file because that file is also where the host program was.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa8_p32",
        options: &[
            "-DHEAD_GRP_SIZE=8",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa8_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa8_bf16_p32_h128",
        from: "attn/attention_xqa_gqa8.cu:24-33",
        because: "head_group_size=8, used by common large GQA models such as \
                  Qwen3-32B and Llama-70B-style shapes. Its launcher \
                  FORWARDS to the sm90 member when `current_device_major() \
                  >= 9`, which is why the two exist as a pair rather than as \
                  alternatives.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa8_p32_sm90",
        options: &[
            "-DHEAD_GRP_SIZE=8",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=1",
            "-Dkernel_mha=kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        from: "attn/attention_xqa_gqa8_sm90.cu:26-35",
        because: "Hopper GMMA/TMA, kept in a separate translation unit \
                  because FlashInfer's `xqa/mha.cu` and `xqa/mha_sm90.cu` \
                  intentionally define the same static kernel symbols. It \
                  also passes `enable_pdl = true` unconditionally where the \
                  other five pass `current_device_major() >= 9` — which is \
                  the same predicate, already known true here. NOT READY: \
                  measured at compute_90a it stops on `std::pair` in DEVICE \
                  code (`xqa/mha_sm90.cu:1980`, 12 diagnostics cascading \
                  from that one line), the header set has no `<utility>`, \
                  `shim/cuda.h` has no `CUtensorMap` or \
                  `CUtensorMapDataType_enum` for `xqa/tensorMap.h` to \
                  declare against, and the archive unit compiles \
                  `<xqa/tensorMap.cpp>` first — HOST code building tensor \
                  maps through `cuTensorMapEncodeTiled`, which is a second \
                  and larger host-to-Rust port than `launchMHA` was.",
    },
];

pub const XQA_ROOT: &str = crate::source::carried("attn/attention_xqa_mha.cuh");

pub const XQA_SMEM_BYTES: u32 = 79_488;

const fn options_of(member: usize) -> [&'static str; 18] {
    let extra = XQA_LATTICE[member].options;
    let mut out = [""; 18];
    let mut i = 0;
    while i < XQA_COMMON_OPTIONS.len() {
        out[i] = XQA_COMMON_OPTIONS[i];
        i += 1;
    }
    let mut j = 0;
    while j < extra.len() {
        out[XQA_COMMON_OPTIONS.len() + j] = extra[j];
        j += 1;
    }
    out
}

const OPTIONS_GQA2_P32: [&str; 18] = options_of(0);
const OPTIONS_GQA2_P16: [&str; 18] = options_of(1);
const OPTIONS_GQA4_P32: [&str; 18] = options_of(2);
const OPTIONS_GQA5_P32: [&str; 18] = options_of(3);
const OPTIONS_GQA8_P32: [&str; 18] = options_of(4);

const OPTION_SETS: [&[&str]; 5] =
    [&OPTIONS_GQA2_P32, &OPTIONS_GQA2_P16, &OPTIONS_GQA4_P32, &OPTIONS_GQA5_P32, &OPTIONS_GQA8_P32];

pub static OPTIONS: [&[&str]; 5] = OPTION_SETS;

const fn mha_root(member: usize) -> Root {
    Root::variant(XQA_LATTICE[member].unit, "attn/attention_xqa_mha.cuh")
        .options(OPTION_SETS[member])
        .upstream()
}

pub static ROOT_GQA2_P32: Root = mha_root(0);

pub static ROOT_GQA2_P16: Root = mha_root(1);

pub static ROOT_GQA4_P32: Root = mha_root(2);

pub static ROOT_GQA5_P32: Root = mha_root(3);

pub static ROOT_GQA8_P32: Root = mha_root(4);

pub static ROOTS: [&Root; 5] =
    [&ROOT_GQA2_P32, &ROOT_GQA2_P16, &ROOT_GQA4_P32, &ROOT_GQA5_P32, &ROOT_GQA8_P32];

pub mod inst {

    pub const MHA: [&str; 5] = [
        super::XQA_LATTICE[0].entry,
        super::XQA_LATTICE[1].entry,
        super::XQA_LATTICE[2].entry,
        super::XQA_LATTICE[3].entry,
        super::XQA_LATTICE[4].entry,
    ];
}

pub const XQA_HEAD_DIM: i32 = 128;

pub const XQA_PAGE_SIZE: i32 = 32;

pub const XQA_BLOCK: [u32; 3] = [128, 1, 2];

const MAX_PAGE_BUCKET: i32 = 4096;

#[must_use]
pub const fn page_bucket(max_pages_per_seq: i32) -> i32 {
    let mut bucket = 1i32;
    let pages = if max_pages_per_seq > 1 { max_pages_per_seq } else { 1 };
    while bucket < pages && bucket < MAX_PAGE_BUCKET {
        bucket <<= 1;
    }
    bucket
}

#[must_use]
pub const fn gqa2_page16_enabled() -> bool {
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XqaMember {

    Gqa2Page32,
    Gqa2Page16,
    Gqa4Page32,
    Gqa5Page32,
    Gqa8Page32,
    Gqa8Page32Sm90,
}

impl XqaMember {

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Gqa2Page32 => 0,
            Self::Gqa2Page16 => 1,
            Self::Gqa4Page32 => 2,
            Self::Gqa5Page32 => 3,
            Self::Gqa8Page32 => 4,
            Self::Gqa8Page32Sm90 => 5,
        }
    }

    #[must_use]
    pub fn entry(self) -> &'static str {
        XQA_LATTICE[self.index()].entry
    }

    #[must_use]
    pub fn enrolled_at(self) -> Option<usize> {
        let at = self.index();
        if at < ROOTS.len() { Some(at) } else { None }
    }

    #[must_use]
    pub const fn dispatch(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        match (head_group_ratio, page_size) {
            (2, 32) => Some(Self::Gqa2Page32),
            (2, 16) => Some(Self::Gqa2Page16),
            (4, 32) => Some(Self::Gqa4Page32),
            (5, 32) => Some(Self::Gqa5Page32),
            (8, 32) if major >= 9 => Some(Self::Gqa8Page32Sm90),
            (8, 32) => Some(Self::Gqa8Page32),
            _ => None,
        }
    }

    #[must_use]
    pub fn pick(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        Self::dispatch(head_group_ratio, page_size, major).filter(|m| m.enrolled_at().is_some())
    }
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn decode_supported(
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    major: i32,
) -> bool {
    if num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0 {
        return false;
    }
    let ratio = num_q_heads / num_kv_heads;

    if XqaMember::pick(ratio, XQA_PAGE_SIZE, major).is_none() {
        return false;
    }
    let page_supported =
        page_size == XQA_PAGE_SIZE || (ratio == 2 && page_size == 16 && gqa2_page16_enabled());
    if head_dim != XQA_HEAD_DIM || !page_supported {
        return false;
    }
    if window_left >= 0 || logits_soft_cap > 0.0 {
        return false;
    }
    #[allow(clippy::cast_precision_loss)]
    let default_scale = 1.0f32 / (head_dim as f32).sqrt();
    if sm_scale > 0.0 && (sm_scale - default_scale).abs() > 1.0e-6 {
        return false;
    }
    major >= 9
}

const SCRATCH_ALIGN: usize = 256;

#[derive(Debug, Clone, Copy)]
struct Carve {

    page_table: *mut i32,
    seq_lens: *mut u32,
    scratch: *mut c_void,
    bucket: i32,
}

fn carve(
    float_buffer: *mut c_void,
    float_bytes: usize,
    num_requests: i32,
    max_pages_per_seq: i32,
) -> Result<Carve, Refusal> {
    if float_buffer.is_null() {
        return Err(Refusal::Null { what: "the attention workspace's float buffer" });
    }
    let bucket = page_bucket(max_pages_per_seq);
    let requests = num_requests.unsigned_abs() as usize;
    let page_table_bytes = requests * bucket.unsigned_abs() as usize * size_of::<i32>();
    let seq_lens_bytes = requests * size_of::<u32>();

    let base = float_buffer.addr();
    let p_page_table = base.next_multiple_of(align_of::<i32>());
    let p_seq_lens = (p_page_table + page_table_bytes).next_multiple_of(align_of::<u32>());
    let p_scratch = (p_seq_lens + seq_lens_bytes).next_multiple_of(SCRATCH_ALIGN);
    if p_scratch >= base + float_bytes {
        return Err(Refusal::Wide {
            what: "the XQA workspace carve's scratch offset",
            at: i64::try_from(p_scratch - base).unwrap_or(i64::MAX),
            max: i64::try_from(float_bytes).unwrap_or(i64::MAX).saturating_sub(1),
        });
    }

    Ok(Carve {
        page_table: float_buffer.with_addr(p_page_table).cast(),
        seq_lens: float_buffer.with_addr(p_seq_lens).cast(),
        scratch: float_buffer.with_addr(p_scratch),
        bucket,
    })
}

#[cfg(feature = "_cuda")]
unsafe fn zero_on_stream(ctx: &Ctx<'_>, at: *mut c_void, bytes: usize) -> Result<(), Refusal> {
    use cudarc::runtime::sys::{cudaError, cudaMemsetAsync};

    let code = unsafe { cudaMemsetAsync(at, 0, bytes, ctx.stream().cast()) };
    if code == cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Refusal::Device { why: "the XQA semaphore bank could not be zeroed" })
    }
}

#[cfg(not(feature = "_cuda"))]
unsafe fn zero_on_stream(_ctx: &Ctx<'_>, _at: *mut c_void, _bytes: usize) -> Result<(), Refusal> {
    Err(Refusal::Device { why: "this build selected no CUDA runtime" })
}

#[allow(clippy::too_many_arguments)]
pub fn build_xqa_metadata(
    ctx: &Ctx<'_>,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    page_table: *mut i32,
    seq_lens: *mut u32,
    num_requests: i32,
    max_pages_per_seq: i32,
    page_size: i32) -> Result<(), Refusal> {

    const METADATA_BLOCK: u32 = 128;

    let bucket = page_bucket(max_pages_per_seq);

    ctx.fire(Fire::at("attn/attention_xqa.cuh", "::pie::attn::build_xqa_metadata").apply(Launch::per_row(num_requests.unsigned_abs(), METADATA_BLOCK)), &[
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_table.arg(),
                seq_lens.arg(),
                num_requests.arg(),
                bucket.arg(),
                page_size.arg(),
            ])
}

#[routine(whole)]
pub fn attention_xqa_decode_bf16_prepared(
    ctx: &Ctx<'_>,
    q: In<Tensor<XqaIoHead>>,
    o: Out<Tensor<XqaIoHead>>,
    num_q_heads: Const<i32>,
    num_kv_heads: Const<i32>,
    head_dim: Const<i32>,
    kvc: In<Struct<KvCache>>,
    sm_scale: Const<f32>,
    float_bytes: Const<usize>,
    int_bytes: Const<usize>,
    num_requests: Const<i32>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let num_q_heads = *num_q_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;
    let page_size = kvc.page_size;
    let max_pages_per_seq = kvc.max_pages_per_request;
    let sm_scale = *sm_scale;
    // The XQA carve is launch-local and its sizes are stated by the
    // statement, so the workspace is named scratch rather than a driver
    // answer (was `keys::AttnWorkspaceFloat` / `keys::AttnWorkspaceInt`).
    let float_bytes = usize::try_from(*float_bytes).unwrap_or(usize::MAX);
    let float_buffer = ctx.scratch("attn::xqa_workspace_float", float_bytes)?;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let int_bytes = usize::try_from(*int_bytes).unwrap_or(usize::MAX);
    let int_buffer = ctx.scratch("attn::xqa_workspace_int", int_bytes)?;
    let num_requests = *num_requests;

    let (k_pages, v_pages) = (k_pages, v_pages);
    let (kv_page_indices, kv_page_indptr, kv_last_page_lens) =
        (kv_page_indices, kv_page_indptr, kv_last_page_lens);
    let (float_buffer, float_bytes) = (float_buffer, float_bytes);
    let (int_buffer, int_bytes) = (int_buffer, int_bytes);

    const XQA_CTA_TILE_X: u32 = 256;

    let Some(major) = ctx.compute_capability_major() else {
        return Err(Refusal::Device { why: "the device would not say its compute capability" });
    };
    let major = i32::try_from(major).unwrap_or(0);

    if !decode_supported(
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        -1,
        0.0,
        sm_scale,
        major,
    ) {
        return Err(Refusal::Device { why: "XQA serves neither this shape nor this device" });
    }
    let ratio = num_q_heads / num_kv_heads;
    let Some(member) = XqaMember::dispatch(ratio, page_size, major) else {
        return Err(Refusal::Device { why: "XQA serves neither this shape nor this device" });
    };
    let Some(root_at) = member.enrolled_at() else {
        return Err(Refusal::Absent { what: "a root for the Hopper XQA body" });
    };

    if kv_page_indices.is_null() || kv_page_indptr.is_null() || kv_last_page_lens.is_null() {
        return Err(Refusal::Null { what: "the fire's paged-KV CSR" });
    }

    let batch = num_requests.unsigned_abs();
    let kv_heads = num_kv_heads.unsigned_abs();

    let regions = carve(float_buffer, float_bytes, num_requests, max_pages_per_seq)?;
    let bucket = regions.bucket;
    let max_seq_len = bucket.unsigned_abs() * page_size.unsigned_abs();

    if int_buffer.is_null() {
        return Err(Refusal::Null { what: "the attention workspace's int buffer" });
    }
    let semaphores: *mut u32 = int_buffer.cast();
    let semaphore_bytes = batch as usize * kv_heads as usize * core::mem::size_of::<u32>();
    if semaphore_bytes > int_bytes {
        return Err(Refusal::Wide {
            what: "the XQA semaphore bank",
            at: i64::try_from(semaphore_bytes).unwrap_or(i64::MAX),
            max: i64::try_from(int_bytes).unwrap_or(i64::MAX),
        });
    }

    let multiprocessors = ctx.multiprocessors()?;

    build_xqa_metadata(
        ctx,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        regions.page_table,
        regions.seq_lens,
        num_requests,
        max_pages_per_seq,
        page_size,
    )?;

    unsafe {
        zero_on_stream(ctx, semaphores.cast(), semaphore_bytes)?;
    }

    let afford = (multiprocessors / (batch * kv_heads)).max(1);
    let sub_seqs = afford.min(max_seq_len.div_ceil(XQA_CTA_TILE_X));

    let per_head = u64::from(head_dim.unsigned_abs());
    let in_heads = |elements: u64| u32::try_from(elements / per_head).unwrap_or(u32::MAX);
    let pages = u64::from(page_size.unsigned_abs());
    let stride_page = in_heads(pages * u64::from(kv_heads) * per_head);
    let stride_token = in_heads(u64::from(kv_heads) * per_head);
    let stride_head = in_heads(per_head);

    let cache = KvCacheList::paged(
        k_pages as DevicePtr,
        v_pages as DevicePtr,
        regions.page_table as DevicePtr,
        regions.seq_lens as DevicePtr,
        bucket.unsigned_abs(),
    );

    ctx.fire(
        Fire::at(ROOTS[root_at].file, inst::MHA[root_at])
            .unit(ROOTS[root_at].name)
            .apply(Launch::grid([sub_seqs, kv_heads, batch], XQA_BLOCK).smem(XQA_SMEM_BYTES)),
        &[
                kv_heads.arg(),
                1.0f32.arg(),
                MaybeConst::<f32>::none().arg(),
                o.arg(),
                q.arg(),
                MaybeConst::<f32>::none().arg(),
                cache.arg(),
                batch.arg(),
                1.0f32.arg(),
                MaybeConst::<f32>::none().arg(),
                stride_page.arg(),
                stride_token.arg(),
                stride_head.arg(),
                semaphores.arg(),
                regions.scratch.arg(),
        ],
    )
}
