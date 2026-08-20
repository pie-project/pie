//! FlashInfer's XQA decode: five roots over one text, and the one routine
//! that reaches them. Comments citing `attention_xqa*.cu` quote a deleted
//! archive file frozen at `0dc8e9e9b`; no line numbers, since none resolve.
//!
//! [`build_xqa_metadata`] and the decode share one function and one stream,
//! in that order: the decode reads the page table the build just wrote.
//! [`carve`] places both, plus XQA's own scratch, in the caller's attention
//! workspace rather than `Ctx::scratch`, since XQA sizes its scratch on the
//! device and a captured graph can't survive a scratch moving.
//!
//! The five roots below are one text (`attn/attention_xqa_mha.cuh`) compiled
//! five ways under different `-D`s ([`Root::options`]). The JIT's
//! `--fmad=false` (the archive passed none) makes it stricter, not bit-exact.

use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::by_value;
use core::ffi::c_void;

use crate::jit::{Ctx, Launch, Root};
use crate::jit::abi::MaybeConst;
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::keys;
use kernels::routine::{Asks, In, Out};

/// A device address held as an opaque word.
pub use crate::jit::abi::DevicePtr;
use kernels::Ty;

/// `KVCacheList<true>` — XQA's paged KV cache descriptor, passed by value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct KvCacheList {
    /// `kCacheVLLM` — `GMemCacheHead*`, the K pages.
    pub k_cache: DevicePtr,
    /// `vCacheVLLM` — `GMemCacheHead*`, the V pages.
    pub v_cache: DevicePtr,
    /// `kvCachePageList` — `KVCachePageIndex const*`, shape
    /// `[batchSize][beamWidth][maxNbPagesPerSeq]`.
    pub page_list: DevicePtr,
    /// `seqLenList` — `SeqLenDataType const*`, shape `[batchSize][beamWidth]`.
    pub seq_len_list: DevicePtr,
    /// `maxNbPagesPerSeq` — the third stride of `page_list`, in pages.
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
    /// The descriptor for one paged KV cache.
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

/// The by-value aggregates the XQA units pass, for the typecheck TU.
pub static LAYOUTS: &[crate::jit::Layout] = &[<KvCacheList as crate::jit::ByValue>::LAYOUT];

/// `IOHead` and `OutputHead` — XQA's per-head vector, as a pointee.
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

// `const IOHead*` reads, `OutputHead*` writes — an asymmetry `ptr_abi!` has no
// arm for, so `Abi` is written by hand; `Elem` restates it for `In`/`Out`.
impl kernels::Elem for XqaIoHead {
    // THE ASYMMETRY IS IN THE CARRIERS TOO: a read is a `const IOHead*` and a
    // write an `OutputHead*`, which is why this pair is written out rather
    // than stamped.
    type Read = *const XqaIoHead;
    type Write = *mut XqaIoHead;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        // SAFETY: the trait's obligation, forwarded to the caller.
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        // SAFETY: as above.
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

/// One member of the XQA lattice: a `-D` set, and what it is for.
pub struct XqaVariant {
    /// The `Unit::name` this member gets.
    pub unit: &'static str,
    /// The `-D` set, verbatim, as `Unit::options` would carry it.
    pub options: &'static [&'static str],
    /// The `extern "C"` device entry point this member exports, after its
    /// `-Dkernel_mha=…` rename.
    pub entry: &'static str,
    /// The archive file this member's `#define` block came from.
    pub from: &'static str,
    /// Why this member exists — the measurement its `.cu` carried.
    pub because: &'static str,
}

/// The `-D`s (and one non-`-D` compiler flag) every lattice member shares.
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

/// The six units, as option sets.
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

/// The root all six members compile, carried so a moved file is a build
/// error rather than a silent miss.
pub const XQA_ROOT: &str = crate::source::carried("attn/attention_xqa_mha.cuh");

/// `sizeof(SharedMem)` in `xqa/mha.cu`, measured out of the PTX.
pub const XQA_SMEM_BYTES: u32 = 79_488;

/// One member's full `-D` set: the options every member shares, then its own.
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

const _: () = assert!(
    XQA_COMMON_OPTIONS.len() == 14,
    "XQA_COMMON_OPTIONS changed width; `options_of`'s array must change with it",
);
const _: () = {
    let mut i = 0;
    while i < XQA_LATTICE.len() {
        assert!(
            XQA_LATTICE[i].options.len() == 4,
            "an XQA lattice member states a number of `-D`s `options_of` was not sized for",
        );
        i += 1;
    }
};

const OPTIONS_GQA2_P32: [&str; 18] = options_of(0);
const OPTIONS_GQA2_P16: [&str; 18] = options_of(1);
const OPTIONS_GQA4_P32: [&str; 18] = options_of(2);
const OPTIONS_GQA5_P32: [&str; 18] = options_of(3);
const OPTIONS_GQA8_P32: [&str; 18] = options_of(4);

/// [`OPTIONS`], in `const` so that [`mha_root`] can read it: a `const fn` may
/// not read a `static`, and the five roots are built by one.
const OPTION_SETS: [&[&str]; 5] =
    [&OPTIONS_GQA2_P32, &OPTIONS_GQA2_P16, &OPTIONS_GQA4_P32, &OPTIONS_GQA5_P32, &OPTIONS_GQA8_P32];

/// The five option sets, indexed as `XQA_LATTICE` is.
///
/// [`Root::key`] must span `Root::options`, or a cache-key collision would
/// hand one head-group size the cubin compiled for another.
pub static OPTIONS: [&[&str]; 5] = OPTION_SETS;

/// One lattice member's root: [`XQA_ROOT`] under that member's `-D` set.
///
/// `.upstream()` because the header closure `xqa/mha.cuh` pulls in is listed
/// under `source::UPSTREAM`, not named here file by file.
const fn mha_root(member: usize) -> Root {
    Root::variant(XQA_LATTICE[member].unit, "attn/attention_xqa_mha.cuh")
        .options(OPTION_SETS[member])
        .upstream()
}

/// `attn/attention_xqa_gqa2.cu`'s member: head group 2, 32-token pages.
pub static ROOT_GQA2_P32: Root = mha_root(0);
/// The same head group at a 16-token page — built, and selected by nothing
/// while [`gqa2_page16_enabled`] answers `false`.
pub static ROOT_GQA2_P16: Root = mha_root(1);
/// Head group 4, 32-token pages.
pub static ROOT_GQA4_P32: Root = mha_root(2);
/// Head group 5, 32-token pages.
pub static ROOT_GQA5_P32: Root = mha_root(3);
/// Head group 8, 32-token pages — the Ampere/Ada body.
pub static ROOT_GQA8_P32: Root = mha_root(4);

/// The five roots, indexed as [`XQA_LATTICE`] is.
///
/// One entry short of the lattice's six: `XQA_LATTICE[5]` is the Hopper body
/// (it does not compile; see its `because`), so [`XqaMember::enrolled_at`]
/// checks this array's length rather than a hand-kept flag.
pub static ROOTS: [&Root; 5] =
    [&ROOT_GQA2_P32, &ROOT_GQA2_P16, &ROOT_GQA4_P32, &ROOT_GQA5_P32, &ROOT_GQA8_P32];

/// The instantiations NVRTC is handed, spelled as it is handed them.
pub mod inst {
    /// The five decode entries, indexed as [`XQA_LATTICE`] is.
    ///
    /// Plain `extern "C"` names, not template-ids: `-Dkernel_mha=…` renames
    /// the same entry point per member.
    ///
    /// [`XQA_LATTICE`]: super::XQA_LATTICE
    pub const MHA: [&str; 5] = [
        super::XQA_LATTICE[0].entry,
        super::XQA_LATTICE[1].entry,
        super::XQA_LATTICE[2].entry,
        super::XQA_LATTICE[3].entry,
        super::XQA_LATTICE[4].entry,
    ];
}

/// `attention_xqa.cu`'s `kXqaHeadDim` — the only head width the lattice is
/// instantiated at (`-DHEAD_ELEMS=128`).
pub const XQA_HEAD_DIM: i32 = 128;

/// `attention_xqa.cu`'s `kXqaPageSize` — the page size five of the six
/// lattice members are built for (`-DTOKENS_PER_PAGE=32`).
pub const XQA_PAGE_SIZE: i32 = 32;

/// `xqa/mha.cu`'s `__launch_bounds__(256, nbCtaPerSM)`, derived rather than
/// read off a `<<<>>>` (the block there is a named `dim3`, invisible to
/// anything parsing between `<<<` and `>>>`).
///
/// `ctaShapeInWarps = {4, 1, 2}` and `warp_size = 32` give
/// `dimCta = {32 * 4, 1, 2}` = `(128, 1, 2)`, 256 threads — matching the
/// kernel's own `__launch_bounds__`.
pub const XQA_BLOCK: [u32; 3] = [128, 1, 2];

/// The largest page-table row stride the bucket will grow to —
/// `attention_xqa.cu`'s `bucket < 4096` clamp.
const MAX_PAGE_BUCKET: i32 = 4096;

/// The dense page table's row stride — `attention_xqa.cu`'s page-bucket loop,
/// transcribed.
///
/// A power of two so it stays stable across the small per-step growth a
/// decode causes, which a captured graph's baked addresses cannot survive.
/// Loop rather than [`usize::next_power_of_two`]: they disagree at the clamp
/// (4097 would round to 8192; this clamps to 4096).
#[must_use]
pub const fn page_bucket(max_pages_per_seq: i32) -> i32 {
    let mut bucket = 1i32;
    let pages = if max_pages_per_seq > 1 { max_pages_per_seq } else { 1 };
    while bucket < pages && bucket < MAX_PAGE_BUCKET {
        bucket <<= 1;
    }
    bucket
}

/// `attention_xqa.cu`'s `xqa_gqa2_page16_enabled()` — always `false`, so the
/// lattice's 16-token-page member is built but never selected. Kept as a
/// literal rather than deleted so flipping it back on stays a flag, not a
/// port.
#[must_use]
pub const fn gqa2_page16_enabled() -> bool {
    false
}

/// Which member of the lattice a shape selects.
///
/// The archive spelled this as an `if`/`else if` chain over
/// `head_group_ratio`; as a closed enum, a lattice member added later can't
/// go unmatched without the compiler noticing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XqaMember {
    /// `-DHEAD_GRP_SIZE=2 -DTOKENS_PER_PAGE=32`.
    Gqa2Page32,
    /// `-DHEAD_GRP_SIZE=2 -DTOKENS_PER_PAGE=16`.
    ///
    /// Unreachable today: [`gqa2_page16_enabled`] is `false`, so
    /// [`decode_supported`] refuses page 16 before this can be picked.
    Gqa2Page16,
    /// `-DHEAD_GRP_SIZE=4 -DTOKENS_PER_PAGE=32`.
    Gqa4Page32,
    /// `-DHEAD_GRP_SIZE=5 -DTOKENS_PER_PAGE=32`.
    Gqa5Page32,
    /// `-DHEAD_GRP_SIZE=8 -DTOKENS_PER_PAGE=32`, the Ampere/Ada body.
    Gqa8Page32,
    /// `-DHEAD_GRP_SIZE=8 -DTOKENS_PER_PAGE=32 -DUSE_SM90_MHA=1`.
    ///
    /// `attention_xqa_gqa8.cu` forwards here whenever `major >= 9`, so
    /// [`XqaMember::Gqa8Page32`] never runs on a device XQA serves — and no
    /// root hosts this variant either, so ratio 8 has no XQA decode at all
    /// until one is enrolled.
    Gqa8Page32Sm90,
}

impl XqaMember {
    /// This member's position in [`XQA_LATTICE`], and therefore in [`ROOTS`]
    /// and in [`inst::MHA`] when it has one.
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

    /// The `extern "C"` device entry this member exports, after the
    /// `-Dkernel_mha=…` rename [`XQA_LATTICE`] carries.
    ///
    /// The archive needed the rename to link six same-named `static` kernels
    /// into one archive; NVRTC needs it for a different reason now — the five
    /// roots are one text, and the instantiation string is what tells one
    /// compile of it apart from another.
    #[must_use]
    pub fn entry(self) -> &'static str {
        XQA_LATTICE[self.index()].entry
    }

    /// Where this member's root is, or `None` if nothing hosts it.
    ///
    /// Not `const`: a `const fn` may not read a `static`, and asking
    /// [`ROOTS`] its length is what costs the constness.
    #[must_use]
    pub fn enrolled_at(self) -> Option<usize> {
        let at = self.index();
        if at < ROOTS.len() { Some(at) } else { None }
    }

    /// The archive's dispatch chain, faithfully — which member a shape
    /// selects, with no question asked about whether it can run.
    ///
    /// Kept separate from [`XqaMember::pick`] on purpose: this is the record
    /// to check the deleted C++ against, `pick` is what a fire may act on.
    #[must_use]
    pub const fn dispatch(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        match (head_group_ratio, page_size) {
            (2, 32) => Some(Self::Gqa2Page32),
            (2, 16) => Some(Self::Gqa2Page16),
            (4, 32) => Some(Self::Gqa4Page32),
            (5, 32) => Some(Self::Gqa5Page32),
            // `attention_xqa_gqa8.cu` forwards to sm90 on Hopper+, else runs its own.
            (8, 32) if major >= 9 => Some(Self::Gqa8Page32Sm90),
            (8, 32) => Some(Self::Gqa8Page32),
            _ => None,
        }
    }

    /// The member this shape selects and that something can host — `None`
    /// for a shape outside the lattice or for a member with no root.
    ///
    /// Deliberately does not fall back to [`XqaMember::Gqa8Page32`] (the
    /// Ampere/Ada body, which does have a root): the archive never ran it
    /// above `major >= 9`, so that fallback would silently swap kernels on
    /// exactly the devices XQA is enabled for. Consequence: ratio 8 has no
    /// XQA decode at all until a Hopper root is enrolled; ratios 2, 4 and 5
    /// are unaffected.
    #[must_use]
    pub fn pick(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        Self::dispatch(head_group_ratio, page_size, major).filter(|m| m.enrolled_at().is_some())
    }
}

/// `attention_xqa.cu`'s `xqa_decode_bf16_supported`, transcribed.
///
/// The SM90 floor is a deployment measurement, not a capability one: the
/// non-Hopper lattice members compile clean, but FlashInfer's own wrapper
/// keeps them off SM89 serving because runs could spin indefinitely after
/// graph capture. Do not lower this bound just because something compiles.
///
/// The scale check is an equality test wearing a tolerance, not an
/// application: XQA folds `1/sqrt(head_dim)` into the kernel itself, so a
/// caller-supplied `sm_scale` is only checked against that default (1e-6);
/// `sm_scale <= 0` means "unset" and passes.
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
    // `xqa_ratio_supported` is the lattice's own membership test; asking
    // `XqaMember::pick` is the same question with one fewer place to be wrong.
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

/// The scratch's alignment — `attention_xqa.cu`'s `kSemaphoreAlignment`.
///
/// Aligns the tail of the carve (XQA's own scratch); named for the
/// semaphores because the decode also zeroes a 256-aligned semaphore bank
/// out of the workspace's int half. Only the alignment is shared, not the
/// buffer.
const SCRATCH_ALIGN: usize = 256;

/// Where XQA's three sub-buffers sit inside the attention workspace's float
/// half.
///
/// Every field is a device address. Held together in one value because the
/// three are one layout: computing any of them without the others is how the
/// archive's five copies of this arithmetic got to disagree.
#[derive(Debug, Clone, Copy)]
struct Carve {
    /// `num_requests * page_bucket` signed page indices, zero padded.
    page_table: *mut i32,
    /// One `u32` per request.
    seq_lens: *mut u32,
    /// Everything after them, 256-byte aligned — XQA's own scratch, whose
    /// extent no host expression names (see this module's header).
    scratch: *mut c_void,
    /// The row stride [`page_bucket`] chose: the table's stride, and — times
    /// `page_size` — XQA's maximum sequence length. Carried rather than
    /// recomputed so the number the carve sized for and the number the
    /// kernel strides by can't drift apart.
    bucket: i32,
}

/// The workspace carve — `attention_xqa.cu`'s carve arithmetic, with
/// [`usize::next_multiple_of`] standing in for its hand-rolled `align_up`.
///
/// The overflow check is `>=`, not `>`: a scratch starting at the last byte
/// of the workspace is empty, and the C++ refused that too.
///
/// # Errors
///
/// [`Refusal::Null`] for a workspace with no float half — a case the C++
/// never tested, which otherwise launches against address 0 and faults
/// asynchronously elsewhere. [`Refusal::Wide`] if the three regions don't
/// fit, naming the offset the scratch would start at against the last one a
/// non-empty scratch could take.
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

/// `cudaMemsetAsync(at, 0, bytes)` on this context's stream.
///
/// The semaphore bank is a multi-block rendezvous: every CTA of a split
/// sequence bumps its `(request, kv head)` slot and the last one merges, so
/// a bank that didn't start at zero makes a CTA believe a merge already
/// happened. Must be zeroed on the stream that runs the kernel, before it.
///
/// # Safety
///
/// `at` must address `bytes` of live device memory, and this context's stream
/// must be live across the call.
#[cfg(feature = "_cuda")]
unsafe fn zero_on_stream(ctx: &Ctx<'_>, at: *mut c_void, bytes: usize) -> Result<(), Refusal> {
    use cudarc::runtime::sys::{cudaError, cudaMemsetAsync};

    // SAFETY: the caller's obligation, forwarded.
    let code = unsafe { cudaMemsetAsync(at, 0, bytes, ctx.stream().cast()) };
    if code == cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Refusal::Device { why: "the XQA semaphore bank could not be zeroed" })
    }
}

/// See the `_cuda` twin.
///
/// # Safety
///
/// None to discharge: this build has no runtime to memset through.
#[cfg(not(feature = "_cuda"))]
unsafe fn zero_on_stream(_ctx: &Ctx<'_>, _at: *mut c_void, _bytes: usize) -> Result<(), Refusal> {
    Err(Refusal::Device { why: "this build selected no CUDA runtime" })
}

/// XQA's dense page table and sequence lengths, built from the paged KV
/// cache's ragged CSR.
///
/// Not a routine or trace symbol — see the module header — so it stays out
/// of [`ROUTINES`]; [`attention_xqa_decode_bf16_prepared`] issues it as the
/// first of its two same-stream launches. `max_pages_per_seq` is the
/// caller's number; the kernel wants [`page_bucket`] of it, the row stride
/// the caller's number rounds up to.
///
/// `call()`'s contract: the three CSR pointers, `page_table`
/// (`num_requests * page_bucket(..)` `i32`s) and `seq_lens` (`num_requests`
/// `u32`s) are live/writable device arrays of those extents.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty request set — a zero `grid.x` refuses
/// rather than launching — and whatever the compile, load or launch refuses.
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
    /// The metadata build's block width — `attention_xqa.cu`'s literal `128`.
    ///
    /// Not a derived value: the page loop strides by `blockDim.x` and the
    /// sequence-length write is gated on `threadIdx.x == 0`, so any width
    /// computes the same result. This one is just a citation.
    const METADATA_BLOCK: u32 = 128;

    let bucket = page_bucket(max_pages_per_seq);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
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

/// `attn::attention_xqa_decode_bf16_prepared` — the paged decode, and the
/// page table it reads.
///
/// One body for all five lattice members, with the member chosen rather than
/// compiled in. `_prepared` names this launch's own first step:
/// [`build_xqa_metadata`] writes the page table [`carve`] laid out, and this
/// function reads it back on the same stream, right after.
///
/// The multi-block split mirrors `xqa/mha.cu`'s grid: CTAs affordable per
/// `(request, kv head)` versus CTAs needed to cover `maxSeqLen`, smaller
/// wins. `maxSeqLen` is the *bucketed* length ([`page_bucket`] `* page_size`),
/// not the caller's raw one, so the table's stride and the kernel's notion
/// of the sequence length stay one number.
///
/// Not reproduced: `cudaPeekAtLastError` (would blame the wrong kernel for
/// an unrelated async fault) and `enable_pdl` ([`Launch`] has no field for
/// it). [`XQA_SMEM_BYTES`] exceeds the default 48 KiB cap but needs no
/// explicit `cudaFuncSetAttribute`; `jit::launch::issue` raises the
/// per-`(device, entry point)` cap when `Launch::smem` exceeds it.
///
/// `call()`'s contract: `q`/`o` address `num_requests * num_q_heads` live
/// heads each, `k_pages`/`v_pages` the layer's page arena, the CSR pointers
/// `num_requests`-shaped arrays, and `float_buffer`/`int_buffer` the
/// workspace's two halves at their stated extents.
///
/// # Errors
///
/// [`Refusal::Device`]/[`Refusal::Absent`] for an unsupported shape, device,
/// or unenrolled lattice member; [`Refusal::Empty`] for an empty request
/// set; [`Refusal::Null`]/[`Refusal::Wide`] for a missing or overflowing
/// workspace half; and whatever the compile, load or launch refuses — all
/// raised before the first launch, so a refusal enqueues nothing.
#[routine(whole)]
pub fn attention_xqa_decode_bf16_prepared(
    ctx: &Ctx<'_>,
    q: In<Tensor<XqaIoHead>>,
    o: Out<Tensor<XqaIoHead>>) -> Result<(), Refusal> {
    // BACK TO ASKS: THE POOL AND THE HEAD COUNTS ARE THE FIRE'S TABLE. Every
    // one was `Env<keys::*>` at HEAD and `driver-cuda` answers all six. A
    // `Const` mark promises the STATEMENT carries the number, and a trace text
    // never sees a deployment -- so `attention_xqa_decode` stated an EMPTY run
    // and this routine could not fire at all.
    //
    // The two head counts are adjacent and both plain per-fire numbers; a GQA
    // shape is exactly where they diverge, so a swap would launch the wrong
    // kernel and nothing about the grid would say so.
    let num_q_heads = ctx.ask::<i32, keys::NumQHeads>()?;
    let num_kv_heads = ctx.ask::<i32, keys::KvNumHeads>()?;
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;
    let max_pages_per_seq = ctx.ask::<i32, keys::KvMaxPagesPerRequest>()?;
    let sm_scale = ctx.ask::<f32, keys::SmScale>()?;
    let float_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceFloat>()?;
    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let float_bytes = ctx.ask::<usize, keys::AttnWorkspaceFloatBytes>()?;
    let int_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()?;
    let int_bytes = ctx.ask::<usize, keys::AttnWorkspaceIntBytes>()?;
    let num_requests = ctx.ask::<i32, keys::RequestCount>()?;
    // Unwrapped once, up front: `**` is `Env`'s two `Deref` hops to the fact
    // inside it.
    let (k_pages, v_pages) = (k_pages, v_pages);
    let (kv_page_indices, kv_page_indptr, kv_last_page_lens) =
        (kv_page_indices, kv_page_indptr, kv_last_page_lens);
    let (float_buffer, float_bytes) = (float_buffer, float_bytes);
    let (int_buffer, int_bytes) = (int_buffer, int_bytes);

    /// `xqa/mha.cu`'s `ctaTile.x` — the sequence-length step one CTA covers:
    /// `warpTile.x (64) * ctaShapeInWarps.x (4)` = 256, the denominator in
    /// the multi-block split below.
    const XQA_CTA_TILE_X: u32 = 256;

    let Some(major) = ctx.compute_capability_major() else {
        return Err(Refusal::Device { why: "the device would not say its compute capability" });
    };
    let major = i32::try_from(major).unwrap_or(0);
    // `window_left`/`logits_soft_cap` are pinned off (`-1`/`0.0`) rather than
    // threaded through: this launcher never offered them. The next four
    // arguments are same-typed and positional — a swap compiles clean and
    // silently checks the wrong shape.
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

    // The semaphore bank is `num_requests * num_kv_heads` `u32`s in the
    // workspace's int half; checked before anything is enqueued.
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

    // SAFETY: the budget check above is what makes this extent right.
    unsafe {
        zero_on_stream(ctx, semaphores.cast(), semaphore_bytes)?;
    }

    let afford = (multiprocessors / (batch * kv_heads)).max(1);
    let sub_seqs = afford.min(max_seq_len.div_ceil(XQA_CTA_TILE_X));

    // Strides are in *heads*, not bytes or elements: `in_heads` divides the
    // element product by `per_head` so page/token/head strides share one unit.
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

    // `xqa/mha.cu`'s parameter list under this option set: `SPEC_DEC=0`,
    // `SLIDING_WINDOW=0`, `LOW_PREC_OUTPUT=0`, `BEAM_WIDTH=1` and no 4-bit KV
    // cache each drop a group of operands from the archive's full signature.
    //
    // SAFETY: `call()`'s contract — every pointer here addresses live device
    // memory of the extent the kernel reads it as; `cache` outlives the launch.
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
