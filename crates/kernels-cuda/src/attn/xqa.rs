//! FlashInfer's XQA decode: five roots over one text, and the one routine
//! that reaches them.
//!
//! # Every `attention_xqa*.cu:NNN` citation below is a line in a DELETED file
//!
//! Said once here rather than forty times inline. The six archive translation
//! units this file is the host half of —
//!
//! ```text
//!   kernels/attn/attention_xqa.cu           436 lines
//!   kernels/attn/attention_xqa_gqa2.cu      194
//!   kernels/attn/attention_xqa_gqa2_p16.cu  194
//!   kernels/attn/attention_xqa_gqa4.cu      194
//!   kernels/attn/attention_xqa_gqa8.cu      244
//!   kernels/attn/attention_xqa_gqa8_sm90.cu 197
//! ```
//!
//! — are **deleted**, along with `kernels/attn/attention_flashinfer_hopper_stub.cpp`,
//! which was chained to them. Their line numbers were correct against the last
//! commit that contained them, `0dc8e9e9b`, and `git show
//! 0dc8e9e9b:crates/kernels-cuda/kernels/attn/attention_xqa.cu` — the
//! archive crate's tree at that revision — is how a reader checks one.
//!
//! The citations are kept rather than stripped because each one is evidence
//! for a decision, and a decision whose evidence has been deleted is a
//! decision no one can re-check. They are **quotations from history**, not
//! pointers into the tree; nothing automated should try to resolve them.
//!
//! # ONE routine and TWO launches, which is how the order is stated
//!
//! `attn::build_xqa_metadata` is not a symbol a trace can state and never
//! was: no `dsl::cuda` builder emits it, no lowering names it, and
//! [`build_xqa_metadata`] has no row in any table. What a trace states is
//! `attn::attention_xqa_decode_bf16_prepared` — ONE statement — and the dense
//! page table it reads has to have been written before it runs.
//!
//! That dependency used to be an obligation written in the table,
//! `Prepare::FireWide`, and discharged by nobody. It was weaker than it
//! looked: the variant is constructed at NO site in this workspace, the row
//! in `not_yet_crossed.rs` left `needs` at `Prepare::None`, and the field's
//! one reader is a `{:?}` inside a `model-ir` diagnostic. **So the
//! order is stated the only way a routine can state one: both launches are
//! issued by [`attention_xqa_decode_bf16_prepared`], on `ctx.stream()`, in
//! that order.** `sample::lm_head_gemv_argmax_int8` and
//! `x::attn::compact_page_csr` are the same shape — a routine whose second
//! launch reads what its first wrote.
//!
//! ## What that costs, stated rather than discovered
//!
//! The build is a function of the fire's page CSR alone, and the CSR is
//! fire-wide, so every LAYER now recomputes the same bytes: one 128-thread
//! block per request, per layer, where a fire-wide discharge would have paid
//! it once. That is the price of the ordering being structural instead of
//! stated, and it is the cheap side of the trade — the alternative on offer
//! is a decode reading whatever the workspace held from the last fire, which
//! is a plausible answer rather than a missing one.
//!
//! # The carve is NOT `Ctx::scratch`, and the reason is a device fact
//!
//! [`carve`] lays the page table, the sequence lengths and XQA's own scratch
//! end to end inside the caller's attention workspace. [`Ctx::scratch`] is
//! this crate's growable named device scratch and is the obvious home for
//! three buffers nothing outside XQA reads. It is wrong for all three:
//!
//! * **XQA's `scratch` has no host-computable size.** It is segmented on the
//!   DEVICE — `xqa/mha.cuh:2605`'s `MemSegmenter{scratch}` over
//!   `SMemWarpRowMax` and `Array2D<LdGrain, …>` — out of types no host
//!   expression in this tree names, which is why *"everything after the
//!   other two"* is the only extent it has ever been given.
//!   `Ctx::scratch(name, bytes)` needs `bytes`.
//! * **Growing a scratch is `cudaFree` + `cudaMalloc`**, and neither is legal
//!   inside a stream capture — which is exactly where `bind::run_captured`
//!   fires this. A growth is also a NEW ADDRESS, and [`page_bucket`]'s whole
//!   reason for being a power of two is that a captured graph bakes these
//!   addresses and must not see them move.
//! * The attention workspace is arena-stable and the driver's memory planner
//!   already sized it. Three process-lifetime allocations beside it would be
//!   memory nothing counted.
//!
//! **The carve is the one thing here that can be silently wrong**, and the
//! answer to that is not to move it: it is that one [`carve`] call now serves
//! both launches, inside one function, where the archive had five copies of
//! the arithmetic in four files joined by nothing a compiler checks.
//!
//! # THE JIT'D XQA DOES NOT MATCH THE ARCHIVE BIT FOR BIT
//!
//! `runtime::nvrtc::options` passes `--fmad=false --prec-div=true
//! --prec-sqrt=true` on every compile and the archive's CMake passed none of
//! the three, so nvcc built these same instantiations under `--fmad=true`: it
//! contracts multiply-adds and the JIT refuses to. The JIT is the stricter of
//! the two and the direction of the disagreement is known, but a
//! bit-exactness claim would be false by construction (`new-horizon.md`
//! §62.8).
//!
//! # The `-D` lattice, and what it costs
//!
//! `attn/attention_xqa_mha.cuh` is `#include "xqa/mha.cuh"` and two lines
//! of ours. Everything that varies across the lattice is a `-D`, so the five
//! roots below are the SAME text five times and are told apart by
//! [`Root::options`] alone — including `-Dkernel_mha=…`, which is why the five
//! entry points have five names and why [`inst::MHA`] is a plain identifier
//! rather than a template-id.

use crate::by_value;
use core::ffi::c_void;

use crate::jit::{Ctx, Family, Launch, Root, Routine};
use crate::routine;
use crate::jit::Abi;
use crate::jit::abi::MaybeConst;
use kernels::Refusal;

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
    /// The `extern "C"` device entry point this member exports, after the
    pub entry: &'static str,
    /// The archive file this member's `#define` block came from, with the
    pub from: &'static str,
    /// Why this member exists — the measurement its `.cu` carried.
    pub because: &'static str,
}

/// The twelve `-D`s every member of the lattice shares, plus the one that is
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
    // `mha_stdheaders.cuh`, `utils.h` and `mha.h` carry the device-side half
    // of a `std` -- `move`, `forward`, `numeric_limits`, tuple compare -- with
    // no execution-space annotations, because upstream compiled them with
    // `nvcc`, which defaults them to `__host__ __device__`. NVRTC's JIT mode
    // defaults them to `__host__` and refuses. This is the flag NVRTC itself
    // names in the diagnostic, and `attn/attention_mla_fa2` already carries it.
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

/// The root all six members compile, carried so a moved file is a compile
pub const XQA_ROOT: &str = crate::source::carried("attn/attention_xqa_mha.cuh");

/// `sizeof(SharedMem)` (`xqa/mha.cu:409`), measured out of the PTX.
pub const XQA_SMEM_BYTES: u32 = 79_488;

/// One member's `-D` set: the thirteen shared with every other, then the
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
/// XQA compiles ONE root five ways by `-D` alone, so these are what tell the
/// five apart in a cache key — [`Root::key`] spans `Root::options` for exactly
/// that reason, and a key that did not would hand `HEAD_GRP_SIZE=8` the cubin
/// `HEAD_GRP_SIZE=2` compiled.
pub static OPTIONS: [&[&str]; 5] = OPTION_SETS;

// ===========================================================================
// The roots: one prepare, and one text compiled five ways
// ===========================================================================

/// One lattice member's root: [`XQA_ROOT`] under that member's `-D` set.
///
/// `.upstream()` on every one of them. `attention_xqa_mha.cuh`'s body is
/// `#include "xqa/mha.cuh"`, and the fifteen-file closure below it is
/// carried as UPSTREAM — `source::UPSTREAM` is what `attn/flashinfer/` and
/// `attn/xqa/` are listed in — so the library set answers not one name of it.
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
/// Five entries against the lattice's six, and the gap is the whole of what
/// [`XqaMember::enrolled_at`] answers: `XQA_LATTICE[5]` is the Hopper body,
/// which does not compile (see its `because`) and therefore has no root here.
/// The mirror is an INDEX rather than a hand-kept `bool` list, so a sixth root
/// landing at the end of this array is the whole re-enablement.
pub static ROOTS: [&Root; 5] =
    [&ROOT_GQA2_P32, &ROOT_GQA2_P16, &ROOT_GQA4_P32, &ROOT_GQA5_P32, &ROOT_GQA8_P32];

/// The instantiations NVRTC is handed, spelled as it is handed them.
pub mod inst {
    /// The five decode entries, indexed as [`XQA_LATTICE`] is.
    ///
    /// **Not template-ids, and that is XQA's own doing.** `kernel_mha` is
    /// `extern "C"` (`xqa/mha.h:273`'s `CUBIN_EXPORT`, under
    /// `GENERATE_CUBIN`), so what NVRTC lowers is a plain entry name. The five
    /// differ from one another only because `-Dkernel_mha=…` renames it per
    /// member — which is why the rename survived the port even though there is
    /// no linker left to collide in.
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

// ===========================================================================
// The host program: what the archive's six `.cu` files decided
// ===========================================================================

/// `attention_xqa.cu:198`'s `kXqaHeadDim`, and the only head width the lattice
/// is instantiated at (`-DHEAD_ELEMS=128`).
pub const XQA_HEAD_DIM: i32 = 128;

/// `attention_xqa.cu:198`'s `kXqaPageSize` — the page size five of the six
/// lattice members are built for (`-DTOKENS_PER_PAGE=32`).
pub const XQA_PAGE_SIZE: i32 = 32;

/// `xqa/mha.cu:2760`'s `__launch_bounds__(256, nbCtaPerSM)`, derived rather
/// than read off a `<<<>>>`.
///
/// `launchMHAFlashInfer` declares its block as a NAMED `dim3`, which is
/// invisible to anything parsing the text between `<<<` and `>>>`:
///
/// ```text
/// xqa/mha.cu:76        ctaShapeInWarps = {4, 1, 2}
/// xqa/utils.cuh:256    warp_size = 32
/// xqa/mha.cu:~2999     dim3 dimCta{warp_size * ctaShapeInWarps.x,
///                                  ctaShapeInWarps.y, ctaShapeInWarps.z}
/// ```
///
/// so `{32 * 4, 1, 2}` = **(128, 1, 2)**, 256 threads, which is what the
/// `__launch_bounds__` on the kernel independently says.
pub const XQA_BLOCK: [u32; 3] = [128, 1, 2];

/// The largest page-table row stride the bucket will grow to —
/// `attention_xqa.cu:277`'s `bucket < 4096`.
const MAX_PAGE_BUCKET: i32 = 4096;

/// The dense page table's row stride — `attention_xqa.cu:274-279`, verbatim.
///
/// ```text
/// int bucket = 1;
/// const int pages = std::max(1, max_pages_per_seq);
/// while (bucket < pages && bucket < 4096) bucket <<= 1;
/// return bucket;
/// ```
///
/// **The stride is a power of two on purpose.** It has to be stable across the
/// small per-step changes in `max_pages_per_seq` that a growing decode
/// produces, because the decode hands XQA `page_bucket * page_size` as the
/// maximum sequence length and re-shaping the buffer every fire would
/// invalidate a captured graph's baked addresses.
///
/// The clamp is on `bucket`, not on `pages`, so a request set wanting more
/// than 4096 pages gets 4096 and the decode's own shape checks are what refuse
/// it. Transcribed as a loop rather than as `next_power_of_two` because the
/// two differ at the clamp: `4097usize.next_power_of_two()` is 8192, and this
/// answers 4096.
#[must_use]
pub const fn page_bucket(max_pages_per_seq: i32) -> i32 {
    let mut bucket = 1i32;
    let pages = if max_pages_per_seq > 1 { max_pages_per_seq } else { 1 };
    while bucket < pages && bucket < MAX_PAGE_BUCKET {
        bucket <<= 1;
    }
    bucket
}

/// `attention_xqa.cu:227`'s `xqa_gqa2_page16_enabled()`.
///
/// `false`, and it has always been `false`. The 16-token-page member of the
/// lattice is built and never selected; the flag is the switch that would
/// select it. Transcribed rather than dropped because a constant `false` in
/// Rust is the same claim the C++ made and deleting it would turn flipping a
/// flag back into a port.
#[must_use]
pub const fn gqa2_page16_enabled() -> bool {
    false
}

/// Which member of the lattice a shape selects.
///
/// The archive spelled this as five `detail::launch_attention_xqa_decode_bf16_*`
/// declarations and an `if`/`else if` chain over `head_group_ratio`. It is an
/// enum here because the answer is one of a closed set, and the C++ shape let
/// a sixth arm be added without anyone noticing the lattice had grown.
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
    /// `attention_xqa_gqa8.cu` forwarded to this when
    /// `current_device_major() >= 9`, so on every device [`decode_supported`]
    /// admits, [`XqaMember::Gqa8Page32`] is in fact never the one that runs.
    ///
    /// **This is the one member nothing can host**, and [`XqaMember::pick`] no
    /// longer answers it: [`ROOTS`] has five entries and this is the sixth
    /// lattice position, so [`XqaMember::enrolled_at`] is `None` BY
    /// CONSTRUCTION rather than by a hand-kept list.
    ///
    /// # Why the variant survives a body it does not have
    ///
    /// Deleting it was the other option and it is worse three ways, all of
    /// them about what the enum is FOR:
    ///
    /// 1. **It would not answer the dispatch question, only hide it.**
    ///    [`XqaMember::dispatch`] must still say something for `(8, 32)` at
    ///    `major >= 9`. Without this variant the only spellings are
    ///    `Some(Gqa8Page32)` — a silent fallback to a different kernel on
    ///    exactly the devices XQA runs on — or `None`, which erases the fact
    ///    that Hopper had a distinct body at all.
    /// 2. **The enum mirrors the lattice, and the mirror is the check.**
    ///    [`XQA_LATTICE`] has six members and this has six variants; five
    ///    variants against six lattice entries would leave the gap expressed
    ///    nowhere in Rust.
    /// 3. **[`XqaMember::entry`] is the only Rust statement of the symbol.**
    ///    `kernel_mha_xqa_gqa8_sm90_bf16_p32_h128` is what a re-enablement has
    ///    to export.
    ///
    /// A variant that reads as a supported configuration IS a hazard — the
    /// answer is to make it unreachable by construction and say so at every
    /// place a reader arrives, not to delete the record.
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
    /// The rename is not decoration. The archive renamed the HOST launcher six
    /// ways because six translation units defining the same `static` kernel
    /// symbols were about to be linked into one archive; under NVRTC each root
    /// is its own module, but the five roots here are ONE text and the
    /// instantiation string is what tells a compile of it apart from the other
    /// four. The collision moved; it did not go away.
    #[must_use]
    pub fn entry(self) -> &'static str {
        XQA_LATTICE[self.index()].entry
    }

    /// Where this member's root is, or `None` if nothing hosts it.
    ///
    /// **Flipping this to `Some` is the whole re-enablement**: a sixth root
    /// appended to [`ROOTS`] makes this answer for the Hopper body and nothing
    /// else changes. [`XqaMember::pick`] reads it and nothing else does.
    ///
    /// Not `const`, and the reason is the mirror: a `const fn` may not read a
    /// `static`, so asking [`ROOTS`] how many roots there are is what costs
    /// the constness. A hand-written `5` here would buy it back and would be
    /// the second statement of a number that has exactly one.
    #[must_use]
    pub fn enrolled_at(self) -> Option<usize> {
        let at = self.index();
        if at < ROOTS.len() { Some(at) } else { None }
    }

    /// The archive's dispatch chain, faithfully — which member a shape
    /// SELECTS, with no question asked about whether it can run.
    ///
    /// `attention_xqa.cu:290-436`'s `if`/`else if` over `head_group_ratio`,
    /// and `attention_xqa_gqa8.cu:96`'s forward to the Hopper body. `major` is
    /// a parameter rather than a query because that forward is a dispatch
    /// decision, not a device fact this function should be reaching out for.
    ///
    /// **Separate from [`XqaMember::pick`] on purpose.** This is the RECORD of
    /// what the archive did and is what a reader checks the deleted C++
    /// against; `pick` is what a fire may act on. Collapsing them would mean
    /// either losing the record or letting the record be fired.
    #[must_use]
    pub const fn dispatch(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        match (head_group_ratio, page_size) {
            (2, 32) => Some(Self::Gqa2Page32),
            (2, 16) => Some(Self::Gqa2Page16),
            (4, 32) => Some(Self::Gqa4Page32),
            (5, 32) => Some(Self::Gqa5Page32),
            // `attention_xqa_gqa8.cu`: the gqa8 launcher forwards to the sm90
            // body on Hopper and above, and runs its own below.
            (8, 32) if major >= 9 => Some(Self::Gqa8Page32Sm90),
            (8, 32) => Some(Self::Gqa8Page32),
            _ => None,
        }
    }

    /// The member this shape selects AND that something can host — `None` for
    /// a shape outside the lattice and `None` for a member with no root.
    ///
    /// # What it does NOT do, and why that is the point
    ///
    /// It does **not** fall back to [`XqaMember::Gqa8Page32`], the Ampere/Ada
    /// body, which has a root and would run. That was the tempting edit and it
    /// is the wrong one twice over. The archive never ran that body above
    /// `major >= 9`, so choosing it is a silent change of kernel on exactly
    /// the devices XQA is enabled for; and `attention_xqa.cu:237-240` (quoted
    /// in [`decode_supported`]) records that the non-Hopper instantiations
    /// *compile* but were kept off SM89 serving because runs could spin after
    /// graph capture. A fallback would be a wrong answer that looks like a
    /// supported configuration.
    ///
    /// # The consequence, stated so nobody discovers it
    ///
    /// [`decode_supported`] ends in `major >= 9`, so XQA decode is offered
    /// ONLY on Hopper and above; this refuses `ratio == 8` there. **Head-group
    /// ratio 8 therefore has no XQA decode at all**, on any device, until a
    /// Hopper root is enrolled. Ratios 2, 4 and 5 are unaffected.
    #[must_use]
    pub fn pick(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        Self::dispatch(head_group_ratio, page_size, major).filter(|m| m.enrolled_at().is_some())
    }
}

/// `attention_xqa.cu:217-242`'s `xqa_decode_bf16_supported`, transcribed.
///
/// ```text
/// if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return false;
/// const int ratio = num_q_heads / num_kv_heads;
/// if (!xqa_ratio_supported(ratio)) return false;
/// const bool page_supported =
///     page_size == kXqaPageSize ||
///     (ratio == 2 && page_size == 16 && xqa_gqa2_page16_enabled());
/// if (head_dim != kXqaHeadDim || !page_supported) return false;
/// if (window_left >= 0 || logits_soft_cap > 0.f) return false;
/// const float default_scale = 1.0f / std::sqrt((float)head_dim);
/// if (sm_scale > 0.f && std::abs(sm_scale - default_scale) > 1.0e-6f) return false;
/// return current_device_major() >= 9;
/// ```
///
/// # The SM90 floor is a deployment measurement, not a capability one
///
/// `attention_xqa.cu:237-240`, verbatim, because it is the sentence that stops
/// someone lowering the bound after finding that the code compiles:
///
/// > FlashInfer's public XQA wrapper only enables this path on SM90+. The
/// > Ampere/Ada csrc instantiations compile, but local SM89 TP2 serving runs
/// > can spin indefinitely after graph capture, so keep those devices on the
/// > regular FlashInfer decode path.
///
/// The NVRTC port re-confirms the first clause from the other side: every
/// non-Hopper member of the lattice compiles clean at `compute_89`, 0 errors.
/// *Compiling* was never the question.
///
/// # The scale test is an equality test wearing a tolerance
///
/// XQA folds `1/sqrt(head_dim)` into the kernel and takes `qScale` as a
/// separate multiplier that the launcher passes as `1.0`. So a caller-supplied
/// `sm_scale` is not applied — it is CHECKED, against the default, at 1e-6.
/// Anything else is a different kernel's job. `sm_scale <= 0` means "unset"
/// and passes.
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
    //
    // `major` IS consulted here: `pick` filters on `enrolled_at`, and the one
    // unenrolled member is `Gqa8Page32Sm90`, which `dispatch` selects for the
    // 8-at-32 pair exactly when `major >= 9`. So this call refuses ratio 8 on
    // Hopper and above — and since the last line of this function is
    // `major >= 9`, **ratio 8 has no XQA decode on any device**. Deliberate,
    // and argued at `pick`.
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

// ===========================================================================
// The workspace carve, which is what the two launches share
// ===========================================================================

/// The scratch's alignment — `attention_xqa.cu:49`'s
/// `constexpr std::size_t kSemaphoreAlignment = 256`.
///
/// It aligns the tail of the carve, which is the region the decode hands XQA
/// as its scratch. Named for the semaphores because the same 256 governs the
/// semaphore bank the decode zeroes out of the workspace's int half; only the
/// alignment is shared, not the buffer.
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
    /// The row stride [`page_bucket`] chose, which the decode needs twice: as
    /// the table's stride and, times `page_size`, as XQA's maximum sequence
    /// length.
    ///
    /// Carried rather than recomputed so that the number the carve sized for
    /// and the number the kernel strides by are one value.
    bucket: i32,
}

/// The workspace carve — `attention_xqa.cu:291-309`, transcribed.
///
/// ```text
/// page_table_bytes = num_requests * page_bucket * sizeof(i32)
/// seq_lens_bytes   = num_requests * sizeof(u32)
/// p_page_table = align_up(float_buffer,                    alignof(i32))
/// p_seq_lens   = align_up(p_page_table + page_table_bytes, alignof(u32))
/// p_scratch    = align_up(p_seq_lens   + seq_lens_bytes,   256)
/// if (p_scratch >= float_buffer + float_bytes) throw
/// ```
///
/// `align_up` is `attention_xqa.cu:51-53`'s `align_up_ptr(p, a)`, whose body
/// is `(p + a - 1) / a * a`; it is spelled with [`usize::next_multiple_of`]
/// rather than transcribed, because the two agree for every `a > 0` and the
/// standard-library name is the one a reader does not have to check.
///
/// # The refusal is `>=`, not `>`
///
/// Carried exactly. A scratch that starts at the last byte of the workspace
/// has nothing in it, so the C++ refused an EMPTY scratch as well as an
/// overflowing one, and it refused it before any launch. Relaxing this to `>`
/// would hand XQA a zero-length region that it would write through.
///
/// # A refusal, where the C++ threw and where the port panicked
///
/// `attention_xqa.cu:308` threw `"xqa decode: attention workspace too small"`
/// and the shim that caught it is gone. This was a `panic!` for as long as
/// nothing could reach it; a trace statement can reach it now, and a fire
/// path may not abort a process over an argument. Same direction as
/// `attn::write_kv_explicit_bf16`'s assert that became a [`Refusal`].
///
/// # Errors
///
/// [`Refusal::Null`] for a workspace with no float half — **the C++ made no
/// such test**, and its arithmetic only caught it when `float_bytes` was
/// zero, so a workspace that is null and claims bytes produced a launch
/// against address 0, reported asynchronously as an unrelated fault at
/// whatever synchronised next. [`Refusal::Wide`] if the three regions do not
/// fit, naming the offset the scratch would have started at against the last
/// offset a non-empty scratch could take.
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
/// `attention_xqa_gqa2.cu:152`. The semaphore bank is a multi-block
/// rendezvous — every CTA of a split sequence bumps its `(request, kv head)`
/// slot and the last one merges — so a bank that did not start at zero makes a
/// CTA believe a merge it has to wait for already happened. It must be zeroed
/// ON the stream that will run the kernel and BEFORE it, which is here: the
/// archive's note that the memset "must happen on the stream before the fire"
/// was written when the host program did not have a stream to reach.
///
/// # Safety
///
/// `at` must address `bytes` of live device memory, and this context's stream
/// must be live across the call.
#[cfg(feature = "_cuda")]
unsafe fn zero_on_stream(ctx: &Ctx, at: *mut c_void, bytes: usize) -> Result<(), Refusal> {
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
unsafe fn zero_on_stream(_ctx: &Ctx, _at: *mut c_void, _bytes: usize) -> Result<(), Refusal> {
    Err(Refusal::Device { why: "this build selected no CUDA runtime" })
}

/// XQA's dense page table and sequence lengths.
///
/// XQA's decode entry point wants a dense zero-padded page table and a flat
/// `seq_lens` array; the paged KV cache carries a ragged CSR. This is the
/// transform.
///
/// **Not a routine and not a trace symbol.** Nothing lowers to it — see this
/// module's header — so it is left out of [`ROUTINES`] deliberately, and its
/// one caller is [`attention_xqa_decode_bf16_prepared`], which issues it as
/// the first of its two launches. A `routine!` here would put a row into
/// `crate::sigs()` under the `attn` namespace that `model-ir` could
/// look up and no trace could produce.
///
/// `max_pages_per_seq` is the CALLER's number and the kernel is handed
/// [`page_bucket`] of it, which is the launcher's own substitution
/// (`attention_xqa.cu:320` passed `page_bucket` into that parameter): the
/// kernel's parameter is the ROW STRIDE it fills, and the caller's number is
/// only what the stride was rounded up from. Computing it here rather than
/// taking it is what keeps the stride the caller carved for and the stride the
/// kernel writes at one expression apart from each other.
///
/// # Ordering
///
/// This build and the decode that reads it back are on ONE stream, in that
/// order, because one function issues both. See the module header for what
/// that replaced.
///
/// What the caller must guarantee, as `call()` states it: the three CSR
/// pointers address `num_requests`-shaped live device arrays, `page_table`
/// `num_requests * page_bucket(max_pages_per_seq)` writable `i32`s, `seq_lens`
/// `num_requests` writable `u32`s, and the stream is live across the launch.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty request set — a legal no-op that must not
/// reach a launch, because a zero `grid.x` is a refusal — and whatever the
/// compile, the load or the launch refuses.
#[allow(clippy::too_many_arguments)]
pub fn build_xqa_metadata(
    ctx: &Ctx,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    page_table: *mut i32,
    seq_lens: *mut u32,
    num_requests: i32,
    max_pages_per_seq: i32,
    page_size: i32,
) -> Result<(), Refusal> {
    /// The metadata build's block width: `attention_xqa.cu:313`'s literal `128`.
    ///
    /// Not an extent and not a reduction width — the page loop is
    /// `for (p = threadIdx.x; p < max_pages_per_seq; p += blockDim.x)` and the
    /// sequence length is written under `if (threadIdx.x == 0)`, so every width
    /// computes the same page table and the same sequence lengths. The number is a
    /// citation, not a derivation.
    const METADATA_BLOCK: u32 = 128;

    // `attention_xqa.cu:290` — `if (num_requests <= 0) return;`
    let bucket = page_bucket(max_pages_per_seq);
    // `attention_xqa.cu:313`, transcribed:
    //
    //     build_xqa_metadata_kernel<<<num_requests, 128, 0, stream>>>(
    //
    // `num_requests` is `grid.x` because the kernel indexes the request by
    // `blockIdx.x`. `smem` is 0: the kernel declares no shared memory at all —
    // the sequence length is written by lane 0 out of registers, not reduced.
    //
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_xqa.cuh",
            "::pie::attn::build_xqa_metadata",
            Launch::per_row(num_requests.unsigned_abs(), METADATA_BLOCK),
            &[
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_table.arg(),
                seq_lens.arg(),
                num_requests.arg(),
                bucket.arg(),
                page_size.arg(),
            ],
        )
    }
}

/// `attn::attention_xqa_decode_bf16_prepared` — the paged decode, and the
/// page table it reads.
///
/// The archive spelled this five times, once per lattice member
/// (`attention_xqa_gqa2.cu:109-193` and four identical twins), each recomputing
/// the same geometry. One body here is that arithmetic done once, with the
/// member chosen rather than compiled in.
///
/// # `_prepared` names the launch above it, not a promise the caller made
///
/// The symbol was `_prepared` because the archive's decode read a page table
/// some earlier `Prepare::FireWide` was supposed to have written. This body
/// writes it — [`build_xqa_metadata`] is its first launch, [`carve`] serves
/// both, and the two are ordered by the stream because one function issued
/// them. The name is kept because it is what a trace states; what it means
/// changed, and the module header says how.
///
/// # The multi-block split, cited
///
/// ```text
/// xqa/defines.h:101    ALLOW_MULTI_BLOCK_MODE defaults to true
/// xqa/mha.cu:~2990     nbSubSeqPerSeq = allowMultiBlockMode
///                        ? min(max(1u, multiProcessorCount / (batchSize * nbKHeads)),
///                              divUp(maxSeqLen, ctaTile.x))
///                        : 1
/// xqa/mha.cu:~2996     dim3 dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize}
/// ```
///
/// The first term is *"how many CTAs can I afford per (request, kv head)"* and
/// the second is *"how many are there sequence to cover"*; the smaller wins.
/// No root overrides `ALLOW_MULTI_BLOCK_MODE`, so the ternary's false arm is
/// unreachable and is not written.
///
/// # `maxSeqLen` is the BUCKETED length, not the caller's
///
/// `page_bucket * page_size`, and `attention_xqa_gqa2.cu:176` is where that
/// substitution is made. The page table's row stride is the power-of-two
/// bucket, the kernel strides by it, and the sequence length it is told about
/// has to be the strided one. `maxNbPagesPerSeq` inside the cache descriptor
/// is `exactDiv(maxSeqLen, tokensPerPage)`, which is the bucket again.
///
/// # What is NOT reproduced, and why
///
/// * `CUDA_CHECK(cudaPeekAtLastError())` after the launch. A peek-and-throw in
///   a fire path attributes an unrelated async fault to the next kernel that
///   happens to run.
/// * `enable_pdl`. It is the ONLY thing `xqa/hostUtils.h`'s `makeLaunchConfig`
///   adds over a plain launch: one
///   `cudaLaunchAttributeProgrammaticStreamSerialization`. With PDL off a bare
///   launch reproduces the launch exactly; [`Launch`] has no field to carry it
///   and `jit::launch` issues no attribute but the cooperative one.
///
/// # The shared memory needs no flag
///
/// [`XQA_SMEM_BYTES`] is 79,488, well over the 48 KiB a launch may ask for
/// before anyone has asked the driver for more, and `jit::launch::issue`
/// raises the per-`(device, entry point)` cap whenever `Launch::smem` exceeds
/// it. That is the readable half of `configureKernel()`
/// (`xqa/mha.cu:2955`) discharged by putting the number in the launch — and it
/// discharges what `attention_xqa.hpp` recorded as an OPEN obligation, *"an
/// undischarged per-device `cudaFuncSetAttribute` under TP>1"*, because a per
/// `(device, function)` key is exactly what a per-process C++ static
/// initializer could not be.
///
/// # The semaphore bank is the workspace's INT half
///
/// `attention_xqa_gqa2.cu:145-151` threw `"xqa gqa2 decode: semaphore
/// workspace too small"` on a budget the caller owned; here the caller states
/// the extent and this body checks it, which is the same test with one fewer
/// place to be wrong. The bank is a multi-block rendezvous and must be zeroed
/// on the stream ahead of the launch — see [`zero_on_stream`].
///
/// What the caller must guarantee, as `call()` states it: `q` and `o` address
/// `num_requests * num_q_heads` live heads each, `k_pages`/`v_pages` the
/// layer's page arena, the three CSR pointers `num_requests`-shaped live
/// device arrays, and `float_buffer`/`int_buffer` the two halves of an
/// attention workspace of the stated extents.
///
/// # Errors
///
/// [`Refusal::Device`] for a shape or a device XQA does not serve,
/// [`Refusal::Absent`] for a shape whose lattice member has no root (see
/// [`XqaMember::pick`]), [`Refusal::Empty`] for an empty request set,
/// [`Refusal::Null`] for a workspace or CSR half that is not there,
/// [`Refusal::Wide`] if the carve or the semaphore bank does not fit its half
/// of the workspace, and whatever the compile, the load or the launch
/// refuses. **Every one of them is raised before the first launch**, so a
/// refused decode has enqueued nothing — which is what keeps a half-run pair
/// off the stream.
#[allow(clippy::too_many_arguments)]
pub fn attention_xqa_decode_bf16_prepared(
    ctx: &Ctx,
    q: *const XqaIoHead,
    o: *mut XqaIoHead,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    float_buffer: *mut c_void,
    float_bytes: usize,
    int_buffer: *mut c_void,
    int_bytes: usize,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    max_pages_per_seq: i32,
    sm_scale: f32,
) -> Result<(), Refusal> {
    /// `xqa/mha.cu:96`'s `ctaTile.x`, the sequence-length step one CTA covers.
    ///
    /// `warpTile = {64, roundUp(nbValidRows, 16)}` and
    /// `ctaTile.x = warpTile.x * ctaShapeInWarps.x` = `64 * 4` = 256. It is the
    /// denominator in the multi-block split below, so it is a geometry input and
    /// not a comment.
    const XQA_CTA_TILE_X: u32 = 256;

    let Some(major) = ctx.compute_capability_major() else {
        return Err(Refusal::Device { why: "the device would not say its compute capability" });
    };
    let major = i32::try_from(major).unwrap_or(0);
    // `attention_xqa.cu:285` — the gate runs before the empty-set test,
    // because a shape that is wrong is wrong whether or not anything asked.
    // `window_left` and `logits_soft_cap` are the two the caller pinned:
    // `/*window_left=*/-1, /*logits_soft_cap=*/0.f` at `attention_xqa.cu:287`.
    if !decode_supported(num_q_heads, num_kv_heads, head_dim, page_size, -1, 0.0, sm_scale, major) {
        return Err(Refusal::Device { why: "XQA serves neither this shape nor this device" });
    }
    // `dispatch` rather than `pick`, so the two refusals stay apart: a shape
    // outside the lattice and a member no root hosts are different answers
    // with different fixes, and `pick`'s `None` cannot tell them apart.
    // `decode_supported` above has already refused the first case at
    // `XQA_PAGE_SIZE`; this is the same test at the page size the caller
    // passed.
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

    // ONE carve for both launches: the metadata build WRITES the page table
    // and the sequence lengths and the decode READS them back, so the two
    // cannot disagree about where they are.
    let regions = carve(float_buffer, float_bytes, num_requests, max_pages_per_seq)?;
    let bucket = regions.bucket;
    let max_seq_len = bucket.unsigned_abs() * page_size.unsigned_abs();

    // `attention_xqa_gqa2.cu:145-151` — the bank is `num_requests *
    // num_kv_heads` `u32`s and lives in the workspace's INT half, so the
    // budget is a question about the caller's workspace and is asked here,
    // before anything is enqueued.
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

    // Asked HERE rather than beside its one use below, which is the last
    // refusal this function can raise: a device that will not say how many
    // multiprocessors it has must not do so with the metadata build already
    // on the stream and no decode coming to read it.
    let multiprocessors = ctx.multiprocessors()?;

    // THE FIRST LAUNCH. Everything above refuses; nothing above enqueues.
    // `max_pages_per_seq` rather than `bucket` is passed on purpose: the
    // build applies `page_bucket` itself, and one expression computing the
    // stride for both sides is what stops the carve and the kernel disagreeing
    // about it.
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

    // `attention_xqa_gqa2.cu:152`, and it starts every fire at zero.
    //
    // SAFETY: the caller's assertion about the workspace's int half,
    // forwarded — the budget above is what makes the extent right.
    unsafe {
        zero_on_stream(ctx, semaphores.cast(), semaphore_bytes)?;
    }

    let afford = (multiprocessors / (batch * kv_heads)).max(1);
    let sub_seqs = afford.min(max_seq_len.div_ceil(XQA_CTA_TILE_X));

    // `attention_xqa_gqa2.cu:168-174` computes the three strides in ELEMENTS,
    // and `launchMHAFlashInfer` then divides each by
    // `validElemsPerHead / CacheElemConverter::ElemsPerContainer` to get the
    // `uint32_t` "in heads" the kernel takes. With `-DCACHE_ELEM_ENUM=0` the
    // container is one element, so the divisor is `validElemsPerHead`, which
    // is `HEAD_ELEMS` — and `decode_supported` has already refused any
    // `head_dim` but that. Both halves are written out rather than cancelled
    // because the cancellation is what stops holding the day the cache is
    // quantised.
    let per_head = u64::from(head_dim.unsigned_abs());
    let in_heads = |elements: u64| u32::try_from(elements / per_head).unwrap_or(u32::MAX);
    let pages = u64::from(page_size.unsigned_abs());
    let stride_page = in_heads(pages * u64::from(kv_heads) * per_head);
    let stride_token = in_heads(u64::from(kv_heads) * per_head);
    let stride_head = in_heads(per_head);

    // `KVCacheList<true> const cacheList{...}` — a LOCAL, and it has to be:
    // `Abi::arg` answers a borrow of it (`ArgValue::Bytes`), and the launch
    // copies out of that borrow. See `x/abi.rs`'s header for the frame this
    // shape exists to avoid.
    let cache = KvCacheList::paged(
        k_pages as DevicePtr,
        v_pages as DevicePtr,
        regions.page_table as DevicePtr,
        regions.seq_lens as DevicePtr,
        bucket.unsigned_abs(),
    );

    // `xqa/mha.cu:2783`'s parameter list under this option set: `SPEC_DEC=0`
    // drops the four q-sequence operands, `SLIDING_WINDOW=0` the window,
    // `LOW_PREC_OUTPUT=0` the output scale, `BEAM_WIDTH=1` the beam params and
    // `ENABLE_4BIT_KV_CACHE` off the three scale-factor strides. Fifteen are
    // left, and these are they.
    //
    // `qScale` and `kvCacheScale` are `1.0f` because XQA folds
    // `1/sqrt(head_dim)` into the kernel; `decode_supported` CHECKED
    // `sm_scale` against that default rather than applying it.
    //
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as, and `cache` outlives
    // the launch.
    unsafe {
        ctx.launch_at(
            ROOTS[root_at],
            inst::MHA[root_at],
            Launch::grid([sub_seqs, kv_heads, batch], XQA_BLOCK).smem(XQA_SMEM_BYTES),
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
}

/// XQA's one trace symbol.
///
/// [`build_xqa_metadata`] is deliberately absent — it is a launch this
/// routine issues, not a statement anything lowers to, and a row for it would
/// be a symbol `model-ir` could look up and no trace could produce.
///
/// `whole` for the reason the deleted row gave: the build walks
/// `kv_page_indptr` and `kv_last_page_lens`, which are R-shaped, so a row
/// window would leave that arithmetic pointing at another request.
pub static ROUTINES: &[Routine] = &[routine!(attention_xqa_decode_bf16_prepared, whole)];

/// XQA's symbol is `attn`'s, as a trace names it.
///
/// A second family under the same namespace as [`crate::attn`], which
/// `Family::routine` resolves by name after the prefix and `sigs()`' own
/// duplicate test still refuses to see twice. Merging the two would mean
/// moving five roots, a six-member lattice and a by-value aggregate into a
/// 4,000-line file to save a `static`.
pub static FAMILY: Family = crate::family!(ROUTINES);

#[cfg(test)]
mod tests {
    use super::{
        MAX_PAGE_BUCKET, OPTIONS, ROOTS, ROUTINES, SCRATCH_ALIGN, XQA_LATTICE, XQA_ROOT, XqaMember,
        carve, decode_supported, inst, page_bucket,
    };
    use kernels::Refusal;

    /// Five roots for six lattice members, and every one of them is the same
    /// text under a different option set.
    #[test]
    fn the_roots_and_the_lattice_agree() {
        assert_eq!(ROOTS.len(), XQA_LATTICE.len() - 1, "the Hopper member has no root");
        for (at, root) in ROOTS.iter().enumerate() {
            assert_eq!(root.name, XQA_LATTICE[at].unit);
            assert_eq!(inst::MHA[at], XQA_LATTICE[at].entry);
            assert_eq!(root.text, XQA_ROOT, "one text, five roots");
            assert_eq!(root.options, OPTIONS[at], "and the options are what differ");
            assert_eq!(root.headers, crate::jit::Headers::LibraryAndUpstream);
        }
        assert!(XqaMember::Gqa8Page32Sm90.enrolled_at().is_none());
        assert_eq!(XqaMember::Gqa8Page32.enrolled_at(), Some(4));
    }

    /// One text five ways is five cubins, and the cache key is what says so.
    #[test]
    fn the_five_roots_key_apart() {
        let keys: Vec<String> = ROOTS.iter().map(|r| r.key(inst::MHA[0], "sm_90")).collect();
        for (at, key) in keys.iter().enumerate() {
            for other in &keys[at + 1..] {
                assert_ne!(key, other, "two members would share a cubin");
            }
        }
    }

    /// `xqa_decode_page_bucket` — `attention_xqa.cu:274-279`.
    #[test]
    fn the_page_bucket_is_a_clamped_power_of_two() {
        assert_eq!(page_bucket(0), 1, "and the floor is on `pages`, not on `bucket`");
        assert_eq!(page_bucket(1), 1);
        assert_eq!(page_bucket(3), 4);
        assert_eq!(page_bucket(32), 32);
        assert_eq!(page_bucket(33), 64);
        assert_eq!(page_bucket(MAX_PAGE_BUCKET), MAX_PAGE_BUCKET);
        assert_eq!(page_bucket(MAX_PAGE_BUCKET + 1), MAX_PAGE_BUCKET, "clamped, not rounded up");
    }

    /// Ratio 8 selects a body nothing hosts, on the only devices XQA runs on.
    #[test]
    fn ratio_eight_selects_a_body_nothing_hosts() {
        assert_eq!(XqaMember::dispatch(8, 32, 9), Some(XqaMember::Gqa8Page32Sm90));
        assert_eq!(XqaMember::pick(8, 32, 9), None, "and `pick` refuses rather than falling back");
        assert_eq!(XqaMember::dispatch(8, 32, 8), Some(XqaMember::Gqa8Page32));
        assert!(!decode_supported(64, 8, 128, 32, -1, 0.0, 0.0, 9), "so the gate refuses it");
        assert!(decode_supported(16, 8, 128, 32, -1, 0.0, 0.0, 9), "ratio 2 is unaffected");
        assert!(!decode_supported(16, 8, 128, 32, -1, 0.0, 0.0, 8), "and the floor is SM90");
    }

    /// One symbol, and it is the one a trace states.
    ///
    /// `build_xqa_metadata` is a launch this family issues and not a symbol
    /// anything lowers to, so a second row here would be a declaration no
    /// trace could produce. `whole` is the deleted row's column, carried.
    #[test]
    fn the_family_declares_the_decode_and_only_the_decode() {
        assert_eq!(ROUTINES.len(), 1);
        assert_eq!(ROUTINES[0].name, "attention_xqa_decode_bf16_prepared");
        assert_eq!(super::FAMILY.symbol(&ROUTINES[0]), "attn::attention_xqa_decode_bf16_prepared");
        assert!(ROUTINES[0].whole, "the build walks an R-shaped CSR");
    }

    /// The carve is `attention_xqa.cu:291-309`, and the three regions do not
    /// overlap.
    #[test]
    fn the_carve_lays_three_regions_end_to_end() {
        // A fake base far from zero: `carve` only does address arithmetic and
        // dereferences nothing, so the number just has to be aligned and to
        // leave room for the sizes below.
        let base = 1 << 20;
        let buffer = core::ptr::without_provenance_mut::<core::ffi::c_void>(base);
        let regions = carve(buffer, 1 << 20, 4, 30).expect("a megabyte holds four 32-page rows");
        assert_eq!(regions.bucket, 32, "30 pages round up to a 32-page stride");
        assert_eq!(regions.page_table.addr(), base);
        assert_eq!(regions.seq_lens.addr(), base + 4 * 32 * 4, "the table, then the lengths");
        assert_eq!(
            regions.scratch.addr(),
            (base + 4 * 32 * 4 + 4 * 4).next_multiple_of(SCRATCH_ALIGN),
            "and the scratch is 256-aligned",
        );
    }

    /// A workspace too small refuses, where the C++ threw and the port
    /// panicked — and an EMPTY scratch refuses too, which is the `>=`.
    #[test]
    fn the_carve_refuses_rather_than_panicking() {
        let buffer = core::ptr::without_provenance_mut::<core::ffi::c_void>(1 << 20);
        assert!(matches!(carve(buffer, 16, 4, 30), Err(Refusal::Wide { .. })));
        // Exactly enough for the two written regions and nothing after them:
        // the scratch would start at the last byte and have nothing in it.
        let full = 4 * 32 * 4 + 4 * 4;
        assert!(matches!(carve(buffer, full, 4, 30), Err(Refusal::Wide { .. })));
        assert!(matches!(
            carve(core::ptr::null_mut(), 1 << 20, 4, 30),
            Err(Refusal::Null { .. }),
        ));
    }

    /// The scale is CHECKED against `1/sqrt(head_dim)`, never applied.
    #[test]
    fn a_caller_supplied_scale_is_an_equality_test() {
        let default = 1.0f32 / 128.0f32.sqrt();
        assert!(decode_supported(16, 8, 128, 32, -1, 0.0, default, 9));
        assert!(decode_supported(16, 8, 128, 32, -1, 0.0, 0.0, 9), "unset passes");
        assert!(!decode_supported(16, 8, 128, 32, -1, 0.0, default * 2.0, 9));
        assert!(!decode_supported(16, 8, 128, 32, 0, 0.0, default, 9), "a window refuses");
        assert!(!decode_supported(16, 8, 128, 32, -1, 30.0, default, 9), "a soft cap refuses");
    }
}
