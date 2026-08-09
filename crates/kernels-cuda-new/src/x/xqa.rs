//! XQA's by-value aggregate — pinned field-for-field against a layout
//! **measured out of NVRTC's PTX** — and the five-member lattice it
//! unblocked.
//!
//! Two halves, in that order: the `KVCacheList<true>` mirror that made
//! `kernel_mha`'s parameter list expressible at all, then the units,
//! contract and refusal that enrol it.
//!
//! # Why this exists
//!
//! `kernel_mha` (`csrc/vendor/xqa/mha.cuh:2801`) takes
//! `KVCacheList<usePagedKVCache> const cacheList` **by value** — forty bytes
//! of pointers and a count, in the middle of its parameter list. Every other
//! XQA parameter is a scalar or a pointer and crosses as one `u64` cell;
//! this one does not, and until it could cross, the six `attention_xqa*.cu`
//! host programs could not be retired.
//!
//! `northstar.md` §5.1 predicted this file's arrival and named the failure
//! mode it has to defend against:
//!
//! > `ArgValue::Bytes` is written and unused. Its first caller … is where
//! > the tag bypass gets tested — and note the failure mode: **a wrong
//! > bypass is a launch with a garbage struct, not a type error.**
//!
//! It was half right about the caller: XQA's `KVCacheList` got here before
//! `attn`'s `MLAParams`. It was entirely right about the failure mode, which
//! is why nothing below was read off the header.
//!
//! # Where the numbers came from
//!
//! **Not from reading the header.** `mhaUtils.cuh:241-257` gives the field
//! ORDER, and the order is all a human can safely read off — the struct's
//! shape depends on `ENABLE_4BIT_KV_CACHE`, which inserts two more pointers
//! in the middle, and on `beamWidth`, and neither is visible at the
//! declaration site.
//!
//! The offsets were measured by `nvrtc-probes/xqa_kvcachelist.py`, which
//! compiles the real struct under the real carried header set and the real
//! `XQA_COMMON_OPTIONS` and reads the offsets back out of the emitted PTX.
//! **Re-running that script is how these assertions get checked against a new
//! upstream.** Its method is `nvrtc-probes/params_layout.py`'s, whose banner
//! records which constant-expression routes NVRTC rejects and why:
//! `offsetof` and `__builtin_offsetof` are both unavailable, the cast of a
//! null-member address is rejected, and only
//! `(unsigned)((char*)&((S*)0)->b - (char*)(S*)0)` — the *difference* of two
//! pointers — folds.
//!
//! Measured, `rc=0`, NVRTC 13.0, `compute_89`, `-std=c++17 -default-device`:
//!
//! ```text
//! ENABLE_4BIT_KV_CACHE=0  beamWidth=1  tokensPerPage=32
//! sizeof(KVCachePageIndex)=4  sizeof(SeqLenDataType)=4  sizeof(ptr)=8
//!
//! KVCacheList<true>: sizeof=40 alignof=8
//!      0  kCacheVLLM         GMemCacheHead*
//!      8  vCacheVLLM         GMemCacheHead*
//!     16  kvCachePageList    KVCachePageIndex const*
//!     24  seqLenList         SeqLenDataType const*
//!     32  maxNbPagesPerSeq   uint32_t
//! ```
//!
//! This confirms the transcription `families/attn.rs` made by eye — *"four
//! pointers plus a `uint32_t`: 40 bytes, 8-aligned"* — which is the first
//! time in this migration that reading and measuring have agreed. The reason
//! they agree is the next section, and it is the thing to check before
//! trusting any future XQA mirror.
//!
//! # The trap that cannot reach this struct, and why that is not luck
//!
//! `params_layout.txt` records three transcription traps the FA2 mirrors hit,
//! all of them the same kind:
//!
//! | what the declaration suggests | what it is |
//! |---|---|
//! | `uint_fastdiv` — one `u32` | **24 bytes**, shifting everything after it by twenty |
//! | CuTe's `dA` — three integers | **8 bytes**; the static modes are compressed out |
//! | two `paged_kv_t`s with equal `sizeof` | **different interiors** — `num_heads` at +24 vs +20 |
//!
//! Every one of them is a **nested aggregate** whose own size or interior
//! disagrees between `csrc/shim`'s headers and CCCL's. The probe measures the
//! *shim's* layout, because the shim is what NVRTC is handed; a mirror built
//! from CCCL's numbers would be wrong in the interior while agreeing on
//! `sizeof`, which is the worst possible way to be wrong.
//!
//! `KVCacheList<true>` has **no nested aggregate**. It is four pointers and a
//! `uint32_t`, and both header sets agree that a pointer is eight bytes and a
//! `uint32_t` is four. So these five offsets are correct under the shim's
//! headers *and* under CCCL's, which makes this the only XQA-side mirror for
//! which the shim/CCCL split is not a live hazard — worth writing down
//! precisely because the next XQA aggregate will not have that property.
//!
//! # What this mirror claims, and its precondition
//!
//! It claims to be `KVCacheList<true>` **under `ENABLE_4BIT_KV_CACHE == 0`**.
//! That is not an assumption: `XQA_COMMON_OPTIONS` (`families/attn.rs`) sets
//! `CACHE_ELEM_ENUM=0` for all six lattice members, and
//! `ENABLE_4BIT_KV_CACHE` is derived from it. Should a member ever compile
//! with `CACHE_ELEM_ENUM=4`, `kSfCacheVLLM`/`vSfCacheVLLM` appear at 16 and
//! 24, everything after shifts by sixteen, and `sizeof` becomes 56 — so the
//! `static_assert`s [`typecheck_tu`](crate::x::abi::typecheck_tu) emits from
//! [`LAYOUTS`] fail on that unit and name the struct. **That is the check.**
//! The Rust-side assertions cannot see a `-D`; the C++ ones compile under the
//! member's own options and can.
//!
//! `KVCacheList<false>` — the contiguous cache — measured `sizeof=24
//! alignof=8`, `data@0`, `seqLenList@8`, `capacity@16`. It is recorded here
//! and **not mirrored**, because no lattice member instantiates it: all six
//! set `USE_PAGED_KV_CACHE=1`. Data only for what has a reading consumer;
//! this paragraph is the reading consumer, and a `#[repr(C)]` struct nothing
//! passes would not be.
//!
//! # The lattice is enrolled below, and this section is the handover it
//! replaces
//!
//! This module was the floor half only — the mirror, its layout, and its
//! [`Abi`](crate::x::Abi) impl — and asked for the rest: *"The `unit!`,
//! `contract!` and `bind!` for the six-member lattice belong here too and
//! are yours to add."* They are added, and three things about how differ
//! from what that sentence assumed:
//!
//! * **There is no `unit!`.** It hardcodes `options: &[]` and the five units
//!   are nothing but an option set. [`UNITS`] carries the exact grammar
//!   patch that would fix it, and the separate reason the hand-written form
//!   is better here anyway.
//! * **Five members, not six.** The Hopper one does not compile; [`UNITS`]
//!   states the measurement and the consequence.
//! * **The `bind!` is a `none:` arm**, and its comment names the two `Cx`
//!   queries and the one device call that would turn it into a bind.
//!
//! The instruction that mattered is honoured as written: **the `unit!` entry
//! came before the bind.** The by-value path bypasses the runtime's
//! per-operand tag check by construction, so the typecheck TU is the only
//! thing standing between a transposed parameter and a launch reading a
//! garbage struct — and a garbage struct is not a type error, it is a fault
//! or, worse, an answer. [`PARAMS`] is what that TU compares, and it is
//! generated from the declared types rather than written, so the fifteen
//! spellings cannot drift from the fifteen parameters.

use crate::by_value;
use crate::device::DeviceKernel;
use crate::families::attn::{XQA_COMMON_OPTIONS, XQA_LATTICE, XQA_ROOT};
use crate::unit::Unit;
use crate::x::abi::MaybeConst;
use crate::{bind, contract};
use core::ffi::c_void;
use core::ptr::NonNull;
use kernels::{Cap, KernelSig, Operand, Prepare, Ty};

/// A device address held as an opaque word.
///
/// Re-exported so a mirror's fields read as one vocabulary. See
/// [`x::abi::DevicePtr`](crate::x::abi::DevicePtr) for why these are `u64`
/// and not `*mut T`.
pub use crate::x::abi::DevicePtr;

/// `KVCacheList<true>` — XQA's paged KV cache descriptor, passed by value.
///
/// Mirrors `csrc/vendor/xqa/mhaUtils.cuh:241-257` under
/// `ENABLE_4BIT_KV_CACHE == 0`. Nothing here may be dereferenced on the host:
/// every field is a device address or a device-side count.
///
/// Field names are Rust's, not C++'s; the C++ spelling of each travels in
/// [`LAYOUTS`] and is what the `static_assert`s name, so a reader chasing a
/// failed assertion never has to guess which Rust field moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct KvCacheList {
    /// `kCacheVLLM` — `GMemCacheHead*`, the K pages.
    pub k_cache: DevicePtr,
    /// `vCacheVLLM` — `GMemCacheHead*`, the V pages.
    pub v_cache: DevicePtr,
    /// `kvCachePageList` — `KVCachePageIndex const*`, shape
    /// `[batchSize][beamWidth][2][maxNbPagesPerSeq]`.
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
    ///
    /// Every argument is a device address the caller already holds; this
    /// function only assembles them, which is why it is not `unsafe` — no
    /// pointer is read here, and the launch that reads them is.
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
///
/// `unit!` cannot derive this: `PARAMS` holds C++ spellings and a spelling
/// cannot be turned back into a type. So it is written once, here, and handed
/// to [`typecheck_tu`](crate::x::abi::typecheck_tu) with `PARAMS`.
///
/// A `KVCacheList<true>` parameter that appears in a unit whose `LAYOUTS`
/// omits it still typechecks by name — `is_same_v` catches a wrong type — but
/// loses the `sizeof`/`offsetof` assertions, which are the half that catches
/// a header whose FIELDS moved under a type whose NAME did not.
pub static LAYOUTS: &[crate::x::Layout] = &[<KvCacheList as crate::x::ByValue>::LAYOUT];

// ===========================================================================
// The lattice, enrolled.
//
// Five units on one root, one contract, one refusal. What follows is the
// other half of this module — the half `# For xqa-nvrtc` above asked for —
// and the three things a reader should check first are these:
//
//  1. THERE IS NO `unit!` HERE AND THAT IS NOT A SHORTCUT. `unit!`
//     hardcodes `options: &[]` (`x/macros.rs:239`) and the whole of what
//     distinguishes these five units is a `-D` set. The macro cannot
//     express them today. The exact grammar patch that would let it is
//     written out below `UNITS`, for whoever owns `x/macros.rs`; the
//     hand-written form is also strictly better here for a reason that has
//     nothing to do with the gap, and that reason is written there too.
//
//  2. THE SIXTH MEMBER IS NOT ENROLLED. `families::attn::XQA_LATTICE`'s
//     last entry is marked NOT READY and argues it at length: at
//     `compute_90a` the Hopper body stops on `std::pair` in DEVICE code,
//     the carried header set has no `CUtensorMap`, and the archive unit
//     compiled a HOST `.cpp` that would be a second and larger port. A
//     `Unit` for it would be a compile that fails in `tests/units.rs` on a
//     GPU box; enrolling five and saying why is the honest count.
//
//  3. THE CONTRACT'S SYMBOL AND THE ROWS' SYMBOLS ARE DIFFERENT, which is
//     unusual in this directory and is the shape of the family. The
//     contract is the DSL-level `attn::attention_xqa_decode_bf16_prepared`,
//     one symbol a trace may say; the rows are the five `extern "C"`
//     `__global__`s the `-Dkernel_mha=…` renames produce, which is what
//     `cuModuleGetFunction` resolves. The host program picks among them
//     (`driver-cuda/src/fire/xqa.rs::XqaMember::pick`), so one contract
//     over five rows is the arrangement and not an accident.
// ===========================================================================

/// `IOHead` and `OutputHead` — XQA's per-head vector, as a pointee.
///
/// `xqa/mha.h:58` is `using IOHead = Vec<InputElem, validElemsPerHead>`, and
/// `:76` makes `OutputHead` `conditional_t<lowPrecOutput, GMemCacheHead,
/// InputHead>` — with `LOW_PREC_OUTPUT=0` in [`XQA_COMMON_OPTIONS`] the two
/// are ONE type under two names, which is why one pointee serves both and
/// why the `const` half spells `IOHead` while the `mut` half spells
/// `OutputHead`. Each parameter spells itself the way its own declaration
/// does; `is_same_v` in the typecheck TU sees through both aliases.
///
/// # Why not `*const bf16`
///
/// Because it is not `bf16*`. `Vec<device::bf16, 128>` is an aggregate of
/// 128 elements, not a pointer to one, and `is_same_v<const bf16*, const
/// IOHead*>` is false — the typecheck TU would fail on the spelling, which
/// is the good outcome and the reason not to reach for the near-miss. The
/// row world's `Ty` has no word for it either, so [`Ty::Buf`]/[`Ty::BufMut`]
/// carry the address opaquely and [`Abi::CPP`](crate::x::Abi::CPP) carries
/// the truth. That is `x/abi.rs`'s own `u64` precedent applied: *"the buffer
/// crosses correctly today — it is opaque to the host and only its width is
/// wrong in the tag."*
///
/// # Why the impls are written by hand
///
/// [`ptr_abi!`] is a plain `macro_rules!` private to `x/abi.rs`, which this
/// module does not own. It does not need to: `x/abi.rs`'s header states the
/// rule — *"adding a crossing type means writing one impl, next to the
/// kernel that needed it, and nothing else in the tree changes"* — and
/// `x/attn.rs`'s `StructuredMaskParams` is the worked precedent. The
/// nullable forms `ptr_abi!` would also generate are not written, because no
/// XQA parameter is a nullable head pointer: `output` and `q` are asserted
/// non-null by the launcher's own shape.
///
/// [`ptr_abi!`]: crate::x::abi
/// [`Ty::Buf`]: kernels::Ty::Buf
/// [`Ty::BufMut`]: kernels::Ty::BufMut
pub enum XqaIoHead {}

impl crate::x::Abi for *const XqaIoHead {
    const CPP: &'static str = "const IOHead*";
    const TY: Ty = Ty::Buf;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Ptr(*self as *mut c_void)
    }
}

impl crate::x::Abi for *mut XqaIoHead {
    const CPP: &'static str = "OutputHead*";
    const TY: Ty = Ty::BufMut;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Ptr(self.cast::<c_void>())
    }
}

/// `kernel_mha`'s parameter list, written ONCE and expanded three ways.
///
/// The three spellings a parameter list needs — the C++ types for the
/// typecheck TU, the [`kernels::Operand`] rows for `Args::bind`, and the
/// launcher's own Rust signature — are all generated from the list below, so
/// a parameter cannot be added to one and missed in another and cannot be
/// reordered in one alone. That is exactly what `unit!` does for every other
/// family; this is the same guarantee spelled locally, because `unit!`
/// cannot carry [`Unit::options`] and these five units are nothing BUT an
/// option set.
///
/// # The list is `mha.cuh:2783-2814` under the lattice's own defines
///
/// `kernel_mha`'s declaration is eight `#if`s deep and the effective list
/// depends on every one of them. Under [`XQA_COMMON_OPTIONS`] —
/// `SPEC_DEC=0`, `SLIDING_WINDOW=0`, `LOW_PREC_OUTPUT=0`, `BEAM_WIDTH=1`,
/// `CACHE_ELEM_ENUM=0` and so `ENABLE_4BIT_KV_CACHE=0` — five blocks vanish
/// and fifteen parameters remain. Each of the five would have inserted
/// parameters in the MIDDLE, which is why the option set is part of this
/// declaration's meaning and not a compile flag beside it: a member built
/// with `SPEC_DEC=1` has a different parameter list and would need a
/// different `PARAMS`, not a different `-D`.
///
/// `semaphores` and `scratch` carry `= nullptr` defaults at the declaration.
/// A default argument is not part of a function's TYPE, so the typecheck
/// TU's `is_same_v` neither sees them nor should; they are recorded as
/// [`Abi::NULLABLE`](crate::x::Abi::NULLABLE) instead, which is where the
/// fact belongs — `mha.cuh:1598` asserts `!isMultiBlock || (semaphores !=
/// nullptr && scratch != nullptr)`, so null is legal and only conditionally.
/// `qScalePtr`, `kvScalePtr` and `attentionSinks` are nullable for the same
/// reason and by the same evidence: `:1592`, `:1593` and `:2545` each test
/// the pointer against null in DEVICE code and fall back to the scalar.
macro_rules! kernel_mha_declaration {
    ($($pname:ident : $pty:ty),* $(,)?) => {
        /// Each parameter's C++ type, for the typecheck TU.
        ///
        /// From [`Abi::CPP`](crate::x::Abi::CPP) and never from a literal,
        /// so a spelling cannot drift from the type it describes. All five
        /// units share it: they differ by `-D` and not by signature.
        pub static PARAMS: &[&str] = &[$(<$pty as crate::x::Abi>::CPP),*];

        /// The same list as operand rows.
        ///
        /// Every `Source::Unbound`, because a device row has no sources in
        /// fn-world — a `fn` binds its own arguments. This is what `unit!`'s
        /// `@rows` accumulator writes for every other family.
        static OPERANDS: &[Operand] = &[$(Operand {
            name: stringify!($pname),
            ty: <$pty as crate::x::Abi>::TY,
            nullable: <$pty as crate::x::Abi>::NULLABLE,
            source: ::kernels::Source::Unbound,
        }),*];

        /// The typed launcher, one for all five members.
        ///
        /// `symbol` picks which — `XqaMember::symbol()` in
        /// `driver-cuda/src/fire/xqa.rs` is the function that answers it —
        /// and everything else is the device text's own parameter list, in
        /// its own order, in Rust types. ONE launcher and not five, which is
        /// the part `unit!` could not have given: it emits a `raw::` module
        /// per unit, so six invocations would have produced six identical
        /// `kernel_mha`s in six modules and a caller would have had to pick
        /// the module AND the symbol to say one thing.
        #[cfg(feature = "_cuda")]
        pub mod raw {
            #[allow(unused_imports)]
            use super::*;
            use crate::x::launch::Launch;

            /// Launch one member of the XQA lattice.
            ///
            /// # Safety
            ///
            /// Every pointer must address live device memory of the extent
            /// this kernel will read or write, `cache_list`'s five fields
            /// must be device addresses of a paged cache laid out as
            /// `KVCacheList<true>` describes, and `stream` must be live
            /// across the launch.
            ///
            /// `launch.smem` must be `XQA_SMEM_BYTES` (79 488), which is
            /// over the 48 KiB default and therefore drives
            /// `runtime::module::raise_dynamic_smem_cap`. A smaller value
            /// launches a kernel whose `SharedMem` does not fit.
            #[allow(clippy::too_many_arguments, unused_unsafe)]
            pub unsafe fn kernel_mha(
                symbol: &'static str,
                launch: Launch,
                $($pname: $pty,)*
                stream: *mut c_void,
            ) {
                unsafe {
                    crate::x::fire::fire(
                        symbol,
                        launch,
                        // `arg` borrows. For a scalar or a pointer that is a
                        // copy either way; for `cache_list` the borrow is of
                        // THIS binding, which lives across the call, and
                        // `Args::bind` copies the forty bytes out before
                        // `fire` returns.
                        &[$(<$pty as crate::x::Abi>::arg(&$pname)),*],
                        stream,
                    );
                }
            }
        }
    };
}

kernel_mha_declaration! {
    nb_k_heads: u32,
    q_scale: f32,
    q_scale_ptr: MaybeConst<f32>,
    output: *mut XqaIoHead,
    q: *const XqaIoHead,
    attention_sinks: MaybeConst<f32>,
    cache_list: KvCacheList,
    batch_size: u32,
    kv_cache_scale: f32,
    kv_scale_ptr: MaybeConst<f32>,
    kv_stride_page: u32,
    kv_stride_token: u32,
    kv_stride_head: u32,
    semaphores: Option<NonNull<u32>>,
    scratch: Option<NonNull<c_void>>,
}

/// The file every row cites, for a `KernelSig`'s `file`.
const FILE: &str = "attn/attention_xqa_mha.cuh";

/// One member's `-D` set: the thirteen shared with every other, then the
/// four that are its own.
///
/// Read out of [`XQA_COMMON_OPTIONS`] and [`XQA_LATTICE`] rather than
/// restated, so the option sets `families::attn` measured and the option
/// sets NVRTC is handed are the same array and cannot drift. That is why
/// `XQA_LATTICE` is a `const` and not a `static`: a `const` may not read a
/// `static`, and this concatenation happens at compile time.
const fn options_of(member: usize) -> [&'static str; 17] {
    let extra = XQA_LATTICE[member].options;
    let mut out = [""; 17];
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

// Seventeen is thirteen plus four, and both halves are checked here rather
// than trusted: `options_of` writes into a fixed-width array, so a shared
// option added without widening it would be a const-eval index panic at
// build time, and a member with a fifth `-D` would silently leave the
// seventeenth slot empty. An empty string in an option array is a valid
// argument to NVRTC and compiles to nothing, which is the shape this crate's
// rules call a stub.
const _: () = assert!(
    XQA_COMMON_OPTIONS.len() == 13,
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

const OPTIONS_GQA2_P32: [&str; 17] = options_of(0);
const OPTIONS_GQA2_P16: [&str; 17] = options_of(1);
const OPTIONS_GQA4_P32: [&str; 17] = options_of(2);
const OPTIONS_GQA5_P32: [&str; 17] = options_of(3);
const OPTIONS_GQA8_P32: [&str; 17] = options_of(4);

/// One row's `KernelSig`, which is the same for all five but for the symbol.
///
/// `name` and `symbol` are the same string, for `unit!`'s reason: `name` is
/// `emit_c_shim`'s `pie_k_` stem and no device row is shimmed, so the only
/// honest answer is the symbol itself.
const fn row_sig(symbol: &'static str) -> KernelSig {
    KernelSig {
        name: symbol,
        symbol,
        file: Some(FILE),
        operands: OPERANDS,
        ..crate::x::contract::SIG_BASE
    }
}

/// The five device entry points, by the name the `-Dkernel_mha=…` rename
/// gives them.
///
/// Taken from [`XQA_LATTICE`]`[i].entry`, which is the table that also
/// states which archive `.cu` each `#define` block came from.
static ROW_SIGS: [KernelSig; 5] = [
    row_sig(XQA_LATTICE[0].entry),
    row_sig(XQA_LATTICE[1].entry),
    row_sig(XQA_LATTICE[2].entry),
    row_sig(XQA_LATTICE[3].entry),
    row_sig(XQA_LATTICE[4].entry),
];

/// Each member's one row.
///
/// # `template_path` carries a leading `::` and that is load-bearing
///
/// [`DeviceKernel::instantiation`](crate::device::DeviceKernel::instantiation)
/// prefixes `::pie_cuda_driver::kernels::` to any path that does not begin
/// with `::`. XQA's entry points are not in that namespace and are not in
/// any namespace: `mha.h:274` defines `CUBIN_EXPORT` as `extern "C"` under
/// `GENERATE_CUBIN`, which every member sets, so each `__global__` is a
/// C-linkage symbol at global scope and its lowered name is its source name.
/// Without the `::` the JIT would ask NVRTC for a name expression naming a
/// function that does not exist.
///
/// Every row is [`DeviceKernel::PLAIN`]: `kernel_mha` is not a template, so
/// there is nothing for `elem` to pick. `families::attn::MLA_NAIVE_ROWS`
/// makes the same argument at length.
///
/// [`DeviceKernel::PLAIN`]: crate::device::DeviceKernel::PLAIN
static ROWS: [[DeviceKernel; 1]; 5] = [
    [DeviceKernel {
        sig: &ROW_SIGS[0],
        template_path: "::kernel_mha_xqa_gqa2_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[1],
        template_path: "::kernel_mha_xqa_gqa2_bf16_p16_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[2],
        template_path: "::kernel_mha_xqa_gqa4_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[3],
        template_path: "::kernel_mha_xqa_gqa5_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[4],
        template_path: "::kernel_mha_xqa_gqa8_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
];

/// `attn/attention_xqa_mha_gqa2_p32` — head group 2 at a 32-token page.
///
/// Qwen3-0.6B and Qwen3-1.7B shapes; see [`XQA_LATTICE`]`[0].because`.
pub const XQA_GQA2_P32: Unit = Unit {
    name: XQA_LATTICE[0].unit,
    root: XQA_ROOT,
    rows: &ROWS[0],
    options: &OPTIONS_GQA2_P32,
};

/// `attn/attention_xqa_mha_gqa2_p16` — the same head group at a 16-token
/// page.
///
/// Unreachable today and kept deliberately: `xqa_decode_page_bucket` never
/// returns 16 because `xqa_gqa2_page16_enabled()` returns false, and
/// deleting the member would make flipping that flag a port rather than a
/// flag. [`XQA_LATTICE`]`[1].because` is the full argument.
pub const XQA_GQA2_P16: Unit = Unit {
    name: XQA_LATTICE[1].unit,
    root: XQA_ROOT,
    rows: &ROWS[1],
    options: &OPTIONS_GQA2_P16,
};

/// `attn/attention_xqa_mha_gqa4_p32` — head group 4, Qwen3-4B and Qwen3-8B
/// shapes.
pub const XQA_GQA4_P32: Unit = Unit {
    name: XQA_LATTICE[2].unit,
    root: XQA_ROOT,
    rows: &ROWS[2],
    options: &OPTIONS_GQA4_P32,
};

/// `attn/attention_xqa_mha_gqa5_p32` — head group 5, the ratio
/// Llama-3.1-8B-shaped models use.
pub const XQA_GQA5_P32: Unit = Unit {
    name: XQA_LATTICE[3].unit,
    root: XQA_ROOT,
    rows: &ROWS[3],
    options: &OPTIONS_GQA5_P32,
};

/// `attn/attention_xqa_mha_gqa8_p32` — head group 8, Qwen3-32B and
/// Llama-70B-style shapes.
///
/// Its archive launcher forwarded to the Hopper member on `major >= 9`.
/// That member is not enrolled ([`UNITS`] says why), so this unit is the
/// only gqa8 compile the JIT has.
pub const XQA_GQA8_P32: Unit = Unit {
    name: XQA_LATTICE[4].unit,
    root: XQA_ROOT,
    rows: &ROWS[4],
    options: &OPTIONS_GQA8_P32,
};

/// The five units this family compiles.
///
/// # Five and not six
///
/// [`XQA_LATTICE`]`[5]` — `attn/attention_xqa_mha_gqa8_p32_sm90` — is NOT
/// enrolled, and its own entry in that table is where the measurement lives:
/// at `compute_90a` it stops on `std::pair` in DEVICE code
/// (`xqa/mha_sm90.cu:1980`, twelve diagnostics cascading from one line),
/// `csrc/shim/cuda.h` declares no `CUtensorMap` or
/// `CUtensorMapDataType_enum` for `xqa/tensorMap.h` to declare against, and
/// the archive unit compiled `<xqa/tensorMap.cpp>` — HOST code driving
/// `cuTensorMapEncodeTiled`, a second and larger port than `launchMHA` was.
/// `csrc/vendor/xqa/` deliberately does not carry `mha_sm90.cu` or
/// `tensorMap.{h,cpp}` for the same reason.
///
/// **The consequence, stated so it is not discovered.**
/// `driver-cuda/src/fire/xqa.rs::XqaMember::pick` REFUSES the 8-at-32 pair on
/// `major >= 9`, and that gate was written against this paragraph. The split
/// there is worth knowing from this side: `XqaMember::dispatch` still answers
/// `Gqa8Page32Sm90` because it is the record of what the archive did, and
/// `pick` filters it out through `XqaMember::enrolled`, a five-of-six mirror
/// of [`UNITS`] maintained by hand because `kernels-cuda-new` has no
/// dependency on `driver-cuda`. **Flipping one `enrolled` arm to `true` is
/// the whole re-enablement**, so a Hopper unit landing in [`UNITS`] needs one
/// edit there and no dispatch surgery.
///
/// Its `entry()` is `kernel_mha_xqa_gqa8_sm90_bf16_p32_h128`, which no unit
/// hosts; before the gate, firing it would have reached `x::fire::fire` and
/// panicked naming the symbol. The refusal is now a `DecodeDecline::MemberNotEnrolled`
/// at the top of the plan instead, which is the same fact delivered before
/// the JIT rather than inside it. Nothing reaches either today —
/// `plan_decode` has no caller and every deployment in the tree states
/// `xqa_decode: false`.
///
/// **What it costs, said plainly**: `decode_supported` ends in `major >= 9`,
/// so refusing ratio 8 there means head-group ratio 8 has no XQA decode on
/// any device until this member is enrolled. Ratios 2, 4 and 5 are
/// unaffected. The alternative was falling back to the Ampere/Ada body on
/// devices the archive never ran it on, which is a wrong answer wearing the
/// shape of a supported configuration.
///
/// # `unit!` could not have written these, and the exact patch that would
///
/// `unit!` hardcodes `options: &[]` (`x/macros.rs:239`) and has no clause to
/// state one. Five units over ONE root whose only difference is a `-D` set
/// are precisely the case it cannot express. The grammar addition is one
/// optional clause on the unit header and one field:
///
/// ```ignore
/// //  x/macros.rs, the `unit!` header line:
/// -   unit $unit:ident = $uname:literal, text = $utext:expr, file = $ufile:literal;
/// +   unit $unit:ident = $uname:literal, text = $utext:expr, file = $ufile:literal
/// +       $(, options = $uopts:expr)? ;
///
/// //  and in the `Unit` it builds:
/// -           options: &[],
/// +           options: $crate::unit!(@opts $($uopts)?),
///
/// //  with two terminal arms:
/// +   (@opts) => { &[] };
/// +   (@opts $uopts:expr) => { $uopts };
/// ```
///
/// That patch belongs to whoever owns `x/macros.rs` and is offered rather
/// than taken. **It would not have been used here even if it existed**, and
/// the reason is worth stating so the patch is judged on its own case and
/// not on this one: `unit!` emits a `raw::` module per invocation, so six
/// invocations produce six byte-identical `kernel_mha` launchers in six
/// modules, and a caller choosing among five members would have to name the
/// MODULE and the SYMBOL to say one thing. One launcher taking a `symbol`
/// is what this family wants, and it is what `x::fire::fire`'s own signature
/// is shaped for.
pub static UNITS: &[Unit] =
    &[XQA_GQA2_P32, XQA_GQA2_P16, XQA_GQA4_P32, XQA_GQA5_P32, XQA_GQA8_P32];

contract! {
    /// XQA's paged decode, the whole fire.
    ///
    /// One symbol over five compiled members: the host program picks by
    /// `(head_group_ratio, page_size, major)` and fires once. The archive
    /// spelled that choice as six translation units and a C++ dispatcher in
    /// `attn/attention_xqa.cu`; the choice now lives in
    /// `driver-cuda/src/fire/xqa.rs::XqaMember::pick` and the compile lives
    /// in [`UNITS`].
    ///
    /// # `whole` and `lacks Scores` are the deleted row's, verbatim
    ///
    /// `table::attn`'s `xqa_decode` row said: *"its prepare is fire-wide
    /// (R-shaped), so the kernel cannot be given a row window — `whole`. And
    /// no capture variant of it exists, so it cannot publish scores —
    /// `lacks Scores`. Both are hand-written rules today: the first is the
    /// model body's `window_one && c.xqa_decode` test, the second a C++
    /// throw."*
    ///
    /// Both survive as `Contract` fields. The C++ throw does not survive,
    /// because `attn/attention_xqa.cu` is deleted — so `lacks: &[Cap::Scores]`
    /// is now the ONLY thing in the tree that refuses a scores capture on
    /// this symbol, and it refuses it at model load rather than at the fire,
    /// which is the better of the two.
    ///
    /// # `needs: Prepare::FireWide`
    ///
    /// `model-compiler` reads this to decide what to stage. It is unchanged
    /// from the row and is what makes `whole` consistent: a fire-wide
    /// prepare cannot be handed a row window, so a windowed dispatch of this
    /// symbol is refused before anything is staged.
    XQA_DECODE_BF16_PREPARED = "attn::attention_xqa_decode_bf16_prepared" as xqa_decode {
        whole: true,
        needs: Prepare::FireWide,
        lacks: &[Cap::Scores],
    }
}

bind! {
    // NOT A BIND, AND THE REASON IS THREE FACTS ABOUT `Cx` RATHER THAN ONE
    // ABOUT XQA. The host program EXISTS and is complete:
    // `driver-cuda/src/fire/xqa.rs` is 959 lines and `plan_decode` returns a
    // fully-populated `XqaLaunch` — grid, block, smem, semaphore count, the
    // three kv strides and the member. What it cannot become here is a
    // `bind!` body, for reasons a future reader should be able to check
    // rather than take:
    //
    //  1. IT IS IN THE OTHER CRATE. `driver-cuda` depends on
    //     `kernels-cuda-new`, so a body here cannot call `plan_decode`. That
    //     is a fact about the dependency direction and not a difficulty:
    //     `fire/xqa.rs` is a DRIVER-side host program, the same shape as
    //     `fire/mla_naive.rs`, and both are waiting on a caller rather than
    //     on a bind.
    //
    //  2. TWO OF ITS INPUTS ARE FACTS `Cx` DOES NOT STATE. The deleted row
    //     sourced `workspace: AttentionWorkspaceView <- Source::Attn(..)`
    //     and `sm_scale: F32 <- Source::Attn("sm_scale")`. `Cx` has
    //     `kv_layer`, `plan`, `head_dim`, `num_q_heads`, `num_kv_heads`,
    //     `arg_in`, `arg_out` and `aux` — and no query for either of those
    //     two. THE EXACT PATCH, for whoever owns `x/cx.rs`, is two `query!`
    //     lines and their `Facts` methods:
    //
    //         query!(
    //             /// The attention workspace this fire was given.
    //             attn_workspace -> AttentionWorkspaceView, "the attention workspace"
    //         );
    //         query!(
    //             /// The softmax scale this fire was planned with.
    //             sm_scale -> f32, "the attention softmax scale"
    //         );
    //
    //  3. THE SEMAPHORE BANK MUST BE ZEROED BEFORE THE LAUNCH, and a bind
    //     body has no device API — `bind!`'s own doc says so: *"no `&mut` in
    //     scope, no device API and no allocator"*. `plan_decode` reports
    //     `semaphore_count` and states in its own doc that the
    //     `cudaMemsetAsync` is the CALLER's, which is a seam a driver op has
    //     and a bind body does not.
    //
    // Landing 2 and 3 turns this arm into a bind and nothing else in this
    // file changes. Until then `Route::Unbound` refuses the symbol at MODEL
    // LOAD with the sentence below, which is where a missing capability
    // should surface — and it refuses nothing today, because every
    // deployment in the tree states `xqa_decode: false` and
    // `model-compiler`'s `attention_xqa_decode` is emitted only under
    // `c.xqa_decode`.
    XQA_DECODE_BF16_PREPARED => { none: "xqa decode's host program is \
        `driver-cuda/src/fire/xqa.rs::plan_decode`, a driver-side program in \
        the crate above this one: `Cx` states neither the attention \
        workspace nor `sm_scale`, and the semaphore bank's zeroing is a \
        device call no bind body may make" },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five enrolled units and the lattice describe the same five.
    ///
    /// `name` and `options` are READ from `XQA_LATTICE` and cannot drift;
    /// `template_path` is a literal and can, because there is no `const`
    /// string concatenation to build `"::" + entry` with. This is the check
    /// that closes that one gap, and it is the only hand-copied string in
    /// the enrolment.
    #[test]
    fn every_row_names_the_lattice_entry_it_compiles() {
        assert_eq!(UNITS.len(), 5, "the sixth member is deliberately not enrolled");
        for (i, unit) in UNITS.iter().enumerate() {
            assert_eq!(unit.name, XQA_LATTICE[i].unit);
            assert_eq!(unit.rows.len(), 1, "one `__global__` per option set");
            assert_eq!(
                unit.rows[0].sig.symbol, XQA_LATTICE[i].entry,
                "the row's symbol is not the name `-Dkernel_mha=` gives it"
            );
            assert_eq!(
                unit.rows[0].template_path,
                format!("::{}", XQA_LATTICE[i].entry),
                "`template_path` must be the entry at GLOBAL scope: `CUBIN_EXPORT` is \
                 `extern \"C\"` under `GENERATE_CUBIN`, and a path without the leading \
                 `::` is qualified into `::pie_cuda_driver::kernels::`"
            );
            assert_eq!(unit.options.len(), 17, "thirteen shared and four its own");
            assert_eq!(&unit.options[..13], XQA_COMMON_OPTIONS);
            assert_eq!(&unit.options[13..], XQA_LATTICE[i].options);
        }
    }

    /// The Hopper member is absent from every list, not merely from `UNITS`.
    ///
    /// An enrolment that left a `Unit` const behind would be a compile
    /// nobody can reach and a name a grep would find — which is how a reader
    /// concludes a member is enrolled when it is not.
    #[test]
    fn the_sm90_member_is_enrolled_nowhere() {
        let sm90 = XQA_LATTICE[5].entry;
        assert!(
            UNITS.iter().all(|u| u.rows.iter().all(|r| r.sig.symbol != sm90)),
            "the sm90 member is enrolled; `UNITS`'s doc says why it must not be"
        );
        assert!(
            crate::unit::unit_of(sm90).is_none(),
            "some unit hosts the sm90 entry -- `XqaMember::pick` would then launch a \
             kernel that was never measured to compile"
        );
    }

    /// Fifteen parameters, and the three spellings agree on the count.
    #[test]
    fn the_parameter_list_is_one_list() {
        assert_eq!(PARAMS.len(), 15, "`mha.cuh:2783` under the lattice's own defines");
        assert_eq!(PARAMS.len(), OPERANDS.len());
        assert_eq!(OPERANDS[6].name, "cache_list", "the by-value aggregate is the seventh");
        assert_eq!(
            PARAMS[6],
            <KvCacheList as crate::x::ByValue>::LAYOUT.cpp,
            "the parameter's spelling and the asserted layout's must be one string"
        );
    }
}
