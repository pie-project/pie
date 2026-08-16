//! ② KERNEL SIGNATURES — the vocabulary (`.wiki/tart/dsl.md` ②).
//!
//! A [`KernelSig`] is one symbol's contract, once per symbol; a wrapper name
//! like `_region`/`_planned`/`_capture` encodes the DISPATCH CONTEXT, which
//! belongs to the call site rather than to the kernel.
//!
//! The rows themselves live in each backend crate beside the `.cuh`/`.metal`
//! they describe. Both tables are written in the same words and neither
//! backend owns those words, so the words are here.

pub mod bind;
pub mod routine;

// The facts a launcher can name, as types rather than as words. One fact,
// one type, one spelling of it in the whole tree.
pub mod keys;

// The shader backends' operand vocabulary, written once: closed and identical
// in metal, vulkan and wgpu, generic over `shader::ShaderValue`. Not CUDA's.
pub mod shader;

pub use routine::{Arg, Backend, Env, KernelFn, Provenance, Refusal, Routine};
// `Layout` is the ALLOCATION's shape; `Stride` is a row pitch deliberately not
// an `i32`.
pub use routine::{Elem, Layout, Region, Stride};
// The CUDA plane's operand wrappers, which carry a whole rectangle. The shader
// planes' `InSlot`/`OutSlot`/`InRow`/`OutRow` carry an address instead and live
// in `routine` beside these; the two vocabularies are disjoint in use.
pub use routine::{Aux, Bank, In, InOut, Out, Param, ParamF32, Unbound, Weight};

/// A capability a seam may ask of the kernel covering its rows. Named after
/// the seam vocabulary (`.wiki/tart/dsl.md` ①), because that is what a
/// `lacks` line refuses to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    /// The attention scores, published for an `attn.out` observer.
    Scores,
    /// The page-mask sink an `attn.q` tap writes.
    PageMaskSink,
}

/// How a kernel turns a rectangle into a thread grid.
///
/// A variant is a **shape of launch, not a kernel**: `Elementwise` serves every
/// 256-wide pointwise pass and `PerHead` both the q/k/v split and the KV
/// append, so a new kernel that launches like an existing one costs nothing.
/// This is data: the arithmetic each variant names stays in the driver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LaunchRule {
    /// The row has not said. A backend must REFUSE rather than guess: a
    /// guessed grid runs a kernel over the wrong extent, which no hardware
    /// reports. Same meaning an operand's absent `Source` has.
    #[default]
    Unstated,
    /// Affine GEMV: four outputs per simdgroup, two simdgroups per
    /// threadgroup, rounded up.
    Qmv,
    /// Row-wise norm: one threadgroup per row, four elements per thread,
    /// capped at the widest threadgroup the backend allows.
    Rms,
    /// Rope: half the rotary channels, per head.
    Rope,
    /// Pointwise over one row, 256-wide — residual adds, embeddings, silu-mul.
    /// Rows stack flat: `width * rows` threads on one axis.
    Elementwise,
    /// Pointwise with the row on its own grid axis rather than stacked flat —
    /// what a gather whose rows are not contiguous needs
    /// (`embed_gather_mb`). One rule apart from [`LaunchRule::Elementwise`]:
    /// the two agree at one row and disagree above it.
    ElementwiseRows,
    /// One threadgroup per head, `head_dim` wide — the q/k/v split, the KV
    /// append. The row is the third grid axis.
    PerHead,
    /// Single-pass decode attention: one 1024-thread threadgroup per query
    /// head, rows on the second axis.
    SdpaVector,
    /// Tiled prefill attention: the same 1024-thread threadgroup per query
    /// head, but the second axis counts TILES of query rows rather than rows.
    SdpaTiled,
    /// The same 32-row tile as [`LaunchRule::SdpaTiled`], on a threadgroup of
    /// 128 threads rather than 1024.
    SdpaMma,
    /// Pointwise over every head's channels, 256-wide.
    PerHeadElementwise,
    /// Gated norm over the value heads.
    GatedRms,
    /// One threadgroup as wide as the expert count PER ROW — the router's
    /// top-k, which `route.metal` indexes with `tgid.y`.
    RouterLane,
    /// ONE threadgroup as wide as the expert count, whatever the row count —
    /// the counting sort, which reduces across all `(row, slot)` pairs
    /// through threadgroup atomics and stripes them over its own lanes.
    RouterSort,
    /// One threadgroup per row, as wide as the row, capped at 256.
    RouteRows,
    /// Routed GEMV: [`LaunchRule::Qmv`] per row, per expert slot.
    RoutedQmv,
    /// Pointwise over the launch's INPUT width with the row on its own axis —
    /// a statement that reads one packed buffer and writes several, where the
    /// output widths are each a fraction of the work. The QKV split is the
    /// case: three outputs, and the grid has to cover their sum.
    SplitPacked,
    /// Affine GEMM: the batched projection, tiled over rows and columns.
    Qmm,
    /// One block per (row, head), 128 wide, with **two head-wide float arrays
    /// staged in shared memory** — the gated-delta recurrence and its chunked
    /// prefill, which read `q` and `k` out of `2 · head_dim` floats the launch
    /// hands them.
    ///
    /// Too LITTLE dynamic shared memory is not a launch failure: the kernels
    /// take their second array as `smem + head_dim`, so a short allocation
    /// writes `k` into what the next block is reading.
    RecurrentScan,
    /// One block per row, a fixed 256 wide, no shared memory — a scatter whose
    /// body strides its own row.
    PerRow,
    /// One block per COLUMN, 64 wide, the row axis walked inside the block —
    /// the short causal convolution, which carries a per-channel state across
    /// the tokens it convolves.
    PerChannel,
    /// Flat pointwise over the launch's INPUT extent — `rows · in_width`
    /// elements, the row folded into the index.
    ElementwiseIn,
    /// One block per row, 256 wide, with **one float of shared scratch per row
    /// of the rectangle** — a causal score buffer over the fire's own tokens.
    RowScores,
    /// One block per row PER HEAD — `rows · (width / head_dim)` blocks of the
    /// reduction's own width, falling back to `rows` when the statement named
    /// no head width.
    RowsPerHead,
    /// `ceil(rows / 256)` blocks of 256 — ONE THREAD per row, not one block.
    RowsFlat,
    /// A grid-stride slab: `min(ceil(units / 256), 1024)` blocks of 256, where
    /// `units` is the vectorised element count. The cap is the contract: the
    /// kernel strides its own extent, and one without that loop launched this
    /// way computes a prefix and reports success.
    Slab,
    /// A 16x16 block over a rectangle — `ceil(width / 16)` by `ceil(rows / 16)`
    /// blocks, the only rule here whose block is not one-dimensional.
    Tile16,
    /// One warp per (head, row), heads on `grid.y` and rows on `grid.z` —
    /// `dim3(1, heads, rows)` at 32 threads.
    AxialRope,
    /// The recurrence tiled by warps over the VALUE width — `dim3(rows, heads,
    /// ceil(value_width / 4))` at 128 threads, nothing shared.
    WarpTiledScan,
    /// One block per row, **128** wide, no shared memory — the same grid
    /// [`LaunchRule::PerRow`] states at half its block. The width is a
    /// NUMERICS contract wherever a kernel folds warp partials serially.
    PerRowNarrow,
    /// The reference paged attention's PREFILL grid —
    /// `dim3(requests, rows, q_heads)` at 128 threads, with
    /// `(head_dim + 128) * sizeof(float)` of DYNAMIC shared memory.
    PagedScores,
    /// The same kernel family's DECODE grid — `dim3(rows, q_heads)` at 128
    /// threads with the same `(head_dim + 128) * sizeof(float)`.
    PagedScoresDecode,
    /// MLA's fused prepare — `dim3(rows, 1 + ceil(q_heads / heads_per_block))`
    /// at 256 threads, nothing shared.
    MlaPrepare,
    /// One block per (row, packed head) — `dim3(rows, q_heads + kv_heads)` at
    /// 256 threads, nothing shared.
    RowsPackedHeads,
    /// [`LaunchRule::RowsPackedHeads`] at **128** threads. `BLOCK` is the
    /// template argument sizing the kernel's `__shared__` array, so a 256-wide
    /// launch of the `<128>` instantiation reads slots nothing wrote.
    RowsPackedHeadsNarrow,
    /// One WARP per (row, packed head), flattened —
    /// `ceil(rows * (q_heads + kv_heads) / (256 / 32))` blocks of 256.
    WarpPackedHeads,
    /// [`LaunchRule::RoutedQmv`]'s two axes TRANSPOSED —
    /// `dim3(ceil(width / 8), rows * experts_per_token)` at 256 threads.
    RoutedQmvTransposed,
    /// A third grid axis over an ALTUP STREAM count — `dim3(rows, streams,
    /// ceil(width / streams / 128))` at 128 threads.
    AltUpStreams,
    /// [`LaunchRule::RoutedQmv`]'s two axes at a **quad** tile, over a
    /// **stacked** output — `dim3(rows * experts_per_token,
    /// ceil((width / experts_per_token) / 16))` at **128** threads.
    RoutedQmvQuad,
    /// **Exactly one block** of 256 threads, whatever the rectangle — the
    /// grid is a literal `1` the host wrote and not a quotient.
    Single,
    /// [`LaunchRule::Single`] at ONE WARP — `<<<1, 32>>>`.
    SingleWarp,
    /// One block per REQUEST, 256 wide, nothing shared — [`LaunchRule::PerRow`]'s
    /// launch over [`crate::LaunchRule`]'s other row-shaped axis. A request
    /// count is not a row count; the two coincide only on a pure decode.
    PerRequest,
}

impl LaunchRule {
    /// Every variant, so a caller can enumerate the vocabulary rather than
    /// remember it.
    pub const ALL: &'static [Self] = &[
        Self::Unstated,
        Self::Qmv,
        Self::Rms,
        Self::Rope,
        Self::Elementwise,
        Self::ElementwiseRows,
        Self::PerHead,
        Self::SdpaVector,
        Self::SdpaTiled,
        Self::SdpaMma,
        Self::PerHeadElementwise,
        Self::GatedRms,
        Self::RouterLane,
        Self::RouterSort,
        Self::RouteRows,
        Self::RoutedQmv,
        Self::SplitPacked,
        Self::Qmm,
        Self::RecurrentScan,
        Self::PerRow,
        Self::PerChannel,
        Self::ElementwiseIn,
        Self::RowScores,
        Self::RowsPerHead,
        Self::RowsFlat,
        Self::Slab,
        Self::Tile16,
        Self::AxialRope,
        Self::WarpTiledScan,
        Self::PerRowNarrow,
        Self::PagedScores,
        Self::PagedScoresDecode,
        Self::MlaPrepare,
        Self::RowsPackedHeads,
        Self::RowsPackedHeadsNarrow,
        Self::WarpPackedHeads,
        Self::RoutedQmvTransposed,
        Self::AltUpStreams,
        Self::RoutedQmvQuad,
        Self::Single,
        Self::SingleWarp,
        Self::PerRequest,
    ];

    /// A discriminant, used only to prove [`Self::ALL`] is complete.
    const fn index(self) -> usize {
        match self {
            Self::Unstated => 0,
            Self::Qmv => 1,
            Self::Rms => 2,
            Self::Rope => 3,
            Self::Elementwise => 4,
            Self::ElementwiseRows => 5,
            Self::PerHead => 6,
            Self::SdpaVector => 7,
            Self::SdpaTiled => 8,
            Self::SdpaMma => 9,
            Self::PerHeadElementwise => 10,
            Self::GatedRms => 11,
            Self::RouterLane => 12,
            Self::RouterSort => 13,
            Self::RouteRows => 14,
            Self::RoutedQmv => 15,
            Self::SplitPacked => 16,
            Self::Qmm => 17,
            Self::RecurrentScan => 18,
            Self::PerRow => 19,
            Self::PerChannel => 20,
            Self::ElementwiseIn => 21,
            Self::RowScores => 22,
            Self::RowsPerHead => 23,
            Self::RowsFlat => 24,
            Self::Slab => 25,
            Self::Tile16 => 26,
            Self::AxialRope => 27,
            Self::WarpTiledScan => 28,
            Self::PerRowNarrow => 29,
            Self::PagedScores => 30,
            Self::PagedScoresDecode => 31,
            Self::MlaPrepare => 32,
            Self::RowsPackedHeads => 33,
            Self::RowsPackedHeadsNarrow => 34,
            Self::WarpPackedHeads => 35,
            Self::RoutedQmvTransposed => 36,
            Self::AltUpStreams => 37,
            Self::RoutedQmvQuad => 38,
            Self::Single => 39,
            Self::SingleWarp => 40,
            Self::PerRequest => 41,
        }
    }
}

// `ALL` is complete and in discriminant order, checked at COMPILE time: a new
// variant makes `index` non-exhaustive, and forgetting to list it here makes
// this assertion fail. Neither can be missed by a reviewer.
const _: () = {
    assert!(LaunchRule::ALL.len() == 42);
    let mut i = 0;
    while i < LaunchRule::ALL.len() {
        assert!(LaunchRule::ALL[i].index() == i);
        i += 1;
    }
};

/// One point of one instantiation axis, and the text it contributes to a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    /// What varies. Prose, for a reader of the table; the matcher never reads
    /// it.
    pub what: &'static str,
    /// The suffixes this axis can contribute, in the order a name spells them.
    /// Exactly one is present in any entrypoint the axis reaches.
    ///
    /// A point MAY be `""`, for an axis whose default specialisation adds no
    /// text. Two orderings are load-bearing, and [`KernelSig::covers`] checks
    /// them rather than asserting them:
    ///
    /// * the empty point goes LAST, because matching is first-wins and an
    ///   empty suffix matches everything;
    /// * a longer point goes before a shorter one it ends with (`_p32_sg8`
    ///   before `_p32`), for the same reason.
    pub points: &'static [&'static str],
}

/// What one operand of a launcher is, in words neither backend owns.
///
/// Deliberately SMALL, and not a type system: `q`, `k` and `k_pages` are all
/// [`Ty::BufMut`], because how a kernel reads its own tensor is its business.
/// This describes what a CALLER must know to place an argument -- word width,
/// whether the callee writes through it, and whether it may be absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// An opaque device buffer the launcher may WRITE through (`void*`).
    BufMut,
    /// An opaque device buffer the launcher only reads (`const void*`).
    Buf,
    /// A read-only device array of `i32` — positions, and the like.
    I32s,
    /// A read-only device array of `i64`. One row needs it — kimi_k3's
    /// hash routing reads a `[vocab, K]` token-to-expert table — and it is
    /// its own kind rather than a `Buf` because `const void*` and
    /// `const int64_t*` are both pointers: only the DECLARED width makes the
    /// substitution a compile error instead of a stride bug.
    I64s,
    /// A read-only device array of `u32` — the CSR/indptr family.
    U32s,
    /// A read-only device array of `u8` — the per-row validity masks.
    U8s,
    /// A device array of `f32` the launcher WRITES — the built tables.
    F32sMut,
    /// A read-only device array of `f32` — softmax scales, sinks, biases.
    F32s,
    /// A device array of `i32` the launcher WRITES.
    I32sMut,
    /// A device array of `u32` the launcher WRITES — the indptrs a plan
    /// builds, as opposed to the ones a dispatch reads.
    U32sMut,
    /// A device array of `u8` the launcher WRITES — the packed masks.
    U8sMut,
    /// A host scalar.
    I32,
    /// A host scalar. Distinct from [`Ty::I32`] because the headers spell it
    /// `std::uint32_t`, and a mirror that guessed `i32` would be a silent
    /// sign bug on any value above 2^31 rather than a compile error.
    U32,
    /// ── POINTER ARRAYS ────────────────────────────────────────────
    ///
    /// A batched or grouped launch is handed an ARRAY of buffers rather
    /// than one, and two independent things can be const about it: the
    /// array itself, and what its entries point at. Four kinds, because
    /// all four combinations are live and C++ will silently accept
    /// three of them where a fourth was meant.

    /// `const void* const*` — an array the launcher reads, of buffers it
    /// reads.
    BufArray,
    /// `void* const*` — an array the launcher reads, of buffers it
    /// writes. Every batched GEMM's destination list.
    BufArrayMut,
    /// `const void**` — an array the launcher WRITES, of buffers to be
    /// read. What a pointer builder fills for a later batched call.
    BufArrayOut,
    /// `void**` — an array the launcher writes, of buffers to be
    /// written. The destination half of the same builder.
    BufArrayOutMut,
    /// `const std::uint8_t* const*` — the same shape as [`Ty::BufArray`]
    /// with the element type spelled, which the MXFP4 banks need: their
    /// entries are packed nibbles and block scales, and a `void*` row
    /// would let a bf16 bank through.
    U8Array,
    /// A read-only device array of `u16`. Two launchers spell their
    /// bf16 buffers this way rather than as `void*`, and the pilot will
    /// not let a `Buf` row stand in: the pointer types differ, so the
    /// forward is a conversion C++ refuses.
    U16s,
    /// A device array of `u16` the launcher WRITES.
    U16sMut,
    /// A read-only device array of `i8` — an int8 LM head's weights.
    I8s,
    /// A device array of `i8` the launcher WRITES — an INT8 quantiser's
    /// destination.
    I8sMut,
    /// ── THE TWO SIXTEEN-BIT FORMATS, NAMED APART ──────────────────
    ///
    /// Distinct kinds because `pie::bf16` and `pie::f16` are distinct STRUCTS
    /// and not typedefs: as typedefs a row that swapped them would have
    /// nothing to swap. [`Ty::U16s`] is the collapsed spelling -- the WIDTH and
    /// nothing about the format -- so it stands in for neither.
    ///
    /// `Bf16s`: a read-only device array of `pie::bf16`.
    Bf16s,
    /// A read-only device array of `pie::f16`. See [`Ty::Bf16s`].
    F16s,
    /// ── AND THE SAME TWO, WRITTEN ─────────────────────────────────
    ///
    /// A device array of `pie::bf16` the launcher WRITES.
    Bf16sMut,
    /// A device array of `pie::f16` the launcher WRITES. See
    /// [`Ty::Bf16sMut`].
    F16sMut,
    /// `const std::int32_t* const*` — an array of int32 buffers the
    /// launcher reads. The WNA16 expert banks are packed as `int32`
    /// words rather than opaque bytes, and a `BufArray` row would hand
    /// their kernel a void array it dereferences as int32.
    I32Array,
    /// `MoeActivation` — which activation a fused MoE leg runs. A
    /// caller-stated enum like [`Ty::Dtype`], and for the same reason:
    /// the declaration knows which, and a driver that inferred it from
    /// a config would be choosing.
    MoeActivation,
    /// `Mxfp4RowSelect` — which half of an interleaved MXFP4 bank a
    /// repack reads.
    Mxfp4RowSelect,
    /// The NVLink P2P all-reduce instance a fused collective is issued
    /// through — `kernels::comm::CustomAllReduce*`.
    ///
    /// A HANDLE, like [`Ty::Stream`]: the arm is given it and never asks. It
    /// exists because the fused landing was a METHOD, and a method has no
    /// address the generated ABI can forward to.
    CustomAllReduce,
    /// The element type a buffer is stored in — `DType`, a
    /// `std::uint8_t`-backed enum class.
    ///
    /// Its own kind and not [`Ty::U32`]: widening would fail to compile, which
    /// is the right answer for the wrong reason. Spelling it means the shim
    /// forwards the enum the header declares.
    Dtype,
    /// `pie::attn::KvScheme` — which quantisation a paged KV bank is
    /// stored under (`enum class … : ::std::uint8_t`,
    /// `attn/attention_naive_paged.cuh:187`).
    KvScheme,
    /// `pie::attn::KvDType` — which element type a paged KV bank stores
    /// (`enum class … : ::std::uint8_t`,
    /// `attn/attention_naive_paged.cuh:198`). See [`Ty::KvScheme`], whose
    /// parameter it follows in both kernels.
    ///
    /// Not [`Ty::Dtype`], which is `::pie_cuda_driver::DType` — a different
    /// enumeration with a different member list, declared in a different
    /// header. `naive_paged_attn` takes this one and the two do not convert.
    KvDType,
    /// `__nv_fp8_interpretation_t` — WHICH fp8 encoding a byte-wide KV page
    /// holds, `__NV_E4M3` or `__NV_E5M2`. CUDA's own, from `<cuda_fp8.h>`.
    ///
    /// FOUR BYTES AND NOT ONE. [`Ty::KvScheme`] and [`Ty::KvDType`] cross as a
    /// byte because their C++ states the underlying type; this is an unscoped
    /// C enum with no fixed one, so its width is a toolchain observation and
    /// not a promise the header makes -- and a one-byte cell against a
    /// four-byte parameter mis-marshals every argument after it. Hence
    /// asserted rather than assumed, by a `static_assert` the shim emits.
    Fp8Kind,
    /// A host scalar spelled `long long` — the recurrent state's slot
    /// stride, which is an ELEMENT count into a multi-gigabyte arena and
    /// so was widened deliberately. Its own kind for `Ty::U32`'s reason:
    /// a mirror that guessed `int` is a silent truncation, not an error.
    I64,
    /// A host byte count, spelled `std::size_t`. Its width is the platform's,
    /// which is why it is not [`Ty::U32`] widened by hand.
    Usize,
    /// A scalar that rides the PRECEDING packed struct rather than a buffer of
    /// its own.
    ///
    /// `RowGatherParams` packs width and count into one buffer; the count is
    /// the struct's second FIELD. A row lists it so the driver supplies the
    /// value, and this says "append it to the scalars, bind nothing".
    InPacked,
    /// A host scalar.
    F32,
    /// A host flag. Spelled `bool` in C++, so it is ONE byte and not `i32`;
    /// a binding that gets this wrong is a silent stack-layout bug rather
    /// than a compile error, which is why it is its own kind.
    Bool,
    /// The stream the launch is ordered on.
    Stream,
    /// The cuBLAS handle a library-issued launch is ordered through.
    ///
    /// Taken instead of a stream, the stream being set on the handle. A row
    /// may NOT state one: a handle is the service's, not the statement's, and
    /// `abi::device_typecheck` refuses a row that names it.
    CublasHandle,

    // ---- The struct-shaped operands. ----
    //
    // The passing mode is the launcher's choice, so it is recorded in `cpp()`
    // rather than spelled at each use. What separates the modes is what the
    // RUST side can say: a `#[repr(C)]` mirror by value, the same mirror behind
    // a `*const`, or `*const c_void` where the C++ never defines the type.
    //
    // A row cannot pick wrong by accident: the shim initialises a function
    // pointer, and a function pointer takes no conversions.
    /// The attention scratch, by value — [`Ty::Usize`]-sized buffers the
    /// driver owns and the kernels only read out of. Five words, so it is
    /// cheaper to copy than to chase.
    AttentionWorkspaceView,
    /// One layer's paged KV, by value.
    KvCacheLayerView,
    /// One layer's paged MLA cache, by value.
    MlaCacheLayerView,
    /// FlashInfer's decode plan, by `const&`. **Incomplete** in the header —
    /// `struct DecodePlanCache;` and nothing more — so this is a handle the
    /// driver holds and hands back, never a layout.
    DecodePlanCache,
    /// FlashInfer's prefill plan, by `const&`. Incomplete, as above.
    PrefillPlanCache,
    /// The MLA plan, by `const&`. Incomplete, as above.
    MlaPlanCache,
    /// The sm90 prefill schedule, by `const&`. Unlike the plan caches this
    /// one IS defined — a POD of offsets and extents — so Rust can mirror it.
    HopperPrefillPlan,
    /// Original-YaRN scaling, by `const*`. POD, and a pointer rather than a
    /// reference because it is optional: see `nullable`.
    YarnOriginalParams,
    /// A read-only device array of `pie::attn::StructuredMaskParams` —
    /// the per-lane structured-mask descriptors `attn::pack_structured_mask`
    /// reads. POD (three `u32`s), so Rust mirrors it and the array crosses
    /// as `*const StructuredMaskParams`.
    StructuredMasks,
}

/// What a statement supplies for one argument of a [`Ty`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binds {
    /// A pointer the kernel reads: an operand or a weight.
    Reads,
    /// A pointer the kernel writes: a result.
    Writes,
    /// Nothing a statement places.
    Nothing,
}

impl Ty {
    /// How C++ spells this, for a generated declaration.
    pub const fn cpp(self) -> &'static str {
        match self {
            // No C spelling: it is a FIELD of the preceding struct, so the
            // header already names it there.
            Ty::InPacked => "::std::uint32_t",
            Ty::BufMut => "void*",
            Ty::Buf => "const void*",
            Ty::I32s => "const ::std::int32_t*",
            Ty::I64s => "const ::std::int64_t*",
            Ty::BufArray => "const void* const*",
            Ty::BufArrayMut => "void* const*",
            Ty::BufArrayOut => "const void**",
            Ty::BufArrayOutMut => "void**",
            Ty::U8Array => "const ::std::uint8_t* const*",
            Ty::CustomAllReduce => "::pie::comm::CustomAllReduce*",
            Ty::I8s => "const ::std::int8_t*",
            Ty::I8sMut => "::std::int8_t*",
            Ty::Bf16s => "const ::pie::bf16*",
            Ty::F16s => "const ::pie::f16*",
            Ty::Bf16sMut => "::pie::bf16*",
            Ty::F16sMut => "::pie::f16*",
            Ty::I32Array => "const ::std::int32_t* const*",
            Ty::MoeActivation => "::pie::moe::MoeActivation",
            Ty::Mxfp4RowSelect => "::pie::quant::Mxfp4RowSelect",
            Ty::U16s => "const ::std::uint16_t*",
            Ty::U16sMut => "::std::uint16_t*",
            Ty::Dtype => "::pie_cuda_driver::DType",
            Ty::KvScheme => "::pie::attn::KvScheme",
            Ty::KvDType => "::pie::attn::KvDType",
            // CUDA's own, and unqualified by anything of ours: `<cuda_fp8.h>`
            // declares it at global scope.
            Ty::Fp8Kind => "::__nv_fp8_interpretation_t",
            Ty::I64 => "long long",
            Ty::U32s => "const ::std::uint32_t*",
            Ty::U8s => "const ::std::uint8_t*",
            Ty::F32sMut => "float*",
            Ty::F32s => "const float*",
            Ty::I32sMut => "::std::int32_t*",
            Ty::U32sMut => "::std::uint32_t*",
            Ty::U8sMut => "::std::uint8_t*",
            Ty::I32 => "int",
            Ty::U32 => "::std::uint32_t",
            Ty::Usize => "::std::size_t",
            Ty::F32 => "float",
            Ty::Bool => "bool",
            Ty::Stream => "cudaStream_t",
            Ty::CublasHandle => "cublasHandle_t",
            Ty::AttentionWorkspaceView => "::pie_cuda_driver::AttentionWorkspaceView",
            Ty::KvCacheLayerView => "::pie_cuda_driver::KvCacheLayerView",
            Ty::MlaCacheLayerView => "::pie_cuda_driver::MlaCacheLayerView",
            Ty::DecodePlanCache => "const ::pie::attn::DecodePlanCache&",
            Ty::PrefillPlanCache => "const ::pie::attn::PrefillPlanCache&",
            Ty::MlaPlanCache => "const ::pie::attn::MlaPlanCache&",
            Ty::HopperPrefillPlan => "const ::pie::attn::HopperPrefillPlan&",
            Ty::YarnOriginalParams => "const ::pie::attn::YarnOriginalParams*",
            Ty::StructuredMasks => "const ::pie::attn::StructuredMaskParams*",
        }
    }

    /// How Rust spells this on an `extern "C"` declaration.
    pub const fn rust(self) -> &'static str {
        match self {
            Ty::InPacked => "u32",
            Ty::BufMut => "*mut ::core::ffi::c_void",
            Ty::Buf => "*const ::core::ffi::c_void",
            Ty::I32s => "*const i32",
            Ty::I64s => "*const i64",
            Ty::BufArray => "*const *const ::core::ffi::c_void",
            Ty::BufArrayMut => "*const *mut ::core::ffi::c_void",
            Ty::BufArrayOut => "*mut *const ::core::ffi::c_void",
            Ty::BufArrayOutMut => "*mut *mut ::core::ffi::c_void",
            Ty::U8Array => "*const *const u8",
            Ty::CustomAllReduce => "*mut ::core::ffi::c_void",
            Ty::I8s => "*const i8",
            Ty::I8sMut => "*mut i8",
            // THE WIDTH, AND ONLY THE WIDTH -- deliberately the same spelling
            // `U16s` gets, because Rust has no bf16 and no f16. The format is
            // checked in the C++, against the instantiation the row names.
            Ty::Bf16s | Ty::F16s => "*const u16",
            Ty::Bf16sMut | Ty::F16sMut => "*mut u16",
            Ty::I32Array => "*const *const i32",
            Ty::MoeActivation => "u32",
            Ty::Mxfp4RowSelect => "i32",
            Ty::U16s => "*const u16",
            Ty::U16sMut => "*mut u16",
            Ty::Dtype => "u8",
            // A `u8`, like `Ty::Dtype`, and for its reason: the enum's
            // underlying type is stated in the C++ (`: ::std::uint8_t`), so
            // the crossing is a byte and no mirror is owed.
            Ty::KvScheme | Ty::KvDType => "u8",
            // FOUR bytes, and the difference from the two above is the whole
            // point: their width is stated in the C++ and this one's is not.
            Ty::Fp8Kind => "u32",
            Ty::I64 => "::core::ffi::c_longlong",
            Ty::U32s => "*const u32",
            Ty::U8s => "*const u8",
            Ty::F32sMut => "*mut f32",
            Ty::F32s => "*const f32",
            Ty::I32sMut => "*mut i32",
            Ty::U32sMut => "*mut u32",
            Ty::U8sMut => "*mut u8",
            Ty::I32 => "::core::ffi::c_int",
            Ty::U32 => "u32",
            Ty::Usize => "usize",
            Ty::F32 => "f32",
            Ty::Bool => "bool",
            Ty::Stream | Ty::CublasHandle => "*mut ::core::ffi::c_void",
            // Unqualified on purpose: a generated binding is placed in a
            // module that has the mirrors in scope, and this crate does not
            // know — and must not have to know — which module that is.
            Ty::AttentionWorkspaceView => "AttentionWorkspaceView",
            Ty::KvCacheLayerView => "KvCacheLayerView",
            Ty::MlaCacheLayerView => "MlaCacheLayerView",
            // Incomplete in the C++, so there is no layout to mirror and the
            // honest Rust type is the pointer a `const&` already is.
            Ty::DecodePlanCache | Ty::PrefillPlanCache | Ty::MlaPlanCache => {
                "*const ::core::ffi::c_void"
            }
            Ty::HopperPrefillPlan => "*const HopperPrefillPlan",
            Ty::YarnOriginalParams => "*const YarnOriginalParams",
            Ty::StructuredMasks => "*const StructuredMaskParams",
        }
    }

    /// What a STATEMENT has to supply for an argument of this type.
    ///
    /// Constness is the whole rule and not a heuristic: a `const` in the ABI
    /// means the kernel does not write through the pointer, and a statement's
    /// OUTPUT is exactly a pointer it does. The kinds that are neither are the
    /// ones no statement can name, which is why [`Binds::Nothing`] is not
    /// "unknown".
    #[must_use]
    pub const fn binds(self) -> Binds {
        match self {
            // Written through: the statement's results.
            Ty::BufMut
            | Ty::F32sMut
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut
            | Ty::U16sMut
            | Ty::I8sMut
            | Ty::Bf16sMut
            | Ty::F16sMut
            // `void* const*` and `void**` -- MoE's per-expert destination
            // arrays. The kernel writes through the inner pointers, which is
            // what makes these results and not operands.
            | Ty::BufArrayMut
            | Ty::BufArrayOutMut => Binds::Writes,
            // Read through: the statement's operands and its weights, which
            // this cannot tell apart and does not try to.
            Ty::Buf
            | Ty::I32s
            | Ty::I64s
            | Ty::U32s
            | Ty::U8s
            | Ty::F32s
            | Ty::U16s
            | Ty::I8s
            | Ty::Bf16s
            | Ty::F16s
            | Ty::BufArray
            | Ty::BufArrayOut
            | Ty::U8Array
            | Ty::I32Array => Binds::Reads,
            // Nothing a statement places: scalars, enums, handles, views and
            // plan descriptors. A routine takes these from the fire.
            Ty::I32
            | Ty::U32
            | Ty::I64
            | Ty::Usize
            | Ty::F32
            | Ty::Bool
            | Ty::InPacked
            | Ty::MoeActivation
            | Ty::Mxfp4RowSelect
            | Ty::Dtype
            | Ty::KvScheme
            | Ty::KvDType
            | Ty::Fp8Kind
            | Ty::Stream
            | Ty::CublasHandle
            | Ty::CustomAllReduce
            | Ty::AttentionWorkspaceView
            | Ty::KvCacheLayerView
            | Ty::MlaCacheLayerView
            | Ty::DecodePlanCache
            | Ty::PrefillPlanCache
            | Ty::MlaPlanCache
            | Ty::HopperPrefillPlan
            | Ty::YarnOriginalParams
            | Ty::StructuredMasks => Binds::Nothing,
        }
    }

    /// Whether [`rust`](Self::rust) names a type the generated declaration
    /// does not itself define.
    ///
    /// The six below spell an UNQUALIFIED `#[repr(C)]` mirror, so a binding
    /// using one compiles only where that mirror is in scope.
    #[must_use]
    pub const fn needs_mirror(self) -> bool {
        matches!(
            self,
            Ty::AttentionWorkspaceView
                | Ty::KvCacheLayerView
                | Ty::MlaCacheLayerView
                | Ty::HopperPrefillPlan
                | Ty::YarnOriginalParams
                | Ty::StructuredMasks
        )
    }
}


/// One operand as `#[routine]` read it off the launcher's own signature.
///
/// The same three questions [`Operand`] answers, asked of a Rust `fn`: what is
/// this parameter called, may it be null, and where does a driver get it.
/// There is no `ty` -- `KernelFn::ARGS` already carries it, and a second copy
/// would only be a way to disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Derived {
    /// The Rust parameter's name, which is also how the fact was recognised.
    pub name: &'static str,
    /// The parameter's type admits a null: `Option<NonNull<_>>` or
    /// `MaybeConst<_>`. The same claim [`Operand::nullable`] makes by hand,
    /// except that here the compiler is the one asserting it. Nothing reads
    /// it, so it is a measurement rather than a switch a binder may act on.
    /// `Or<_>` was a third spelling that REFUSED a null where these permit
    /// one; it is deleted.
    pub nullable: bool,
    /// Where a driver would bind this slot from, or `None` if the
    /// signature does not say. A weight is never derived — see the
    /// macro's docs, and [`Ty::binds`], which has always said that a weight
    /// and an input are both `const T*`.
    pub source: Option<Source>,
    /// The signature SAID this, rather than the macro counting to it.
    ///
    /// NOT the same as `source.is_some()`: a bare `*const T` derives `In(0)`
    /// and is unstated, and an `Env<f32>` named `rms_eps` derives from its
    /// NAME and is also unstated. Only positional sources are walked.
    pub stated: bool,
}

/// The operand column `#[routine]` read off a launcher's signature.
///
/// `#[routine]` emits `impl Derivation for <fn>` against a unit struct of the
/// SAME spelling: Rust's value and type namespaces are separate, so the name
/// is the function in expression position and the marker in type position.
///
/// `KernelFn::ARGS` cannot carry this. `ARGS` is built from parameter TYPES,
/// and a source is not a type -- `#[source(..)]` is consumed at expansion, and
/// a bare `*const T` is the same type at every position it could derive from.
pub trait Derivation {
    /// The column, in the order the signature takes it.
    const DERIVED: &'static [Derived];
}

/// Is this source the named fact `key`, in a `const` context?
///
/// `Source::Named` CANNOT BE MATCHED IN A CONST: `matches!` on a `&str` is
/// `E0658` and `==` fails the same way, `PartialEq` not yet being a const
/// trait (rust-lang/rust#143874). The derived-column pins need this.
#[must_use]
pub const fn source_is_named(s: &Option<Source>, key: &str) -> bool {
    let Some(Source::Named(actual)) = s else {
        return false;
    };
    let (a, b) = (actual.as_bytes(), key.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Which indexed channel a [`Source::Slot`] counts within.
///
/// `ParamF32` is a kind and not a type because the params channel is a byte
/// run with no element type: "the Nth param read as f32" is a different
/// CHANNEL, not a different reading of one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    /// The statement's `i`-th operand, as a device pointer.
    In,
    /// The statement's `i`-th result, as a device pointer.
    Out,
    /// The `i`-th weight the statement NAMES, resolved through the
    /// binder.
    Weight,
    /// The `i`-th scalar the statement carries (`Launch`'s params).
    Param,
    /// The same slot read as a FLOAT.
    ///
    /// The param channel is untyped `u32` and a scale is a float. gemma-4
    /// fires one kernel four times with four constants the HOST derives, and
    /// the driver held a name-to-arithmetic table to get them back; the
    /// statement carries the number instead.
    ParamF32,
    /// The staged parameter BLOCK: every scalar the statement carries, laid
    /// out as one struct and bound as one buffer.
    ///
    /// A shader plane stages this where CUDA passes scalars individually,
    /// which is why the kind exists here and nowhere in the CUDA rows. There
    /// is one per launch, so the index is always zero.
    Params,
    /// The `i`-th FOREIGN value the join collected for this statement.
    ///
    /// nemotron's mamba block wires values ACROSS statements, none of which
    /// their own statements carry. Its own source and not `Slot(Kind::In, _)`
    /// because the INDEX is the join's convention, not the trace's.
    Aux,
    /// The trailing-dims product of the `i`-th result — what a row of
    /// it is worth in elements.
    OutWidth,
    /// The same for the `i`-th operand.
    InWidth,
    /// Rows times the `i`-th result's row width — the ELEMENT count a
    /// flat launcher takes where a row-shaped one takes both.
    OutElements,
}

/// Where a bound argument comes from.
///
/// `PartialEq` but not `Eq`: a [`Lit::F32`] is a float. Nothing here is a map
/// key — the comparisons are all "is this operand sourced from X" — and a
/// `Source` const cannot appear in a pattern, which is why [`Source::Named`]
/// carries the key's `&'static str` rather than a `keys::X::SOURCE`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// A literal VALUE — for the arguments a launcher takes that no
    /// statement and no context carries: an `interleaved` flag a family
    /// never sets, a `beta` of zero, an optional buffer left absent.
    Lit(Lit),
    /// An operand reached by INDEX: the `i`-th of its [`Kind`]. The same
    /// spelling works in expression and pattern position.
    Slot(Kind, u8),
    /// A fact reached by NAME.
    ///
    /// The payload is the key's string, spelled `<keys::X as keys::Fact>::KEY`
    /// in expression AND pattern position. Not a preference: an associated
    /// `&'static str` const is a legal pattern, so TWO KEYS WITH THE SAME
    /// STRING make the second arm an unreachable-pattern warning. A string
    /// literal loses that, which is why `#[source(Named(..))]` is refused.
    Named(&'static str),
    /// A CHAIN: the first if the statement carries one, the second
    /// otherwise.
    ///
    /// Not a convenience: a statement's scalar and a fire-wide fact are
    /// alternatives only where a per-layer number exists. gemma-4's
    /// full-attention layers rotate a quarter of each head and its sliding
    /// layers all of one, so no fire-wide `rotary_width` is right for both,
    /// while every single-shape deployment states nothing and means the
    /// fire's.
    ///
    /// Zero is absent: a grid axis of zero launches nothing.
    ///
    /// By reference because a `Source` cannot contain itself by value.
    Or(&'static Source, &'static Source),
    /// A PRODUCT of two sources.
    ///
    /// Arithmetic on facts is not a fact: `logit_softcap`'s grid is one flat
    /// span, `width * rows`, and a probe moving facts one at a time sees the
    /// argument move twice and can name neither.
    Times(&'static Source, &'static Source),
    /// A QUOTIENT of two sources, refused when the divisor is zero.
    ///
    /// `rms_strided_head_row` normalizes each head of a row separately and
    /// needs how many there are, which is the row's width over the length of
    /// one head -- and the divisor is itself a chain, because a statement may
    /// carry the head length and the fire answers when it does not.
    Over(&'static Source, &'static Source),
}

/// What a [`Source::Lit`] holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lit {
    /// An absent optional pointer. Typed by the operand it binds — the
    /// row already says which pointer is missing.
    Null,
    /// A flag.
    Bool(bool),
    /// A float. `1.702` is gpt-oss's GLU alpha; the row carries the
    /// number, and each consumer spells it its own way.
    F32(f32),
    /// An integer.
    I32(i32),
}

/// One kernel's contract.
pub struct KernelSig {
    /// The dsl-side name (what a model text spells).
    pub name: &'static str,
    /// The C++ launcher symbol the trace records.
    pub symbol: &'static str,
    /// The kernel REFUSES a row split: it may not be stated inside a peel's
    /// regions, because its addressing (a fire-wide prepare, a padded staging
    /// buffer) is not row-offsettable. `model-compiler`'s `OpKind::Peel` is
    /// the op this refuses, and its `check_plan` is what enforces the refusal.
    pub whole: bool,
    /// Which OUTPUTS this kernel writes over which INPUTS, as
    /// `(output index, input index)` pairs.
    ///
    /// A fact about the KERNEL and not about any statement using it -- every
    /// call of `residual_add` is in-place -- which is why it lives here.
    pub in_place: &'static [(u32, u32)],
    /// On a union tail layer this dispatch pairs the DEPTH PREFIX plan (and
    /// its dedicated workspace) instead of the fire's own plan.
    pub depth_prefix_plan: bool,
    /// The arguments the routine takes, in the order its `fn` takes them, and
    /// who supplies each.
    ///
    /// DERIVED, from [`routine::KernelFn::ARGS`], so it cannot disagree with
    /// the body. Empty means the row was written by hand — the three tables
    /// that predate the routine shape — and not that the routine is nullary.
    pub args: &'static [(Ty, Provenance)],
    /// Which side of the statement each argument sits on, in the same order.
    ///
    /// DERIVED, from [`routine::KernelFn::SIDES`]. See [`routine::Side`].
    pub sides: &'static [routine::Side],
    /// The axes `symbol` is instantiated over, if it names a FAMILY of
    /// entrypoints rather than one.
    ///
    /// Empty is the CUDA case and the default: a launcher there is an authored
    /// C++ function, so one row is one symbol and [`sig_in`] matches it whole.
    ///
    /// Metal's are generated: `quant/qmm_t.metal` stamps one template body
    /// over `(group, bits) × (bm, bn)`, so 54 entrypoints are one kernel at 54
    /// points. Enumerating them as rows would state the macro's job by hand.
    ///
    /// Not a Metal-only idea: CUDA writes the same product into FILENAMES and
    /// cannot state it, each of those being separately authored.
    pub axes: &'static [Axis],
}

// `lacks` IS NOT SURPLUS -- ITS READER IS MISSING. Zero readers, fourteen
// writers across four backends, every one a real capability negation. The
// consumer that ought to refuse routing a scores-needing seam onto a kernel
// that lacks them does not exist. DO NOT DELETE.
//
// `Cap` IS A HOMONYM: `model_ir::seam::Cap` is a second, independent enum that
// also has `Scores` and `PageMaskSink`, so a grep cannot attribute a hit.

impl KernelSig {
    /// Does `symbol` name this row — as the kernel itself, or at one point of
    /// its axes?
    ///
    /// Both are legitimate: a model text states the KERNEL, a driver or audit
    /// the POINT. Nothing BETWEEN them resolves -- the axes are peeled from
    /// the END, so a half-spelled name is refused rather than rounded.
    pub fn covers(&self, symbol: &str) -> bool {
        self.symbol == symbol || self.covers_point(symbol)
    }

    /// `symbol` is this row at one point of its axes — not the bare base.
    ///
    /// Order is the whole implementation: the axes are declared in the order a
    /// name spells them, so this peels suffixes from the end, one axis at a
    /// time, and what must remain is the base. That refuses a name a point
    /// short of a real entrypoint, and refuses a permuted spelling.
    pub fn covers_point(&self, symbol: &str) -> bool {
        if self.axes.is_empty() {
            return false;
        }
        let mut rest = symbol;
        for axis in self.axes.iter().rev() {
            match axis
                .points
                .iter()
                .find(|point| rest.len() > point.len() && rest.ends_with(**point))
            {
                Some(point) => rest = &rest[..rest.len() - point.len()],
                None => return false,
            }
        }
        rest == self.symbol
    }

    /// Every entrypoint this row names: the product of its axes, appended in
    /// declaration order. One element (the symbol itself) when there are none.
    pub fn entrypoints(&self) -> Vec<String> {
        let mut out = vec![self.symbol.to_string()];
        for axis in self.axes {
            out = out
                .iter()
                .flat_map(|stem| {
                    axis.points
                        .iter()
                        .map(move |point| format!("{stem}{point}"))
                })
                .collect();
        }
        out
    }
}

// `KernelSig` survives the `kernel!` row DSL that used to build it, and so
// does `sig_in`: `kernels-cuda` synthesises rows at run time from its routine
// registry, because `model-compiler` asks a backend two questions about every
// symbol it lowers. That is a lookup, not a table.


/// The contract for one symbol, in `table`.
///
/// Exact matches on the symbol are tried first and across the WHOLE table,
/// before any row is allowed to claim `symbol` as a point of its axes. Without
/// that two-pass order a row could swallow a sibling whose base happens to end
/// in one of its points, and which row won would depend on declaration order.
pub fn sig_in(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    table
        .iter()
        .find(|k| k.symbol == symbol)
        .or_else(|| table.iter().find(|k| k.covers_point(symbol)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFFINE: Axis = Axis {
        what: "affine group and width",
        points: &[
            "_gs_32_b_4",
            "_gs_64_b_4",
            "_gs_128_b_4",
            "_gs_32_b_8",
            "_gs_64_b_8",
            "_gs_128_b_8",
        ],
    };
    const TILE: Axis = Axis {
        what: "routed GEMM tile",
        points: &["_bm_16_bn_16", "_bm_32_bn_32", "_bm_64_bn_64"],
    };
    const DTYPE: Axis = Axis {
        what: "activation dtype",
        points: &["_bfloat16"],
    };

    /// A row at rest, for a test that cares about two columns and not sixteen.
    ///
    /// `kernel!` used to spell this: a macro whose whole body was a struct
    /// literal with defaults. It is written out here because the macro is gone
    /// -- no table in the tree holds a row any more, and a DSL for writing
    /// rows was an invitation to write one.
    const BASE: KernelSig = KernelSig {
        name: "",
        symbol: "",
        whole: false,
        in_place: &[],
        depth_prefix_plan: false,
        args: &[],
        sides: &[],
        axes: &[],
    };

    static TABLE: &[KernelSig] = &[
        KernelSig {
            name: "qmv",
            symbol: "affine_qmv_fast",
            axes: &[DTYPE, AFFINE],
            ..BASE
        },
        KernelSig {
            name: "qmm_t",
            symbol: "affine_qmm_t",
            axes: &[DTYPE, AFFINE, TILE],
            ..BASE
        },
        // A base that is ALSO a legal entrypoint, next to its dtyped form.
        KernelSig {
            name: "route_sort",
            symbol: "moe_route_sort",
            ..BASE
        },
        KernelSig {
            name: "router",
            symbol: "router_topk",
            axes: &[DTYPE],
            ..BASE
        },
    ];

    fn named(symbol: &str) -> Option<&'static str> {
        sig_in(TABLE, symbol).map(|k| k.name)
    }

    #[test]
    fn a_row_covers_every_point_of_its_axes() {
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_64_b_4"), Some("qmv"));
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_128_b_8"), Some("qmv"));
        assert_eq!(
            named("affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64"),
            Some("qmm_t")
        );
    }

    /// The axes are peeled from the END in declaration order, so a name that
    /// stops short of a full point set is NOT covered.
    #[test]
    fn a_partial_or_permuted_spelling_is_refused() {
        assert_eq!(named("affine_qmm_t_bfloat16_gs_64_b_4"), None); // no tile
        assert_eq!(named("affine_qmm_t_bm_16_bn_16"), None); // no dtype/affine
        // Right points, wrong order.
        assert_eq!(named("affine_qmm_t_bfloat16_bm_16_bn_16_gs_64_b_4"), None);
        // A point that is not on the axis.
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_16_b_4"), None);
    }

    /// A row whose base is itself an entrypoint keeps it, and does not get
    /// eaten by a sibling that could peel to the same text.
    #[test]
    fn a_row_resolves_by_its_base_and_by_every_point() {
        assert_eq!(named("moe_route_sort"), Some("route_sort"));
        assert_eq!(named("router_topk_bfloat16"), Some("router"));
        // The BASE resolves too, and this is not a convenience: a model text
        // states the kernel, not the instantiation.
        assert_eq!(named("router_topk"), Some("router"));
        assert_eq!(named("affine_qmm_t"), Some("qmm_t"));
    }

    /// CUDA's rows carry no axes, and this is the assertion that the addition
    /// changed nothing for them: an axisless row matches its symbol and
    /// nothing else, prefix or suffix.
    #[test]
    fn an_axisless_row_is_unchanged_by_the_axis_machinery() {
        assert_eq!(named("moe_route_sort_bfloat16"), None);
        assert_eq!(named("moe_route_sor"), None);
        assert_eq!(named("xmoe_route_sort"), None);
    }

    /// The `sdpa_paged_decode` case, and the reason `points` may hold `""`.
    #[test]
    fn an_axis_may_have_a_point_that_adds_no_text() {
        const DIM: Axis = Axis {
            what: "head dim",
            points: &["_d_64", "_d_128"],
        };
        // Longest first, empty last: both orderings are load-bearing.
        const PAGE: Axis = Axis {
            what: "page table width and simdgroup count",
            points: &["_p32_sg8", "_p32", ""],
        };
        static T: &[KernelSig] = &[KernelSig {
            name: "sdpa_paged",
            symbol: "sdpa_paged_decode",
            axes: &[DTYPE, DIM, PAGE],
            ..BASE
        }];

        for name in [
            "sdpa_paged_decode_bfloat16_d_128",
            "sdpa_paged_decode_bfloat16_d_128_p32",
            "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
        ] {
            assert!(sig_in(T, name).is_some(), "{name}");
        }
        // Still not a licence to match anything.
        assert!(sig_in(T, "sdpa_paged_decode_bfloat16_d_256").is_none());
        assert!(sig_in(T, "sdpa_paged_decode_bfloat16").is_none());
        assert_eq!(T[0].entrypoints().len(), 2 * 3);
    }

    /// `covers` and `entrypoints` are two directions on one relation, and the
    /// audit script trusts both. Round-trip them.
    #[test]
    fn everything_a_row_generates_is_something_it_covers() {
        for row in TABLE {
            for name in row.entrypoints() {
                assert_eq!(
                    sig_in(TABLE, &name).map(|k| k.name),
                    Some(row.name),
                    "{name}"
                );
            }
        }
    }
}
