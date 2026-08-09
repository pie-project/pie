//! ② KERNEL SIGNATURES — the vocabulary. The rows live with the kernels
//! (`.wiki/tart/dsl.md` ②).
//!
//! `dsl::cuda` has ten wrappers over five attention kernels because
//! `_region` / `_planned` / `_capture` / `_dequant` encode the DISPATCH
//! CONTEXT in the wrapper name. The context is a property of the call site;
//! what belongs to the kernel is its symbol and its contract. A [`KernelSig`]
//! is that contract, once per symbol.
//!
//! Four declarations, each replacing something that is a hand-written runtime
//! rule today:
//!
//! | declaration | replaces |
//! |---|---|
//! | `whole`   | `if c.head_dim_padded \|\| (window_one && c.xqa_decode)` in the model body |
//! | `lacks`   | "a score-wanting program under XQA fails loudly PTIR-side" (a C++ throw) |
//! | `needs`   | the prepare a stated kernel obligates, named nowhere |
//! | `sink`    | `emit_cuda::emit_masked_pages_bracket`'s hardcoded page substitution |
//!
//! `whole` is CHECKED at trace time — which is load time, since a declaration
//! is traced when the model loads. The other three are declared but not yet
//! consumed: `needs`/`sink` are the emitter's knowledge until the launch ABI
//! flattens (migration step 6), and `lacks` needs the deployment's
//! servable-seam set, which is the support-matrix work. Declaring them first
//! is the point — the table is where they land, and it exists.
//!
//! ## Why this is its own crate
//!
//! The rows are in [`kernels-cuda`](../kernels_cuda/index.html) and
//! [`kernels-metal`](../kernels_metal/index.html), one crate per backend,
//! each beside the `.cu`/`.metal` it describes — so a new kernel is one
//! source file and one table row in the same directory and the same diff
//! hunk. Both tables have to be written in the same words, and neither
//! backend owns those words, so they are here.
//!
//! Bare-named for the same reason [`driver`](../driver/index.html) is: it is
//! the shared floor under a `-`-prefixed pair, holding what both members
//! speak rather than anything either one does. Nothing depends on it but the
//! two tables and the compiler that reads them, and it depends on nothing at
//! all — a row must be writable next to its kernel without dragging a
//! dependency graph along.

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
///
/// This is data. The arithmetic each variant names stays in the driver, beside
/// the doc comment that explains it — the same split [`Prepare`] and
/// [`Source`] already make between what a contract STATES and what a backend
/// DOES about it.
///
/// The alternative was a `const` expression grammar on the row. Every rule in
/// use is uniformly `source -> max -> min -> divide-rounding-up -> multiply`,
/// so it fits; it was rejected because spelling a rule as
/// `Term { floor: 1, cap: 1024, div_ceil: 32, mul: 32 }` loses the sentence
/// that says why, and those sentences carry findings — one of them records
/// that a round-up is the difference between computing every output and
/// silently dropping the last few.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LaunchRule {
    /// The row has not said. A backend must REFUSE rather than guess: a
    /// guessed grid runs a kernel over the wrong extent, which no hardware
    /// reports. Same meaning [`Source::Unbound`] has for an operand.
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
    /// (`embed_gather_mb`). One rule apart from [`LaunchRule::Elementwise`]
    /// because the two agree at one row and disagree above it, which is
    /// exactly the distinction a row has to be able to state.
    ElementwiseRows,
    /// One threadgroup per head, `head_dim` wide — the q/k/v split, the KV
    /// append. The row is the third grid axis.
    PerHead,
    /// Single-pass decode attention: one 1024-thread threadgroup per query
    /// head, rows on the second axis.
    SdpaVector,
    /// Pointwise over every head's channels, 256-wide.
    PerHeadElementwise,
    /// Gated norm over the value heads.
    GatedRms,
    /// One threadgroup as wide as the expert count, rounded to a simd multiple.
    RouterLane,
    /// One threadgroup per row, as wide as the row, capped at 256.
    RouteRows,
    /// Routed GEMV: [`LaunchRule::Qmv`] per row, per expert slot.
    RoutedQmv,
    /// Affine GEMM: the batched projection, tiled over rows and columns.
    /// Distinct from [`LaunchRule::Qmv`] because it is a different kernel with
    /// a different name, not the same one launched wider — which is what makes
    /// the M>1 lane a ROW's statement rather than a mode the driver picks.
    Qmm,
}

/// The host-side plan a kernel's contract obligates: stated so a reader of
/// the model text can see which prepare a launch drags in, rather than
/// reading the driver to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prepare {
    /// No host plan.
    None,
    /// The FlashInfer decode plan (per fire, per layer group).
    DecodePlan,
    /// The FlashInfer ragged prefill plan.
    PrefillPlan,
    /// The custom-mask plan (`attn_page_mask`'s consumer).
    CustomPlan,
    /// XQA's fire-wide prepare — R-shaped, so it cannot be built per row
    /// window. This is why `xqa_decode` is also `whole`.
    FireWide,
    /// MLA's plan (`kernels::attn::plan_attention_mla_bf16`), which is its own kind
    /// rather than a FlashInfer plan under another name: it is built from
    /// `kv_lora_rank` and `qk_rope_head_dim` — a latent KV geometry no other
    /// prepare here has a field for — and it is cached in an `MlaPlanCache`
    /// the dispatch borrows, not in the shared attention workspace.
    MlaPlan,
}

/// One point of one instantiation axis, and the text it contributes to a name.
///
/// See [`KernelSig::axes`] for why a row has these at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    /// What varies. Prose, for a reader of the table; the matcher never reads
    /// it.
    pub what: &'static str,
    /// The suffixes this axis can contribute, in the order a name spells them.
    /// Exactly one is present in any entrypoint the axis reaches.
    ///
    /// A point MAY be `""`, for an axis whose default specialisation adds no
    /// text — `sdpa_paged_decode<…, 0, false, 32>` is spelled
    /// `sdpa_paged_decode_bfloat16_d_128` and the two others are `…_p32` and
    /// `…_p32_sg8`, off ONE template. Two rules follow and both are checked by
    /// [`KernelSig::covers`]'s ordering rather than asserted:
    ///
    /// * the empty point goes LAST, because matching is first-wins and an
    ///   empty suffix matches everything;
    /// * a longer point goes before a shorter one it ends with (`_p32_sg8`
    ///   before `_p32`), for the same reason.
    pub points: &'static [&'static str],
}

/// What one operand of a launcher is, in words neither backend owns.
///
/// This is deliberately a SMALL vocabulary. It is not a type system and it is
/// not trying to describe what a buffer contains — `q`, `k` and `k_pages` are
/// all [`Ty::BufMut`], because how a kernel reads its own tensor is the
/// kernel's business. What it has to describe is exactly what a CALLER must
/// know to place an argument: how wide the word is, whether the callee may
/// write through it, and whether it may be absent.
///
/// The element-typed array kinds exist because the C++ spells them
/// (`const std::uint32_t*` for a CSR array, `const std::int32_t*` for
/// positions) and losing that would make the generated declaration a `void*`
/// that no longer proves anything — see [`crate`]'s note on why the shim is
/// generated rather than written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// An opaque device buffer the launcher may WRITE through (`void*`).
    BufMut,
    /// An opaque device buffer the launcher only reads (`const void*`).
    Buf,
    /// A read-only device array of `i32` — positions, and the like.
    I32s,
    /// A read-only device array of `i64`. One row needs it — kimi_k3's
    /// hash routing reads a `[vocab, K]` token-to-expert table — and it
    /// is its own kind rather than a `Buf` because the pilot caught
    /// exactly that substitution: `const void*` and `const int64_t*` are
    /// both pointers, so only the DECLARED width makes the mismatch a
    /// compile error instead of a stride bug.
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
    ///
    /// Naming reads outside-in — `BufArray` is the array the launcher
    /// READS of buffers it READS; `Mut` on the end means the BUFFERS are
    /// writable; `Out` in the middle means the ARRAY is.
    ///
    /// The distinction is not pedantry. `gemm::batched_act_x_wt_bf16`
    /// takes its destination array as `void* const*` — it writes the
    /// buffers and reads the array — while the nemotron pointer
    /// BUILDERS take `void**`, because building the array is what they
    /// do. A row that swapped them would compile as `-fpermissive` and
    /// hand a builder an array it must not write.

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
    /// A HANDLE, like [`Ty::Stream`] and [`Ty::CublasHandle`]: the arm is
    /// given it and never asks for it. It exists because the fused
    /// landing was a METHOD, and a method has no address the generated
    /// ABI can forward to — the free form takes the instance first, and
    /// the row can then describe the call.
    CustomAllReduce,
    /// The element type a buffer is stored in — `DType`, a
    /// `std::uint8_t`-backed enum class.
    ///
    /// It is a scalar the CALLER states, not a property the launcher
    /// discovers, and that is the whole reason the scaled GEMM entry
    /// points take it: the storage a weight is in used to reach them
    /// inside a `WeightView`, and a driver built that descriptor by
    /// looking at a per-layer struct no statement mentioned. The
    /// descriptor is assembled INSIDE the launcher now and the caller
    /// passes what the declaration said.
    ///
    /// Its own kind rather than [`Ty::U32`]: an enum class does not
    /// convert from an integer, so a row that widened it would not
    /// compile -- which is the answer wanted, but for the wrong reason.
    /// Spelling it means the shim forwards the enum the header declares.
    Dtype,
    /// A host scalar spelled `long long` — the recurrent state's slot
    /// stride, which is an ELEMENT count into a multi-gigabyte arena and
    /// so was widened deliberately. Its own kind for `Ty::U32`'s reason:
    /// a mirror that guessed `int` is a silent truncation, not an error.
    I64,
    /// A host byte count, spelled `std::size_t`. Its width is the platform's,
    /// which is why it is not [`Ty::U32`] widened by hand.
    Usize,
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
    /// A launcher is "anything that issues device work", so the MLA absorb
    /// pair are launchers even though the work is `cublasGemmStridedBatchedEx`
    /// and not a kernel of ours. This is what they take instead of a stream —
    /// the stream is set on the handle.
    CublasHandle,

    // ---- The struct-shaped operands. ----
    //
    // Four passing modes, and the mode is the launcher's choice rather than
    // the row author's, so it is recorded here in `cpp()` rather than spelled
    // at each use. What separates them is what the RUST side can say:
    //
    //   by value, POD     a `#[repr(C)]` mirror, its layout proven by
    //                     `emit_layout_assertions`
    //   by const ref/ptr, POD    the same mirror, behind a `*const`
    //   by const ref, INCOMPLETE the C++ never defines the type, so Rust
    //                     has nothing to mirror and gets `*const c_void`
    //
    // A row cannot pick the wrong one by accident: the shim initialises a
    // function pointer, and a function pointer takes no conversions.
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
    /// A read-only device array of `attn::StructuredMaskParams` — the
    /// per-lane structured-mask descriptors `attn::pack_structured_mask`
    /// reads. POD (three `u32`s), so Rust mirrors it and the array crosses
    /// as `*const StructuredMaskParams`.
    ///
    /// Unlike the descriptor kinds this file lost, nothing about it is a
    /// ROUTE: the packer reads every lane's kind and window the same way,
    /// and no caller is choosing a kernel by what it finds inside. It is
    /// an operand shaped like a struct, which is why it survives where
    /// `WeightView` did not.
    StructuredMasks,
}

impl Ty {
    /// How C++ spells this, for a generated declaration.
    ///
    /// The pointer kinds are spelled with the fixed-width `std::` names the
    /// headers use, not `int`/`unsigned`, so the generated text is the same
    /// text a reader finds in the header it is checked against.
    pub const fn cpp(self) -> &'static str {
        match self {
            Ty::BufMut => "void*",
            Ty::Buf => "const void*",
            Ty::I32s => "const ::std::int32_t*",
            Ty::I64s => "const ::std::int64_t*",
            Ty::BufArray => "const void* const*",
            Ty::BufArrayMut => "void* const*",
            Ty::BufArrayOut => "const void**",
            Ty::BufArrayOutMut => "void**",
            Ty::U8Array => "const ::std::uint8_t* const*",
            Ty::CustomAllReduce => "::pie_cuda_driver::kernels::comm::CustomAllReduce*",
            Ty::I8s => "const ::std::int8_t*",
            Ty::I32Array => "const ::std::int32_t* const*",
            Ty::MoeActivation => "::pie_cuda_driver::kernels::moe::MoeActivation",
            Ty::Mxfp4RowSelect => "::pie_cuda_driver::kernels::quant::Mxfp4RowSelect",
            Ty::U16s => "const ::std::uint16_t*",
            Ty::U16sMut => "::std::uint16_t*",
            Ty::Dtype => "::pie_cuda_driver::DType",
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
            Ty::DecodePlanCache => "const ::pie_cuda_driver::kernels::attn::DecodePlanCache&",
            Ty::PrefillPlanCache => "const ::pie_cuda_driver::kernels::attn::PrefillPlanCache&",
            Ty::MlaPlanCache => "const ::pie_cuda_driver::kernels::attn::MlaPlanCache&",
            Ty::HopperPrefillPlan => "const ::pie_cuda_driver::kernels::attn::HopperPrefillPlan&",
            Ty::YarnOriginalParams => "const ::pie_cuda_driver::kernels::attn::YarnOriginalParams*",
            Ty::StructuredMasks => "const ::pie_cuda_driver::kernels::attn::StructuredMaskParams*",
        }
    }

    /// How Rust spells this on an `extern "C"` declaration.
    ///
    /// `Stream` lands as a plain opaque pointer rather than any driver type:
    /// `cudaStream_t` is `CUstream_st*` and `CUstream` is the same pointer,
    /// so a driver-side binding can pass its own handle without a conversion,
    /// and this crate does not have to name either API to say so.
    pub const fn rust(self) -> &'static str {
        match self {
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
            Ty::I32Array => "*const *const i32",
            Ty::MoeActivation => "u32",
            Ty::Mxfp4RowSelect => "i32",
            Ty::U16s => "*const u16",
            Ty::U16sMut => "*mut u16",
            Ty::Dtype => "u8",
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
}

/// One operand of a launcher, in the position the launcher takes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operand {
    /// The C++ parameter's name. Prose — nothing matches on it — but it is
    /// what makes a row diffable against the header by eye, and what a
    /// generated binding names its argument.
    pub name: &'static str,
    /// What has to be placed here.
    pub ty: Ty,
    /// The launcher accepts a null here and means something by it.
    ///
    /// Not checkable by the C++ compiler — every pointer accepts null — which
    /// is exactly why it belongs in the table. `rope_write_kv_bf16`'s
    /// `row_valid` says "may be null" in a comment today, and a comment is
    /// not something a binding can be generated from.
    pub nullable: bool,
    /// WHERE the value comes from when a driver binds this slot.
    ///
    /// The signature says what TYPE goes here and the row's arity says
    /// how many; neither says that `q` is the statement's first OUTPUT
    /// and `positions` is the fire's. That correspondence is what every
    /// hand-written arm encodes, one arm at a time, and it is the last
    /// thing standing between a table that describes a call and a table
    /// a call can be GENERATED from.
    ///
    /// [`Source::Unbound`] — the default — means the row has not said,
    /// and a generator skips it exactly as it skips a row with no
    /// operands. Filling it is per-row work like the signature was.
    pub source: Source,
}

/// Where a bound argument comes from.
///
/// The vocabulary is deliberately small and describes the STATEMENT and
/// the FIRE, never a family: an operand that could only be sourced from
/// a workspace field is an operand whose arm is not shareable, which is
/// the same boundary `ExecCtx` draws.
/// `PartialEq` but not `Eq`: a [`Lit::F32`] is a float, and a float is
/// not a total order. Nothing here is a map key — the comparisons are
/// all "is this operand sourced from X", which partial equality answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// The row has not stated it; nothing may be generated from this.
    Unbound,
    /// The statement's `i`-th operand, as a device pointer.
    In(u8),
    /// The statement's `i`-th result, as a device pointer.
    Out(u8),
    /// The `i`-th weight the statement NAMES, resolved through the
    /// binder.
    Weight(u8),
    /// The `i`-th scalar the statement carries (`Launch`'s params).
    Param(u8),
    /// The same slot read as a FLOAT.
    ///
    /// The param channel is untyped `u32` — "what each slot means is the
    /// SYMBOL's contract" — and a scale is a float. gemma-4 fires one
    /// kernel four times per fire with four different constants, all
    /// derived from dims the HOST knows, and the driver held a
    /// name-to-arithmetic table to get them back. The statement carries
    /// the number instead, in the bits the slot already has room for.
    ///
    /// WHAT IS STILL BLOCKED ON THIS, and it is one thing wearing three
    /// hats. Every row below is fully derivable EXCEPT for a load-time
    /// number the model-side FACTS do not carry:
    ///
    /// * gemma-4's per-layer rope theta (`rope_partial_bf16`,
    ///   `qk_rmsnorm_rope_bf16_rounded`) — the driver reads it off the
    ///   weights struct, which is loaded, not traced.
    /// * gemma-4's per-layer PLE scalar
    ///   (`rmsnorm_residual_add_scale_rmsnorm_bf16`) — the same.
    /// * gpt-oss's five yarn constants (`rope_yarn_original_bf16`) — in
    ///   the forward config, not in `GptOssFacts`.
    ///
    /// Each is one fixture field away, and the field is the work: a
    /// declaration states what the deployment IS, so the number has to
    /// be the deployment's real one. `dsl::cuda::rope_partial`'s note
    /// says why inventing them is worse than a driver reading a config
    /// value, and that judgment is what these rows are waiting on rather
    /// than any missing vocabulary.
    ParamF32(u8),
    /// The fire's POSITIONS, advanced to the rectangle's first row.
    ///
    /// `Ctx("positions")` would bind the fire's base, and a rectangle
    /// that starts partway in would then rotate its own rows against
    /// another rectangle's positions — the same defect `ArmCtx::row`
    /// exists for, on the one fire input that is token-rowed. So it gets
    /// the same treatment and its own source rather than a `Ctx` a
    /// reader has to notice is special.
    ///
    /// The DEVICE-WINDOW call forms are the exception and they do not
    /// use this: their kernel reads the split off a device word and
    /// wants the fire's positions unadvanced. Those are hand-written
    /// arms, which is where a per-call-form fact belongs.
    Positions,
    /// The rectangle's row count.
    ///
    /// The fire's, and only right for a statement whose rows ARE the
    /// fire's — which is most of them and not all. Where a value's own
    /// leading extent is the answer, [`Source::OutRows`] says so and
    /// covers this case too.
    Rows,
    /// The `i`-th result if the statement declares one, and the
    /// enclosing value-producing guard's value otherwise.
    ///
    /// A REGION LAUNCH declares no result: the guard owns the value and
    /// its arms bind it, so which value that is depends on where the
    /// statement sits rather than on what it says. qwen3.5's recurrence
    /// three-way and gpt-oss's attention chain are the same shape, and
    /// a row whose spellings appear both ways — the decode step states
    /// its result, the prefill spellings do not — cannot say `Out` and
    /// cannot say `Ctx`.
    ResultOrRegion(u8),
    /// The `i`-th result's LEADING extent, resolved for this fire.
    ///
    /// `Rows` for a token-shaped value, a constant for a fixed one, and
    /// for the MoE aligned path the padded block-major count — which is
    /// `Dim::MoeAlignedRoutes`, the one extent in the tree that is
    /// neither the fire's rows nor a load-time number. Five hand-written
    /// forwards restate the formula for it; a row that says `OutRows`
    /// gets it from the one place that computes it.
    OutRows(u8),
    /// The same for the `i`-th operand.
    InRows(u8),
    /// The trailing-dims product of the `i`-th result — what a row of
    /// it is worth in elements.
    OutWidth(u8),
    /// The same for the `i`-th operand.
    InWidth(u8),
    /// Rows times the `i`-th result's row width — the ELEMENT count a
    /// flat launcher takes where a row-shaped one takes both.
    OutElements(u8),
    /// The same for the `i`-th OPERAND.
    ///
    /// The MoE routing kernels want the ROUTE count, which is the fire's
    /// tokens times `top_k` — and `topk_idx` is `[Tokens, top_k]`, so it
    /// is exactly that operand's element count. A product the table has
    /// no arithmetic for, read off a value that already is it.
    InElements(u8),
    /// Dimension `d` of the `i`-th operand. The routed combine reads
    /// `[Tokens, top_k, H]` and both extents come off it.
    InDim(u8, u8),
    /// Dimension `d` of the `i`-th result, which is how a head count
    /// reaches a launcher: the shape says `[Tokens, heads, dim]`.
    OutDim(u8, u8),
    /// A named field of the executing context — the stream, the handle,
    /// `eps`, the head geometry.
    ///
    /// The name is the FIELD's, and nothing else. It used to carry the
    /// C++ context object's struct nesting (`"arm.stream"`), which made
    /// this vocabulary speak one consumer's shape: every other consumer
    /// then had to strip a prefix it never had. A row names a fact; where
    /// that fact sits inside a driver's context is the driver's business.
    Ctx(&'static str),
    /// [`Source::Ctx`], plus a GUARD: the generated branch fires only
    /// when the field is non-zero, and a family that leaves it zero
    /// keeps its own arm.
    ///
    /// This exists because of one number. gemma-4 alternates its rope
    /// theta per layer, so the single `rope_theta` a context can carry
    /// is the wrong one for half that model, and the family says so by
    /// leaving the field zero — a convention the hand-written shared
    /// rope arm already had (`if (c.rope_theta == 0.f) return false;`).
    /// When the rope rows started generating, the generated branch ran
    /// FIRST and had no such refusal: it would have rotated half of
    /// gemma-4 by nothing, silently, past an arm written to prevent
    /// exactly that.
    ///
    /// So the refusal belongs to the ROW rather than to whichever arm
    /// happens to be reading the field. Zero means "not this family's",
    /// which is a claim about the context field and not about any one
    /// call site.
    CtxNonZero(&'static str),
    /// A literal VALUE — for the arguments a launcher takes that no
    /// statement and no context carries: an `interleaved` flag a family
    /// never sets, a `beta` of zero, an optional buffer left absent.
    ///
    /// A value, not a string. It was `&'static str` holding C++ source
    /// text (`"1.702f"`, `"nullptr"`, `"false"`), which meant every
    /// consumer that was not C++ had to PARSE another language's syntax
    /// to find out what number the row meant. `1.702f` is not a float in
    /// any language that reads this table; it was a float in the one
    /// language that wrote it.
    Lit(Lit),
}

/// What a [`Source::Lit`] holds.
///
/// Deliberately small. A row states a constant the launcher needs and no
/// statement supplies; anything that wants arithmetic or a name is a
/// different `Source`, not a richer literal.
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
    /// The source file that defines the entry point, relative to the backend's
    /// shader directory.
    ///
    /// Metal compiles at run time from `(path, entry name)`, so a symbol alone
    /// cannot be built — and the only things that knew the file were the
    /// hand-written per-family plans, which is one more fact a driver knew and
    /// a statement did not. `None` means the row has not said; a backend that
    /// links its kernels (CUDA) never needs it, because the linker knows.
    pub file: Option<&'static str>,
    /// How a rectangle becomes a thread grid. See [`LaunchRule`].
    pub launch: LaunchRule,
    /// The kernel REFUSES a row split: it may not be stated inside a peel's
    /// regions, because its addressing (a fire-wide prepare, a padded staging
    /// buffer) is not row-offsettable. `model-compiler`'s `OpKind::Peel` is
    /// the op this refuses, and its `check_plan` is what enforces the refusal.
    pub whole: bool,
    /// The host plan its contract obligates.
    pub needs: Prepare,
    /// Capabilities this kernel cannot serve — a seam asking for one of these
    /// over rows this kernel covers is unservable.
    pub lacks: &'static [Cap],
    /// Where a sink-writing seam's output lands, if this kernel accepts one
    /// (`sink pages -> kv.pages`).
    pub sink: Option<&'static str>,
    /// Which OUTPUTS this kernel writes over which INPUTS, as
    /// `(output index, input index)` pairs.
    ///
    /// `launch_residual_add_bf16(y, x, n)` writes its result over `y`, so
    /// its row says `in_place = &[(0, 0)]`. That is a fact about the
    /// KERNEL and not about any statement using it — every call of it is
    /// in-place — which is why it lives here rather than at the call
    /// site.
    ///
    /// A LIST because one pair cannot say it. `launch_rope_bf16(q, k,
    /// ...)` rotates both tensors where they lie, which is two
    /// independent aliases, and the single-index form this field used to
    /// have could express neither of them together nor either of them
    /// honestly.
    ///
    /// `lower::Buffers` is what reads it: an in-place op's output takes
    /// its operand's OFFSET instead of an allocation of its own. Without
    /// that, a text accumulating into a WINDOW (gemma3n's per-layer
    /// embedding, added back into K-1 corrected AltUp streams) would
    /// produce fresh values nothing downstream reads, and the streams
    /// would silently stay pre-update — and an executor that binds
    /// operands from the trace hands one pointer to a kernel that means
    /// to read and write it, having written nothing there.
    pub in_place: &'static [(u32, u32)],
    /// On a union tail layer this dispatch pairs the DEPTH PREFIX plan (and
    /// its dedicated workspace) instead of the fire's own plan.
    ///
    /// This was the `PrefixPlanSwap` half of the retired per-op `DepthRole` —
    /// a word the IR carried on one launch per layer of every depth-declaring
    /// trace, restating a fact about the KERNEL. Migration step 5 moved it
    /// here.
    pub depth_prefix_plan: bool,
    /// The operands `symbol` takes, in order — the launch ABI, as data.
    ///
    /// Empty means UNSTATED, not "takes nothing": a launcher that genuinely
    /// took no operands would still take a stream. Rows are being filled in a
    /// family at a time (`rope` is the pilot), so an empty list is how a row
    /// says it has not been done yet, and nothing may infer a nullary call
    /// from one.
    ///
    /// Stating this is what lets the DECLARATION be generated from the row
    /// instead of written beside it. That turns the crate's own invariant —
    /// every symbol resolves to exactly one declaration, every declaration
    /// has exactly one row — from a check into a tautology, the same way
    /// deriving `symbol` from the module path did for names. It is also the
    /// only way this contract can be PROVEN: a generated shim that calls the
    /// real function makes a wrong row a C++ compile error, so the compiler
    /// is the oracle and no golden can drift.
    ///
    /// Default arguments are not representable and that is deliberate. A row
    /// lists every operand the callee has, defaulted or not, because a
    /// caller that is not C++ cannot omit one — and because a default is a
    /// choice the table should be able to see.
    pub operands: &'static [Operand],
    /// What the launcher RETURNS, spelled as C++ spells it.
    ///
    /// `""` — the default — means `void`, which is what a launcher is
    /// nearly always. Three are not: `gemv3_bf16`, `rmsnorm_bf16_tuned`
    /// and `lm_head_argmax_chunked` return `bool`, and the bool means
    /// "did the fused/tuned form run" rather than "did it succeed".
    ///
    /// It is on the ROW rather than inferred because the shim has to
    /// declare the forwarding pointer's full type: a `void` forward to a
    /// `bool` launcher is a conversion C++ refuses, which is how these
    /// three were found. Stating it is also the honest reading — a
    /// launcher that answers something is a different contract from one
    /// that only acts, and a table that could not tell them apart was
    /// describing the second and meaning the first.
    ///
    /// The generated entry point returns the same type; a caller that
    /// ignores it is doing what the C++ call sites already do.
    pub returns: &'static str,
    /// The axes `symbol` is instantiated over, if it names a FAMILY of
    /// entrypoints rather than one.
    ///
    /// Empty is the CUDA case and the default: a launcher there is an authored
    /// C++ function, so one row is one symbol and [`sig_in`] matches it whole.
    ///
    /// Metal's are generated. `quantized_qmm_t.metal` holds one template body
    /// and a macro that stamps it over `(group, bits) × (bm, bn)`, so `54` of
    /// its entrypoints are one kernel evaluated at 54 points. Enumerating them
    /// as 54 rows would state the macro's job a second time, by hand, and
    /// `.wiki/kernel-refactor.md` §5's own rule — *would the two share one C++
    /// definition?* — says they are not distinct kernels. So the row is the
    /// base and the axes are declared beside it.
    ///
    /// This is not a Metal-only idea. CUDA writes the same product into
    /// FILENAMES (`attn/flashinfer_hd{64,128,256,512}.cu`,
    /// `attn/xqa_gqa{2,4,8}.cu`) and cannot state it, because each of those is
    /// separately authored. When that changes, the axis is already spelled
    /// here.
    pub axes: &'static [Axis],
}

impl KernelSig {
    /// Does `symbol` name this row at one point of its axes?
    ///
    /// Order matters and is the whole implementation: the axes are declared in
    /// the order a name spells them, so this peels suffixes from the END, one
    /// axis at a time, and what must remain is the base. That refuses
    /// `qmm_t_bfloat16_gs_64_b_4` (a `bm`/`bn` short of a real entrypoint) and
    /// refuses a permuted spelling, both of which a "contains all the points"
    /// test would wave through.
    /// Does `symbol` name this row — as the kernel itself, or at one point of
    /// its axes?
    ///
    /// Both are legitimate and they come from different places. **A model text
    /// states the KERNEL**, because the axis point is a deployment fact: which
    /// affine format a checkpoint is, how wide its heads are. `dsl::metal`
    /// records `affine_qmv_fast` and the driver resolves
    /// `affine_qmv_fast_bfloat16_gs_64_b_4` at load, from `AffineFormat`. **The
    /// driver and the audit name the POINT**, because that is what a pipeline
    /// is built from.
    ///
    /// So the base resolves, and so does every point. What does not resolve is
    /// anything between them: [`Self::covers_point`] peels the axes from the
    /// END in declaration order, so a half-spelled name is refused rather than
    /// rounded to the nearest row.
    pub fn covers(&self, symbol: &str) -> bool {
        self.symbol == symbol || self.covers_point(symbol)
    }

    /// `symbol` is this row at one point of its axes — not the bare base.
    ///
    /// Order is the whole implementation: the axes are declared in the order a
    /// name spells them, so this peels suffixes from the end, one axis at a
    /// time, and what must remain is the base. That refuses
    /// `qmm_t_bfloat16_gs_64_b_4` (a tile short of a real entrypoint) and
    /// refuses a permuted spelling, both of which a "contains all the points"
    /// test would wave through.
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
    ///
    /// This is the other half of [`KernelSig::covers`], and the reason both
    /// exist: `covers` answers "is this name mine", `entrypoints` answers
    /// "what are all of mine", and `scripts/metal-kernel-audit.py` compares
    /// the second against the shader tree. A row that generates a name no
    /// shader instantiates, or misses one that exists, fails there — which is
    /// the invariant `.wiki/kernel-metal-refactor.md` §6 (1) states.
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

/// Declare one kernel. The syntax is `.wiki/tart/dsl.md` ②'s.
///
/// Operand shapes used to be excluded from this on the grounds that they
/// stayed with the emitter until the launch ABI flattened, and that stating
/// them here would duplicate it. They are admitted now, through
/// [`KernelSig::operands`] and the [`operands!`] macro, because flattening
/// the ABI is precisely what a stated operand list DOES: once the row carries
/// the signature, the C++ declaration and every non-C++ binding are generated
/// from it, and the emitter reads the row rather than knowing a second copy.
///
/// Exported so the two backend tables can declare rows in the same words. It
/// names [`KernelSig`], [`Prepare`] and [`Cap`] through `$crate`, so a table
/// crate needs no `use` beyond the macro itself.
#[macro_export]
macro_rules! kernel {
    ($name:ident $symbol:literal $(, $key:ident = $value:expr)* $(,)?) => {
        $crate::KernelSig {
            name: stringify!($name),
            symbol: $symbol,
            $($key: $value,)*
            ..$crate::KernelSig {
                name: "",
                symbol: "",
                file: None,
                launch: $crate::LaunchRule::Unstated,
                whole: false,
                needs: $crate::Prepare::None,
                lacks: &[],
                sink: None,
                in_place: &[],
                depth_prefix_plan: false,
                operands: &[],
                returns: "",
                axes: &[],
            }
        }
    };
}

/// An operand list, spelled the way the C++ declaration reads.
///
/// `name: Ty`, in the callee's parameter order, with `| null` on the ones
/// that accept an absent pointer:
///
/// ```ignore
/// operands![
///     q: BufMut, k: BufMut,
///     positions: I32s,
///     row_valid: U8s | null,
///     num_tokens: I32, theta: F32,
///     stream: Stream,
/// ]
/// ```
///
/// `| null` rather than a `?` suffix on purpose: `?` is a token a `tt` would
/// swallow ahead of the `,` separating two operands, and the arm would then
/// depend on macro lookahead rather than on anything a reader can see.
#[macro_export]
macro_rules! operands {
    ($($name:ident : $ty:ident $(| $null:ident)? $(<- $src:expr)?),* $(,)?) => {
        &[$($crate::Operand {
            name: stringify!($name),
            ty: $crate::Ty::$ty,
            nullable: $crate::operands!(@nullable $($null)?),
            source: $crate::operands!(@source $($src)?),
        }),*]
    };
    (@source) => { $crate::Source::Unbound };
    (@source $src:expr) => { $src };
    (@nullable) => { false };
    (@nullable null) => { true };
}

/// The contract for one symbol, in `table`.
///
/// A linear scan: the tables are ~100 and ~90 rows, and the call sites are
/// load-time (a declaration is traced when the model loads), not per-fire.
///
/// Exact matches on the symbol are tried first and across the WHOLE table,
/// before any row is allowed to claim `symbol` as a point of its axes. Without
/// that two-pass order a row could swallow a sibling whose base happens to end
/// in one of its points, and which row won would depend on declaration order.
///
/// CUDA's rows carry no axes, so for them the second pass never fires and this
/// is the same linear scan it always was.
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

    static TABLE: &[KernelSig] = &[
        kernel!(qmv "affine_qmv_fast", axes = &[DTYPE, AFFINE]),
        kernel!(qmm_t "affine_qmm_t", axes = &[DTYPE, AFFINE, TILE]),
        // A base that is ALSO a legal entrypoint, next to its dtyped form.
        kernel!(route_sort "moe_route_sort"),
        kernel!(router "router_topk", axes = &[DTYPE]),
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
    /// stops short of a full point set is NOT covered. This is the case a
    /// "contains all the points" test would wave through, and it is exactly
    /// the shape of the bug the table exists to catch: `decode_psos.cpp`
    /// building `"affine_qmm_t" + q` and forgetting the tile.
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
        // states the kernel, not the instantiation, because the affine format
        // is a checkpoint fact the lowering does not have. `dsl::metal` records
        // `affine_qmv_fast`; the driver resolves the point at load.
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
    ///
    /// Three macros in `sdpa_paged.metal` stamp ONE template —
    /// `sdpa_paged_decode<itype, d, v, sink, PAGES, FIXED, SG>` — at
    /// `<…, 0, false, 32>`, `<…, 32, true, 32>` and `<…, 32, true, 8>`. Same
    /// body, three points, and the first contributes no text.
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
        static T: &[KernelSig] =
            &[kernel!(sdpa_paged "sdpa_paged_decode", axes = &[DTYPE, DIM, PAGE])];

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
        assert_eq!(T[0].entrypoints().len(), 1 * 2 * 3);
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
