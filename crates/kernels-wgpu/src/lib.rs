//! WEBGPU's kernel signature table — one row per KERNEL in `kernels/`, and one
//! row is many entrypoints.
//!
//! ## A window where this crate did not compile, and what it cost
//!
//! `The column is the boundary now, and eighteen arms are what it could not
//! carry` (79c145590) redesigned `kernels::routine`, and its own message names
//! the surface it was checked against: *"`cargo check -p kernels -p
//! kernels-cuda -p driver-cuda --features cuda-12`"*. `kernels-cuda` was
//! migrated with it; the shader backends were not. Checked out alone, that
//! commit gave **743 errors here and 799 in `kernels-metal`** — a migration in
//! flight rather than a regression to bisect, and the metal sibling being in
//! the identical state was the evidence.
//!
//! Gone for that window was the vocabulary a routine signature is written in:
//! `Ask`, `Block`, `Else`, `Held`, `InSlot`, `Nth`, `Null`, `OutSlot`, `Over`,
//! `ParamOr`, `ParamOrLit`, `Reckoned`, `Say`, `Times`, plus twelve `keys`.
//! **`kernels: restore the shader planes' slot vocabulary beside CUDA's`
//! (64022e07e) put it back**, so nothing here needs porting and the note
//! survives only for what it is evidence of.
//!
//! Which is this. `a1fcf1bc9` reached `origin/rewrite` without building: its
//! tree passed the gate — `driver-wgpu` 23, `kernels-wgpu` 37/13/55/4, serving
//! 22 of 22 in release — and then `git rebase` moved it onto a base the gate
//! had never seen, and it went out unchecked. **A rebase is a change to the
//! tree and deserves the same gate a change to the source does.** This branch
//! rebases before every commit onto a very active upstream, so that is not a
//! rare hazard here; it is the normal one.
//!
//! The second lesson is about the triage itself. Between the break and the
//! restore I wrote that this crate needed a hundred signatures ported, which
//! was true of the tree in front of me and false of the tree an hour later.
//! **A note that describes a live outage has to be timestamped by something**,
//! and this one is now dated by the two commit hashes either side of it rather
//! than left reading as a standing fact.
//!
//! ## Why this crate is the one that needs no toolchain
//!
//! `kernels-cuda` costs nothing at build time and `libnvrtc.so` at RUN time: a
//! kernel there is text the process compiles when it first fires one, so a
//! process that cannot `dlopen` NVRTC cannot fire anything. `kernels-vulkan`
//! costs `glslc`, because Vulkan has no runtime shader compiler at all and a
//! pipeline is built from a SPIR-V module that something else had to produce.
//! `kernels-metal` costs a Mac to RUN, though not to read.
//!
//! WGSL costs nothing. `wgpu` carries `naga`, a WGSL front end written in Rust,
//! so the process that dispatches a kernel is the process that compiled it —
//! the Metal model, with the compiler in the dependency graph instead of in the
//! operating system. That is why this crate has no `native` feature: there is
//! no build product to gate, so the table, the shaders and every structural
//! test are reachable on any machine that can build Rust.
//!
//! ## The coverage is `kernels-metal`'s, deliberately
//!
//! Row for row, axis for axis, point for point: **100 kernels over 481
//! entrypoints**, the same names in the same ten families, and
//! `tests/entrypoints.rs` pins all three of those numbers against
//! `kernels-metal`'s own source.
//!
//! That is not imitation for its own sake. `crates/kernels` is the
//! backend-neutral vocabulary — [`kernels::KernelSig`], [`kernels::Axis`],
//! [`kernels::Ty`], [`kernels::Operand`], [`kernels::Source`],
//! [`kernels::LaunchRule`] — and `model-ir` resolves a traced program
//! against it through `kernels::sig_in` without ever learning which backend it
//! is compiling for. A backend table is therefore not a design surface. It is
//! an *answer* to a question the compiler already asked, and the answer must be
//! the same shape on every backend or the compiler's plan stops being portable.
//!
//! A divergence between two tables is then a STATEMENT that one backend covers
//! something the other does not, rather than an accident nobody wrote down. If
//! you add a wgpu-only kernel you are no longer porting — you are forking the
//! vocabulary, and that belongs in `crates/kernels` first.
//!
//! ## The WebGPU launch ABI, which is this backend's one real divergence
//!
//! A Metal row is positional over BUFFERS: `constant int& n [[buffer(4)]]` is a
//! buffer like any other, so a scalar and a tensor occupy the same kind of slot
//! and the row's index is the binding. WebGPU does not work that way, and
//! neither does Vulkan — but the two disagree about the answer, so the rule
//! here is a third one and it is worth stating precisely.
//!
//! Vulkan sends the scalar run to a `layout(push_constant)` block. **WebGPU has
//! no push constants.** `wgpu` exposes them as `Features::PUSH_CONSTANTS`,
//! which is a native-only extension no WebGPU implementation is obliged to
//! offer and which the browser backend cannot offer at all. A table that
//! depended on it would be a table that runs on `wgpu` and not on WebGPU, which
//! gives up the only thing this backend has that its siblings do not.
//!
//! So the scalar run becomes **one uniform buffer**, and the two runs are put
//! in two different bind GROUPS rather than sharing one numbering:
//!
//! * every operand whose [`kernels::Ty`] is a BUFFER kind takes the next
//!   `@group(0) @binding(N)` storage slot, in row order, from 0;
//! * every operand whose kind is a SCALAR (`I32`, `U32`, `F32`, `Usize`,
//!   `Bool`) becomes the next field of the single
//!   `@group(1) @binding(0) var<uniform> params` struct, in row order;
//! * [`kernels::Ty::InPacked`] takes neither, for the reason
//!   [`Binding::Packed`] gives.
//!
//! **Two groups, not one, and the reason is not tidiness.** Vulkan's binding
//! numbers are shared between its buffers and nothing else, because its scalars
//! left the numbering entirely; a WebGPU uniform is a binding like any other, so
//! putting it in group 0 would give it an index that MOVES with the row's
//! buffer count. Every shader in a family would then declare its params block at
//! a different number than its neighbour, and a family's shaders are one file.
//! Group 1 binding 0 is the same in all 100 rows, so a shader states it once.
//!
//! `bindings` WAS the rule as code, so that a shell binding a launch and a test
//! checking a shader computed it from one place rather than agreeing by habit.
//! It read a row's operand list, and there are no rows: a ROUTINE's signature
//! states the same list in a form the compiler checks, and the shell numbers
//! the handles an arm mints in the order the body asked for them. The rule
//! above is still the rule — it is a property of the API, not of any kernel —
//! and `driver-wgpu::lowering::routine::bind` is where it is now applied.
//!
//! ### The trap this rule exists to make unrepresentable
//!
//! `kernels-vulkan`'s own notes record sixty entrypoints that read a descriptor
//! the shell never wrote, every one of them a shader author transcribing
//! *Metal's* buffer indices into a Vulkan `binding`. The two numbers differ by
//! however many scalars precede the operand. They must never be copied across,
//! and the same is true here — with the extra wrinkle that a WGSL binding that
//! is declared and never written is not a validation error either. `wgpu` will
//! refuse a bind group that does not MATCH its layout, which is more than
//! Vulkan does, but the layout is derived from the same wrong reading.
//!
//! The OFFSETS are the other half, and turning a field index into one is not
//! multiplication: WGSL's uniform address space aligns a member to its own
//! size, so an eight-byte value after a lone four-byte one starts at 8 and not
//! at 4, and the struct itself rounds up to 16. `uniform_layout` computed that
//! from a row; `driver-wgpu::reflect::Declared::uniform_offsets` reads it off
//! the MODULE with naga, which is the shader itself rather than a description
//! of it, and `driver-wgpu::lowering::routine::bind` packs against that.
//!
//! ## What a shell that RUNS these has to do
//!
//! Three things a shader cannot check for itself. They are written down here
//! rather than discovered, because two of the three already cost the Vulkan
//! port a debugging session.
//!
//! * **Grids are workgroup-granular.** `dispatch_workgroups` counts
//!   WORKGROUPS where Metal's `dispatchThreads` counts threads, so a shell that
//!   ports a Metal grid arithmetically launches a 256th of it. Every pointwise
//!   body here guards its own tail against the length of the buffer it writes,
//!   so an overshoot is harmless. An UNDERSHOOT is not, and it is the direction
//!   that fails silently: a lane that never launches writes nothing, the gap
//!   reads back as whatever the buffer held, and the dispatch completes.
//!   A routine takes the shape as an ARGUMENT, and
//!   `driver-wgpu::lowering::arm::Handles::stated` supplies it: the
//!   statement's scalar where there is one, the fire's where there is not, and
//!   never a zero. Four `KernelSig` columns used to name those params by
//!   index; see their retirement in `kernels`.
//! * **Ask the adapter for its limits.** WebGPU's guaranteed floor is **8**
//!   storage buffers per shader stage, and `sdpa_paged_decode` binds eleven. A
//!   shell that requests `Limits::downlevel_defaults()` will fail to create
//!   those pipelines on hardware that would have run them.
//!   `over_downlevel_storage_limit` named them off the table;
//!   `driver-wgpu::Device::unreachable` names them off the MODULES, against
//!   the adapter's real limit rather than the guaranteed floor.
//! * **A workgroup size is fixed when the module compiles.** WGSL's
//!   `@workgroup_size` is a compile-time attribute (an `override` may size it,
//!   but not a uniform), where a Metal threadgroup is sized at dispatch. A body
//!   ported from Metal that assumed "one lane per channel" is correct only up
//!   to its own declared width — the defect `kernels-vulkan` records in
//!   `gated_rms`, which this tree inherits the fix for and not the bug.

pub use kernels::{Axis, Cap, KernelSig};

mod capability;
pub use crate::capability::Capability;

pub mod preproc;
pub use crate::preproc::{Directive, Malformed, Variant, expand, instantiations};

pub mod source;
pub use crate::source::{Missing, SOURCES, entrypoint_source, source};

pub mod axes;

// This backend's instantiation of the `kernels` routine machinery: the
// `Backend` impl, the argument types, and the `Encode` trait a body dispatches
// through. Declares no routine itself -- the family modules do.
pub mod routine;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod ptir;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

/// This crate's backend, under the one name `#[routine]` knows it by.
///
/// The attribute is shared by all four planes and cannot name any of them, so
/// each aliases its own here and the macro writes `crate::Plane`.
pub type Plane = crate::routine::Wgpu;

/// Every routine this crate declares.
///
/// A DISTRIBUTED SLICE, so nothing enumerates: `#[routine]` puts each row in a
/// `linkme_ROUTINES` link section beside its own `fn`, and the linker hands
/// back the bounds. There is no membership list to add a routine to, which is
/// the last hand-written thing about one -- and the last that could be
/// forgotten, leaving a routine compiled, correct and unreachable.
///
/// The order is LINK ORDER and no reader may depend on it. Nothing does:
/// lookups match on the full symbol, which `kernels-cuda`'s
/// `no_symbol_is_declared_twice` keeps unique -- one section, four crates, so
/// that proof covers this plane's rows as much as its own.
// THE SLICE'S NAME IS A LINK-SECTION NAME, AND IT IS GLOBAL.
//
// `linkme` keys a distributed slice on the STATIC's identifier, not on the
// crate that declares it, so four crates each declaring `ROUTINES` are four
// declarations of one slice -- and `linkme` 0.3.37 refuses that outright
// ("duplicate #[distributed_slice] with name \"ROUTINES\"") the moment two of
// them are linked into one binary, which `kernels`' cross-backend agreement
// test is. Before that version it was worse than a refusal: the sections
// merged, and a sweep over one plane's rows walked another's.
//
// So the declaration wears the plane's name and the ALIAS wears the one
// `#[routine]` emits. `crate::ROUTINES` still resolves, at no cost to a
// reader, and the link section is this crate's alone.
#[::linkme::distributed_slice]
pub static WGPU_ROUTINES: [::kernels::routine::Routine<Plane>];

/// The slice under the name `#[routine]` registers into.
pub use WGPU_ROUTINES as ROUTINES;

/// The family tables, concatenated.
///
/// A `const fn` fold rather than a `Vec`, so the whole table stays a `&'static`
/// the compiler can read at load with no allocation — the same shape both
/// siblings use for the same reason.
pub static KERNELS: &[KernelSig] = &CONCAT;


const FAMILIES: &[&[KernelSig]] = &[
    attn::KERNELS,
    layout::KERNELS,
    mlp::KERNELS,
    moe::KERNELS,
    norm::KERNELS,
    ptir::KERNELS,
    quant::KERNELS,
    rope::KERNELS,
    sample::KERNELS,
    ssm::KERNELS,
];

const fn total() -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < FAMILIES.len() {
        n += FAMILIES[i].len();
        i += 1;
    }
    n
}

const N: usize = total();

const EMPTY: KernelSig = KernelSig {
    name: "",
    symbol: "",
    whole: false,
    depth_prefix_plan: false,
    args: &[],
    sources: &[],
    derived: &[],
    axes: &[],
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name,
        symbol: k.symbol,
        whole: k.whole,
        sources: k.sources,
        derived: k.derived,
        depth_prefix_plan: k.depth_prefix_plan,
        args: k.args,
        axes: k.axes,
    }
}

// `static` and not `const`: `KERNELS` borrows it, so a `const` would be
// materialised at each use site, and this one is thousands of rows.
static CONCAT: [KernelSig; N] = {
    let mut out = [EMPTY; N];
    let mut at = 0;
    let mut f = 0;
    while f < FAMILIES.len() {
        let family = FAMILIES[f];
        let mut i = 0;
        while i < family.len() {
            out[at] = copy_sig(&family[i]);
            at += 1;
            i += 1;
        }
        f += 1;
    }
    out
};

/// Every entrypoint the table names, sorted.
///
/// One for one, the set of variants the `// pie:instantiate` directives in
/// `kernels/` declare.
#[must_use]
pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = KERNELS.iter().flat_map(KernelSig::entrypoints).collect();
    out.extend(
        RETIRED
            .iter()
            .flat_map(|family| family.iter().map(|n| (*n).to_owned())),
    );
    out.sort();
    out
}

/// The entrypoints of the families whose `kernel!` rows have been RETIRED.
///
/// The crossing moves who NAMES an entrypoint, not whether it exists: the
/// shader is still in the tree, `Embedded` still holds it, and the driver
/// still dispatches it — through `driver-wgpu/src/lowering/arm.rs` rather
/// than through a row. So the name has to be stated somewhere, or every sweep
/// keyed on [`entrypoints()`] would read a family that crossed SUCCESSFULLY
/// as a family whose shaders had vanished, and stop covering it in silence.
///
/// This list shrinks to nothing in the other direction: when the last family
/// crosses, `KERNELS` is empty and this is the whole census. That is
/// `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 4, and it is why this is a
/// list of families rather than one flat slice.
const RETIRED: &[&[&str]] = &[
    sample::ENTRYPOINTS,
    ptir::ENTRYPOINTS,
    mlp::ENTRYPOINTS,
    norm::ENTRYPOINTS,
    layout::ENTRYPOINTS,
    rope::ENTRYPOINTS,
    quant::ENTRYPOINTS,
    moe::ENTRYPOINTS,
    ssm::ENTRYPOINTS,
    attn::ENTRYPOINTS,
];

/// The rows that have been retired, by the name their `kernel!` call had.
///
/// `RETIRED` answers *"what can still be dispatched"*; this answers *"what
/// used to be a row here and is one in `kernels-metal` still"*. The parity
/// tests in `tests/entrypoints.rs` scrape both crates' sources and compare row
/// for row, so during the crossing they need to know which of the sibling's
/// rows have no counterpart here on purpose.
#[must_use]
pub fn retired_rows() -> &'static [&'static str] {
    &[
        "argmax_logits",
        "copy_logits_bf16",
        "geglu_tanh",
        "geglu_tanh_strided",
        "gptoss_swiglu",
        "silu_mul",
        "silu_mul_strided",
        "encode_u4_bf16",
        "encode_u4_f32",
        "mxfp4_dequant_bf16",
        "add_bias",
        "gated_rms",
        "gated_rms_strided",
        "layer_scalar_mul",
        "residual_add",
        "residual_add_strided",
        "rms_residual",
        "rms_residual_scaled",
        "rms_single_row",
        "rms_strided_head_row",
        "rms_strided_row",
        "vnorm_single_row",
        "embed_gather_4bit",
        "embed_gather_mb_4bit",
        "embed_gather_scaled_4bit",
        "embed_gather_scaled_mb_4bit",
        "ple_combine",
        "row_gather",
        "neox_decode",
        "neox_freqs_decode",
        "neox_freqs_mb",
        "neox_mb",
        "neox_prop_decode",
        "neox_prop_mb",
        "neox_strided",
        "cast_qmm_input_bfloat16_to_float16",
        "cast_qmm_input_strided_bfloat16_to_float16",
        "combine_sorted",
        "gdn_core",
        "gdn_core_recurrent",
        "gdn_core_recurrent_prefill",
        "gdn_core_recurrent_slotted",
        "gdn_core_slotted",
        "gdn_prep",
        "gdn_prep_prefill",
        "gdn_prep_slotted",
        "logit_softcap",
        "mxfp4_qmm_t_routed_bias",
        "mxfp4_qmv_routed_bias",
        "gate",
        "kv_append",
        "kv_append_paged",
        "sdpa_paged_decode",
        "sdpa_paged_decode_sink",
        "sdpa_paged_mma",
        "sdpa_paged_mma_sink",
        "sdpa_paged_tiled",
        "sdpa_paged_tiled_sink",
        "sdpa_paged_tiled_strided",
        "sdpa_vector_decode",
        "sdpa_vector_decode_sink",
        "sdpa_vector_decode_swa",
        "router_topk",
        "qmv_routed",
        "q_gate_split",
        "qmm_splitk_reduce",
        "qmm_splitk_reduce_f32",
        "qmm_t",
        "qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
        "qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
        "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
        "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
        "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
        "qmm_t_bias",
        "qmm_t_bias_fp16_precast",
        "qmm_t_fp16_precast",
        "qmm_t_residual",
        "qmm_t_residual_fp16_precast",
        "qmm_t_routed",
        "qmm_t_routed_fp16",
        "qmm_t_splitk",
        "qmm_t_splitk_f32",
        "qmm_t_splitk_fp16_precast",
        "qmm_t_splitk_fp16_precast_f32",
        "qmm_t_strided",
        "qmm_t_strided_fp16_precast",
        "qmm_t_strided_fp16_precast_residual",
        "qmm_t_strided_residual",
        "qmv_fast",
        "qmv_fast_residual",
        "qmv_routed_bias",
        "qmv_tail",
        "qmv_tail_bias",
        "qmv_wide_strided",
        "route_gather",
        "route_sort",
        "router_topk_scaled",
        "shared_expert_combine",
        "shared_expert_combine_strided",
        "split_qkv_bf16",
    ]
}

/// Every entrypoint whose row is gone and whose routine now answers for it.
#[must_use]
pub fn retired() -> Vec<&'static str> {
    RETIRED.iter().flat_map(|f| f.iter().copied()).collect()
}

/// Every routine this backend has crossed.
///
/// The families that have crossed, flattened. One line per family in
/// `CROSSED`, and the list is what [`KERNELS`] is being emptied into.
#[must_use]
pub fn routines() -> Vec<&'static routine::Routine> {
    
    ROUTINES.iter().collect()
}

/// Every routine this backend has crossed, with the backend forgotten.
///
/// The other half of [`KERNELS`] during the crossing: a kernel is a ROW until
/// its family is ported and a ROUTINE afterwards, and the union of the two is
/// what must still be the hundred. `.wiki/kernel-x/refactor-bigplan.md` §8
/// makes that the progress bar, and `kernels::routine::Declared` is the view
/// three backends' incompatible `Routine<B>` types can share.
#[must_use]
pub fn declared() -> Vec<kernels::routine::Declared> {
    ROUTINES
        .iter()
        .map(kernels::routine::Routine::declared)
        .collect()
}

/// The row that covers `symbol`, or `None`.
///
/// A thin forward to [`kernels::sig_in`] over this crate's table, so a caller
/// that has the symbol and not the table does not have to name both.
#[must_use]
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(KERNELS, symbol)
}

/// Where one operand rides. See this module's header for the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    /// `@group(0) @binding(N) var<storage, ...>`.
    Storage(u32),
    /// The `N`-th field of the `@group(1) @binding(0) var<uniform>` block.
    Uniform(u32),
    /// Nowhere of its own: a FIELD of the packed buffer ahead of it.
    ///
    /// [`kernels::Ty::InPacked`] is how a row says "the driver has to supply
    /// this value, but it does not get a slot" — the value belongs to a struct
    /// some earlier `Buf` operand already binds, so the driver writes it while
    /// filling that buffer and the shader reads it as a struct member.
    ///
    /// Metal could fold this into the scalar run, because there a packed slot
    /// IS the buffer and a trailing scalar lands in the same argument. Both
    /// this backend and Vulkan split the two runs, so folding it into the
    /// uniform block would push a word no shader reads and leave the struct
    /// field unwritten — the defect this variant exists to make
    /// unrepresentable. `layout/row_gather` is the row that has it: it declares
    /// no uniform block at all while the row states `count`, because `count` is
    /// the second field of the `RowGatherParams` struct that a storage buffer
    /// already binds.
    Packed,
}

/// WebGPU's guaranteed floor for storage buffers in one shader stage.
///
/// `wgpu::Limits::downlevel_defaults().max_storage_buffers_per_shader_stage`,
/// restated as a number so this crate can name it without depending on `wgpu`.
/// A kernel above it is not wrong; it is one whose pipeline needs a device
/// that reports more than the floor. `driver-wgpu::Device::unreachable` is how
/// a shell finds out which, before a model is loaded — counted off the modules
/// that have to be bound, and against the adapter's REAL limit rather than
/// this floor.
pub const DOWNLEVEL_STORAGE_BUFFERS: u32 = 8;

/// One scalar's place in the uniform block, in BYTES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformField {
    /// The operand's name, as the row spells it.
    pub name: &'static str,
    /// Byte offset from the start of the block.
    pub offset: u32,
    /// Width in bytes: four, or eight for a 64-bit value.
    pub size: u32,
    /// Whether the field is a 64-bit value with no WGSL scalar type.
    ///
    /// WGSL has `u32`, `i32` and `f32` and no 64-bit integer at all. A
    /// [`kernels::Ty::Usize`] or [`kernels::Ty::I64`] operand is therefore
    /// declared as `vec2<u32>` — low word first — which is what gives it an
    /// eight-byte alignment as well as an eight-byte width, and is the same
    /// answer `kernels-vulkan` arrived at from the other direction when a
    /// validation layer objected to its `uint64_t` requiring `shaderInt64`.
    pub split: bool,
}

/// WebGPU's guaranteed floor for one uniform binding's size, in bytes.
///
/// `wgpu::Limits::downlevel_defaults().max_uniform_buffer_binding_size` is
/// 16 KiB, which no row comes close to. It is named so `tests/entrypoints.rs`
/// can pin the ceiling the way `kernels-vulkan` pins its 128-byte push limit —
/// the check is cheap and the failure it prevents is a pipeline that refuses to
/// build on the one device that mattered.
pub const DOWNLEVEL_UNIFORM_BYTES: u32 = 16 * 1024;

/// Whether a kind crosses as a device allocation rather than as a value.
///
/// Read off the KIND and not off a list of operand names, so a row that grows a
/// buffer cannot land in the uniform block by omission. The struct-shaped and
/// handle kinds of the CUDA vocabulary are not reachable from a row here —
/// there is no stream and no cuBLAS handle in WebGPU — so they answer `false`,
/// and a row that used one would put a plan cache in a uniform: a failure at
/// the row, where it can be read, rather than a silent binding.
///
/// THROUGH [`kernels::Ty::binds`] AND NOT AN ENUMERATION HERE. The list this
/// replaced had drifted twice over: `Ty::Buf`, `Ty::F32s`, `Ty::I32s`,
/// `Ty::U32s` and `Ty::U8s` each appeared TWICE — the residue of the
/// `Buf`/`BufMut` merge rewriting every `XMut` arm to `X` — and `Ty::Bf16s`
/// was absent, which would have put an activation buffer in a uniform the
/// moment a signature named its element. A classification stated once in
/// `kernels` cannot drift from itself, and a `Ty` added later is classified
/// there rather than in three backends.
pub const fn is_buffer(ty: kernels::Ty) -> bool {
    !matches!(ty.binds(), kernels::Binds::Nothing)
}

#[cfg(test)]
mod tests {
    use super::*;

    // RETIRED: `sdpa_paged_decode` was the example and `attn` has retired.
    //
    // It asserted the launch ABI's central claim — that a row's STORAGE
    // bindings and its UNIFORM fields are two independent numberings, so an
    // operand in one does not shift a field in the other.
    // `every_row_places_every_operand_exactly_once` makes the same claim over
    // every row that remains rather than over one example, and the routine
    // plane's `bind` keeps the two apart structurally: buffers come from the
    // body's handles and scalars from its `ArgValue`s, and neither can
    // renumber the other.

    // RETIRED: `row_gather` was the only row stating a PACKED operand, and
    // `layout` has crossed.
    //
    // The test asserted that a `Ty::InPacked` operand takes no `@group(0)`
    // slot of its own — it is a FIELD of the params struct an earlier buffer
    // binds — and `row_gather_bfloat16` was the one row in the table it could
    // ask about. `bindings()` still answers `Binding::Packed` and
    // `every_row_places_every_operand_exactly_once` still walks every row it
    // returns, so the function is not unchecked; what is gone is the one
    // example, and inventing a synthetic row to keep the sentence would be
    // asserting that this file can build a struct, not that the table does.
    //
    // The claim lives on the routine side now: `layout::row_gather` passes
    // `InPacked(count)` as a scalar and
    // `driver-wgpu::lowering::routine::bind` appends it to the storage
    // block's run rather than giving it a binding.
    //
    // RETIRED: THE TABLE IS EMPTY, so there is no unstated row to pick.
    //
    // It asserted that a row carrying axes and a name and no operands answers
    // with nothing rather than with a nullary layout — not unlaunchable (a
    // shell falls back to the lowered plan's own argument order, which is what
    // `driver-metal` does) but not launchable from HERE, and the difference
    // was the whole content of the check.
    //
    // The row it picked was replaced four times as each became armable:
    // `sdpa_paged_tiled`, then `gdn_core`, then `gate`, then
    // `silu_mul_strided` — which the comment above called "the LAST: it has no
    // routine on any backend, so no arm can ever retire it". That was
    // inherited from metal and it was wrong here; see `mlp::KERNELS`.
    //
    // The rule survives where it is now reachable: `driver-wgpu`'s
    // `lowering::routine::bind` builds the same two numberings from a body's
    // handles and `ArgValue`s, and `driver-wgpu::tests::arena`'s walks assert
    // it over every rectangle of every real lowering rather than over one row.

    /// The table is `kernels-metal`'s coverage, and these are the numbers.
    ///
    /// Pinned here as well as in `tests/entrypoints.rs` because a `cargo test
    /// -p kernels-wgpu --lib` should be able to say whether the port is whole.
    #[test]
    fn the_table_is_one_hundred_rows_over_four_hundred_and_eighty_one_entrypoints() {
        // Rows PLUS retired rows: `refactor-bigplan.md` §7 empties the table
        // family by family, and the hundred is what the two together name.
        assert_eq!(
            KERNELS.len() + retired_rows().len(),
            100,
            "one row per kernel in `kernels/`, retired or not"
        );
        assert_eq!(entrypoints().len(), 481, "the product of every row's axes");
        assert_eq!(
            KERNELS.len(),
            0,
            "THE TABLE IS EMPTY. Every one of the hundred kernels this crate \
             carries is reached through a ROUTINE and an ARM, and nothing in \
             this crate describes a launch positionally any more. It was 100 \
             rows over 481 entrypoints when the port landed, 52 of them \
             unstated. `refactor-bigplan.md` §7 Stage 3 is COMPLETE for wgpu, \
             which was the last backend holding rows."
        );
    }

    // RETIRED: `kv_append` was the last row here with a 64-bit uniform field
    // after a narrow one, and `attn` has retired.
    //
    // It asserted WGSL's alignment on a real row: a `Usize` is declared
    // `vec2<u32>`, so it is eight-aligned as well as eight wide and a `u32`
    // before it leaves four bytes of padding. The rule itself is not gone —
    // `uniform_layout` still applies it and
    // `every_row_places_every_operand_exactly_once` still walks every row it
    // returns — and the routine plane makes the same run in
    // `driver-wgpu::lowering::routine::bind`, whose own packer states the rule
    // and is exercised by every armed kernel that carries a scalar.

    // RETIRED: THE TABLE IS EMPTY, so `max()` has nothing to take.
    //
    // It asserted that no row's uniform block exceeds what WebGPU guarantees,
    // and — because 16 KiB is not a real constraint here, unlike the 128 bytes
    // `kernels-vulkan` pins — that the widest row was far under it, so the
    // ceiling would stop being decorative before it stopped being true.
    //
    // It refused to go vacuous (`.expect("the table is not empty")`) and that
    // is how it announced itself when the last three rows went, rather than
    // passing over an empty iterator. The rule now lives where the bytes are
    // actually built: `Ceiling::UniformBinding` refuses a block over the
    // ADAPTER's real limit at run time, and
    // `driver-wgpu::lowering`'s walk asserts `call.uniform.len() <=
    // DOWNLEVEL_UNIFORM_BYTES` over every rectangle every real lowering
    // produces — which is a stronger statement than this one made, since a
    // routine's block is packed per call and a row's was a static shape.

    // RETIRED: every row over the downlevel storage floor was an attention
    // row, and `attn` has retired.
    //
    // It named the rows binding more than eight storage buffers — the
    // WebGPU downlevel limit — so that a NEW one could not appear unnoticed.
    // `Ceiling::StorageBinding` is where that is refused at run time against
    // the adapter's real limit, which is the check that matters and is not
    // keyed on the table; and `driver-wgpu::device`'s
    // `every_entrypoint_in_the_tree_builds_a_pipeline_on_this_adapter` builds all 481
    // entrypoints, so a kernel the adapter cannot lay out fails there by name.
}

/// What a generic routine's element type must be on this plane.
///
/// `#[routine]` puts this on every type parameter that states no bound of its
/// own, so a generic signature reads the same on all four planes and the
/// plane's own requirement is said HERE, once. A shader plane binds a handle
/// and asks nothing more of an element than that it be one.
pub trait RoutineElem: kernels::Elem {}

impl<T: kernels::Elem> RoutineElem for T {}
