//! VULKAN's kernel signature table — one row per KERNEL in `kernels/`, and one
//! row is many entrypoints.
//!
//! ## Why this is shaped like Metal's and not like CUDA's
//!
//! `kernels-cuda` has one row per launcher symbol and no axes at all: its
//! device text is a C++ template, but a routine there NAMES the single
//! instantiation it wants and its row is DERIVED from that routine, so a point
//! of the template that nothing launches is not a row. A Slang compute shader
//! is the other extreme: it has exactly ONE entry point and it is always
//! called `main`, so a name like
//! `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32` cannot be a symbol at all. It
//! is the name of a SPIR-V MODULE — one `.spv`, compiled from one `.slang` with
//! one `-D` set, exactly as llama.cpp's `vulkan-shaders-gen` produces its
//! `matmul_q4_k_f16_f32.spv` and its 900 siblings.
//!
//! That lands this table on Metal's shape rather than CUDA's, for nearly
//! Metal's reason: an entrypoint here is GENERATED from a template evaluated at
//! a point, so enumerating the product by hand would state the generator's job
//! a second time. A row carries its [`Axis`]es and the product is the
//! entrypoint set.
//!
//! **The coverage is `kernels-metal`'s, deliberately.** Row for row, axis for
//! axis, point for point: 99 kernels over 480 entrypoints, the same names in
//! the same ten families. That is not imitation for its own sake — it is what
//! makes the two backends comparable. `model-ir` checks a lowered plan
//! against whichever table the deployment selected, so a text that runs on
//! Metal either runs here or names exactly which row it wanted; and a
//! divergence between the two tables is then a STATEMENT that one backend
//! covers something the other does not, rather than an accident nobody wrote
//! down.
//!
//! ## The Vulkan launch ABI, which is this backend's one real divergence
//!
//! A Metal row is positional over BUFFERS: `constant int& n [[buffer(4)]]` is a
//! buffer like any other, so a scalar and a tensor occupy the same kind of slot
//! and the row's index is the binding. Vulkan does not work that way. Binding a
//! four-byte scalar as a storage buffer costs a descriptor write, a device
//! allocation and a barrier for a number that fits in a push constant, and no
//! Vulkan backend written by anyone does it — llama.cpp puts every scalar in
//! one `layout(push_constant)` block and every tensor in a `binding`.
//!
//! So the row is read in TWO passes, and the rule is mechanical:
//!
//! * every operand whose [`kernels::Ty`] is a BUFFER kind (`Buf`, `BufMut`,
//!   `I32s`, `U32s`, `U8s`, `F32s`, and the rest of the pointer family) takes
//!   the next `layout(std430, binding = N)` slot, in row order, from 0;
//! * every operand whose kind is a SCALAR (`I32`, `U32`, `F32`, `Usize`,
//!   `Bool`, `InPacked`) takes the next field of the single
//!   `layout(push_constant)` block, in row order.
//!
//! Both passes read the SAME row in the SAME order, so nothing about which
//! operand is which moves; only where the value rides does. [`bindings`] is the
//! rule as code, so a shell binding a launch and a test checking a shader
//! compute it from one place rather than agreeing by habit.
//!
//! The rule is stated here rather than per-row on purpose: it is a property of
//! the API, not of any kernel, and a table that spelled it 99 times would let
//! row 100 spell it differently.
//!
//! ## What keeps it honest
//!
//! Three checks, at three distances — the same three the Metal tree has, with
//! the middle one doing more work because the generator is ours:
//!
//! * `kernels`' own unit tests pin the matcher — that a row covers every point
//!   of its axes and refuses a partial or permuted spelling.
//! * `tests/entrypoints.rs` pins the table's product against the shader tree,
//!   by reading the `// pie:instantiate` directives out of `kernels/`.
//! * `scripts/vulkan-kernel-audit.py` reads those same directives — the ones
//!   the BUILD compiles from, so the audit and the build cannot disagree about
//!   what exists — and
//!   `--compile` runs `slangc` over every one of them, which proves a declared
//!   variant is a variant that builds.
//!
//! And from the other end, `model-ir`'s `kernels::check_plan` refuses any
//! launched symbol no row declares, so a lowered text cannot state a kernel
//! this table has not heard of.
//!
//! ## The validation layer, which is not optional and is not installed
//!
//! Everything above is a comparison between two descriptions. Whether a DRIVER
//! agrees is a separate question, and the driver these were developed against
//! answers a malformed request by building the pipeline anyway. Three real
//! defects survived a green suite for weeks because of it: the coopmat tier
//! not naming `vulkanMemoryModel`, the baseline tier requiring an optional
//! `shaderInt64`, and 120 entrypoints declaring a push block wider than the
//! range their row builds.
//!
//! `tests/gpu.rs` enables `VK_LAYER_KHRONOS_validation` when the loader can see
//! it, with synchronization and GPU-assisted validation on, and an ERROR ends
//! the process. It is a soft dependency, because a build machine will not have
//! the layer and "no validation here" must not be a test failure — which does
//! mean a clean CI run is weaker evidence than a clean local one.
//!
//! It does not have to be installed system-wide. `apt-get download
//! vulkan-validationlayers`, `dpkg-deb -x` it somewhere, rewrite the
//! manifest's `library_path` to the absolute path of the extracted `.so`, and
//! point `VK_LAYER_PATH` at the directory holding the manifest.
//!
//! ## What a shell that RUNS these has to do
//!
//! There is no `driver-vulkan` yet, so these are written down here rather than
//! discovered by whoever writes one. All three are things a shader cannot
//! check for itself.
//!
//! * **Grids are workgroup-granular.** `vkCmdDispatch` counts WORKGROUPS where
//!   Metal's `dispatchThreads` counts threads, so a shell that ports a Metal
//!   grid arithmetically will round UP and launch invocations with no work.
//!   Every pointwise body here guards its own tail against the bound length of
//!   the buffer it writes, so an overshoot is harmless. An UNDERSHOOT is not,
//!   and it is the direction that fails silently: a lane that never launches
//!   writes nothing, the gap reads back as whatever the buffer held, and the
//!   dispatch completes. `KernelSig::grid_param`, `head_param` and
//!   `heads_param` say which of the STATEMENT's params give the shape, and a
//!   shell that builds a grid from the fire's numbers instead gets a wrong
//!   answer on any deployment that states two head shapes.
//! * **Enable `robustBufferAccess` unless there is a reason not to.** It makes
//!   an out-of-range access defined and discarded rather than undefined, which
//!   turns the worst residual class of shader bug from memory corruption into
//!   a wrong number. The GPU tests enable it and say plainly that doing so
//!   makes the tail guards unobservable -- the guards are there for a shell
//!   that turns it off.
//! * **One tensor, one descriptor.** Index arithmetic in these shaders is
//!   32-bit, which is safe because
//!   `VkPhysicalDeviceLimits::maxStorageBufferRange` is a `uint32_t` and so a
//!   bound range is at most 4 GiB - 1. A shell that means to address more than
//!   that has a binding problem to solve, not a shader to change.
//! * **An UNSTATED row does not describe a layout, and mistaking that for an
//!   empty one is fatal.** 56 of the 99 rows name no operands.
//!   [`buffer_count`] answers 0 for them honestly -- the row describes nothing
//!   -- but the shader behind the name still declares its bindings, so a
//!   layout built from such a row is missing every descriptor the module
//!   reads. That is not a request a driver rejects: on the machine this was
//!   written against it is a segmentation fault inside
//!   `vkCreateComputePipelines`.
//!
//!   It does NOT mean those 292 entrypoints are unlaunchable, which is what
//!   this said first. `driver-metal/src/lowering/dispatch.rs` shows the other
//!   source: where `sig.operands.is_empty()`, it falls back to the lowered
//!   plan's own argument order. The row's operand list is a reordering and
//!   verification layer over the plan, not the only description of it, and a
//!   Vulkan shell has the same fallback -- it needs a descriptor COUNT at
//!   layout time, and the plan has one. What a shell must not do is build a
//!   layout from the row and dispatch anyway.
//!
//! ## Reading this without a GPU, and without a Vulkan SDK
//!
//! All of the above runs anywhere. `default-features = false` gives the table
//! and nothing else — which is what `model-ir` wants and all it wants —
//! and `native` adds the `slangc` pass that turns the Slang tree into SPIR-V.
//! Unlike Metal, that pass is not optional for a shell that means to RUN:
//! `vkCreateShaderModule` takes words, not source.

mod capability;
pub use crate::capability::Capability;

use kernels::KernelSig;
// Named only by the doc links above, which rustdoc still has to resolve.
#[allow(unused_imports)]
use kernels::Axis;

pub mod axes;

/// The routine machinery, this backend's instantiation of it.
///
/// The crossing described by `.wiki/kernel-x/vulkan-refactor.md`: a kernel
/// becomes an ordinary `fn` whose table row is derived from its signature,
/// replacing a `kernel!` row whose launch rule and operand list were data for
/// `driver-vulkan` to interpret. Families cross one at a time and the two
/// tables coexist until the last one has.
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

/// The family tables, concatenated.
///
/// A `const fn` fold rather than a `Vec`, so the whole table stays a `&'static`
/// the compiler can read at load with no allocation — the same shape both
/// sibling tables use for the same reason.
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
    file: None,
    launch: kernels::LaunchRule::Unstated,
    whole: false,
    lacks: &[],
    sink: None,
    in_place: &[],
    depth_prefix_plan: false,
    // A standing fact about this table, no longer annotating a field: no
    // kernel here is a mamba block, and `Prepare::Ssm` does not appear in
    // this file. That is why the aux-slot field this crate used to carry
    // was always empty. Step 9 measured that field CUDA-only and
    // consolidated it onto `kernels_cuda::x::Contract`, so the reason
    // outlived the field.
    args: &[],
    operands: &[],
    axes: &[],
    grid_param: None,
    head_param: None,
    heads_param: None,
    rows_param: None,
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name,
        symbol: k.symbol,
        file: k.file,
        launch: k.launch,
        whole: k.whole,
        lacks: k.lacks,
        sink: k.sink,
        in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan,
        args: k.args,
        operands: k.operands,
        axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param,
        heads_param: k.heads_param,
        rows_param: k.rows_param,
    }
}

/// The families laid end to end.
///
/// A `static` rather than a `const`: `KERNELS` borrows it, and borrowing a
/// `const` promotes a fresh copy of the whole table rather than pointing at
/// one. The initialiser is still a const-evaluated loop, because a `static`
/// may be built by one.
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
/// The set `scripts/vulkan-kernel-audit.py` compares against the shader tree,
/// and — one for one — the set of `.spv` module names a `native` build writes.
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
/// shader is still in the tree, `build.rs` still compiles it, and the driver
/// still dispatches it -- through `driver-vulkan/src/arm.rs`'s stem lookup
/// rather than through a row. So the name has to be stated somewhere, or
/// `tests/entrypoints.rs` would read a family that crossed successfully as a
/// family whose shaders had vanished, and the comparison against
/// `kernels-metal` -- which has not retired these rows -- would report drift
/// where there is none.
///
/// This list shrinks to nothing in the other direction: when the last family
/// crosses, `KERNELS` is empty and this is the whole census. That is
/// `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 4, and it is why this is a
/// list of families rather than one flat slice.
const RETIRED: &[&[&str]] = &[
    sample::ENTRYPOINTS,
    ptir::ENTRYPOINTS,
    mlp::ENTRYPOINTS,
    layout::ENTRYPOINTS,
    rope::ENTRYPOINTS,
    norm::ENTRYPOINTS,
    ssm::ENTRYPOINTS,
    moe::ENTRYPOINTS,
    attn::ENTRYPOINTS,
    quant::ENTRYPOINTS,
];

/// The rows that have been retired, by the name their `kernel!` call had.
///
/// [`RETIRED`] answers "what can still be dispatched"; this answers "what used
/// to be a row here and is one in `kernels-metal` still". The parity tests in
/// `tests/entrypoints.rs` scrape both crates' sources and compare row for row,
/// so during the crossing they need to know which of the sibling's rows have
/// no counterpart here on purpose. It empties when the last family crosses and
/// the parity tests retire with it.
#[must_use]
pub fn retired_rows() -> &'static [&'static str] {
    &[
        "argmax_logits",
        "copy_logits_bf16",
        "geglu_tanh",
        "geglu_tanh_strided",
        "gptoss_swiglu",
        "silu_mul",
        "embed_gather_4bit",
        "embed_gather_mb_4bit",
        "embed_gather_scaled_4bit",
        "embed_gather_scaled_mb_4bit",
        "ple_combine",
        "row_gather",
        "neox_decode",
        "neox_mb",
        "neox_prop_decode",
        "neox_prop_mb",
        "neox_freqs_decode",
        "neox_freqs_mb",
        "neox_strided",
        "gated_rms",
        "gated_rms_strided",
        "layer_scalar_mul",
        "add_bias",
        "residual_add",
        "residual_add_strided",
        "rms_residual",
        "rms_residual_scaled",
        "rms_single_row",
        "rms_strided_head_row",
        "rms_strided_row",
        "vnorm_single_row",
        "gdn_core",
        "gdn_core_recurrent",
        "gdn_core_recurrent_prefill",
        "gdn_core_recurrent_slotted",
        "gdn_core_slotted",
        "gdn_prep",
        "gdn_prep_prefill",
        "gdn_prep_slotted",
        "router_topk",
        "router_topk_scaled",
        "route_sort",
        "route_gather",
        "combine_sorted",
        "shared_expert_combine",
        "shared_expert_combine_strided",
        "qmv_routed",
        "qmv_routed_bias",
        "mxfp4_qmv_routed_bias",
        "qmm_t_routed",
        "qmm_t_routed_fp16",
        "mxfp4_qmm_t_routed_bias",
        "sdpa_paged_decode",
        "sdpa_paged_decode_sink",
        "sdpa_paged_tiled",
        "sdpa_paged_tiled_sink",
        "sdpa_paged_tiled_strided",
        "sdpa_paged_mma",
        "sdpa_paged_mma_sink",
        "sdpa_vector_decode",
        "sdpa_vector_decode_swa",
        "sdpa_vector_decode_sink",
        "kv_append",
        "kv_append_paged",
        "split_qkv_bf16",
        "gate",
        "q_gate_split",
        "logit_softcap",
        "cast_qmm_input_bfloat16_to_float16",
        "cast_qmm_input_strided_bfloat16_to_float16",
        "encode_u4_bf16",
        "encode_u4_f32",
        "mxfp4_dequant_bf16",
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
        "qmv_tail",
        "qmv_tail_bias",
        "qmv_wide_strided",
    ]
}

/// Every entrypoint whose row is gone and whose routine now answers for it.
#[must_use]
pub fn retired() -> Vec<&'static str> {
    RETIRED.iter().flat_map(|f| f.iter().copied()).collect()
}

/// Every routine this backend has crossed, with the backend forgotten.
///
/// The other half of [`KERNELS`] during the crossing: a kernel is a ROW until
/// its family is ported and a ROUTINE afterwards, and the union of the two is
/// what must still be the hundred.
/// `.wiki/kernel-x/refactor-bigplan.md` §8 makes that the progress bar, and
/// `kernels::routine::Declared` is the view three backends' incompatible
/// `Routine<B>` types can share.
///
/// Exposing it is what enters this backend into
/// `kernels/tests/shader_backends_agree.rs`, which from here on compares every
/// crossed body's TRACE arguments against the row it still has beside it. That
/// is the check a port most needs: a body that drops an operand, or takes two
/// in the other order, compiles, dispatches and returns `Ok` -- it binds the
/// wrong buffer to the wrong slot and computes a plausible number.
#[must_use]
pub fn declared() -> Vec<kernels::routine::Declared> {
    CROSSED
        .iter()
        .flat_map(|family| family.iter().map(kernels::routine::Routine::declared))
        .collect()
}

/// The families that have crossed. One line per family, and the list is what
/// [`KERNELS`] is being emptied into.
const CROSSED: &[&[routine::Routine]] = &[
    attn::ROUTINES,
    layout::ROUTINES,
    mlp::ROUTINES,
    moe::ROUTINES,
    norm::ROUTINES,
    ptir::ROUTINES,
    quant::ROUTINES,
    rope::ROUTINES,
    sample::ROUTINES,
    ssm::ROUTINES,
];

/// Every crossed routine, with its body still attached.
///
/// [`declared`] forgets the backend so that three crates can be compared;
/// this keeps it, which is what lets `tests/routines.rs` actually RUN each
/// body against a recorder and ask what it bound. The two views exist for
/// different questions and neither is derivable from the other.
#[must_use]
pub fn routines() -> Vec<&'static routine::Routine> {
    CROSSED.iter().copied().flatten().collect()
}

/// Where one operand rides. See [`bindings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    /// `layout(std430, binding = N)`.
    Buffer(u32),
    /// The `N`-th field of the push-constant block.
    Push(u32),
    /// Nowhere of its own: a FIELD of the packed buffer ahead of it.
    ///
    /// [`kernels::Ty::InPacked`] is how a row says "the driver has to supply
    /// this value, but it does not get a slot" — the value belongs to a struct
    /// some earlier `Buf` operand already binds, so the driver writes it while
    /// filling that buffer and the shader reads it as a struct member.
    ///
    /// Metal could fold this into the scalar run, because there a packed slot
    /// IS the buffer and a trailing scalar lands in the same argument. Vulkan
    /// splits the two runs, so folding it into the push block would push a word
    /// no shader reads and leave the struct field unwritten — the defect this
    /// variant exists to make unrepresentable.
    Packed,
}

/// Which descriptor binding each operand of `sig` takes, and which
/// push-constant field, under the two-pass rule this module's own docs state.
///
/// Both runs are indexed from zero and both follow the row's order, so the
/// answer for operand `k` is a function of the row alone — which is the point:
/// a shell binding a launch and a test checking a shader compute the same thing
/// from the same place.
///
/// A row with no operands is UNSTATED (see [`KernelSig::operands`]), and this
/// answers with an empty vector rather than inventing a nullary layout.
#[must_use]
pub fn bindings(sig: &KernelSig) -> Vec<Binding> {
    let mut buffers = 0;
    let mut pushes = 0;
    sig.operands
        .iter()
        .map(|op| {
            if matches!(op.ty, kernels::Ty::InPacked) {
                // Consumes neither run: see `Binding::Packed`.
                Binding::Packed
            } else if is_buffer(op.ty) {
                let at = buffers;
                buffers += 1;
                Binding::Buffer(at)
            } else {
                let at = pushes;
                pushes += 1;
                Binding::Push(at)
            }
        })
        .collect()
}

/// How many descriptor bindings a row's pipeline layout declares.
#[must_use]
pub fn buffer_count(sig: &KernelSig) -> u32 {
    sig.operands.iter().filter(|op| is_buffer(op.ty)).count() as u32
}

/// One scalar's place in the push-constant block, in BYTES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushField {
    /// The operand's name, as the row spells it.
    pub name: &'static str,
    /// Byte offset from the start of the block.
    pub offset: u32,
    /// Width in bytes: four or eight.
    pub size: u32,
}

/// The push-constant block's byte layout, which [`Binding::Push`] does NOT give.
///
/// `Binding::Push(n)` is a field INDEX, and a driver needs an offset. Turning
/// one into the other is not multiplication, because a push block follows
/// std430: a member is aligned to its own width, so an eight-byte scalar after
/// a lone four-byte one starts at 8 and not at 4. `attn/kv_write.slang` is
/// exactly that shape --
///
/// ```text
/// int head_dim; uint64_t k_head_stride; uint64_t k_seq_stride;
/// ```
///
/// -- so the naive sum of widths says 20 bytes and the real block is 24, with
/// four bytes of padding after the first field. A driver that packs by
/// concatenation writes both strides four bytes low and the shader reads two
/// halves of two different numbers. Nothing reports it: Vulkan does not know
/// what the bytes were supposed to mean.
///
/// That padding used to exist only as a hand-computed constant in a GPU test.
/// It is derived here instead, so the test and any future driver get it from
/// one place — which is the same reason [`bindings`] exists rather than each
/// of them counting buffers by hand. Two other readers have since gone:
/// `examples/dump_layout.rs` printed this layout and
/// `scripts/vulkan-kernel-audit.py --bindings` compared it to what the shaders
/// declare. Both are deleted, so this function is now the only statement of
/// the push ABI and a shader that disagrees with it is caught on a device or
/// not at all.
#[must_use]
pub fn push_layout(sig: &KernelSig) -> Vec<PushField> {
    let mut at = 0u32;
    sig.operands
        .iter()
        .filter(|op| !is_buffer(op.ty) && !matches!(op.ty, kernels::Ty::InPacked))
        .map(|op| {
            let size = push_width(op.ty);
            at = at.next_multiple_of(size);
            let field = PushField {
                name: op.name,
                offset: at,
                size,
            };
            at += size;
            field
        })
        .collect()
}

/// The push block's total size in bytes, padding included.
///
/// Rounded up to the block's own alignment — the widest member — because that
/// is what a `VkPushConstantRange` covering the whole block has to be, and
/// because `vkCmdPushConstants` takes a size that must be a multiple of four.
#[must_use]
pub fn push_size(sig: &KernelSig) -> u32 {
    let fields = push_layout(sig);
    let Some(last) = fields.last() else { return 0 };
    let align = fields.iter().map(|f| f.size).max().unwrap_or(4);
    (last.offset + last.size).next_multiple_of(align)
}

/// Every scalar kind a row can name is four or eight bytes wide.
fn push_width(ty: kernels::Ty) -> u32 {
    match ty {
        kernels::Ty::Usize | kernels::Ty::I64 => 8,
        _ => 4,
    }
}

/// Whether a kind crosses as a device allocation rather than as a value.
///
/// Read off the KIND and not off a list of operand names, so a row that grows a
/// buffer cannot land in the push block by omission. The struct-shaped and
/// handle kinds of the CUDA vocabulary are not reachable from a Vulkan row —
/// there is no stream and no cuBLAS handle here — so they answer `false`, and a
/// row that used one would put a plan cache in a push constant: a failure at
/// the row, where it can be read, rather than a silent binding.
const fn is_buffer(ty: kernels::Ty) -> bool {
    use kernels::Ty;
    matches!(
        ty,
        Ty::BufMut
            | Ty::Buf
            | Ty::I32s
            | Ty::I64s
            | Ty::U32s
            | Ty::U8s
            | Ty::F32sMut
            | Ty::F32s
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut
            | Ty::U16s
            | Ty::U16sMut
            | Ty::I8s
            | Ty::BufArray
            | Ty::BufArrayMut
            | Ty::BufArrayOut
            | Ty::BufArrayOutMut
            | Ty::U8Array
            | Ty::I32Array
            | Ty::StructuredMasks
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two runs are independent and both start at zero — the property a
    /// shell's binder and a shader's `layout` qualifiers have to agree on.
    #[test]
    fn buffers_and_push_constants_are_numbered_independently() {
        let row = kernels::sig_in(KERNELS, "rms_single_row_bfloat16").expect("a stated row");
        assert_eq!(
            bindings(row),
            vec![
                Binding::Buffer(0), // x
                Binding::Buffer(1), // w
                Binding::Buffer(2), // out
                Binding::Buffer(3), // params
            ]
        );

        let row = kernels::sig_in(KERNELS, "affine_qmv_fast_bfloat16_gs_64_b_4").expect("a row");
        assert_eq!(
            bindings(row),
            vec![
                Binding::Buffer(0), // w
                Binding::Buffer(1), // scales
                Binding::Buffer(2), // biases
                Binding::Buffer(3), // x
                Binding::Buffer(4), // y
                Binding::Push(0),   // in_vec_size
                Binding::Push(1),   // out_vec_size
            ]
        );
        assert_eq!(buffer_count(row), 5);
    }

    /// A packed field consumes NEITHER run.
    ///
    /// Found by comparing every shader's push block against the table field by
    /// field: `row_gather.slang` declared no push constants at all, because the
    /// count it needs is a member of the params struct it already binds. The
    /// table was right and `bindings()` was wrong — it had inherited Metal's
    /// "append to the scalars" reading, which would have had the driver push a
    /// word the shader never reads while `p.count` stayed whatever the params
    /// buffer happened to hold.
    /// Asked of whatever packed row the table still holds rather than of a
    /// named one: this used to name `row_gather_bfloat16`, and that row was
    /// retired when `layout` crossed. What has to stay true is the RULE, and
    /// the rule is about a kind of operand, not about one kernel.
    #[test]
    fn a_packed_field_takes_no_slot_of_its_own() {
        let packed: Vec<&KernelSig> = KERNELS
            .iter()
            .filter(|r| bindings(r).contains(&Binding::Packed))
            .collect();
        for row in packed {
            let bound = bindings(row);
            let buffers = bound
                .iter()
                .filter(|b| matches!(b, Binding::Buffer(_)))
                .count();
            assert_eq!(
                buffer_count(row),
                u32::try_from(buffers).expect("a small count"),
                "`{}`'s packed field took a descriptor slot",
                row.name
            );
            assert!(
                !bound.iter().any(|b| matches!(b, Binding::Push(_))),
                "`{}` packs a field into a block it already binds, so it \
                 pushes nothing -- a pushed word here is one the shader never \
                 reads, while the struct member stays whatever the params \
                 buffer happened to hold",
                row.name
            );
        }
    }

    /// An unstated row gets no layout rather than a nullary one.
    ///
    /// Asked of whatever unstated row the table still holds rather than of a
    /// named one: this used to name `argmax_logits_bfloat16`, and that row was
    /// retired when `sample` crossed. A test that pins a row by name is a test
    /// that fails when the refactor succeeds, which is the wrong way round.
    #[test]
    fn an_unstated_row_has_no_bindings() {
        let unstated: Vec<&KernelSig> = KERNELS.iter().filter(|r| r.operands.is_empty()).collect();
        assert!(
            !unstated.is_empty(),
            "no row states nothing, so this proves nothing -- delete it"
        );
        for row in unstated {
            assert!(bindings(row).is_empty(), "`{}` got a layout", row.name);
        }
    }

    /// `maxPushConstantsSize` is 128 bytes on the floor of the desktop Vulkan
    /// implementations (and llama.cpp treats 128 as the number to respect), so
    /// a row whose scalars overflow it is a row whose launch cannot be issued.
    ///
    /// This used to sum the widths, which is the wrong number: a push block is
    /// std430, so an eight-byte scalar after a lone four-byte one is preceded
    /// by four bytes of padding, and the sum UNDER-counts. Under-counting is
    /// the dangerous direction for a ceiling — it lets a row that really does
    /// overflow pass — so it asks [`push_size`] now.
    #[test]
    fn no_row_overflows_the_push_constant_floor() {
        for row in KERNELS {
            let bytes = push_size(row);
            assert!(
                bytes <= 128,
                "`{}` wants {bytes} bytes of push constants; the floor is 128",
                row.symbol
            );
        }
    }

    /// The padding is real, and this is the row that has it.
    ///
    /// `attn/kv_write.slang` declares `int head_dim; uint64_t k_head_stride;
    /// uint64_t k_seq_stride;`, so the block is 4 + 4 pad + 8 + 8 = 24 and not
    /// the 20 that adding the widths gives. A driver packing by concatenation
    /// writes both strides four bytes low, and the shader reads two halves of
    /// two different numbers with nothing to report it.
    #[test]
    fn an_eight_byte_scalar_after_a_four_byte_one_is_padded() {
        let row = kernels::sig_in(KERNELS, "kv_append_bfloat16").expect("a row");
        let fields = push_layout(row);
        let places: Vec<(&str, u32)> = fields.iter().map(|f| (f.name, f.offset)).collect();
        assert_eq!(
            places,
            vec![("head_dim", 0), ("k_head_stride", 8), ("k_seq_stride", 16)]
        );
        assert_eq!(push_size(row), 24);

        // A block of one width needs no padding, and must not acquire any.
        let plain =
            kernels::sig_in(KERNELS, "affine_qmv_routed_bias_bfloat16_gs_64_b_4").expect("a row");
        assert_eq!(
            push_layout(plain)
                .iter()
                .map(|f| f.offset)
                .collect::<Vec<_>>(),
            vec![0, 4, 8, 12, 16]
        );
        assert_eq!(push_size(plain), 20);
    }
}
