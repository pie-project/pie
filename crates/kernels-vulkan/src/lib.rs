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

/// The family tables, concatenated. THERE ARE NO ROWS.
///
/// This was `&CONCAT`: a `FAMILIES` list, a `total()`/`N` const fold, an
/// `EMPTY` filler row, a field-by-field `copy_sig` because `KernelSig` is not
/// `Copy` in a const context, and a const-evaluated loop that laid ten family
/// tables end to end so the whole hundred stayed a `&'static` the compiler
/// could read at load with no allocation. About a hundred and forty lines, and
/// the last family crossing made all of it a machine for concatenating
/// nothing.
///
/// The NAME stays, and stays deliberately, for exactly the reason
/// `kernels-metal` gives at its own copy of this line:
/// `kernels/tests/shader_backends_agree.rs` reads all three backends through
/// it, and an empty table and an absent one are the same shape to that gate
/// but not the same fact. This backend FINISHED. `refactor-bigplan.md` §7
/// Stage 5 deletes `KernelSig` itself, once `kernels-wgpu` -- the last with
/// rows -- can write this line too.
///
/// Nothing in this crate reads it, and nothing in `driver-vulkan` names
/// `KernelSig` at all.
pub static KERNELS: &[kernels::KernelSig] = &[];

// The rest of what stood here, and where it went:
//
// `pub static KERNELS: &[KernelSig] = &CONCAT` was backed by the
// `FAMILIES` list, the `total()`/`N` const fold, the `EMPTY` row, `copy_sig`,
// and the const-evaluated `CONCAT` loop that laid ten family tables end to
// end -- a hundred and forty lines whose only job was to make one `&'static
// [KernelSig]` the compiler could read at load with no allocation.
//
// All ten families are `&[]`. Every one has crossed to a routine, so the fold
// concatenated nothing into a zero-length array and `KERNELS` was an empty
// slice that thirty-odd readers across three crates went on consulting -- each
// of them, on the day its family crossed, quietly checking nothing.
//
// What names this crate's kernels now is `retired_rows()`, which is the whole
// hundred, and `routines()`, which is what serves them. `entrypoints()` below
// is unchanged in meaning and simpler in fact: it was the table's names plus
// the retired ones, and it is the retired ones.
//
// `kernels-wgpu` is the last crate holding rows. When it finishes, the
// `kernel!` macro and `KernelSig`'s sixteen fields go with it -- Stage 5 --
// and nothing in this crate will have to move for that to happen.

/// Every entrypoint the table names, sorted.
///
/// The set `scripts/vulkan-kernel-audit.py` compares against the shader tree,
/// and — one for one — the set of `.spv` module names a `native` build writes.
pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = RETIRED
        .iter()
        .flat_map(|family| family.iter().map(|n| (*n).to_owned()))
        .collect();
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
        "silu_mul_strided",
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

// THE LAYOUT HALF STOOD HERE -- `Binding`, `bindings`, `buffer_count`,
// `PushField`, `push_layout`, `push_size`, `push_width`, and the `#[cfg(test)]
// mod tests` that held them to six transcribed rows. About four hundred lines
// answering one question: given a `KernelSig`, which of its operands is a
// storage binding, which is a field of the push block, and at what std430
// offset does each land.
//
// Nothing asks it. `driver-vulkan` was the only caller and it names
// `KernelSig` nowhere at all now: an arm states its own operands in ordinary
// Rust, `binding::params_from` decides between a push block and a staged
// buffer, and `encode::Encoder` writes the scalars at the offsets the MODULE
// declares -- read out of the SPIR-V rather than derived from a row, which is
// the same arithmetic with the second description taken out of the middle.
//
// The padding rule is the part worth not losing, so it is written down here
// and checked on a device rather than described twice: `attn/kv_write.slang`
// declares `int head_dim; PIE_STRIDE k_head_stride; PIE_STRIDE k_seq_stride;`
// with `PIE_STRIDE` an eight-byte `uint2`, so the block is 4 + 4 pad + 8 + 8 =
// 24 and NOT the 20 that adding the widths gives. A packer that concatenates
// writes both strides four bytes low and the shader reads two halves of two
// different numbers with nothing to report it. `driver-vulkan`'s
// `the_scalars_this_crate_packs_are_the_ones_the_shader_addresses_with`
// transcribes those twenty-four bytes and submits them to a real GPU, and
// `driver-vulkan/tests/rules.rs` holds all 39 blocks of this tree to what an
// independent SPIR-V walk measured.
//
// The 128-byte push ceiling that `no_row_overflows_the_push_constant_floor`
// guarded is now the device's own `maxPushConstantsSize`, which
// `Pipelines::get` takes and `driver-vulkan`'s
// `the_tier_this_device_selects_is_one_it_can_actually_load` states outright.
