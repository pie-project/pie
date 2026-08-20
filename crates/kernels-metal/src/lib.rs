//! METAL's kernel signature table — one row per KERNEL in `kernels/`, and one
//! row is many entrypoints.
//!
//! ## Why this is not shaped like CUDA's
//!
//! `kernels-cuda` has one row per launcher symbol and no axes at all. Its
//! device text is a C++ template like anything else, but a routine there NAMES
//! the single instantiation it wants and its row is DERIVED from that routine,
//! so a point of the template that nothing launches is not a row. An MSL
//! entrypoint is generated: `quant/qmm_t.metal` holds one template body and
//! a macro that stamps it over `(group × bits × row tile × column tile)`, so 54
//! of its entrypoints are one kernel evaluated at 54 points.
//!
//! Measured by `scripts/metal-kernel-audit.py`, which prints the file count
//! as it runs: **481 entrypoints over 100 kernels.** Those two numbers are
//! pinned a few files away, by `tests/entrypoints.rs::the_table_is_one_
//! hundred_kernels_over_four_hundred_and_eighty_one_entrypoints`, and this
//! sentence used to state a third -- a file count -- that nothing pinned
//! and that had drifted by one while both of the pinned numbers drifted
//! too. A number in prose beside a test that holds it is a number that
//! gets fixed; a number in prose beside nothing is a number that rots.
//! Enumerating the 481 would state the macro's job a
//! second time, by hand, and `.wiki/kernel-refactor.md` §5's own test — *would
//! the two share one C++ definition?* — answers that they are not distinct
//! kernels. So a row carries its [`Axis`]es and the product is the entrypoint
//! set. `.wiki/kernel-metal-refactor.md` §2 is the argument in full.
//!
//! The consequence worth stating on the way in: **the table is now where the
//! shader tree's coverage is written down.** `qmv_fast` is compiled for six
//! affine formats and `qmv_routed` for one; before this that difference existed
//! only as a name the driver would fail to find at model load.
//!
//! ## What keeps it honest
//!
//! Two checks, and a third that used to close the loop:
//!
//! * `kernels`' own unit tests pin the matcher — that a row covers every point
//!   of its axes and refuses a partial or permuted spelling.
//! * `tests/entrypoints.rs` pins the table's product against the routine name
//!   tables, and pins its size at 100 rows over 481 entrypoints.
//! * Nothing pins that product against the SHADERS.
//!   `scripts/metal-kernel-audit.py --table` did, by preprocessing them the
//!   way the Metal runtime does and diffing the result against
//!   `examples/entrypoints.rs`. That example is deleted and the mode is
//!   retired, so an entrypoint a `.metal` instantiates and no row declares
//!   reaches a device as a nil pipeline rather than a red check. It was a
//!   script rather than a test because expanding `instantiate_*` needs a C
//!   preprocessor; the Vulkan and WGSL siblings still hold the same invariant
//!   inside `cargo test`, where a variant is declared on a line rather than
//!   stamped by a macro.
//!
//! And from the other end, `model-ir`'s `kernels::check_plan` refuses any
//! launched symbol no row declares, so a lowered `*.metal.*` text cannot state
//! a kernel this table has not heard of.
//!
//! ## Reading this without a Mac
//!
//! All of the above runs on Linux. Metal compiles its shaders at RUN time, so
//! `default-features = false` gives the table and nothing else — which is what
//! `model-ir` wants and all it wants — and `native` adds only the staging
//! of `ptir_rng.generated.metal` out of `tensor-compiler`.

pub use kernels::KernelSig;

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
pub type Plane = crate::routine::Metal;

/// Every routine this crate declares.
///
/// A DISTRIBUTED SLICE, so nothing enumerates: `#[routine]` puts each row in a
/// `linkme_ROUTINES` link section beside its own `fn`, and the linker hands
/// back the bounds. There is no membership list to add a routine to, which is
/// the last hand-written thing about one -- and the last that could be
/// forgotten, leaving a routine compiled, correct and unreachable.
///
/// The order is LINK ORDER and no reader may depend on it. Nothing does:
/// lookups match on the full symbol, which `no_symbol_is_declared_twice`
/// keeps unique.
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
pub static METAL_ROUTINES: [::kernels::routine::Routine<Plane>];

/// The slice under the name `#[routine]` registers into.
pub use METAL_ROUTINES as ROUTINES;

/// EMPTY. Every family has retired its rows.
///
/// It is a `&[]` written here rather than ten empty slices folded together:
/// this was a `const fn` concatenation of the ten family tables, kept as a
/// `const` so the whole hundred stayed a `&'static` the compiler could read
/// at load with no allocation. Sixty lines of it -- a `total()` fold, an
/// `EMPTY` filler, a field-by-field `copy_sig` because `KernelSig` is not
/// `Copy` in a const context -- and the last row leaving made all of it a
/// machine for concatenating nothing.
///
/// The name stays because `kernels/tests/shader_backends_agree.rs` reads all
/// three backends' tables through it, and an empty table and an absent one
/// are the same shape to that gate but not the same fact: this backend
/// FINISHED. `refactor-bigplan.md` §7 Stage 5 deletes `KernelSig` itself,
/// once the last backend can also write this line.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints of the families whose `kernel!` rows have been RETIRED.
///
/// Stated at the SOURCE rather than at each consumer: one edit instead of N,
/// every sweep keeps its coverage untouched, and the census stays whole so
/// the comparison against a backend that has not retired the same rows keeps
/// meaning what it meant.
///
/// This list grows as families cross and the table shrinks under it. When the
/// last one crosses, `KERNELS` is empty and this is the whole census — which
/// is `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 4, and why it is a list
/// of families rather than one flat slice.
pub const RETIRED: &[&[(&str, &str)]] = &[
    attn::ENTRYPOINTS,
    layout::ENTRYPOINTS,
    mlp::ENTRYPOINTS,
    moe::ENTRYPOINTS,
    norm::ENTRYPOINTS,
    ptir::ENTRYPOINTS,
    quant::ENTRYPOINTS,
    rope::ENTRYPOINTS,
    sample::ENTRYPOINTS,
    ssm::ENTRYPOINTS,
];

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
/// in the other order, compiles, dispatches and returns `Ok` — it binds the
/// wrong buffer to the wrong slot and computes a plausible number.
///
/// It was exposed EMPTY for one commit before the first family crossed, so
/// that the wiring was proven separately from the thing it carries. Both
/// families here are ones no text names — `argmax_logits` and
/// `copy_logits_bf16` are dark on this backend — which is why they went first:
/// a crossing that cannot change what any model computes is the one to make
/// the mistakes on.
#[must_use]
pub fn declared() -> Vec<kernels::routine::Declared> {
    ROUTINES
        .iter()
        .map(kernels::routine::Routine::declared)
        .chain(ELSEWHERE.iter().copied())
        .collect()
}

/// The signatures behind [`DECLARED_ELSEWHERE`], written out by hand.
///
/// A name in [`DECLARED_ELSEWHERE`] resolves through [`kernel_of`] and then
/// finds NO routine, and `model-ir`'s `stated_in` falls back to a permissive
/// empty signature for one. That fallback is safe for the arity check it was
/// written for -- `arity_problem` returns immediately on an empty arg list --
/// and it is NOT safe for the other question the same lookup answers:
/// `Stated::in_place` reads the aliasing off the `sources` column, and an
/// empty column says a launch aliases nothing.
///
/// For `rms_rope` that answer is wrong and silently so. The kernel rotates
/// the tensor it norms IN PLACE -- `x` is an `InOut` and the shader binds one
/// buffer for both halves -- so the statement's result and its operand are
/// one allocation. Read as aliasing nothing, `model-compiler` gives the
/// result a region of its own, the fused dispatch writes over its INPUT
/// region, and every consumer downstream reads the untouched region the
/// planner handed out: `kv_append` appends the unnormed, unrotated keys and
/// attention reads a slot the q projection never reached. Fluent and wrong,
/// on a model that answers correctly the moment `fused_qk_rope` is off.
///
/// So the column is stated here rather than left to the fallback. It is the
/// signature `kernels-vulkan`'s `norm::rms_rope` derives, copied field for
/// field, and it leaves with the name when Metal grows the kernel.
pub const ELSEWHERE: &[kernels::routine::Declared] = &[kernels::routine::Declared {
    name: "rms_rope",
    namespace: "norm",
    args: &[kernels::Ty::Bf16sMut, kernels::Ty::Bf16s],
    sources: &[
        // `InOut<Tensor<bf16>>`: result 0 over operand 0, which is the whole
        // point of the entry.
        Some(kernels::Source::Alias(0, 0)),
        Some(kernels::Source::Or(
            &kernels::Source::Named("weight"),
            &kernels::Source::Slot(kernels::Kind::Weight, 0),
        )),
    ],
    whole: false,
    depth_prefix_plan: false,
    derived: &[
        kernels::Derived { name: "x", nullable: false },
        kernels::Derived { name: "w", nullable: false },
    ],
}];

/// Every entrypoint this backend can dispatch, sorted. The set
/// `scripts/metal-kernel-audit.py` compares against the shader tree.
///
/// Rows plus [`RETIRED`], because those are two different questions now. A
/// row's `axes` generate its entrypoints, so this used to be BOTH "what the
/// table says" and "what the backend can do" — deleting a row separates them
/// for the first time, and every sweep keyed on this function would silently
/// follow the table. `device_kernels.rs` compiles what this returns; a
/// crossed family dropping out of it would stop being built anywhere, its
/// shader still in the tree and still fired by a routine, with nothing
/// failing.
pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = RETIRED
        .iter()
        .flat_map(|family| family.iter().map(|(_, name)| (*name).to_owned()))
        .collect();
    out.sort();
    out
}

/// The KERNEL a dispatchable symbol belongs to, by name.
///
/// `model-ir` refuses a model at load whose launched symbol no backend
/// declares, and asks `whether this kernel consumes its whole operand` of the
/// same lookup. Both used to be `kernels::sig_in`, which found the row whose
/// `symbol` matched exactly or whose `axes` generated the point — and both
/// silently stopped answering when the last row left [`KERNELS`], because an
/// empty table declares nothing and every metal text would have been refused
/// at trace time.
///
/// So the question moves to the two lists that outlived the rows. A text may
/// spell either end: `dsl::metal` states BASE names (`affine_qmv_fast`, never
/// `affine_qmv_fast_bfloat16_gs_64_b_4` — the suffix is the driver's binding
/// of a checkpoint fact) while a lowering may carry an instantiated point, and
/// `sig_in` accepted both, so this does too.
///
/// LONGEST match wins, and the rule is not cosmetic: `rms_norm` is a prefix of
/// `rms_norm_gated`, and `affine_qmm_t` of `affine_qmm_t_splitk`. A first
/// match would answer for the wrong kernel — and would answer with the wrong
/// `whole`, which is a fact the compiler plans buffers against. The boundary
/// is an underscore for the same reason: a point is a SUFFIX appended to a
/// base, so a name that merely starts with another's letters is not one of it.
///
/// This is the same rule `driver-metal`'s `lowering::routine::crossed` uses,
/// applied to a different list: that one asks which ROUTINE dispatches a
/// symbol, over the stems its registry states. This asks which of the HUNDRED
/// a symbol is, over the names the shader tree has. They agree wherever both
/// are defined, and they have to: a symbol that resolved here and nowhere
/// there would be a kernel the compiler planned and the driver could not fire.
///
/// The symbol must be one this backend can actually dispatch -- one of the
/// hundred names, or one of the 481 points the census holds. The prefix rule
/// alone would accept `affine_qmv_fast_typo`, and accepting it is exactly the
/// load-time refusal `model-ir` exists to make: a row's `axes` used to
/// enumerate the legal points, and the census is that enumeration now.
#[must_use]
pub fn kernel_of(symbol: &str) -> Option<&'static str> {
    static CENSUS: std::sync::OnceLock<std::collections::BTreeSet<&'static str>> =
        std::sync::OnceLock::new();
    let census = CENSUS.get_or_init(|| {
        RETIRED
            .iter()
            .flat_map(|family| family.iter().map(|(_, name)| *name))
            .chain(DECLARED_ELSEWHERE.iter().copied())
            .collect()
    });

    KERNELS
        .iter()
        .map(|k| k.name)
        .chain(retired_rows().iter().copied())
        .filter(|name| {
            symbol == *name || (census.contains(symbol) && at_word_boundary(symbol, name))
        })
        .max_by_key(|name| name.len())
}

/// Every `(shader file, entrypoint)` a retired family reaches.
///
/// Metal compiles at RUN time from a path and a name, so the two are one fact
/// and a consumer that has the name still has to be told the file. `KERNELS`
/// carried both in one row; ten `ENTRYPOINTS` lists carry both now, and this
/// is the concatenation. `driver-metal/tests/device_kernels.rs` builds every
/// pair against a device, which is the sweep that would otherwise have gone
/// quiet the moment the last row retired.
#[must_use]
pub fn shaders() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&str, &str)> = RETIRED.iter().flat_map(|f| f.iter().copied()).collect();
    out.sort_unstable();
    out
}


/// Points this backend NAMES but does not build, and the whole of that set.
///
/// [`RETIRED`] is the census of what can still be dispatched, and every
/// name in it rides a `(file, entrypoint)` pair that `device_kernels.rs`
/// opens against a real device. These are neither: they resolve so that
/// `model-ir`'s `check_plan` can check a text, and there is no shader
/// behind them on this backend.
///
/// Why a text this backend cannot run reaches a check that asks it: Vulkan
/// consumes the metal-flavoured plan, so `stated_in(Metal, ..)` is asked
/// about every symbol a VULKAN launch names. It needs only [`kernel_of`] to
/// answer, and falls back to a permissive empty signature when no routine is
/// declared. Putting the point HERE rather than in an `ENTRYPOINTS` row is
/// what keeps the macOS sweep from trying to compile a file that does not
/// exist.
///
/// Each name is debt, and it is one line each: when Metal grows the kernel,
/// the name moves to its family's `ENTRYPOINTS` and leaves here.
pub const DECLARED_ELSEWHERE: &[&str] = &["rms_rope_bfloat16"];

/// The rows that have been retired, by the name their `kernel!` call had.
///
/// [`RETIRED`] answers "what can still be dispatched"; this answers "what
/// used to be a row here". The cross-backend parity tests scrape every
/// backend's sources and compare row for row, so while the crossing is in
/// flight they need to know which rows are absent on purpose. It empties when
/// the last family crosses and those tests retire with it.
#[must_use]
pub fn retired_rows() -> &'static [&'static str] {
    &[
        "add_bias",
        "argmax_logits",
        "cast_qmm_input_bfloat16_to_float16",
        "cast_qmm_input_strided_bfloat16_to_float16",
        "combine_sorted",
        "copy_logits_bf16",
        "embed_gather_4bit",
        "embed_gather_mb_4bit",
        "embed_gather_scaled_4bit",
        "embed_gather_scaled_mb_4bit",
        "encode_u4_bf16",
        "encode_u4_f32",
        "gate",
        "gated_rms",
        "gated_rms_strided",
        "gdn_core",
        "gdn_core_recurrent",
        "gdn_core_recurrent_prefill",
        "gdn_core_recurrent_slotted",
        "gdn_core_slotted",
        "gdn_prep",
        "gdn_prep_prefill",
        "gdn_prep_slotted",
        "geglu_tanh",
        "geglu_tanh_strided",
        "gptoss_swiglu",
        "kv_append",
        "kv_append_paged",
        "layer_scalar_mul",
        "logit_softcap",
        "mxfp4_dequant_bf16",
        "mxfp4_qmm_t_routed_bias",
        "mxfp4_qmv_routed_bias",
        "neox_decode",
        "neox_freqs_decode",
        "neox_freqs_mb",
        "neox_mb",
        "neox_prop_decode",
        "neox_prop_mb",
        "neox_strided",
        "ple_combine",
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
        "qmv_routed",
        "qmv_routed_bias",
        "qmv_tail",
        "qmv_tail_bias",
        "qmv_wide_strided",
        "residual_add",
        "residual_add_strided",
        "rms_residual",
        "rms_residual_scaled",
        // Declared here and NOWHERE ELSE on this backend: there is no
        // `rms_rope` entry in `norm.rs`'s `ENTRYPOINTS` and no `.metal` body
        // behind it. That is deliberate and it is the narrowest thing that
        // works.
        //
        // Vulkan consumes the metal-flavoured plan text, so `model-ir`'s
        // `check_plan` asks `stated_in(Metal, ..)` about every symbol a
        // VULKAN launch names, and `stated_in` needs only `kernel_of` to
        // answer -- it falls back to a permissive `Stated { args: &[] }` when
        // no routine is declared, and `arity_problem` returns immediately on
        // an empty arg list. `kernel_of` matches a base name against this
        // census directly, without consulting `ENTRYPOINTS` at all.
        //
        // Adding an `ENTRYPOINTS` row instead would oblige a `.metal` file,
        // because `device_kernels.rs` opens every `(file, entry)` pair on a
        // real device -- and that file could not be compiled, run or measured
        // from the machine this fusion was written on. Shipping an
        // unverifiable shader to satisfy a name lookup is a worse trade than
        // a name that resolves and dispatches nowhere, especially as the
        // statement is gated so no Metal text can name it.
        //
        // The debt is real and it is one line: when Metal grows the fused
        // kernel, this comment goes and an `ENTRYPOINTS` row arrives.
        "rms_rope",
        "rms_single_row",
        "rms_strided_head_row",
        "rms_strided_row",
        "route_gather",
        "route_sort",
        "router_topk",
        "router_topk_scaled",
        "row_gather",
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
        "shared_expert_combine",
        "shared_expert_combine_strided",
        "silu_mul",
        "silu_mul_strided",
        "split_qkv_bf16",
        "vnorm_single_row",
    ]
}

/// Every entrypoint whose row is gone and whose routine now answers for it.
#[must_use]
pub fn retired() -> Vec<&'static str> {
    RETIRED
        .iter()
        .flat_map(|f| f.iter().map(|(_, name)| *name))
        .collect()
}

/// Whether `name` appears in `symbol` bounded by underscores on both sides.
///
/// NOT `starts_with`, and the difference is a whole family of kernels. A row's
/// name is usually a prefix of its points — `rms_single_row_bfloat16` — but
/// not always: `quant`'s are spelled `affine_qmv_fast_bfloat16_gs_64_b_4`,
/// where the row is `qmv_fast` and `affine_` is a QUALIFIER the row name does
/// not carry. A prefix rule finds nothing there.
///
/// It never had to. `KERNELS` held every row until Stage 4, and the caller
/// above resolved these through the table's own axis expansion; the prefix
/// rule was only ever the fallback for a retired family, and the first
/// families to retire were the ones whose rows ARE prefixes. When the last
/// row went, every `quant` and `moe` symbol stopped resolving and `model-ir`
/// refused to build any metal text at all — the load-time check firing on the
/// whole fleet.
///
/// The boundary is required on both sides for the reason the prefix rule
/// required it on one: a point is appended with a separator, so `qmv_fast`
/// must not claim `qmv_fastest_bfloat16`, and the qualifier is prepended with
/// one, so it must not claim `xqmv_fast_bfloat16` either.
fn at_word_boundary(symbol: &str, name: &str) -> bool {
    let mut from = 0;
    while let Some(at) = symbol[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before = start == 0 || symbol.as_bytes()[start - 1] == b'_';
        let after = end == symbol.len() || symbol.as_bytes()[end] == b'_';
        if before && after {
            return true;
        }
        from = start + 1;
    }
    false
}

/// What a generic routine's element type must be on this plane.
///
/// `#[routine]` puts this on every type parameter that states no bound of its
/// own, so a generic signature reads the same on all four planes and the
/// plane's own requirement is said HERE, once. A shader plane binds a handle
/// and asks nothing more of an element than that it be one.
pub trait RoutineElem: kernels::Elem {}

impl<T: kernels::Elem> RoutineElem for T {}
