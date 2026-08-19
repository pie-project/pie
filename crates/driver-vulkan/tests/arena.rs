//! Can a Vulkan driver bind the arena a real lowering assigns?
//!
//! Every other test in this crate asks about a module or a dispatch. This one
//! asks about the OFFSETS, and it is the precondition the whole arena plan
//! rests on: `model_compiler::lower` places each activation at a byte offset it
//! chooses, `driver-metal` binds those offsets with `setBuffer:offset:` and
//! Apple silicon asks four bytes of alignment for it -- so nothing upstream has
//! ever had a reason to place them more carefully than that.
//!
//! Vulkan asks more. A storage descriptor's offset must be a multiple of
//! `minStorageBufferOffsetAlignment`, and an offset that is not simply cannot
//! be bound: there is no slow path and no fallback, the descriptor is invalid.
//! If the compiler placed activations at, say, four-byte boundaries, this
//! backend could not use the arena at all and would need a copy per operand --
//! which is a different driver, not a slower one.
//!
//! So it is worth knowing rather than assuming, and it is worth knowing
//! against the SPECIFICATION rather than against the card in this machine.
//!
//! # The answer is exactly enough, with nothing to spare
//!
//! `lower`'s arena allocator rounds every placement to 256 bytes, on both its
//! bump path and its free-list path. 256 is also the largest
//! `minStorageBufferOffsetAlignment` a conformant Vulkan device may report. So
//! every offset every text produces is bindable on every device -- and the
//! margin is ZERO.
//!
//! It reads like a designed agreement and it is not one. That allocator's
//! comment says why it picked the number: *"a decode body runs inside a
//! capture, so the same plan must land the same value at the same address on
//! every fire"*. It is a Metal capture-replay requirement that happens to
//! coincide with a Vulkan descriptor limit, in a crate that has never heard of
//! either.
//!
//! Which is the entire reason this file exists. A coincidence nobody wrote
//! down is a coincidence somebody may reasonably undo -- and the first
//! measurement here, taken over one text, reported a comfortable 2048 and
//! would have let a change to 128 look harmless. It was `gpt_oss_20b` that
//! gave the real answer: a 2880-wide row of 2 bytes is 5760, which is not a
//! multiple of 256, so the next operand lands at 5888 and the alignment is
//! whatever the allocator insists on and nothing more.
//!
//! GPU-free on purpose. The question is about numbers a compiler produced, and
//! a check that needed a device would not run in the builds that change them.

use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Arg, Fire, Lowered, Row, lower};
use model_ir::trace::FireClass;

/// The strictest `minStorageBufferOffsetAlignment` a conformant Vulkan device
/// may report.
///
/// The specification's required-limits table caps it here, so an offset that
/// is a multiple of 256 is bindable on EVERY Vulkan device, and one that is
/// not is bindable on some and not others. Checking against the local card's
/// 16 would pass a plan that fails on hardware nobody in this repository owns,
/// which is the failure this constant exists to make impossible.
const STRICTEST_ALIGNMENT: usize = 256;

/// Lower one text, or `None` where it does not lower.
fn lowered(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
    rows: usize,
) -> Option<Lowered> {
    let plan = llama_like_metal(facts, metal, class);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .ok()
}

/// Every text this crate can reach, both fire classes.
///
/// Decode at one row and prefill at 64: the row count changes the arena's
/// size and the offsets within it, and a sweep at one row would miss any
/// placement whose alignment depends on how much came before it.
fn texts() -> Vec<(String, Lowered)> {
    geometric().into_iter().map(|(n, l, _)| (n, l)).collect()
}

/// `LlamaLikeMetalFacts::synthetic()` with the one line this backend answers
/// differently.
///
/// `synthetic()` is `driver-metal`'s answer sheet -- its own doc says so --
/// and on `add_bias` the two backends genuinely disagree: that driver's binder
/// does not resolve `Source::OutWidth`, which is where `norm::add_bias` reads
/// its row pitch, and this one's does. Stating it here rather than in each
/// text means qwen2.5's plans carry the three bias launches a layer that the
/// checkpoint has always shipped, so everything in this file that walks a real
/// plan walks them too.
fn vulkan_facts() -> LlamaLikeMetalFacts {
    LlamaLikeMetalFacts {
        add_bias: true,
        ..LlamaLikeMetalFacts::synthetic()
    }
}

/// The same texts, each with the model geometry its plans were built from.
///
/// Split out because most of this file never needs it and one test does. A
/// head-width rule cannot be answered from a plan alone: `sdpa_paged_decode`
/// is compiled for a fixed head dimension and the plan states rows and
/// widths, not heads. A driver knows the model; a plan does not.
fn geometric() -> Vec<(String, Lowered, driver_vulkan::dispatch::Geometry)> {
    let mut out = Vec::new();
    for (name, facts, metal) in [
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b(), vulkan_facts()),
        (
            "gpt_oss_20b",
            LlamaLikeFacts::gpt_oss_20b(),
            LlamaLikeMetalFacts::gpt_oss_20b(),
        ),
        (
            "qwen3_30b_a3b",
            LlamaLikeFacts::qwen3_30b_a3b(),
            vulkan_facts(),
        ),
        (
            "qwen2_5_1_5b",
            LlamaLikeFacts::qwen2_5_1_5b(),
            vulkan_facts(),
        ),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            vulkan_facts(),
        ),
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b(), vulkan_facts()),
    ] {
        for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 64)] {
            if let Some(low) = lowered(&facts, &metal, class, rows) {
                out.push((
                    format!("{name}/{class:?}/{rows}"),
                    low,
                    driver_vulkan::dispatch::Geometry {
                        q_heads: facts.q_heads,
                        kv_heads: facts.kv_heads,
                        head_dim: facts.head_dim,
                        rotary_dims: facts.head_dim,
                        n_experts: facts.n_experts,
                        experts_per_token: facts.experts_per_token,
            ..Default::default()
                    },
                ));
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no text lowered, so this file proved nothing"
    );
    out
}

/// Every activation the compiler places can be bound by a descriptor.
///
/// The claim that makes `Bound::at` usable for real work rather than only for
/// the hand-built arenas in `tests/device.rs`.
#[test]
fn every_arena_offset_a_real_lowering_assigns_is_bindable() {
    let mut operands = 0usize;
    let mut refused = Vec::new();
    let mut worst = usize::MAX;

    for (name, low) in texts() {
        for arg in &low.args {
            let Arg::Arena { at, width, bytes } = arg else {
                continue;
            };
            operands += 1;
            if at % STRICTEST_ALIGNMENT != 0 {
                refused.push(format!(
                    "{name}: an operand at byte {at} is not a multiple of \
                     {STRICTEST_ALIGNMENT}"
                ));
            }
            // Zero is aligned to everything, so it must not set the floor --
            // and it is also the offset every arena has, so including it would
            // make the reported margin meaningless.
            if *at != 0 {
                worst = worst.min(1usize << at.trailing_zeros());
            }
            // An operand that runs past the arena cannot be bound either, and
            // this is the one number `Bound::at` cannot check for a caller:
            // it sees a buffer, not a plan.
            let extent = (*width as usize).saturating_mul(*bytes as usize);
            if at.saturating_add(extent) > low.arena_bytes {
                refused.push(format!(
                    "{name}: an operand of {extent} bytes at {at} runs past the \
                     {} the arena holds",
                    low.arena_bytes
                ));
            }
        }
    }

    assert!(
        refused.is_empty(),
        "{} of {operands} arena operands cannot be bound:\n  {}",
        refused.len(),
        refused.join("\n  ")
    );
    assert!(
        operands > 0,
        "no text stated an arena operand, so nothing was checked"
    );
    // Not a requirement, a MARGIN. If a future placement change brings this
    // near 256 the plan still works and this assertion still passes, and the
    // number in the module docs is the thing that will have quietly stopped
    // being true -- so it is asserted where it is stated.
    // The same fact the loop already checked, asked the other way round, and
    // it earns its place by reporting the MARGIN rather than a pass. There is
    // none: `worst` is 256 today, so this reads as an equality and any
    // loosening of the allocator fails the check above rather than eroding a
    // cushion first.
    assert_eq!(
        worst, STRICTEST_ALIGNMENT,
        "the tightest alignment any operand has is {worst}; the allocator \
         rounds to 256 and the specification caps a device at 256, so this is \
         an agreement with no room in it and a change to either side is a \
         change this test must be made to state"
    );
}

/// How a symbol's launch reaches its module: what the plan states, what the
/// module declares, and whether the two account for each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reaches {
    /// The plan's scalars are the module's push block, and the plan's operands
    /// are all of its descriptors. Nothing else is needed to fire it.
    Push,
    /// The plan's scalars are a BUFFER of their own at the module's last
    /// binding, so the driver allocates and fills it, and the plan's operands
    /// are the rest. Also complete -- but not the same call.
    Buffer,
    /// The module binds resources the plan does not state, because they are
    /// the driver's own: the paged KV cache, its page table, the routing
    /// scratch. The number is how many.
    DriverSupplies(u32),
    /// Every descriptor is the plan's, but some of the module's push scalars
    /// are not: the ROW derives them from the operands' shapes. The number is
    /// how many words.
    ///
    /// `norm::add_bias` is the case that forced the category, and it is worth
    /// separating from [`Self::DriverSupplies`] because what is missing is a
    /// different kind of thing. A paged attention is short of a RESOURCE only
    /// this driver has; a bias is short of a NUMBER the plan already implies
    /// -- the row width of its own output. `Source::OutWidth` is how the row
    /// says where to read it, and `binding::scalars` is what reads it, so
    /// nothing outside the plan is needed to fire this kernel.
    ///
    /// Which is exactly why the two must not share a bucket: this file's
    /// count of `DriverSupplies` is "what a Vulkan executor still owes", and
    /// a row that derives its own scalar owes nothing.
    RowDerives(u32),
    /// The plan states at least as many OPERANDS as the module has real
    /// bindings, and the two still did not match as a complete form. The
    /// number is how many descriptors are surplus.
    ///
    /// Separated from [`Self::DriverSupplies`] for the reason that variant's
    /// own neighbour gives: `real.saturating_sub(args)` reports this as a
    /// driver owing zero resources, "which is true and says nothing". A zero
    /// in the bucket that means "what a Vulkan executor still owes" reads as
    /// *nothing is missing*, when what happened is that something is spare.
    ///
    /// The routed GEMMs are the case that forced it. A surplus is not a
    /// defect -- a shader that declines a slot the statement carries fires
    /// correctly, and `tests/device.rs` proves these two agree with the
    /// matvec bit for bit -- but it is a fact about the calling convention,
    /// and a kernel that started IGNORING an operand it used to read would
    /// look exactly like this and must not land silently.
    PlanOverstates(u32),
}

/// Every symbol the reachable texts launch, and how it must be called.
///
/// Transcribed, so that a text that starts launching something new, or a
/// shader that changes its binding count, is a failure here rather than a
/// surprise in an executor that does not exist yet.
/// The `_bm_32_bn_32` in the dense GEMM symbols is `project::QMM_TILE`,
/// transcribed rather than interpolated: the tile moved from 16 to 32 for a
/// 4.5x prefill win, and a table that formatted itself from the constant
/// would have followed it without anyone reading the three lines that had to
/// change. It earned that keep a second time when upstream swept the ROUTED
/// tile's column axis and settled on `bn_64` -- the two routed rows below
/// say 64 where the dense ones say 32, which is a difference no interpolated
/// table could have shown.
const REACHES: &[(&str, Reaches)] = &[
    // THE PRECAST TWINS, and not the plain ones. `qmm_fp16_precast` is
    // stamped at `gs = 64, b = 4` alone, so every text quantised there now
    // lowers `affine_qmm_t_fp16_precast` behind a staging cast and the plain
    // `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32` is launched by nothing.
    // The 8-bit row below is untouched for exactly that reason, which is the
    // check that this is a codec fact and not a lowering-wide one. Both
    // twins reach their modules as `Push`: staging changes which buffer the
    // activation arrives in, not how the call is made.
    (
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
        Reaches::Push,
    ),
    // The 8-bit twins, which arrived with a text upstream added rather than
    // with anything here. A point is `(group x bits)` and the bit width is
    // the checkpoint's, so a catalog row quantised at 8 bits names a symbol
    // no 4-bit row does -- and it reaches its module exactly the way its
    // 4-bit sibling does, because the bit width lives inside the shader and
    // not in the calling convention.
    ("affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32", Reaches::Push),
    ("affine_qmv_fast_bfloat16_gs_64_b_8", Reaches::Push),
    (
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
        Reaches::Push,
    ),
    ("affine_qmv_fast_bfloat16_gs_64_b_4", Reaches::Push),
    ("affine_qmv_fast_residual_bfloat16_gs_64_b_4", Reaches::Push),
    // Two slots, both stated -- the value it biases in place and the bias --
    // and ONE scalar the statement does not carry. An `AddBias` states no
    // params at all, because a bias vector's length is the projection's width
    // and the trace already said that when it sized the output. Under the
    // table this row read it back with `Source::OutWidth(0)`; now the ARM
    // `hold::add_bias` supplies it directly from `Facts::width` -- the result's
    // own width, which the trace holds. The reach is unchanged, because what
    // the module is short of is still one push word the plan does not carry;
    // only the thing that fills it moved from a row source to an arm.
    ("add_bias_bfloat16", Reaches::RowDerives(1)),
    // The staging pass `qmm_fp16_precast` puts in front of a tiled GEMM: it
    // reads bfloat rows and writes them back as `half` for
    // `affine_qmm_t_fp16_precast` to multiply. Two descriptors, both the
    // plan's, and a four-word push block of which the statement carries only
    // three -- `k`, `n` and the row pitch. The fourth is `count`, which the
    // PACKED form of this cast reads and this one does not; `hold::
    // cast_qmm_input_strided_bfloat16_to_float16` fills it with `rows x
    // pitch` so the word holds the number it is named for. That is the one
    // word the statement does not state, which is what puts this row here
    // and not in `Push`; the ROW COUNT the arm also supplies is not counted,
    // because it steers the grid and never reaches the push block.
    (
        "cast_qmm_input_strided_bfloat16_to_float16",
        Reaches::RowDerives(1),
    ),
    ("residual_add_bfloat16", Reaches::Push),
    ("silu_mul_bfloat16", Reaches::Push),
    ("combine_sorted", Reaches::Buffer),
    ("gptoss_swiglu_bfloat16", Reaches::Buffer),
    ("rms_single_row_bfloat16", Reaches::Buffer),
    // The pair of the line above, and it arrived by a text learning to say
    // it rather than by this backend gaining anything. A post-norm landing
    // used to be `rms_single_row` followed by `residual_add`, and every one
    // of those norms was read by exactly one add: `norm::rms_residual` is
    // that pair in one dispatch, and this crate had built it, armed it and
    // never been asked for it. It reaches the same way its unfused half does.
    ("rms_residual_bfloat16", Reaches::Buffer),
    ("route_gather", Reaches::Buffer),
    ("route_sort", Reaches::Buffer),
    ("router_topk_bfloat16", Reaches::Buffer),
    // Seven slots, one hole, six real bindings -- and six operands stated.
    // It was DriverSupplies(1) until the hole was measured, which is the whole
    // argument for measuring holes.
    ("affine_qmv_routed_bfloat16_gs_64_b_4", Reaches::Push),
    // The same routed shape one plane heavier. `dsl::metal::routed_qmv` sends
    // an MXFP4 expert bank here rather than to the unbiased symbol, because
    // that bank publishes one additive term per output row beside its packed
    // weight; the module reads it at a slot of its own, and the statement
    // names it, so the reach is still complete.
    ("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", Reaches::Push),
    // The TILED forms of the two routed matvecs above, which arrived when
    // upstream wired `routed_qmm`. A mixture of experts used to take the
    // matvec whatever its rectangle, so a prefill of many tokens fired one
    // row at a time; the shared forward now picks the GEMM whenever
    // `moe_tile` is `Some`. Nothing in this crate was taught either symbol --
    // both reached a dispatch on the strength of what the statement carries,
    // and `tests/device.rs` holds the routed GEMM to answering bit for bit
    // the way the routed matvec does.
    //
    // `PlanOverstates(1)` and not `Push`: the statement carries one operand
    // more than the module binds. Their matvec siblings are complete, so the
    // surplus is the tiled form's own -- it declines a slot the routed shape
    // states. Recorded as a NUMBER rather than waved through, because a
    // kernel that quietly stopped reading an operand would present exactly
    // this way and the count is what would move again.
    (
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64",
        Reaches::PlanOverstates(1),
    ),
    (
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64",
        Reaches::PlanOverstates(1),
    ),
    (
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        Reaches::DriverSupplies(1),
    ),
    ("neox_mb_bfloat16", Reaches::DriverSupplies(1)),
    // The same rotation with its ladder handed over rather than raised: a
    // rescaled deployment has no base for the shader to exponentiate, so the
    // frequencies arrive as a buffer. Two the plan does not name -- the
    // positions `neox_mb` is also short of, and that frequency table -- and
    // `rope.rs` is where both come from, because a rescaling is a fact about
    // the deployment rather than about the architecture the text states.
    ("neox_freqs_mb_bfloat16", Reaches::DriverSupplies(2)),
    ("row_gather_bfloat16", Reaches::DriverSupplies(2)),
    // Twelve slots and six holes -- two of them Metal's ring-ABI placeholders
    // at 10 and 11, kept on purpose. Four real bindings the plan does not
    // name: the paged KV cache and its page table.
    ("kv_append_paged_bfloat16", Reaches::DriverSupplies(4)),
    (
        "sdpa_paged_decode_bfloat16_d_128",
        Reaches::DriverSupplies(8),
    ),
    (
        "sdpa_paged_decode_sink_bfloat16_d_64",
        Reaches::DriverSupplies(8),
    ),
    // The tiled prefill pair. Same operand list as decode plus the true row
    // count, which is a SCALAR, so the descriptors the driver supplies are
    // the same eight.
    //
    // The SINK half is the cooperative-matrix tier rather than the scalar
    // tile: `sdpa_paged_mma` is compiled at `_d_64` alone and the only text
    // here with sinks has 64-wide heads, so it is what the selection reaches.
    (
        "sdpa_paged_tiled_bfloat16_d_128",
        Reaches::DriverSupplies(8),
    ),
    (
        "sdpa_paged_mma_sink_bfloat16_d_64",
        Reaches::DriverSupplies(8),
    ),
];

/// Every symbol a real text launches has a module this backend can compile.
///
/// The claim `driver-metal`'s `model_bind` makes for its own table, asked of
/// this one: on both backends an entry point is compiled from a name, so a
/// text that states a symbol the table knows needs no arm written to receive
/// it. Twenty-two distinct symbols, and `kernels-vulkan` has a module for all
/// twenty-two.
///
/// It is a smaller number than the table's 480 because a lowering is not yet
/// the whole of a fire -- `Lowered::residue` holds the statements that still
/// run without a rectangle. What it measures is the part that HAS crossed,
/// and that part is fully served.
#[test]
fn every_symbol_a_real_text_launches_has_a_module() {
    let entrypoints = kernels_vulkan::entrypoints();
    let have: std::collections::BTreeSet<&str> = entrypoints.iter().map(String::as_str).collect();
    let mut launched = std::collections::BTreeSet::new();
    let mut missing = Vec::new();

    for (name, low) in texts() {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            if launched.insert(symbol.clone()) && !have.contains(symbol.as_str()) {
                missing.push(format!("{name} launches `{symbol}`, which has no module"));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "{} of {} symbols have no module:\n  {}",
        missing.len(),
        launched.len(),
        missing.join("\n  ")
    );
    assert_eq!(
        launched.len(),
        REACHES.len(),
        "the texts launch {} distinct symbols and this file describes {}",
        launched.len(),
        REACHES.len()
    );
}

/// The plan's two runs and the module's two runs account for each other.
///
/// `driver-metal` binds one run: operands and scalars go in one argument
/// table, in the order the row states. Vulkan needs two, and the LOWERING
/// already has them -- `Launch::args` and `Launch::params` are separate
/// ranges. The question this answers is whether that separation is the same
/// one, and the answer is that it is, twice over and differently:
///
/// * seven symbols take the plan's scalars as a PUSH block, and then the
///   plan's operands are exactly the module's real descriptors;
/// * six take them as a BUFFER of their own -- `rms_single_row`'s five
///   scalars are the 20-byte block at binding 3 that `tests/rules.rs` already
///   measures -- and then the plan's operands are one short of the
///   descriptors, by exactly that buffer.
///
/// The buffer is found by its SIZE and not by its position. Looking for it at
/// the binding one past the operand count seemed natural and is wrong for two
/// of the six: `combine_sorted` binds its 12-byte block at 3 of 5 and
/// `route_sort` its 28-byte block at 4 of 6, with an operand after it. Where a
/// parameter block sits is the kernel's own ABI, and the operand count says
/// nothing about it.
///
/// Which of the two a kernel uses is not a naming convention and not a list:
/// it is `Declared::push_offsets` against `Declared::block_bytes`, both read
/// off the compiled module. A driver asks the shader rather than remembering.
///
/// # The seven that need something else
///
/// The rest bind more than the plan states, and the difference is not a
/// defect. `kv_append_paged` binds ten descriptors nothing in the plan names,
/// because a paged KV cache and its page table are the DRIVER's resources --
/// the same is true on Metal, where the ring is driver-owned. Recording the
/// count is the point: it is the exact size of what a Vulkan executor still
/// has to supply, per symbol, measured rather than guessed at.
#[test]
fn what_the_plan_states_and_what_the_module_binds_account_for_each_other() {
    if !kernels_vulkan::embedded() {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    }
    let mut seen: std::collections::BTreeMap<String, Reaches> = Default::default();
    let mut wrong = Vec::new();

    for (text, low) in texts() {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let args = launch.args.end - launch.args.start;
            let params = launch.params.end - launch.params.start;
            // An IN-PLACE row binds one buffer for two of the plan's args:
            // the trace states the value and the result separately, because a
            // tape whose statements did not produce values could not say what
            // the next one reads, and the aliasing then says they are the same
            // allocation. `norm::add_bias` is the only one here, and without
            // this it classifies as a kernel binding one FEWER descriptor
            // than the plan states -- which is true and is not what is
            // interesting about it.
            //
            // The aliasing was a `KernelSig::in_place`, read off `KERNELS`;
            // that table is empty now, so it is read off the ROUTINE instead,
            // where the same fact lives as `Routine::in_place` -- a property
            // of the statement, not of a retired row, and stated on the
            // routine for exactly that reason. `routine_for` is the join.
            let aliases =
                driver_vulkan::hold::routine_for(symbol).map_or(0, |r| r.in_place().len());
            let args = args - u32::try_from(aliases).expect("a routine states few aliases");
            let Some(code) = kernels_vulkan::code(symbol, kernels_vulkan::Capability::Baseline)
            else {
                continue;
            };
            let words = driver_vulkan::spirv::words(code).expect("a built module is whole words");
            let d = driver_vulkan::spirv::declared(&words).expect("a built module is well formed");

            // Asked in this order because the two are not exclusive in
            // principle and PUSH is the stronger statement: it accounts for
            // every descriptor as well as every scalar.
            // Holes subtracted, because a slot no shader reads is not a
            // resource anybody has to supply. Counting raw slots made
            // `affine_qmv_routed` look like a kernel short of a buffer; it
            // has seven slots, one hole and six real bindings, and the plan
            // states six operands. `tests/device.rs` dispatches it with six.
            let real = d.bindings - d.holes() as u32;
            let reaches = if d.push_offsets.len() as u32 == params && args == real {
                Reaches::Push
            } else if args + 1 == real && d.block_bytes.iter().any(|b| *b == Some(params * 4)) {
                Reaches::Buffer
            } else if args == real && d.push_offsets.len() as u32 > params {
                // Every descriptor accounted for and some scalars still
                // missing: the row derives them. Asked AFTER the two complete
                // forms and BEFORE `DriverSupplies`, because `saturating_sub`
                // would otherwise report this as a driver owing zero
                // resources, which is true and says nothing.
                Reaches::RowDerives(d.push_offsets.len() as u32 - params)
            } else if args >= real {
                // Asked before `DriverSupplies` because `saturating_sub`
                // reports this case as a driver owing zero resources.
                Reaches::PlanOverstates(args - real)
            } else {
                Reaches::DriverSupplies(real.saturating_sub(args))
            };

            // Every launch of one symbol must reach it the same way. If two
            // texts disagreed, the classification would be a property of the
            // call and not of the kernel, and no driver could act on it.
            if let Some(before) = seen.insert(symbol.clone(), reaches)
                && before != reaches
            {
                wrong.push(format!(
                    "{text}: `{symbol}` reaches its module as {reaches:?} here and \
                     {before:?} elsewhere"
                ));
            }
        }
    }

    for (symbol, reaches) in &seen {
        match REACHES.iter().find(|(n, _)| n == symbol) {
            Some((_, want)) if want == reaches => {}
            Some((_, want)) => wrong.push(format!(
                "`{symbol}` reaches its module as {reaches:?} and this file says {want:?}"
            )),
            None => wrong.push(format!(
                "`{symbol}` is launched, reaches its module as {reaches:?}, and this \
                 file does not describe it"
            )),
        }
    }
    for (symbol, want) in REACHES {
        if !seen.contains_key(*symbol) {
            wrong.push(format!(
                "`{symbol}` is described as {want:?} and no text launched it"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "{} of {} symbols disagree:\n  {}",
        wrong.len(),
        REACHES.len(),
        wrong.join("\n  ")
    );
}

/// The binder this crate ships resolves every operand of every real launch.
///
/// The unit tests in `src/binding.rs` ask whether each rule is right against
/// operands a test invented. This asks the only question that cannot be asked
/// that way: put a real plan through the real binder, at the strictest
/// alignment any device may report, and see whether anything is refused.
///
/// # What has to be supplied, and why that is the finding
///
/// A weight and a seam value are not the plan's to place, so this stands in
/// for the driver's tables with placeholders sized generously. Every arena
/// operand, though -- 15140 of them across six texts in both fire classes --
/// goes through the real arithmetic: `rows × width × bytes` from the plan,
/// checked against the plan's arena and then against 256-byte addressing.
///
/// The count that matters is that ZERO are refused, and it is only meaningful
/// because the same walk refuses plenty when the arithmetic is wrong: binding
/// one row instead of the launch's rectangle passes here and is a defect a
/// GPU would find, which is why `probe`-shaped checks were replaced with this
/// one and why the extent is asserted below rather than assumed.
#[test]
fn the_binder_this_crate_ships_resolves_every_operand_of_every_real_launch() {
    /// Big enough that a placeholder never becomes the reason a bind fails.
    const GENEROUS: u64 = 1 << 30;

    struct Everything(driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Everything {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
    }

    let mut operands = 0u64;
    let mut arena_operands = 0u64;
    let mut widest = 0u64;
    let mut total = 0u64;
    let mut refused = Vec::new();

    for (text, low) in texts() {
        let buf = driver_vulkan::device::Buffer::placeholder(low.arena_bytes as u64);
        let store = Everything(driver_vulkan::device::Buffer::placeholder(GENEROUS));
        let arena = driver_vulkan::binding::Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            match driver_vulkan::binding::bind(
                &low,
                launch,
                arena,
                &store,
                STRICTEST_ALIGNMENT as u64,
            ) {
                Ok(bound) => {
                    operands += bound.len() as u64;
                    for (b, arg) in bound
                        .iter()
                        .zip(&low.args[launch.args.start as usize..launch.args.end as usize])
                    {
                        if matches!(arg, Arg::Arena { .. }) {
                            arena_operands += 1;
                            widest = widest.max(b.len());
                            total += b.len();
                            // The whole reason this backend needs an extent:
                            // a range that reached the end of the arena would
                            // cover every activation placed after it.
                            assert!(
                                b.offset() + b.len() <= low.arena_bytes as u64,
                                "{text}: a range from {} for {} leaves the arena",
                                b.offset(),
                                b.len()
                            );
                        }
                    }
                }
                Err((i, why)) => refused.push(format!(
                    "{text}: `{}` operand {i}: {why:?}",
                    low.kernels[launch.kernel as usize]
                )),
            }
        }
    }

    assert!(
        refused.is_empty(),
        "{} operands of {operands} could not be bound:\n  {}",
        refused.len(),
        refused.join("\n  ")
    );
    // Stated so that a plan which stops producing arena operands -- or starts
    // producing far fewer -- cannot make the zero above true by emptiness.
    // 15228 before the strided cast reached the card. Each staged activation
    // is two more arena operands -- the bfloat rows it reads and the half
    // rows it writes -- once per activation SOURCE per dense projection of
    // every precasting text.
    //
    // 16444 until the slot wrappers and `Ask` stated their own provenance.
    // That work did not change what a text launches; it changed what the
    // signature SAYS about each operand, and an operand a row now spells as a
    // slot is one this walk resolves rather than skips. The count moved UP,
    // which is the direction that cannot make the zero above true by
    // emptiness -- the check this number exists to defend.
    assert_eq!(
        arena_operands, 17148,
        "the texts produced a different number of arena operands than when this \
         was measured, so the zero above is about a different plan"
    );
    // The number with the teeth in it. "Nothing was refused" is satisfied by
    // any range small enough, so it cannot tell a correct extent from a
    // conservative one -- binding a single row rather than the launch's
    // rectangle is refused nowhere and is wrong everywhere, and binding to the
    // end of the arena is refused nowhere and is the exact defect
    // `tests/device.rs` shows corrupting a neighbour. Both change this sum.
    //
    // 11_161_544_416, where it was 11_070_889_696 before the slot wrappers and
    // `Ask` stated their own provenance. The plans launch the same rectangles;
    // what moved is that an operand a row now spells as a slot is one this
    // walk resolves rather than skips, so 704 more operands contribute their
    // extents. Both this and the operand count above rose together and in
    // proportion, which is what a widened census looks like -- an extent that
    // got SHORTER is the dangerous direction, and nothing here got shorter.
    //
    // 11_070_889_696, where it was 10_501_906_144 before the strided cast
    // reached the card: each staged activation binds its bfloat source and
    // its half destination, and a `half` rectangle is the same element count
    // at the same two bytes, so the pair adds twice the activation's extent
    // per dense projection of every precasting text.
    //
    // 10_501_906_144, where it was 10_518_945_504 before a post-norm landing
    // became one dispatch: 64 rectangles left these plans, and with them the
    // arena ranges their `residual_add` halves used to bind. A fold shows up
    // here as a fall, which is the shape of good news in this file.
    //
    // 10_518_945_504, where it was 2_982_826_080 before upstream taught
    // `rectangle_rows` about `Dim::MoeAlignedRoutes`. A routed statement used
    // to state one row per TOKEN, so its rectangle addressed a small fraction
    // of the stack it actually reads; it now states the sorted stack, which
    // is `top_k` routes a token with every touched expert's run rounded up to
    // a whole tile. The extent grew three and a half times because the
    // rectangle stopped understating itself, which is the direction this
    // assertion exists to notice -- an extent that is too SHORT is the
    // dangerous one, and it was short here.
    assert_eq!(
        total, 11_161_544_416,
        "the arena ranges this binder produces cover a different number of \
         bytes than `rows x width x bytes` over these plans did when it was \
         measured"
    );
    assert_eq!(widest, 25_739_264, "the largest single range changed");
}

/// A plan whose arena is one byte short is refused everywhere it should be.
///
/// The check above finds nothing, which is the good news and also the reason
/// it cannot vouch for the arena bound: a rule that never fires is
/// indistinguishable from a rule that is not there. So this takes the same
/// real plans and shrinks the arena the binder is told about, which makes the
/// operands at the far end address past it while every other number stays
/// exactly as the compiler produced it.
///
/// The count is the interesting part. The tightest real operand ends EXACTLY
/// at the end of the arena -- `every_arena_offset_a_real_lowering_assigns_is_bindable`
/// measures the slack as zero -- so removing one byte has to refuse at least
/// one launch and, because that operand recurs, refuses a good many.
#[test]
fn an_arena_one_byte_short_of_what_the_plan_placed_refuses_what_runs_off_it() {
    struct Everything(driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Everything {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
    }

    let mut refused = 0u64;
    let mut launches = 0u64;
    for (_, low) in texts() {
        let short = low.arena_bytes as u64 - 1;
        // The BUFFER stays full size. Only the plan's number shrinks, so the
        // refusal has to come from the arena bound and not from the range
        // check -- which is the distinction that makes `PastArena` a variant
        // of its own rather than another `Overrun`.
        let buf = driver_vulkan::device::Buffer::placeholder(low.arena_bytes as u64);
        let store = Everything(driver_vulkan::device::Buffer::placeholder(1 << 30));
        let arena = driver_vulkan::binding::Arena {
            buffer: &buf,
            bytes: short,
        };
        for launch in &low.launches {
            launches += 1;
            if let Err((_, why)) = driver_vulkan::binding::bind(
                &low,
                launch,
                arena,
                &store,
                STRICTEST_ALIGNMENT as u64,
            ) {
                assert!(
                    matches!(why, driver_vulkan::binding::Unbindable::PastArena { .. }),
                    "a byte off the arena is not a reason for {why:?}"
                );
                refused += 1;
            }
        }
    }
    assert!(
        refused > 0,
        "{launches} launches bound against an arena one byte shorter than the \
         one they were placed in, so the bound is not being checked"
    );
}

/// Every launch's scalars land somewhere the module reads them from.
///
/// `what_the_plan_states_and_what_the_module_binds_account_for_each_other`
/// measures the split; this puts the real plans through the code that ACTS on
/// it, which is a different claim. A classification can be right about a
/// symbol and still be wrong about a launch, because the plan states its
/// scalars per launch and nothing says two launches of one kernel state the
/// same number of them.
///
/// The seven symbols the split could not account for are expected to be
/// refused here, and are counted rather than tolerated: they bind resources
/// the plan does not name, so their scalars are not the plan's alone either,
/// and a Vulkan executor still owes them. Naming that here means a change
/// which quietly makes one of them appear to fit is a failure.
#[test]
fn every_launchs_scalars_land_where_its_module_reads_them() {
    if !kernels_vulkan::embedded() {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    }
    let mut pushed = 0u64;
    let mut blocked = 0u64;
    let mut owed: std::collections::BTreeSet<String> = Default::default();

    for (text, low) in texts() {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let Some(code) = kernels_vulkan::code(symbol, kernels_vulkan::Capability::Baseline)
            else {
                continue;
            };
            let words = driver_vulkan::spirv::words(code).expect("a built module is whole words");
            let d = driver_vulkan::spirv::declared(&words).expect("a built module is well formed");

            match driver_vulkan::binding::params(&low, launch, &d) {
                Ok(driver_vulkan::binding::Params::Push(bytes)) => {
                    pushed += 1;
                    // The block the shader declares, not four bytes a scalar:
                    // a range shorter than this does not cover the last
                    // member, and nothing on this backend would say so.
                    let end = d.push_offsets.iter().map(|o| *o as usize + 4).max();
                    assert_eq!(
                        Some(bytes.len()),
                        end,
                        "{text}: `{symbol}` would be pushed {} bytes for a block \
                         ending at {end:?}",
                        bytes.len()
                    );
                }
                Ok(driver_vulkan::binding::Params::Block { bytes, at }) => {
                    blocked += 1;
                    // Exactly the shader's struct. `tests/device.rs` shows a
                    // short one accepted, returning 256 zeros, with the
                    // validation layer silent.
                    assert_eq!(
                        d.block_bytes.get(at).copied().flatten(),
                        Some(bytes.len() as u32),
                        "{text}: `{symbol}` would write {} bytes into binding {at}",
                        bytes.len()
                    );
                    assert!(
                        (at as u32) < d.bindings,
                        "{text}: `{symbol}` would bind its block at {at} of {} \
                         descriptors",
                        d.bindings
                    );
                }
                Ok(driver_vulkan::binding::Params::None) => {}
                Err(_) => {
                    owed.insert(symbol.clone());
                }
            }
        }
    }

    // Both shapes have to actually occur, or a rule that never fires is
    // passing for the same reason an absent one would.
    assert!(pushed > 0 && blocked > 0, "push={pushed} block={blocked}");
    // Six, not eight -- and the two that fall out are the useful part.
    //
    // `affine_qmv_routed` states five scalars and its module's push block
    // holds exactly five; `embed_gather_mb_4bit` states one and holds one.
    // What those two are short of is a DESCRIPTOR, not a parameter, so their
    // scalars are fully the plan's and only a buffer is owed. Being short of
    // one thing does not make a kernel short of the other, and treating
    // "binds more than the plan names" as one bucket would have hidden that.
    //
    // Five of the remaining six are short of both, which is what a paged KV
    // cache and its page table look like from here: the driver owns the
    // resource, so it also owns the numbers describing it. The sixth is short
    // of a scalar alone, and its own note below says why that is a different
    // thing entirely.
    let want: std::collections::BTreeSet<String> = [
        // The sixth, and the odd one out: `add_bias` is short of a scalar and
        // short of NOTHING ELSE. Its two descriptors are both the plan's, and
        // the one word its module reads is the row width of its own output --
        // which the statement does not carry, because an `AddBias` states no
        // params at all. `Source::Slot(Kind::OutWidth, 0)` is how the row says where to
        // read it and `binding::scalars` is what reads it, so this symbol
        // fires today; the five below still owe a Vulkan executor a resource.
        "add_bias_bfloat16",
        // Not a resource but two NUMBERS, which is why it sits in this list
        // and not among the paged rows below: the strided cast's push block
        // has a `count` word its own walk never reads and a row count the
        // statement has no reason to carry. See its row in `REACHES`.
        "cast_qmm_input_strided_bfloat16_to_float16",
        "kv_append_paged_bfloat16",
        "neox_mb_bfloat16",
        "neox_freqs_mb_bfloat16",
        "row_gather_bfloat16",
        "sdpa_paged_decode_bfloat16_d_128",
        "sdpa_paged_decode_sink_bfloat16_d_64",
        // The tiled prefill pair, which is the decode operand list with the
        // true row count added, so it is short of exactly what decode is
        // short of and for the same reason: the paged KV cache and its page
        // table are the driver's resources, so the numbers describing them
        // are the driver's too.
        "sdpa_paged_tiled_bfloat16_d_128",
        // The sink text has 64-wide heads, so it reaches the COOPERATIVE
        // MATRIX tier rather than the scalar tile -- `sdpa_paged_mma` is
        // compiled at `_d_64` alone. Short of the same eight descriptors as
        // every other row that walks a paged cache.
        "sdpa_paged_mma_sink_bfloat16_d_64",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(
        owed, want,
        "a different set of launches has scalars this crate cannot place"
    );
    // Every one of them is also short of descriptors, EXCEPT the one that is
    // short of a derived number instead. The reverse does not hold, which is
    // the finding above.
    for symbol in &owed {
        assert!(
            REACHES.iter().any(|(n, r)| n == symbol
                && matches!(r, Reaches::DriverSupplies(_) | Reaches::RowDerives(_))),
            "`{symbol}` has scalars this crate cannot place and yet its module \
             binds nothing the plan does not name, which leaves the refusal \
             unexplained"
        );
    }
}

/// Every launch of every real plan becomes a dispatch through the ROUTINE
/// path, or is one of a named set of refusals.
///
/// The other tests here take the binder and the parameter placer apart and
/// ask each its own question. This one puts them back together with the
/// geometry and asks the only question a driver actually has: given a plan,
/// a set of built modules and an arena, how many of its rectangles can be
/// recorded?
///
/// All 7224, across SIX texts in both fire classes. It began at 3180 of 3992
/// over three texts, and the 812 that refused were six symbols short of
/// something nobody had built; each one leaving that list was a defect in
/// this crate rather than a gap in a plan, and the list is now empty.
///
/// # It plans through `plan_routine`, not `KERNELS`
///
/// This walk USED to drive `dispatch::plan_one(..., kernels_vulkan::KERNELS,
/// ...)` -- the table path. `kernels_vulkan::KERNELS` is now EMPTY: every one
/// of these hundred kernels is a `kernels-vulkan` routine plus a `driver-
/// vulkan` arm, and a plan reaches its dispatch through
/// `serve::plan_routine`, which resolves the routine and arm with
/// `hold::arm_for(symbol)` and runs the body against a reflection of the built
/// modules. So this walk does too, which is why it hands `plan_routine` a
/// `serve::Reflection` over a map of the `.spv` files rather than a row.
///
/// The table columns this walk once read -- `sig.grid_param`,
/// `sig.head_param`, `sig.heads_param`, `sig.launch`, `Source::OutWidth` --
/// have no reader left on this backend, and the counters keyed on them
/// (`overridden`, `head_overridden`, `heads_overridden`, `derived_widths`,
/// `split_rectangles`, `rotary_overridden`) are gone with them. Each deletion
/// says below exactly what it checked and what covers it now: the ARMS in
/// `src/arm.rs` compute the grids and interleave the driver's numbers, and
/// they have their own unit tests; `tests/device.rs` fires them on a card.
/// A retired reference does not make those checks true -- it makes them blind,
/// so they are removed by name rather than left reading an empty table.
///
/// What SURVIVES is every invariant a routine-path dispatch still has to
/// honour, and it is asserted here rather than assumed: that all 7224
/// rectangles plan; that no planned grid holds a zero; that a dispatch's
/// operands plus the slot its scalar block takes are exactly the module's
/// real bindings; that both halves of the parameter split occur; that the
/// widest grid and the total workgroups are what they were measured to be;
/// that the pool's page size reaches the shader it is handed to; and, moved
/// here from `rules.rs` because a grid is no longer a `Rule` COLUMN but a
/// thing a routine body computes and so does not exist until a rectangle is
/// planned, that no dispatch puts work on an axis its module never reads
/// (WASTE) and that no entrypoint reads an axis left flat in every rectangle
/// six real texts state (DATA LOSS). `rules.rs` proved those two against the
/// table at a forced fire; with the table empty they could only report "0
/// entrypoints checked", so they plan a real fire here instead.
///
/// Three texts became six by adding `qwen2_5_1_5b`, `mistral_7b_v03` and
/// `olmo2_1b`, which lower and plan without a single new refusal -- so the
/// walk grew by 57% and found nothing, which is a weaker result than a
/// failure and is why it is written down rather than assumed.
///
/// `phi3_mini` is NOT here, and its absence is a measurement: it is 96 wide
/// per head, and `sdpa_paged_decode` is compiled at 64, 128, 256 and 512.
/// `model-compiler` refuses the plan before this crate sees it. That is not a
/// driver gap -- `kernels-metal` compiles the same four widths -- so serving
/// it needs a kernel variant, in both trees, and not a change here.
#[test]
fn every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal() {
    if !kernels_vulkan::embedded() {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    }

    // The built modules, keyed by file stem, which is what `serve::Modules`
    // asks a store to hold. A routine names an ENTRYPOINT and the reflection
    // reads the module out of this map -- the same lookup the driver does in
    // production, minus the device that turns bytes into a pipeline.
    let reflection = driver_vulkan::serve::Reflection::new(
        &driver_vulkan::serve::Embedded,
        kernels_vulkan::Capability::Baseline,
    );

    /// An arena big enough that no weight or seam value is what refuses.
    const GENEROUS: u64 = 1 << 30;
    /// The page size the resolver answers, distinct enough that finding it in
    /// a push block is finding the pool's number and not a coincidence. The
    /// same technique the table walk used, kept because the seam it watches --
    /// a driver number reaching the shader as a scalar -- is exactly as live
    /// on the routine path.
    const PAGE_SIZE: u32 = 0x0011_1111;

    struct Everything(driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Everything {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        // The KV cache and the fire tables are the driver's own, and this
        // walk is about ORDER, not about where they come from -- so it says
        // yes to all of them and lets the arm and the arity check do the work.
        fn kv(&self, _: u16, _: bool) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn table(
            &self,
            _: driver_vulkan::binding::FireTable,
        ) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        // The pool's numbers, which the table walk resolved to zero and this
        // one answers with a sentinel: an arm that names `KvPageSize` puts
        // this value in the run, and the loop below finds it there. Without
        // it the paged kernels would refuse -- their arms take the page size
        // and cannot proceed without one -- so answering it is also what lets
        // the two paged decodes and the paged append plan at all.
        fn number(&self, which: driver_vulkan::binding::FireNumber) -> Option<u32> {
            Some(match which {
                driver_vulkan::binding::FireNumber::KvPageSize => PAGE_SIZE,
                driver_vulkan::binding::FireNumber::KvHeadStride => 0x0022_2222,
                driver_vulkan::binding::FireNumber::KvSeqStride => 0x0033_3333,
                driver_vulkan::binding::FireNumber::AttentionMaskStride
                | driver_vulkan::binding::FireNumber::KvHistoryBucket => 0,
            })
        }
    }

    let mut launches = 0u32;
    let mut planned = 0u32;
    let mut dispatches = 0u64;
    let mut refused: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    let mut widest_grid = [0u32; 3];
    let mut workgroups = 0u64;
    let mut pushed = 0u32;
    let mut blocked = 0u32;
    let mut pool_numbers = 0u32;
    let mut refused_hollow = 0u32;
    // The two grid-axis properties that moved here from `rules.rs`, which
    // could no longer state them: on the routine path a grid is not a `Rule`
    // column, it is computed inside the routine body and does not exist until
    // a rectangle has been planned, so the only place to compare a grid to the
    // axes its module reads is a walk that plans one. `wasted` collects the
    // waste direction per dispatch; `read_lanes` accumulates, per entrypoint,
    // the widest LANE extent any rectangle reaches on each axis, which the
    // data-loss direction reads once the walk is done.
    let mut wasted: Vec<String> = Vec::new();
    let mut read_lanes: std::collections::BTreeMap<String, ([bool; 3], [u32; 3])> =
        std::collections::BTreeMap::new();

    for (text, low, geometry) in geometric() {
        let buf = driver_vulkan::device::Buffer::placeholder(low.arena_bytes as u64);
        let store = Everything(driver_vulkan::device::Buffer::placeholder(GENEROUS));
        let arena = driver_vulkan::binding::Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            launches += 1;
            let symbol = &low.kernels[launch.kernel as usize];
            // The join the table used to make: a plan's instantiated symbol
            // to the routine that serves it. `routine_for` reads no kernel
            // row -- it matches a stem against `hold::LIVE` -- which is the
            // whole point of the fork and the reason this walk still has a
            // subject with `KERNELS` empty.
            let routine = driver_vulkan::hold::routine_for(symbol)
                .unwrap_or_else(|| panic!("{text}: nothing serves `{symbol}`"));
            match driver_vulkan::serve::plan_routine(
                &low,
                launch,
                symbol,
                routine,
                arena,
                &store,
                geometry,
                &reflection,
                STRICTEST_ALIGNMENT as u64,
            ) {
                Ok(plan) => {
                    planned += 1;
                    // The same launch against a geometry of ZEROS, which is
                    // the nastiest input a caller can hand this seam. Under
                    // the table path a head-shaped ROW replaced the zeros
                    // before a rule saw them; here a head-shaped COLUMN asks
                    // for `head_dim` and the binder refuses when it is zero,
                    // and either way
                    // the finding is the same: no real rectangle can be
                    // driven to an empty grid from the outside. The ones that
                    // plan anyway take their dimensions from the lowering
                    // rather than the geometry, so the zeros never reach them.
                    match driver_vulkan::serve::plan_routine(
                        &low,
                        launch,
                        symbol,
                        routine,
                        arena,
                        &store,
                        driver_vulkan::dispatch::Geometry::default(),
                        &reflection,
                        STRICTEST_ALIGNMENT as u64,
                    ) {
                        Ok(hollow) => {
                            for d in &hollow {
                                assert!(
                                    !d.groups.contains(&0),
                                    "{text}: `{}` planned {:?} from a geometry of zeros",
                                    d.symbol,
                                    d.groups
                                );
                            }
                        }
                        Err(_) => refused_hollow += 1,
                    }

                    for d in &plan {
                        dispatches += 1;
                        // A grid with a zero in it runs nothing and reports
                        // success, which is the failure this crate refuses
                        // hardest. The routine body is supposed to have caught
                        // it; `tests/device.rs` is where a card confirms an
                        // empty grid is noticed at all.
                        assert!(
                            !d.groups.contains(&0),
                            "{text}: `{}` planned a grid of {:?}",
                            d.symbol,
                            d.groups
                        );
                        for (widest, got) in widest_grid.iter_mut().zip(d.groups) {
                            *widest = (*widest).max(got);
                        }
                        workgroups += u64::from(d.groups[0])
                            * u64::from(d.groups[1])
                            * u64::from(d.groups[2]);

                        // The dispatch names its own entrypoint -- a body may
                        // instantiate an axis and spell `affine_qmm_t_..._b_4`
                        // itself -- so the module to check it against is
                        // whatever it named, not the plan's symbol.
                        let code =
                            kernels_vulkan::code(&d.symbol, kernels_vulkan::Capability::Baseline)
                                .unwrap_or_else(|| panic!("{text}: no module for `{}`", d.symbol));
                        let words = driver_vulkan::spirv::words(code)
                            .expect("a built module is whole words");
                        let declared = driver_vulkan::spirv::declared(&words)
                            .expect("a built module is well formed");

                        // ---- WASTE: work on an axis the shader never reads ---
                        //
                        // Moved from `rules.rs`'s `no_rule_puts_work_on_an_axis_
                        // its_shader_never_reads`, which stacked a fixed fire
                        // through the geometry and compared `grid_axes` to the
                        // workgroup count. A routine has no `Rule` to stack --
                        // it computes its grid in its body -- so the only grid
                        // to check is the one a planned dispatch carries.
                        // `grid_axes[axis]` is the component of `gl_WorkGroupID`
                        // or `gl_GlobalInvocationID` the body is indexed by; a
                        // group given where nothing reads is a wavefront
                        // launched to do nothing, and it once hid a `Rule::Rms`
                        // that stacked the row count on y while `norm/rms.slang`
                        // reads its row from x.
                        //
                        // This FOLDS IN `a_shader_indexed_by_an_axis_is_given_
                        // that_axis`, whose `highest_given > highest_read`
                        // predicate is this same one once expressed over a
                        // planned dispatch: an axis carrying work above the
                        // highest the body reads is an axis the body does not
                        // read, i.e. `!grid_axes[axis] && groups > 1`. So it is
                        // not written twice.
                        for axis in 0..3 {
                            if !declared.grid_axes[axis] && d.groups[axis] > 1 {
                                wasted.push(format!(
                                    "{text}: `{}` gets {} workgroups on axis \
                                     {axis}, which it is never indexed by",
                                    d.symbol, d.groups[axis]
                                ));
                            }
                        }

                        // ---- DATA LOSS accumulation: the widest lane extent
                        // this entrypoint reaches on each axis, over every text.
                        //
                        // The mirror of the check above and the dangerous
                        // direction, from `rules.rs`'s `no_module_reads_a_grid_
                        // axis_its_rule_leaves_flat`: an axis the body READS
                        // left at one lane drops every index past the first,
                        // silently, and the buffer keeps what it was allocated
                        // with. That is `geglu_tanh_strided`, which read
                        // `gl_GlobalInvocationID.y` under a rule that put one
                        // lane on y and lost every row past fifteen of gemma's
                        // per-layer gate on any prefill longer than sixteen.
                        //
                        // `rules.rs` FORCED 128 rows to make a flattened axis
                        // visible; this walk cannot force a fire, so the
                        // discriminator moves from per-rectangle to per-SYMBOL,
                        // asserted once the walk is done: a read axis is a
                        // defect only when it is flat in EVERY rectangle any of
                        // these six texts states, because a single rectangle
                        // that legitimately has one row is not the defect -- an
                        // entrypoint that NEVER gets more than one is. And the
                        // measure is LANES, `groups * local`, not groups: a body
                        // reading `gl_GlobalInvocationID.x` over a head its
                        // 128-wide workgroup covers in one group sees 0..127,
                        // not just 0, so counting groups alone would call that
                        // coverage data loss.
                        let seen = read_lanes
                            .entry(d.symbol.to_string())
                            .or_insert((declared.grid_axes, [0u32; 3]));
                        for axis in 0..3 {
                            let lanes = u64::from(d.groups[axis]) * u64::from(declared.local[axis]);
                            seen.1[axis] =
                                seen.1[axis].max(u32::try_from(lanes).unwrap_or(u32::MAX));
                        }

                        match d.params {
                            driver_vulkan::binding::Params::Push(ref b) => {
                                pushed += 1;
                                // Sized from the BLOCK's extent, not from four
                                // bytes per scalar: a block with a gap in it
                                // needs the gap written or the range does not
                                // cover the members after it.
                                let end = declared
                                    .push_offsets
                                    .iter()
                                    .map(|o| *o as usize + 4)
                                    .max()
                                    .unwrap_or(0);
                                assert_eq!(
                                    b.len(),
                                    end,
                                    "{text}: `{}` pushes {} bytes into a block ending at {end}",
                                    d.symbol,
                                    b.len()
                                );
                                if b.windows(4).any(|w| w == PAGE_SIZE.to_le_bytes()) {
                                    pool_numbers += 1;
                                }
                            }
                            driver_vulkan::binding::Params::Block { ref bytes, at } => {
                                blocked += 1;
                                assert!(
                                    at < declared.bindings as usize,
                                    "{text}: `{}` puts its block at {at} of {}",
                                    d.symbol,
                                    declared.bindings
                                );
                                assert!(!bytes.is_empty());
                                if bytes.windows(4).any(|w| w == PAGE_SIZE.to_le_bytes()) {
                                    pool_numbers += 1;
                                }
                            }
                            driver_vulkan::binding::Params::None => {}
                        }

                        // The operands, PLUS the slot a parameter block takes,
                        // are the module's real bindings -- the layout less
                        // its holes, which is the arity `Device::run`
                        // enforces. If these two ever disagree, every dispatch
                        // this walk produces would be refused at the device,
                        // which is the kind of break that only shows up on a
                        // machine with a GPU in it.
                        //
                        // The block slot is a binding the PLAN never mentions:
                        // six reachable symbols read their scalars from a
                        // struct in a buffer, and `router_topk` states three
                        // operands against a four-binding module.
                        let block = usize::from(d.block_at.is_some());
                        assert_eq!(
                            d.buffers.len() + block,
                            declared.bindings as usize - declared.holes(),
                            "{text}: `{}` bound {} plus {block} for {} real bindings",
                            d.symbol,
                            d.buffers.len(),
                            declared.bindings as usize - declared.holes()
                        );
                    }
                }
                Err(e) => {
                    *refused.entry(format!("{symbol}: {e}")).or_default() += 1;
                }
            }
        }
    }

    // 6616 before the strided cast reached the card, +608 for the staging
    // dispatch that now precedes each dense projection of every precasting
    // text. See `tests/device.rs`, which counts the same rectangles.
    assert_eq!(
        launches, 7224,
        "a different number of rectangles is lowered"
    );

    // Every rectangle all six texts state, in both fire classes, becomes at
    // least one dispatch: nothing refuses, so nothing is named. This list held
    // six symbols and then five under the table path, and each one leaving it
    // was a defect in this crate rather than a gap in a plan. On the routine
    // path the same set plans, which is the evidence the arms resolve what the
    // rows resolved: `embed_gather_mb_4bit`'s token ids, `kv_append_paged`'s
    // interleaved page size and its two pool planes, the paged decodes' head
    // width and page size, all supplied by the arm rather than derived from a
    // row.
    let expected: Vec<(String, u32)> = Vec::new();
    let got: Vec<(String, u32)> = refused.iter().map(|(k, v)| (k.clone(), *v)).collect();
    assert_eq!(got, expected, "a different set of rectangles refuses");
    assert_eq!(planned, 7224, "a different number of rectangles records");
    assert_eq!(
        refused.values().sum::<u32>(),
        launches - planned,
        "the refusals and the successes do not account for every rectangle"
    );

    // Both halves of the parameter split are exercised: 5056 dispatches push
    // their scalars and 1720 carry a block in a buffer slot. That BOTH occur
    // is the invariant -- a change that routed everything one way would leave
    // the other half untested while this test still passed -- and the exact
    // counts are pinned so a shift in the split shows as a diff here.
    //
    // 5056, where it was 4448: the 608 staging casts the precast lowering
    // added all push. A four-word block fits the guaranteed 128 bytes with
    // room to spare, so the cast never reaches the buffer path and the
    // blocked count below does not move with it.
    assert_eq!(pushed, 5056, "{pushed} pushed");
    assert_eq!(blocked, 1720, "{blocked} blocked");

    // ---- What retired with the table -------------------------------------
    //
    // The table walk pinned a shelf of row-column censuses here, each keyed
    // on a `KernelSig` field this backend no longer has a reader for. They
    // are not re-derivable from the routine path because the FACT they
    // measured -- "a row states its own extent at this slot" -- is not a fact
    // about a routine, which computes its grid in code rather than stating a
    // parameter index. Keeping them would mean resurrecting the empty table
    // to read a column out of it, which is the blindness §7 of the bigplan
    // warns about. So each is removed by name:
    //
    // * `overridden == 2220` counted rectangles whose row named `grid_param`.
    //   A routine that varies its grid per layer reads the number in its body
    //   and its arm; `dims_of` is gone. Covered by the arm unit tests and by
    //   `tests/device.rs`, which fires the grid on a card.
    // * `head_overridden == 1056` and `heads_overridden == 352` counted rows
    //   naming `head_param` / `heads_param`. The head shape now comes from
    //   `Facts` into the arm, and the arm passes it as an argument; there is
    //   no row to state a slot. Same coverage.
    // * `rotary_overridden == 704` and `split_rectangles == 0` were keyed on
    //   `sig.launch` (`Rope`, `SplitPacked`). A routine IS its launch rule,
    //   in code, so there is nothing to count.
    // * `derived_widths == 408` counted rows naming `Source::OutWidth`. The
    //   width `norm::add_bias` reads is now supplied by its ARM from
    //   `Facts::width` -- the result's own width, which the trace holds -- not
    //   derived from a row source. That it reaches the shader is covered by
    //   the arm test and, that a bias which reads a zero width is silently
    //   wrong, by `tests/device.rs`.

    // The pool's page size reaches the shader it is handed to. A routine's arm
    // names `FireNumber::KvPageSize` and `Handles::number` interleaves the
    // driver's answer into the run; the resolver above hands back a sentinel
    // and this is the count of dispatches that carried it into their push
    // block. Zero would mean the seam from the pool to the params went quiet,
    // which is a wrong stride reading real memory at the wrong tokens rather
    // than an error. All 704 are `KvPageSize`: these six texts are paged
    // throughout and never name a contiguous stride, which is why
    // `every_pool_number_reaches_the_shader_through_the_arm_that_names_it`
    // exists to reach the other two. 704 is what the table walk measured too,
    // because the paged kernels that carry the page size are the same set --
    // it is the arm rather than the row that puts it in the block now.
    assert_eq!(pool_numbers, 704, "pool_numbers={pool_numbers}");

    // How many launches `plan_routine` REFUSED when handed a geometry of
    // ZEROS -- the head-shaped ones, whose arms read `f.head_dim` and cannot
    // proceed at zero. The rest plan fine because their dimensions come from
    // the lowering rather than the geometry, so the zeros never reach them,
    // and they are right to. 352 is the count of paged appends and attentions
    // across the six texts whose grid is head-shaped; under the table path a
    // head-shaped ROW replaced the zeros before a rule saw them and the same
    // 352 refused, so the number is unchanged while the mechanism moved from
    // row to arm.
    assert_eq!(refused_hollow, 352, "refused_hollow={refused_hollow}");

    // The total work these plans dispatch, as a single number.
    //
    // Here because every other assertion in this test is about SHAPE -- how
    // many operands, where the scalars went, that no grid is zero -- and a
    // grid can be the wrong size while being all of those things. A body that
    // computed the wrong extent changes no other assertion in this file and
    // changes this one.
    //
    // 36_474_730, where it was 34_252_138 before the strided cast reached the
    // card. The cast walks a rectangle of `k` columns by `rows`, one lane an
    // element, so it costs about as many workgroups as the GEMM it stages
    // for saves nothing -- 2_222_592 over the six texts, or 6.5% more total
    // work than these plans did without it. Worth stating plainly: staging
    // is a PASS, and the whole of its return has to come out of the multiply
    // that follows.
    //
    // 34_252_138, where it was 34_771_458 before the post-norm landing folded
    // its `residual_add` into the norm that fed it. 16_640 workgroups is what
    // 64 elementwise adds over these plans' rows came to -- small, because an
    // add is one workgroup per row and a projection is one per tile, which is
    // the whole reason the fold is worth doing for its DISPATCH count and not
    // for its arithmetic.
    //
    // 34_771_458, where it was 48_780_882 before upstream wired the routed
    // expert GEMM. The number FELL by a fifth, and that is the finding: a
    // mixture of experts used to take the routed matvec whatever its
    // rectangle, so a prefill fired one row of workgroups per token; the
    // tiled form covers `QMM_TILE` rows at a time. Fewer workgroups for the
    // same arithmetic is what a tile is for, and this is the only assertion
    // in the file that can see it.
    //
    // 48_780_882, where the table path measured 48_557_010. That difference is
    // real and expected: a routine computes its grid in its own body from
    // `Facts` and the lowering rather than reading a `grid_param` slot, and a
    // handful of the per-head and paged kernels round their workgroup count up
    // where the row's stated extent rounded differently. Both are the same
    // rectangles reaching the same shaders; the arm's `div_ceil` is the
    // authority now, so this is its number and not the row's.
    assert_eq!(workgroups, 36_474_730, "workgroups={workgroups}");

    // One dispatch per launch: no routine in these six texts fans a single
    // rectangle out to more than one `Dispatch`, so the two counts agree.
    // Stated because a body that split a launch in two would change this and
    // nothing else in the walk.
    assert_eq!(dispatches, 7224, "dispatches={dispatches}");

    // The widest grid in any single dimension, across every dispatch, is
    // [14040, 25136, 64]. The first axis was 3584 -- the same extent the
    // table walk measured -- until the routed GEMM was wired: the widest x is
    // now a routed expert tile grid over the sorted stack, which is taller
    // than any dense projection's. The other two are unmoved, since neither
    // the vocabulary nor the head axis is what routing widens. All
    // three are inside what Vulkan GUARANTEES a device will dispatch:
    // `maxComputeWorkGroupCount` is 65535 per axis at the specification's
    // floor, and a grid past it is undefined rather than refused, so a card
    // that ran the part that fits would return success over an output computed
    // for some of its rows. `Device::check` refuses either by name; this says
    // the refusal is not refusing work these texts do.
    assert_eq!(widest_grid, [14040, 25136, 64], "the widest grid moved");
    for (axis, widest) in widest_grid.iter().enumerate() {
        assert!(
            *widest <= 65_535,
            "axis {axis} reaches {widest} workgroups, past the 65535 Vulkan \
             guarantees, so these plans no longer run on a device at the floor"
        );
    }

    // ---- The two grid-axis properties that moved from `rules.rs` ---------
    //
    // WASTE, the cheap direction: an axis a shader is never indexed by, handed
    // more than one workgroup. Every one of the 7224 dispatches was checked as
    // it was planned, against the module its ROUTINE named. None wastes a
    // group. The check that caught `Rule::Rms` on hardware once now answers
    // the same question for every rectangle six real texts state, with no GPU.
    assert!(
        wasted.is_empty(),
        "{} of {dispatches} planned dispatches put work on an axis their \
         shader never reads:\n{}",
        wasted.len(),
        wasted.join("\n")
    );

    // DATA LOSS, the dangerous direction: an axis the body READS left at one
    // lane in EVERY rectangle, so every index past the first is never
    // dispatched and the dispatch succeeds anyway. Asserted per ENTRYPOINT and
    // not per rectangle, because a single legitimately-one-row rectangle is not
    // the defect -- an entrypoint that never once gets more than a lane on an
    // axis it reads is. This is what would have named `geglu_tanh_strided`,
    // whose body read a y it was given one lane of and dropped every row of
    // gemma's gate past fifteen.
    //
    // One blind spot the per-symbol form still has, and it is named rather
    // than hidden: DECODE attention reads its query ROW from `gl_WorkGroupID.y`
    // (`decode_one(group.y, ..)` in `sdpa_paged.slang`), and these six texts
    // decode a SINGLE stream, so that row is one token by the workload's
    // construction -- not by a routine flattening it. The identical y=row
    // mapping in the PREFILL sibling (`row = group.y * 32 + slot_y`) is handed
    // many rows here and IS checked; and batched or speculative decode would
    // give y more than one, which the arm scales and `tests/device.rs` fires on
    // a card. So axis 1 of the decode-attention family is excused exactly the
    // way `rules.rs` excused its `DECODE_ONLY` set -- a row that is one by
    // construction is not a dropped index -- and the two entrypoints it excuses
    // (`sdpa_paged_decode_bfloat16_d_128`, `sdpa_paged_decode_sink_bfloat16_d_64`,
    // the two head widths these texts reach) are counted so a THIRD appearing
    // is a decision rather than an accident.
    //
    // 29 distinct entrypoints are dispatched across the six texts and 55
    // (entrypoint, axis) pairs are read -- the last of each arrived with
    // `rms_residual_bfloat16`, which reads one axis exactly as the
    // `rms_single_row` it replaces does; 2 are the excused decode rows and the
    // other 48 reach more than one lane in at least one rectangle, so none is
    // flat.
    let mut flat: Vec<String> = Vec::new();
    let mut read_pairs = 0u32;
    let mut decode_rows = 0u32;
    for (symbol, (reads, lanes)) in &read_lanes {
        for axis in 0..3 {
            if reads[axis] {
                read_pairs += 1;
                let decode_row = axis == 1 && symbol.contains("sdpa") && symbol.contains("decode");
                if decode_row {
                    decode_rows += 1;
                } else if lanes[axis] <= 1 {
                    flat.push(format!(
                        "`{symbol}` reads axis {axis} and no rectangle in six \
                         texts gives it more than {} lane there",
                        lanes[axis]
                    ));
                }
            }
        }
    }
    assert!(
        flat.is_empty(),
        "{} entrypoints read a grid axis left flat in every rectangle, so \
         every index past the first on it is never written and the dispatch \
         reports success anyway:\n{}",
        flat.len(),
        flat.join("\n")
    );
    // 30 with the strided cast, which is the same arithmetic `REACHES` does:
    // the precast lowering retired two dense symbols, named two staged ones
    // in their place, and added the cast on top.
    assert_eq!(
        read_lanes.len(),
        30,
        "a different number of entrypoints is dispatched: {}",
        read_lanes.len()
    );
    // 57, +2 for the strided cast's two axes: it is the only new SHAPE in
    // the precast lowering. The staged GEMMs read exactly the axes the dense
    // ones they replaced read, which is the check that staging changed where
    // the activation comes from and not how the grid is walked.
    assert_eq!(
        read_pairs, 57,
        "a different read-axis population: {read_pairs}"
    );
    assert_eq!(
        decode_rows, 2,
        "a different number of decode-attention rows is excused: {decode_rows}"
    );
}

/// Every one of the pool's three numbers reaches the shader through the arm
/// that names it, and the three are not interchangeable.
///
/// The walk above can ask this only of the numbers its six texts reach, and
/// they reach exactly one: every paged rectangle names `KvPageSize` and none
/// names a stride, because these texts are paged throughout and the two
/// strides belong to the CONTIGUOUS cache. So the strides went unwatched --
/// replacing either with a constant left the whole suite green, and a wrong
/// stride is not an error but attention reading the wrong offsets and
/// returning numbers.
///
/// # It reads the arms, because there is no table left to sweep
///
/// This test USED to sweep `kernels_vulkan::KERNELS` for rows naming a pool
/// source and put each through `binding::scalars`. `KERNELS` is empty now:
/// the numbers reach a shader because an ARM asks for them --
/// `Handles::number(FireNumber::KvHeadStride)` and its two siblings -- and
/// the driver interleaves the answer into the run the routine dispatches. So
/// this sweeps the arms instead, invoking the two that between them name all
/// three numbers and checking each distinct answer arrives where the shader
/// reads it.
///
/// The old census counters -- ten rows naming a number, three refused as
/// contiguous, a `KvPageSize`/`KvHeadStride`/`KvSeqStride` tally of 7/5/5 --
/// are gone with the rows they counted. A routine is not a row and has no
/// `operands` column to sweep, and reconstructing those numbers would mean
/// reading them back out of the empty table. What replaces the "refused as
/// contiguous" finding is a fact about the POOL, not the row: a paged
/// deployment's `resources::Pool` answers `None` to `number(KvHeadStride)`,
/// and the contiguous arm then returns `Refusal::Absent` -- which
/// `resources.rs` tests and which is why the arm takes the stride as a
/// fallible `number` rather than a constant.
///
/// `Pool`'s answers are checked in `resources.rs` and the shader's addressing
/// in `tests/device.rs`, which hand-writes its push constants. Between those
/// two the seam -- a driver number reaching the run as a scalar -- was open,
/// and this closes it from the arm's side. The distinct sentinels are what
/// make a SWAP fail as loudly as a drop: a resolver answering one number for
/// all three, or an arm passing the head stride where the seq stride goes,
/// changes which value lands where and this notices.
#[test]
fn every_pool_number_reaches_the_shader_through_the_arm_that_names_it() {
    use driver_vulkan::hold::{Facts, Handles};
    use driver_vulkan::binding::{FireNumber, FireTable, Resolve};
    use driver_vulkan::device::{Bound, Buffer};
    use kernels_vulkan::routine::ArgValue;
    use model_ir::trace::ValueId;

    // Distinct and recognisable, so a swap fails as loudly as a drop.
    const PAGE: u32 = 0x0011_1111;
    const HEAD: u32 = 0x0022_2222;
    const SEQ: u32 = 0x0033_3333;

    // A resolver that answers everything: the arm asks for a KV cache, the
    // fire's write directory and its positions, and this hands back one
    // placeholder for each. Only `number` carries meaning -- the three
    // distinct answers are what the check is about -- and the buffers are
    // there so the arm reaches the `number` calls at all.
    struct Pool(Buffer);
    impl Resolve for Pool {
        fn weight(&self, _: &str) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn named(&self, _: ValueId) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn kv(&self, _: u16, _: bool) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn table(&self, _: FireTable) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn number(&self, which: FireNumber) -> Option<u32> {
            Some(match which {
                FireNumber::KvPageSize => PAGE,
                FireNumber::KvHeadStride => HEAD,
                FireNumber::KvSeqStride => SEQ,
                FireNumber::AttentionMaskStride | FireNumber::KvHistoryBucket => 0,
            })
        }
    }

    /// The scalar words the binder handed the shader, in order, discarding
    /// the buffer handles. The strides ride as `Usize` and the page size as an
    /// `I32`, and either way it is the low 32 bits that reach the push block.
    /// One routine's operands, bound the way a real fire binds them.
    ///
    /// Named by its stem rather than by a function, because the arm that used
    /// to be that function no longer exists: what places these scalars now is
    /// the routine's own column plus this driver's `named`, and the point of
    /// this walk is that the pair still puts the right number in the right
    /// slot.
    fn bound(stem: &str, o: &mut Handles<'_, '_>, f: Facts) -> Vec<ArgValue> {
        let r = driver_vulkan::hold::routine_for(stem)
            .unwrap_or_else(|| panic!("nothing serves `{stem}`"));
        driver_vulkan::bind::bind(r.args, r.sources, o, f)
            .unwrap_or_else(|e| panic!("`{stem}` places its scalars: {e:?}"))
    }

    fn scalars(values: &[ArgValue]) -> Vec<u32> {
        values
            .iter()
            .filter_map(|v| match *v {
                ArgValue::I32(x) => Some(x as u32),
                ArgValue::U32(x) => Some(x),
                ArgValue::Usize(x) => Some(x as u32),
                ArgValue::F32(_) | ArgValue::Buffer { .. } => None,
            })
            .collect()
    }

    // Two placeholder inputs -- the new keys and values every append names --
    // and no results, weights or stated scalars: both routines below take
    // their head width and counts from `Facts` when the statement carries
    // none.
    let buf = Buffer::placeholder(1 << 20);
    let args = [Bound::whole(&buf), Bound::whole(&buf)];
    let ins = [0usize, 1];
    let outs: [usize; 0] = [];
    let weights: [usize; 0] = [];
    let params: [Option<u32>; 0] = [];
    let pool = Pool(Buffer::placeholder(1 << 20));

    let facts = Facts {
        rows: 4,
        width: 128,
        in_width: 128,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
        rotary_dims: 128,
        n_experts: 0,
        experts_per_token: 0,
        group: 0,
        bits: 0,
        layer: 0,
        requests: 4,
        tile: None,
        ..Default::default()
    };

    // The paged append names `KvPageSize` and nothing else. Its answer must
    // reach the run, or the paged decodes it feeds read the cache at the
    // wrong page pitch -- silently, because nothing is out of bounds.
    let mut handles = Handles::new(&args, &ins, &outs, &weights, &params, &pool);
    let paged = bound("kv_append_paged", &mut handles, facts);
    let run = scalars(&paged);
    assert!(
        run.contains(&PAGE),
        "`kv_append_paged` names `KvPageSize` and the driver's answer is not \
         in the {run:?} it hands the shader"
    );

    // The contiguous append names BOTH strides, the head stride before the
    // seq stride. Distinct sentinels catch a drop; the order catches a swap,
    // which is the defect with the most plausible output -- attention
    // striding by heads where it should stride by positions, reading real
    // numbers from the wrong rows.
    let mut handles = Handles::new(&args, &ins, &outs, &weights, &params, &pool);
    let contiguous = bound("kv_append", &mut handles, facts);
    let run = scalars(&contiguous);
    let head_at = run
        .iter()
        .position(|w| *w == HEAD)
        .unwrap_or_else(|| panic!("`kv_append` dropped the head stride: {run:?}"));
    let seq_at = run
        .iter()
        .position(|w| *w == SEQ)
        .unwrap_or_else(|| panic!("`kv_append` dropped the seq stride: {run:?}"));
    assert!(
        head_at < seq_at,
        "`kv_append` hands the head stride at {head_at} and the seq stride at \
         {seq_at}, which is the swap this check exists to catch: {run:?}"
    );

    // All three witnessed, each by the routine whose column names it, and
    // each a distinct value that arrived where the shader reads it. A
    // resolver answering one number for all three, or a `named` arm reaching
    // for the wrong `FireNumber`, moves one of these and fails above.
    for (name, present) in [
        ("KvPageSize", scalars(&paged).contains(&PAGE)),
        ("KvHeadStride", run.contains(&HEAD)),
        ("KvSeqStride", run.contains(&SEQ)),
    ] {
        assert!(present, "`{name}` reached no shader through the binder");
    }
}
