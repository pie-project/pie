//! Can this backend bind, place and dispatch what a REAL lowering produces?
//!
//! Every other test in this crate builds its `Lowered` by hand. That checks
//! that the binder, the geometry and the dispatch planner agree with a plan a
//! TEST invented, and says nothing about whether they agree with the plan
//! `model-compiler` produces for an architecture somebody serves. This file is
//! the other question: six real texts, both fire classes, twelve lowerings,
//! 6920 rectangles, put through the code this crate ships.
//!
//! It is `driver-vulkan/tests/arena.rs`'s question asked of this backend, and
//! the interesting part is where the two answers differ. Three places, and
//! each one changes what a check can claim.
//!
//! # Why every count in this file is pinned, and what that costs
//!
//! Each check asserts the SIZE of what it walked — 16540 arena operands, 6920
//! rectangles, 26584 operands bound — because the failure these checks exist
//! to prevent has a silent twin: a sweep that iterated nothing passes exactly
//! as loudly as one that iterated everything and agreed. A `> 0` floor would
//! let the coverage shrink to one text without saying so.
//!
//! The cost is real and should be stated rather than discovered. **These
//! numbers move whenever `crates/model` changes a text.** Rebasing onto
//! upstream moved every one of them in a single afternoon — 14660 became
//! 15140, 6440 became 6680 — and eight tests went red at once for no reason
//! anybody would call a defect. It happened again when `moe_tile` arrived:
//! 15140 became 15356, 24576 became 24792, and the dispatched work FELL by
//! fourteen million workgroups, which is the tile doing its job.
//!
//! And again when upstream landed *"the kernel written for gemma-4 was never
//! called by gemma-4"*: that family's norm now folds its residual, so
//! `rms_residual_bfloat16` joins the launched symbols (28 -> 29) and the
//! separate `residual_add` rectangles it replaces go away -- 6680 -> 6616
//! rectangles, 24792 -> 24664 operands, 15356 -> 15228 arena operands. Two
//! rectangles per gemma layer become one, which is what a fold is, and the
//! count FALLING while a symbol is ADDED is the shape that says so.
//!
//! And again when the dense projections moved to the fp16 PRECAST path
//! ("metal: run the GEMM in half where the device has no bfloat matrix
//! unit"). That path stages each activation to fp16 in a dispatch of its own
//! instead of converting it once per output tile, so a rectangle APPEARS
//! beside every projection it touches: 6616 -> 6920 rectangles (+608),
//! and 6920 -> 6920 (-304) when `norm::rms_rope` folded two dispatches
//! into one,
//! 17148 -> 16540 arena operands, 26584 -> 26584 bound, and 29 -> 30 symbols
//! -- two replaced by their precast twins and `cast_qmm_input_strided` added,
//! which no text had ever launched. A count that rises because work was
//! HOISTED out of a loop is the direction to want.
//!
//! That is the right trade anyway, and it is worth saying why rather than
//! leaving the next person to re-derive it. A number that moves says *the
//! plans changed*, which is a fact worth one line of a diff: somebody reading
//! that diff can see the coverage grew rather than shrank. A floor says
//! nothing, and a floor is what a sweep silently degrades into.
//!
//! But it is only the right trade because updating them is MECHANICAL — the
//! assertion prints both numbers, and the new one is the answer. If it ever
//! stops being mechanical, or if these start being updated without anybody
//! looking at the direction they moved, the honest fix is to assert the
//! coverage (how many texts, how many classes, how many distinct symbols)
//! rather than the volume.
//!
//! # The offsets: the same agreement, with the same nothing to spare
//!
//! `lower`'s arena allocator rounds every placement to 256 bytes, on both its
//! bump path and its free-list path. WebGPU's `min_storage_buffer_offset_
//! alignment` has a guaranteed floor of 256 --
//! `wgpu::Limits::downlevel_defaults()`, restated by this crate as
//! [`driver_wgpu::facts::GUARANTEED_STORAGE_ALIGNMENT`] so the portable half
//! can name it without `wgpu` present -- so every offset every text produces
//! is bindable in every implementation, browser included, and the margin is
//! ZERO.
//!
//! It reads like a designed agreement and it is not one. That allocator's
//! comment says why it picked the number: *"a decode body runs inside a
//! capture, so the same plan must land the same value at the same address on
//! every fire"*. A Metal capture-replay requirement that happens to coincide
//! with a WebGPU buffer-binding limit, in a crate that has never heard of
//! either.
//!
//! Which is why the coincidence is measured over SIX architectures rather than
//! one. Taken over `qwen3_0_6b` alone the answer is a comfortable 2048, and a
//! change of the allocator to 128 would have looked harmless. `gpt_oss_20b`
//! gives the real answer: a 2880-wide row of 2 bytes is 5760, which is not a
//! multiple of 256, so the next operand lands at 5888 and the alignment is
//! whatever the allocator insists on and nothing more.
//!
//! The UNIFORM side is a question the Vulkan file does not have, because
//! Vulkan's scalars leave the binding numbering entirely and this backend's
//! ride a buffer. `min_uniform_buffer_offset_alignment` has the same
//! guaranteed floor of 256 and is a DIFFERENT limit, and here the agreement is
//! not zero-margin, it is absent: every scalar block a real lowering produces
//! is a handful of words, so no block's own size is a multiple of the
//! granularity a suballocated one would have to start at.
//! [`every_arena_offset_a_real_lowering_assigns_is_bindable`] states that,
//! because it is the reason `device::Device::uniform` gives every launch a
//! buffer of its own at offset zero rather than packing a frame's blocks into
//! one.
//!
//! # The modules: unconditional here, conditional there
//!
//! `driver-vulkan` reads its `.spv` out of `kernels-vulkan`'s rlib, and three
//! of its eight checks still return early with a printed reason when `slangc`
//! did not run -- the three that need a module, because `kernels-vulkan/native`
//! is off by default and an empty module table is what that means. There is no
//! such condition here. `kernels_wgpu::entrypoint_source` hands back compile-ready WGSL for
//! all 481 entrypoints out of the rlib, `naga` is a WGSL front end written in
//! Rust, and [`driver_wgpu::reflect::entrypoint`] turns a name into a
//! [`Declared`] with nothing on disk and no adapter.
//!
//! So those three are strictly stronger here than the templates they are
//! ported from: the build that changes a plan is the build that checks it
//! against the shader it will run. A fourth goes further --
//! [`every_symbol_a_real_text_launches_has_a_module`] can PARSE every symbol
//! its texts launch where the Vulkan one can only look the name up in a table.
//! Nothing here is behind `native`, and nothing prints SKIP.
//!
//! # The launch ABI: the parameter check points the other way
//!
//! WebGPU has no push constants, so `driver-vulkan`'s `Params::Push`/
//! `Params::Block` split does not exist: a launch's scalars are the fields of
//! ONE `@group(1) @binding(0)` uniform block, or -- where the routine binds
//! them as a buffer -- a struct at a `@group(0)` slot.
//! [`binding::Params`] carries one variant with the slot as data.
//!
//! The check inverts with it. Vulkan's `layout-10069` finding is that a push
//! block DECLARED wider than the pipeline's range is a validation error, so
//! there the module must not ask for more than the layout promised. A WebGPU
//! uniform binding is a buffer: WGSL requires the BOUND range to cover the
//! struct, and `wgpu` refuses one that is too small. So the direction here is
//! that the shell must not offer LESS than the module's struct needs, being
//! over is fine, and [`every_launchs_scalars_land_where_its_module_reads_them`]
//! is written that way round -- with the module's own field offsets held
//! against the offsets the routine's `bind` packs to, since a block of the
//! right SIZE with the fields in the wrong PLACES is a shader reading a stride
//! where a head count belongs.
//!
//! # A Metal-authored text on a WebGPU backend, on purpose
//!
//! `llama_like_metal` is the METAL lowering of the llama-like forward pass and
//! this is the wgpu shell. That is correct rather than convenient:
//! `kernels-wgpu` is `kernels-metal`'s coverage row for row, axis for axis,
//! point for point -- 100 kernels over 481 entrypoints, pinned against
//! `kernels-metal`'s own source by that crate's `tests/entrypoints.rs` -- so a
//! Metal-authored text names symbols this backend has BY CONSTRUCTION, and a
//! plan compiled for one is a plan the other can bind.
//!
//! If it ever names one this crate does not have, that is a coverage
//! divergence between two backends that are supposed to cover the same
//! kernels, and
//! [`every_symbol_a_real_text_launches_has_a_module`] is where it surfaces --
//! as a missing module rather than as a `Undispatchable::Unknown` at a fire.
//!
//! # The unstated rows, and where that question went
//!
//! This file used to open a section here on the 56 of the table's 100 rows
//! that stated no operands, on `.wiki/new-driver/vulkan.md` §13's argument
//! that they were still launchable, and on the second `reorder` pass
//! [`every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`]
//! ran to prove it.
//!
//! `kernels_wgpu::KERNELS` is empty. There are no rows, stated or unstated,
//! and the fallback they were the argument for is not on any path a plan
//! takes: [`driver_wgpu::dispatch::plan_one`] finds an arm for every one of
//! the 481 entrypoints and never reaches `plan_by_row`. The question did not
//! get answered, it stopped having a subject -- see the `// RETIRED:` block
//! on that test, which says so in those terms. What a routine leaves empty is
//! now `Placed::Nothing`, and the accounting for it is `Declared::holes`.
//!
//! # GPU-free, all eleven
//!
//! The questions are about numbers a compiler produced and modules `naga` can
//! read, and a check that needed a device would not run in the builds that
//! change them. [`binding::Placeholder`] is a size and nothing else, so this
//! file compiles and passes with no features at all.

use std::collections::{BTreeMap, BTreeSet};

use driver_wgpu::binding::{
    Arena, FireNumber, FireTable, ParamSlot, Params, Placeholder, Resolve, Unbindable,
};
use driver_wgpu::dispatch::{Built, Geometry, Sources};
use driver_wgpu::lowering::routine::Stated;
use driver_wgpu::reflect::Declared;
use kernels_wgpu::Capability;
use kernels_wgpu::routine::ArgValue;
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Arg, Fire, Launch, Lowered, Row, lower};
use model_ir::trace::{FireClass, ValueId};

/// The strictest `min_storage_buffer_offset_alignment` a WebGPU implementation
/// may hold a caller to.
///
/// The limit is a MAXIMUM a caller may request and 256 is the value every
/// implementation must accept, so an offset that divides 256 binds everywhere
/// -- including in a browser, where the number is whatever the page's
/// implementation decided -- and one that does not binds on some machines and
/// not others. The adapter in this machine reports 32 (and 64 for the uniform
/// limit), so checking against the local one would pass a plan that fails on
/// hardware nobody in this repository owns, which is the failure this constant
/// exists to make impossible.
const STRICTEST_ALIGNMENT: u64 = driver_wgpu::facts::GUARANTEED_STORAGE_ALIGNMENT as u64;

/// The same floor for the OTHER limit: where a scalar block may start.
///
/// A different question that happens to share a value, which is why
/// `facts.rs` states it as its own constant rather than folding the two. This
/// file uses it to say what a shell may not do with the uniform blocks a real
/// lowering produces.
const UNIFORM_ALIGNMENT: u32 = driver_wgpu::facts::GUARANTEED_UNIFORM_ALIGNMENT;

/// Big enough that a stand-in for a weight or a fire table never becomes the
/// reason something is refused.
const GENEROUS: u64 = 1 << 30;

/// Lower one text, or `None` where it does not lower.
fn lowered(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
    rows: usize,
) -> Option<Lowered> {
    let plan = llama_like_metal(facts, metal, class);
    // `multi_token` ON A PREFILL, and it was missing. Without it a
    // "prefill at 64" is sixty-four SINGLE-TOKEN rows through the prefill
    // text -- a batched decode wearing a prefill's name -- and when upstream
    // gave a batched decode its own attention lane those rows stopped
    // reaching `sdpa_paged_tiled_bfloat16_d_128` and `sdpa_paged_mma_sink_
    // bfloat16_d_64`. Two symbols left `REACHES` and read exactly like a
    // stale table.
    //
    // Three readings were measured and refuted before this one: that the
    // lane had a row threshold (a 64-row multi-token prefill still reaches
    // the tiled attention), that the facts differed (`wgpu_facts` is
    // `serving.rs`'s `backend_facts` line for line), and that a family had
    // silently dropped out of the sweep (the `if let Some(..)` above is now
    // an unwrap and none of the eight fires it).
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            multi_token: matches!(class, FireClass::Prefill),
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

/// `LlamaLikeMetalFacts::synthetic()` with the one line this backend answers
/// differently.
///
/// `synthetic()` is `driver-metal`'s answer sheet -- its own doc says the
/// `add_bias: false` there is read off the ABSENCE of a `Source::OutWidth` arm
/// in that driver's binder -- and this backend's binder has one:
/// `binding::scalars`' `derived` closure answers `OutWidth(i)` from the
/// launch's own last output width. So `norm::add_bias` is launchable here, and
/// stating it once means qwen2.5's plans carry the three bias launches a layer
/// its checkpoint has always shipped, and everything in this file that walks a
/// real plan walks them.
fn wgpu_facts() -> LlamaLikeMetalFacts {
    LlamaLikeMetalFacts {
        add_bias: true,
        // MIRRORS THE STAMP for the same reason `add_bias` does: this file
        // sweeps every text the crate can reach, and a fact it does not carry
        // is a text it does not reach. With `false` here the fused
        // `rms_rope_bfloat16` plan -- a THREE-buffer launch with a rotation in
        // it, unlike either dispatch it replaces -- is swept by nothing, and
        // the arena rules below would go on passing while saying nothing about
        // the one new rectangle in the deployment.
        fused_qk_rope: true,
        ..LlamaLikeMetalFacts::synthetic()
    }
}

/// Every text this crate can reach, both fire classes.
///
/// Decode at one row and prefill at 64: the row count changes the arena's size
/// and the offsets within it, and it changes every launch's RECTANGLE, so a
/// sweep at one row would miss both a placement whose alignment depends on how
/// much came before it and an extent that only reaches the end of the arena
/// when 64 rows of it are bound.
fn texts() -> Vec<(String, Lowered)> {
    geometric().into_iter().map(|(n, l, _)| (n, l)).collect()
}

/// The same texts, each with the model geometry its plans were built from.
///
/// Split out because most of this file never needs it and one test does. A
/// head-width rule cannot be answered from a plan alone: `sdpa_paged_decode` is
/// compiled for a fixed head dimension and the plan states rows and widths, not
/// heads. A driver knows the model; a plan does not.
fn geometric() -> Vec<(String, Lowered, Geometry)> {
    let mut out = Vec::new();
    for (name, facts, metal) in [
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b(), wgpu_facts()),
        (
            "gpt_oss_20b",
            LlamaLikeFacts::gpt_oss_20b(),
            LlamaLikeMetalFacts::gpt_oss_20b(),
        ),
        (
            "qwen3_30b_a3b",
            LlamaLikeFacts::qwen3_30b_a3b(),
            wgpu_facts(),
        ),
        ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b(), wgpu_facts()),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            wgpu_facts(),
        ),
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b(), wgpu_facts()),
    ] {
        for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 64)] {
            // UNWRAPPED, not skipped, for the reason `modules()` states forty
            // lines below about `driver-vulkan`: a walk that drops what it
            // cannot build reports a denominator it did not measure. This was
            // an `if let Some(..)` and it cost a whole afternoon — two symbols
            // vanished from `REACHES`, and the obvious readings (a stale pin, a
            // changed attention lane, a row-count threshold) were each measured
            // and refuted, because the actual event was a FAMILY dropping out
            // of the sweep without saying so.
            let low = lowered(&facts, &metal, class, rows).unwrap_or_else(|| {
                panic!(
                    "`{name}` no longer lowers at {class:?}/{rows}. Every symbol \
                     only it reaches leaves `REACHES` when this happens, which \
                     reads exactly like a stale table and is not one."
                )
            });
            {
                out.push((
                    format!("{name}/{class:?}/{rows}"),
                    low,
                    Geometry {
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
    // `phi3_mini` is in the catalog and is deliberately NOT in the list above:
    // it is 96 wide per head, `sdpa_paged_decode` is compiled at 64, 128, 256
    // and 512, and the declaration names `sdpa_paged_decode_bfloat16_d_96`.
    // `model_ir::trace`'s signature check refuses it -- by PANICKING, not
    // by returning, which is why this is a note and not an assertion: adding
    // the text here would abort the process rather than skip a lowering. Not a
    // driver gap either, since `kernels-metal` compiles the same four widths;
    // serving it needs a kernel variant in every tree, not a change here.
    assert_eq!(
        out.len(),
        12,
        "six architectures in two fire classes lower; a different number did"
    );
    out
}

/// What each symbol's module declares, parsed once.
///
/// A `Declared` is a `naga` parse and a walk, which is a few hundred
/// microseconds, and the walks below would ask for the same twenty-one modules
/// 6920 times. Cached here for the same reason a driver caches a PIPELINE and
/// not a reflection: the answer is a property of the name.
fn modules<'a>(lows: impl IntoIterator<Item = &'a Lowered>) -> BTreeMap<String, Declared> {
    let mut out: BTreeMap<String, Declared> = BTreeMap::new();
    for low in lows {
        for symbol in &low.kernels {
            if out.contains_key(symbol) {
                continue;
            }
            // Unwrapped, not skipped. `driver-vulkan`'s walk skipped a launch
            // whose module it could not open and got its own DENOMINATOR wrong
            // by 200 -- it reported 3792 rectangles in a plan that has 3992.
            // There is no such condition here anyway: the source is in the
            // rlib.
            let declared = driver_wgpu::reflect::entrypoint(symbol, Capability::Baseline)
                .unwrap_or_else(|why| panic!("no module for `{symbol}`: {why}"));
            out.insert(symbol.clone(), declared);
        }
    }
    out
}

/// A resolver that answers everything, so that a refusal is never about a
/// table this file did not build.
///
/// The weights, the seam values, the KV cache and the fire tables are the
/// driver's own and none of them is the plan's to place. What is under test is
/// the ARITHMETIC over the plan's offsets and the ORDER the row asks for, so
/// this says yes to all of them with one generously sized allocation and lets
/// the real checks do the work.
struct Everything(Placeholder);

impl Resolve for Everything {
    type Buffer = Placeholder;
    fn weight(&self, _: &str) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn named(&self, _: ValueId) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn kv(&self, _: u16, _: bool) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn table(&self, _: FireTable) -> Option<&Placeholder> {
        Some(&self.0)
    }
}

/// The same, with three recognisable answers where a pool's numbers go.
///
/// Distinct per number, so that a driver which SWAPPED two strides fails as
/// loudly as one that dropped both. A resolver answering one value for all
/// three would pass a check that only asked "is something there", and swapping
/// a head stride for a sequence stride is the defect with the most plausible
/// output: attention striding by heads where it should stride by positions,
/// reading real numbers out of the wrong rows.
struct Sentinels(Placeholder);

impl Resolve for Sentinels {
    type Buffer = Placeholder;
    fn weight(&self, _: &str) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn named(&self, _: ValueId) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn kv(&self, _: u16, _: bool) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn table(&self, _: FireTable) -> Option<&Placeholder> {
        Some(&self.0)
    }
    fn number(&self, which: FireNumber) -> Option<u32> {
        Some(sentinel(which))
    }
}

/// The sentinel [`Sentinels`] answers for one of the pool's numbers.
///
/// One reading, used twice: the resolver hands these to the driver, and the
/// walks below look for them in the bytes a dispatch carries. A constant
/// written in two places is a constant that can be changed in one of them.
const fn sentinel(which: FireNumber) -> u32 {
    match which {
        FireNumber::KvPageSize => 0x0011_1111,
        FireNumber::KvHeadStride => 0x0022_2222,
        FireNumber::KvSeqStride => 0x0033_3333,
        FireNumber::AttentionMaskStride => 0x0044_4444,
    }
}

/// Whether a dispatch's scalar block carries a word.
///
/// Bytes, because that is what the shader gets. `Params::Block` is the run as
/// it will be written into a buffer, so a number that reached it is four
/// consecutive little-endian bytes somewhere in it and a number that did not is
/// nowhere in it. Aligned to four on purpose: the packer never places a scalar
/// off a word boundary, so an unaligned coincidence is not a hit.
fn carries(params: &Params, word: u32) -> bool {
    let Params::Block { bytes, .. } = params else {
        return false;
    };
    let want = word.to_le_bytes();
    bytes.chunks_exact(4).any(|c| c == want)
}

/// What one launch's BODY states, run the way `dispatch::plan_one` runs it.
///
/// `plan_one` finds the arm, builds the handles, runs the body and binds; a
/// `Dispatch` is the far end of that and cannot say what the body PASSED. This
/// repeats the two public calls in the middle -- `hold::Handles` and
/// `routine::state` -- because the argument list a body passes is the thing a
/// row's `operands` column used to state, and the walks below are about it.
///
/// `None` where the symbol is unarmed or the body refuses, which is the
/// caller's to judge: a refusal is `plan_one`'s to report and
/// `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
/// is where it is reported.
fn stated_by(
    low: &Lowered,
    launch: &Launch,
    geometry: Geometry,
    numbers: &BTreeMap<FireNumber, u32>,
) -> Option<Vec<Stated>> {
    let symbol = low.kernels[launch.kernel as usize].as_str();
    let body = driver_wgpu::lowering::routine::armed(symbol)?;
    let args = &low.args[launch.args.start as usize..launch.args.end as usize];
    let scalars = &low.params[launch.params.start as usize..launch.params.end as usize];
    // The two widths `dispatch::widths` feeds the arm: the LAST widthed
    // operand's and the FIRST's. Mirrored rather than called, because it is
    // private -- and mirroring is what makes this a second reading.
    let widths = || {
        args.iter().filter_map(|arg| match arg {
            Arg::Arena { width, .. } | Arg::Named { width, .. } => Some(*width),
            Arg::Weight(_) | Arg::Raised { .. } => None,
        })
    };
    let facts = driver_wgpu::lowering::hold::facts(
        symbol,
        launch.rows.end - launch.rows.start,
        geometry,
        low.n_requests,
        widths().next_back().unwrap_or(0),
        widths().next().unwrap_or(0),
    );
    let mut handles = driver_wgpu::lowering::hold::Handles::with_numbers(
        args,
        driver_wgpu::lowering::routine::results(body),
        scalars,
        numbers,
    );
    let mut views = driver_wgpu::lowering::views::Views::default();
    let taken =
        driver_wgpu::lowering::bind::bind(body.args, body.sources, &mut handles, facts, &mut views).ok()?;
    // ON A PLANNER THAT CAN ANSWER. `state`'s has no fire behind it, so every
    // body that reaches for a fact -- which after the marks migration is most
    // of them -- refuses `Unstated` and this helper returns `None`. A walk
    // built on it measured nothing, and the assertions below could not tell
    // "no rectangle does this" from "no rectangle was read".
    let handles = core::cell::RefCell::new(handles);
    driver_wgpu::lowering::routine::stating(body, &taken, &handles, facts).ok()
}

/// The offsets a body's scalars pack to, from WGSL's alignment rule.
///
/// A SECOND reading of `routine::bind`'s packing, written from the rule rather
/// than from the code, and that is the whole point of it: a helper that asked
/// `bind` where it put a field would only be checking `bind` against itself.
/// Every value is aligned to its own width -- four for a word, eight for a
/// `vec2<u32>` -- which is what WGSL requires of a host-shareable struct's
/// members and what a shader therefore reads at.
fn packed_offsets(scalars: &[ArgValue]) -> Vec<u32> {
    let mut at = 0usize;
    let mut out = Vec::with_capacity(scalars.len());
    for value in scalars {
        let width = match value {
            ArgValue::Usize(_) => 8,
            // A body cannot pass a buffer as a scalar: `routine::state`
            // separates the handles out. Counted as a word so that a change
            // there shows up as a misplacement rather than as a silent skip.
            // `Shaped` IS A BUFFER WITH ITS RECTANGLE. It reaches a scalar
            // run for the same reason `Buffer` does -- it does not -- and is
            // counted the same way so a change there shows as a misplacement.
            ArgValue::I32(_)
            | ArgValue::U32(_)
            | ArgValue::F32(_)
            | ArgValue::Buffer(_)
            | ArgValue::Shaped { .. } => 4,
            // A RAISED VIEW is not a scalar and never lands here: `state`
            // splits it off with `Buffer` and `Shaped` -- host data the body
            // already read, packing nothing -- and `packed_offsets` is
            // fed the scalar half. Reaching this arm would mean the split
            // above went wrong, so it panics rather than assigning a width
            // that would silently hide the misplacement.
            ArgValue::Raised(_) => unreachable!(
                "a raised view reached the scalar run -- `routine::state` \
                 should have split it off with the handles"
            ),
        };
        at = at.next_multiple_of(width);
        out.push(u32::try_from(at).expect("a block of a few words"));
        at += width;
    }
    out
}

// RETIRED: `struct_slot` and `gaps`, two readings of a kernel's ABI off the
// TABLE.
//
// `struct_slot` answered where a row put its parameter STRUCT, by walking
// `kernels_wgpu::bindings` for the `Binding::Storage(at)` under a
// `Source::Param` operand. `gaps` answered how many `@group(0)` slots a row
// stated and nothing filled -- `Source::Unbound`, bounded by the module's own
// binding count, which is how `kv_append_paged` keeps seven ring-ABI
// placeholders without shifting the operands behind them.
//
// Both went BLIND, not true. `kernels_wgpu::KERNELS` is empty, so there is no
// `KernelSig` to read either off, and a helper that answers `None` and `0` for
// every kernel is not a helper that started agreeing with the reflection --
// it is one with nothing left to read. The two facts still exist and are read
// from the plan instead: `Dispatch::block_at` says where the struct went, and
// the slots the body leaves empty are what `Placed::Nothing` takes and
// `Declared::holes` names.

/// A raise key this test recognises as a live carrier the routine binder
/// consumes before the low-level [`driver_wgpu::binding::bind`] would be
/// asked to resolve it.
///
/// # Why the check names them rather than pattern-matching on the variant
///
/// [`Arg::Raised`] on its own means "the plan states a host aggregate the
/// driver builds", and after the marks migration every real plan carries a
/// handful: `kv_cache`, `attention_mask`, `attn.split_policy`,
/// `recurrent_state`, `rope.frequencies`. In real driver flow the ROUTINE
/// binder calls `views::raise` on each -- exactly at `lowering/routine.rs`'s
/// operand loop -- before any single-arg `binding::resolve` sees it, so a
/// raise is not a failure of this test's arithmetic; it just does not
/// belong to this test's subject at all.
///
/// The list is stated as a CLOSED set rather than `matches!(_, Arg::Raised
/// { .. })` because that is what makes the sweep still catch a real regression:
/// a plan that grew a raise nothing on this backend answers would land here as
/// a genuinely unbindable operand -- and the low-level `binding::resolve`
/// refusing it with `NotOnThisPlane { key }` for a key that is not in this
/// list stays a failure, not a silent skip. Adding a key here is a
/// deliberate act, matched at the routine binder in `lowering/views.rs`.
fn is_live_raise(key: &str) -> bool {
    matches!(
        key,
        "kv_cache"
            | "recurrent_state"
            | "attention_mask"
            | "attn.split_policy"
            | "rope.frequencies"
    )
}

/// The per-arg walk `driver_wgpu::binding::bind` runs, split so a `Raised` is
/// counted as "not this binder's" rather than as a refusal.
///
/// # Why this stands in for `bind`
///
/// The low-level binder walks every arg and calls `binding::resolve`, and
/// `resolve` refuses [`Arg::Raised`] with `NotOnThisPlane { key }` -- that is
/// correct: a raised operand is a HOST aggregate the routine binder in
/// `lowering/routine.rs:718` consumes through `views::raise` before any
/// single-arg resolve ever sees it. So the real driver never calls the flat
/// `bind` on a plan that carries raises; it walks `Handles::asked()` and
/// asks `resolve` per non-raise operand.
///
/// This helper follows that split for the tests below: `bind_arena_operands`
/// answers what `bind` would have answered for the ARENA/NAMED/WEIGHT operands
/// -- with the same per-argument widening the flat binder does, cribbed from
/// `binding::bind` -- and counts raises apart so the caller can hold them
/// against the plan's own bookkeeping.
///
/// A refusal from the resolver is returned INDEXED into the launch's arg span,
/// exactly like `bind` did; and a raise whose key is not in [`is_live_raise`]
/// stays a refusal, so a plan that stops speaking one of the tier-1/plane
/// vocabulary words this driver understands still fails here.
fn bind_arena_operands<'a, R: Resolve>(
    lowered: &Lowered,
    launch: &Launch,
    arena: Arena<'a, R::Buffer>,
    resolver: &'a R,
    min_offset: u64,
) -> (
    Vec<(usize, driver_wgpu::binding::Bound<'a, R::Buffer>)>,
    usize,
    Vec<(usize, Unbindable)>,
) {
    let span = launch.args.start as usize..launch.args.end as usize;
    let mut bound = Vec::with_capacity(span.len());
    let mut raised = 0usize;
    let mut refused = Vec::new();
    let covered = launch.rows.end - launch.rows.start;
    for (i, arg) in lowered.args[span.clone()].iter().enumerate() {
        if let Arg::Raised { key, .. } = arg {
            if is_live_raise(key) {
                raised += 1;
                continue;
            }
            // A raise key this driver does not build a view for stays a
            // refusal, because that is the same fact `binding::resolve` would
            // have surfaced and the same fact the routine binder would have
            // refused -- one step earlier -- if the plan grew one.
            refused.push((i, Unbindable::NotOnThisPlane { key: key.clone() }));
            continue;
        }
        let own = lowered.arg_rows.get(span.start + i).copied().unwrap_or(0);
        let widened;
        let against = if own > covered {
            widened = Launch {
                rows: launch.rows.start..launch.rows.start + own,
                ..launch.clone()
            };
            &widened
        } else {
            launch
        };
        match driver_wgpu::binding::resolve(arg, against, arena, resolver, min_offset) {
            Ok(b) => bound.push((i, b)),
            Err(why) => refused.push((i, why)),
        }
    }
    (bound, raised, refused)
}

/// Every activation the compiler places can be bound by a `BufferBinding`.
///
/// The claim that makes `Bound::within` usable for real work rather than only
/// for the hand-built arenas in `tests/device.rs`, plus the uniform-side
/// question that has no Vulkan counterpart.
#[test]
fn every_arena_offset_a_real_lowering_assigns_is_bindable() {
    let all = geometric();
    let mut operands = 0usize;
    let mut refused = Vec::new();
    let mut worst = usize::MAX;

    for (name, low, _) in &all {
        for arg in &low.args {
            let Arg::Arena { at, width, bytes } = arg else {
                continue;
            };
            operands += 1;
            if !(*at as u64).is_multiple_of(STRICTEST_ALIGNMENT) {
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
            // One row of it, which is all an `Arg` alone can say. The
            // RECTANGLE -- rows times this -- is what the binder computes and
            // what `the_binder_this_crate_ships_resolves_every_operand_of_
            // every_real_launch` walks; this is the weaker claim that even one
            // row of the operand is inside the arena the plan sized.
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
    // Stated exactly so that the zero above cannot be true by emptiness: a
    // plan that stopped placing activations in the arena would refuse nothing
    // and prove nothing.
    assert_eq!(
        operands, 16540,
        "the texts placed a different number of arena operands than when this \
         was measured, so the zero above is about a different plan"
    );
    // Not a requirement, a MARGIN, and there is none. `worst` is 256 today, so
    // this reads as an equality: any loosening of the allocator fails the
    // refusal check above rather than eroding a cushion first, and any
    // implementation that asked for more than the guaranteed floor would be
    // asking for more than the specification lets it.
    assert_eq!(
        worst, STRICTEST_ALIGNMENT as usize,
        "the tightest alignment any operand has is {worst}; the allocator \
         rounds to 256 and WebGPU's guaranteed floor is 256, so this is an \
         agreement with no room in it and a change to either side is a change \
         this test must be made to state"
    );

    // THE UNIFORM SIDE, which `driver-vulkan` has no question for: its scalars
    // leave the binding numbering through `vkCmdPushConstants`, and this
    // backend's are a buffer that has to start somewhere.
    //
    // No plan places these, so there is no offset to check against a real
    // lowering -- what there is instead is the reason a shell must not invent
    // one. The blocks are TINY, and that is the whole finding: none of them is
    // a multiple of the 256 a suballocated block would have to start at, so a
    // shell packing a frame's blocks into one buffer end to end produces an
    // unbindable offset at the second launch.
    //
    // `binding::Params` carries no offset at all, which is what makes that
    // unrepresentable rather than merely unwise: the only offset this driver
    // can produce for a uniform block is zero, and zero divides everything.
    //
    // Measured over the blocks a real lowering PRODUCES, not over the blocks a
    // table states. `kernels_wgpu::KERNELS` is empty; every symbol resolves
    // through its routine, and the extent of a routine's block is
    // `Declared::uniform_bytes` -- the shader's own span. So the census that
    // used to read `uniform_size` off eleven rows now reads `Params::Block`
    // off every rectangle the twelve real lowerings plan, which is the same
    // question asked of a strictly larger corpus.
    //
    // ONE HALF OF THE OLD CLAIM IS GONE AND IT DID NOT BECOME TRUE, IT MOVED.
    // The old form also asserted every block was a multiple of 16 -- a row
    // could state a block WGSL cannot lay out in the uniform address space,
    // and no row did. A routine cannot: its block is `naga`'s span for the
    // struct it read out of the shader, so five words is 20 bytes here where
    // the row rounded to 32. `Device::uniform` is what rounds the BUFFER up
    // now, with `next_multiple_of(16).max(16)`, and `crate::device` is where
    // that rule is checked. There is no longer a plane on which this file
    // could state an unlayable block.
    let mods = modules(all.iter().map(|(_, low, _)| low));
    let store = Everything(Placeholder(GENEROUS));
    let mut blocks: BTreeSet<u32> = BTreeSet::new();
    let mut carrying = 0usize;
    let mut rectangles = 0usize;
    let mut silent = Vec::new();

    for (name, low, geometry) in &all {
        let buf = Placeholder(low.arena_bytes as u64);
        let arena = Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            rectangles += 1;
            let symbol = &low.kernels[launch.kernel as usize];
            let declared = &mods[symbol];
            let module = driver_wgpu::geometry::Module::loaded(symbol, declared);
            // `plan_all` AND NOT `plan_one`. A rectangle is one statement and
            // may be more than one dispatch: `attn`'s split decode cuts a
            // row's key range into slices and merges them, which is two
            // entrypoints over one launch. `plan_one` refuses that by name
            // (`Undispatchable::Multiple`) and this walk would have counted
            // the refusal as a defect in a corpus that has none.
            //
            // Each pass is asked the question separately, because each pass
            // binds its own block: a merge that offered no scalars to a module
            // that reads them is the same silent bind-group rejection as a
            // single-pass launch doing it, and asking only the first would
            // have stopped seeing half of the split fleet.
            let planned = driver_wgpu::dispatch::plan_all(
                low,
                launch,
                Built { module, declared },
                Sources {
                    arena,
                    resolver: &store,
                    min_offset: STRICTEST_ALIGNMENT,
                },
                *geometry,
            )
            .unwrap_or_else(|why| panic!("{name}: `{symbol}` plans a rectangle: {why}"));
            for pass in &planned {
                match pass.params {
                    Params::Block {
                        ref bytes,
                        at: ParamSlot::Uniform,
                    } => {
                        carrying += 1;
                        blocks.insert(u32::try_from(bytes.len()).expect("a block of a few words"));
                    }
                    // A STORAGE block is a buffer in `@group(0)`, placed by the
                    // arena at an offset the walk above already holds to 256. It
                    // is not this question.
                    Params::Block {
                        at: ParamSlot::Storage(_),
                        ..
                    } => {}
                    // A module that reads a block and a dispatch that offers none
                    // is a bind group `wgpu` rejects at encode, and it would reach
                    // this file as a block of zero bytes rather than as a refusal.
                    Params::None => {
                        if !declared.uniform_offsets.is_empty() {
                            silent.push(format!(
                                "{name}: `{symbol}` offers no scalars against a module that \
                             reads {} of them",
                                declared.uniform_offsets.len()
                            ));
                        }
                    }
                }
            }
        }
    }

    assert!(
        silent.is_empty(),
        "{} rectangles offer no block to a module that reads one:\n  {}",
        silent.len(),
        silent.join("\n  ")
    );
    assert_eq!(
        rectangles, 6920,
        "the texts planned a different number of rectangles than when this was \
         measured, so the census below is about a different corpus"
    );
    // Stated so the filter cannot be true by emptiness: a fork that stopped
    // producing uniform blocks would satisfy every assertion under it.
    assert_ne!(
        carrying, 0,
        "no rectangle of any real lowering carries a uniform block, so nothing \
         below is measuring the driver's scalars"
    );
    let widest_block = *blocks
        .iter()
        .next_back()
        .expect("a rectangle that carries a block states a size");
    assert!(
        widest_block < UNIFORM_ALIGNMENT,
        "every scalar block is smaller than the granularity a suballocated one \
         would have to start at, so a shell that packed them would be placing \
         blocks at offsets no implementation has to accept; the widest is \
         {widest_block}"
    );
    assert_eq!(
        blocks
            .iter()
            .filter(|b| b.is_multiple_of(UNIFORM_ALIGNMENT))
            .count(),
        0,
        "a rectangle now carries a block whose own size is a multiple of the \
         uniform alignment, which is the first block a shell could pack \
         without rounding"
    );
}

/// Every symbol a real text launches has a module this backend can compile.
///
/// The claim `driver-metal`'s `model_bind` makes for its own table, asked of
/// this one: an entry point is compiled from a NAME, so a text that states a
/// symbol the table knows needs no arm written to receive it. Twenty-six
/// distinct symbols, and `kernels-wgpu` has WGSL for all twenty-six — the
/// number is `REACHES.len()`, asserted below, and it said twenty-one here
/// until the assertion had moved twice without the sentence.
///
/// Two things make this stronger than its Vulkan counterpart. It does not
/// skip: there is no build directory to look in, so "the shaders were not
/// built" is not a state this backend has. And it does not stop at the name --
/// `reflect::entrypoint` expands the variant's includes and `//#if` arms,
/// hands the result to `naga`, and refuses a source that is not one
/// dispatchable compute module, so a symbol that is in the table and whose
/// WGSL does not parse fails here rather than at a fire.
///
/// It is a smaller number than the table's 490 because a lowering is not yet
/// the whole of a fire -- `Lowered::residue` holds the statements that still
/// run without a rectangle. What it measures is the part that HAS crossed, and
/// that part is fully served.
#[test]
fn every_symbol_a_real_text_launches_has_a_module() {
    let table: BTreeSet<String> = kernels_wgpu::entrypoints().into_iter().collect();
    // 481 -> 489 -> 490 AS A FAMILY CROSSED. The last is `rms_rope`. `kernels-wgpu` pins the same number from
    // the owning side in its `tests/entrypoints.rs`, and `reflect.rs`'s census
    // sweep pins it from this crate's; all three move together and a
    // disagreement is settled at the table. It is repeated here because this
    // walk needs a DENOMINATOR -- the sentence below is about the corpus being
    // a fraction of the table, and a fraction of an unpinned number says
    // nothing.
    assert_eq!(
        table.len(),
        490,
        "the table states a different set of names"
    );

    let texts = texts();
    let mut launched = BTreeSet::new();
    let mut missing = Vec::new();
    let mut unreadable = Vec::new();
    let mut tiers = 0usize;

    for (name, low) in &texts {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            if !launched.insert(symbol.clone()) {
                continue;
            }
            // The TABLE first, because that is where a coverage divergence
            // shows up: these plans are compiled against `kernels-metal`'s
            // vocabulary, and the two tables are supposed to be one table.
            if !table.contains(symbol.as_str()) {
                missing.push(format!(
                    "{name} launches `{symbol}`, which this backend's table \
                     does not state"
                ));
                continue;
            }
            for tier in Capability::ALL {
                match driver_wgpu::reflect::entrypoint(symbol, tier) {
                    Ok(declared) => {
                        tiers += 1;
                        // A workgroup of zero would divide a fire's extent
                        // into a grid of nothing, and `naga` does not forbid
                        // it here because WGSL forbids it at parse.
                        assert!(
                            declared.local.iter().all(|axis| *axis > 0),
                            "`{symbol}` at {tier:?} declares a workgroup of {:?}",
                            declared.local
                        );
                    }
                    // ORDINARY above baseline and a defect at it: a tier with
                    // no source is how a driver learns to fall back, and the
                    // baseline is the floor a browser gets.
                    Err(driver_wgpu::reflect::Unreadable::NoSource(_))
                        if tier != Capability::Baseline => {}
                    Err(why) => {
                        unreadable.push(format!("{name}: `{symbol}` at {tier:?}: {why}"));
                    }
                }
            }
        }
    }

    assert!(
        missing.is_empty(),
        "{} of {} symbols are not in this backend's table:\n  {}",
        missing.len(),
        launched.len(),
        missing.join("\n  ")
    );
    assert!(
        unreadable.is_empty(),
        "{} module(s) the table names cannot be read:\n  {}",
        unreadable.len(),
        unreadable.join("\n  ")
    );
    // `REACHES` AND `SPLIT` TOGETHER ARE THE DESCRIPTION. Every launched
    // symbol is classified by exactly one of them: `REACHES` says how a
    // single-dispatch statement's operands account against its module, and
    // `SPLIT` names the statements that plan into more than one module and
    // therefore have no single account to give. The union is what "this file
    // describes it" means, and a symbol in NEITHER is a statement nobody has
    // looked at -- which is the whole reason both are enumerations.
    let described = |s: &str| REACHES.iter().any(|(n, _)| *n == s) || SPLIT.contains(&s);
    let undescribed: Vec<&str> = launched
        .iter()
        .map(std::string::String::as_str)
        .filter(|s| !described(s))
        .collect();
    let stale: Vec<&str> = REACHES
        .iter()
        .map(|(n, _)| *n)
        .chain(SPLIT.iter().copied())
        .filter(|n| !launched.iter().any(|s| s == n))
        .collect();
    assert_eq!(
        launched.len(),
        REACHES.len() + SPLIT.len(),
        "the texts launch {} distinct symbols and this file describes {}; \
         launched but undescribed: {undescribed:?}; described but never launched: {stale:?}",
        launched.len(),
        REACHES.len() + SPLIT.len()
    );
    // Every launched symbol at every tier that HAS a source, pinned because
    // the loop above passes vacuously for a tier that has none.
    //
    // This used to read `tiers == launched.len()` -- one module per symbol,
    // every one of them BASELINE -- and the note under it said that was a
    // property of the whole tree, because no `// pie:instantiate` line carried
    // an `@fp16` or `@subgroup` tag. Several now do -- the reduction ladders
    // that `attn.rs` measured as a third of the largest kernel in a decode,
    // and now the PREFILL attention, whose `@subgroup` tier splits a dot the
    // `PIE_TX` lanes used to duplicate. So the count is `launched.len()` plus
    // one per launched symbol that has a second tier.
    //
    // WHAT THE OLD ASSERTION WAS ACTUALLY GUARDING is still guarded, and it is
    // the half that matters: every launched symbol has a BASELINE module, so a
    // core-WebGPU adapter -- a browser -- still serves every one of these
    // plans. `serve::pick` falls back and the loop above proves the fallback
    // exists. The extra pairs are optimisations on top of that, not a new
    // requirement.
    const TIERED: usize = 3;
    assert_eq!(
        tiers,
        launched.len() + TIERED,
        "a different number of (symbol, tier) pairs have a module"
    );
    // Pinned, and it MOVED once already: upstream changed a text to launch
    // `neox_freqs_mb` beside `neox_mb` — a rope that reads a precomputed
    // `inv_freq` where its sibling derives the rotation from a `base` scalar,
    // which is what a rescaled context needs and cannot state a base for. The
    // number moving is the news; it arriving as a failure rather than as a
    // silence is the point of pinning it.
    // 22 became 24 when upstream's texts started launching the 8-bit GEMM
    // rungs (`..._gs_64_b_8_bm_16_bn_32` and `affine_qmv_fast_bfloat16_gs_64_
    // b_8`) beside the 4-bit pair. Growth, and the direction is the news.
    //
    // 24 became 26 when upstream's texts started launching the TILED paged
    // attention beside the decode one. The texts changed, not this file: the
    // symbols were always launchable, and the corpus began naming them.
    //
    // 28 became 29 with `rms_residual_bfloat16`: upstream pointed gemma-4's
    // norm at the residual-folding form it had always had a kernel for. One
    // symbol ADDED while the rectangle count FELL, which is what a fold is.
    //
    // 30 became 31 with `rms_rope_bfloat16`, and the rectangle count fell by
    // 304 in the same move -- see the note at the re-bind pin below.
    //
    // 31 became 33 WITHOUT A SYMBOL BEING ADDED, which is the first time this
    // number has moved for that reason and is why the message above is now
    // wrong about what it counts. `tiers` counts (symbol, tier) PAIRS that
    // have a module, and two of the launched symbols -- `affine_qmv_fast` and
    // `affine_qmv_fast_residual` at `gs_64_b_4` -- gained an `@subgroup`
    // variant that folds their tail ladder in registers. `launched.len()` is
    // still 31 and every one of the 31 still has a baseline module, so a
    // core-WebGPU adapter serves these plans unchanged.
    //
    // It briefly read 35. `pie_workgroup_sum`'s subgroup arm was repaired and
    // pointed at `rms_single_row_bfloat16` and `rms_rope_bfloat16`, and it
    // measured a TIE -- see `common/reduce.inc.wgsl` for the numbers and for
    // why the arm is kept, compilable and correct, with nothing minting it.
    //
    // 33 became 34, again without a symbol being added: `sdpa_paged_tiled_
    // bfloat16_d_128` gained an `@subgroup` variant. The prefill's dot was
    // computed identically by each of a row's `PIE_TX` lanes, and the tiered
    // arm stripes it across them and folds with a butterfly instead -- see
    // `dot_row_split` in `attn/sdpa_paged.wgsl`. Baseline is unchanged and
    // still serves every plan.
    assert_eq!(tiers, 34, "a different number of (symbol, tier) pairs");
}

/// How a symbol's launch reaches its module: what the plan states, what the
/// module declares, and whether the two account for each other.
///
/// Ported from `driver-vulkan`'s `Reaches` with the two variants that mean
/// something different here renamed, because keeping the words `Push` and
/// `Buffer` would describe an API this backend does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reaches {
    /// The launch states no scalars and the module declares no uniform block,
    /// and the plan's operands are all of its bindings.
    ///
    /// Its own variant rather than a zero-field [`Self::Uniform`]: this is the
    /// shape whose `Params` is `None`, and a launch that binds no parameter
    /// group at all is a different call from one that binds an empty block.
    Bare,
    /// The launch's scalars are exactly the fields of the module's
    /// `@group(1) @binding(0)` block, and the plan's operands are all of its
    /// `@group(0)` bindings. Nothing else is needed to fire it.
    Uniform,
    /// The launch's scalars are a STRUCT at this `@group(0)` binding, which
    /// the row named by giving a `Param` operand a buffer kind, and the plan's
    /// operands are the rest.
    ///
    /// The slot is carried because it is not derivable from the operand count:
    /// `route_sort` puts its 28-byte block at 4 of 6 with an operand after it.
    /// Where a parameter struct sits is the kernel's own ABI.
    Storage(u32),
    /// The plan states this many operands the BODY does not bind, on purpose.
    ///
    /// The mirror of [`Self::DriverSupplies`], and it needed its own variant
    /// because the subtraction that produced that one was SATURATING: an
    /// over-count came out as `DriverSupplies(0)`, which reads as "the driver
    /// owes nothing" and means the opposite of what happened.
    ///
    /// The real case is `moe::qmm_t_routed`, whose arm documents it: the
    /// statement carries `x`, `pad` and `tile_expert`, where `pad` is the
    /// sort's padded row count read on the DEVICE. Metal binds `pad` five
    /// times over to fill its argument table's holes; `moe/qmm_t_routed.wgsl`
    /// declares six dense bindings and takes its extent through the grid, so
    /// nothing binds `pad` here.
    ///
    /// A declined operand is not a defect and is not a gap: a gap is a slot
    /// the MODULE declares and nobody fills, and this is an operand the
    /// STATEMENT carries and this backend has no slot for. The balance check
    /// below is what proves the fire is still whole -- it counts what the arm
    /// actually bound, which is why it passed while this classification was
    /// wrong.
    BodyDeclines(u32),
    /// The module binds this many `@group(0)` entries the plan does not state,
    /// because they are the DRIVER's own: the paged KV cache, its page table,
    /// the routing scratch.
    ///
    /// The row's own gaps are subtracted first. A slot the row leaves
    /// unsourced is nobody's debt -- nothing fills it and the shader does not
    /// read it -- and counting it here would say a WebGPU executor owes a
    /// resource that does not exist.
    DriverSupplies(u32),
    /// Every binding is accounted for, and the module's block holds this many
    /// more words than the statement carries: the ROW supplies them.
    ///
    /// Separated from [`Self::DriverSupplies`] because what is missing is a
    /// different kind of thing. A paged attention is short of a RESOURCE only
    /// this driver has; `norm::add_bias` is short of a NUMBER the plan already
    /// implies -- the row width of its own output, which an `AddBias`
    /// statement does not carry because the trace said it when it sized the
    /// output. `Source::Slot(Kind::OutWidth, 0)` is how the row says where to read it and
    /// `binding::scalars` is what reads it, so nothing outside the plan is
    /// needed to fire this kernel.
    RunGrows(u32),
    /// Every binding is accounted for, and the STATEMENT carries this many
    /// more words than the module's block declares: the extra ones are
    /// slot-holders and no shader reads them.
    ///
    /// The mirror of [`Self::RunGrows`], and it exists for the reason
    /// [`Self::BodyDeclines`] does -- the subtraction that produced that one
    /// SATURATES, so a statement with more words than the block came out as
    /// `RunGrows(0)`, which reads as "the row supplies nothing" and means the
    /// opposite of what happened. This file has now been bitten by the same
    /// saturation twice, once per axis.
    ///
    /// It happens for two different reasons and the variant does not
    /// distinguish them, because what a driver does about either is the same:
    /// the block is packed from the fields the MODULE declares, in the
    /// module's order, and a stated word with nowhere to land does not reach
    /// the GPU. `affine_qmm_t`'s extra two are SELECTORS -- `group`, `bits`,
    /// `bm`, `bn` pick the entrypoint and the block is `{ k, n }` -- and
    /// gpt-oss's one is a field that died.
    ///
    /// The real case is `mlp::gptoss_swiglu`. `GptOssSwiGluParams` opened with
    /// a per-row element count that the grid supplies and no body reads, and a
    /// struct has to carry a dead field to keep the live ones at their
    /// offsets. A uniform packed from the marks the statement PASSES does not,
    /// so the block is `{ limit, alpha }` and the dead word never reaches the
    /// GPU -- but the word is still STATED, because `Const` slots are the
    /// statement's run counted in order and `limit` sits at word 1. The
    /// slot-holder goes when the DSL stops stating it.
    BlockDeclines(u32),
}

/// Every symbol the reachable texts launch, and how it must be called.
///
/// Transcribed, so that a text that starts launching something new, or a
/// shader that changes its binding count, is a failure here rather than a
/// surprise in a fire.
/// The statements this driver plans as more than one dispatch.
///
/// One entry, and it is `attn`'s split decode. `sdpa_paged_decode` cuts a
/// row's key range into eight slices so that a fire too narrow to fill this
/// GPU has workgroups to spare, writes a partial and a log-sum-exp per slice,
/// and then merges them -- `_split_` and then `_merge_`, two entrypoints over
/// one launch.
///
/// It is written down here rather than inferred because splitting is a
/// DRIVER's decision about its own occupancy and not a property of the
/// statement: the same trace is one dispatch on a machine with fewer cores
/// and on any backend that does not split. A statement that starts splitting
/// should therefore fail this list until somebody says so on purpose, which
/// is the same rule `REACHES` states for itself.
///
/// `sdpa_paged_decode_sink` is NOT here: it is a complete unsplit kernel and
/// still names one module. That asymmetry is real and is why this is a list
/// and not a prefix test.
const SPLIT: &[&str] = &["sdpa_paged_decode_bfloat16_d_128"];

const REACHES: &[(&str, Reaches)] = &[
    // *** THE MARKS MIGRATION RE-PINNED EVERY LINE BELOW. ***
    //
    // Before the sweep, most of these symbols reached their modules as
    // `Uniform` or `DriverSupplies(N)`: the DSL emitted a packed uniform
    // block that carried each kernel's fixed numbers alongside the
    // driver-supplied resources, so the accounting balanced through the
    // block. The sweep retired the block for those bodies: `head_dim`,
    // `rows`, the strides, the tile widths -- everything the kernel had
    // been reading out of the uniform -- are `Const<i32>` marks the routine
    // states, and the SHADER declares one storage binding per Const<Tensor>
    // instead of consuming a uniform field. So the module's binding count
    // rose against a statement whose declared count did not, and the
    // ledger flipped from "uniform absorbs the difference" (`Uniform`) to
    // "the block declares one word the shader does not read" or "the
    // driver supplies a resource the plan did not name". The values below
    // are what each symbol classifies as post-marks, and the prose above
    // each entry is the historical reasoning for the SHAPE the classification
    // took -- kept because the marks migration did not change what a
    // kernel does, only where the numbers ride.
    //
    // The 8-bit rungs, which the texts started launching when upstream added
    // them. Same shape as the 4-bit pair below and described separately
    // because this table is an enumeration, not a pattern: a symbol nobody
    // wrote down is a symbol nobody looked at.
    //
    // The GEMM tile moved from `bm_16` to `bm_32` upstream: three symbols
    // REPLACED, not added, which is why the count below did not move. That is
    // the direction a reader wants from this diff -- a tile change, not new
    // coverage.
    (
        // FOUR STATED WORDS AND A TWO-FIELD BLOCK. `group`, `bits`, `bm` and
        // `bn` are the statement's marks -- they were facts the driver
        // recovered from the SYMBOL and they choose the entrypoint -- while
        // the block the body fills is `{ k, n }`, both read off the operands'
        // own rectangles. Four words stated, two words in the uniform.
        //
        // This read `RunGrows(0)` and the note beside it said, correctly, that
        // the run runs past the plan's block. It could not say so in the
        // VALUE, because the subtraction that produced it saturated at zero --
        // the same trap `BodyDeclines` was added for, on the other axis. Now
        // it does.
        //
        // BlockDeclines(2) -> (1) WHEN THE TILED GEMM LEARNED ITS ROW COUNT.
        // `qmm_t.wgsl`'s `Params` ends with `m` now so that `write_out` can
        // return on `row >= m` and a partial tile can be rounded up instead of
        // refused -- see `geometry.rs`'s `Rule::Qmm` arm. That is one more word
        // the body reads, so one fewer word the statement carries that the block
        // declines. The four stated marks did not move; the uniform came up to
        // meet them.
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32",
        Reaches::BlockDeclines(2),
    ),
    ("affine_qmv_fast_bfloat16_gs_64_b_8", Reaches::BlockDeclines(1)),
    // THE DENSE PROJECTIONS MOVED TO THE PRECAST PATH. Upstream's "metal: run
    // the GEMM in half where the device has no bfloat matrix unit" points the
    // qwen3.5 and qwen3.6 texts at `_fp16_precast`, so these two are REPLACED
    // by their precast twins and a third symbol appears that no text launched
    // before: the staging pass that rounds an activation to fp16 ONCE instead
    // of once per output tile.
    //
    // On this backend `_fp16_precast` is fp16 STORAGE and an fp32 multiply --
    // `quant/qmm_t.wgsl`'s header says so at length -- because WebGPU's
    // matrix-unit feature is deliberately outside this crate's tier list. Same
    // bindings either way, which is why these two read alike.
    //
    // `Uniform` -> `RunGrows(1)` FOR THE SAME REASON THE ROW ABOVE MOVED. The
    // tiled GEMM's `Params` gained `m`, so the body's run is one word longer
    // than the statement's -- and `RunGrows` is the right side of the ledger
    // for it: the driver supplies the extra word from the fire, which is
    // where a row count comes from. It is not the statement's to state.
    (
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
        Reaches::Uniform,
    ),
    (
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
        Reaches::Uniform,
    ),
    // `RunGrows(0)` and not `Uniform`: the statement carries THREE scalars --
    // `model-dsl::metal::cast_qmm_input` states `[k, k, k]` -- and the body
    // forwards two, because `qmm_t.wgsl`'s strided cast arm declares
    // `Params { k, row_stride }` and nothing more. The `n` between them is the
    // unread slot the packed and strided forms share, and the `count` the
    // packed form takes is `rows * k`, a fire's number this one derives from
    // the grid. Both are marked `Env` on the routine so the block is packed
    // from what the shader actually reads.
    (
        // `RunGrows(0)` until the row stride became a `Const<i32>` -- the
        // rectangle the text laid out, not the fire's -- which is one more
        // word than the plan fills.
        "cast_qmm_input_strided_bfloat16_to_float16",
        Reaches::Uniform,
    ),
    ("affine_qmv_fast_bfloat16_gs_64_b_4", Reaches::BlockDeclines(1)),
    (
        "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
        Reaches::BlockDeclines(1),
    ),
    // Seven bindings, six of them the plan's and one the row's own gap: the
    // unbiased routed QMV keeps the bias slot its biased twin reads, and the
    // module declares it and never touches it. `DriverSupplies(1)` until the
    // gap was counted, which is the whole argument for counting gaps.
    // RunGrows RATHER THAN Uniform SINCE THE MARKS MIGRATION: the routed
    // matvecs read their slot strides and their group/bit counts as
    // `Const<i32>` now -- the statement carries what the driver used to
    // recover from the symbol -- so the run these modules read grows past
    // the uniform block the plan alone filled.
    ("affine_qmv_routed_bfloat16_gs_64_b_4", Reaches::RunGrows(1)),
    // The MXFP4 twin, and the newest row here: it stated NO operands until
    // recently, which made it the one operand-less row a real plan could name.
    // Now it says where its buffers go, and its unread `biases` slot is the
    // same kind of gap -- the codec has no bias plane, so nothing fills it.
    (
        "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
        Reaches::RunGrows(1),
    ),
    // The BATCHED routed pair, which the texts started launching when upstream
    // added `Qwen35MetalFacts::moe_tile` -- the opt-in that turns a MoE
    // prefill's expert banks from one matvec per row into a tiled GEMM.
    //
    // `BodyDeclines(1)` and not a defect: the statement carries `x`, `pad` and
    // `tile_expert`, and `pad` is the sort's padded row count read on the
    // DEVICE because the host cannot know it -- it depends on the routing.
    // Metal binds `pad` five times over to fill the holes its argument table
    // leaves between slot 6 and `tile_expert` at 12; these modules declare six
    // DENSE bindings and take their extent through the grid, so nothing binds
    // `pad` here. `lowering::hold::qmm_t_routed`'s doc is where that is
    // written down, and it skips the index rather than renumbering, so
    // `tile_expert` stays `Input(2)`.
    //
    // The `bn_64` in these two names is a TILE and not a signature: the
    // column axis moved from 32 upstream and `BodyDeclines(1)` is unchanged
    // by it, because a decomposition does not alter what a body binds. See
    // the note beside the disagreement assertion for what that drift cost.
    (
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64",
        Reaches::BodyDeclines(1),
    ),
    (
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64",
        Reaches::BodyDeclines(1),
    ),
    // Two slots, both stated -- the value it biases in place and the bias --
    // and ONE word the statement does not carry.
    ("add_bias_bfloat16", Reaches::Uniform),
    ("residual_add_bfloat16", Reaches::BlockDeclines(1)),
    ("silu_mul_bfloat16", Reaches::BlockDeclines(1)),
    // # NO SYMBOL IN THIS TABLE IS `Storage` ANY MORE
    //
    // These seven were, and the slot each carried was the point: `route_sort`
    // put its 28-byte block at 4 of 6 with an operand AFTER it, so where a
    // parameter struct sits is the kernel's own ABI and not a function of the
    // operand count.
    //
    // Then the packed blocks came apart. `MoeRouteParams`, `ExpertCombineParams`,
    // `RmsParams` and `GptOssSwiGluParams` became the marks their statements
    // pass, and on this plane a mark rides the `@group(1)` uniform -- so every
    // one of them is `Uniform` now and `@group(0)` is dense from zero. The
    // operand that used to sit after a block moved DOWN by one, which is the
    // half of this that a fixture gets wrong quietly: `route_sort`'s `inv` is
    // at 4 rather than 5.
    //
    // `Reaches::Storage` is KEPT, and so is its slot, because a shader may
    // declare a storage params block again -- the variant is what would say so
    // by name. Its absence from this table is asserted in
    // `every_launchs_scalars_land_where_its_module_reads_them`, which counts
    // the three shapes over the whole corpus and now requires the storage one
    // to be zero.
    ("combine_sorted", Reaches::BlockDeclines(1)),
    // TWO WORDS WHERE THE STATEMENT PASSES THREE, and the odd one out in this
    // group. `GptOssSwiGluParams` opened with a per-row element count the grid
    // supplies and no body reads; a struct had to carry it to keep `limit` and
    // `alpha` at their offsets and a uniform packed from the marks does not.
    // The mark stays because `Const` slots are the statement's run counted in
    // order and `limit` sits at word 1. See `Reaches::BlockDeclines`.
    ("gptoss_swiglu_bfloat16", Reaches::BlockDeclines(2)),
    ("rms_single_row_bfloat16", Reaches::BlockDeclines(1)),
    // The residual-folding twin: upstream's "the kernel written for gemma-4 was
    // never called by gemma-4" pointed that family's norm at this symbol, so a
    // text started launching it. Its residual used to be the operand AFTER the
    // block at 3 and is at 3 itself now, with `s` behind it at 4 in the scaled
    // form.
    ("rms_residual_bfloat16", Reaches::BlockDeclines(1)),
    ("route_gather", Reaches::BlockDeclines(1)),
    ("route_sort", Reaches::Uniform),
    // Five bindings with a GAP at 3: the unscaled top-k declares a per-expert
    // scale buffer its body never reads, and the row leaves the slot empty so
    // that the scaled twin's numbering is the same numbering. The gap moved
    // down with everything else when the block left `@group(0)`.
    ("router_topk_bfloat16", Reaches::BlockDeclines(1)),
    // One driver-owned table each: the token ids an embedding gathers by, the
    // positions a rope turns by, and the sampling indices a row gather reads.
    (
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        Reaches::BlockDeclines(2),
    ),
    ("neox_mb_bfloat16", Reaches::BlockDeclines(1)),
    // ONE, the same one, because the fusion changed what a dispatch computes
    // and not who supplies its operands. `rms_rope` is `rms_single_row`
    // (`Uniform`, above) and `neox_mb` (`DriverSupplies(1)`) in a single
    // launch: it takes the norm's gain from the text and the rope's positions
    // from the driver, so the union of a `Uniform` row and a
    // `DriverSupplies(1)` row is a `DriverSupplies(1)` row.
    ("rms_rope_bfloat16", Reaches::BlockDeclines(1)),
    // TWO, where its sibling above supplies one, and the difference is the
    // whole reason this row exists. `neox_mb` derives its rotation from a
    // `base` scalar the text states; `neox_freqs_mb` reads a precomputed
    // `inv_freq` table the DRIVER builds, because a model whose rope is
    // rescaled -- YaRN, or a stretched context -- cannot state a base at all.
    // So the driver supplies the positions AND the frequency table.
    //
    // It arrived here by upstream changing a text rather than this crate
    // changing anything, and the first guess at this line said one. The check
    // named the number it measured, which is the difference between a test
    // that reports a fact and one that asks you to go and find it.
    ("neox_freqs_mb_bfloat16", Reaches::BlockDeclines(1)),
    ("row_gather_bfloat16", Reaches::BlockDeclines(1)),
    // Twelve bindings, six of them the row's ring-ABI gaps, two of them the
    // plan's, and FOUR the driver's: both sides of the paged cache and the two
    // tables saying where this fire writes.
    ("kv_append_paged_bfloat16", Reaches::DriverSupplies(3)),
    // Eight each: the cache's two sides, the page indices and their indptr,
    // the positions, the request map and the two mask tables.
    // `sdpa_paged_decode_bfloat16_d_128` STOOD HERE, as
    // `DriverSupplies(8)`, until it began to split. It is in `SPLIT` above
    // now: a statement planned as two modules has no single account to
    // reach one by, and the per-pass balance check is what covers it
    // instead. The sink below is the same family and does not split, which
    // is why the row shape is still worth reading.
    (
        "sdpa_paged_decode_sink_bfloat16_d_64",
        Reaches::DriverSupplies(3),
    ),
    // The tiled pair supplies the same EIGHT. Its eighteenth operand is the
    // fire's row count, and that is a scalar rather than a binding -- so the
    // count here does not move even though the row grew.
    (
        "sdpa_paged_tiled_bfloat16_d_128",
        Reaches::DriverSupplies(4),
    ),
    // The MMA sink, which is the tiled pair's operand list exactly -- Metal's
    // entrypoint names over a scalar body, since WGSL has no matrix unit to
    // make them differ.
    //
    // It stands where `sdpa_paged_tiled_sink_bfloat16_d_64` used to. This list
    // is what the TEXTS launch, and upstream's lowering moved a sinked prefill
    // from the tiled sink to this one in the same rebase that added
    // `LaunchRule::SdpaMma` -- which is how the MMA rows came to be stated at
    // all, and how their five-scalars-into-a-seven-field-block was found.
    (
        "sdpa_paged_mma_sink_bfloat16_d_64",
        Reaches::DriverSupplies(4),
    ),
];

/// The plan's two runs and the module's two runs account for each other.
///
/// `driver-metal` binds one run: operands and scalars go in one argument
/// table, in the order the row states. WebGPU needs two BIND GROUPS, and the
/// LOWERING already has two runs -- `Launch::args` and `Launch::params` are
/// separate ranges. The question this answers is whether that separation is
/// the same one, and the answer is that it is, in five shapes:
///
/// * six symbols hand their scalars to the module's uniform block and their
///   operands to every one of its `@group(0)` bindings;
/// * two state no scalars at all against a module that declares no block;
/// * six put their scalars in a STRUCT at a `@group(0)` slot the body names,
///   and their operands in the rest;
/// * one is short of a WORD, which its body derives from the launch;
/// * six are short of RESOURCES only the driver has.
///
/// # Holes are not what they are on the other backend, and it changes the sum
///
/// `driver-vulkan` subtracts a module's holes before comparing, because there
/// a hole is a binding number with NO declaration -- `glslc` deletes the
/// declaration of a buffer a variant never reads -- so the descriptor set
/// needs a slot the plan has nothing to put in.
///
/// `naga` deletes nothing. A hole here is a binding that EXISTS and that this
/// entry point happens not to read, so the bind group still needs an entry
/// there, and `wgpu` validates a group against its layout entry for entry.
/// Which means the count that balances is the module's WHOLE binding set, and
/// the thing that fills an unread slot is the body's own `Placed::Nothing`.
///
/// So this asserts what `driver-vulkan` cannot: **the slots the body leaves
/// empty are exactly the slots the module does not read.** If a body grew a
/// gap where the shader reads, the dispatch would be refused by
/// `Unlayoutable::Unfilled`; if a shader stopped reading a slot the body
/// fills, a real tensor would be bound to an entry nothing looks at, which
/// costs a descriptor and says the body is describing a variant it no longer
/// serves.
///
/// # Re-anchored
///
/// The five shapes above were read off a ROW: `struct_slot` found the block's
/// slot in `kernels_wgpu::bindings`, `gaps` counted the row's `Source::Unbound`
/// operands, and `sig.in_place()` named the aliases. `kernels_wgpu::KERNELS` is
/// empty and every one of those readings is gone. Each is taken from what a
/// real plan PRODUCES instead -- the block's slot from `Dispatch::block_at`,
/// the aliases from `Routine::in_place`, and the balance from the buffer list
/// the arm and the body between them filled. The classification is the same
/// classification and `REACHES` is unchanged, which is the point: the
/// information moved and the claim did not.
#[test]
fn what_the_plan_states_and_what_the_module_binds_account_for_each_other() {
    let all = geometric();
    let mods = modules(all.iter().map(|(_, low, _)| low));
    let store = Everything(Placeholder(GENEROUS));
    let mut seen: BTreeMap<String, Reaches> = BTreeMap::new();
    let mut wrong = Vec::new();
    let mut launches = 0u32;

    for (text, low, geometry) in &all {
        let buf = Placeholder(low.arena_bytes as u64);
        let arena = Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            launches += 1;
            let symbol = &low.kernels[launch.kernel as usize];
            let declared = &mods[symbol];
            let module = driver_wgpu::geometry::Module::loaded(symbol, declared);
            // EVERY symbol is armed now, so this is an assertion and not a
            // skip. A symbol that is neither armed nor rowed cannot be planned
            // at all, and a walk that quietly stepped over it would report a
            // full account of a corpus it had thrown part of away.
            let routine = driver_wgpu::lowering::routine::armed(symbol)
                .unwrap_or_else(|| panic!("`{symbol}` is armed: nothing plans without a routine"));
            let mut passes = driver_wgpu::dispatch::plan_all(
                low,
                launch,
                Built { module, declared },
                Sources {
                    arena,
                    resolver: &store,
                    min_offset: STRICTEST_ALIGNMENT,
                },
                *geometry,
            )
            .unwrap_or_else(|why| panic!("{text}: `{symbol}` plans a rectangle: {why}"));

            // A SPLIT LAUNCH IS NOT ONE ACCOUNT, and this is where that stops
            // being a detail. `Reaches` compares what the STATEMENT states --
            // its operand count, its aliases, its scalar run -- against what
            // ONE module declares, and a split decode is two modules over one
            // statement: the split pass takes a scratch buffer the statement
            // never mentions and the merge pass reads that scratch instead of
            // the cache. There is no single number the pair reaches its
            // module by, because there is no single module.
            //
            // So the classification is skipped for them and the BALANCE is
            // not. That is the stronger of the two claims anyway -- every
            // pass fills its own module's bindings exactly, with its own
            // holes -- and it is asked here against each pass's own
            // `Declared` rather than the statement symbol's.
            //
            // The list is pinned rather than inferred for the same reason
            // `REACHES` is an enumeration: a statement that starts splitting
            // is a driver decision somebody made, and it should fail here
            // until it is written down.
            if passes.len() > 1 {
                assert!(
                    SPLIT.contains(&symbol.as_str()),
                    "{text}: `{symbol}` planned {} passes and this file \
                     describes it as a single dispatch",
                    passes.len()
                );
                for pass in &passes {
                    let own = driver_wgpu::reflect::entrypoint(pass.symbol, Capability::Baseline)
                        .unwrap_or_else(|why| panic!("no module for `{}`: {why}", pass.symbol));
                    let filled = pass.buffers.len() + usize::from(pass.block_at.is_some());
                    if filled + own.holes() != own.bindings as usize {
                        wrong.push(format!(
                            "{text}: `{symbol}`'s pass `{}` fills {filled} of its \
                             module's {} bindings and leaves {} unread, which does \
                             not balance",
                            pass.symbol,
                            own.bindings,
                            own.holes()
                        ));
                    }
                }
                continue;
            }
            let d = passes.remove(0);

            let params = launch.params.end - launch.params.start;
            // An IN-PLACE routine binds one buffer for two of the plan's args:
            // the trace states the value and the result separately, because a
            // tape whose statements did not produce values could not say what
            // the next one reads, and the routine then says they are the same
            // allocation. `norm::add_bias` is the only one here, and without
            // this it classifies as a kernel binding one FEWER entry than the
            // plan states -- which is true and is not what is interesting
            // about it.
            let args = (launch.args.end - launch.args.start)
                - u32::try_from(routine.in_place().len()).expect("a routine states few aliases");
            let block = match d.params {
                Params::Block {
                    at: ParamSlot::Storage(at),
                    ..
                } => Some(at),
                Params::Block {
                    at: ParamSlot::Uniform,
                    ..
                }
                | Params::None => None,
            };
            let gaps = u32::try_from(declared.holes()).expect("a module declares few bindings");
            let uniform = declared.uniform_offsets.len() as u32;

            // What the body and the plan together account for, against what
            // the module declares. Asked FIRST, because a buffer account that
            // does not balance makes the scalar comparison meaningless: a
            // kernel short of the KV cache is short of the numbers describing
            // it too.
            let accounted = args + gaps + u32::from(block.is_some());
            if accounted > declared.bindings {
                println!(
                    "OVER-ACCOUNTED `{symbol}`: args {args} + gaps {gaps} + block {} = {accounted}, module declares {} bindings, uniform fields {uniform}, params {params}",
                    u32::from(block.is_some()),
                    declared.bindings,
                );
            }
            let reaches = if accounted > declared.bindings {
                Reaches::BodyDeclines(accounted - declared.bindings)
            } else if accounted != declared.bindings {
                Reaches::DriverSupplies(declared.bindings - accounted)
            } else if let Some(at) = block {
                Reaches::Storage(at)
            } else if uniform == params {
                if uniform == 0 {
                    Reaches::Bare
                } else {
                    Reaches::Uniform
                }
            } else if uniform > params {
                Reaches::RunGrows(uniform - params)
            } else {
                // NOT `RunGrows` OF A SATURATED SUBTRACTION. See
                // `BlockDeclines`: this arm used to fall through to the same
                // expression and report zero, which says the row supplies
                // nothing when what happened is that the statement carries a
                // word the block does not.
                Reaches::BlockDeclines(params - uniform)
            };

            // The body's gaps and the module's unread bindings are the same
            // slots. Not a tautology: the left side is the list the ARM filled
            // and the BODY ordered, and the right side is what `naga` read out
            // of the WGSL. `Placed::Nothing` is what takes a position and
            // leaves no entry, and this is the only place the two are held
            // together over real launches.
            let filled = d.buffers.len() + usize::from(d.block_at.is_some());
            if filled + declared.holes() != declared.bindings as usize {
                wrong.push(format!(
                    "{text}: `{symbol}` fills {filled} of its module's {} bindings \
                     and leaves {} unread, which does not balance",
                    declared.bindings,
                    declared.holes()
                ));
            }

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

    // The routed pair's tile is TRANSCRIBED into `REACHES` above, and it has
    // drifted once already: `ROUTED_QMM_TILE` went (32, 32) -> (32, 64) in
    // `7bda96b66` and this file kept saying `bn_32`, so the assertion below
    // reported "4 of 29 symbols disagree" and left the reader to work out that
    // two of the four were the other two under a new name. `driver-vulkan`'s
    // `gemm_agrees` reached the same wall over `QMM_TILE` and reads the
    // constant now instead of transcribing it.
    //
    // `REACHES` cannot: it is a `const` and its own header insists it stays an
    // ENUMERATION -- "a symbol nobody wrote down is a symbol nobody looked
    // at". So the transcription stays and this says what happened to it.
    {
        let (bm, bn) = model::shared::llama_like::project::ROUTED_QMM_TILE;
        for stem in [
            "affine_qmm_t_routed_bfloat16_gs_64_b_4",
            "mxfp4_qmm_t_routed_bias_bfloat16",
        ] {
            let want = format!("{stem}_bm_{bm}_bn_{bn}");
            assert!(
                REACHES.iter().any(|(symbol, _)| *symbol == want),
                "`ROUTED_QMM_TILE` is ({bm}, {bn}) and `REACHES` has no \
                 `{want}`. The tile moved upstream: REPLACE the routed rows' \
                 suffix rather than adding rows, because these are the same \
                 two symbols at a new tile and the count must not grow."
            );
        }
    }

    assert!(
        wrong.is_empty(),
        "{} of {} symbols disagree:\n  {}",
        wrong.len(),
        REACHES.len(),
        wrong.join("\n  ")
    );
    // The denominator, so that a classification agreeing over nothing cannot
    // pass: every rectangle of every text was classified, not every symbol
    // once.
    assert_eq!(
        launches, 6920,
        "a different number of rectangles was walked"
    );
    assert_eq!(
        seen.len(),
        REACHES.len(),
        "a different set was reached, and every symbol these texts launch is \
         described above"
    );
}

/// The binder this crate ships resolves every operand of every real launch.
///
/// The unit tests in `src/binding.rs` ask whether each rule is right against
/// operands a test invented. This asks the only question that cannot be asked
/// that way: put a real plan through the real binder, at the strictest
/// alignment any implementation may hold it to, and see whether anything is
/// refused.
///
/// # What has to be supplied, and why that is the finding
///
/// A weight and a seam value are not the plan's to place, so [`Everything`]
/// stands in for the driver's tables. Every arena operand, though -- 17148 of
/// them across six architectures in both fire classes -- goes through the real
/// arithmetic: `rows × width × bytes` from the plan, checked against the plan's
/// arena and then against 256-byte addressing.
///
/// The count that matters is that ZERO are refused, and it is only meaningful
/// because the same walk refuses plenty when the arithmetic is wrong: binding
/// one row instead of the launch's rectangle passes here and is a defect a GPU
/// would find, which is why the extent is asserted below rather than assumed.
///
/// This backend also refuses two things Vulkan does not, and neither fires: a
/// zero-length range, because WebGPU has no empty binding, and a `scale.`
/// constant riding a weight slot, because there is no binding that means
/// "nothing". Both would show up in `refused` by name.
#[test]
fn the_binder_this_crate_ships_resolves_every_operand_of_every_real_launch() {
    let mut operands = 0u64;
    let mut arena_operands = 0u64;
    let mut widest = 0u64;
    let mut total = 0u64;
    let mut refused = Vec::new();

    for (text, low) in texts() {
        // A SIZE and nothing else -- `binding::Placeholder` is the whole of
        // what this arithmetic needs from an allocation, which is why this
        // file needs no adapter and no `native` build. `driver-vulkan`'s
        // counterpart names `device::Buffer`, so its arena test cannot compile
        // without the feature that pulls in a loader.
        let buf = Placeholder(low.arena_bytes as u64);
        let store = Everything(Placeholder(GENEROUS));
        let arena = Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            // The low-level `binding::bind` walks every arg and calls
            // `binding::resolve`, and `resolve` refuses `Arg::Raised` by
            // design -- see its comment: a raised operand is a HOST
            // aggregate the ROUTINE binder in `lowering/routine.rs:718`
            // consumes through `views::raise` before any per-arg resolve
            // would see it. So the real driver never puts a raise through
            // this path, and neither can this test.
            //
            // `bind_arena_operands` above is that split: it walks the same
            // arg span with the same per-argument widening the flat binder
            // does, and it partitions the result into (1) real bounds for
            // arena/named/weight operands, (2) a count of raises the
            // routine binder would have consumed, and (3) refusals. A raise
            // whose key is NOT in the live-carrier list stays a refusal, so
            // a plan that grew an unbindable name still fails the sweep
            // below; the same fact `NotOnThisPlane { key }` states, kept
            // rather than filtered.
            let (bounds, raises, this_refused) = bind_arena_operands(
                &low, launch, arena, &store, STRICTEST_ALIGNMENT,
            );
            operands += bounds.len() as u64 + raises as u64;
            for (i, why) in this_refused {
                refused.push(format!(
                    "{text}: `{}` operand {i}: {why}",
                    low.kernels[launch.kernel as usize]
                ));
            }
            for (i, b) in &bounds {
                let arg = &low.args[launch.args.start as usize + i];
                if matches!(arg, Arg::Arena { .. }) {
                    arena_operands += 1;
                    widest = widest.max(b.len());
                    total += b.len();
                    // The whole reason this backend needs an extent: a
                    // range that reached the end of the arena would
                    // cover every activation placed after it, and
                    // WGSL's bounds checking confines a stray index to
                    // the BOUND range -- so an operand bound too long
                    // is a silent wrong answer rather than a fault.
                    assert!(
                        b.offset() + b.len() <= low.arena_bytes as u64,
                        "{text}: a range from {} for {} leaves the arena",
                        b.offset(),
                        b.len()
                    );
                }
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
    //
    // RE-PINNED FROM 25976 TO 28736 for the marks migration. The old count
    // was the sum of what the flat `binding::bind` accepted; that count fell
    // to whatever came before the first `Arg::Raised` in each launch's args
    // and was uncomparable across plans, so the sweep replaced the flat call
    // with the per-arg walk this test's `bind_arena_operands` runs. The
    // 2760 added are the raises the routine binder in
    // `lowering/routine.rs:718` would consume through `views::raise` -- one
    // per `kv_cache`, `attention_mask`, `attn.split_policy`,
    // `recurrent_state`, `rope.frequencies` on each launch that names one --
    // and this test now counts them because a raise IS a plan operand, just
    // one this binder does not resolve. The arena and named/weight halves
    // still bind through `binding::resolve` exactly as before.
    assert_eq!(operands, 28736, "a different number of operands was bound");
    assert_eq!(
        arena_operands, 16540,
        "the texts produced a different number of arena operands than when this \
         was measured, so the zero above is about a different plan"
    );
    // The number with the teeth in it. "Nothing was refused" is satisfied by
    // any range small enough, so it cannot tell a correct extent from a
    // conservative one -- binding a single row rather than the launch's
    // rectangle is refused nowhere and is wrong everywhere, and binding to the
    // end of the arena is refused nowhere and is the exact defect that lets a
    // kernel scribble on its neighbour. Both change this sum.
    // RE-MEASURED after `crates/kernels` gained `rows_param` and the shared
    // MoE rows started stating it. `route_gather`'s rows are the SORTED STACK
    // -- one per route, `tokens * experts_per_token` of them -- not the fire's
    // tokens, and the three routed matvecs name `out_vec_size` as their grid
    // extent rather than their rectangle's width. Both are row content this
    // crate had not copied, so the plans this file walks changed shape when
    // they were.
    // RE-MEASURED again, and by far the largest move: 2,982,826,080 to
    // 10,518,945,504 when upstream's `moe_tile` turned a MoE prefill's expert
    // banks from one matvec per row into a tiled GEMM.
    //
    // A 3.5x jump is exactly the direction this assertion exists to catch, so
    // it was checked rather than re-pinned. **It is the rectangle, exactly.**
    // The batched arm's rectangle is the PADDED SORTED STACK -- 140 tiles of
    // 32 for qwen3-30b-a3b's 64-token prefill, so `span` is 4480 where the
    // matvec's grid extent was `out_vec_size` -- and its `x` operand binds
    // 18,350,080 bytes, which is 4480 x 2048 x 2 to the byte. The binder is
    // covering the rows the launch states and no more.
    //
    // DOWN by 17,039,360 when gemma-4's norm began folding its residual: 64
    // `residual_add` rectangles went away and each bound 266,240 bytes across
    // its operands. Fewer rectangles binding fewer bytes is the whole content
    // of that upstream change.
    // UP by 1,035,483,936 when `Lowered::arg_rows` let an operand state a row
    // space wider than its launch's rectangle. The epilogue gather is the
    // case: its rectangle is `n_requests` rows and its INPUT spans the token
    // stream, so measured by the launch alone it bound ONE row and WGSL
    // clamped the rest of its reads to zero. Binding the rows the operand
    // actually spans is more bytes by construction, and it is the whole point
    // of the field -- a jump in this direction is the fix, not a regression.
    assert_eq!(
        total, 12_117_156_352,
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
/// indistinguishable from a rule that is not there. So this takes the same real
/// plans and shrinks the arena the binder is TOLD about, which makes the
/// operands at the far end address past it while every other number stays
/// exactly as the compiler produced it.
///
/// Twelve refusals across TEN of the twelve texts, and the distribution is the
/// interesting part. Ten lowerings have at least one rectangle that ends
/// EXACTLY at the end of its arena -- the margin the module docs call zero,
/// measured from the other side, since a plan one row wider would not fit the
/// arena it was given. The two that do not are the two mixture models at
/// decode, whose arenas hold more than a single row's rectangles reach.
#[test]
fn an_arena_one_byte_short_of_what_the_plan_placed_refuses_what_runs_off_it() {
    let mut refused = 0u64;
    let mut launches = 0u64;
    let mut texts_refusing = 0u64;

    for (text, low) in texts() {
        let short = low.arena_bytes as u64 - 1;
        // The BUFFER stays full size. Only the plan's number shrinks, so the
        // refusal has to come from the arena bound and not from the range
        // check -- which is the distinction that makes `PastArena` a variant of
        // its own rather than another `Overrun`.
        let buf = Placeholder(low.arena_bytes as u64);
        let store = Everything(Placeholder(GENEROUS));
        let arena = Arena {
            buffer: &buf,
            bytes: short,
        };
        let before = refused;
        for launch in &low.launches {
            launches += 1;
            // The per-arg walk `bind_arena_operands` runs, for the same
            // reason `the_binder_this_crate_ships_resolves_every_operand_of_
            // every_real_launch` uses it: the flat `binding::bind` refuses
            // every launch that carries a raise as its first non-arena
            // operand, and every real plan does. What this test's subject
            // is -- an ARENA operand that runs one byte off a shrunk arena
            // -- happens under the routine binder's split in real fire, so
            // the walk here honours the same split: raises are skipped
            // (they never touch the arena bound), and every arena/named
            // resolve still runs. A `PastArena` from any of them is the
            // finding.
            let (_, _, this_refused) = bind_arena_operands(
                &low, launch, arena, &store, STRICTEST_ALIGNMENT,
            );
            for (_, why) in this_refused {
                assert!(
                    matches!(why, Unbindable::PastArena { .. }),
                    "{text}: a byte off the arena is not a reason for {why}"
                );
                refused += 1;
            }
        }
        if refused > before {
            texts_refusing += 1;
        }
    }

    // 7,224 became 6,920 and 30 symbols became 31 when `norm::rms_rope`
    // landed: one symbol ADDED while 304 rectangles FELL, the same shape
    // `rms_residual` made and for the same reason. A fusion is always this
    // pair of moves, and a pin that only tracked the count would let a
    // fusion and a deletion look alike.
    assert_eq!(
        launches, 6920,
        "a different number of rectangles was re-bound"
    );
    // Two more, for the same reason the byte total moved: see the note above
    // it. A wider rectangle is a rectangle with more ways to run off the end.
    assert_eq!(
        refused, 15,
        "{launches} launches bound against an arena one byte shorter than the \
         one they were placed in, and a different number ran off it"
    );
    assert_eq!(
        texts_refusing, 12,
        "a different number of texts has a rectangle ending at the end of its \
         arena"
    );
}

/// Every launch's scalars land where its module reads them from.
///
/// `what_the_plan_states_and_what_the_module_binds_account_for_each_other`
/// measures the split; this puts the real plans through the code that ACTS on
/// it, which is a different claim. A classification can be right about a symbol
/// and still be wrong about a launch, because the plan states its scalars per
/// launch and nothing says two launches of one kernel state the same number of
/// them.
///
/// # The check points the opposite way from its Vulkan template
///
/// There, `Params::Push` is held against the declared block's extent because a
/// push block wider than the pipeline's range is a validation error: the module
/// must not ask for more than the layout promised. Here a uniform block is a
/// BUFFER, WGSL requires the bound range to cover the struct, and `wgpu`
/// refuses a binding that is too small -- so the shell must not offer LESS than
/// the module needs and being over is fine.
///
/// Which is exactly the state these launches are in, and the two numbers are
/// worth keeping apart. `Declared::uniform_bytes` is `naga`'s span for the
/// struct -- 20 for five words -- and a shell allocating a BUFFER rounds that
/// to 16, because WGSL gives a host-shareable struct an alignment of at least
/// 16. `Device::uniform` is where the rounding lives. Over, never under, at
/// every launch here.
///
/// # And the size is not the whole of it
///
/// A block of the right length with its fields in the wrong PLACES is a shader
/// reading a stride where a head count belongs, and nothing reports it: a
/// uniform buffer is bytes. So the offsets the body's scalars PACK to are held
/// against the offsets the module READS at, field for field. Two independent
/// readings of one layout: `packed_offsets` walks WGSL's alignment rule over
/// the values `routine::state` says the body passes, and `Declared` walks the
/// struct `naga` parsed out of the shader. `routine::bind` is the code between
/// them and is not consulted by either.
///
/// A PREFIX and not an equality, deliberately. `bind` pads a short run out to
/// the struct's extent, so a body that passes four words to a five-field
/// struct leaves the fifth reading zero -- a real question, and one about what
/// a body states rather than about where the driver puts it.
/// `crates/kernels-wgpu/tests/routines.rs` is where a body's argument list is
/// held to its own module. Counted here, so that the count cannot grow
/// unremarked.
///
/// # Re-anchored, and what that cost
///
/// This used to ask `binding::scalars`, which builds a run from a ROW, and
/// compared `kernels_wgpu::uniform_layout`'s offsets to the module's.
/// `kernels_wgpu::KERNELS` is empty; there is no row to compare, and the
/// question moved rather than closed -- the row's `operands` column is now the
/// argument list a body passes, and `routine::state` is where it is read. The
/// walk is over ALL 6920 rectangles now instead of the 1136 that had rows, and
/// over the 7376 DISPATCHES they plan into -- a split decode is two
/// entrypoints over one statement and each has its own block to place.
#[test]
fn every_launchs_scalars_land_where_its_module_reads_them() {
    let all = geometric();
    let mods = modules(all.iter().map(|(_, low, _)| low));
    let store = Everything(Placeholder(GENEROUS));
    let numbers = BTreeMap::new();
    let mut uniform = 0u64;
    let mut storage = 0u64;
    let mut bare = 0u64;
    let mut owed: BTreeSet<String> = BTreeSet::new();
    let mut misplaced: BTreeSet<String> = BTreeSet::new();
    let mut short: BTreeSet<String> = BTreeSet::new();
    let mut whole_struct = 0u64;
    let mut module_alone = 0u64;
    let mut body_alone = 0u64;
    let mut eight_byte = 0u64;
    // Every module any PASS names, which is a superset of `mods`: the split
    // decode's two entrypoints are named by no statement.
    let mut passed: BTreeMap<String, Declared> = BTreeMap::new();

    for (text, low, geometry) in &all {
        let buf = Placeholder(low.arena_bytes as u64);
        let arena = Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let declared = &mods[symbol];
            let module = driver_wgpu::geometry::Module::loaded(symbol, declared);
            let states = stated_by(low, launch, *geometry, &numbers);
            // A 64-bit scalar has no WGSL type, so a shader declares it as a
            // `vec2<u32>` and a body passes it as `ArgValue::Usize`. Counted so
            // the ZERO below can say the case is unwitnessed here rather than
            // passing silently.
            if states.iter().flatten().any(|state| {
                state
                    .scalars
                    .iter()
                    .any(|v| matches!(v, ArgValue::Usize(_)))
            }) {
                eight_byte += 1;
            }

            // `plan_all`, AND EACH PASS AGAINST ITS OWN MODULE. A split
            // decode is two entrypoints over one statement, and the two do
            // not read the same block -- the split pass takes the partition
            // count and the merge pass takes how many partials to fold. So
            // `declared` above, which is the STATEMENT symbol's module, is
            // the right one to open the launch with and the wrong one to
            // hold either pass's scalars against.
            //
            // `stated_by` returns one `Stated` per dispatch in the same
            // order, which is what makes the zip below meaningful rather
            // than a coincidence: pass `i`'s packed run is compared with pass
            // `i`'s module and with nothing else.
            let planned = driver_wgpu::dispatch::plan_all(
                low,
                launch,
                Built { module, declared },
                Sources {
                    arena,
                    resolver: &store,
                    min_offset: STRICTEST_ALIGNMENT,
                },
                *geometry,
            );
            let passes: Vec<_> = match planned {
                Ok(list) => list,
                Err(why) => {
                    owed.insert(format!("{symbol}: {why}"));
                    Vec::new()
                }
            };
            for (d, state) in passes.iter().zip(states.iter().flatten()) {
                // MEMOISED. `reflect::entrypoint` expands the `//#include`
                // tree and hands it to `naga`, and this walk asks once per
                // dispatch over seven thousand of them; without the cache the
                // test took two minutes and spent all of it re-parsing the
                // same forty modules.
                let declared = passed
                    .entry(d.symbol.to_owned())
                    .or_insert_with(|| {
                        driver_wgpu::reflect::entrypoint(d.symbol, Capability::Baseline)
                            .unwrap_or_else(|why| panic!("no module for `{}`: {why}", d.symbol))
                    })
                    .clone();
                let declared = &declared;
                let symbol = &d.symbol;
                match d.params {
                    Params::Block {
                        ref bytes,
                        at: ParamSlot::Uniform,
                    } => {
                        uniform += 1;
                        // The block the SHADER declares, not four bytes a
                        // scalar: a run written end to end would place every
                        // field after a gap at the wrong offset.
                        assert_eq!(
                            bytes.len(),
                            declared.uniform_bytes as usize,
                            "{text}: `{symbol}` offers {} bytes for a block of {}",
                            bytes.len(),
                            declared.uniform_bytes
                        );
                        // Every field the module reads is inside what the shell
                        // offers. The direction that matters: short reads as
                        // zeros and a zero pitch is a plausible number.
                        for offset in &declared.uniform_offsets {
                            assert!(
                                *offset as usize + 4 <= bytes.len(),
                                "{text}: `{symbol}` reads a field at {offset} out of \
                                 {} bytes",
                                bytes.len()
                            );
                        }
                        // The body's packing and the module's fields, offset
                        // for offset. Collected rather than asserted in place
                        // so a disagreement names every symbol it holds for and
                        // not just the first rectangle to reach it.
                        let packed = packed_offsets(&state.scalars);
                        if declared.uniform_offsets.starts_with(&packed) {
                            if packed.len() == declared.uniform_offsets.len() {
                                whole_struct += 1;
                            } else {
                                short.insert(format!(
                                    "{symbol}: passes {} of {} fields",
                                    packed.len(),
                                    declared.uniform_offsets.len()
                                ));
                            }
                        } else {
                            misplaced.insert(format!(
                                "{symbol}: packs its scalars at {packed:?} and the \
                                 module reads them at {:?}",
                                declared.uniform_offsets
                            ));
                        }
                    }
                    Params::Block {
                        ref bytes,
                        at: ParamSlot::Storage(at),
                    } => {
                        storage += 1;
                        // Exactly the shader's struct. `tests/device.rs` shows
                        // a short one accepted and read back as zeros past its
                        // end.
                        assert_eq!(
                            declared.block_bytes.get(at as usize).copied().flatten(),
                            Some(u32::try_from(bytes.len()).expect("a block of a few words")),
                            "{text}: `{symbol}` writes {} bytes into binding {at}",
                            bytes.len()
                        );
                        assert!(
                            at < declared.bindings,
                            "{text}: `{symbol}` binds its block at {at} of {}",
                            declared.bindings
                        );
                        // Where the SLOT says the struct goes and where the
                        // buffer list says it went. Two readings of one
                        // placement: a driver that filled one and not the other
                        // would bind a block into a slot the shader reads as an
                        // operand, and `descriptors` would number the rest of
                        // the list off by one.
                        assert_eq!(
                            d.block_at,
                            Some(at as usize),
                            "{text}: `{symbol}` names its block's slot twice and \
                             disagrees with itself"
                        );
                    }
                    Params::None => {
                        bare += 1;
                        assert!(
                            declared.uniform_offsets.is_empty(),
                            "{text}: `{symbol}` places nothing against a module with \
                             {} uniform fields",
                            declared.uniform_offsets.len()
                        );
                    }
                }
            }

            // The MODULE-ONLY placer, which needs no row and never did: it
            // takes the statement's run whole and holds it against the block
            // `naga` parsed. Counted rather than asserted, because for a body
            // that interleaves a pool number or derives a width it is the wrong
            // question and its answer is the measurement -- the run a body
            // passes is not the statement's run.
            match driver_wgpu::binding::params(low, launch, declared) {
                Ok(_) => module_alone += 1,
                Err(_) => body_alone += 1,
            }
        }
    }

    // NOTHING is owed, which is a stronger result than the Vulkan template
    // gets: six symbols there have scalars that crate cannot place, five of
    // them because the driver owns the resource and therefore the numbers
    // describing it. Here the body builds the run -- interleaving `KvPageSize`
    // where `kv_append_paged` puts it, taking three of `neox_mb`'s four,
    // deriving `add_bias`'s output width -- so every launch of every text
    // places.
    assert!(
        owed.is_empty(),
        "{} launches have scalars this crate cannot place:\n  {}",
        owed.len(),
        owed.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );
    assert!(
        misplaced.is_empty(),
        "{} kernels pack their scalars where their module does not read \
         them:\n  {}",
        misplaced.len(),
        misplaced.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );
    if !short.is_empty() {
        eprintln!(
            "  {} kernels pass fewer scalars than their module declares fields, \
             and the rest of the struct reads zero:\n  {}",
            short.len(),
            short.iter().cloned().collect::<Vec<_>>().join("\n  ")
        );
    }
    // Every DISPATCH takes exactly one of the three shapes. Stated as an
    // identity rather than as three literals: the split between uniform and
    // storage is a property of which shaders declare a block where, and moves
    // whenever a kernel's ABI does, but a dispatch that took NO shape would be
    // one this walk silently dropped.
    //
    // 6920 -> 7072, AND THE 152 ARE NOT NEW RECTANGLES. The corpus is still
    // 6920 launches -- the number the other walks in this file pin -- and 152
    // of them are the split decode, which plans two dispatches over one
    // statement. A parameter block belongs to a MODULE and each pass has its
    // own, so this walk counts passes where the others count launches. The
    // difference between the two is exactly the extra passes, which is the
    // arithmetic worth writing down: if it ever is not, either a launch
    // stopped planning or a pass stopped being counted.
    assert_eq!(
        uniform + storage + bare,
        7072,
        "a different number of dispatches take each parameter shape"
    );
    // And the shape that occurs occurs, or a branch that never runs is passing
    // for the same reason an absent one would.
    //
    // # THE STORAGE SHAPE IS GONE, AND ZERO IS THE ANSWER NOW
    //
    // This asserted that BOTH homes are used, which was true while a params
    // block was a `@group(0)` storage struct on some shaders and a `@group(1)`
    // uniform on others. The packed blocks came apart -- `RmsParams`,
    // `GegluStridedParams`, `GptOssSwiGluParams`, the moe router's and the
    // sort's -- and every one of them landed in the uniform, because a
    // descriptor for a handful of scalars buys nothing a uniform does not.
    // `GegluParams { unused: u32 }` did not land anywhere: it was a whole
    // binding for a field no body read.
    //
    // So `storage == 0` is the fact, and asserting it is worth more than
    // deleting the line. It is the trip-wire for the shape coming back: a
    // shader that reaches for a storage params block again fails HERE, with
    // the count, rather than at whichever fixture happens to bind it.
    assert_ne!(uniform, 0, "no rectangle carries a uniform block");
    assert_eq!(
        storage, 0,
        "{storage} rectangles carry a STORAGE parameter block. Every params \
         struct on this plane is a `@group(1)` uniform -- if one is a storage \
         binding again, that is an ABI this walk and every device fixture \
         beside it were written against the absence of."
    );
    // The offset comparison above is a prefix, so it is satisfied by a body
    // that passes NOTHING. This is what stops that from being a pass.
    assert_ne!(
        whole_struct, 0,
        "no rectangle's body fills its module's uniform struct exactly, so the \
         offset comparison above is only ever checking a prefix of one"
    );
    assert_eq!(
        module_alone + body_alone,
        6920,
        "the module-only placer was asked about a different number of launches"
    );
    // Non-zero on the side that MATTERS: if every launch's run could be placed
    // by the module alone, the body's argument list would be doing no work and
    // the walk above would be checking a copy.
    assert_ne!(
        body_alone, 0,
        "every launch's run fits its module's block unaided, so no body \
         interleaves, drops or derives a scalar and this walk is checking a \
         copy"
    );
    // ZERO, and stated as zero because it names what these texts do NOT reach.
    // A body passing `ArgValue::Usize` is the only shape where a field is eight
    // bytes wide and the packer has to align to eight -- `kv_append` and the
    // contiguous vector decodes, on a cache none of these texts configures. So
    // `routine::bind`'s eight-byte alignment is unwitnessed by any real plan
    // and its own unit tests are the whole of its cover. A deployment on a
    // contiguous cache moves this number.
    assert_eq!(
        eight_byte, 0,
        "a text now launches a body that passes an eight-byte scalar, so the \
         packer's wider alignment is reachable from a real plan and should be \
         checked here"
    );
}

/// Every launch of every real plan becomes a dispatch, or is a named refusal.
///
/// The other tests here take the binder and the parameter placer apart and ask
/// each its own question. This one puts them back together with the geometry
/// and asks the only question a driver actually has: given a plan, the modules
/// this crate can parse and an arena, how many of its rectangles can be
/// recorded?
///
/// All 6920, across six architectures in both fire classes, and nothing is
/// refused. The
/// Vulkan port reached that number by removing six symbols from its refusal
/// list one at a time, each removal a defect in that crate rather than a gap in
/// a plan; this backend was written after those findings and starts there.
///
/// The exact totals are asserted rather than a "most of them" threshold,
/// because the Vulkan port gave three separate reasons to insist on it: a walk
/// that matched kernel rows by string equality planned 432 of 3992 and reported
/// "no kernel row" for sixteen symbols that all exist; a walk that skipped a
/// module it could not open got its denominator wrong by 200; and a walk that
/// checked arity before placing the scalars refused 1439 rectangles across nine
/// symbols that dispatch perfectly well.
///
/// # RETIRED: the unstated-row fallback was exercised here, deliberately
///
/// It asserted that `binding::reorder`, handed a REAL operand-less row out of
/// the shipped table, produced for every one of these 6920 rectangles exactly
/// the slots `binding::bind` produces -- the plan's own args in the plan's own
/// order. That is `.wiki/new-driver/vulkan.md` §13's claim, and 56 of the
/// table's 100 rows and 292 of its entrypoints depended on it. The census
/// beside it pinned the operand-less rows at seven with seven entrypoints and
/// asserted every one stated `LaunchRule::Unstated`.
///
/// It went BLIND, not true. `kernels_wgpu::KERNELS` is empty: there is no
/// operand-less row to pick, `unstated[0]` would panic on an empty slice, and
/// the census counts zero rows in a table of zero rows -- which is not the
/// claim passing, it is the subject being gone. Nor is the claim reachable any
/// other way from here: `plan_one` consults `routine::armed` first and every
/// symbol is armed, so no real rectangle falls through to `plan_by_row` and
/// none reaches `reorder` at all -- and `reorder` has since been deleted with
/// the rest of the row path, so the claim has no subject on either plane. The
/// routine plane's nearest equivalent -- a body that asks for the statement's
/// operands in the order IT names them, which is not the same claim -- is
/// `Handles::input`/`output`, exercised by every arm and pinned by
/// `handles_are_minted_in_the_order_the_body_asks`.
#[test]
fn every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal() {
    let all = geometric();
    let mods = modules(all.iter().map(|(_, low, _)| low));

    let mut launches = 0u32;
    let mut planned = 0u32;
    let mut refused: BTreeMap<String, u32> = BTreeMap::new();
    let mut widest_grid = [0u32; 3];
    let mut workgroups = 0u64;
    let mut uniform = 0u32;
    let mut storage = 0u32;
    let mut refused_hollow = 0u32;
    let mut pool_numbers = 0u32;
    let mut numbers_seen: BTreeSet<FireNumber> = BTreeSet::new();
    let mut lost_sentinels: BTreeSet<String> = BTreeSet::new();
    let mut foreign: BTreeSet<String> = BTreeSet::new();
    let mut arena_bound = 0u32;
    // Every module any PASS names, memoised: `reflect::entrypoint` expands an
    // include tree and hands it to `naga`, and this walk asks once per
    // dispatch over seven thousand of them.
    let mut passed: BTreeMap<String, Declared> = BTreeMap::new();

    for (text, low, geometry) in &all {
        let buf = Placeholder(low.arena_bytes as u64);
        let store = Everything(Placeholder(GENEROUS));
        let sentinels = Sentinels(Placeholder(GENEROUS));
        let arena = Arena {
            buffer: &buf,
            bytes: low.arena_bytes as u64,
        };
        for launch in &low.launches {
            launches += 1;
            let symbol = &low.kernels[launch.kernel as usize];
            let declared = &mods[symbol];
            let module = driver_wgpu::geometry::Module::loaded(symbol, declared);
            let sources = Sources {
                arena,
                resolver: &store,
                min_offset: STRICTEST_ALIGNMENT,
            };

            // THE PLAN'S OWN OPERANDS, bound the plan's own way, kept beside
            // the planned rectangle so the walk below can tell an arena buffer
            // apart from a stand-in the driver supplied.
            //
            // `bind_arena_operands` runs the split the routine binder does in
            // real fire (see the helper's own comment): raises are consumed
            // by `views::raise` and never touch the arena, so this per-arg
            // walk is the same shape `binding::bind` used to be for a
            // pre-raises plan. Only the arena/named/weight halves are
            // handled here, which is what `plain` is compared against below:
            // a driver-bound RESOURCE whose buffer is the arena's is one of
            // the statement's own operands, at exactly the range the plan
            // placed it, and a raise cannot be one of those on this plane.
            let (plain_bounds, _raises, plain_refused) = bind_arena_operands(
                low, launch, arena, &store, STRICTEST_ALIGNMENT,
            );
            assert!(
                plain_refused.is_empty(),
                "{text}: the statement's operands bind against the arena that \
                 holds them: {plain_refused:?}"
            );
            let plain: Vec<_> = plain_bounds.into_iter().map(|(_, b)| b).collect();

            // `plan_all`, AND EVERY PASS WALKED. A rectangle is one statement
            // and may be more than one dispatch -- the split decode cuts a
            // key range into slices and merges them -- so `plan_one` refused
            // 152 of this corpus's 6920 launches by name and the walk counted
            // the refusal as though the statement had no plan.
            //
            // `planned` still counts LAUNCHES, which is what makes it
            // comparable with `launches` above; the per-pass work below runs
            // once per dispatch. Each pass is held against ITS OWN module,
            // because a split pass and a merge pass declare different binding
            // sets and neither is the statement symbol's.
            match driver_wgpu::dispatch::plan_all(
                low,
                launch,
                Built { module, declared },
                sources,
                *geometry,
            ) {
                Ok(passes) => {
                    planned += 1;
                    for (pass, d) in passes.iter().enumerate() {
                        let declared = &passed
                            .entry(d.symbol.to_owned())
                            .or_insert_with(|| {
                                driver_wgpu::reflect::entrypoint(d.symbol, Capability::Baseline)
                                    .unwrap_or_else(|why| {
                                        panic!("no module for `{}`: {why}", d.symbol)
                                    })
                            })
                            .clone();
                        // A grid with a zero in it runs nothing and reports
                        // success, which is the failure this crate refuses
                        // hardest. `plan_one` is supposed to have caught it.
                        assert!(
                            !d.groups.contains(&0),
                            "{text}: `{symbol}` planned a grid of {:?}",
                            d.groups
                        );
                        for (widest, got) in widest_grid.iter_mut().zip(d.groups) {
                            *widest = (*widest).max(got);
                        }
                        workgroups += u64::from(d.groups[0])
                            * u64::from(d.groups[1])
                            * u64::from(d.groups[2]);

                        // The same launch against a geometry of ZEROS, which is the
                        // nastiest input a caller can hand this seam. None gets
                        // through, and the measured reason is not `plan_one`'s
                        // empty-grid check but the row overrides: a head-shaped row
                        // STATES its head width and head count, so the zeros are
                        // replaced before a rule sees them, and the rows that do
                        // not state them refuse geometrically instead. So this
                        // block does not witness that check; it witnesses that no
                        // real rectangle can be driven to an empty grid from
                        // outside.
                        if let Some(hollow) = driver_wgpu::dispatch::plan_all(
                            low,
                            launch,
                            Built { module, declared },
                            sources,
                            Geometry::default(),
                        )
                        .ok()
                        .and_then(|list| list.into_iter().nth(pass))
                        {
                            assert!(
                                !hollow.groups.contains(&0),
                                "{text}: `{symbol}` planned {:?} from a geometry of zeros",
                                hollow.groups
                            );
                        } else {
                            refused_hollow += 1;
                        }

                        // AND THE SEAM MOVED, so the hollow case has to move with
                        // it. A head width used to reach a body through
                        // `Facts::head_dim`, and a geometry of zeros was therefore
                        // the way to hand one nothing. It is a `Const<i32>` now --
                        // the checkpoint's, which the STATEMENT carries -- so the
                        // fire's zeros no longer reach it and the block above stops
                        // witnessing anything for that family.
                        //
                        // What a caller can still hand it nothing THROUGH is the
                        // params run, so that is zeroed here. A body that divides
                        // by its stated width, or sizes a group by it, must refuse
                        // rather than dispatch an empty grid.
                        let mut hollowed = low.clone();
                        hollowed.params = vec![0; low.params.len()];
                        if let Some(hollow) = driver_wgpu::dispatch::plan_all(
                            &hollowed,
                            launch,
                            Built { module, declared },
                            sources,
                            Geometry::default(),
                        )
                        .ok()
                        .and_then(|list| list.into_iter().nth(pass))
                        {
                            assert!(
                                !hollow.groups.contains(&0),
                                "{text}: `{symbol}` planned {:?} from a statement \
                             of zeros",
                                hollow.groups
                            );
                        } else {
                            refused_hollow += 1;
                        }

                        // THE POOL'S NUMBERS REACH THE SHADER. A body may pass
                        // a number that belongs to the POOL rather than to the
                        // statement -- the KV page size, the cache's two strides,
                        // the attention mask's -- and `Handles::fire_number` is
                        // where an arm asks for one. The walk's own resolver
                        // answers `None` to every number, so all of them read as
                        // zero in `d`; the SAME rectangle planned against a
                        // resolver with recognisable answers is what says which
                        // ones arrived and where.
                        //
                        // A wrong stride is not an error. It is attention reading
                        // the wrong offsets and returning numbers.
                        let Some(told) = driver_wgpu::dispatch::plan_all(
                            low,
                            launch,
                            Built { module, declared },
                            Sources {
                                arena,
                                resolver: &sentinels,
                                min_offset: STRICTEST_ALIGNMENT,
                            },
                            *geometry,
                        )
                        .ok()
                        .and_then(|list| list.into_iter().nth(pass)) else {
                            lost_sentinels.insert(format!(
                                "{symbol}: plans against a resolver that answers nothing \
                             and refuses one that answers everything"
                            ));
                            continue;
                        };
                        for which in [
                            FireNumber::KvPageSize,
                            FireNumber::KvHeadStride,
                            FireNumber::KvSeqStride,
                            FireNumber::AttentionMaskStride,
                        ] {
                            if !carries(&told.params, sentinel(which)) {
                                continue;
                            }
                            pool_numbers += 1;
                            numbers_seen.insert(which);
                            // The word is the POOL's and nothing in the statement
                            // carries it, so it cannot have come from the run: the
                            // same rectangle planned against a resolver that
                            // answers nothing has a zero where this word is. That
                            // is what makes the hit above evidence of a resolver
                            // call rather than of a coincidence in the scalars.
                            assert!(
                                !carries(&d.params, sentinel(which)),
                                "{text}: `{symbol}` hands the shader {which:?}'s sentinel \
                             even when the resolver answers nothing, so the word did \
                             not come from the pool"
                            );
                        }

                        match d.params {
                            Params::Block {
                                at: ParamSlot::Uniform,
                                ref bytes,
                            } => {
                                uniform += 1;
                                // A uniform block takes NO place in the `@group(0)`
                                // list -- it is a bind group of its own -- so a
                                // dispatch that recorded one here would shift every
                                // storage entry after it, and `wgpu` would accept
                                // the set if the kinds happened to line up.
                                assert_eq!(
                                    d.block_at, None,
                                    "{text}: `{symbol}` puts its uniform block in the \
                                 storage group"
                                );
                                assert_eq!(bytes.len(), declared.uniform_bytes as usize);
                            }
                            Params::Block {
                                at: ParamSlot::Storage(at),
                                ref bytes,
                            } => {
                                storage += 1;
                                assert!(!bytes.is_empty());
                                assert!(
                                    at < declared.bindings,
                                    "{text}: `{symbol}` puts its block at {at} of {}",
                                    declared.bindings
                                );
                                // The DENSE index, which is what a bind group is
                                // written from: a slot the body leaves empty takes
                                // no entry. They agree on every module in this
                                // tree, and this is where that is measured rather
                                // than assumed.
                                assert_eq!(
                                    d.block_at,
                                    Some(at as usize),
                                    "{text}: `{symbol}`'s block is binding {at} and \
                                 entry {:?} of the group",
                                    d.block_at
                                );
                            }
                            Params::None => {}
                        }

                        // The operands, PLUS the slot a parameter struct takes,
                        // PLUS the slots the body leaves empty, are the module's
                        // whole binding set. `driver-vulkan` asserts the same sum
                        // with its holes SUBTRACTED, and that is a Vulkan
                        // coincidence rather than a rule: there a hole is an
                        // undeclared number, here it is a declared binding this
                        // entry point does not read, and `wgpu` validates a group
                        // against its layout entry for entry either way.
                        let block = usize::from(d.block_at.is_some());
                        let empty = declared.holes();
                        assert_eq!(
                            d.buffers.len() + block + empty,
                            declared.bindings as usize,
                            "{text}: `{symbol}` bound {} plus {block} plus {empty} for \
                         {} bindings",
                            d.buffers.len(),
                            declared.bindings
                        );

                        // AND EVERY ARENA BUFFER IT BOUND IS ONE OF THIS LAUNCH'S
                        // OWN OPERANDS, at exactly the range `binding::bind`
                        // places it.
                        //
                        // The claim the retired fallback used to make, moved to
                        // the plane the operands now travel on. An arm asks for
                        // its operands by position -- `Handles::input(0)`,
                        // `output(0)` -- and a body then orders them for its
                        // shader; a mis-indexed ask binds a real tensor of the
                        // right length to the wrong slot, and no shape check
                        // anywhere notices, because these are storage buffers of
                        // matching extent.
                        //
                        // Told apart from the driver's own resources by BUFFER:
                        // the arena is `Placeholder(low.arena_bytes)` and every
                        // stand-in is `Placeholder(GENEROUS)`, and `Placeholder`
                        // compares by value. So a weight, a fire table or the KV
                        // cache is not this question and is skipped, and what is
                        // left is the statement's own activations.
                        for bound in &d.buffers {
                            if *bound.buffer() != buf {
                                continue;
                            }
                            arena_bound += 1;
                            if !plain.contains(bound) {
                                foreign.insert(format!(
                                    "{symbol}: binds {} bytes at {} of the arena, which is \
                                 not one of the {} operands the statement carries",
                                    bound.len(),
                                    bound.offset(),
                                    plain.len()
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    *refused.entry(format!("{symbol}: {e}")).or_default() += 1;
                }
            }
        }
    }

    assert_eq!(
        launches, 6920,
        "a different number of rectangles is lowered"
    );
    assert_eq!(planned, 6920, "a different number of rectangles records");
    assert!(
        lost_sentinels.is_empty(),
        "{} kernels plan against a resolver that answers nothing and refuse one \
         that answers everything:\n  {}",
        lost_sentinels.len(),
        lost_sentinels
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    assert!(
        foreign.is_empty(),
        "{} kernels bind a range of the arena that is not one of the \
         statement's operands:\n  {}",
        foreign.len(),
        foreign.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );
    // The denominator for the line above, so that "nothing foreign" cannot be
    // true because nothing was looked at.
    //
    // NOT `operands`, and the difference is the reason this is a floor rather
    // than the equality the rest of this file prefers. A rectangle may bind
    // one operand twice -- `Vector::head` passes the same cache plane's
    // strides twice, and a body may read an operand it also writes -- while an
    // `in_place` pair binds ONE buffer for two of the plan's args and an
    // `unbound` slot binds none. So the plan's 16540 arena operands and the
    // ranges a body binds are two different counts, and only the direction is
    // a rule: every planned rectangle carries at least one range of the arena,
    // because a statement that touched no activation would not be a statement.
    //
    // The exact number is printed rather than pinned, because this file cannot
    // derive it from anything it already asserts and a literal nobody can
    // re-derive is a literal that gets updated without being read.
    assert!(
        arena_bound >= planned,
        "{arena_bound} arena ranges were bound across {planned} rectangles, so \
         some rectangle reached a shader without touching the arena at all"
    );
    eprintln!("{arena_bound} arena ranges bound across {planned} rectangles");

    // Nothing is refused, so nothing is named. Every rectangle all six
    // architectures state, in both fire classes, becomes a dispatch.
    let expected: Vec<String> = Vec::new();
    let got: Vec<String> = refused.keys().cloned().collect();
    assert_eq!(got, expected, "a different set of rectangles refuses");
    assert_eq!(
        refused.values().sum::<u32>(),
        launches - planned,
        "the refusals and the successes do not account for every rectangle"
    );

    // The parameter home that exists is exercised, and the one that does not
    // is asserted absent. See the same pair in
    // `every_launchs_scalars_land_where_its_module_reads_them`: every packed
    // block on this plane became a `@group(1)` uniform, so a storage params
    // binding is not a smaller half of the split, it is a shape no shader has.
    assert_ne!(uniform, 0, "no dispatch takes a uniform block");
    assert_eq!(
        storage, 0,
        "{storage} dispatches take a STORAGE parameter block, and no shader on \
         this plane declares one"
    );
    assert!(
        uniform + storage <= planned,
        "more dispatches carry a block than were planned"
    );
    // RETIRED: six census counters over the table's geometry columns.
    //
    // `overridden`, `head_overridden`, `heads_overridden`, `rotary_overridden`,
    // `split_rectangles` and `derived_widths` counted rectangles whose ROW
    // named a `grid_param`, `head_param`, `heads_param`, a `LaunchRule::Rope`
    // rotary width, `LaunchRule::SplitPacked` or a `Source::OutWidth`, and
    // asserted 288 / 352 / 352 / 0 / 0 / 0. Beside them, `dims_of` was asked
    // the same rectangle against a deliberately LYING `Geometry` -- head width
    // plus seven, head count plus seven, rotary width plus 1024 -- and had to
    // answer with the row's stated word rather than the fire's, which is what
    // caught an override that was reading the fire on one rule and the row on
    // another.
    //
    // They went BLIND, not true. `kernels_wgpu::KERNELS` is empty, so
    // `sig.grid_param` and its four siblings do not exist to read and
    // `dims_of` has no row to answer from: the counters would sum zero over an
    // empty table and the liar would never be asked anything. The rule is not
    // gone -- a statement still names its own extent, its head shape and its
    // rotary width, and a body reads those same words through
    // `Handles::stated` -- but it is no longer expressible as a column census,
    // because there is no column. `crates/kernels-wgpu/tests/routines.rs` is
    // where a body's reading of the statement's run is held, and `dims_of`'s
    // own unit tests in `src/dispatch.rs` -- which synthesize the row they
    // read -- are the whole of what still exercises the override itself.
    //
    // What is lost, stated plainly: nothing in this file now witnesses the
    // extent a statement names travelling from `Launch::params` to a grid over
    // REAL plans. The grid TOTAL below is the remaining cover, and it is a
    // checksum rather than a derivation.

    // How many of these launches `plan_one` REFUSED when handed a geometry of
    // zeros -- the head-shaped ones. The rest plan fine because their
    // dimensions come from the lowering rather than from the geometry, and they
    // are right to.
    //
    // Stated as a non-zero rather than as a literal: the count is a property
    // of which arms's grids come from a param the statement zeroes and moves
    // when a family is armed, but a ZERO would mean no rectangle anywhere
    // refuses a hollow fire, which is the finding this block exists to make.
    //
    // The upper bound this line used to carry -- `refused_hollow < planned` --
    // was witnessing that "not every launch reads head_dim from the geometry,
    // so a hollow one still plans through". The marks migration retired that
    // seam: `head_dim`, `rows`, the strides are `Const<i32>` marks now, so
    // when the statement zeros them EVERY grid-forming param goes to zero and
    // the plan refuses everywhere. Which is the whole point of moving the
    // number to a Const -- a caller can hand it nothing THROUGH the params
    // and every arm refuses. So `arena_bound` above (15212 real ranges across
    // 6920 rectangles) is the evidence the plans were reached at all; this
    // count says how many of those reached rectangles guard themselves
    // against a zeroed statement, and the migration has raised it to
    // "essentially all of them".
    assert_ne!(
        refused_hollow, 0,
        "no launch refused a geometry of zeros, so nothing here holds the seam \
         against a caller that hands it nothing"
    );
    // WHICH of the pool's numbers these texts actually carry to a shader.
    //
    // The page size and the attention mask's stride, and NOT the cache's two
    // strides: these texts are paged throughout and the strides belong to the
    // contiguous cache. That is why the loop above cannot be the whole check,
    // and why the arms themselves are swept separately in
    // `every_arm_naming_a_pool_number_is_handed_that_number_and_not_another`.
    assert_eq!(
        numbers_seen.iter().copied().collect::<Vec<_>>(),
        vec![FireNumber::KvPageSize, FireNumber::AttentionMaskStride],
        "a different set of the pool's numbers reaches a shader from these texts"
    );
    assert_ne!(
        pool_numbers, 0,
        "no rectangle carries one of the pool's numbers, so the resolver's \
         answers are not reaching a shader at all"
    );

    // The total work these plans dispatch, as a single number.
    //
    // Here because every other assertion in this test is about SHAPE -- how
    // many operands, where the scalars went, that no grid is zero -- and a grid
    // can be the wrong size while being all of those things. A body that read
    // the fire's row count where it should read the statement's changes no
    // other assertion in this file and changes this one.
    //
    // It is the number the ROUTINE plane produces, and it is the same number
    // the table plane produced: `bind` divides the body's own lanes by the
    // module's `@workgroup_size`, and while both planes were live the
    // twice-derived control in this file compared them rectangle for
    // rectangle -- see its retirement below, which is why it is described here
    // rather than named. The history below is therefore still the history of
    // this number, read off a different derivation of it.
    //
    // It is NOT the 35,473,250 `driver-vulkan` pins over the same twelve plans,
    // and the difference is a real one rather than a rounding: these grids are
    // divided by the WGSL's own `@workgroup_size`, and this tree's is not
    // always the GLSL's. `rope/neox.wgsl` is `@workgroup_size(1)` -- one
    // invocation owning one rotary pair -- and it alone dispatches 22,663,680
    // of these workgroups, which is why the two totals cannot be compared and
    // why this one is measured from the modules rather than transcribed.
    // 48,945,410 before the shared MoE rows started stating their extents,
    // and the +342,024 is entirely `route_gather`: `rows_param = Some(4)` puts
    // its rows on the SORTED STACK, so it now covers all of its own output
    // instead of a quarter of it at `top_k = 4`.
    //
    // It went the other way first. Reading `grid_param` without also reading
    // it in `Rule::RoutedQmv` -- which took the rectangle's `width`, `k` times
    // one result -- put this at 147,555,722, three times over. Both readings
    // are the row's, and this number is what says the pair is consistent.
    assert_eq!(
        // DOWN, and down is the news here. The tile moved from `bm_16` to
        // `bm_32`, so a prefill covers its rows in half as many row tiles and
        // the GEMM asks for fewer workgroups. A count that had gone UP after a
        // tile grew would be the thing worth stopping for.
        //
        // Down again by 16,640 -- 260 workgroups each for the 64 gemma-4
        // `residual_add` rectangles the folding norm replaced. Work that is
        // not dispatched because it was folded into its neighbour.
        //
        // And by 502,680 more when the routed tile's column axis went from 32
        // to 64: half as many column tiles over the same rows. Down after a
        // tile GREW, which is the direction that says the sweep found a real
        // point rather than a wider dispatch.
        //
        // Down again by 318,432 when upstream STATED `sdpa_paged_tiled` and
        // its sink, and the reason is worth having: while the row was
        // unstated the driver launched it on the TRACE's grid, which puts
        // ROWS on y. The stated `LaunchRule::SdpaTiled` puts TILES there, and
        // the shader's tiled arm covers 32 rows per group -- so the old grid
        // was launching thirty-two times the groups it needed and relying on
        // `row >= n_rows` to make the surplus do nothing. Wasteful, not wrong,
        // and now neither: 176 rectangles at up to `q_heads * 62` groups each
        // is the whole difference.
        // Down by 14,009,424 -- by far the largest single move this number has
        // made -- when upstream added `Qwen35MetalFacts::moe_tile`, the opt-in
        // that lets a MoE prefill take a tiled GEMM over the expert banks
        // instead of one matvec per row. That is the whole point of a tile, so
        // the direction is the one to want: the two new symbols
        // (`affine_qmm_t_routed_...` and `mxfp4_qmm_t_routed_bias_...`) do
        // the same arithmetic in 29% fewer workgroups.
        //
        // Down again by 502,680 when `ROUTED_QMM_TILE` went from (32, 32)
        // to (32, 64) in `7bda96b66`: half as many column tiles over the
        // same expert widths. That commit swept the axis rather than
        // arguing it -- gemma-4-26b-a4b 433.4 tok/s at `bn = 32` against
        // 436.7 at 64 -- and the two routed rows in `REACHES` above are the
        // SAME two symbols at the new tile, which is why the count there
        // did not move either.
        // UP BY 1,395,064 WITH THE MARKS MIGRATION, and up is the direction to
        // read carefully. It is the strided norms: `rms_strided_row` and
        // `rms_strided_head_row` sized their threadgroup from `x.width`, which
        // is the PITCH across several norms and not one norm's axis. Their
        // axis is `params[1]` and a `Const<i32>` now, so the group is sized by
        // the norm rather than by the whole strided row -- more groups, each
        // covering what it should, where before one oversized group covered a
        // row and masked the surplus. Wasteful became right, and right costs
        // more workgroups.
        //
        // And by 177,320 more when `rms_single_row` joined its strided twin:
        // its axis is `params[1]` too, and it had been reading `x.width`. A
        // row that holds one norm makes the two the same number, and one that
        // holds several does not -- which is exactly the case the old reading
        // sized wrong.
        //
        // DOWN BY 4,740,096 when `quant/qmv.wgsl` grew `PIE_MT`. A reducing
        // matvec workgroup owns eight output columns and reads their weights
        // over the whole of K; giving it FOUR activation rows instead of one
        // divides the workgroup count on x by four and the weight traffic with
        // it. The arithmetic is the same arithmetic -- the saving is that the
        // packed matrix is read once per four tokens rather than once per
        // token, which for a 512-token prefill of a 1B lm head was 67 GiB of
        // it. `geometry::lanes(Rule::Qmv)` and `quant::qmv_grid`'s callers
        // both state the quartering, and they have to agree.
        //
        // DOWN BY 22,253,400 -- the largest move this number has made, and it
        // is one file. `rope/neox.wgsl` was `@workgroup_size(1)` because it
        // read the rotated pair count off `num_workgroups.x`, where the count
        // is a STRIDE and has to be exact: widen the group and the host rounds
        // that count up and every pair rotates against the wrong partner. The
        // file's header had weighed the price and recorded the way out without
        // taking it, and the price was that this ONE shader dispatched
        // twenty-two and a half million of the thirty-three million counted
        // here.
        //
        // `rotary` is a field of all three of its uniform blocks now, so
        // `pairs` comes off the block and the x axis is free. `rope_grid`
        // divides it twice: by two, because an invocation owns a four-byte
        // word and so covers two pairs -- the old grid launched a workgroup
        // per pair and half of them returned at the guard -- and then by the
        // 32-wide group. Sixty-four times fewer, and the `i0 >= pairs` guard
        // was already covering both round-ups.
        //
        // Measured on an M4 while the change was made: a 512-row prefill of
        // llama-3.2-1B went from 262,144 one-thread workgroups per rotation to
        // 8,192 of thirty-two lanes, pp512 850 to 897 tok/s. This number is
        // the reason that was worth doing and the record that it was done.
        //
        // UP BY 2,095,584 at `d0fb52657`, and this is the rare rise that is a
        // gain. The quantized matvec gives a workgroup FOUR output columns
        // where it gave eight, so the column axis needs twice the groups over
        // the same weights. Fewer columns a group is what let the scale and
        // the bias hoist to once per lane-block, which is only sound while a
        // block cannot straddle two quantization groups -- and the narrowest
        // group this tree stamps is 32, so eight words of four-bit codes would
        // read one scale for two groups' weights. Wrong, not slow.
        //
        // NOT `2ef735054` beside it, which doubled `PIE_QMV_VPT` to four words
        // a lane. That halves the K loop's trip count inside a group and
        // changes no grid, so this number is blind to it -- which is the right
        // shape: this counts dispatched work, not work per lane.
        //
        // UP BY 3,188,864 WHEN THE PAGED DECODE SPLIT, and this rise is the
        // point of the change rather than a cost of it. `sdpa_paged_decode`
        // is two dispatches now: `_split_` cuts a row's key range into eight
        // slices so that a fire with more query heads than the GPU has cores
        // still has workgroups to hand it, and `_merge_` folds the eight
        // partials back. Eight times the attention groups plus a cheap merge
        // pass, over 152 rectangles.
        //
        // This total counts DISPATCHED WORKGROUPS and not work, and the
        // distinction is the whole reason a split is worth making: each of
        // the eight covers an eighth of the key range, so the arithmetic is
        // the same arithmetic spread over more groups. A number that measured
        // work would not have moved at all.
        workgroups,
        16_182_066,
        "the plans dispatch a different amount of work"
    );
    // The third dimension is a prefill's rows, and only the paged decodes put
    // anything there. A backend that flattened the grid to two dimensions would
    // have looked correct on every other rectangle.
    assert_eq!(
        // x went 3584 -> 14040 with `moe_tile`, and the widest is NOT the new
        // GEMM. It is `gptoss_swiglu_bfloat16` and `silu_mul_bfloat16`, the
        // elementwise kernels over the ROUTED STACK -- and the stack is what
        // grew: enabling the tile makes `route_sort` pad every expert's rows
        // up to `tile_rows`, so qwen3-30b-a3b's 64-token prefill sorts into
        // 4480 rows (140 tiles of 32) where it packed 512 before.
        //
        // `13440 = 4480 * 768 / 256` to the element, which is what says this
        // is a grid over real rows and not a runaway. Padding is the price of
        // tiling and it is paid here; the tile still wins, which is why the
        // total workgroup count above FELL by fourteen million in the same
        // change.
        //
        // y DOUBLED 25,136 -> 50,272 at `d0fb52657`, from the same halving of
        // the matvec's output columns per workgroup that put 2,095,584 on the
        // total above. It is llama-3.2-1B's lm head: 128,256 columns over four
        // a group is 32,064 -- not this -- so the widest is the 4-bit qmv on
        // gpt-oss's 201,088-column vocabulary, which at four columns a group
        // and the `_wide_` row's own division lands exactly twice where it
        // landed at eight. An axis that doubles when the divisor halves is the
        // arithmetic agreeing with itself; an axis that did anything else here
        // would be the news.
        widest_grid,
        [14040, 50272, 64],
        "the widest grid in any dimension changed"
    );
}

/// Every arm that names one of the pool's numbers is handed that number, and
/// the four are not interchangeable.
///
/// The walk above can only ask this of arms its twelve lowerings reach, and
/// they reach exactly two of the four numbers: the page size and the attention
/// mask's stride, because these texts are paged throughout and the cache's two
/// strides belong to the CONTIGUOUS pool. So the strides go unwatched there --
/// replacing either with a constant leaves that walk green, and a wrong stride
/// is not an error but attention reading the wrong offsets and returning
/// numbers.
///
/// `Pool`'s answers are checked in `resources.rs` and the shader's addressing
/// in `tests/device.rs`, which hand-writes its parameter bytes. Between those
/// two the seam is open. This closes it from the ARM REGISTRY's side rather
/// than a text's, so a kernel armed tomorrow is covered whether or not a text
/// reaches it.
///
/// # Re-anchored
///
/// This used to sweep `kernels_wgpu::KERNELS` for rows naming
/// `Source::KvPageSize`, `KvHeadStride`, `KvSeqStride`, `AttentionMaskStride`
/// or `Rows`, and put each through `binding::scalars` against a synthesized
/// launch and a synthesized module, comparing the whole run word for word
/// against `expected_run`. The table is empty and none of that has a subject.
///
/// `Handles::fire_number` is where an arm asks the fire for one of these
/// numbers now, so the sweep is over the ARMS: every entrypoint
/// `kernels-wgpu` ships, run twice against a generous synthetic statement --
/// once with a pool that states NO page size, which is a contiguous cache, and
/// once with a pool that states one, which is a paged cache. What an arm hands
/// its body is a `Vec<ArgValue>` and the sentinels are recognisable in it, so
/// the two runs together say which numbers each arm asked for and which it was
/// given.
///
/// `Source::Rows` has no counterpart here and is not swept: the fire's row
/// count reaches a body through `Facts::rows`, which is not a resolver call
/// and cannot be answered by the wrong pool.
#[test]
fn every_arm_naming_a_pool_number_is_handed_that_number_and_not_another() {
    // A statement generous enough that an arm does not refuse for want of an
    // operand or a scalar. The question here is which NUMBERS an arm asks the
    // fire for, and an arm that never runs answers nothing.
    //
    // Built per-symbol just below rather than once: the marks migration
    // added `Ty::Raised` at specific input slots on the attention family
    // (`In<Struct<KvCache>>`, `In<Struct<AttentionMask>>`,
    // `In<Struct<AttnSplit>>`), and those slots must carry `Arg::Raised`
    // with the specific key `views::raise` matches on -- an `Arg::Arena`
    // there refuses `Unstated`. A per-symbol shape lets the sdpa arms bind
    // through the routine binder without breaking every other kernel that
    // reads slot 1 as a buffer. The raise slots below are read off each
    // routine's `#[routine]` signature in `crates/kernels-wgpu/src/attn.rs`
    // and would move if the signature moves; the mapping is stated here
    // rather than derived because `Ty::Raised` carries no key -- the key
    // rides through the `raise!` macro to the `Handles::raised_key` lookup,
    // so the fixture and the signature are two facts that have to agree.
    let arg_shape = |symbol: &str| -> Vec<Arg> {
        let raises: &[(usize, &str)] = if symbol.starts_with("sdpa_paged_decode") {
            // `queries`, `kvc`, `positions`, `request_of_token`, `maskv`,
            // `split`: six inputs, three raised. `_sink` shares the same
            // input surface -- the sink table is a `Const<Tensor<..>>`.
            &[(1, "kv_cache"), (4, "attention_mask"), (5, "attn.split_policy")]
        } else if symbol.starts_with("sdpa_vector_decode") {
            &[(1, "kv_cache")]
        } else if symbol.starts_with("kv_append") {
            // `k_new`, `v_new`, `kvc`, `positions`: four inputs, one raised.
            // The write half of the same paged view the decodes read.
            &[(2, "kv_cache")]
        } else {
            &[]
        };
        (0..24u32)
            .map(|i| {
                if let Some((_, key)) = raises.iter().find(|(at, _)| *at == i as usize) {
                    Arg::Raised {
                        value: 0,
                        key: (*key).to_owned(),
                    }
                } else {
                    Arg::Arena {
                        at: i as usize * 4096,
                        width: 512,
                        bytes: 2,
                    }
                }
            })
            .chain((0..8).map(|i| Arg::Weight(format!("layer.0.w{i}"))))
            .collect()
    };
    // A RUN EVERY ARMED BODY CAN READ. It was `1000 + i` -- distinct
    // sentinels, which was right while the widths came from the FIRE, and
    // wrong the moment a head width became a `Const<i32>` the statement
    // carries: no point is compiled at a 1003-wide head, so every attention
    // body refused and this sweep ran nothing. 64 is a width all four decode
    // families instantiate, and a positive window admits the sliding pair.
    let scalars: Vec<u32> = vec![64; 24];
    // A plausible fire, and every field non-zero: an arm dividing by a head
    // count would be measuring this file's laziness rather than the driver.
    let geometry = Geometry {
        q_heads: 8,
        kv_heads: 4,
        head_dim: 64,
        rotary_dims: 64,
        n_experts: 8,
        experts_per_token: 4,
        ..Default::default()
    };
    // A CONTIGUOUS pool states no page size, which is how `contiguous_pool`
    // tells the two apart. Both maps answer every other number, so a
    // difference between the two runs is about the page size and nothing else.
    let contiguous: BTreeMap<FireNumber, u32> = [
        FireNumber::KvHeadStride,
        FireNumber::KvSeqStride,
        FireNumber::AttentionMaskStride,
    ]
    .into_iter()
    .map(|which| (which, sentinel(which)))
    .collect();
    let mut paged = contiguous.clone();
    paged.insert(FireNumber::KvPageSize, sentinel(FireNumber::KvPageSize));

    let mut strided: BTreeSet<&'static str> = BTreeSet::new();
    let mut page_sized: BTreeSet<&'static str> = BTreeSet::new();
    let mut masked: BTreeSet<&'static str> = BTreeSet::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut ran = 0u32;

    for symbol in kernels_wgpu::entrypoints() {
        let Some(routine) = driver_wgpu::lowering::routine::armed(&symbol) else {
            continue;
        };
        let results = driver_wgpu::lowering::routine::results(routine);
        let facts = driver_wgpu::lowering::hold::facts(&symbol, 1, geometry, 1, 512, 512);
        // THE WIDTH THIS SYMBOL IS COMPILED FOR, read off its own name. One
        // flat run cannot serve the decode families: `sdpa_vector_decode_sink`
        // exists at `_d_64` alone and `sdpa_vector_decode_swa` at `_d_256` and
        // `_d_512`, so any single number refuses one of them -- and a refusal
        // here is a body this sweep never ran.
        let scalars: Vec<u32> = match symbol
            .rsplit_once("_d_")
            .and_then(|(_, d)| d.split('_').next().and_then(|d| d.parse::<u32>().ok()))
        {
            // AT THE SLOT EACH FAMILY READS IT FROM, which is not the same
            // slot. `sdpa_vector_decode` takes `[scale, head_dim, q_heads]`;
            // the `swa` and `sink` forms put a WINDOW between the scale and
            // the width, so theirs is at 2. A run of one number everywhere
            // would state a window of 256 and a q-head count to match.
            Some(d) => {
                let at = usize::from(symbol.contains("_swa") || symbol.contains("_sink")) + 1;
                let mut run = scalars.clone();
                run[at] = d;
                // AND THE HEAD COUNT THAT GOES WITH IT. The query row this
                // file synthesizes is 512 wide and a row is heads laid end to
                // end, so a width of `d` means `512 / d` of them -- state one
                // and not the other and the body refuses a row that does not
                // divide.
                run[at + 1] = (512 / d).max(1);
                run
            }
            None => scalars.clone(),
        };
        let args = arg_shape(&symbol);
        let mut open = driver_wgpu::lowering::hold::Handles::with_numbers(
            &args,
            results,
            &scalars,
            &contiguous,
        );
        // THROUGH THE BODY, NOT THE COLUMN. The cache's strides are
        // `ctx.ask::<_, keys::KvHeadStride>()` calls now -- a fact only the
        // fire can answer left the parameter list, which is the whole of the
        // marks migration -- so binding the column alone never sees them and
        // this sweep witnessed nothing at all. `Stated::scalars` is what the
        // body PASSED, in its own order, which is the order a swap moves.
        let fired_scalars = |taken: Vec<ArgValue>,
                             open: driver_wgpu::lowering::hold::Handles<'_>|
         -> Vec<ArgValue> {
            let cell = core::cell::RefCell::new(open);
            driver_wgpu::lowering::routine::stating(routine, &taken, &cell, facts)
                .ok()
                .map(|stated| {
                    stated
                        .iter()
                        .flat_map(|one| one.scalars.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or(taken)
        };
        // Each binder pass takes its own `Views`. `Ty::Raised` operands
        // build a boxed host view whose address lives in the returned
        // `ArgValue::Raised`; the two passes below are independent binds
        // over different `Handles`, so they mint independent views too.
        let mut free_views = driver_wgpu::lowering::views::Views::default();
        let free =
            driver_wgpu::lowering::bind::bind(routine.args, routine.sources, &mut open, facts, &mut free_views)
                .ok()
                .map(|taken| fired_scalars(taken, open));
        let mut walled =
            driver_wgpu::lowering::hold::Handles::with_numbers(&args, results, &scalars, &paged);
        let mut walled_views = driver_wgpu::lowering::views::Views::default();
        let held =
            driver_wgpu::lowering::bind::bind(routine.args, routine.sources, &mut walled, facts, &mut walled_views)
                .ok()
                .map(|taken| fired_scalars(taken, walled));
        if free.is_none() && held.is_none() {
            continue;
        }
        ran += 1;

        // ON A CONTIGUOUS POOL: an arm handed one of the cache's two strides
        // is handed BOTH, and the head stride comes first.
        //
        // Order, not presence, because presence cannot see a swap: both
        // strides are in the run either way, and swapping them is the defect
        // with the most plausible output -- attention striding by heads where
        // it should stride by positions, reading real numbers from the wrong
        // rows. The two sentinels are distinct, so their POSITIONS in what the
        // arm hands its body are what a swap moves.
        if let Some(took) = &free {
            let head = position(took, sentinel(FireNumber::KvHeadStride));
            let seq = position(took, sentinel(FireNumber::KvSeqStride));
            match (head, seq) {
                (Some(h), Some(q)) => {
                    strided.insert(routine.name);
                    if h >= q {
                        wrong.push(format!(
                            "`{}` hands its body the sequence stride at {q} and the \
                             head stride at {h}, which is the wrong way round",
                            routine.name
                        ));
                    }
                }
                (Some(_), None) | (None, Some(_)) => wrong.push(format!(
                    "`{}` is handed one of the cache's two strides and not the other",
                    routine.name
                )),
                (None, None) => {}
            }
        }

        // ON A PAGED POOL: no arm is handed either stride, whether or not it
        // asked for one.
        //
        // The claim `contiguous_pool` exists to make, and the one the table
        // path made by refusing the row outright. Both strides mean "walk the
        // cache with no page table", and this driver's pool is
        // `[page, token, head, dim]`: handing them over makes the launch
        // succeed against the wrong tokens. An arm that reads them must REFUSE
        // here rather than answer, and an arm that does not read them must not
        // acquire one.
        if let Some(took) = &held {
            for which in [FireNumber::KvHeadStride, FireNumber::KvSeqStride] {
                if position(took, sentinel(which)).is_some() {
                    wrong.push(format!(
                        "`{}` is handed {which:?} over a pool that states a page \
                         size, so it walks the cache with no page table",
                        routine.name
                    ));
                }
            }
            if position(took, sentinel(FireNumber::KvPageSize)).is_some() {
                page_sized.insert(routine.name);
            }
            if position(took, sentinel(FireNumber::AttentionMaskStride)).is_some() {
                masked.insert(routine.name);
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "{} arms are handed a pool number they did not name:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
    // The denominator. A registry sweep that ran no arm would satisfy every
    // line above, and a synthetic statement too thin to satisfy the bodies is
    // exactly how that happens.
    assert!(
        ran > 100,
        "{ran} of `kernels-wgpu`'s entrypoints could be run at all, so the \
         statement this file synthesizes is too thin to reach the arms"
    );
    // WHICH arms walk the cache contiguously, named rather than counted: the
    // set is small, `contiguous_pool`'s own documentation states it, and an arm
    // joining or leaving it is a change to which kernels this driver may serve
    // on a paged pool.
    assert_eq!(
        strided.iter().copied().collect::<Vec<_>>(),
        vec![
            "kv_append",
            "sdpa_vector_decode",
            "sdpa_vector_decode_sink",
            "sdpa_vector_decode_swa",
        ],
        "a different set of arms reads the contiguous cache's strides, and only \
         these are witnessed: {strided:?}"
    );
    // And the two numbers a PAGED pool does answer are witnessed, or the walk
    // above is only ever proving absence.
    assert!(
        !page_sized.is_empty(),
        "no arm is handed the pool's page size, so `Handles::fire_number` is \
         answering nothing and the absences above are not evidence"
    );
    assert!(
        !masked.is_empty(),
        "no arm is handed the attention mask's stride, so the absences above \
         are not evidence"
    );
}

/// Where a word first appears in what an arm hands its body, if it appears.
///
/// By VALUE and across the integer variants, because an arm chooses the width:
/// `kv_append` hands its strides over as `Usize` and `paged` hands the page
/// size over as `I32`, and a sentinel that survived the cast is the same
/// number either way. Buffers and floats are excluded -- a handle is an index
/// into a bound list and a float is not one of these numbers, so a match on
/// either would be a coincidence rather than a resolver's answer.
fn position(args: &[ArgValue], word: u32) -> Option<usize> {
    args.iter().position(|value| match value {
        ArgValue::U32(n) => *n == word,
        ArgValue::I32(n) => n.cast_unsigned() == word,
        ArgValue::Usize(n) => *n == u64::from(word),
        // A HANDLE IS NOT ONE OF THESE NUMBERS, whether or not it arrived
        // with its rectangle beside it. A RAISED VIEW is not either: it
        // carries a HOST address, and a host address matching a sentinel
        // is a coincidence rather than a resolver's answer.
        ArgValue::Buffer(_)
        | ArgValue::Shaped { .. }
        | ArgValue::F32(_)
        | ArgValue::Raised(_) => false,
    })
}

/// Every row count the projection guard admits to the GEMM is one the
/// geometry can launch.
///
/// # The defect this was written to measure
///
/// `llama_like`'s projection is a GUARD, not a Rust `if`: the arm that fires
/// is chosen per fire, at lowering time. Its predicate was
/// `GuardPred::TokensGT(tile - 1)` — enough rows — while the kernel needs a
/// whole number of tiles. The comment directly above it quoted the
/// precondition it then failed to test: *"its header says the driver only
/// selects it when `M % BM == 0`"*. 32 rows is two whole 16-row tiles, 35 is
/// not, and both are `> 15`, so a real `pie run` of *"What is the capital of
/// France? Answer in one word."* — 35 tokens after templating — died at the
/// first projection:
///
///     Unplannable { symbol: "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
///                   why: Ungeometric { why: PartialTile { rows: 35, tile: 16 } } }
///
/// Fifteen row counts in sixteen were in that state, on Metal, Vulkan and
/// wgpu alike — `qmm_multi_batch` is `true` on all three.
///
/// # What fixed it, and why the fix was not a driver's
///
/// `driver-metal`'s `Ungeometric::PartialTile` says the answer is *"a text
/// that states the pair with a predicate on rows"*, as though the text did
/// not. It did — this guard has had a GEMM arm and a GEMV arm all along. The
/// mechanism was not missing; the predicate was wrong, and `GuardPred` could
/// not spell the right one. `GuardPred::TokensMultipleOf(k)` is that variant:
/// one arm in `model_compiler::lower` (where every guard is evaluated, in ONE
/// place), one wire slot, one line in the text.
///
/// No driver could have reached it. `m` is in no entrypoint's uniform block —
/// `qmm_t.wgsl`: *"the launch names the grid and nothing else"* — so the row
/// overhang cannot be guarded in the shader either without changing the
/// shared table's operand list.
///
/// # What this asserts now
///
/// Not "35 works", which one predicate makes true and any other could make
/// true by accident. It sweeps EVERY row count from 1 to four tiles, asks the
/// plan which arm came out, and asks the geometry whether that arm's grid
/// exists — two answers computed by different code, in different crates, that
/// have to agree for every count. A predicate that admitted one row too many
/// or one too few fails on the count it got wrong, by number.
///
/// # And the two have since stopped needing to agree
///
/// `qmm_t.wgsl` takes its row count in `Params` now and discards the
/// overhang, so `Rule::Qmm` rounds a partial tile up instead of refusing it —
/// the geometry has a grid for every row count and the halves of the sweep no
/// longer mirror each other. The GEMM half is unchanged and is still the
/// claim that matters; the GEMV half became a LEDGER of the counts the text's
/// predicate withholds from a path that could now take them, which is a gap
/// in `GuardPred::TokensMultipleOf` and not in this driver.
#[test]
fn every_row_count_the_guard_sends_to_the_gemm_has_a_grid() {
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let metal = wgpu_facts();
    let tile = metal.qmm_tile.0.max(1) as usize;
    assert!(
        metal.qmm_multi_batch,
        "the guard only exists for such a build"
    );

    // The GEMM's own symbol, built the way the text builds it, so the tile the
    // geometry reads is the tile the plan named.
    let symbol = format!(
        "affine_qmm_t{}",
        model_dsl::metal::affine_gemm_point(metal.proj_repr, metal.affine_bits, metal.qmm_tile,)
    );
    let module = driver_wgpu::reflect::entrypoint(&symbol, kernels_wgpu::Capability::Baseline)
        .map(|d| driver_wgpu::geometry::Module::loaded(&symbol, &d))
        .unwrap_or_else(|e| panic!("`{symbol}` is a module this build has: {e}"));

    let mut gemm_counts = Vec::new();
    // Row counts the guard sends to the MATVEC that the GEMM could have
    // taken. Empty until the tiled GEMM learned to round a partial tile up;
    // see the ledger under the sweep.
    let mut withheld = Vec::new();
    for rows in 1..=4 * tile {
        let low = lowered(&facts, &metal, FireClass::Prefill, rows)
            .unwrap_or_else(|| panic!("qwen3 lowers at {rows} rows"));
        let takes_gemm = low.kernels.iter().any(|k| k.starts_with("affine_qmm_t"));
        let grid = driver_wgpu::geometry::groups(
            driver_wgpu::geometry::Rule::Qmm,
            driver_wgpu::geometry::Dims {
                rows: rows as u32,
                width: 1024,
                in_width: 1024,
                ..Default::default()
            },
            module,
        );
        if takes_gemm {
            gemm_counts.push(rows);
            assert!(
                grid.is_ok(),
                "the guard sent {rows} rows to `{symbol}`, which the geometry \
                 refuses: {:?}",
                grid.unwrap_err()
            );
        } else {
            // THE OTHER DIRECTION USED TO BE A REFUSAL AND IS NOW A GAP.
            //
            // This asserted `grid.is_err()`: a count the guard kept from the
            // GEMM was a count the geometry would have refused anyway, so the
            // two agreed and neither was doing the other's work. That held
            // while `Rule::Qmm` raised `PartialTile`.
            //
            // It does not any more. `qmm_t.wgsl` takes its row count in
            // `Params` and `write_out` returns on `row >= m`, so the arm
            // rounds a partial tile up and every row count from 1 to four
            // tiles has a grid. The geometry stopped refusing; the TEXT's
            // guard did not stop withholding.
            //
            // So the claim is inverted rather than dropped. The grid must
            // exist -- if it ever does not, the round-up regressed and this
            // says so on the count it broke on -- and the count is recorded
            // as work the driver could take and the text does not offer.
            withheld.push(rows);
            assert!(
                grid.is_ok(),
                "{rows} rows took the GEMV arm and the GEMM has no grid for it \
                 either: {:?}. The tiled GEMM is supposed to round a partial \
                 tile up now",
                grid.unwrap_err()
            );
        }
    }

    // AND THE GAP IS EXACTLY THE NON-MULTIPLES, which is the useful form for
    // it to be in.
    //
    // `GuardPred::TokensMultipleOf(bm)` is the text's predicate and it is a
    // modulus, so it withholds thirty-one prompt lengths in thirty-two from
    // the tiled path and sends them to the matvec instead. `geometry.rs`'s
    // `Rule::Qmm` arm records what that cost when the refusal was real: a
    // 496-token prefill read 529.2 tok/s against 512's 1238.1, and 1187.2
    // once the bound was in place.
    //
    // The bound IS in place here, on the driver's side. What has not moved is
    // the predicate, and it is not this crate's to move -- the guard is
    // evaluated in `model_compiler::lower` from a `GuardPred` a shared text
    // states, so relaxing it is a change every backend takes at once and one
    // that needs the other two drivers to round up first. `driver-metal` and
    // `driver-vulkan` still raise `PartialTile`.
    //
    // This list is therefore a LEDGER of that gap rather than a complaint
    // about it. When the predicate relaxes, `withheld` empties and this
    // assertion is the thing that notices.
    assert_eq!(
        withheld,
        (1..=4 * tile)
            .filter(|rows| !rows.is_multiple_of(tile))
            .collect::<Vec<_>>(),
        "the guard withholds a different set of row counts than the \
         non-multiples of its tile"
    );

    // And the set is the multiples of the tile, stated as a list rather than
    // as a rule, because a rule here would be the predicate agreeing with
    // itself.
    assert_eq!(
        gemm_counts,
        vec![tile, 2 * tile, 3 * tile, 4 * tile],
        "the GEMM arm is taken at exactly the tile-aligned row counts"
    );
}

/// Every launch of a real plan, at the token count this seam CLAIMS, fits the
/// device limit it will be dispatched under.
///
/// # The claim this measures
///
/// `engine/src/driver/backend/wgpu.rs` reports a `max_forward_tokens` and a
/// `max_forward_requests`. Those are not descriptions -- the scheduler FORMS
/// BATCHES under them, so a fire of four thousand tokens is a fire this driver
/// has said it will take. Nothing had checked that it can, and it could not:
/// the seam stated 4096 and the largest that fits is 4095.
///
/// The bound that makes this a real question is `MAX_WORKGROUPS_PER_DIMENSION`,
/// and `geometry.rs`'s own note says how close it is: *"a `Rule::Elementwise`
/// launch over a 4096-wide hidden and 32 rows is 131072 lanes, which is 512
/// workgroups at 256 wide -- fine -- but the same rule over a 151936-wide
/// vocabulary at 16 rows is 9496 workgroups, and a 64-row prefill of it is
/// 37984. The margin is one order of magnitude, not six."* One order of
/// magnitude is four doublings, and the claim is a factor of sixty-four above
/// the largest row count anything else in this file lowers.
///
/// # What it does NOT do
///
/// It does not dispatch. `plan_one` is the same code the fire path runs, asked
/// of the same modules over the same plan, and a grid it produces here is the
/// grid it produces there. What a GPU would add is whether the memory exists,
/// which is a different claim and one `max_page_refs` makes separately.
///
/// # Re-anchored
///
/// This used to ask `dispatch::rule_of` for the row's launch rule,
/// `kernels::sig_in` for the row itself, `dims_of` for the dims that rule reads
/// and `geometry::groups_within` for the grid. `kernels_wgpu::KERNELS` is empty
/// -- `rule_of` is always `Err` and `sig_in` always `None`, so every launch was
/// skipped and the `checked > 0` guard is what said so, loudly, instead of
/// letting the walk pass over nothing.
///
/// `plan_one` gives the planned rectangle's real dimensions directly and is
/// what a fire actually calls, so the limit is now held against the grid the
/// driver will encode rather than against a re-derivation of it. That is
/// strictly closer to the question, and it moves where the ceiling is applied.
/// `plan_one` deliberately does NOT apply it -- `dispatch.rs`'s own note says
/// why: the limit is the adapter's and a `Dispatch` is the kernel's -- so a
/// grid past 65535 does not arrive here as a refusal, it arrives as a number.
/// This test is therefore the one that compares, exactly as
/// `Device::refusals` does at encode time, and a launch over the line reads as
/// an axis in the report below rather than as an `Undispatchable`.
#[test]
fn every_launch_at_the_claimed_token_ceiling_fits_the_device() {
    // What the seam MEASURES, not what it used to state. It reported the
    // literal 4096 until this test failed on it: at 4096 tokens
    // `rms_single_row` wants 65,536 workgroups on one axis and the device's
    // limit is 65,535 -- over by exactly one. The seam searches for the
    // boundary now (`widest_fire`), and for this model the answer is 4095.
    //
    // Stated here as a literal ON PURPOSE. If the seam's search changes, this
    // should fail and be read, rather than quietly following it.
    const CLAIMED_TOKENS: usize = 4095;
    const LIMIT: u32 = driver_wgpu::geometry::MAX_WORKGROUPS_PER_DIMENSION;

    let facts = LlamaLikeFacts::qwen3_0_6b();
    let metal = wgpu_facts();
    let low = lowered(&facts, &metal, FireClass::Prefill, CLAIMED_TOKENS)
        .unwrap_or_else(|| panic!("qwen3 lowers at the {CLAIMED_TOKENS} tokens it is promised"));
    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
        ..Default::default()
    };
    let mods = modules(std::iter::once(&low));

    let store = Everything(Placeholder(GENEROUS));
    let buf = Placeholder(low.arena_bytes as u64);
    let arena = Arena {
        buffer: &buf,
        bytes: low.arena_bytes as u64,
    };

    let mut checked = 0usize;
    let mut widest = 0u32;
    let mut over: Vec<String> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        let Some(declared) = mods.get(symbol) else {
            continue;
        };
        let module = driver_wgpu::geometry::Module::loaded(symbol, declared);
        match driver_wgpu::dispatch::plan_one(
            &low,
            launch,
            Built { module, declared },
            Sources {
                arena,
                resolver: &store,
                min_offset: STRICTEST_ALIGNMENT,
            },
            geometry,
        ) {
            Ok(d) => {
                checked += 1;
                for axis in d.groups {
                    widest = widest.max(axis);
                    if axis > LIMIT {
                        over.push(format!(
                            "`{symbol}` asks for a grid of {:?}, and {axis} is past \
                             the {LIMIT} a dispatch may name on one axis",
                            d.groups
                        ));
                    }
                }
            }
            // A refusal is not a grid past the limit, but at THIS token count
            // it is the same failure to the scheduler: a fire it was told this
            // driver would take, that this driver will not encode. Collected
            // rather than panicked on, so the report names every one.
            Err(e) => refused.push(format!("`{symbol}`: {e}")),
        }
    }

    assert!(
        refused.is_empty(),
        "{} launches of a {CLAIMED_TOKENS}-token prefill do not plan, which \
         this seam says it takes:\n  {}",
        refused.len(),
        refused.join("\n  ")
    );
    assert!(
        over.is_empty(),
        "{} launches of a {CLAIMED_TOKENS}-token prefill ask for more \
         workgroups on one axis than the device allows:\n  {}",
        over.len(),
        over.join("\n  ")
    );
    assert!(
        checked > 0,
        "no launch of a {CLAIMED_TOKENS}-token prefill was checked, so this \
         test measures nothing"
    );
    // The margin, printed rather than asserted: it is a fact about this model
    // at this ceiling, and pinning it would fail on the next model rather than
    // on a defect. What IS asserted is that nothing crossed the line.
    eprintln!(
        "{checked} launches at {CLAIMED_TOKENS} tokens, widest grid {widest} of {LIMIT} \
         ({}% of the limit)",
        widest as u64 * 100 / u64::from(LIMIT)
    );
}

// RETIRED: it has nothing left to compare, and its instrument is deleted.
//
// THIS WAS THE CENTRAL CONTROL OF THE WHOLE REFACTOR. For every rectangle of
// every real lowering in the text corpus it derived the dispatch TWICE -- once
// through `plan_one`'s fork, which resolves the arm and runs the routine body,
// and once through `plan_by_row`, which read the kernel's row -- and compared
// every field: module, entrypoint, grid, buffer list, offsets, scalar bytes.
//
// It is why a LIVE family could be armed at all. `sample` and `ptir` were dark
// -- no text names either symbol -- so their crossings could only be argued.
// `mlp` was the first family the corpus actually fires, and argument was not
// enough. Rectangles compared, family by family: mlp 352, norm 1700, the
// five-family batch 2764, attn 704, gate/router/qmv 432.
//
// Because the comparison needed the very row it was about to delete, the
// commit shape was forced: ARM, COMPARE, DELETE, in one commit per family. It
// asserted `armable.is_empty()` -- no armed family may still hold rows -- so
// a window could never be left open unused, and it printed its totals rather
// than passing silently at zero.
//
// What it caught, which review had not:
//
// * `residual_add` asked for a 2-D grid against a `gid.x`-only shader. 63 of
//   64 rows untouched, and the dispatch reports success.
// * `gate` reads `input(0)` on metal and vulkan -- the tensor `output(0)`
//   already aliases -- computing `attn *= sigmoid(attn)`. Fixed in both.
// * the three transcode encoders forwarded no scalars at all, so their
//   `@group(1)` uniform arrived empty and the shader read zero groups.
//
// IT WENT BLIND, NOT TRUE, and it went blind by succeeding: the last row it
// could have compared was `silu_mul_strided`'s, and deleting that row is what
// finished the job it existed to police. `plan_by_row` is gone with the
// columns it read, so the comparison cannot be written again even in
// principle.
//
// NOTHING REPLACES IT, and nothing should: a control that holds a new plane to
// an old one has no work left once the old plane is deleted. What holds the
// routine plane now is everything else in this file -- the walks that plan
// every rectangle of every real lowering through `plan_one` and check it
// against the MODULE (`every_launchs_scalars_land_where_its_module_reads_them`
// compares a body's packing against `Declared::uniform_offsets`;
// `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
// walks all 6920) -- plus, across backends,
// `two_backends_that_crossed_the_same_kernel_agree_on_its_signature`, which is
// routine against routine over 199 kernels and never needed a table.
