//! Can this backend bind, place and dispatch what a REAL lowering produces?
//!
//! Every other test in this crate builds its `Lowered` by hand. That checks
//! that the binder, the geometry and the dispatch planner agree with a plan a
//! TEST invented, and says nothing about whether they agree with the plan
//! `model-compiler` produces for an architecture somebody serves. This file is
//! the other question: six real texts, both fire classes, twelve lowerings,
//! 6584 rectangles, put through the code this crate ships.
//!
//! It is `driver-vulkan/tests/arena.rs`'s question asked of this backend, and
//! the interesting part is where the two answers differ. Three places, and
//! each one changes what a check can claim.
//!
//! # Why every count in this file is pinned, and what that costs
//!
//! Each check asserts the SIZE of what it walked — 14948 arena operands, 6584
//! rectangles, 24288 operands bound — because the failure these checks exist
//! to prevent has a silent twin: a sweep that iterated nothing passes exactly
//! as loudly as one that iterated everything and agreed. A `> 0` floor would
//! let the coverage shrink to one text without saying so.
//!
//! The cost is real and should be stated rather than discovered. **These
//! numbers move whenever `crates/model` changes a text.** Rebasing onto
//! upstream moved every one of them in a single afternoon — 14660 became
//! 14948, 6440 became 6584 — and eight tests went red at once for no reason
//! anybody would call a defect.
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
//! not zero-margin, it is absent: the widest scalar block any row states is 64
//! bytes and `kernels_wgpu::uniform_size` rounds to 16, so no block's own size
//! is a multiple of the granularity a suballocated one would have to start at.
//! [`every_arena_offset_a_real_lowering_assigns_is_bindable`] states that,
//! because it is the reason `device::Device::uniform` gives every launch a
//! buffer of its own at offset zero rather than packing a frame's blocks into
//! one.
//!
//! # The modules: unconditional here, conditional there
//!
//! `driver-vulkan` reads `.spv` files out of `PIE_KERNELS_VULKAN_SPV_DIR`, and
//! THREE of its eight checks return early with a printed reason when `glslc`
//! did not run -- the three that need a module. There is no such condition
//! here. `kernels_wgpu::entrypoint_source` hands back compile-ready WGSL for
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
//! ONE `@group(1) @binding(0)` uniform block, or -- where the row says so by
//! giving a `Param` operand a buffer kind -- a struct at a `@group(0)` slot.
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
//! against the row's, since a block of the right SIZE with the fields in the
//! wrong PLACES is a shader reading a stride where a head count belongs.
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
//! If it ever names one this table does not have, that is a coverage
//! divergence between two tables that are supposed to be the same table, and
//! [`every_symbol_a_real_text_launches_has_a_module`] is where it surfaces --
//! as a missing module rather than as a `Undispatchable::Unknown` at a fire.
//!
//! # The unstated rows, measured rather than assumed
//!
//! 56 of the table's 100 rows state no operands, covering 292 of its 481
//! entrypoints, and `.wiki/new-driver/vulkan.md` §13 is the argument that they
//! are not unlaunchable: `driver-metal` falls back to the lowered plan's own
//! argument order, and [`binding::reorder`] carries that fallback.
//!
//! NONE of the twelve lowerings reaches one today, and that is a measurement
//! with a date on it. `mxfp4_qmv_routed_bias` was the one operand-less row a plan
//! could name -- `model-compiler`'s routed-QMV site picks it by a `match` on
//! the weight repr -- and it was given a stated operand list, so `gpt_oss_20b`
//! now launches a row that says where its buffers go. The fallback is
//! therefore exercised deliberately in
//! [`every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`]:
//! every one of the 6584 real rectangles is put through `reorder` a second
//! time under a REAL unstated row of the shipped table, and has to come back
//! as exactly the plan's own operands in the plan's own order. All 56 of those
//! rows also state `LaunchRule::Unstated`, so no real one of them could reach
//! a grid even if a text named it, and that is asserted rather than described.
//!
//! # GPU-free, all eight
//!
//! The questions are about numbers a compiler produced and modules `naga` can
//! read, and a check that needed a device would not run in the builds that
//! change them. [`binding::Placeholder`] is a size and nothing else, so this
//! file compiles and passes with no features at all.

use std::collections::{BTreeMap, BTreeSet};

use driver_wgpu::binding::{
    Arena, FireNumber, FireTable, ParamSlot, Params, Placeholder, Resolve, Slot, Unbindable,
};
use driver_wgpu::dispatch::{Built, Geometry, Sources};
use driver_wgpu::reflect::Declared;
use kernels::{KernelSig, Source};
use kernels_wgpu::{Binding, Capability};
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Arg, Fire, Launch, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

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
/// file uses it to say what a shell may not do with the uniform blocks the
/// rows state.
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
            if let Some(low) = lowered(&facts, &metal, class, rows) {
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
                    },
                ));
            }
        }
    }
    // `phi3_mini` is in the catalog and is deliberately NOT in the list above:
    // it is 96 wide per head, `sdpa_paged_decode` is compiled at 64, 128, 256
    // and 512, and the declaration names `sdpa_paged_decode_bfloat16_d_96`.
    // `model_compiler::trace`'s signature check refuses it -- by PANICKING, not
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
/// 6584 times. Cached here for the same reason a driver caches a PIPELINE and
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
        Some(match which {
            FireNumber::KvPageSize => 0x0011_1111,
            FireNumber::KvHeadStride => 0x0022_2222,
            FireNumber::KvSeqStride => 0x0033_3333,
        })
    }
}

/// The sentinel a row's source is supposed to be handed, or `None` for a
/// source that is not the pool's.
fn sentinel(src: Source) -> Option<u32> {
    match src {
        Source::KvPageSize => Some(0x0011_1111),
        Source::KvHeadStride => Some(0x0022_2222),
        Source::KvSeqStride => Some(0x0033_3333),
        _ => None,
    }
}

/// The `@group(0)` binding a row puts its parameter STRUCT at, if it has one.
///
/// Read off `kernels_wgpu::bindings`, which is the ABI as code, rather than
/// found by size the way `driver-vulkan` has to find it: a row says "the rest
/// of this run is a struct, and it starts here" by giving a `Param` operand a
/// buffer kind, and the table can therefore answer where. The reflection is a
/// CHECK on that answer below and not the source of it.
fn struct_slot(sig: &KernelSig) -> Option<u32> {
    kernels_wgpu::bindings(sig)
        .into_iter()
        .zip(sig.operands)
        .find_map(|(binding, operand)| match (binding, operand.source) {
            (Binding::Storage(at), Source::Param(_) | Source::ParamF32(_)) => Some(at),
            _ => None,
        })
}

/// The `@group(0)` binding numbers the row states and nothing fills, within
/// the module's layout.
///
/// `Source::Unbound` is how a row says "this slot exists and nothing supplies
/// it": `kv_append_paged` keeps seven placeholders so that the rest of its row
/// stays where a shared ring ABI put them, and the unbiased routed QMV keeps
/// the bias slot its biased twin uses. They are the row's gaps rather than the
/// driver's debts, which is why the accounting below counts them separately
/// from what an executor still owes.
///
/// Bounded by the module's binding count on purpose. `kv_append_paged` states
/// THIRTEEN buffer slots against a twelve-binding module, and the thirteenth is
/// the seventh placeholder, which `binding::descriptors` drops as a tail past
/// the layout; a gap that is not in the layout is not a slot anybody has to
/// account for.
fn gaps(sig: &KernelSig, declared: &Declared) -> u32 {
    kernels_wgpu::bindings(sig)
        .into_iter()
        .zip(sig.operands)
        .filter(|(binding, operand)| match binding {
            Binding::Storage(at) => {
                matches!(operand.source, Source::Unbound) && *at < declared.bindings
            }
            Binding::Uniform(_) | Binding::Packed => false,
        })
        .count() as u32
}

/// Every activation the compiler places can be bound by a `BufferBinding`.
///
/// The claim that makes `Bound::within` usable for real work rather than only
/// for the hand-built arenas in `tests/device.rs`, plus the uniform-side
/// question that has no Vulkan counterpart.
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
        operands, 14948,
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
    // one. The blocks are TINY: the widest any of the 100 rows states is 64
    // bytes and every one is a multiple of 16, because WGSL gives a
    // host-shareable struct an alignment of at least 16. None is a multiple of
    // the 256 a suballocated block would have to start at, so a shell packing
    // a frame's blocks into one buffer end to end produces an unbindable
    // offset at the second launch.
    //
    // `binding::Params` carries no offset at all, which is what makes that
    // unrepresentable rather than merely unwise: the only offset this driver
    // can produce for a uniform block is zero, and zero divides everything.
    let blocks: Vec<u32> = kernels_wgpu::KERNELS
        .iter()
        .map(kernels_wgpu::uniform_size)
        .filter(|bytes| *bytes > 0)
        .collect();
    assert_eq!(
        blocks.len(),
        25,
        "a different number of rows state a scalar block"
    );
    assert!(
        blocks.iter().all(|b| b.is_multiple_of(16)),
        "a row states a block WGSL cannot lay out in the uniform address space"
    );
    let widest_block = blocks.iter().copied().max().expect("rows state blocks");
    assert_eq!(widest_block, 64, "the widest scalar block changed");
    assert!(
        widest_block < UNIFORM_ALIGNMENT,
        "every scalar block is smaller than the granularity a suballocated one \
         would have to start at, so a shell that packed them would be placing \
         blocks at offsets no implementation has to accept"
    );
    assert_eq!(
        blocks
            .iter()
            .filter(|b| b.is_multiple_of(UNIFORM_ALIGNMENT))
            .count(),
        0,
        "a row now states a block whose own size is a multiple of the uniform \
         alignment, which is the first row a shell could pack without rounding"
    );
}

/// Every symbol a real text launches has a module this backend can compile.
///
/// The claim `driver-metal`'s `model_bind` makes for its own table, asked of
/// this one: an entry point is compiled from a NAME, so a text that states a
/// symbol the table knows needs no arm written to receive it. Twenty-one
/// distinct symbols, and `kernels-wgpu` has WGSL for all twenty-one.
///
/// Two things make this stronger than its Vulkan counterpart. It does not
/// skip: there is no build directory to look in, so "the shaders were not
/// built" is not a state this backend has. And it does not stop at the name --
/// `reflect::entrypoint` expands the variant's includes and `//#if` arms,
/// hands the result to `naga`, and refuses a source that is not one
/// dispatchable compute module, so a symbol that is in the table and whose
/// WGSL does not parse fails here rather than at a fire.
///
/// It is a smaller number than the table's 481 because a lowering is not yet
/// the whole of a fire -- `Lowered::residue` holds the statements that still
/// run without a rectangle. What it measures is the part that HAS crossed, and
/// that part is fully served.
#[test]
fn every_symbol_a_real_text_launches_has_a_module() {
    let table: BTreeSet<String> = kernels_wgpu::entrypoints().into_iter().collect();
    assert_eq!(
        table.len(),
        481,
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
    assert_eq!(
        launched.len(),
        REACHES.len(),
        "the texts launch {} distinct symbols and this file describes {}",
        launched.len(),
        REACHES.len()
    );
    // Every launched symbol at every tier that HAS a source, pinned because
    // the loop above passes vacuously for a tier that has none -- and the
    // number says something worth knowing: it is 21, one per symbol, so every
    // module these plans need is a BASELINE module.
    //
    // That is a property of the whole tree rather than of these texts. No
    // `// pie:instantiate` line carries an `@fp16` or `@subgroup` tag, so
    // `entrypoint_source` answers `NoVariant` above baseline for all 481
    // entrypoints and a driver walking `Capability::PREFERENCE` lands on
    // baseline whatever the adapter allows. Core WebGPU is what these plans
    // run on, which is the tier a browser gets.
    assert_eq!(
        tiers,
        launched.len(),
        "a different number of (symbol, tier) pairs have a module"
    );
    // Pinned, and it MOVED once already: upstream changed a text to launch
    // `neox_freqs_mb` beside `neox_mb` — a rope that reads a precomputed
    // `inv_freq` where its sibling derives the rotation from a `base` scalar,
    // which is what a rescaled context needs and cannot state a base for. The
    // number moving is the news; it arriving as a failure rather than as a
    // silence is the point of pinning it.
    assert_eq!(tiers, 22, "a different number of symbols was launched");
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
    /// The module binds this many `@group(0)` entries the plan does not state,
    /// because they are the DRIVER's own: the paged KV cache, its page table,
    /// the routing scratch.
    ///
    /// The row's own gaps are subtracted first. A slot the row leaves
    /// `Unbound` is nobody's debt -- nothing fills it and the shader does not
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
    /// output. `Source::OutWidth(0)` is how the row says where to read it and
    /// `binding::scalars` is what reads it, so nothing outside the plan is
    /// needed to fire this kernel.
    RunGrows(u32),
}

/// Every symbol the reachable texts launch, and how it must be called.
///
/// Transcribed, so that a text that starts launching something new, or a
/// shader that changes its binding count, is a failure here rather than a
/// surprise in a fire.
const REACHES: &[(&str, Reaches)] = &[
    (
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32",
        Reaches::Uniform,
    ),
    (
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
        Reaches::Uniform,
    ),
    ("affine_qmv_fast_bfloat16_gs_64_b_4", Reaches::Uniform),
    (
        "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
        Reaches::Uniform,
    ),
    // Seven bindings, six of them the plan's and one the row's own gap: the
    // unbiased routed QMV keeps the bias slot its biased twin reads, and the
    // module declares it and never touches it. `DriverSupplies(1)` until the
    // gap was counted, which is the whole argument for counting gaps.
    ("affine_qmv_routed_bfloat16_gs_64_b_4", Reaches::Uniform),
    // The MXFP4 twin, and the newest row here: it stated NO operands until
    // recently, which made it the one operand-less row a real plan could name.
    // Now it says where its buffers go, and its unread `biases` slot is the
    // same kind of gap -- the codec has no bias plane, so nothing fills it.
    ("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", Reaches::Uniform),
    // Two slots, both stated -- the value it biases in place and the bias --
    // and ONE word the statement does not carry.
    ("add_bias_bfloat16", Reaches::RunGrows(1)),
    ("residual_add_bfloat16", Reaches::Bare),
    ("silu_mul_bfloat16", Reaches::Bare),
    ("combine_sorted", Reaches::Storage(3)),
    ("gptoss_swiglu_bfloat16", Reaches::Storage(3)),
    ("rms_single_row_bfloat16", Reaches::Storage(3)),
    ("route_gather", Reaches::Storage(3)),
    // The one with an operand AFTER its block, which is why `Storage` carries
    // the slot instead of being derived from the operand count.
    ("route_sort", Reaches::Storage(4)),
    // A block at 3 of 5 and a GAP at 4: the unscaled top-k declares a
    // per-expert scale buffer its body never reads, and the row leaves the slot
    // empty so that the scaled twin's numbering is the same numbering.
    ("router_topk_bfloat16", Reaches::Storage(3)),
    // One driver-owned table each: the token ids an embedding gathers by, the
    // positions a rope turns by, and the sampling indices a row gather reads.
    (
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        Reaches::DriverSupplies(1),
    ),
    ("neox_mb_bfloat16", Reaches::DriverSupplies(1)),
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
    ("neox_freqs_mb_bfloat16", Reaches::DriverSupplies(2)),
    ("row_gather_bfloat16", Reaches::DriverSupplies(1)),
    // Twelve bindings, six of them the row's ring-ABI gaps, two of them the
    // plan's, and FOUR the driver's: both sides of the paged cache and the two
    // tables saying where this fire writes.
    ("kv_append_paged_bfloat16", Reaches::DriverSupplies(4)),
    // Eight each: the cache's two sides, the page indices and their indptr,
    // the positions, the request map and the two mask tables.
    (
        "sdpa_paged_decode_bfloat16_d_128",
        Reaches::DriverSupplies(8),
    ),
    (
        "sdpa_paged_decode_sink_bfloat16_d_64",
        Reaches::DriverSupplies(8),
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
/// * six put their scalars in a STRUCT at a `@group(0)` slot the row names,
///   and their operands in the rest;
/// * one is short of a WORD, which its row derives from the launch;
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
/// the thing that fills an unread slot is the row's own `Unbound` gap.
///
/// So this asserts what `driver-vulkan` cannot: **the slots the row leaves
/// empty are exactly the slots the module does not read.** If a row grew a gap
/// where the shader reads, the dispatch would be refused by
/// `Unlayoutable::Unfilled`; if a shader stopped reading a slot the row fills,
/// a real tensor would be bound to an entry nothing looks at, which costs a
/// descriptor and says the row is describing a variant it no longer serves.
#[test]
fn what_the_plan_states_and_what_the_module_binds_account_for_each_other() {
    let texts = texts();
    let mods = modules(texts.iter().map(|(_, low)| low));
    let mut seen: BTreeMap<String, Reaches> = BTreeMap::new();
    let mut wrong = Vec::new();
    let mut launches = 0u32;

    for (text, low) in &texts {
        for launch in &low.launches {
            launches += 1;
            let symbol = &low.kernels[launch.kernel as usize];
            let declared = &mods[symbol];
            let sig = kernels_wgpu::sig(symbol).expect("a launched symbol has a row");
            let params = launch.params.end - launch.params.start;
            // An IN-PLACE row binds one buffer for two of the plan's args: the
            // trace states the value and the result separately, because a tape
            // whose statements did not produce values could not say what the
            // next one reads, and the row then says they are the same
            // allocation. `norm::add_bias` is the only one here, and without
            // this it classifies as a kernel binding one FEWER entry than the
            // plan states -- which is true and is not what is interesting
            // about it.
            let args = (launch.args.end - launch.args.start)
                - u32::try_from(sig.in_place.len()).expect("a row states few aliases");
            let block = struct_slot(sig);
            let gaps = gaps(sig, declared);
            let uniform = declared.uniform_offsets.len() as u32;

            // What the row and the plan together account for, against what the
            // module declares. Asked FIRST, because a buffer account that does
            // not balance makes the scalar comparison meaningless: a kernel
            // short of the KV cache is short of the numbers describing it too.
            let accounted = args + gaps + u32::from(block.is_some());
            let reaches = if accounted != declared.bindings {
                Reaches::DriverSupplies(declared.bindings.saturating_sub(accounted))
            } else if let Some(at) = block {
                Reaches::Storage(at)
            } else if uniform == params {
                if uniform == 0 {
                    Reaches::Bare
                } else {
                    Reaches::Uniform
                }
            } else {
                Reaches::RunGrows(uniform.saturating_sub(params))
            };

            // The row's gaps and the module's unread bindings are the same
            // slots. Not a tautology: `gaps` is read off the TABLE and
            // `holes()` off the parsed WGSL, and nothing but this holds them
            // together.
            if gaps != declared.holes() as u32 {
                wrong.push(format!(
                    "{text}: `{symbol}` leaves {gaps} slots unbound and its \
                     module declares {} it never reads",
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
        launches, 6584,
        "a different number of rectangles was walked"
    );
    assert_eq!(seen.len(), REACHES.len(), "a different set was reached");
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
/// stands in for the driver's tables. Every arena operand, though -- 14948 of
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
            match driver_wgpu::binding::bind(&low, launch, arena, &store, STRICTEST_ALIGNMENT) {
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
                Err((i, why)) => refused.push(format!(
                    "{text}: `{}` operand {i}: {why}",
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
    assert_eq!(operands, 24288, "a different number of operands was bound");
    assert_eq!(
        arena_operands, 14948,
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
    assert_eq!(
        total, 2_964_655_200,
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
            if let Err((_, why)) =
                driver_wgpu::binding::bind(&low, launch, arena, &store, STRICTEST_ALIGNMENT)
            {
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

    assert_eq!(
        launches, 6584,
        "a different number of rectangles was re-bound"
    );
    // Two more, for the same reason the byte total moved: see the note above
    // it. A wider rectangle is a rectangle with more ways to run off the end.
    assert_eq!(
        refused, 14,
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
/// struct -- 20 for five words -- and `kernels_wgpu::uniform_size` rounds to 16
/// because WGSL gives a host-shareable struct an alignment of at least 16, so
/// the row says 32 where the module says 20. `Device::uniform` allocates by the
/// same rounding. Over, never under, at every launch here.
///
/// # And the size is not the whole of it
///
/// A block of the right length with its fields in the wrong PLACES is a shader
/// reading a stride where a head count belongs, and nothing reports it: a
/// uniform buffer is bytes. So the module's own field offsets are held against
/// `kernels_wgpu::uniform_layout`'s, which is what the driver would write by if
/// it wrote by the row. They agree at every field of every launched module.
#[test]
fn every_launchs_scalars_land_where_its_module_reads_them() {
    let texts = texts();
    let mods = modules(texts.iter().map(|(_, low)| low));
    let store = Everything(Placeholder(GENEROUS));
    let mut uniform = 0u64;
    let mut storage = 0u64;
    let mut bare = 0u64;
    let mut owed: BTreeSet<String> = BTreeSet::new();
    let mut module_alone = 0u64;
    let mut row_alone = 0u64;
    let mut split_fields = 0u64;

    for (text, low) in &texts {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let declared = &mods[symbol];
            let sig = kernels_wgpu::sig(symbol).expect("a launched symbol has a row");
            // A 64-bit scalar has no WGSL type, so the row declares it as a
            // `vec2<u32>` and `binding` writes its LOW word and leaves the high
            // one zero -- right for every stride a `Lowered::params` can carry,
            // since that run is a `Vec<u32>`. Counted so the ZERO below can say
            // the case is unwitnessed here rather than passing silently.
            if kernels_wgpu::uniform_layout(sig)
                .iter()
                .any(|field| field.split)
            {
                split_fields += 1;
            }

            match driver_wgpu::binding::scalars(sig, low, launch, declared, &store) {
                Ok(Params::Block {
                    bytes,
                    at: ParamSlot::Uniform,
                }) => {
                    uniform += 1;
                    // The block the SHADER declares, not four bytes a scalar:
                    // a run written end to end would place every field after a
                    // gap at the wrong offset.
                    assert_eq!(
                        bytes.len(),
                        declared.uniform_bytes as usize,
                        "{text}: `{symbol}` would offer {} bytes for a block of {}",
                        bytes.len(),
                        declared.uniform_bytes
                    );
                    // Every field the module reads is inside what the shell
                    // offers. The direction that matters: short reads as zeros
                    // and a zero pitch is a plausible number.
                    for offset in &declared.uniform_offsets {
                        assert!(
                            *offset as usize + 4 <= bytes.len(),
                            "{text}: `{symbol}` reads a field at {offset} out of \
                             {} bytes",
                            bytes.len()
                        );
                    }
                    // The row's layout and the module's, field for field.
                    let row: Vec<u32> = kernels_wgpu::uniform_layout(sig)
                        .iter()
                        .map(|field| field.offset)
                        .collect();
                    assert_eq!(
                        row, declared.uniform_offsets,
                        "{text}: `{symbol}` puts its scalars at the row's offsets \
                         and the module reads them at the module's"
                    );
                    // The allocation a shell makes from the row is never
                    // shorter than the struct the module declares. This is the
                    // inversion, stated as an inequality because being over is
                    // legal and being under is the defect.
                    assert!(
                        kernels_wgpu::uniform_size(sig) >= declared.uniform_bytes,
                        "{text}: `{symbol}` would be allocated {} bytes for a \
                         struct of {}",
                        kernels_wgpu::uniform_size(sig),
                        declared.uniform_bytes
                    );
                }
                Ok(Params::Block {
                    bytes,
                    at: ParamSlot::Storage(at),
                }) => {
                    storage += 1;
                    // Exactly the shader's struct. `tests/device.rs` shows a
                    // short one accepted and read back as zeros past its end.
                    assert_eq!(
                        declared.block_bytes.get(at as usize).copied().flatten(),
                        Some(bytes.len() as u32),
                        "{text}: `{symbol}` would write {} bytes into binding {at}",
                        bytes.len()
                    );
                    assert!(
                        at < declared.bindings,
                        "{text}: `{symbol}` would bind its block at {at} of {}",
                        declared.bindings
                    );
                    // The ROW said where the struct goes and the reflection
                    // agrees. Two readings of one placement: the table's
                    // `bindings` and `naga`'s sized block.
                    assert_eq!(
                        struct_slot(sig),
                        Some(at),
                        "{text}: `{symbol}` places its struct where the row does not"
                    );
                }
                Ok(Params::None) => {
                    bare += 1;
                    assert!(
                        declared.uniform_offsets.is_empty(),
                        "{text}: `{symbol}` places nothing against a module with \
                         {} uniform fields",
                        declared.uniform_offsets.len()
                    );
                }
                Err(why) => {
                    owed.insert(format!("{symbol}: {why}"));
                }
            }

            // The MODULE-ONLY placer, which is what an unstated row falls back
            // to. Counted rather than asserted, because for a STATED row it is
            // the wrong question and its answer is the measurement: the run a
            // row spells is not the statement's run. 1588 of these launches
            // state a number of scalars the module's block cannot hold on its
            // own -- the row indexes only part of the statement's run
            // (`neox_mb` reads three of four), interleaves the pool's page
            // size, or derives a width -- and every one of them places
            // correctly through the row above.
            match driver_wgpu::binding::params(low, launch, declared) {
                Ok(_) => module_alone += 1,
                Err(_) => row_alone += 1,
            }
        }
    }

    // NOTHING is owed, which is a stronger result than the Vulkan template
    // gets: six symbols there have scalars that crate cannot place, five of
    // them because the driver owns the resource and therefore the numbers
    // describing it. Here `binding::scalars` builds the run from the ROW --
    // interleaving `KvPageSize` where `kv_append_paged` puts it, taking three
    // of `neox_mb`'s four, deriving `add_bias`'s output width -- so every
    // launch of every text places.
    assert!(
        owed.is_empty(),
        "{} launches have scalars this crate cannot place:\n  {}",
        owed.len(),
        owed.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );
    // All three shapes actually occur, or a rule that never fires is passing
    // for the same reason an absent one would.
    assert_eq!(
        (uniform, storage, bare),
        (4352, 1720, 512),
        "a different number of launches take each parameter shape"
    );
    assert_eq!(
        (module_alone, row_alone),
        (4852, 1732),
        "a different number of launches need the row to build their run"
    );
    // ZERO, and stated as zero because it names what these texts do NOT reach.
    // Three rows in the table -- `kv_append` and the two contiguous vector
    // decodes -- carry a 64-bit stride as a `vec2<u32>`, which is the only
    // shape where a field is eight bytes wide and the shell writes four. No
    // text here launches one, so `binding`'s low-word write is unwitnessed by
    // any real plan and its own unit tests are the whole of its cover. A
    // deployment on a contiguous cache moves this number.
    assert_eq!(
        split_fields, 0,
        "a text now launches a row with a 64-bit uniform field, so the low-word \
         write is reachable from a real plan and should be checked here"
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
/// All 6584, across six architectures in both fire classes, and nothing is
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
/// # The unstated-row fallback is exercised here, deliberately
///
/// 56 rows state no operands and none of these texts reaches one, so the
/// fallback `binding::reorder` carries for them -- bind the plan's own args in
/// the plan's own order -- would go unwalked. It is walked: every rectangle is
/// put through `reorder` a second time under a REAL operand-less row of the
/// shipped table, and the slots that come back have to be exactly what
/// `binding::bind` produces. That is `.wiki/new-driver/vulkan.md` §13's claim,
/// held against 6584 real launches rather than against an argument.
#[test]
fn every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal() {
    let all = geometric();
    let mods = modules(all.iter().map(|(_, low, _)| low));

    // A row of the SHIPPED table, not one this file invented: the fallback is
    // about rows that exist, and picking one out of `KERNELS` means a table
    // that stopped having any would fail here rather than silently stop
    // testing the path.
    let unstated: Vec<&KernelSig> = kernels_wgpu::KERNELS
        .iter()
        .filter(|sig| sig.operands.is_empty())
        .collect();
    assert_eq!(
        (
            unstated.len(),
            unstated
                .iter()
                .map(|s| s.entrypoints().len())
                .sum::<usize>()
        ),
        (56, 292),
        "a different number of rows state no operands"
    );
    // Every one of them also states no launch RULE, so none could reach a grid
    // even if a text named it: `geometry::Ungeometric::Unstated` is where that
    // is refused, and it is a different sentence from "the row exists and this
    // backend has no shader for its rule".
    assert!(
        unstated
            .iter()
            .all(|sig| matches!(sig.launch, kernels::LaunchRule::Unstated)),
        "an operand-less row now states a launch rule, so the fallback can \
         reach a grid and this test has to say what happens then"
    );
    let plan_order = unstated[0];

    let mut launches = 0u32;
    let mut planned = 0u32;
    let mut refused: BTreeMap<String, u32> = BTreeMap::new();
    let mut fell_back = 0u32;
    let mut widest_grid = [0u32; 3];
    let mut workgroups = 0u64;
    let mut uniform = 0u32;
    let mut storage = 0u32;
    let mut refused_hollow = 0u32;
    let mut overridden = 0u32;
    let mut rotary_overridden = 0u32;
    let mut head_overridden = 0u32;
    let mut heads_overridden = 0u32;
    let mut split_rectangles = 0u32;
    let mut pool_numbers = 0u32;
    let mut derived_widths = 0u32;

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

            // THE UNSTATED-ROW FALLBACK, on the same real rectangle. A row that
            // states no operands has never told anyone an order, so the trace's
            // is the only one there is -- and `driver-metal` binds exactly
            // that. 292 entrypoints depend on it.
            let fallback = driver_wgpu::binding::reorder(
                plan_order,
                low,
                launch,
                arena,
                &store,
                STRICTEST_ALIGNMENT,
            )
            .expect("the plan's own operands bind in the plan's own order");
            let plain = driver_wgpu::binding::bind(low, launch, arena, &store, STRICTEST_ALIGNMENT)
                .expect("the same operands, bound the same way");
            assert_eq!(
                fallback.len(),
                plain.len(),
                "{text}: `{symbol}` falls back to {} slots for {} operands",
                fallback.len(),
                plain.len()
            );
            for (slot, bound) in fallback.iter().zip(&plain) {
                assert!(
                    matches!(slot, Slot::Buffer(b) if b == bound),
                    "{text}: `{symbol}`'s fallback slot is not the plan's operand"
                );
            }
            fell_back += 1;

            match driver_wgpu::dispatch::plan_one(
                low,
                launch,
                kernels_wgpu::KERNELS,
                Built { module, declared },
                sources,
                *geometry,
            ) {
                Ok(d) => {
                    planned += 1;
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
                    workgroups +=
                        u64::from(d.groups[0]) * u64::from(d.groups[1]) * u64::from(d.groups[2]);

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
                    if let Ok(hollow) = driver_wgpu::dispatch::plan_one(
                        low,
                        launch,
                        kernels_wgpu::KERNELS,
                        Built { module, declared },
                        sources,
                        Geometry::default(),
                    ) {
                        assert!(
                            !hollow.groups.contains(&0),
                            "{text}: `{symbol}` planned {:?} from a geometry of zeros",
                            hollow.groups
                        );
                    } else {
                        refused_hollow += 1;
                    }

                    let sig = kernels_wgpu::sig(symbol).expect("the walk found a row");
                    // The VALUE a row states at a slot, not merely that it
                    // states one. `!=` against a lying fire proves the answer
                    // did not come from the fire; it does not prove it came
                    // from the row, and an override off by one survived exactly
                    // that gap.
                    let value = |index: Option<u8>| -> Option<u32> {
                        let i = index?;
                        let at = launch.params.start as usize + i as usize;
                        (at < launch.params.end as usize)
                            .then(|| low.params.get(at).copied().unwrap_or(0))
                            .filter(|n| *n > 0)
                    };
                    let states = |index: Option<u8>| value(index).is_some();
                    if states(sig.grid_param) {
                        overridden += 1;
                    }
                    if states(sig.head_param) {
                        head_overridden += 1;
                    }
                    if states(sig.heads_param) {
                        heads_overridden += 1;
                        // Every row that states a head COUNT states a head
                        // WIDTH, which is why the two counts below are not
                        // independent -- and why they are still counted
                        // separately: a union would be 1056 either way, so
                        // `heads_param` going to zero would leave it unchanged
                        // and the suite green.
                        assert!(
                            states(sig.head_param),
                            "`{symbol}` states a head count and no head width, so \
                             the two overrides are no longer nested"
                        );
                    }
                    if matches!(sig.launch, kernels::LaunchRule::SplitPacked) {
                        split_rectangles += 1;
                    }

                    // Counting rows that STATE a head shape does not witness
                    // `dims_of` USING it, and the difference is not academic:
                    // across these texts the stated value equals the fire's
                    // and both override lines are no-ops, so deleting them
                    // leaves every other number here unchanged. The model that
                    // separates them is gemma-4, which is not one of the texts.
                    //
                    // So the fire is made to disagree on purpose.
                    let liar = Geometry {
                        head_dim: geometry.head_dim + 7,
                        kv_heads: geometry.kv_heads + 7,
                        rotary_dims: geometry.rotary_dims + 1024,
                        ..*geometry
                    };
                    let told = driver_wgpu::dispatch::dims_of(sig, low, launch, liar);
                    if states(sig.head_param) {
                        assert_eq!(
                            Some(told.head_dim),
                            value(sig.head_param),
                            "{symbol} states a head width and `dims_of` answered \
                             with something else"
                        );
                    }
                    if states(sig.heads_param) {
                        assert_eq!(
                            Some(told.kv_heads),
                            value(sig.heads_param),
                            "{symbol} states a head count and `dims_of` answered \
                             with something else"
                        );
                    }
                    // A rope row's rotary width comes from the STATEMENT
                    // because a model may rotate only part of its head --
                    // gemma-4 turns 128 of 512 -- and for these texts the
                    // stated width equals the fire's, so dropping the override
                    // changed nothing anywhere.
                    if states(sig.grid_param) && matches!(sig.launch, kernels::LaunchRule::Rope) {
                        rotary_overridden += 1;
                        assert_eq!(
                            Some(told.rotary_dims),
                            value(sig.grid_param),
                            "{symbol} states a grid and `dims_of` answered with a \
                             different rotary width"
                        );
                    }

                    // THE POOL'S NUMBERS REACH THE SHADER. A row may name a
                    // number that belongs to the pool rather than to the
                    // statement -- the KV page size and the cache's two strides
                    // -- and `binding::scalars` interleaves the driver's answer
                    // into the run at the row's position. The walk's own
                    // resolver answers `None` to every number, so all three
                    // read as zero above; a second resolver with recognisable
                    // answers runs beside it so no pinned count moves.
                    //
                    // A wrong stride is not an error. It is attention reading
                    // the wrong offsets and returning numbers.
                    for src in [
                        Source::KvPageSize,
                        Source::KvHeadStride,
                        Source::KvSeqStride,
                    ] {
                        if !sig.operands.iter().any(|o| o.source == src) {
                            continue;
                        }
                        pool_numbers += 1;
                        let got =
                            driver_wgpu::binding::scalars(sig, low, launch, declared, &sentinels)
                                .expect("the row's scalars place");
                        let bytes = match got {
                            Params::Block { ref bytes, .. } => bytes.clone(),
                            Params::None => Vec::new(),
                        };
                        let want = sentinel(src).expect("a pool source").to_le_bytes();
                        assert!(
                            bytes.windows(4).any(|w| w == want),
                            "{text}: `{symbol}` names {src:?} and the driver's \
                             answer is not in the {} bytes it hands the shader",
                            bytes.len()
                        );
                    }

                    // THE ROW'S DERIVED WIDTH REACHES THE SHADER, for the same
                    // reason and with the same shape of check.
                    // `Source::OutWidth(0)` is a number NOTHING in the
                    // statement carries -- an `AddBias` states no params at all
                    // -- so if `binding::scalars` read the wrong output, or
                    // dropped the source and left the run short, the module
                    // would get a zero and every lane would return before
                    // writing. That failure is silent in the direction that
                    // matters: a bias never added is a projection missing a
                    // small constant, which stays fluent.
                    if sig
                        .operands
                        .iter()
                        .any(|o| matches!(o.source, Source::OutWidth(_)))
                    {
                        derived_widths += 1;
                        let got =
                            driver_wgpu::binding::scalars(sig, low, launch, declared, &sentinels)
                                .expect("the row's scalars place");
                        let bytes = match got {
                            Params::Block { ref bytes, .. } => bytes.clone(),
                            Params::None => Vec::new(),
                        };
                        let width = low.args[launch.args.start as usize..launch.args.end as usize]
                            .iter()
                            .filter_map(|a| match a {
                                Arg::Arena { width, .. } | Arg::Named { width, .. } => Some(*width),
                                Arg::Weight(_) => None,
                            })
                            .next_back()
                            .expect("a widthed operand");
                        assert!(
                            width > 0 && bytes.windows(4).any(|w| w == width.to_le_bytes()),
                            "{text}: `{symbol}` names its output's width and \
                             {width} is not in the {} bytes it hands the shader",
                            bytes.len()
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
                            // written from: a slot the row leaves empty takes
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
                    // PLUS the slots the row leaves empty, are the module's
                    // whole binding set. `driver-vulkan` asserts the same sum
                    // with its holes SUBTRACTED, and that is a Vulkan
                    // coincidence rather than a rule: there a hole is an
                    // undeclared number, here it is a declared binding this
                    // entry point does not read, and `wgpu` validates a group
                    // against its layout entry for entry either way.
                    let block = usize::from(d.block_at.is_some());
                    let empty = gaps(sig, declared) as usize;
                    assert_eq!(
                        d.buffers.len() + block + empty,
                        declared.bindings as usize,
                        "{text}: `{symbol}` bound {} plus {block} plus {empty} for \
                         {} bindings",
                        d.buffers.len(),
                        declared.bindings
                    );
                }
                Err(e) => {
                    *refused.entry(format!("{symbol}: {e}")).or_default() += 1;
                }
            }
        }
    }

    assert_eq!(
        launches, 6584,
        "a different number of rectangles is lowered"
    );
    assert_eq!(planned, 6584, "a different number of rectangles records");
    assert_eq!(
        fell_back, 6584,
        "a different number of rectangles went through the unstated-row path"
    );

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

    // Both parameter homes are exercised. Stated because a change that routed
    // everything one way would leave the other untested while this passed.
    assert_eq!(
        (uniform, storage),
        (4352, 1720),
        "a different number of dispatches take each parameter home"
    );
    // Stated exactly, because the fallback is silent: a row whose stated extent
    // went missing would take the fire's number and normalise over the wrong
    // width, producing numbers rather than an error, and this count going to
    // zero is the only thing that would say so.
    // 1788 before the shared MoE rows started stating `grid_param` on the
    // three routed matvecs and `rows_param` on `route_gather`. This crate had
    // not copied either, so 432 rectangles were taking the fire's number where
    // the row names one -- which is what this count exists to notice.
    assert_eq!(
        overridden, 2220,
        "a different number of rectangles states its own extent"
    );
    // Counted separately, and that was not tidying. `head_param` is 1056 and
    // `heads_param` is 352, and the loop above asserts the second set is a
    // SUBSET of the first -- so a union would be 1056 either way and could not
    // witness `heads_param` at all: it going to zero would leave the number
    // unchanged and this file green.
    assert_eq!(
        head_overridden, 1056,
        "a different number of rectangles states a head width"
    );
    assert_eq!(
        heads_overridden, 352,
        "a different number of rectangles states a head count"
    );
    assert_eq!(
        rotary_overridden, 704,
        "a different number of rope rectangles states its rotary width"
    );
    // ZERO, and asserted as zero because that is the fact `dims_of`'s
    // `in_width` note rests on: no text here reaches `split_qkv`, so nothing
    // consumes `in_width` and replacing it with a constant is invisible. A text
    // that starts splitting its projections makes this number move, which is
    // when the note stops being true.
    assert_eq!(
        split_rectangles, 0,
        "a text now reaches `Rule::SplitPacked`, so `in_width` is no longer \
         unwitnessed"
    );
    // How many of these launches `plan_one` REFUSED when handed a geometry of
    // zeros -- the head-shaped ones. The rest plan fine because their
    // dimensions come from the lowering rather than from the geometry, and they
    // are right to.
    assert_eq!(
        refused_hollow, 352,
        "a different number of launches refused a geometry of zeros"
    );
    // All 704 are `KvPageSize`, in the two paged decodes and the paged append.
    // NOT ONE names a stride: these texts are paged throughout, and the
    // strides belong to the contiguous cache. That is why the loop above cannot
    // be the whole check, and why the rows themselves are swept separately in
    // `every_row_naming_a_pool_number_is_handed_that_number_and_not_another`.
    assert_eq!(
        pool_numbers, 704,
        "a different number of rectangles names one of the pool's numbers"
    );
    // qwen2.5 alone, and three a layer in each of its two fire classes: 28
    // layers x 3 biases x 2 classes. Pinned rather than asserted as non-zero,
    // because the failure it witnesses is a bias that is not added, and a text
    // that quietly stopped stating one would leave the check above passing over
    // nothing.
    assert_eq!(
        derived_widths, 312,
        "a different number of rectangles names a width the row derives"
    );
    // The total work these plans dispatch, as a single number.
    //
    // Here because every other assertion in this test is about SHAPE -- how
    // many operands, where the scalars went, that no grid is zero -- and a grid
    // can be the wrong size while being all of those things. Dropping
    // `dims_of`'s statement override changes no other assertion in this file
    // and changes this one.
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
        workgroups, 49_287_434,
        "the plans dispatch a different amount of work"
    );
    // The third dimension is a prefill's rows, and only the paged decodes put
    // anything there. A backend that flattened the grid to two dimensions would
    // have looked correct on every other rectangle.
    assert_eq!(
        widest_grid,
        [3584, 25136, 64],
        "the widest grid in any dimension changed"
    );
}

/// Every row that names one of the pool's numbers is handed that number, and
/// the three are not interchangeable.
///
/// The walk above can only ask this of rows its twelve lowerings reach, and
/// they reach exactly one of the three numbers: all 704 are `KvPageSize`,
/// because these texts are paged throughout and the two strides belong to the
/// contiguous cache. So the strides go unwatched there -- replacing either with a constant
/// leaves that walk green, and a wrong stride is not an error but attention
/// reading the wrong offsets and returning numbers.
///
/// `Pool`'s answers are checked in `resources.rs` and the shader's addressing
/// in `tests/device.rs`, which hand-writes its parameter bytes. Between those
/// two the seam is open. This closes it from the TABLE's side rather than a
/// text's, so a row added tomorrow is covered whether or not a text reaches it.
#[test]
fn every_row_naming_a_pool_number_is_handed_that_number_and_not_another() {
    let store = Sentinels(Placeholder(GENEROUS));
    // A real lowering, borrowed for its shape and then emptied of scalars.
    // Hand-building a `Lowered` would be a second definition of the type to
    // keep in step for no gain -- nothing below reads any field but `params`.
    let mut low = texts().swap_remove(0).1;
    let mut rows = 0u32;
    let mut refused_rows = 0u32;
    let mut named = BTreeMap::<String, u32>::new();

    for sig in kernels_wgpu::KERNELS {
        let wanted: Vec<Source> = sig
            .operands
            .iter()
            .map(|o| o.source)
            .filter(|s| sentinel(*s).is_some())
            .collect();
        if wanted.is_empty() {
            continue;
        }
        rows += 1;

        // A launch whose statement carries as many scalars as the row's `Param`
        // slots index, and a module wide enough to hold the run -- the question
        // here is the RUN's contents, not its placement, which `binding`'s own
        // tests already pin.
        let params = sig
            .operands
            .iter()
            .filter_map(|o| match o.source {
                Source::Param(i) | Source::ParamF32(i) => Some(u32::from(i) + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        low.params = (0..params).map(|i| 1000 + i).collect();
        let launch = Launch {
            kernel: 0,
            rows: 0..1,
            layers: 0..1,
            op: 0,
            args: 0..0,
            params: 0..params,
            peel: None,
            cond: Launch::NO_COND,
        };
        // A uniform block of EXACTLY the run's length in fields, because
        // `binding` refuses a run no shape can hold rather than truncating it.
        // One field per word: this stands in for the module, and the modules
        // these rows really have are checked field for field by
        // `every_launchs_scalars_land_where_its_module_reads_them`.
        let words = run_words(sig, params);
        let declared = Declared {
            local: [1, 1, 1],
            bindings: 1,
            used: vec![true],
            reads_workgroup_count: false,
            grid_axes: [true, false, false],
            uniform_offsets: (0..words).map(|i| i * 4).collect(),
            uniform_bytes: (words * 4).next_multiple_of(16),
            block_bytes: vec![None],
        };
        // A row wanting a STRIDE is refused, and that is the answer this
        // driver owes it. Both strides mean "walk the cache with no page
        // table", and the pool is `[page, token, head, dim]`: handing them
        // over makes the launch succeed against the wrong tokens. The
        // refusal is checked here too, rather than only in the unit test,
        // because this is the sweep that walks the SHIPPED table -- a row
        // added tomorrow that names a stride is refused or this fails.
        if wanted
            .iter()
            .any(|s| matches!(s, Source::KvHeadStride | Source::KvSeqStride))
        {
            let why = driver_wgpu::binding::scalars(sig, &low, &launch, &declared, &store)
                .expect_err("a contiguous stride is refused, not answered");
            assert!(
                format!("{why}").contains("pool is paged"),
                "`{}` is refused for the reason it should be: {why}",
                sig.symbol
            );
            refused_rows += 1;
            for src in wanted {
                *named.entry(format!("{src:?}")).or_default() += 1;
            }
            continue;
        }
        let got = driver_wgpu::binding::scalars(sig, &low, &launch, &declared, &store)
            .unwrap_or_else(|why| panic!("`{}`: {why}", sig.symbol));
        let bytes = match got {
            Params::Block { ref bytes, .. } => bytes.clone(),
            Params::None => Vec::new(),
        };
        // The WHOLE run, not "is the number in there somewhere". Presence alone
        // cannot see a swap: both strides are in the same run either way, and
        // swapping them is the defect with the most plausible output --
        // attention striding by heads where it should stride by positions,
        // reading real numbers from the wrong rows.
        //
        // Compared at the OFFSETS the block declares, which is where
        // `params_from` writes: the run is one word per field and the field
        // offsets here are four apart, so the bytes are the run.
        let want: Vec<u8> = expected_run(sig, &low, params)
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        assert_eq!(
            bytes[..want.len()],
            want[..],
            "`{}` hands the shader a different run than its row spells",
            sig.symbol
        );
        // The tail is the block's padding to 16 and holds nothing, so a run
        // that ran long would be caught above rather than hidden here.
        assert!(
            bytes[want.len()..].iter().all(|b| *b == 0),
            "`{}` writes past the run it spells",
            sig.symbol
        );
        for src in wanted {
            *named.entry(format!("{src:?}")).or_default() += 1;
        }
    }

    // Every one of the three is witnessed by at least one row. Stated because
    // the loop passes vacuously for a number no row names, which is precisely
    // the state the strides are in as far as any text goes.
    assert_eq!(named.len(), 3, "only these are witnessed: {named:?}");
    // Stated exactly. Six rows in the whole table name one of these, and a row
    // losing its source is not an error anywhere -- the number simply stops
    // being written and the shader reads whatever the statement left in that
    // slot.
    assert_eq!(rows, 6, "a different number of rows names a pool number");
    // Three of the six ROWS walk the cache contiguously -- `kv_append`,
    // `sdpa_vector_decode` and `sdpa_vector_decode_swa`. (The strides appear
    // five times each in the tally below because a row names both a key and a
    // value stride; this counts rows.) The other three name only
    // `KvPageSize` and are answered.
    assert_eq!(
        refused_rows, 3,
        "a different number of rows walks the cache with no page table"
    );
    assert_eq!(
        named,
        [
            ("KvHeadStride".to_string(), 5u32),
            ("KvPageSize".to_string(), 3),
            ("KvSeqStride".to_string(), 5),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
        "a different set of rows names each number"
    );
}

/// How many words the run for one row is, given a statement of `params`
/// scalars.
///
/// Mirrors `binding::scalars`' walk rather than guessing, because the two
/// disagreeing is the failure this file cannot see: a block sized from a guess
/// would refuse rows that are fine and pass rows that are not.
fn run_words(sig: &KernelSig, params: u32) -> u32 {
    let mut n = 0u32;
    for operand in sig.operands {
        match operand.source {
            _ if operand.ty == kernels::Ty::InPacked => n += 1,
            Source::KvPageSize | Source::KvHeadStride | Source::KvSeqStride => n += 1,
            Source::Param(i) | Source::ParamF32(i) => {
                if matches!(operand.ty, kernels::Ty::Buf | kernels::Ty::BufMut) {
                    n += params.saturating_sub(u32::from(i));
                } else {
                    n += 1;
                }
            }
            _ => {}
        }
    }
    n
}

/// The run one row spells, word for word, built from the row rather than from
/// `binding`.
///
/// A second reading of the same table. That is the point: the check it feeds is
/// that `binding::scalars` and the row agree, and a helper that asked `binding`
/// would only be checking it against itself.
fn expected_run(sig: &KernelSig, low: &Lowered, params: u32) -> Vec<u32> {
    let stated: Vec<u32> = (0..params).map(|i| 1000 + i).collect();
    let mut run = Vec::new();
    for operand in sig.operands {
        if operand.ty == kernels::Ty::InPacked {
            run.push(match operand.source {
                Source::RequestCount => low.n_requests,
                _ => 0,
            });
            continue;
        }
        match operand.source {
            Source::KvPageSize | Source::KvHeadStride | Source::KvSeqStride => {
                run.push(sentinel(operand.source).expect("a pool source"));
            }
            Source::Param(i) | Source::ParamF32(i) => {
                if matches!(operand.ty, kernels::Ty::Buf | kernels::Ty::BufMut) {
                    run.extend_from_slice(stated.get(usize::from(i)..).unwrap_or(&[]));
                } else {
                    run.push(stated.get(usize::from(i)).copied().unwrap_or(0));
                }
            }
            _ => {}
        }
    }
    run
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
///     Unplannable { symbol: "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32",
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
        model_compiler::dsl::metal::affine_gemm_point(
            metal.proj_repr,
            metal.affine_bits,
            metal.qmm_tile,
        )
    );
    let module = driver_wgpu::reflect::entrypoint(&symbol, kernels_wgpu::Capability::Baseline)
        .map(|d| driver_wgpu::geometry::Module::loaded(&symbol, &d))
        .unwrap_or_else(|e| panic!("`{symbol}` is a module this build has: {e}"));

    let mut gemm_counts = Vec::new();
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
            // The other direction, so this is not one claim checked against
            // itself: a count the guard KEEPS from the GEMM is a count the
            // geometry would have refused. If they ever both say yes here, the
            // guard has become needlessly narrow rather than wrong.
            assert!(
                grid.is_err(),
                "{rows} rows took the GEMV arm, but the GEMM's grid exists for \
                 it -- the guard is refusing work it could do"
            );
        }
    }

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
/// It does not dispatch. `groups_within` is the same arithmetic the fire path
/// runs, asked of the same modules over the same plans, and a grid that fits
/// here is one no `PastDeviceLimit` can reject there. What a GPU would add is
/// whether the memory exists, which is a different claim and one
/// `max_page_refs` makes separately.
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
    };
    let mods = modules(std::iter::once(&low));

    let mut checked = 0usize;
    let mut widest = 0u32;
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        let Some(declared) = mods.get(symbol) else {
            continue;
        };
        let module = driver_wgpu::geometry::Module::loaded(symbol, declared);
        let Ok(rule) = driver_wgpu::dispatch::rule_of(kernels_wgpu::KERNELS, symbol) else {
            continue;
        };
        let sig = kernels::sig_in(kernels_wgpu::KERNELS, symbol);
        let dims = match sig {
            Some(sig) => driver_wgpu::dispatch::dims_of(sig, &low, launch, geometry),
            None => continue,
        };
        match driver_wgpu::geometry::groups_within(rule, dims, module, LIMIT) {
            Ok(g) => {
                checked += 1;
                widest = widest.max(g.into_iter().max().unwrap_or(0));
            }
            // A rule this backend does not serve is not this test's business;
            // `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
            // is where that lives.
            Err(driver_wgpu::geometry::Ungeometric::Unruled(_)) => {}
            Err(e) => {
                panic!("`{symbol}` at {CLAIMED_TOKENS} tokens, which this seam says it takes: {e}")
            }
        }
    }

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
