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
use model_compiler::trace::FireClass;

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

/// Where the module directory is, when the shaders were built.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

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
}

/// Every symbol the reachable texts launch, and how it must be called.
///
/// Transcribed, so that a text that starts launching something new, or a
/// shader that changes its binding count, is a failure here rather than a
/// surprise in an executor that does not exist yet.
const REACHES: &[(&str, Reaches)] = &[
    ("affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32", Reaches::Push),
    (
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
        Reaches::Push,
    ),
    ("affine_qmv_fast_bfloat16_gs_64_b_4", Reaches::Push),
    ("affine_qmv_fast_residual_bfloat16_gs_64_b_4", Reaches::Push),
    // Two slots, both stated -- the value it biases in place and the bias --
    // and ONE scalar the statement does not carry. An `AddBias` states no
    // params at all, because a bias vector's length is the projection's width
    // and the trace already said that when it sized the output. The row reads
    // it back with `Source::OutWidth(0)`.
    ("add_bias_bfloat16", Reaches::RowDerives(1)),
    ("residual_add_bfloat16", Reaches::Push),
    ("silu_mul_bfloat16", Reaches::Push),
    ("combine_sorted", Reaches::Buffer),
    ("gptoss_swiglu_bfloat16", Reaches::Buffer),
    ("rms_single_row_bfloat16", Reaches::Buffer),
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
    (
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        Reaches::DriverSupplies(1),
    ),
    ("neox_mb_bfloat16", Reaches::DriverSupplies(1)),
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
];

/// Every symbol a real text launches has a module this backend can compile.
///
/// The claim `driver-metal`'s `model_bind` makes for its own table, asked of
/// this one: on both backends an entry point is compiled from a name, so a
/// text that states a symbol the table knows needs no arm written to receive
/// it. Nineteen distinct symbols, and `kernels-vulkan` has a module for all
/// nineteen.
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
    let Some(dir) = SPV_DIR else {
        eprintln!("no modules: build with `--features native` and `glslc` on PATH");
        return;
    };
    let dir = std::path::Path::new(dir);
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
            // the next one reads, and the row then says they are the same
            // allocation. `norm::add_bias` is the only one here, and without
            // this it classifies as a kernel binding one FEWER descriptor
            // than the plan states -- which is true and is not what is
            // interesting about it.
            let args = args
                - u32::try_from(
                    kernels::sig_in(kernels_vulkan::KERNELS, symbol)
                        .map_or(0, |sig| sig.in_place.len()),
                )
                .expect("a row states few aliases");
            let Ok(code) = std::fs::read(dir.join(format!("{symbol}.spv"))) else {
                continue;
            };
            let words = driver_vulkan::spirv::words(&code).expect("a built module is whole words");
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
/// operand, though -- 14948 of them across six texts in both fire classes --
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
        fn named(
            &self,
            _: model_compiler::trace::ValueId,
        ) -> Option<&driver_vulkan::device::Buffer> {
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
    assert_eq!(
        arena_operands, 14948,
        "the texts produced a different number of arena operands than when this \
         was measured, so the zero above is about a different plan"
    );
    // The number with the teeth in it. "Nothing was refused" is satisfied by
    // any range small enough, so it cannot tell a correct extent from a
    // conservative one -- binding a single row rather than the launch's
    // rectangle is refused nowhere and is wrong everywhere, and binding to the
    // end of the arena is refused nowhere and is the exact defect
    // `tests/device.rs` shows corrupting a neighbour. Both change this sum.
    assert_eq!(
        total, 2_368_916_160,
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
        fn named(
            &self,
            _: model_compiler::trace::ValueId,
        ) -> Option<&driver_vulkan::device::Buffer> {
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
    let Some(dir) = SPV_DIR else {
        eprintln!("no modules: build with `--features native` and `glslc` on PATH");
        return;
    };
    let dir = std::path::Path::new(dir);
    let mut pushed = 0u64;
    let mut blocked = 0u64;
    let mut owed: std::collections::BTreeSet<String> = Default::default();

    for (text, low) in texts() {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let Ok(code) = std::fs::read(dir.join(format!("{symbol}.spv"))) else {
                continue;
            };
            let words = driver_vulkan::spirv::words(&code).expect("a built module is whole words");
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
        // params at all. `Source::OutWidth(0)` is how the row says where to
        // read it and `binding::scalars` is what reads it, so this symbol
        // fires today; the five below still owe a Vulkan executor a resource.
        "add_bias_bfloat16",
        "kv_append_paged_bfloat16",
        "neox_mb_bfloat16",
        "row_gather_bfloat16",
        "sdpa_paged_decode_bfloat16_d_128",
        "sdpa_paged_decode_sink_bfloat16_d_64",
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

/// Every launch of every real plan becomes a dispatch, or is one of six
/// symbols already known to be waiting on something nobody has built.
///
/// The other tests here take the binder and the parameter placer apart and
/// ask each its own question. This one puts them back together with the
/// geometry and asks the only question a driver actually has: given a plan,
/// a set of built modules and an arena, how many of its rectangles can be
/// recorded?
///
/// All 6584, across SIX texts in both fire classes. It began at 3180 of 3992
/// over three texts, and the 812 that refused were six symbols short of
/// something nobody had built; each one leaving that list was a defect in
/// this crate rather than a gap in a plan, and the list is now empty.
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
///
/// The exact totals are asserted rather than a "most of them" threshold, and
/// building this walk gave three separate reasons to insist on that:
///
/// * the first version found a kernel row by string equality, when the table
///   states AXES and a plan names POINTS on them. It planned 432 of the 3992
///   this walk had then, and
///   reported "no kernel row" for sixteen symbols that all exist and all have
///   modules built for them. A threshold test would have called that a pass.
/// * the second skipped a launch whose module it could not open, so the
///   denominator itself was wrong by 200 -- it reported 3792 rectangles in a
///   plan that has 3992.
/// * the third checked arity before placing the scalars, and so counted the
///   parameter BLOCK's binding as a missing operand. It refused 1439
///   rectangles across nine symbols that dispatch perfectly well.
#[test]
fn every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal() {
    let Some(dir) = SPV_DIR else {
        eprintln!("no modules: build with `--features native` and `glslc` on PATH");
        return;
    };
    let dir = std::path::Path::new(dir);

    /// An arena big enough that no weight or seam value is what refuses.
    const GENEROUS: u64 = 1 << 30;

    struct Everything(driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Everything {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn named(
            &self,
            _: model_compiler::trace::ValueId,
        ) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        // The KV cache and the fire tables are the driver's own, and this
        // walk is about ORDER, not about where they come from -- so it says
        // yes to all of them and lets the arity check do the work. What the
        // walk would measure otherwise is the absence of a cache allocator,
        // which it is not testing.
        fn kv(&self, _: u16, _: bool) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn table(
            &self,
            _: driver_vulkan::binding::FireTable,
        ) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
    }

    let mut launches = 0u32;
    let mut planned = 0u32;
    let mut refused: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    let mut widest_grid = [0u32; 3];
    let mut workgroups = 0u64;
    let mut pushed = 0u32;
    let mut blocked = 0u32;
    let mut overridden = 0u32;
    let mut split_rectangles = 0u32;
    let mut rotary_overridden = 0u32;
    let mut pool_numbers = 0u32;
    let mut derived_widths = 0u32;
    let mut head_overridden = 0u32;
    let mut heads_overridden = 0u32;

    let mut refused_hollow = 0_u32;
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
            // Read rather than skipped. The version that skipped got the
            // DENOMINATOR wrong, which is worse than getting the numerator
            // wrong: it silently shrank the question.
            let code = std::fs::read(dir.join(format!("{symbol}.spv")))
                .unwrap_or_else(|e| panic!("{text}: no module for `{symbol}`: {e}"));
            let words = driver_vulkan::spirv::words(&code).expect("whole words");
            let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
            let module = driver_vulkan::geometry::Module::named(
                symbol,
                [declared.local[0], declared.local[1], declared.local[2]],
            );
            match driver_vulkan::dispatch::plan_one(
                &low,
                launch,
                kernels_vulkan::KERNELS,
                driver_vulkan::dispatch::Built {
                    module,
                    declared: &declared,
                },
                driver_vulkan::dispatch::Sources {
                    arena,
                    resolver: &store,
                    min_offset: STRICTEST_ALIGNMENT as u64,
                },
                geometry,
            ) {
                Ok(d) => {
                    planned += 1;
                    // The same launch against a geometry of ZEROS, which is
                    // the nastiest input a caller can hand this seam:
                    // fourteen rule-and-field pairs answer a grid with a zero
                    // in it (`geometry.rs` pins the set), and an empty grid
                    // runs nothing and reports success.
                    //
                    // None gets through. The measured reason is not
                    // `plan_one`'s empty-grid check but the row overrides --
                    // a head-shaped row STATES its head width and head count,
                    // so the geometry's zeros are replaced before a rule ever
                    // sees them, and the rows that do not state them refuse
                    // geometrically instead. So this block does not witness
                    // that check; it witnesses that no real rectangle can be
                    // driven to an empty grid from the outside.
                    if let Ok(hollow) = driver_vulkan::dispatch::plan_one(
                        &low,
                        launch,
                        kernels_vulkan::KERNELS,
                        driver_vulkan::dispatch::Built {
                            module,
                            declared: &declared,
                        },
                        driver_vulkan::dispatch::Sources {
                            arena,
                            resolver: &store,
                            min_offset: STRICTEST_ALIGNMENT as u64,
                        },
                        driver_vulkan::dispatch::Geometry::default(),
                    ) {
                        assert!(
                            !hollow.groups.contains(&0),
                            "{text}: `{symbol}` planned {:?} from a geometry of zeros",
                            hollow.groups
                        );
                    } else {
                        refused_hollow += 1;
                    }
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

                    // Whether the STATEMENT overrode the fire's extent. A row
                    // that names `grid_param` says its rule's dimension
                    // varies per layer in a way no fire-wide number can
                    // express, and `dims_of` reading it is the difference
                    // between a driver that normalises over the right width
                    // and one that produces plausible numbers over the wrong
                    // one.
                    let sig = kernels::sig_in(kernels_vulkan::KERNELS, symbol)
                        .expect("the walk found a row");
                    // The VALUE a row states at that slot, not merely that
                    // it states one. `!=` against a lying fire proves the
                    // answer did not come from the fire; it does not prove it
                    // came from the row, and an override off by one survived
                    // exactly that gap.
                    let value = |index: Option<u8>| -> Option<u32> {
                        let i = index?;
                        let at = launch.params.start as usize + i as usize;
                        (at < launch.params.end as usize)
                            .then(|| low.params.get(at).copied().unwrap_or(0))
                            .filter(|n| *n > 0)
                    };
                    let states = |index: Option<u8>| {
                        index.is_some_and(|i| {
                            let at = launch.params.start as usize + i as usize;
                            at < launch.params.end as usize
                                && low.params.get(at).copied().unwrap_or(0) > 0
                        })
                    };
                    if states(sig.grid_param) {
                        overridden += 1;
                    }
                    if states(sig.head_param) {
                        head_overridden += 1;
                    }
                    if states(sig.heads_param) {
                        heads_overridden += 1;
                    }

                    // THE POOL'S NUMBERS REACH THE SHADER, checked here
                    // because nothing else was checking it. A row may name a
                    // number that belongs to the pool rather than to the
                    // statement -- the KV page size and the cache's two
                    // strides -- and `binding::scalars` interleaves the
                    // driver's answer into the run at the row's position.
                    //
                    // Replacing either stride with 999 left the whole suite
                    // green. The walk's resolver answers `None` to every
                    // number, so all three read as zero here, and the device
                    // tests that DO know the strides hand-write their push
                    // constants and never go through this path. Between them
                    // the seam from `Pool` to the params was unwatched, and a
                    // wrong stride is not an error -- it is attention reading
                    // the wrong offsets and returning numbers.
                    //
                    // A second resolver with three recognisable answers, run
                    // beside the real one so no pinned count moves. Distinct
                    // per number, so a swap fails as loudly as a drop.
                    struct Sentinels(driver_vulkan::device::Buffer);
                    impl driver_vulkan::binding::Resolve for Sentinels {
                        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
                            Some(&self.0)
                        }
                        fn named(
                            &self,
                            _: model_compiler::trace::ValueId,
                        ) -> Option<&driver_vulkan::device::Buffer> {
                            Some(&self.0)
                        }
                        fn number(&self, which: driver_vulkan::binding::FireNumber) -> Option<u32> {
                            Some(match which {
                                driver_vulkan::binding::FireNumber::KvPageSize => 0x0011_1111,
                                driver_vulkan::binding::FireNumber::KvHeadStride => 0x0022_2222,
                                driver_vulkan::binding::FireNumber::KvSeqStride => 0x0033_3333,
                            })
                        }
                    }
                    let wants = |src: kernels::Source| sig.operands.iter().any(|o| o.source == src);
                    let sentinel_of = |src: kernels::Source| match src {
                        kernels::Source::KvPageSize => 0x0011_1111u32,
                        kernels::Source::KvHeadStride => 0x0022_2222,
                        _ => 0x0033_3333,
                    };
                    for src in [
                        kernels::Source::KvPageSize,
                        kernels::Source::KvHeadStride,
                        kernels::Source::KvSeqStride,
                    ] {
                        if !wants(src) {
                            continue;
                        }
                        pool_numbers += 1;
                        let store = Sentinels(driver_vulkan::device::Buffer::placeholder(GENEROUS));
                        let got =
                            driver_vulkan::binding::scalars(sig, &low, launch, &declared, &store)
                                .expect("the row's scalars place");
                        let bytes = match got {
                            driver_vulkan::binding::Params::Push(ref b) => b.clone(),
                            driver_vulkan::binding::Params::Block { ref bytes, .. } => {
                                bytes.clone()
                            }
                            driver_vulkan::binding::Params::None => Vec::new(),
                        };
                        let want = sentinel_of(src).to_le_bytes();
                        assert!(
                            bytes.windows(4).any(|w| w == want),
                            "{text}: `{symbol}` names {src:?} and the driver's answer is not in \
                             the {} bytes it hands the shader",
                            bytes.len()
                        );
                    }

                    // THE ROW'S DERIVED WIDTH REACHES THE SHADER, for the
                    // same reason the pool's numbers are checked above and
                    // with the same shape of check. `Source::OutWidth(0)` is
                    // a number NOTHING in the statement carries -- an
                    // `AddBias` states no params at all -- so if
                    // `binding::scalars` read the wrong output, or dropped
                    // the source and left the run short, the module would get
                    // a zero and every lane would return before writing.
                    //
                    // That failure is silent in the direction that matters:
                    // a bias never added is a projection missing a small
                    // constant, which stays fluent. So the width is asserted
                    // to be IN the bytes, and it is the width the plan states
                    // for the launch's own output rather than a constant this
                    // test knows.
                    if sig
                        .operands
                        .iter()
                        .any(|o| matches!(o.source, kernels::Source::OutWidth(_)))
                    {
                        derived_widths += 1;
                        let store = Sentinels(driver_vulkan::device::Buffer::placeholder(GENEROUS));
                        let got =
                            driver_vulkan::binding::scalars(sig, &low, launch, &declared, &store)
                                .expect("the row's scalars place");
                        let bytes = match got {
                            driver_vulkan::binding::Params::Push(ref b) => b.clone(),
                            driver_vulkan::binding::Params::Block { ref bytes, .. } => {
                                bytes.clone()
                            }
                            driver_vulkan::binding::Params::None => Vec::new(),
                        };
                        let width = low.args[launch.args.start as usize..launch.args.end as usize]
                            .iter()
                            .filter_map(|a| match a {
                                model_compiler::lower::Arg::Arena { width, .. }
                                | model_compiler::lower::Arg::Named { width, .. } => Some(*width),
                                model_compiler::lower::Arg::Weight(_) => None,
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

                    // Counting rows that STATE a head shape does not witness
                    // `dims_of` USING it, and the difference is not academic:
                    // deleting either override left this whole file green,
                    // because across these six texts the stated value
                    // equals the fire's and the two lines are no-ops. The
                    // model that separates them is gemma-4, which is not one
                    // of the texts here.
                    //
                    // So the fire is made to disagree on purpose. A geometry
                    // carrying head shapes nothing states is handed to the
                    // same launch, and `dims_of` must still answer with what
                    // the row said. Now deleting an override is a failure.
                    let liar = driver_vulkan::dispatch::Geometry {
                        head_dim: geometry.head_dim + 7,
                        kv_heads: geometry.kv_heads + 7,
                        rotary_dims: geometry.rotary_dims + 1024,
                        ..geometry
                    };
                    let told = driver_vulkan::dispatch::dims_of(sig, &low, launch, liar);
                    if states(sig.head_param) {
                        assert_eq!(
                            Some(told.head_dim),
                            value(sig.head_param),
                            "{symbol} states a head width and `dims_of` answered with something else"
                        );
                    }
                    if states(sig.heads_param) {
                        assert_eq!(
                            Some(told.kv_heads),
                            value(sig.heads_param),
                            "{symbol} states a head count and `dims_of` answered with something else"
                        );
                    }
                    // The same no-op, one field over. A rope row's rotary
                    // width comes from the STATEMENT because a model may
                    // rotate only part of its head -- gemma-4 turns 128 of
                    // 512 -- and for these six texts the stated width
                    // equals the fire's, so dropping the override changed
                    // nothing anywhere. 400 rope rectangles state a grid and
                    // every one of them is now answered from the row.
                    if matches!(sig.launch, kernels::LaunchRule::SplitPacked) {
                        split_rectangles += 1;
                    }
                    if states(sig.grid_param) && matches!(sig.launch, kernels::LaunchRule::Rope) {
                        rotary_overridden += 1;
                        assert_eq!(
                            Some(told.rotary_dims),
                            value(sig.grid_param),
                            "{symbol} states a grid and `dims_of` answered with a different \
                             rotary width"
                        );
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
                                "{text}: `{symbol}` pushes {} bytes into a block ending at {end}",
                                b.len()
                            );
                        }
                        driver_vulkan::binding::Params::Block { ref bytes, at } => {
                            blocked += 1;
                            assert!(
                                at < declared.bindings as usize,
                                "{text}: `{symbol}` puts its block at {at} of {}",
                                declared.bindings
                            );
                            assert!(!bytes.is_empty());
                        }
                        driver_vulkan::binding::Params::None => {}
                    }

                    // The operands, PLUS the slot a parameter block takes,
                    // are the module's real bindings -- the layout less its
                    // holes, which is the arity `Device::run` enforces. If
                    // these two ever disagree, every dispatch this walk
                    // produces would be refused at the device, which is the
                    // kind of break that only shows up on a machine with a
                    // GPU in it.
                    //
                    // The block slot is a binding the PLAN never mentions:
                    // six reachable symbols read their scalars from a struct
                    // in a buffer, and `router_topk` states three operands
                    // against a four-binding module.
                    let block = usize::from(matches!(
                        d.params,
                        driver_vulkan::binding::Params::Block { .. }
                    ));
                    assert_eq!(
                        d.buffers.len() + block,
                        declared.bindings as usize - declared.holes(),
                        "{text}: `{symbol}` bound {} plus {block} for {} real bindings",
                        d.buffers.len(),
                        declared.bindings as usize - declared.holes()
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

    // Nothing is refused, so nothing is named. Every rectangle all six
    // texts state, in both fire classes, becomes a dispatch.
    //
    // This list held six symbols and then five, and each one leaving it was a
    // defect in this crate rather than a gap in a plan.
    //
    // `embed_gather_mb_4bit` went when binding stopped being positional: its
    // row states `TokenIds`, which the driver owns, so four operands against
    // five bindings was never short of anything.
    //
    // `kv_append_paged`, `neox_mb` and `row_gather` went when the scalar run
    // started being built from the ROW instead of taken whole. A row indexes
    // into the statement's run and may use only part of it -- `neox_mb` reads
    // three of four -- and it interleaves numbers the driver resolves, so
    // `kv_append_paged`'s page size lands BETWEEN its two stated scalars.
    //
    // The two paged decodes went last and twice over: first their page size,
    // then their grid. A module compiled for 128-wide heads cannot serve the
    // 1-wide default, and nothing but the driver knows the model's head
    // dimension -- a plan states rows and widths. That refusal is the first
    // time `head_param` fired at all.
    let expected: Vec<String> = Vec::new();
    let got: Vec<String> = refused.keys().cloned().collect();
    assert_eq!(got, expected, "a different set of rectangles refuses");
    assert_eq!(
        refused.values().sum::<u32>(),
        launches - planned,
        "the refusals and the successes do not account for every rectangle"
    );

    // Both halves of the parameter split are exercised. Stated because a
    // change that routed everything one way would leave the other half
    // untested while this test still passed.
    assert!(
        pushed > 0 && blocked > 0,
        "{pushed} pushed, {blocked} blocked"
    );
    // Stated exactly, because the fallback is silent: a row whose stated
    // extent went missing would take the fire's number and normalise over the
    // wrong width, producing numbers rather than an error, and this count
    // going to zero is the only thing that would say so.
    //
    // Was 710, all of them `rms_single_row`'s reduction axis. The other 400
    // are the two paged decodes, which could not be planned at all until the
    // walk started stating the model's geometry.
    assert_eq!(
        overridden, 1788,
        "a different number of rectangles states its own extent"
    );
    // `head_param` and `heads_param` fired ZERO times for as long as the walk
    // handed `plan_one` a default geometry, and `dims_of`'s two other
    // overrides were carried untested on a green suite. They fire now, and
    // the number is stated so that it going back to zero is visible rather
    // than comfortable.
    //
    // Counted separately, and that was not a tidying. One count of the UNION
    // was 600 -- and `head_param` alone is 600 while `heads_param` is 200, so
    // the rows stating a head count are a strict subset of those stating a
    // head width. The union could not have witnessed `heads_param` at all:
    // it going to zero would have left the number unchanged and the suite
    // green. Two numbers, so each override answers for itself.
    // Stated so that the seam going unexercised is visible. It was zero for
    // as long as nothing asked, which is how it stayed broken-able.
    //
    // All 400 are `KvPageSize`, in the two paged decodes and the paged
    // append. NOT ONE names a stride: these six texts are paged throughout,
    // and the strides belong to the contiguous cache. That is why the loop
    // above cannot be the whole check, and why the rows themselves are swept
    // separately below.
    // How many of those same launches `plan_one` REFUSED when handed a
    // geometry of ZEROS -- 352 of 1788, the head-shaped ones. The rest plan
    // fine because their dimensions come from the lowering rather than from
    // the geometry, so the zeros never reach them, and they are right to.
    //
    // Every one of those refusals is `Ungeometric` -- 193 of them from the
    // two paged decodes, whose module width cannot match a zero head, and the
    // rest from rules that need a shape the zeros destroyed. NOT ONE is
    // `Undispatchable::Empty`, which is the measurement this comment exists
    // for: deleting `plan_one`'s `groups.contains(&0)` check still leaves this
    // file green, because nothing reachable from a real plan and an arbitrary
    // geometry produces an empty grid. That check is witnessed by
    // `tests/device.rs` alone, where a zero grid is recorded and the card
    // notices. Said here so the next reader does not mistake this number for
    // a check of the refusal.
    assert_eq!(
        refused_hollow, 352,
        "a different number of launches refused a geometry of zeros"
    );
    assert_eq!(
        pool_numbers, 704,
        "a different number of rectangles names one of the pool's numbers"
    );
    // Two texts, three biases a layer, both fire classes: qwen2.5's 28
    // layers make 168 and gpt-oss's 24 make 144. Pinned rather than asserted
    // as non-zero, because the failure this witnesses is a bias that is not
    // added, and a text that quietly stopped stating one would leave the
    // check above passing over nothing.
    //
    // Was qwen2.5 alone until gpt-oss started stating its attention biases
    // too, and the 144 that arrived needed nothing here: an `AddBias` states
    // no scalars whatever states it, so the row deriving the width was
    // already the whole of it.
    assert_eq!(
        derived_widths, 312,
        "a different number of rectangles names a width the row derives"
    );
    // ZERO, and asserted as zero because that is the fact `dims_of`'s
    // `in_width` note rests on: no text here reaches `split_qkv`, so nothing
    // consumes `in_width` and replacing it with a constant is invisible. A
    // text that starts splitting its projections makes this number move,
    // which is when the note stops being true.
    assert_eq!(
        split_rectangles, 0,
        "a text now reaches `Rule::SplitPacked`, so `in_width` is no longer unwitnessed"
    );
    assert_eq!(
        rotary_overridden, 704,
        "a different number of rope rectangles states its rotary width"
    );
    assert_eq!(
        head_overridden, 1056,
        "a different number of rectangles states a head width"
    );
    assert_eq!(
        heads_overridden, 352,
        "a different number of rectangles states a head count"
    );
    // The total work these plans dispatch, as a single number.
    //
    // Here because every other assertion in this test is about SHAPE -- how
    // many operands, where the scalars went, that no grid is zero -- and a
    // grid can be the wrong size while being all of those things. Dropping
    // `dims_of`'s statement override changes no other assertion in this file
    // and changes this one, which is the whole reason it is stated.
    assert_eq!(
        workgroups, 35_504_450,
        "the plans dispatch a different amount of work"
    );
    // The third dimension was 1 across every text until the paged decodes
    // could be planned: they are the only rows that put anything on z, and
    // 64 is a prefill's rows. A backend that flattened the grid to two
    // dimensions would have looked correct on every plan before this one.
    assert_eq!(
        widest_grid,
        [3584, 25136, 64],
        "the widest grid in any dimension changed"
    );
}

/// Every row that names one of the pool's numbers is handed that number, and
/// the three are not interchangeable.
///
/// The walk above can only ask this of rows its six texts reach, and they
/// reach exactly one of the three numbers: all 400 are `KvPageSize`, because
/// these texts are paged throughout and the two strides belong to the
/// contiguous cache. So the strides went unwatched -- replacing either with a
/// constant left the whole suite green, and a wrong stride is not an error but
/// attention reading the wrong offsets and returning numbers.
///
/// `Pool`'s answers are checked in `resources.rs` and the shader's addressing
/// in `tests/device.rs`, which hand-writes its push constants. Between those
/// two the seam was open. This closes it from the table's side rather than a
/// text's, so a row added tomorrow is covered whether or not a text reaches
/// it.
#[test]
fn every_row_naming_a_pool_number_is_handed_that_number_and_not_another() {
    use kernels::Source;

    // Distinct and recognisable, so a swap fails as loudly as a drop. A
    // resolver answering one number for all three would pass a check that
    // only asked "is something there".
    fn sentinel(src: Source) -> Option<u32> {
        match src {
            Source::KvPageSize => Some(0x0011_1111),
            Source::KvHeadStride => Some(0x0022_2222),
            Source::KvSeqStride => Some(0x0033_3333),
            _ => None,
        }
    }
    struct Sentinels(driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Sentinels {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn named(
            &self,
            _: model_compiler::trace::ValueId,
        ) -> Option<&driver_vulkan::device::Buffer> {
            Some(&self.0)
        }
        fn number(&self, which: driver_vulkan::binding::FireNumber) -> Option<u32> {
            Some(match which {
                driver_vulkan::binding::FireNumber::KvPageSize => 0x0011_1111,
                driver_vulkan::binding::FireNumber::KvHeadStride => 0x0022_2222,
                driver_vulkan::binding::FireNumber::KvSeqStride => 0x0033_3333,
            })
        }
    }

    let store = Sentinels(driver_vulkan::device::Buffer::placeholder(1 << 30));
    // A real lowering, borrowed for its shape and then emptied of scalars.
    // Hand-building a `Lowered` would be a second definition of the type to
    // keep in step for no gain -- nothing below reads any field but `params`.
    let mut low = texts().swap_remove(0).1;
    let mut rows = 0u32;
    let mut named = std::collections::BTreeMap::<String, u32>::new();
    for sig in kernels_vulkan::KERNELS {
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

        // A launch whose statement carries as many scalars as the row's
        // `Param` slots index, and a module wide enough to hold the run --
        // the question here is the RUN's contents, not its placement, which
        // `binding`'s own tests already pin.
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
        let launch = model_compiler::lower::Launch {
            kernel: 0,
            rows: 0..1,
            layers: 0..1,
            op: 0,
            args: 0..0,
            params: 0..params,
            peel: None,
            cond: 0,
        };
        // A push block of EXACTLY the run's length, because `binding`
        // refuses a run a block cannot hold rather than truncating it -- see
        // `scalars_neither_shape_can_hold_are_refused_rather_than_truncated`.
        // The length is the row's own: one word per operand that carries a
        // scalar, plus the tail a `Buf` at a `Param` slot swallows.
        let words = run_words(sig, params);
        let declared = driver_vulkan::spirv::Declared {
            local: [1, 1, 1],
            bindings: 1,
            used: vec![true],
            reads_workgroup_count: false,
            grid_axes: [true, false, false],
            push_offsets: (0..words).map(|i| i * 4).collect(),
            block_bytes: vec![None],
        };
        let got = driver_vulkan::binding::scalars(sig, &low, &launch, &declared, &store)
            .unwrap_or_else(|e| panic!("`{}`: {e:?}", sig.symbol));
        let bytes = match got {
            driver_vulkan::binding::Params::Push(ref b) => b.clone(),
            driver_vulkan::binding::Params::Block { ref bytes, .. } => bytes.clone(),
            driver_vulkan::binding::Params::None => Vec::new(),
        };
        // The WHOLE run, not "is the number in there somewhere". Presence
        // alone cannot see a swap: both strides are in the same run either
        // way, and swapping them is the defect with the most plausible
        // output -- attention striding by heads where it should stride by
        // positions, reading real numbers from the wrong rows.
        let want: Vec<u8> = expected_run(sig, &low, params)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            bytes, want,
            "`{}` hands the shader a different run than its row spells",
            sig.symbol
        );
        for src in wanted {
            *named.entry(format!("{src:?}")).or_default() += 1;
        }
    }

    // Every one of the three is witnessed by at least one row. Stated because
    // the loop passes vacuously for a number no row names, which is precisely
    // the state the strides were in.
    assert_eq!(named.len(), 3, "only these are witnessed: {named:?}");
    // Stated exactly. Six rows in the whole table name one of these, and a
    // row losing its source is not an error anywhere -- the number simply
    // stops being written and the shader reads whatever the statement left
    // in that slot.
    assert_eq!(rows, 6, "a different number of rows names a pool number");
    assert_eq!(
        named,
        [
            ("KvHeadStride".to_string(), 5u32),
            ("KvPageSize".to_string(), 3),
            ("KvSeqStride".to_string(), 5),
        ]
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>(),
        "a different set of rows names each number"
    );
}

/// How many words the run for one row is, given a statement of `params`
/// scalars.
///
/// Mirrors `binding::scalars`' walk rather than guessing, because the two
/// disagreeing is the failure this file cannot see: a block sized from a guess
/// would refuse rows that are fine and pass rows that are not.
fn run_words(sig: &kernels::KernelSig, params: u32) -> u32 {
    let mut n = 0u32;
    for o in sig.operands {
        match o.source {
            _ if o.ty == kernels::Ty::InPacked => n += 1,
            kernels::Source::KvPageSize
            | kernels::Source::KvHeadStride
            | kernels::Source::KvSeqStride => n += 1,
            kernels::Source::Param(i) | kernels::Source::ParamF32(i) => {
                if matches!(o.ty, kernels::Ty::Buf | kernels::Ty::BufMut) {
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
/// A second reading of the same table. That is the point: the check it feeds
/// is that `binding::scalars` and the row agree, and a helper that asked
/// `binding` would only be checking it against itself.
fn expected_run(
    sig: &kernels::KernelSig,
    low: &model_compiler::lower::Lowered,
    params: u32,
) -> Vec<u32> {
    let stated: Vec<u32> = (0..params).map(|i| 1000 + i).collect();
    let mut run = Vec::new();
    for o in sig.operands {
        if o.ty == kernels::Ty::InPacked {
            run.push(match o.source {
                kernels::Source::RequestCount => low.n_requests,
                _ => 0,
            });
            continue;
        }
        match o.source {
            kernels::Source::KvPageSize => run.push(0x0011_1111),
            kernels::Source::KvHeadStride => run.push(0x0022_2222),
            kernels::Source::KvSeqStride => run.push(0x0033_3333),
            kernels::Source::Param(i) | kernels::Source::ParamF32(i) => {
                if matches!(o.ty, kernels::Ty::Buf | kernels::Ty::BufMut) {
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
