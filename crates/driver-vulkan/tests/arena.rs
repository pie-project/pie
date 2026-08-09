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

/// The same texts, each with the model geometry its plans were built from.
///
/// Split out because most of this file never needs it and one test does. A
/// head-width rule cannot be answered from a plan alone: `sdpa_paged_decode`
/// is compiled for a fixed head dimension and the plan states rows and
/// widths, not heads. A driver knows the model; a plan does not.
fn geometric() -> Vec<(String, Lowered, driver_vulkan::dispatch::Geometry)> {
    let mut out = Vec::new();
    for (name, facts, metal) in [
        (
            "qwen3_0_6b",
            LlamaLikeFacts::qwen3_0_6b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            "gpt_oss_20b",
            LlamaLikeFacts::gpt_oss_20b(),
            LlamaLikeMetalFacts::gpt_oss_20b(),
        ),
        (
            "qwen3_30b_a3b",
            LlamaLikeFacts::qwen3_30b_a3b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
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
/// operand, though -- 9618 of them across three texts in both fire classes --
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
        arena_operands, 9618,
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
        total, 1_057_900_480,
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
    // Five, not seven -- and the two that fall out are the useful part.
    //
    // `affine_qmv_routed` states five scalars and its module's push block
    // holds exactly five; `embed_gather_mb_4bit` states one and holds one.
    // What those two are short of is a DESCRIPTOR, not a parameter, so their
    // scalars are fully the plan's and only a buffer is owed. Being short of
    // one thing does not make a kernel short of the other, and treating
    // "binds more than the plan names" as one bucket would have hidden that.
    //
    // The remaining five are short of both, which is what a paged KV cache
    // and its page table look like from here: the driver owns the resource,
    // so it also owns the numbers describing it.
    let want: std::collections::BTreeSet<String> = [
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
    // Every one of them is also short of descriptors. The reverse does not
    // hold, which is the finding above.
    for symbol in &owed {
        assert!(
            REACHES
                .iter()
                .any(|(n, r)| n == symbol && matches!(r, Reaches::DriverSupplies(_))),
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
/// 3180 of 3992. The other 812 are the six symbols the earlier tests already
/// record as short of something nobody has built -- `kv_append_paged` and the
/// two paged-decode attentions want a KV cache and a page table this crate
/// does not allocate, `neox_mb` and `row_gather` want scalars a driver
/// derives, `embed_gather_mb_4bit` wants a descriptor the plan does not name.
/// Nothing NEW refuses, which is the claim: joining three layers that each
/// pass alone did not introduce a fourth failure.
///
/// The exact totals are asserted rather than a "most of them" threshold, and
/// building this walk gave three separate reasons to insist on that:
///
/// * the first version found a kernel row by string equality, when the table
///   states AXES and a plan names POINTS on them. It planned 432 of 3992 and
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
    let mut heads_overridden = 0u32;

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
                    if states(sig.head_param) || states(sig.heads_param) {
                        heads_overridden += 1;
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
        launches, 3992,
        "a different number of rectangles is lowered"
    );
    assert_eq!(planned, 3992, "a different number of rectangles records");

    // Nothing is refused, so nothing is named. Every rectangle all three
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
        overridden, 1110,
        "a different number of rectangles states its own extent"
    );
    // `head_param` and `heads_param` fired ZERO times for as long as the walk
    // handed `plan_one` a default geometry, and `dims_of`'s two other
    // overrides were carried untested on a green suite. They fire now, and
    // the number is stated so that it going back to zero is visible rather
    // than comfortable.
    assert_eq!(
        heads_overridden, 600,
        "a different number of rectangles states a head shape"
    );
    // The total work these plans dispatch, as a single number.
    //
    // Here because every other assertion in this test is about SHAPE -- how
    // many operands, where the scalars went, that no grid is zero -- and a
    // grid can be the wrong size while being all of those things. Dropping
    // `dims_of`'s statement override changes no other assertion in this file
    // and changes this one, which is the whole reason it is stated.
    assert_eq!(
        workgroups, 23_025_115,
        "the plans dispatch a different amount of work"
    );
    // The third dimension was 1 across every text until the paged decodes
    // could be planned: they are the only rows that put anything on z, and
    // 64 is a prefill's rows. A backend that flattened the grid to two
    // dimensions would have looked correct on every plan before this one.
    assert_eq!(
        widest_grid,
        [2048, 25136, 64],
        "the widest grid in any dimension changed"
    );
}
