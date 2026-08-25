//! What the compiled modules declare, read out of the SPIR-V.
//!
//! This file was the geometry sweep: [`driver_vulkan::geometry`] divided a
//! thread extent by a workgroup size, the extent came from a table row's
//! `Rule`, the size came from `[numthreads(...)]` that only `slangc` can see,
//! and nothing made them agree except the twelve sweeps that stood here.
//!
//! The rows are gone. A routine states its own grid in its own body, so there
//! is no `Rule` to hold up against a module until a rectangle is planned --
//! and planning rectangles is `tests/arena.rs`'s job, over 6,680 of them that
//! a real model states. `every_module` below names each retired sweep and what
//! took it over, because a sweep whose reference implementation retires does
//! not become wrong, it becomes BLIND, and a blind sweep that still passes is
//! worse than no sweep at all.
//!
//! What is left is the part that never needed a row: three walks that read the
//! shader tree and compare it to what this crate says about it. The workgroup
//! width every module bets on, the parameter blocks it measures, and the
//! binding holes it calls holes.
//!
//! No GPU is involved. The modules are `kernels-vulkan`'s embedded table.

use driver_vulkan::{Declared, spirv};

/// Skip with a reason when there are no modules, rather than pass silently.
/// **THE SKIP IS A FACT, SO IT IS ASSERTED RATHER THAN PRINTED.**
///
/// Every sweep below opens with `modules!()`, which returns early when
/// `kernels-vulkan` embedded no SPIR-V. That is the honest thing for a sweep
/// over shader modules to do — but `cargo test` hides stderr, so the whole
/// file reported `5 passed` in a configuration where it read nothing at all,
/// and reported `5 passed` in the configuration where it read 496 modules,
/// and the two are indistinguishable from the outside. The suite ran for
/// months in the first one: `native` was the only route to a module table and
/// `native` does not compile.
///
/// So the emptiness gets a test of its own. Nothing here can be vacuous
/// without this failing, which is the difference between a suite that skips
/// and a suite that is merely absent.
#[test]
fn the_module_table_is_empty_only_when_nothing_was_asked_to_fill_it() {
    let n = kernels_vulkan::MODULES.len();
    if cfg!(feature = "device") {
        assert_eq!(
            n, 698,
            "`device` is on, so `kernels-vulkan` was asked for its SPIR-V and \
             the table holds {n} rather than 698. If a shader was ADDED or a \
             variant instantiated, raise this number — that edit is the \
             record. If it FELL, the sweeps below are reading less than they \
             read yesterday and every one of them still says `ok`, which is \
             the failure this test exists for. A count near zero means \
             `slangc` did not run at all.",
        );
    } else {
        assert_eq!(
            n, 0,
            "`device` is off and the table holds {n} module(s). The sweeps \
             below are then reading a table nobody stated they wanted, which \
             means the feature no longer says what compiles the shaders.",
        );
    }
}

macro_rules! modules {
    () => {
        if !kernels_vulkan::embedded() {
            eprintln!(
                "no modules: build with `-p driver-vulkan --features native` \
                 (or any profile that pulls kernels-vulkan/native) and have \
                 `slangc` on PATH"
            );
            return;
        }
    };
}

/// What a module declares, read through the crate's own loader.
///
/// The tests used to carry a second SPIR-V parser. Two parsers of the same
/// bytes is two things to keep right, and the one under test is the one that
/// matters -- so this calls it, and a defect in it now fails these sweeps
/// instead of being masked by a copy that happens to agree.
fn declared(code: &[u8]) -> Declared {
    let words = spirv::words(code).expect("a built module is whole words");
    spirv::declared(&words).expect("a built module is well formed")
}

/// Every compiled module, by entrypoint name, with what it declares.
///
/// `table()` STOOD HERE. It paired each name with `kernels::sig_in(KERNELS,
/// ..)`, so every sweep in this file was really a comparison of two columns:
/// a `Rule` a table row stated, and a `[numthreads(..)]` the module baked in.
/// `kernels_vulkan::KERNELS` is empty -- a routine computes its own grid in
/// its own body now, and there is no `Rule` to compare against until a
/// rectangle is actually planned. So the pairing does not become wrong; it
/// becomes IMPOSSIBLE, and the twelve sweeps that rested on it were deleted
/// rather than left to pass over an empty vector. Say which, and what took
/// over, in the same edit:
///
/// * `every_entrypoint_is_launched_over_its_whole_extent`,
///   `a_module_that_reads_its_workgroup_count_is_launched_exactly`,
///   `only_the_rules_that_are_allowed_to_vary_do`,
///   `an_awkward_shape_still_covers_its_tail` and
///   `every_rule_the_table_names_is_one_this_driver_can_lay_out` asked whether
///   `geometry::groups(rule, dims, module)` covered a stated extent. TAKEN
///   OVER BY `tests/arena.rs`, which walks 6,680 real rectangles through
///   `serve::plan_routine` and records the workgroups every one of them
///   dispatches -- the same arithmetic, over shapes a model states rather than
///   the one 64x4096 fire this file invented.
/// * `no_rule_puts_work_on_an_axis_its_shader_never_reads`,
///   `no_module_reads_a_grid_axis_its_rule_leaves_flat` and
///   `a_shader_indexed_by_an_axis_is_given_that_axis` are the three that found
///   REAL defects (`Rule::Rms` stacking rows on a `y` that `norm/rms.slang`
///   never read; `geglu_tanh_strided` reading `gl_GlobalInvocationID.y` under
///   a flat rule). MOVED TO `tests/arena.rs`, asserted per SYMBOL over the
///   whole walk rather than per rectangle -- a real plan cannot force
///   `rows: 64` the way this file did, so a single rectangle being flat on an
///   axis proves nothing and only the symbol never once using it does.
/// * `no_row_declares_fewer_buffers_than_its_module_binds`,
///   `the_push_block_a_row_packs_is_the_one_its_module_declares`,
///   `every_row_packs_and_every_scalar_lands_on_its_own_field` and
///   `a_rows_writable_buffers_are_the_ones_its_module_may_write` compared
///   `KernelSig.operands` against a module's bindings. TAKEN OVER BY the
///   `binding.rs` and `encode.rs` unit tests, which check the same ABI against
///   the arms that state it, and by `tests/device.rs`, which submits it.
///
/// What survives here is what never needed a row: three sweeps that read the
/// modules and nothing else.
fn every_module() -> Vec<(String, Declared)> {
    kernels_vulkan::entrypoints()
        .into_iter()
        .filter_map(|name| {
            let code = kernels_vulkan::code(&name, kernels_vulkan::Capability::Baseline)?;
            Some((name, declared(code)))
        })
        .collect()
}

///
/// `maxComputeWorkGroupInvocations` is guaranteed to be at least 128 and no
/// more; anything above that is a device that happens to allow it. A module
/// wider than 128 is a deliberate bet on the hardware, so the ones that make it
/// are counted rather than waved through — if the count moves, someone widened
/// a shader without deciding to.
#[test]
fn a_module_wider_than_the_guaranteed_floor_is_a_deliberate_bet() {
    modules!();
    let mut over = std::collections::BTreeMap::new();
    let mut checked = 0usize;
    for (name, d) in every_module() {
        checked += 1;
        let invocations = d.local[0] * d.local[1] * d.local[2];
        assert!(
            invocations <= 1024,
            "`{name}` wants {invocations} invocations per workgroup, past the \
             1024 that is the most any device in this tree reports"
        );
        if invocations > 128 {
            *over.entry(invocations).or_insert(0usize) += 1;
        }
    }
    // 1024 is the router's lane-per-expert sort; 512 and 256 are the wide
    // SDPA head dimensions and the pointwise family.
    assert!(
        over.keys().all(|n| [256, 512, 1024].contains(n)),
        "a module is wider than the guaranteed 128 at an unexpected size: {over:?}"
    );
    // The sweep used to be filtered by a table row, so a table that emptied
    // would have left this passing over nothing. It walks the directory now,
    // and says how far it walked.
    assert!(
        checked > 400,
        "only {checked} modules were measured; the shader tree is 480 wide"
    );
}

/// Every parameter block this tree declares, and its size.
///
/// A transcribed table rather than a derivation, for the reason
/// `driver-metal`'s `packed_params_cover_the_struct` gives: a check whose
/// expectation is computed the same way as the thing it checks agrees with
/// itself. These numbers were read off the compiled modules by an independent
/// SPIR-V walk written in another language.
///
/// # It was 45 rows, and 38 of them left in one edit
///
/// A parameter block used to be how nearly every launch in this tree got its
/// scalars: a struct in a storage buffer, bound after the data, filled by the
/// host. The shaders moved theirs into `[[vk::push_constant]]` ranges instead,
/// and a scalar that travels in a push range is not a binding at all -- so 38
/// of these rows did not shrink or move, they stopped existing.
///
/// That is a real change and not a lost measurement, and the difference
/// matters enough to say how it was told apart. A block that vanished because
/// a walk stopped finding blocks would leave the module still declaring it;
/// these modules declare a PUSH RANGE where the block used to be, which
/// `the_push_ranges_are_where_the_parameter_blocks_went` below counts. The
/// two sweeps are the two halves of one census, and 38 rows leaving one of
/// them while the other holds 669 modules is the shape a MOVE has.
///
/// Two of the departed -- `router_topk` at 16 bytes and `combine_sorted` at 12
/// -- were the same two blocks the Metal driver was found packing short, which
/// is the sort of defect this table exists for and is worth keeping the name
/// of even though the rows are gone.
///
/// # Then the six `rms_rope` rows left the same way
///
/// They are the 39th through 44th, and they left after the paragraph above
/// was written -- which is why that paragraph is kept: it said at length why
/// the fused norm+rope KEPT a buffer, and the reason it gave has since stopped
/// being true. `rms_rope.slang` declares `struct Push` and a
/// `[[vk::push_constant]] ConstantBuffer<Push>` where `RmsRopeParams` on
/// binding 2 used to be, and `position` moved up into the binding the block
/// vacated.
///
/// The move is recorded rather than inferred, by
/// [`the_fused_norm_rope_moved_its_block_into_a_push_range`] below: all six
/// declare exactly nine four-byte push fields at offsets 0 through 32 -- the
/// nine `RmsRopeParams` scalars, in order, whole -- and no parameter block at
/// any binding. Nine fields arriving as six blocks leave is a MOVE; a block
/// that vanished from a walk that stopped walking would leave nothing behind
/// it.
///
/// # What is left, and why it kept a buffer
///
/// `argmax_logits` at 40 bytes, alone. It is over the 128-byte floor no device
/// may go under, so this is not a size question -- it is the one launch whose
/// scalars `binding::params` places in a buffer, because it places a launch's
/// scalars in a push range OR a parameter buffer and never both, and this one
/// needs the buffer for what else it carries.
///
/// A table of one row is a table that has nearly finished being needed, and
/// the count assertion is what makes it still worth having: it is now the
/// claim that no module has GROWN a block, which is the direction the port is
/// not going and therefore the direction a mistake would show up in.
const PARAM_BLOCKS: &[(&str, u32, u32)] = &[("argmax_logits_bfloat16", 2, 40)];

/// The six `rms_rope` blocks did not vanish, they became push ranges.
///
/// Stated as its own test because `PARAM_BLOCKS` losing six rows and the push
/// census gaining six modules are, separately, both consistent with a walk
/// that broke: the block sweep would find nothing and the push sweep's floor
/// is `> 600` out of 686, which six modules do not move. Only holding the two
/// against ONE module family tells a move from a loss.
///
/// Nine fields at four-byte spacing is `struct Push` in `rms_rope.slang`,
/// field for field, and the offsets are stated rather than just the count
/// because a struct that reordered would keep the count.
#[test]
fn the_fused_norm_rope_moved_its_block_into_a_push_range() {
    modules!();
    let mut seen = 0usize;
    for &(name, code) in kernels_vulkan::MODULES {
        if !name.starts_with("rms_rope") {
            continue;
        }
        seen += 1;
        let d = declared(code);
        assert_eq!(
            d.push_offsets,
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
            "`{name}` does not carry the nine `RmsRopeParams` scalars as a \
             push range"
        );
        assert!(
            d.block_bytes.iter().all(Option::is_none),
            "`{name}` declares a parameter block as well as its push range, \
             and `binding::params` places a launch's scalars in one or the \
             other and never both"
        );
    }
    assert_eq!(seen, 6, "the `rms_rope` family is six modules wide");
}

/// The block sizes this crate derives are the ones an independent walk read.
///
/// Two claims at once, and the second is the one that would rot. That the
/// size agrees is the arithmetic being right. That there is exactly one row
/// -- no module has grown a parameter block this table does not know about,
/// and the one that has it has not lost it -- is what keeps a new kernel from
/// arriving with an unchecked ABI and this file still passing.
///
/// The second claim is doing MORE work than it was, not less, now that the
/// scalars have gone to push ranges. One is a small enough number that the
/// buffer is the exception, and a second appearing is a launch that could not
/// fit a push range and nobody said so.
#[test]
fn the_parameter_blocks_this_crate_measures_are_the_ones_the_modules_declare() {
    modules!();
    let mut found: Vec<(String, u32, u32)> = Vec::new();
    let mut disagreed = Vec::new();

    for &(name, code) in kernels_vulkan::MODULES {
        let d = declared(code);
        for (binding, size) in d.block_bytes.iter().enumerate() {
            let Some(size) = size else { continue };
            found.push((name.to_owned(), binding as u32, *size));
            let want = PARAM_BLOCKS
                .iter()
                .find(|(n, b, _)| *n == name && *b == binding as u32);
            match want {
                Some((_, _, bytes)) if bytes == size => {}
                Some((_, _, bytes)) => disagreed.push(format!(
                    "{name} binding {binding}: this crate says {size} bytes, the \
                     transcription says {bytes}"
                )),
                None => disagreed.push(format!(
                    "{name} binding {binding} is a {size}-byte block the table \
                     does not know about"
                )),
            }
        }
    }

    for (name, binding, bytes) in PARAM_BLOCKS {
        if !found.iter().any(|(n, b, _)| n == name && b == binding) {
            disagreed.push(format!(
                "{name} binding {binding} is a {bytes}-byte block the table \
                 states and this crate did not find"
            ));
        }
    }

    assert!(
        disagreed.is_empty(),
        "{} of {} parameter blocks disagree:\n  {}",
        disagreed.len(),
        PARAM_BLOCKS.len(),
        disagreed.join("\n  ")
    );
}

/// The other half of the census: the scalars that left the buffers are in
/// push ranges, and the ranges fit the floor every device guarantees.
///
/// Without this, the sweep above would have been free to keep passing as the
/// tree emptied out from under it. Thirty-eight rows left it in one edit, and
/// "the modules stopped declaring parameter blocks" reads exactly the same
/// whether the scalars moved to a push range or stopped being passed at all --
/// which is not a hypothetical failure, since a body that forwards no scalars
/// into a block that arrives zeroed is a loop that runs no iterations and
/// reports success.
///
/// So: 674 of the 686 modules declare a push range. The twelve that do not are
/// the launches whose every scalar is a buffer's length, which a shader reads
/// off the binding rather than being told.
///
/// The widths are stated as a floor rather than exactly, because instantiating
/// a family at a new tile adds modules without changing any ABI and an exact
/// count would fail for that. What is exact is the CEILING: 128 bytes is the
/// `maxPushConstantsSize` every Vulkan device must offer, and a range past it
/// is a launch that will not fire on a conformant driver. The widest here ends
/// at offset 48 -- `gdn_prep`'s prefill arm, thirteen words -- so the tree is
/// using a bit over a third of the guarantee.
#[test]
fn the_push_ranges_are_where_the_parameter_blocks_went() {
    modules!();
    let mut modules = 0usize;
    let mut with_push = 0usize;
    let mut widest = (0usize, 0u32, String::new());

    for &(name, code) in kernels_vulkan::MODULES {
        let d = declared(code);
        modules += 1;
        let Some(&last) = d.push_offsets.last() else {
            continue;
        };
        with_push += 1;
        if d.push_offsets.len() > widest.0 {
            widest = (d.push_offsets.len(), last, name.to_owned());
        }
    }

    assert!(
        modules > 600 && with_push > 600,
        "{with_push} of {modules} modules declare a push range; the shader tree \
         is 686 modules wide and 674 of them do, so a walk this short is a walk \
         that stopped finding them rather than a tree that stopped having them"
    );
    assert!(
        modules - with_push <= 20,
        "{} modules declare no push range at all, up from the twelve whose only \
         scalars are buffer lengths -- a launch that lost its scalars reads the \
         same as one that never had any, and only this number tells them apart",
        modules - with_push
    );
    // A push field is at most 16 bytes (a `float4`), so `last + 16` bounds the
    // range without this test needing to know the widths `slangc` chose.
    assert!(
        widest.1 + 16 <= 128,
        "`{}` puts a push field at offset {}, and {} bytes is past the 128 that \
         is the `maxPushConstantsSize` every Vulkan device must offer",
        widest.2,
        widest.1,
        widest.1 + 16
    );
}

/// The bindings a module skips are the ones this crate reports as holes.
///
/// `Declared::bindings` is one past the highest, so it and the decorated set
/// disagree wherever `slangc` dropped a binding a variant never reads. On Metal
/// that costs nothing; on Vulkan the descriptor set still carries a slot there
/// and something has to decide what goes in it, so the count is load-bearing.
///
/// Measured across the whole tree rather than sampled: 165 of 666 modules have
/// at least one hole and there are 406 in all. Both numbers are stated because
/// a walk that silently stopped finding holes would otherwise look like a tree
/// that stopped having them.
///
/// The hole COUNT moved when the tree was ported from GLSL to Slang -- 358
/// became 406 -- while the set of holed modules did not: the same 165, and the
/// same deepest one. So the change is `slangc` eliminating more unread buffers
/// than `glslc` did, which is exactly the behaviour this test exists to keep
/// the driver tolerant of, and not a shader that stopped declaring something
/// it reads. A module that dropped a binding it USES fails on device, and all
/// 40 kernel proofs and the on-device rule cross-checks pass.
#[test]
fn the_bindings_a_module_skips_are_the_ones_this_crate_calls_holes() {
    modules!();
    let mut modules = 0u32;
    let mut holed = 0u32;
    let mut holes = 0usize;
    let mut widest = 0usize;

    for &(name, code) in kernels_vulkan::MODULES {
        let d = declared(code);
        modules += 1;
        // The invariant that makes `holes()` meaningful at all: `used` is
        // indexed by binding number, so it has to be as long as the layout or
        // a hole at the end would read as an absence instead.
        assert_eq!(
            d.used.len(),
            d.bindings as usize,
            "{name}: {} slots and {} of them accounted for",
            d.bindings,
            d.used.len()
        );
        // One past the HIGHEST means the last slot is always decorated. A walk
        // that reported a trailing hole would be reporting a `bindings` that
        // does not mean what it says.
        if d.bindings > 0 {
            assert!(
                *d.used.last().expect("a non-empty set"),
                "{name}: the highest binding is a hole, so `bindings` is not \
                 one past it"
            );
        }
        if d.holes() > 0 {
            holed += 1;
            holes += d.holes();
            if d.holes() > widest {
                widest = d.holes();
                eprintln!("WIDEST {name} {}", d.holes());
            }
        }
    }

    // 666/165/406 became 675/173/418 with the flash decode's nine modules.
    // Twelve of the nine's holes are DELIBERATE, and they are the same shape
    // twice over: `sdpa_paged_decode_split` inherits the eleven-binding decode
    // header and writes no `out_` (binding 3) and reads no `sinks` (binding
    // 10), and the four sinkless `sdpa_paged_decode_combine` modules declare
    // `sinks` at binding 1 and never read it. Both are a variant sharing a
    // header with its siblings, which is what most of the 406 already were.
    // 681 became 691 across two waves that landed the same hour, five modules
    // each: the ssm family (the two conv arms, the `[b | a]` gate row and the
    // two delta scans) and the gemm/layout/norm wave (the dense matmul's two
    // arms, the two row cuts, and `norm.mul_scalar`'s stated factor).
    //
    // 173/418 did NOT move, and both waves owe the same reason from two
    // directions. The ssm five each declare every binding they stamp: the one
    // operand an arm has that its sibling lacks is `indptr`, and both conv and
    // delta APPEND it rather than reserving it a slot in the middle. The other
    // five declare their buffers densely from zero and RENUMBER where they
    // vary -- `layer_scalar_mul_stated` puts `out_` at binding 1 where the read
    // arm has it at 2, which is the alternative to a hole and is why this count
    // is unchanged.
    //
    // 675 became 681 with the fused norm+rope's six, and 173/418 did NOT
    // move: not one of the six has a hole. That is worth a line because the
    // family looked like it should have some -- the `freqs` arms declare an
    // `inv_freq` at binding 4 that the plain arms do not -- but the extra
    // binding is the LAST one rather than an interior one, so the plain arms
    // simply declare four and stop. A family whose optional buffer is
    // appended costs no holes; one whose optional buffer sits in the middle
    // costs a hole in every sibling, which is most of the 418.
    //
    // 691 BECAME 694 WITH THE LSE PLANE AND THE SINK RESCALE, and this is the
    // first wave in a while where the holes moved with the modules: 173/418
    // became 175/420, two of the three new modules carrying one hole each.
    //
    // The two are `sdpa_paged_{decode,tiled}_lse_bfloat16_d_64`, and their hole
    // is the paragraph above's second shape rather than its first. They inherit
    // the eleven-binding decode header, they read `queries` through
    // `attention_mask_enabled` at 0..9 and they write the lse at 11 -- but an
    // `_lse` reading reaches no SINK, so binding 10 is declared, never touched,
    // and dropped. That is an optional buffer sitting in the MIDDLE, which is
    // the case that costs a hole; had the lse been given binding 10 and the sink
    // moved out, the two arms would have cost none and every sinkless sibling in
    // the family would have gained one. 11 is also `partials`' number under
    // `PIE_SPLIT`, and the two are never compiled together -- a split holds only
    // its own slice's denominator.
    //
    // The third, `attn_sink_rescale_bfloat16`, is a NEW FILE with a header of
    // its own: four buffers declared densely from zero and all four read, so no
    // hole. A family that does not inherit cannot inherit a gap.
    //
    // 694 BECAME 697 WHEN THREE POINTS THAT COULD NOT FIRE WERE GIVEN ARMS,
    // and none of the three moved the holes:
    //
    // * `embed_bfloat16` is a NEW FILE (`layout/embed.slang`) and declares
    //   three buffers densely from zero, all read. `layout.embed` had no dense
    //   gather at all -- it asked `Staged::bank` for a quantised table's
    //   sidecars unconditionally, and every catalog row states a `bf16`
    //   embedding.
    // * `neox_yarn_mb_bfloat16` is a `PIE_YARN` variant of `rope/neox.slang`,
    //   whose header is two buffers and a push block. `rope.yarn` had asked
    //   for a precomputed ladder no driver stages.
    // * `gptoss_swiglu_strided_bfloat16` is a `PIE_GPTOSS_STRIDED` variant of
    //   `mlp/gated.slang`, three buffers, all read. The flat arm indexes gate,
    //   up and out by one id, which a packed `[gate | up]` row cannot use.
    //
    // Three modules, zero holes, and the reason is the same each time: a file
    // whose bindings are declared densely from zero and all read has none, and
    // a variant that adds only push words changes no binding at all.
    // 697 BECAME 698 WITH THE UNSORTED COMBINE, and it costs no hole either:
    // `expert_combine` is a `PIE_EXPERT_COMBINE` arm of `moe/route.slang` with
    // three buffers declared densely from zero and all three read. It is what
    // `moe.weighted_sum` was missing — the sorted arm beside it folds through
    // an inverse permutation `route_sort` writes and no point of this plane
    // claims, so the point was in the claim table and could fire for nothing.
    assert_eq!(modules, 698, "a different number of modules is built");
    assert_eq!(holed, 175, "a different number of modules has a hole");
    assert_eq!(holes, 420, "a different number of holes in all");
    // `cast_qmm_input_bfloat16_to_float16` is the deepest: it shares a header
    // with the matmul family it feeds, reads two of the thirteen bindings that
    // header declares, and `slangc` drops the other eleven. A driver counting
    // slots would go looking for eleven buffers that do not exist.
    //
    // Stated so that a module quietly becoming mostly holes is visible, and
    // because six -- `kv_append_paged`'s deliberate ring-ABI gap -- was the
    // guess this replaced. The accidental ones are deeper than the intentional
    // one.
    assert_eq!(widest, 11, "the most holes in one module changed");
}
