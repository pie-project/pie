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
//! No GPU is involved. The modules are files.

use driver_vulkan::{Declared, spirv};

/// Where a `native` build of `kernels-vulkan` left the modules.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// Skip with a reason when there are no modules, rather than pass silently.
macro_rules! modules {
    () => {
        match SPV_DIR {
            Some(d) => std::path::Path::new(d),
            None => {
                eprintln!(
                    "no modules: build with `-p driver-vulkan --features native` \
                     (or any profile that pulls kernels-vulkan/native) and have \
                     `slangc` on PATH"
                );
                return;
            }
        }
    };
}

/// What a module declares, read through the crate's own loader.
///
/// The tests used to carry a second SPIR-V parser. Two parsers of the same
/// bytes is two things to keep right, and the one under test is the one that
/// matters -- so this calls it, and a defect in it now fails these sweeps
/// instead of being masked by a copy that happens to agree.
fn declared(path: &std::path::Path) -> Option<Declared> {
    let code = std::fs::read(path).ok()?;
    let words = spirv::words(&code).expect("a built module is whole words");
    Some(spirv::declared(&words).expect("a built module is well formed"))
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
    let dir = match SPV_DIR {
        Some(d) => std::path::Path::new(d),
        None => return Vec::new(),
    };
    kernels_vulkan::entrypoints()
        .into_iter()
        .filter_map(|name| {
            let d = declared(&dir.join(format!("{name}.spv")))?;
            Some((name, d))
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
    let _ = modules!();
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
/// SPIR-V walk written in another language, and two of them -- `router_topk`
/// at 16 and `combine_sorted` at 12 -- are the same two blocks the Metal
/// driver was found packing short.
const PARAM_BLOCKS: &[(&str, u32, u32)] = &[
    ("affine_encode_u4_bf16", 4, 8),
    ("affine_encode_u4_f32", 4, 8),
    ("argmax_logits_bfloat16", 2, 40),
    ("combine_sorted", 3, 12),
    ("gated_rms_bfloat16", 4, 8),
    ("gated_rms_strided_bfloat16", 4, 8),
    ("gdn_core_bfloat16", 11, 44),
    ("gdn_core_recurrent_bfloat16", 10, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_16_v_1", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_16_v_2", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_16_v_4", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_32_v_2", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_32_v_4", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_32_v_8", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_4_v_1", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_8_v_1", 5, 44),
    ("gdn_core_recurrent_prefill_bfloat16_l_8_v_2", 5, 44),
    ("gdn_core_recurrent_slotted_bfloat16", 10, 44),
    ("gdn_core_slotted_bfloat16", 11, 44),
    ("gdn_prep_bfloat16", 12, 44),
    ("gdn_prep_prefill_bfloat16", 12, 44),
    ("gdn_prep_slotted_bfloat16", 12, 44),
    ("geglu_tanh_strided_bfloat16", 3, 20),
    ("gptoss_swiglu_bfloat16", 3, 12),
    ("logit_softcap_bfloat16", 2, 8),
    ("mxfp4_dequant_bf16", 3, 8),
    ("ple_combine_bfloat16", 3, 8),
    ("rms_residual_bfloat16", 3, 20),
    ("rms_residual_scaled_bfloat16", 3, 20),
    ("rms_single_row_bfloat16", 3, 20),
    ("rms_strided_head_row_bfloat16", 3, 20),
    ("rms_strided_row_bfloat16", 3, 20),
    ("route_gather", 3, 28),
    ("route_sort", 4, 28),
    ("router_topk_bfloat16", 3, 16),
    ("router_topk_scaled_bfloat16", 3, 16),
    ("row_gather_bfloat16", 3, 8),
    ("split_qkv_bf16", 4, 8),
    ("vnorm_single_row_bfloat16", 2, 8),
];

/// The block sizes this crate derives are the ones an independent walk read.
///
/// Two claims at once, and the second is the one that would rot. That the 39
/// sizes agree is the arithmetic being right. That there are exactly 39 -- no
/// module has grown a parameter block this table does not know about, and none
/// has lost one -- is what keeps a new kernel from arriving with an unchecked
/// ABI and this file still passing.
#[test]
fn the_parameter_blocks_this_crate_measures_are_the_ones_the_modules_declare() {
    let dir = modules!();
    let mut found: Vec<(String, u32, u32)> = Vec::new();
    let mut disagreed = Vec::new();

    for entry in std::fs::read_dir(dir).expect("the module directory is readable") {
        let path = entry.expect("a directory entry").path();
        if path.extension().is_none_or(|e| e != "spv") {
            continue;
        }
        let name = path
            .file_stem()
            .expect("a name")
            .to_string_lossy()
            .into_owned();
        let Some(d) = declared(&path) else { continue };
        for (binding, size) in d.block_bytes.iter().enumerate() {
            let Some(size) = size else { continue };
            found.push((name.clone(), binding as u32, *size));
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
    let dir = modules!();
    let mut modules = 0u32;
    let mut holed = 0u32;
    let mut holes = 0usize;
    let mut widest = 0usize;

    for entry in std::fs::read_dir(dir).expect("the module directory reads") {
        let path = entry.expect("an entry").path();
        if path.extension().is_none_or(|e| e != "spv") {
            continue;
        }
        let Some(d) = declared(&path) else { continue };
        modules += 1;
        // The invariant that makes `holes()` meaningful at all: `used` is
        // indexed by binding number, so it has to be as long as the layout or
        // a hole at the end would read as an absence instead.
        assert_eq!(
            d.used.len(),
            d.bindings as usize,
            "{}: {} slots and {} of them accounted for",
            path.display(),
            d.bindings,
            d.used.len()
        );
        // One past the HIGHEST means the last slot is always decorated. A walk
        // that reported a trailing hole would be reporting a `bindings` that
        // does not mean what it says.
        if d.bindings > 0 {
            assert!(
                *d.used.last().expect("a non-empty set"),
                "{}: the highest binding is a hole, so `bindings` is not one \
                 past it",
                path.display()
            );
        }
        if d.holes() > 0 {
            holed += 1;
            holes += d.holes();
            if d.holes() > widest {
                widest = d.holes();
                eprintln!("WIDEST {} {}", path.display(), d.holes());
            }
        }
    }

    assert_eq!(modules, 666, "a different number of modules is built");
    assert_eq!(holed, 165, "a different number of modules has a hole");
    assert_eq!(holes, 406, "a different number of holes in all");
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
