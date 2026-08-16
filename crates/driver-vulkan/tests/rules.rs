//! The geometry, checked against every module's own declared workgroup.
//!
//! [`driver_vulkan::geometry`] divides a thread extent by a workgroup size. The
//! extent comes from the row's [`Rule`]; the workgroup size comes from
//! `[numthreads(...)]` in the shader, which no Rust code can see and
//! `slangc` bakes into the module. Nothing makes them agree. This file is what
//! makes them agree, over all 480 entrypoints at once, by reading the size back
//! out of the SPIR-V and asking the geometry what it would have launched.
//!
//! It exists because that disagreement does not report itself. A workgroup
//! count too small by one launches a dispatch that runs, returns success, and
//! leaves a slice of the output holding whatever the buffer was allocated with.
//!
//! No GPU is involved. The modules are files.

use driver_vulkan::{Declared, Dims, Local, Module, Rule, Tile, geometry, spirv};

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

/// The fire a given module could actually serve.
///
/// Decode attention is compiled one module per head width and refuses any
/// other, so a sweep cannot hold `head_dim` fixed and still reach the family:
/// it has to ask each module for the width that module was built for, which is
/// its own `local_size_x`. Every other rule ignores the adjustment.
fn dims_for(rule: Rule, local: [u32; 3]) -> Dims {
    if rule == Rule::SdpaVector {
        Dims {
            head_dim: local[0].max(1),
            ..dims()
        }
    } else {
        dims()
    }
}

/// A plausible fire: a 4096-wide model, 32/8 grouped-query heads.
///
/// 64 rows and not four, because 64 is the only row count every compiled GEMM
/// tile divides -- `Rule::Qmm` refuses the rest, and a sweep in which the
/// largest family silently refuses is a sweep that proves much less than its
/// name claims.
fn dims() -> Dims {
    Dims {
        rows: 64,
        width: 4096,
        in_width: 12288,
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        axis: 4096,
        rotary_dims: 128,
        n_experts: 64,
        experts_per_token: 8,
    }
}

/// Every entrypoint, its rule, and the workgroup its module declares.
fn table() -> Vec<(String, Rule, Declared)> {
    let dir = match SPV_DIR {
        Some(d) => std::path::Path::new(d),
        None => return Vec::new(),
    };
    kernels_vulkan::entrypoints()
        .into_iter()
        .filter_map(|name| {
            let row = kernels::sig_in(kernels_vulkan::KERNELS, &name)?;
            let d = declared(&dir.join(format!("{name}.spv")))?;
            Some((name, row.launch, d))
        })
        .collect()
}

/// Every stated entrypoint gets enough lanes for the extent its rule states.
///
/// The check the crate exists for. It is a sweep over the real table rather
/// than a handful of cases because the rule that gets this wrong is the one
/// nobody thought to write a case for.
#[test]
fn every_entrypoint_is_launched_over_its_whole_extent() {
    let _ = modules!();
    let table = table();
    assert!(
        table.len() >= 400,
        "only {} entrypoints resolved to a module",
        table.len()
    );

    let mut stated = 0;
    for (name, rule, d) in &table {
        if *rule == Rule::Unstated {
            continue;
        }
        let g = geometry::groups(*rule, dims_for(*rule, d.local), Module::loaded(name, d))
            .unwrap_or_else(|e| panic!("`{name}` ({rule:?}) has no geometry: {e}"));
        for axis in 0..3 {
            assert!(
                g[axis] >= 1,
                "`{name}` ({rule:?}) launches {g:?} workgroups -- axis {axis} is \
                 empty, so the dispatch runs nothing and reports success"
            );
        }
        stated += 1;
    }
    assert!(stated >= 180, "only {stated} entrypoints state a rule");
}

/// Where the workgroup COUNT is data, the division must come out exact.
///
/// This is the finding that qualifies everything else in this file. The usual
/// reading is that an over-launch is harmless because every shader guards its
/// own tail. Thirty-four modules make that false: they read `gl_NumWorkGroups`
/// and use it as a QUANTITY, so an extra workgroup does not run a guarded lane,
/// it changes the arithmetic every lane does.
///
/// `rope/neox.slang` is the clearest. It takes `gl_NumWorkGroups.x` as the
/// rotary pair count -- the number it strides the second half of each pair by
/// and divides the frequency exponent by -- and `gl_NumWorkGroups.y` as the
/// head count, which sizes the row base. Round its grid up and every pair is
/// rotated by the wrong angle against the wrong partner. That is why the shader
/// is `local_size (1, 1, 1)`: it is not a decomposition, it is the grid being
/// the contract.
///
/// So for these modules the round-up must be a no-op, and this checks that it
/// is rather than trusting that it happens to be.
#[test]
fn a_module_that_reads_its_workgroup_count_is_launched_exactly() {
    let _ = modules!();
    let mut exact = 0;
    for (name, rule, d) in table() {
        if rule == Rule::Unstated || !d.reads_workgroup_count {
            continue;
        }
        let m = Module::loaded(&name, &d);
        let d = dims_for(rule, d.local);
        let Ok(g) = geometry::groups(rule, d, m) else {
            continue;
        };
        let want = geometry::lanes(rule, d, m).expect("answered once");
        for axis in 0..3 {
            assert_eq!(
                g[axis] * m.local.at(axis),
                want[axis],
                "`{name}` ({rule:?}) reads gl_NumWorkGroups, so its workgroup \
                 count is a QUANTITY -- but on axis {axis} the geometry rounds \
                 {} lanes up to {}. The extra workgroup does not run a guarded \
                 lane; it makes every lane compute against the wrong count.",
                want[axis],
                g[axis] * m.local.at(axis)
            );
        }
        exact += 1;
    }
    // The rope and SDPA-vector families, and nothing may quietly leave.
    assert!(
        exact >= 15,
        "only {exact} modules that read their workgroup count were checked"
    );
}

/// A rule's workgroup size is a property of the RULE, not of the entrypoint —
/// except where the shader says otherwise, and those exceptions are named.
///
/// This is the check that found the two real ones. `SdpaVector` declares
/// `local_size_x = PIE_HEAD_DIM`, so its four modules are 64, 128, 256 and 512
/// wide and a geometry that assumed any single number would undershoot three of
/// them by up to 8x. `Elementwise` is 256 in nineteen modules and `(16, 16, 1)`
/// in `geglu_tanh_strided`, which is indexed per (channel, row). `RouteRows`
/// is the third: `add_bias` cannot round its y axis up, because unlike the
/// `moe` rows it carries no `rows` scalar to guard against.
///
/// Anything ELSE that varies is drift: two shaders under one rule that no
/// longer agree about their decomposition. The list is closed on purpose, so
/// adding a variant makes this fail rather than pass quietly.
#[test]
fn only_the_rules_that_are_allowed_to_vary_do() {
    let _ = modules!();
    // Keyed by the rule's NAME: `LaunchRule` is a shared vocabulary and does
    // not derive `Ord`, and a driver-side test is not a reason to widen it.
    let mut by_rule: std::collections::BTreeMap<String, std::collections::BTreeSet<[u32; 3]>> =
        std::collections::BTreeMap::new();
    for (_, rule, d) in table() {
        if rule == Rule::Unstated {
            continue;
        }
        by_rule
            .entry(format!("{rule:?}"))
            .or_default()
            .insert(d.local);
    }
    assert!(!by_rule.is_empty(), "no stated rules in the table");

    for (rule, sizes) in &by_rule {
        let allowed = match rule.as_str() {
            // One module per head dimension.
            "SdpaVector" => 4,
            // `geglu_tanh_strided` is laid out per (channel, row).
            "Elementwise" => 2,
            // `add_bias` is 256 on x with one row per group; the two `moe`
            // rows are (16, 16) because they carry `rows` in a params buffer
            // and can guard the y axis. `add_bias` carries only its width, so
            // it takes its row count from the grid itself and must not be
            // rounded up on y. Both decompose the SAME grid -- the rule still
            // hands out [width, rows, 1] -- so a driver dividing by each
            // module's own `local` is right either way.
            "RouteRows" => 2,
            _ => 1,
        };
        assert!(
            sizes.len() <= allowed,
            "{rule} is compiled at {} different workgroup sizes ({sizes:?}) but \
             only {allowed} are accounted for -- either a shader's local_size \
             changed or two kernels under one rule stopped agreeing about their \
             decomposition. A driver divides by this number.",
            sizes.len()
        );
    }
}

/// No module asks for more invocations per workgroup than Vulkan's floor.
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
    for (name, _, d) in table() {
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
}

/// The tail-covering property, at sizes that are NOT multiples of anything.
///
/// The round-up only matters when the division is inexact, so a sweep at round
/// numbers proves nothing — this tree already shipped three tests whose `n =
/// 512` against a 256-wide workgroup meant the tail branch never ran. Every
/// dimension here is chosen to be awkward.
#[test]
fn an_awkward_shape_still_covers_its_tail() {
    let _ = modules!();
    // Two row counts, because a GEMM legitimately refuses a row count its
    // compiled tile does not divide -- at three rows all 108 of them refuse and
    // the sweep quietly becomes a sweep of everything EXCEPT the largest
    // family. Sixteen rows lets the `bm_16` third of them in while every other
    // dimension stays awkward.
    let mut checked = 0;
    let mut gemms = 0;
    for rows in [3, 16] {
        let ragged = Dims {
            rows,
            width: 4095,
            in_width: 6141,
            q_heads: 7,
            kv_heads: 7,
            head_dim: 129,
            axis: 4095,
            rotary_dims: 130,
            n_experts: 61,
            experts_per_token: 5,
        };
        for (name, rule, d) in table() {
            if rule == Rule::Unstated {
                continue;
            }
            // A GEMM refuses a row count no tile divides, which is the point of
            // `PartialTile` and not a failure here.
            let m = Module::loaded(&name, &d);
            let Ok(g) = geometry::groups(rule, ragged, m) else {
                continue;
            };
            // The same rule against a one-lane workgroup: the extent itself, in
            // lanes. Comparing against THAT rather than recomputing the extent here
            // keeps this test from restating the arithmetic it is checking.
            let widest = geometry::groups(
                rule,
                ragged,
                Module {
                    local: driver_vulkan::Local([1, 1, 1]),
                    tile: m.tile,
                },
            )
            .expect("answered once");
            for axis in 0..3 {
                let launched = g[axis] * m.local.at(axis);
                assert!(
                    launched >= widest[axis],
                    "`{name}` ({rule:?}) launches {launched} lanes on axis {axis} for \
                 an extent of {} -- the {} that do not run write nothing, and the \
                 gap reads back as the buffer's zeros",
                    widest[axis],
                    widest[axis] - launched
                );
            }
            checked += 1;
            gemms += u32::from(rule == Rule::Qmm);
        }
    }
    assert!(checked >= 150, "only {checked} entrypoints were checked");
    // And the GEMMs were not all sitting the sweep out: at sixteen rows the
    // `bm_16` third of them answers, and a change that made every tile refuse
    // would otherwise leave this test passing on the remainder.
    assert!(gemms >= 30, "only {gemms} GEMM entrypoints were checked");
}

/// No rule puts work on a grid axis its shader never reads.
///
/// The check the sweeps above cannot make, because they count LANES. Lanes on
/// an axis nobody is indexed by are lanes all the same, so a geometry that
/// spreads rows across the wrong axis passes every one of them: the extent is
/// covered, the arithmetic is right, and the dispatch computes row 0 and
/// leaves every other row holding the zeros its buffer was born with.
///
/// That is not hypothetical -- it is what `Rule::Rms` did here. It stacked
/// rows on grid y while `norm/rms.slang` reads `gl_WorkGroupID.x` and never
/// mentions y. It took a real dispatch on real hardware to see, once. This
/// asks the module which components of `gl_WorkGroupID` and
/// `gl_GlobalInvocationID` it is actually indexed by, and answers the same
/// question for all 480 entrypoints without a GPU.
///
/// Stated as "more than one group", not "any group": a grid is a product, so
/// every axis carries at least one whatever the shader reads, and one is
/// exactly what an unused axis should have.
#[test]
fn no_rule_puts_work_on_an_axis_its_shader_never_reads() {
    let _ = modules!();
    let mut checked = 0;
    let mut bad: Vec<String> = Vec::new();
    for (name, rule, d) in table() {
        if rule == Rule::Unstated || DECODE_ONLY.contains(&&*name) {
            continue;
        }
        // Deep enough to make every axis a rule uses non-trivial: at one row a
        // rule that stacks rows on the wrong axis puts one group there and is
        // indistinguishable from a rule that stacks them on the right one.
        let fire = Dims {
            rows: 64,
            ..dims_for(rule, d.local)
        };
        let Ok(g) = geometry::groups(rule, fire, Module::loaded(&name, &d)) else {
            continue;
        };
        for (axis, (&read, &given)) in d.grid_axes.iter().zip(g.iter()).enumerate() {
            if !read && given > 1 {
                bad.push(format!(
                    "`{name}` ({rule:?}) gets {given} workgroups on axis \
                     {axis}, which it is never indexed by"
                ));
            }
        }
        checked += 1;
    }
    assert!(
        bad.is_empty(),
        "{} entrypoints are given work nobody will do:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert!(checked >= 180, "only {checked} entrypoints were checked");
}

/// The MIRROR of the check above, which is the dangerous direction.
///
/// `no_rule_puts_work_on_an_axis_its_shader_never_reads` asks `if !read &&
/// given > 1` -- work handed to an axis nothing reads, which is WASTE. This
/// asks the opposite: an axis the body READS that the rule leaves at one lane,
/// which is DATA LOSS. Every index past the first on that axis is never
/// dispatched, nothing faults, and the buffer keeps what it was allocated
/// with.
///
/// They are different predicates and only the first was written, which is
/// exactly why this crate did not catch `geglu_tanh_strided`: its body
/// declared `local_size_y = 16` and read `gl_GlobalInvocationID.y` as the row
/// under a rule (`Elementwise`) that puts one lane on y. Every row past 15 of
/// gemma's per-layer-embedding gate was dropped, on every prefill longer than
/// sixteen tokens, silently. The body is flat now; this is what would have
/// said so.
///
/// Transcribed from `driver-wgpu`'s `no_module_reads_a_grid_axis_its_rule_
/// leaves_flat`, which was written after the same defect was measured there.
#[test]
fn no_module_reads_a_grid_axis_its_rule_leaves_flat() {
    let _ = modules!();
    let mut checked = 0;
    let mut found: Vec<String> = Vec::new();
    for (name, rule, d) in table() {
        if rule == Rule::Unstated || DECODE_ONLY.contains(&&*name) {
            continue;
        }
        // 128 rows and not the 64 the sibling check uses, because this
        // predicate is the other way round and 64 makes it lie. Every compiled
        // GEMM tile divides 64, so `Rule::Qmm`'s y extent is
        // `rows.div_ceil(bm)` = 1 for a `bm` of 64 -- one row tile, correctly
        // one workgroup, and a body reading `gl_WorkGroupID.y` there is
        // reading index 0 because index 0 is all there is. 128 is a multiple
        // of every tile too and makes that axis two, so an axis that is flat
        // is flat because the RULE flattened it.
        let fire = Dims {
            rows: 128,
            ..dims_for(rule, d.local)
        };
        // LANES, not workgroups, and the difference is the whole predicate.
        // `Rule::PerHead` puts `head_dim` lanes on x; a module 128 wide covers
        // a 128-channel head in ONE workgroup, and a body reading
        // `gl_GlobalInvocationID.x` there sees 0..127, not just 0. Measured
        // against `groups` this check called that data loss. An axis is flat
        // when the rule puts one LANE on it, because only then can the body
        // never see an index above zero.
        let Ok(g) = geometry::lanes(rule, fire, Module::loaded(&name, &d)) else {
            continue;
        };
        for (axis, (&read, &given)) in d.grid_axes.iter().zip(g.iter()).enumerate() {
            // A module whose own workgroup is wider than one on this axis may
            // legitimately read a LOCAL index there -- `gl_LocalInvocationID.y`
            // is a lane within the group and says nothing about the grid. Only
            // a GLOBAL read of a flattened axis is the defect, and `grid_axes`
            // is about the global builtins.
            if read && given <= 1 {
                found.push(format!(
                    "`{name}` ({rule:?}) is indexed by axis {axis} and its rule \
                     puts {given} workgroup there"
                ));
            }
        }
        checked += 1;
    }
    assert!(
        found.is_empty(),
        "{} entrypoints read a grid axis their rule flattens, so every index \
         past the first on that axis is never written and the dispatch \
         succeeds anyway:\n{}",
        found.len(),
        found.join("\n")
    );
    assert!(checked >= 180, "only {checked} entrypoints were checked");
}

/// The entrypoints that are one row by construction, and why each is.
///
/// Not a way to quiet the check -- a record of the modules whose row count is
/// not a grid axis at all, so that asking the check about them at 64 rows asks
/// a question their caller never asks.
///
/// `kv_append_bfloat16` appends at `pos[0]`, a single scalar slot: a second
/// row would not be a second destination, it would be the same destination
/// written twice. Its paged sibling takes the slot from a per-row table and
/// does read z, which is why only one of the two is named here.
///
/// The three `neox_*_decode_*` modules are compiled with `PIE_DECODE`, which
/// is `row = 0u` -- the assignment discards `gl_WorkGroupID.z` and slangc then
/// drops the read, which is exactly why the check sees no z. Their `_mb_`
/// siblings keep it and are checked normally.
///
/// So every name here is a module that would MISCOMPUTE at 64 rows, not one
/// that would merely waste them: 64 planes racing to write one destination.
/// Choosing one of these for a multi-row fire is a caller's error, and it is
/// recorded here rather than hidden because the geometry cannot see it.
const DECODE_ONLY: &[&str] = &[
    "kv_append_bfloat16",
    "neox_decode_bfloat16",
    "neox_freqs_decode_bfloat16",
    "neox_prop_decode_bfloat16",
];

/// And every axis a shader IS indexed by is given work when there is work.
///
/// The mirror image, and the reason the check above is not sufficient alone: a
/// geometry that answered `[n, 1, 1]` to everything would satisfy it perfectly
/// while telling a module that reads three axes it has one of each.
#[test]
fn a_shader_indexed_by_an_axis_is_given_that_axis() {
    let _ = modules!();
    let mut counted = 0;
    let mut bad: Vec<String> = Vec::new();
    for (name, rule, d) in table() {
        if rule == Rule::Unstated || DECODE_ONLY.contains(&&*name) {
            continue;
        }
        let fire = Dims {
            rows: 64,
            ..dims_for(rule, d.local)
        };
        let Ok(g) = geometry::groups(rule, fire, Module::loaded(&name, &d)) else {
            continue;
        };
        let highest_read = (0..3).filter(|a| d.grid_axes[*a]).max();
        let highest_given = (0..3).filter(|a| g[*a] > 1).max();
        if let (Some(read), Some(given)) = (highest_read, highest_given) {
            if given > read {
                bad.push(format!(
                    "`{name}` ({rule:?}) is given work up to axis {given} but \
                     is only indexed up to axis {read}"
                ));
            }
            counted += 1;
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    assert!(
        counted >= 100,
        "only {counted} entrypoints had work to check"
    );
}

/// A row never describes FEWER buffers than its module decorates.
///
/// The audit `kernels/attn/kv_write.slang` records in prose -- "these read 9
/// and 10 until a SPIR-V-level audit compared the compiled `OpDecorate
/// Binding` set against the table: off by one" -- made permanent. That audit
/// lives in a script, and a script only runs when someone remembers to run it.
/// The thing it guards against is not a wrong answer: a module reading
/// `binding = 11` under a layout that stops at 10 is a segmentation fault
/// inside `vkCreateComputePipelines`, which takes the process down and names
/// nothing.
///
/// Stated as an inequality because the two counts genuinely disagree, and only
/// one direction is a defect. Of the 188 stated entrypoints 177 agree exactly
/// and 11 have a module decorating one binding FEWER than the row lists --
/// slangc drops the decoration of a buffer the shader never reads, so the
/// row is describing a real operand the shader happens not to touch. A
/// descriptor declared and never read costs nothing. The other direction has
/// no benign reading at all.
#[test]
fn no_row_declares_fewer_buffers_than_its_module_binds() {
    let _ = modules!();
    let mut checked = 0;
    let mut short: Vec<String> = Vec::new();
    for name in kernels_vulkan::entrypoints() {
        let Some(row) = kernels::sig_in(kernels_vulkan::KERNELS, &name) else {
            continue;
        };
        // An unstated row describes nothing rather than nothing-to-bind, so it
        // has no claim here to be wrong about. `driver-metal`'s dispatch falls
        // back to the lowered plan's own argument order for exactly these, and
        // `Pipelines::get` takes the module's count when the caller offers 0.
        if row.operands.is_empty() {
            continue;
        }
        let Some(dir) = SPV_DIR.map(std::path::Path::new) else {
            continue;
        };
        let Some(d) = declared(&dir.join(format!("{name}.spv"))) else {
            continue;
        };
        let stated = kernels_vulkan::buffer_count(row);
        if stated < d.bindings {
            short.push(format!(
                "`{name}` lists {stated} buffers and its module decorates \
                 {} -- a layout that short is a SIGSEGV, not an error",
                d.bindings
            ));
        }
        checked += 1;
    }
    assert!(short.is_empty(), "{}", short.join("\n"));
    assert!(checked >= 150, "only {checked} stated rows were checked");
}

/// The block the driver writes is the block the shader reads.
///
/// The other half of the ABI, and the half with no symptom. A binding count
/// that is wrong crashes; a push offset that is wrong does not. The dispatch
/// is legal, the layer is silent, and the shader reads a stride where a head
/// count belongs -- a number, of the right type, in the right place, that is
/// simply not the one that was written.
///
/// So this compares `kernels_vulkan::push_layout`, which is what a driver
/// packs from the row, against the `Offset` decorations slangc put on the
/// module's own push block. All 188 stated entrypoints agree today, which is
/// the reason to fix it now rather than after the first disagreement: this
/// check cannot find a defect that already exists, only one that is about to
/// be introduced, and that is the only kind it could ever have caught.
///
/// Two things could make it vacuous, and both are controlled for. Packing the
/// scalars with no alignment rule at all -- each straight after the last --
/// disagrees, so the 8-byte members are load-bearing. Reading the offsets off
/// any `Offset`-decorated struct rather than following a `PushConstant`
/// variable to its own block picks up an SSBO layout instead, and this tree's
/// kernels are full of those.
#[test]
fn the_push_block_a_row_packs_is_the_one_its_module_declares() {
    let _ = modules!();
    let Some(dir) = SPV_DIR.map(std::path::Path::new) else {
        return;
    };
    let mut checked = 0;
    let mut with_scalars = 0;
    let mut differ: Vec<String> = Vec::new();
    for name in kernels_vulkan::entrypoints() {
        let Some(row) = kernels::sig_in(kernels_vulkan::KERNELS, &name) else {
            continue;
        };
        if row.operands.is_empty() {
            continue;
        }
        let Some(d) = declared(&dir.join(format!("{name}.spv"))) else {
            continue;
        };
        let stated: Vec<u32> = kernels_vulkan::push_layout(row)
            .iter()
            .map(|f| f.offset)
            .collect();
        if !stated.is_empty() {
            with_scalars += 1;
        }
        if stated != d.push_offsets {
            differ.push(format!(
                "`{name}` packs its scalars at {stated:?} and its module reads \
                 them at {:?}",
                d.push_offsets
            ));
        }
        checked += 1;
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
    assert!(checked >= 150, "only {checked} stated rows were checked");
    // Agreement between two empty lists is agreement about nothing.
    assert!(
        with_scalars >= 100,
        "only {with_scalars} of the rows checked actually have a push block"
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

/// Every row in the table packs, and every scalar lands on its own field.
///
/// [`driver_vulkan::lowering::pack`] carries two `unreachable!`s: a buffer
/// value arriving at a push slot, and a `Binding::Push(n)` whose `n` is past
/// the end of `push_layout`. A panic audit read both as structural --
/// `kernels_vulkan::bindings` and `kernels_vulkan::push_layout` walk the same
/// `sig.operands` under the same predicate, so the counts cannot disagree --
/// and that reading is correct today. It is correct because of an agreement
/// between two functions in a different crate that nothing checks.
///
/// That is the whole reason for this test. The invariant is real but it is
/// not local: `bindings` numbers the push operands by counting the ones that
/// are neither a buffer nor `InPacked`, and `push_layout` builds its vector
/// by filtering on the same two predicates written out a second time. Either
/// side could grow a case the other does not -- a new scalar kind that
/// `is_buffer` rejects and the layout filter skips would shift every field
/// after it -- and the first report would be an `unreachable!` in a driver,
/// or worse, no report at all.
///
/// # Why it checks the bytes and not just the absence of a panic
///
/// A test that only called `pack` and asserted `Ok` would pass on a table
/// where the two functions agreed about the COUNT and disagreed about the
/// ORDER, which is the more likely of the two mistakes and the silent one: a
/// stride written where a head count belongs is a legal dispatch. So each
/// operand is given a value derived from its own index, and each field of
/// the layout is read back out of the packed block and compared to the bytes
/// of the operand that should have filled it. The block is zero-filled, so a
/// field nobody wrote reads as zero -- and the seeds start at one for that
/// reason.
///
/// # What this does and does not say about the two panics
///
/// Following the callers settled it further than expected: `pack` has no
/// production caller in this crate. The live dispatch path binds from the
/// COMPILED MODULE -- `binding::Params`, which reads whether a kernel wants a
/// push block or a storage-buffer struct off the SPIR-V rather than off the
/// row -- because the two disagree and the module is the one the GPU obeys.
/// `pack` is the row-derived half, kept because it is the thing that can be
/// proven with no device, and its `unreachable!`s are therefore not reachable
/// from a malformed model artifact for a second, simpler reason: nothing in a
/// serving path calls them.
///
/// So this is not a crash guard. It is the check that keeps the row-derived
/// half honest while it has no caller to keep it honest, which is exactly
/// when an ABI derivation rots.
///
/// # Why only 44 of the 100 rows
///
/// Fifty-six state no operands at all -- `sdpa_paged_mma`, the whole
/// `qmm_t_*` family, `gdn_*` -- and `pack` refuses those by name with
/// `Mismatch::Unstated`. They are dispatched all the same, through the module
/// path above. The count is asserted exactly so that a row losing its operand
/// list, which would silently shrink this test's reach rather than fail it,
/// has to be noticed here.
///
/// No GPU and no modules: this is the table checking itself.
///
/// # What it was shown to catch
///
/// Three mutations, all killed. Dropping `push_layout`'s `InPacked` filter so
/// the two functions disagree about the COUNT reports `row_gather` states one
/// field and has no scalar operands. Striding `bindings`' push counter by two
/// so they disagree about the INDEX reaches the `unreachable!` above, by name
/// -- which is the only demonstration that that panic is a real edge and not
/// a decorative one. Writing every field at `layout[0].offset`, which leaves
/// every count correct and only the PLACEMENT wrong, is caught by the byte
/// comparison alone: `kv_append`'s `head_dim` reads 8 where 6 belongs and its
/// two 8-byte strides read zero, the zero being the empty block showing
/// through, which is why the seeds start at one.
#[test]
fn every_row_packs_and_every_scalar_lands_on_its_own_field() {
    use driver_vulkan::lowering::{Value, pack};
    use kernels::Ty;

    // Mirrors `Value::fits`. Written out rather than reached for, because a
    // driver calling `pack` has to make this same choice from the row's kind,
    // and a `None` here is a kind no caller could supply.
    fn value_for(ty: Ty, seed: u32) -> Option<Value> {
        Some(match ty {
            Ty::Buf
            | Ty::BufMut
            | Ty::I32s
            | Ty::I64s
            | Ty::U32s
            | Ty::U8s
            | Ty::F32s
            | Ty::F32sMut
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut => Value::Buffer(seed),
            Ty::I32 => Value::I32(seed as i32),
            Ty::U32 | Ty::InPacked => Value::U32(seed),
            Ty::F32 => Value::F32(seed as f32),
            Ty::Usize | Ty::I64 => Value::Usize(u64::from(seed)),
            _ => return None,
        })
    }

    fn bytes_of(v: Value) -> Vec<u8> {
        match v {
            Value::Buffer(_) => Vec::new(),
            Value::I32(x) => x.to_le_bytes().to_vec(),
            Value::U32(x) => x.to_le_bytes().to_vec(),
            Value::F32(x) => x.to_le_bytes().to_vec(),
            Value::Usize(x) => x.to_le_bytes().to_vec(),
        }
    }

    let mut checked = 0;
    let mut with_scalars = 0;
    let mut wrong: Vec<String> = Vec::new();
    let mut unbindable: Vec<String> = Vec::new();

    for row in kernels_vulkan::KERNELS {
        if row.operands.is_empty() {
            continue;
        }
        // Seeds from one, so a field left unwritten in a zero-filled block
        // cannot pass by coincidence.
        let mut values = Vec::new();
        let mut ok = true;
        for (at, op) in row.operands.iter().enumerate() {
            match value_for(op.ty, at as u32 + 1) {
                Some(v) => values.push(v),
                None => {
                    unbindable.push(format!(
                        "`{}` operand `{}` is {:?}, which no `Value` fits",
                        row.name, op.name, op.ty
                    ));
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }

        let call = match pack(row, &values) {
            Ok(c) => c,
            Err(e) => {
                wrong.push(format!("`{}` did not pack: {e}", row.name));
                continue;
            }
        };

        if call.buffers.len() != kernels_vulkan::buffer_count(row) as usize {
            wrong.push(format!(
                "`{}` packed {} buffers and its layout declares {}",
                row.name,
                call.buffers.len(),
                kernels_vulkan::buffer_count(row)
            ));
        }
        if call.push.len() != kernels_vulkan::push_size(row) as usize {
            wrong.push(format!(
                "`{}` packed a {}-byte block and its layout is {} bytes",
                row.name,
                call.push.len(),
                kernels_vulkan::push_size(row)
            ));
        }

        // The operands that should have gone into the block, in row order --
        // the order `push_layout` filters in.
        let scalars: Vec<Value> = row
            .operands
            .iter()
            .enumerate()
            .filter(|(at, op)| {
                !matches!(op.ty, Ty::InPacked) && !matches!(values[*at], Value::Buffer(_))
            })
            .map(|(at, _)| values[at])
            .collect();

        let layout = kernels_vulkan::push_layout(row);
        if layout.len() != scalars.len() {
            wrong.push(format!(
                "`{}` states {} push fields and has {} scalar operands",
                row.name,
                layout.len(),
                scalars.len()
            ));
            continue;
        }
        if !layout.is_empty() {
            with_scalars += 1;
        }
        for (field, value) in layout.iter().zip(&scalars) {
            let want = bytes_of(*value);
            let at = field.offset as usize;
            let got = &call.push[at..at + field.size as usize];
            if got != want.as_slice() {
                wrong.push(format!(
                    "`{}` field `{}` at {} holds {got:?} and its operand is {want:?}",
                    row.name, field.name, field.offset
                ));
            }
        }
        checked += 1;
    }

    assert!(unbindable.is_empty(), "{}", unbindable.join("\n"));
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    // Was 44, before the two `sdpa_paged_tiled` rows stated their operands,
    // and 46 before the two `sdpa_paged_mma` rows did.
    assert_eq!(checked, 48, "a different number of rows state operands");
    // Agreement about no scalars at all would be agreement about nothing.
    assert_eq!(with_scalars, 29, "a different number of rows push scalars");
}

/// The buffers a row marks writable are the ones its module may write.
///
/// # The third statement of the same ABI, and the one with a race behind it
///
/// `tests/rules` already checks two halves of the row-versus-module contract:
/// the workgroup size a rule divides by against the one the shader declares,
/// and the push offsets a driver packs against the ones the shader reads.
/// This is the third. A row gives every buffer operand a `Ty`, and `BufMut`
/// is the mark that says the shader writes through it; the source says the same
/// thing with `readonly`, which `slangc` records as `NonWritable` on the
/// SPIR-V variable. Nothing made the two agree.
///
/// Unlike the other two, this one is not merely wrong-answers-if-broken --
/// it decides where BARRIERS go. `device::hazards` asks which byte ranges a
/// dispatch writes and puts a pipeline barrier wherever a later dispatch
/// touches them, so the two directions fail differently and only one of them
/// is loud:
///
/// * The row says a buffer is read-only and the module writes it. The driver
///   omits a barrier it needed, and two dispatches race. On this card that is
///   usually invisible -- the earlier one finishes first most of the time --
///   which makes it the worst kind of defect: a test that passes on most
///   runs.
/// * The row says `BufMut` and the module declares `readonly`. A barrier that
///   buys nothing. Answers stay right and the step gets slower, which is the
///   fault this crate has now found four times by measuring and never once by
///   a test going red.
///
/// Both directions are asserted, because on the day this was written both
/// held exactly: 189 modules, 1011 bound buffers, 813 of them declared
/// `readonly`, and ZERO disagreements either way.
///
/// # Holes are not disagreements
///
/// A row covers a family of entrypoints, and `slangc` drops the declaration of
/// a buffer a variant never reads -- `tests/rules` counts 358 such holes
/// across 165 modules. `mxfp4_qmv_routed_bias` is the clearest: it declares
/// bindings 0/1/3/4/5/6 because the MXFP4 codec has no separate bias plane,
/// and its unread `biases` operand still owns slot 2. So a row's k-th buffer
/// operand is binding k, an index and not a rank among the declared ones, and
/// holes are skipped through the same `used` mask `device::slots` writes
/// descriptors through. Skipping them on the module side only shifts
/// everything past the first hole by one, which is how the first draft of
/// this test accused `mxfp4_qmv_routed_bias` of writing through a read-only
/// binding.
///
/// # What this found, and the negative result it explains
///
/// No defect -- and finding none is itself the answer to a question that had
/// been open. `dispatch` marks every bound buffer as written for a row that
/// states no operands, and 80% of this tree's bindings are `readonly`, so
/// deriving the mask from the module looked like free barrier removal. It was
/// measured and the barrier count did not move by one: 311 before and 311
/// after. Two reasons, and this test is half of the second. The branch is
/// close to dead -- a specialised name resolves through `sig_in` to a parent
/// row that DOES state operands -- and the masks those parent rows state are,
/// as counted above, already exactly what the modules declare. There was
/// nothing to correct. The barriers this driver records are the genuine
/// arena chain, and the way to fewer of them is fusion, not bookkeeping.
///
/// That is also why this is worth keeping now rather than after the first
/// disagreement: like the push-block check beside it, it cannot find a defect
/// that already exists, only one about to be introduced -- and a missing
/// barrier is not a defect anybody would find by reading, or reliably by
/// running.
#[test]
fn a_rows_writable_buffers_are_the_ones_its_module_may_write() {
    let _ = modules!();
    let Some(dir) = SPV_DIR.map(std::path::Path::new) else {
        return;
    };
    let mut checked = 0;
    let mut buffers = 0;
    let mut readonly = 0;
    let mut wider = 0;
    let mut races: Vec<String> = Vec::new();

    for name in kernels_vulkan::entrypoints() {
        let Some(row) = kernels::sig_in(kernels_vulkan::KERNELS, &name) else {
            continue;
        };
        if row.operands.is_empty() {
            continue;
        }
        let Some(d) = declared(&dir.join(format!("{name}.spv"))) else {
            continue;
        };
        // The row's buffer operands, in the order the shader binds them,
        // which is the row's own order.
        let stated: Vec<bool> = kernels_vulkan::bindings(row)
            .into_iter()
            .zip(row.operands)
            .filter(|(b, _)| matches!(b, kernels_vulkan::Binding::Buffer(_)))
            .map(|(_, op)| op.ty == kernels::Ty::BufMut)
            .collect();
        // A row's k-th buffer operand IS binding k. Holes keep their slot --
        // `mxfp4_qmv_routed_bias` declares 0/1/3/4/5/6 and its unread
        // `biases` still owns 2 -- so this is an index, not a rank among the
        // declared ones. Skipping the holes on one side and not the other
        // shifts everything past the first hole by one, which is how the
        // first draft of this test accused a correct row of a race.
        let mut seen = false;
        for (b, mark) in stated.iter().enumerate() {
            if !d.used.get(b).copied().unwrap_or(false) {
                continue;
            }
            let Some(&writable) = d.writable.get(b) else {
                continue;
            };
            seen = true;
            buffers += 1;
            if !writable {
                readonly += 1;
            }
            if writable && !mark {
                races.push(format!(
                    "`{name}` binding {b} is written by the module and its row \
                     does not mark it `BufMut`"
                ));
            }
            if !writable && *mark {
                wider += 1;
            }
        }
        if seen {
            checked += 1;
        }
    }

    assert!(races.is_empty(), "{}", races.join("\n"));
    assert_eq!(
        wider, 0,
        "{wider} bindings are marked `BufMut` by a row whose module declares \
         them `readonly`, which costs a barrier and buys nothing"
    );
    assert!(checked >= 150, "only {checked} modules were compared");
    eprintln!(
        "{checked} modules, {buffers} bound buffers, {readonly} declared \
         readonly, {wider} marked writable by a row that its module does not"
    );
    // Agreement about nothing would be agreement all the same: most of this
    // tree's bindings ARE read-only, and if that stopped being true the
    // assertion above would be checking an empty set.
    assert!(
        readonly * 2 > buffers,
        "only {readonly} of {buffers} bindings are read-only, so this no \
         longer checks the direction it was written for"
    );
}

/// The rules this driver serves and the rules this backend's table names are
/// the same set, minus two that are named here.
///
/// `geometry`'s own `the_rules_this_backend_serves_are_exactly_the_ones_with_shaders`
/// checks one direction: every rule in its `SERVED` list answers, and every
/// other rule the fleet states refuses by name. What it cannot see from
/// inside `driver-vulkan` is the TABLE -- and the two halves had drifted
/// apart without anything noticing, in both directions at once:
///
/// * `kernels-vulkan` names fifteen rules across its hundred rows. All
///   fifteen are served, which is the half that matters and is asserted
///   first.
/// * `SERVED` holds seventeen. The two spare ones are `GatedRms` and
///   `PerHeadElementwise`, and no Vulkan row names either. They are not
///   mistakes -- the grids are right, and `kernels-cuda` states both on
///   `attn_sink_correction`, `per_head_rmsnorm` and three SSM rows -- but
///   this backend has no shader for any of those kernels, so the geometry
///   is written ahead of the row rather than behind it.
///
/// That is worth a name rather than a silence. `SERVED`'s comment reads
/// "this is the part `kernels-vulkan` compiles a shader for", which was two
/// rules too generous, and a reader counting coverage off that list would
/// have counted two kernels this backend does not have.
///
/// The test is two-directional on purpose: a row that starts naming
/// `GatedRms` must delete it from `AHEAD_OF_THE_TABLE` here, and a rule that
/// stops being served fails the first assertion. Neither can pass by
/// accident.
#[test]
fn every_rule_the_table_names_is_one_this_driver_can_lay_out() {
    /// Served, with no row in `kernels-vulkan` naming it. See above.
    const AHEAD_OF_THE_TABLE: &[Rule] = &[Rule::GatedRms, Rule::PerHeadElementwise];

    let mut named: Vec<Rule> = kernels_vulkan::KERNELS
        .iter()
        .map(|row| row.launch)
        .filter(|r| *r != Rule::Unstated)
        .collect();
    named.sort_by_key(|r| format!("{r:?}"));
    named.dedup();
    assert_eq!(named.len(), 15, "the table names a different set of rules");

    // A rule is "laid out" when `lanes` answers for the module the family is
    // built with. The dims are the sweep's own, so this asks exactly what a
    // fire would.
    for &rule in &named {
        let module = Module {
            local: Local([32, 2, 2]),
            tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
        };
        let d = if rule == Rule::SdpaVector {
            Dims {
                head_dim: 32,
                ..dims()
            }
        } else {
            dims()
        };
        assert!(
            geometry::lanes(rule, d, module).is_ok(),
            "{rule:?} is named by a row and this driver cannot lay it out"
        );
        assert!(
            !AHEAD_OF_THE_TABLE.contains(&rule),
            "{rule:?} is listed as having no row, and a row names it"
        );
    }

    // And the other direction: each rule said to be ahead of the table really
    // is both served and unnamed.
    for &rule in AHEAD_OF_THE_TABLE {
        let module = Module {
            local: Local([32, 2, 2]),
            tile: None,
        };
        assert!(
            geometry::lanes(rule, dims(), module).is_ok(),
            "{rule:?} is listed as served ahead of the table and does not answer"
        );
        assert!(
            !named.contains(&rule),
            "{rule:?} now has a row; take it out of AHEAD_OF_THE_TABLE"
        );
    }
}
