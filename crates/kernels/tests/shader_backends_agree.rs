//! The three shader backends' tables, compared.
//!
//! `.wiki/kernel-x/refactor-bigplan.md` §0: the three are **one table written
//! down three times**. Measured over all 300 statements: the 100 names are the
//! same set in all three, `launch` is identical in 100 of 100 rows, `axes` in
//! 100 of 100, `operands` agrees on 94 of 100 — and the statements that differ
//! are drift rather than difference.
//!
//! **Nothing in this workspace compared them**, which is why the drift was
//! found by three greps in an afternoon rather than by a test, and why it had
//! been there long enough that nobody could say which side was right. This is
//! the missing comparison, and it is the same defect class
//! `refactor-plan-followup.md` §0 named once and generally: *a hand-transcribed
//! fact outlived the test that checked it.*
//!
//! # Why it is here and not later
//!
//! The bigplan's §3 gate compares `ROUTINES` — derived argument lists — and
//! cannot exist until the ports do. This one compares the tables that exist
//! today, and it is Stage 0's verification: *"settle the seven drifted
//! statements and make the three tables agree, so what gets frozen into three
//! sets of bodies is one decision rather than an accident."* A cleanup that
//! lands without it blesses whatever it finds.
//!
//! When a backend crosses to routines its rows leave `KERNELS`, and this test
//! compares what is left. It goes quiet family by family and the §3 gate takes
//! over; that hand-off is deliberate and is why both exist.
//!
//! # What is compared, and what is deliberately not
//!
//! Compared: the name set, and per name `launch`, `axes`, the operand list
//! (type, nullability and source, in order), the four `*_param` columns,
//! `whole`, `in_place` and `depth_prefix_plan`. These are facts about the
//! KERNEL.
//!
//! Not compared: `file` (three shader languages, three extensions), `symbol`
//! spellings beyond the row name, and anything about a device. Those are
//! properly per-backend, and a gate that failed on them would be a gate people
//! learn to edit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use kernels::KernelSig;

/// One backend's table, under the name this file reports it by.
struct Table {
    what: &'static str,
    rows: &'static [KernelSig],
    /// The names this backend's table used to hold. See [`retired_rows`].
    retired: &'static [&'static str],
}

fn tables() -> Vec<Table> {
    vec![
        Table {
            what: "metal",
            rows: kernels_metal::KERNELS,
            retired: kernels_metal::retired_rows(),
        },
        Table {
            what: "vulkan",
            rows: kernels_vulkan::KERNELS,
            retired: kernels_vulkan::retired_rows(),
        },
        Table {
            what: "wgpu",
            rows: kernels_wgpu::KERNELS,
            retired: kernels_wgpu::retired_rows(),
        },
    ]
}

/// A statement the three tables are ALLOWED to disagree about, and why.
///
/// Every entry needs a sentence. A bare name would make this list the place
/// disagreements go to stop being questions, which is the failure mode of
/// every exceptions list — so the test requires the sentence to be non-empty
/// and the entry to still be disagreeing.
///
/// These are the seven `refactor-bigplan.md` §1.1 found. **Neither looks like
/// a backend difference**; both look like a transcription made once and not
/// made again, and Stage 0 is where they get settled with the code in front of
/// whoever settles them. Until then they are written down rather than
/// tolerated silently.
const DRIFTED: &[(&str, &str)] = &[
    (
        "sdpa_paged_decode",
        "SETTLED, and metal is the wrong one. The mask stride is a FIRE fact: \
         a custom mask is a rectangle the DRIVER stages, one byte per (row, \
         key), so its pitch is whatever the driver made the widest row -- \
         which is why `Source::AttentionMaskStride` exists at all. metal reads \
         `Source::Param(3)` instead, the TEXT's scalar, and `llama_like` \
         states a literal 0 there with the comment that this text carries no \
         custom mask. `driver-metal` does stage masks (it resolves \
         `AttentionMask` and `AttentionMaskEnabled`) and mentions \
         `AttentionMaskStride` in ZERO places, so a deployment with a custom \
         mask reads every row's mask at pitch 0. Six rows share it; that \
         crate's fix. Stays here until then.",
    ),
    (
        "sdpa_paged_decode_sink",
        "As `sdpa_paged_decode`: the mask stride is a fire fact and metal \
         reads a text scalar that is zero.",
    ),
    (
        "sdpa_paged_mma",
        "As `sdpa_paged_decode`: the mask stride is a fire fact and metal \
         reads a text scalar that is zero.",
    ),
    (
        "sdpa_paged_mma_sink",
        "As `sdpa_paged_decode`: the mask stride is a fire fact and metal \
         reads a text scalar that is zero.",
    ),
    (
        "sdpa_paged_tiled",
        "As `sdpa_paged_decode`: the mask stride is a fire fact and metal \
         reads a text scalar that is zero.",
    ),
    (
        "sdpa_paged_tiled_sink",
        "As `sdpa_paged_decode`: the mask stride is a fire fact and metal \
         reads a text scalar that is zero.",
    ),
    (
        "route_gather",
        "SETTLED, and vulkan is the wrong one. wgpu states `rows_param = \
         Some(4)` because this statement's rows are the SORTED STACK -- \
         `MoeRouteParams::padded` -- and not the fire's tokens; given the \
         fire's count the gather covers a quarter of its own output at \
         `top_k = 4` and leaves the rest whatever the arena held. \
         `driver-vulkan` cannot state it: it reads `rows_param` in ZERO places \
         and its `Dims.rows` is unconditionally `launch.rows.end - \
         launch.rows.start`. The fix is two-part and is that crate's -- the \
         column on the row, and the driver reading it. Stays here until then.",
    ),
];

/// Everything about a row that is a fact about the KERNEL rather than a fact
/// about a device, rendered so two backends' answers can be compared.
fn kernel_facts(sig: &KernelSig) -> String {
    let operands: Vec<String> = sig
        .operands
        .iter()
        .map(|o| {
            format!(
                "{}:{:?}{}<-{:?}",
                o.name,
                o.ty,
                if o.nullable { "?" } else { "" },
                o.source
            )
        })
        .collect();
    format!(
        "launch={:?} axes={:?} grid={:?} head={:?} heads={:?} rows={:?} \
         whole={} in_place={:?} depth_prefix_plan={} operands=[{}]",
        sig.launch,
        sig.axes.iter().map(|a| a.points).collect::<Vec<_>>(),
        sig.grid_param,
        sig.head_param,
        sig.heads_param,
        sig.rows_param,
        sig.whole,
        sig.in_place,
        sig.depth_prefix_plan,
        operands.join(", ")
    )
}

/// The three tables name the same kernels.
///
/// Not a coincidence and not an aspiration: `kernels-wgpu`'s own table test
/// says the row count *"is `kernels-metal`'s, and that is the point rather
/// than a coincidence: this backend's coverage is DEFINED as its sibling's"*.
/// Each of the three asserts its own count against 100 and none of them
/// compares the NAMES, so three tables could hold a hundred rows each and
/// disagree about which hundred.
#[test]
fn the_three_shader_backends_name_the_same_kernels() {
    let tables = tables();
    let sets: Vec<(&str, BTreeSet<&str>)> = tables
        .iter()
        .map(|t| (t.what, t.rows.iter().map(|r| r.name).collect()))
        .collect();

    // Ported rows LEAVE `KERNELS`, so a backend part-way through the crossing
    // is a subset rather than a mismatch. The union is what must still be the
    // hundred, and a name in one table and absent from another is only news
    // when the second table has not started crossing.
    //
    // The union is over rows, ROUTINES and RETIREMENTS, and it needed all
    // three in that order. Rows alone read 99 once all three backends had
    // retired `sample`: `argmax_logits` was in no table and the invariant had
    // not been broken, the question had. Adding routines fixed that, because
    // a kernel is a row until its family crosses and a routine afterwards.
    // Then metal crossed all ten families, and `silu_mul_strided` -- DARK,
    // with a row's name and no routine anywhere -- fell out of both planes at
    // once. `retired_rows` is each backend's own record of what it let go,
    // which is what keeps this a claim about the SHADER TREE rather than
    // about three lists that shrink.
    let crossed = [
        kernels_wgpu::declared(),
        kernels_vulkan::declared(),
        kernels_metal::declared(),
    ];
    let union: BTreeSet<&str> = sets
        .iter()
        .flat_map(|(_, s)| s.iter().copied())
        .chain(crossed.iter().flat_map(|d| d.iter().map(|x| x.name)))
        .chain(tables.iter().flat_map(|t| t.retired.iter().copied()))
        .collect();
    assert_eq!(
        union.len(),
        100,
        "the union of the three tables is {} kernels, not the hundred all \
         three declare",
        union.len()
    );

    for (what, set) in &sets {
        if set.len() == 100 {
            continue;
        }
        // A shrinking table is a port in progress; say so rather than fail,
        // and name what it has given up so the countdown is readable.
        let gone: Vec<&str> = union.difference(set).copied().collect();
        println!("{what} has crossed {} rows: {gone:?}", gone.len());
    }
}

/// Where the three tables disagree, the disagreement is written down.
///
/// This is `refactor-bigplan.md` §1.1 as a test. It compares only the columns
/// that are facts about the kernel — see the module docs for what is left out
/// and why — and it fails on a disagreement that is not in [`DRIFTED`], and on
/// a [`DRIFTED`] entry that has stopped disagreeing.
#[test]
fn the_three_tables_disagree_only_where_it_is_written_down() {
    let tables = tables();
    let mut by_name: BTreeMap<&str, Vec<(&str, String)>> = BTreeMap::new();
    for t in &tables {
        for row in t.rows {
            by_name
                .entry(row.name)
                .or_default()
                .push((t.what, kernel_facts(row)));
        }
    }

    let excused: BTreeMap<&str, &str> = DRIFTED.iter().copied().collect();
    assert_eq!(
        excused.len(),
        DRIFTED.len(),
        "a kernel is named twice in DRIFTED"
    );
    for (name, why) in DRIFTED {
        assert!(
            why.len() > 30,
            "`{name}` is excused without a sentence saying which backend is \
             different and why, which makes this list the place questions go \
             to stop being questions"
        );
    }

    let mut disagreeing = BTreeSet::new();
    let mut report = Vec::new();
    for (name, answers) in &by_name {
        // One backend holding the row is not a disagreement: the others may
        // have crossed to routines.
        if answers.len() < 2 {
            continue;
        }
        let first = &answers[0].1;
        if answers.iter().all(|(_, facts)| facts == first) {
            continue;
        }
        disagreeing.insert(*name);
        if !excused.contains_key(name) {
            report.push(format!(
                "`{name}`:\n{}",
                answers
                    .iter()
                    .map(|(what, facts)| format!("    {what:8} {facts}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }

    assert!(
        report.is_empty(),
        "the three shader tables disagree about {} statement(s) that nothing \
         explains. Each is a fact about the KERNEL, so at most one of the \
         answers is right — settle it, or add it to DRIFTED with a sentence \
         saying which backend is different and why:\n\n{}",
        report.len(),
        report.join("\n\n")
    );

    // STALE means SETTLED, not UNOBSERVABLE.
    //
    // An entry stops being checkable the moment a backend retires the row it
    // was about, and Stage 4 retires all of them. Metal finished first, and
    // the six `sdpa_paged_*` entries — every one of which says *"metal is the
    // wrong one"* — immediately looked settled: the two tables still holding a
    // row are wgpu's and vulkan's, and those two always agreed. Deleting them
    // would have recorded progress that did not happen and erased a defect
    // that is still there.
    //
    // So an entry is stale only when EVERY table still states the name and
    // they agree. Once one has retired it, the drift is carried by
    // `the_two_settled_drifts_are_still_true_of_the_drivers_they_name`, which
    // reads the DRIVERS and does not care what any table says — and which
    // still reports `AttentionMaskStride` in zero places in `driver-metal`.
    let all = tables.len();
    let settled: Vec<&str> = excused
        .keys()
        .copied()
        .filter(|n| !disagreeing.contains(n) && by_name.get(n).is_some_and(|a| a.len() == all))
        .collect();
    assert!(
        settled.is_empty(),
        "{settled:?} no longer disagree, so their DRIFTED entries are stale. \
         Delete them — that edit is what records the progress."
    );
}

/// A kernel is a ROW or a ROUTINE, and during a family's port it is both.
///
/// `refactor-bigplan.md` §8 asks for the union of `KERNELS` and `ROUTINES` to
/// still be the hundred names per backend, and for the two to be **pairwise
/// disjoint**. Disjoint is the end state: §7 Stage 3 says a family's bodies,
/// its driver arms and its `kernel!` rows land in ONE commit, so both planes
/// exist only *within* a family and never across one.
///
/// This tree does not meet that today, and what is written down here is the
/// exact list rather than the claim -- or even a count of it.
///
/// A family's rows cannot come off when its bodies land, because deleting a
/// row means the driver can reach the routine instead, and NOTHING in this
/// workspace can yet. A row states which trace operand fills each shader
/// binding (`In(0), Weight(0), Out(0)`); a `Routine`'s signature states the
/// types and their [`Provenance`], which is `Trace` or `Env` and not WHICH
/// trace operand. That mapping lives only in the row. The arm that would
/// carry it is `refactor-bigplan.md` §7's "driver arms", and grepping the
/// workspace for a consumer of `Provenance` finds tests and nothing else --
/// including in `kernels-cuda-new`, the port every other backend is copying.
/// So Stage 3's one-commit rule is aspirational for all three of us until
/// somebody builds that seam, and until then every crossed family is stated
/// twice.
///
/// Which makes a COUNT the wrong instrument. `<= DUAL` cannot tell a family
/// crossing from a family being un-crossed while another crosses, and a bare
/// number does not say which kernels are living in two planes -- which is the
/// thing a person reading a failure actually needs. So the list is written
/// out. Adding a family is one edit that names it; removing a family's rows
/// is one edit that deletes it; and a name appearing in both planes without
/// anybody writing it down is a failure, which is what the count was for.
#[test]
fn the_kernels_stated_twice_are_the_ones_written_down() {
    /// Names held by BOTH a table and a routine set, per backend.
    ///
    /// Sorted, because they are compared as sets and a set has one spelling.
    ///
    /// wgpu's six are its `layout` family, crossed first because the
    /// axis-suffix fork of `wgpu-refactor.md` §4 needed a family with two
    /// instantiation axes to settle at all. vulkan's are `sample`, the
    /// one-kernel family that exercises the whole surface, and then `mlp` and
    /// `layout` and `rope` and `ssm` and `norm` and `moe` and `attn` and
    /// `quant` and `ptir`, in `refactor-bigplan.md` §7's size order.
    ///
    /// **metal's is empty.** Not because it never crossed but because it has
    /// finished: a name is stated twice only while a family's rows and its
    /// routines both exist, and metal has no rows. That is what this list
    /// measures, and reaching zero is what it was counting toward.
    const DUAL: &[(&str, &[&str])] = &[
        (
            "wgpu",
            &[
                // SORTED, because the test compares this against a sorted set
                // and an out-of-order name reads as a mismatch of twenty
                // against twenty.
                //
                // `norm` is the first LIVE family on this backend -- every
                // layer of every model names `rms_single_row`, and a mixture
                // names `residual_add`. The rest are the two DARK families
                // and `layout`, crossed for the reason `kernels-metal` gives
                // at its own crossing of the same pair: no text names those
                // symbols here, so a mistake in the entrypoint, the grid or
                // the argument order cannot change what any model computes.
                "add_bias",
                // NEITHER `argmax_logits` NOR `copy_logits_bf16` is here any
                // more: Stage 3 took both rows off as their arms landed, so
                // each is stated once, by its routine, and this list is the
                // names stated TWICE. The two dark families, on all three
                // backends now.
                "cast_qmm_input_bfloat16_to_float16",
                "cast_qmm_input_strided_bfloat16_to_float16",
                "combine_sorted",
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
                "split_qkv_bf16",
                "vnorm_single_row",
            ],
        ),
        (
            "vulkan",
            &[
                "add_bias",
                // `argmax_logits` and `copy_logits_bf16` are NOT here any
                // more: `7d2945eac` took their rows off, so they are stated
                // once, by the routine. Vulkan is the fleet's first.
                "cast_qmm_input_bfloat16_to_float16",
                "cast_qmm_input_strided_bfloat16_to_float16",
                "combine_sorted",
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
                "split_qkv_bf16",
                "vnorm_single_row",
            ],
        ),
        // Metal's two first crossings are both kernels no text names —
        // `argmax_logits` and `copy_logits_bf16` are dark on that backend, so
        // the crossing could not change what any model computes. That is the
        // property to pick a first family for, and it is why these are the
        // first two rather than the ones a driver leans on.
        (
            "metal",
            // EMPTY, and it is the milestone. Every metal family has retired
            // its `kernel!` rows, so no name on this backend is stated twice
            // any more -- the routine is the only statement of every one of
            // the hundred. Ninety-eight names were deleted from this list in
            // one commit, which is `refactor-bigplan.md` §7 Stage 4 finished
            // here, and the countdown above fell by the same ninety-eight.
            //
            // The entry stays rather than being dropped: the loop below
            // requires every backend it reads to appear, and an absent
            // backend and a crossed one would otherwise look the same.
            &[],
        ),
    ];

    // JOINING THIS GATE, for the next backend to port a family: expose
    //
    //     pub fn declared() -> Vec<kernels::routine::Declared>
    //
    // over your `ROUTINES` -- `Routine::declared()` is the per-row view -- and
    // add one line below. That is the whole of it, and from then on your
    // routines are compared against your own rows while both exist, and
    // against the other backends' once two of you have ported the same
    // family.
    //
    // `kernels-metal` has crossed ALL TEN. It started with `sample` and
    // `ptir` — its two one-kernel families, both dark, so neither crossing
    // could change what a model computes — and finished the remaining eight
    // once `driver-metal` stopped resolving symbols through the table at all.
    // Its `KERNELS` is an empty slice, which is what makes the union
    // assertion below load-bearing rather than arithmetic: it is the only
    // line left that proves the hundred are all still stated somewhere.
    let crossed: Vec<(&str, Vec<&str>, BTreeSet<&str>)> = vec![
        (
            "wgpu",
            kernels_wgpu::declared().iter().map(|d| d.name).collect(),
            kernels_wgpu::KERNELS.iter().map(|r| r.name).collect(),
        ),
        (
            "vulkan",
            kernels_vulkan::declared().iter().map(|d| d.name).collect(),
            kernels_vulkan::KERNELS.iter().map(|r| r.name).collect(),
        ),
        (
            "metal",
            kernels_metal::declared().iter().map(|d| d.name).collect(),
            kernels_metal::KERNELS.iter().map(|r| r.name).collect(),
        ),
    ];

    for (what, routines, rows) in &crossed {
        let names: BTreeSet<&str> = routines.iter().copied().collect();
        assert_eq!(
            names.len(),
            routines.len(),
            "{what} declares a routine name twice"
        );

        // Rows, routines AND the names the rows gave up. A kernel a backend
        // has fully crossed is in neither plane -- metal's `silu_mul_strided`
        // is DARK, so it has a row's name and no routine at all, and once its
        // row retired nothing else would have carried the name. `retired_rows`
        // is each backend's own statement of what it let go, so this stays a
        // claim about the hundred rather than about two lists that shrink.
        let retired: BTreeSet<&str> = tables()
            .iter()
            .find(|t| t.what == *what)
            .map_or(BTreeSet::new(), |t| t.retired.iter().copied().collect());
        let union: BTreeSet<&str> = names
            .union(rows)
            .copied()
            .chain(retired.iter().copied())
            .collect();
        assert_eq!(
            union.len(),
            100,
            "{what}'s rows, routines and retirements together are {} kernels, \
             not the hundred. A port that loses a name loses it silently: the \
             row is gone, no routine answers for it, and nothing recorded it.",
            union.len()
        );

        let both: Vec<&str> = names.intersection(rows).copied().collect();
        let written: Vec<&str> = DUAL
            .iter()
            .find(|(b, _)| b == what)
            .map_or(&[][..], |(_, names)| names)
            .to_vec();
        assert_eq!(
            both,
            written,
            "{what} states {} kernels in both a table and a routine set, and \
             DUAL above writes down {}. Both directions are a real event and \
             neither is a formality: a name here that is not there is a \
             family ported without anyone recording that it now lives in two \
             places at once, and a name there that is not here is a family \
             whose rows came off -- delete the line, because that edit is the \
             record of it.",
            both.len(),
            written.len()
        );
    }

    let crossed_backends: BTreeSet<&str> = crossed.iter().map(|(w, _, _)| *w).collect();
    for (what, _) in DUAL {
        assert!(
            crossed_backends.contains(what),
            "DUAL writes down `{what}`, which is not in the list of backends \
             this gate reads. A backend cannot be checked by being mentioned."
        );
    }
}

/// A ported routine takes exactly the operands its row states.
///
/// `refactor-bigplan.md` §3's cross-backend gate compares `Routine::args`
/// between backends, and cannot run until two of them have ported the same
/// family. This is the same comparison turned inward, where it is available
/// today: **while a kernel is stated twice, the routine's TRACE-provenance
/// arguments, in order, must be exactly the row's operand types.**
///
/// It is the check the port itself most needs. A body that drops an operand,
/// or takes two in the other order, still compiles, still dispatches and still
/// returns `Ok` — it binds the wrong buffer to the wrong slot and computes
/// nonsense. Nothing else in either plane compares them.
///
/// `Env` arguments are excluded on purpose and are the interesting half: they
/// are the facts a row POINTED AT and did not carry — `width` and `rows` were
/// `LaunchRule::Elementwise` reading the rectangle, `group` and `bits` were an
/// axis suffix on the entrypoint name. That a signature states them and a row
/// could not is the whole reason for the refactor, so the gate requires the
/// trace half to match and says nothing about the rest.
#[test]
fn a_ported_routine_takes_exactly_the_operands_its_row_states() {
    let backends: Vec<(&str, Vec<kernels::routine::Declared>, &[KernelSig])> = vec![
        ("wgpu", kernels_wgpu::declared(), kernels_wgpu::KERNELS),
        (
            "vulkan",
            kernels_vulkan::declared(),
            kernels_vulkan::KERNELS,
        ),
        ("metal", kernels_metal::declared(), kernels_metal::KERNELS),
    ];

    let mut compared = 0usize;
    for (what, declared, table) in backends {
        let rows: BTreeMap<&str, &KernelSig> = table.iter().map(|r| (r.name, r)).collect();
        for d in declared {
            let Some(row) = rows.get(d.name) else {
                // Its rows are gone: the family has fully crossed, and §3's
                // cross-backend gate is what covers it from then on.
                continue;
            };
            // An UNSTATED row makes no claim to compare against. 285 of
            // `kernels-vulkan`'s 481 entrypoints are dispatched through the
            // plan's own operand order rather than through a stated list --
            // `vulkan-refactor.md` §8, deferred deliberately -- and
            // `sample::argmax_logits` is one of them. Comparing a body's four
            // buffers against an empty list would fail on the one thing that is
            // not a disagreement: there is no second statement to disagree with.
            //
            // This is not a hole the gate leaves open. A row with no operands
            // never protected anything, and the body is now the ONLY statement of
            // the binding order, which is strictly more than the row said.
            if row.operands.is_empty() {
                continue;
            }
            let stated: Vec<kernels::Ty> = row.operands.iter().map(|o| o.ty).collect();
            let taken: Vec<kernels::Ty> = d
                .args
                .iter()
                .filter(|(_, prov)| *prov == kernels::routine::Provenance::Trace)
                .map(|(ty, _)| *ty)
                .collect();
            assert_eq!(
                taken, stated,
                "{what}'s `{}` takes {taken:?} where its row states {stated:?}. A \
             body that drops an operand or reorders two still compiles, still \
             dispatches and still returns Ok -- it binds the wrong buffer to \
             the wrong slot. In this tree that is not hypothetical: the shader \
             binds `rms_single_row` as 0=x, 1=w, 2=out and the trace hands \
             over In(0), Out(0), Weight(0), so a positional bind reads the \
             output as the weight.",
                d.name
            );
            assert_eq!(
                d.whole, row.whole,
                "{what}'s `{}` disagrees with its row about `whole`",
                d.name
            );
            assert_eq!(
                d.in_place, row.in_place,
                "{what}'s `{}` disagrees with its row about `in_place`",
                d.name
            );
            assert_eq!(
                d.depth_prefix_plan, row.depth_prefix_plan,
                "{what}'s `{}` disagrees with its row about `depth_prefix_plan`",
                d.name
            );
            compared += 1;
        }
        let _ = what;
    }
    assert!(
        compared > 0,
        "no kernel is stated twice, so this compared nothing. When that is \
         true because every family has crossed, delete this test -- §3's \
         cross-backend gate is its successor."
    );
    println!("{compared} ported routines compared against their rows");
}

/// The two settled drifts say something about a DRIVER, and this is what
/// keeps those sentences true.
///
/// `DRIFTED` does not merely record that two tables differ — for
/// `route_gather` and the six sdpa rows it says *which backend is wrong and
/// why*, and the why is a claim about a crate this test does not own:
///
/// * `driver-vulkan` reads `rows_param` **nowhere**, so it could not honour
///   the column even if its row carried it;
/// * `driver-metal` resolves `Source::AttentionMask` but mentions
///   `AttentionMaskStride` **nowhere**, so it stages a mask and reads its
///   pitch from a text scalar.
///
/// A judgement about another crate, written into a comment, is the shape this
/// workspace has had to correct repeatedly — so it is asked as a question
/// instead of asserted as an answer. When either driver learns to resolve its
/// column this fails, which is the moment the `DRIFTED` entry stops being
/// true and someone should be looking at it.
///
/// It reads sources rather than calling anything, because neither driver
/// builds on every host that runs this test.
#[test]
fn the_two_settled_drifts_are_still_true_of_the_drivers_they_name() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf();

    /// How many times `needle` appears in a crate's `src`.
    fn mentions(dir: &std::path::Path, needle: &str) -> usize {
        fn walk(dir: &std::path::Path, needle: &str, n: &mut usize) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, needle, n);
                } else if path.extension().is_some_and(|x| x == "rs") {
                    *n += std::fs::read_to_string(&path)
                        .unwrap_or_default()
                        .matches(needle)
                        .count();
                }
            }
        }
        let mut n = 0;
        walk(dir, needle, &mut n);
        n
    }

    let vulkan = root.join("driver-vulkan/src");
    let metal = root.join("driver-metal/src");
    assert!(
        vulkan.is_dir() && metal.is_dir(),
        "the two sibling drivers are not where this test looks: {} and {}",
        vulkan.display(),
        metal.display()
    );

    assert_eq!(
        mentions(&vulkan, "rows_param"),
        0,
        "`driver-vulkan` has learned to read `rows_param`. That is the defect \
         `DRIFTED[\"route_gather\"]` names being half-fixed — check whether \
         `kernels-vulkan`'s row now states `rows_param = Some(4)` too, and if \
         it does, delete the entry."
    );

    assert_eq!(
        mentions(&metal, "AttentionMaskStride"),
        0,
        "`driver-metal` has learned to resolve `AttentionMaskStride`. That is \
         the defect `DRIFTED[\"sdpa_paged_decode\"]` names being half-fixed — \
         check whether its six rows now read `Source::AttentionMaskStride`, \
         and if they do, delete the six entries."
    );

    // And the other half of that claim: metal DOES stage masks, so reading
    // their pitch as a text constant is a defect rather than an absence.
    //
    // Two spellings, because `4d2753b4d` deleted metal's table interpreter and
    // with it the `Source::` enum this used to look for: the same fact is now
    // spelled `FireTable::AttentionMask`, staged by an ARM. The needle
    // followed the rename rather than the claim being dropped — metal still
    // stages masks, which is what makes reading their pitch as a text
    // constant a defect rather than an absence.
    assert!(
        mentions(&metal, "Source::AttentionMask") + mentions(&metal, "FireTable::AttentionMask")
            > 0,
        "`driver-metal` no longer resolves a mask under either spelling. If it \
         has stopped supporting custom masks, the mask-stride entries are not \
         a defect any more and the sentence in DRIFTED is wrong."
    );
}

/// A retired row's ENTRYPOINTS are still in the backend's census.
///
/// The one thing `RETIRED` exists for, asked directly. A row's `axes`
/// GENERATED its entrypoints, so `entrypoints()` used to mean both *"what the
/// table says"* and *"what this backend can do"*; deleting a row separates
/// them, and every sweep keyed on `entrypoints()` follows the table. On wgpu
/// that silently stopped compiling `argmax_logits_bfloat16` on a real adapter
/// while passing, and the loss compounds to a sweep that builds nothing.
///
/// Each crate answers it by folding a `RETIRED` list back in. This does not
/// care how: it asks whether the three censuses are still the SAME 481, which
/// is derivable from the crates as they are and needs no accessor any of them
/// might not have. A backend that retires a family and forgets to state its
/// entrypoints fails here, whichever backend goes first.
#[test]
fn retiring_a_row_does_not_shrink_a_backends_census() {
    let censuses = [
        ("wgpu", kernels_wgpu::entrypoints()),
        ("vulkan", kernels_vulkan::entrypoints()),
        ("metal", kernels_metal::entrypoints()),
    ];
    let (first, reference) = (&censuses[0].0, &censuses[0].1);
    for (what, census) in &censuses[1..] {
        assert_eq!(
            census,
            reference,
            "`kernels-{what}`'s census is {} entrypoints and `kernels-{first}`'s \
             is {}. The crossing moves who NAMES an entrypoint, never whether \
             it exists, so a backend part-way through Stage 3 must still name \
             every one of them -- through its rows, or through a `RETIRED` \
             list once those rows are gone. Whichever side is short has \
             deleted rows without stating what they named, and every sweep \
             keyed on `entrypoints()` there has stopped covering the \
             difference in silence.",
            census.len(),
            reference.len(),
        );
    }
    assert_eq!(reference.len(), 481, "the shared shader census");
}

/// A kernel one backend has crossed and another has not is still compared.
///
/// This closes the hand-off between the two gates above, which had a hole in
/// it. `the_three_tables_disagree_only_where_it_is_written_down` skips a
/// kernel that only one table still holds, because the others may have
/// crossed; `two_backends_that_crossed_the_same_kernel_agree_on_its_signature`
/// only sees kernels that TWO backends have crossed. So a kernel that backend
/// A has crossed and backend B has not was compared by nothing at all — and
/// crossing a family is exactly the act that moves kernels into that state,
/// eleven at a time.
///
/// It is not hypothetical. `neox_freqs_mb`'s vulkan row was BARE, and its
/// crossing records what that cost: a statement had nothing to name and named
/// the DECODE symbol instead, whose shader assigns `row = 0`, so a prefill
/// rotated row zero once per row and left the rest exactly as they arrived.
/// The wgpu row for the same kernel states its operands. Had vulkan crossed
/// rope one commit earlier, that difference would have left the tables and
/// entered nothing.
///
/// The two planes ARE comparable: a routine's TRACE-provenance arguments, in
/// order, are the row's operand list — which is what
/// `a_ported_routine_takes_exactly_the_operands_its_row_states` asserts within
/// a backend. This asserts it across two.
#[test]
fn a_kernel_crossed_on_one_backend_is_compared_against_the_row_on_another() {
    let crossed: Vec<(&str, Vec<kernels::routine::Declared>)> = vec![
        ("wgpu", kernels_wgpu::declared()),
        ("vulkan", kernels_vulkan::declared()),
        ("metal", kernels_metal::declared()),
    ];

    let mut compared = 0usize;
    let mut report = Vec::new();
    for (what, declared) in &crossed {
        for d in declared {
            let taken: Vec<kernels::Ty> = d
                .args
                .iter()
                .filter(|(_, prov)| *prov == kernels::routine::Provenance::Trace)
                .map(|(ty, _)| *ty)
                .collect();

            for t in tables() {
                if t.what == *what {
                    continue;
                }
                let Some(row) = t.rows.iter().find(|r| r.name == d.name) else {
                    continue;
                };
                // An unstated row makes no claim -- there is nothing to
                // disagree with, and the signature is now the only statement
                // of the binding order this kernel has ever had.
                if row.operands.is_empty() {
                    continue;
                }
                let stated: Vec<kernels::Ty> = row.operands.iter().map(|o| o.ty).collect();
                if taken != stated {
                    report.push(format!(
                        "`{}`: {what}'s routine takes {taken:?}, {}'s row \
                         states {stated:?}",
                        d.name, t.what
                    ));
                }
                compared += 1;
            }
        }
    }

    assert!(
        report.is_empty(),
        "a crossed routine and a table row describe the same kernel \
         differently. One of the two is wrong about what the shader binds, and \
         the crossing is what would have hidden it -- a ported row leaves the \
         table and stops being compared:\n  {}",
        report.join("\n  ")
    );
    assert!(
        compared > 0,
        "nothing was compared. Either every backend has crossed the same \
         kernels, in which case the test above covers them and this can go, or \
         the plane lookup has stopped finding rows."
    );
    println!("{compared} routine-against-row pairs compared across backends");
}

/// The countdown: `REMAINING` rows to 0.
///
/// `refactor-bigplan.md` §8. `KERNELS.len()` summed across the three, against
/// a constant that only ever goes DOWN. It makes the dual-maintenance window
/// visible and makes *"we will finish this later"* a number rather than a
/// sentence — which is what stops Stage 5, the only stage that pays for the
/// other four, from being permanently one backend away.
#[test]
fn the_three_tables_only_ever_lose_rows() {
    /// Lower this when a family crosses. It may not be raised.
    ///
    /// 300 → 192. vulkan retired `sample`, `ptir` and four more families;
    /// wgpu retired `sample` and `ptir`, one arm at a time; **metal retired
    /// ALL TEN**, which is `refactor-bigplan.md` §7 Stage 4 finished on one
    /// backend: `kernels_metal::KERNELS` is an empty slice, and the driver
    /// that used to read it resolves every symbol through the stem its
    /// routine registry states.
    ///
    /// A hundred rows is what the stage costs to reach, and what it buys is
    /// that no reader of an `operands`, `launch` or `*_param` column is left
    /// on this backend. Stage 5 deletes those columns, and it can only run
    /// once the LAST backend is here.
    const REMAINING: usize = 192;

    let total: usize = tables().iter().map(|t| t.rows.len()).sum();
    assert!(
        total <= REMAINING,
        "the three tables hold {total} rows and this test allows {REMAINING}. \
         A row was ADDED to a table that is being emptied — if that is \
         deliberate, say so here."
    );
    assert_eq!(
        total, REMAINING,
        "the three tables hold {total} rows, down from {REMAINING}. Lower the \
         constant: that edit is the progress bar."
    );
}

/// How many crossed routines have NO row to be checked against, per backend.
///
/// [`a_ported_routine_takes_exactly_the_operands_its_row_states`] skips a row
/// whose `operands` list is empty, and its comment argues — correctly — that
/// an unstated row never protected anything and the body is now the only
/// statement of the binding order.
///
/// What that argument does not do is say HOW MANY. The answer was FIFTY-ONE,
/// the same fifty-one names on all three backends, which is over half of
/// what each has crossed. metal's is NINETY-NINE now, which is every routine
/// it has: its table is empty, so there is no row to compare against for any
/// of them and the row check is inert on that backend entirely. The rise is
/// the retirement, not a regression -- see the second cause below. I went looking because wgpu's `ssm` crossed with
/// all eight of its rows unfilled and I had just written "all eight
/// signatures matched vulkan's on the first gate run" — true, and true via a
/// DIFFERENT test than the one I had in mind. The row check compared nothing
/// for those eight; what compared them was
/// [`two_backends_that_crossed_the_same_kernel_agree_on_its_signature`],
/// which is routine-against-routine and is independent.
///
/// The number being identical three ways is the useful part: it is a property
/// of the shared TABLE — `vulkan-refactor.md` §8's deferral, where 285 of 481
/// entrypoints are dispatched through the plan's own operand order — and not
/// of any backend's port.
///
/// So the number is asserted. It is not a defect and it does not have to
/// reach zero — filling a row for a symbol no text names would be inventing a
/// claim to check against. It has to be VISIBLE, because "this check ran"
/// and "this check compared something" are different facts and only one of
/// them was being reported.
///
/// What covers those eight instead is real and worth naming: `kernels-wgpu`'s
/// `every_routine_binds_a_buffer_for_every_binding_its_module_declares`
/// measures each dispatch against the parsed `naga` module, which is this
/// backend's own shader rather than a sibling's signature.
#[test]
fn the_crossed_routines_with_no_row_to_check_are_counted() {
    /// Per backend: how many crossed routines have an unstated row.
    ///
    /// May fall freely. A RISE means either a family crossed against rows
    /// that state nothing, or a family retired its rows -- both allowed, both
    /// deliberate, and the second is the end state. metal is at its own
    /// routine count because it has finished: 99 unchecked out of 99 crossed.
    const UNCHECKED: &[(&str, usize)] = &[("metal", 99), ("vulkan", 55), ("wgpu", 51)];

    let backends = [
        ("wgpu", kernels_wgpu::declared(), kernels_wgpu::KERNELS),
        (
            "vulkan",
            kernels_vulkan::declared(),
            kernels_vulkan::KERNELS,
        ),
        ("metal", kernels_metal::declared(), kernels_metal::KERNELS),
    ];
    for (what, declared, table) in backends {
        let rows: BTreeMap<&str, &KernelSig> = table.iter().map(|r| (r.name, r)).collect();
        let mut blind: Vec<&str> = Vec::new();
        let mut retired: Vec<&str> = Vec::new();
        for d in declared {
            match rows.get(d.name) {
                Some(row) if row.operands.is_empty() => blind.push(d.name),
                // Stage 3 took the row off. The row check has nothing to
                // compare for the same reason, so it belongs in the same
                // count — but it is tracked separately because this cause is
                // PERMANENT and the other one is a row waiting to be filled in.
                None => {
                    blind.push(d.name);
                    retired.push(d.name);
                }
                Some(_) => {}
            }
        }
        blind.sort_unstable();
        retired.sort_unstable();
        let want = UNCHECKED
            .iter()
            .find(|(n, _)| *n == what)
            .unwrap_or_else(|| panic!("`{what}` has no entry in UNCHECKED"))
            .1;
        assert_eq!(
            blind.len(),
            want,
            "{what} has {} crossed routines the row check cannot compare -- \
             the row states no operands, or Stage 3 already took the row off \
             -- and this test writes down {want}. Those routines pass the row \
             check without it comparing anything. {blind:?}",
            blind.len()
        );

        // The check `d2f5c6330` wanted and could not run: metal's retirements
        // were in a private `RETIRED` with no accessor. Every crate has
        // `retired_rows()` now, so a routine whose row is gone must be one
        // the crate SAYS it gave up. Subset, not equality -- a retired row
        // with no routine at all is legal and metal has one, `silu_mul_
        // strided`, which is dark. What this forbids is the other direction:
        // a row deleted and not recorded, which reads here as a crossing that
        // happened and everywhere else as a kernel that never existed.
        let recorded: BTreeSet<&str> = tables()
            .iter()
            .find(|t| t.what == what)
            .map_or(BTreeSet::new(), |t| t.retired.iter().copied().collect());
        let unrecorded: Vec<&str> = retired
            .iter()
            .copied()
            .filter(|n| !recorded.contains(n))
            .collect();
        assert!(
            unrecorded.is_empty(),
            "{what} has crossed routines whose rows are gone from `KERNELS` \
             and absent from `retired_rows()`: {unrecorded:?}. The row was \
             deleted without the deletion being stated, so every sweep that \
             folds the retirements back in -- the census, the union above -- \
             is short by exactly these."
        );
    }
}

/// Two backends that crossed the same kernel state the same signature.
///
/// This is `refactor-bigplan.md` §3's gate proper, and until now it had
/// nothing to run on: it needs one kernel ported by two backends, and `layout`
/// is the first -- `kernels-wgpu` crossed it to settle the axis-suffix
/// question and `kernels-vulkan` crossed it as §7's second real family.
///
/// What it compares is everything a `Declared` carries, which is everything
/// that is NOT device-shaped: the argument types and their provenance, whether
/// the statement consumes its whole operand, whether it joins the depth-prefix
/// plan, and which of its operands must be given the same address. Grids,
/// tiers, workgroup sizes and entrypoint spellings are properly per-backend --
/// §2 is the argument for why the bodies are not shared at all -- and none of
/// them are here.
///
/// The reason this is worth a test rather than a convention: §1 measured the
/// three `kernel!` tables and found them to be ONE table written three times.
/// A hundred identical names, `launch` identical 100/100, `axes` identical
/// 100/100, wgpu and vulkan's `operands` identical 100/100, and seven
/// statements differing workspace-wide -- every one of which is written down
/// in `DRIFTED` above. The refactor moves that table into three crates' worth
/// of separate `fn` signatures, which is exactly the move that lets them drift
/// silently for the first time. So the agreement stops being a measurement
/// somebody once took and becomes a thing that fails.
///
/// A real divergence is not impossible, and it is not this test's business to
/// forbid one. It is this test's business to make it an EDIT: a backend that
/// genuinely needs a different operand list adds itself here the way `DRIFTED`
/// records the seven, with the reason written next to it.
/// A crossed kernel two backends declare differently, and why.
///
/// The routine-level twin of [`DRIFTED`], and it works the same way: an entry
/// needs a sentence, and it may only be deleted — when the two agree, the
/// entry is stale and this test says so.
const DIVERGED: &[(&str, &str)] = &[
    // Vulkan's descriptor set holds buffers and its push block holds scalars,
    // and they are two namespaces: `sdpa_sliding.slang` gives `sinks` binding
    // 4, right behind `out_`, because a buffer cannot sit past a push
    // constant. Metal has one flat argument table, and the same shader there
    // declares `sinks [[buffer(14)]]` -- after `window` and both row pitches,
    // so the sinked and unsinked forms share a prefix and the sink is
    // appended. Both are the right order for their own ABI and neither can
    // take the other's without renumbering a shader that is already correct.
    (
        "sdpa_vector_decode_sink",
        "vulkan must put every buffer ahead of its push constants; Metal's \
         flat argument table appends the sink plane past the scalars instead",
    ),
    // `ssm/gdn_prep.slang` declares `struct Push { int row_pitch; int n_scan; }`
    // under `PIE_SCAN` and binds it as a push constant; Metal's counterpart
    // reads the same two numbers out of the params block it already binds, so
    // there they are grid-only. Metal's scan also binds a leading `pad`
    // buffer the Slang module has no binding for at all. Both are in the
    // shaders, and neither is expressible as one signature.
    (
        "gdn_core_recurrent_prefill",
        "vulkan pushes row_pitch and n_scan that metal reads from its params \
         block, and metal binds a leading pad buffer the slang module does \
         not declare",
    ),
    (
        "gdn_prep_prefill",
        "vulkan pushes row_pitch and n_scan that metal reads from its params \
         block, so the same two numbers are bound on one backend and \
         grid-only on the other",
    ),
    // The norm family's four strided/gated forms. Metal sizes the
    // THREADGROUP on the axis -- `grid::rms` has always been `axis / 4`
    // threads, capped at 1024 -- while every Slang module here is compiled at
    // a fixed 256 and walks the axis in a loop. So metal needs the axis (or
    // the head width) as a grid fact and vulkan does not, in exactly the four
    // places where the axis is not ALSO needed to count the norms in a row.
    //
    // The four PACKED forms are not here, and that is the useful half of this
    // finding: there the axis divides the row into `width / axis` reductions,
    // so both backends need both numbers, vulkan was launching one workgroup
    // per row, and it took metal's signature rather than an excuse.
    (
        "gated_rms",
        "metal's threadgroup is the value-head width and vulkan's is a fixed \
         256 that loops, so the head dim is a grid fact on one backend and \
         a shader constant on the other",
    ),
    (
        "gated_rms_strided",
        "metal's threadgroup is the value-head width and vulkan's is a fixed \
         256 that loops, so the head dim is a grid fact on one backend and \
         a shader constant on the other",
    ),
    (
        "rms_strided_row",
        "metal sizes the threadgroup on the axis and vulkan compiles a fixed \
         256-wide workgroup that walks it, and a pitched row holds exactly \
         one norm so neither backend needs the axis for the grid's EXTENT",
    ),
    (
        "rms_strided_head_row",
        "metal sizes the threadgroup on the axis and vulkan compiles a fixed \
         256-wide workgroup that walks it, and the head is its own grid axis \
         on both, so the axis is only a group width and only metal's varies",
    ),
    // The moe family's six, in two groups of three, and both groups are a
    // shape this list already carries.
    //
    // The three routed GEMMs take a `pad`. Their entrypoints declare buffers
    // 0..=6 and then `tile_expert` at TWELVE, so one argument-table ordinal
    // can serve both the routed GEMM and the routed matvec -- and a Metal
    // argument table is a contiguous run, so the holes must still hold an
    // address. `gdn_core_recurrent_prefill` above is excused for the same
    // reason and says so at more length. The MXFP4 one pads six slots rather
    // than five: it declares nothing at 2 either, because the codec has no
    // zero point to bind where affine puts `biases`.
    (
        "qmm_t_routed",
        "metal binds a pad at slots 7..=11 because its entrypoint declares \
         tile_expert at buffer 12, which is the routed matvec's numbering \
         kept so one argument table serves both pipelines",
    ),
    (
        "qmm_t_routed_fp16",
        "metal binds a pad at slots 7..=11 because its entrypoint declares \
         tile_expert at buffer 12, which is the routed matvec's numbering \
         kept so one argument table serves both pipelines",
    ),
    (
        "mxfp4_qmm_t_routed_bias",
        "metal binds a pad at slot 2 and at 8..=11: tile_expert is at buffer \
         12 and MXFP4 has no zero point to bind where the affine codec puts \
         its biases, so six slots are holes rather than five",
    ),
    // The three routing kernels size their threadgroup on the EXPERT COUNT --
    // `route.metal`'s top-k reduces across simdgroups, so the group is the
    // expert count rounded to a whole simdgroup and clamped at 1024. Every
    // Slang module here is compiled at a flat 1024 and strides. So the expert
    // count is a grid fact on one backend and a shader constant on the other,
    // which is the `norm` family's divergence in a second family.
    (
        "router_topk",
        "metal's threadgroup is the expert count rounded to a simdgroup and \
         vulkan's is a flat 1024 that strides, so the count is a grid fact \
         on one backend and a shader constant on the other",
    ),
    (
        "router_topk_scaled",
        "metal's threadgroup is the expert count rounded to a simdgroup and \
         vulkan's is a flat 1024 that strides, so the count is a grid fact \
         on one backend and a shader constant on the other",
    ),
    (
        "route_sort",
        "metal's ONE threadgroup is as wide as the expert count and vulkan's \
         is a flat 1024 that strides, so the count is a grid fact on one \
         backend and a shader constant on the other",
    ),
];

// The other six `gdn_*` kernels were parked here as UNRESOLVED and are not
// parked here any more, because both halves of that finding were settled
// rather than excused and both crates moved.
//
// (a) ELEMENT TYPES: metal now says `F32s`/`F32sMut` in the four positions
// where `gdn_core.metal` and `gdn_prep.metal` declare `device float*`. The
// typed spelling was not vulkan's preference, it was what all three trees
// declare, so metal took it.
//
// (b) ARITY: metal's extra trailing `Env<i32>` was the better statement and
// not an extra. Its grid takes `rows` and `v_heads` where vulkan took their
// PRODUCT, and the shader takes the product apart again -- `hv = z % Hv`,
// `row = z / Hv` -- so a body handed the product cannot tell one
// factorisation from another and two different routings with the same
// product address different state. Vulkan took metal's.
//
// What is left above is the part that really is two different shaders.

#[test]
fn two_backends_that_crossed_the_same_kernel_agree_on_its_signature() {
    let backends: Vec<(&str, Vec<kernels::routine::Declared>)> = vec![
        ("wgpu", kernels_wgpu::declared()),
        ("vulkan", kernels_vulkan::declared()),
        ("metal", kernels_metal::declared()),
    ];

    // name -> the backends that have crossed it, with what they declared.
    let mut by_name: BTreeMap<&str, Vec<(&str, kernels::routine::Declared)>> = BTreeMap::new();
    for (what, declared) in &backends {
        for d in declared {
            by_name.entry(d.name).or_default().push((what, *d));
        }
    }

    let excused: BTreeMap<&str, &str> = DIVERGED.iter().copied().collect();
    assert_eq!(
        excused.len(),
        DIVERGED.len(),
        "a kernel is named twice in DIVERGED"
    );
    for (name, why) in DIVERGED {
        assert!(
            why.len() > 40,
            "`{name}` is excused without a sentence saying what differs and \
             why, which makes this list the place questions go to stop being \
             questions"
        );
    }

    let mut compared = 0usize;
    let mut report: Vec<String> = Vec::new();
    for (name, ports) in &by_name {
        let Some(((first_what, first), rest)) = ports.split_first() else {
            continue;
        };
        if excused.contains_key(name) {
            // Still compared, so a DIVERGED entry that has stopped diverging
            // fails below rather than sitting here forever.
            if rest.iter().all(|(_, other)| other.args == first.args) {
                panic!(
                    "`{name}` is in DIVERGED and the backends now agree. \
                     Delete the entry -- that edit is what records it being \
                     settled."
                );
            }
            compared += rest.len();
            continue;
        }
        for (what, other) in rest {
            if other.args != first.args {
                report.push(format!(
                    "`{name}`: {what} takes {:?}, {first_what} takes {:?}",
                    other.args, first.args
                ));
            }
            assert_eq!(
                (other.whole, other.depth_prefix_plan, other.in_place),
                (first.whole, first.depth_prefix_plan, first.in_place),
                "`{name}` is stated differently in {what} and {first_what}. \
                 These three are facts about how a TRACE may use the kernel -- \
                 whether it consumes its whole operand, whether it joins the \
                 depth-prefix plan, which operands must be aliased -- and a \
                 trace does not know which backend will run it."
            );
            compared += 1;
        }
    }

    assert!(
        report.is_empty(),
        "{} crossed kernel(s) are declared differently by two backends. A \
         signature is now the only statement of an operand list, so this is \
         either a port that mistyped one or a real divergence nobody wrote \
         down -- record it in DIVERGED with the reason:\n  {}",
        report.len(),
        report.join("\n  ")
    );

    // Thirty-four, which is a count of COMPARISONS and not of kernels: a name
    // two backends carry is one, a name three carry is two. The six of
    // `layout`, crossed by all three, are therefore twelve of it; plus
    // `argmax_logits`, which metal and vulkan both crossed and which was the
    // first kernel this gate compared across two backends that arrived at it
    // independently; plus `mlp`'s four, the first LIVE ones -- every gemma and
    // every gpt-oss layer names one, so unlike `argmax_logits` these are
    // signatures a model's output depends on; plus `ssm`'s eight, which is
    // where the gate first paid for itself. Two of those eight are in DIVERGED
    // and are counted here anyway: an excused divergence is a comparison whose
    // answer is written down, not one that stopped happening. The other six were
    // settled rather than excused, and settling them moved both crates --
    // metal's four float planes stopped being spelled `Buf` when the MSL says
    // `device float*`, and vulkan stopped folding `rows * v_heads` into one
    // number the shader takes apart again. The thirty-fifth is
    // `copy_logits_bf16`, which wgpu and vulkan crossed against shader text
    // that does NOT agree -- wgpu's writes two vocabulary entries per lane and
    // vulkan's writes one -- and the signatures match anyway, because a grid
    // is not an operand. That is the gate's shape working as intended: it
    // compares what a caller must pass, and leaves what a lane does to the
    // family tests that can see the shader.
    //
    // Plus `norm`'s twelve, and they paid the same way: metal's crossing
    // showed that the four PACKED reductions need `width` as well as `axis`,
    // because their base is `group * axis_size` and a row holds `width / axis`
    // of them. Vulkan was launching one workgroup per ROW, which normalizes
    // the first head of a q/k norm and leaves the other thirty-one as the
    // projection wrote them. It took metal's signature. Four of the twelve are
    // in DIVERGED for a difference that really is two shaders -- metal sizes
    // its threadgroup on the axis, vulkan compiles a fixed 256 and loops.
    //
    // Plus `moe`'s thirteen. Six of those are excused just above, in two
    // groups of three, and the other seven matched on the first attempt --
    // including all three routed matvecs, whose twelve arguments are the
    // longest signature either backend has agreed on without an edit.
    //
    // Plus `attn`'s sixteen, fifteen of which matched at once. That includes
    // `kv_append_paged`'s sixteen slots, six of them a ring ABI neither
    // backend reads and both name one by one.
    //
    // Plus `quant`'s thirty-one -- the largest family and the one that needed
    // the least argument, because both backends were ported from the same MLX
    // bodies and the 303 instantiation names are spelled identically. Every
    // one of them agreed.
    assert_eq!(
        compared, 198,
        "{compared} kernels are crossed by more than one backend and compared \
         here, and this test expects 198. It may only RISE: a fall means a name \
         stopped matching its counterpart, which is how a comparison stops \
         happening without anybody deleting it."
    );
}

/// `kernels-metal::kernel_of` answers what a table with ROWS still answers.
///
/// Metal finished Stage 4 first, so its `KERNELS` is empty and every symbol
/// now resolves through the retired-row fallback instead of through a row's
/// own axis expansion. That fallback is a text rule over names, and a text
/// rule can be wrong in a way nothing on that backend can see any more —
/// there is no table left to check it against.
///
/// `kernels-wgpu` still has 98 rows, and `kernels::sig_in` resolves the same
/// 481 entrypoints through them. So the check is available exactly once, from
/// here, while at least one backend is behind: the two answers must be the
/// same name for every point of the shared census.
///
/// # What it caught
///
/// The fallback matched a row name only as a PREFIX. That is right for
/// `rms_single_row_bfloat16` and wrong for `affine_qmv_fast_bfloat16_gs_64_b_4`,
/// where the row is `qmv_fast` and `affine_` is a qualifier the row name does
/// not carry — so every `quant` and `moe` symbol resolved to `None`, and
/// `model-ir`'s load-time check refused to build ANY metal text. The first
/// families to retire were the ones whose rows happen to be prefixes, which is
/// why it surfaced only at the last one.
#[test]
fn metals_symbol_lookup_agrees_with_a_table_that_still_has_rows() {
    let census = kernels_wgpu::entrypoints();
    assert_eq!(census.len(), 481, "the shared shader census");

    let mut checked = 0u32;
    let mut wrong = Vec::new();
    for point in &census {
        let Some(row) = kernels::sig_in(kernels_wgpu::KERNELS, point) else {
            // A family wgpu has retired too. Neither side has a row, so
            // there is nothing to compare — and the count below is what
            // keeps that from quietly becoming everything.
            continue;
        };
        checked += 1;
        match kernels_metal::kernel_of(point) {
            Some(name) if name == row.name => {}
            other => wrong.push(format!(
                "`{point}`: wgpu says `{}`, metal says {other:?}",
                row.name
            )),
        }
    }

    assert!(
        wrong.is_empty(),
        "{} of {checked} entrypoints resolve to a different kernel on the two \
         backends:\n  {}",
        wrong.len(),
        wrong.join("\n  "),
    );
    assert!(
        checked > 400,
        "only {checked} entrypoints were compared. This check is only \
         available while a backend is BEHIND metal's Stage 4, and it stops \
         being available family by family as wgpu catches up — when it reads \
         nothing, delete it rather than letting it pass on an empty loop",
    );
}
