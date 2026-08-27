//! P6, as THIS SHELL bakes it: the exclusive-workspace rule, and the fork
//! assignment it leaves behind.
//!
//! **HOST-ONLY. NO DEVICE.** Everything here is `compile` against the profile
//! `Shell::load` builds, which is a pure function — the point is precisely
//! that a machine with no GPU can say what a machine with one will do.
//!
//! # What the rule is
//!
//! `kernels_cuda::Ctx::scratch` hands back a slab keyed by a static NAME:
//! process-global, grown but never shrunk, and deliberately not per stream,
//! because an entry that allocated per fire could not be captured. Two
//! launches inside one slab at the same instant stage over each other, and
//! the fire computes anyway — nothing faults, a logit moves.
//!
//! The compiler cannot know this: no `Operands` method says which entries
//! reach a slab, and a backend-neutral pass that did would be a compiler that
//! knows a backend's allocator. So the shell says it, as
//! [`driver_cuda::EXCLUSIVE`], and P6 turns it into a dependency edge.
//!
//! # What it costs, said plainly
//!
//! It takes every LINEAR-ATTENTION layer of qwen and kimi off the table. Their
//! two arms — `ssm_gated_delta` beside `ssm_gated_delta_chunked`, over
//! disjoint classes, writing disjoint values — are otherwise a textbook
//! concurrency candidate, and both stage through `attn.ssm_gdn_chunk_qk`. The
//! overlap is real and unavailable; what makes it available is a slab keyed by
//! `(name, stream)` in the kernels plane, which is a change to the frozen side
//! of the seam and is not this wave's. Their full-attention layers still fork,
//! and so do glm-5's and gpt-oss's: those arms are flashinfer entries whose
//! workspace is a `ScheduleSeat` per plan value (build log 21), already
//! disjoint by construction.
//!
//! Gemma is the SKU that forks most, and it is the one the axis campaign
//! built: three attention arms and a qkv pair, in every one of its layers.

use model_compiler::{Budgets, DeviceProfile, compile};
use model_dsl::Platform;
use model_ir::{Operands, Plan};

/// The profile `Shell::load` builds, minus the device probe: the compiler's
/// defaults, this shell's exclusive list, and the L40S SM count the gates run
/// on.
fn profile() -> DeviceProfile {
    DeviceProfile {
        sms: 142,
        exclusive: driver_cuda::EXCLUSIVE
            .iter()
            .map(|op| (*op).to_string())
            .collect(),
        ..DeviceProfile::default()
    }
}

fn budgets_for(plan: &Plan) -> Budgets {
    let seats = plan
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

fn claims_a_slab(plan: &Plan, region: &model_compiler::Region) -> Vec<&'static str> {
    region
        .nodes
        .clone()
        .filter_map(|node| plan.nodes.get(node as usize))
        .map(|node| node.op.name())
        .filter(|name| driver_cuda::EXCLUSIVE.contains(name))
        .collect()
}

#[test]
fn no_two_regions_that_claim_a_slab_are_ever_scheduled_together() {
    let profile = profile();
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        let plan = trace(Platform::Cuda);
        let Ok(baked) = compile(&plan, &budgets_for(&plan), &profile) else {
            continue;
        };
        for &(a, b) in &baked.forks.pairs {
            let (x, y) = (
                claims_a_slab(&plan, &baked.regions[a as usize]),
                claims_a_slab(&plan, &baked.regions[b as usize]),
            );
            if !x.is_empty() && !y.is_empty() {
                wrong.push(format!(
                    "`{sku}`: regions {a} and {b} run beside each other and both \
                     claim a process-global slab — {x:?} against {y:?}",
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **WHAT THE SLAB RULE COSTS, SAID IN THE ONE PLACE IT IS VISIBLE.** The
/// compiler's own catalog test asks the same question at the backend-neutral
/// default, where `exclusive` is empty; this is it with the CUDA answer in,
/// and the rows that MOVE between the two are the rule's whole price.
///
/// **THE PREDICATE IS PINNED, NOT THE COUNT.** How many regions a SKU forks
/// follows from how many layers its model text declares, and those move when
/// somebody edits a family. What must not move without something being wrong
/// is which SKUs the slab rule silences, and by how much RELATIVE to the
/// neutral bake — so the assertion is a comparison between the two profiles
/// rather than a table of numbers, and the numbers ride in the message.
#[test]
fn the_slab_rule_silences_the_arms_that_share_a_staging_plane() {
    let cuda = profile();
    let neutral = DeviceProfile {
        exclusive: Vec::new(),
        ..profile()
    };

    let mut table: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        let plan = trace(Platform::Cuda);
        let free = compile(&plan, &budgets_for(&plan), &neutral).expect("bakes");
        let bound = compile(&plan, &budgets_for(&plan), &cuda).expect("bakes");
        let forked = |baked: &model_compiler::Baked| {
            baked.regions.iter().filter(|r| r.stream != 0).count()
        };
        table.push(format!(
            "  {sku}: neutral {} forked, this shell {} forked",
            forked(&free),
            forked(&bound),
        ));

        // The rule only ever takes forks away: it adds edges to the DAG and
        // an edge cannot create a candidate.
        if forked(&bound) > forked(&free) {
            wrong.push(format!(
                "`{sku}`: the slab rule ADDED forks — {} against the neutral bake's \
                 {}, and an extra dependency edge cannot make a candidate",
                forked(&bound),
                forked(&free),
            ));
        }

        // And where it takes them, it is because two regions really do both
        // claim a slab. Nothing else may lose a fork.
        let claims = bound
            .regions
            .iter()
            .filter(|region| !claims_a_slab(&plan, region).is_empty())
            .count();
        if forked(&bound) < forked(&free) && claims < 2 {
            wrong.push(format!(
                "`{sku}` lost forks to a rule that names {claims} slab-claiming \
                 regions in the whole plan",
            ));
        }
    }

    // **GEMMA IS THE SKU THE RULE COSTS NOTHING**, and it is the subject: its
    // three attention arms are flashinfer entries whose workspace is a
    // `ScheduleSeat` per plan value (build log 21), so none of them is on the
    // list at all.
    let plan = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("the catalog ships gemma")(
        Platform::Cuda,
    );
    let bound = compile(&plan, &budgets_for(&plan), &cuda).expect("bakes");
    if bound.forks.streams < 3 {
        wrong.push(format!(
            "gemma asked for {} streams under this shell's profile, and its three \
             attention arms take no slab",
            bound.forks.streams,
        ));
    }

    assert!(
        wrong.is_empty(),
        "\n{}\n\nthe whole table:\n{}\n",
        wrong.join("\n"),
        table.join("\n"),
    );
}

/// The off arm, through the shell's own door: `PIE_CUDA_STREAMS=off` is the
/// artifact P6 never ran on, not a shell that declines to use one it baked.
///
/// The env read itself is `serve::streams_from_env` and is private; what this
/// asserts is the property it exists to produce, which is the one a
/// measurement's off arm depends on.
#[test]
fn the_off_arm_is_the_artifact_and_not_a_flag() {
    let off = DeviceProfile {
        side_streams: 0,
        ..profile()
    };
    for (sku, _, trace, _) in model::catalog() {
        let plan = trace(Platform::Cuda);
        let baked = compile(&plan, &budgets_for(&plan), &off).expect("bakes");
        assert!(
            baked.regions.iter().all(|r| {
                r.stream == 0 && r.wait.is_empty() && r.open.is_none() && r.close.is_none()
            }),
            "`{sku}` forked with the streams off",
        );
        assert_eq!(baked.forks.events, 0, "`{sku}` minted an event with the streams off");
        assert_eq!(baked.forks.streams, 1, "`{sku}` asked for a stream it will not open");
    }
}
