//! P6, as THIS SHELL bakes it: the exclusive-workspace rule, and the fork
//! assignment it leaves behind.
//!
//! **HOST-ONLY. NO DEVICE.** Everything here is `compile` against the profile
//! `Shell::load` builds, which is a pure function — the point is precisely
//! that a machine with no GPU can say what a machine with one will do.
//!
//! # What the rule is, and why it now names nobody
//!
//! `kernels_cuda::Ctx::scratch` used to hand back a slab keyed by a static
//! NAME: process-global, grown but never shrunk, and deliberately not per
//! stream, because an entry that allocated per fire could not be captured.
//! Two launches inside one slab at the same instant stage over each other,
//! and the fire computes anyway — nothing faults, a logit moves. The compiler
//! cannot know that: no `Operands` method says which entries reach a slab,
//! and a backend-neutral pass that did would be a compiler that knows a
//! backend's allocator. So the shell said it, as [`engine_cuda::EXCLUSIVE`] —
//! eleven op names, read off the four `kernels-cuda` modules that call
//! `Ctx::scratch` — and P6 turned it into a dependency edge.
//!
//! It cost every LINEAR-ATTENTION layer of qwen and kimi. Their two arms —
//! `ssm_gated_delta` beside `ssm_gated_delta_chunked`, over disjoint classes,
//! writing disjoint values — are otherwise a textbook concurrency candidate,
//! and both stage through `attn.ssm_gdn_chunk_qk`.
//!
//! **The key is `(arena, name, stream)` now**, so two arms of a fork group
//! take two slabs and there is nothing to order apart: `EXCLUSIVE` is empty
//! and those layers fork. [`WAS`] below is the list it used to hold, kept so
//! the gate can price the change instead of asserting the absence of one.
//!
//! # What the tests pin
//!
//! The PREDICATE, not the count. How many regions a SKU forks follows from
//! how many layers its model text declares, and those move when somebody
//! edits a family. What must not move without something being wrong is the
//! RELATION between two bakes of the same plan — so every assertion here is a
//! comparison between profiles, and the numbers ride in the message.

use model_compiler::{Budget, DeviceProfile, compile};
use model_dsl::Platform;
use model_ir::{Operands, Trace};

/// **THE ELEVEN NAMES THE SHELL USED TO PUBLISH**, kept as the BEFORE of a
/// measurement rather than as a rule.
///
/// Every one of them is an op whose `kernels-cuda` entry reaches
/// `Ctx::scratch` — `attn/ssm.rs` (the staging planes), `attn/pool.rs` (the
/// boundary rope side channel), `attn/index.rs` (the top-k score plane) and
/// `linear/lora.rs` (the correction's waist). Under the old name-keyed slab
/// each was a device-wide workspace and P6 had to serialize them; under the
/// per-stream slab each is a workspace per stream and none of them is.
///
/// A test that only asserted "the empty list costs nothing" would be true of
/// any empty list. Baking against this one is what says the arms it silenced
/// are the arms that came back.
const WAS: [&str; 11] = [
    "attention.ssm_causal_conv1d",
    "attention.ssm_causal_conv1d_chunked",
    "attention.ssm_gdn_prep",
    "attention.ssm_gated_delta",
    "attention.ssm_gated_delta_chunked",
    "attention.ssm_kda_step",
    "attention.ssm_kda_chunked",
    "attention.index_topk",
    "attention.pool_boundary_decode",
    "attention.pool_boundary_prefill",
    "linear.lora_correct",
];

/// The profile `Shell::load` builds, minus the device probe: the compiler's
/// defaults, this shell's exclusive list, and the L40S SM count the gates run
/// on.
fn profile() -> DeviceProfile {
    DeviceProfile {
        sms: 142,
        exclusive: engine_cuda::EXCLUSIVE
            .iter()
            .map(|op| (*op).to_string())
            .collect(),
        ..DeviceProfile::default()
    }
}

/// The same profile as it was before the slab was keyed per stream.
fn as_it_was() -> DeviceProfile {
    DeviceProfile {
        exclusive: WAS.iter().map(|op| (*op).to_string()).collect(),
        ..profile()
    }
}

fn budgets_for(trace: &Trace) -> Budget {
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

fn claims(list: &[&str], trace: &Trace, region: &model_compiler::Region) -> Vec<&'static str> {
    region
        .nodes
        .clone()
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| node.op.name())
        .filter(|name| list.contains(name))
        .collect()
}

fn forked(compiled: &model_compiler::CompiledModel) -> usize {
    compiled.regions.iter().filter(|r| r.stream != 0).count()
}

/// **THE INVARIANT, WHICHEVER NAMES THE LIST HOLDS.** Two regions that both
/// claim an entry the shell called exclusive are never scheduled beside each
/// other. It is vacuous today because the list is empty, and it is the test
/// that stops being vacuous the moment somebody puts a name back on it —
/// which is exactly when the property matters.
#[test]
fn no_two_regions_that_claim_a_slab_are_ever_scheduled_together() {
    let profile = profile();
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let Ok(compiled) = compile(&trace, &budgets_for(&trace), &profile) else {
            continue;
        };
        for &(a, b) in &compiled.streams.pairs {
            let (x, y) = (
                claims(&engine_cuda::EXCLUSIVE, &trace, &compiled.regions[a as usize]),
                claims(&engine_cuda::EXCLUSIVE, &trace, &compiled.regions[b as usize]),
            );
            if !x.is_empty() && !y.is_empty() {
                wrong.push(format!(
                    "`{sku}`: regions {a} and {b} run beside each other and both \
                     claim a device-wide workspace — {x:?} against {y:?}",
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **WHAT THE PER-STREAM SLAB BOUGHT, PRICED IN THE ONE PLACE IT IS
/// VISIBLE.** Bake every SKU twice — once against the eleven names the shell
/// used to publish, once against the empty list it publishes now — and the
/// difference is the whole of what the old key cost.
///
/// Two claims, and the second is the one that could fail:
///
/// 1. **The new bake is the NEUTRAL bake.** With nothing exclusive, the CUDA
///    profile and the backend-neutral default agree region for region, which
///    is the statement that this shell no longer asks the compiler for a
///    dependency edge it cannot derive.
/// 2. **Every SKU the old list silenced now forks**, and it is exactly the
///    SKUs whose plans name one of the eleven. A SKU that names none of them
///    must be unmoved — if one moved, the two profiles differ for a reason
///    that has nothing to do with slabs and the reading below is wrong.
#[test]
fn the_per_stream_slab_gives_back_the_arms_the_name_key_silenced() {
    let now = profile();
    let was = as_it_was();
    let neutral = DeviceProfile {
        exclusive: Vec::new(),
        ..profile()
    };

    let mut table: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut recovered = 0usize;

    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let before = compile(&trace, &budgets_for(&trace), &was).expect("bakes");
        let after = compile(&trace, &budgets_for(&trace), &now).expect("bakes");
        let free = compile(&trace, &budgets_for(&trace), &neutral).expect("bakes");
        let named = trace
            .nodes
            .iter()
            .any(|node| WAS.contains(&node.op.name()));

        table.push(format!(
            "  {sku}: was {}/{}/{}, is {}/{}/{}  (streams/events/forked){}",
            before.streams.streams,
            before.streams.events,
            forked(&before),
            after.streams.streams,
            after.streams.events,
            forked(&after),
            if named { "  ← names one of the eleven" } else { "" },
        ));

        // 1. The empty list IS the neutral profile, so the two bakes agree.
        if forked(&after) != forked(&free) || after.streams.events != free.streams.events {
            wrong.push(format!(
                "`{sku}`: this shell's profile bakes {} forked regions and {} events \
                 where the backend-neutral one bakes {} and {}, and the shell's \
                 exclusive list is empty",
                forked(&after),
                after.streams.events,
                forked(&free),
                free.streams.events,
            ));
        }

        // 2. A plan that names none of the eleven cannot have moved.
        if !named && forked(&after) != forked(&before) {
            wrong.push(format!(
                "`{sku}` names none of the eleven and still moved, {} forked against \
                 {} — the two profiles differ for a reason that is not the slab",
                forked(&after),
                forked(&before),
            ));
        }
        // The rule only ever took forks away, so the new bake can only add.
        if forked(&after) < forked(&before) {
            wrong.push(format!(
                "`{sku}` LOST forks when the edges were removed: {} against {}",
                forked(&after),
                forked(&before),
            ));
        }
        if forked(&after) > forked(&before) {
            recovered += 1;
        }
    }

    // **AND SOMETHING HAS TO HAVE MOVED.** The whole wave is the claim that
    // the eleven names cost real overlap; a table where nothing changed would
    // mean they cost nothing and this test proves nothing.
    if recovered == 0 {
        wrong.push(
            "no SKU forks more than it did under the eleven names — either the \
             catalog stopped declaring linear attention or the profile is not \
             reaching the compiler"
                .to_string(),
        );
    }

    assert!(
        wrong.is_empty(),
        "\n{}\n\nthe whole table:\n{}\n",
        wrong.join("\n"),
        table.join("\n"),
    );
    eprintln!("{}", table.join("\n"));
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
        let trace = trace(Platform::Cuda);
        let compiled = compile(&trace, &budgets_for(&trace), &off).expect("bakes");
        assert!(
            compiled.regions.iter().all(|r| {
                r.stream == 0 && r.wait.is_empty() && r.open.is_none() && r.close.is_none()
            }),
            "`{sku}` forked with the streams off",
        );
        assert_eq!(compiled.streams.events, 0, "`{sku}` minted an event with the streams off");
        assert_eq!(compiled.streams.streams, 1, "`{sku}` asked for a stream it will not open");
    }
}
