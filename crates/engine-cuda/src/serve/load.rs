#![allow(unsafe_code)]
use poem_compiler::arena::{LiveSlot, Peak};
use poem_compiler::{Budget, Budgets, CompiledModel, DeviceProfile};

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::{
    Exports, Feeds, corrected_classes, decoding_of, group_masked_classes, landing_requests,
    masked_classes, media_classes, one_direction_per_mask, plain_of, read_only_by_decoders,
    regions_lane_shifting, regions_launching_schedules, regions_shifting, schedule_streams,
    wide_classes,
};
use crate::inputs::Inputs;
use crate::program::Plane as ProgramPlane;
use crate::record::Bodies as GraphCache;
use crate::store::kv::{self, Paging};
use crate::store::rs::Buffers;
use crate::store::{KvSize, Pools, StatedKv};
use crate::weights::Weights;

use super::{Boot, Golden, Released, Shell};

/// The least `max_forward_tokens` is served at when the card is short: one
/// prefill chunk a frame, which serves, and under it the halving stops.
const TOKENS_FLOOR: u32 = 1024;

/// Where the kernels' compiled code lands under the cache directory.
#[cfg(not(feature = "rocm"))]
const CODE_DIR: &str = kernels_cuda::disk::CUBINS;
#[cfg(feature = "rocm")]
const CODE_DIR: &str = kernels_cuda::disk::CODE_OBJECTS;

pub(super) fn bake(boot: &mut Boot<'_>) -> Result<Baked> {
    super::diag::publish(&boot.knobs.diagnostics);
    if let Some(guard) = super::diag::on().sync_guard {
        crate::device::guard::arm()?;
        if guard == super::diag::SyncGuard::Report {
            crate::device::guard::speak();
        }
    }
    // SAFETY: the boot's communicator is held by the engine for the life of
    // the shell it loads.
    let mut device = Context::bind(boot.ordinal, unsafe { boot.comm.as_ref() })?;
    // What the card may hold while this load runs, resolved here before the
    // load takes anything: every later reading of its room is under it.
    let ceiling = crate::store::load_ceiling(boot.allowance, true)?;
    let (major, minor) = device.capability();
    let measured = crate::chip::facts(
        u32::try_from(major).unwrap_or(0),
        u32::try_from(minor).unwrap_or(0),
        device.device().num_sm,
    );
    let keys = crate::stored::Keys::of(boot.ordinal, device.name());
    let (resolved, report) = crate::chip::resolve_keyed(
        device.name(),
        &measured,
        boot.knobs.profiles.as_deref(),
        &keys,
    );
    let chip = match boot.knobs.chip.clone() {
        // A pinned profile pins decisions, not the device: the rates it
        // leaves unstated are this device's, as its own profile states them.
        Some(chip) => {
            let mut tuned = resolved.tuned().clone();
            tuned.lay(chip.tuned());
            chip.with_tuned(tuned)
        }
        None => {
            crate::chip::SHELF.tell_untuned(device.name(), &report);
            resolved
        }
    };
    // A device whose profile lacks the roofline's constants measures them
    // here, before ready, and keeps them in the user's profile.
    let room = {
        let (free, total) = crate::store::device_memory()?;
        crate::device::elastic::room_under(ceiling, free, total)
    };
    let chip = crate::rates::ensure(
        chip,
        boot.ordinal,
        device.name(),
        &keys,
        boot.knobs.profiles.as_deref(),
        boot.cache_dir,
        room,
        boot.knobs.measuring,
    )
    .map_err(Fault::Card)?;
    // What the rates step timed (none where every rate was stated).
    let timed_at_open = chip.tuned().timed.clone();
    device.set_tuning(kernels_cuda::tuning::Tuning::of(&chip));
    // The product times the device's profile holds choose between the
    // kernels that can serve a decode product; arming notes the shapes it
    // lacks, measured once it has armed (`Staged::finish`).
    let products = chip.rates().product_ns.clone().unwrap_or_default();
    // How the device's profile says this machine lands weights, under a
    // matching key: a load that finds it measures nothing of it.
    let landing = chip.rates().landing;
    device.set_routes(std::sync::Arc::new(
        kernels_cuda::linear::routes::Routes::of(&products),
    ));
    let engine = crate::chip::Engine::of(&chip);
    // The most rows one prompt lays: its context, where the model keeps a
    // lane's rows (kv or recurrent state); a model that keeps none (a
    // diffusion transformer's plan) bounds a lane's rows by nothing the
    // context sizes, only by the rows the load states. Every rung of the
    // lattice to it is a candidate: the fire top (`engine::load::fire_top`)
    // is chosen among them below, that bound and memory its only bounds. A
    // check's load sized for one kind of its fires lays the rungs to their
    // rows.
    let prompt_rows = if boot.trace.caches.is_empty() && boot.budget.max_tokens > 0 {
        boot.budget.max_tokens
    } else {
        boot.context.max(1)
    };
    let lattice_top = match boot.knobs.every_rung {
        Some(super::EveryRung::To(rows)) => rows,
        _ => prompt_rows,
    };
    if boot.budget.max_tokens == 0 {
        boot.budget.max_tokens = lattice_top;
        // only where it at least halves the fire: a trunk of few value heads seats enough lanes that the
        // wide fire feeds its experts better (Qwen3.6-35B-A3B pf64 lost 4% at 5120 rows of 8192)
        if let Some(fit) = walk_fire_fit(&boot.trace, engine.walk_fire_rows, device.device().num_sm)
            && fit * 2 <= boot.budget.max_tokens
        {
            eprintln!(
                "engine-cuda: fires of {fit} rows where {} were asked: the chunked gated delta \
                 walk's blocks fit one wave (engine.walk_fire_rows)",
                boot.budget.max_tokens
            );
            boot.budget.max_tokens = fit;
        }
        if engine.fire_rows > 0 && engine.fire_rows < boot.budget.max_tokens {
            boot.budget.max_tokens = engine.fire_rows;
        }
    }
    if engine.host_planes {
        let freed = boot.residency.host_declared(&boot.trace)?;
        if freed > 0 {
            eprintln!(
                "engine-cuda: {:.1} GiB of planes the package declares `host` serve from \
                 pinned host memory (engine.host_planes)",
                freed as f64 / f64::from(1u32 << 30)
            );
        }
    }
    // So do the routed banks it streams an expert at a time.
    if !boot.knobs.dense_weights() && !boot.knobs.diagnostics.dense_experts {
        let banks = crate::store::layout::streamed_experts(&boot.trace);
        match host_copy(boot, &banks)? {
            HostCopy::Filed => {
                boot.residency
                    .file_weights(&boot.trace, &banks, true, false)?;
            }
            HostCopy::Mapped => {
                boot.residency
                    .file_weights(&boot.trace, &banks, true, true)?;
            }
            HostCopy::Pinned => {
                boot.residency.host_experts(&boot.trace, &banks)?;
            }
        }
    }
    // Unstated, a seat a lane: every lane a fire batches has a slot to land
    // in, so the lattice's widest decode is the fire limit's, not a fixed
    // count's.
    boot.slots = poem_compiler::state_slots(boot.slots, &boot.trace, boot.budget.max_lanes.max(1));

    kernels_cuda::disk::install(boot.cache_dir);

    let stated_buckets = std::mem::take(&mut boot.budget.buckets);
    // Unstated, a load's fires are rungs of the lattice its rates were timed
    // at: asked other tokens (the walk fit, `engine.fire_rows`, a stated
    // `max_forward_tokens`), its top is the rung at or below them.
    if stated_buckets.is_empty() {
        boot.budget.max_tokens =
            crate::api::on_lattice(&engine, lattice_top, boot.budget.max_tokens);
    }
    // A fire carries a row a lane at least: lanes past its rows are fired as
    // waves of as many as it has rows for.
    boot.budget.max_lanes = boot.budget.max_lanes.min(boot.budget.max_tokens);
    boot.budget.buckets = crate::api::lattice(
        stated_buckets.clone(),
        boot.budget.max_tokens,
        engine.lattice_steps,
        engine.prefill_lattice_from,
        engine.prefill_lattice_steps,
    );

    let grouped: &[&str] = if engine.grouped { &crate::GROUPED } else { &[] };
    let mut profile = DeviceProfile::of(&chip, device.device().num_sm, &crate::EXCLUSIVE, grouped);
    let mut kernels = poem_compiler::fuse::flags::on(&chip, &crate::FUSIONS);
    if !boot.knobs.diagnostics.fuse_chains {
        kernels.clear();
    }
    #[cfg(feature = "rocm")]
    {
        boot.trace = crate::split_arm::split_into_arms(boot.trace.clone());
    }
    boot.trace = poem_compiler::fuse::fuse(boot.trace.clone(), kernels);
    #[cfg(not(feature = "rocm"))]
    {
        let tuning = kernels_cuda::tuning::Tuning::of(&chip);
        boot.trace = crate::grouped_decode::rewrite_with(
            boot.trace.clone(),
            tuning.decode_grouped_prefill_from(),
            tuning.decode_grouped_prefill_lse(),
            tuning.decode_grouped_prefill_window(),
            kernels_cuda::attn::decode_gqa::Gate::of(tuning),
        );
        boot.trace = crate::narrow::gdn_out_bf16(boot.trace.clone(), || tuning.ssm_gdn_out_bf16());
    }
    // A join's independent branches side by side, where the profile forks
    // them onto side streams (`compiler.branches`).
    if profile.branches {
        boot.trace = poem_compiler::branch::split(boot.trace.clone());
    }
    // `compiler.synth_anchors`: each matmul and the pointwise ops over its
    // product folded into one `fused.anchored`, before the chains fold, so
    // a product's epilogue is its contraction skeleton's.
    #[cfg(not(feature = "rocm"))]
    let anchored = if profile.synth_anchors && boot.world.size == 1 {
        let skinny = !profile.synth_regions
            || kernels_cuda::tuning::Tuning::of(&chip).gemm_skinny_rows() > 0;
        let (folded, anchored) = crate::synth::fold_anchors_with(boot.trace.clone(), skinny);
        boot.trace = folded;
        anchored
    } else {
        std::collections::BTreeMap::new()
    };
    // ROCm holds no contraction skeletons: a profile that asks for anchored
    // regions there is refused, not served without them.
    #[cfg(feature = "rocm")]
    let anchored = if profile.synth_anchors {
        return Err(Fault::Unsupported(
            "`compiler.synth_anchors` folds a product's epilogue into its contraction \
             skeleton, which ROCm does not hold; serve it with the knob off"
                .to_owned(),
        ));
    } else {
        std::collections::BTreeMap::new()
    };
    profile.synth_anchors = false;
    // `compiler.synth_regions`: each chain of elementwise and row ops the row
    // skeleton serves folded into one `fused.region`, its region kernels-ir's.
    // Folded once, here, the trace is the one the load runs and every plan
    // after is of it as it stands.
    let regions = if profile.synth_regions {
        let (folded, regions) =
            poem_compiler::fuse::chain::fold(boot.trace.clone(), &crate::synth::serves_region);
        boot.trace = folded;
        profile.synth_regions = false;
        regions
    } else {
        std::collections::BTreeMap::new()
    };
    // The tables the weight cache streams by their rows keep their copy on
    // the pinned host tier: a row not cached is read from it in place. Told
    // of the trace as the load runs it: a table whose gather a rewrite
    // above folded into another op is read whole, and stays on the device.
    if !boot.knobs.dense_weights() {
        let streamed = crate::store::layout::streamed_rows(&boot.trace);
        match host_copy(boot, &streamed)? {
            HostCopy::Filed => {
                boot.residency
                    .file_weights(&boot.trace, &streamed, false, false)?;
            }
            HostCopy::Mapped => {
                boot.residency
                    .file_weights(&boot.trace, &streamed, false, true)?;
            }
            HostCopy::Pinned => {
                boot.residency.host_rows(&boot.trace, &streamed)?;
            }
        }
    }
    if boot.knobs.diagnostics.trace_census {
        let mut census: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for node in &boot.trace.nodes {
            *census.entry(poem_ir::Operands::name(&node.op)).or_insert(0) += 1;
        }
        eprintln!(
            "[trace-census] {} nodes: {census:?}",
            boot.trace.nodes.len()
        );
    }
    // The activation arena and the attention workspaces are sized from
    // `max_forward_tokens`, and on a wide model at a long envelope they are
    // gigabytes before a sequence seats. The card's share after the weight
    // tier is what they must fit in, half of it, so the cache rows, the
    // bodies and the guests' programs have the other half; the token budget
    // halves toward its floor until they do, and the lattice follows.
    let (free, total) = crate::store::device_memory()?;
    let weight_tier = crate::store::weight_tier_bytes(
        &boot.trace,
        boot.residency.resident_bytes(&boot.trace)?,
        0,
    )?;
    let after_weights =
        crate::device::elastic::room_under(ceiling, free, total).saturating_sub(weight_tier);
    // Each byte charged once: what the card holds now (this process's
    // context and kernel modules among it; no weight has landed yet), the
    // weight tier to come, and the driver's floor.
    let room_line = room_line(boot.allowance, ceiling, total - free, weight_tier, after_weights);
    eprintln!("engine-cuda: {room_line}");
    let after_weights = held_beside(boot, after_weights)?;
    // A declared context is the model's, not the operator's: it is served at
    // the longest one sequence of which takes half the room this line leaves
    // (this process's context and modules already out of it), and the fit
    // shortens it further where the load does not seat it
    // (`Pools::declare_context`); a card that seats none is refused by name.
    if boot.context_declared {
        let fits = crate::store::declared_context_within(
            &boot.trace,
            boot.page_size,
            boot.context,
            after_weights,
        )?;
        if fits < boot.context {
            eprintln!(
                "engine-cuda: the model declares a {}-token context and one sequence at it \
                 takes more than half of the room the load has past the weights; serving \
                 {fits} tokens. State `[engine] max_context` to choose.",
                boot.context
            );
            boot.context = fits;
        }
    }
    let facts = kv::probe(&boot.trace)?;
    let derive_voxels = boot
        .voxels
        .as_ref()
        .is_some_and(|ladder| ladder.max_voxels == 0);
    let least_voxels = |ladder: &poem_compiler::VoxelLadder| {
        let clips = ladder.max_clips;
        ladder.clone().at(clips)
    };
    let voxels = boot.voxels.as_ref().map(|ladder| {
        if derive_voxels {
            least_voxels(ladder)
        } else {
            ladder.clone()
        }
    });
    let budgets_at = |budget: &Budget, voxels: &Option<poem_compiler::VoxelLadder>| Budgets {
        tokens: budget.clone(),
        patches: boot.patches.clone(),
        voxels: voxels.clone(),
    };
    // A rung's fire carries a row a lane at least: its lanes are the asked
    // lanes it has rows for, as the refit's rungs are.
    let budget_at = |tokens: u32| {
        let mut budget = boot.budget.clone();
        budget.max_lanes = budget.max_lanes.min(tokens);
        budget.max_tokens = tokens;
        let mut buckets: Vec<u32> = stated_buckets
            .iter()
            .copied()
            .filter(|bucket| *bucket < tokens)
            .collect();
        if !stated_buckets.is_empty() {
            buckets.push(tokens);
        }
        budget.buckets = crate::api::lattice(
            buckets,
            tokens,
            engine.lattice_steps,
            engine.prefill_lattice_from,
            engine.prefill_lattice_steps,
        );
        budget
    };
    let asked_tokens = boot.budget.max_tokens;
    let asked = boot.budget.clone();
    let window = crate::store::window_of(&boot.trace)?;
    // A fire of `tokens`' working set (the activation arena, the attention
    // workspaces and one fire's windowed kv rows) and the launches its
    // compiled plan issues; the compile's error where it does not compile.
    let laid = |tokens: u32,
                voxels: &Option<poem_compiler::VoxelLadder>|
     -> std::result::Result<(u64, u32), String> {
        let windowed = Paging::of(boot.page_size, boot.context, boot.slots, 1)
            .ok()
            .and_then(|paging| {
                crate::store::window_fire_bytes(&boot.trace, paging.windowed(window, tokens)).ok()
            })
            .unwrap_or(u64::MAX);
        let budget = budget_at(tokens);
        let compiled =
            poem_compiler::compile_axes(&boot.trace, &budgets_at(&budget, voxels), &profile)
                .map_err(|e| e.to_string())?;
        let bytes = compiled
            .arena
            .bytes
            .saturating_add(
                crate::inputs::attention_workspace_bytes(
                    &budget,
                    engine.fire_lanes_of(budget.max_lanes),
                    boot.page_size,
                    &facts,
                    &device.device(),
                    device.toggles(),
                    &crate::exports::schedule_streams(&boot.trace, &compiled),
                )
                .map_err(|e| e.to_string())?,
            )
            .saturating_add(windowed);
        let launches = u32::try_from(compiled.template().len()).unwrap_or(u32::MAX);
        Ok((bytes, launches))
    };
    let working = |tokens: u32, voxels: &Option<poem_compiler::VoxelLadder>| -> u64 {
        laid(tokens, voxels).map_or(u64::MAX, |(bytes, _)| bytes)
    };
    // The fire top, the ruling's (`crate::top`, `engine::load::fire_top`):
    // of every bucket of the asked lattice (to the context), each spanning
    // its fire's working set and needing one sequence beside it, over the
    // room the card has after the weight tier. The bodies are priced when
    // they arm, ahead of the cache rows' fill (`store::holds`), and a card
    // that holds none of them past the rung is refused there by name.
    let weighed;
    let tokens = {
        let paging = Paging::of(boot.page_size, boot.context, boot.slots, 1)?;
        let mut candidates = Vec::new();
        for rung in budget_at(asked_tokens).buckets {
            let plan = laid(rung, &voxels).map_err(|why| crate::top::no_plan(&why));
            let sequence =
                crate::store::least_sequence_bytes(&boot.trace, paging.windowed(window, rung))?;
            candidates.push(crate::top::Candidate {
                tokens: rung,
                plan,
                beside: sequence,
            });
        }
        // The lanes one fire carries (`Shell::fire_lanes`): the runtime
        // fires a wider wave as fires of this many, and a rung's check fire
        // (`RungFacts::check_fire`) lays its rows over them.
        let fire_lanes = engine.fire_lanes_of(boot.budget.max_lanes);
        let rungs = crate::top::rung_facts(
            &candidates,
            boot.budget.max_lanes,
            fire_lanes,
            after_weights,
            |rows| rows,
        );
        let counts = crate::top::counts(&boot.trace)?;
        // A card on which no rung fits beside one sequence is refused by
        // name, with the room it had: too small for this load as stated.
        let (top, estimate) = crate::top::choose(
            &rungs,
            after_weights,
            prompt_rows,
            &chip.rates(),
            crate::top::shape(&chip)?,
            device.name(),
            &counts,
        )
        .map_err(|fault| match fault {
            Fault::Card(why) => Fault::Card(format!(
                "{room_line}; one sequence at {} tokens takes {} MiB beside a fire: {why}",
                boot.context,
                candidates.first().map_or(0, |c| c.beside) >> 20,
            )),
            fault => fault,
        })?;
        // Which pricing the estimate gave the dense matmuls, said once: the
        // waves of the kernels hipBLASLt states, or the measured curve as
        // timed and why its solutions went unread.
        #[cfg(feature = "rocm")]
        eprintln!(
            "engine-cuda: the matmuls weighed by {}",
            kernels_cuda::hip::lt_kernel::pricing()
        );
        eprintln!(
            "engine-cuda: {}",
            engine::load::line_of(
                &top,
                top.tokens,
                top.span_bytes,
                after_weights,
                chip.tuned()
            )
        );
        weighed = Some(crate::top::Weighed {
            rungs: rungs.clone(),
            estimate,
            live: after_weights,
            fire_top: top.tokens,
        });
        let widest = crate::top::widest(&rungs).unwrap_or(top.tokens);
        match boot.knobs.every_rung {
            None => top.tokens,
            // Past the top (`engine::load::check::laid_top`), so a top set
            // too low shows in the check.
            Some(super::EveryRung::PastTop) => {
                let lattice: Vec<u32> = rungs.iter().map(|r| r.tokens).collect();
                engine::load::check::laid_top(&lattice, top.tokens).min(widest)
            }
            Some(super::EveryRung::To(_)) => widest,
        }
    };
    // The derived voxel rung's room is still the configured share of the
    // card past the weights (`engine.working_set_percent`).
    let share = after_weights / 100 * u64::from(engine.working_set_percent);
    let voxels = match voxels.clone() {
        Some(ladder) if derive_voxels => {
            let floor = ladder.max_clips;
            let paging = Paging::of(boot.page_size, boot.context, boot.slots, 1)?;
            let cache = crate::store::least_sequence_bytes(&boot.trace, paging)?;
            let room = share.min(after_weights.saturating_sub(cache));
            let fitted = poem_compiler::voxels_within(room, floor, |rows| {
                let at = ladder.clone().at(rows);
                let seat = crate::voxels::Seat::of(&boot.trace, &at);
                working(tokens, &Some(at)).saturating_add(seat.payload_bytes())
            });
            eprintln!(
                "engine-cuda: the voxel rung derived: {fitted} rows of the widest voxel \
                 value fit the working set beside {tokens} tokens"
            );
            Some(ladder.at(fitted))
        }
        voxels => voxels,
    };
    let cache_room = (boot.slots == 0).then(|| {
        let seat = voxels.as_ref().map_or(0, |ladder| {
            crate::voxels::Seat::of(&boot.trace, ladder).payload_bytes()
        });
        after_weights
            .saturating_sub(working(tokens, &voxels))
            .saturating_sub(seat)
    });
    if tokens != asked_tokens {
        boot.budget = budget_at(tokens);
    }
    boot.voxels = voxels;
    if let Some(room) = cache_room {
        boot.slots = poem_compiler::fit_frame_slots(
            &boot.trace,
            boot.voxels.as_mut(),
            room,
            engine::fit::least_state_slots(1),
            boot.runahead.frames_in_flight,
        );
    }
    let budgets = Budgets {
        tokens: boot.budget.clone(),
        patches: boot.patches.clone(),
        voxels: boot.voxels.clone(),
    };
    let compiled = poem_compiler::compile_axes(&boot.trace, &budgets, &profile)?;
    if boot.knobs.diagnostics.arm_trace {
        let mut streams: std::collections::BTreeMap<u32, Vec<String>> =
            std::collections::BTreeMap::new();
        for (at, region) in compiled.template().iter().enumerate() {
            let op = boot
                .trace
                .nodes
                .get(region.nodes.start as usize)
                .map_or("?", |node| poem_ir::Operands::name(&node.op));
            if op.starts_with("attn.") || op.starts_with("ssm.") {
                streams.entry(region.stream).or_default().push(format!(
                    "r{at}:{}:{:?}",
                    op.trim_start_matches("attn."),
                    region.mask.iter().collect::<Vec<_>>()
                ));
            }
        }
        for (stream, regions) in &streams {
            eprintln!(
                "[arm-trace] attention regions on stream {stream}: {}",
                regions.join(" ")
            );
        }
        let Peak { at, slots } = compiled.arena.peak();
        eprintln!(
            "[arm-trace] budget: max_tokens {} max_lanes {} buckets {:?}",
            boot.budget.max_tokens, boot.budget.max_lanes, boot.budget.buckets
        );
        eprintln!(
            "[arm-trace] arena prefix for {} readouts: {} MiB",
            boot.budget.max_lanes,
            compiled.arena.prefix_for(u64::from(boot.budget.max_lanes)) >> 20
        );
        eprintln!(
            "[arm-trace] arena reserved {} MiB (live bound {} MiB), peak at node {at}: {} slots, {} MiB",
            compiled.arena.bytes >> 20,
            compiled.arena.live_bound() >> 20,
            slots.len(),
            slots.iter().map(|slot| slot.bytes).sum::<u64>() >> 20
        );
        for LiveSlot {
            value,
            span,
            offset,
            bytes,
        } in slots.iter().take(16)
        {
            let what = match &boot.trace.values[value.0 as usize].def {
                poem_ir::Def::Op(node) => {
                    poem_ir::Operands::name(&boot.trace.nodes[*node as usize].op).to_string()
                }
                other => format!("{other:?}").chars().take(20).collect(),
            };
            let rows = match &compiled.arena.placements[value.0 as usize] {
                poem_compiler::arena::Placement::Arena {
                    rows, width, dtype, ..
                } => format!("{rows:?} x {width} {dtype:?}"),
                _ => String::new(),
            };
            eprintln!(
                "[arm-trace]   value {} {what}: {} MiB at {} MiB, live nodes {}..{} [{rows}]",
                value.0,
                bytes >> 20,
                offset >> 20,
                span.first,
                span.last
            );
        }
    }
    Ok(Baked {
        device,
        engine,
        compiled,
        budgets,
        profile,
        regions,
        anchored,
        keys,
        products,
        landing,
        asked,
        ceiling,
        weighed,
        timed_at_open,
    })
}

/// Where the weight cache's copy of a streamable weight lies off the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostCopy {
    /// In the serving file's mapping, copied into pool slots at the first
    /// weight residency and never read where it lies.
    Filed,
    /// In the serving file's mapping, a miss read where it lies.
    Mapped,
    /// On the host tier: pinned, a miss read where it lies; or, for a load
    /// told its weights are held, pageable, seated in the pool before any
    /// fire and read by none (`Tier::open`).
    Pinned,
}

/// Where the weights `params` the weight cache takes are kept off the
/// device.
///
/// A load whose weights the runtime holds whole in the pool for its life
/// (`Knobs::weights_held`) leaves them in the model's serving file, where
/// the file holds each as the trace reads it; no fire reads a host copy.
/// A load that streams reads its misses from a copy: the file's mapping
/// where the device faults mapped pages in, else pinned host memory, which
/// is refused where the machine does not have it rather than left to the
/// kernel to kill.
fn host_copy(boot: &Boot<'_>, params: &[usize]) -> Result<HostCopy> {
    if params.is_empty() {
        return Ok(HostCopy::Pinned);
    }
    // A plane the load lays in another form (its codes requantized, its
    // scales laid in atoms) is not in the file as a slot holds it, whatever
    // its size: its copy is laid in place at load.
    let bytes = crate::weights::laid_bytes(&boot.trace)?;
    let laid = crate::weights::laid_planes(&boot.trace);
    // So is a bank ROCm lays in the matrix core's lane order, codes and
    // exponents (`store::layout::lane_ordered`), where a copy is read: a
    // load told its weights are held reads the file as it lies, and its
    // slots are laid in lane order on the device once filled.
    let lanes: Vec<usize> = params
        .iter()
        .filter(|_| !boot.knobs.weights_held)
        .filter(|&&codes| crate::store::layout::lane_ordered(&boot.trace, codes))
        .flat_map(|&codes| crate::store::layout::packed_planes(&boot.trace, codes))
        .flatten()
        .map(|plane| plane as usize)
        // And an fp8 bank's codes ROCm puts in the FP8 matrix core's order
        // (`store::layout::fp8_ordered`), laid alike: in a copy where one is
        // read, in its slots once filled under a load told its weights are
        // held. Its factors lie as the file holds them.
        .chain(
            params
                .iter()
                .filter(|_| !boot.knobs.weights_held)
                .filter(|&&codes| crate::store::layout::fp8_ordered(&boot.trace, codes))
                .copied(),
        )
        .collect();
    let relaid = |param: usize| {
        lanes.contains(&param)
            || laid
                .iter()
                .any(|laid| laid.codes == param || laid.scales == param)
    };
    let in_file = !params.iter().any(|&param| relaid(param))
        && crate::checkpoint_serving::Serving::open(boot.checkpoint, &boot.trace)
            .is_some_and(|serving| serving.holds(params, &bytes));
    let pinned: u64 = params.iter().map(|&param| bytes[param]).sum();
    if boot.knobs.weights_held {
        if in_file {
            return Ok(HostCopy::Filed);
        }
        eprintln!(
            "engine-cuda: {} MiB of weights the pool will hold are copied to the host at \
             load (pageable, freed once seated): {} does not hold their planes as this plan \
             reads them (a plane decoded at load), so their slots cannot be filled from the \
             file",
            pinned >> 20,
            boot.checkpoint.display()
        );
    } else if in_file && crate::experts::pageable_access() {
        return Ok(HostCopy::Mapped);
    }
    let have = crate::weights::available_memory();
    if pinned > have {
        return Err(Fault::Residency(format!(
            "this load keeps a host copy of the weights its cache takes, {:.1} GiB, \
             and this machine has {:.1} GiB of memory left for it. {}",
            pinned as f64 / f64::from(1u32 << 30),
            have as f64 / f64::from(1u32 << 30),
            if boot.knobs.weights_held {
                "Its checkpoint's planes need decoding at load, so they cannot be read \
                 from the file; give the host more memory."
            } else {
                "Give the device room to hold them all (they are then held on the device \
                 and read from the file), or use a device with pageable memory access \
                 (HMM)."
            }
        )));
    }
    Ok(HostCopy::Pinned)
}

/// The load's room as the fit reads it, each byte once: the ceiling the
/// allowance sets, what the card holds now (this process's context and
/// kernel modules, and any other process's, before a weight lands), the
/// weight tier to come and the driver's floor, and what is left live.
fn room_line(
    allowance: engine::Allowance,
    ceiling: u64,
    in_use: u64,
    weights: u64,
    live: u64,
) -> String {
    let mib = |bytes: u64| bytes >> 20;
    let held = match allowance.budget {
        // The ceiling is what other holders held at the load's start plus
        // the budget: past that, the card's bytes in use are this load's.
        Some(budget) => format!(
            "{} MiB of it this process's context and modules, charged to the {} MiB budget",
            mib(in_use.saturating_sub(ceiling.saturating_sub(budget))),
            mib(budget)
        ),
        None => "no budget stated".to_owned(),
    };
    format!(
        "the load's room: ceiling {} MiB; the card holds {} MiB before the weights land ({held}); \
         weight tier {} MiB; driver floor {} MiB; {} MiB live past the weights",
        mib(ceiling),
        mib(in_use),
        mib(weights),
        mib(crate::device::elastic::safety_floor_bytes()),
        mib(live),
    )
}

/// `room` less the banks the weight cache takes, where the card holds them
/// beside it: they are not on the device yet, and once seated they are the
/// pool's for the load's life, as a resident bank's bytes are, so no fire's
/// working set is sized into them. A deployment whose banks exceed the room
/// streams them, and its fires are sized in the room as it is.
fn held_beside(boot: &Boot<'_>, room: u64) -> Result<u64> {
    let bytes = crate::weights::laid_bytes(&boot.trace)?;
    let cached: u64 = boot
        .residency
        .cached_experts()
        .iter()
        .map(|&param| bytes[param].next_multiple_of(crate::weights::ALIGN))
        .sum();
    Ok(if cached < room { room - cached } else { room })
}

/// The token and lane budgets fitted to what the card has with the weights
/// resident. What a fire's working set takes at those budgets (the
/// activation arena, the attention workspaces, the decoded-weight tiles, the
/// guests' programs and the readout indices) is held to half of the card
/// past the weights, the half `bake` sizes the token budget against before
/// the weights are down, and to what leaves one sequence beside it. The
/// tile yields first, held to the widest plane fired at token rows (a plane
/// past it is served by the fused-dequant arm), then the token budget
/// halves toward its floor, then the lanes halve toward one, and what their
/// cut frees grows the token budget back toward `asked`'s, the budget
/// before `bake` cut it at the asked lanes; `Pools::fit_the_card` still
/// refuses past that.
#[allow(clippy::too_many_arguments)]
fn refit(
    boot: &mut Boot<'_>,
    device: &Context,
    compiled: CompiledModel,
    budgets: Budgets,
    profile: &DeviceProfile,
    paging: Paging,
    table: &crate::run::WeightTable,
    engine: crate::chip::Engine,
    asked: &Budget,
    ceiling: u64,
) -> Result<(CompiledModel, Budgets, Vec<u64>)> {
    let (free, total) = crate::store::device_memory()?;
    let live = crate::device::elastic::room_under(ceiling, free, total);
    let live = held_beside(boot, live)?;
    let facts = kv::probe(&boot.trace)?;
    let window = crate::store::window_of(&boot.trace)?;
    let (asked_lanes, asked_tokens) = (boot.budget.max_lanes, boot.budget.max_tokens);
    let budget_at = |lanes: u32, tokens: u32| {
        let mut budget = boot.budget.clone();
        budget.max_lanes = lanes.min(tokens);
        budget.max_tokens = tokens;
        let mut buckets: Vec<u32> = asked
            .buckets
            .iter()
            .copied()
            .filter(|bucket| *bucket < tokens)
            .collect();
        buckets.push(tokens);
        budget.buckets = crate::api::lattice(
            buckets,
            tokens,
            engine.lattice_steps,
            engine.prefill_lattice_from,
            engine.prefill_lattice_steps,
        );
        budget
    };
    let budgets_at = |budget: &Budget| Budgets {
        tokens: budget.clone(),
        patches: boot.patches.clone(),
        voxels: boot.voxels.clone(),
    };
    let tiles_of = |planes: &[DecodedPlanes], scoped: bool| -> Vec<u64> {
        planes
            .iter()
            .map(|plane| {
                if scoped {
                    plane.at_tokens
                } else {
                    plane.widest
                }
            })
            .collect()
    };
    // A workspace the plan refuses fails the refit by name; a budget that
    // does not compile is no fit.
    let working_at = |lanes: u32, tokens: u32, scoped: bool| -> Result<u64> {
        let budget = budget_at(lanes, tokens);
        let lanes = budget.max_lanes;
        let Ok(compiled) = poem_compiler::compile_axes(&boot.trace, &budgets_at(&budget), profile)
        else {
            return Ok(u64::MAX);
        };
        let tiles: u64 = tiles_of(
            &decoded_weight_planes(&boot.trace, &compiled, table),
            scoped,
        )
        .iter()
        .map(|plane| crate::store::decoded_weight_reserve(*plane, 1))
        .fold(0u64, u64::saturating_add);
        Ok(compiled
            .arena
            .bytes
            .saturating_add(crate::inputs::attention_workspace_bytes(
                &budget,
                engine.fire_lanes_of(budget.max_lanes),
                boot.page_size,
                &facts,
                &device.device(),
                device.toggles(),
                &schedule_streams(&boot.trace, &compiled),
            )?)
            .saturating_add(tiles)
            .saturating_add(
                u64::from(lanes)
                    .saturating_mul(u64::from(tokens))
                    .saturating_mul(8),
            ))
    };
    // What one sequence leaves a fire, and within it the
    // half the cache rows are owed when the floors can still reach it.
    let room_at = |tokens: u32, owed: bool| -> Result<u64> {
        let sequence =
            crate::store::least_sequence_bytes(&boot.trace, paging.windowed(window, tokens))?;
        let room = live.saturating_sub(sequence);
        Ok(if owed {
            room.min(live / 100 * u64::from(engine.working_set_percent))
        } else {
            room
        })
    };
    let floor = TOKENS_FLOOR.min(asked_tokens);
    let fit = |owed: bool, scoped: bool| -> Result<Option<(u32, u32)>> {
        let mut tokens = asked_tokens;
        while tokens > floor && working_at(asked_lanes, tokens, scoped)? > room_at(tokens, owed)? {
            tokens = crate::api::on_lattice(&engine, asked.max_tokens, tokens / 2).max(floor);
        }
        let room = room_at(tokens, owed)?;
        // At these rows the workspace planned at every lane count already
        // (the loop above), so a refusal here has surfaced.
        let lanes = crate::store::tokens_within(asked_lanes.min(tokens), 1, room, |lanes| {
            working_at(lanes, tokens, scoped).unwrap_or(u64::MAX)
        });
        // A 1024-lane boot of MiniMax-M3.1-Flash mini60 on an H100 cut its fires to
        // 1024 rows at 1024 lanes, then the lanes to 512: one 717-token prompt a fire.
        // Back at 8192 rows: pf64-512 7.05k -> 10.35k tok/s (vLLM 9.03k at 1024
        // sequences), pf1024-512 7.92k -> 10.37k (9.00k), c1024 2.69k -> 2.71k (2.72k).
        // The rows grow back toward the top rung the load chose, never past it:
        // the lattice's top above it is a rung the choice already weighed and
        // passed over, and what the load reserved for (the activation span,
        // the guests' carve) is the chosen top's.
        let tokens = if lanes < asked_lanes {
            widened(
                tokens,
                asked_tokens,
                |rows| crate::api::on_lattice(&engine, asked.max_tokens, rows),
                |wider| Ok(working_at(lanes, wider, scoped)? <= room_at(wider, owed)?),
            )?
        } else {
            tokens
        };
        let room = room_at(tokens, owed)?;
        Ok((working_at(lanes, tokens, scoped)? <= room).then_some((tokens, lanes)))
    };
    let planes = decoded_weight_planes(&boot.trace, &compiled, table);
    if working_at(asked_lanes, asked_tokens, false)? <= room_at(asked_tokens, true)? {
        return Ok((compiled, budgets, tiles_of(&planes, false)));
    }
    // The tile stays whole while the half can be met with it; only one
    // sequence is worth the fused-dequant arm.
    let (scoped, fitted) = match fit(true, false)? {
        Some(fitted) => (false, Some(fitted)),
        None => (true, fit(false, true)?),
    };
    if scoped {
        for (stream, plane) in planes.iter().enumerate() {
            if plane.widest > plane.at_tokens {
                eprintln!(
                    "engine-cuda: the decoded-weight tile on stream {stream} is held to {} MiB, \
                     the widest plane fired at token rows, not the {} MiB plane fired at \
                     readout rows: at {asked_lanes} lanes it leaves no room for one sequence on \
                     this card. A plane past the tile is served by the fused-dequant arm.",
                    crate::store::decoded_weight_reserve(plane.at_tokens, 1) >> 20,
                    crate::store::decoded_weight_reserve(plane.widest, 1) >> 20,
                );
            }
        }
    }
    let Some((tokens, lanes)) = fitted else {
        return Ok((compiled, budgets, tiles_of(&planes, true)));
    };
    if (lanes, tokens) == (asked_lanes, asked_tokens) {
        return Ok((compiled, budgets, tiles_of(&planes, scoped)));
    }
    eprintln!(
        "engine-cuda: a fire of {} tokens and {asked_lanes} lanes is served at {tokens} \
         and {lanes}: at the asked budgets a fire's working set (activation arena, attention \
         workspaces, decoded-weight tiles) takes {} MiB of the {} MiB this \
         card has after the weight tier under the load's allowance, past the half the cache \
         rows are owed and what one sequence at the declared context needs.",
        asked_tokens,
        working_at(asked_lanes, asked_tokens, scoped)? >> 20,
        live >> 20,
    );
    boot.budget = budget_at(lanes, tokens);
    let budgets = budgets_at(&boot.budget);
    let compiled = poem_compiler::compile_axes(&boot.trace, &budgets, profile)?;
    let planes = decoded_weight_planes(&boot.trace, &compiled, table);
    Ok((compiled, budgets, tiles_of(&planes, scoped)))
}

/// The rows a fire whose lanes were cut grows back to: doubling on the
/// lattice (`on`, the rung at or below) while the wider fire `fits`, up to
/// `top`, the rung the load chose and reserved for, and never past it.
fn widened(
    tokens: u32,
    top: u32,
    on: impl Fn(u32) -> u32,
    mut fits: impl FnMut(u32) -> Result<bool>,
) -> Result<u32> {
    let mut tokens = tokens;
    while tokens < top {
        let wider = on(tokens.saturating_mul(2).min(top));
        if wider <= tokens || !fits(wider)? {
            break;
        }
        tokens = wider;
    }
    Ok(tokens)
}

impl Shell {
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        Self::stage(boot)?.finish()
    }

    /// A load up to its bodies: weights, arena, inputs and cache rows down,
    /// nothing fired. [`Staged::finish`] arms the bodies; an expert-parallel
    /// group wires its ranks between the two, since arming fires their
    /// collectives.
    pub fn stage(boot: Boot<'_>) -> Result<Staged> {
        let mut boot = boot;
        crate::program::launch::state_queue_masks(&boot.knobs.queue_masks);
        let Baked {
            mut device,
            engine,
            compiled,
            budgets,
            profile,
            regions: folded,
            anchored,
            keys,
            products,
            landing,
            asked,
            ceiling,
            weighed,
            timed_at_open,
        } = bake(&mut boot)?;
        // What shapes this deployment's bodies, beside the engine build
        // the record's key carries ([`super::measured`]).
        let bodies_shape = format!(
            "{}|{}|{:?}|{}|{}|{}|{:?}",
            super::measured::checkpoint_identity(boot.checkpoint),
            boot.trace.name,
            profile,
            boot.page_size,
            boot.context,
            boot.slots,
            boot.kv_budget,
        );
        // Every generated kernel compiled and loaded now, before a fire can
        // ask for one.
        let regions =
            crate::synth::Regions::load(&boot.trace, &folded, &anchored, &device.ctx().tuning())?;
        let (free_before, device_total) = crate::store::device_memory()?;
        crate::device::held::heap_mark();
        // The pool's address space is reserved for every page its bytes
        // could hold, the stated budget or the card's room before the
        // weights land; `Pools::fit_the_card` sizes it within that.
        // Stated tokens are the pages they fill; else every page the stated
        // bytes, or the card's room before the weights land, could hold.
        let pages = match boot.kv_budget {
            Some(KvSize::Tokens(tokens)) => {
                let pages = StatedKv::pages_for(tokens, boot.page_size);
                let sequence = u64::from(boot.context.div_ceil(boot.page_size.max(1)));
                if pages < sequence {
                    return Err(Fault::Residency(format!(
                        "`[engine.memory] kv_device_tokens = {tokens}` fills {pages} kv pages \
                         of {} tokens, and one sequence at the {}-token context takes \
                         {sequence}; state at least {} tokens or a shorter `[engine] \
                         max_context`",
                        boot.page_size,
                        boot.context,
                        sequence * u64::from(boot.page_size),
                    )));
                }
                pages
            }
            stated => crate::store::page_ceiling(
                &boot.trace,
                boot.page_size,
                boot.context,
                match stated {
                    Some(KvSize::Bytes(bytes)) => bytes,
                    _ => crate::device::elastic::room_under(ceiling, free_before, device_total),
                },
            )?,
        };
        let paging = Paging::of(boot.page_size, boot.context, boot.slots, pages)?
            .windowed(
                crate::store::window_of(&boot.trace)?,
                boot.budget.max_tokens,
            )
            .seated(boot.budget.max_lanes)
            .spare(engine.window_spare);
        // A class attending both ways decodes its planes over all its rows once, at load.
        let decoding = poem_exec::feeds::classes_reading(&boot.trace, &compiled, |def| {
            matches!(def, poem_ir::Def::Weight(w)
                if crate::weights::decodes_at_load(&boot.trace.params[*w as usize]))
        });
        let decode_dense = crate::exports::wide_classes(&boot.trace, &compiled)
            .iter()
            .any(|class| decoding.contains(class));
        let decoded_dense = if decode_dense {
            crate::weights::decoded_dense_bytes(&boot.trace)
        } else {
            0
        };
        let accounting = crate::store::admit_the_card(
            boot.allowance,
            boot.residency.resident_bytes(&boot.trace)?,
            decoded_dense,
            &boot.trace,
            paging,
            match boot.kv_budget {
                Some(KvSize::Bytes(bytes)) => Some(bytes),
                _ => None,
            },
        )?;

        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB before weights",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        // Rewritten in place by the CUDA build's load-time copies.
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.residency.clone(),
            device.stream(),
            checkpoint::plan::StorageTarget {
                ep_rank: boot.expert.rank,
                ..checkpoint::plan::StorageTarget::for_backend(
                    checkpoint::types::BackendKind::Cuda,
                    boot.world.rank,
                    boot.world.size,
                )
            },
            decode_dense,
            // A load told its weights are held seats every bank before a
            // fire: its banks' host copy is pageable (`Tier::open`).
            boot.knobs.weights_held,
            landing,
        )?;
        if let Some(measured) = weights.landing() {
            match crate::rates::save_landing(boot.knobs.profiles.as_deref(), device.name(), &keys, measured) {
                Ok(Some(path)) => eprintln!(
                    "engine-cuda: how this machine lands weights, measured, saved to {path}"
                ),
                Ok(None) => {}
                Err(why) => eprintln!("engine-cuda: the landing's rates were not saved: {why}"),
            }
        }
        // `kernel.gemm.fp8_o_down`: grouped projections on fp8b128, as vLLM
        // reads DeepSeek-V4's `wo_a` (the wide fires on tcgen05).
        #[cfg(feature = "cuda")]
        if device.ctx().tuning().fp8_o_down() && kernels_cuda::linear::gemm_sm100::supported() {
            weights.requantize_grouped(&boot.trace, device.stream())?;
        }
        // `kernel.gemm.fp8_shadow`: block-FP8 dense planes keep a bf16 copy;
        // without it, where the FP8 MMA reads 128x128 blocks, the planes whose
        // rows carry their own exponents do.
        #[cfg(feature = "cuda")]
        if !kernels_cuda::linear::gemm_sm100::supported()
            && (device.ctx().tuning().gemm_fp8_shadow()
                || kernels_cuda::linear::moe::fp8_w8a8_capable())
        {
            let mixed = !device.ctx().tuning().gemm_fp8_shadow();
            weights.shadow_fp8_dense(&boot.trace, device.stream(), mixed)?;
        }
        // `kernel.gemm.fp8_mx`: block-FP8 dense planes keep 1x32 exponents.
        #[cfg(feature = "cuda")]
        if device.ctx().tuning().gemm_fp8_mx() && !kernels_cuda::linear::gemm_sm100::supported() {
            weights.spread_fp8_dense(&boot.trace, device.stream())?;
        }
        // A load with no MXFP4 bank asks nothing of the copies below.
        #[cfg(feature = "cuda")]
        let mxfp4 = boot
            .trace
            .params
            .iter()
            .any(|param| param.dtype == poem_ir::Dtype::Mxfp4);
        // `kernel.moe.mxfp4_fp8_from`: MXFP4 banks keep a block-FP8 copy.
        #[cfg(feature = "cuda")]
        if mxfp4
            && !kernels_cuda::linear::gemm_sm100::supported()
            && kernels_cuda::linear::moe::fp8_w8a8_capable()
            && device.ctx().tuning().mxfp4_fp8_from() > 0
        {
            weights.shadow_mxfp4_banks(&boot.trace, device.stream())?;
        }
        // `kernel.moe.mxfp4_bf16_from`: MXFP4 banks keep a bf16 copy on Hopper.
        #[cfg(feature = "cuda")]
        if mxfp4
            && kernels_cuda::jit::cache::arch() == Some("sm_90")
            && device.ctx().tuning().mxfp4_bf16_from() > 0
        {
            weights.copy_mxfp4_banks_bf16(
                &boot.trace,
                device.stream(),
                device.ctx().tuning().mxfp4_bf16_down(),
                device.ctx().tuning().mxfp4_bf16_keep(),
            )?;
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after weights resident",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        let (graphs, recording) = boot.knobs.modes();
        let (compiled, budgets, tiles) = refit(
            &mut boot,
            &device,
            compiled,
            budgets,
            &profile,
            paging,
            weights.table(),
            engine,
            &asked,
            ceiling,
        )?;
        let paging = paging
            .windowed(
                crate::store::window_of(&boot.trace)?,
                boot.budget.max_tokens,
            )
            .seated(boot.budget.max_lanes)
            .spare(engine.window_spare);
        device.open_lanes(
            compiled.streams.streams.saturating_sub(1),
            compiled.streams.events,
        )?;
        let mut wants_if = false;
        let mut wants_switch = false;
        for region in &compiled.regions {
            match region.lowering {
                poem_compiler::Lowering::AlwaysLaunch => {}
                poem_compiler::Lowering::If => wants_if = true,
                poem_compiler::Lowering::Switch { .. } => wants_switch = true,
            }
        }
        if wants_if || wants_switch {
            device.open_conditional()?;
            let warmed = |what: &str, outcome: core::result::Result<(), kernels_cuda::Error>| {
                outcome.map_err(|why| Fault::Unbound {
                    what: format!(
                        "the {what} this artifact's baked conditional needs, which \
                         answered {why}"
                    ),
                })
            };
            if wants_if {
                warmed(
                    "conditional setter",
                    kernels_cuda::runtime::graph::set_conditional(
                        device.ctx(),
                        0,
                        0,
                        0,
                        false,
                        kernels_cuda::runtime::graph::Arm::Warm,
                        0,
                    ),
                )?;
            }
            if wants_switch {
                warmed(
                    "switch setter",
                    kernels_cuda::runtime::graph::set_switch(
                        device.ctx(),
                        0,
                        0,
                        0,
                        0,
                        kernels_cuda::runtime::graph::Arm::Warm,
                        0,
                    ),
                )?;
            }
            crate::device::ctx::sync(device.stream())?;
        }

        let mut facts = kv::probe(&boot.trace)?;
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;
        crate::window::no_grouped_window_is_also_a_prepare_window(&compiled)?;
        one_direction_per_mask(&boot.trace, &compiled)?;
        let masked = masked_classes(&boot.trace, &compiled);
        let mask_read = crate::exports::mask_reading_classes(&boot.trace, &compiled);
        let group_masked = group_masked_classes(&boot.trace, &compiled);
        let wide = wide_classes(&boot.trace, &compiled);
        let corrected = corrected_classes(&boot.trace, &compiled);
        let landing = landing_requests(&boot.trace.facts, &compiled.classes);
        let decoding = decoding_of(&landing);
        // A schedule only decoding classes read sees a row a lane: its
        // partials are sized for the lanes, not the widest fire's rows.
        facts.mark_rows_a_lane(&read_only_by_decoders(&boot.trace, &compiled, &decoding));
        let feeds = Feeds::of(&boot.trace, &compiled);
        let stated_feeds = crate::exports::stated_feeds(&boot.trace, &feeds.ragged);
        let tiers = crate::record::Tiers {
            plain: plain_of(
                &landing,
                &Exports::of(&boot.trace, &compiled, boot.context, boot.page_size)?.outputs,
            ),
        };
        let drafting_words: std::collections::HashSet<u64> = compiled
            .classes
            .classes
            .iter()
            .enumerate()
            .filter(|(class, _)| {
                tiers.plain.contains(*class) && !landing[*class].iter().any(poem_ir::Request::plain)
            })
            .flat_map(|(_, class)| class.words.iter().copied())
            .collect();
        let media = media_classes(&boot.trace, &compiled);
        let shifted = regions_shifting(&boot.trace, &compiled);
        let lane_shifted = regions_lane_shifting(&boot.trace, &compiled);
        let staging = crate::exports::regions_staging(&boot.trace, &compiled);
        let schedule_readers = regions_launching_schedules(&boot.trace, &compiled);
        crate::voxels::relabel_conv_weights(&device, &boot.trace, weights.table())?;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after weights",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        // The fires that run the side streams carve their own arena
        // (`CompiledModel::arena_forked`): the larger of the two is what the
        // pool keeps under its end for a fire's activation.
        let forked_bytes = compiled
            .arena_forked
            .as_ref()
            .map_or(0, |forked| forked.bytes);
        let arena_bytes = compiled.arena.bytes.max(forked_bytes);
        // What allocates on the device beside the pools allocates before
        // they read the card: the guests' fan streams, and on ROCm what the
        // runtime and its libraries take at their first use. A flat range
        // is sized to what the card then has free, and nothing allocates
        // beside it after but arming, which the bodies record holds.
        #[cfg(feature = "rocm")]
        kernels_cuda::hip::open().map_err(|why| Fault::Unbound {
            what: format!("the HIP runtime's first use: {why}"),
        })?;
        // The streams a serving load copies on: synchronous copies on the
        // null stream (the weights' upload among them), fires and read-backs
        // on the device's, taps on its tap stream, tier moves on its swap.
        #[cfg(feature = "rocm")]
        kernels_cuda::hip::warm_copies(&[
            core::ptr::null_mut(),
            device.stream().cast(),
            device.tap_stream().cast(),
            device.swap_stream().cast(),
        ])
        .map_err(|why| Fault::Unbound {
            what: format!("a copying stream's first copy: {why}"),
        })?;
        let mut programs = ProgramPlane::new(crate::program::compile::Disk::rooted(
            boot.cache_dir.map(|dir| dir.join(CODE_DIR)),
        ));
        programs.open_fan()?;
        let pools = Pools::reserve(
            device.ordinal(),
            ceiling,
            &boot.trace,
            paging,
            boot.kv_budget,
            &facts,
            &budgets,
            arena_bytes,
            boot.knobs.diagnostics.chunk_bytes,
            // `untabled-experts`: the banks stay on the host tier with no
            // shape, so no table reads them.
            if boot.knobs.diagnostics.untabled_experts {
                &[]
            } else {
                boot.residency.cached_experts()
            },
            !boot.knobs.dense_weights(),
            boot.knobs.diagnostics.one_window,
            boot.knobs.diagnostics.windowed_state,
            boot.budget.max_tokens,
        )?;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after pools",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        let mut pools = pools;
        if boot.context_declared {
            pools.declare_context();
        }
        let buffers = Buffers::reserve(&boot.trace, budgets.tokens.max_tokens, &pools)?;
        // Each row-streamed table's cache opens over its host copy.
        let hosts: Vec<(u32, u64)> = pools
            .row_params()
            .into_iter()
            .filter_map(|param| match weights.table().0.get(param as usize) {
                Some(Some(crate::run::WeightRow::Dense(table))) => Some((param, table.ptr)),
                _ => None,
            })
            .collect();
        pools.seat_rows(&hosts)?;
        // And each expert shape's, over its banks' host copies.
        let hosts: Vec<(u32, u64)> = pools
            .expert_params()
            .into_iter()
            .filter_map(|param| match weights.table().0.get(param as usize) {
                Some(Some(crate::run::WeightRow::Dense(bank))) => Some((param, bank.ptr)),
                Some(Some(crate::run::WeightRow::Planes { codes, .. })) => Some((param, codes.ptr)),
                _ => None,
            })
            .collect();
        // A packed bank's planes share its codes' table.
        let groups: Vec<(u32, Vec<Option<u32>>)> = pools
            .expert_params()
            .into_iter()
            .filter_map(|param| {
                let planes = crate::store::layout::packed_planes(&boot.trace, param as usize);
                (!planes.is_empty()).then_some((param, planes))
            })
            .collect();
        pools.seat_experts(&hosts, &groups)?;
        // The banks laid in lane order, which a slot filled from the file
        // (or a copy pinned from it) is laid in once filled.
        pools.lane_banks(
            pools
                .expert_params()
                .into_iter()
                .filter(|&param| crate::store::layout::lane_ordered(&boot.trace, param as usize))
                .filter_map(|param| {
                    let planes = crate::store::layout::packed_planes(&boot.trace, param as usize);
                    let scales = (*planes.get(1)?)?;
                    let shape = &boot.trace.params[param as usize].shape;
                    Some(crate::store::LaneBank {
                        codes: param,
                        scales,
                        rows: shape[1],
                        k: 2 * shape[2..].iter().product::<u64>(),
                    })
                })
                .collect(),
        );
        // The fp8 banks put in the FP8 matrix core's order, which a slot
        // filled from the file (or a copy pinned from it) is put in once
        // filled.
        pools.fp8_banks(
            pools
                .expert_params()
                .into_iter()
                .filter(|&param| crate::store::layout::fp8_ordered(&boot.trace, param as usize))
                .filter_map(|param| {
                    let shape = &boot.trace.params[param as usize].shape;
                    Some(crate::store::Fp8Bank {
                        codes: param,
                        rows: *shape.get(1)?,
                        k: shape[2..].iter().product::<u64>(),
                    })
                })
                .collect(),
        );
        // The activation rungs a pool may drop to (SPEC decision 6), each a
        // bucket's carve laid at the reservation's end.
        let rungs = activation_rungs(&boot.trace, &budgets, &compiled)?;
        // The own-row pages of a load that keeps rings lie at the span's
        // top, above every carve (`store::OwnRows`).
        let own_rows = pools.own_rows_bytes();
        let arena_bytes = arena_bytes + own_rows;
        eprintln!(
            "engine-cuda: activation rungs {:?} under an arena of {arena_bytes} bytes",
            rungs
                .iter()
                .map(|(tokens, map)| (*tokens, map.bytes))
                .collect::<Vec<_>>(),
        );
        let mut arena = Arena::reserve(arena_bytes)
            .keeping_top(own_rows)
            .with_rungs(rungs);
        // The arena is backed here, at load, whole, out of the elastic pool
        // the allowance leaves past the weights: every carve a fire lays,
        // wherever in the span, and a fire reading out past one row a lane
        // (a verify reads every row), so no fire grows it. One the pool
        // does not hold refuses the load by name.
        arena
            .ensure(&mut pools, arena_bytes)
            .map_err(|why| match why {
                Fault::Ceiling { need, have, .. } | Fault::OutOfMemory { need, have } => {
                    Fault::Residency(format!(
                        "the card does not hold this deployment's activation arena: it backs \
                         {need} bytes at load, and under the load's allowance (`[engine.memory] \
                         budget_device`, or the whole card) the elastic pool past the weights \
                         holds {have}. Raise `[engine.memory] budget_device`, or serve fewer \
                         `max_forward_tokens`."
                    ))
                }
                other => other,
            })?;
        if arena.has_rungs() {
            pools.fit_range_whole();
        }
        let predicate = crate::store::rs::Predicate::reserve(
            boot.budget.max_lanes,
            crate::store::rs::snapshots(&boot.trace, |k_dim| {
                #[cfg(not(feature = "rocm"))]
                {
                    kernels_cuda::ssm::gdn_lands_snapshots(device.ctx(), k_dim)
                }
                #[cfg(feature = "rocm")]
                {
                    let _ = k_dim;
                    true
                }
            }) > 0,
            pools.ids_a_lane(),
        )?;
        pools.seat_tables(
            crate::window::lane_capacity(boot.budget.max_lanes),
            boot.budget
                .max_lanes
                .saturating_mul(crate::store::rs::SNAPSHOTS),
        );
        // A page table a kv shape: the full rows' and the windowed rows'.
        let page_tables = pools.geometries();
        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (
                    poem_ir::Def::Input(poem_ir::RuntimeInput::Patches),
                    poem_ir::Ty::Tensor { shape, dtype },
                ) = (&decl.def, &decl.ty)
                else {
                    return None;
                };
                let width: u64 = shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        poem_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product();
                let element = poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0);
                Some(crate::inputs::PatchSeat {
                    rows: u64::from(ladder.max_patches),
                    row_bytes: width * element,
                    images: u64::from(ladder.max_images),
                    dtype: *dtype,
                })
            })
        });
        let mrope_seat = boot.trace.values.iter().any(|decl| {
            matches!(
                decl.def,
                poem_ir::Def::Input(poem_ir::RuntimeInput::MropePositions)
            )
        });
        let patch_fold = patch_fold(&boot.trace);
        let voxels = match boot.voxels.as_ref() {
            Some(ladder) if compiled.order_for(poem_ir::RowAxis::Voxels).is_some() => Some(
                crate::voxels::Store::reserve(crate::voxels::Seat::of(&boot.trace, ladder))?,
            ),
            _ => None,
        };
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(
                node.op,
                poem_ir::Operation::Gather(poem_ir::Gather::ScatterLiveRows { .. })
            )
        });
        if let Some(value) = feeds.unlanded.first() {
            return Err(Fault::Unbound {
                what: format!(
                    "value {}, a merge over a runtime input this shell cannot land: only a \
                     lane or row guest input under a conjunction of facts is landed in a \
                     merged column before the walk",
                    value.0
                ),
            });
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after buffers",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        let inputs = Inputs::reserve(
            &boot.budget,
            engine.fire_lanes_of(boot.budget.max_lanes),
            paging,
            page_tables,
            &facts,
            compiled.classes.classes.len(),
            compiled.template().len(),
            poem_exec::fire::max_runs(&compiled),
            poem_exec::fire::fragmentable(&compiled),
            device.device(),
            device.toggles(),
            boot.runahead,
            patch_seat,
            mrope_seat,
            &feeds.seats(),
            feeds.selections.len(),
            !mask_read.is_empty() || !wide.is_empty(),
            &schedule_streams(&boot.trace, &compiled),
            pools.ids_a_lane(),
            buffers.as_ref().map(crate::store::rs::Buffers::page_tokens),
            // A beam's fork copies a page's rows of every kv plane the pool
            // holds (`Pools::page_planes`).
            pools.page_planes().len(),
            // A hybrid beam's fork copies a page's first rows of each
            // buffered plane (`BEAM-PAGE-LISTS.md` §11).
            buffers
                .as_ref()
                .map(crate::store::rs::Buffers::page_planes)
                .transpose()?
                .map_or(0, |planes| planes.len()),
            // A stretch a packing an attention stating a mask reads.
            stated_feeds.len(),
        )?;

        let exports = Exports::of(&boot.trace, &compiled, boot.context, boot.page_size)?;
        // A capturing lane's rows are its reader's; the fire stages only
        // their addresses, one a lane of the widest window it lays.
        let capture = crate::scores::CaptureLayout::of(
            &exports,
            boot.context,
            boot.page_size,
            crate::window::lane_capacity(boot.budget.max_lanes),
        );
        if let Some(layout) = &capture {
            programs.set_score_bytes(layout.row_bytes());
        }
        // A beam group's page lists count in kv pages to its bind's depth,
        // and a page it frees waits out the frames in flight.
        // A beam group's width is bound by the shared memory its step lays
        // its tables in (`channel::beam_max_lanes`): a wider bind is refused
        // by name.
        programs.set_beam_bounds(
            boot.page_size,
            u32::from(boot.runahead.frames_in_flight),
            kernels_cuda::runtime::channel::beam_max_lanes(&device.ctx()),
        );

        let airborne = crate::settle::Airborne::new();
        let mut pools = pools;
        pools.watch(airborne.clone());
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after inputs",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        // A run of the activation's span, laid with the staged inputs.
        let readout_rows = crate::device::Buffer::lent(
            (boot.budget.max_lanes as usize)
                .saturating_mul(boot.budget.max_tokens as usize)
                .saturating_mul(size_of::<u64>()),
        );
        // A routed bank on the host that no table of the weight cache reads
        // is staged by its fire: the most experts one fire routes to, laid
        // here with the staged inputs. A lane of it for each stream whose
        // regions run such a select.
        let expert_staging = {
            let mut streams = vec![0u32; boot.trace.nodes.len()];
            for region in compiled.template() {
                for node in region.nodes.clone() {
                    if let Some(stream) = streams.get_mut(node as usize) {
                        *stream = region.stream;
                    }
                }
            }
            crate::experts::staging::Staging::bound(
                &crate::experts::staging::Staging::selects(
                    &boot.trace,
                    weights.table(),
                    &pools.expert_params(),
                    crate::device::alloc::is_host_pointer,
                    |node| streams[node],
                )?,
                budgets.tokens.max_tokens,
            )
        };
        // It is a staged run of the activation's span: the fit seats it or
        // refuses the load, naming it (`Staging::short`).
        if let Some(fact) = expert_staging.fact() {
            eprintln!("{fact}");
        }
        let adapter_seats = weights.adapter_seats();
        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let compiled_towered = compiled.order_for(poem_ir::RowAxis::Patches).is_some();
        // A tensor-parallel rank holds its band of every page's heads, not
        // the page: it reads nothing out.
        let kv_taps = (boot.world.size <= 1)
            .then(|| crate::tap::row_layers(&boot.trace).and_then(|rows| pools.tap_groups(&rows)))
            .flatten();
        let mut shell = Shell {
            ring_runs: None,
            ring_seen: crate::serve::RingSeen::default(),
            weighed,
            device,
            accounting,
            trace: boot.trace,
            compiled,
            regions,
            ep: None,
            keys,
            products,
            routes_measuring: boot.knobs.measuring,
            measured: timed_at_open,
            forced_bucket: None,
            budget: budgets.tokens.clone(),
            budgets,
            patch_seat,
            mrope_seat,
            drops_patch_rows,
            towered: compiled_towered,
            patch_fold,
            runahead: boot.runahead,
            voxels,
            weights,
            arena,
            pools,
            buffers,
            predicate,
            inputs,
            facts,
            page_tables,
            masked,
            mask_read,
            group_masked,
            stated_feeds,
            wide,
            feeds,
            adapter_fact,
            corrected,
            decoding,
            drafting_words,
            tiers,
            landing,
            armed: None,
            media,
            shifted,
            lane_shifted,
            staging,
            schedule_readers,
            adapters: crate::blob::Adapters::new(adapter_seats),
            held: vec![0; boot.slots as usize],
            readout_rows,
            capture,
            // Lent at the stand-in span until laid, as the readout rows
            // are: the size-only walk stages into it and copies nothing.
            expert_staged: expert_staging.at(crate::sizing::STAND_IN),
            expert_staging,
            exports,
            graphs,
            copies: engine.copies,
            seated_first: engine.seated_first,
            seat_to_end: engine.seat_to_end,
            seat_spread: engine.seat_spread,
            prefill_lanes: engine.prefill_lanes,
            fire_lanes: engine.fire_lanes,
            pad: recording.pad(),
            golden_arm: Golden::Off,
            bodies: recording.bodies(),
            bodies_mem: 0,
            bodies_record: None,
            bodies_from_record: false,
            arming_at_load: false,
            rearming: false,
            arm_only: None,
            walked_launches: std::collections::HashMap::new(),
            bodies_took: 0,
            measuring: false,
            workspace: None,
            inputs_at: 0,
            decoded_tiles: Vec::new(),
            arming: false,
            armed_body: None,
            arm_cold: false,
            arm_rs: crate::record::RsBody::Folds,
            arm_deferred: false,
            preseated_as: None,
            card: (0, 0),
            held_for_fires: 0,
            live: engine::live::Live::armed(),
            segments: std::collections::HashMap::new(),
            windows_memo: Vec::new(),
            cache: {
                let mut cache = GraphCache::new();
                cache.watch(airborne.clone());
                cache
            },
            programs,
            settlement: crate::settle::Settlement::open(boot.runahead.staging_depth())?,
            taps: Vec::new(),
            kv_taps,
            tap_sink: None,
            airborne,
            owed: super::Owed::new(),
            guest_faults: Vec::new(),
            spare_events: Vec::new(),
            rs_ahead: std::collections::HashMap::new(),
            written: engine::extent::Written::default(),
            _released: Released,
        };
        if !shell.graphs.records() || !shell.bodies {
            eprintln!(
                "engine-cuda: a test's pinned modes ({:?}, {recording:?}): every fire \
                 walks eagerly (~470 kernel launches of host time per decode step) with \
                 nothing captured",
                shell.graphs
            );
        }
        // The cache rows are declared last, from what the card has left: once
        // with the bodies' allowance held ahead of them so arming is not
        // starved, and again after arming, when the bodies have taken what
        // they take and the rest is the rows' to declare. What the guests'
        // programs will take at the admitted lane count is held through both,
        // since they arrive after the pool is declared and the fires that
        // bring them are past static admission by then.
        // A guest program's epilogue launches a block a lane, so a flight
        // past the card's SMs runs in waves anyway: a wider group flies in
        // several, and its scratch is held at this width, not at the lanes.
        let flight = 1 << shell.device.device().num_sm.max(1).ilog2();
        shell.programs.set_flight(flight);
        shell.programs.set_fire_lanes(shell.fire_lanes());
        // A fire that replays a recorded body carves for the body's lane
        // capacity, not its own lanes: the rungs stated to the runtime
        // follow that carve, so state is never lent under a carve a fire
        // cut to the rung still lays.
        if shell.records_bodies() {
            shell
                .arena
                .reach_lanes(crate::window::lane_capacity(shell.budget.max_lanes));
        }
        // A program's fires lay their tables and scratch in its pairs'
        // state, which the runtime takes in `Instance` slots at bind
        // (`Shell::guest_state`): nothing is held for them here.
        let programs = 0u64;
        // A plane stored as codes and scales is decoded to bf16 before a
        // prefill's dense gemm, into one tile a stream: a scratch slab like
        // any other, laid with the kernels' workspaces once the lattice is
        // walked.
        shell.decoded_tiles = tiles
            .iter()
            .map(|plane| crate::store::decoded_weight_reserve(*plane, 1))
            .collect();
        shell.held_for_fires = programs;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] decoded-weight tiles over {} stream(s) ({})",
                tiles.len(),
                tiles
                    .iter()
                    .map(|plane| format!("{}", plane >> 20))
                    .collect::<Vec<String>>()
                    .join("/"),
            );
        }
        let footprint = |shell: &Shell| footprint(shell, device_total, free_before);
        // The bodies are held what they took when this deployment last
        // armed whole, where it is recorded; with no record, everything the
        // card has past one working set, and the cache rows fill what they
        // leave once they are captured. One number: arming's allowance is
        // the hold, a load fact like the span, and no figure is chosen.
        let mut recorded = false;
        if shell.records_bodies()
            && let Some(dir) = boot.cache_dir
        {
            // A record measured with the device's copy queues is not this
            // process's when they are off (each holds memory of its own).
            #[cfg(feature = "rocm")]
            let copies = format!("|{:?}", kernels_cuda::hip::sys::copy_queues());
            #[cfg(not(feature = "rocm"))]
            let copies = String::new();
            let record =
                super::measured::path(dir, &format!("{bodies_shape}|{:?}{copies}", shell.budget));
            if let Some(measured) = super::measured::read(&record) {
                recorded = true;
                let held = usize::try_from(measured.hold()).unwrap_or(usize::MAX);
                if boot.knobs.diagnostics.arm_trace {
                    eprintln!(
                        "[arm-trace] bodies held {} MiB: {} bodies took {} MiB, their captures {} \
                         MiB, when this deployment last armed whole",
                        held >> 20,
                        measured.bodies,
                        measured.bytes >> 20,
                        measured.captured >> 20,
                    );
                }
                shell.bodies_from_record = true;
                shell.bodies_mem = held;
            }
            shell.bodies_record = Some(record);
        }
        // A flat range cannot grow after its bodies name it, so a load that
        // has no record of what they take measures it: no estimate stands
        // in for the figure.
        shell.measuring =
            shell.records_bodies() && !crate::device::elastic::mapped_on_demand() && !recorded;
        let bodies = shell.bodies_hold();
        let resident = footprint(&shell);
        // What the kernels' scratch slabs take, walked size-only through
        // the compositions the lattice fires before the range is first
        // backed: every slab is laid in the activation's span, past its
        // arenas, at the size the walk asked of it. A take answers its
        // place, and nothing is allocated for it. The fit comes after, so
        // the cache rows take what the walk leaves.
        // The staged inputs' slab is a run of that span as well, past the
        // workspaces: one address every body names, under no rung's carve.
        let walked = if shell.pools.fresh() {
            shell.size_arming()?
        } else {
            None
        };
        let placed = walked.as_ref().map(crate::sizing::Footprint::placed);
        let workspace = placed.as_ref().map_or(0, |placed| {
            kernels_cuda::jit::Slabs::placed_bytes(placed) as u64
        });
        if placed.is_some()
            && let Some(buffers) = shell.buffers.as_mut()
        {
            buffers.lay_reach(&mut shell.pools, true)?;
        }
        let staged = workspace.next_multiple_of(INPUTS_ALIGN);
        let runs: u64 = shell.staged_runs().iter().sum();
        let at = shell.pools.widen_activation(staged.saturating_add(runs))?;
        shell.inputs_at = at + staged;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] size-only walk: {} slabs of {} MiB and the staged inputs' {} MiB \
                 laid in the activation's span: {}",
                placed.as_ref().map_or(0, Vec::len),
                workspace >> 20,
                shell.inputs.bytes() >> 20,
                walked.as_ref().map_or_else(String::new, |walked| walked
                    .slabs
                    .iter()
                    .take(8)
                    .map(|(name, bytes)| format!("{name}={}", bytes >> 20))
                    .collect::<Vec<String>>()
                    .join(" "))
            );
        }
        shell.workspace = placed.map(|slabs| Workspace {
            at,
            bytes: workspace,
            slabs,
        });
        // A load that walked nothing lays the rs scratch for the most any
        // fire may ask.
        if shell.workspace.is_none()
            && let Some(buffers) = shell.buffers.as_mut()
        {
            buffers.lay_reach(&mut shell.pools, false)?;
        }
        let fitted = match shell
            .pools
            .fit_the_card(bodies, shell.held_for_fires, false, &resident)
        {
            // A record's bodies no longer fit beside one working set (the
            // card or the budget is tighter than when it was taken): the
            // load measures again, arming what fits and stating the rest
            // eager, as a load with no record does.
            Err(Fault::Residency(why)) if shell.bodies_from_record => {
                eprintln!(
                    "engine-cuda: the bodies' record no longer fits ({why}); this load measures \
                     them again"
                );
                shell.bodies_from_record = false;
                shell.measuring = !crate::device::elastic::mapped_on_demand();
                let bodies = shell.bodies_hold();
                shell
                    .pools
                    .fit_the_card(bodies, shell.held_for_fires, false, &resident)
            }
            fitted => fitted,
        }
        .map_err(|why| shell.expert_staging.short(why))?;
        // Arming measures what the bodies take where no record (the home's
        // or a committed one) held it.
        if shell.records_bodies() && !shell.bodies_from_record {
            shell.measured.push("bodies".to_string());
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] claim before arming: {} pages, {} chunks, room {} MiB (fresh {})",
                fitted.fit,
                fitted.chunks,
                fitted.room >> 20,
                shell.pools.fresh()
            );
        }
        shell.lay_workspace()?;
        // Arming is allowed what the fit held the bodies: a record's figure,
        // or all the card has past the least range.
        shell.bodies_mem = usize::try_from(fitted.bodies).unwrap_or(usize::MAX);
        Ok(Staged {
            shell,
            rank: boot.world.rank,
            arm_trace: boot.knobs.diagnostics.arm_trace,
            weights_held: boot.knobs.weights_held,
            device_total,
            free_before,
            profiles: boot.knobs.profiles.clone(),
        })
    }
}

/// What a buffer the device allocates is aligned to, kept for the staged
/// inputs' slab in the span.
const INPUTS_ALIGN: u64 = 4096;

/// The kernels' workspaces, laid in the activation's span past its arenas:
/// the slabs a size-only walk of the lattice asked for, each at the most it
/// asked.
#[derive(Debug)]
pub(crate) struct Workspace {
    /// Where it starts in the activation's span.
    at: u64,
    bytes: u64,
    slabs: Vec<(kernels_cuda::jit::SlabKey, usize)>,
}

impl Shell {
    /// The runs of the activation's span a fire stages into and reads out
    /// of, in the order they lie from `inputs_at`: the staged inputs' slab,
    /// the readout rows' table, the voxel store, the rs fold predicates and
    /// the host-resident experts' staging, the guests' samplers' lane
    /// arenas and the capture rows' tables, each to [`INPUTS_ALIGN`].
    fn staged_runs(&self) -> [u64; 7] {
        [
            self.inputs.bytes(),
            self.readout_rows.bytes() as u64,
            self.voxels.as_ref().map_or(0, crate::voxels::Store::bytes),
            self.predicate.bytes(),
            self.expert_staging.bytes(),
            self.lane_arenas_bytes(),
            self.capture
                .as_ref()
                .map_or(0, |layout| layout.table_bytes() as u64),
        ]
        .map(|bytes| bytes.next_multiple_of(INPUTS_ALIGN))
    }

    /// Where a guest's sampler meets across its blocks: the most lanes'
    /// spans one of its launches lays, at the plane's flight on this device.
    fn lane_arenas_bytes(&self) -> u64 {
        crate::program::launch::Arenas::bytes_for(
            self.programs.flight(),
            self.device.device().num_sm,
        )
    }

    /// Backs the workspaces where the range now lies, zeroes them, and has
    /// every take of their slabs answer its place there.
    fn lay_workspace(&mut self) -> Result<()> {
        if let Some(buffers) = &self.buffers {
            buffers.lay_scratch(&mut self.pools)?;
        }
        let [
            inputs,
            readouts,
            voxels,
            predicates,
            experts,
            arenas,
            scores,
        ] = self.staged_runs();
        let staged = inputs + readouts + voxels + predicates + experts + arenas + scores;
        self.pools
            .hold_activation_at(self.inputs_at, staged, false)?;
        let base = self.pools.activation_base() + self.inputs_at;
        crate::device::zero_span(base, usize::try_from(staged).unwrap_or(0))?;
        self.inputs.lay(base);
        self.readout_rows.lay(base + inputs);
        if let Some(store) = self.voxels.as_mut() {
            store.lay(base + inputs + readouts);
        }
        self.predicate.lay(base + inputs + readouts + voxels)?;
        self.expert_staged = self
            .expert_staging
            .at(base + inputs + readouts + voxels + predicates);
        self.programs.set_arenas(crate::program::launch::Arenas {
            at: base + inputs + readouts + voxels + predicates + experts,
            bytes: arenas,
        });
        if let Some(layout) = self.capture.as_mut() {
            layout.lay(base + inputs + readouts + voxels + predicates + experts + arenas);
        }
        // The kernels a guest's wave fires are the engine's, not the
        // program's: loaded here, before ready, so no first fire builds them.
        kernels_cuda::runtime::channel::load_wave_kernels()?;
        kernels_cuda::runtime::channel::load_gather_kernels()?;
        eprintln!(
            "engine-cuda: guests' samplers meet in {} KiB of lane arenas laid at load ({} \
             lanes a flight on {} units), no lane's",
            arenas >> 10,
            self.programs.flight(),
            self.device.device().num_sm,
        );
        let Some(workspace) = &self.workspace else {
            return Ok(());
        };
        self.pools
            .hold_activation_at(workspace.at, workspace.bytes, false)?;
        let base = self.pools.activation_base() + workspace.at;
        crate::device::zero_span(base, usize::try_from(workspace.bytes).unwrap_or(0))?;
        // SAFETY: `base` starts the span just held backed and zeroed, of the
        // bytes these slabs take; it stays held as long as the pools do, and
        // the shell drops its contexts' slabs with them.
        unsafe {
            kernels_cuda::jit::Slabs::place(base as *mut core::ffi::c_void, &workspace.slabs);
        }
        Ok(())
    }
}

/// A shell whose load stopped short of arming its bodies ([`Shell::stage`]).
pub struct Staged {
    pub shell: Shell,
    rank: u32,
    arm_trace: bool,
    /// The load was told its weights are held (`Knobs::weights_held`).
    weights_held: bool,
    device_total: u64,
    free_before: u64,
    /// Where the device's user profile lies (the product times a load
    /// measures are saved there).
    profiles: Option<std::path::PathBuf>,
}

impl Staged {
    /// Arms the bodies and fits the cache rows to what the card has left.
    pub fn finish(self) -> Result<Shell> {
        let Staged {
            mut shell,
            rank,
            arm_trace,
            weights_held,
            device_total,
            free_before,
            profiles,
        } = self;
        if arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB before arming",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        // Weights left in the serving file are on no tier a fire can read:
        // they are seated in the pool before the lattice is armed over them.
        // So are those a load told they are held keeps a host copy of
        // (planes laid at load, which the file does not hold as served).
        let filed = shell
            .weights
            .experts()
            .is_some_and(|tier| tier.plan().files());
        let held_copied = weights_held
            && shell
                .weights
                .experts()
                .is_some_and(|tier| !tier.plan().cached_experts().is_empty());
        if filed || held_copied {
            shell.preseat_weights(filed, held_copied)?;
        }
        // A load whose weights the pool does not hold whole streams them:
        // armed now, its bodies would read every expert from the host copy,
        // over the bus, fire after fire. It arms at the starting weight
        // residency instead, over the slots the cache then holds, and the
        // fit keeps the bodies' allowance out of the cache rows till then,
        // when it settles ([`Shell::settle_fit`]).
        let deferred =
            shell.ep.is_none() && shell.records_bodies() && !shell.pools.weights_seated_whole();
        if deferred {
            shell.arm_deferred = true;
            eprintln!(
                "engine-cuda: this load streams its weights; its bodies arm at the starting \
                 weight residency, over the slots the cache holds then"
            );
        } else {
            shell.arming_at_load = true;
            let armed = shell.arm_bodies();
            shell.arming_at_load = false;
            armed?;
            shell.routes_measured(profiles.as_deref())?;
            if shell.measuring {
                shell.measured_refit(device_total, free_before)?;
            }
        }
        let resident = footprint(&shell, device_total, free_before);
        // Whatever armed a body, the range stays where it lies: the bodies
        // name addresses in it, and a fit that reserved it anew (a flat
        // range's does) would leave them naming the old one.
        if shell.cache.body_stats().census.bodies > 0 {
            shell.pools.pin();
        }
        // No fire of the load's own is still in flight over the chunks the
        // last fit lets go and lays again.
        shell.device.synchronize()?;
        let fitted = if deferred && shell.arena.has_rungs() {
            // The bodies arm after the runtime has read the pool, so the
            // fit that follows them cannot be the one it is stated: it is
            // stated the range whole, what that fit may yet give the cache
            // rows with it, and keeps the pool under it by what this fit
            // holds back for the bodies' allowance and the activation no
            // fire has backed yet (`Shell::settle_fit` lifts it). The
            // state's chunks then end where the activation starts, so what
            // a rung lends and what lies under it are one span.
            let fitted = shell.pools.fit_the_card(
                shell.bodies_hold(),
                shell.held_for_fires,
                false,
                &resident,
            )?;
            shell.pools.state_whole();
            fitted
        } else {
            // Everything the load puts on the device is there, its bodies
            // among it: the pool is what the card has left, and the range
            // is stated as exactly the chunks backed below. Arming fired
            // every composition, so the activation is at its high water; a
            // load that records no bodies has fired none, and keeps the
            // whole span for its fires; one that arms later and lends no
            // rung keeps the bodies' allowance out as well.
            let bodies = if deferred { shell.bodies_hold() } else { 0 };
            let settled = shell.records_bodies() && !deferred;
            shell
                .pools
                .fit_and_state(bodies, shell.held_for_fires, settled, &resident)?
        };
        // A load that arms later may have had its range reserved anew by
        // its fit: the workspaces are laid where it now lies.
        shell.lay_workspace()?;
        // The load's last fit backs what it took, here and not later: every
        // page the load serves is backed before anything else can take the
        // card's room, so serving maps nothing (a dying process holding the
        // driver's lock stalls no fire of this one's), and the pool the
        // runtime is stated is the pool the card backed.
        shell.back_whole()?;
        if let Some(kv) = shell.pools.stated_kv() {
            eprintln!(
                "engine-cuda: kv_device_tokens = {}: the device pool holds {} kv pages of {} \
                 tokens ({} asked), {} chunks of {} MiB: {} for the kv pages and the shapes \
                 beside them, {} for the weights held whole",
                kv.tokens,
                shell.pools.paging().pages(),
                kv.page_tokens,
                kv.pages,
                kv.chunks(),
                kv.chunk_bytes >> 20,
                kv.kv_chunks,
                kv.weight_chunks,
            );
        }
        if arm_trace {
            eprintln!(
                "[arm-trace] cache rows sized to {} pages ({} MiB, {} sequences at the declared \
                 context) of the {} MiB the card hands out for them ({resident})",
                fitted.fit,
                shell.pools.declared_bytes() >> 20,
                fitted.fit / u64::from(shell.pools.paging().pages_per_slot),
                fitted.room >> 20,
            );
            if let Some(free) = crate::device::alloc::free_bytes() {
                let ledger = ledger(&shell, device_total, free_before);
                let audit = ledger.audit(device_total.saturating_sub(free));
                eprintln!(
                    "[arm-trace] ledger: {ledger}; the device reports {} MiB in use: {} MiB no \
                     owner entered",
                    audit.device >> 20,
                    audit.gap() >> 20,
                );
            }
            eprintln!(
                "[arm-trace] after arming: {} MiB asked past the laid workspaces",
                kernels_cuda::jit::Slabs::unplaced_bytes() >> 20,
            );
        }
        shell.card = (device_total, free_before);
        if rank != 0 {
            shell.programs.set_shadow(true);
        }
        // Only a warm fire waits on the device, and a load whose bodies were
        // held to nothing fires none: a fault raised by what the load put
        // down asynchronously would otherwise be the first request's to
        // find, as a sticky error from whichever call touches the context
        // next, under a banner that said ready.
        shell.device.synchronize().map_err(|why| {
            Fault::program(
                "serve::load",
                format!(
                    "the device faulted under this load's own work ({why}); a context that \
                     has faulted serves nothing, so the load is refused rather than reported \
                     ready"
                ),
            )
        })?;
        Ok(shell)
    }
}

fn footprint(shell: &Shell, device_total: u64, free_before: u64) -> Footprint {
    Footprint {
        total: device_total,
        before: device_total.saturating_sub(free_before),
        weights: shell.weights.bytes(),
        arena: shell.arena.bytes(),
        inputs: shell.inputs.bytes(),
        buffers: shell.buffer_bytes(),
        readout: shell.readout_rows.bytes() as u64,
        experts: shell.expert_staging.bytes(),
        cache_rows: shell.pools.committed_bytes(),
        bodies: shell.cache.body_stats().census.bytes as u64,
    }
}

impl Shell {
    /// The staging a fire lays a host-resident bank's experts in, as the
    /// load bounded it (`experts::staging`).
    #[must_use]
    pub fn expert_staging(&self) -> &crate::experts::staging::Staging {
        &self.expert_staging
    }

    /// What the fit holds the bodies ahead of the cache rows' fill: none on
    /// a load that records none; what they took when this deployment last
    /// armed whole, where that is recorded; else everything past one working
    /// set ([`crate::store::MEASURING`]).
    fn bodies_hold(&self) -> u64 {
        if !self.records_bodies() {
            0
        } else if self.bodies_from_record {
            self.bodies_mem as u64
        } else {
            crate::store::MEASURING
        }
    }

    /// The device's memory as this load holds it now ([`ledger`]).
    #[must_use]
    pub fn ledger(&self) -> engine::memory::Ledger {
        ledger(self, self.card.0, self.card.1)
    }

    /// A measuring load's second fit: the bodies armed over the least range
    /// The decode product shapes the armed bodies met with no measured time
    /// (`kernels_cuda::linear::routes`): each chosen by its decode body's
    /// time under each kernel (`Shell::routes_by_body`), saved in the
    /// device's user profile, and the lattice left armed under the choices.
    /// A load whose profile holds every shape measures nothing.
    fn routes_measured(&mut self, profiles: Option<&std::path::Path>) -> Result<()> {
        let Some(routes) = self.device.ctx().routes().cloned() else {
            return Ok(());
        };
        let wanted = routes.wanted();
        if wanted.is_empty() {
            eprintln!(
                "engine-cuda: decode products' kernels from the profile: every shape the bodies \
                 hold reused, 0 measured"
            );
            return Ok(());
        }
        // A load that records no bodies serves every fire as an eager walk:
        // there is no body to time a product's kernel by, so it keeps the
        // routes it has, saves no choice to the profile, and says so.
        if !self.records_bodies() {
            eprintln!(
                "engine-cuda: card: routes unmeasured, no recorded bodies: {} decode product \
                 shape(s) keep their routes as they stand, and none is saved",
                wanted.len()
            );
            return Ok(());
        }
        // Under `[engine] measure = false` nothing is timed: each shape no
        // layer states keeps its stated kernel, by name, and none is saved.
        if self.routes_measuring == engine::rates::Measuring::Skip {
            eprintln!(
                "engine-cuda: rates: skipped: `[rates] product_ns` for {} decode product \
                 shape(s): nothing timed (`[engine] measure = false`); each keeps its stated \
                 kernel",
                wanted.len()
            );
            return Ok(());
        }
        let started = std::time::Instant::now();
        let measured = self.routes_by_body(routes, &wanted)?;
        if measured.is_empty() {
            // Every shape a decode body times is the profile's: nothing is
            // measured or saved; the rest (rows no decode body carries) keep
            // their stated kernel.
            eprintln!(
                "engine-cuda: decode products' kernels from the profile: 0 measured; {} shape(s) \
                 at rows no decode body carries keep their stated kernel",
                wanted.len()
            );
            return Ok(());
        }
        self.measured
            .push(format!("product_ns ({} decode product times)", measured.len()));
        let saved = crate::rates::save_product_times(
            profiles,
            self.device.name(),
            &self.keys,
            &self.products,
            &measured,
        )
        .map_err(Fault::Card)?;
        let chosen = self.device.ctx().routes().cloned();
        let said: Vec<String> = wanted
            .iter()
            .map(|(s, noted)| {
                let us = |kernel| {
                    measured
                        .iter()
                        .find(|(shape, k, ..)| shape == s && *k == kernel)
                        .map_or(f64::NAN, |&(.., ns, _)| ns / 1e3)
                };
                let pick = chosen
                    .as_ref()
                    .map_or(noted.candidates.first().copied().unwrap_or(0), |c| {
                        c.pick(*s, 0, &noted.candidates)
                    });
                let times: Vec<String> = noted
                    .candidates
                    .iter()
                    .map(|&kernel| format!("{kernel:08x} {:.1} us", us(kernel)))
                    .collect();
                format!(
                    "{:?} {}x{} m{}: step {} -> {pick:08x}",
                    s.kind,
                    s.rows,
                    s.k,
                    s.m,
                    times.join(" / "),
                )
            })
            .collect();
        eprintln!(
            "engine-cuda: chose the kernel of {} decode product shapes by their bodies' times in \
             {:.1} s ({}); {}",
            said.len(),
            started.elapsed().as_secs_f64(),
            saved.map_or_else(|| "not saved: no profile directory".to_string(), |p| format!("saved to {p}")),
            said.join("; "),
        );
        Ok(())
    }

    /// took `bodies_took`; they are forgotten, the range is fitted to what
    /// the card has past that hold, and the lattice armed again over it.
    /// Its record holds the figure, so a later boot fits once.
    fn measured_refit(&mut self, device_total: u64, free_before: u64) -> Result<()> {
        let started = std::time::Instant::now();
        let mut hold = self.bodies_took;
        let weighed = self.cache.body_stats().census.bytes as u64;
        self.cache.forget_bodies();
        // A flat range was allocated at the least range the bodies armed
        // over; where the card backs more, it is let go (the weights
        // preseated in it with it) so the refit reserves it anew at what the
        // card backs, and the weights are seated again in it below.
        let preseated = self.pools.preseated();
        self.device.synchronize()?;
        let refilled = self.pools.free_for_refit()?;
        let resident = footprint(self, device_total, free_before);
        let fitted = match self
            .pools
            .fit_the_card(hold, self.held_for_fires, false, &resident)
        {
            // What the captures took (the device's free bytes they cost)
            // past one working set no longer fits: an arming that stopped for
            // room holds what its bodies weigh, the rest stated eager.
            Err(Fault::Residency(_)) if weighed < hold => {
                hold = weighed;
                self.pools
                    .fit_the_card(hold, self.held_for_fires, false, &resident)?
            }
            fitted => fitted?,
        };
        if refilled && let (true, Some((filed, copied))) = (preseated, self.preseated_as) {
            let began = std::time::Instant::now();
            self.preseat_weights(filed, copied)?;
            let chunks = self
                .pools
                .preseat_plan()
                .map_or(0, |plan| self.pools.weight_chunks(&plan).len());
            eprintln!(
                "engine-cuda: a measuring boot's flat range held the least range the bodies \
                 armed over; it was reserved anew at {} chunks, and the weights the pool holds \
                 ({chunks} chunks) were seated in it again from their copy in {:.1} s",
                fitted.chunks,
                began.elapsed().as_secs_f64()
            );
        }
        self.lay_workspace()?;
        self.bodies_mem = usize::try_from(hold).unwrap_or(usize::MAX);
        self.rearming = true;
        self.arming_at_load = true;
        let armed = self.arm_bodies();
        self.arming_at_load = false;
        self.rearming = false;
        armed?;
        eprintln!(
            "engine-cuda: a measuring boot: with no record of what this deployment's bodies \
             take, they armed over the least range that serves and took {} MiB; the range was \
             fitted again to the rest ({} chunks) and the lattice armed over it in {:.1} s. \
             Later boots hold the measured figure and fit once",
            hold >> 20,
            fitted.chunks,
            started.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    /// Two load facts of this shell
    /// ([`engine::chunk::ChunkFacts::held_bytes`] and `outside_bytes`):
    /// what the fit holds back of the chunks it stated, and what the driver
    /// holds outside the engine's span, as the ledger enters it (read by
    /// the policy, taken out of no range).
    #[must_use]
    pub fn held_and_outside(&self) -> (u64, u64) {
        (
            self.pools.kept_back_bytes(),
            self.ledger().bytes(engine::memory::Class::Outside),
        )
    }

    /// Settles the fit of a load that armed at its starting weight
    /// residency. Its last fit ran with nothing fired: it kept the bodies'
    /// allowance and the activation's whole span out of the cache rows
    /// ([`Staged::finish`]). The lattice has fired now, so the activation
    /// is at its high water and the bodies hold what they take: the rest is
    /// the cache rows', as on a load that armed before its last fit.
    ///
    /// The range stays where it lies (the bodies name it) and the chunks
    /// stated to the runtime stand: what moves is how many of them the fit
    /// holds back ([`Self::held_and_outside`]), which the residency answers
    /// and the runtime lifts its pool's hold-back by. What that opens is
    /// backed when the runtime comes to take it (`Pools::back_whole`),
    /// before the load returns.
    pub(super) fn settle_fit(&mut self) -> Result<()> {
        self.pools.pin();
        // A pool stated as the chunks it backs has nothing held back to
        // lift: its load held the bodies' allowance out of it.
        if !self.pools.stated_whole() {
            return Ok(());
        }
        let resident = footprint(self, self.card.0, self.card.1);
        let before = (self.pools.declared_bytes(), self.pools.kept_back_bytes());
        let fitted = self
            .pools
            .fit_the_card(0, self.held_for_fires, true, &resident)?;
        if super::diag::on().arm_trace {
            eprintln!(
                "[arm-trace] settled at the starting weight residency: cache rows {} -> {} MiB \
                 ({} sequences at the declared context) of the {} MiB the card hands out for \
                 them ({resident}); {} -> {} MiB of the stated chunks held back",
                before.0 >> 20,
                self.pools.declared_bytes() >> 20,
                fitted.fit / u64::from(self.pools.paging().pages_per_slot),
                fitted.room >> 20,
                before.1 >> 20,
                self.pools.kept_back_bytes() >> 20,
            );
        }
        Ok(())
    }
}

/// The device's memory as this load holds it ([`engine::memory`]): each
/// owner under its class, over what the device has for this process (all of
/// it but what others held before the load began).
fn ledger(shell: &Shell, device_total: u64, free_before: u64) -> engine::memory::Ledger {
    use engine::memory::{Class, Ledger};
    let held = footprint(shell, device_total, free_before);
    let mut ledger = Ledger::new(shell.pools.chunk_bytes(1), device_total);
    ledger.enter(Class::Model, "weights", held.weights);
    let workspaces = shell
        .workspace
        .as_ref()
        .map_or(0, |workspace| workspace.bytes);
    let rs_scratch = shell
        .buffers
        .as_ref()
        .map_or(0, |buffers| buffers.scratch_bytes());
    ledger.enter(Class::Fire, "activation", held.arena);
    ledger.enter(Class::Fire, "workspaces", workspaces);
    ledger.enter(Class::Fire, "rs scratch", rs_scratch);
    if crate::device::elastic::mapped_on_demand() {
        ledger.enter(Class::State, "cache rows", held.cache_rows);
    } else {
        // A flat range is backed whole once allocated: the state's chunks
        // hold the device whether committed or not, and so does the
        // activation's span past the carve, the workspaces and the rs
        // scratch laid in it.
        let span = shell.pools.activation_span_bytes();
        ledger.enter(
            Class::State,
            "cache rows",
            shell.pools.backed_bytes().saturating_sub(span),
        );
        ledger.enter(
            Class::Fire,
            "activation span spare",
            span.saturating_sub(
                held.arena + workspaces + rs_scratch + shell.staged_runs().iter().sum::<u64>(),
            ),
        );
    }
    ledger.enter(Class::Fire, "inputs", held.inputs);
    ledger.enter(Class::Fire, "readout rows", held.readout);
    ledger.enter(Class::Fire, "staged experts", held.experts);
    // The heaps' chunks are this process's whatever load made them, and a
    // load carves its own cells from them (the readout rows among them),
    // so they are entered whole, less the cells entered by their owners,
    // and the device held before the load is entered without them.
    let heaps = crate::device::held::heap_bytes() as u64;
    let heaps_before = heaps.saturating_sub(crate::device::held::heap_bytes_since() as u64);
    ledger.enter(
        Class::Outside,
        "heap chunks",
        heaps.saturating_sub(if shell.readout_rows.carved() {
            held.readout
        } else {
            0
        }),
    );
    ledger.enter(
        Class::Outside,
        "idle shelf",
        crate::device::held::kept_bytes() as u64,
    );
    ledger.enter(Class::Outside, "graph bodies", held.bodies);
    ledger.enter(
        Class::Outside,
        "before the load",
        held.before.saturating_sub(heaps_before),
    );
    // Where the allocator's own takes are known, the driver's and its
    // libraries' (streams' queues, code objects) are measured as what the
    // device holds past them, the bodies apart; the audit then states only
    // what this load allocated that no owner entered.
    if let (Some(malloced), Some(free)) = (
        crate::device::held::malloced_since(),
        crate::device::alloc::free_bytes(),
    ) {
        let in_use = device_total.saturating_sub(free);
        let library = crate::device::held::blas_library_since();
        if library > 0 {
            ledger.enter(Class::Outside, "hipBLASLt", library);
        }
        ledger.enter(
            Class::Outside,
            "driver",
            in_use
                .saturating_sub(held.before)
                .saturating_sub(malloced)
                .saturating_sub(held.bodies)
                .saturating_sub(library),
        );
    }
    ledger
}

/// What holds the device when the cache rows come to be declared, in bytes:
/// what was in use before this load began (other processes, and this one's
/// context) and each thing the load has put down since. The elastic budget
/// is what the load's allowance leaves after all of it.
struct Footprint {
    total: u64,
    before: u64,
    weights: u64,
    arena: u64,
    inputs: u64,
    buffers: u64,
    readout: u64,
    /// The host-resident experts' staging (`experts::staging`).
    experts: u64,
    cache_rows: u64,
    bodies: u64,
}

impl core::fmt::Display for Footprint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "the device holds {} bytes, {} of them in use before this load began; this load \
             put down weights {}, activation arena {}, inputs {}, buffered activations {}, \
             readout rows {}, staged experts {}, cache rows {}, graph bodies {}",
            self.total,
            self.before,
            self.weights,
            self.arena,
            self.inputs,
            self.buffers,
            self.readout,
            self.experts,
            self.cache_rows,
            self.bodies,
        )
    }
}

/// The widest plane one stream decodes to bf16 before a dense gemm, and the
/// widest among those fired at token rows: the lm_head is fired at readout
/// rows and is the widest plane of a text model by far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DecodedPlanes {
    widest: u64,
    at_tokens: u64,
}

/// Per stream, the planes the linear and attention nodes launched on it
/// decode to bf16 before a dense gemm, at the `[n, k]` the trace declares,
/// two bytes an element; an embedding gathers from the codes and a repacked
/// plane has its own kernels. A linear's plane is fired at the rows its
/// output lands. Index 0 is the main stream, `n` the `n`th side stream.
fn decoded_weight_planes(
    trace: &poem_ir::Trace,
    compiled: &CompiledModel,
    table: &crate::run::WeightTable,
) -> Vec<DecodedPlanes> {
    use poem_ir::Operands;

    let plane_bytes = |id: poem_ir::ValueId| -> Option<u64> {
        let decl = trace.values.get(id.0 as usize)?;
        let poem_ir::Def::Weight(w) = &decl.def else {
            return None;
        };
        let poem_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            return None;
        };
        table
            .0
            .get(*w as usize)
            .and_then(Option::as_ref)
            .filter(|row| {
                matches!(
                    row,
                    crate::run::WeightRow::Planes {
                        repacked: false,
                        lanes: false,
                        ..
                    }
                )
            })?;
        Some(
            shape
                .iter()
                .map(|dim| match dim {
                    poem_ir::Dim::Const(n) => *n,
                    _ => 1,
                })
                .fold(2u64, u64::saturating_mul),
        )
    };
    let token_rows = |id: poem_ir::ValueId| -> bool {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        let poem_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            return false;
        };
        matches!(
            shape.first(),
            Some(dim) if dim.axis().is_some()
                && !matches!(
                    dim,
                    poem_ir::Dim::Lanes | poem_ir::Dim::LanesPlus(_) | poem_ir::Dim::Readouts
                )
        )
    };
    let mut planes: Vec<DecodedPlanes> =
        vec![DecodedPlanes::default(); compiled.streams.streams.max(1) as usize];
    let mut inputs: Vec<poem_ir::ValueId> = Vec::new();
    let mut outputs: Vec<poem_ir::ValueId> = Vec::new();
    for region in compiled.template() {
        let Some(slot) = planes.get_mut(region.stream as usize) else {
            continue;
        };
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            let at_tokens = match &node.op {
                // A low-precision product decodes no bf16 tile.
                #[cfg(not(feature = "rocm"))]
                poem_ir::Operation::Linear(
                    poem_ir::Linear::Matmul {
                        quant: Some(_),
                        logits: false,
                        ..
                    }
                    | poem_ir::Linear::MoeSelect {
                        bias: None,
                        quantized: true,
                        quant: Some(_),
                        ..
                    },
                ) => continue,
                #[cfg(not(feature = "rocm"))]
                poem_ir::Operation::Fused(
                    poem_ir::Fused::MoeSelectGlu { select, .. }
                    | poem_ir::Fused::MoeExperts { select, .. }
                    | poem_ir::Fused::MoeMega { select, .. },
                ) if matches!(
                    **select,
                    poem_ir::Linear::MoeSelect {
                        bias: None,
                        quantized: true,
                        quant: Some(_),
                        ..
                    }
                ) =>
                {
                    continue;
                }
                // A routed select reads its bank's codes as they are, whatever
                // the arm (the register-dequant and staged kernels, the
                // grouped and per-route ones): it decodes no bf16 tile.
                #[cfg(not(feature = "rocm"))]
                poem_ir::Operation::Linear(
                    poem_ir::Linear::MoeSelect {
                        bias: None,
                        quantized: false,
                        quant: None,
                        ..
                    }
                    | poem_ir::Linear::MoeSelect {
                        bias: Some(_),
                        quantized: false,
                        quant: None,
                        ..
                    }
                    | poem_ir::Linear::MoeSelect {
                        bias: None,
                        quantized: true,
                        quant: None,
                        ..
                    },
                )
                | poem_ir::Operation::Fused(
                    poem_ir::Fused::MoeSelectGlu { .. }
                    | poem_ir::Fused::MoeExperts { .. }
                    | poem_ir::Fused::MoeMega { .. },
                ) => continue,
                // A fused projection runs its matmul when no fused kernel
                // takes it, and decodes the same plane that matmul would.
                poem_ir::Operation::Linear(_)
                | poem_ir::Operation::Fused(
                    poem_ir::Fused::MatmulGlu { .. }
                    | poem_ir::Fused::MatmulSwigluMatmul { .. }
                    | poem_ir::Fused::MatmulBias { .. }
                    | poem_ir::Fused::MatmulMoeRoute { .. }
                    | poem_ir::Fused::MatmulSplitRows { .. }
                    | poem_ir::Fused::MatmulSoftcap { .. },
                ) => {
                    outputs.clear();
                    node.op.outputs(&mut outputs);
                    outputs.iter().any(|id| token_rows(*id))
                }
                #[cfg(feature = "rocm")]
                poem_ir::Operation::Fused(
                    poem_ir::Fused::MoeSelectGlu { .. }
                    | poem_ir::Fused::MoeExperts { .. }
                    | poem_ir::Fused::MoeMega { .. },
                ) => {
                    outputs.clear();
                    node.op.outputs(&mut outputs);
                    outputs.iter().any(|id| token_rows(*id))
                }
                poem_ir::Operation::Attn(_) | poem_ir::Operation::Ssm(_) => true,
                _ => continue,
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                if let Some(bytes) = plane_bytes(*id) {
                    slot.widest = slot.widest.max(bytes);
                    if at_tokens {
                        slot.at_tokens = slot.at_tokens.max(bytes);
                    }
                }
            }
            // On ROCm an fp8b128 projection is one dense e4m3 plane, decoded
            // to bf16 for a prefill's GEMM like a split-plane one; on CUDA
            // without tcgen05 so is its W8A16 product past
            // `w8a16::GEMM_UNTIL` rows.
            #[cfg(feature = "rocm")]
            let decoded = true;
            #[cfg(all(feature = "cuda", not(feature = "rocm")))]
            let decoded = !kernels_cuda::linear::gemm_sm100::supported();
            #[cfg(any(feature = "rocm", feature = "cuda"))]
            if decoded
                && let poem_ir::Operation::Linear(poem_ir::Linear::Matmul {
                    w,
                    scales: Some(_),
                    quant: Some(_),
                    logits: false,
                    ..
                }) = &node.op
                && let Some(poem_ir::Ty::Tensor { shape, .. }) =
                    trace.values.get(w.0 as usize).map(|decl| &decl.ty)
            {
                let bytes = shape
                    .iter()
                    .map(|dim| match dim {
                        poem_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .fold(2u64, u64::saturating_mul);
                slot.widest = slot.widest.max(bytes);
                if at_tokens {
                    slot.at_tokens = slot.at_tokens.max(bytes);
                }
            }
        }
    }
    planes
}

fn patch_fold(trace: &poem_ir::Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Gather(
                poem_ir::Gather::MergeRows { side, .. } | poem_ir::Gather::PoolRows { side, .. },
            ) => Some(side.saturating_mul(*side)),
            _ => None,
        })
        .fold(1u32, |fold, side| fold.saturating_mul(side))
        .max(1)
}

fn adapter_fact(classes: &poem_ir::ClassTable, corrected: &poem_ir::ClassSet) -> Option<u32> {
    classes.adapter_fact(corrected)
}

pub(super) struct Baked {
    pub(super) device: Context,
    pub(super) engine: crate::chip::Engine,
    pub(super) compiled: CompiledModel,
    pub(super) budgets: Budgets,
    pub(super) profile: DeviceProfile,
    /// The region each `fused.region` of the trace names, by key.
    pub(super) regions: std::collections::BTreeMap<u64, kernels_ir::Region>,
    /// The anchored region each `fused.anchored` of the trace names, by key.
    pub(super) anchored: std::collections::BTreeMap<u64, kernels_ir::Anchored>,
    /// The budget asked before the card cut its tokens at the asked lanes.
    /// What this load's stored measurements are keyed by.
    pub(super) keys: crate::stored::Keys,
    /// The decode products' times the device's profile states under a
    /// matching key, which a measurement of more joins.
    pub(super) products: Vec<profile::fire::ProductTime>,
    /// How the device's profile says this machine lands weights.
    pub(super) landing: Option<profile::fire::Landing>,
    pub(super) asked: Budget,
    /// What the card may hold in all under the load's allowance, resolved
    /// before the load took anything (`elastic::ceiling_of`).
    pub(super) ceiling: u64,
    /// The rungs the top was weighed of, and the estimate that weighed them.
    pub(super) weighed: Option<crate::top::Weighed>,
    /// What the rates step timed at open (none where every rate was stated).
    pub(super) timed_at_open: Vec<String>,
}

/// The widest fire whose lanes of `rows` prompt rows put no more (lane,
/// value head) blocks of the trace's widest chunked gated delta walk than the
/// card has SMs; `None` where `rows` is 0 or the trace has no such walk.
fn walk_fire_fit(trace: &poem_ir::Trace, rows: u32, sms: u32) -> Option<u32> {
    if rows == 0 {
        return None;
    }
    let heads = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Ssm(poem_ir::Ssm::GatedDeltaChunked { v_heads, .. }) => {
                Some(*v_heads)
            }
            _ => None,
        })
        .max()?;
    Some((sms / heads.max(1)).max(1) * rows)
}

/// The activation rungs below the largest (SPEC decision 6): for every
/// power of two of tokens, the largest bucket at most that many tokens and
/// its carve, where that carve is smaller than the arena for every fire.
/// Smallest first.
fn activation_rungs(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
    compiled: &CompiledModel,
) -> Result<Vec<(u32, poem_compiler::ArenaMap)>> {
    let max = budgets.tokens.max_tokens;
    let mut tokens: Vec<u32> = engine::memory::rungs(1, max)
        .into_iter()
        .filter_map(|rung| {
            budgets
                .tokens
                .buckets
                .iter()
                .copied()
                .filter(|&bucket| bucket <= rung && bucket < max)
                .max()
        })
        .collect();
    tokens.dedup();
    let mut rungs = Vec::new();
    for rung in tokens {
        let map = poem_compiler::arena_rung(trace, budgets, compiled, rung)?;
        if map.bytes < compiled.arena.bytes {
            rungs.push((rung, map));
        }
    }
    Ok(rungs)
}

#[cfg(test)]
mod tests {
    use super::widened;

    /// A lattice to 13778, as the MI300X laid it.
    const LATTICE: [u32; 21] = [
        512, 1024, 2048, 2234, 2436, 2656, 2897, 3159, 3445, 3757, 4096, 4467, 4871, 5312, 5793,
        6317, 6889, 7513, 8192, 9742, 13778,
    ];

    fn on(rows: u32) -> u32 {
        LATTICE
            .iter()
            .copied()
            .filter(|rung| *rung <= rows)
            .max()
            .unwrap_or(LATTICE[0])
    }

    /// A tight-budget load whose lanes were cut grows its fire back to the
    /// top rung it chose (3445), not to the lattice's top (13778) that every
    /// wider fire would also fit: what it reserved is the chosen top's.
    #[test]
    fn a_cut_fire_grows_back_to_the_chosen_top_and_no_further() {
        let mut asked = Vec::new();
        let rows = widened(1024, 3445, on, |wider| {
            asked.push(wider);
            Ok(true)
        })
        .unwrap();
        assert_eq!(rows, 3445);
        assert!(asked.iter().all(|wider| *wider <= 3445), "{asked:?}");
    }

    /// It stops at the widest rung that fits below the top.
    #[test]
    fn a_cut_fire_stops_where_the_wider_fire_does_not_fit() {
        let rows = widened(512, 3445, on, |wider| Ok(wider <= 2048)).unwrap();
        assert_eq!(rows, 2048);
    }

    /// At the top already, nothing widens.
    #[test]
    fn a_fire_at_its_top_stays() {
        let rows = widened(3445, 3445, on, |_| panic!("nothing past the top is asked")).unwrap();
        assert_eq!(rows, 3445);
    }
}
