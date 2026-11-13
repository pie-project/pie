#![allow(unsafe_code)]
use poem_compiler::arena::{LiveSlot, Peak};
use poem_compiler::{Budget, Budgets, CompiledModel, DeviceProfile};

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::{
    Exports, Feeds, corrected_classes, decoding_of, landing_requests, masked_classes,
    media_classes, one_direction_per_mask, plain_of, read_only_by_decoders, regions_lane_shifting,
    regions_launching_schedules, regions_shifting, schedule_streams, wide_classes,
};
use crate::inputs::Inputs;
use crate::program::Plane as ProgramPlane;
use crate::record::Bodies as GraphCache;
use crate::store::Pools;
use crate::store::kv::{self, Paging};
use crate::store::rs::Buffers;
use crate::weights::Weights;

use super::{Boot, Golden, Graphs, Released, Shell};

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
    let chip = match boot.knobs.chip.clone() {
        Some(chip) => chip,
        None => {
            let (major, minor) = device.capability();
            let measured = crate::chip::facts(
                u32::try_from(major).unwrap_or(0),
                u32::try_from(minor).unwrap_or(0),
                device.device().num_sm,
            );
            let (chip, report) =
                crate::chip::resolve(device.name(), &measured, boot.knobs.profiles.as_deref());
            crate::chip::SHELF.tell_untuned(device.name(), &report);
            chip
        }
    };
    device.set_tuning(kernels_cuda::tuning::Tuning::of(&chip));
    let engine = crate::chip::Engine::of(&chip);
    let bodies_stated = boot.knobs.chip.is_some()
        || crate::chip::states_recording(device.name(), boot.knobs.profiles.as_deref());
    if boot.budget.max_tokens == 0 {
        let params: u64 = boot
            .trace
            .params
            .iter()
            .map(|p| p.shape.iter().product::<u64>())
            .sum();
        boot.budget.max_tokens = match engine.forward_tokens {
            0 => crate::api::DEFAULT_FORWARD_TOKENS,
            tokens
                if params < 2 << 30 || sparse_experts(&boot.trace) || !recurrent(&boot.trace) =>
            {
                tokens.max(crate::api::DEFAULT_FORWARD_TOKENS)
            }
            tokens => tokens,
        };
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
    if !boot.knobs.diagnostics.dense_weights && !boot.knobs.diagnostics.dense_experts {
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
        kernels.retain(|rule| crate::UNCHAINED.contains(&rule.kernel()));
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
            tuning.decode_grouped_prefill_from,
            tuning.decode_grouped_prefill_lse,
            tuning.decode_grouped_prefill_window,
            kernels_cuda::attn::decode_gqa::Gate::of(tuning),
        );
        boot.trace = crate::narrow::gdn_out_bf16(boot.trace.clone(), tuning.ssm_gdn_out_bf16);
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
        let skinny =
            !profile.synth_regions || kernels_cuda::tuning::Tuning::of(&chip).gemm_skinny_rows > 0;
        let (folded, anchored) = crate::synth::fold_anchors_with(boot.trace.clone(), skinny);
        boot.trace = folded;
        anchored
    } else {
        std::collections::BTreeMap::new()
    };
    #[cfg(feature = "rocm")]
    let anchored = std::collections::BTreeMap::new();
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
    if !boot.knobs.diagnostics.dense_weights {
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
    if !boot.knobs.diagnostics.gumbel_direct {
        eta_compiler::codegen::cuda::fused::GUMBEL_DIRECT
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
    // The activation arena and the attention workspaces are sized from
    // `max_forward_tokens`, and on a wide model at a long envelope they are
    // gigabytes before a sequence seats. The card's share after the weight
    // tier is what they must fit in, half of it, so the cache rows, the
    // bodies and the guests' programs have the other half; the token budget
    // halves toward its floor until they do, and the lattice follows.
    let (free, total) = crate::store::device_memory()?;
    let after_weights =
        crate::device::elastic::budget_bytes(free, total, boot.allowance).saturating_sub(
            crate::store::weight_tier_bytes(&boot.trace, boot.residency.device_demand(), 0)?,
        );
    let after_weights = held_beside(boot, after_weights)?;
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
    let budget_at = |tokens: u32| {
        let mut budget = boot.budget.clone();
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
    let floor = boot.budget.max_lanes.max(TOKENS_FLOOR).min(asked_tokens);
    let window = crate::store::window_of(&boot.trace)?;
    let working = |tokens: u32, voxels: &Option<poem_compiler::VoxelLadder>| -> u64 {
        let windowed = Paging::of(boot.page_size, boot.context, boot.slots, 1)
            .ok()
            .and_then(|paging| {
                crate::store::window_fire_bytes(&boot.trace, paging.windowed(window, tokens)).ok()
            })
            .unwrap_or(u64::MAX);
        let budget = budget_at(tokens);
        let Ok(compiled) =
            poem_compiler::compile_axes(&boot.trace, &budgets_at(&budget, voxels), &profile)
        else {
            return u64::MAX;
        };
        compiled
            .arena
            .bytes
            .saturating_add(crate::inputs::attention_workspace_bytes(
                &budget,
                &facts,
                &device.device(),
                device.toggles(),
                &crate::exports::schedule_streams(&boot.trace, &compiled),
            ))
            .saturating_add(windowed)
    };
    let working_at = |tokens: u32| working(tokens, &voxels);
    let share = after_weights / 100 * u64::from(engine.working_set_percent);
    let tokens = crate::store::tokens_within(asked_tokens, floor, share, working_at);
    // The halving can land a whole step short: the widest bucket of the asked
    // lattice past it that still fits serves more rows a fire.
    let tokens = budget_at(asked_tokens)
        .buckets
        .iter()
        .copied()
        .filter(|&rows| rows > tokens && rows < asked_tokens)
        .rev()
        .find(|&rows| working_at(rows) <= share)
        .unwrap_or(tokens);
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
            .saturating_sub(crate::store::BODIES_FLOOR_BYTES)
    });
    if tokens != asked_tokens {
        eprintln!(
            "engine-cuda: a fire of {asked_tokens} tokens is served at {tokens}: at \
             {asked_tokens} the activation arena, attention workspaces and one fire's windowed \
             kv rows take {} MiB of the \
             {} MiB this card has after the weight tier under the load's allowance, and \
             the cache rows, graph bodies and guest programs need the other half.",
            working_at(asked_tokens) >> 20,
            after_weights >> 20,
        );
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
        bodies_stated,
        compiled,
        budgets,
        profile,
        regions,
        anchored,
        asked,
        stated_buckets: !stated_buckets.is_empty(),
    })
}

/// The token and lane budgets fitted to what the card has with the weights
/// resident. What a fire's working set takes at those budgets (the
/// activation arena, the attention workspaces, the decoded-weight tiles, the
/// guests' programs, the capturing lanes' score planes and the readout
/// indices) is held to half of the card past the weights, the half `bake`
/// sizes the token budget against before the weights are down, and to what
/// leaves one sequence and the bodies' floor beside it. The tile yields
/// first, held to the widest plane fired at token rows (a plane past it is
/// served by the fused-dequant arm), then the token budget halves toward its
/// floor, then the lanes halve toward one, and what their cut frees grows the
/// token budget back toward `asked`'s, the budget before `bake` cut it at the
/// asked lanes; `Pools::fit_the_card` still refuses past that.
#[allow(clippy::too_many_arguments)]
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
    let in_file = !params.iter().any(|&param| {
        laid.iter()
            .any(|laid| laid.codes == param || laid.scales == param)
    }) && crate::checkpoint_serving::Serving::open(boot.checkpoint, &boot.trace)
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
) -> Result<(CompiledModel, Budgets, Vec<u64>)> {
    let (free, total) = crate::store::device_memory()?;
    let live = crate::device::elastic::budget_bytes(free, total, boot.allowance);
    let live = held_beside(boot, live)?;
    let facts = kv::probe(&boot.trace)?;
    let window = crate::store::window_of(&boot.trace)?;
    let out_row = Exports::of(&boot.trace, &compiled)?.readout_row_bytes();
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
    let working_at = |lanes: u32, tokens: u32, scoped: bool| -> u64 {
        let budget = budget_at(lanes, tokens);
        let lanes = budget.max_lanes;
        let Ok(compiled) = poem_compiler::compile_axes(&boot.trace, &budgets_at(&budget), profile)
        else {
            return u64::MAX;
        };
        let tiles: u64 = tiles_of(
            &decoded_weight_planes(&boot.trace, &compiled, table),
            scoped,
        )
        .iter()
        .map(|plane| crate::store::decoded_weight_reserve(*plane, 1))
        .fold(0u64, u64::saturating_add);
        compiled
            .arena
            .bytes
            .saturating_add(crate::inputs::attention_workspace_bytes(
                &budget,
                &facts,
                &device.device(),
                device.toggles(),
                &schedule_streams(&boot.trace, &compiled),
            ))
            .saturating_add(tiles)
            .saturating_add(crate::store::program_scratch_reserve(
                lanes.min(device.device().num_sm),
                out_row,
            ))
            .saturating_add(
                u64::from(lanes)
                    .saturating_mul(u64::from(tokens))
                    .saturating_mul(8),
            )
    };
    // What one sequence and the bodies' floor leave a fire, and within it the
    // half the cache rows are owed when the floors can still reach it.
    let room_at = |tokens: u32, owed: bool| -> Result<u64> {
        let sequence =
            crate::store::least_sequence_bytes(&boot.trace, paging.windowed(window, tokens))?;
        let room = live
            .saturating_sub(sequence)
            .saturating_sub(crate::store::BODIES_FLOOR_BYTES);
        Ok(if owed {
            room.min(live / 100 * u64::from(engine.working_set_percent))
        } else {
            room
        })
    };
    let floor = TOKENS_FLOOR.min(asked_tokens);
    let fit = |owed: bool, scoped: bool| -> Result<Option<(u32, u32)>> {
        let mut tokens = asked_tokens;
        while tokens > floor && working_at(asked_lanes, tokens, scoped) > room_at(tokens, owed)? {
            tokens = (tokens / 2).max(floor);
        }
        let room = room_at(tokens, owed)?;
        let lanes = crate::store::tokens_within(asked_lanes.min(tokens), 1, room, |lanes| {
            working_at(lanes, tokens, scoped)
        });
        // A 1024-lane boot of MiniMax-M3.1-Flash mini60 on an H100 cut its fires to
        // 1024 rows at 1024 lanes, then the lanes to 512: one 717-token prompt a fire.
        // Back at 8192 rows: pf64-512 7.05k -> 10.35k tok/s (vLLM 9.03k at 1024
        // sequences), pf1024-512 7.92k -> 10.37k (9.00k), c1024 2.69k -> 2.71k (2.72k).
        let mut tokens = tokens;
        while lanes < asked_lanes && tokens < asked.max_tokens {
            let wider = tokens.saturating_mul(2).min(asked.max_tokens);
            if working_at(lanes, wider, scoped) > room_at(wider, owed)? {
                break;
            }
            tokens = wider;
        }
        let room = room_at(tokens, owed)?;
        Ok((working_at(lanes, tokens, scoped) <= room).then_some((tokens, lanes)))
    };
    let planes = decoded_weight_planes(&boot.trace, &compiled, table);
    if working_at(asked_lanes, asked_tokens, false) <= room_at(asked_tokens, true)? {
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
         workspaces, decoded-weight tiles, guest programs) takes {} MiB of the {} MiB this \
         card has after the weight tier under the load's allowance, past the half the cache \
         rows are owed and what one sequence at the declared context and the bodies' floor \
         need.",
        asked.max_tokens,
        working_at(asked_lanes, asked_tokens, scoped) >> 20,
        live >> 20,
    );
    boot.budget = budget_at(lanes, tokens);
    let budgets = budgets_at(&boot.budget);
    let compiled = poem_compiler::compile_axes(&boot.trace, &budgets, profile)?;
    let planes = decoded_weight_planes(&boot.trace, &compiled, table);
    Ok((compiled, budgets, tiles_of(&planes, scoped)))
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
        let Baked {
            mut device,
            engine,
            bodies_stated,
            compiled,
            budgets,
            profile,
            regions: folded,
            anchored,
            asked,
            stated_buckets,
        } = bake(&mut boot)?;
        // Every generated kernel compiled and loaded now, before a fire can
        // ask for one.
        let regions =
            crate::synth::Regions::load(&boot.trace, &folded, &anchored, &device.ctx().tuning())?;
        let (free_before, device_total) = crate::store::device_memory()?;
        crate::device::held::heap_mark();
        // The pool's address space is reserved for every page its bytes
        // could hold, the stated budget or the share of the card before the
        // weights land; `Pools::fit_the_card` sizes it within that.
        let pool_bytes = boot.kv_budget.unwrap_or_else(|| {
            crate::device::elastic::budget_bytes(free_before, device_total, boot.allowance)
        });
        let paging = Paging::of(
            boot.page_size,
            boot.context,
            boot.slots,
            crate::store::page_ceiling(&boot.trace, boot.page_size, boot.context, pool_bytes)?,
        )?
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
            boot.residency.device_demand(),
            decoded_dense,
            &boot.trace,
            paging,
            boot.kv_budget,
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
        )?;
        // `kernel.gemm.fp8_o_down`: grouped projections on fp8b128, as vLLM
        // reads DeepSeek-V4's `wo_a` (the wide fires on tcgen05).
        #[cfg(feature = "cuda")]
        if device.ctx().tuning().fp8_o_down && kernels_cuda::linear::gemm_sm100::supported() {
            weights.requantize_grouped(&boot.trace, device.stream())?;
        }
        // `kernel.gemm.fp8_shadow`: block-FP8 dense planes keep a bf16 copy;
        // without it, where the FP8 MMA reads 128x128 blocks, the planes whose
        // rows carry their own exponents do.
        #[cfg(feature = "cuda")]
        if !kernels_cuda::linear::gemm_sm100::supported()
            && (device.ctx().tuning().gemm_fp8_shadow
                || kernels_cuda::linear::moe::fp8_w8a8_capable())
        {
            let mixed = !device.ctx().tuning().gemm_fp8_shadow;
            weights.shadow_fp8_dense(&boot.trace, device.stream(), mixed)?;
        }
        // `kernel.gemm.fp8_mx`: block-FP8 dense planes keep 1x32 exponents.
        #[cfg(feature = "cuda")]
        if device.ctx().tuning().gemm_fp8_mx && !kernels_cuda::linear::gemm_sm100::supported() {
            weights.spread_fp8_dense(&boot.trace, device.stream())?;
        }
        // `kernel.moe.mxfp4_fp8_from`: MXFP4 banks keep a block-FP8 copy.
        #[cfg(feature = "cuda")]
        if device.ctx().tuning().mxfp4_fp8_from > 0
            && !kernels_cuda::linear::gemm_sm100::supported()
            && kernels_cuda::linear::moe::fp8_w8a8_capable()
        {
            weights.shadow_mxfp4_banks(&boot.trace, device.stream())?;
        }
        // `kernel.moe.mxfp4_bf16_from`: MXFP4 banks keep a bf16 copy on Hopper.
        #[cfg(feature = "cuda")]
        if device.ctx().tuning().mxfp4_bf16_from > 0
            && kernels_cuda::jit::cache::arch() == Some("sm_90")
        {
            weights.copy_mxfp4_banks_bf16(
                &boot.trace,
                device.stream(),
                device.ctx().tuning().mxfp4_bf16_down,
                device.ctx().tuning().mxfp4_bf16_keep,
            )?;
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after weights resident",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
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
        let wide = wide_classes(&boot.trace, &compiled);
        let corrected = corrected_classes(&boot.trace, &compiled);
        let landing = landing_requests(&boot.trace.facts, &compiled.classes);
        let decoding = decoding_of(&landing);
        // A schedule only decoding classes read sees a row a lane: its
        // partials are sized for the lanes, not the widest fire's rows.
        facts.mark_rows_a_lane(&read_only_by_decoders(&boot.trace, &compiled, &decoding));
        let feeds = Feeds::of(&boot.trace, &compiled);
        let tiers = crate::record::Tiers {
            plain: plain_of(&landing, &Exports::of(&boot.trace, &compiled)?.outputs),
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
        let pools = Pools::reserve(
            device.ordinal(),
            boot.allowance,
            &boot.trace,
            paging,
            boot.kv_budget,
            &facts,
            &budgets,
            arena_bytes,
            boot.knobs.diagnostics.chunk_bytes,
            boot.residency.cached_experts(),
            !boot.knobs.diagnostics.dense_weights,
            boot.knobs.diagnostics.one_window,
        )?;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after pools",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        let mut pools = pools;
        let buffers =
            Buffers::reserve(&boot.trace, paging, budgets.tokens.max_tokens, &mut pools)?;
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
        // Committed whole: a fire reading out past one row a lane (a verify
        // reads every row) would otherwise grow it outside the fit.
        let unscored = compiled.arena.scores_from(&boot.trace).max(
            compiled
                .arena_forked
                .as_ref()
                .map_or(0, |forked| forked.scores_from(&boot.trace)),
        );
        // The activation rungs a pool may drop to (SPEC decision 6), each a
        // bucket's carve laid at the reservation's end.
        let rungs = activation_rungs(&boot.trace, &budgets, &compiled)?;
        eprintln!(
            "engine-cuda: activation rungs {:?} under an arena of {arena_bytes} bytes",
            rungs
                .iter()
                .map(|(tokens, map)| (*tokens, map.bytes))
                .collect::<Vec<_>>(),
        );
        // What each carve laid at the end reads before its scores.
        let prefixes: Vec<(u64, u64)> = std::iter::once(&compiled.arena)
            .chain(compiled.arena_forked.as_ref())
            .chain(rungs.iter().map(|(_, map)| map))
            .map(|map| (map.bytes, map.scores_from(&boot.trace)))
            .collect();
        let mut arena = Arena::reserve(arena_bytes).with_rungs(rungs, prefixes);
        arena.ensure(&mut pools, unscored)?;
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
            !masked.is_empty() || !wide.is_empty(),
            &schedule_streams(&boot.trace, &compiled),
            pools.ids_a_lane(),
            buffers.as_ref().map(crate::store::rs::Buffers::page_tokens),
        )?;

        let exports = Exports::of(&boot.trace, &compiled)?;

        let airborne = crate::settle::Airborne::new();
        let mut pools = pools;
        pools.watch(airborne.clone());
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after inputs",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        let readout_rows = crate::device::Buffer::zeroed(
            (boot.budget.max_lanes as usize)
                .saturating_mul(boot.budget.max_tokens as usize)
                .saturating_mul(size_of::<u64>()),
        )?;
        let adapter_seats = weights.adapter_seats();
        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let compiled_towered = compiled.order_for(poem_ir::RowAxis::Patches).is_some();
        // A tensor-parallel rank holds its band of every page's heads, not
        // the page: it reads nothing out.
        let kv_taps = (boot.world.size <= 1)
            .then(|| crate::tap::row_layers(&boot.trace).and_then(|rows| pools.tap_groups(&rows)))
            .flatten();
        let mut shell = Shell {
            device,
            accounting,
            trace: boot.trace,
            compiled,
            regions,
            ep: None,
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
            exports,
            graphs: engine.graphs,
            copies: engine.copies,
            seated_first: engine.seated_first,
            seat_to_end: engine.seat_to_end,
            seat_spread: engine.seat_spread,
            prefill_lanes: engine.prefill_lanes,
            fire_lanes: engine.fire_lanes,
            lattice: (!stated_buckets && engine.prefill_lattice_from > 0).then_some((
                engine.lattice_steps,
                engine.prefill_lattice_from,
                engine.prefill_lattice_steps,
            )),
            pad: engine.recording.pad(),
            golden_arm: Golden::Off,
            bodies: engine.recording.bodies(),
            bodies_mem: (engine.recording.bodies_mem() as usize).saturating_mul(1 << 20),
            // An operator's allowance (the user's profile, or a pinned one)
            // is a hard cap; a profile's default may stretch while arming.
            bodies_stated,
            bodies_ceiling: 0,
            workspace: None,
            decoded_tiles: Vec::new(),
            arming: false,
            armed_body: None,
            arm_cold: false,
            arm_rs: crate::record::RsBody::Folds,
            arm_deferred: false,
            arm_stepped: boot.knobs.arm_stepped,
            stepping: None,
            lent: None,
            live: engine::live::Live::armed(),
            segments: std::collections::HashMap::new(),
            windows_memo: Vec::new(),
            cache: {
                let mut cache = GraphCache::new();
                cache.watch(airborne.clone());
                cache
            },
            programs: ProgramPlane::new(crate::program::compile::Disk::rooted(
                boot.cache_dir.map(|dir| dir.join(CODE_DIR)),
            )),
            settlement: crate::settle::Settlement::open(boot.runahead.staging_depth())?,
            taps: Vec::new(),
            kv_taps,
            tap_sink: None,
            airborne,
            owed: super::Owed::new(),
            spare_events: Vec::new(),
            rs_ahead: std::collections::HashMap::new(),
            _released: Released,
        };
        if !shell.graphs.records() {
            eprintln!(
                "engine-cuda: engine.graphs is {}, a diagnostic mode — every fire \
                 walks eagerly (~470 kernel launches of host time per decode step) \
                 with nothing captured; a profile serves `on` with bodies",
                match shell.graphs {
                    Graphs::Off => "off",
                    Graphs::Shaped => "shaped",
                    Graphs::On => "on",
                }
            );
        } else if !shell.bodies {
            eprintln!(
                "engine-cuda: engine.recording keeps no bodies under engine.graphs = on, a \
                 diagnostic arm — bodies are the only recorded path, so every fire walks \
                 eagerly (~470 kernel launches of host time per decode step) with nothing \
                 captured; a profile serves `bodies`"
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
        let programs = crate::store::program_scratch_reserve(
            boot.budget.max_lanes.min(flight),
            shell.exports.readout_row_bytes(),
        );
        // A plane stored as codes and scales is decoded to bf16 before a
        // prefill's dense gemm, into one tile a stream: a scratch slab like
        // any other, laid with the kernels' workspaces once the lattice is
        // walked.
        shell.decoded_tiles = tiles
            .iter()
            .map(|plane| crate::store::decoded_weight_reserve(*plane, 1))
            .collect();
        let held_for_fires = programs;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] held for fires: guest programs {} MiB at {} lanes; decoded-weight \
                 tiles over {} stream(s) ({})",
                programs >> 20,
                boot.budget.max_lanes,
                tiles.len(),
                tiles
                    .iter()
                    .map(|plane| format!("{}", plane >> 20))
                    .collect::<Vec<String>>()
                    .join("/"),
            );
        }
        let footprint = |shell: &Shell| footprint(shell, device_total, free_before);
        let bodies = if shell.records_bodies() {
            shell.bodies_mem as u64
        } else {
            0
        };
        let resident = footprint(&shell);
        // On ROCm, arming the tall buckets grows the kernels' scratch slabs
        // (top-k scores, chunked-recurrence planes, attention partials) after
        // this fit, by about the activation arena again at a 16K-token
        // budget; held for them here, the cache rows leave the bodies the
        // room they were allowed (GLM's mini armed 11 of 49 compositions
        // without it, its prefill buckets walking eagerly). A hybrid's slots
        // are cut at this fit for good, and on CUDA its chunked recurrence
        // grows them the same way (Kimi-K3's mini: 3.1 GB of KDA planes at
        // 8192 tokens left an RTX 5090's rows short of one sequence).
        let scratch_growth = if (cfg!(feature = "rocm")
            || shell.trace.nodes.iter().any(|node| {
                matches!(
                    node.op,
                    poem_ir::Operation::Ssm(poem_ir::Ssm::KdaChunked { .. })
                )
            }))
            && shell.records_bodies()
        {
            shell.arena.bytes()
        } else {
            0
        };
        let mut fitted = shell.pools.fit_the_card(
            bodies,
            held_for_fires.saturating_add(scratch_growth),
            false,
            !shell.bodies_stated,
            &resident,
        )?;
        // What arming will grow the kernels' scratch slabs by, walked
        // size-only through the compositions it arms before the range is
        // first backed: the fit holds that, not a stretch or an arena's
        // worth guessed for it, and the cache rows take the rest.
        if shell.records_bodies() && shell.pools.fresh() {
            let began = std::time::Instant::now();
            let walked = shell.size_arming()?;
            let walk_ms = began.elapsed().as_secs_f64() * 1000.0;
            // The slabs this shell's contexts take are laid in the
            // activation's span, past its arenas, at the size the walk asked
            // of each: a take answers its place, and nothing grows. What is
            // left to grow by allocation is the slabs of other arenas.
            let arena = shell.device.ctx().slabs().id();
            let placed: Vec<(kernels_cuda::jit::SlabKey, usize)> = walked
                .keys
                .iter()
                .filter(|slab| slab.key.arena == arena)
                .map(|slab| (slab.key, slab.asked))
                .collect();
            let grows = walked
                .keys
                .iter()
                .filter(|slab| slab.key.arena != arena)
                .fold(0u64, |sum, slab| sum.saturating_add(slab.grown as u64));
            let workspace = kernels_cuda::jit::Slabs::placed_bytes(&placed) as u64;
            if workspace > 0 {
                let at = shell.pools.activation_bytes();
                shell.pools.widen_activation(workspace)?;
                shell.workspace = Some(Workspace {
                    at,
                    bytes: workspace,
                    slabs: placed,
                });
            }
            let margin = (grows / 50).max(SIZED_MARGIN);
            if boot.knobs.diagnostics.arm_trace {
                eprintln!(
                    "[arm-trace] size-only walk ({walk_ms:.0} ms): {} slabs of {} MiB laid in the \
                     activation's span; {} MiB of other slabs grow (held {} MiB with its margin, \
                     against {} MiB guessed): {}",
                    shell.workspace.as_ref().map_or(0, |w| w.slabs.len()),
                    workspace >> 20,
                    grows >> 20,
                    (grows + margin) >> 20,
                    (scratch_growth + fitted.stretch) >> 20,
                    walked
                        .slabs
                        .iter()
                        .take(8)
                        .map(|(name, bytes)| format!("{name}={}", bytes >> 20))
                        .collect::<Vec<String>>()
                        .join(" ")
                );
            }
            let first = (fitted.fit, fitted.chunks, fitted.room >> 20);
            fitted = shell.pools.fit_the_card(
                bodies,
                held_for_fires.saturating_add(grows).saturating_add(margin),
                false,
                false,
                &resident,
            )?;
            if boot.knobs.diagnostics.arm_trace {
                eprintln!(
                    "[arm-trace] fit before arming: {} pages, {} chunks, room {} MiB guessed; \
                     {} pages, {} chunks, room {} MiB walked (fresh {})",
                    first.0,
                    first.1,
                    first.2,
                    fitted.fit,
                    fitted.chunks,
                    fitted.room >> 20,
                    shell.pools.fresh()
                );
            }
        }
        // The most a profile's allowance may stretch to while arming.
        shell.bodies_ceiling = usize::try_from(fitted.bodies).unwrap_or(usize::MAX);
        // The carves laid at the activation's end, backed where the range
        // now lies (the fit may have reserved it anew).
        shell.arena.ensure_rungs(&mut shell.pools)?;
        shell.lay_workspace()?;
        let afforded = fitted.bodies - fitted.stretch;
        if afforded < bodies {
            eprintln!(
                "engine-cuda: [engine] bodies_mem {} MiB is held down to {} MiB on this card: \
                 with the weights, activations and inputs resident, one sequence at the declared \
                 context and the guests' programs seated, {} MiB is left for the graph bodies \
                 and the cache rows together, and the bodies take at most a quarter of it. \
                 State a smaller bodies_mem to choose it.",
                bodies >> 20,
                afforded >> 20,
                fitted.spare >> 20,
            );
            shell.bodies_mem = usize::try_from(afforded).unwrap_or(usize::MAX);
        }
        Ok(Staged {
            shell,
            held_for_fires,
            rank: boot.world.rank,
            arm_trace: boot.knobs.diagnostics.arm_trace,
            weights_held: boot.knobs.weights_held,
            device_total,
            free_before,
        })
    }
}

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
    /// Backs the workspaces where the range now lies, zeroes them, and has
    /// every take of their slabs answer its place there.
    fn lay_workspace(&mut self) -> Result<()> {
        if let Some(buffers) = &self.buffers {
            buffers.lay_scratch(&mut self.pools)?;
        }
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

/// The least the fit holds past a size-only walk's slabs: the slabs grow
/// as the walk grew them, but arming also takes a few allocations of its
/// own beside them (graph bodies' upload staging, events).
const SIZED_MARGIN: u64 = 64 << 20;

/// A shell whose load stopped short of arming its bodies ([`Shell::stage`]).
pub struct Staged {
    pub shell: Shell,
    held_for_fires: u64,
    rank: u32,
    arm_trace: bool,
    /// The load was told its weights are held (`Knobs::weights_held`).
    weights_held: bool,
    device_total: u64,
    free_before: u64,
}

impl Staged {
    /// Arms the bodies and fits the cache rows to what the card has left.
    pub fn finish(self) -> Result<Shell> {
        let Staged {
            mut shell,
            held_for_fires,
            rank,
            arm_trace,
            weights_held,
            device_total,
            free_before,
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
        // fit keeps the bodies' allowance out of the cache rows till then.
        let deferred =
            shell.ep.is_none() && shell.records_bodies() && !shell.pools.weights_seated_whole();
        if deferred {
            shell.arm_deferred = true;
            eprintln!(
                "engine-cuda: this load streams its weights; its bodies arm at the starting \
                 weight residency, over the slots the cache holds then"
            );
        } else {
            // The rungs past the smallest arm in the lane time the caller
            // gives, over slots it lends: of the kv list and the records,
            // so not where a fire also names windowed pages.
            shell.arm_stepped = shell.arm_stepped
                && shell.ep.is_none()
                && !shell.pools.has_windowed();
            shell.arm_bodies()?;
        }
        let stepping = shell.stepping.is_some();
        let resident = footprint(&shell, device_total, free_before);
        // Arming fired every composition, so the activation is at its high
        // water; a load that records no bodies, or arms at its starting
        // weight residency, has fired none yet, and keeps the bodies'
        // allowance out of the cache rows till it arms.
        // One whose rungs past the smallest arm later keeps what is left of
        // the allowance out likewise, and its range where the bodies armed
        // name it.
        let settled = shell.records_bodies() && !deferred && !stepping;
        let bodies = if deferred {
            shell.bodies_mem as u64
        } else if stepping {
            shell.pools.pin();
            (shell.bodies_mem as u64).saturating_sub(shell.cache.body_stats().census.bytes as u64)
        } else {
            0
        };
        let fitted = shell
            .pools
            .fit_the_card(bodies, held_for_fires, settled, false, &resident)?;
        // A load that arms later may have had its range reserved anew by
        // this fit: the workspaces are laid where it now lies.
        shell.lay_workspace()?;
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
                "[arm-trace] after arming: {} MiB asked past the laid workspaces; slabs grown by \
                 allocation: {}",
                kernels_cuda::jit::Slabs::unplaced_bytes() >> 20,
                kernels_cuda::jit::Slabs::census()
                    .iter()
                    .map(|(name, bytes)| format!("{name}={}K", bytes >> 10))
                    .collect::<Vec<String>>()
                    .join(" ")
            );
        }
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

/// Whether `trace` carries a recurrent state (a gated delta or Mamba
/// mixer): the narrower profile fire serves those; a dense attention model's
/// prefill GEMMs are compute-bound and run best at the full 8192 rows.
fn recurrent(trace: &poem_ir::Trace) -> bool {
    trace
        .nodes
        .iter()
        .any(|node| matches!(node.op, poem_ir::Operation::Ssm(_)))
}

fn sparse_experts(trace: &poem_ir::Trace) -> bool {
    let dims = |v: poem_ir::ValueId| match &trace.values[v.0 as usize].ty {
        poem_ir::Ty::Tensor { shape, .. } => shape.clone(),
        poem_ir::Ty::Struct(_) => Vec::new(),
    };
    trace.nodes.iter().any(|node| match node.op {
        poem_ir::Operation::Linear(poem_ir::Linear::MoeSelect { bank, routes, .. }) => {
            matches!(
                (dims(bank).first(), dims(routes).last()),
                (Some(poem_ir::Dim::Const(experts)), Some(poem_ir::Dim::Const(top_k)))
                    if *experts > 2 * *top_k
            )
        }
        _ => false,
    })
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
        cache_rows: shell.pools.committed_bytes(),
        bodies: shell.cache.body_stats().census.bytes as u64,
        scratch: kernels_cuda::jit::Slabs::census()
            .iter()
            .map(|(_, bytes)| *bytes as u64)
            .sum(),
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
    ledger.enter(Class::State, "cache rows", held.cache_rows);
    ledger.enter(Class::Fire, "activation", held.arena);
    ledger.enter(
        Class::Fire,
        "workspaces",
        shell.workspace.as_ref().map_or(0, |workspace| workspace.bytes),
    );
    ledger.enter(Class::Fire, "inputs", held.inputs);
    ledger.enter(Class::Fire, "readout rows", held.readout);
    ledger.enter(Class::Fire, "grown slabs", held.scratch);
    ledger.enter(
        Class::Fire,
        "rs scratch",
        shell.buffers.as_ref().map_or(0, |buffers| buffers.scratch_bytes()),
    );
    ledger.enter(
        Class::Outside,
        "heap chunks",
        crate::device::held::heap_bytes_since() as u64,
    );
    ledger.enter(
        Class::Outside,
        "idle shelf",
        crate::device::held::kept_bytes() as u64,
    );
    ledger.enter(Class::Outside, "graph bodies", held.bodies);
    ledger.enter(Class::Outside, "before the load", held.before);
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
    cache_rows: u64,
    bodies: u64,
    scratch: u64,
}

impl core::fmt::Display for Footprint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "the device holds {} bytes, {} of them in use before this load began; this load \
             put down weights {}, activation arena {}, inputs {}, buffered activations {}, \
             readout rows {}, cache rows {}, graph bodies {}, kernel scratch {}",
            self.total,
            self.before,
            self.weights,
            self.arena,
            self.inputs,
            self.buffers,
            self.readout,
            self.cache_rows,
            self.bodies,
            self.scratch,
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
    /// The bodies' allowance is the operator's (a user or pinned profile
    /// states `[engine] recording`): a hard cap.
    pub(super) bodies_stated: bool,
    pub(super) compiled: CompiledModel,
    pub(super) budgets: Budgets,
    pub(super) profile: DeviceProfile,
    /// The region each `fused.region` of the trace names, by key.
    pub(super) regions: std::collections::BTreeMap<u64, kernels_ir::Region>,
    /// The anchored region each `fused.anchored` of the trace names, by key.
    pub(super) anchored: std::collections::BTreeMap<u64, kernels_ir::Anchored>,
    /// The budget asked before the card cut its tokens at the asked lanes.
    pub(super) asked: Budget,
    /// Whether the load stated its buckets (no default lattice to coarsen).
    pub(super) stated_buckets: bool,
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

/// The activation rungs below the largest (SPEC decision 6): for each of
/// [`poem_compiler::ACTIVATION_RUNGS`], the largest bucket at most that
/// many tokens and its carve, where that carve is smaller than the arena
/// for every fire. Smallest first.
fn activation_rungs(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
    compiled: &CompiledModel,
) -> Result<Vec<(u32, poem_compiler::ArenaMap)>> {
    let max = budgets.tokens.max_tokens;
    let mut tokens: Vec<u32> = poem_compiler::ACTIVATION_RUNGS
        .iter()
        .filter_map(|&rung| {
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
