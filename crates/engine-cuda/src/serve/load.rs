use poem_compiler::{Budget, Budgets, CompiledModel, DeviceProfile};

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::{
    Exports, Feeds, corrected_classes, decoding_of, landing_requests, masked_classes,
    media_classes, plain_of, regions_lane_shifting, regions_launching_schedules, regions_shifting,
    schedule_streams,
};
use crate::inputs::Inputs;
use crate::program::Plane as ProgramPlane;
use crate::record::Bodies as GraphCache;
use crate::store::Pools;
use crate::store::kv::{self, Paging};
use crate::store::rs::Buffers;
use crate::weights::Weights;

use super::{Boot, FireCost, Golden, Graphs, Shell};

/// The least `max_forward_tokens` is served at when the card is short: one
/// prefill chunk a frame, which serves, and under it the halving stops.
const TOKENS_FLOOR: u32 = 1024;

pub(super) fn bake(boot: &mut Boot<'_>) -> Result<Baked> {
    super::diag::publish(&boot.knobs.diagnostics);
    // SAFETY: the boot's communicator is held by the engine for the life of
    // the shell it loads.
    let device = Context::bind(boot.ordinal, unsafe { boot.comm.as_ref() })?;

    kernels_cuda::disk::install(boot.cache_dir);

    let stated_buckets = std::mem::take(&mut boot.budget.buckets);
    boot.budget.buckets = crate::api::lattice(stated_buckets.clone(), boot.budget.max_tokens);

    let mut profile = boot.profile.take().unwrap_or(DeviceProfile {
        sms: device.device().num_sm,
        ..DeviceProfile::default()
    });
    if let Some(streams) = boot.knobs.side_streams {
        profile.side_streams = streams;
    }
    profile.exclusive = crate::EXCLUSIVE
        .iter()
        .map(|op| (*op).to_string())
        .collect();
    profile.grouped = if boot.knobs.grouped {
        crate::GROUPED.iter().map(|op| (*op).to_string()).collect()
    } else {
        Vec::new()
    };
    let kernels: &[&str] = if boot.knobs.diagnostics.fuse_chains {
        &crate::FUSED
    } else {
        &crate::UNCHAINED
    };
    boot.trace = poem_compiler::fuse::fuse(boot.trace.clone(), kernels);
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
        crate::device::elastic::budget_bytes(free, total, boot.knobs.gpu_mem_utilization)
            .saturating_sub(crate::store::weight_tier_bytes(
                &boot.trace,
                boot.residency.device_demand(),
                0,
            )?);
    let facts = kv::probe(&boot.trace)?;
    let budgets_at = |budget: &Budget| Budgets {
        tokens: budget.clone(),
        patches: boot.patches.clone(),
        voxels: boot.voxels.clone(),
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
        budget.buckets = crate::api::lattice(buckets, tokens);
        budget
    };
    let asked_tokens = boot.budget.max_tokens;
    let floor = boot.budget.max_lanes.max(TOKENS_FLOOR).min(asked_tokens);
    let window = crate::store::window_of(&boot.trace)?;
    let working_at = |tokens: u32| -> u64 {
        let windowed = Paging::of(boot.page_size, boot.context, boot.slots, 1)
            .ok()
            .and_then(|paging| {
                crate::store::window_fire_bytes(&boot.trace, paging.windowed(window, tokens)).ok()
            })
            .unwrap_or(u64::MAX);
        let budget = budget_at(tokens);
        let Ok(compiled) = poem_compiler::compile_axes(&boot.trace, &budgets_at(&budget), &profile)
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
                &crate::exports::schedule_streams(&boot.trace, &compiled),
            ))
            .saturating_add(windowed)
    };
    let tokens = crate::store::tokens_within(asked_tokens, floor, after_weights / 2, working_at);
    if tokens != asked_tokens {
        eprintln!(
            "engine-cuda: [engine] max_forward_tokens {asked_tokens} is served at {tokens}: at \
             {asked_tokens} the activation arena, attention workspaces and one fire's windowed \
             kv rows take {} MiB of the \
             {} MiB this card has after the weight tier under [engine] gpu_mem_utilization, and \
             the cache rows, graph bodies and guest programs need the other half. State a \
             smaller max_forward_tokens to choose it, or a larger gpu_mem_utilization.",
            working_at(asked_tokens) >> 20,
            after_weights >> 20,
        );
        boot.budget = budget_at(tokens);
    }
    let budgets = budgets_at(&boot.budget);
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
            if op.starts_with("attention.") {
                streams.entry(region.stream).or_default().push(format!(
                    "r{at}:{}:{:?}",
                    op.trim_start_matches("attention."),
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
        let (at, slots) = compiled.arena.peak();
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
            slots.iter().map(|(_, _, _, bytes)| *bytes).sum::<u64>() >> 20
        );
        for (value, span, offset, bytes) in slots.iter().take(16) {
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
        compiled,
        budgets,
        profile,
    })
}

/// The token and lane budgets fitted to what the card has with the weights
/// resident. What a fire's working set takes at those budgets (the
/// activation arena, the attention workspaces, the decoded-weight tiles, the
/// guests' programs, the score planes and the readout indices) is held to
/// half of the card past the weights, the half `bake` sizes the token budget
/// against before the weights are down, and to what leaves one sequence and
/// the bodies' floor beside it. The tile yields first, held to the widest
/// plane fired at token rows (a plane past it is served by the fused-dequant
/// arm), then the token budget halves toward its floor, then the lanes
/// halve toward one; `Pools::fit_the_card` still refuses past that.
#[allow(clippy::too_many_arguments)]
fn refit(
    boot: &mut Boot<'_>,
    device: &Context,
    compiled: CompiledModel,
    budgets: Budgets,
    profile: &DeviceProfile,
    paging: Paging,
    table: &crate::run::WeightTable,
) -> Result<(CompiledModel, Budgets, Vec<u64>)> {
    let (free, total) = crate::store::device_memory()?;
    let live = crate::device::elastic::budget_bytes(free, total, boot.knobs.gpu_mem_utilization);
    let facts = kv::probe(&boot.trace)?;
    let window = crate::store::window_of(&boot.trace)?;
    let exports = Exports::of(&boot.trace, &compiled)?;
    let out_row = match exports.out {
        Some(out) => kv::width_of(&boot.trace, out)?.saturating_mul(4),
        None => 0,
    };
    let score_values: Vec<poem_ir::ValueId> =
        exports.scores.iter().map(|export| export.value).collect();
    let score_heads = score_heads(&boot.trace, &exports);
    let (asked_lanes, asked_tokens) = (boot.budget.max_lanes, boot.budget.max_tokens);
    let budget_at = |lanes: u32, tokens: u32| {
        let mut budget = boot.budget.clone();
        budget.max_lanes = lanes.min(tokens);
        budget.max_tokens = tokens;
        let mut buckets: Vec<u32> = budget
            .buckets
            .iter()
            .copied()
            .filter(|bucket| *bucket < tokens)
            .collect();
        buckets.push(tokens);
        budget.buckets = crate::api::lattice(buckets, tokens);
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
                &schedule_streams(&boot.trace, &compiled),
            ))
            .saturating_add(tiles)
            .saturating_add(crate::store::program_scratch_reserve(
                lanes.min(device.device().num_sm),
                out_row,
            ))
            .saturating_add(crate::scores::bytes_for(
                &score_values,
                score_heads,
                lanes,
                boot.world,
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
        Ok(if owed { room.min(live / 2) } else { room })
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
        "engine-cuda: [engine] max_forward_tokens {asked_tokens} and max_forward_requests \
         {asked_lanes} are served at {tokens} and {lanes}: at the asked budgets a fire's \
         working set (activation arena, attention workspaces, decoded-weight tiles, guest \
         programs) takes {} MiB of the {} MiB this card has after the weight tier under \
         [engine] gpu_mem_utilization, past the half the cache rows are owed and what one \
         sequence at the declared context and the bodies' floor need. State smaller budgets \
         to choose them, or a larger gpu_mem_utilization.",
        working_at(asked_lanes, asked_tokens, scoped) >> 20,
        live >> 20,
    );
    boot.budget = budget_at(lanes, tokens);
    let budgets = budgets_at(&boot.budget);
    let compiled = poem_compiler::compile_axes(&boot.trace, &budgets, profile)?;
    let planes = decoded_weight_planes(&boot.trace, &compiled, table);
    Ok((compiled, budgets, tiles_of(&planes, scoped)))
}

/// The query heads a score export carries per layer, zero when none is.
fn score_heads(trace: &poem_ir::Trace, exports: &Exports) -> u32 {
    exports
        .scores
        .first()
        .and_then(|export| match &trace.values[export.value.0 as usize].ty {
            poem_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                poem_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                _ => None,
            }),
            poem_ir::Ty::Struct(_) => None,
        })
        .unwrap_or(0)
}

impl Shell {
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let mut boot = boot;
        let Baked {
            mut device,
            compiled,
            budgets,
            profile,
        } = bake(&mut boot)?;
        let (free_before, device_total) = crate::store::device_memory()?;
        let paging = Paging::of(
            boot.page_size,
            boot.context,
            boot.slots,
            u64::from(boot.pages),
        )?
        .windowed(
            crate::store::window_of(&boot.trace)?,
            boot.budget.max_tokens,
        );
        let decode_dense = landing_requests(&boot.trace.facts, &compiled.classes)
            .iter()
            .flatten()
            .any(poem_ir::Request::denoise);
        let decoded_dense = if decode_dense {
            crate::weights::decoded_dense_bytes(&boot.trace)
        } else {
            0
        };
        let accounting = crate::store::admit_the_card(
            boot.knobs.gpu_mem_utilization,
            boot.residency.device_demand(),
            decoded_dense,
            &boot.trace,
            paging,
        )?;

        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB before weights",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
            );
        }
        let mut weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.residency.clone(),
            device.stream(),
            checkpoint::plan::StorageTarget::for_backend(
                checkpoint::types::BackendKind::Cuda,
                boot.world.rank,
                boot.world.size,
            ),
            decode_dense,
            boot.deferred_tier,
        )?;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after weights resident",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
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
        )?;
        let paging = paging.windowed(
            crate::store::window_of(&boot.trace)?,
            boot.budget.max_tokens,
        );
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
                    kernels_cuda::graph::set_conditional(
                        device.ctx(),
                        0,
                        0,
                        0,
                        false,
                        kernels_cuda::graph::Arm::Warm,
                        0,
                    ),
                )?;
            }
            if wants_switch {
                warmed(
                    "switch setter",
                    kernels_cuda::graph::set_switch(
                        device.ctx(),
                        0,
                        0,
                        0,
                        0,
                        kernels_cuda::graph::Arm::Warm,
                        0,
                    ),
                )?;
            }
            crate::device::ctx::sync(device.stream())?;
        }

        let facts = kv::probe(&boot.trace)?;
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;
        crate::window::no_grouped_window_is_also_a_prepare_window(&compiled)?;
        let masked = masked_classes(&boot.trace, &compiled);
        let corrected = corrected_classes(&boot.trace, &compiled);
        let landing = landing_requests(&boot.trace.facts, &compiled.classes);
        let decoding = decoding_of(&landing);
        let feeds = Feeds::of(&boot.trace, &compiled);
        // A wide body is captured from representative lanes and keyed by the
        // whole wide list, so only classes a synthetic lane can stand in for
        // join it: text-stream, non-denoising, port-free.
        let tiers = crate::record::Tiers {
            plain: plain_of(&landing),
            wide: landing
                .iter()
                .enumerate()
                .filter(|(class, requests)| {
                    !requests.is_empty()
                        && requests.iter().all(|request| {
                            !request.denoise()
                                && request.stream() == poem_ir::Stream::Text
                                && request.reading().is_none()
                        })
                        && !feeds
                            .ports
                            .iter()
                            .any(|(_, readers)| readers.contains(*class))
                })
                .map(|(class, _)| class as u32)
                .collect(),
        };
        let media = media_classes(&boot.trace, &compiled);
        let shifted = regions_shifting(&boot.trace, &compiled);
        let lane_shifted = regions_lane_shifting(&boot.trace, &compiled);
        let schedule_readers = regions_launching_schedules(&boot.trace, &compiled);
        crate::voxels::relabel_conv_weights(&device, &boot.trace, weights.table())?;
        weights.rotate(&boot.trace, &compiled)?;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after rotate",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
            );
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after weights",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
            );
        }
        let pools = Pools::reserve(
            device.ordinal(),
            boot.knobs.gpu_mem_utilization,
            &boot.trace,
            paging,
            &facts,
        )?;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after pools",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
            );
        }
        let buffers = Buffers::reserve(&boot.trace, paging, &pools)?;
        let mut pools = pools;
        if let Some(buffers) = &buffers {
            pools.seat_buffered(buffers.slot_bytes(), buffers.map_unit());
        }
        // Committed whole: a fire reading out past one row a lane (a verify
        // reads every row) would otherwise grow it outside the fit.
        let mut arena = Arena::reserve(&compiled.arena, &pools)?;
        arena.ensure(&mut pools, compiled.arena.bytes)?;
        let predicate = crate::store::rs::Predicate::reserve(boot.budget.max_lanes)?;
        let spaces = boot
            .trace
            .caches
            .iter()
            .filter_map(|row| match row {
                poem_ir::CacheRow::Kv { space, .. } => Some(*space as usize + 1),
                poem_ir::CacheRow::State { .. } => None,
            })
            .max()
            .unwrap_or(0);
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
                    embed_taps: declared_width(&boot.trace, poem_ir::RuntimeInput::PatchEmbedRows),
                    embed_weights: declared_width(
                        &boot.trace,
                        poem_ir::RuntimeInput::PatchEmbedWeights,
                    ) > 0,
                })
            })
        });
        let self_cond_taps = u32::try_from(declared_width(
            &boot.trace,
            poem_ir::RuntimeInput::SelfCondRows,
        ))
        .unwrap_or(u32::MAX);
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
                poem_ir::Operation::Layout(poem_ir::Layout::ScatterLiveRows { .. })
            )
        });
        if let Some(value) = feeds.unlanded.first() {
            return Err(Fault::Unbound {
                what: format!(
                    "value {}, a merge over a runtime input this shell cannot land: only a \
                     float port (latents, lane vector, context, axis positions) under a \
                     conjunction of facts is landed in a merged column before the walk",
                    value.0
                ),
            });
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after buffers",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
            );
        }
        let inputs = Inputs::reserve(
            &boot.budget,
            paging,
            spaces,
            &facts,
            compiled.classes.classes.len(),
            compiled.template().len(),
            poem_exec::fire::max_runs(&compiled),
            poem_exec::fire::fragmentable(&compiled),
            device.device(),
            boot.runahead,
            patch_seat,
            mrope_seat,
            u64::from(self_cond_taps),
            &feeds.seats(),
            feeds.selections.len(),
            !masked.is_empty(),
            &schedule_streams(&boot.trace, &compiled),
        )?;

        let exports = Exports::of(&boot.trace, &compiled)?;

        let score_heads = score_heads(&boot.trace, &exports);
        let score_values: Vec<poem_ir::ValueId> =
            exports.scores.iter().map(|export| export.value).collect();
        let scores = crate::scores::Scores::reserve(
            &score_values,
            score_heads,
            boot.budget.max_lanes,
            boot.world,
        )?;

        let airborne = crate::settle::Airborne::new();
        let mut pools = pools;
        pools.watch(airborne.clone());
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB after inputs",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
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
        let mut shell = Shell {
            device,
            accounting,
            trace: boot.trace,
            compiled,
            budget: budgets.tokens.clone(),
            budgets,
            patch_seat,
            mrope_seat,
            self_cond_taps,
            drops_patch_rows,
            towered: compiled_towered,
            patch_fold,
            runahead: boot.runahead,
            voxels,
            weights,
            arena,
            pools,
            buffers,
            rs_scratch: None,
            predicate,
            inputs,
            facts,
            spaces,
            masked,
            feeds,
            adapter_fact,
            corrected,
            decoding,
            tiers,
            landing,
            armed: None,
            media,
            shifted,
            lane_shifted,
            schedule_readers,
            adapters: crate::blob::Adapters::new(adapter_seats),
            scores,
            held: vec![0; boot.slots as usize],
            readout_rows,
            exports,
            graphs: boot.graphs,
            copies: boot.knobs.copies,
            pad: boot.knobs.pad(),
            golden_arm: Golden::Off,
            bodies: boot.knobs.bodies(),
            bodies_mem: (boot.knobs.bodies_mem() as usize).saturating_mul(1 << 20),
            decoded_tiles: Vec::new(),
            arming: false,
            armed_body: None,
            segments: std::collections::HashMap::new(),
            windows_memo: Vec::new(),
            last: FireCost::default(),
            cache: {
                let mut cache = GraphCache::new();
                cache.watch(airborne.clone());
                cache
            },
            programs: ProgramPlane::new(crate::program::compile::Disk::rooted(
                boot.cache_dir
                    .map(|dir| dir.join(kernels_cuda::disk::CUBINS)),
            )),
            settlement: crate::settle::Settlement::open(boot.runahead.staging_depth())?,
            airborne,
            owed: None,
            guest_landed: crate::device::graph::Event::new()?,
        };
        if shell.weights.rotating() && shell.graphs.records() {
            eprintln!(
                "engine-cuda: [engine] graphs is on but this load armed a dense rotor, \
                 so every fire walks eagerly and nothing is recorded — a rotation's \
                 backpressure is a host cursor and a replayed graph has no walk{}",
                if shell.bodies {
                    "; the bodies path's load-time arming is skipped for the same reason, \
                     since every rung it climbed would execute its warm fires and capture \
                     nothing"
                } else {
                    ""
                }
            );
        }
        if !shell.graphs.records() {
            eprintln!(
                "engine-cuda: [engine] graphs is {}, a diagnostic mode — every fire \
                 walks eagerly (~470 kernel launches of host time per decode step) \
                 with nothing captured; leave the key unstated to serve bodies",
                match shell.graphs {
                    Graphs::Off => "off",
                    Graphs::Shaped => "shaped",
                    Graphs::On => "on",
                }
            );
        } else if !shell.bodies {
            eprintln!(
                "engine-cuda: [engine] bodies is off under [engine] graphs = on, a \
                 diagnostic arm — bodies are the only recorded path, so every fire walks \
                 eagerly (~470 kernel launches of host time per decode step) with nothing \
                 captured; leave the key unstated to serve them"
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
        let flight = shell.device.device().num_sm;
        shell.programs.set_flight(flight);
        let programs = crate::store::program_scratch_reserve(
            boot.budget.max_lanes.min(flight),
            shell.out_width().map_or(0, |width| width.saturating_mul(4)),
        );
        // A plane stored as codes and scales is decoded to bf16 before a
        // prefill's dense gemm, into one tile a stream. The tile is taken
        // now, at the widest plane the lane fit left each stream, so the
        // fit sees it resident and no fire — a capture least of all — has to
        // grow it: a load whose bodies are held to nothing arms no body at
        // load, and its first fire is a capture.
        shell.decoded_tiles = tiles
            .iter()
            .map(|plane| crate::store::decoded_weight_reserve(*plane, 1))
            .collect();
        let mut decoded = 0u64;
        for (stream, plane) in tiles.iter().enumerate() {
            let tile = crate::store::decoded_weight_reserve(*plane, 1);
            let ctx = match stream {
                0 => Some(shell.device.ctx()),
                n => shell.device.side_ctx().get(n - 1).copied(),
            };
            let Some(ctx) = ctx else {
                continue;
            };
            kernels_cuda::linear::quant::warm_decoded_weight(ctx, tile).map_err(|why| {
                Fault::Residency(format!(
                    "the card does not hold this deployment: the decoded-weight tile stream \
                     {stream} decodes its widest plane into, {tile} bytes, would not allocate \
                     beside the weights, activations and inputs ({why}). Lower `[model] \
                     device_weight_budget`, raise `[engine] gpu_mem_utilization`, or free what \
                     else holds the device."
                ))
            })?;
            decoded = decoded.saturating_add(tile);
        }
        let held_for_fires = programs;
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] held for fires: guest programs {} MiB at {} lanes; decoded-weight \
                 tiles taken at load: {} MiB over {} stream(s) ({})",
                programs >> 20,
                boot.budget.max_lanes,
                decoded >> 20,
                tiles.len(),
                tiles
                    .iter()
                    .map(|plane| format!("{}", plane >> 20))
                    .collect::<Vec<String>>()
                    .join("/"),
            );
        }
        let footprint = |shell: &Shell| Footprint {
            total: device_total,
            before: device_total.saturating_sub(free_before),
            weights: shell.weights.bytes(),
            arena: shell.arena.bytes(),
            inputs: shell.inputs.bytes(),
            buffers: shell.buffer_bytes(),
            readout: shell.readout_rows.bytes() as u64,
            scores: shell
                .scores
                .as_ref()
                .map_or(0, crate::scores::Scores::bytes),
            cache_rows: shell.pools.committed_bytes(),
            bodies: shell.cache.body_stats().census.bytes as u64,
            scratch: kernels_cuda::jit::Slabs::census()
                .iter()
                .map(|(_, bytes)| *bytes as u64)
                .sum(),
        };
        let bodies = if shell.records_bodies() {
            shell.bodies_mem as u64
        } else {
            0
        };
        let resident = footprint(&shell);
        let fitted = shell
            .pools
            .fit_the_card(bodies, held_for_fires, &resident)?;
        if fitted.bodies < bodies {
            eprintln!(
                "engine-cuda: [engine] bodies_mem {} MiB is held down to {} MiB on this card: \
                 with the weights, activations and inputs resident, one sequence at the declared \
                 context and the guests' programs seated, {} MiB is left for the graph bodies \
                 and the cache rows together, and the bodies take at most a quarter of it. \
                 State a smaller bodies_mem to choose it.",
                bodies >> 20,
                fitted.bodies >> 20,
                fitted.spare >> 20,
            );
            shell.bodies_mem = usize::try_from(fitted.bodies).unwrap_or(usize::MAX);
        }
        if (fitted.slots as usize) < shell.held.len() {
            eprintln!(
                "engine-cuda: [engine] max_state_slots {} is served at {} on this card: the \
                 recurrent slab is declared for every slot, and at {} it left no room for one \
                 sequence's kv pages beside it, so the slots were cut to what leaves the pages \
                 half of the room past one sequence; the pool seats {} sequences at the \
                 declared context. State a smaller max_state_slots to choose it.",
                shell.held.len(),
                fitted.slots,
                shell.held.len(),
                fitted.fit / u64::from(shell.pools.paging().pages_per_slot),
            );
            shell.held.truncate(fitted.slots as usize);
        }
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB before arming",
                crate::device::nodes::free_bytes().unwrap_or(0) >> 20
            );
        }
        shell.arm_bodies()?;
        let resident = footprint(&shell);
        let fitted = shell.pools.fit_the_card(0, held_for_fires, &resident)?;
        if fitted.fit != fitted.asked {
            let paging = shell.pools.paging();
            eprintln!(
                "engine-cuda: the pool was declared {} pages ({} MiB), past the {} MiB this card \
                 hands out for the cache rows under [engine] gpu_mem_utilization once {} MiB is \
                 held for the programs guests register at {} lanes{} ({resident}); sized to {} \
                 pages ({} MiB), {} sequences at the declared context. State [engine] \
                 max_total_pages to choose the count.",
                fitted.asked,
                shell.pools.declared_at(fitted.asked) >> 20,
                fitted.room >> 20,
                programs >> 20,
                boot.budget.max_lanes,
                if fitted.held.saturating_sub(fitted.bodies) < held_for_fires {
                    format!(
                        ", held down to {} MiB so one sequence and the bodies' floor seat",
                        fitted.held.saturating_sub(fitted.bodies) >> 20
                    )
                } else {
                    String::new()
                },
                fitted.fit,
                shell.pools.declared_bytes() >> 20,
                fitted.fit / u64::from(paging.pages_per_slot),
            );
        }
        if boot.world.rank != 0 {
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

/// What holds the device when the cache rows come to be declared, in bytes:
/// what was in use before this load began (other processes, and this one's
/// context) and each thing the load has put down since. The elastic budget
/// is what `[engine] gpu_mem_utilization` leaves after all of it.
struct Footprint {
    total: u64,
    before: u64,
    weights: u64,
    arena: u64,
    inputs: u64,
    buffers: u64,
    readout: u64,
    scores: u64,
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
             readout rows {}, attention scores {}, cache rows {}, graph bodies {}, kernel \
             scratch {}",
            self.total,
            self.before,
            self.weights,
            self.arena,
            self.inputs,
            self.buffers,
            self.readout,
            self.scores,
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
                poem_ir::Operation::Linear(_) => {
                    outputs.clear();
                    node.op.outputs(&mut outputs);
                    outputs.iter().any(|id| token_rows(*id))
                }
                poem_ir::Operation::Attention(_) => true,
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
        }
    }
    planes
}

fn declared_width(trace: &poem_ir::Trace, which: poem_ir::RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (poem_ir::Def::Input(named), poem_ir::Ty::Tensor { shape, .. }) =
                (&decl.def, &decl.ty)
            else {
                return None;
            };
            if *named != which {
                return None;
            }
            Some(
                shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        poem_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product(),
            )
        })
        .unwrap_or(0)
}

fn patch_fold(trace: &poem_ir::Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Layout(
                poem_ir::Layout::MergeRows { side, .. } | poem_ir::Layout::PoolRows { side, .. },
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
    pub(super) compiled: CompiledModel,
    pub(super) budgets: Budgets,
    pub(super) profile: DeviceProfile,
}
