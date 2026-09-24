use model_compiler::{Budget, Budgets, CompiledModel, DeviceProfile};

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
    let device = Context::bind(boot.ordinal, boot.comm)?;

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
    boot.trace = model_ir::fuse::residual_norm(boot.trace.clone());
    if boot.knobs.diagnostics.fuse_chains {
        boot.trace = model_ir::fuse::residual_chains(boot.trace.clone());
        boot.trace = model_ir::fuse::gemm_epilogues(boot.trace.clone());
        boot.trace = model_ir::fuse::modulation(boot.trace.clone());
        boot.trace = model_ir::fuse::q_norm_rope(boot.trace.clone());
        boot.trace = model_ir::fuse::embed_select(boot.trace.clone());
    }
    if boot.knobs.diagnostics.trace_census {
        let mut census: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for node in &boot.trace.nodes {
            *census
                .entry(model_ir::Operands::name(&node.op))
                .or_insert(0) += 1;
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
    let working_at = |tokens: u32| -> u64 {
        let budget = budget_at(tokens);
        let Ok(compiled) =
            model_compiler::compile_axes(&boot.trace, &budgets_at(&budget), &profile)
        else {
            return u64::MAX;
        };
        compiled
            .arena
            .prefix_for(u64::from(budget.max_lanes))
            .saturating_add(crate::inputs::attention_workspace_bytes(
                &budget,
                &facts,
                &device.device(),
                &crate::exports::schedule_streams(&boot.trace, &compiled),
            ))
    };
    let tokens = crate::store::tokens_within(asked_tokens, floor, after_weights / 2, working_at);
    if tokens != asked_tokens {
        eprintln!(
            "engine-cuda: [engine] max_forward_tokens {asked_tokens} is served at {tokens}: at \
             {asked_tokens} the activation arena and attention workspaces take {} MiB of the \
             {} MiB this card has after the weight tier under [engine] gpu_mem_utilization, and \
             the cache rows, graph bodies and guest programs need the other half. State a \
             smaller max_forward_tokens to choose it, or a larger gpu_mem_utilization.",
            working_at(asked_tokens) >> 20,
            after_weights >> 20,
        );
        boot.budget = budget_at(tokens);
    }
    let budgets = budgets_at(&boot.budget);
    let compiled = model_compiler::compile_axes(&boot.trace, &budgets, &profile)?;
    if boot.knobs.diagnostics.arm_trace {
        let mut streams: std::collections::BTreeMap<u32, Vec<String>> =
            std::collections::BTreeMap::new();
        for (at, region) in compiled.template().iter().enumerate() {
            let op = boot
                .trace
                .nodes
                .get(region.nodes.start as usize)
                .map_or("?", |node| model_ir::Operands::name(&node.op));
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
                model_ir::Def::Op(node) => {
                    model_ir::Operands::name(&boot.trace.nodes[*node as usize].op).to_string()
                }
                other => format!("{other:?}").chars().take(20).collect(),
            };
            let rows = match &compiled.arena.placements[value.0 as usize] {
                model_compiler::arena::Placement::Arena {
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
    })
}

impl Shell {
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let mut boot = boot;
        let Baked {
            mut device,
            compiled,
            budgets,
        } = bake(&mut boot)?;
        let (free_before, device_total) = crate::store::device_memory()?;
        device.open_lanes(
            compiled.streams.streams.saturating_sub(1),
            compiled.streams.events,
        )?;
        let mut wants_if = false;
        let mut wants_switch = false;
        for region in &compiled.regions {
            match region.lowering {
                model_compiler::Lowering::AlwaysLaunch => {}
                model_compiler::Lowering::If => wants_if = true,
                model_compiler::Lowering::Switch { .. } => wants_switch = true,
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
        let landing = landing_requests(boot.classify, &compiled.classes);
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
                                && request.stream() == model_ir::Stream::Text
                                && request.reading() == 0
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
        let paging = Paging::of(
            boot.page_size,
            boot.context,
            boot.slots,
            u64::from(boot.pages),
        )?;
        let decode_dense = landing.iter().flatten().any(model_ir::Request::denoise);
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
        let unbuffered = buffers
            .is_none()
            .then(|| crate::store::rs::unbuffered(&boot.trace))
            .flatten();
        let mut pools = pools;
        let mut arena = Arena::reserve(&compiled.arena, &pools)?;
        arena.ensure(
            &mut pools,
            compiled.arena.prefix_for(u64::from(boot.budget.max_lanes)),
        )?;
        let predicate = crate::store::rs::Predicate::reserve(boot.budget.max_lanes)?;
        let spaces = boot
            .trace
            .caches
            .iter()
            .filter_map(|row| match row {
                model_ir::CacheRow::Kv { space, .. } => Some(*space as usize + 1),
                model_ir::CacheRow::State { .. } => None,
            })
            .max()
            .unwrap_or(0);
        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (
                    model_ir::Def::Input(model_ir::RuntimeInput::Patches),
                    model_ir::Ty::Tensor { shape, dtype },
                ) = (&decl.def, &decl.ty)
                else {
                    return None;
                };
                let width: u64 = shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        model_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product();
                let element = model_compiler::arena::elem_bytes(*dtype).unwrap_or(0);
                Some(crate::inputs::PatchSeat {
                    rows: u64::from(ladder.max_patches),
                    row_bytes: width * element,
                    images: u64::from(ladder.max_images),
                    dtype: *dtype,
                    embed_taps: declared_width(&boot.trace, model_ir::RuntimeInput::PatchEmbedRows),
                    embed_weights: declared_width(
                        &boot.trace,
                        model_ir::RuntimeInput::PatchEmbedWeights,
                    ) > 0,
                })
            })
        });
        let self_cond_taps = u32::try_from(declared_width(
            &boot.trace,
            model_ir::RuntimeInput::SelfCondRows,
        ))
        .unwrap_or(u32::MAX);
        let mrope_seat = boot.trace.values.iter().any(|decl| {
            matches!(
                decl.def,
                model_ir::Def::Input(model_ir::RuntimeInput::MropePositions)
            )
        });
        let patch_fold = patch_fold(&boot.trace);
        let voxels = match boot.voxels.as_ref() {
            Some(ladder) if compiled.order_for(model_ir::RowAxis::Voxels).is_some() => Some(
                crate::voxels::Store::reserve(crate::voxels::Seat::of(&boot.trace, ladder))?,
            ),
            _ => None,
        };
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(
                node.op,
                model_ir::Operation::Layout(model_ir::Layout::ScatterLiveRows { .. })
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
            model_exec::fire::max_runs(&compiled),
            model_exec::fire::fragmentable(&compiled),
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

        let score_heads = exports
            .scores
            .first()
            .and_then(
                |export| match &boot.trace.values[export.value.0 as usize].ty {
                    model_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                        model_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                        _ => None,
                    }),
                    model_ir::Ty::Struct(_) => None,
                },
            )
            .unwrap_or(0);
        let score_values: Vec<model_ir::ValueId> =
            exports.scores.iter().map(|export| export.value).collect();
        let scores =
            crate::scores::Scores::reserve(&score_values, score_heads, boot.budget.max_lanes)?;

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
        let compiled_towered = compiled.order_for(model_ir::RowAxis::Patches).is_some();
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
            unbuffered,
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
            classify: boot.classify,
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
        let programs = crate::store::program_scratch_reserve(
            boot.budget.max_lanes,
            shell.out_width().map_or(0, |width| width.saturating_mul(4)),
        );
        let decoded = crate::store::decoded_weight_reserve(
            widest_affine_plane(&shell.trace, shell.weights.table()),
            shell.compiled.streams.streams,
        );
        let held_for_fires = programs.saturating_add(decoded);
        if boot.knobs.diagnostics.arm_trace {
            eprintln!(
                "[arm-trace] held for fires: guest programs {} MiB at {} lanes, decoded-weight \
                 tiles {} MiB ({} MiB widest plane on {} stream(s))",
                programs >> 20,
                boot.budget.max_lanes,
                decoded >> 20,
                widest_affine_plane(&shell.trace, shell.weights.table()) >> 20,
                shell.compiled.streams.streams,
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
                 sequence's kv pages beside it; the pool seats {} sequences at the declared \
                 context. State a smaller max_state_slots to choose it.",
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
                 held for the programs guests register at {} lanes and {} MiB for the \
                 decoded-weight tiles on {} stream(s){} ({resident}); sized to {} pages ({} \
                 MiB), {} sequences at the declared context. State [engine] max_total_pages \
                 to choose the count.",
                fitted.asked,
                shell.pools.declared_at(fitted.asked) >> 20,
                fitted.room >> 20,
                programs >> 20,
                boot.budget.max_lanes,
                decoded >> 20,
                shell.compiled.streams.streams,
                if fitted.held < held_for_fires {
                    format!(
                        ", held down to {} MiB together so one sequence seats",
                        fitted.held >> 20
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

/// The widest plane this load decodes to bf16 before a dense gemm: every
/// weight the table seats as codes and scales, at the `[n, k]` the trace
/// declares for it, two bytes an element.
fn widest_affine_plane(trace: &model_ir::Trace, table: &crate::run::WeightTable) -> u64 {
    trace
        .values
        .iter()
        .filter_map(|decl| {
            let model_ir::Def::Weight(w) = &decl.def else {
                return None;
            };
            let model_ir::Ty::Tensor { shape, .. } = &decl.ty else {
                return None;
            };
            table
                .0
                .get(*w as usize)
                .and_then(Option::as_ref)
                .filter(|row| matches!(row, crate::run::WeightRow::Planes { .. }))?;
            Some(
                shape
                    .iter()
                    .map(|dim| match dim {
                        model_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .fold(2u64, u64::saturating_mul),
            )
        })
        .max()
        .unwrap_or(0)
}

fn declared_width(trace: &model_ir::Trace, which: model_ir::RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (model_ir::Def::Input(named), model_ir::Ty::Tensor { shape, .. }) =
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
                        model_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product(),
            )
        })
        .unwrap_or(0)
}

fn patch_fold(trace: &model_ir::Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            model_ir::Operation::Layout(
                model_ir::Layout::MergeRows { side, .. } | model_ir::Layout::PoolRows { side, .. },
            ) => Some(side.saturating_mul(*side)),
            _ => None,
        })
        .fold(1u32, |fold, side| fold.saturating_mul(side))
        .max(1)
}

fn adapter_fact(classes: &model_ir::ClassTable, corrected: &model_ir::ClassSet) -> Option<u32> {
    classes.adapter_fact(corrected)
}

pub(super) struct Baked {
    pub(super) device: Context,
    pub(super) compiled: CompiledModel,
    pub(super) budgets: Budgets,
}
