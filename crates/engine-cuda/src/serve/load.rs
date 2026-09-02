//! Boot: bind the device, bake the artifact, land the checkpoint, reserve, arm.

use model_compiler::{Budgets, CompiledModel, DeviceProfile};

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::{
    Exports, corrected_classes, decoding_of, landing_requests, masked_classes, media_classes,
    regions_lane_shifting, regions_shifting,
};
use crate::inputs::Inputs;
use crate::program::Plane as ProgramPlane;
use crate::record::Bodies as GraphCache;
use crate::store::Pools;
use crate::store::kv::{self, Paging};
use crate::store::rs::Buffers;
use crate::weights::Weights;

use super::{Boot, FireCost, Golden, Graphs, Shell};

/// The cold prefix both doors run: bind the device, settle the compiler's
/// inputs, bake the artifact. `boot` is widened in place (its lattice).
pub(super) fn bake(boot: &mut Boot<'_>) -> Result<Baked> {
    let device = Context::bind(boot.ordinal)?;

    // One-shot: whichever load arrives first states the kernel cache root.
    kernels_cuda::disk::install(boot.cache_dir);

    // The shape lattice is a compiler input, filled by the load door's policy.
    boot.budget.buckets = crate::api::lattice(
        std::mem::take(&mut boot.budget.buckets),
        boot.budget.max_tokens,
    );

    let mut profile = boot.profile.take().unwrap_or(DeviceProfile {
        sms: device.device().num_sm,
        ..DeviceProfile::default()
    });
    // P6's off arm bakes a different artifact rather than declining a graph.
    if let Some(streams) = boot.knobs.side_streams {
        profile.side_streams = streams;
    }
    profile.exclusive = crate::EXCLUSIVE
        .iter()
        .map(|op| (*op).to_string())
        .collect();
    // The grouped arm names the same ops or none; the list is never the caller's.
    profile.grouped = if boot.knobs.grouped {
        crate::GROUPED.iter().map(|op| (*op).to_string()).collect()
    } else {
        Vec::new()
    };
    let budgets = Budgets {
        tokens: boot.budget.clone(),
        patches: boot.patches.clone(),
    };
    let compiled = model_compiler::compile_axes(&boot.trace, &budgets, &profile)?;
    Ok(Baked {
        device,
        compiled,
        budgets,
    })
}

impl Shell {
    /// Boot: bind, bake, land, reserve, arm.
    ///
    /// # Errors
    ///
    /// [`Fault::Bake`] for a plan these budgets do not admit, [`Fault::Load`]
    /// for a checkpoint the contract does not fit, [`Fault::Device`] for the
    /// residency, [`Fault::Unbound`] for a plan naming a seat this shell does
    /// not bind, [`Fault::Golden`] for an armed body that answers other than
    /// its walk.
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let mut boot = boot;
        let Baked {
            mut device,
            compiled,
            budgets,
        } = bake(&mut boot)?;
        device.open_lanes(
            compiled.streams.streams.saturating_sub(1),
            compiled.streams.events,
        )?;
        // A conditional body needs its own stream, and its setter's module
        // must be resident before any capture: warm each spelling the artifact
        // baked, once, here.
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
        // The op-vocabulary scans, read once off the bake.
        let masked = masked_classes(&boot.trace, &compiled);
        let corrected = corrected_classes(&boot.trace, &compiled);
        let landing = landing_requests(boot.classify, &compiled.classes);
        let decoding = decoding_of(&landing);
        let media = media_classes(&boot.trace, &compiled);
        let shifted = regions_shifting(&boot.trace, &compiled);
        let lane_shifted = regions_lane_shifting(&boot.trace, &compiled);
        let paging = Paging::of(boot.page_size, boot.context, boot.slots)?;
        // The accounting sentence refuses ahead of every allocation.
        let accounting = crate::store::admit_the_card(
            boot.knobs.gpu_mem_utilization,
            boot.residency.device_demand(),
            &boot.trace,
            paging,
        )?;

        let mut weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.residency.clone(),
            device.stream(),
        )?;
        weights.rotate(&boot.trace, &compiled)?;
        let arena = Arena::reserve(&compiled.arena)?;
        let pools = Pools::reserve(
            device.ordinal(),
            boot.knobs.gpu_mem_utilization,
            &boot.trace,
            paging,
            &facts,
        )?;
        let buffers = Buffers::reserve(&boot.trace, paging)?;
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
        // The patch seat: the deployment's ceilings, the plan's own row width.
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
        let mrope_seat = boot.trace.values.iter().any(|decl| {
            matches!(
                decl.def,
                model_ir::Def::Input(model_ir::RuntimeInput::MropePositions)
            )
        });
        let patch_fold = patch_fold(&boot.trace);
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(
                node.op,
                model_ir::Operation::Layout(model_ir::Layout::ScatterLiveRows { .. })
            )
        });
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
        )?;

        let exports = Exports::of(&boot.trace, &compiled)?;

        // The score slab, carved off the `attn.scores` exports the text wrote.
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
        // The readout's row-pointer tables, at the ceiling by construction.
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
            drops_patch_rows,
            towered: compiled_towered,
            patch_fold,
            weights,
            arena,
            pools,
            buffers,
            predicate,
            inputs,
            facts,
            spaces,
            masked,
            adapter_fact,
            corrected,
            decoding,
            landing,
            classify: boot.classify,
            armed: None,
            media,
            shifted,
            lane_shifted,
            adapters: crate::blob::Adapters::new(adapter_seats),
            scores,
            held: vec![0; boot.slots as usize],
            readout_rows,
            exports,
            graphs: boot.graphs,
            copies: boot.knobs.copies,
            // The three bodies words and the pad, derived from one `Recording`.
            pad: boot.knobs.pad(),
            golden: boot.knobs.golden(),
            golden_arm: Golden::Off,
            bodies: boot.knobs.bodies(),
            // Megabytes to bytes, once, at the seam the boot document crosses.
            bodies_mem: (boot.knobs.bodies_mem() as usize).saturating_mul(1 << 20),
            arming: false,
            armed_body: None,
            segments: std::collections::HashMap::new(),
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
        // A rotating load never records; say so once, at load.
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
        // The diagnostic arms print one line at load.
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
        // The arming pass is the last thing the load does; only the golden can fail it.
        shell.arm_bodies()?;
        Ok(shell)
    }
}

/// How wide a runtime input the plan declares is — the product of every dim
/// past the leading row one, or `0` when no value of the trace names it.
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

/// How many patch rows this plan folds into one, or `1` for a plan that folds nothing.
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

/// Which bit of a fact word decides the correction window, or `None` when no
/// single bit does (none qualifies, or two do).
fn adapter_fact(classes: &model_ir::ClassTable, corrected: &model_ir::ClassSet) -> Option<u32> {
    if corrected.is_empty() {
        return None;
    }
    let mut found = None;
    for bit in 0..u64::BITS {
        if classes.mask & (1u64 << bit) == 0 {
            continue;
        }
        let decides = classes.classes.iter().enumerate().all(|(at, class)| {
            let runs = corrected.contains(at);
            class
                .words
                .iter()
                .all(|word| ((word >> bit) & 1 == 1) == runs)
        });
        if decides {
            if found.is_some() {
                return None;
            }
            found = Some(bit);
        }
    }
    found
}

/// What [`bake`] answers.
pub(super) struct Baked {
    pub(super) device: Context,
    pub(super) compiled: CompiledModel,
    pub(super) budgets: Budgets,
}
