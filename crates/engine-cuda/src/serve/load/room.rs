//! The room a load has on the card: where its host copies lie, the line it
//! states, what the driver holds beside it, the refit of its rungs, and the
//! ledger and footprint of what it holds.

#[allow(clippy::wildcard_imports)]
use super::*;

/// Where the weight cache's copy of a streamable weight lies off the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HostCopy {
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
pub(super) fn host_copy(boot: &Boot<'_>, params: &[usize]) -> Result<HostCopy> {
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
        tracing::info!(
            "{} MiB of weights the pool will hold are copied to the host at \
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
pub(super) fn room_line(
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
pub(super) fn held_beside(boot: &Boot<'_>, room: u64) -> Result<u64> {
    Ok(room - beside(boot, room)?.0)
}

/// What [`held_beside`] holds out of `room`, and the banks it names, as a
/// refusal of the room left says them: nothing where the banks exceed it.
pub(super) fn beside(boot: &Boot<'_>, room: u64) -> Result<(u64, String)> {
    let bytes = crate::weights::laid_bytes(&boot.trace)?;
    let cached: u64 = boot
        .residency
        .cached_experts()
        .iter()
        .map(|&param| bytes[param].next_multiple_of(crate::weights::ALIGN))
        .sum();
    if cached >= room || cached == 0 {
        return Ok((0, String::new()));
    }
    let names: Vec<String> = boot
        .residency
        .cached_experts()
        .iter()
        .filter_map(|&param| boot.trace.params.get(param))
        .map(|param| format!("`{}`", param.name))
        .collect();
    Ok((
        cached,
        format!(
            "; the weight cache seats the routed bank(s) {} ({} MiB) beside the fires, out of \
             that room",
            names.join(", "),
            cached >> 20
        ),
    ))
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
pub(super) fn refit(
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
                tracing::info!(
                    "the decoded-weight tile on stream {stream} is held to {} MiB, \
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
    tracing::info!(
        "a fire of {} tokens and {asked_lanes} lanes is served at {tokens} \
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
pub(super) fn widened(
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

pub(super) fn footprint(shell: &Shell, device_total: u64, free_before: u64) -> Footprint {
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

/// The device's memory as this load holds it ([`engine::memory`]): each
/// owner under its class, over what the device has for this process (all of
/// it but what others held before the load began).
pub(super) fn ledger(shell: &Shell, device_total: u64, free_before: u64) -> engine::memory::Ledger {
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
    // Each run staged in the activation's span past the workspaces, under
    // its own name: what a flat range's span spare leaves out is entered.
    let [inputs, readouts, voxels, predicates, experts, arenas, scores, rings] =
        shell.staged_runs();
    for (name, bytes) in [
        ("inputs", inputs),
        ("readout rows", readouts),
        ("voxels", voxels),
        ("predicates", predicates),
        ("staged experts", experts),
        ("lane arenas", arenas),
        ("capture table", scores),
        ("ring runs", rings),
    ] {
        ledger.enter(Class::Fire, name, bytes);
    }
    ledger.enter(Class::Outside, "graph bodies", held.bodies);
    ledger.enter(Class::Outside, "before the load", held.before);
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
pub(super) struct Footprint {
    pub(super) total: u64,
    pub(super) before: u64,
    pub(super) weights: u64,
    pub(super) arena: u64,
    pub(super) inputs: u64,
    pub(super) buffers: u64,
    pub(super) readout: u64,
    /// The host-resident experts' staging (`experts::staging`).
    pub(super) experts: u64,
    pub(super) cache_rows: u64,
    pub(super) bodies: u64,
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
pub(super) struct DecodedPlanes {
    pub(super) widest: u64,
    pub(super) at_tokens: u64,
}

/// Per stream, the planes the linear and attention nodes launched on it
/// decode to bf16 before a dense gemm, at the `[n, k]` the trace declares,
/// two bytes an element; an embedding gathers from the codes and a repacked
/// plane has its own kernels. A linear's plane is fired at the rows its
/// output lands. Index 0 is the main stream, `n` the `n`th side stream.
pub(super) fn decoded_weight_planes(
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
