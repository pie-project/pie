//! The card's fire top, derived at `pie tune cuda --rates`: the widest rung
//! of the lattice (`crate::api::rung_lattice`) whose activation arena alone
//! fits the card's memory beside the smallest model the catalog serves on
//! CUDA (the least weight bytes of its non-miniature deployments: a
//! miniature is a stated catalog fact, a test's prefix of a whole model) and
//! one sequence of it at the context a load seats when the model states
//! none. Nothing narrower binds a fire: its elementwise kernels count and
//! index their planes as `i64`, and its other kernels lay rows on the grid's
//! x axis. No wider fire can lay on the card for any model, so the matmul
//! grid is timed to it, a load's candidates run to it, and the estimate
//! picks each model's top below it.

use profile::Profile;

/// What the fire top was derived from, stated in the profile's
/// `[provenance]` beside the grid timed to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireTop {
    /// The widest rung whose arena fits.
    pub tokens: u32,
    /// The smallest non-miniature CUDA deployment, the arena compiled.
    pub model: String,
    /// Its weight bytes.
    pub weights: u64,
    /// One sequence of it at [`CONTEXT`].
    pub sequence: u64,
    /// The card's memory.
    pub memory: u64,
}

/// The context one sequence is held at: the one a load seats where neither
/// the model nor the operator states one (`engine::Budgets`' default).
fn context() -> u32 {
    engine::Budgets::default().max_context
}

/// The lanes a fire carries, as a served CUDA load is asked them.
const LANES: u32 = 512;

/// The bytes of `trace`'s weights: each parameter's elements at its dtype's
/// bits.
fn weight_bytes(trace: &poem_ir::Trace) -> u64 {
    trace
        .params
        .iter()
        .map(|p| (p.shape.iter().product::<u64>() * p.dtype.bits()).div_ceil(8))
        .sum()
}

/// The fire top on a card of `memory` bytes and `sms` SMs under `chip`.
pub fn derive(chip: &Profile, sms: u32, memory: u64) -> Result<FireTop, String> {
    use poem::Platform;
    let (weights, row, trace) = poem_compiler::catalog::deployments()
        .filter(|d| !d.model.mini && d.check(Platform::Cuda).is_ok())
        .map(|d| {
            let trace = d.trace(Platform::Cuda);
            (weight_bytes(&trace), d, trace)
        })
        .min_by_key(|(bytes, d, _)| (*bytes, d.name.clone()))
        .ok_or("the catalog lists no deployment CUDA serves")?;
    let paging = crate::store::kv::Paging::of(16, context(), 1, 1).map_err(|e| e.to_string())?;
    let sequence = crate::store::least_sequence_bytes(&trace, paging).map_err(|e| e.to_string())?;
    let engine = crate::chip::Engine::of(chip);
    let grouped: &[&str] = if engine.grouped { &crate::GROUPED } else { &[] };
    let profile = poem_compiler::DeviceProfile::of(chip, sms, &crate::EXCLUSIVE, grouped);
    let trace =
        poem_compiler::fuse::fuse(trace, poem_compiler::fuse::flags::on(chip, &crate::FUSIONS));
    // The arena at `rows`: the larger of the in-order and forked carves,
    // without their score planes, as a load's rung weighs it.
    let arena = |rows: u32| -> Option<u64> {
        let mut budget = poem_compiler::Budget::new(LANES.min(rows), rows);
        budget.buckets = vec![rows];
        let compiled =
            poem_compiler::compile_axes(&trace, &poem_compiler::Budgets::of(budget), &profile)
                .ok()?;
        Some(
            compiled.arena.scores_from(&trace).max(
                compiled
                    .arena_forked
                    .as_ref()
                    .map_or(0, |forked| forked.scores_from(&trace)),
            ),
        )
    };
    let beside = weights.saturating_add(sequence);
    // Every point of the lattice a `u32` of rows reaches, ascending, up to
    // the first whose arena does not fit: an arena grows with its rows.
    let mut lattice = crate::api::rung_lattice(&engine, u32::MAX);
    lattice.pop();
    let mut top = None;
    for rows in lattice {
        match arena(rows) {
            Some(span) if span.saturating_add(beside) <= memory => top = Some(rows),
            _ => break,
        }
    }
    let tokens = top.ok_or_else(|| {
        format!(
            "no rung of {} fits this card's {} MiB beside its {} MiB of weights",
            row.name,
            memory >> 20,
            weights >> 20
        )
    })?;
    Ok(FireTop {
        tokens,
        model: row.name.clone(),
        weights,
        sequence,
        memory,
    })
}
