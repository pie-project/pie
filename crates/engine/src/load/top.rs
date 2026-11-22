//! What every engine weighs its top rung by ([`super::top_rung`]): the
//! model's counts ([`Counts`], from the trace and what the engine alone
//! knows of where its params lie), the estimate over the device's rates
//! (the profile's `[rates]`), the one refusal of a device without them, and
//! the load's one line on the rung chosen. Each engine offers its own rungs
//! (their spans, launches, padded rows and why it cannot lay one).

use std::fmt;

use poem_ir::{Def, Dim, Operation, Trace, Ty};

use super::rung::{FireEstimate, GemmShape, NoRung, RungFacts, TopRung, TopWhy};

/// What a fire of the model reads and computes: the estimate's counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counts {
    /// The weight bytes every fire reads: the planes the device keeps, less
    /// the token tables (a gather reads a row of them a token) and the
    /// routed banks.
    pub weight_bytes: u64,
    /// The bytes of every routed bank.
    pub expert_bytes: u64,
    /// `(k, E)` of the routed banks: the experts a row picks, of how many.
    pub experts: Option<(u32, u32)>,
    /// The matmuls a row takes, by shape: the planes of `n` columns at depth
    /// `k` merged (their count, flops and bytes), dense and routed apart, the
    /// output head's aside. Ordered by `(n, k)`.
    pub gemm_shapes: Vec<GemmShape>,
    /// The output head's bytes on the device, read every fire by the
    /// readout alone (not in `weight_bytes`).
    pub head_bytes: u64,
    /// The output head's flops a sampled row: two a head element. A fire
    /// reads out once a sampled row, not once a row; the head's bytes are
    /// `head_bytes`.
    pub readout_flops: u64,
    /// The output head's depth (the model's hidden width).
    pub readout_depth: u32,
    /// The flops a row takes a key it attends, summed over the layers: four
    /// a query element (the scores' product and the values').
    pub attn_flops_per_row_key: u64,
    /// The kv bytes a key read takes, summed over the kv rows attentions
    /// read: every plane of the row at its element's size.
    pub kv_bytes_per_key: u64,
    /// The widest head the trace's attentions read (0 with none): the width
    /// the attention rate is read at.
    pub attn_head_dim: u32,
    /// The recurrent state elements a row steps, summed over the layers: a
    /// scan's state row (a gated delta's `v_heads · k_dim · v_dim`), once a
    /// state the scans read however many forms of the scan the trace holds.
    pub state_elements_per_row: u64,
    /// The dispatches the attentions take: one a layer that attends.
    pub attn_dispatches: u32,
    /// The dispatches the scans take: a conv and a scan a recurrent layer.
    pub scan_dispatches: u32,
    /// The activation bytes a row's per-row ops read and write, one an op
    /// (each its own dispatch): every operand a row of the fire's tokens
    /// wide, at its element's size.
    pub elementwise_ops: Vec<u64>,
    /// The contraction of the largest dense plane: the width its matmul
    /// kernel is picked at, which a fire's rows pad to the tile of.
    pub width: u32,
}

/// What the engine alone knows of a trace's params: the bytes each takes
/// on its device (one a param, in order), which are token tables (a gather
/// reads a row of them a token) and which are routed banks.
#[derive(Debug, Clone, Copy)]
pub struct Kept<'a> {
    pub plane_bytes: &'a [u64],
    pub tables: &'a [usize],
    pub banks: &'a [usize],
}

impl Counts {
    /// The counts of `trace`, its params as `kept` states them, and its
    /// attentions' `(flops a row a key, kv bytes a key)` (all layers
    /// together: `poem_exec::store::kv::attention_counts`).
    #[must_use]
    pub fn of(trace: &Trace, kept: Kept<'_>, attention: (u64, u64)) -> Counts {
        let (mut weight_bytes, mut expert_bytes) = (0u64, 0u64);
        let (mut largest, mut width) = (0u64, 0u64);
        let heads = heads(trace);
        let classes = poem_ir::residency::residency(trace);
        let (mut readout, mut readout_depth) = (0u64, 0u64);
        let experts = routing(trace);
        // `(n, k, routed)` to the shape's planes, elements a row and bytes.
        let mut shapes: std::collections::BTreeMap<(u32, u32, bool), (u32, u64, u64)> =
            std::collections::BTreeMap::new();
        let mut head_bytes = 0u64;
        let to = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        for (at, param) in trace.params.iter().enumerate() {
            if heads.contains(&at) {
                // The output head: read whole every fire (a tied table as
                // much as its own plane), multiplied once a lane.
                let (rows, cols) = rectangle(&param.shape);
                let bytes = kept.plane_bytes.get(at).copied().unwrap_or(0);
                head_bytes = head_bytes.saturating_add(bytes);
                readout = readout.saturating_add(rows.saturating_mul(cols));
                readout_depth = readout_depth.max(cols);
                continue;
            }
            // A token table a gather reads a row of a token: no plane a
            // fire multiplies by, whatever tier it lies in.
            if kept.tables.contains(&at)
                || matches!(
                    classes.get(at),
                    Some(poem_ir::residency::Residency::Rows(_))
                )
            {
                continue;
            }
            let bytes = kept.plane_bytes.get(at).copied().unwrap_or(0);
            if kept.banks.contains(&at) {
                expert_bytes = expert_bytes.saturating_add(bytes);
                // `[E, n, k...]`: an expert's plane, `k` of `E` a row.
                if let [e, n, rest @ ..] = param.shape.as_slice()
                    && !rest.is_empty()
                {
                    let k: u64 = rest.iter().product();
                    let per_row = match experts {
                        Some((picks, _)) => n.saturating_mul(k).saturating_mul(u64::from(picks)),
                        // A bank whose routing does not read: every expert,
                        // every row.
                        None => e.saturating_mul(*n).saturating_mul(k),
                    };
                    let seat = shapes.entry((to(*n), to(k), true)).or_default();
                    *seat = (seat.0 + 1, seat.1 + per_row, seat.2 + bytes);
                }
            } else {
                weight_bytes = weight_bytes.saturating_add(bytes);
                if param.shape.len() >= 2 {
                    let (n, k) = rectangle(&param.shape);
                    let elements = n.saturating_mul(k);
                    let seat = shapes.entry((to(n), to(k), false)).or_default();
                    *seat = (seat.0 + 1, seat.1 + elements, seat.2 + bytes);
                    if elements > largest {
                        largest = elements;
                        width = k;
                    }
                }
            }
        }
        let (attn_flops_per_row_key, kv_bytes_per_key) = attention;
        Counts {
            weight_bytes,
            expert_bytes,
            experts,
            gemm_shapes: shapes
                .into_iter()
                .map(|((n, k, routed), (planes, elements, bytes))| GemmShape {
                    n,
                    k,
                    flops_per_row: elements.saturating_mul(2),
                    planes,
                    bytes,
                    routed,
                })
                .collect(),
            head_bytes,
            readout_flops: readout.saturating_mul(2),
            readout_depth: to(readout_depth),
            attn_flops_per_row_key,
            kv_bytes_per_key,
            attn_head_dim: attn_head_dim(trace),
            state_elements_per_row: state_elements(trace).0,
            attn_dispatches: attn_dispatches(trace),
            scan_dispatches: state_elements(trace).1.saturating_mul(2),
            elementwise_ops: elementwise_bytes(trace),
            width: u32::try_from(width).unwrap_or(u32::MAX),
        }
    }
}

/// The rectangle a param of `shape` is read as: its first axis the rows,
/// the rest its width.
#[must_use]
pub fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

/// The params the trace's output head multiplies by (a `Matmul` marked
/// `logits`, or its softcapped fusion).
fn heads(trace: &Trace) -> Vec<usize> {
    let param = |value: poem_ir::ValueId| match trace.values.get(value.0 as usize) {
        Some(decl) => match decl.def {
            Def::Weight(w) => Some(w as usize),
            _ => None,
        },
        None => None,
    };
    let mut out: Vec<usize> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Linear(poem_ir::Linear::Matmul {
                w, logits: true, ..
            })
            | Operation::Fused(poem_ir::Fused::MatmulSoftcap { w, .. }) => param(*w),
            _ => None,
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The widest head of `trace`'s attentions: their prefill, decode, dense
/// and ragged forms' stated head width.
fn attn_head_dim(trace: &Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attn(
                poem_ir::Attn::Prefill { head_dim, .. }
                | poem_ir::Attn::Decode { head_dim, .. }
                | poem_ir::Attn::Dense { head_dim, .. }
                | poem_ir::Attn::Ragged { head_dim, .. },
            ) => Some(*head_dim),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The state elements a row steps, over every recurrent scan of `trace`:
/// each distinct state row its gated-delta, KDA and Mamba-2 scans read,
/// at its slab's elements (one slot's).
fn state_elements(trace: &Trace) -> (u64, u32) {
    use poem_ir::Operands as _;
    let mut states: Vec<usize> = Vec::new();
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        let Operation::Ssm(scan) = &node.op else {
            continue;
        };
        if !matches!(
            scan,
            poem_ir::Ssm::GatedDelta { .. }
                | poem_ir::Ssm::GatedDeltaChunked { .. }
                | poem_ir::Ssm::Kda { .. }
                | poem_ir::Ssm::KdaChunked { .. }
                | poem_ir::Ssm::Mamba2 { .. }
                | poem_ir::Ssm::Mamba2Chunked { .. }
        ) {
            continue;
        }
        inputs.clear();
        node.op.inputs(&mut inputs);
        for value in &inputs {
            if let Some(decl) = trace.values.get(value.0 as usize)
                && let Def::Cache(row) = decl.def
                && matches!(
                    trace.caches.get(row as usize),
                    Some(poem_ir::CacheRow::State { .. })
                )
                && !states.contains(&(row as usize))
            {
                states.push(row as usize);
            }
        }
    }
    let elements = states
        .iter()
        .filter_map(|&row| match trace.caches.get(row) {
            Some(poem_ir::CacheRow::State { slab, .. }) => Some(slab.iter().product::<u64>()),
            _ => None,
        })
        .sum();
    (elements, u32::try_from(states.len()).unwrap_or(u32::MAX))
}

/// The dispatches `trace`'s attentions take: one a layer that attends (its
/// prefill and decode forms one dispatch, whichever a fire takes), one a
/// node where no layer is stated.
fn attn_dispatches(trace: &Trace) -> u32 {
    let mut layers: Vec<u32> = Vec::new();
    let mut unlayered = 0u32;
    for node in &trace.nodes {
        if !matches!(
            node.op,
            Operation::Attn(
                poem_ir::Attn::Prefill { .. }
                    | poem_ir::Attn::Decode { .. }
                    | poem_ir::Attn::Dense { .. }
                    | poem_ir::Attn::Ragged { .. }
            )
        ) {
            continue;
        }
        match node.layer {
            Some(layer) if !layers.contains(&layer) => layers.push(layer),
            Some(_) => {}
            None => unlayered += 1,
        }
    }
    u32::try_from(layers.len()).unwrap_or(u32::MAX) + unlayered
}

/// The activation bytes a row's per-row ops read and write: each operand
/// whose rows are the fire's tokens, its row at its element's size (an op in
/// place reads its row and writes it back, both counted), over the
/// elementwise nodes (norms over the hidden width and over each head,
/// RoPE, activations, residual adds), the q / k / v splits, the fused
/// norm and RoPE forms, and the kv append (its k and v rows read and
/// written to the pool).
fn elementwise_bytes(trace: &Trace) -> Vec<u64> {
    use poem_ir::Operands as _;
    let row_bytes = |value: &poem_ir::ValueId| -> u64 {
        let Some(decl) = trace.values.get(value.0 as usize) else {
            return 0;
        };
        let Ty::Tensor { shape, dtype } = &decl.ty else {
            return 0;
        };
        let rows = match shape.first() {
            Some(Dim::Tokens) => 1,
            Some(Dim::TokensTimes(n)) => u64::from(*n),
            _ => return 0,
        };
        let mut elements = rows;
        for d in &shape[1..] {
            match d {
                Dim::Const(n) => elements = elements.saturating_mul(*n),
                _ => return 0,
            }
        }
        elements.saturating_mul(dtype.bits()) / 8
    };
    let mut operands = Vec::new();
    let mut ops = Vec::new();
    for node in &trace.nodes {
        // Each counted node's operands, and how many times its rows move:
        // once each, or twice for an append (read, then written to the
        // pool, whose own value is no row of the fire's).
        let times = match &node.op {
            Operation::Elemwise(_) => 1,
            Operation::Gather(
                poem_ir::Gather::SplitQkv { .. }
                | poem_ir::Gather::SplitQGate { .. }
                | poem_ir::Gather::SplitQkvPadded { .. },
            ) => 1,
            Operation::Fused(
                poem_ir::Fused::ResidualAddRmsnorm { .. }
                | poem_ir::Fused::RmsnormResidualAdd { .. }
                | poem_ir::Fused::NormRope { .. }
                | poem_ir::Fused::NormRopePair { .. }
                | poem_ir::Fused::NormModulate { .. }
                | poem_ir::Fused::GatedResidualNormModulate { .. }
                | poem_ir::Fused::ResBlendRmsnorm { .. },
            ) => 1,
            Operation::Fused(
                poem_ir::Fused::NormRopeLastAppend { .. }
                | poem_ir::Fused::QGateNormRopeAppend { .. }
                | poem_ir::Fused::QkvFusedQknormRopeVnormWrite { .. }
                | poem_ir::Fused::RopeYarnAppend { .. }
                | poem_ir::Fused::SplitRopeYarnAppend { .. },
            ) => 2,
            Operation::Attn(
                poem_ir::Attn::KvAppend { .. }
                | poem_ir::Attn::MlaKvAppend { .. }
                | poem_ir::Attn::PoolKvAppend { .. }
                | poem_ir::Attn::IndexKvAppend { .. },
            ) => 2,
            _ => continue,
        };
        operands.clear();
        node.op.inputs(&mut operands);
        node.op.outputs(&mut operands);
        let bytes: u64 = operands.iter().map(row_bytes).sum();
        ops.push(bytes.saturating_mul(times));
    }
    ops
}

/// `(k, E)` of the trace's routed selects: the routes a row carries (the
/// routes' width) and the experts its bank holds (the bank's first axis).
fn routing(trace: &Trace) -> Option<(u32, u32)> {
    let shape = |value: poem_ir::ValueId| match trace.values.get(value.0 as usize) {
        Some(decl) => match (&decl.def, &decl.ty) {
            (Def::Weight(w), _) => trace.params.get(*w as usize).map(|p| p.shape.clone()),
            (_, Ty::Tensor { shape, .. }) => Some(
                shape
                    .iter()
                    .map(|d| match d {
                        Dim::Const(n) => *n,
                        _ => 0,
                    })
                    .collect(),
            ),
            _ => None,
        },
        None => None,
    };
    trace.nodes.iter().find_map(|node| match &node.op {
        Operation::Linear(poem_ir::Linear::MoeSelect { bank, routes, .. }) => {
            let e = *shape(*bank)?.first()?;
            let k = *shape(*routes)?.get(1)?;
            Some((u32::try_from(k).ok()?, u32::try_from(e).ok()?))
        }
        _ => None,
    })
}

/// The estimate of a fire's time on a device of `rates`, or the first rate
/// the model's work needs that the device's profile leaves unstated: the
/// matmul curve and the bandwidths always, the attention curve where the
/// model attends, the scan rate where it carries recurrent state.
pub fn estimate(
    counts: &Counts,
    rates: &profile::fire::Rates,
) -> Result<FireEstimate, &'static str> {
    let constants = crate::roofline::Constants::of(rates)?;
    if counts.attn_flops_per_row_key > 0 && rates.attn_tflops_by_keys.is_none() {
        return Err("attn_tflops_by_keys");
    }
    if counts.state_elements_per_row > 0 && rates.scan_gelems.is_none() {
        return Err("scan_gelems");
    }
    Ok(FireEstimate {
        rates: constants,
        attn_head_dim: counts.attn_head_dim,
        dense_bytes: counts.weight_bytes,
        expert_bytes: counts.expert_bytes,
        experts: counts.experts,
        gemm_shapes: counts.gemm_shapes.clone(),
        head_bytes: counts.head_bytes,
        readout_flops: counts.readout_flops as f64,
        readout_depth: counts.readout_depth,
        attn_flops_per_row_key: counts.attn_flops_per_row_key as f64,
        kv_bytes_per_key: counts.kv_bytes_per_key as f64,
        state_elements_per_row: counts.state_elements_per_row as f64,
        elementwise_ops: counts.elementwise_ops.clone(),
        attn_dispatches: counts.attn_dispatches,
        scan_dispatches: counts.scan_dispatches,
    })
}

/// Why no top rung was chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum Refused {
    /// The device's profile leaves `key` of `[rates]` unstated, which the
    /// model's work is weighed by: nothing weighs its rungs.
    NoRates { device: String, key: &'static str },
    /// No rung offered is feasible.
    NoRung(NoRung),
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::NoRates { device, key } => write!(
                f,
                "device {device} has no rates in its profile for this model: `[rates] {key}` is \
                 unstated (or stated at nothing), so nothing weighs its top rung; `pie tune <backend> --rates` measures \
                 it"
            ),
            Refused::NoRung(no) => no.fmt(f),
        }
    }
}

impl std::error::Error for Refused {}

/// The fire top ([`super::fire_top`]) of `rungs` (every rung the engine's
/// lattice offers) a prompt of `context` holds, for `device`, whose profile
/// states `rates`, over a live range of `live` bytes, by the estimate from
/// the rates alone (its matmuls padding nothing: [`choose_tiled`] states the
/// backend's tile). A device whose profile states no rates has nothing to
/// weigh the rungs by, and the load is refused, naming it.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    context: u32,
    rates: &profile::fire::Rates,
    device: &str,
    counts: &Counts,
) -> Result<(TopRung, FireEstimate), Refused> {
    choose_tiled(rungs, live, context, rates, None, device, counts)
}

/// [`choose`] with the backend's matmul tile rows (`row_tile`), each
/// plane's rows padded to its kernel's tile.
pub fn choose_tiled(
    rungs: &[RungFacts],
    live: u64,
    context: u32,
    rates: &profile::fire::Rates,
    row_tile: Option<crate::roofline::RowTile>,
    device: &str,
    counts: &Counts,
) -> Result<(TopRung, FireEstimate), Refused> {
    choose_shaped(rungs, live, context, rates, device, counts, |c| {
        Ok(match &row_tile {
            Some(tile) => c.with_row_tile(tile.clone()),
            None => c,
        })
    })
}

/// [`choose_tiled`] with the backend's kernels laid on the estimate's
/// constants by `shape` (its tile rows, its few-rows kernel): the key it
/// refuses by where the device's profile states none.
pub fn choose_shaped(
    rungs: &[RungFacts],
    live: u64,
    context: u32,
    rates: &profile::fire::Rates,
    device: &str,
    counts: &Counts,
    shape: impl FnOnce(
        crate::roofline::Constants,
    ) -> Result<crate::roofline::Constants, &'static str>,
) -> Result<(TopRung, FireEstimate), Refused> {
    let refused = |key| Refused::NoRates {
        device: device.to_owned(),
        key,
    };
    let mut estimate = estimate(counts, rates).map_err(refused)?;
    estimate.rates = shape(estimate.rates).map_err(refused)?;
    let top = super::rung::fire_top(rungs, live, &estimate, context).map_err(Refused::NoRung)?;
    Ok((top, estimate))
}

/// The load's one line on its top rung: which (`top` tokens), by what, and
/// the `span` it reserves of the `live` range; the rates where this load
/// measured them.
#[must_use]
pub fn line_of(
    rung: &TopRung,
    top: u32,
    span: u64,
    live: u64,
    tuned: &profile::fire::Tuned,
) -> String {
    let r = &tuned.rates;
    let rates = format!(
        "{:.1} TFLOP/s bf16, {:.0} GB/s, launch {} ns",
        r.gemm_tflops_by_rows
            .as_ref()
            .and_then(|w| w.last())
            .and_then(|(_, c)| c.last())
            .map_or(0.0, |&(_, t)| t),
        r.hbm_gbps.unwrap_or(0.0),
        r.launch_ns.unwrap_or(0)
    );
    let by = match rung.why {
        TopWhy::Argmax { eff, live_share } => {
            format!("by the estimate: eff {eff:.3} at a live share of {live_share:.3}")
        }
        TopWhy::OnlyFeasible => "as the one feasible rung".to_string(),
    };
    let gib = live as f64 / f64::from(1u32 << 30);
    match measured_line(tuned) {
        Some(measured) => format!(
            "top rung {top} {by}; span {} MiB of {gib:.1} GiB live; {measured}",
            span >> 20
        ),
        None => format!(
            "top rung {top} {by}; span {} MiB of {gib:.1} GiB live; the profile's rates: {rates}",
            span >> 20
        ),
    }
}

/// What a load's line says of rates it measured itself
/// (`engine::rates::rates_at_load`): `measured the roofline at load: <the
/// constants>; saved to <path>` (or kept for this load alone); `None` where
/// its profile stated them.
#[must_use]
pub fn measured_line(tuned: &profile::fire::Tuned) -> Option<String> {
    if !tuned.measured_this_load {
        return None;
    }
    let r = &tuned.rates;
    // A curve's widest point, its rate where the device runs fastest.
    let top = |c: &[(u32, f64)]| c.iter().map(|&(_, t)| t).fold(0.0, f64::max);
    let constants = format!(
        "launch {} ns, matmul to {:.1} TFLOP/s, attention to {:.1} TFLOP/s, {:.0} GB/s, \
         per-row ops {:.0} GB/s, {:.0} G scan elements/s",
        r.launch_ns.unwrap_or(0),
        r.gemm_tflops_by_rows
            .as_ref()
            .map_or(0.0, |w| w.iter().map(|(_, c)| top(c)).fold(0.0, f64::max)),
        r.attn_tflops_by_keys.as_deref().map_or(0.0, top),
        r.hbm_gbps.unwrap_or(0.0),
        r.elementwise_gbps.unwrap_or(0.0),
        r.scan_gelems.unwrap_or(0.0),
    );
    let cut = if tuned.frame_cut {
        " (a frame stopped at the device's bound on one submission before it settled)"
    } else {
        ""
    };
    Some(match &tuned.saved_to {
        Some(path) => format!("measured the roofline at load{cut}: {constants}; saved to {path}"),
        None => {
            format!("measured the roofline at load{cut}: {constants}; kept for this load alone")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn rung(tokens: u32, span: u64, infeasible: Option<&'static str>) -> RungFacts {
        RungFacts {
            tokens,
            lanes: tokens.min(256),
            keys: tokens,
            launches: 100,
            padded_rows: tokens,
            span_bytes: span,
            infeasible,
            fire_lanes: 256,
        }
    }

    fn counts() -> Counts {
        Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![GemmShape::dense(4096, 4096, 1 << 30, 1 << 30)],
            head_bytes: 0,
            attn_flops_per_row_key: 1 << 10,
            kv_bytes_per_key: 1 << 10,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![0],
            attn_dispatches: 28,
            scan_dispatches: 0,
            width: 4096,
        }
    }

    fn rates() -> profile::fire::Rates {
        profile::fire::Rates {
            launch_ns: Some(5),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 5.0), (4096, 10.0)])]),
            attn_tflops_by_keys: Some(vec![(2048, 7.0)]),
            hbm_gbps: Some(100.0),
            elementwise_gbps: Some(50.0),
            scan_gelems: None,
            ..profile::fire::Rates::default()
        }
    }

    #[test]
    fn without_rates_the_load_is_refused_by_name() {
        let rungs = [rung(1024, 100 * MIB, None), rung(2048, 150 * MIB, None)];
        let bare = profile::fire::Rates::default();
        let err = choose(&rungs, 1 << 33, u32::MAX, &bare, "Card X", &counts()).unwrap_err();
        assert!(
            err.to_string()
                .contains("device Card X has no rates in its profile"),
            "{err}"
        );
        assert!(
            err.to_string()
                .contains("`pie tune <backend> --rates` measures it"),
            "{err}"
        );
        // No feasible rung at all: the ruling's refusal.
        let none = [rung(1024, 300 * MIB, Some("past one binding"))];
        let err = choose(&none, 1 << 33, u32::MAX, &rates(), "Card X", &counts()).unwrap_err();
        assert!(matches!(err, Refused::NoRung(_)), "{err}");
        assert!(
            err.to_string()
                .contains("1024 tokens (span 300 MiB): past one binding"),
            "{err}"
        );
    }

    #[test]
    fn the_estimate_reads_the_rates_and_the_counts() {
        let e = estimate(&counts(), &rates()).unwrap();
        assert_eq!(e.gemm_shapes, counts().gemm_shapes);
        assert_eq!(
            (
                e.rates.gemm_by_rows.clone(),
                e.rates.attn_by_keys.clone(),
                e.rates.bytes_per_ns,
                e.rates.elementwise_bytes_per_ns,
                e.rates.launch_ns
            ),
            (
                vec![(4096, vec![(256, 5_000.0), (4096, 10_000.0)])],
                vec![(2048, 7_000.0)],
                100.0,
                50.0,
                5.0
            )
        );
        // No recurrent state: no scan rate read.
        assert_eq!(e.rates.scan_elements_per_ns, f64::INFINITY);
        assert_eq!(e.dense_bytes, 1 << 30);
        // Each rate the model's work needs, unstated, is named.
        for (key, unstated) in [
            (
                "hbm_gbps",
                profile::fire::Rates {
                    hbm_gbps: None,
                    ..rates()
                },
            ),
            (
                "gemm_tflops_by_rows",
                profile::fire::Rates {
                    gemm_tflops_by_rows: None,
                    ..rates()
                },
            ),
            (
                "elementwise_gbps",
                profile::fire::Rates {
                    elementwise_gbps: None,
                    ..rates()
                },
            ),
            (
                "attn_tflops_by_keys",
                profile::fire::Rates {
                    attn_tflops_by_keys: None,
                    ..rates()
                },
            ),
        ] {
            assert_eq!(estimate(&counts(), &unstated).map(|_| ()), Err(key));
        }
        let recurrent = Counts {
            state_elements_per_row: 1 << 20,
            ..counts()
        };
        assert_eq!(
            estimate(&recurrent, &rates()).map(|_| ()),
            Err("scan_gelems")
        );
        let scanned = profile::fire::Rates {
            scan_gelems: Some(3.0),
            ..rates()
        };
        assert_eq!(
            estimate(&recurrent, &scanned)
                .unwrap()
                .rates
                .scan_elements_per_ns,
            3.0
        );
        // A model attending nothing reads no attention rate.
        let attending_nothing = Counts {
            attn_flops_per_row_key: 0,
            ..counts()
        };
        let no_attn = profile::fire::Rates {
            attn_tflops_by_keys: None,
            ..rates()
        };
        assert!(estimate(&attending_nothing, &no_attn).is_ok());
    }

    #[test]
    fn the_refusal_names_the_key_unstated() {
        let rungs = [rung(1024, 100 * MIB, None)];
        let rates = profile::fire::Rates {
            attn_tflops_by_keys: None,
            ..rates()
        };
        let err = choose(&rungs, 1 << 33, u32::MAX, &rates, "Card X", &counts()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "device Card X has no rates in its profile for this model: `[rates] \
             attn_tflops_by_keys` is unstated (or stated at nothing), so nothing weighs its top \
             rung; `pie tune <backend> --rates` measures it"
        );
        // A rate that settled at nothing is refused by the same name, never
        // divided by.
        let zero = profile::fire::Rates {
            attn_tflops_by_keys: Some(vec![(512, 0.0), (8192, 4.0)]),
            ..self::rates()
        };
        let err = choose(&rungs, 1 << 33, u32::MAX, &zero, "Card X", &counts()).unwrap_err();
        assert!(
            err.to_string().contains("`[rates] attn_tflops_by_keys`"),
            "{err}"
        );
    }

    /// A trace of `nodes` over values of `[tokens, width]` bf16 each (one a
    /// width), and a cache value (no row of the fire's) at the end.
    fn per_row(widths: &[u64], nodes: Vec<Operation>) -> Trace {
        let bf16 = |width: u64| poem_ir::ValueDecl {
            def: Def::Op(0),
            ty: Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(width)],
                dtype: poem_ir::Dtype::Bf16,
            },
        };
        let mut values: Vec<poem_ir::ValueDecl> = widths.iter().map(|&w| bf16(w)).collect();
        values.push(poem_ir::ValueDecl {
            def: Def::Cache(0),
            ty: Ty::Tensor {
                shape: vec![Dim::Const(1024), Dim::Const(256)],
                dtype: poem_ir::Dtype::Bf16,
            },
        });
        Trace {
            name: "per-row".into(),
            platform: poem_ir::Platform::Vulkan,
            params: Vec::new(),
            caches: Vec::new(),
            values,
            nodes: nodes
                .into_iter()
                .map(|op| poem_ir::Node {
                    op,
                    guard: poem_ir::Guard::Always,
                    layer: None,
                })
                .collect(),
            outputs: Vec::new(),
            drafter: None,
            facts: poem_ir::Facts::default(),
            inputs: Vec::new(),
        }
    }

    fn v(i: u32) -> poem_ir::ValueId {
        poem_ir::ValueId(i)
    }

    #[test]
    fn a_norm_over_the_hidden_and_over_each_head_reads_and_writes_its_row() {
        // rms over a 1024-wide row: 2 KiB in, 2 KiB out; over q's 16 heads
        // of 128 (a per-head norm): 4 KiB in and out.
        let rms = |x, y, axis| {
            Operation::Elemwise(poem_ir::Elemwise::Rmsnorm {
                x: v(x),
                weight: poem_ir::NormWeight::None,
                axis,
                eps: 1e-6,
                y: v(y),
            })
        };
        let trace = per_row(
            &[1024, 1024, 2048, 2048],
            vec![
                rms(0, 1, poem_ir::NormAxis::Row),
                rms(2, 3, poem_ir::NormAxis::Head(128)),
            ],
        );
        assert_eq!(
            elementwise_bytes(&trace).iter().sum::<u64>(),
            2 * 2048 + 2 * 4096
        );
    }

    #[test]
    fn the_qkv_split_reads_its_packed_row_and_writes_three() {
        // 2048 + 512 + 512 packed: 6 KiB in, 4 + 1 + 1 KiB out.
        let trace = per_row(
            &[3072, 2048, 512, 512],
            vec![Operation::Gather(poem_ir::Gather::SplitQkv {
                packed: v(0),
                q_width: 2048,
                kv_width: 512,
                q: v(1),
                k: v(2),
                v: v(3),
            })],
        );
        assert_eq!(
            elementwise_bytes(&trace).iter().sum::<u64>(),
            6144 + 4096 + 1024 + 1024
        );
    }

    #[test]
    fn rope_turns_q_and_k_in_place() {
        // q (2048 wide) and k (512) read and written; the positions are no
        // bf16 row here (a cache-shaped value).
        let trace = per_row(
            &[2048, 512],
            vec![Operation::Elemwise(poem_ir::Elemwise::Rope {
                q: v(0),
                k: Some(v(1)),
                positions: v(2),
                head_dim: 128,
                theta: 1e6,
                turn: poem_ir::Turn::Full { interleaved: false },
                freqs: None,
                q_out: v(0),
                k_out: Some(v(1)),
            })],
        );
        assert_eq!(
            elementwise_bytes(&trace).iter().sum::<u64>(),
            2 * (4096 + 1024)
        );
    }

    #[test]
    fn the_kv_append_reads_its_rows_and_writes_them_to_the_pool() {
        // k and v rows of 512 read, then written: the pool's own value is no
        // row of the fire's.
        let trace = per_row(
            &[512, 512],
            vec![Operation::Attn(poem_ir::Attn::KvAppend {
                k: v(0),
                v: Some(v(1)),
                cache: v(2),
                write_page: v(2),
                write_offset: v(2),
            })],
        );
        assert_eq!(
            elementwise_bytes(&trace).iter().sum::<u64>(),
            2 * (1024 + 1024)
        );
    }

    #[test]
    fn a_param_is_read_as_its_first_axis_by_the_rest() {
        assert_eq!(rectangle(&[4096, 8, 128]), (4096, 1024));
        assert_eq!(rectangle(&[7]), (7, 1));
        assert_eq!(rectangle(&[]), (1, 1));
    }

    #[test]
    fn the_line_names_the_rung_its_share_and_the_rates() {
        let top = TopRung {
            tokens: 2048,
            lanes: 256,
            span_bytes: 512 * MIB,
            why: TopWhy::Argmax {
                eff: 0.9,
                live_share: 0.25,
            },
        };
        let tuned = profile::fire::Tuned {
            rates: profile::fire::Rates {
                gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0), (8192, 249.0)])]),
                hbm_gbps: Some(654.0),
                launch_ns: Some(1620),
                ..profile::fire::Rates::default()
            },
            measured_this_load: false,
            saved_to: None,
            frame_cut: false,
            refused: None,
            ..profile::fire::Tuned::default()
        };
        assert_eq!(
            line_of(&top, 2048, 512 * MIB, 2 << 30, &tuned),
            "top rung 2048 by the estimate: eff 0.900 at a live share of 0.250; span 512 MiB \
             of 2.0 GiB live; the profile's rates: 249.0 TFLOP/s bf16, 654 GB/s, launch 1620 ns"
        );
        // Measured by this load and saved: the line says so, and where.
        let measured = profile::fire::Tuned {
            measured_this_load: true,
            saved_to: Some("/home/u/.pie/profiles/cuda/user/nvidia-l40s.toml".into()),
            ..tuned
        };
        let line = line_of(&top, 2048, 512 * MIB, 2 << 30, &measured);
        assert!(
            line.ends_with(
                "measured the roofline at load: launch 1620 ns, matmul to 249.0 TFLOP/s, \
                 attention to 0.0 TFLOP/s, 654 GB/s, per-row ops 0 GB/s, 0 G scan elements/s; \
                 saved to /home/u/.pie/profiles/cuda/user/nvidia-l40s.toml"
            ),
            "{line}"
        );
    }
}
