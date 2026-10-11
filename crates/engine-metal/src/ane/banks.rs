//! The GPU's banks of a split, all views into the whole banks: the leading
//! `keep` rows of gate and up (or of a projection) by offset, and the
//! leading `keep` columns of down by a row stride wider than the width.

use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;

#[derive(Clone, Copy)]
pub struct Split {
    pub gate: Bank,
    pub up: Bank,
    pub down: Bank,
    /// The intermediate channels the GPU keeps.
    pub keep: u32,
}

/// One dense MLP the compiler fused, with its weights as Q4 banks.
pub struct Mlp {
    /// The MLP's index among the fused ones, in trace order; the weight
    /// sets alternate by its parity.
    pub layer: u32,
    /// The op's `gate_up` value, which is how a dispatch names the layer.
    pub key: poem_ir::ValueId,
    pub gate_up: Bank,
    pub down: Bank,
    pub intermediate: u32,
}

/// The affine 4-bit bank in groups of 64 behind `id`, which is what the
/// hand-off kernels dequantize, if that is what `id` is.
fn q4_bank(
    trace: &poem_ir::Trace,
    weights: &crate::weights::Weights,
    id: poem_ir::ValueId,
) -> Option<Bank> {
    match trace.values[id.0 as usize].def {
        poem_ir::Def::Weight(at) => match weights.table().0.get(at as usize).copied().flatten() {
            Some(crate::run::WeightRow::Planes(bank))
                if bank.bits == 4 && bank.group == 64 && bank.codes.dtype == Dtype::U4g64 =>
            {
                Some(bank)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Every fused dense MLP whose weights are affine 4-bit planes in groups
/// of 64.
pub fn mlps(trace: &poem_ir::Trace, weights: &crate::weights::Weights) -> Vec<Mlp> {
    let bank = |id| q4_bank(trace, weights, id);
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Fused(poem_ir::Fused::MlpSwiglu {
                gate_up,
                down,
                intermediate,
                ..
            }) => Some((gate_up, down, *intermediate)),
            _ => None,
        })
        .enumerate()
        .filter_map(|(layer, (gate_up, down, intermediate))| {
            Some(Mlp {
                layer: layer as u32,
                key: *gate_up,
                gate_up: bank(*gate_up)?,
                down: bank(*down)?,
                intermediate,
            })
        })
        .collect()
}

/// A plain projection the Neural Engine may take trailing columns of.
pub struct Projection {
    /// The projection's index among those of its shape, in trace order.
    pub layer: u32,
    /// The op's weight value, which is how a dispatch names it.
    pub key: poem_ir::ValueId,
    pub bank: Bank,
}

/// A projection must land at least this many columns to be worth a
/// hand-off of its own.
pub const MIN_PROJECTION_COLUMNS: u32 = 2048;

/// `PIE_ANE_PROJECTIONS=0|off|false` keeps every projection on the GPU;
/// the MLPs still split.
#[cfg(target_vendor = "apple")]
fn projections_enabled() -> bool {
    !matches!(
        std::env::var("PIE_ANE_PROJECTIONS").as_deref(),
        Ok("0" | "off" | "false")
    )
}

/// Every `linear.matmul` over an affine 4-bit bank wide enough to split,
/// grouped by `(contraction, columns)` and numbered within the group.
#[cfg(target_vendor = "apple")]
pub fn projections(
    trace: &poem_ir::Trace,
    weights: &crate::weights::Weights,
) -> Vec<((u32, u32), Vec<Projection>)> {
    let mut groups: Vec<((u32, u32), Vec<Projection>)> = Vec::new();
    if !projections_enabled() {
        return groups;
    }
    for node in &trace.nodes {
        let poem_ir::Operation::Linear(poem_ir::Linear::Matmul { w, .. }) = &node.op else {
            continue;
        };
        let Some(bank) = q4_bank(trace, weights, *w) else {
            continue;
        };
        let (k, n) = (bank.codes.width, bank.codes.rows);
        if n < MIN_PROJECTION_COLUMNS
            || !n.is_multiple_of(kernels_metal::ane::ffn::UNIT)
            || kernels_metal::ane::mil::segment_of(k).is_none()
        {
            continue;
        }
        let group = match groups.iter_mut().find(|(shape, _)| *shape == (k, n)) {
            Some((_, group)) => group,
            None => {
                groups.push(((k, n), Vec::new()));
                &mut groups.last_mut().expect("just pushed").1
            }
        };
        group.push(Projection {
            layer: group.len() as u32,
            key: *w,
            bank,
        });
    }
    groups
}

/// `rows` rows of the bank from `start`, as views.
pub fn rows_of(handles: &Handles, bank: &Bank, start: u32, rows: u32) -> Result<Bank> {
    let plane = |t: Tensor, bits: u32| -> Result<Tensor> {
        let row = u64::from(t.width) * u64::from(bits) / 8;
        let buf = if start == 0 {
            t.buf
        } else {
            handles.cut(t.buf, row * u64::from(start), row * u64::from(rows))?
        };
        Ok(Tensor::new(buf, rows, t.width, t.dtype))
    };
    Ok(Bank {
        codes: plane(bank.codes, bank.bits)?,
        mpp_codes: None,
        scales: plane(bank.scales, 16)?,
        biases: bank.biases.map(|b| plane(b, 16)).transpose()?,
        group: bank.group,
        bits: bank.bits,
        ld: bank.ld,
    })
}

/// The leading `cols` columns of every row of the bank: the same planes
/// read `cols` wide at the bank's own row stride.
#[must_use]
pub fn columns_of(bank: &Bank, cols: u32) -> Bank {
    let narrow = |t: Tensor, width: u32| Tensor::new(t.buf, t.rows, width, t.dtype);
    let groups = cols / bank.group;
    Bank {
        codes: narrow(bank.codes, cols),
        mpp_codes: bank.mpp_codes,
        scales: narrow(bank.scales, groups),
        biases: bank.biases.map(|b| narrow(b, groups)),
        group: bank.group,
        bits: bank.bits,
        ld: bank.ld,
    }
}

pub fn split(handles: &Handles, mlp: &Mlp, keep: u32) -> Result<Split> {
    Ok(Split {
        gate: rows_of(handles, &mlp.gate_up, 0, keep)?,
        up: rows_of(handles, &mlp.gate_up, mlp.intermediate, keep)?,
        down: columns_of(&mlp.down, keep),
        keep,
    })
}

#[allow(clippy::mut_from_ref)]
pub fn host(buffer: &Buffer) -> Result<&mut [u8]> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLBuffer;
        let len = buffer.bytes() as usize;
        let base = buffer.raw().contents().as_ptr().cast::<u8>();
        // SAFETY: a zeroed buffer is shared storage, so its contents are
        // host-visible for its whole life; the slice is as long as it is.
        Ok(unsafe { std::slice::from_raw_parts_mut(base, len) })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = buffer;
        Err(crate::error::Fault::Deviceless)
    }
}

/// A zeroed `rows x width` buffer of `dtype`, bound as a tensor.
pub fn staging(
    device: &Context,
    handles: &Handles,
    rows: u32,
    width: u32,
    dtype: Dtype,
    keep: &mut Vec<Buffer>,
) -> Result<Tensor> {
    let bytes = u64::from(rows) * u64::from(width) * dtype.bits() / 8;
    let buffer = Buffer::zeroed(device, bytes)?;
    let tensor = Tensor::new(handles.bind(&buffer, 0, bytes)?, rows, width, dtype);
    keep.push(buffer);
    Ok(tensor)
}
