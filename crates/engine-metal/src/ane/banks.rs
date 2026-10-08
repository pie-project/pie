use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;

#[derive(Clone, Copy)]
pub struct Split {
    pub gate: Bank,
    pub up: Bank,
    pub down: Bank,
    pub keep: u32,
}

pub struct Mlp {
    pub layer: u32,
    pub gate_up: Bank,
    pub down: Bank,
    pub intermediate: u32,
}

pub fn mlps(trace: &poem_ir::Trace, weights: &crate::weights::Weights) -> Vec<Mlp> {
    let bank = |id: poem_ir::ValueId| match trace.values[id.0 as usize].def {
        poem_ir::Def::Weight(at) => match weights.table().0.get(at as usize).copied().flatten() {
            Some(crate::run::WeightRow::Planes(bank))
                if bank.bits == 4 && bank.group == 64 && bank.codes.dtype == Dtype::U4g64 =>
            {
                Some(bank)
            }
            _ => None,
        },
        _ => None,
    };
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Linear(poem_ir::Linear::MlpAne {
                gate_up,
                down,
                intermediate,
                layer,
                ..
            }) => Some(Mlp {
                layer: *layer,
                gate_up: bank(*gate_up)?,
                down: bank(*down)?,
                intermediate: *intermediate,
            }),
            _ => None,
        })
        .collect()
}

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
    })
}

pub fn leading_columns(
    device: &Context,
    handles: &Handles,
    bank: &Bank,
    cols: u32,
    keep: &mut Vec<Buffer>,
) -> Result<Bank> {
    let mut plane = |t: Tensor, bits: u32, out: u32| -> Result<Tensor> {
        let row_in = (u64::from(t.width) * u64::from(bits) / 8) as usize;
        let row_out = (u64::from(out) * u64::from(bits) / 8) as usize;
        let whole = handles.read(t.buf, (row_in * t.rows as usize) as u64)?;
        let bytes: Vec<u8> = whole
            .chunks_exact(row_in)
            .flat_map(|row| &row[..row_out])
            .copied()
            .collect();
        let buffer = Buffer::zeroed(device, bytes.len() as u64)?;
        host(&buffer)?[..bytes.len()].copy_from_slice(&bytes);
        let handle = handles.bind(&buffer, 0, bytes.len() as u64)?;
        keep.push(buffer);
        Ok(Tensor::new(handle, t.rows, out, t.dtype))
    };
    let groups = cols / bank.group;
    Ok(Bank {
        codes: plane(bank.codes, bank.bits, cols)?,
        mpp_codes: None,
        scales: plane(bank.scales, 16, groups)?,
        biases: bank.biases.map(|b| plane(b, 16, groups)).transpose()?,
        group: bank.group,
        bits: bank.bits,
    })
}

pub fn split(
    device: &Context,
    handles: &Handles,
    mlp: &Mlp,
    keep: u32,
    buffers: &mut Vec<Buffer>,
) -> Result<Split> {
    Ok(Split {
        gate: rows_of(handles, &mlp.gate_up, 0, keep)?,
        up: rows_of(handles, &mlp.gate_up, mlp.intermediate, keep)?,
        down: leading_columns(device, handles, &mlp.down, keep, buffers)?,
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
        Ok(unsafe { std::slice::from_raw_parts_mut(base, len) })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = buffer;
        Err(crate::error::Fault::Deviceless)
    }
}

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
