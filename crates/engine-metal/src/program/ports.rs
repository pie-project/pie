use std::sync::atomic::{AtomicU64, Ordering};

use engine::ports::{Envelope, resolves};
use eta_exec::{ExecPlan, Value};
use eta_ir::Dtype;
use eta_ir::registry::{GeometryClass, Port};
use eta_ir::types::name_or_unknown;

use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor, Rings};
use crate::device::Buffer;

static RESOLVED: AtomicU64 = AtomicU64::new(0);

#[must_use]
pub fn resolved() -> u64 {
    RESOLVED.load(Ordering::Relaxed)
}

/// The tokens of a fire a pass ahead of it puts: `bytes` at `at` of `slab`,
/// the cell that pass writes, copied in on the device once the queue has run it.
#[derive(Clone, Debug)]
pub struct Fed {
    pub slab: Buffer,
    pub at: u64,
    pub bytes: u64,
}

/// `unlanded[channel]` marks a channel whose next cell an airborne pass owes:
/// its tokens are fed on the device, and any other port read off it is
/// refused, since the host would read the cell before it is written.
pub fn resolve(
    plan: &ExecPlan,
    class: GeometryClass,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
    unlanded: &[bool],
) -> Result<(Envelope, Option<Fed>)> {
    let mut out = Envelope::default();
    let mut fed = None;
    RESOLVED.fetch_add(1, Ordering::Relaxed);
    for binding in &plan.package.ports {
        if !resolves(class, binding.port) {
            continue;
        }
        if !binding.is_const && unlanded.get(binding.channel as usize).copied() == Some(true) {
            if binding.port != Port::EmbedTokens {
                return Err(Fault::program(
                    "program::ports",
                    format!(
                        "port {}'s channel {} is put by a pass still airborne, and only \
                         `embed_tokens` is fed on the device; the host would read this \
                         port before it is written",
                        binding.port.name(),
                        binding.channel
                    ),
                ));
            }
            let (cursor, shape) = cell_of(binding.port, binding.channel, cursors, shapes)?;
            if !matches!(shape.dtype, Dtype::I32 | Dtype::U32) {
                return Err(Fault::program(
                    "program::ports",
                    format!(
                        "port {}'s channel {} holds {}, and a token is an index",
                        binding.port.name(),
                        binding.channel,
                        name_or_unknown(shape.dtype)
                    ),
                ));
            }
            let channel = binding.channel as usize;
            fed = Some(Fed {
                slab: rings.slab(channel)?.clone(),
                at: rings.cell_offset(channel, cursor.head)?,
                bytes: shape.cell_bytes() as u64,
            });
            // The rows are the cell's; their ids are the device's.
            out.tokens = Some(vec![0; shape.numel]);
            continue;
        }
        if binding.port == Port::AttnMask {
            if binding.is_const {
                continue;
            }
            out.mask = Some(read_bool_cell(
                binding.port,
                binding.channel,
                rings,
                cursors,
                shapes,
            )?);
            continue;
        }
        let slot = match binding.port {
            Port::EmbedIndptr => &mut out.qo_indptr,
            Port::EmbedTokens => &mut out.tokens,
            Port::Positions => &mut out.positions,
            Port::Slots => &mut out.slots,
            Port::KvLen => &mut out.kv_len,
            Port::Fold => &mut out.fold,
            _ => continue,
        };
        let value = if binding.is_const {
            let folded = plan
                .const_ports
                .iter()
                .find(|folded| folded.port == binding.port)
                .ok_or_else(|| {
                    Fault::program(
                        "program::ports",
                        format!(
                            "port {} is declared const and no folded value was kept for it",
                            binding.port.name()
                        ),
                    )
                })?;
            as_u32(binding.port, &folded.value)?
        } else {
            read_cell(binding.port, binding.channel, rings, cursors, shapes)?
        };
        *slot = Some(value);
    }
    Ok((out, fed))
}

fn read_cell(
    port: Port,
    channel: u32,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Vec<u32>> {
    let (cursor, shape) = cell_of(port, channel, cursors, shapes)?;
    if !matches!(shape.dtype, Dtype::I32 | Dtype::U32) {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} holds {}, and a geometry index is not an \
                 activation",
                port.name(),
                name_or_unknown(shape.dtype)
            ),
        ));
    }
    let cell = rings.read_cell(channel as usize, cursor.head)?;
    Ok(cell
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn read_bool_cell(
    port: Port,
    channel: u32,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Vec<bool>> {
    let (cursor, shape) = cell_of(port, channel, cursors, shapes)?;
    if shape.dtype != Dtype::Bool {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} holds {}, and an attention mask is a \
                 rectangle of bools",
                port.name(),
                name_or_unknown(shape.dtype)
            ),
        ));
    }
    let cell = rings.read_cell(channel as usize, cursor.head)?;
    if cell.len() != shape.cell_bytes() {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} declares {} bool lane(s) — {} packed byte(s) \
                 — and its cell is {} byte(s)",
                port.name(),
                shape.numel,
                shape.cell_bytes(),
                cell.len()
            ),
        ));
    }
    Ok(unpack_bits(&cell, shape.numel))
}

fn unpack_bits(packed: &[u8], numel: usize) -> Vec<bool> {
    (0..numel)
        .map(|lane| packed[lane / 8] >> (lane % 8) & 1 != 0)
        .collect()
}

fn cell_of(
    port: Port,
    channel: u32,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<(Cursor, ChannelShape)> {
    let index = channel as usize;
    let (Some(cursor), Some(shape)) = (cursors.get(index).copied(), shapes.get(index).copied())
    else {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {} names channel {channel}, which this instance does not carry",
                port.name()
            ),
        ));
    };
    if cursor.tail == cursor.head {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {} names channel {channel}, whose ring holds nothing committed \
                 (head and tail are both {}); the port's value for this fire is the \
                 cell the guest's own pass takes, and no pass has published one",
                port.name(),
                cursor.head
            ),
        ));
    }
    Ok((cursor, shape))
}

fn as_u32(port: Port, value: &Value) -> Result<Vec<u32>> {
    match value {
        Value::U32(lanes) => Ok(lanes.clone()),
        Value::I32(lanes) => Ok(lanes.iter().map(|&lane| lane as u32).collect()),
        other => Err(Fault::program(
            "program::ports",
            format!(
                "the folded constant for port {} is {}, and a geometry index is not an \
                 activation",
                port.name(),
                name_or_unknown(other.dtype())
            ),
        )),
    }
}
