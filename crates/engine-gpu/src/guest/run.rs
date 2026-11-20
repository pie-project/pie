//! What a guest pass's run shares: the readouts it is offered, and values
//! to and from the session's heap.

use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::Value;
use eta_ir::Dtype;

use crate::backend::Backend;
use crate::error::{Fault, Result};

pub struct Readout<'a, B: Backend> {
    /// The output it holds.
    pub output: &'a str,

    pub seat: &'a B::Buffer,

    pub at: u64,

    pub width: u32,

    /// Bf16 values widen; f32 and i32 copy as they are.
    pub dtype: Dtype,
}

impl<B: Backend> Clone for Readout<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for Readout<'_, B> {}

/// The seats a pass reads output root `root` from, `values` values of it,
/// in the order its rows lie: row `r` of an output the pass reads at lane
/// `root.lane` is the readout row of the instance's lane `root.lane + r`,
/// each through that lane's own seat (`seats[k]` is what the fire read out
/// for the pass's lane `k`). A root a lane's seat holds whole reads that
/// seat alone. Each entry is a seat and the values read from it, rows whose
/// seats lie end to end one entry.
pub fn row_seats<'a, B: Backend>(
    seats: &[Vec<Readout<'a, B>>],
    root: &LaunchValue,
    values: u32,
) -> Result<Vec<(Readout<'a, B>, u32)>> {
    let name = root.output.as_deref().unwrap_or("");
    let seat_of = |lane: usize| {
        seats
            .get(lane)
            .and_then(|offered| offered.iter().find(|seat| seat.output == name))
            .cloned()
    };
    let first = root.lane as usize;
    let Some(seat) = seat_of(first) else {
        return Err(Fault::Program {
            at: "guest::run",
            why: format!(
                "this pass reads output {name:?} of lane {first}, which the fire read out none of"
            ),
        });
    };
    if values <= seat.width {
        return Ok(vec![(seat, values)]);
    }
    let row = seat.width.max(1);
    if values % row != 0 {
        return Err(Fault::Program {
            at: "guest::run",
            why: format!(
                "this pass reads {values} values of output {name:?} from lane {first}, whose \
                 readout holds {row}: not whole rows of it"
            ),
        });
    }
    (0..(values / row) as usize)
        .map(|r| {
            let lane = first + r;
            let seat = seat_of(lane).ok_or_else(|| Fault::Program {
                at: "guest::run",
                why: format!(
                    "this pass reads row {r} of output {name:?} at lane {first}, which is lane \
                     {lane}'s readout row, and the fire read out none for that lane"
                ),
            })?;
            if seat.width != row {
                return Err(Fault::Program {
                    at: "guest::run",
                    why: format!(
                        "row {r} of output {name:?} is lane {lane}'s readout of {} values, and \
                         its row 0 holds {row}",
                        seat.width
                    ),
                });
            }
            Ok((seat, row))
        })
        .try_fold(Vec::new(), |mut runs: Vec<(Readout<'a, B>, u32)>, next| {
            // Rows whose seats lie end to end in one buffer read as one run.
            let (seat, values) = next?;
            let elem = if seat.dtype == Dtype::Bf16 { 2 } else { 4 };
            match runs.last_mut() {
                Some((run, held))
                    if std::ptr::eq(run.seat, seat.seat)
                        && run.dtype == seat.dtype
                        && run.at + u64::from(*held) * elem == seat.at =>
                {
                    *held += values;
                    run.width = *held;
                }
                _ => runs.push((seat, values)),
            }
            Ok(runs)
        })
}

/// The first seat of `seats` (`seats[k]` lane `k`'s) no output root of
/// `package` reads a row of, as its output and lane: a root of `held` bytes
/// read at lane `k` reads rows `k..k + rows`, a seat's width a row.
/// `held(stage, local)` is a stage's value's bytes.
pub fn unread_seat<B: Backend>(
    package: &LaunchPackage,
    held: impl Fn(usize, u32) -> Option<u64>,
    seats: &[Vec<Readout<'_, B>>],
) -> Result<Option<(String, usize)>> {
    let mut read: Vec<(usize, &str)> = Vec::new();
    for (global, root) in package.values.iter().enumerate() {
        if root.source != ValueOrigin::Output {
            continue;
        }
        let Some(name) = root.output.as_deref() else {
            continue;
        };
        for at in 0..package.plans.len() {
            let Some(local) = local_of(package, at, global as u32)? else {
                continue;
            };
            let values = held(at, local).unwrap_or(0) / 4;
            let first = root.lane as usize;
            let row = seats
                .get(first)
                .and_then(|offered| offered.iter().find(|seat| seat.output == name))
                .map_or(values, |seat| u64::from(seat.width.max(1)));
            let rows = values.div_ceil(row.max(1)).max(1) as usize;
            read.extend((first..first + rows).map(|lane| (lane, name)));
        }
    }
    Ok(seats.iter().enumerate().find_map(|(lane, offered)| {
        offered
            .iter()
            .find(|seat| !read.contains(&(lane, seat.output)))
            .map(|seat| (seat.output.to_string(), lane))
    }))
}

pub fn local_of(package: &LaunchPackage, at: usize, global: u32) -> Result<Option<u32>> {
    match package.plan_local(global) {
        Some((stage, local)) if stage == at => Ok(local),
        Some((stage, _)) => Err(Fault::Program {
            at: "guest::run",
            why: format!("value {global} is stage {stage}'s and this is stage {at}"),
        }),
        None => Err(Fault::Program {
            at: "guest::run",
            why: format!("value {global} belongs to no stage of this package"),
        }),
    }
}

pub fn root_bytes(root: &LaunchValue, value: &Value, span: u64) -> Vec<u8> {
    let bytes = to_heap(value);
    if root.source != ValueOrigin::Const {
        return bytes;
    }
    let lane = match value {
        Value::Bool(_) => 1,
        Value::Bf16(_) => 2,
        _ => 4,
    };
    let span = usize::try_from(span).unwrap_or(0);
    if bytes.len() != lane || span <= lane || !span.is_multiple_of(lane) {
        return bytes;
    }
    bytes.repeat(span / lane)
}

pub fn to_heap(value: &Value) -> Vec<u8> {
    match value {
        Value::F32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::I32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::U32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::Bool(cells) => cells.iter().map(|&b| u8::from(b)).collect(),
        Value::Bf16(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

pub fn from_heap(raw: &[u8], dtype: Dtype, held: u64) -> Value {
    let word = |i: usize| {
        let at = i * 4;
        u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
    };
    match dtype {
        Dtype::Bool => Value::Bool(raw.iter().map(|&b| b != 0).collect()),
        Dtype::I32 => Value::I32((0..(held / 4) as usize).map(|i| word(i) as i32).collect()),
        Dtype::U32 => Value::U32((0..(held / 4) as usize).map(word).collect()),
        Dtype::Bf16 => Value::Bf16(
            (0..(held / 2) as usize)
                .map(|i| u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]))
                .collect(),
        ),
        Dtype::F32 => Value::F32(
            (0..(held / 4) as usize)
                .map(|i| f32::from_bits(word(i)))
                .collect(),
        ),
        other => panic!("{other:?} is not a dtype a guest heap holds"),
    }
}
