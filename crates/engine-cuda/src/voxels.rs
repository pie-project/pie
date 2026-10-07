use poem_compiler::VoxelLadder;
use poem_exec::fire::LaneRow;
use poem_ir::Dtype;

use crate::device::Buffer;
use crate::error::{Fault, Result};
use kernels_cuda::Tensor;

#[derive(Debug, Clone)]
pub struct Seat {
    pub rows: u64,
    pub clips: u64,
    pub channels: u32,
    pub widths: Vec<u32>,
    pub dtype: Dtype,
    pub token_patch: Option<[u32; 3]>,
}

impl Seat {
    fn elem_bytes(&self) -> u64 {
        poem_compiler::arena::elem_bytes(self.dtype).unwrap_or(0)
    }

    fn row_bytes(&self) -> u64 {
        u64::from(self.channels) * self.elem_bytes()
    }

    #[must_use]
    pub fn of(trace: &poem_ir::Trace, ladder: &VoxelLadder) -> Seat {
        let mut channels = 0u32;
        let mut widths: Vec<u32> = Vec::new();
        let mut dtype = Dtype::Bf16;
        let mut token_patch = None;
        for decl in &trace.values {
            match (&decl.def, &decl.ty) {
                (
                    poem_ir::Def::Input(poem_ir::RuntimeInput::Voxels { channels: c, .. }),
                    poem_ir::Ty::Tensor { dtype: d, .. },
                ) => {
                    channels = channels.max(*c);
                    if !widths.contains(c) {
                        widths.push(*c);
                    }
                    dtype = *d;
                }
                (poem_ir::Def::Input(poem_ir::RuntimeInput::TokenGrid { p }), _) => {
                    token_patch = Some(*p);
                }
                _ => {}
            }
        }
        widths.sort_unstable();
        Seat {
            rows: u64::from(ladder.max_voxels),
            clips: u64::from(ladder.max_clips),
            channels,
            widths,
            dtype,
            token_patch,
        }
    }
}

pub struct Store {
    seat: Seat,
    buffer: Buffer,
    grid: u64,
    token_grid: u64,
    slots: u64,
    payload: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Handles {
    pub grid: Tensor,
    pub token_grid: Option<Tensor>,
    pub voxels: Option<Tensor>,
    pub slots: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct Clips<'a> {
    pub lane: u32,
    pub clips: &'a [[u32; 3]],
    pub payload: &'a [u8],
}

impl Clips<'_> {
    #[must_use]
    pub fn voxels(&self) -> u64 {
        self.clips
            .iter()
            .map(|[t, h, w]| u64::from(*t) * u64::from(*h) * u64::from(*w))
            .sum()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Tables {
    pub grid: Vec<i32>,
    pub token_grid: Vec<i32>,
    pub payload: Vec<u8>,
    pub channels: u32,
    pub rows: u64,
    pub slots: Vec<i32>,
}

impl Tables {
    pub fn of(
        seat: &Seat,
        lanes: &[LaneRow],
        of_lane: &[Option<Clips<'_>>],
        slot_of: &[u32],
        fed: u32,
    ) -> Result<Tables> {
        let clips_total: usize = lanes.iter().map(|lane| lane.clips as usize).sum();
        let voxels_total: usize = lanes.iter().map(|lane| lane.voxels as usize).sum();
        let elem = seat.elem_bytes() as usize;
        let mut channels = 0u32;
        let host_fed = of_lane
            .iter()
            .flatten()
            .any(|shot| !shot.payload.is_empty());
        if seat.channels > 0 && !host_fed {
            if fed != 0 && !seat.widths.contains(&fed) {
                return Err(Fault::VoxelPayload {
                    lane: lanes.first().map_or(0, |lane| lane.source),
                    what: "the channel-fed voxel port is not a width the plan reads",
                });
            }
            channels = fed;
        }
        if seat.channels > 0 && host_fed {
            for lane in lanes {
                let Some(shot) = of_lane.get(lane.source as usize).copied().flatten() else {
                    continue;
                };
                let voxels = shot.voxels() as usize;
                let per_row = shot.payload.len().checked_div(voxels).unwrap_or(0);
                let width = u32::try_from(per_row / elem.max(1)).unwrap_or(u32::MAX);
                if elem == 0
                    || voxels == 0
                    || per_row * voxels != shot.payload.len()
                    || per_row % elem != 0
                    || !seat.widths.contains(&width)
                {
                    return Err(Fault::VoxelPayload {
                        lane: lane.source,
                        what: "the payload is not one port row per voxel of the clips at a \
                               width the plan reads",
                    });
                }
                if channels != 0 && channels != width {
                    return Err(Fault::VoxelPayload {
                        lane: lane.source,
                        what: "two lanes fed voxel ports of two widths in one fire (M0: one \
                               voxel class per fire)",
                    });
                }
                channels = width;
            }
        }
        let row_bytes = channels as usize * elem;
        let mut grid = vec![0i32; clips_total * 4];
        let mut token_grid = if seat.token_patch.is_some() {
            vec![0i32; clips_total * 4]
        } else {
            Vec::new()
        };
        let mut payload = if host_fed {
            vec![0u8; voxels_total * row_bytes]
        } else {
            Vec::new()
        };
        let mut slots = vec![0i32; clips_total];
        for lane in lanes {
            let Some(shot) = of_lane.get(lane.source as usize).copied().flatten() else {
                continue;
            };
            if shot.clips.len() != lane.clips as usize || shot.voxels() != u64::from(lane.voxels) {
                return Err(Fault::VoxelPayload {
                    lane: lane.source,
                    what: "the clips composed are not the clips submitted",
                });
            }
            let need = shot.voxels() as usize * row_bytes;
            if seat.channels > 0 && host_fed {
                let at = lane.voxel_offset as usize * row_bytes;
                payload[at..at + need].copy_from_slice(shot.payload);
            }
            let mut row: i64 = i64::from(lane.voxel_offset);
            let mut token_row: i64 = i64::from(lane.row_offset);
            let slot = slot_of.get(lane.source as usize).copied().unwrap_or(0) as i32;
            for (c, [t, h, w]) in shot.clips.iter().enumerate() {
                slots[lane.clip_offset as usize + c] = slot;
                let at = (lane.clip_offset as usize + c) * 4;
                grid[at..at + 4].copy_from_slice(&[*t as i32, *h as i32, *w as i32, row as i32]);
                row += i64::from(*t) * i64::from(*h) * i64::from(*w);
                if let Some(p) = seat.token_patch {
                    if p.contains(&0) || t % p[0] != 0 || h % p[1] != 0 || w % p[2] != 0 {
                        return Err(Fault::VoxelPayload {
                            lane: lane.source,
                            what: "a clip's box does not divide by the plan's token patch",
                        });
                    }
                    let [tt, th, tw] = [t / p[0], h / p[1], w / p[2]];
                    token_grid[at..at + 4].copy_from_slice(&[
                        tt as i32,
                        th as i32,
                        tw as i32,
                        token_row as i32,
                    ]);
                    token_row += i64::from(tt) * i64::from(th) * i64::from(tw);
                }
            }
            if seat.token_patch.is_some()
                && token_row != i64::from(lane.row_offset) + i64::from(lane.rows)
            {
                return Err(Fault::VoxelPayload {
                    lane: lane.source,
                    what: "the lane's token rows are not its clips' token count",
                });
            }
        }
        Ok(Tables {
            grid,
            token_grid,
            payload,
            channels,
            rows: voxels_total as u64,
            slots,
        })
    }
}

impl Store {
    pub fn reserve(seat: Seat) -> Result<Store> {
        let align = |bytes: u64| bytes.next_multiple_of(256);
        let grid = 0u64;
        let token_grid = align(seat.clips * 16);
        let slots = token_grid
            + if seat.token_patch.is_some() {
                align(seat.clips * 16)
            } else {
                0
            };
        let payload = slots + align(seat.clips * 4);
        let total = payload + align(seat.rows * seat.row_bytes());
        Ok(Store {
            seat,
            buffer: Buffer::zeroed(total as usize)?,
            grid,
            token_grid,
            slots,
            payload,
        })
    }

    #[must_use]
    pub fn seat(&self) -> Seat {
        self.seat.clone()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buffer.bytes() as u64
    }

    pub fn stage(&mut self, stream: *mut core::ffi::c_void, tables: &Tables) -> Result<Handles> {
        let clips = (tables.grid.len() / 4) as u64;
        if clips > self.seat.clips {
            return Err(Fault::Ceiling {
                what: "clips",
                need: clips,
                have: self.seat.clips,
            });
        }
        let rows = tables.rows;
        if rows > self.seat.rows {
            return Err(Fault::Ceiling {
                what: "voxel rows",
                need: rows,
                have: self.seat.rows,
            });
        }
        let base = self.buffer.ptr();
        self.buffer
            .stage(stream, self.grid, i32_bytes(&tables.grid))?;
        if !tables.token_grid.is_empty() {
            self.buffer
                .stage(stream, self.token_grid, i32_bytes(&tables.token_grid))?;
        }
        self.buffer
            .stage(stream, self.slots, i32_bytes(&tables.slots))?;
        if !tables.payload.is_empty() {
            self.buffer.stage(stream, self.payload, &tables.payload)?;
        }
        Ok(Handles {
            grid: Tensor::new(base + self.grid, clips as u32, 4, Dtype::I32),
            token_grid: (!tables.token_grid.is_empty())
                .then(|| Tensor::new(base + self.token_grid, clips as u32, 4, Dtype::I32)),
            voxels: (tables.channels > 0).then(|| {
                Tensor::new(
                    base + self.payload,
                    rows as u32,
                    tables.channels,
                    self.seat.dtype,
                )
            }),
            slots: Tensor::new(base + self.slots, clips as u32, 1, Dtype::I32),
        })
    }
}

#[must_use]
pub(crate) fn host_grid(
    trace: &poem_ir::Trace,
    port_grid: &[i32],
    grid: poem_ir::ValueId,
) -> Option<Vec<i32>> {
    use poem_ir::{Def, Operation, RuntimeInput, Spatial};

    let mut rules = Vec::new();
    let mut at = grid;
    loop {
        if matches!(
            trace.values[at.0 as usize].def,
            Def::Input(RuntimeInput::Grid)
        ) {
            break;
        }
        let step = trace.nodes.iter().find_map(|node| match &node.op {
            Operation::Spatial(Spatial::Grid { grid, rule, y }) if *y == at => Some((*grid, *rule)),
            _ => None,
        })?;
        rules.push(step.1);
        at = step.0;
        if rules.len() > trace.values.len() {
            return None;
        }
    }
    rules.reverse();
    let mut table = port_grid.to_vec();
    for rule in rules {
        table = rule.apply(&table)?;
    }
    Some(table)
}

fn i32_bytes(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` has no padding and any bit pattern is a valid byte.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

pub fn port_bytes(values: &[f32], element: Dtype) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        Dtype::Bf16 => Ok(values
            .iter()
            .flat_map(|&v| crate::adapter::bf16_bits(v).to_le_bytes())
            .collect()),
        Dtype::F32 => Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => {
            Err("a voxel submission against a plan whose port element is neither `bf16` nor `f32`")
        }
    }
}

pub(crate) fn relabel_conv_weights(
    device: &crate::device::Context,
    trace: &poem_ir::Trace,
    weights: &crate::run::WeightTable,
) -> Result<()> {
    for (at, param) in trace.params.iter().enumerate() {
        let poem_ir::ParamLayout::ConvTapsMajor { c_in, taps } = param.layout else {
            continue;
        };
        let Some(crate::run::WeightRow::Dense(handle)) = weights.0.get(at).copied().flatten()
        else {
            return Err(Fault::Unbound {
                what: format!(
                    "`{}`, a convolution weight that did not land as one dense plane",
                    param.name
                ),
            });
        };
        let bytes = usize::try_from(u64::from(handle.rows) * u64::from(handle.width) * 2)
            .unwrap_or(usize::MAX);
        let aside = Buffer::zeroed(bytes)?;
        crate::device::alloc::copy_d2d(device.stream(), aside.ptr(), handle.ptr, bytes)?;
        let natural = Tensor::new(aside.ptr(), handle.rows, handle.width, handle.dtype);
        let mut relabelled = handle;
        kernels_cuda::spatial::conv_weight_taps_major(
            device.ctx(),
            natural,
            c_in,
            taps,
            &mut relabelled,
        )
        .map_err(|fault| Fault::Unbound {
            what: format!("`{}`: {fault}", param.name),
        })?;
        device.synchronize()?;
    }
    Ok(())
}
