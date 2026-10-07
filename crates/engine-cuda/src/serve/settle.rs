use engine::fire::{LayerScores, Readout};

use crate::error::{Fault, Result};

use super::{Enqueued, Shell};

#[derive(Debug, Clone)]
pub(crate) struct Readback {
    pub(crate) lanes: Vec<Option<(engine::fire::ReadoutSeam, kernels_cuda::Tensor)>>,
    pub(crate) columns: Vec<(u32, kernels_cuda::Tensor)>,
    pub(crate) last_row: Vec<u32>,
    pub(crate) first_row: Vec<u32>,
    pub(crate) lane_rows: Vec<u32>,
    pub(crate) captures: Vec<bool>,
    pub(crate) pixels: Option<PixelsReadback>,
}

#[derive(Debug, Clone)]
pub(crate) struct PixelsReadback {
    pub(crate) plane: kernels_cuda::Tensor,
    pub(crate) grid: kernels_cuda::Tensor,
    pub(crate) lane_clips: Vec<(u32, u32)>,
}

#[derive(Debug, Default)]
pub struct Settled {
    pub logits: Vec<Vec<f32>>,
    pub rows: Vec<u32>,
    pub scores: Vec<Vec<LayerScores>>,
    pub seams: Vec<engine::fire::ReadoutSeam>,
    pub pixels: Vec<super::Pixels>,
    pub(super) readback: Option<Readback>,
}

pub struct Done {
    pub at: engine::StepDone,
    pub sink: engine::CompletionSink,
}

impl Shell {
    pub fn settle_step<'a>(&mut self, enqueued: Enqueued<'a>, done: Option<Done>) -> Result<Settled>
    where
        Self: 'a,
    {
        let Enqueued {
            mut prepared,
            launches: _,
            readback,
        } = enqueued;

        let slot = prepared.slot.take();
        drop(prepared);

        let at = self.settlement.claim()?;
        let airborne = self.airborne.clone();
        airborne.enter();

        let ordered = self
            .settlement
            .event(at)
            .record(self.device.stream())
            .and_then(|()| self.settlement.event(at).wait(self.device.notify_stream()));
        if let Err(fault) = ordered {
            let _ = self.device.synchronize();
            airborne.abandon();
            self.settlement.recycler().give(at);
            drop(slot);
            return Err(fault);
        }

        if let Some(tier) = self.weights.experts() {
            let _ = tier.drain(self.device.notify_stream());
        }

        let recycler = self.settlement.recycler();
        let posted = self.device.host_fn(Box::new(move || {
            drop(slot);
            recycler.give(at);
            airborne.leave();
            if let Some(done) = done {
                (done.sink)(done.at, engine::StepOutcome::Committed);
            }
        }));
        if let Err(fault) = posted {
            let _ = self.device.synchronize();
            self.airborne.abandon();
            self.settlement.recycler().give(at);
            return Err(fault);
        }

        Ok(Settled {
            logits: Vec::new(),
            rows: Vec::new(),
            scores: Vec::new(),
            seams: readback.as_ref().map_or_else(Vec::new, |readback| {
                readback
                    .lanes
                    .iter()
                    .map(|lane| lane.map_or(engine::fire::ReadoutSeam::Pixels, |(seam, _)| seam))
                    .collect()
            }),
            pixels: Vec::new(),
            readback,
        })
    }

    pub fn read_out(&mut self, settled: &mut Settled) -> Result<()> {
        self.read_out_rows(settled, &[])
    }

    pub fn read_out_rows(&mut self, settled: &mut Settled, want: &[Readout]) -> Result<()> {
        self.device.synchronize()?;
        let Some(readback) = settled.readback.as_ref() else {
            return Ok(());
        };

        let lanes = readback.last_row.len();
        let mut taken = vec![Vec::new(); lanes];
        let mut counts = vec![0u32; lanes];
        let mut raw = Vec::new();
        for lane in 0..lanes {
            let logits = readback.lanes[lane].map_or_else(
                || kernels_cuda::Tensor::new(0, 0, 0, poem_ir::Dtype::Bf16),
                |(_, plane)| plane,
            );
            let width = logits.width as usize;
            let element = if logits.dtype == poem_ir::Dtype::F32 {
                4
            } else {
                2
            };
            raw.clear();
            raw.resize(width * element, 0);
            let owned = readback.lane_rows[lane];
            if owned == 0 || logits.width == 0 {
                continue;
            }
            let chosen: Vec<u32> = match want.get(lane) {
                None | Some(Readout::Last) => vec![readback.last_row[lane]],
                Some(Readout::None) => Vec::new(),
                Some(Readout::Rows(rows)) => {
                    if rows.len() as u32 > owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: rows.len() as u64,
                            have: u64::from(owned),
                        });
                    }
                    (0..rows.len() as u32)
                        .map(|i| readback.first_row[lane] + i)
                        .collect()
                }
            };
            let mut values = Vec::with_capacity(chosen.len() * width);
            for row in &chosen {
                self.arena.read(
                    logits.ptr + u64::from(*row) * width as u64 * element as u64,
                    &mut raw,
                )?;
                if element == 4 {
                    values.extend(
                        raw.chunks_exact(4)
                            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]])),
                    );
                } else {
                    values.extend(
                        raw.chunks_exact(2)
                            .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]]))),
                    );
                }
            }
            counts[lane] = u32::try_from(chosen.len()).unwrap_or(u32::MAX);
            taken[lane] = values;
        }

        let mut scores: Vec<Vec<LayerScores>> = vec![Vec::new(); lanes];
        if !readback.columns.is_empty() {
            let mut mass: Vec<u8> = Vec::new();
            for lane in 0..lanes {
                if !readback.captures[lane] {
                    continue;
                }
                let rows = readback.lane_rows[lane];
                let first = readback.first_row[lane];
                let mut layers = Vec::with_capacity(readback.columns.len());
                for (layer, column) in &readback.columns {
                    let heads = column.width;
                    let bytes = rows as usize * heads as usize * 4;
                    mass.clear();
                    mass.resize(bytes, 0);
                    self.arena.read(
                        column.ptr + u64::from(first) * u64::from(heads) * 4,
                        &mut mass,
                    )?;
                    layers.push(LayerScores {
                        layer: *layer,
                        rows,
                        heads,
                        lse: mass
                            .chunks_exact(4)
                            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                            .collect(),
                    });
                }
                scores[lane] = layers;
            }
        }

        let mut pixels: Vec<super::Pixels> = vec![(Vec::new(), Vec::new()); lanes];
        if let Some(seat) = readback.pixels.as_ref() {
            let clips = seat.grid.rows as usize;
            let mut grid = vec![0u8; clips * 16];
            self.arena.read(seat.grid.ptr, &mut grid)?;
            let grid: Vec<i32> = grid
                .chunks_exact(4)
                .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let channels = seat.plane.width as usize;
            let element = poem_compiler::arena::elem_bytes(seat.plane.dtype).unwrap_or(0) as usize;
            let mut raw: Vec<u8> = Vec::new();
            for (&(first, count), answer) in seat.lane_clips.iter().zip(pixels.iter_mut()) {
                let mut values = Vec::new();
                let mut boxes = Vec::with_capacity(count as usize);
                for clip in first..first + count {
                    let at = clip as usize * 4;
                    let [t, h, w, off] = [grid[at], grid[at + 1], grid[at + 2], grid[at + 3]];
                    let voxels = t as usize * h as usize * w as usize;
                    boxes.push([t as u32, h as u32, w as u32]);
                    let bytes = voxels * channels * element;
                    raw.clear();
                    raw.resize(bytes, 0);
                    self.arena.read(
                        seat.plane.ptr + off as u64 * channels as u64 * element as u64,
                        &mut raw,
                    )?;
                    match seat.plane.dtype {
                        poem_ir::Dtype::F32 => values.extend(
                            raw.chunks_exact(4)
                                .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])),
                        ),
                        _ => values.extend(
                            raw.chunks_exact(2)
                                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]]))),
                        ),
                    }
                }
                *answer = (values, boxes);
            }
        }

        settled.logits = taken;
        settled.rows = counts;
        settled.scores = scores;
        settled.pixels = pixels;
        Ok(())
    }
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
