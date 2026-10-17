use engine::fire::{LayerScores, Readout};

use crate::error::{Fault, Result};

use super::{Enqueued, Shell};

#[derive(Debug, Clone)]
pub(crate) struct Readback {
    pub(crate) lanes: Vec<Option<(engine::fire::ReadoutSeam, kernels_rocm::Tensor)>>,
    pub(crate) columns: Vec<(u32, kernels_rocm::Tensor)>,
    pub(crate) last_row: Vec<u32>,
    pub(crate) first_row: Vec<u32>,
    pub(crate) lane_rows: Vec<u32>,
    pub(crate) captures: Vec<bool>,
    pub(crate) pixels: Option<PixelsReadback>,
}

#[derive(Debug, Clone)]
pub(crate) struct PixelsReadback {
    pub(crate) plane: kernels_rocm::Tensor,
    pub(crate) grid: kernels_rocm::Tensor,
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

struct Tap(*const u32);

// SAFETY: the tap is the shell's, alive as long as it is, and read only by
// the host function of the step that filled it, before that step's event is
// given back for another to claim.
unsafe impl Send for Tap {}

fn written(reports: Vec<super::KvReport>, tap: Option<Tap>) -> Vec<engine::KvWrites> {
    let folds_from = tap_rows(&reports);
    let mut folds = 0usize;
    reports
        .into_iter()
        .map(|report| {
            let rows = report.positions.len();
            let tokens = report.tokens.unwrap_or_else(|| match &tap {
                // SAFETY: the copy into the tap was ordered before the event
                // this host function waits on, and spans these rows.
                Some(Tap(host)) => unsafe {
                    std::slice::from_raw_parts(host.add(report.first_row as usize), rows).to_vec()
                },
                None => Vec::new(),
            });
            let fold = report.fold.or_else(|| {
                report.fold_at?;
                let Tap(host) = tap.as_ref()?;
                folds += 1;
                // SAFETY: as above; each read-back count follows the rows.
                Some(unsafe { *host.add(folds_from + folds - 1) })
            });
            engine::KvWrites {
                tag: report.tag,
                tokens,
                positions: report.positions,
                pages: report.pages,
                offsets: report.offsets,
                fold,
            }
        })
        .collect()
}

fn tap_rows(reports: &[super::KvReport]) -> usize {
    if reports.iter().all(|report| report.tokens.is_some()) {
        return 0;
    }
    reports
        .iter()
        .map(|report| report.first_row as usize + report.positions.len())
        .max()
        .unwrap_or(0)
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

        let reports = if done.is_some() {
            std::mem::take(&mut prepared.kv_reports)
        } else {
            Vec::new()
        };
        let slot = prepared.slot.take();
        drop(prepared);

        let at = self.settlement.claim()?;
        let tap = match self.tap_tokens(at, &reports) {
            Ok(tap) => tap,
            Err(fault) => {
                self.settlement.recycler().give(at);
                drop(slot);
                return Err(fault);
            }
        };
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
            // Read before the event goes back: its tap is the next step's then.
            let outcome = if reports.is_empty() {
                engine::StepOutcome::Committed
            } else {
                engine::StepOutcome::Wrote(written(reports, tap))
            };
            drop(slot);
            recycler.give(at);
            airborne.leave();
            if let Some(done) = done {
                (done.sink)(done.at, outcome);
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

    /// In stream order, before the next fire writes its own over them.
    fn tap_tokens(&mut self, at: u32, reports: &[super::KvReport]) -> Result<Option<Tap>> {
        let cells: Vec<u64> = reports.iter().filter_map(|report| report.fold_at).collect();
        if reports.iter().all(|report| report.tokens.is_some()) && cells.is_empty() {
            return Ok(None);
        }
        let rows = tap_rows(reports);
        let bytes = (rows + cells.len()) * 4;
        let at = at as usize;
        if self.taps.len() <= at {
            self.taps.resize_with(at + 1, || None);
        }
        if self.taps[at].as_ref().is_none_or(|tap| tap.bytes() < bytes) {
            let floor = self.budget.max_tokens as usize * 4;
            self.taps[at] = Some(crate::device::Pinned::mapped_uninit(bytes.max(floor))?);
        }
        let tap = self.taps[at].as_ref().expect("made just above");
        if rows > 0 {
            crate::device::copy_d2d(
                self.device.stream(),
                tap.device(),
                self.inputs.tokens_at(),
                rows * 4,
            )?;
        }
        for (index, &cell) in cells.iter().enumerate() {
            crate::device::copy_d2d(
                self.device.stream(),
                tap.device() + ((rows + index) * 4) as u64,
                cell,
                4,
            )?;
        }
        Ok(Some(Tap(tap.host().cast::<u32>().cast_const())))
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
                || kernels_rocm::Tensor::new(0, 0, 0, poem_ir::Dtype::Bf16),
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
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|word| f32::from_le_bytes(*word)),
                    );
                } else {
                    values.extend(
                        raw.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|pair| bf16(u16::from_le_bytes(*pair))),
                    );
                }
            }
            counts[lane] = u32::try_from(chosen.len()).unwrap_or(u32::MAX);
            taken[lane] = values;
        }

        let mut scores: Vec<Vec<LayerScores>> = vec![Vec::new(); lanes];
        if !readback.columns.is_empty() {
            let mut mass: Vec<u8> = Vec::new();
            for (lane, scored) in scores.iter_mut().enumerate() {
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
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|word| f32::from_le_bytes(*word))
                            .collect(),
                    });
                }
                *scored = layers;
            }
        }

        let mut pixels: Vec<super::Pixels> = vec![(Vec::new(), Vec::new()); lanes];
        if let Some(seat) = readback.pixels.as_ref() {
            let clips = seat.grid.rows as usize;
            let mut grid = vec![0u8; clips * 16];
            self.arena.read(seat.grid.ptr, &mut grid)?;
            let grid: Vec<i32> = grid
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| i32::from_le_bytes(*w))
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
                            raw.as_chunks::<4>()
                                .0
                                .iter()
                                .map(|w| f32::from_le_bytes(*w)),
                        ),
                        _ => values.extend(
                            raw.as_chunks::<2>()
                                .0
                                .iter()
                                .map(|pair| bf16(u16::from_le_bytes(*pair))),
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
