#![allow(unsafe_code)]
use engine::fire::Readout;
use kernels_cuda::runtime::channel::WordRun;

use crate::error::{Fault, Result};
use crate::program::Fired;

use super::{Enqueued, Shell};

#[derive(Debug, Clone)]
pub(crate) struct Readback {
    /// The output each lane reads back, and where its rows are when it is
    /// not laid out by clips of its own.
    pub(crate) names: Vec<String>,
    pub(crate) lanes: Vec<Option<(String, kernels_cuda::Tensor)>>,
    /// Each lane's rows of its output's plane, in the order its readout
    /// names them: readout-gathered outputs at the lane's readout indices,
    /// row outputs at its own token rows.
    pub(crate) rows: Vec<Vec<u32>>,
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
    /// The output each lane read back.
    pub outputs: Vec<String>,
    pub pixels: Vec<super::Pixels>,
    pub(super) readback: Option<Readback>,
}

struct Tap(*const u32);

// SAFETY: the tap is the shell's, alive as long as it is, and read only by
// the host function of the step that filled it, before that step's event is
// given back for another to claim.
unsafe impl Send for Tap {}

fn written(reports: Vec<super::KvReport>, tap: Option<&Tap>) -> Vec<engine::KvWrites> {
    let folds_from = tap_rows(&reports);
    let mut folds = 0usize;
    reports
        .into_iter()
        .map(|report| {
            let rows = report.positions.len();
            let tokens = report.tokens.unwrap_or_else(|| match tap {
                // SAFETY: the copy into the tap was ordered before the event
                // this host function waits on, and spans these rows.
                Some(Tap(host)) => unsafe {
                    std::slice::from_raw_parts(host.add(report.first_row as usize), rows).to_vec()
                },
                None => Vec::new(),
            });
            let fold = report.fold.or_else(|| {
                report.fold_at?;
                let Tap(host) = tap?;
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
                end: report.end,
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

#[derive(Clone)]
pub struct Done {
    pub at: engine::StepDone,
    pub sink: engine::CompletionSink,
}

impl Shell {
    pub fn settle_step<'a>(&mut self, enqueued: Enqueued<'a>, done: Option<Done>) -> Result<Settled>
    where
        Self: 'a,
    {
        let built = kernels_cuda::tally::counts().built;
        let settled = self.settle_fire(enqueued, done);
        self.count_built_in_fire(built);
        settled
    }

    fn settle_fire<'a>(&mut self, enqueued: Enqueued<'a>, done: Option<Done>) -> Result<Settled>
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
        // Each beam group's common slot (§11.3), read after the fault words.
        let commons: Vec<(u64, engine::InstanceId)> =
            if done.is_some() && prepared.page_faults.is_some() {
                prepared.page_work.commons.clone()
            } else {
                Vec::new()
            };
        // The lanes whose own page lists the device resolved: a fault word
        // each, read with the commit words.
        let page_faults: Vec<(u64, usize, engine::InstanceId)> = match prepared.page_faults {
            Some(words) if done.is_some() => prepared
                .page_work
                .faults
                .iter()
                .enumerate()
                .map(|(index, &(lane, instance))| (words + index as u64 * 4, lane, instance))
                .collect(),
            _ => Vec::new(),
        };
        let lanes = u32::try_from(prepared.lanes.len()).unwrap_or(u32::MAX);
        // Its staging slot comes back as the step lands.
        let slot = prepared.slot.take().map(crate::inputs::SlotGuard::landing);
        drop(prepared);

        // The guest passes this step launched that land with it: their
        // commit words are read where the step lands, each faulting its own
        // lanes where it did not commit.
        let seq = self.airborne.next_seq();
        let mut flying: Vec<super::enqueue::Flying> = Vec::new();
        for batch in self
            .owed
            .iter_mut()
            .filter(|batch| batch.seq == seq && batch.told.is_none())
        {
            batch.told = Some(done.is_some());
            if done.is_some() {
                flying.extend(batch.launched.iter().cloned());
            }
        }
        let faults = std::mem::take(&mut self.guest_faults);
        if done.is_none()
            && let Some(fault) = faults.first()
        {
            let _ = self.device.synchronize();
            drop(slot);
            return Err(Fault::program("serve::fire", fault.reason.clone()));
        }

        // A failure past the enqueue is the step lost on the device: every
        // lane of it faults.
        let lost = |done: Option<Done>, fault: Fault| -> Result<()> {
            match done {
                Some(done) => {
                    (done.sink)(
                        done.at,
                        engine::StepOutcome::lost(lanes, format!("cuda step lost: {fault}")),
                    );
                    Ok(())
                }
                None => Err(fault),
            }
        };

        let commits: Vec<u64> = flying
            .iter()
            .map(|flying| flying.commit)
            .chain(page_faults.iter().map(|&(word, _, _)| word))
            .chain(commons.iter().map(|&(word, _)| word))
            .collect();
        let event = self.settlement.claim()?;
        let at = event.at();
        let tap = match self.tap_tokens(at, &reports, &commits) {
            Ok(tap) => tap,
            Err(fault) => {
                let _ = self.device.synchronize();
                drop(event);
                drop(slot);
                lost(done, fault)?;
                return Ok(Settled::default());
            }
        };
        let commits_from = tap_rows(&reports)
            + reports
                .iter()
                .filter(|report| report.fold_at.is_some())
                .count();
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
            drop(event);
            drop(slot);
            lost(done, fault)?;
            return Ok(Settled::default());
        }

        // Its settle event comes back as the step lands.
        let event = event.landing();
        let told = done.clone();
        let posted = self.device.host_fn(Box::new(move || {
            // Read before the event goes back: its tap is the next step's then.
            let mut outcome = if reports.is_empty() {
                engine::StepOutcome::committed()
            } else {
                engine::StepOutcome::wrote(written(reports, tap.as_ref()))
            };
            for fault in faults {
                outcome.fault_lanes(fault.lanes, Some(fault.instance), &fault.reason);
            }
            if let Some(Tap(host)) = tap.as_ref() {
                for (index, flying) in flying.iter().enumerate() {
                    // SAFETY: as in `written`; the commit words follow the
                    // fold counts.
                    let word = unsafe { *host.add(commits_from + index) };
                    if word == 0 {
                        let reason =
                            super::enqueue::fault_of(Fired::Declined, flying.instance, flying.at)
                                .expect("a declined pass faults");
                        outcome.fault_lanes(flying.lanes.clone(), Some(flying.instance), &reason);
                    }
                }
                for (index, &(_, instance)) in commons.iter().enumerate() {
                    // SAFETY: as above; the common slot words follow the
                    // fault words.
                    let word = unsafe {
                        *host.add(commits_from + flying.len() + page_faults.len() + index)
                    };
                    outcome.beam_common.push((instance, word));
                }
                for (index, &(_, lane, instance)) in page_faults.iter().enumerate() {
                    // SAFETY: as above; the page-list fault words follow the
                    // commit words.
                    let word = unsafe { *host.add(commits_from + flying.len() + index) };
                    if word != 0 {
                        let lane32 = lane as u32;
                        outcome.fault_lanes(
                            lane32..lane32 + 1,
                            Some(instance),
                            &super::pages::fault_reason(word, lane),
                        );
                    }
                }
            }
            drop(slot);
            drop(event);
            airborne.leave();
            if let Some(done) = done {
                (done.sink)(done.at, outcome);
            }
        }));
        if let Err(fault) = posted {
            let _ = self.device.synchronize();
            // The step's landing, never posted, gave its event back as it
            // dropped.
            self.airborne.abandon();
            lost(told, fault)?;
            return Ok(Settled::default());
        }

        Ok(Settled {
            logits: Vec::new(),
            rows: Vec::new(),
            outputs: readback
                .as_ref()
                .map_or_else(Vec::new, |readback| readback.names.clone()),
            pixels: Vec::new(),
            readback,
        })
    }

    /// In stream order, before the next fire writes its own over them.
    fn tap_tokens(
        &mut self,
        at: u32,
        reports: &[super::KvReport],
        commits: &[u64],
    ) -> Result<Option<Tap>> {
        let cells: Vec<u64> = reports
            .iter()
            .filter_map(|report| report.fold_at)
            .chain(commits.iter().copied())
            .collect();
        if reports.iter().all(|report| report.tokens.is_some()) && cells.is_empty() {
            return Ok(None);
        }
        let rows = tap_rows(reports);
        // The cells' gather table follows the words, in the tap itself.
        let table_at = ((rows + cells.len()) * 4).next_multiple_of(16);
        let bytes = table_at + cells.len() * size_of::<WordRun>();
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
        // One gather launch for every cell. A cell may be host memory (a
        // guest pass's commit word is a mapped pinned slot): a copy call
        // between two host spans runs on the host once the stream drains,
        // so it would hold the host a whole step behind the device.
        if !cells.is_empty() {
            let table: Vec<WordRun> = cells
                .iter()
                .enumerate()
                .map(|(index, &src)| WordRun {
                    src,
                    dst_word: u32::try_from(rows + index).unwrap_or(u32::MAX),
                    words: 1,
                })
                .collect();
            // SAFETY: the tap at `at` is this step's alone until its host
            // function gives it back, and it spans `bytes`; `WordRun` is
            // plain `repr(C)` data, written as its bytes before the launch
            // that reads it.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    table.as_ptr().cast::<u8>(),
                    tap.host().add(table_at),
                    table.len() * size_of::<WordRun>(),
                );
            }
            kernels_cuda::runtime::channel::gather_words(
                self.device.ctx(),
                tap.device() + table_at as u64,
                tap.device(),
                u32::try_from(table.len()).unwrap_or(u32::MAX),
            )?;
        }
        Ok(Some(Tap(tap.host().cast::<u32>().cast_const())))
    }

    pub fn read_out(&mut self, settled: &mut Settled) -> Result<()> {
        self.read_out_rows(settled, &[])
    }

    pub fn read_out_rows(&mut self, settled: &mut Settled, want: &[Readout]) -> Result<()> {
        self.device.synchronize()?;
        let stream = self.device.stream();
        let Some(readback) = settled.readback.as_ref() else {
            return Ok(());
        };

        let lanes = readback.rows.len();
        let mut taken = vec![Vec::new(); lanes];
        let mut counts = vec![0u32; lanes];
        let mut raw = Vec::new();
        for lane in 0..lanes {
            let logits = readback.lanes[lane].as_ref().map_or_else(
                || kernels_cuda::Tensor::new(0, 0, 0, poem_ir::Dtype::Bf16),
                |(_, plane)| *plane,
            );
            let width = logits.width as usize;
            let element = if logits.dtype == poem_ir::Dtype::F32 {
                4
            } else {
                2
            };
            raw.clear();
            raw.resize(width * element, 0);
            let planned = &readback.rows[lane];
            let Some(&last) = planned.last() else {
                continue;
            };
            if logits.width == 0 {
                continue;
            }
            let chosen: Vec<u32> = match want.get(lane) {
                None | Some(Readout::Last) => vec![last],
                Some(Readout::None) => Vec::new(),
                Some(Readout::Rows(rows)) => {
                    if rows.len() > planned.len() {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: rows.len() as u64,
                            have: planned.len() as u64,
                        });
                    }
                    planned[..rows.len()].to_vec()
                }
            };
            let mut values = Vec::with_capacity(chosen.len() * width);
            for row in &chosen {
                self.arena.read(
                    &self.pools,
                    stream,
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

        let mut pixels: Vec<super::Pixels> = vec![(Vec::new(), Vec::new()); lanes];
        if let Some(seat) = readback.pixels.as_ref() {
            let clips = seat.grid.rows as usize;
            let mut grid = vec![0u8; clips * 16];
            self.arena
                .read(&self.pools, stream, seat.grid.ptr, &mut grid)?;
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
                        &self.pools,
                        stream,
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
        settled.pixels = pixels;
        Ok(())
    }
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
