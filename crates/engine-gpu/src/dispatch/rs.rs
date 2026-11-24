use kernels_slang::{Error, Tensor};
use poem_ir::ValueId;

use crate::backend::{Backend, Buffer as _};
use crate::rs::Seat;
use crate::run::Run;

fn extended_origin(
    indptr: &[i32],
    lanes: &[engine::rs::LanePlan],
    lane0: u32,
    r: usize,
) -> (u32, u32) {
    let mut begin = indptr[r].max(0) as u32;
    for j in 0..r {
        begin += lanes.get(lane0 as usize + j).map_or(0, |plan| plan.replay);
    }
    let replay = lanes.get(lane0 as usize + r).map_or(0, |plan| plan.replay);
    (begin, replay)
}

/// The arena buffers a fire's batched recurrent copies read and write, each
/// bound whole ([`kernels_slang::gather::rows::copy_runs`]): at most two, by
/// their buffer.
#[derive(Default)]
struct Arenas {
    bound: Vec<(u64, Tensor)>,
}

impl<B: Backend> Run<'_, B> {
    /// Where byte `at` of handle `handle`'s binding lies among `arenas`: its
    /// arena buffer (bound whole, the first two a fire names) and its word.
    fn arena_place(
        &self,
        op: &'static str,
        arenas: &mut Arenas,
        (handle, at): (u32, u64),
        bytes: u64,
    ) -> Result<(kernels_slang::gather::rows::Space, u32), Error> {
        use kernels_slang::gather::rows::Space;
        let (buffer, base) = {
            let binding = self.handles().get(handle).ok_or_else(|| Error::Backend {
                op,
                detail: format!("handle {handle}, which no row answers"),
            })?;
            if at.saturating_add(bytes) > binding.extent() {
                return Err(Error::Backend {
                    op,
                    detail: format!(
                        "a run of {bytes} bytes at {at} passes the {} bytes handle {handle} is \
                         bound at",
                        binding.extent()
                    ),
                });
            }
            (binding.buffer().clone(), binding.offset())
        };
        let first = base + at;
        if !first.is_multiple_of(4) {
            return Err(Error::Backend {
                op,
                detail: format!("a run at byte {first}, which is not a whole word"),
            });
        }
        let word = u32::try_from(first / 4).map_err(|_| Error::Backend {
            op,
            detail: format!("a run at byte {first}, past what a word offset addresses"),
        })?;
        let slab = buffer.slab_id();
        let place = match arenas.bound.iter().position(|(id, _)| *id == slab) {
            Some(at) => at,
            None if arenas.bound.len() < 2 => {
                let reservation = self
                    .handles()
                    .bind(&buffer, 0, buffer.bytes())
                    .map_err(|fault| Self::kernel(fault, op))?;
                let words = u32::try_from(buffer.bytes() / 4).unwrap_or(u32::MAX);
                arenas.bound.push((
                    slab,
                    Tensor::new(reservation, words, 1, poem_ir::Dtype::U32),
                ));
                arenas.bound.len() - 1
            }
            None => {
                return Err(Error::Backend {
                    op,
                    detail: "a fire's recurrent copies name a third arena buffer; they bind two"
                        .to_string(),
                });
            }
        };
        Ok((if place == 0 { Space::ArenaA } else { Space::ArenaB }, word))
    }

    /// The runs of `run`'s rows of `plane` between the pool and `rect` from
    /// its row `row0`, into the pool where `into_slab`, out of it elsewhere.
    #[allow(clippy::too_many_arguments)]
    fn rs_runs(
        &self,
        op: &'static str,
        seat: &Seat,
        plane: usize,
        run: &engine::rs::Run,
        rect: Tensor,
        row0: u32,
        into_slab: bool,
        arenas: &mut Arenas,
        runs: &mut Vec<kernels_slang::gather::rows::CopyRun>,
    ) -> Result<(), Error> {
        use kernels_slang::gather::rows::{CopyRun, Space};
        let row_bytes = seat.layout.planes[plane].row_bytes;
        let mut done = 0u32;
        while done < run.count {
            let (segment, at, room) = seat
                .locate_segment(run, plane, run.from + done)
                .map_err(|fault| Self::kernel(fault, op))?;
            let take = room.min(run.count - done);
            let bytes = u64::from(take) * row_bytes;
            let rows_at = u64::from(row0 + done) * row_bytes;
            let arena = self.arena_place(op, arenas, (rect.buf, rows_at), bytes)?;
            if !at.is_multiple_of(4) || !bytes.is_multiple_of(4) {
                return Err(Error::Backend {
                    op,
                    detail: format!("a pool run of {bytes} bytes at {at}, not whole words"),
                });
            }
            let pool = (Space::Segment(segment), u32::try_from(at / 4).unwrap_or(u32::MAX));
            let words = u32::try_from(bytes / 4).unwrap_or(u32::MAX);
            let (from, to) = if into_slab { (arena, pool) } else { (pool, arena) };
            runs.push(CopyRun { from, to, words });
            done += take;
        }
        Ok(())
    }

    /// Moves `runs` in one dispatch a [`kernels_slang::gather::rows::COPY_RUNS`].
    fn rs_flush(
        &self,
        seat: &Seat,
        arenas: &Arenas,
        runs: &[kernels_slang::gather::rows::CopyRun],
    ) -> Result<(), Error> {
        // A fire moving no run still dispatches its copies, empty, beside
        // the arena its planes lie in: its commands are its family's.
        let Some(&(_, a)) = arenas.bound.first() else {
            return Ok(());
        };
        let b = arenas.bound.get(1).map_or(a, |&(_, b)| b);
        kernels_slang::gather::rows::copy_runs(self.ctx, a, b, seat.pool, runs, seat.run_dispatches)
    }

    fn kernel(fault: crate::error::Fault, op: &'static str) -> Error {
        Error::Backend {
            op,
            detail: fault.to_string(),
        }
    }

    pub(crate) fn rs_committed(&self, seat: &Seat) -> kernels_slang::ssm::Committed {
        kernels_slang::ssm::Committed {
            replay: seat.replay,
            commit: seat.commit,
            slots: seat.slots,
            lane0: self.window().span.lane_offset,
        }
    }

    pub(crate) fn rs_extend(
        &self,
        op: &'static str,
        seat: &Seat,
        value: ValueId,
    ) -> Result<Tensor, Error> {
        let plane = *seat.layout.in_of.get(&value.0).ok_or_else(|| Error::Backend {
            op,
            detail: format!(
                "value {} is no recurrence input this load buffers (the layout read {} plane(s))",
                value.0,
                seat.layout.planes.len()
            ),
        })?;
        let ext = seat.ext_in[plane].ok_or_else(|| Error::Backend {
            op,
            detail: format!(
                "value {} lies in a plane read where it lies, which no fire extends",
                value.0
            ),
        })?;
        let own = self.tensor(value);
        let row_bytes = seat.layout.planes[plane].row_bytes;
        let indptr = self.qo_indptr_host();
        let lane0 = self.window().span.lane_offset;
        let mut arenas = Arenas::default();
        let mut runs = Vec::new();
        for r in 0..indptr.len().saturating_sub(1) {
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let (begin, replay) = extended_origin(indptr, &seat.lanes, lane0, r);
            let Some(plan) = seat.lanes.get(lane0 as usize + r) else {
                continue;
            };
            // The buffered slots the lane's lineage replays, run after run
            // in slot order: every one of its set's (`Lineage::Set`), or the
            // ones its mask admits (`Lineage::Admitted`).
            let mut at = begin;
            for run in &plan.gathers {
                self.rs_runs(op, seat, plane, run, ext, at, false, &mut arenas, &mut runs)?;
                at += run.count;
            }
            let own_at = u64::from(indptr[r].max(0) as u32) * row_bytes;
            let bytes = u64::from(rows) * row_bytes;
            if bytes > 0 {
                let from = self.arena_place(op, &mut arenas, (own.buf, own_at), bytes)?;
                let to = self.arena_place(
                    op,
                    &mut arenas,
                    (ext.buf, u64::from(begin + replay) * row_bytes),
                    bytes,
                )?;
                runs.push(kernels_slang::gather::rows::CopyRun {
                    from,
                    to,
                    words: u32::try_from(bytes / 4).unwrap_or(u32::MAX),
                });
            }
            if let Some(run) = &plan.scatter {
                let row0 = indptr[r].max(0) as u32;
                self.rs_runs(op, seat, plane, run, own, row0, true, &mut arenas, &mut runs)?;
            }
        }
        // Every lane's runs in one dispatch: however its rows lie in the
        // pool, the fire's commands are the same.
        self.rs_flush(seat, &arenas, &runs)?;
        Ok(ext)
    }

    pub(crate) fn rs_out(
        &self,
        op: &'static str,
        seat: &Seat,
        value: ValueId,
    ) -> Result<Tensor, Error> {
        let region = *seat
            .layout
            .out_of
            .get(&value.0)
            .ok_or_else(|| Error::Backend {
                op,
                detail: format!(
                    "value {} is no recurrence output this load extends",
                    value.0
                ),
            })?;
        let ext = *seat.ext_out.get(region).ok_or_else(|| Error::Backend {
            op,
            detail: format!(
                "value {} lands extended in a fire whose lanes all carry their rows, which \
                 extends none",
                value.0
            ),
        })?;
        seat.ext.borrow_mut().insert(value.0, ext);
        Ok(ext)
    }

    pub(crate) fn rs_ext_of(
        &self,
        op: &'static str,
        seat: &Seat,
        value: ValueId,
    ) -> Result<Tensor, Error> {
        seat.ext
            .borrow()
            .get(&value.0)
            .copied()
            .ok_or_else(|| Error::Backend {
                op,
                detail: format!(
                    "value {} was not landed extended by an earlier recurrent op of this window",
                    value.0
                ),
            })
    }

    pub(crate) fn rs_land(
        &self,
        op: &'static str,
        seat: &Seat,
        ext: Tensor,
        dest: ValueId,
    ) -> Result<(), Error> {
        let target = self.tensor(dest);
        let row_bytes = u64::from(ext.width)
            * poem_compiler::arena::elem_bytes(ext.dtype).ok_or_else(|| Error::Backend {
                op,
                detail: format!("{:?} has no element size", ext.dtype),
            })?;
        let indptr = self.qo_indptr_host();
        let lane0 = self.window().span.lane_offset;
        let mut arenas = Arenas::default();
        let mut runs = Vec::new();
        for r in 0..indptr.len().saturating_sub(1) {
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let (begin, replay) = extended_origin(indptr, &seat.lanes, lane0, r);
            let bytes = u64::from(rows) * row_bytes;
            if bytes == 0 {
                continue;
            }
            let from = self.arena_place(
                op,
                &mut arenas,
                (ext.buf, u64::from(begin + replay) * row_bytes),
                bytes,
            )?;
            let to = self.arena_place(
                op,
                &mut arenas,
                (target.buf, u64::from(indptr[r].max(0) as u32) * row_bytes),
                bytes,
            )?;
            runs.push(kernels_slang::gather::rows::CopyRun {
                from,
                to,
                words: u32::try_from(bytes / 4).unwrap_or(u32::MAX),
            });
        }
        self.rs_flush(seat, &arenas, &runs)
    }
}
