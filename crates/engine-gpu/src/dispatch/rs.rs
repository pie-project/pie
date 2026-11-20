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

impl<B: Backend> Run<'_, B> {
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

    fn rs_copy(
        &self,
        op: &'static str,
        src: (u32, u64),
        dst: (u32, u64),
        bytes: u64,
    ) -> Result<(), Error> {
        if bytes == 0 {
            return Ok(());
        }
        // Each end is its reservation bound whole from byte 0, the run a word
        // offset in it: the same view whatever binding or run a copy names,
        // so one group a pair of reservations. The run is refused past its
        // binding's bound range (as a cut of it would be), and `copy_words`
        // refuses it past the reservation.
        let whole = |(handle, at): (u32, u64)| -> Result<(Tensor, u32), Error> {
            let (buffer, base) = {
                let binding = self.handles().get(handle).ok_or_else(|| Error::Backend {
                    op,
                    detail: format!("handle {handle}, which no row answers"),
                })?;
                if at.saturating_add(bytes) > binding.extent() {
                    return Err(Error::Backend {
                        op,
                        detail: format!(
                            "a run of {bytes} bytes at {at} passes the {} bytes handle {handle}                              is bound at",
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
            let reservation = self
                .handles()
                .bind(&buffer, 0, buffer.bytes())
                .map_err(|fault| Self::kernel(fault, op))?;
            let words = u32::try_from(buffer.bytes() / 4).unwrap_or(u32::MAX);
            let first = u32::try_from(first / 4).map_err(|_| Error::Backend {
                op,
                detail: format!(
                    "a run at byte {}, past what a word offset addresses",
                    base + at
                ),
            })?;
            Ok((
                Tensor::new(reservation, words, 1, poem_ir::Dtype::U32),
                first,
            ))
        };
        let ((src, src_at), (dst, dst_at)) = (whole(src)?, whole(dst)?);
        kernels_slang::gather::rows::copy_words(self.ctx, src, src_at, dst, dst_at, bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn rs_move(
        &self,
        op: &'static str,
        seat: &Seat,
        plane: usize,
        run: &engine::rs::Run,
        rect: Tensor,
        row0: u32,
        into_slab: bool,
    ) -> Result<(), Error> {
        let row_bytes = seat.layout.planes[plane].row_bytes;
        let mut done = 0u32;
        while done < run.count {
            let (pool, at, room) = seat
                .locate(run, plane, run.from + done)
                .map_err(|fault| Self::kernel(fault, op))?;
            let take = room.min(run.count - done);
            let bytes = u64::from(take) * row_bytes;
            let rows_at = u64::from(row0 + done) * row_bytes;
            if into_slab {
                self.rs_copy(op, (rect.buf, rows_at), (pool, at), bytes)?;
            } else {
                self.rs_copy(op, (pool, at), (rect.buf, rows_at), bytes)?;
            }
            done += take;
        }
        Ok(())
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
        let ext = seat.ext_in[plane];
        let own = self.tensor(value);
        let row_bytes = seat.layout.planes[plane].row_bytes;
        let indptr = self.qo_indptr_host();
        let lane0 = self.window().span.lane_offset;
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
                self.rs_move(op, seat, plane, run, ext, at, false)?;
                at += run.count;
            }
            let own_at = u64::from(indptr[r].max(0) as u32) * row_bytes;
            self.rs_copy(
                op,
                (own.buf, own_at),
                (ext.buf, u64::from(begin + replay) * row_bytes),
                u64::from(rows) * row_bytes,
            )?;
            if let Some(run) = &plan.scatter {
                self.rs_move(op, seat, plane, run, own, indptr[r].max(0) as u32, true)?;
            }
        }
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
        let ext = seat.ext_out[region];
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
        for r in 0..indptr.len().saturating_sub(1) {
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let (begin, replay) = extended_origin(indptr, &seat.lanes, lane0, r);
            self.rs_copy(
                op,
                (ext.buf, u64::from(begin + replay) * row_bytes),
                (target.buf, u64::from(indptr[r].max(0) as u32) * row_bytes),
                u64::from(rows) * row_bytes,
            )?;
        }
        Ok(())
    }
}
