//! The device half of the recurrent-state verbs: a buffered lane's
//! recurrence runs over its extended rows (replayed buffer rows, then its
//! own), its own rows are stashed into the buffer, and the outputs of its own
//! rows land back where the plan reads them. Every move is a gather or a
//! scatter over the index maps staged for the window (`rs::WindowMap`).

use kernels_xla::attn::ssm::Committed;
use kernels_xla::{Error, Tensor};
use poem_ir::ValueId;

use crate::rs::{Seat, WindowInputs};
use crate::run::Run;
use crate::trace::{Root, Source};

impl Run<'_> {
    fn rs_window(&self, op: &'static str, seat: &Seat) -> Result<WindowInputs, Error> {
        let span = self.window().span;
        seat.maps
            .get(&(span.lane_offset, span.lanes, span.row_offset, span.rows))
            .copied()
            .ok_or_else(|| Error::Backend {
                op,
                detail: "the walk runs the recurrence over a window the fire staged no map for"
                    .to_string(),
            })
    }

    fn rs_temp(&self, rows: u32, width: u32, dtype: poem_ir::Dtype) -> Tensor {
        self.handles().root(Root {
            source: Source::Temp,
            dtype,
            rows: rows.max(1),
            width,
        })
    }

    pub(crate) fn rs_committed(&self, seat: &Seat) -> Committed {
        Committed {
            replay: seat.replay,
            commit: seat.commit,
            slots: seat.slots,
            lane0: self.window().span.lane_offset,
        }
    }

    /// `value`'s rows extended by each lane's replayed buffer rows; the
    /// lane's own rows are stashed into the buffer on the way.
    pub(crate) fn rs_extend(
        &self,
        op: &'static str,
        seat: &Seat,
        value: ValueId,
    ) -> Result<Tensor, Error> {
        let plane = *seat
            .layout
            .in_of
            .get(&value.0)
            .ok_or_else(|| Error::Backend {
                op,
                detail: format!("value {} is no recurrence input this load buffers", value.0),
            })?;
        let map = self.rs_window(op, seat)?;
        let spec = seat.layout.planes[plane];
        let ext = self.rs_temp(map.rows_ext, spec.width, spec.dtype);
        let own = self.tensor(value);
        let buffer = seat.buffers[plane];
        self.ctx().emit(&mut |cx| {
            let buf = cx.read(buffer)?;
            let mine = cx.read(own)?;
            let n = i64::from(map.rows_ext.max(1));
            let flat = |cx: &mut kernels_xla::Cx<'_>, t: Tensor, len: i64| {
                let v = cx.read(t)?;
                let all = cx.ty(v).elements();
                let v = cx.reshape(v, &[all])?;
                Ok::<_, Error>(cx.slice_axis(v, 0, 0, len)?)
            };
            let from_buffer = flat(cx, map.from_buffer, n)?;
            let buffer_row = flat(cx, map.buffer_row, n)?;
            let own_row = flat(cx, map.own_row, n)?;
            let replayed = cx.take_rows(buf, buffer_row)?;
            let owned = cx.take_rows(mine, own_row)?;
            let zero = cx.like_i(from_buffer, 0);
            let pick = cx.compare(kernels_xla::hlo::Cmp::Ne, from_buffer, zero)?;
            let width = cx.dims(replayed)[1];
            let pick = cx.broadcast(pick, &[n, width], &[0])?;
            let rows = cx.select(pick, replayed, owned)?;
            cx.write(ext, rows)?;
            let rows_own = i64::from(own.rows);
            let stash = flat(cx, map.stash, rows_own)?;
            let next = cx.put_rows(buf, stash, mine, kernels_xla::hlo::Combine::Set)?;
            cx.write(buffer, next)
        })?;
        Ok(ext)
    }

    /// The extended plane an op of this window lands `value` into.
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
        let map = self.rs_window(op, seat)?;
        let spec = seat.layout.regions[region];
        let ext = self.rs_temp(map.rows_ext, spec.width, spec.dtype);
        seat.ext.borrow_mut().insert(value.0, ext);
        Ok(ext)
    }

    /// The extended plane an earlier op of this window landed `value` into.
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

    /// The own rows of an extended output, landed where the plan reads
    /// `dest`.
    pub(crate) fn rs_land(
        &self,
        op: &'static str,
        seat: &Seat,
        ext: Tensor,
        dest: ValueId,
    ) -> Result<(), Error> {
        let map = self.rs_window(op, seat)?;
        let target = self.tensor(dest);
        self.ctx().emit(&mut |cx| {
            let v = cx.read(ext)?;
            let land = cx.read(map.land)?;
            let all = cx.ty(land).elements();
            let land = cx.reshape(land, &[all])?;
            let land = cx.slice_axis(land, 0, 0, i64::from(target.rows))?;
            let rows = cx.take_rows(v, land)?;
            cx.write(target, rows)
        })
    }
}
