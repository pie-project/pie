//! The device half of the recurrent-state verbs: a buffered lane's
//! recurrence runs over its extended rows (replayed buffer rows, then its
//! own), its own rows are stashed into the buffer, and the outputs of its own
//! rows land back where the plan reads them. Every move is a gather or a
//! scatter over the index maps staged for the window (`rs::WindowMap`).

use crate::handle::Committed;
use crate::handle::Tensor;
use kernels_xla::Error;
use poem_ir::ValueId;

use crate::rs::{CarriedInputs, Seat, WindowInputs};
use crate::run::Run;
use crate::trace::{Root, Source};

/// Buffer rows `q · tokens + token` (grain index `q`) as rows of a plane
/// whose blocks `map` places: `(q · stride + offset) · tokens + token`. A
/// dropped row (`rs::DROP`) stays out of range.
fn plane_rows(
    cx: &mut crate::emit::Cx<'_>,
    rows: kernels_xla::hlo::Val,
    map: crate::store::Map,
    tokens: i64,
) -> Result<kernels_xla::hlo::Val, Error> {
    if map.stride == 1 && map.offset == 0 {
        return Ok(rows);
    }
    let t = cx.like_i(rows, tokens);
    let q = cx.div(rows, t)?;
    let token = cx.rem(rows, t)?;
    let stride = cx.like_i(rows, map.stride);
    let offset = cx.like_i(rows, map.offset);
    let block = cx.mul(q, stride)?;
    let block = cx.add(block, offset)?;
    let row = cx.mul(block, t)?;
    let row = cx.add(row, token)?;
    let drop = cx.like_i(rows, i64::from(crate::rs::DROP));
    let dropped = cx.compare(kernels_xla::hlo::Cmp::Eq, rows, drop)?;
    Ok(cx.select(dropped, rows, row)?)
}

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

    /// The carried rows of the current window when it runs each layer's
    /// decode form (a window of single-token lanes), or none.
    pub(crate) fn rs_carried(&self, seat: &Seat) -> Option<CarriedInputs> {
        let span = self.window().span;
        seat.carried
            .get(&(span.lane_offset, span.lanes, span.row_offset, span.rows))
            .copied()
    }

    /// Layout plane `plane`'s rows each lane of the window carries, as the
    /// decode step reads them: `[lanes · rows, width]` in the plane's dtype,
    /// lane-major (rows past a lane's count read buffer row 0, unread).
    pub(crate) fn rs_carried_plane(
        &self,
        op: &'static str,
        seat: &Seat,
        carried: &CarriedInputs,
        plane: usize,
    ) -> Result<Tensor, Error> {
        let spec = *seat
            .layout
            .planes
            .get(plane)
            .ok_or_else(|| Error::Backend {
                op,
                detail: format!("no layout plane {plane}"),
            })?;
        let n = self.window().span.lanes * carried.rows.max(1);
        let out = self.rs_temp(n, spec.width, spec.dtype);
        let (buffer, at) = seat.buffers[plane];
        let tokens = i64::from(seat.page_tokens.max(1));
        let rows = carried.buffer_rows;
        self.ctx().emit(&mut |cx| {
            let buf = cx.read(buffer)?;
            let r = cx.read(rows)?;
            let all = cx.ty(r).elements();
            let r = cx.reshape(r, &[all])?;
            let r = cx.slice_axis(r, 0, 0, i64::from(n))?;
            let r = plane_rows(cx, r, at, tokens)?;
            let got = cx.take_rows(buf, r)?;
            cx.write(out, got)
        })?;
        Ok(out)
    }

    /// Columns `col0..col0 + width` of `src` (the window's own rows) into
    /// layout plane `plane` at the buffer rows the window stashes its own
    /// rows into (`WindowMap::stash`): a plane no trace value carries whole
    /// (a gated-delta row's keys, its corrected values).
    pub(crate) fn rs_stash_cols(
        &self,
        op: &'static str,
        seat: &Seat,
        plane: usize,
        src: Tensor,
        col0: u32,
    ) -> Result<(), Error> {
        let spec = *seat
            .layout
            .planes
            .get(plane)
            .ok_or_else(|| Error::Backend {
                op,
                detail: format!("no layout plane {plane}"),
            })?;
        let map = self.rs_window(op, seat)?;
        let (buffer, at) = seat.buffers[plane];
        let tokens = i64::from(seat.page_tokens.max(1));
        let rows_own = i64::from(src.rows);
        let (lo, hi) = (i64::from(col0), i64::from(col0 + spec.width));
        self.ctx().emit(&mut |cx| {
            let buf = cx.read(buffer)?;
            let mine = cx.read(src)?;
            let mine = cx.slice_axis(mine, 1, lo, hi)?;
            let elem = cx.elem(buf);
            let mine = cx.convert(mine, elem);
            let stash = cx.read(map.stash)?;
            let all = cx.ty(stash).elements();
            let stash = cx.reshape(stash, &[all])?;
            let stash = cx.slice_axis(stash, 0, 0, rows_own)?;
            let stash = plane_rows(cx, stash, at, tokens)?;
            let next = cx.put_rows(buf, stash, mine, kernels_xla::hlo::Combine::Set)?;
            cx.write(buffer, next)
        })
    }

    pub(crate) fn rs_temp(&self, rows: u32, width: u32, dtype: poem_ir::Dtype) -> Tensor {
        self.handles().root(Root {
            source: Source::Temp,
            dtype,
            rows: rows.max(1),
            width,
        })
    }

    /// The rollback tables of a recurrence over `state`'s row: each lane's
    /// slot of the record piece that row lies in.
    pub(crate) fn rs_committed(&self, seat: &Seat, state: ValueId) -> Committed {
        let record = self.record_of(state);
        Committed {
            replay: seat.replay,
            commit: seat.commit,
            slots: seat.slots[record],
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
        let (buffer, at) = seat.buffers[plane];
        let tokens = i64::from(seat.page_tokens.max(1));
        self.ctx().emit(&mut |cx| {
            let buf = cx.read(buffer)?;
            let mine = cx.read(own)?;
            let n = i64::from(map.rows_ext.max(1));
            let flat = |cx: &mut crate::emit::Cx<'_>, t: Tensor, len: i64| {
                let v = cx.read(t)?;
                let all = cx.ty(v).elements();
                let v = cx.reshape(v, &[all])?;
                Ok::<_, Error>(cx.slice_axis(v, 0, 0, len)?)
            };
            let from_buffer = flat(cx, map.from_buffer, n)?;
            let buffer_row = flat(cx, map.buffer_row, n)?;
            let buffer_row = plane_rows(cx, buffer_row, at, tokens)?;
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
            let stash = plane_rows(cx, stash, at, tokens)?;
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
