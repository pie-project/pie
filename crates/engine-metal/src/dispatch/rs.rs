use kernels_metal::attn::{self, ssm::Committed};
use kernels_metal::{Error, Tensor, layout};
use poem_ir::{Attention, Operation, ValueId};

use crate::rs::{Convolved, Seat};
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

impl Run<'_> {
    fn kernel(fault: crate::error::Fault, op: &'static str) -> Error {
        Error::Backend {
            op,
            detail: fault.to_string(),
        }
    }

    pub(crate) fn rs_committed(&self, seat: &Seat) -> Committed {
        Committed {
            replay: seat.replay,
            commit: seat.commit,
            slots: seat.slots,
            lane0: self.window().span.lane_offset,
        }
    }

    /// Queues one copy of `bytes` from `src` to `dst` on `copies`.
    fn rs_copy(
        &self,
        op: &'static str,
        copies: &mut Vec<(Tensor, Tensor, u64)>,
        src: (u32, u64),
        dst: (u32, u64),
        bytes: u64,
    ) -> Result<(), Error> {
        if bytes == 0 {
            return Ok(());
        }
        let src = self
            .handles()
            .cut(src.0, src.1, bytes)
            .map_err(|fault| Self::kernel(fault, op))?;
        let dst = self
            .handles()
            .cut(dst.0, dst.1, bytes)
            .map_err(|fault| Self::kernel(fault, op))?;
        copies.push((
            Tensor::new(src, 1, 1, poem_ir::Dtype::U32),
            Tensor::new(dst, 1, 1, poem_ir::Dtype::U32),
            bytes,
        ));
        Ok(())
    }

    /// Lands the copies one recurrent op queued, four to a launch. They are
    /// disjoint — each lane's rows, its buffer pages, and the extended plane
    /// it lands them in are its own — so a launch runs them side by side
    /// where one launch a copy paid the device's launch cost per row run.
    fn rs_flush(&self, copies: &[(Tensor, Tensor, u64)]) -> Result<(), Error> {
        for chunk in copies.chunks(4) {
            layout::copy_words_batch(self.ctx(), chunk)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rs_move(
        &self,
        op: &'static str,
        copies: &mut Vec<(Tensor, Tensor, u64)>,
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
            let (at, room) = seat
                .locate(run, plane, run.from + done)
                .map_err(|fault| Self::kernel(fault, op))?;
            let take = room.min(run.count - done);
            let bytes = u64::from(take) * row_bytes;
            let rows_at = u64::from(row0 + done) * row_bytes;
            if into_slab {
                self.rs_copy(op, copies, (rect.buf, rows_at), (seat.pool, at), bytes)?;
            } else {
                self.rs_copy(op, copies, (seat.pool, at), (rect.buf, rows_at), bytes)?;
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
        let mut copies = Vec::new();
        for r in 0..indptr.len().saturating_sub(1) {
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let (begin, replay) = extended_origin(indptr, &seat.lanes, lane0, r);
            let Some(plan) = seat.lanes.get(lane0 as usize + r) else {
                continue;
            };
            if let Some(run) = &plan.gather {
                self.rs_move(op, &mut copies, seat, plane, run, ext, begin, false)?;
            }
            let own_at = u64::from(indptr[r].max(0) as u32) * row_bytes;
            self.rs_copy(
                op,
                &mut copies,
                (own.buf, own_at),
                (ext.buf, u64::from(begin + replay) * row_bytes),
                u64::from(rows) * row_bytes,
            )?;
            if let Some(run) = &plan.scatter {
                self.rs_move(
                    op,
                    &mut copies,
                    seat,
                    plane,
                    run,
                    own,
                    indptr[r].max(0) as u32,
                    true,
                )?;
            }
        }
        self.rs_flush(&copies)?;
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
        if crate::diag::on().rs_trace {
            eprintln!(
                "rs_land {op}: ext buf {} rows {} width {} {:?} -> target buf {} rows {} width {} {:?}; window {:?}",
                ext.buf,
                ext.rows,
                ext.width,
                ext.dtype,
                target.buf,
                target.rows,
                target.width,
                target.dtype,
                self.qo_indptr_host()
            );
        }
        let row_bytes = u64::from(ext.width)
            * poem_compiler::arena::elem_bytes(ext.dtype).ok_or_else(|| Error::Backend {
                op,
                detail: format!("{:?} has no element size", ext.dtype),
            })?;
        let indptr = self.qo_indptr_host();
        let lane0 = self.window().span.lane_offset;
        let mut copies = Vec::new();
        for r in 0..indptr.len().saturating_sub(1) {
            let rows = (indptr[r + 1] - indptr[r]).max(0) as u32;
            let (begin, replay) = extended_origin(indptr, &seat.lanes, lane0, r);
            self.rs_copy(
                op,
                &mut copies,
                (ext.buf, u64::from(begin + replay) * row_bytes),
                (target.buf, u64::from(indptr[r].max(0) as u32) * row_bytes),
                u64::from(rows) * row_bytes,
            )?;
        }
        self.rs_flush(&copies)
    }

    /// Whether `gates` is read by one node alone, a gated-delta scan.
    pub(crate) fn read_only_by_scan(&self, gates: ValueId) -> bool {
        matches!(
            self.sole_reader(gates).map(|node| &node.op),
            Some(Operation::Attention(
                Attention::SsmGatedDelta { gates: read, .. }
                    | Attention::SsmGatedDeltaChunked { gates: read, .. }
            )) if *read == gates
        )
    }

    /// Runs the gate prep over the extended `[b | a]` rows, landing the
    /// window's own gates in `gates`.
    pub(crate) fn rs_prep(
        &self,
        seat: &Seat,
        (ba, dt_bias, a_log): (ValueId, Tensor, Tensor),
        gates: ValueId,
    ) -> Result<(), Error> {
        const OP: &str = "attention.ssm_gdn_prep";
        let ext_ba = self.rs_extend(OP, seat, ba)?;
        let ext_gates = self.rs_out(OP, seat, gates)?;
        attn::ssm::gdn_prep(self.ctx(), ext_ba, dt_bias, a_log, ext_gates)?;
        self.rs_land(OP, seat, ext_gates, gates)
    }

    /// Whether a split of `x` at `width` into `left` and `right` is left to
    /// the verify convs that alone read `left`, one an arm, through the page
    /// table: each reads its rows from `x` and lands their `right` itself,
    /// before the rest of its arm reads them (a gated-delta scan reads none).
    pub(crate) fn rs_split_into_conv(
        &self,
        x: ValueId,
        width: u32,
        left: ValueId,
        right: ValueId,
    ) -> bool {
        let Some(seat) = self.rs_seat() else {
            return false;
        };
        let node = |at: u32| self.nodes().get(at as usize);
        let convs = self.readers(left);
        let convolves = |at: &u32| {
            matches!(
                node(*at).map(|node| &node.op),
                Some(Operation::Attention(
                    Attention::SsmCausalConv1d { x: input, conv_width, dilation, .. }
                        | Attention::SsmCausalConv1dChunked { x: input, conv_width, dilation, .. }
                )) if *input == left
                    && attn::ssm::causal_conv1d_verify_fits(width, *conv_width, *dilation)
            )
        };
        let after_its_conv = |at: &u32| {
            let Some(reader) = node(*at) else {
                return false;
            };
            matches!(
                reader.op,
                Operation::Attention(
                    Attention::SsmGatedDelta { .. } | Attention::SsmGatedDeltaChunked { .. }
                )
            ) || convs
                .iter()
                .any(|conv| conv < at && node(*conv).is_some_and(|conv| conv.guard == reader.guard))
        };
        let z = self.tensor(right).width;
        let fits = !convs.is_empty()
            && convs.iter().all(convolves)
            && self.readers(right).iter().all(after_its_conv)
            && z <= width
            && z.is_multiple_of(4)
            && seat.paged(self.handles(), left).is_some();
        if fits {
            seat.split.borrow_mut().insert(left.0, (x, right));
        }
        fits
    }

    /// Runs the window's conv into the extended plane, landing the window's
    /// own rows in `y`, and returns the extended plane.
    pub(crate) fn rs_convolve(
        &self,
        seat: &Seat,
        conv: &Convolved,
        y: ValueId,
    ) -> Result<Tensor, Error> {
        const OP: &str = "attention.ssm_causal_conv1d_committed";
        let ext_y = self.rs_out(OP, seat, y)?;
        let split = seat.split.borrow().get(&conv.x.0).copied();
        // The verify conv lands the window's own rows itself, and through the
        // page table reads and lands the buffer's too, and the split's other
        // half where it reads the packed rows.
        let paged = seat.paged(self.handles(), conv.x);
        let fits =
            attn::ssm::causal_conv1d_verify_fits(ext_y.width, conv.conv_width, conv.dilation);
        if let (Some((packed, right)), Some(_), true) = (split, paged, fits) {
            return attn::ssm::causal_conv1d_verify(
                self.ctx(),
                self.tensor(packed),
                paged,
                Some(attn::ssm::VerifySplit {
                    z: self.tensor(right),
                }),
                self.qo_indptr(),
                &self.rs_committed(seat),
                conv.weight,
                &self.recurrent(conv.state),
                conv.conv_width,
                conv.dilation,
                ext_y,
                self.tensor(y),
                self.rs_snapshots(),
            )
            .map(|()| ext_y);
        }
        // A split left to a conv that cannot take it is cut here.
        if let Some((packed, right)) = split {
            layout::split_rows(
                self.ctx(),
                self.tensor(packed),
                ext_y.width,
                self.tensor(conv.x),
                self.tensor(right),
            )?;
        }
        if fits {
            let source = match paged {
                Some(_) => self.tensor(conv.x),
                None => self.rs_extend(OP, seat, conv.x)?,
            };
            attn::ssm::causal_conv1d_verify(
                self.ctx(),
                source,
                paged,
                None,
                self.qo_indptr(),
                &self.rs_committed(seat),
                conv.weight,
                &self.recurrent(conv.state),
                conv.conv_width,
                conv.dilation,
                ext_y,
                self.tensor(y),
                self.rs_snapshots(),
            )?;
            return Ok(ext_y);
        }
        let ext_x = self.rs_extend(OP, seat, conv.x)?;
        attn::ssm::causal_conv1d_committed_landing(
            self.ctx(),
            ext_x,
            self.qo_indptr(),
            &self.rs_committed(seat),
            conv.weight,
            &self.recurrent(conv.state),
            conv.conv_width,
            conv.dilation,
            ext_y,
            self.rs_snapshots(),
        )?;
        self.rs_land(OP, seat, ext_y, y)?;
        Ok(ext_y)
    }
}
