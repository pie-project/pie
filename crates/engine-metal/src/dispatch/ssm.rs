use crate::run::Run;
use kernels_metal::ssm;
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::{Operands, Ssm};

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ssm(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// Whether this fire buffers `x`, a recurrence input, in the rs list:
    /// a conv whose output no deferred recurrence reads commits as it runs.
    fn rs_buffers(&self, x: poem_ir::ValueId) -> bool {
        self.rs_seat()
            .is_some_and(|seat| seat.layout.in_of.contains_key(&x.0))
    }

    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_metal::Error> {
        match op {
            // The committed walk's conv takes no bias.
            Ssm::CausalConv1d {
                bias: Some(_), x, ..
            }
            | Ssm::CausalConv1dChunked {
                bias: Some(_), x, ..
            } if self.rs_buffers(*x) => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Ssm::Mamba2 { .. } | Ssm::Mamba2Chunked { .. } if self.rs_seat().is_some() => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Ssm::Mamba2 {
                xbc,
                dt,
                z,
                a_log,
                dt_bias,
                d,
                norm,
                eps,
                state,
                heads,
                groups,
                state_dim,
                y,
            }
            | Ssm::Mamba2Chunked {
                xbc,
                dt,
                z,
                a_log,
                dt_bias,
                d,
                norm,
                eps,
                state,
                heads,
                groups,
                state_dim,
                y,
            } => {
                let m = ssm::Mamba2 {
                    a_log: self.tensor(*a_log)?,
                    dt_bias: self.tensor(*dt_bias)?,
                    d: self.tensor(*d)?,
                    norm: self.tensor(*norm)?,
                    eps: *eps,
                    heads: *heads,
                    groups: *groups,
                };
                let (dt, z, y) = (self.tensor(*dt), self.tensor(*z), self.tensor(*y));
                if matches!(op, Ssm::Mamba2 { .. }) {
                    return ssm::mamba2(
                        self.ctx(),
                        self.tensor(*xbc)?,
                        dt?,
                        z?,
                        m,
                        &self.recurrent(*state)?,
                        *state_dim,
                        y?,
                    );
                }
                ssm::mamba2_chunked(
                    self.ctx(),
                    self.ragged(*xbc)?,
                    dt?,
                    z?,
                    m,
                    &self.recurrent(*state)?,
                    *state_dim,
                    y?,
                    self.rs_snapshots(*state)?,
                )
            }
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            }
            | Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } if self.rs_buffers(*x) => {
                let seat = self.rs_seat().expect("guarded");
                let conv = crate::rs::Convolved {
                    x: *x,
                    weight: self.tensor(*weight)?,
                    state: *state,
                    conv_width: *conv_width,
                    dilation: *dilation,
                };
                self.rs_convolve(&seat, &conv, *y).map(|_| ())
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } if self.rs_seat().is_some() => {
                let seat = self.rs_seat().expect("guarded");
                let prep = (*ba, self.tensor(*dt_bias)?, self.tensor(*a_log)?);
                // The scan that alone reads the gates derives them as it
                // stages its rows, and lands the window's own.
                if self.read_only_by_scan(*gates) {
                    seat.prepped.borrow_mut().insert(gates.0, prep);
                    return Ok(());
                }
                self.rs_prep(&seat, prep, *gates)
            }
            Ssm::GatedDelta {
                qkv,
                z: _,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            }
            | Ssm::GatedDeltaChunked {
                qkv,
                z: _,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.gated_delta_committed";
                let seat = self.rs_seat().expect("guarded");
                let ext_qkv = self.rs_ext_of(OP, &seat, *qkv)?;
                let prepped = seat.prepped.borrow_mut().remove(&gates.0);
                if let Some((ba, dt_bias, a_log)) = prepped
                    && ssm::gated_delta_verify_fits(*k_dim, *v_dim)
                {
                    let paged = seat.paged(self.handles(), ba);
                    let ba = match paged {
                        Some(_) => self.tensor(ba)?,
                        None => self.rs_extend(OP, &seat, ba)?,
                    };
                    // The corrected planes (keys, u) each own row lands:
                    // through the page table, or into the planes' scratch at
                    // its own row and scattered into the buffer after.
                    let in_of = |id: poem_ir::ValueId| seat.layout.in_of.get(&id.0).copied();
                    let planes = in_of(*qkv).zip(in_of(*y)).zip(in_of(*gates));
                    let corrected = match (planes, paged) {
                        (None, _) => None,
                        (Some(_), Some(_)) => seat
                            .paged(self.handles(), *qkv)
                            .zip(seat.paged(self.handles(), *y))
                            .zip(seat.paged(self.handles(), *gates))
                            .map(|((keys, u), scalars)| ssm::Corrected {
                                keys: keys.slab,
                                key_row_bytes: keys.row_bytes,
                                u: u.slab,
                                u_row_bytes: u.row_bytes,
                                scalars: scalars.slab,
                                scalars_row_bytes: scalars.row_bytes,
                            }),
                        (Some(((kp, up), sp)), None) => Some(ssm::Corrected {
                            keys: seat.ext_in[kp],
                            key_row_bytes: plane_row_bytes(&seat, kp, OP)?,
                            u: seat.ext_in[up],
                            u_row_bytes: plane_row_bytes(&seat, up, OP)?,
                            scalars: seat.ext_in[sp],
                            scalars_row_bytes: plane_row_bytes(&seat, sp, OP)?,
                        }),
                    };
                    // The beam groups' folds first:
                    // each group's state takes the rows its beams share
                    // before any of its lanes reads it.
                    if let Some(folds) = seat.folds {
                        let (Some(ba_paged), Some(corrected)) = (paged, corrected) else {
                            return Err(kernels_metal::Error::Backend {
                                op: OP,
                                detail: "a beam group folds its shared rows before the step, and \
                                      the scan reads its rows through no page table and \
                                      corrected planes: a group fold reads them there"
                                    .to_string(),
                            });
                        };
                        ssm::gated_delta_group_fold(
                            self.ctx(),
                            folds,
                            self.window_lanes()?,
                            self.rs_committed(&seat, *state)?.slots,
                            (a_log, dt_bias),
                            ba_paged,
                            corrected,
                            &self.recurrent(*state)?,
                            (*k_heads, *v_heads, *k_dim, *v_dim),
                        )?;
                    }
                    // A decode window (a row of its own every lane, few
                    // carried) reads its carried rows as a linear attention.
                    if let (Some(ba_paged), Some(corrected)) = (paged, corrected)
                        && self.rs_snapshots(*state)?.is_none()
                        && self.carried_window(&seat)?
                    {
                        return ssm::gated_delta_carried(
                            self.ctx(),
                            self.tuning(),
                            ext_qkv,
                            self.qo_indptr()?,
                            &self.rs_committed(&seat, *state)?,
                            &ssm::VerifyGates {
                                ba,
                                dt_bias,
                                a_log,
                                gates: self.tensor(*gates)?,
                            },
                            ba_paged,
                            corrected,
                            &self.recurrent(*state)?,
                            *k_heads,
                            *v_heads,
                            *k_dim,
                            *v_dim,
                            self.tensor(*y)?,
                            self.dots_table(&seat, *k_heads)?,
                        );
                    }
                    let scattered = planes.filter(|_| paged.is_none());
                    let landed = ssm::gated_delta_verify(
                        self.ctx(),
                        ext_qkv,
                        self.qo_indptr()?,
                        &self.rs_committed(&seat, *state)?,
                        &ssm::VerifyGates {
                            ba,
                            dt_bias,
                            a_log,
                            gates: self.tensor(*gates)?,
                        },
                        paged,
                        corrected,
                        &self.recurrent(*state)?,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        self.tensor(*y)?,
                        self.rs_snapshots(*state)?,
                    );
                    if let (Ok(()), Some(((kp, up), sp))) = (&landed, scattered) {
                        self.rs_scatter(OP, &seat, kp, seat.ext_in[kp])?;
                        self.rs_scatter(OP, &seat, up, seat.ext_in[up])?;
                        self.rs_scatter(OP, &seat, sp, seat.ext_in[sp])?;
                    }
                    landed?;
                    // A compact fire's decode lanes beside wider ones read
                    // their carried rows from their buffers as a decode
                    // window's do (the verify scan passed them over).
                    if let (Some(ba_paged), Some(corrected)) = (paged, corrected)
                        && self.window_has_compact_decode(&seat)
                    {
                        ssm::gated_delta_carried(
                            self.ctx(),
                            self.tuning(),
                            ext_qkv,
                            self.qo_indptr()?,
                            &self.rs_committed(&seat, *state)?,
                            &ssm::VerifyGates {
                                ba,
                                dt_bias,
                                a_log,
                                gates: self.tensor(*gates)?,
                            },
                            ba_paged,
                            corrected,
                            &self.recurrent(*state)?,
                            *k_heads,
                            *v_heads,
                            *k_dim,
                            *v_dim,
                            self.tensor(*y)?,
                            self.dots_table(&seat, *k_heads)?,
                        )?;
                    }
                    return Ok(());
                }
                // Heads the verify scan does not stage take the gate prep
                // the verify scan would have folded in.
                if let Some(prep) = prepped {
                    self.rs_prep(&seat, prep, *gates)?;
                }
                let ext_gates = self.rs_ext_of(OP, &seat, *gates)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                ssm::gated_delta_committed_landing(
                    self.ctx(),
                    self.tuning(),
                    ext_qkv,
                    self.qo_indptr()?,
                    &self.rs_committed(&seat, *state)?,
                    ext_gates,
                    &self.recurrent(*state)?,
                    seat.work,
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    ext_y,
                    self.rs_snapshots(*state)?,
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                bias,
            } => ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x)?,
                self.tensor(*weight)?,
                bias.map(|b| self.tensor(b)).transpose()?,
                &self.recurrent(*state)?,
                *conv_width,
                *dilation,
                self.tensor(*y)?,
            ),
            Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                bias,
            } => ssm::causal_conv1d_chunked_landing(
                self.ctx(),
                self.ragged(*x)?,
                self.tensor(*weight)?,
                bias.map(|b| self.tensor(b)).transpose()?,
                &self.recurrent(*state)?,
                *conv_width,
                *dilation,
                self.tensor(*y)?,
                self.rs_snapshots(*state)?,
            ),
            Ssm::ShortConv { .. } | Ssm::ShortConvChunked { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Ssm::DynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => {
                // The convolved rows' projections on the register tile read
                // their input table as the convolution lays it.
                let (x, coeff, base, out) = (
                    self.ragged(*x),
                    self.tensor(*coeff),
                    self.tensor(*base),
                    self.tensor(*y),
                );
                match self.producer_table(*y)? {
                    Some((table, counters)) => ssm::dyn_conv::block_dyn_conv_table(
                        self.ctx(),
                        x?,
                        coeff?,
                        base?,
                        *side,
                        *taps,
                        *group,
                        out?,
                        table,
                        counters,
                    ),
                    None => ssm::dyn_conv::block_dyn_conv(
                        self.ctx(),
                        x?,
                        coeff?,
                        base?,
                        *side,
                        *taps,
                        *group,
                        out?,
                    ),
                }
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => ssm::gdn_prep(
                self.ctx(),
                self.tensor(*ba)?,
                self.tensor(*dt_bias)?,
                self.tensor(*a_log)?,
                self.tensor(*gates)?,
            ),
            Ssm::GatedDelta {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => ssm::gated_delta(
                self.ctx(),
                self.tuning(),
                self.tensor(*qkv)?,
                self.tensor(*z)?,
                self.tensor(*gates)?,
                &self.recurrent(*state)?,
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y)?,
            ),
            Ssm::GatedDeltaChunked {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => ssm::gated_delta_chunked_landing(
                self.ctx(),
                self.tuning(),
                self.ragged(*qkv)?,
                self.tensor(*z)?,
                self.tensor(*gates)?,
                &self.recurrent(*state)?,
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y)?,
                self.rs_snapshots(*state)?,
            ),
            Ssm::Kda {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            }
            | Ssm::KdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.kda_committed";
                let seat = self.rs_seat().expect("guarded");
                let ext_mixed = self.rs_ext_of(OP, &seat, *mixed)?;
                let ext_f = self.rs_extend(OP, &seat, *f)?;
                let ext_b = self.rs_extend(OP, &seat, *b)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                ssm::kda_committed(
                    self.ctx(),
                    ext_mixed,
                    self.qo_indptr()?,
                    &self.rs_committed(&seat, *state)?,
                    ext_f,
                    ext_b,
                    self.tensor(*dt_bias)?,
                    self.tensor(*a_log)?,
                    &self.recurrent(*state)?,
                    seat.work,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                    ext_y,
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Ssm::Kda {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } => ssm::kda_step(
                self.ctx(),
                self.tensor(*mixed)?,
                self.tensor(*f)?,
                self.tensor(*b)?,
                self.tensor(*dt_bias)?,
                self.tensor(*a_log)?,
                &self.recurrent(*state)?,
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                self.tensor(*y)?,
            ),
            Ssm::KdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } => ssm::kda_chunked(
                self.ctx(),
                self.ragged(*mixed)?,
                self.tensor(*f)?,
                self.tensor(*b)?,
                self.tensor(*dt_bias)?,
                self.tensor(*a_log)?,
                &self.recurrent(*state)?,
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                self.tensor(*y)?,
            ),
        }
    }
}

/// A buffered plane's bytes a token, as the kernels take it.
fn plane_row_bytes(
    seat: &crate::rs::Seat,
    plane: usize,
    op: &'static str,
) -> Result<u32, kernels_metal::Error> {
    u32::try_from(seat.layout.planes[plane].row_bytes).map_err(|_| kernels_metal::Error::Backend {
        op,
        detail: "a buffered plane's row is past a u32 of bytes".to_string(),
    })
}

impl crate::run::Run<'_> {
    /// Whether this window's lanes each run one row of their own over their
    /// carried rows (any number of them): a decode step the carried scan
    /// serves.
    /// The per-head table a carried window's scan reads (`ssm::DotsTable`),
    /// in the seat's work area (the verify path's, unused by the carried
    /// scan): refused by name where the window's lanes carry more rows than
    /// it holds for each lane and key head.
    pub(crate) fn dots_table(
        &self,
        seat: &crate::rs::Seat,
        k_heads: u32,
    ) -> Result<ssm::DotsTable, kernels_metal::Error> {
        let lane0 = self.window()?.span.lane_offset as usize;
        let lanes = self.qo_indptr_host()?.len().saturating_sub(1);
        let max_rep = seat
            .lanes
            .iter()
            .skip(lane0)
            .take(lanes)
            .map(|plan| plan.replay)
            .max()
            .unwrap_or(0);
        // The table holds as many rows as the seat's work area gives a lane
        // and key head; a lane carrying more (a beam whose lineages stay
        // apart, a long run-ahead) is served all the same: the scan makes
        // each chunk's reductions itself (`ssm::DotsTable::fits`).
        let rows = u32::try_from(seat.layout.work_elems * 4 / 16 / u64::from(k_heads.max(1)))
            .unwrap_or(u32::MAX);
        Ok(ssm::DotsTable {
            table: seat.work,
            rows,
            max_rep,
        })
    }

    /// Whether a lane of this window takes fewer extended rows than it
    /// carries: a compact fire's decode lane, which the carried scan reads.
    pub(crate) fn window_has_compact_decode(&self, seat: &crate::rs::Seat) -> bool {
        let Ok((lane0, lanes)) = self.window_lanes() else {
            return false;
        };
        (lane0..lane0.saturating_add(lanes)).any(|lane| {
            let lane = lane as usize;
            match (seat.ext_replay.get(lane), seat.lanes.get(lane)) {
                (Some(&ext), Some(plan)) => ext < plan.replay,
                _ => false,
            }
        })
    }

    /// This window's first lane and its lane count: the lanes its
    /// recurrent ops run.
    pub(crate) fn window_lanes(&self) -> crate::run::Ran<(u32, u32)> {
        let lanes = self.qo_indptr_host()?.len().saturating_sub(1);
        Ok((
            self.window()?.span.lane_offset,
            u32::try_from(lanes).unwrap_or(u32::MAX),
        ))
    }

    pub(crate) fn carried_window(&self, seat: &crate::rs::Seat) -> crate::run::Ran<bool> {
        let indptr = self.qo_indptr_host()?;
        let lane0 = self.window()?.span.lane_offset as usize;
        Ok(indptr
            .windows(2)
            .enumerate()
            .all(|(r, pair)| pair[1] - pair[0] == 1 && seat.lanes.get(lane0 + r).is_some()))
    }
}
