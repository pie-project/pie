use kernels_xla::conv;
use kernels_xla::linear::moe::ExpertGlu;
use poem_exec::{
    DispatchConv, DispatchElemwise, DispatchFused, DispatchLinear, DispatchProbe, KernelError,
};
use poem_ir::{Conv, Fused, GluAct, GridRule, Linear, TimePad};

use poem_ir::Operands;

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        let Fused::MoeExperts {
            select,
            glu,
            down,
            norm,
            sum,
            gate_add,
        } = op
        else {
            return Err(KernelError::Unsupported { op: op.name() });
        };
        // The routed MLP on one sort where the kernel takes it, else its
        // parts as the trace states them.
        let fused = norm.is_none() && self.moe_experts(select, glu, down, sum)?;
        if !fused {
            DispatchLinear::dispatch(self, select)?;
            DispatchLinear::dispatch(self, glu)?;
            DispatchLinear::dispatch(self, down)?;
            if let Some(norm) = norm {
                DispatchElemwise::dispatch(self, norm)?;
            }
            DispatchLinear::dispatch(self, sum)?;
        }
        match gate_add {
            Some(gate_add) => DispatchLinear::dispatch(self, gate_add),
            None => Ok(()),
        }
    }
}

impl Run<'_> {
    /// `kernels_xla::linear::moe::experts` over a plain gate-up select, its
    /// packed SwiGLU or GeGLU, a plain down select on the same routes and
    /// their weighted sum; `false` (nothing lowered) for any other form, or
    /// where the kernel declines the fire.
    fn moe_experts(
        &mut self,
        select: &Linear,
        glu: &Linear,
        down: &Linear,
        sum: &Linear,
    ) -> Result<bool, KernelError> {
        let plain = |op: &Linear| match op {
            Linear::MoeSelect {
                x,
                bank,
                bias: None,
                routes,
                input_scales: None,
                quantized: false,
                quant: None,
                y,
            } => Some((*x, *bank, *routes, *y)),
            _ => None,
        };
        let (Some((x, up_bank, routes, packed)), Some((hidden, down_bank, down_routes, routed))) =
            (plain(select), plain(down))
        else {
            return Ok(false);
        };
        let Linear::Glu {
            packed: Some(glu_in),
            gate: None,
            up: None,
            weights: None,
            intermediate,
            act,
            limit: None,
            y: glu_out,
        } = glu
        else {
            return Ok(false);
        };
        let act = match act {
            GluAct::Swiglu => ExpertGlu::Swiglu,
            GluAct::GegluTanh => ExpertGlu::GegluTanh,
            _ => return Ok(false),
        };
        let Linear::MoeCombine {
            routed: summed,
            weights: Some(weights),
            shared: None,
            y,
        } = sum
        else {
            return Ok(false);
        };
        if *glu_in != packed || *glu_out != hidden || down_routes != routes || *summed != routed {
            return Ok(false);
        }
        if self.maybe_stored(up_bank).is_some()
            || self.maybe_stored(down_bank).is_some()
            || self.banked(up_bank).is_some()
            || self.banked(down_bank).is_some()
        {
            return Ok(false);
        }
        let (up_streamed, down_streamed) =
            (self.streamed_bank(up_bank), self.streamed_bank(down_bank));
        if let Some(b) = up_streamed.or(down_streamed) {
            self.cut_for(b, routes).map_err(crate::error::kernel)?;
        }
        let (up_t, down_t) = (self.tensor(up_bank), self.tensor(down_bank));
        let mut fused = false;
        crate::lower::lower(self.ctx(), |l| {
            let x_v = l.read(self.tensor(x))?;
            let up_v = match up_streamed {
                Some(b) => l.streamed_bank(b, up_t.width)?,
                None => l.read(up_t)?,
            };
            let routes_v = l.read(self.tensor(routes))?;
            let down_v = match down_streamed {
                Some(b) => l.streamed_bank(b, down_t.width)?,
                None => l.read(down_t)?,
            };
            let weights_v = l.read(self.tensor(*weights))?;
            let o_y = self.tensor(*y);
            let tuning = l.tuning();
            let v = kernels_xla::linear::moe::experts(
                l.f(),
                x_v,
                up_v,
                routes_v,
                self.tensor(packed).landing(),
                act,
                *intermediate,
                down_v,
                self.tensor(routed).landing(),
                weights_v,
                o_y.landing(),
                tuning,
            )?;
            match v {
                Some(v) => {
                    fused = true;
                    l.land(o_y, v)
                }
                None => Ok(()),
            }
        })
        .map_err(crate::error::kernel)?;
        Ok(fused)
    }
}

impl DispatchConv for Run<'_> {
    fn dispatch(&mut self, op: &Conv) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.spatial(op).map_err(crate::error::kernel)
    }
}

/// A probed value is read, in f32, right after the node that writes it ran:
/// the arena reuses its root for later values, so reading it at the end of
/// the program would read somebody else.
impl DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &poem_ir::Node) {
        if self.probes().is_empty() {
            return;
        }
        let mut outputs = Vec::new();
        node.op.outputs(&mut outputs);
        for value in outputs {
            if !self.probes().contains(&value) {
                continue;
            }
            let t = self.tensor(value);
            let mut read = None;
            let emitted = self.ctx().emit(&mut |cx| {
                let v = cx.read(t)?;
                read = Some(cx.convert(v, kernels_xla::hlo::Elem::F32));
                Ok(())
            });
            if let (Ok(()), Some(read)) = (emitted, read) {
                self.record_probe(value, read);
            }
        }
    }
}

fn rule(rule: GridRule) -> conv::GridRule {
    match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => conv::GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => conv::GridRule::Upsample {
            factor,
            keep_first_frame,
        },
        GridRule::Shuffle { r, trim_t } => conv::GridRule::Shuffle { r, trim_t },
        GridRule::Unshuffle { r } => conv::GridRule::Unshuffle { r },
        GridRule::AvgDown { factor } => conv::GridRule::AvgDown { factor },
    }
}

impl Run<'_> {
    /// A convolution weight in the tap-major order `conv::conv3d::conv3d` reads
    /// (`[C_out, taps · C_in]`, input channel inner), relabelled from the
    /// checkpoint's natural `[C_out, C_in · taps]` (engine-cuda
    /// `voxels::relabel_conv_weights`, done per program here: the plan
    /// declares every such weight `ConvTapsMajor { c_in, taps }`).
    fn taps_major(
        &self,
        w: crate::handle::Tensor,
        c_in: u32,
        taps: u32,
    ) -> Result<crate::handle::Tensor, kernels_xla::Error> {
        if taps <= 1 || c_in <= 1 {
            return Ok(w);
        }
        if u64::from(w.width) != u64::from(c_in) * u64::from(taps) {
            return Err(kernels_xla::Error::Backend {
                op: "conv.conv3d",
                detail: format!(
                    "a weight {} wide does not hold {c_in} input channels by {taps} taps",
                    w.width
                ),
            });
        }
        let out = self.temp(w.rows, w.width, w.dtype);
        self.ctx().emit(&mut |cx| {
            let v = cx.read(w)?;
            let v = cx.reshape(v, &[i64::from(w.rows), i64::from(c_in), i64::from(taps)])?;
            let v = cx.transpose(v, &[0, 2, 1])?;
            cx.write(out, v)
        })?;
        Ok(out)
    }

    /// The fire's clip slot table, or a refusal naming `op`.
    fn clip_slot_table(
        &self,
        op: &'static str,
    ) -> Result<crate::handle::Tensor, kernels_xla::Error> {
        self.clip_slots()
            .ok_or_else(|| kernels_xla::Error::Backend {
                op,
                detail:
                    "a frame cache needs the fire's clip slot table, which no lane of it staged"
                        .to_string(),
            })
    }

    /// engine-cuda's `dispatch/spatial.rs`, entry for entry. The lane tables
    /// are device data, so a convolution runs the table-as-data
    /// `conv::conv3d::conv3d`; `conv3d_boxed` (clip boxes as static attributes)
    /// needs the boxes on the host, which the walk does not hold.
    fn spatial(&mut self, op: &Conv) -> Result<(), kernels_xla::Error> {
        match op {
            Conv::Grid { grid, rule: how, y } => crate::lower::lower(self.ctx(), |l| {
                let grid_v = l.read(self.tensor(*grid))?;
                let o_y = self.tensor(*y);
                let v = conv::rule::derive_grid(l.f(), grid_v, rule(*how), o_y.landing())?;
                l.land(o_y, v)
            }),
            Conv::Conv3d {
                x,
                grid,
                w,
                bias,
                k,
                stride,
                pad,
                pad_back,
                causal_t,
                time_pad,
                dilation,
                depthwise,
                cache,
                y_grid,
                y,
            } => {
                if *dilation != [1, 1, 1] || *depthwise {
                    return Err(kernels_xla::Error::Unsupported {
                        op: "conv.conv3d (dilated or depthwise)",
                    });
                }
                let conv = conv::Conv3d {
                    k: *k,
                    stride: *stride,
                    pad: *pad,
                    pad_back: *pad_back,
                    causal_t: *causal_t,
                    time_pad: match time_pad {
                        TimePad::Zero => conv::TimePad::Zero,
                        TimePad::Replicate => conv::TimePad::Replicate,
                    },
                };
                let bases = [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)];
                let x = self.tensor(*x);
                let grid = self.tensor(*grid);
                let bias = bias.map(|b| self.tensor(b));
                let w = self.taps_major(self.tensor(*w), x.width, conv.taps())?;
                let Some(state) = cache else {
                    return crate::lower::lower(self.ctx(), |l| {
                        let y_grid_v = l.read(self.tensor(*y_grid))?;
                        let grid_v = l.read(grid)?;
                        let x_v = l.read(x)?;
                        let cache_v = l.read_opt(None)?;
                        let w_v = l.read(w)?;
                        let bias_v = l.read_opt(bias)?;
                        let o_y = self.tensor(*y);
                        let v = conv::conv3d::conv3d(
                            l.f(),
                            x_v,
                            grid_v,
                            w_v,
                            bias_v,
                            conv,
                            cache_v,
                            o_y.landing(),
                            y_grid_v,
                            bases,
                        )?;
                        l.land(o_y, v)
                    });
                };
                // The `pad[0]` frames before each clip: read from its slot,
                // stood in front of frame 0, then this fire's tail written
                // back over them.
                let frames = conv.pad[0];
                let slab_pool = self.recurrent(*state);
                let slab = slab_pool.state;
                let slot_ids = self.clip_slot_table("conv.conv3d")?;
                let frame_cache = self.temp(x.rows.saturating_mul(frames), x.width, x.dtype);
                crate::lower::lower(self.ctx(), |l| {
                    let grid_v = l.read(grid)?;
                    let slot_ids_v = l.read_slots(slot_ids, slab_pool.map)?;
                    let slab_v = l.read(slab)?;
                    let o_cache = frame_cache;
                    let v = conv::cache::cache_gather(
                        l.f(),
                        slab_v,
                        slot_ids_v,
                        grid_v,
                        frames,
                        o_cache.landing(),
                    )?;
                    l.land(o_cache, v)
                })?;
                crate::lower::lower(self.ctx(), |l| {
                    let y_grid_v = l.read(self.tensor(*y_grid))?;
                    let grid_v = l.read(grid)?;
                    let x_v = l.read(x)?;
                    let cache_v = l.read_opt(Some(frame_cache))?;
                    let w_v = l.read(w)?;
                    let bias_v = l.read_opt(bias)?;
                    let o_y = self.tensor(*y);
                    let v = conv::conv3d::conv3d(
                        l.f(),
                        x_v,
                        grid_v,
                        w_v,
                        bias_v,
                        conv,
                        cache_v,
                        o_y.landing(),
                        y_grid_v,
                        bases,
                    )?;
                    l.land(o_y, v)
                })?;
                crate::lower::lower(self.ctx(), |l| {
                    let grid_v = l.read(grid)?;
                    let slot_ids_v = l.read_slots(slot_ids, slab_pool.map)?;
                    let x_v = l.read(x)?;
                    let cache_v = l.read(frame_cache)?;
                    let slab_v = l.read(slab)?;
                    let o_slab = slab;
                    let v = conv::cache::cache_store(
                        l.f(),
                        x_v,
                        cache_v,
                        slot_ids_v,
                        grid_v,
                        frames,
                        slab_v,
                        bases[0],
                    )?;
                    l.land(o_slab, v)
                })
            }
            Conv::GroupNorm {
                x,
                grid,
                groups,
                weight,
                bias,
                eps,
                silu,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let grid_v = l.read(self.tensor(*grid))?;
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let bias_v = l.read(self.tensor(*bias))?;
                let o_y = self.tensor(*y);
                let v = conv::group_norm::group_norm(
                    l.f(),
                    x_v,
                    grid_v,
                    *groups,
                    weight_v,
                    bias_v,
                    *eps,
                    *silu,
                    o_y.landing(),
                    self.grid_base(*grid, *x),
                )?;
                l.land(o_y, v)
            }),
            Conv::Upsample {
                x,
                grid,
                factor,
                keep_first_frame,
                y_grid,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let y_grid_v = l.read(self.tensor(*y_grid))?;
                let grid_v = l.read(self.tensor(*grid))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = conv::resample::upsample_nearest(
                    l.f(),
                    x_v,
                    grid_v,
                    *factor,
                    *keep_first_frame,
                    o_y.landing(),
                    y_grid_v,
                    [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
                )?;
                l.land(o_y, v)
            }),
            Conv::PixelShuffle {
                x,
                grid,
                r,
                trim_t,
                y_grid,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let y_grid_v = l.read(self.tensor(*y_grid))?;
                let grid_v = l.read(self.tensor(*grid))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = conv::resample::pixel_shuffle(
                    l.f(),
                    x_v,
                    grid_v,
                    *r,
                    *trim_t,
                    o_y.landing(),
                    y_grid_v,
                    [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
                )?;
                l.land(o_y, v)
            }),
            Conv::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let y_grid_v = l.read(self.tensor(*y_grid))?;
                let grid_v = l.read(self.tensor(*grid))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = conv::resample::pixel_unshuffle(
                    l.f(),
                    x_v,
                    grid_v,
                    *r,
                    o_y.landing(),
                    y_grid_v,
                    [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
                )?;
                l.land(o_y, v)
            }),
            Conv::AvgDown {
                x,
                grid,
                factor,
                group,
                y_grid,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let y_grid_v = l.read(self.tensor(*y_grid))?;
                let grid_v = l.read(self.tensor(*grid))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = conv::resample::avg_down(
                    l.f(),
                    x_v,
                    grid_v,
                    *factor,
                    *group,
                    o_y.landing(),
                    y_grid_v,
                    [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
                )?;
                l.land(o_y, v)
            }),
            Conv::CacheStore {
                x,
                grid,
                frames,
                cache,
                x_out: _,
            } => {
                let base = self.grid_base(*grid, *x);
                let x = self.tensor(*x);
                let slab_pool = self.recurrent(*cache);
                let slab = slab_pool.state;
                let slot_ids = self.clip_slot_table("conv.cache_store")?;
                crate::lower::lower(self.ctx(), |l| {
                    let grid_v = l.read(self.tensor(*grid))?;
                    let slot_ids_v = l.read_slots(slot_ids, slab_pool.map)?;
                    let x_v = l.read(x)?;
                    let cache_v = l.read(x)?;
                    let slab_v = l.read(slab)?;
                    let o_slab = slab;
                    let v = conv::cache::cache_store(
                        l.f(),
                        x_v,
                        cache_v,
                        slot_ids_v,
                        grid_v,
                        *frames,
                        slab_v,
                        base,
                    )?;
                    l.land(o_slab, v)
                })
            }
            Conv::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let y_grid_v = l.read(self.tensor(*tgrid))?;
                let grid_v = l.read(self.tensor(*grid))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let bases = [self.grid_base(*grid, *x), self.grid_base(*tgrid, *y)];
                let v = conv::resample::patchify(
                    l.f(),
                    x_v,
                    grid_v,
                    *p,
                    o_y.landing(),
                    y_grid_v,
                    bases,
                )?;
                l.land(o_y, v)
            }),
            Conv::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let y_grid_v = l.read(self.tensor(*grid))?;
                let grid_v = l.read(self.tensor(*tgrid))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let bases = [self.grid_base(*tgrid, *x), self.grid_base(*grid, *y)];
                let v = conv::resample::unpatchify(
                    l.f(),
                    x_v,
                    grid_v,
                    *p,
                    o_y.landing(),
                    y_grid_v,
                    bases,
                )?;
                l.land(o_y, v)
            }),
            Conv::Cells {
                tgrid,
                positions,
                axis,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let tgrid_v = l.read(self.tensor(*tgrid))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_y = self.tensor(*y);
                let base = self.grid_base(*tgrid, *y);
                let v = conv::rule::cells(l.f(), tgrid_v, positions_v, *axis, o_y.landing(), base)?;
                l.land(o_y, v)
            }),
        }
    }
}
