use kernels_xla::spatial;
use poem_exec::{DispatchCustomCuda, DispatchProbe, DispatchSpatial, KernelError};
use poem_ir::{CustomCuda, GridRule, Spatial, TimePad, VoxelSegment};

use poem_ir::Operands;

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
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

fn segment(segment: VoxelSegment) -> spatial::Segment {
    match segment {
        VoxelSegment::Clip => spatial::Segment::Lane,
        VoxelSegment::Frames(n) => spatial::Segment::Frames(n),
    }
}

fn rule(rule: GridRule) -> spatial::GridRule {
    match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => spatial::GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => spatial::GridRule::Upsample {
            factor,
            keep_first_frame,
        },
        GridRule::Shuffle { r, trim_t } => spatial::GridRule::Shuffle { r, trim_t },
        GridRule::Unshuffle { r } => spatial::GridRule::Unshuffle { r },
        GridRule::AvgDown { factor } => spatial::GridRule::AvgDown { factor },
    }
}

impl Run<'_> {
    /// A convolution weight in the tap-major order `spatial::conv3d` reads
    /// (`[C_out, taps · C_in]`, input channel inner), relabelled from the
    /// checkpoint's natural `[C_out, C_in · taps]` (engine-cuda
    /// `voxels::relabel_conv_weights`, done per program here: the plan
    /// declares every such weight `ConvTapsMajor { c_in, taps }`).
    fn taps_major(
        &self,
        w: kernels_xla::Tensor,
        c_in: u32,
        taps: u32,
    ) -> Result<kernels_xla::Tensor, kernels_xla::Error> {
        if taps <= 1 || c_in <= 1 {
            return Ok(w);
        }
        if u64::from(w.width) != u64::from(c_in) * u64::from(taps) {
            return Err(kernels_xla::Error::Backend {
                op: "spatial.conv3d",
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
    fn clip_slot_table(&self, op: &'static str) -> Result<kernels_xla::Tensor, kernels_xla::Error> {
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
    /// `spatial::conv3d`; `conv3d_boxed` (clip boxes as static attributes)
    /// needs the boxes on the host, which the walk does not hold.
    fn spatial(&mut self, op: &Spatial) -> Result<(), kernels_xla::Error> {
        match op {
            Spatial::Grid { grid, rule: how, y } => {
                spatial::derive_grid(self.ctx(), self.tensor(*grid), rule(*how), self.tensor(*y))
            }
            Spatial::Conv3d {
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
                cache,
                y_grid,
                y,
            } => {
                let conv = spatial::Conv3d {
                    k: *k,
                    stride: *stride,
                    pad: *pad,
                    pad_back: *pad_back,
                    causal_t: *causal_t,
                    time_pad: match time_pad {
                        TimePad::Zero => spatial::TimePad::Zero,
                        TimePad::Replicate => spatial::TimePad::Replicate,
                    },
                };
                let x = self.tensor(*x);
                let grid = self.tensor(*grid);
                let bias = bias.map(|b| self.tensor(b));
                let w = self.taps_major(self.tensor(*w), x.width, conv.taps())?;
                let Some(state) = cache else {
                    return spatial::conv3d(
                        self.ctx(),
                        x,
                        grid,
                        w,
                        bias,
                        conv,
                        None,
                        self.tensor(*y),
                        self.tensor(*y_grid),
                    );
                };
                // The `pad[0]` frames before each clip: read from its slot,
                // stood in front of frame 0, then this fire's tail written
                // back over them.
                let frames = conv.pad[0];
                let slab = self.recurrent(*state).state;
                let slot_ids = self.clip_slot_table("spatial.conv3d")?;
                let frame_cache = self.temp(x.rows.saturating_mul(frames), x.width, x.dtype);
                spatial::cache_gather(self.ctx(), slab, slot_ids, grid, frames, frame_cache)?;
                spatial::conv3d(
                    self.ctx(),
                    x,
                    grid,
                    w,
                    bias,
                    conv,
                    Some(frame_cache),
                    self.tensor(*y),
                    self.tensor(*y_grid),
                )?;
                spatial::cache_store(self.ctx(), x, frame_cache, slot_ids, grid, frames, slab)
            }
            Spatial::GroupNorm {
                x,
                grid,
                groups,
                weight,
                bias,
                eps,
                silu,
                y,
            } => spatial::group_norm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *groups,
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                *silu,
                self.tensor(*y),
            ),
            Spatial::Attention {
                q,
                k,
                v,
                grid,
                segment: how,
                sm_scale,
                y,
            } => spatial::attention(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*grid),
                segment(*how),
                *sm_scale,
                self.tensor(*y),
            ),
            Spatial::UpsampleNearest {
                x,
                grid,
                factor,
                keep_first_frame,
                y_grid,
                y,
            } => spatial::upsample_nearest(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *keep_first_frame,
                self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::PixelShuffle {
                x,
                grid,
                r,
                trim_t,
                y_grid,
                y,
            } => spatial::pixel_shuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                *trim_t,
                self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => spatial::pixel_unshuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::AvgDown {
                x,
                grid,
                factor,
                group,
                y_grid,
                y,
            } => spatial::avg_down(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *group,
                self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::CacheStore {
                x,
                grid,
                frames,
                cache,
                x_out: _,
            } => {
                let x = self.tensor(*x);
                let slab = self.recurrent(*cache).state;
                let slot_ids = self.clip_slot_table("spatial.cache_store")?;
                spatial::cache_store(
                    self.ctx(),
                    x,
                    x,
                    slot_ids,
                    self.tensor(*grid),
                    *frames,
                    slab,
                )
            }
            Spatial::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => spatial::patchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *p,
                self.tensor(*y),
                self.tensor(*tgrid),
            ),
            Spatial::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => spatial::unpatchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*tgrid),
                *p,
                self.tensor(*y),
                self.tensor(*grid),
            ),
        }
    }
}
