use model_exec::{DispatchCustomCuda, KernelError};
use model_ir::{CustomCuda, Operands};

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl model_exec::DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &model_ir::Spatial) -> Result<(), KernelError> {
        self.spatial(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn spatial(&mut self, op: &model_ir::Spatial) -> Result<(), kernels_metal::Error> {
        use kernels_metal::spatial;
        use model_ir::Spatial;
        match op {
            Spatial::Grid { grid, rule, y } => spatial::rule::derive_grid(
                self.ctx(),
                self.tensor(*grid),
                match *rule {
                    model_ir::GridRule::Conv {
                        k,
                        stride,
                        pad,
                        pad_back,
                        causal_t,
                    } => spatial::rule::GridRule::Conv {
                        k,
                        stride,
                        pad,
                        pad_back,
                        causal_t,
                    },
                    model_ir::GridRule::Upsample {
                        factor,
                        keep_first_frame,
                    } => spatial::rule::GridRule::Upsample {
                        factor,
                        keep_first_frame,
                    },
                    model_ir::GridRule::Shuffle { r, trim_t } => {
                        spatial::rule::GridRule::Shuffle { r, trim_t }
                    }
                    model_ir::GridRule::Unshuffle { r } => spatial::rule::GridRule::Unshuffle { r },
                    model_ir::GridRule::AvgDown { factor } => {
                        spatial::rule::GridRule::AvgDown { factor }
                    }
                },
                self.tensor(*y),
            ),
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
                if cache.is_some() {
                    return Err(kernels_metal::Error::Unsupported {
                        op: "spatial.conv3d (with a frame cache)",
                    });
                }
                spatial::conv::conv3d(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*grid),
                    self.tensor(*w),
                    bias.map(|bias| self.tensor(bias)),
                    None,
                    spatial::conv::Conv3d {
                        k: *k,
                        stride: *stride,
                        pad: *pad,
                        pad_back: *pad_back,
                        causal_t: *causal_t,
                        time_pad: match time_pad {
                            model_ir::TimePad::Zero => spatial::conv::TimePad::Zero,
                            model_ir::TimePad::Replicate => spatial::conv::TimePad::Replicate,
                        },
                    },
                    self.tensor(*y_grid),
                    self.tensor(*y),
                )
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
            } => {
                let clips = self.tensor(*grid).rows;
                let (partials, stats) = self.spatial_moments(clips, *groups).ok_or_else(|| {
                    kernels_metal::Error::Backend {
                        op: "spatial.group_norm",
                        detail: format!(
                            "this load reserved no moment plane for {clips} clip(s) at \
                             {groups} group(s)"
                        ),
                    }
                })?;
                spatial::norm::group_norm(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*grid),
                    *groups,
                    self.tensor(*weight),
                    self.tensor(*bias),
                    *eps,
                    *silu,
                    partials,
                    stats,
                    self.tensor(*y),
                )
            }
            Spatial::Attention {
                q,
                k,
                v,
                grid,
                segment,
                sm_scale,
                y,
            } => spatial::attn::attention(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*grid),
                match segment {
                    model_ir::VoxelSegment::Clip => spatial::attn::Segment::Clip,
                    model_ir::VoxelSegment::Frames(n) => spatial::attn::Segment::Frames(*n),
                },
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
            } => spatial::resample::upsample_nearest(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *keep_first_frame,
                self.tensor(*y_grid),
                self.tensor(*y),
            ),
            Spatial::PixelShuffle {
                x,
                grid,
                r,
                trim_t,
                y_grid,
                y,
            } => spatial::resample::pixel_shuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                *trim_t,
                self.tensor(*y_grid),
                self.tensor(*y),
            ),
            Spatial::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => spatial::resample::pixel_unshuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                self.tensor(*y_grid),
                self.tensor(*y),
            ),
            Spatial::AvgDown {
                x,
                grid,
                factor,
                group,
                y_grid,
                y,
            } => spatial::resample::avg_down(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *group,
                self.tensor(*y_grid),
                self.tensor(*y),
            ),
            Spatial::CacheStore {
                x,
                grid,
                frames,
                cache,
                x_out: _,
            } => {
                let x = self.tensor(*x);
                let grid = self.tensor(*grid);
                let pool = self.recurrent(*cache);
                let slots = self.clip_slots().ok_or(kernels_metal::Error::Backend {
                    op: "spatial.cache_store",
                    detail: "a frame-cache store needs the fire's clip slot table, which no \
                             lane of it staged"
                        .to_string(),
                })?;
                spatial::resample::cache_store(self.ctx(), x, x, slots, grid, *frames, pool.state)
            }
            Spatial::Patchify { .. } | Spatial::Unpatchify { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
        }
    }
}

impl model_exec::DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &model_ir::Node) {
        use model_ir::Operands as _;
        if !crate::diag::on().nan_check {
            return;
        }
        let Some(flags) = self.bindings().nan_flags else {
            return;
        };
        let mut outs: Vec<model_ir::ValueId> = Vec::new();
        node.op.outputs(&mut outs);
        for id in outs {
            if !self.resolvable(id) {
                continue;
            }
            let t = self.tensor(id);
            if !matches!(
                t.dtype,
                model_ir::Dtype::F32 | model_ir::Dtype::Bf16 | model_ir::Dtype::F16
            ) {
                continue;
            }
            let _ = kernels_metal::tripwire::nan_check(
                self.ctx(),
                t,
                flags,
                id.0,
                crate::diag::on().nan_limit,
            );
        }
    }
}
