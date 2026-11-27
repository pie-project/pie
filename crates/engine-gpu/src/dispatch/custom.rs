use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::backend::Backend;
use crate::run::Run;

impl<B: Backend> DispatchFused for Run<'_, B> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl<B: Backend> poem_exec::DispatchConv for Run<'_, B> {
    fn dispatch(&mut self, op: &poem_ir::Conv) -> Result<(), KernelError> {
        self.conv(op).map_err(crate::error::kernel)
    }
}

impl<B: Backend> Run<'_, B> {
    fn conv(&mut self, op: &poem_ir::Conv) -> Result<(), kernels_slang::Error> {
        use kernels_slang::conv::{conv3d, group_norm, resample, rule};
        use poem_ir::Conv;
        let ctx = self.ctx;
        let t = |id: poem_ir::ValueId| self.tensor(id);
        match op {
            Conv::Grid { grid, rule, y } => rule::derive_grid(
                ctx,
                t(*grid),
                match *rule {
                    poem_ir::GridRule::Conv {
                        k,
                        stride,
                        pad,
                        pad_back,
                        causal_t,
                    } => rule::GridRule::Conv {
                        k,
                        stride,
                        pad,
                        pad_back,
                        causal_t,
                    },
                    poem_ir::GridRule::Upsample {
                        factor,
                        keep_first_frame,
                    } => rule::GridRule::Upsample {
                        factor,
                        keep_first_frame,
                    },
                    poem_ir::GridRule::Shuffle { r, trim_t } => {
                        rule::GridRule::Shuffle { r, trim_t }
                    }
                    poem_ir::GridRule::Unshuffle { r } => rule::GridRule::Unshuffle { r },
                    poem_ir::GridRule::AvgDown { factor } => rule::GridRule::AvgDown { factor },
                },
                t(*y),
            ),
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
                if cache.is_some() {
                    return Err(kernels_slang::Error::Unsupported {
                        op: "conv.conv3d (with a frame cache)",
                    });
                }
                conv3d::conv3d(
                    ctx,
                    t(*x),
                    t(*grid),
                    t(*w),
                    bias.map(t),
                    conv3d::Conv3d {
                        k: *k,
                        stride: *stride,
                        pad: *pad,
                        pad_back: *pad_back,
                        causal_t: *causal_t,
                        time_pad: match time_pad {
                            poem_ir::TimePad::Zero => conv3d::TimePad::Zero,
                            poem_ir::TimePad::Replicate => conv3d::TimePad::Replicate,
                        },
                        dilation: *dilation,
                        depthwise: *depthwise,
                    },
                    t(*y_grid),
                    t(*y),
                    [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
                )
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
            } => {
                let clips = self.tensor(*grid).rows;
                let (partials, stats) = self.spatial_moments(clips, *groups).ok_or_else(|| {
                    kernels_slang::Error::Backend {
                        op: "conv.group_norm",
                        detail: format!(
                            "this load reserved no moment plane for {clips} clip(s) at \
                             {groups} group(s)"
                        ),
                    }
                })?;
                group_norm::group_norm(
                    ctx,
                    t(*x),
                    t(*grid),
                    *groups,
                    t(*weight),
                    t(*bias),
                    *eps,
                    *silu,
                    partials,
                    stats,
                    t(*y),
                    self.grid_base(*grid, *x),
                )
            }
            Conv::Upsample {
                x,
                grid,
                factor,
                keep_first_frame,
                y_grid,
                y,
            } => resample::upsample_nearest(
                ctx,
                t(*x),
                t(*grid),
                *factor,
                *keep_first_frame,
                t(*y_grid),
                t(*y),
                [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
            ),
            Conv::PixelShuffle {
                x,
                grid,
                r,
                trim_t,
                y_grid,
                y,
            } => resample::pixel_shuffle(
                ctx,
                t(*x),
                t(*grid),
                *r,
                *trim_t,
                t(*y_grid),
                t(*y),
                [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
            ),
            Conv::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => resample::pixel_unshuffle(
                ctx,
                t(*x),
                t(*grid),
                *r,
                t(*y_grid),
                t(*y),
                [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
            ),
            Conv::AvgDown {
                x,
                grid,
                factor,
                group,
                y_grid,
                y,
            } => resample::avg_down(
                ctx,
                t(*x),
                t(*grid),
                *factor,
                *group,
                t(*y_grid),
                t(*y),
                [self.grid_base(*grid, *x), self.grid_base(*y_grid, *y)],
            ),
            Conv::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => resample::pixel_unshuffle(
                ctx,
                t(*x),
                t(*grid),
                *p,
                t(*tgrid),
                t(*y),
                [self.grid_base(*grid, *x), self.grid_base(*tgrid, *y)],
            ),
            Conv::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => resample::pixel_shuffle(
                ctx,
                t(*x),
                t(*tgrid),
                *p,
                0,
                t(*grid),
                t(*y),
                [self.grid_base(*tgrid, *x), self.grid_base(*grid, *y)],
            ),
            Conv::Cells {
                tgrid,
                positions,
                axis,
                y,
            } => rule::cells(
                ctx,
                t(*tgrid),
                t(*positions),
                *axis,
                t(*y),
                self.grid_base(*tgrid, *y),
            ),
            Conv::CacheStore { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
        }
    }
}

impl<B: Backend> poem_exec::DispatchProbe for Run<'_, B> {
    fn probe(&mut self, node: &poem_ir::Node) {
        self.split_at(node);
    }
}
