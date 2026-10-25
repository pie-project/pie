use kernels_cuda::conv;
use kernels_cuda::tensor::Tensor;
use poem_exec::{DispatchConv, KernelError};
use poem_ir::{Conv, GridRule, TimePad};

use crate::run::Run;

impl DispatchConv for Run<'_> {
    fn dispatch(&mut self, op: &Conv) -> Result<(), KernelError> {
        self.conv(op).map_err(crate::error::kernel)
    }
}

fn rule(rule: GridRule) -> kernels_cuda::elemwise::posenc::GridRule {
    match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => kernels_cuda::elemwise::posenc::GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => kernels_cuda::elemwise::posenc::GridRule::Upsample {
            factor,
            keep_first_frame,
        },
        GridRule::Shuffle { r, trim_t } => {
            kernels_cuda::elemwise::posenc::GridRule::Shuffle { r, trim_t }
        }
        GridRule::Unshuffle { r } => kernels_cuda::elemwise::posenc::GridRule::Unshuffle { r },
        GridRule::AvgDown { factor } => {
            kernels_cuda::elemwise::posenc::GridRule::AvgDown { factor }
        }
    }
}

impl Run<'_> {
    fn conv(&mut self, op: &Conv) -> Result<(), kernels_cuda::Error> {
        match op {
            Conv::Grid { grid, rule: how, y } => kernels_cuda::elemwise::posenc::derive_grid(
                self.ctx(),
                self.tensor(*grid),
                rule(*how),
                &mut self.tensor(*y),
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
                cache,
                y_grid,
                y,
            } => {
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
                let x = self.tensor(*x);
                let grid = self.tensor(*grid);
                let bias = bias.map(|b| self.tensor(b));
                let Some(state) = cache else {
                    return conv::conv3d(
                        self.ctx(),
                        x,
                        grid,
                        self.tensor(*w),
                        bias,
                        conv,
                        None,
                        &mut self.tensor(*y),
                        self.tensor(*y_grid),
                    );
                };
                let frames = conv.pad[0];
                let pool = self.recurrent(*state);
                let slot_ids = self
                    .clip_slots()
                    .ok_or_else(|| kernels_cuda::Error::Backend {
                        op: "conv.conv3d",
                        detail:
                            "a causal convolution with a frame cache needs the fire's clip slot \
                             table, which no lane of it staged"
                                .to_string(),
                    })?;
                let rows = x.rows.saturating_mul(frames);
                let bytes = u64::from(rows) * u64::from(x.width) * 2;
                let scratch = self.ctx().scratch(
                    "conv.conv3d",
                    "conv.conv3d.frame_cache",
                    usize::try_from(bytes).unwrap_or(usize::MAX),
                )? as u64;
                let mut frame_cache = Tensor::new(scratch, rows, x.width, x.dtype);
                conv::cache_gather(
                    self.ctx(),
                    pool.slab,
                    slot_ids,
                    grid,
                    frames,
                    &mut frame_cache,
                )?;
                conv::conv3d(
                    self.ctx(),
                    x,
                    grid,
                    self.tensor(*w),
                    bias,
                    conv,
                    Some(frame_cache),
                    &mut self.tensor(*y),
                    self.tensor(*y_grid),
                )?;
                let mut slab = pool.slab;
                conv::cache_store(
                    self.ctx(),
                    x,
                    frame_cache,
                    slot_ids,
                    grid,
                    frames,
                    &mut slab,
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
            } => kernels_cuda::elemwise::norm::group_norm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *groups,
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                *silu,
                &mut self.tensor(*y),
            ),
            Conv::Upsample {
                x,
                grid,
                factor,
                keep_first_frame,
                y_grid,
                y,
            } => conv::upsample_nearest(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *keep_first_frame,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Conv::PixelShuffle {
                x,
                grid,
                r,
                trim_t,
                y_grid,
                y,
            } => conv::pixel_shuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                *trim_t,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Conv::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => conv::pixel_unshuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Conv::AvgDown {
                x,
                grid,
                factor,
                group,
                y_grid,
                y,
            } => conv::avg_down(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *group,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Conv::CacheStore {
                x,
                grid,
                frames,
                cache,
                x_out: _,
            } => {
                let x = self.tensor(*x);
                let grid = self.tensor(*grid);
                let pool = self.recurrent(*cache);
                let slot_ids = self
                    .clip_slots()
                    .ok_or_else(|| kernels_cuda::Error::Backend {
                        op: "conv.cache_store",
                        detail:
                            "a frame-cache store needs the fire's clip slot table, which no lane                              of it staged"
                                .to_string(),
                    })?;
                let mut slab = pool.slab;
                conv::cache_store(self.ctx(), x, x, slot_ids, grid, *frames, &mut slab)
            }
            Conv::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => conv::patchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *p,
                &mut self.tensor(*y),
                self.tensor(*tgrid),
            ),
            Conv::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => conv::unpatchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*tgrid),
                *p,
                &mut self.tensor(*y),
                self.tensor(*grid),
            ),
            Conv::Cells {
                tgrid,
                positions,
                axis,
                y,
            } => kernels_cuda::elemwise::posenc::cells(
                self.ctx(),
                self.tensor(*tgrid),
                self.tensor(*positions),
                *axis,
                &mut self.tensor(*y),
            ),
        }
    }
}
