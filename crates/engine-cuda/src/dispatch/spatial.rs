use kernels_cuda::spatial;
use kernels_cuda::tensor::Tensor;
use model_exec::{DispatchSpatial, KernelError};
use model_ir::{GridRule, Spatial, TimePad, VoxelSegment};

use crate::run::Run;

impl DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
        self.spatial(op).map_err(crate::error::kernel)
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
    fn spatial(&mut self, op: &Spatial) -> Result<(), kernels_cuda::Error> {
        match op {
            Spatial::Grid { grid, rule: how, y } => spatial::derive_grid(
                self.ctx(),
                self.tensor(*grid),
                rule(*how),
                &mut self.tensor(*y),
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
                let Some(state) = cache else {
                    return spatial::conv3d(
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
                        op: "spatial.conv3d",
                        detail:
                            "a causal convolution with a frame cache needs the fire's clip slot \
                             table, which no lane of it staged"
                                .to_string(),
                    })?;
                let rows = x.rows.saturating_mul(frames);
                let bytes = u64::from(rows) * u64::from(x.width) * 2;
                let scratch = self.ctx().scratch(
                    "spatial.conv3d",
                    "spatial.conv3d.frame_cache",
                    usize::try_from(bytes).unwrap_or(usize::MAX),
                )? as u64;
                let mut frame_cache = Tensor::new(scratch, rows, x.width, x.dtype);
                spatial::cache_gather(
                    self.ctx(),
                    pool.slab,
                    slot_ids,
                    grid,
                    frames,
                    &mut frame_cache,
                )?;
                spatial::conv3d(
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
                spatial::cache_store(
                    self.ctx(),
                    x,
                    frame_cache,
                    slot_ids,
                    grid,
                    frames,
                    &mut slab,
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
            } => spatial::group_norm(
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                let grid = self.tensor(*grid);
                let pool = self.recurrent(*cache);
                let slot_ids = self
                    .clip_slots()
                    .ok_or_else(|| kernels_cuda::Error::Backend {
                        op: "spatial.cache_store",
                        detail:
                            "a frame-cache store needs the fire's clip slot table, which no lane                              of it staged"
                                .to_string(),
                    })?;
                let mut slab = pool.slab;
                spatial::cache_store(self.ctx(), x, x, slot_ids, grid, *frames, &mut slab)
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
                self.tensor(*grid),
            ),
        }
    }
}
