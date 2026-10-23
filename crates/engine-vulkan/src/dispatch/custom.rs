use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl poem_exec::DispatchConv for Run<'_> {
    fn dispatch(&mut self, op: &poem_ir::Conv) -> Result<(), KernelError> {
        self.conv(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
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
                    },
                    t(*y_grid),
                    t(*y),
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
            ),
            Conv::PixelShuffle {
                x,
                grid,
                r,
                trim_t,
                y_grid,
                y,
            } => resample::pixel_shuffle(ctx, t(*x), t(*grid), *r, *trim_t, t(*y_grid), t(*y)),
            Conv::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => resample::pixel_unshuffle(ctx, t(*x), t(*grid), *r, t(*y_grid), t(*y)),
            Conv::AvgDown {
                x,
                grid,
                factor,
                group,
                y_grid,
                y,
            } => resample::avg_down(ctx, t(*x), t(*grid), *factor, *group, t(*y_grid), t(*y)),
            Conv::CacheStore { .. } | Conv::Patchify { .. } | Conv::Unpatchify { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
        }
    }
}

impl poem_exec::DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &poem_ir::Node) {
        use std::io::Write;
        self.split_at(node);
        let Some(dir) = crate::probe::dir() else {
            return;
        };
        if let Err(fault) = crate::probe::flush() {
            eprintln!("probe: flush failed: {fault}");
            return;
        }
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        let seq = crate::probe::next_seq();
        let mut manifest = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(dir.join("manifest.txt"))
            .ok();
        for id in outs {
            let Some(decl) = self.values().get(id.0 as usize) else {
                continue;
            };
            let poem_ir::Ty::Tensor { dtype, .. } = &decl.ty else {
                continue;
            };
            let elem: u64 = match dtype {
                poem_ir::Dtype::Bf16 | poem_ir::Dtype::F16 => 2,
                poem_ir::Dtype::F32 | poem_ir::Dtype::I32 | poem_ir::Dtype::U32 => 4,
                _ => continue,
            };
            let t = self.tensor(id);
            let total = u64::from(t.rows) * u64::from(t.width) * elem;
            if total == 0 || t.buf == kernels_slang::ABSENT {
                continue;
            }
            let Ok(bytes) = self.handles().read(t.buf, total) else {
                continue;
            };
            let name = format!(
                "{seq:04}_{}_L{}_v{}.bin",
                node.op.name().replace(['.', '/'], "_"),
                node.layer.map_or(-1, i64::from),
                id.0
            );
            let _ = std::fs::write(dir.join(&name), &bytes);
            if let Some(m) = manifest.as_mut() {
                let _ = writeln!(
                    m,
                    "{seq} {} {} v{} {:?} {} {} {name}",
                    node.op.name(),
                    node.layer.map_or(-1, i64::from),
                    id.0,
                    dtype,
                    t.rows,
                    t.width
                );
            }
        }
    }
}
