use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl poem_exec::DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &poem_ir::Spatial) -> Result<(), KernelError> {
        self.spatial(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn spatial(&mut self, op: &poem_ir::Spatial) -> Result<(), kernels_vulkan::Error> {
        use kernels_vulkan::spatial;
        use poem_ir::Spatial;
        match op {
            Spatial::Grid { grid, rule, y } => spatial::rule::derive_grid(
                self.ctx(),
                self.tensor(*grid),
                match *rule {
                    poem_ir::GridRule::Conv {
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
                    poem_ir::GridRule::Upsample {
                        factor,
                        keep_first_frame,
                    } => spatial::rule::GridRule::Upsample {
                        factor,
                        keep_first_frame,
                    },
                    poem_ir::GridRule::Shuffle { r, trim_t } => {
                        spatial::rule::GridRule::Shuffle { r, trim_t }
                    }
                    poem_ir::GridRule::Unshuffle { r } => spatial::rule::GridRule::Unshuffle { r },
                    poem_ir::GridRule::AvgDown { factor } => {
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
                    return Err(kernels_vulkan::Error::Unsupported {
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
                            poem_ir::TimePad::Zero => spatial::conv::TimePad::Zero,
                            poem_ir::TimePad::Replicate => spatial::conv::TimePad::Replicate,
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
                    kernels_vulkan::Error::Backend {
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
                    poem_ir::VoxelSegment::Clip => spatial::attn::Segment::Clip,
                    poem_ir::VoxelSegment::Frames(n) => spatial::attn::Segment::Frames(*n),
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
            Spatial::CacheStore { .. } => Err(kernels_vulkan::Error::Unsupported {
                op: "spatial.cache_store",
            }),
            Spatial::Patchify { .. } | Spatial::Unpatchify { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
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
            if total == 0 || t.buf == kernels_vulkan::ABSENT {
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
