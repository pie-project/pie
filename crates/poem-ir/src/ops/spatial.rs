use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimePad {
    Zero,
    Replicate,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoxelSegment {
    #[default]
    Clip,
    Frames(u32),
}

impl VoxelSegment {
    #[must_use]
    pub fn frames(self, t: u32, frame: u32) -> (u32, u32) {
        match self {
            VoxelSegment::Clip => (0, t),
            VoxelSegment::Frames(0) => (frame, frame),
            VoxelSegment::Frames(per) => {
                let begin = frame / per * per;
                (begin, (begin + per).min(t))
            }
        }
    }

    #[must_use]
    pub fn bounds(self, [t, h, w]: [u32; 3], row: u32) -> Option<(u32, u32)> {
        let plane = h * w;
        if plane == 0 || row >= t * plane {
            return None;
        }
        let (begin, end) = self.frames(t, row / plane);
        Some((begin * plane, end * plane))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GridRule {
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
    },
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    Shuffle {
        r: [u32; 3],
        trim_t: u32,
    },
    Unshuffle {
        r: [u32; 3],
    },
    AvgDown {
        factor: [u32; 3],
    },
}

impl GridRule {
    #[must_use]
    pub fn out_extent(self, [t, h, w]: [u32; 3]) -> Option<[u32; 3]> {
        match self {
            GridRule::Conv {
                k,
                stride,
                pad,
                pad_back,
                causal_t,
            } => {
                let axis = |n: u32, k: u32, s: u32, front: u32, back: u32| {
                    (n + front + back)
                        .checked_sub(k)
                        .map(|span| span / s.max(1) + 1)
                };
                let back_t = if causal_t { 0 } else { pad_back[0] };
                Some([
                    axis(t, k[0], stride[0], pad[0], back_t)?,
                    axis(h, k[1], stride[1], pad[1], pad_back[1])?,
                    axis(w, k[2], stride[2], pad[2], pad_back[2])?,
                ])
            }
            GridRule::Upsample {
                factor,
                keep_first_frame,
            } => {
                let t_out = if keep_first_frame && t > 0 {
                    1 + (t - 1) * factor[0]
                } else {
                    t * factor[0]
                };
                Some([t_out, h * factor[1], w * factor[2]])
            }
            GridRule::Shuffle { r, trim_t } => {
                let t_out = (t * r[0]).checked_sub(trim_t).filter(|n| *n > 0)?;
                Some([t_out, h * r[1], w * r[2]])
            }
            GridRule::Unshuffle { r } => {
                if r.contains(&0) || t % r[0] != 0 || h % r[1] != 0 || w % r[2] != 0 {
                    return None;
                }
                Some([t / r[0], h / r[1], w / r[2]])
            }
            GridRule::AvgDown { factor } => {
                if factor.contains(&0) || h % factor[1] != 0 || w % factor[2] != 0 {
                    return None;
                }
                Some([t.div_ceil(factor[0]), h / factor[1], w / factor[2]])
            }
        }
    }

    #[must_use]
    pub fn growth(self) -> u32 {
        match self {
            GridRule::Conv { .. } | GridRule::Unshuffle { .. } | GridRule::AvgDown { .. } => 1,
            GridRule::Upsample { factor, .. } => factor[0] * factor[1] * factor[2],
            GridRule::Shuffle { r, .. } => r[0] * r[1] * r[2],
        }
    }

    #[must_use]
    pub fn apply(self, grid: &[i32]) -> Option<Vec<i32>> {
        let mut out = Vec::with_capacity(grid.len());
        let mut off: i64 = 0;
        for clip in grid.as_chunks::<4>().0 {
            let boxed = [clip[0], clip[1], clip[2]].map(|n| u32::try_from(n).ok());
            let [t, h, w] = self.out_extent([boxed[0]?, boxed[1]?, boxed[2]?])?;
            out.extend_from_slice(&[t as i32, h as i32, w as i32, i32::try_from(off).ok()?]);
            off += i64::from(t) * i64::from(h) * i64::from(w);
        }
        Some(out)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Spatial {
    Grid {
        grid: ValueId,
        rule: GridRule,
        y: ValueId,
    },
    Conv3d {
        x: ValueId,
        grid: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
        time_pad: TimePad,
        cache: Option<ValueId>,
        y_grid: ValueId,
        y: ValueId,
    },
    GroupNorm {
        x: ValueId,
        grid: ValueId,
        groups: u32,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        silu: bool,
        y: ValueId,
    },
    Attention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        grid: ValueId,
        segment: VoxelSegment,
        sm_scale: f32,
        y: ValueId,
    },
    UpsampleNearest {
        x: ValueId,
        grid: ValueId,
        factor: [u32; 3],
        keep_first_frame: bool,
        y_grid: ValueId,
        y: ValueId,
    },
    PixelShuffle {
        x: ValueId,
        grid: ValueId,
        r: [u32; 3],
        trim_t: u32,
        y_grid: ValueId,
        y: ValueId,
    },
    PixelUnshuffle {
        x: ValueId,
        grid: ValueId,
        r: [u32; 3],
        y_grid: ValueId,
        y: ValueId,
    },
    AvgDown {
        x: ValueId,
        grid: ValueId,
        factor: [u32; 3],
        group: u32,
        y_grid: ValueId,
        y: ValueId,
    },
    CacheStore {
        x: ValueId,
        grid: ValueId,
        frames: u32,
        cache: ValueId,
        x_out: ValueId,
    },
    Patchify {
        x: ValueId,
        grid: ValueId,
        p: [u32; 3],
        tgrid: ValueId,
        y: ValueId,
    },
    Unpatchify {
        x: ValueId,
        tgrid: ValueId,
        p: [u32; 3],
        grid: ValueId,
        y: ValueId,
    },
}

impl Operands for Spatial {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Grid { grid, .. } => sink.push(*grid),
            Self::Conv3d {
                x,
                grid,
                w,
                bias,
                cache,
                y_grid,
                ..
            } => {
                sink.extend([*x, *grid, *w]);
                sink.extend(bias.iter().copied());
                sink.extend(cache.iter().copied());
                sink.push(*y_grid);
            }
            Self::GroupNorm {
                x,
                grid,
                weight,
                bias,
                ..
            } => sink.extend([*x, *grid, *weight, *bias]),
            Self::Attention { q, k, v, grid, .. } => sink.extend([*q, *k, *v, *grid]),
            Self::UpsampleNearest {
                x, grid, y_grid, ..
            }
            | Self::PixelShuffle {
                x, grid, y_grid, ..
            }
            | Self::PixelUnshuffle {
                x, grid, y_grid, ..
            }
            | Self::AvgDown {
                x, grid, y_grid, ..
            } => sink.extend([*x, *grid, *y_grid]),
            Self::CacheStore { x, grid, cache, .. } => sink.extend([*x, *grid, *cache]),
            Self::Patchify { x, grid, tgrid, .. } => sink.extend([*x, *grid, *tgrid]),
            Self::Unpatchify { x, tgrid, grid, .. } => sink.extend([*x, *tgrid, *grid]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Grid { y, .. }
            | Self::Conv3d { y, .. }
            | Self::GroupNorm { y, .. }
            | Self::Attention { y, .. }
            | Self::UpsampleNearest { y, .. }
            | Self::PixelShuffle { y, .. }
            | Self::PixelUnshuffle { y, .. }
            | Self::AvgDown { y, .. }
            | Self::Patchify { y, .. }
            | Self::Unpatchify { y, .. } => sink.push(*y),
            Self::CacheStore { x_out, .. } => sink.push(*x_out),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Grid { .. }
            | Self::Conv3d { .. }
            | Self::GroupNorm { .. }
            | Self::Attention { .. }
            | Self::UpsampleNearest { .. }
            | Self::PixelShuffle { .. }
            | Self::PixelUnshuffle { .. }
            | Self::AvgDown { .. }
            | Self::Patchify { .. }
            | Self::Unpatchify { .. } => {}
            Self::CacheStore { x_out, x, .. } => sink.push((*x_out, *x)),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Grid { .. } => "spatial.grid",
            Self::Conv3d { .. } => "spatial.conv3d",
            Self::GroupNorm { .. } => "spatial.group_norm",
            Self::Attention { .. } => "spatial.attention",
            Self::UpsampleNearest { .. } => "spatial.upsample_nearest",
            Self::PixelShuffle { .. } => "spatial.pixel_shuffle",
            Self::PixelUnshuffle { .. } => "spatial.pixel_unshuffle",
            Self::AvgDown { .. } => "spatial.avg_down",
            Self::CacheStore { .. } => "spatial.cache_store",
            Self::Patchify { .. } => "spatial.patchify",
            Self::Unpatchify { .. } => "spatial.unpatchify",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GridRule, VoxelSegment};

    #[test]
    fn spatial_every_case() {
        a_grid_rule_maps_boxes_the_way_torch_does();
        a_trimmed_shuffle_drops_its_anchor_frames_from_the_box();
        an_avg_down_pads_its_time_axis_to_the_block();
        a_voxel_segment_reads_its_block_off_the_clips_box();
    }

    fn a_grid_rule_maps_boxes_the_way_torch_does() {
        let conv = GridRule::Conv {
            k: [3, 3, 3],
            stride: [1, 2, 2],
            pad: [1, 1, 1],
            pad_back: [1, 1, 1],
            causal_t: false,
        };
        assert_eq!(conv.out_extent([5, 7, 8]), Some([5, 4, 4]));
        let causal = GridRule::Conv {
            k: [3, 3, 3],
            stride: [1, 1, 1],
            pad: [2, 1, 1],
            pad_back: [2, 1, 1],
            causal_t: true,
        };
        assert_eq!(causal.out_extent([5, 7, 8]), Some([5, 7, 8]));
        let down = GridRule::Conv {
            k: [1, 3, 3],
            stride: [1, 2, 2],
            pad: [0, 0, 0],
            pad_back: [0, 1, 1],
            causal_t: false,
        };
        assert_eq!(down.out_extent([1, 64, 64]), Some([1, 32, 32]));
        assert_eq!(down.out_extent([1, 7, 9]), Some([1, 3, 4]));
        let up = GridRule::Upsample {
            factor: [2, 2, 2],
            keep_first_frame: true,
        };
        assert_eq!(up.out_extent([3, 4, 4]), Some([5, 8, 8]));
        assert_eq!(up.growth(), 8);
        assert_eq!(
            GridRule::Unshuffle { r: [1, 2, 2] }.out_extent([1, 3, 4]),
            None
        );
        let table = up.apply(&[1, 2, 2, 0, 3, 4, 4, 4]).expect("both boxes map");
        assert_eq!(table, vec![1, 4, 4, 0, 5, 8, 8, 16]);
    }

    fn a_trimmed_shuffle_drops_its_anchor_frames_from_the_box() {
        let plain = GridRule::Shuffle {
            r: [2, 2, 2],
            trim_t: 0,
        };
        let trimmed = GridRule::Shuffle {
            r: [2, 2, 2],
            trim_t: 1,
        };
        assert_eq!(plain.out_extent([3, 4, 5]), Some([6, 8, 10]));
        assert_eq!(trimmed.out_extent([3, 4, 5]), Some([5, 8, 10]));
        assert_eq!(trimmed.growth(), 8);
        assert_eq!(
            trimmed.apply(&[1, 1, 1, 0, 3, 4, 5, 1]),
            Some(vec![1, 2, 2, 0, 5, 8, 10, 4])
        );
        assert_eq!(
            GridRule::Shuffle {
                r: [1, 2, 2],
                trim_t: 1
            }
            .out_extent([1, 2, 2]),
            None
        );
    }

    fn an_avg_down_pads_its_time_axis_to_the_block() {
        let two = GridRule::AvgDown { factor: [2, 2, 2] };
        assert_eq!(two.out_extent([4, 8, 12]), Some([2, 4, 6]));
        assert_eq!(two.out_extent([1, 8, 12]), Some([1, 4, 6]));
        assert_eq!(
            GridRule::Unshuffle { r: [2, 2, 2] }.out_extent([1, 8, 12]),
            None
        );
        assert_eq!(two.out_extent([3, 8, 12]), Some([2, 4, 6]));
        assert_eq!(two.out_extent([2, 7, 12]), None);
        assert_eq!(
            GridRule::AvgDown { factor: [1, 2, 2] }.out_extent([5, 8, 12]),
            Some([5, 4, 6])
        );
        assert_eq!(
            GridRule::AvgDown { factor: [1, 1, 1] }.out_extent([5, 8, 12]),
            Some([5, 8, 12])
        );
        assert_eq!(two.growth(), 1);
        assert_eq!(
            two.apply(&[1, 2, 2, 0, 4, 2, 2, 1]),
            Some(vec![1, 1, 1, 0, 2, 1, 1, 1])
        );
    }

    fn a_voxel_segment_reads_its_block_off_the_clips_box() {
        let clip = [3, 2, 2];
        assert_eq!(VoxelSegment::Clip.bounds(clip, 0), Some((0, 12)));
        assert_eq!(VoxelSegment::Clip.bounds(clip, 11), Some((0, 12)));
        for (row, want) in [(0, (0, 4)), (3, (0, 4)), (4, (4, 8)), (11, (8, 12))] {
            assert_eq!(VoxelSegment::Frames(1).bounds(clip, row), Some(want));
        }
        assert_eq!(VoxelSegment::Frames(2).bounds(clip, 0), Some((0, 8)));
        assert_eq!(VoxelSegment::Frames(2).bounds(clip, 9), Some((8, 12)));
        assert_eq!(VoxelSegment::Frames(1).bounds(clip, 12), None);
    }
}
