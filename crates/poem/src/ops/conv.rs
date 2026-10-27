use super::*;
use poem_ir::{Conv, GridRule, ParamLayout, RowAxis, TimePad};

fn grid_ty() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Clips, Dim::Const(4)],
        dtype: Dtype::I32,
    }
}

pub(super) fn expect_grid(what: &str, grid: &Value) {
    assert!(
        grid.rows() == Dim::Clips && grid.width() == 4 && grid.dtype() == Dtype::I32,
        "{what} must be a `[Clips, 4]` i32 grid table, not {:?}",
        grid.ty()
    );
}

pub(super) fn expect_voxels(what: &str, x: &Value) {
    assert!(
        x.rows().axis() == Some(RowAxis::Voxels),
        "{what} must live on the voxel axis, not {:?}",
        x.rows()
    );
}

fn scaled(rows: Dim, (num, den): (u32, u32)) -> Dim {
    let (n, d) = rows
        .voxel_scale()
        .unwrap_or_else(|| panic!("{rows:?} is not a voxel row count"));
    Dim::voxels_scaled(n * num, d * den)
}

fn volume(r: [u32; 3]) -> u32 {
    r[0] * r[1] * r[2]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvShape {
    pub k: [u32; 3],
    pub stride: [u32; 3],
    pub pad: [u32; 3],
    pub pad_back: [u32; 3],
    pub causal_t: bool,
    pub time_pad: TimePad,
}

impl ConvShape {
    #[must_use]
    pub const fn conv2d(k: [u32; 2], stride: [u32; 2], pad: [u32; 2]) -> ConvShape {
        ConvShape {
            k: [1, k[0], k[1]],
            stride: [1, stride[0], stride[1]],
            pad: [0, pad[0], pad[1]],
            pad_back: [0, pad[0], pad[1]],
            causal_t: false,
            time_pad: TimePad::Zero,
        }
    }

    #[must_use]
    pub const fn conv3d(k: [u32; 3], stride: [u32; 3], pad: [u32; 3]) -> ConvShape {
        ConvShape {
            k,
            stride,
            pad,
            pad_back: pad,
            causal_t: false,
            time_pad: TimePad::Zero,
        }
    }

    #[must_use]
    pub const fn pad_back(mut self, back: [u32; 3]) -> ConvShape {
        self.pad_back = back;
        self
    }

    #[must_use]
    pub const fn same3() -> ConvShape {
        ConvShape::conv3d([3, 3, 3], [1, 1, 1], [1, 1, 1])
    }

    #[must_use]
    pub const fn causal(mut self, time_pad: TimePad) -> ConvShape {
        self.causal_t = true;
        self.pad[0] = self.k[0] - 1;
        self.pad_back[0] = 0;
        self.time_pad = time_pad;
        self
    }

    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }

    #[must_use]
    pub const fn rule(&self) -> GridRule {
        GridRule::Conv {
            k: self.k,
            stride: self.stride,
            pad: self.pad,
            pad_back: self.pad_back,
            causal_t: self.causal_t,
        }
    }
}

#[must_use]
pub fn grid(grid: &Value, rule: GridRule) -> Value {
    expect_grid("`conv::grid`'s input", grid);
    let r = grid.rec();
    let y = r.fresh(grid_ty());
    r.push(
        Conv::Grid {
            grid: grid.id(),
            rule,
            y: y.id(),
        },
        &[grid],
    );
    y
}

#[must_use]
pub fn conv3d(
    x: &Value,
    grid: &Value,
    w: &Weight,
    bias: Option<&Weight>,
    conv: ConvShape,
    cache: Option<ValueId>,
) -> (Value, Value) {
    expect_voxels("`conv::conv3d`'s input", x);
    expect_grid("`conv::conv3d`'s grid", grid);
    assert_eq!(x.dtype(), Dtype::Bf16, "`conv::conv3d` reads bf16 rows");
    let taps = conv.taps();
    // A convolution reads its weight taps major, `c_in` channels minor: the
    // order its kernels walk, which a package need not state.
    let laid = ParamLayout::ConvTapsMajor {
        c_in: x.width() as u32,
        taps,
    };
    assert!(
        w.layout == ParamLayout::Natural || w.layout == laid,
        "`{}` is laid {:?}; this convolution reads it {laid:?}",
        w.name,
        w.layout
    );
    let w = &Weight {
        layout: laid,
        ..w.clone()
    };
    assert!(
        w.shape.len() == 2 && w.shape[1] == x.width() * u64::from(taps),
        "`{}` is {:?}; `[C_out, {}·{taps}]` is what {} input channels convolve through",
        w.name,
        w.shape,
        x.width(),
        x.width()
    );
    if let Some(b) = bias {
        assert!(
            b.shape == vec![w.dim(0)] && b.dtype == Dtype::F32,
            "`{}` is {:?} {:?}; a conv bias is `[C_out]` f32",
            b.name,
            b.shape,
            b.dtype
        );
    }
    assert!(
        cache.is_none() || (conv.causal_t && conv.pad[0] > 0),
        "a frame cache is read only under `causal_t` with a front pad"
    );
    let r = x.rec();
    let y_grid = self::grid(grid, conv.rule());
    let y = r.fresh(tensor(
        scaled(x.rows(), conv.rule().row_scale()),
        w.dim(0),
        x.dtype(),
    ));
    r.push(
        Conv::Conv3d {
            x: x.id(),
            grid: grid.id(),
            w: r.weight(w),
            bias: bias.map(|b| r.weight(b)),
            k: conv.k,
            stride: conv.stride,
            pad: conv.pad,
            pad_back: conv.pad_back,
            causal_t: conv.causal_t,
            time_pad: conv.time_pad,
            cache,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

#[must_use]
pub fn group_norm(
    x: &Value,
    grid: &Value,
    groups: u32,
    weight: &Weight,
    bias: &Weight,
    eps: f32,
    silu: bool,
) -> Value {
    expect_voxels("`conv::group_norm`'s input", x);
    expect_grid("`conv::group_norm`'s grid", grid);
    assert!(
        groups > 0 && x.width().is_multiple_of(u64::from(groups)),
        "{} channels do not split into {groups} groups",
        x.width()
    );
    for (what, w) in [("weight", weight), ("bias", bias)] {
        assert!(
            w.shape == vec![x.width()] && w.dtype == Dtype::F32,
            "the group norm {what} `{}` is {:?} {:?}; expected [{}] f32",
            w.name,
            w.shape,
            w.dtype,
            x.width()
        );
    }
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Conv::GroupNorm {
            x: x.id(),
            grid: grid.id(),
            groups,
            weight: r.weight(weight),
            bias: r.weight(bias),
            eps,
            silu,
            y: y.id(),
        },
        &[x, grid],
    );
    y
}

#[must_use]
pub fn upsample_nearest(
    x: &Value,
    grid: &Value,
    factor: [u32; 3],
    keep_first_frame: bool,
) -> (Value, Value) {
    expect_voxels("`conv::upsample_nearest`'s input", x);
    expect_grid("`conv::upsample_nearest`'s grid", grid);
    assert!(
        volume(factor) > 0,
        "an upsample factor of {factor:?} is empty"
    );
    let rule = GridRule::Upsample {
        factor,
        keep_first_frame,
    };
    let r = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = r.fresh(tensor(
        scaled(x.rows(), rule.row_scale()),
        x.width(),
        x.dtype(),
    ));
    r.push(
        Conv::Upsample {
            x: x.id(),
            grid: grid.id(),
            factor,
            keep_first_frame,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

#[must_use]
pub fn pixel_shuffle(x: &Value, grid: &Value, r: [u32; 3]) -> (Value, Value) {
    pixel_shuffle_trimming(x, grid, r, 0)
}

#[must_use]
pub fn pixel_shuffle_trimming(x: &Value, grid: &Value, r: [u32; 3], trim_t: u32) -> (Value, Value) {
    expect_voxels("`conv::pixel_shuffle`'s input", x);
    expect_grid("`conv::pixel_shuffle`'s grid", grid);
    let vol = volume(r);
    assert!(
        vol > 0 && x.width().is_multiple_of(u64::from(vol)),
        "{} channels do not unpack by a {r:?} block",
        x.width()
    );
    let rule = GridRule::Shuffle { r, trim_t };
    let rec = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = rec.fresh(tensor(
        scaled(x.rows(), rule.row_scale()),
        x.width() / u64::from(vol),
        x.dtype(),
    ));
    rec.push(
        Conv::PixelShuffle {
            x: x.id(),
            grid: grid.id(),
            r,
            trim_t,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

#[must_use]
pub fn pixel_unshuffle(x: &Value, grid: &Value, r: [u32; 3]) -> (Value, Value) {
    expect_voxels("`conv::pixel_unshuffle`'s input", x);
    expect_grid("`conv::pixel_unshuffle`'s grid", grid);
    let vol = volume(r);
    assert!(vol > 0, "an unshuffle block of {r:?} is empty");
    let rule = GridRule::Unshuffle { r };
    let rec = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = rec.fresh(tensor(
        scaled(x.rows(), rule.row_scale()),
        x.width() * u64::from(vol),
        x.dtype(),
    ));
    rec.push(
        Conv::PixelUnshuffle {
            x: x.id(),
            grid: grid.id(),
            r,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

#[must_use]
pub fn avg_down(x: &Value, grid: &Value, factor: [u32; 3], group: u32) -> (Value, Value) {
    expect_voxels("`conv::avg_down`'s input", x);
    expect_grid("`conv::avg_down`'s grid", grid);
    let vol = volume(factor);
    assert!(vol > 0, "an avg-down block of {factor:?} is empty");
    let widened = x.width() * u64::from(vol);
    assert!(
        group > 0 && widened.is_multiple_of(u64::from(group)),
        "{widened} widened channels do not fold into groups of {group}"
    );
    let rule = GridRule::AvgDown { factor };
    let rec = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = rec.fresh(tensor(
        scaled(x.rows(), rule.row_scale()),
        widened / u64::from(group),
        x.dtype(),
    ));
    rec.push(
        Conv::AvgDown {
            x: x.id(),
            grid: grid.id(),
            factor,
            group,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

#[must_use]
pub fn store_frames(x: &Value, grid: &Value, cache: ValueId, frames: u32) -> Value {
    expect_voxels("`conv::store_frames`'s input", x);
    expect_grid("`conv::store_frames`'s grid", grid);
    assert!(frames > 0, "a cache of no frames is no cache");
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Conv::CacheStore {
            x: x.id(),
            grid: grid.id(),
            frames,
            cache,
            x_out: x_out.id(),
        },
        &[x, grid],
    );
    x_out
}

#[must_use]
pub fn patchify(x: &Value, grid: &Value, p: [u32; 3], tgrid: &Value) -> Value {
    expect_voxels("`conv::patchify`'s input", x);
    expect_grid("`conv::patchify`'s grid", grid);
    expect_grid("`conv::patchify`'s token grid", tgrid);
    let vol = volume(p);
    assert!(vol > 0, "a patch of {p:?} is empty");
    let r = x.rec();
    let y = r.fresh(tensor(Dim::Tokens, x.width() * u64::from(vol), x.dtype()));
    r.push(
        Conv::Patchify {
            x: x.id(),
            grid: grid.id(),
            p,
            tgrid: tgrid.id(),
            y: y.id(),
        },
        &[x, grid, tgrid],
    );
    y
}

#[must_use]
pub fn unpatchify(x: &Value, tgrid: &Value, p: [u32; 3], grid: &Value) -> Value {
    assert!(
        x.rows().axis() == Some(RowAxis::Tokens),
        "`conv::unpatchify` reads token rows, not {:?}",
        x.rows()
    );
    expect_grid("`conv::unpatchify`'s token grid", tgrid);
    expect_grid("`conv::unpatchify`'s grid", grid);
    let vol = volume(p);
    assert!(
        vol > 0 && x.width().is_multiple_of(u64::from(vol)),
        "{} channels do not unpack by a {p:?} patch",
        x.width()
    );
    let r = x.rec();
    let y = r.fresh(tensor(Dim::Voxels, x.width() / u64::from(vol), x.dtype()));
    r.push(
        Conv::Unpatchify {
            x: x.id(),
            tgrid: tgrid.id(),
            p,
            grid: grid.id(),
            y: y.id(),
        },
        &[x, tgrid, grid],
    );
    y
}

/// Each token row's cell, as [`poem_ir::Conv::Cells`].
#[must_use]
pub fn cells(tgrid: &Value, positions: &Value, axis: u32) -> Value {
    expect_grid("`conv::cells`'s token grid", tgrid);
    assert!(
        matches!(positions.ty(), Ty::Tensor { shape, dtype: Dtype::I32 }
            if shape.len() == 1 && shape[0].axis() == Some(RowAxis::Tokens)),
        "`conv::cells` reads each token row's i32 runtime position, not {:?}",
        positions.ty()
    );
    assert!(
        axis < 4,
        "a rope turns at most four axes, not column {axis}"
    );
    let r = tgrid.rec();
    let y = r.fresh(tensor(
        positions.rows(),
        u64::from(axis.max(2) + 1 + 3),
        Dtype::F32,
    ));
    r.push(
        Conv::Cells {
            tgrid: tgrid.id(),
            positions: positions.id(),
            axis,
            y: y.id(),
        },
        &[tgrid, positions],
    );
    y
}
