use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, kernel, refuse, stated,
};
use crate::tensor::Tensor;
use crate::voxels::lane_pair;
use dtype::Dtype;

const FILE: &str = "conv/conv3d.cu";

const OP: &str = "conv.conv3d";

const DIRECT_TILE: [u32; 2] = [64, 64];

const DIRECT_BLOCK: u32 = 256;

const MMA_TILE: [u32; 2] = [128, 128];

const MMA_STAGES: u32 = 3;

const MMA_BLOCK: u32 = 256;

const MMA_SMEM: u32 = MMA_STAGES * (MMA_TILE[0] + MMA_TILE[1]) * (32 + 8) * 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimePad {
    Zero,
    Replicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3d {
    pub k: [u32; 3],
    pub stride: [u32; 3],
    pub pad: [u32; 3],
    pub pad_back: [u32; 3],
    pub causal_t: bool,
    pub time_pad: TimePad,
    pub dilation: [u32; 3],
    pub depthwise: bool,
}

impl Conv3d {
    #[must_use]
    pub const fn conv2d(k: [u32; 2], stride: [u32; 2], pad: [u32; 2]) -> Self {
        Self {
            k: [1, k[0], k[1]],
            stride: [1, stride[0], stride[1]],
            pad: [0, pad[0], pad[1]],
            pad_back: [0, pad[0], pad[1]],
            causal_t: false,
            time_pad: TimePad::Zero,
            dilation: [1, 1, 1],
            depthwise: false,
        }
    }

    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }

    #[must_use]
    pub fn out_extent(&self, [t, h, w]: [u32; 3]) -> Option<[u32; 3]> {
        let axis = |n: u32, k: u32, s: u32, front: u32, back: u32| {
            (n + front + back)
                .checked_sub(k)
                .map(|span| span / s.max(1) + 1)
        };
        let back_t = if self.causal_t { 0 } else { self.pad_back[0] };
        // The taps spread `dilation` apart span `(k - 1) · dilation + 1`.
        let spread = |a: usize| (self.k[a] - 1) * self.dilation[a] + 1;
        Some([
            axis(t, spread(0), self.stride[0], self.pad[0], back_t)?,
            axis(h, spread(1), self.stride[1], self.pad[1], self.pad_back[1])?,
            axis(w, spread(2), self.stride[2], self.pad[2], self.pad_back[2])?,
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvPath {
    Auto,
    Direct,
    TensorCore,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Geom {
    c_in: i32,
    c_out: i32,
    kt: i32,
    kh: i32,
    kw: i32,
    st: i32,
    sh: i32,
    sw: i32,
    pt: i32,
    ph: i32,
    pw: i32,
    causal: i32,
    replicate: i32,
    lanes: i32,
    rows_out: i32,
    x_base: i32,
    o_base: i32,
    dt: i32,
    dh: i32,
    dw: i32,
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    cache: Option<Tensor>,
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    conv3d_on(
        ctx,
        ConvPath::Auto,
        x,
        grid,
        w,
        bias,
        conv,
        cache,
        o,
        o_grid,
        bases,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_on(
    ctx: &Ctx,
    path: ConvPath,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    cache: Option<Tensor>,
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert_eq!(w.dtype, Dtype::Bf16, "`{OP}` reads a bf16 weight");
    debug_assert_eq!(o.dtype, Dtype::Bf16, "`{OP}` lands bf16");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let c_in = count(OP, "the input channels", x.width)?;
    let c_out = count(OP, "the output channels", o.width)?;
    let rows_out = count(OP, "the output rows", o.rows)?;
    for (axis, v) in ["kt", "kh", "kw"].into_iter().zip(conv.k) {
        count(OP, axis, v)?;
    }
    for (axis, v) in ["st", "sh", "sw"].into_iter().zip(conv.stride) {
        count(OP, axis, v)?;
    }
    let taps = conv.taps();
    if conv.depthwise && x.width != o.width {
        return Err(refuse(
            OP,
            format!(
                "a depthwise convolution keeps its channels, and this one maps {} into {}",
                x.width, o.width
            ),
        ));
    }
    let per = if conv.depthwise { 1 } else { x.width };
    if w.rows != o.width || w.width != taps.saturating_mul(per) {
        return Err(refuse(
            OP,
            format!(
                "the weight is {}x{}; this convolution reads [{}, {}] = [C_out, kt*kh*kw*C_in]",
                w.rows,
                w.width,
                o.width,
                taps.saturating_mul(per)
            ),
        ));
    }
    if let Some(b) = bias
        && (b.dtype != Dtype::F32 || b.elements() != u64::from(o.width))
    {
        return Err(refuse(
            OP,
            format!(
                "the bias is {}x{} {:?}; expected {} f32",
                b.rows, b.width, b.dtype, o.width
            ),
        ));
    }
    if let Some(c) = cache {
        if !conv.causal_t || conv.pad[0] == 0 {
            return Err(refuse(
                OP,
                "a frame cache is read only under `causal_t` with a front pad",
            ));
        }
        if c.dtype != Dtype::Bf16 || c.width != x.width {
            return Err(refuse(
                OP,
                format!(
                    "the cache is {}x{} {:?}; it holds `[frames, C_in]` bf16",
                    c.rows, c.width, c.dtype
                ),
            ));
        }
    }

    let geom = Geom {
        c_in,
        c_out,
        kt: stated(OP, conv.k[0])?,
        kh: stated(OP, conv.k[1])?,
        kw: stated(OP, conv.k[2])?,
        st: stated(OP, conv.stride[0])?,
        sh: stated(OP, conv.stride[1])?,
        sw: stated(OP, conv.stride[2])?,
        pt: stated(OP, conv.pad[0])?,
        ph: stated(OP, conv.pad[1])?,
        pw: stated(OP, conv.pad[2])?,
        causal: i32::from(conv.causal_t),
        replicate: i32::from(conv.time_pad == TimePad::Replicate),
        lanes,
        rows_out,
        x_base: stated(OP, bases[0])?,
        o_base: stated(OP, bases[1])?,
        dt: stated(OP, conv.dilation[0])?,
        dh: stated(OP, conv.dilation[1])?,
        dw: stated(OP, conv.dilation[2])?,
    };
    let args = |o: &Tensor| {
        [
            x.arg(),
            grid.arg(),
            w.arg(),
            bias.map_or(ArgValue::ABSENT, |b| b.arg()),
            cache.map_or(ArgValue::ABSENT, |c| c.arg()),
            o.arg(),
            o_grid.arg(),
            ArgValue::of(&geom),
        ]
    };
    if conv.depthwise {
        let elements = u64::from(o.rows) * u64::from(o.width);
        let lanes = u32::try_from(elements).map_err(|_| {
            refuse(
                OP,
                format!("{elements} outputs do not fit a 32-bit launch extent"),
            )
        })?;
        return ctx.fire(
            OP,
            Fire::at(kernel(FILE, "::pie::conv::conv3d_depthwise"))
                .apply(Launch::flat(lanes, DIRECT_BLOCK)),
            &args(o),
        );
    }

    let vectorised = x.width.is_multiple_of(8)
        && aligned16(x.ptr)
        && aligned16(w.ptr)
        && cache.is_none_or(|c| aligned16(c.ptr))
        && (!o.width.is_multiple_of(8) || aligned16(o.ptr));
    // Every GPU pie serves is sm_80 or later, which has the tensor cores.
    let tensor_core = vectorised;
    let mma = match path {
        ConvPath::Auto => tensor_core,
        ConvPath::Direct => false,
        ConvPath::TensorCore => {
            if !tensor_core {
                return Err(refuse(
                    OP,
                    format!(
                        "the tensor-core kernel needs C_in % 8 == 0 ({}) and 16-byte planes",
                        x.width
                    ),
                ));
            }
            true
        }
    };

    let (entry, tile, block, smem) = if mma {
        (
            format!("::pie::conv::conv3d_mma<{MMA_STAGES}>"),
            MMA_TILE,
            MMA_BLOCK,
            MMA_SMEM,
        )
    } else {
        (
            format!("::pie::conv::conv3d_direct<{vectorised}>"),
            DIRECT_TILE,
            DIRECT_BLOCK,
            0,
        )
    };
    let launch = Launch::grid(
        [o.rows.div_ceil(tile[0]), o.width.div_ceil(tile[1]), 1],
        [block, 1, 1],
    )
    .smem(smem);
    ctx.fire(OP, Fire::at(kernel(FILE, &entry)).apply(launch), &args(o))
}

pub fn conv_weight_taps_major(
    ctx: &Ctx,
    src: Tensor,
    c_in: u32,
    taps: u32,
    dst: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.conv_weight_taps_major";
    const BLOCK: u32 = 256;
    dtype_dispatch!(OP, src.dtype, { Bf16 => () });
    debug_assert_eq!(dst.dtype, Dtype::Bf16, "`{OP}` lands bf16");
    let c_in = count(OP, "the input channels", c_in)?;
    let taps = count(OP, "the tap count", taps)?;
    let c_out = count(OP, "the output channels", src.rows)?;
    let k = u64::from(src.width);
    if k != u64::from(c_in.unsigned_abs()) * u64::from(taps.unsigned_abs())
        || dst.rows != src.rows
        || dst.width != src.width
    {
        return Err(refuse(
            OP,
            format!(
                "{}x{} into {}x{} is not a relabelling of [C_out, {c_in}*{taps}]",
                src.rows, src.width, dst.rows, dst.width
            ),
        ));
    }
    let (blocks, _) = crate::voxels::flat_elements(OP, src, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::conv::conv_weight_taps_major"))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[src.arg(), dst.arg(), c_out.arg(), c_in.arg(), taps.arg()],
    )
}
