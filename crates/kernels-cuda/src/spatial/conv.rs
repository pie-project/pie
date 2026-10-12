use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, refuse, stated,
};
use crate::spatial::lane_pair;
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "spatial/conv.cuh";

const OP: &str = "spatial.conv3d";

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
        Some([
            axis(t, self.k[0], self.stride[0], self.pad[0], back_t)?,
            axis(h, self.k[1], self.stride[1], self.pad[1], self.pad_back[1])?,
            axis(w, self.k[2], self.stride[2], self.pad[2], self.pad_back[2])?,
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
    if w.rows != o.width || w.width != taps.saturating_mul(x.width) {
        return Err(refuse(
            OP,
            format!(
                "the weight is {}x{}; this convolution reads [{}, {}] = [C_out, kt*kh*kw*C_in]",
                w.rows,
                w.width,
                o.width,
                taps.saturating_mul(x.width)
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
    };

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
            crate::jit::symbol(&format!("::pie::spatial::conv3d_mma<{MMA_STAGES}>")),
            MMA_TILE,
            MMA_BLOCK,
            MMA_SMEM,
        )
    } else {
        (
            if vectorised {
                "::pie::spatial::conv3d_direct<true>"
            } else {
                "::pie::spatial::conv3d_direct<false>"
            },
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
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(launch),
        &[
            x.arg(),
            grid.arg(),
            w.arg(),
            bias.map_or(ArgValue::ABSENT, |b| b.arg()),
            cache.map_or(ArgValue::ABSENT, |c| c.arg()),
            o.arg(),
            o_grid.arg(),
            ArgValue::Bytes {
                ptr: std::ptr::from_ref(&geom).cast(),
                len: size_of::<Geom>(),
            },
        ],
    )
}

pub fn conv_weight_taps_major(
    ctx: &Ctx,
    src: Tensor,
    c_in: u32,
    taps: u32,
    dst: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.conv_weight_taps_major";
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
    let (blocks, _) = crate::spatial::flat_elements(OP, src, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::spatial::conv_weight_taps_major")
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[src.arg(), dst.arg(), c_out.arg(), c_in.arg(), taps.arg()],
    )
}
