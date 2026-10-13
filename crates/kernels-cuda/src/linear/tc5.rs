//! Native datacenter-Blackwell GEMMs: one persistent tcgen05 kernel
//! (`linear/tc5.cuh`) specialized per kind and tile at JIT time, fed by
//! TMA tensor maps encoded here. `y[m][n] = alpha * sum_k act[m][k] * w[n][k]`
//! with both operands K-major and `y` bf16, `m x n` with row stride `n`.
//!
//! Block scales are 512-byte atoms of 128 rows x 4 scale columns (see
//! [`atom_offset`]), atoms of one 128-row block consecutive along K, rows
//! padded to whole blocks with finite (zero) bytes. A scale column covers
//! `gran` K: 16 for NVFP4's ue4m3, 32 or 128 for ue8m0.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys as dr;

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, refuse, symbol};

const OP: &str = "linear.tc5";
const FILE: &str = "linear/tc5.cuh";
const THREADS: u32 = 224;
const SMEM_CAP: u32 = 232_448;

/// An operand's element type as it lies in global memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    Bf16,
    /// fp8 e4m3, one a byte.
    E4m3,
    /// fp4 e2m1, two a byte, the even element in the low nibble.
    E2m1,
}

/// A block-scale tensor in the atom layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scales {
    pub ptr: u64,
    /// The K a scale covers: 16 (ue4m3), 32 or 128 (ue8m0).
    pub gran: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    pub ptr: u64,
    pub elem: Elem,
    pub scales: Option<Scales>,
}

impl Operand {
    #[must_use]
    pub const fn plain(ptr: u64, elem: Elem) -> Self {
        Self {
            ptr,
            elem,
            scales: None,
        }
    }

    #[must_use]
    pub const fn scaled(ptr: u64, elem: Elem, scales: u64, gran: u32) -> Self {
        Self {
            ptr,
            elem,
            scales: Some(Scales { ptr: scales, gran }),
        }
    }
}

/// The byte of scale `(row, col)` in an atom-layout tensor whose rows carry
/// `cols` scale columns.
#[must_use]
pub const fn atom_offset(row: usize, col: usize, cols: usize) -> usize {
    let atoms_k = cols.div_ceil(4);
    ((row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4
}

/// The bytes an atom-layout scale tensor for `rows x k` at `gran` takes.
#[must_use]
pub const fn atom_bytes(rows: usize, k: usize, gran: usize) -> usize {
    rows.div_ceil(128) * k.div_ceil(gran).div_ceil(4) * 512
}

/// Whether this device and NVRTC run the tcgen05 kernels.
#[must_use]
pub fn supported() -> bool {
    crate::jit::cache::features().tcgen05
}

/// bf16 x bf16 -> bf16, f32 accumulation.
pub fn gemm_bf16(ctx: &Ctx, act: u64, w: u64, y: u64, m: u32, n: u32, k: u32) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::plain(act, Elem::Bf16),
        Operand::plain(w, Elem::Bf16),
        y,
        [m, n, k],
        1.0,
    )
}

/// fp8 e4m3 W8A8 with DeepSeek's blocks: one ue8m0 scale per 1x128 of the
/// activation and per 128x128 of the weight, the weight's spread to every
/// row of its block (gran 128 on both).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block(
    ctx: &Ctx,
    act: u64,
    act_scales: u64,
    w: u64,
    w_scales: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::scaled(act, Elem::E4m3, act_scales, 128),
        Operand::scaled(w, Elem::E4m3, w_scales, 128),
        y,
        [m, n, k],
        1.0,
    )
}

/// NVFP4 W4A4: e2m1 with a ue4m3 scale a 16, `alpha` the product of the
/// two global scales.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4(
    ctx: &Ctx,
    act: u64,
    act_scales: u64,
    w: u64,
    w_scales: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::scaled(act, Elem::E2m1, act_scales, 16),
        Operand::scaled(w, Elem::E2m1, w_scales, 16),
        y,
        [m, n, k],
        alpha,
    )
}

/// W4A8: MXFP4 weights (e2m1, ue8m0 a 32) against e4m3 activations with a
/// ue8m0 scale a `act_gran` (32 or 128).
#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a8(
    ctx: &Ctx,
    act: u64,
    act_scales: u64,
    act_gran: u32,
    w: u64,
    w_scales: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::scaled(act, Elem::E4m3, act_scales, act_gran),
        Operand::scaled(w, Elem::E2m1, w_scales, 32),
        y,
        [m, n, k],
        1.0,
    )
}

/// The kernel's operand kind, and each side's format code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Bf16,
    Fp8,
    Mx,
    Nvf4,
}

impl Kind {
    const fn code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::Fp8 => 1,
            Self::Mx => 2,
            Self::Nvf4 => 3,
        }
    }

    const fn stage_k(self) -> u32 {
        match self {
            Self::Bf16 => 64,
            Self::Nvf4 => 256,
            Self::Fp8 | Self::Mx => 128,
        }
    }

    fn of(act: Operand, w: Operand) -> Result<Self, String> {
        let gran = |o: Operand| o.scales.map(|s| s.gran);
        match (act.elem, w.elem, gran(act), gran(w)) {
            (Elem::Bf16, Elem::Bf16, None, None) => Ok(Self::Bf16),
            (Elem::E4m3, Elem::E4m3, None, None) => Ok(Self::Fp8),
            (Elem::E2m1, Elem::E2m1, Some(16), Some(16)) => Ok(Self::Nvf4),
            (Elem::E4m3 | Elem::E2m1, Elem::E4m3 | Elem::E2m1, Some(a), Some(b))
                if matches!(a, 32 | 128) && matches!(b, 32 | 128) =>
            {
                Ok(Self::Mx)
            }
            other => Err(format!(
                "no tcgen05 kind multiplies {other:?} (act elem, w elem, act gran, w gran): \
                 bf16 x bf16, e4m3 x e4m3, nvfp4 (gran 16) or e4m3/e2m1 with ue8m0 at 32/128"
            )),
        }
    }

    /// The instruction descriptor's format code for one side.
    const fn format(self, elem: Elem) -> u32 {
        match (self, elem) {
            (Self::Bf16, _) | (Self::Nvf4, _) => 1,
            (_, Elem::E2m1) => 5,
            _ => 0,
        }
    }
}

const fn atoms_k(kind: Kind, gran: u32) -> u32 {
    match kind {
        Kind::Bf16 | Kind::Fp8 => 0,
        Kind::Mx | Kind::Nvf4 => {
            let atoms = kind.stage_k() / gran / 4;
            if atoms == 0 { 1 } else { atoms }
        }
    }
}

/// The dynamic shared memory the kernel lays out, mirrored from it, plus
/// the 1024 bytes it may spend aligning the ring.
const fn smem_bytes(kind: Kind, gran_p: u32, gran_q: u32, plan: Plan) -> u32 {
    let Plan {
        swap,
        pair,
        bn,
        stages,
    } = plan;
    let stage = 128 * 128
        + bn / if pair { 2 } else { 1 } * 128
        + atoms_k(kind, gran_p) * 512
        + atoms_k(kind, gran_q) * bn.div_ceil(128) * 512;
    let out = (stages * stage).div_ceil(1024) * 1024;
    let bar = out + if swap { 0 } else { 4 * 2 * 2048 };
    bar + (2 * stages + 12) * 8 + 8 + 1024
}

/// One shape's specialization: which side is the MMA's M, whether two CTAs
/// share a 256-row MMA, the Q tile, and the ring depth that fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub swap: bool,
    /// A cluster of two CTAs runs each 256-row tile (`cta_group::2`).
    pub pair: bool,
    pub bn: u32,
    pub stages: u32,
}

impl Plan {
    /// Rows of 64 or fewer put the weight on the 128-row side, the rows on
    /// a narrow Q tile. Otherwise a pair of CTAs runs 256 x 256 tiles, each
    /// streaming half of both operands' bytes a lone CTA would, unless
    /// lone 128 x 128 tiles fill the `sms` enough better to outweigh it
    /// (they stream at about half the rate).
    #[must_use]
    pub fn choose(m: u32, n: u32, sms: u32) -> Self {
        if m <= 64 {
            return Self {
                swap: true,
                pair: false,
                bn: m.next_multiple_of(32).max(32),
                stages: 0,
            };
        }
        let waves = |rows: u32, bn: u32, per: u32| {
            (m.div_ceil(rows) * n.div_ceil(bn)).div_ceil((sms / per).max(1))
        };
        let pair = 10 * waves(256, 256, 2) <= 9 * waves(128, 128, 1);
        Self {
            swap: false,
            pair,
            bn: if pair { 256 } else { 128 },
            stages: 0,
        }
    }

    fn staged(mut self, kind: Kind, gran_p: u32, gran_q: u32) -> Self {
        let mut stages = 8;
        while stages > 2 && smem_bytes(kind, gran_p, gran_q, Self { stages, ..self }) > SMEM_CAP {
            stages -= 1;
        }
        self.stages = stages;
        self
    }
}

/// `y = alpha * act . w^T` for `[m, n, k]`, choosing the plan.
pub fn gemm(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    [m, n, k]: [u32; 3],
    alpha: f32,
) -> Result<(), Error> {
    let sms = ctx.multiprocessors().unwrap_or(148);
    gemm_with(ctx, act, w, y, [m, n, k], alpha, Plan::choose(m, n, sms))
}

/// As [`gemm`], under a stated plan (its `stages` is refit to the kind).
#[allow(clippy::too_many_arguments)]
pub fn gemm_with(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    [m, n, k]: [u32; 3],
    alpha: f32,
    plan: Plan,
) -> Result<(), Error> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        return Err(refuse(OP, "an empty contraction"));
    }
    if !supported() {
        return Err(refuse(
            OP,
            "this device has no tcgen05 (sm_100a/sm_103a), or NVRTC is older than 12.9",
        ));
    }
    let kind = Kind::of(act, w).map_err(|why| refuse(OP, why))?;
    if !n.is_multiple_of(8) {
        return Err(refuse(OP, format!("n = {n} is not a whole number of 8s")));
    }
    // Whole 32-row quarters: a scaled Q tile narrower than its scale atom
    // starts its scale columns at its own quarter.
    if !matches!(plan.bn, 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256) {
        return Err(refuse(OP, format!("a Q tile of {} rows", plan.bn)));
    }
    if plan.pair && (plan.swap || plan.bn < 64) {
        return Err(refuse(
            OP,
            "a CTA pair splits an unswapped Q tile of 64 or more",
        ));
    }
    let (p, q) = if plan.swap { (w, act) } else { (act, w) };
    let (rows_p, rows_q) = if plan.swap { (n, m) } else { (m, n) };
    let gran = |o: Operand| o.scales.map_or(128, |s| s.gran);
    let plan = plan.staged(kind, gran(p), gran(q));
    let smem = smem_bytes(kind, gran(p), gran(q), plan);
    let ctas = if plan.pair { 2 } else { 1 };

    let map_p = tensor_map(kind, p, rows_p, k, 128)?;
    let map_q = tensor_map(kind, q, rows_q, k, plan.bn / ctas)?;
    let map_y = if plan.swap {
        [0; 16]
    } else {
        output_map(y, m, n)?
    };
    let atoms = |o: Operand, rows: u32, row_blocks: u32| match o.scales {
        Some(s) => scale_map(s, rows, k, atoms_k(kind, s.gran), row_blocks),
        None => Ok([0; 16]),
    };
    let map_sfp = atoms(p, rows_p, 1)?;
    let map_sfq = atoms(q, rows_q, plan.bn.div_ceil(128))?;
    let tiles = rows_p.div_ceil(128 * ctas) * rows_q.div_ceil(plan.bn);
    let grid = ctas * tiles.min(ctx.multiprocessors().unwrap_or(148) / ctas);
    let entry = symbol(&format!(
        "::pie::linear::tc5_gemm<{}, {}, {}, {}, {}, {}, {}, {}, {ctas}>",
        kind.code(),
        kind.format(p.elem),
        kind.format(q.elem),
        gran(p),
        gran(q),
        plan.bn,
        plan.stages,
        u32::from(plan.swap),
    ));
    let int = |v: u32| i32::try_from(v).map_err(|_| refuse(OP, format!("{v} overflows an int")));
    let map = |m: &[u64; 16]| ArgValue::Bytes {
        ptr: m.as_ptr().cast(),
        len: 128,
    };
    let values = [
        map(&map_p),
        map(&map_q),
        map(&map_y),
        map(&map_sfp),
        map(&map_sfq),
        ArgValue::Ptr(y),
        ArgValue::I32(int(rows_p)?),
        ArgValue::I32(int(rows_q)?),
        ArgValue::I32(int(k)?),
        ArgValue::I32(int(n)?),
        ArgValue::F32(alpha),
    ];
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(
            Launch::grid([grid, 1, 1], [THREADS, 1, 1])
                .smem(smem)
                .cluster(ctas),
        ),
        &values,
    )
}

/// An atom-layout scale tensor as TMA reads it: [row block][atom][128 words],
/// boxes of `atoms` atoms by `row_blocks` row blocks, zero past either end.
fn scale_map(
    s: Scales,
    rows: u32,
    k: u32,
    atoms: u32,
    row_blocks: u32,
) -> Result<[u64; 16], Error> {
    if !s.ptr.is_multiple_of(16) {
        return Err(refuse(OP, "TMA needs the scales 16-byte aligned"));
    }
    let along = k.div_ceil(s.gran).div_ceil(4);
    encode(Map {
        ptr: s.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        dims: [128, along, rows.div_ceil(128)],
        strides: [512, u64::from(along) * 512],
        boxes: [128, atoms, row_blocks],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    })
}

/// The tensor map of a K-major `rows x k` operand, boxes of 128 bytes of K
/// by `box_rows`, swizzled 128B.
fn tensor_map(
    kind: Kind,
    o: Operand,
    rows: u32,
    k: u32,
    box_rows: u32,
) -> Result<[u64; 16], Error> {
    use dr::CUtensorMapDataType as T;

    // (type, inner extent in the type's elements, row stride in bytes, box width)
    let (ty, inner, stride, box_k) = match (kind, o.elem) {
        (_, Elem::Bf16) => (T::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, k, u64::from(k) * 2, 64),
        (_, Elem::E4m3) => (T::CU_TENSOR_MAP_DATA_TYPE_UINT8, k, u64::from(k), 128),
        (Kind::Nvf4, Elem::E2m1) => (
            T::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            k / 2,
            u64::from(k) / 2,
            128,
        ),
        (_, Elem::E2m1) => (
            T::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            k,
            u64::from(k) / 2,
            128,
        ),
    };
    // Packed fp4 rows are 16-byte strides; unpacked (16U4_ALIGN16B) ones
    // must be whole 64-byte runs.
    let whole = if ty == T::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B {
        128
    } else {
        32
    };
    if o.elem == Elem::E2m1 && !k.is_multiple_of(whole) {
        return Err(refuse(
            OP,
            format!("an fp4 k of {k} is not a whole number of {whole}s for this kind"),
        ));
    }
    if !stride.is_multiple_of(16) || !o.ptr.is_multiple_of(16) {
        return Err(refuse(
            OP,
            format!("TMA needs 16-byte rows and base; this operand's row is {stride} bytes"),
        ));
    }
    encode(Map {
        ptr: o.ptr,
        ty,
        dims: [inner, rows, 0],
        strides: [stride, 0],
        boxes: [box_k, box_rows, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })
}

/// The bf16 `m x n` output, in the 32 x 32 boxes an epilogue warp stores
/// from its 64-byte-swizzled stage.
fn output_map(y: u64, m: u32, n: u32) -> Result<[u64; 16], Error> {
    if !y.is_multiple_of(16) {
        return Err(refuse(OP, "TMA needs the output 16-byte aligned"));
    }
    encode(Map {
        ptr: y,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [n, m, 0],
        strides: [u64::from(n) * 2, 0],
        boxes: [32, 32, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Map {
    ptr: u64,
    ty: dr::CUtensorMapDataType,
    /// A zero third extent makes the map 2D.
    dims: [u32; 3],
    strides: [u64; 2],
    boxes: [u32; 3],
    swizzle: dr::CUtensorMapSwizzle,
}

/// Encoding is host-only and cheap; the cache spares the driver call on a
/// replayed shape.
fn encode(m: Map) -> Result<[u64; 16], Error> {
    static MAPS: OnceLock<Mutex<HashMap<Map, [u64; 16]>>> = OnceLock::new();
    let maps = MAPS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = maps
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&m)
    {
        return Ok(*found);
    }
    let dims = m.dims.map(u64::from);
    let steps = [1u32, 1, 1];
    let rank = if m.dims[2] == 0 { 2 } else { 3 };
    let mut map: dr::CUtensorMap = unsafe { core::mem::zeroed() };
    // SAFETY: every pointer is to a live local of the length the rank says.
    let code = unsafe {
        dr::cuTensorMapEncodeTiled(
            &raw mut map,
            m.ty,
            rank,
            m.ptr as *mut core::ffi::c_void,
            dims.as_ptr(),
            m.strides.as_ptr(),
            m.boxes.as_ptr(),
            steps.as_ptr(),
            dr::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            m.swizzle,
            dr::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            dr::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(refuse(
            OP,
            format!(
                "cuTensorMapEncodeTiled answered {code:?} for a {:?} {} x {} at {:#x}",
                m.ty, m.dims[1], m.dims[0], m.ptr
            ),
        ));
    }
    // SAFETY: CUtensorMap is 128 opaque bytes, 64-aligned.
    let raw: [u64; 16] = unsafe { core::mem::transmute(map) };
    maps.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(m, raw);
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_atom_is_the_layout_tcgen05_cp_lands() {
        assert_eq!(atom_offset(0, 0, 4), 0);
        assert_eq!(atom_offset(0, 3, 4), 3);
        assert_eq!(atom_offset(32, 0, 4), 4);
        assert_eq!(atom_offset(1, 0, 4), 16);
        assert_eq!(atom_offset(127, 3, 4), 511);
        assert_eq!(
            atom_offset(0, 4, 8),
            512,
            "atoms of a row block run along K"
        );
        assert_eq!(atom_offset(128, 0, 8), 1024);
        assert_eq!(atom_bytes(129, 4096, 16), 2 * 64 * 512);
    }

    #[test]
    fn every_kind_fits_its_ring() {
        for (kind, g) in [
            (Kind::Bf16, 128),
            (Kind::Fp8, 128),
            (Kind::Mx, 32),
            (Kind::Nvf4, 16),
        ] {
            for bn in [32, 64, 128, 256] {
                let plan = Plan {
                    swap: false,
                    pair: bn >= 64,
                    bn,
                    stages: 0,
                }
                .staged(kind, g, g);
                assert!(plan.stages >= 3, "{kind:?} bn={bn}: {plan:?}");
                assert!(smem_bytes(kind, g, g, plan) <= SMEM_CAP);
            }
        }
    }
}
