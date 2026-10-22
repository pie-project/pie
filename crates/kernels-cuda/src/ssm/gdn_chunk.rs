use super::*;

const GDN_CHUNK: &str = "ssm/gdn_chunk.cu";

const GDN_CHUNK_SM80: &str = "ssm/gdn_chunk_sm80.cu";

/// The value columns one block of the chunked scan carries: half a head,
/// so a lone long request still spreads over twice its heads' blocks. A
/// captured fire states its bucket's lanes, not the live ones, so the width
/// cannot follow the batch.
const SCAN_BV: u32 = 64;

/// A quarter head a block, where a block cannot hold half a head's scan
/// (99 KiB: an RTX 30/40/50, an A40, an L4, a DGX Spark).
const SCAN_BV_NARROW: u32 = 32;

/// The shared memory one block of the chunked scan holds at key width
/// `k_dim` and `bv` value columns: its prep is smaller.
const fn scan_smem_at(k_dim: u32, bv: u32) -> u32 {
    const CH: u32 = 64;
    const LD: u32 = 136;
    const LDC: u32 = 72;
    let ldv = bv + 8;
    (3 * CH * LD + CH * ldv + CH * LDC + k_dim * ldv + 2 * CH * ldv) * 2 + 2 * CH * FLOAT
}

/// The value columns a 128-wide prefill's two-launch scan takes a block,
/// where the profile turns the scan on and this device's blocks can opt
/// into its shared memory at half a head, else a quarter.
fn fused_scan(ctx: &Ctx, k_dim: u32) -> Option<u32> {
    if !ctx.tuning().ssm_gdn_fused_scan {
        return None;
    }
    let cap = ctx.smem_per_block_optin()?;
    [SCAN_BV, SCAN_BV_NARROW]
        .into_iter()
        .find(|&bv| cap >= scan_smem_at(k_dim, bv))
}

/// Whether a prefill of the gated delta rule at key width `k_dim` lands its
/// state snapshots on this device: the two-launch scan and the serial walk
/// seat them, the wmma launches and the tcgen05 walk do not.
#[must_use]
pub fn gdn_lands_snapshots(ctx: &Ctx, k_dim: u32) -> bool {
    k_dim == 128 && fused_scan(ctx, k_dim).is_some()
}

/// `pie::ssm::sm100::Params`, field for field.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Tc5Params {
    gates: u64,
    state: u64,
    slot_ids: u64,
    qo_indptr: u64,
    commit_len: u64,
    begin_at: u64,
    write_state_mask: u64,
    out: u64,
    win: u64,
    slot_stride: i64,
    lanes: i32,
    k_heads: i32,
    v_heads: i32,
    write_state: i32,
    q_scale: f32,
    out_bf16: i32,
}

/// The one-launch tcgen05 walk (`attn/gdn_tc5.cuh`), where the profile
/// asks for it and the device and operands admit it: 128-wide heads, rows
/// TMA can read. `Ok(false)` leaves the prefill to the scans.
#[cfg(feature = "cuda")]
fn gated_delta_chunked_tc5(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<bool, Error> {
    use crate::linear::tc5::{Map, encode, supported};
    use cudarc::driver::sys as dr;
    const GDN_CHUNK_SM100: &str = "ssm/gdn_chunk_sm100.cu";
    /// `pie::ssm::sm100::smem_bytes()`.
    const SMEM: u32 = 213_504 + 1024;
    if !(ctx.tuning().ssm_gdn_tc5
        && supported()
        && shape.k_dim == 128
        && shape.v_dim == 128
        && shape.v_heads.is_multiple_of(shape.k_heads.max(1))
        && (shape.conv_dim * 2).is_multiple_of(16)
        && aligned16(qkv.data.ptr)
        && aligned16(y.ptr)
        && state.slot_stride_elems > 0
        && state.snapshots.count == 0)
    {
        return Ok(false);
    }
    let lanes = requests(op, qkv)?;
    seated(op, state, "sm100::gdn_chunked", true, true, true)?;
    let map = crate::attn::bf16_map(
        qkv.data.ptr,
        [shape.conv_dim, qkv.data.rows, 1],
        [
            u64::from(shape.conv_dim) * 2,
            u64::from(shape.conv_dim) * 2 * u64::from(qkv.data.rows.max(1)),
        ],
        [64, 64, 1],
    )?;
    // A whole chunk's output, 64 rows of a value head, stored by TMA, in
    // bf16 where the trace narrowed it for its gated norm.
    let out_bf16 = y.dtype == Dtype::Bf16;
    let out_bytes: u64 = if out_bf16 { 2 } else { 4 };
    let out_width = shape.v_heads * shape.v_dim;
    let map_out = encode(Map {
        ptr: y.ptr,
        ty: if out_bf16 {
            dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        } else {
            dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        },
        dims: [out_width, y.rows, 1],
        strides: [
            u64::from(out_width) * out_bytes,
            u64::from(out_width) * out_bytes * u64::from(y.rows.max(1)),
        ],
        boxes: [128, 64, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    })?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (shape.k_dim as f32).sqrt().recip();
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let ArgValue::Ptr(win) = ctx.stage() else {
        return Ok(false);
    };
    let params = Tc5Params {
        gates: gates.ptr,
        state: state.slab.ptr,
        slot_ids: state.slot_ids.ptr,
        qo_indptr: qkv.indptr.ptr,
        commit_len: state.commit_len.ptr,
        begin_at: state.begin_at.ptr,
        write_state_mask: state.write_state_mask.ptr,
        out: y.ptr,
        win,
        slot_stride: state.slot_stride_elems,
        lanes: int(lanes),
        k_heads: int(shape.k_heads),
        v_heads: int(shape.v_heads),
        write_state: i32::from(state.write_state),
        q_scale,
        out_bf16: i32::from(out_bf16),
    };
    let tiles = lanes.saturating_mul(shape.v_heads);
    let grid = tiles.min(ctx.multiprocessors().unwrap_or(148)).max(1);
    ctx.fire(
        op,
        Fire::at(GDN_CHUNK_SM100, symbol("::pie::ssm::sm100::gdn_chunked"))
            .apply(Launch::grid([grid, 1, 1], [384, 1, 1]).smem(SMEM)),
        &[
            ArgValue::of(&map),
            ArgValue::of(&map_out),
            ArgValue::of(&params),
        ],
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_chunked_fla(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &RecurrentPool,
    y: &mut Tensor,
    bv: u32,
) -> Result<(), Error> {
    const CH: u32 = 64;
    const BLOCK: u32 = 128;
    const LD: u32 = 136;
    const LDC: u32 = 72;
    const LDA: u32 = 68;
    let lanes = requests(op, qkv)?;
    seated(op, state, "ssm_gdn_chunk", true, true, true)?;
    let (k_dim, v_dim, v_heads, k_heads) = (shape.k_dim, shape.v_dim, shape.v_heads, shape.k_heads);
    // A snapshot cuts a lane's chunk where it lands: one more chunk slot each.
    let slots = shape.n.div_ceil(CH) + lanes + state.snapshots.count;
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    let padded = Delta {
        n: shape.n + CH,
        ..*shape
    };
    let gcum = plane(ctx, op, "attn.ssm_gdn_chunk_gcum", padded.elems(v_heads, 1))?;
    let qn = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_qn",
        padded.elems(k_heads, k_dim),
    )?;
    let kn = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kn",
        padded.elems(k_heads, k_dim),
    )?;
    let w = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_w",
        padded.elems(v_heads, k_dim),
    )?;
    let u = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_u",
        padded.elems(v_heads, v_dim),
    )?;
    let pm = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_p",
        u64::from(slots) * u64::from(v_heads) * u64::from(CH * CH),
    )?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    let prep_smem =
        2 * CH * LD * 2 + 2 * CH * LDC * 2 + CH * LDA * FLOAT + 3 * CH * FLOAT + 4 * 256 * FLOAT;
    ctx.fire(
        op,
        Fire::at(GDN_CHUNK_SM80, "::pie::ssm::gdn_chunk_prep<128, 128>")
            .apply(Launch::grid([v_heads, slots, 1], [2 * BLOCK, 1, 1]).smem(prep_smem)),
        &[
            qkv.data.arg(),
            gates.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qn),
            ArgValue::Ptr(kn),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(pm),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            stated(op, shape.conv_dim)?.arg(),
            q_scale.arg(),
            snap,
            snap_rows,
            ctx.stage(),
        ],
    )?;
    let scan_smem = scan_smem_at(k_dim, bv);
    let entry = if bv == SCAN_BV {
        "::pie::ssm::gdn_chunk_scan<64>"
    } else {
        "::pie::ssm::gdn_chunk_scan<32>"
    };
    ctx.fire(
        op,
        Fire::at(GDN_CHUNK_SM80, entry).apply(
            Launch::grid([lanes * v_heads * (v_dim / bv), 1, 1], [2 * BLOCK, 1, 1]).smem(scan_smem),
        ),
        &[
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qn),
            ArgValue::Ptr(kn),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(pm),
            state.slab.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
        ],
    )
}

/// The chunked rule in three wmma launches over the staged planes: the
/// update directions per chunk, the state walk per sixteen value columns,
/// and the output per chunk. No block holds more than 41 KiB of shared
/// memory, so every card runs it.
#[allow(clippy::too_many_arguments)]
fn gated_delta_chunked_split(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    staged: &DeltaStaged,
    qkv: RaggedTensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const CH: u32 = 64;
    const BVT: u32 = 16;
    const BLOCK: u32 = 256;
    let lanes = requests(op, qkv)?;
    seated(op, state, "ssm_gdn_chunk", true, true, true)?;
    if state.snapshots.count > 0 {
        return Err(refuse(
            op,
            "this fire lands state snapshots and the three wmma launches have no seat for them",
        ));
    }
    let (k_dim, v_dim, v_heads, k_heads) = (shape.k_dim, shape.v_dim, shape.v_heads, shape.k_heads);
    let slots = shape.n.div_ceil(CH) + lanes;
    let padded = Delta {
        n: shape.n + CH,
        ..*shape
    };
    let per_slot = |heads: u32, width: u32| u64::from(slots) * u64::from(heads) * u64::from(width);
    let gcum = plane(ctx, op, "attn.ssm_gdn_chunk_gcum", shape.elems(v_heads, 1))?;
    let qb = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_qb",
        padded.elems(k_heads, k_dim),
    )?;
    let kb = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kb",
        padded.elems(k_heads, k_dim),
    )?;
    let kt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kt",
        per_slot(k_heads, k_dim * CH),
    )?;
    let w = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_w",
        padded.elems(v_heads, k_dim),
    )?;
    let u = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_u",
        padded.elems(v_heads, v_dim),
    )?;
    let vnt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_vnt",
        per_slot(v_heads, v_dim * CH),
    )?;
    let hct = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_hct",
        per_slot(v_heads, v_dim * k_dim),
    )?;
    let wu_smem = CH * CH * FLOAT + CH * k_dim * 2 + CH * CH * 2 + 4 * CH * FLOAT;
    ctx.fire(
        op,
        Fire::at(GDN_CHUNK_SM80, "::pie::ssm::gdn_chunk_wu<128, 128>")
            .apply(Launch::grid([slots, v_heads, 1], [BLOCK, 1, 1]).smem(wu_smem)),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qb),
            ArgValue::Ptr(kb),
            ArgValue::Ptr(kt),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(
            GDN_CHUNK_SM80,
            "::pie::ssm::gdn_chunk_h<::pie::ssm::state_bf16, 128, 128, 16>",
        )
        .apply(Launch::grid(
            [lanes * v_heads * (v_dim / BVT), 1, 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(kt),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(vnt),
            ArgValue::Ptr(hct),
            state.slab.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(GDN_CHUNK_SM80, "::pie::ssm::gdn_chunk_o<128, 128, 16>")
            .apply(Launch::grid([slots, v_heads, 1], [BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qb),
            ArgValue::Ptr(kb),
            ArgValue::Ptr(vnt),
            ArgValue::Ptr(hct),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked_serial(
    ctx: &Ctx,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    SERIAL_SCAN.with(|flag| flag.set(true));
    let out = gated_delta_chunked(ctx, qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, y);
    SERIAL_SCAN.with(|flag| flag.set(false));
    out
}

thread_local! {
    static SERIAL_SCAN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    ctx: &Ctx,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";

    const BK_MAX_FLA: u32 = 128;

    const BV_FLA: u32 = 128;

    const WARP_TILED_K_MAX: u32 = 256;

    const SCAN_WARPS: u32 = 4;

    let _ = z;
    dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    // A short fire (a verify, a fold) scans its rows in order instead.
    let scan = y.rows <= ctx.tuning().ssm_gdn_scan_until;
    if y.dtype == Dtype::Bf16 {
        // Narrowed for its gated norm: the tcgen05 walk lands bf16 itself;
        // any other walk lands f32 in a plane, cast after.
        let wide_y = Tensor::new(y.ptr, y.rows, y.width, Dtype::F32);
        let shape = Delta::of(OP, qkv.data, gates, &wide_y, k_heads, v_heads, k_dim, v_dim)?;
        #[cfg(feature = "cuda")]
        if k_dim == 128
            && v_dim == 128
            && !SERIAL_SCAN.with(core::cell::Cell::get)
            && !scan
            && gated_delta_chunked_tc5(ctx, OP, &shape, qkv, gates, state, y)?
        {
            return Ok(());
        }
        let _ = shape;
        // A staged window lands its rows past its base, within twice the
        // fire's rows.
        let plane_at = plane(ctx, OP, "attn.ssm_gdn_wide_out", 2 * y.elements())?;
        let mut wide = Tensor::new(plane_at, y.rows, y.width, Dtype::F32);
        gated_delta_chunked(
            ctx, qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, &mut wide,
        )?;
        let width = stated(OP, y.width)?;
        let rows = nonzero(OP, "rows", y.rows)?;
        return ctx.fire(
            OP,
            Fire::at(GDN_CHUNK_SM80, "::pie::ssm::gdn_narrow_rows").apply(Launch::grid(
                [rows, y.width.div_ceil(4 * 128), 1],
                [128, 1, 1],
            )),
            &[wide.arg(), y.arg(), width.arg(), ctx.stage()],
        );
    }
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    let wide =
        k_dim == BK_MAX_FLA && v_dim == BV_FLA && !SERIAL_SCAN.with(core::cell::Cell::get) && !scan;
    #[cfg(feature = "cuda")]
    if wide && gated_delta_chunked_tc5(ctx, OP, &shape, qkv, gates, state, y)? {
        return Ok(());
    }
    let scan_bv = if wide { fused_scan(ctx, k_dim) } else { None };
    let fused = scan_bv.is_some();
    if let Some(bv) = scan_bv
        && shape.conv_dim.is_multiple_of(8)
        && (k_heads * k_dim).is_multiple_of(8)
        && state.slot_stride_elems % 8 == 0
        && [qkv.data.ptr, y.ptr, state.slab.ptr]
            .iter()
            .all(|&at| aligned16(at))
    {
        return gated_delta_chunked_fla(ctx, OP, &shape, qkv, gates, state, y, bv);
    }
    let staged = shape.stage(ctx, OP, qkv.data, gates, StagePlanes::Chunk)?;
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    if wide && !fused {
        return gated_delta_chunked_split(ctx, OP, &shape, &staged, qkv, state, y);
    }
    if k_dim <= BK_MAX_FLA && v_dim.is_multiple_of(BV_FLA) {
        seated(OP, state, "gdn_chunked_fla", true, true, true)?;
        return ctx.fire(
            OP,
            Fire::at(
                GDN_CHUNK,
                "::pie::ssm::gdn_chunked_fla<::pie::ssm::state_bf16, 128, 128>",
            )
            .apply(
                Launch::grid([v_dim / BV_FLA, lanes, v_heads], [BV_FLA, 1, 1])
                    .smem(2 * BK_MAX_FLA * FLOAT),
            ),
            &[
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                state.slab.arg(),
                state.slot_ids.arg(),
                qkv.indptr.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                stated(OP, k_dim)?.arg(),
                stated(OP, v_dim)?.arg(),
                state.write_state.arg(),
                state.commit_len.arg(),
                state.write_state_mask.arg(),
                state.begin_at.arg(),
                state.fused_decay.arg(),
                snap,
                snap_rows,
                snap_slots,
                ctx.stage(),
            ],
        );
    }
    if state.snapshots.count > 0 {
        return Err(refuse(
            OP,
            "this fire lands state snapshots and the warp-tiled scan has no seat for them",
        ));
    }

    if k_dim <= WARP_TILED_K_MAX {
        seated(OP, state, "gdn_chunked_warp_tiled", true, false, false)?;
        return ctx.fire(
            OP,
            Fire::at(
                GDN_CHUNK,
                "::pie::ssm::gdn_chunked_warp_tiled<::pie::ssm::state_bf16, false>",
            )
            .apply(Launch::grid(
                [lanes, v_heads, v_dim.div_ceil(SCAN_WARPS)],
                [SCAN_WARPS * WARP, 1, 1],
            )),
            &[
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                state.slab.arg(),
                state.slot_ids.arg(),
                qkv.indptr.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                stated(OP, k_dim)?.arg(),
                stated(OP, v_dim)?.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ctx.stage(),
            ],
        );
    }

    seated(OP, state, "gdn_chunked", false, false, false)?;
    let (q_norm, k_norm) = if v_heads == k_heads {
        (staged.q_norm, staged.k_norm)
    } else {
        let wide = shape.elems(v_heads, k_dim);
        let repeated = plane(ctx, OP, "attn.ssm_gdn_chunk_repeat", 2 * wide)?;
        let (q, k) = (repeated, repeated + wide * u64::from(FLOAT));
        for (src, dst) in [(staged.q_norm, q), (staged.k_norm, k)] {
            ctx.fire(
                OP,
                Fire::at(GDN_PREP, "::pie::ssm::repeat_heads_f32<::pie::ssm::f32>")
                    .apply(Launch::grid([shape.n, v_heads, 1], [BLOCK, 1, 1])),
                &[
                    ArgValue::Ptr(src),
                    ArgValue::Ptr(dst),
                    stated(OP, k_heads)?.arg(),
                    stated(OP, v_heads)?.arg(),
                    stated(OP, k_dim)?.arg(),
                    stated(OP, v_heads / k_heads)?.arg(),
                    ctx.stage(),
                ],
            )?;
        }
        (q, k)
    };
    ctx.fire(
        OP,
        Fire::at(
            GDN_CHUNK,
            "::pie::ssm::gdn_chunked<::pie::ssm::state_bf16, false>",
        )
        .apply(Launch::grid([lanes, v_heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_dim * FLOAT)),
        &[
            ArgValue::Ptr(q_norm),
            ArgValue::Ptr(k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A B200 or H100 block opts into 227 KiB, an A100 into 163 KiB; an
    /// RTX 30/40/50 or a DGX Spark into 99 KiB, short of the scan.
    #[test]
    fn the_scan_fits_a_datacenter_block_and_not_a_99_kib_one() {
        assert!(scan_smem_at(128, SCAN_BV) <= 232_448);
        assert!(scan_smem_at(128, SCAN_BV) <= 166_912);
        assert!(scan_smem_at(128, SCAN_BV) > 101_376);
        assert!(scan_smem_at(128, SCAN_BV_NARROW) <= 101_376);
    }
}
