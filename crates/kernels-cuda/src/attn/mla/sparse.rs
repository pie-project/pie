//! The selected attention a block a row on tcgen05 (`attn/mla_sparse.cuh`).
pub(super) const SPARSE_SM100: &str = "attn/mla/sparse_sm100.cu";

use super::{Ctx, Error, KvPool, Tensor};
use crate::jit::{Arg, ArgValue, Fire, Launch, aligned16, symbol};
use crate::linear::gemm_sm100::supported;
use dtype::Dtype;

/// `pie::attn::mla::Params`, field for field.
#[repr(C)]
struct Params {
    o: u64,
    lse: u64,
    q: u64,
    q_pe: u64,
    qo_indptr: u64,
    page_indptr: u64,
    page_indices: u64,
    last_page_lens: u64,
    selection: u64,
    live: u64,
    requests: i32,
    causal: i32,
    runs: i32,
    page: i32,
    top_k: i32,
    rows: i32,
    q_stride: i32,
    qpe_stride: i32,
    o_stride: i32,
    scale_log2: f32,
    part_o: u64,
    part_lse: u64,
    splits: i32,
    dense: i32,
    kv: u64,
    kv_width: i32,
}

/// `pie::attn::mla::Layout<kRope, kE4>::kSmem`: the ring (two slots of
/// 64 keys' bf16 latent and rotated dims twice, or four of their e4m3
/// codes, five without rotated dims), P twice, the score exchange, six key lists, barriers, and the
/// 1024-byte alignment.
const fn smem(rope: i32, e4: bool) -> u32 {
    // Four e4m3 slots with rotated dims (4 x (32768 + 8192)) and five
    // without (5 x 32768) both come to 160 KiB.
    let ring = if e4 {
        5 * 32768
    } else {
        2 * (65536 + if rope > 0 { 16384 } else { 0 })
    };
    ring + 2 * 8192 + 16384 + 6 * 64 * 4 + 1024 + 1024
}

pub(super) struct Shape<'a> {
    pub q: Tensor,
    pub qo_indptr: u64,
    pub q_pe: Tensor,
    pub selection: Tensor,
    pub pool: &'a KvPool,
    pub heads: i32,
    pub kv_lora_rank: i32,
    pub rope: i32,
    pub requests: i32,
    pub sm_scale: f32,
    pub causal: bool,
    pub lse: u64,
}

/// Fires the pass where the profile says and the shape is the one it is
/// built for: 64 heads over a 512-wide bf16 latent and a 64-wide rotated
/// part; `None` where it does not serve.
pub(super) fn fire(
    ctx: &Ctx,
    op: &'static str,
    s: &Shape<'_>,
    o: &mut Tensor,
) -> Option<Result<(), Error>> {
    let pool = s.pool;
    let fp8 = fp8_rows(pool);
    // Where the tcgen05 pass serves, fp8 rows at any head count are read
    // 64 heads a pass (a rank's share under tensor parallelism, or more
    // than 64); elsewhere they take the path any other shape takes.
    if fp8 && s.heads != 64 && s.heads > 0 && ctx.tuning().mla_sparse && supported() {
        return Some(by_sixty_four(ctx, op, s, o));
    }
    let cells = pool.keys.rows;
    let fits = ctx.tuning().mla_sparse
        && supported()
        && s.heads == 64
        && s.kv_lora_rank == 512
        && (s.rope == 0 || s.rope == 64)
        && s.q.dtype == Dtype::Bf16
        && (pool.keys.dtype == Dtype::Bf16 || fp8)
        && pool.page_size > 0
        && cells > 0
        && s.selection.width > 0
        && o.width == 64 * 512
        && s.q.width == 64 * 512
        && [s.q.ptr, o.ptr, pool.keys.ptr]
            .iter()
            .all(|&at| aligned16(at))
        && (!fp8 || u32::try_from(s.rope).is_ok_and(|rope| pool.keys.width == 512 + rope))
        && (s.rope == 0
            || ((fp8 || (pool.values.dtype == Dtype::Bf16 && aligned16(pool.values.ptr)))
                && aligned16(s.q_pe.ptr)
                && s.q_pe.width == 64 * 64));
    if !fits {
        // An fp8 pool falls to the tensor-core kernel, which widens it.
        return None;
    }
    Some(launch(ctx, op, s, o))
}

/// An fp8 pool's pass over a head count other than 64, 64 heads a pass:
/// each pass reads its heads where they lie in the rows (the kernel
/// strides its rows), or, a pass short of 64 heads, copies them into
/// 64-head rows and its output back out. A head's attention reads no
/// other head's, so a padded head's rows (left as they are) touch nothing
/// kept. Each pass's log-sum-exps (64 a row in the kernel) are copied to
/// their heads likewise.
fn by_sixty_four(ctx: &Ctx, op: &'static str, s: &Shape<'_>, o: &mut Tensor) -> Result<(), Error> {
    let heads = u32::try_from(s.heads).unwrap_or(0);
    let rope = u32::try_from(s.rope).unwrap_or(u32::MAX);
    if s.kv_lora_rank != 512
        || !(rope == 0 || rope == 64)
        || s.q.width != heads * 512
        || o.width != heads * 512
        || (rope > 0 && s.q_pe.width != heads * rope)
    {
        return Err(crate::jit::refuse(
            op,
            format!(
                "fp8 latent rows are read 64 heads a pass, over 512 latent and 64 or no \
                 rotated dims, and this read is {} heads of {} latent and {} rotated dims",
                s.heads, s.kv_lora_rank, s.rope
            ),
        ));
    }
    let rows = o.rows;
    if rows == 0 {
        return Ok(());
    }
    let at = |ptr: u64, head: u32, width: u32| ptr + u64::from(head) * u64::from(width) * 2;
    // The kernel lands 64 log-sum-exps a row: a pass's land in a plane of
    // their own and are copied out (an f32 is two bf16s wide).
    let lse = if s.lse == 0 {
        0
    } else {
        ctx.scratch_shared(op, "attn.mla_sparse.pad_lse", rows as usize * 64 * 4)? as usize as u64
    };
    let lse_out = |first: u32, held: u32| -> Result<(), Error> {
        if lse == 0 {
            return Ok(());
        }
        let to = at(s.lse, first, 2);
        copy_rows(ctx, op, to, heads * 2, lse, 64 * 2, held * 2, rows)
    };
    for first in (0..heads).step_by(64) {
        let held = (heads - first).min(64);
        if held == 64 {
            // A whole pass reads its heads in place, at the rows' own pitch.
            let q = Tensor {
                ptr: at(s.q.ptr, first, 512),
                ..s.q
            };
            let q_pe = Tensor {
                ptr: at(s.q_pe.ptr, first, rope),
                ..s.q_pe
            };
            let mut out = Tensor {
                ptr: at(o.ptr, first, 512),
                ..*o
            };
            let pass = Shape {
                q,
                q_pe,
                heads: 64,
                lse,
                ..*s
            };
            passed(ctx, op, &pass, &mut out)?;
            lse_out(first, held)?;
            continue;
        }
        let wide = |name, width: u32| -> Result<Tensor, Error> {
            let bytes = rows as usize * 64 * width as usize * 2;
            let ptr = ctx.scratch_shared(op, name, bytes)? as usize as u64;
            Ok(Tensor::new(ptr, rows, 64 * width, Dtype::Bf16))
        };
        let q = wide("attn.mla_sparse.pad_q", 512)?;
        let from = at(s.q.ptr, first, 512);
        copy_rows(
            ctx,
            op,
            q.ptr,
            64 * 512,
            from,
            heads * 512,
            held * 512,
            rows,
        )?;
        let q_pe = if rope > 0 {
            let q_pe = wide("attn.mla_sparse.pad_q_pe", rope)?;
            let from = at(s.q_pe.ptr, first, rope);
            copy_rows(
                ctx,
                op,
                q_pe.ptr,
                64 * rope,
                from,
                heads * rope,
                held * rope,
                rows,
            )?;
            q_pe
        } else {
            s.q_pe
        };
        let mut out = wide("attn.mla_sparse.pad_o", 512)?;
        let pass = Shape {
            q,
            q_pe,
            heads: 64,
            lse,
            ..*s
        };
        passed(ctx, op, &pass, &mut out)?;
        let to = at(o.ptr, first, 512);
        copy_rows(
            ctx,
            op,
            to,
            heads * 512,
            out.ptr,
            64 * 512,
            held * 512,
            rows,
        )?;
        lse_out(first, held)?;
    }
    Ok(())
}

/// One 64-head pass, refused where the pass would not serve.
fn passed(ctx: &Ctx, op: &'static str, s: &Shape<'_>, o: &mut Tensor) -> Result<(), Error> {
    let pool = s.pool;
    let fits = ctx.tuning().mla_sparse
        && supported()
        && pool.page_size > 0
        && pool.keys.rows > 0
        && s.selection.width > 0
        && s.q.dtype == Dtype::Bf16
        && [
            s.q.ptr,
            o.ptr,
            pool.keys.ptr,
            u64::from(s.q.width) * 2,
            u64::from(o.width) * 2,
        ]
        .iter()
        .all(|&at| aligned16(at))
        && u32::try_from(s.rope).is_ok_and(|rope| pool.keys.width == 512 + rope)
        && (s.rope == 0 || (aligned16(s.q_pe.ptr) && aligned16(u64::from(s.q_pe.width) * 2)));
    if !fits {
        return Err(crate::jit::refuse(
            op,
            "fp8 latent rows are read by the tcgen05 sparse pass alone, and this \
             shape or device is not one it serves",
        ));
    }
    launch(ctx, op, s, o)
}

/// `rows` rows of `width` bf16s from `src` (a row every `src_pitch`) to
/// `dst` (a row every `dst_pitch`), on the fire's stream.
#[allow(clippy::too_many_arguments)]
fn copy_rows(
    ctx: &Ctx,
    op: &'static str,
    dst: u64,
    dst_pitch: u32,
    src: u64,
    src_pitch: u32,
    width: u32,
    rows: u32,
) -> Result<(), Error> {
    use cudarc::runtime::sys as rt;
    // SAFETY: both spans are live device memory the fire owns, `rows`
    // rows of their pitches, on its stream.
    let code = unsafe {
        rt::cudaMemcpy2DAsync(
            dst as usize as *mut core::ffi::c_void,
            dst_pitch as usize * 2,
            src as usize as *const core::ffi::c_void,
            src_pitch as usize * 2,
            width as usize * 2,
            rows as usize,
            rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().cast(),
        )
    };
    if code != rt::cudaError::cudaSuccess {
        return Err(crate::jit::refuse(
            op,
            format!(
                "`cudaMemcpy2DAsync` answered {} laying a pass's heads",
                code as i32
            ),
        ));
    }
    Ok(())
}

pub(super) use super::fp8_rows;

fn launch(ctx: &Ctx, op: &'static str, s: &Shape<'_>, o: &mut Tensor) -> Result<(), Error> {
    let rows = o.rows;
    if rows == 0 {
        return Ok(());
    }
    let pool = s.pool;
    let cells = pool.keys.rows;
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let map = |ptr: u64, width: u32, box_rows: u32| {
        crate::attn::bf16_map(
            ptr,
            [width, cells, 0],
            [u64::from(width) * 2, 0],
            [64, box_rows, 1],
        )
    };
    let fp8 = fp8_rows(pool);
    // fp8 rows: one u8 view of the plane, 128 codes a box, stands for
    // every map (the rotated codes come from the same rows).
    let (map_ckv, map_ckv_run, map_kpe) = if fp8 {
        use crate::linear::gemm_sm100::{Map, encode};
        use cudarc::driver::sys as dr;
        let u8_map = |box_rows: u32| {
            encode(Map {
                ptr: pool.keys.ptr,
                ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                dims: [pool.keys.width, cells, 0],
                strides: [u64::from(pool.keys.width), 0],
                boxes: [128, box_rows, 1],
                swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            })
        };
        let rows = u8_map(1)?;
        (rows, u8_map(16)?, rows)
    } else {
        let map_ckv = map(pool.keys.ptr, 512, 1)?;
        let map_ckv_run = map(pool.keys.ptr, 512, 16)?;
        // No rotated part: the map is never read; any valid one stands in.
        let map_kpe = if s.rope > 0 {
            map(pool.values.ptr, 64, 1)?
        } else {
            map_ckv
        };
        (map_ckv, map_ckv_run, map_kpe)
    };
    let live = match ctx.stage() {
        ArgValue::Ptr(at) => at,
        _ => 0,
    };
    // Few rows (a decode) deal each row's tiles over splits, so the
    // multiprocessors fill in one wave: as many as fit them whole, at
    // most a split a tile of the selection, and at most 16 (a split's
    // Q load and partial write cost about a tile's work of their own).
    let sms = ctx.multiprocessors().unwrap_or(148);
    let tiles = s.selection.width.div_ceil(64).max(1);
    let splits = (sms / rows.max(1)).min(tiles).clamp(1, 16);
    let (part_o, part_lse) = if splits > 1 {
        let units = rows as usize * splits as usize * 64;
        (
            ctx.scratch_shared(op, "attn.mla_sparse.part_o", units * 512 * 2)? as usize as u64,
            ctx.scratch_shared(op, "attn.mla_sparse.part_lse", units * 4)? as usize as u64,
        )
    } else {
        (0, 0)
    };
    let params = Params {
        o: o.ptr,
        lse: s.lse,
        q: s.q.ptr,
        q_pe: s.q_pe.ptr,
        qo_indptr: s.qo_indptr,
        page_indptr: pool.page_indptr.ptr,
        page_indices: pool.page_indices.ptr,
        last_page_lens: pool.last_page_lens.ptr,
        selection: s.selection.ptr,
        live,
        requests: s.requests,
        causal: i32::from(s.causal),
        runs: 1,
        page: pool.page_size,
        top_k: int(s.selection.width),
        rows: int(rows),
        q_stride: int(s.q.width),
        qpe_stride: int(s.q_pe.width),
        o_stride: int(o.width),
        scale_log2: s.sm_scale * core::f32::consts::LOG2_E,
        part_o,
        part_lse,
        splits: int(splits),
        dense: i32::from(ctx.tuning().mla_dense_short),
        kv: pool.keys.ptr,
        kv_width: int(pool.keys.width),
    };
    if fp8 && splits == 1 {
        // A prefill's rows on 128-key tiles and whole scores.
        // Five 32 KB halves, the rope atom and Q's staging's third atom
        // with rotated dims, Q's two atoms, two P buffers.
        let rope = if s.rope > 0 { 2 * 8192 + 8192 } else { 0 };
        let smem = 5 * 32768 + rope + 2 * 8192 + 2 * 8192 + 4 * 128 * 4 + 1024 + 2048 + 1024;
        return ctx.fire(
            op,
            Fire::at(
                SPARSE_SM100,
                symbol(&format!("::pie::attn::mla::mla_sparse_e4<{}>", s.rope)),
            )
            .apply(Launch::grid([rows.min(sms), 1, 1], [384, 1, 1]).smem(smem)),
            &[ArgValue::of(&params)],
        );
    }
    ctx.fire(
        op,
        Fire::at(
            SPARSE_SM100,
            symbol(&format!("::pie::attn::mla::mla_sparse<{}, {fp8}>", s.rope)),
        )
        .apply(Launch::grid([(rows * splits).min(sms), 1, 1], [384, 1, 1]).smem(smem(s.rope, fp8))),
        &[
            ArgValue::of(&map_ckv),
            ArgValue::of(&map_ckv_run),
            ArgValue::of(&map_kpe),
            ArgValue::of(&params),
        ],
    )?;
    if splits == 1 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(SPARSE_SM100, "::pie::attn::mla::mla_sparse_fold")
            .apply(Launch::grid([rows, 64, 1], [128, 1, 1]).programmatic(ctx.tuning().pdl)),
        &[
            ArgValue::Ptr(part_o),
            ArgValue::Ptr(part_lse),
            o.arg(),
            ArgValue::Ptr(s.lse),
            int(splits).arg(),
            int(o.width).arg(),
            ctx.stage(),
        ],
    )
}
