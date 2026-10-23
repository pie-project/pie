//! The streaming conv's frame cache: gather the kept frames, store the new.

use crate::conv::{
    Boxes, Located, add_k, boxes, cmp_k, exclusive, k, lanes_of, locate, mul_k, nonzero, spread,
    take_elems, take1, unit_where_invalid,
};
use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Combine, Elem, Func, ScatterDims, Val};
use crate::tensor::Tensor;
use dtype::Dtype;

/// Where each cache row sits: its lane (clips hold `frames * h*w` rows each,
/// packed in lane order from 0) and its row inside that lane's block.
fn cache_rows(
    f: &mut Func,
    grid: Val,
    frames: u32,
    rows: i64,
) -> Result<(Located, Boxes, Val), Error> {
    let bx = boxes(f, grid)?;
    let plane = bx.plane(f)?;
    let count = mul_k(f, plane, i64::from(frames))?;
    let start = exclusive(f, count)?;
    let at = locate(f, start, count, rows)?;
    let base = take1(f, start, at.lane)?;
    let b = bx.at(f, at.lane)?;
    let b = unit_where_invalid(f, b, at.valid)?;
    Ok((at, b, base))
}

fn cache_checks(
    op: &'static str,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    cache: Tensor,
    frames: u32,
) -> Result<u32, Error> {
    // The frames are bf16; a slab kept in f32 (a shell that stores every
    // recurrent state wide) holds them exactly.
    if cache.dtype != Dtype::Bf16 || !matches!(slab.dtype, Dtype::Bf16 | Dtype::F32) {
        return Err(refuse(op, "the cache is bf16 and the slab bf16 or f32"));
    }
    let lanes = lanes_of(op, "input", grid)?;
    if slot_ids.dtype != Dtype::I32 || slot_ids.elements() < u64::from(lanes) {
        return Err(refuse(
            op,
            format!(
                "the slot table is {}x{} {:?}; one i32 per lane is needed",
                slot_ids.rows, slot_ids.width, slot_ids.dtype
            ),
        ));
    }
    nonzero(op, "the cached frames", frames)?;
    nonzero(op, "the channel count", cache.width)?;
    nonzero(op, "the slab's slot width", slab.width)?;
    Ok(lanes)
}

/// Each lane's slot index, `[rows]`, from the first `lanes` entries of the
/// slot table.
fn slots_at(cx: &mut Cx<'_>, slot_ids: Tensor, lanes: u32, lane: Val) -> Result<Val, Error> {
    let s = cx.read(slot_ids)?;
    let s = cx.reshape(s, &[slot_ids.elements() as i64])?;
    let s = cx.slice_axis(s, 0, 0, i64::from(lanes))?;
    take1(cx, s, lane)
}

/// Reads each clip's `frames` cached frames out of its slot of the slab
/// (`[slots, frames*h*w*C]` flat per slot) into `cache`, clips packed in lane
/// order; what no clip holds, or what runs past a slot, reads zero.
/// kernels-cuda `conv::cache::cache_gather`.
pub fn cache_gather(
    ctx: &Ctx<'_>,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    cache: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.cache_gather";
    let lanes = cache_checks(OP, slab, slot_ids, grid, cache, frames)?;
    ctx.emit(&mut |cx| {
        let (rows, c) = (i64::from(cache.rows), i64::from(cache.width));
        let g = cx.read(grid)?;
        let (at, _, base) = cache_rows(cx, g, frames, rows)?;
        let slot = slots_at(cx, slot_ids, lanes, at.lane)?;
        let dims = [rows, c];
        let row = cx.iota(Elem::I32, &dims, 0);
        let col = cx.iota(Elem::I32, &dims, 1);
        let base = spread(cx, base, &dims, &[0])?;
        let local = cx.sub(row, base)?;
        let local = mul_k(cx, local, c)?;
        let local = cx.add(local, col)?;
        let stride = i64::from(slab.width);
        let fits = cmp_k(cx, Cmp::Lt, local, stride)?;
        let valid = spread(cx, at.valid, &dims, &[0])?;
        let valid = cx.and(valid, fits)?;
        let slot = spread(cx, slot, &dims, &[0])?;
        let idx = mul_k(cx, slot, stride)?;
        let idx = cx.add(idx, local)?;
        let s = cx.read(slab)?;
        let v = take_elems(cx, s, idx, valid)?;
        cx.write(cache, v)
    })
}

/// Writes each clip's last `frames` frames into its slot of the slab: frame
/// `f` of the block is the clip's frame `t - frames + f`, or, for a clip
/// shorter than the block, frame `f + t` of `cache` (the block read before
/// this fire). kernels-cuda `conv::cache::cache_store`; `poem_ir`'s
/// `CacheStore` passes the input as both `x` and `cache`, as CUDA does.
pub fn cache_store(
    ctx: &Ctx<'_>,
    x: Tensor,
    cache: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    slab: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.cache_store";
    let lanes = cache_checks(OP, slab, slot_ids, grid, cache, frames)?;
    if x.dtype != Dtype::Bf16 || x.width != cache.width {
        return Err(refuse(OP, "the input is not `[rows, C_in]` bf16"));
    }
    ctx.emit(&mut |cx| {
        let (rows, c) = (i64::from(cache.rows), i64::from(cache.width));
        let g = cx.read(grid)?;
        let (at, b, base) = cache_rows(cx, g, frames, rows)?;
        let slot = slots_at(cx, slot_ids, lanes, at.lane)?;
        // Per row: the frame of the block and the voxel in the plane.
        let local_row = cx.iota(Elem::I32, &[rows], 0);
        let local_row = cx.sub(local_row, base)?;
        let plane = b.plane(cx)?;
        let fr = cx.div(local_row, plane)?;
        let hw = cx.rem(local_row, plane)?;
        let src_t = cx.add(b.t, fr)?;
        let src_t = add_k(cx, src_t, -i64::from(frames))?;
        let live = cmp_k(cx, Cmp::Ge, src_t, 0)?;
        let x_row = cx.mul(src_t, plane)?;
        let x_row = cx.add(x_row, hw)?;
        let x_row = cx.add(x_row, b.off)?;
        let c_row = cx.add(fr, b.t)?;
        let c_row = cx.mul(c_row, plane)?;
        let c_row = cx.add(c_row, hw)?;
        let c_row = cx.add(c_row, base)?;
        let dims = [rows, c];
        let col = cx.iota(Elem::I32, &dims, 1);
        let at_x = spread(cx, x_row, &dims, &[0])?;
        let at_x = mul_k(cx, at_x, c)?;
        let at_x = cx.add(at_x, col)?;
        let at_c = spread(cx, c_row, &dims, &[0])?;
        let at_c = mul_k(cx, at_c, c)?;
        let at_c = cx.add(at_c, col)?;
        let yes = cx.const_i(Elem::Pred, 1, &dims);
        let xv = cx.read(x)?;
        let from_x = take_elems(cx, xv, at_x, yes)?;
        let cv = cx.read(cache)?;
        let from_c = take_elems(cx, cv, at_c, yes)?;
        let live = spread(cx, live, &dims, &[0])?;
        let value = cx.select(live, from_x, from_c)?;
        // Where it lands.
        let local = spread(cx, local_row, &dims, &[0])?;
        let local = mul_k(cx, local, c)?;
        let local = cx.add(local, col)?;
        let stride = i64::from(slab.width);
        let fits = cmp_k(cx, Cmp::Lt, local, stride)?;
        let valid = spread(cx, at.valid, &dims, &[0])?;
        let valid = cx.and(valid, fits)?;
        let slot = spread(cx, slot, &dims, &[0])?;
        let idx = mul_k(cx, slot, stride)?;
        let idx = cx.add(idx, local)?;
        let far = k(cx, idx, i64::from(i32::MAX));
        let idx = cx.select(valid, idx, far)?;
        let n = rows * c;
        let idx = cx.reshape(idx, &[n, 1])?;
        let value = cx.reshape(value, &[n])?;
        let s = cx.read(slab)?;
        let slab_elem = cx.elem(s);
        let value = cx.convert(value, slab_elem);
        let total = i64::from(slab.rows) * stride;
        let flat = cx.reshape(s, &[total])?;
        let flat = cx.scatter(
            flat,
            idx,
            value,
            &ScatterDims {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..ScatterDims::default()
            },
            Combine::Set,
        )?;
        cx.write(slab, flat)
    })
}
