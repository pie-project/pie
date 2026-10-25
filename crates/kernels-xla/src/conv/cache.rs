//! The streaming conv's frame cache: gather the kept frames, store the new.

use crate::conv::{
    Boxes, Located, add_k, boxes, cmp_k, exclusive, k, lanes_of, locate, mul_k, nonzero, spread,
    take_elems, take1, unit_where_invalid,
};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Combine, Elem, Func, ScatterDims, Val};
use crate::tensor::{Landing, Operand, Stored};
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
    slab: impl Stored,
    slot_ids: impl Stored,
    grid: impl Stored,
    cache: impl Stored,
    frames: u32,
) -> Result<u32, Error> {
    // The frames are bf16; a slab kept in f32 (a shell that stores every
    // recurrent state wide) holds them exactly.
    if cache.dtype() != Dtype::Bf16 || !matches!(slab.dtype(), Dtype::Bf16 | Dtype::F32) {
        return Err(refuse(op, "the cache is bf16 and the slab bf16 or f32"));
    }
    let lanes = lanes_of(op, "input", grid)?;
    if slot_ids.dtype() != Dtype::I32 || slot_ids.elements() < u64::from(lanes) {
        return Err(refuse(
            op,
            format!(
                "the slot table is {}x{} {:?}; one i32 per lane is needed",
                slot_ids.rows(),
                slot_ids.width(),
                slot_ids.dtype()
            ),
        ));
    }
    nonzero(op, "the cached frames", frames)?;
    nonzero(op, "the channel count", cache.width())?;
    nonzero(op, "the slab's slot width", slab.width())?;
    Ok(lanes)
}

/// Each lane's slot index, `[rows]`, from the first `lanes` entries of the
/// slot table.
fn slots_at(f: &mut Func, slot_ids: Operand, lanes: u32, lane: Val) -> Result<Val, Error> {
    let s = slot_ids.val;
    let s = f.reshape(s, &[slot_ids.elements() as i64])?;
    let s = f.slice_axis(s, 0, 0, i64::from(lanes))?;
    take1(f, s, lane)
}

/// Reads each clip's `frames` cached frames out of its slot of the slab
/// (`[slots, frames*h*w*C]` flat per slot) into `cache`, clips packed in lane
/// order; what no clip holds, or what runs past a slot, reads zero.
/// kernels-cuda `conv::cache::cache_gather`.
pub fn cache_gather(
    f: &mut Func,
    slab: Operand,
    slot_ids: Operand,
    grid: Operand,
    frames: u32,
    cache: Landing,
) -> Result<Val, Error> {
    const OP: &str = "conv.cache_gather";
    let lanes = cache_checks(OP, slab, slot_ids, grid, cache, frames)?;
    let (rows, c) = (i64::from(cache.rows), i64::from(cache.width));
    let g = grid.val;
    let (at, _, base) = cache_rows(f, g, frames, rows)?;
    let slot = slots_at(f, slot_ids, lanes, at.lane)?;
    let dims = [rows, c];
    let row = f.iota(Elem::I32, &dims, 0);
    let col = f.iota(Elem::I32, &dims, 1);
    let base = spread(f, base, &dims, &[0])?;
    let local = f.sub(row, base)?;
    let local = mul_k(f, local, c)?;
    let local = f.add(local, col)?;
    let stride = i64::from(slab.width);
    let fits = cmp_k(f, Cmp::Lt, local, stride)?;
    let valid = spread(f, at.valid, &dims, &[0])?;
    let valid = f.and(valid, fits)?;
    let slot = spread(f, slot, &dims, &[0])?;
    let idx = mul_k(f, slot, stride)?;
    let idx = f.add(idx, local)?;
    let s = slab.val;
    let v = take_elems(f, s, idx, valid)?;
    Ok(v)
}

/// Writes each clip's last `frames` frames into its slot of the slab: frame
/// `f` of the block is the clip's frame `t - frames + f`, or, for a clip
/// shorter than the block, frame `f + t` of `cache` (the block read before
/// this fire). kernels-cuda `conv::cache::cache_store`; `poem_ir`'s
/// `CacheStore` passes the input as both `x` and `cache`, as CUDA does.
pub fn cache_store(
    f: &mut Func,
    x: Operand,
    cache: Operand,
    slot_ids: Operand,
    grid: Operand,
    frames: u32,
    slab: Operand,
    x_base: u32,
) -> Result<Val, Error> {
    let grid = crate::conv::rebased(f, grid, x_base)?;
    const OP: &str = "conv.cache_store";
    let lanes = cache_checks(OP, slab, slot_ids, grid, cache, frames)?;
    if x.dtype != Dtype::Bf16 || x.width != cache.width {
        return Err(refuse(OP, "the input is not `[rows, C_in]` bf16"));
    }
    let (rows, c) = (i64::from(cache.rows), i64::from(cache.width));
    let g = grid.val;
    let (at, b, base) = cache_rows(f, g, frames, rows)?;
    let slot = slots_at(f, slot_ids, lanes, at.lane)?;
    // Per row: the frame of the block and the voxel in the plane.
    let local_row = f.iota(Elem::I32, &[rows], 0);
    let local_row = f.sub(local_row, base)?;
    let plane = b.plane(f)?;
    let fr = f.div(local_row, plane)?;
    let hw = f.rem(local_row, plane)?;
    let src_t = f.add(b.t, fr)?;
    let src_t = add_k(f, src_t, -i64::from(frames))?;
    let live = cmp_k(f, Cmp::Ge, src_t, 0)?;
    let x_row = f.mul(src_t, plane)?;
    let x_row = f.add(x_row, hw)?;
    let x_row = f.add(x_row, b.off)?;
    let c_row = f.add(fr, b.t)?;
    let c_row = f.mul(c_row, plane)?;
    let c_row = f.add(c_row, hw)?;
    let c_row = f.add(c_row, base)?;
    let dims = [rows, c];
    let col = f.iota(Elem::I32, &dims, 1);
    let at_x = spread(f, x_row, &dims, &[0])?;
    let at_x = mul_k(f, at_x, c)?;
    let at_x = f.add(at_x, col)?;
    let at_c = spread(f, c_row, &dims, &[0])?;
    let at_c = mul_k(f, at_c, c)?;
    let at_c = f.add(at_c, col)?;
    let yes = f.const_i(Elem::Pred, 1, &dims);
    let xv = x.val;
    let from_x = take_elems(f, xv, at_x, yes)?;
    let cv = cache.val;
    let from_c = take_elems(f, cv, at_c, yes)?;
    let live = spread(f, live, &dims, &[0])?;
    let value = f.select(live, from_x, from_c)?;
    // Where it lands.
    let local = spread(f, local_row, &dims, &[0])?;
    let local = mul_k(f, local, c)?;
    let local = f.add(local, col)?;
    let stride = i64::from(slab.width);
    let fits = cmp_k(f, Cmp::Lt, local, stride)?;
    let valid = spread(f, at.valid, &dims, &[0])?;
    let valid = f.and(valid, fits)?;
    let slot = spread(f, slot, &dims, &[0])?;
    let idx = mul_k(f, slot, stride)?;
    let idx = f.add(idx, local)?;
    let far = k(f, idx, i64::from(i32::MAX));
    let idx = f.select(valid, idx, far)?;
    let n = rows * c;
    let idx = f.reshape(idx, &[n, 1])?;
    let value = f.reshape(value, &[n])?;
    let s = slab.val;
    let slab_elem = f.elem(s);
    let value = f.convert(value, slab_elem);
    let total = i64::from(slab.rows) * stride;
    let flat = f.reshape(s, &[total])?;
    let flat = f.scatter(
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
    Ok(flat)
}
