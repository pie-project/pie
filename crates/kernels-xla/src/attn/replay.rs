//! Bounded replay's floors and a lane's last key (DeepSeek-V4.1):
//! engine-cuda's `attn/kv_window.cu`, `replay_floor` and `kv_ends`.

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Func, Val};
use crate::lanes::{
    clamp_i, cmp_i, count_le, csr_lanes, flat_i32, iota, land_rows, ragged_lanes, take,
};
use crate::tensor::{Landing, Operand, RaggedTensor};

/// Each row's replay floor (i32 a row), and with `mark` the state slab its
/// lanes wrote: with `mark` the position past its lane's last row, written
/// to the lane's state too; without, the floor the lane's state holds (0 for
/// a lane with no state). `slots` is per row, a lane's slot its first row's;
/// a row of no lane lands 0.
pub fn floor(
    f: &mut Func,
    positions: RaggedTensor<Operand>,
    state: Operand,
    slots: Operand,
    mark: bool,
    floors: Landing,
) -> Result<(Val, Option<Val>), Error> {
    const OP: &str = "attn.replay_floor";
    expect(OP, positions.data, &[Dtype::I32])?;
    expect(OP, floors, &[Dtype::I32])?;
    expect(OP, state, &[Dtype::I32])?;
    csr_lanes(OP, positions.indptr)?;
    if state.width != 1 {
        return Err(refuse(OP, "a lane's replay state is one i32"));
    }
    let n = i64::from(floors.rows);
    if positions.data.elements() < floors.rows.into() || slots.elements() == 0 {
        return Err(refuse(OP, "the positions and slots ride the fire's rows"));
    }
    let ip = flat_i32(f, positions.indptr)?;
    let sl = flat_i32(f, slots)?;
    let lanes = ragged_lanes(f, ip, sl)?;
    let l = lanes.count(f);
    let pos = flat_i32(f, positions.data)?;
    let pos = f.slice_axis(pos, 0, 0, n)?;
    let s = i64::from(state.rows);
    let held = cmp_i(f, Cmp::Ge, lanes.slot, 0)?;
    let below = cmp_i(f, Cmp::Lt, lanes.slot, s)?;
    let held = f.and(held, below)?;
    let (lane_floor, next) = if mark {
        let last = f.add(lanes.begin, lanes.span)?;
        let one = f.like_i(last, 1);
        let last = f.sub(last, one)?;
        let last = clamp_i(f, last, 0, n - 1)?;
        let p = take(f, pos, last)?;
        let one = f.like_i(p, 1);
        let fl = f.add(p, one)?;
        let write = f.and(lanes.write, held)?;
        let rows = f.reshape(fl, &[l, 1])?;
        let slab = land_rows(f, state.val, lanes.slot, write, rows)?;
        (fl, Some(slab))
    } else {
        let col = flat_i32(f, state)?;
        let at = clamp_i(f, lanes.slot, 0, s - 1)?;
        let got = take(f, col, at)?;
        let zero = f.like_i(got, 0);
        (f.select(held, got, zero)?, None)
    };
    let t = iota(f, n);
    let lane = count_le(f, lanes.begin, t)?;
    let one = f.like_i(lane, 1);
    let lane = f.sub(lane, one)?;
    let lane = clamp_i(f, lane, 0, l - 1)?;
    let fl = take(f, lane_floor, lane)?;
    let first = f.slice_axis(ip, 0, 0, 1)?;
    let end = f.slice_axis(ip, 0, l, l + 1)?;
    let first = f.broadcast(first, &[n], &[0])?;
    let end = f.broadcast(end, &[n], &[0])?;
    let a = f.compare(Cmp::Ge, t, first)?;
    let b = f.compare(Cmp::Lt, t, end)?;
    let inside = f.and(a, b)?;
    let zero = f.like_i(fl, 0);
    Ok((f.select(inside, fl, zero)?, next))
}

/// Each row's lane's last key position (i32 a row): its pages (the last
/// `last_page_len` deep) less one; -1 for a row of no lane or an empty one.
pub fn kv_ends(
    f: &mut Func,
    request_of_token: Operand,
    page_indptr: Operand,
    last_page_len: Operand,
    page_size: i32,
    ends: Landing,
) -> Result<Val, Error> {
    const OP: &str = "attn.kv_ends";
    expect(OP, request_of_token, &[Dtype::I32])?;
    expect(OP, ends, &[Dtype::I32])?;
    let n = i64::from(ends.rows);
    if request_of_token.elements() < ends.rows.into() {
        return Err(refuse(OP, "the requests ride the fire's rows"));
    }
    let lanes = page_indptr.elements().saturating_sub(1);
    if lanes == 0 || last_page_len.elements() < lanes {
        return Err(refuse(OP, "the pool's tables name no lane"));
    }
    let l = lanes as i64;
    let req = flat_i32(f, request_of_token)?;
    let req = f.slice_axis(req, 0, 0, n)?;
    let ip = flat_i32(f, page_indptr)?;
    let lp = flat_i32(f, last_page_len)?;
    let a = cmp_i(f, Cmp::Ge, req, 0)?;
    let b = cmp_i(f, Cmp::Lt, req, l)?;
    let ok = f.and(a, b)?;
    let r = clamp_i(f, req, 0, l - 1)?;
    let lo = take(f, ip, r)?;
    let one = f.like_i(r, 1);
    let r1 = f.add(r, one)?;
    let hi = take(f, ip, r1)?;
    let pages = f.sub(hi, lo)?;
    let deep = take(f, lp, r)?;
    let one = f.like_i(pages, 1);
    let full = f.sub(pages, one)?;
    let ps = f.like_i(full, i64::from(page_size));
    let e = f.mul(full, ps)?;
    let e = f.add(e, deep)?;
    let e = f.sub(e, one)?;
    let some = cmp_i(f, Cmp::Gt, pages, 0)?;
    let ok = f.and(ok, some)?;
    let none = f.like_i(e, -1);
    Ok(f.select(ok, e, none)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn::paged;
    use crate::hlo::{Elem, Ty};
    use crate::tensor::KvPool;
    use crate::tuning::Tuning;

    fn arg(f: &mut Func, dtype: Dtype, rows: u32, width: u32) -> Operand {
        let elem = match dtype {
            Dtype::Bf16 => Elem::Bf16,
            Dtype::F32 => Elem::F32,
            _ => Elem::I32,
        };
        let val = f.param(Ty::new(elem, &[i64::from(rows), i64::from(width)]), None);
        Operand {
            val,
            rows,
            width,
            dtype,
        }
    }

    fn landing(rows: u32, width: u32, dtype: Dtype) -> Landing {
        Landing { rows, width, dtype }
    }

    #[test]
    fn a_floor_builds_marked_or_read() {
        for mark in [true, false] {
            let mut f = Func::new("main");
            let data = arg(&mut f, Dtype::I32, 7, 1);
            let indptr = arg(&mut f, Dtype::I32, 4, 1);
            let state = arg(&mut f, Dtype::I32, 5, 1);
            let slots = arg(&mut f, Dtype::I32, 7, 1);
            let (fl, next) = floor(
                &mut f,
                RaggedTensor { data, indptr },
                state,
                slots,
                mark,
                landing(7, 1, Dtype::I32),
            )
            .unwrap();
            assert_eq!(f.dims(fl), &[7]);
            assert_eq!(next.is_some(), mark);
            if let Some(next) = next {
                assert_eq!(f.dims(next), &[5, 1]);
            }
            let mut out = vec![fl];
            out.extend(next);
            assert!(f.module("m", &out).contains("stablehlo"));
        }
    }

    #[test]
    fn a_floor_refuses_a_wide_state() {
        let mut f = Func::new("main");
        let data = arg(&mut f, Dtype::I32, 3, 1);
        let indptr = arg(&mut f, Dtype::I32, 2, 1);
        let state = arg(&mut f, Dtype::I32, 4, 2);
        let slots = arg(&mut f, Dtype::I32, 3, 1);
        let got = floor(
            &mut f,
            RaggedTensor { data, indptr },
            state,
            slots,
            false,
            landing(3, 1, Dtype::I32),
        );
        assert!(got.is_err());
    }

    #[test]
    fn kv_ends_build_one_per_row() {
        let mut f = Func::new("main");
        let req = arg(&mut f, Dtype::I32, 6, 1);
        let ip = arg(&mut f, Dtype::I32, 3, 1);
        let lp = arg(&mut f, Dtype::I32, 2, 1);
        let e = kv_ends(&mut f, req, ip, lp, 4, landing(6, 1, Dtype::I32)).unwrap();
        assert_eq!(f.dims(e), &[6]);
    }

    #[test]
    fn a_window_sink_reads_a_shared_plane() {
        let (rows, heads, d) = (3u32, 4u32, 8u32);
        let mut f = Func::new("main");
        let q = arg(&mut f, Dtype::Bf16, rows, heads * d);
        let pos = arg(&mut f, Dtype::I32, rows, 1);
        let req = arg(&mut f, Dtype::I32, rows, 1);
        let keys = arg(&mut f, Dtype::Bf16, 20, d);
        let page_indices = arg(&mut f, Dtype::I32, 5, 1);
        let page_indptr = arg(&mut f, Dtype::I32, 3, 1);
        let pool = KvPool {
            keys,
            values: keys,
            page_indices,
            page_indptr,
            page_size: 4,
            max_pages: 3,
            seq_stride: u64::from(d),
            head_stride: u64::from(d),
        };
        let o = arg(&mut f, Dtype::Bf16, rows, heads * d);
        let lse = arg(&mut f, Dtype::F32, rows, heads);
        let call = paged::window_sink(q, pos, req, 3, d, 0.5, o, lse).unwrap();
        let (call, pool) = call.try_map(&pool, Ok::<_, ()>).unwrap();
        let (o, lse) = paged::attend(&mut f, call, pool, Tuning::DEFAULT).unwrap();
        assert!(lse.is_some());
        assert_eq!(f.ty(o).elements(), i64::from(rows * heads * d));
    }
}
