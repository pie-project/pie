//! A prefill's kv append as one Mosaic kernel: the fire's `k`, `v` rows are
//! copied into VMEM whole, each page run's rows realigned to their page
//! offsets by one sublane rotation of the 32-row aligned window that holds
//! them, laid over the page's old rows where the run covers part of it, and
//! written out as one 16-row page copy. The XLA form gathers both source
//! tiles of every run, picks the rows by a one-hot product and scatters the
//! pages, each a pass over HBM of its own.
//!
//! The runs are `paged::page_runs`'s: run `j` starts at fire row
//! `start[j]`, lands at offset `first[j]` of page `page[j]` and is `len[j]`
//! rows long; the first `total` runs are live.

use crate::hlo::{Built, Elem, Func, Val};
use crate::mosaic::{Body, Grid, ICmp, Kernel, MTy, Operand, Sem, Space, V, Window};
use crate::tensor::PageMap;

/// Where the kernel lands the rows: two planes of their own, or the keys
/// and values planes of the chunk pool (`u32 [R, 128]`), each page `p` of
/// the runs at its plane's page `map(p)`, in the pool's packed rows order
/// (a page `[16, w]` as `[16 · w / 128, 128]`, reshaped in VMEM).
pub(crate) enum Planes {
    Two {
        keys: Val,
        values: Val,
    },
    /// `maps`: one a plane written (one plane alone, or keys and values).
    /// `ordered`: the planes' pages lie in the row-pair order
    /// (`crate::hlo::PoolOrder`): u32 row `c · 16 + t` the pair `c` of
    /// page row `t`, where the packed rows order lays it at `t · w / 256 + c`.
    Pool {
        pool: Val,
        maps: Vec<PageMap>,
        slots: i64,
        ordered: bool,
    },
}

/// Rows a page holds for the kernel (a bf16 tile).
pub(crate) const PS: i64 = 16;

/// The VMEM the kernel holds for `n` fire rows of width `w` and `m` runs.
fn vmem(n: i64, w: i64, m: i64) -> i64 {
    // both planes' rows, the staged pages and the old pages, and the window
    // temporaries
    2 * n * w * 2 + 2 * 2 * m * PS * w * 2 + 64 * w * 4 * 4
}

/// Whether the kernel takes a fire of `n` rows into `[slots, w]` planes of
/// `ps`-row pages with `m` runs (`PIE_XLA_KV_KERNEL=0` turns it off).
pub(crate) fn fits(n: i64, w: i64, slots: i64, ps: i64, m: i64, vmem_cap: i64) -> bool {
    !std::env::var("PIE_XLA_KV_KERNEL").is_ok_and(|v| v == "0")
        && ps == PS
        && n >= 2 * PS
        && n % PS == 0
        && w % 128 == 0
        && slots % PS == 0
        && vmem(n, w, m) <= vmem_cap
}

/// The planes `keys`, `values` (`[slots, w]` bf16) with the fire's rows
/// `k`, `v` (`[n, w]` bf16) laid in by the runs (`[m]` i32 each) and
/// `total` (`[1]` i32); written in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write(
    f: &mut Func,
    srcs: &[Val],
    planes: Planes,
    runs: [Val; 4],
    total: Val,
    vmem_cap: i64,
) -> Built<Vec<Val>> {
    let (n, w) = (f.dims(srcs[0])[0], f.dims(srcs[0])[1]);
    let m = f.dims(runs[0])[0];
    // planes written: each its own source rows
    let np = srcs.len();
    let any_of = |array: Vec<i64>, elem: Elem| Operand {
        array,
        elem,
        block: vec![],
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    };
    let any = |rows: i64| any_of(vec![rows, w], Elem::Bf16);
    // a page's rows in the scratch: `[PS, w]`, or `[PS · w / 128, 128]` as
    // the pool lays them
    let kk = w / 128;
    let (pooled, maps, slots) = match &planes {
        Planes::Two { keys, .. } => (None, vec![PageMap::IDENTITY; 2], f.dims(*keys)[0]),
        Planes::Pool {
            pool, maps, slots, ..
        } => (Some(f.dims(*pool)[0]), maps.clone(), *slots),
    };
    let ordered = matches!(planes, Planes::Pool { ordered: true, .. });
    if maps.len() != np {
        return Err(crate::hlo::Malformed {
            op: "kv_append",
            detail: format!("{np} source planes for {} written", maps.len()),
        });
    }
    // `direct`: the pool's pages built as its own u32 rows (each row a pair
    // of a page row's 128-lane blocks, low and high) and laid by sublane
    // strided stores, with no relayout of the page in VMEM.
    let direct = pooled.is_some() && kk % 2 == 0;
    if ordered && !direct {
        return Err(crate::hlo::Malformed {
            op: "kv_append",
            detail: format!("rows of {w} in the row-pair order"),
        });
    }
    let half = kk / 2;
    let (tile, laid_elem) = if direct {
        (vec![PS * half, 128], Elem::I32)
    } else if pooled.is_some() {
        (vec![PS * kk, 128], Elem::Bf16)
    } else {
        (vec![PS, w], Elem::Bf16)
    };
    let (ins, outs) = match pooled {
        None => (
            vec![any(n), any(n), any(slots), any(slots)],
            vec![any(slots), any(slots)],
        ),
        Some(r) => {
            let mut ins: Vec<Operand> = (0..np).map(|_| any(n)).collect();
            ins.push(any_of(vec![r, 128], Elem::U32));
            (ins, vec![any_of(vec![r, 128], Elem::U32)])
        }
    };
    let limit = (vmem(n, w, m) * 3 / 2 + (8 << 20)).min(vmem_cap);
    let kern = Kernel {
        name: "kv_append".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![m], vec![m], vec![m], vec![m], vec![1]],
        ins,
        outs,
        vmem_limit: Some(u64::try_from(limit).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![np as i64, n, w], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![np as i64, m, tile[0], tile[1]], laid_elem, Space::Vmem),
            MTy::Ref(vec![np as i64, m, tile[0], tile[1]], laid_elem, Space::Vmem),
            MTy::Sem(vec![4]),
        ],
    };
    let c = kern.build(|b, blk| {
        let pre = &blk.grid.prefetch;
        // each plane's source and destination: the pool read as bf16 rows
        // of 128 for both
        let (src_planes, dst_planes) = if pooled.is_some() {
            let as_ = if direct { Elem::I32 } else { Elem::Bf16 };
            let i = b.ref_bitcast(blk.ins[np], as_)?;
            let o = b.ref_bitcast(blk.outs[0], as_)?;
            (vec![i; np], vec![o; np])
        } else {
            (vec![blk.ins[2], blk.ins[3]], vec![blk.outs[0], blk.outs[1]])
        };
        let (rows, stage, old, sems) = (
            blk.scratch[0],
            blk.scratch[1],
            blk.scratch[2],
            blk.scratch[3],
        );
        let z = b.i32(0);
        let i0 = b.index(0);
        let sem = |b: &mut Body, i: i64| -> Built<V> {
            let at = b.i32(i);
            let s = b.ref_slice(sems, &[at], &[1])?;
            b.ref_squeeze(s)
        };
        let plane_rows = |b: &mut Body, x: i64| -> Built<V> {
            let at = b.i32(x);
            let z = b.i32(0);
            let r = b.ref_slice(rows, &[at, z, z], &[1, n, w])?;
            b.ref_squeeze(r)
        };
        // The fire's rows, both planes, in whole.
        for x in 0..np as i64 {
            let dst = plane_rows(b, x)?;
            let s = sem(b, x)?;
            b.dma_start(blk.ins[x as usize], dst, s)?;
        }
        for x in 0..np as i64 {
            let dst = plane_rows(b, x)?;
            let s = sem(b, x)?;
            b.dma_wait(s, blk.ins[x as usize], dst)?;
        }
        let total = b.load(pre[4], &[i0])?;
        let one = b.i32(1);
        // A run covers part of its page: its page's old rows are read first.
        let partial = |b: &mut Body, j: V| -> Built<V> {
            let f0 = b.load(pre[1], &[j])?;
            let len = b.load(pre[2], &[j])?;
            let z = b.i32(0);
            let ps = b.i32(PS);
            let a = b.cmpi(ICmp::Ne, f0, z)?;
            let c = b.cmpi(ICmp::Ne, len, ps)?;
            b.select(a, a, c)
        };
        let page_ref = |b: &mut Body, plane: V, x: i64, j: V| -> Built<V> {
            let p = b.load(pre[3], &[j])?;
            let map = maps[x as usize];
            let p = if map.is_identity() {
                p
            } else {
                let by = b.i32(map.stride);
                let p = b.muli(p, by)?;
                let from = b.i32(map.offset);
                b.addi(p, from)?
            };
            let rows = b.i32(tile[0]);
            let at = b.muli(p, rows)?;
            let at = b.assume_multiple(at, PS)?;
            let z = b.i32(0);
            b.ref_slice(plane, &[at, z], &tile)
        };
        let slot_ref = |b: &mut Body, buf: V, x: i64, j: V| -> Built<V> {
            let xv = b.i32(x);
            let z = b.i32(0);
            let r = b.ref_slice(buf, &[xv, j, z, z], &[1, 1, tile[0], tile[1]])?;
            let r = b.ref_squeeze(r)?;
            b.ref_squeeze(r)
        };
        // a page's rows as the kernel computes them, and back as laid
        let as_page = |b: &mut Body, x: V| -> Built<V> {
            if pooled.is_some() {
                b.shape_cast(x, &[PS, w])
            } else {
                Ok(x)
            }
        };
        let as_laid = |b: &mut Body, x: V| -> Built<V> {
            if pooled.is_some() {
                b.shape_cast(x, &tile)
            } else {
                Ok(x)
            }
        };
        b.for_range(z, total, one, |b, j| {
            let p = partial(b, j)?;
            b.when(p, |b| {
                for x in 0..np as i64 {
                    let src = page_ref(b, src_planes[x as usize], x, j)?;
                    let dst = slot_ref(b, old, x, j)?;
                    let s = sem(b, 2)?;
                    b.dma_start(src, dst, s)?;
                }
                Ok(())
            })
        })?;
        b.for_range(z, total, one, |b, j| {
            let s0 = b.load(pre[0], &[j])?;
            let f0 = b.load(pre[1], &[j])?;
            let len = b.load(pre[2], &[j])?;
            // The 32-row aligned window that holds the run's rows, and where
            // the row bound for the page's offset 0 sits in it.
            let base = b.subi(s0, f0)?;
            let sixteen = b.i32(PS);
            let biased = b.addi(base, sixteen)?;
            let t = b.divsi(biased, sixteen)?;
            let t = b.subi(t, one)?;
            let t0 = b.muli(t, sixteen)?;
            let zero = b.i32(0);
            let top = b.i32(n - 2 * PS);
            let t0 = b.maxsi(t0, zero)?;
            let t0 = b.minsi(t0, top)?;
            let t0 = b.assume_multiple(t0, PS)?;
            let lead = b.subi(base, t0)?;
            // rotate so the window's row `lead + r` lands at row `r`
            let thirty_two = b.i32(2 * PS);
            let by = b.subi(thirty_two, lead)?;
            let by = b.remsi(by, thirty_two)?;
            let rr = b.iota(&[PS, w], 0);
            let lo = b.broadcast_i(f0, &[PS, w])?;
            let end = b.addi(f0, len)?;
            let hi = b.broadcast_i(end, &[PS, w])?;
            let ge = b.cmpi(ICmp::Sge, rr, lo)?;
            let lt = b.cmpi(ICmp::Slt, rr, hi)?;
            let laid = b.andi(ge, lt)?;
            let p = partial(b, j)?;
            let at = b.as_index(t0)?;
            if direct {
                // the old page lands first where the run covers part of it
                b.when(p, |b| {
                    for x in 0..np as i64 {
                        let src = page_ref(b, src_planes[x as usize], x, j)?;
                        let held = slot_ref(b, old, x, j)?;
                        let s = sem(b, 2)?;
                        b.dma_wait(s, src, held)?;
                    }
                    Ok(())
                })?;
                let rr = b.iota(&[PS, 128], 0);
                let lo = b.broadcast_i(f0, &[PS, 128])?;
                let hi = b.broadcast_i(end, &[PS, 128])?;
                let ge = b.cmpi(ICmp::Sge, rr, lo)?;
                let lt = b.cmpi(ICmp::Slt, rr, hi)?;
                let laid = b.andi(ge, lt)?;
                let sixteen_v = b.splat_i(&[PS, 128], 16)?;
                let high_v = b.splat_i(&[PS, 128], i64::from(0xFFFF_0000u32 as i32))?;
                // a page's f32 rows [PS, w] as the pool's u32 rows into the
                // stage, over the old rows where `merge`
                let pack = |b: &mut Body, tile: V, x: i64, merge: bool| -> Built<()> {
                    let held = slot_ref(b, old, x, j)?;
                    let dst = slot_ref(b, stage, x, j)?;
                    for c in 0..half {
                        // u32 row 4p + c of the page: bf16 lane blocks 2c
                        // (low) and 2c + 1 (high) of page row p; a bf16's
                        // bits are its f32's top half
                        let a = b.slice(tile, &[0, 256 * c], &[PS, 128])?;
                        let a = b.bitcast_v(a, Elem::I32)?;
                        let a = b.shrui(a, sixteen_v)?;
                        let h = b.slice(tile, &[0, 256 * c + 128], &[PS, 128])?;
                        let h = b.bitcast_v(h, Elem::I32)?;
                        let h = b.andi(h, high_v)?;
                        let mut u = b.ori(h, a)?;
                        if ordered {
                            // the pair's rows one run of the page
                            let ci = b.index(c * PS);
                            if merge {
                                let o = b.vload(held, &[ci, i0], &[PS, 128])?;
                                u = b.select(laid, u, o)?;
                            }
                            b.vstore(u, dst, &[ci, i0])?;
                            continue;
                        }
                        let ci = b.index(c);
                        if merge {
                            let o = b.strided_load(held, &[ci, i0], &[half, 1], &[PS, 128])?;
                            u = b.select(laid, u, o)?;
                        }
                        b.strided_store(u, dst, &[ci, i0], &[half, 1])?;
                    }
                    Ok(())
                };
                // A whole page from a page-aligned fire row: its rows as
                // they are (every run of a prompt from a page's start).
                let r16 = b.remsi(s0, sixteen)?;
                let off = b.cmpi(ICmp::Ne, r16, zero)?;
                let un = b.select(p, p, off)?;
                let t1 = b.i32(1);
                let t1 = b.cmpi(ICmp::Ne, t1, zero)?;
                let no = b_false(b)?;
                let al = b.select(un, no, t1)?;
                b.when(al, |b| {
                    let s0a = b.assume_multiple(s0, PS)?;
                    let sa = b.as_index(s0a)?;
                    for x in 0..np as i64 {
                        let src = plane_rows(b, x)?;
                        let tile = b.vload(src, &[sa, i0], &[PS, w])?;
                        let tile = b.convert_f(tile, Elem::F32)?;
                        pack(b, tile, x, false)?;
                    }
                    Ok(())
                })?;
                b.when(un, |b| {
                    for x in 0..np as i64 {
                        let src = plane_rows(b, x)?;
                        let win = b.vload(src, &[at, i0], &[2 * PS, w])?;
                        let win = b.convert_f(win, Elem::F32)?;
                        let win = b.rotate(win, by, 0)?;
                        let tile = b.rows(win, 0, PS)?;
                        pack(b, tile, x, true)?;
                    }
                    Ok(())
                })?;
            } else {
                for x in 0..np as i64 {
                    let src = plane_rows(b, x)?;
                    let win = b.vload(src, &[at, i0], &[2 * PS, w])?;
                    let win = b.convert_f(win, Elem::F32)?;
                    let win = b.rotate(win, by, 0)?;
                    let tile = b.rows(win, 0, PS)?;
                    let tile = b.convert_f(tile, Elem::Bf16)?;
                    let tile = as_laid(b, tile)?;
                    let dst = slot_ref(b, stage, x, j)?;
                    b.vstore_all(tile, dst)?;
                }
                b.when(p, |b| {
                    for x in 0..np as i64 {
                        let src = page_ref(b, src_planes[x as usize], x, j)?;
                        let held = slot_ref(b, old, x, j)?;
                        let s = sem(b, 2)?;
                        b.dma_wait(s, src, held)?;
                        let o = b.vload_all(held)?;
                        let o = as_page(b, o)?;
                        let dst = slot_ref(b, stage, x, j)?;
                        let t = b.vload_all(dst)?;
                        let t = as_page(b, t)?;
                        let merged = b.select(laid, t, o)?;
                        let merged = as_laid(b, merged)?;
                        b.vstore_all(merged, dst)?;
                    }
                    Ok(())
                })?;
            }
            for x in 0..np as i64 {
                let src = slot_ref(b, stage, x, j)?;
                let dst = page_ref(b, dst_planes[x as usize], x, j)?;
                let s = sem(b, 3)?;
                b.dma_start(src, dst, s)?;
            }
            Ok(())
        })?;
        b.for_range(z, total, one, |b, j| {
            for x in 0..np as i64 {
                let src = slot_ref(b, stage, x, j)?;
                let dst = page_ref(b, dst_planes[x as usize], x, j)?;
                let s = sem(b, 3)?;
                b.dma_wait(s, src, dst)?;
            }
            Ok(())
        })
    })?;
    let mut ops = vec![runs[0], runs[1], runs[2], runs[3], total];
    ops.extend_from_slice(srcs);
    match planes {
        Planes::Two { keys, values } => {
            ops.extend([keys, values]);
            c.call_aliased(f, &ops, &[(0, 5 + np), (1, 6 + np)])
        }
        Planes::Pool { pool, .. } => {
            ops.push(pool);
            c.call_aliased(f, &ops, &[(0, 5 + np)])
        }
    }
}

/// An i1 false.
fn b_false(b: &mut Body) -> Built<V> {
    let z = b.i32(0);
    b.cmpi(ICmp::Ne, z, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_writes_both_planes_in_place() {
        let mut f = Func::new("main");
        let k = f.param(crate::hlo::Ty::new(Elem::Bf16, &[1616, 1024]), None);
        let v = f.param(crate::hlo::Ty::new(Elem::Bf16, &[1616, 1024]), None);
        let keys = f.param(crate::hlo::Ty::new(Elem::Bf16, &[4096, 1024]), None);
        let values = f.param(crate::hlo::Ty::new(Elem::Bf16, &[4096, 1024]), None);
        let runs: Vec<Val> = (0..4)
            .map(|_| f.param(crate::hlo::Ty::new(Elem::I32, &[165]), None))
            .collect();
        let total = f.param(crate::hlo::Ty::new(Elem::I32, &[1]), None);
        assert!(fits(1616, 1024, 4096, 16, 165, 110 << 20));
        assert!(!fits(1000, 1024, 4096, 16, 165, 110 << 20));
        let out = write(
            &mut f,
            &[k, v],
            Planes::Two { keys, values },
            [runs[0], runs[1], runs[2], runs[3]],
            total,
            110 << 20,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(f.dims(out[0]), &[4096, 1024]);
    }
}
