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
    k: Val,
    v: Val,
    keys: Val,
    values: Val,
    runs: [Val; 4],
    total: Val,
    vmem_cap: i64,
) -> Built<Vec<Val>> {
    let (n, w) = (f.dims(k)[0], f.dims(k)[1]);
    let slots = f.dims(keys)[0];
    let m = f.dims(runs[0])[0];
    let any = |rows: i64| Operand {
        array: vec![rows, w],
        elem: Elem::Bf16,
        block: vec![],
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    };
    let limit = (vmem(n, w, m) * 3 / 2 + (8 << 20)).min(vmem_cap);
    let kern = Kernel {
        name: "kv_append".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![m], vec![m], vec![m], vec![m], vec![1]],
        ins: vec![any(n), any(n), any(slots), any(slots)],
        outs: vec![any(slots), any(slots)],
        vmem_limit: Some(u64::try_from(limit).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![2, n, w], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![2, m, PS, w], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![2, m, PS, w], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![4]),
        ],
    };
    let c = kern.build(|b, blk| {
        let pre = &blk.grid.prefetch;
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
        for x in 0..2 {
            let dst = plane_rows(b, x)?;
            let s = sem(b, x)?;
            b.dma_start(blk.ins[x as usize], dst, s)?;
        }
        for x in 0..2 {
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
        let page_ref = |b: &mut Body, plane: V, j: V| -> Built<V> {
            let p = b.load(pre[3], &[j])?;
            let ps = b.i32(PS);
            let at = b.muli(p, ps)?;
            let at = b.assume_multiple(at, PS)?;
            let z = b.i32(0);
            b.ref_slice(plane, &[at, z], &[PS, w])
        };
        let slot_ref = |b: &mut Body, buf: V, x: i64, j: V| -> Built<V> {
            let xv = b.i32(x);
            let z = b.i32(0);
            let r = b.ref_slice(buf, &[xv, j, z, z], &[1, 1, PS, w])?;
            let r = b.ref_squeeze(r)?;
            b.ref_squeeze(r)
        };
        b.for_range(z, total, one, |b, j| {
            let p = partial(b, j)?;
            b.when(p, |b| {
                for x in 0..2 {
                    let src = page_ref(b, blk.ins[2 + x as usize], j)?;
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
            for x in 0..2 {
                let src = plane_rows(b, x)?;
                let win = b.vload(src, &[at, i0], &[2 * PS, w])?;
                let win = b.convert_f(win, Elem::F32)?;
                let win = b.rotate(win, by, 0)?;
                let tile = b.rows(win, 0, PS)?;
                let tile = b.convert_f(tile, Elem::Bf16)?;
                let dst = slot_ref(b, stage, x, j)?;
                b.vstore_all(tile, dst)?;
            }
            b.when(p, |b| {
                for x in 0..2 {
                    let src = page_ref(b, blk.ins[2 + x as usize], j)?;
                    let held = slot_ref(b, old, x, j)?;
                    let s = sem(b, 2)?;
                    b.dma_wait(s, src, held)?;
                    let o = b.vload_all(held)?;
                    let dst = slot_ref(b, stage, x, j)?;
                    let t = b.vload_all(dst)?;
                    let merged = b.select(laid, t, o)?;
                    b.vstore_all(merged, dst)?;
                }
                Ok(())
            })?;
            for x in 0..2 {
                let src = slot_ref(b, stage, x, j)?;
                let dst = page_ref(b, blk.outs[x as usize], j)?;
                let s = sem(b, 3)?;
                b.dma_start(src, dst, s)?;
            }
            Ok(())
        })?;
        b.for_range(z, total, one, |b, j| {
            for x in 0..2 {
                let src = slot_ref(b, stage, x, j)?;
                let dst = page_ref(b, blk.outs[x as usize], j)?;
                let s = sem(b, 3)?;
                b.dma_wait(s, src, dst)?;
            }
            Ok(())
        })
    })?;
    let ops = [
        runs[0], runs[1], runs[2], runs[3], total, k, v, keys, values,
    ];
    c.call_aliased(f, &ops, &[(0, 7), (1, 8)])
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
            k,
            v,
            keys,
            values,
            [runs[0], runs[1], runs[2], runs[3]],
            total,
            110 << 20,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(f.dims(out[0]), &[4096, 1024]);
    }
}
