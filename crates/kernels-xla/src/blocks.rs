//! Whole blocks of the chunk pool moved by DMA, as Mosaic kernels: what a
//! pool view's block gather or scatter is on a TPU for elements narrower
//! than a word. XLA reinterprets `u32` as a narrower element through a minor
//! axis the tiled layout pads (shifts and masks over every element read); a
//! DMA through the pool reference bitcast to the element
//! (`tpu.memref_bitcast`, the packed rows order) moves the bytes as they lie,
//! HBM to HBM, one copy a block in flight together.

use crate::hlo::{Built, Elem, Func, Val};
use crate::mosaic::{Body, Grid, ICmp, Kernel, MTy, Operand, Sem, V, Window};

/// Copies a step of a kernel's issue loop.
const UNROLL: i64 = 8;

/// The scoped VMEM a block kernel declares: it moves HBM to HBM.
const VMEM: u64 = 1 << 20;

fn any(array: Vec<i64>, elem: Elem) -> Operand {
    Operand {
        array,
        elem,
        block: vec![],
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    }
}

/// Blocks `at` (`[B]` i32, each in range) of `nr` pool rows of `pool`
/// (`u32 [R, 128]`) as `elem [B · nr · k, 128]` (`k` elements a word).
pub fn gather(f: &mut Func, pool: Val, at: Val, nr: i64, elem: Elem) -> Built<Val> {
    Ok(gather_planes(f, pool, at, elem, &[nr])?[0])
}

/// Several planes' blocks `at` in one kernel, each by its own rows a block
/// (fewer calls: each is a launch and a boundary XLA schedules around).
fn gather_planes(f: &mut Func, pool: Val, at: Val, elem: Elem, planes: &[i64]) -> Built<Vec<Val>> {
    let rows = f.dims(pool)[0];
    let b = f.dims(at)[0];
    let k = i64::from(32 / elem.bits());
    // one wait for every byte holds where each copy is whole (8k, 128)
    // tiles of the element (a 2-row bf16 block hangs it)
    let whole_tiles = planes.iter().all(|&nr| (nr * k) % (8 * k) == 0);
    let np = planes.len() as i64;
    let scratch = vec![MTy::Sem(vec![np])];
    let kern = Kernel {
        name: "pool_gather".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![b]],
        ins: vec![any(vec![rows, 128], Elem::U32)],
        outs: planes
            .iter()
            .map(|&nr| any(vec![b * nr * k, 128], elem))
            .collect(),
        // DMA only: no VMEM of its own, so none held from XLA's placement
        // (the compiler's default scope would be)
        vmem_limit: Some(VMEM),
        scratch,
    };
    let c = kern.build(|bd, blk| {
        let ids = blk.grid.prefetch[0];
        let src = bd.ref_bitcast(blk.ins[0], elem)?;
        let n = bd.i32(b);
        let one = bd.i32(1);
        let z = bd.i32(0);
        let sem_of = |bd: &mut Body, x: i64| -> Built<V> {
            let at = bd.i32(x);
            let s = bd.ref_slice(blk.scratch[0], &[at], &[1])?;
            bd.ref_squeeze(s)
        };
        let copy = |bd: &mut Body, j: V, wait: bool| -> Built<()> {
            let raw = bd.load(ids, &[j])?;
            for (x, &nr) in planes.iter().enumerate() {
                let er = nr * k;
                let z = bd.i32(0);
                let id = raw;
                // into the pool, as a gather clamps
                let top = bd.i32(rows / nr - 1);
                let id = bd.maxsi(id, z)?;
                let id = bd.minsi(id, top)?;
                let erv = bd.i32(er);
                let from = bd.muli(id, erv)?;
                let to = bd.muli(j, erv)?;
                let s = bd.ref_slice(src, &[from, z], &[er, 128])?;
                let d = bd.ref_slice(blk.outs[x], &[to, z], &[er, 128])?;
                let sem = sem_of(bd, x as i64)?;
                let go = |bd: &mut Body, s: V| -> Built<()> {
                    if wait {
                        bd.dma_wait(sem, s, d)
                    } else {
                        bd.dma_start(s, d, sem)
                    }
                };
                go(bd, s)?;
            }
            Ok(())
        };
        // The copies started UNROLL a loop step (the issue loop is scalar
        // work), then one wait for every byte of each output: each copy
        // signals its plane's semaphore with its bytes, and every block
        // lands (none dropped). Blocks of part of a tile are waited for one
        // by one. v6e, 4096 blocks of 16 KiB: 103 us a wait a copy, 54 us so.
        let groups = b / UNROLL;
        if groups > 0 {
            let g = bd.i32(groups);
            bd.for_range(z, g, one, |bd, i| {
                for u in 0..UNROLL {
                    let base = bd.i32(UNROLL);
                    let j = bd.muli(i, base)?;
                    let du = bd.i32(u);
                    let j = bd.addi(j, du)?;
                    copy(bd, j, false)?;
                }
                Ok(())
            })?;
        }
        let rest = bd.i32(groups * UNROLL);
        bd.for_range(rest, n, one, |bd, j| copy(bd, j, false))?;
        if whole_tiles {
            for (x, &out) in blk.outs[..planes.len()].iter().enumerate() {
                let sem = sem_of(bd, x as i64)?;
                bd.dma_wait(sem, out, out)?;
            }
            Ok(())
        } else {
            // copies of part of a tile are not counted as their bytes
            bd.for_range(z, n, one, |bd, j| copy(bd, j, true))
        }
    })?;
    c.call(f, &[at, pool])
}

/// `pool` with `ups` (`elem [B · nr · k, 128]`) landed as blocks `at` (`[B]`
/// i32; a block outside `[0, R / nr)` is dropped), in place.
pub fn scatter(f: &mut Func, pool: Val, at: Val, ups: Val, nr: i64) -> Built<Val> {
    let rows = f.dims(pool)[0];
    let b = f.dims(at)[0];
    let elem = f.elem(ups);
    let k = i64::from(32 / elem.bits());
    let er = nr * k;
    let blocks = rows / nr;
    let kern = Kernel {
        name: "pool_scatter".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![b]],
        ins: vec![
            any(vec![b * er, 128], elem),
            any(vec![rows, 128], Elem::U32),
        ],
        outs: vec![any(vec![rows, 128], Elem::U32)],
        // DMA only: no VMEM of its own, so none held from XLA's placement
        // (the compiler's default scope would be)
        vmem_limit: Some(VMEM),
        scratch: vec![MTy::Sem(vec![1])],
    };
    let c = kern.build(|bd, blk| {
        let ids = blk.grid.prefetch[0];
        let dst = bd.ref_bitcast(blk.outs[0], elem)?;
        let src = blk.ins[0];
        let sem = blk.scratch[0];
        let z = bd.i32(0);
        let s = bd.ref_slice(sem, &[z], &[1])?;
        let sem = bd.ref_squeeze(s)?;
        let n = bd.i32(b);
        let one = bd.i32(1);
        let copy = |bd: &mut Body, j: V, wait: bool| -> Built<()> {
            let id = bd.load(ids, &[j])?;
            let z = bd.i32(0);
            let top = bd.i32(blocks);
            let ge = bd.cmpi(ICmp::Sge, id, z)?;
            let lt = bd.cmpi(ICmp::Slt, id, top)?;
            let ok = bd.andi(ge, lt)?;
            bd.when(ok, |bd| {
                let erv = bd.i32(er);
                let from = bd.muli(j, erv)?;
                let to = bd.muli(id, erv)?;
                let z = bd.i32(0);
                let s = bd.ref_slice(src, &[from, z], &[er, 128])?;
                let d = bd.ref_slice(dst, &[to, z], &[er, 128])?;
                if wait {
                    bd.dma_wait(sem, s, d)
                } else {
                    bd.dma_start(s, d, sem)
                }
            })
        };
        // starts unrolled (see `gather`); a dropped block is not copied, so
        // each copy is waited for by itself
        let groups = b / UNROLL;
        if groups > 0 {
            let g = bd.i32(groups);
            bd.for_range(z, g, one, |bd, i| {
                for u in 0..UNROLL {
                    let base = bd.i32(UNROLL);
                    let j = bd.muli(i, base)?;
                    let du = bd.i32(u);
                    let j = bd.addi(j, du)?;
                    copy(bd, j, false)?;
                }
                Ok(())
            })?;
        }
        let rest = bd.i32(groups * UNROLL);
        bd.for_range(rest, n, one, |bd, j| copy(bd, j, false))?;
        bd.for_range(z, n, one, |bd, j| copy(bd, j, true))
    })?;
    Ok(c.call_aliased(f, &[at, ups, pool], &[(0, 2)])?[0])
}

/// Tokens of a reordered pool (`crate::hlo::PoolOrder`) landed one DMA a
/// token: token `i`'s `rpt` pool rows (`ups` `u32 [N · rpt, 1, 128]`, a row
/// apiece) to pool rows `at[i] + j · tokens` (`at[i]`: its page block's
/// start plus its slot; one past the pool, or negative, drops it).
pub fn scatter_tokens(
    f: &mut Func,
    pool: Val,
    at: Val,
    ups: Val,
    order: crate::hlo::PoolOrder,
) -> Built<Val> {
    let rows = f.dims(pool)[0];
    let n = f.dims(at)[0];
    let (t, rpt) = (order.tokens, order.rpt);
    let kern = Kernel {
        name: "pool_scatter_tokens".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![n]],
        ins: vec![
            any(vec![n * rpt, 1, 128], Elem::U32),
            any(vec![rows, 128], Elem::U32),
        ],
        outs: vec![any(vec![rows, 128], Elem::U32)],
        vmem_limit: Some(VMEM),
        scratch: vec![MTy::Sem(vec![1])],
    };
    let c = kern.build(|bd, blk| {
        let ids = blk.grid.prefetch[0];
        // the pool as [R / T, T, 128]: a token's rows one step of the
        // leading axis apart
        let dst = bd.ref_reshape(blk.outs[0], &[rows / t, t, 128])?;
        let src = blk.ins[0];
        let sem = blk.scratch[0];
        let z = bd.i32(0);
        let s = bd.ref_slice(sem, &[z], &[1])?;
        let sem = bd.ref_squeeze(s)?;
        let nv = bd.i32(n);
        let one = bd.i32(1);
        let copy = |bd: &mut Body, i: V, wait: bool| -> Built<()> {
            let id = bd.load(ids, &[i])?;
            let z = bd.i32(0);
            let top = bd.i32(rows);
            let ge = bd.cmpi(ICmp::Sge, id, z)?;
            let lt = bd.cmpi(ICmp::Slt, id, top)?;
            let ok = bd.andi(ge, lt)?;
            bd.when(ok, |bd| {
                let tv = bd.i32(t);
                let blk_row = bd.divsi(id, tv)?;
                let slot = bd.remsi(id, tv)?;
                let rv = bd.i32(rpt);
                let from = bd.muli(i, rv)?;
                let z = bd.i32(0);
                let s = bd.ref_slice(src, &[from, z, z], &[rpt, 1, 128])?;
                let d = bd.ref_slice(dst, &[blk_row, slot, z], &[rpt, 1, 128])?;
                if wait {
                    bd.dma_wait(sem, s, d)
                } else {
                    bd.dma_start(s, d, sem)
                }
            })
        };
        bd.for_range(z, nv, one, |bd, i| copy(bd, i, false))?;
        bd.for_range(z, nv, one, |bd, i| copy(bd, i, true))
    })?;
    Ok(c.call_aliased(f, &[at, ups, pool], &[(0, 2)])?[0])
}
