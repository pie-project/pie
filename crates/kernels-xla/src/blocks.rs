//! Whole blocks of the chunk pool moved by DMA, as Mosaic kernels: what a
//! pool view's block gather or scatter is on a TPU for elements narrower
//! than a word. XLA reinterprets `u32` as a narrower element through a minor
//! axis the tiled layout pads (shifts and masks over every element read); a
//! DMA through the pool reference bitcast to the element
//! (`tpu.memref_bitcast`, the packed rows order) moves the bytes as they lie,
//! HBM to HBM, one copy a block in flight together.
//!
//! `PIE_XLA_POOL_DMA=0` keeps blocks on the XLA path (off a TPU).

use crate::hlo::{Built, Elem, Func, Val};
use crate::mosaic::{Body, Grid, ICmp, Kernel, MTy, Operand, Sem, V, Window};

/// Whether pool blocks go by DMA.
#[must_use]
pub fn on() -> bool {
    !std::env::var("PIE_XLA_POOL_DMA").is_ok_and(|v| v == "0")
}

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
    let rows = f.dims(pool)[0];
    let b = f.dims(at)[0];
    let k = i64::from(32 / elem.bits());
    let er = nr * k;
    let kern = Kernel {
        name: "pool_gather".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![b]],
        ins: vec![any(vec![rows, 128], Elem::U32)],
        outs: vec![any(vec![b * er, 128], elem)],
        vmem_limit: None,
        scratch: vec![MTy::Sem(vec![1])],
    };
    let c = kern.build(|bd, blk| {
        let ids = blk.grid.prefetch[0];
        let src = bd.ref_bitcast(blk.ins[0], elem)?;
        let out = blk.outs[0];
        let sem = blk.scratch[0];
        let z = bd.i32(0);
        let s = bd.ref_slice(sem, &[z], &[1])?;
        let sem = bd.ref_squeeze(s)?;
        let n = bd.i32(b);
        let one = bd.i32(1);
        let copy = |bd: &mut Body, j: V, wait: bool| -> Built<()> {
            let id = bd.load(ids, &[j])?;
            let erv = bd.i32(er);
            let from = bd.muli(id, erv)?;
            let to = bd.muli(j, erv)?;
            let z = bd.i32(0);
            let s = bd.ref_slice(src, &[from, z], &[er, 128])?;
            let d = bd.ref_slice(out, &[to, z], &[er, 128])?;
            if wait {
                bd.dma_wait(sem, s, d)
            } else {
                bd.dma_start(s, d, sem)
            }
        };
        bd.for_range(z, n, one, |bd, j| copy(bd, j, false))?;
        bd.for_range(z, n, one, |bd, j| copy(bd, j, true))
    })?;
    Ok(c.call(f, &[at, pool])?[0])
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
        vmem_limit: None,
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
        bd.for_range(z, n, one, |bd, j| copy(bd, j, false))?;
        bd.for_range(z, n, one, |bd, j| copy(bd, j, true))
    })?;
    Ok(c.call_aliased(f, &[at, ups, pool], &[(0, 2)])?[0])
}
