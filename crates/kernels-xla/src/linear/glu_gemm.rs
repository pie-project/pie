//! A dense GeGLU's packed gate-up projection and its GLU as one Mosaic
//! kernel: XLA's dot runs a v6e's MXU at ~82% on Gemma-4-31B's 2304-row
//! `[5376, 43008]` gate-up, and the product lands for a pass of its own;
//! the kernel takes the fire's rows whole against `bn`-wide blocks of the
//! gate and the up halves (94%, measured in Pallas) and lands only the GLU.
//!
//! Operands: `x` bf16 `[m, k]`, the rows; `w` bf16 `[k, 2·i]`, the gemm
//! weight as it lands (`[K, N]`, the gate's columns then the up's), passed
//! twice, one block of each half a step. Result: bf16 `[m, i]`, `gelu(g) · u`
//! with each half's product rounded to bf16 first, as the primitives land
//! `packed` before their GLU reads it.

use crate::error::Error;
use crate::hlo::{Elem, Func, Val};
use crate::linear::moe::ExpertGlu;
use crate::mosaic::{Body, Grid, Kernel, Operand as Block, Sem, Window};

/// The scoped VMEM a kernel may take (a v6e holds 128 MiB).
const VMEM_CAP: i64 = 100 << 20;

/// `gelu(x · w_gate) · (x · w_up)` for the first `rows` bf16 rows of `x`
/// `[_, k]` and the landed gate-up weight `w` `[k, 2·intermediate]`; `None`
/// where no block fits.
pub fn geglu(f: &mut Func, x: Val, w: Val, rows: u32, intermediate: i64) -> Result<Option<Val>, Error> {
    let x = super::gemm::head_rows(f, x, rows)?;
    let (m, k) = (f.dims(x)[0], f.dims(x)[1]);
    let i = intermediate;
    if f.elem(x) != Elem::Bf16 || f.elem(w) != Elem::Bf16 || f.dims(w) != [k, 2 * i] || k % 128 != 0 {
        return Ok(None);
    }
    let Some(bn) = [512, 256].into_iter().find(|bn| i % bn == 0) else {
        return Ok(None);
    };
    // Two buffers of the rows, of each half's block and of the result, and
    // the f32 products and GLU of a step.
    let need = |bm: i64| 4 * bm * k + 8 * k * bn + 4 * bm * bn + 12 * bm * bn;
    let Some(bm) = [1, 2, 3, 4]
        .into_iter()
        .filter(|d| m % d == 0)
        .map(|d| m / d)
        .find(|&bm| bm % 16 == 0 && need(bm) <= VMEM_CAP)
    else {
        return Ok(None);
    };
    let off = i / bn;
    let half = |at: i64| Block {
        array: vec![k, 2 * i],
        elem: Elem::Bf16,
        block: vec![k, bn],
        window: Window::Blocked,
        index: Box::new(move |b: &mut Body, g: &Grid| {
            let o = b.i32(at);
            Ok(vec![b.i32(0), b.addi(g.ids[1], o)?])
        }),
    };
    let ins = vec![
        Block {
            array: vec![m, k],
            elem: Elem::Bf16,
            block: vec![bm, k],
            window: Window::Blocked,
            index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
        },
        half(0),
        half(off),
    ];
    let outs = vec![Block {
        array: vec![m, i],
        elem: Elem::Bf16,
        block: vec![bm, bn],
        window: Window::Blocked,
        index: Box::new(|_: &mut Body, g: &Grid| Ok(vec![g.ids[0], g.ids[1]])),
    }];
    let kernel = Kernel {
        name: "glu_gemm".into(),
        grid: vec![m / bm, i / bn],
        semantics: vec![Sem::Parallel, Sem::Parallel],
        prefetch: vec![],
        ins,
        outs,
        vmem_limit: Some(u64::try_from(need(bm) + (8 << 20)).unwrap_or(0)),
        scratch: vec![],
    };
    let c = kernel
        .build(|b, blk| {
            let x = b.vload_all(blk.ins[0])?;
            let product = |b: &mut Body, w: crate::mosaic::V| -> crate::hlo::Built<crate::mosaic::V> {
                let w = b.vload_all(w)?;
                let acc = b.splat_f(&[bm, bn], Elem::F32, 0.0)?;
                let y = b.matmul(x, w, acc, false)?;
                let y = b.convert_f(y, Elem::Bf16)?;
                b.convert_f(y, Elem::F32)
            };
            let g = product(b, blk.ins[1])?;
            let u = product(b, blk.ins[2])?;
            let a = super::moe_gmm::glu_of(b, ExpertGlu::GegluTanh, g, u, &[bm, bn])?;
            let a = b.convert_f(a, Elem::Bf16)?;
            b.vstore_all(a, blk.outs[0])
        })
        .map_err(Error::from)?;
    Ok(Some(c.call(f, &[x, w, w]).map_err(Error::from)?[0]))
}
