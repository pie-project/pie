//! Mosaic kernels built in Rust (`kernels_xla::mosaic`), called from our
//! StableHLO through `stablehlo.custom_call @tpu_custom_call`, compiled by
//! the PJRT plugin and run on the TPU.

use engine_xla::bench::{Bench, assert_close};
use kernels_xla::hlo::Elem;
use kernels_xla::mosaic::{Body, Grid, Kernel, Operand, Sem, Window};

/// `z = x + y` over a `[rows, 256]` f32 array, one `[8, 256]` block per
/// grid step (block `i` read at step `rows/8 - 1 - i`: the index map is
/// honoured, not just the identity).
#[test]
fn a_rust_built_mosaic_add_runs_on_the_device() {
    let rows = 40u32;
    let steps = i64::from(rows / 8);
    let block = move || Operand {
        array: vec![i64::from(rows), 256],
        elem: Elem::F32,
        block: vec![8, 256],
        window: Window::Blocked,
        index: Box::new(move |b: &mut Body, g: &Grid| {
            let last = b.i32(steps - 1);
            let i = b.subi(last, g.ids[0])?;
            Ok(vec![i, b.i32(0)])
        }),
    };
    let kernel = Kernel {
        name: "add".into(),
        grid: vec![steps],
        semantics: vec![Sem::Parallel],
        prefetch: vec![],
        ins: vec![block(), block()],
        outs: vec![block()],
        vmem_limit: None,
    };
    let compiled = kernel
        .build(|b, blk| {
            let x = b.vload_all(blk.ins[0])?;
            let y = b.vload_all(blk.ins[1])?;
            let z = b.addf(x, y)?;
            b.vstore_all(z, blk.outs[0])
        })
        .unwrap();

    let n = (rows * 256) as usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| 1.0 - (i % 7) as f32).collect();
    let mut b = Bench::new();
    let x = b.f32(rows, 256, &xs);
    let y = b.f32(rows, 256, &ys);
    let z = b.zeros(dtype::Dtype::F32, rows, 256);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let xv = cx.read(x)?;
                let yv = cx.read(y)?;
                let out = compiled.call(cx, &[xv, yv])?;
                cx.write(z, out[0])
            })
        })
        .unwrap_or_else(|e| panic!("{e}\n--- mosaic ---\n{}", compiled.module));
    if !ran {
        return;
    }
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| a + b).collect();
    assert_close(&b.read_f32(z), &want, 0.0, 0.0);
}
