//! The bench's vertical slice: a phase emitted by hand through the kernel
//! sink is compiled, run on the simulator, and its bf16 write lands rounded.

use engine_cerebras::bench::{Bench, assert_close, round_bf16};

#[test]
fn a_scaled_sum_lands_rounded_to_bf16() {
    let xs: Vec<f32> = (0..12).map(|i| i as f32 * 0.37 - 2.0).collect();
    let ys: Vec<f32> = (0..12).map(|i| (i as f32).sin()).collect();
    let mut b = Bench::new();
    let x = b.bf16(3, 4, &xs);
    let y = b.bf16(3, 4, &ys);
    let z = b.zeros(dtype::Dtype::Bf16, 3, 4);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let xb = cx.read(x)?;
                let yb = cx.read(y)?;
                let zb = cx.write(z)?;
                cx.for_range("i", 12, |blk| {
                    blk.line(format!(
                        "{}[i] = 1.5 * {}[i] + {}[i];",
                        zb.name, xb.name, yb.name
                    ));
                });
                Ok(())
            })
        })
        .unwrap();
    if !ran {
        return;
    }
    let want: Vec<f32> = b
        .read_f32(x)
        .iter()
        .zip(b.read_f32(y))
        .map(|(p, q)| round_bf16(1.5 * p + q))
        .collect();
    assert_close(&b.read_f32(z), &want, 0.0, 0.0);
}
