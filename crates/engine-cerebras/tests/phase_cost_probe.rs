//! What a phase costs on the simulator: a near-empty kernel over buffers of
//! growing size, on one PE and on several, timed by
//! `PIE_CEREBRAS_TRACE_PHASES` (run with `--nocapture`). The assertion only
//! checks the results; the timings are the point.

use engine_cerebras::bench::{Bench, assert_close};
use kernels_cerebras::elemwise::norm;

fn residual(rows: u32, width: u32) {
    let n = (rows * width) as usize;
    let xs: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.25).collect();
    let mut b = Bench::new();
    let x = b.f32(rows, width, &xs);
    let y = b.f32(rows, width, &ys);
    if !b.run(|ctx| norm::residual_add(ctx, x, y)).unwrap() {
        return;
    }
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| a + b).collect();
    assert_close(&b.read_f32(y), &want, 0.0, 0.0);
}

#[test]
fn a_phase_costs_its_start_up_and_its_words() {
    // SAFETY: tests in this file run alone; the flag only adds stderr lines.
    unsafe { std::env::set_var("PIE_CEREBRAS_TRACE_PHASES", "1") };
    for (rows, width) in [(1u32, 256u32), (1, 4096), (8, 1024), (16, 2048)] {
        eprintln!("residual_add {rows}x{width}:");
        residual(rows, width);
    }
}
