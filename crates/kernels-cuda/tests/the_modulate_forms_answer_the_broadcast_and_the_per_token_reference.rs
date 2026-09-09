#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::modulate;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 37;
const WIDTH: usize = 128;
const LANES: usize = 4;

fn rows_of(lanes: &Option<Vec<i32>>) -> Vec<usize> {
    (0..ROWS)
        .map(|r| lanes.as_ref().map_or(r, |map| map[r] as usize))
        .collect()
}

fn close(got: f32, want: f32, what: &str, at: usize) {
    assert!(
        (got - want).abs() <= 1e-6 + want.abs() / 256.0,
        "{what}[{at}]: {got} against {want}"
    );
}

fn check(lane_of_row: Option<Vec<i32>>) {
    let m_rows = if lane_of_row.is_some() { LANES } else { ROWS };
    let mut lcg = Lcg::seeded(0xd17);
    let (x_raw, x) = lcg.row(ROWS * WIDTH);
    let (y_raw, y) = lcg.row(ROWS * WIDTH);
    let (r_raw, r) = lcg.row(ROWS * WIDTH);
    let (m_raw, m) = lcg.row(m_rows * 2 * WIDTH);
    let (g_raw, g) = lcg.row(m_rows * WIDTH);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let y_at = gpu.up(&y_raw);
    let r_at = gpu.up(&r_raw);
    let m_at = gpu.up(&m_raw);
    let g_at = gpu.up(&g_raw);
    let map_at = lane_of_row.as_ref().map(|map| gpu.up(map));
    let out = [(); 4].map(|()| gpu.zeros(ROWS * WIDTH * 2));
    let ctx = gpu.ctx();

    let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);
    let map = map_at.map(|at| Tensor::new(at, ROWS as u32, 1, Dtype::I32));
    let m_wide = Tensor::new(m_at, m_rows as u32, 2 * WIDTH as u32, Dtype::Bf16);
    let m_tight = Tensor::new(g_at, m_rows as u32, WIDTH as u32, Dtype::Bf16);

    modulate::scale_shift(&ctx, rect(x_at), m_wide, map, &mut rect(out[0])).expect("fires");
    modulate::scale(&ctx, rect(x_at), m_tight, map, &mut rect(out[1])).expect("fires");
    modulate::tanh_gate(&ctx, rect(x_at), m_tight, map, &mut rect(out[2])).expect("fires");
    modulate::gated_residual_add(
        &ctx,
        rect(r_at),
        m_tight,
        rect(y_at),
        map,
        &mut rect(out[3]),
    )
    .expect("fires");
    gpu.sync();

    let got: Vec<Vec<u16>> = out.iter().map(|&at| gpu.down(at, ROWS * WIDTH)).collect();
    let source = rows_of(&lane_of_row);
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let at = row * WIDTH + col;
            let s = m[source[row] * 2 * WIDTH + col];
            let b = m[source[row] * 2 * WIDTH + WIDTH + col];
            let gv = g[source[row] * WIDTH + col];
            close(
                from_bf16(got[0][at]),
                x[at].mul_add(1.0 + s, b),
                "scale_shift",
                at,
            );
            close(from_bf16(got[1][at]), x[at] * (1.0 + gv), "scale", at);
            close(from_bf16(got[2][at]), gv.tanh() * x[at], "tanh_gate", at);
            close(
                from_bf16(got[3][at]),
                gv.mul_add(y[at], r[at]),
                "gated_residual_add",
                at,
            );
        }
    }
}

#[test]
fn the_modulate_forms_answer_the_broadcast_and_the_per_token_reference_every_case() {
    the_forms_answer_the_reference_with_m_read_per_lane();
    the_forms_answer_the_reference_with_m_read_per_token();
    a_modulation_may_write_the_rectangle_it_read();
}

fn the_forms_answer_the_reference_with_m_read_per_lane() {
    let map: Vec<i32> = (0..ROWS).map(|r| (r % LANES) as i32).collect();
    check(Some(map));
}

fn the_forms_answer_the_reference_with_m_read_per_token() {
    check(None);
}

fn a_modulation_may_write_the_rectangle_it_read() {
    let mut lcg = Lcg::seeded(0xa11a5);
    let (x_raw, x) = lcg.row(ROWS * WIDTH);
    let (m_raw, m) = lcg.row(ROWS * 2 * WIDTH);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let m_at = gpu.up(&m_raw);
    let ctx = gpu.ctx();
    let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);

    modulate::scale_shift(
        &ctx,
        rect(x_at),
        Tensor::new(m_at, ROWS as u32, 2 * WIDTH as u32, Dtype::Bf16),
        None,
        &mut rect(x_at),
    )
    .expect("fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(x_at, ROWS * WIDTH);
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let at = row * WIDTH + col;
            let s = m[row * 2 * WIDTH + col];
            let b = m[row * 2 * WIDTH + WIDTH + col];
            assert_eq!(
                got[at],
                to_bf16(x[at].mul_add(1.0 + s, b)),
                "in place at {at}"
            );
        }
    }
}
