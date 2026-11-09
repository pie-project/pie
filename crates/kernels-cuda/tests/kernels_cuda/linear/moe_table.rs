//! Every routed select through an expert table: the contiguous bank with no
//! table, with its own (degenerate) table, and with its experts scattered
//! across a pool in a shuffled order with garbage between (the bank itself
//! garbled after), all land the same bits; a table with two experts
//! swapped does not.

use crate::common::expert_pool::{Plane, degenerate, garble, scatter, swapped};
use crate::common::linear_ref::{t, tab};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, I32};
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::moe::{self, ExpertTable};

/// `fire` under each table over `planes` of `experts` experts, the result
/// `y_bytes` at `y`: the three tables land one result, bit for bit, and a
/// table with experts 0 and 1 swapped (each routed) lands another. Each
/// result is read after `y` is set to a sentinel. `what` names the case.
pub fn same_bits(
    g: &mut Gpu,
    what: &str,
    experts: usize,
    planes: &[Plane],
    (y, y_bytes): (u64, usize),
    mut fire: impl FnMut(ExpertTable),
) {
    let mut run = |g: &mut Gpu, table: ExpertTable| -> Vec<u8> {
        g.fill(y, 0x7f, y_bytes);
        fire(table);
        g.sync();
        g.down::<u8>(y, y_bytes)
    };
    let want = run(g, ExpertTable::RESIDENT);
    let own = degenerate(g, experts, planes);
    let got = run(g, ExpertTable::located(own, (0, 0)));
    assert!(
        got == want,
        "{what}: the contiguous bank's own table moved a bit"
    );
    let pool = scatter(g, experts, planes, 0x51ce ^ experts as u64);
    let wrong = swapped(g, &pool.entries, planes.len(), 0, 1);
    garble(g, experts, planes);
    let got = run(g, ExpertTable::located(pool.table, pool.window()));
    assert!(got == want, "{what}: the scattered table moved a bit");
    let got = run(g, ExpertTable::located(wrong, pool.window()));
    assert!(
        got != want,
        "{what}: a table with two experts swapped landed the same bits"
    );
}

/// Routes that name every expert, experts 0 and 1 among the first.
pub fn routes(tokens: usize, top: usize, experts: usize) -> Vec<i32> {
    tab(tokens * top, |r| {
        ((r / top * 3 + r % top * 5 + r / 7) % experts) as i32
    })
}

/// A bf16 bank `[experts, n, k]` and activations `[rows, k]`, random.
fn bf16_bank(g: &mut Gpu, lcg: &mut Lcg, experts: usize, n: usize, k: usize) -> u64 {
    let (raw, _) = lcg.row(experts * n * k);
    g.up(&raw)
}

fn ctx_of(g: &Gpu) -> Ctx {
    g.ctx()
}

#[test]
fn a_bf16_select_lands_the_same_bits_through_any_table() {
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7ab1e);
    // Decode-sized: the per-route GEMV.
    for (experts, tokens, top, n, k, by_token) in [
        (8usize, 3usize, 2usize, 64usize, 256usize, true),
        (16, 5, 4, 96, 512, false),
    ] {
        let bank = bf16_bank(&mut g, &mut lcg, experts, n, k);
        let rows = if by_token { tokens } else { tokens * top };
        let (xr, _) = lcg.row(rows * k);
        let x = t(g.up(&xr), rows, k, Bf16);
        let r = t(g.up(&routes(tokens, top, experts)), tokens, top, I32);
        let y = g.zeros(tokens * top * n * 2);
        let ctx = ctx_of(&g);
        let stride = (n * k * 2) as u64;
        let what = format!("bf16 select {experts}x{tokens}x{top} {n}x{k}");
        same_bits(
            &mut g,
            &what,
            experts,
            &[Plane::of(bank, stride)],
            (y, tokens * top * n * 2),
            |table| {
                let mut yt = t(y, tokens * top, n, Bf16);
                moe::matmul_select(&ctx, x, t(bank, experts, n * k, Bf16), r, &mut yt, table)
                    .expect("the select fires");
            },
        );
    }
}
