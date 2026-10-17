#![cfg(feature = "cuda")]

// The tcgen05 prefill over key and value planes of their own widths (an
// expanded latent attention's: 192-wide keys, 128-wide values read inside a
// 256-wide plane at its own head stride), its pages a pool's: each row's
// softmax over its request's causal keys, against an f32 walk of the same
// bf16 inputs.

use crate::common::{Gpu, Lcg, from_bf16};
use kernels_cuda::attn::tc5_fmha::{Planes, planes_take, prefill_planes};

const PAGE: usize = 16;

struct Fixture {
    planes: Planes,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    lens: Vec<(usize, usize)>,
}

/// Requests of `(rows, keys)`, the rows the last of the keys; their pages
/// laid out in reverse so no run is the identity.
/// Zero planes, and no host copies, where not `fill` (a timing's).
fn fixture(
    gpu: &mut Gpu,
    lens: &[(usize, usize)],
    heads: usize,
    dqk: usize,
    dv: usize,
    fill: bool,
) -> Fixture {
    let mut lcg = Lcg::seeded(0x5f1 ^ (heads * dqk) as u64);
    let rows: usize = lens.iter().map(|l| l.0).sum();
    let pages: Vec<usize> = lens.iter().map(|l| l.1.div_ceil(PAGE)).collect();
    let total_pages: usize = pages.iter().sum();
    let cells = total_pages * PAGE;
    let vplane = 2 * dv;
    let mut plane = |gpu: &mut Gpu, count: usize| {
        if fill {
            let (raw, exact) = lcg.row(count);
            (gpu.up(&raw), exact)
        } else {
            (gpu.zeros(count * 2), Vec::new())
        }
    };
    let (q_at, q) = plane(gpu, rows * heads * dqk);
    let (k_at, k) = plane(gpu, cells * heads * dqk);
    let (v_at, v) = plane(gpu, cells * heads * vplane);
    let mut qo: Vec<i32> = vec![0];
    let mut kv: Vec<i32> = vec![0];
    let mut last: Vec<i32> = Vec::new();
    for (i, &(r, n)) in lens.iter().enumerate() {
        qo.push(qo.last().unwrap() + r as i32);
        kv.push(kv.last().unwrap() + pages[i] as i32);
        last.push((n - (pages[i] - 1) * PAGE) as i32);
    }
    let indices: Vec<i32> = (0..total_pages as i32).rev().collect();
    let live = [0i32, lens.len() as i32];
    let planes = Planes {
        q: q_at,
        q_stride: (heads * dqk) as u32,
        o: gpu.zeros(rows * heads * dv * 2),
        o_stride: (heads * dv) as u32,
        k: k_at,
        k_head: dqk as u32,
        k_cell: (heads * dqk) as u32,
        v: v_at + (vplane - dv) as u64 * 2,
        v_head: vplane as u32,
        v_cell: (heads * vplane) as u32,
        cells: cells as u32,
        heads: heads as u32,
        dqk: dqk as u32,
        dv: dv as u32,
        qo_indptr: gpu.up(&qo),
        kv_indptr: gpu.up(&kv),
        kv_indices: gpu.up(&indices),
        last_page_len: gpu.up(&last),
        page: PAGE as u32,
        live: gpu.up(&live),
        requests: lens.len() as u32,
        rows: rows as u32,
        causal: true,
        sm_scale: 1.0 / (dqk as f32).sqrt(),
    };
    Fixture {
        planes,
        q,
        k,
        v,
        lens: lens.to_vec(),
    }
}

fn reads(lens: &[(usize, usize)], heads: usize, dqk: usize, dv: usize) {
    let mut gpu = Gpu::open();
    if !planes_take(dqk as u32, dv as u32) {
        eprintln!("no tcgen05 here: skipped");
        return;
    }
    let f = fixture(&mut gpu, lens, heads, dqk, dv, true);
    let ctx = gpu.ctx();
    prefill_planes(&ctx, "test.prefill_planes", &f.planes).expect("the prefill fires");
    gpu.sync();
    let rows: usize = lens.iter().map(|l| l.0).sum();
    let got: Vec<u16> = gpu.down(f.planes.o, rows * heads * dv);
    let vplane = 2 * dv;
    let total_pages: usize = lens.iter().map(|l| l.1.div_ceil(PAGE)).sum();
    // Request i's page j sits at compact page `total - 1 - (its first + j)`.
    let mut page0 = 0usize;
    let mut row0 = 0usize;
    let mut worst = 0f32;
    for &(r, n) in &f.lens {
        let cell = |t: usize| (total_pages - 1 - (page0 + t / PAGE)) * PAGE + t % PAGE;
        for i in 0..r {
            let pos = n - r + i;
            for h in 0..heads {
                let qrow = &f.q[((row0 + i) * heads + h) * dqk..][..dqk];
                let scores: Vec<f32> = (0..=pos)
                    .map(|t| {
                        let krow = &f.k[(cell(t) * heads + h) * dqk..][..dqk];
                        qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * f.planes.sm_scale
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..dv {
                    let want: f32 = (0..=pos)
                        .map(|t| e[t] * f.v[(cell(t) * heads + h) * vplane + (vplane - dv) + d])
                        .sum::<f32>()
                        / sum;
                    let g = from_bf16(got[((row0 + i) * heads + h) * dv + d]);
                    let err = (g - want).abs();
                    assert!(
                        err <= 0.02 + 0.02 * want.abs(),
                        "{lens:?}: row {} head {h} column {d}: {g} against {want}",
                        row0 + i
                    );
                    worst = worst.max(err);
                }
            }
        }
        page0 += n.div_ceil(PAGE);
        row0 += r;
    }
    eprintln!("{lens:?} x {heads} heads: worst {worst}");
}

#[test]
fn the_tcgen05_prefill_reads_keys_and_values_of_their_own_widths() {
    // A whole prompt, one past a tile, a prompt over a cached prefix, a
    // short one.
    reads(&[(300, 300)], 2, 192, 128);
    reads(&[(129, 129), (40, 200), (7, 7)], 3, 192, 128);
    // GLM-5.3's expanded heads: 256 over 256, 64-key tiles.
    reads(&[(129, 129), (40, 200), (7, 7)], 2, 256, 256);
}
