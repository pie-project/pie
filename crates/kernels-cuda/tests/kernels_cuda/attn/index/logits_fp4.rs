//! The FP4 indexer's logits on tcgen05's block-scaled MMA
//! (`kernel.sdpa.index_fp4_tc5`) against the path it replaces, the keys
//! widened to bf16 and scored there: the same scores to the f32 rounding of
//! their sums' order, the same picks but for ties.

use crate::common::attn_ref::{col, offsets, owners, pool, ten, tuned};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, I32, U8};
use kernels_cuda::attn::index::{self, FP4_ROW};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const DIM: usize = 128;

/// An indexer's fire: `heads` heads ranking `top_k` blocks of `ratio`
/// positions over requests of `(rows, positions)` on `page`-position pages.
#[derive(Clone, Copy, Debug)]
struct Case<'a> {
    heads: usize,
    lens: &'a [(usize, usize)],
    page: usize,
    ratio: usize,
    top_k: usize,
}

/// A case's device state: the FP4 key pool (a key a row, pages back to
/// front), the queries and weights.
struct Fire {
    keys: KvPool,
    q: RaggedTensor,
    w: Tensor,
    rows: usize,
    stride: usize,
}

/// `x`'s 32-blocks scaled by powers of two from 2^-4 to 2^4 (`salt` picks
/// the run), so the blocks' exponents differ and each lands in its place.
fn spread(x: &[f32], salt: usize) -> Vec<u16> {
    let at = |i: usize| 2f32.powi(((i / 32 * 5 + salt) % 9) as i32 - 4);
    x.iter()
        .enumerate()
        .map(|(i, v)| to_bf16(v * at(i)))
        .collect()
}

fn build(gpu: &mut Gpu, c: Case) -> Fire {
    let per = c.page / c.ratio;
    let pages: Vec<usize> = c.lens.iter().map(|l| l.1.div_ceil(c.page)).collect();
    let total: usize = pages.iter().sum();
    let ids: Vec<i32> = (0..total as i32).rev().collect();
    let starts = offsets(pages.iter().copied());
    let width = FP4_ROW as usize / c.ratio;
    let plane = ten(
        gpu.zeros(total * per * FP4_ROW as usize),
        total * c.page,
        width,
        U8,
    );
    let last: Vec<i32> = c
        .lens
        .iter()
        .map(|l| ((l.1 - 1) % c.page + 1) as i32)
        .collect();
    let keys = KvPool {
        last_page_lens: col(gpu, &last, I32),
        ..pool(plane, col(gpu, &ids, I32), col(gpu, &starts, I32), c.page)
    };
    // A block's key at its last position, quantized as V4.1 stores it.
    let mut lcg = Lcg::seeded(0xf4 + c.heads as u64 * 31 + c.ratio as u64);
    let (mut src, mut bpos, mut breq) = (vec![], vec![], vec![]);
    for (r, l) in c.lens.iter().enumerate() {
        for j in 0..l.1 / c.ratio {
            src.extend(spread(&lcg.row(DIM).1, j));
            bpos.push((j * c.ratio + c.ratio - 1) as i32);
            breq.push(r as i32);
        }
    }
    let entries = ten(gpu.up(&src), bpos.len(), DIM, Bf16);
    let (bpos, breq) = (col(gpu, &bpos, I32), col(gpu, &breq, I32));
    index::store_fp4(&gpu.ctx(), entries, bpos, breq, &keys).expect("the FP4 keys land");
    let rows = c.lens.iter().map(|l| l.0).sum::<usize>();
    let q = spread(&lcg.row(rows * c.heads * DIM).1, 3);
    let w: Vec<u16> = (0..rows * c.heads)
        .map(|_| to_bf16(lcg.unit().abs() + 0.25))
        .collect();
    let qo = col(gpu, &offsets(c.lens.iter().map(|l| l.0)), I32);
    Fire {
        keys,
        q: RaggedTensor {
            data: ten(gpu.up(&q), rows, c.heads * DIM, Bf16),
            indptr: qo,
        },
        w: ten(gpu.up(&w), rows, c.heads, Bf16),
        rows,
        stride: (total * c.page).div_ceil(c.ratio),
    }
}

/// The ranking under `tuning`: its picks, and the scores it landed.
fn rank(gpu: &mut Gpu, c: Case, f: &Fire, tuning: Tuning) -> (Vec<i32>, Vec<f32>) {
    let sel_at = gpu.zeros(f.rows * c.top_k * 4);
    let ctx = gpu.ctx().with_tuning(tuning);
    let scores = ctx
        .scratch_shared("test", "attn.index_topk_scores", f.rows * f.stride * 4)
        .expect("the scores scratch") as u64;
    // NaN where the scorer writes nothing.
    gpu.fill(scores, 0xff, f.rows * f.stride * 4);
    let (h, kk, r) = (c.heads as u32, c.top_k as u32, c.ratio as u32);
    let sel = &mut ten(sel_at, f.rows, c.top_k, I32);
    let fired = index::topk(&ctx, f.q, Some(f.w), &f.keys, h, DIM as u32, kk, r, sel);
    fired.expect("the ranking fires");
    gpu.sync();
    (
        gpu.down(sel_at, f.rows * c.top_k),
        gpu.down(scores, f.rows * f.stride),
    )
}

/// B200's index decisions, the FP4 scorer on or off.
fn b200(fp4: bool) -> Tuning {
    tuned(|t| {
        (t.index_mma, t.index_radix, t.index_tc5, t.index_fp4_tc5) = (true, true, true, fp4);
    })
}

#[test]
fn fp4_logits_score_as_the_widened_keys_do() {
    #[rustfmt::skip]
    let cases = [
        // V4-Flash's indexer at decode: 64 heads, ratio 4, a row a request.
        Case { heads: 64, lens: &[(1, 8000), (1, 3000), (1, 2100), (1, 90)], page: 256, ratio: 4, top_k: 512 },
        // V4.1-Flash's: 32 heads at ratio 2 and 1, a request's two rows (a draft's).
        Case { heads: 32, lens: &[(1, 5000), (2, 4100)], page: 128, ratio: 2, top_k: 512 },
        Case { heads: 32, lens: &[(1, 3000), (3, 700)], page: 64, ratio: 1, top_k: 256 },
        // A prefill's rows (the old path widens every page and scores on tcgen05).
        Case { heads: 64, lens: &[(300, 2600), (301, 3000)], page: 64, ratio: 4, top_k: 128 },
        Case { heads: 32, lens: &[(299, 1400), (301, 1200)], page: 256, ratio: 2, top_k: 128 },
        // Heads padded to 32; pages of 4 keys (a box a page, 32 a tile) and of
        // 256 (two tiles a page).
        Case { heads: 20, lens: &[(3, 1500)], page: 32, ratio: 2, top_k: 64 },
        Case { heads: 64, lens: &[(2, 1200), (1, 400)], page: 16, ratio: 4, top_k: 64 },
        Case { heads: 32, lens: &[(2, 1900)], page: 512, ratio: 2, top_k: 128 },
    ];
    for c in cases {
        let mut gpu = Gpu::open();
        let f = build(&mut gpu, c);
        let (want_sel, want) = rank(&mut gpu, c, &f, b200(false));
        let (got_sel, got) = rank(&mut gpu, c, &f, b200(true));
        let (mut exact, mut n, mut worst) = (0usize, 0usize, 0f32);
        let mut differ = 0usize;
        for (t, &(r, p)) in owners(c.lens).iter().enumerate() {
            let nkeys = (p + 1) / c.ratio;
            if nkeys <= c.top_k {
                continue;
            }
            let _ = r;
            for j in 0..nkeys {
                let (g, w) = (got[t * f.stride + j], want[t * f.stride + j]);
                // A score's terms are exact products (e2m1 by e2m1 under
                // powers of two); its sums round in another order.
                let bound = 4e-6 * w.abs().max(1.0);
                assert!(
                    (g - w).abs() <= bound,
                    "{c:?}: row {t} block {j}: {g} against {w}"
                );
                worst = worst.max((g - w).abs() / w.abs().max(1.0));
                exact += usize::from(g.to_bits() == w.to_bits());
                n += 1;
            }
            let (gs, ws) = (
                &got_sel[t * c.top_k..][..c.top_k],
                &want_sel[t * c.top_k..][..c.top_k],
            );
            assert!(
                gs.windows(2).all(|p| p[0] < p[1] || p[1] == -1),
                "row {t}: the picks ascend"
            );
            differ += gs.iter().filter(|j| !ws.contains(j)).count();
        }
        assert!(n > 0, "{c:?}: no row scored");
        assert!(
            differ * 2000 <= f.rows * c.top_k,
            "{c:?}: {differ} picks off"
        );
        println!(
            "{} heads, ratio {}, page {}: {exact} of {n} scores bit-exact, worst {worst:.2e}, {differ} picks off",
            c.heads, c.ratio, c.page
        );
    }
}
