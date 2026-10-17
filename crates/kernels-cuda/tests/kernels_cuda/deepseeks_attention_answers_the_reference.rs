#![cfg(feature = "cuda")]

//! DeepSeek-V4's attention launchers (`dsv4::decode`, `dsv4_prefill`'s pass
//! and window-only pass, `pool::attention_lse`) against the host softmax
//! over the window, the compressed entries and the sinks, on bf16 planes and
//! on the rows `dsv4::append` writes.

use crate::common::attn_ref::{Quant, YARN, attend, col, near, picks, pool};
use crate::common::attn_ref::{quantized, ten, tuned, turn, wide};
use crate::common::host::rows_read;
use crate::common::{Gpu, Lcg, captured, forget, host::upload_to, replay};
use dtype::Dtype::{Bf16, F32, I32, U8, U32};
use kernels_cuda::attn::dsv4::{self, Pooled, Rows, Turn};
use kernels_cuda::attn::{dsv4_prefill as pre, pool as pools};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const PAGE: usize = 64;
const PER: usize = 36;
const CELLS: usize = 2 * PER * PAGE;
const H: usize = 64;
const D: usize = 512;

/// Request `r`'s page `i` is pool page `2 i + r`.
fn slot(req: usize, pos: usize) -> usize {
    (2 * (pos / PAGE) + req) * PAGE + pos % PAGE
}

fn paged(gpu: &mut Gpu, keys: Tensor) -> KvPool {
    let pages: Vec<u32> = (0..2 * PER as u32)
        .map(|i| i % PER as u32 * 2 + i / PER as u32)
        .collect();
    let pages = col(gpu, &pages, U32);
    let indptr = col(gpu, &[0, PER as u32, 2 * PER as u32], U32);
    pool(keys, pages, indptr, PAGE)
}

/// `exact` (bf16 at `src`) as a pool of `fmt`'s rows written through the
/// append, and what its rows read as.
fn plane(gpu: &mut Gpu, src: u64, exact: &[f32], fmt: Option<Rows>) -> (KvPool, Vec<f32>) {
    let Some(fmt) = fmt else {
        return (paged(gpu, ten(src, CELLS, D, Bf16)), exact.to_vec());
    };
    let row = fmt.bytes() as usize;
    let at = gpu.zeros(CELLS * row);
    let rows = paged(gpu, ten(at, CELLS, row, U8));
    let page: Vec<u32> = (0..CELLS as u32).map(|c| c / PAGE as u32).collect();
    let off: Vec<u32> = (0..CELLS as u32).map(|c| c % PAGE as u32).collect();
    let (page, off) = (col(gpu, &page, U32), col(gpu, &off, U32));
    let src = ten(src, CELLS, D, Bf16);
    dsv4::append(&gpu.ctx(), src, &rows, page, off).expect("the append fires");
    gpu.sync();
    let read = rows_read(&gpu.down::<u8>(at, CELLS * row), CELLS, PAGE, fmt);
    match fmt {
        // V4.1's rows: its reference quantizers' codes, value for value.
        Rows::Mx8 => assert!(read == quantized(exact, Quant::Mx8), "MXFP8"),
        Rows::Nv4 => assert!(read == quantized(exact, Quant::Nv4), "NVFP4"),
        // V4's: each code within half its step, the rotated tail exact.
        Rows::Ds => near(&read, exact, 1.0 / 16.0, 0.02, "V4's fp8 rows"),
    }
    (rows, read)
}

#[derive(Clone, Copy, Debug)]
enum Pass {
    Decode,
    /// The prefill's pass, as a graph records it where `true`.
    Prefill(bool),
    Window,
}

/// `rows` of `(request, position)`; `ratio` 0 reads no entries, `top_k` 0
/// every visible one; a replay floor a row where `floored`.
#[allow(clippy::too_many_lines)]
fn check(
    pass: Pass,
    rows: &[(usize, usize)],
    [ratio, top_k]: [usize; 2],
    fmts: [Option<Rows>; 2],
    floored: bool,
) {
    let n = rows.len();
    let mut lcg = Lcg::seeded(0xd5f4 + (n * ratio) as u64);
    let (q_raw, q) = lcg.row(n * H * D);
    let (w_raw, w) = lcg.row(CELLS * D);
    let (e_raw, e) = lcg.row(CELLS * D);
    let sinks: Vec<f32> = (0..H).map(|h| (h as f32 * 0.37).sin() * 3.0).collect();
    let floor = |(r, &(_, p)): (usize, &(usize, usize))| (p as i32 - r as i32 * 37 % 180).max(0);
    let floors: Vec<i32> = rows.iter().enumerate().map(floor).collect();
    let floors: Vec<i32> = floors.iter().map(|f| f * i32::from(floored)).collect();
    // Ascending ids a row sees, neighbours sharing most, cut short by -1.
    let mut sel = vec![-1i32; n * top_k];
    for (r, &(_, pos)) in rows.iter().enumerate().filter(|_| top_k > 0) {
        let seen = (0..((pos + 1) / ratio) as i32).filter(|id| (id * 7 + r as i32 / 3) % 3 != 0);
        for (i, id) in seen.take(top_k - r % 4).enumerate() {
            sel[r * top_k + i] = id;
        }
    }
    let mut gpu = Gpu::open();
    let (w_at, e_at) = (gpu.up(&w_raw), gpu.up(&e_raw));
    let (win, w) = plane(&mut gpu, w_at, &w, fmts[0]);
    let (ent, e) = plane(&mut gpu, e_at, &e, fmts[1]);
    let (kv, scale) = ([w, e].concat(), 4.0 / (D as f32).sqrt());
    let (mut want, _) = attend([&q, &kv, &kv], [H, 1, D, D], scale, Some(&sinks), |r, _| {
        let (req, pos) = rows[r];
        let first = (pos + 1).saturating_sub(128).max(floors[r] as usize);
        let visible = (pos + 1).checked_div(ratio).unwrap_or(0);
        let ids = picks(&sel, r, top_k, visible).into_iter();
        let entries = ids.map(|c| CELLS + slot(req, (c + 1) * ratio - 1));
        (first..=pos).map(|p| slot(req, p)).chain(entries).collect()
    });
    let qt = ten(gpu.up(&q_raw), n, H * D, Bf16);
    let pos: Vec<i32> = rows.iter().map(|r| r.1 as i32).collect();
    let req: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let (pos_t, req_t) = (col(&mut gpu, &pos, I32), col(&mut gpu, &req, I32));
    let sink_t = col(&mut gpu, &sinks, F32);
    let floor_t = floored.then(|| col(&mut gpu, &floors, I32));
    let selection = (top_k > 0).then(|| (ten(gpu.up(&sel), n, top_k, I32), top_k as u32));
    let (ctx, op, ratio) = (gpu.ctx(), "attention.pool_attention", ratio as u32);
    let pooled = (ratio > 0).then_some(Pooled {
        entries: &ent,
        ratio,
        selection,
    });
    let staged = pre::Pooled {
        entries: &ent,
        ratio,
        selection,
    };
    let (yarn, t) = (Some(YARN), (floor_t, sink_t, scale));
    let turns = [
        None,
        Some(Turn {
            dim: 64,
            theta: 1e4,
            yarn,
        }),
    ];
    // Each pass bare and with its output's tail turned back.
    for turned in turns {
        let o_at = gpu.zeros(n * H * D * 2);
        let (o, (floor_t, sink_t, s)) = (&mut ten(o_at, n, H * D, Bf16), t);
        let fired = match pass {
            Pass::Decode => dsv4::decode(
                &ctx, qt, pos_t, req_t, floor_t, &win, 128, pooled, sink_t, 64, 512, s, turned, o,
            ),
            Pass::Prefill(safe) => {
                let takes = pre::takes(64, 512, Some(128), &win, &staged);
                assert!(takes, "V4-Flash's shape");
                pre::prefill(
                    &ctx, op, qt, pos_t, req_t, floor_t, safe, &win, &staged, sink_t, s, turned, o,
                )
            }
            Pass::Window if turned.is_some() => continue,
            Pass::Window => pre::window_only(&ctx, op, qt, pos_t, req_t, false, &win, sink_t, s, o),
        };
        fired.expect("the attention fires");
        gpu.sync();
        let got = wide(&gpu.down::<u16>(o_at, n * H * D));
        let what = format!("{pass:?}, {n} rows, {ratio} {top_k}, {fmts:?}, {floored}, {turned:?}");
        near(&got, &want, 4e-2, 0.05, &what);
        for (i, head) in want.chunks_mut(D).enumerate() {
            turn(head, pos[i / H], 64, true, true, yarn);
        }
    }
}

#[test]
fn deepseeks_attention_answers_the_reference() {
    let base = [2, 3, 50, 700, 2047, 2100, 1500, 127];
    let spread = |n: usize| {
        (0..n)
            .map(|r| (r % 2, base[r % 8] + r / 8 % 5))
            .collect::<Vec<_>>()
    };
    // Rows pair up within a request and across its end; the last stands alone.
    let run: Vec<_> = (600..608)
        .map(|p| (0, p))
        .chain((1000..1005).map(|p| (1, p)))
        .collect();
    let short: Vec<_> = (0..5).map(|p| (0, p)).chain(run.iter().copied()).collect();
    let (bf, ds) = ([None; 2], [Some(Rows::Ds); 2]);
    let v41 = [Some(Rows::Mx8), Some(Rows::Nv4)];
    let (dec, win) = (Pass::Decode, Pass::Window);
    let (staged, graph) = (Pass::Prefill(false), Pass::Prefill(true));
    #[rustfmt::skip]
    let cases = [
        // The decode's 8, 4, 2 and 1 blocks a row (1, 8, 16, 40 rows), each format and pool.
        (dec, spread(1), [4, 512], v41, false),
        (dec, spread(8), [0, 0], ds, true),
        (dec, spread(16), [4, 0], bf, false),
        (dec, spread(40), [2, 0], v41, false),
        (dec, spread(40), [4, 64], ds, true),
        // The prefill's CSA and HCA layers, staged and as a graph records them.
        (staged, run.clone(), [4, 64], bf, false),
        (graph, run.clone(), [128, 0], ds, false),
        (graph, run.clone(), [2, 512], v41, true),
        (staged, run.clone(), [1, 512], v41, false),
        (staged, run.clone(), [4, 0], bf, true),
        // A window-only layer, rows short of a window too.
        (win, short.clone(), [0, 0], bf, false),
        (win, short.clone(), [0, 0], ds, false),
    ];
    for (pass, rows, pool, fmts, floored) in cases {
        check(pass, &rows, pool, fmts, floored);
    }
}

/// The prefill's pass recorded over 11 pages replays over the page table
/// it has grown to, landing what a fresh pass lands, bit for bit.
#[test]
fn a_recorded_pass_reads_the_page_tables_its_replay_states() {
    for fmts in [[Rows::Ds; 2], [Rows::Mx8, Rows::Nv4]] {
        let (mut gpu, mut lcg, n) = (Gpu::open(), Lcg::seeded(0x9a), 6usize);
        let (q_raw, _) = lcg.row(n * H * D);
        let (w_raw, w) = lcg.row(CELLS * D);
        let w_at = gpu.up(&w_raw);
        let (win, _) = plane(&mut gpu, w_at, &w, Some(fmts[0]));
        let (ent, _) = plane(&mut gpu, w_at, &w, Some(fmts[1]));
        let (table_at, indptr_at, pos_at) = (gpu.zeros(PER * 4), gpu.zeros(8), gpu.zeros(n * 4));
        let (qt, pos_t) = (ten(gpu.up(&q_raw), n, H * D, Bf16), ten(pos_at, n, 1, I32));
        let (req_t, sink_t) = (
            col(&mut gpu, &[0i32; 6], I32),
            col(&mut gpu, &[0.5f32; H], F32),
        );
        let (o_at, fresh_at) = (gpu.zeros(n * H * D * 2), gpu.zeros(n * H * D * 2));
        let held = |held: usize| {
            let positions: Vec<i32> = (0..n).map(|r| (held * PAGE - 5 - n + r) as i32).collect();
            upload_to(
                table_at,
                &(0..held as u32).map(|i| 2 * i).collect::<Vec<_>>(),
            );
            upload_to(indptr_at, &[0u32, held as u32]);
            upload_to(pos_at, &positions);
        };
        let (page_indices, page_indptr) = (ten(table_at, PER, 1, U32), ten(indptr_at, 2, 1, U32));
        let win = KvPool {
            page_indices,
            page_indptr,
            ..win
        };
        let ent = KvPool {
            page_indices,
            page_indptr,
            ..ent
        };
        let (op, staged) = (
            "attention.pool_attention",
            pre::Pooled {
                entries: &ent,
                ratio: 4,
                selection: None,
            },
        );
        let fire = |at: u64| {
            let (ctx, o) = (gpu.ctx(), &mut ten(at, n, H * D, Bf16));
            let fired = pre::prefill(
                &ctx, op, qt, pos_t, req_t, None, true, &win, &staged, sink_t, 0.04, None, o,
            );
            fired.expect("the pass fires");
        };
        held(11);
        let exec = captured(&gpu.ctx(), || fire(o_at));
        for pages in [11usize, 23, PER] {
            gpu.sync();
            held(pages);
            replay(&gpu.ctx(), exec);
            fire(fresh_at);
            gpu.sync();
            let replayed: Vec<u16> = gpu.down(o_at, n * H * D);
            assert!(replayed.iter().any(|&v| v != 0), "{fmts:?}: the pass lands");
            let fresh: Vec<u16> = gpu.down(fresh_at, n * H * D);
            assert!(replayed == fresh, "{fmts:?}: over {pages} pages");
        }
        forget(exec);
    }
}

/// `pool::attention_lse` (every visible entry) and `_selected` (an
/// index's picks, ids past the visible ones read as nothing, a list ended
/// early) on the scalar walk and the tensor cores, a partial head block too.
#[test]
fn the_pool_attention_answers_the_reference() {
    let cases = [
        (64, 512, 4, 512, true),
        (64, 512, 128, 0, false),
        (20, 256, 4, 64, false),
        (24, 128, 16, 0, true),
    ];
    for (heads, d, ratio, top_k, mma) in cases {
        let (positions, reqs) = (
            [2i32, 3, 50, 700, 2047, 2100, 1500, 9],
            [0i32, 1, 0, 1, 0, 1, 1, 0],
        );
        let mut sel = Vec::new();
        for (r, &p) in positions.iter().enumerate() {
            let visible = (p as usize + 1) / ratio;
            let early = |i: usize| (r == 3 || r == 5) && i >= top_k / 2;
            let past = |i: usize| (r + i) % 9 == 0;
            let id = |i: usize| {
                if past(i) {
                    visible + i % 7
                } else {
                    (i * 37 + r * 11) % (visible.max(1) + 3)
                }
            };
            sel.extend((0..top_k).map(|i| if early(i) { -1 } else { id(i) as i32 }));
        }
        let (mut gpu, mut lcg, n, s) =
            (Gpu::open(), Lcg::seeded(0x9001), 8, 1.0 / (d as f32).sqrt());
        let (q_raw, q) = lcg.row(n * heads * d);
        let (k_raw, k) = lcg.row(CELLS * d);
        let keys = ten(gpu.up(&k_raw), CELLS, d, Bf16);
        let ent = paged(&mut gpu, keys);
        let (want, want_lse) = attend([&q, &k, &k], [heads, 1, d, d], s, None, |r, _| {
            let visible = (positions[r] as usize + 1) / ratio;
            let ids = picks(&sel, r, top_k, visible).into_iter();
            ids.map(|c| slot(reqs[r] as usize, (c + 1) * ratio - 1))
                .collect()
        });
        let qt = ten(gpu.up(&q_raw), n, heads * d, Bf16);
        let (pos_t, req_t) = (col(&mut gpu, &positions, I32), col(&mut gpu, &reqs, I32));
        let (o_at, lse_at) = (gpu.zeros(n * heads * d * 2), gpu.zeros(n * heads * 4));
        let (o, lse) = (
            &mut ten(o_at, n, heads * d, Bf16),
            &mut ten(lse_at, n, heads, F32),
        );
        let ctx = gpu.ctx().with_tuning(tuned(|t| t.pool_mma = mma));
        let (h, w, ratio, k) = (heads as u32, d as u32, ratio as u32, top_k as u32);
        let fired = if top_k > 0 {
            let sel_t = ten(gpu.up(&sel), n, top_k, I32);
            pools::attention_lse_selected(
                &ctx, qt, pos_t, req_t, sel_t, &ent, ratio, k, h, w, s, o, lse,
            )
        } else {
            pools::attention_lse(&ctx, qt, pos_t, req_t, &ent, ratio, h, w, s, o, lse)
        };
        fired.expect("the pool attention fires");
        gpu.sync();
        let what = format!("{heads} heads at {d}, ratio {ratio}, top {top_k}, mma {mma}");
        near(
            &wide(&gpu.down::<u16>(o_at, n * heads * d)),
            &want,
            3e-2,
            0.05,
            &what,
        );
        let lse_got: Vec<f32> = gpu.down(lse_at, n * heads);
        let close = |(g, w): (&f32, &f32)| g == w || (g - w).abs() <= 2e-2 * w.abs().max(1.0);
        assert!(lse_got.iter().zip(&want_lse).all(close), "{what}: lse");
    }
}

/// A fire seats more lanes than it fills (their indptr 0, no pages): no
/// such lane is handed a pool boundary, and the gather (both forms) walks
/// no page a lane does not own (it read past the table, a 700 under
/// compute-sanitizer) and leaves rows past the staged window alone.
#[test]
fn a_padded_lane_closes_no_pool_entry() {
    let mut gpu = Gpu::open();
    let positions: Vec<i32> = (0..16).map(|r| r % 8).chain([3, 7, 11, 15]).collect();
    let data = col(&mut gpu, &positions, I32);
    let indptr = col(&mut gpu, &[0i32, 8, 16, 0, 0, 0], I32);
    let (bpos_at, breq_at, rope_at) = (gpu.zeros(80), gpu.zeros(80), gpu.zeros(80));
    let (bp, br) = (&mut ten(bpos_at, 20, 1, I32), &mut ten(breq_at, 20, 1, I32));
    let (rope, absent, ctx) = (&mut ten(rope_at, 20, 1, I32), Tensor::ABSENT, gpu.ctx());
    let fired =
        pools::boundary_prefill(&ctx, RaggedTensor { data, indptr }, absent, 4, bp, br, rope);
    fired.expect("the boundaries fire");
    gpu.sync();
    let (bpos, breq): (Vec<i32>, Vec<i32>) = (gpu.down(bpos_at, 20), gpu.down(breq_at, 20));
    for (r, &p) in positions.iter().enumerate() {
        let closes = r < 16 && (p + 1) % 4 == 0;
        assert_eq!(bpos[r], if closes { p } else { -1 }, "row {r}: boundary");
        assert!(r >= 16 || breq[r] == r as i32 / 8, "row {r}: owner");
    }
    let mut lcg = Lcg::seeded(0x5eed);
    let kv = ten(gpu.up(&lcg.row(16 * 128).0), 16, 128, Bf16);
    let score = ten(gpu.up(&lcg.row(16 * 128).0), 16, 128, Bf16);
    let (pages, owned) = (
        col(&mut gpu, &[0u32], U32),
        col(&mut gpu, &[0u32, 1, 1, 1], U32),
    );
    let state = pool(kv, pages, owned, 16);
    let bpos = col(&mut gpu, &[7i32, 7, 1_000_000_003], I32);
    let breq = col(&mut gpu, &[0i32, 2, -123_456_789], I32);
    let win_at = gpu.up(&[2u32, 0, 3, 0]);
    for vec in [false, true] {
        let out_at = gpu.up(&[0x3f80u16; 3 * 64]);
        let ctx = gpu.ctx().with_tuning(tuned(|t| t.rows_vec = vec));
        ctx.arm_stage(win_at);
        let out = &mut ten(out_at, 3, 64, Bf16);
        let fired = pools::gather(&ctx, bpos, breq, &state, 64, 4, kv, score, absent, out);
        fired.expect("the gather fires");
        gpu.sync();
        let out: Vec<u16> = gpu.down(out_at, 3 * 64);
        assert!(
            out[..64].iter().any(|&b| b != 0),
            "vec {vec}: the live lane"
        );
        assert!(
            out[64..128].iter().all(|&b| b == 0),
            "vec {vec}: the padded lane"
        );
        assert!(
            out[128..].iter().all(|&b| b == 0x3f80),
            "vec {vec}: past the window"
        );
    }
}
