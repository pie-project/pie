//! The paged attention family against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use engine_xla::handle::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Tensor};
use engine_xla::lower::attn as low;
use kernels_xla::attn::{self, arbiter};

// ------------------------------------------------------------------ fixtures

/// The score capture lowered from handles: the page tables, the query's
/// indptr, the positions, its data, the keys plane, the slab.
#[allow(clippy::too_many_arguments)]
fn capture(
    ctx: &engine_xla::emit::Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    kv: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    observe: u32,
    lane_offset: u32,
    plane_stride: u32,
    plane: u32,
    kv_max: u32,
    requests: u32,
    scores: Tensor,
) -> Result<(), kernels_xla::Error> {
    engine_xla::lower::lower(ctx, |l| {
        let page_indptr = l.read(kv.page_indptr)?;
        let page_indices = l.read(kv.page_indices)?;
        let indptr = l.read(q.indptr)?;
        let positions = l.read(plan.positions)?;
        let data = l.read(q.data)?;
        let keys = l.read(kv.keys)?;
        let slab = l.read(scores)?;
        let tuning = l.tuning();
        let v = attn::score::capture(
            l.f(),
            kernels_xla::tensor::RaggedTensor { data, indptr },
            positions,
            keys,
            page_indptr,
            page_indices,
            kv,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            observe,
            lane_offset,
            plane_stride,
            plane,
            kv_max,
            requests,
            slab,
            tuning,
        )?;
        l.land(scores, v)
    })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| round_bf16(self.next() * scale)).collect()
    }
}

/// A paged pool: `lanes[l]` lists lane `l`'s pages.
struct Pool {
    ps: usize,
    kvh: usize,
    d: usize,
    slots: usize,
    lanes: Vec<Vec<i32>>,
    keys: Vec<f32>,
    values: Vec<f32>,
    max_pages: u32,
}

impl Pool {
    fn new(
        rng: &mut Rng,
        ps: usize,
        kvh: usize,
        d: usize,
        pages: usize,
        lanes: Vec<Vec<i32>>,
        max_pages: u32,
    ) -> Self {
        let slots = pages * ps;
        Self {
            ps,
            kvh,
            d,
            slots,
            keys: rng.bf16s(slots * kvh * d, 1.0),
            values: rng.bf16s(slots * kvh * d, 1.0),
            lanes,
            max_pages,
        }
    }

    fn slot(&self, lane: usize, kp: usize) -> usize {
        self.lanes[lane][kp / self.ps] as usize * self.ps + kp % self.ps
    }

    fn capacity(&self, lane: usize) -> usize {
        self.lanes[lane].len() * self.ps
    }

    fn bind(&self, b: &mut Bench) -> KvPool {
        let w = (self.kvh * self.d) as u32;
        let keys = b.bf16(self.slots as u32, w, &self.keys);
        let values = b.bf16(self.slots as u32, w, &self.values);
        let mut indptr = vec![0i32];
        let mut indices = Vec::new();
        for l in &self.lanes {
            indices.extend_from_slice(l);
            indptr.push(indices.len() as i32);
        }
        let page_indices = b.i32(indices.len() as u32, 1, &indices);
        let page_indptr = b.i32(indptr.len() as u32, 1, &indptr);
        KvPool {
            keys,
            values,
            page_indices,
            page_indptr,
            page_size: self.ps as i32,
            max_pages: self.max_pages,
            seq_stride: w as u64,
            head_stride: self.d as u64,
        }
    }
}

/// Scaled dot-product attention over explicit key lists: `keys(r)` names
/// the key rows row `r` reads (duplicates count twice); `bias(r, h, j)` adds
/// to the scaled logit of the `j`-th listed key. Returns `o` and the base-2
/// lse.
#[allow(clippy::too_many_arguments)]
fn sdpa(
    q: &[f32],
    rows: usize,
    qh: usize,
    kvh: usize,
    d: usize,
    dv: usize,
    k: &[f32],
    v: &[f32],
    scale: f64,
    keys: &dyn Fn(usize) -> Vec<usize>,
    bias: &dyn Fn(usize, usize, usize) -> f64,
    mult: &dyn Fn(usize) -> f64,
) -> (Vec<f32>, Vec<f32>) {
    let g = qh / kvh;
    let mut o = vec![0.0f32; rows * qh * dv];
    let mut lse = vec![f32::NEG_INFINITY; rows * qh];
    for r in 0..rows {
        let list = keys(r);
        for h in 0..qh {
            let kh = h / g;
            if list.is_empty() {
                continue;
            }
            let s: Vec<f64> = list
                .iter()
                .enumerate()
                .map(|(j, &kr)| {
                    let mut dot = 0.0f64;
                    for i in 0..d {
                        dot += f64::from(q[(r * qh + h) * d + i])
                            * f64::from(k[(kr * kvh + kh) * d + i]);
                    }
                    (dot * scale + bias(r, h, j)) * mult(r)
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = e.iter().sum();
            for i in 0..dv {
                let mut acc = 0.0f64;
                for (j, &kr) in list.iter().enumerate() {
                    acc += e[j] * f64::from(v[(kr * kvh + kh) * dv + i]);
                }
                o[(r * qh + h) * dv + i] = round_bf16((acc / l) as f32);
            }
            lse[r * qh + h] = (m * std::f64::consts::LOG2_E + l.log2()) as f32;
        }
    }
    (o, lse)
}

fn no_bias(_: usize, _: usize, _: usize) -> f64 {
    0.0
}

fn unit(_: usize) -> f64 {
    1.0
}

struct Fire {
    positions: Vec<i32>,
    req: Vec<i32>,
    mask: Vec<u8>,
    enabled: Vec<u8>,
    stride: u32,
}

impl Fire {
    fn plain(rows: &[(i32, i32)]) -> Self {
        Self {
            positions: rows.iter().map(|r| r.1).collect(),
            req: rows.iter().map(|r| r.0).collect(),
            mask: vec![0; rows.len()],
            enabled: vec![0; rows.len()],
            stride: 0,
        }
    }

    fn tables(&self, b: &mut Bench) -> (Tensor, Tensor, Tensor, Tensor) {
        let n = self.positions.len() as u32;
        let p = b.i32(n, 1, &self.positions);
        let r = b.i32(n, 1, &self.req);
        let m = b.u8(n, self.stride.max(1), &self.mask);
        let e = b.u8(n, 1, &self.enabled);
        (p, r, m, e)
    }

    fn decode(&self, b: &mut Bench) -> DecodePlan {
        let (positions, request_of_token, mask, mask_enabled) = self.tables(b);
        DecodePlan {
            positions,
            request_of_token,
            mask,
            mask_enabled,
            mask_stride: self.stride,
        }
    }

    fn prefill(&self, b: &mut Bench) -> PrefillPlan {
        let (positions, request_of_token, mask, mask_enabled) = self.tables(b);
        PrefillPlan {
            positions,
            request_of_token,
            mask,
            mask_enabled,
            mask_stride: self.stride,
        }
    }

    /// Keys row `r` admits under the GPU kernels' rules.
    fn admitted(&self, pool: &Pool, r: usize, window: u32, causal: bool) -> Vec<usize> {
        let lane = self.req[r] as usize;
        let qpos = self.positions[r];
        let cap = pool.capacity(lane) as i32;
        let en = self.enabled[r];
        let wide = !causal || en == 2;
        let hi = if wide { cap } else { (qpos + 1).min(cap) };
        let lo = if window > 0 && qpos >= window as i32 {
            qpos - window as i32 + 1
        } else {
            0
        };
        (lo.max(0)..hi.max(0))
            .filter(|&kp| {
                en == 0
                    || ((kp as u32) < self.stride
                        && self.mask[r * self.stride as usize + kp as usize] != 0)
            })
            .map(|kp| pool.slot(lane, kp as usize))
            .collect()
    }
}

fn check_rows(got: &[f32], want: &[f32], width: usize, rows: usize, atol: f32) {
    assert_close(&got[..rows * width], &want[..rows * width], atol, 8e-3);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "a padded row answered a non-finite value"
    );
}

// -------------------------------------------------------------------- tests

#[test]
fn decode_reads_each_rows_lane_through_its_pages() {
    let mut rng = Rng(7);
    let (qh, kvh, d) = (6, 2, 48);
    let pool = Pool::new(
        &mut rng,
        4,
        kvh,
        d,
        9,
        vec![vec![5, 1, 7], vec![2], vec![0, 6, 8]],
        4,
    );
    // Three live rows and one padded row (position -1).
    let fire = Fire::plain(&[(0, 9), (1, 2), (2, 11), (0, -1)]);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let ow = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::decode_lse(qt, &plan, None, d as u32, scale as f32, o, lse)?,
                kv,
            )?;
            low::attend(
                ctx,
                attn::decode(qt, &plan, Some(3), d as u32, scale as f32, ow)?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let (want, want_lse) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-3);
    assert_close(&b.read_f32(lse)[..3 * qh], &want_lse[..3 * qh], 1e-3, 1e-4);
    assert!(
        b.read_f32(lse)[3 * qh..]
            .iter()
            .all(|x| *x == f32::NEG_INFINITY)
    );
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 3, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(ow), &want, qh * d, rows, 1e-3);
}

/// Ragged prefill rows: `(lane, first position, count)` runs, then `pad`
/// padded rows.
fn prefill_rows(runs: &[(i32, i32, i32)], pad: usize) -> Vec<(i32, i32)> {
    let mut rows = Vec::new();
    for &(lane, first, n) in runs {
        for i in 0..n {
            rows.push((lane, first + i));
        }
    }
    rows.extend(std::iter::repeat_n((0, -1), pad));
    rows
}

fn ragged(b: &mut Bench, runs: &[(i32, i32, i32)], data: Tensor) -> RaggedTensor {
    let mut indptr = vec![0i32];
    for r in runs {
        indptr.push(indptr.last().unwrap() + r.2);
    }
    let indptr = b.i32(indptr.len() as u32, 1, &indptr);
    RaggedTensor { data, indptr }
}

#[test]
fn prefill_walks_row_blocks_and_per_row_alike() {
    prefill_alike(4, 2, 32, 300);
}

/// Wide kv heads walk head by head; 620 rows span two row blocks.
#[test]
fn a_wide_headed_prefill_walks_each_kv_head_alone() {
    prefill_alike(8, 2, 128, 620);
}

fn prefill_alike(qh: usize, kvh: usize, d: usize, n0: i32) {
    let mut rng = Rng(11);
    let lane0: Vec<i32> = (0..45).map(|i| (i * 7 + 3) % 50).collect();
    let pool = Pool::new(
        &mut rng,
        16,
        kvh,
        d,
        52,
        vec![lane0, vec![50, 45], vec![46, 51, 47]],
        45,
    );
    let runs = [(0, 720 - n0, n0), (1, 20, 5), (2, 33, 1)];
    let fire = Fire::plain(&prefill_rows(&runs, 2));
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.2;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.prefill(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let w = (qh * d) as u32;
    let o_blocked = b.zeros(Dtype::Bf16, rows as u32, w);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    let o_rows = b.zeros(Dtype::Bf16, rows as u32, w);
    let o_win = b.zeros(Dtype::Bf16, rows as u32, w);
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::prefill_lse(
                    qr, &plan, &kv, None, d as u32, kvh as u32, scale, o_blocked, lse,
                )?,
                kv,
            )?;
            low::attend(
                ctx,
                arbiter::prefill(
                    qr, &plan, &kv, None, d as u32, kvh as u32, scale, o_rows, 1000,
                )?,
                kv,
            )?;
            low::attend(
                ctx,
                attn::prefill(
                    qr,
                    &plan,
                    &kv,
                    Some(100),
                    d as u32,
                    kvh as u32,
                    scale,
                    o_win,
                )?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let (want, want_lse) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| fire.admitted(&pool, r, 0, true),
        &no_bias,
        &unit,
    );
    let live = rows - 2;
    check_rows(&b.read_f32(o_blocked), &want, qh * d, live, 1e-3);
    check_rows(&b.read_f32(o_rows), &want, qh * d, live, 1e-3);
    assert_close(
        &b.read_f32(lse)[..live * qh],
        &want_lse[..live * qh],
        2e-3,
        1e-4,
    );
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| fire.admitted(&pool, r, 100, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o_win), &want, qh * d, live, 1e-3);
}

#[test]
fn a_long_decode_walks_its_keys_in_chunks() {
    long_decode(8, 2, 64);
    // 64-wide heads in pairs (gpt-oss: 8 kv heads of 64).
    long_decode(16, 8, 64);
}

/// Wide heads take the per-kv-head column-slice contraction.
#[test]
fn a_long_decode_with_wide_heads_slices_each_kv_head() {
    long_decode(8, 2, 128);
    long_decode(8, 4, 256);
}

/// A sliding window walks window-wide chunks from each row's window start.
#[test]
fn a_long_windowed_decode_reads_only_its_window() {
    long_decode_in(8, 2, 64, Some(100));
    long_decode_in(8, 2, 256, Some(512));
}

/// A sliding-window decode's walk stops after the last chunk any row admits
/// a key in: while every lane's context is shorter than the window, the
/// chunks past it are skipped. Skipping them changes no bit of the answer:
/// the same fire walked over every chunk of the window, unrolled, as the walk
/// ran before its bound (`Chunks::window_reach` off), answers the same bytes.
#[test]
fn a_window_walk_bounded_by_its_rows_reach_answers_the_same_bits() {
    // (q heads, kv heads, head width, window, most pages a lane holds): the
    // first three keep every context under the window (the bound cuts the
    // walk), the last puts some lanes past it (the bound is the whole walk).
    for (qh, kvh, d, window, most) in [
        (8usize, 2usize, 128usize, 1024u32, 40usize),
        (8, 2, 256, 512, 20),
        (16, 8, 128, 1024, 8),
        (8, 2, 128, 256, 40),
    ] {
        let walked = |reach: bool| -> Option<Vec<u8>> {
            let mut rng = Rng(11);
            let ps = 16;
            let lanes: Vec<Vec<i32>> = (0..64)
                .map(|l| {
                    (0..1 + (l * 37) % most)
                        .map(|p| ((p * 13 + l) % 90) as i32)
                        .collect()
                })
                .collect();
            let pool = Pool::new(&mut rng, ps, kvh, d, 90, lanes, most as u32);
            let rows: Vec<(i32, i32)> = (0..64)
                .map(|l| {
                    let cap = pool.capacity(l) as i32;
                    (l as i32, cap - 1 - (l as i32 % 5).min(cap - 1))
                })
                .collect();
            let fire = Fire::plain(&rows);
            let n = rows.len();
            let q = rng.bf16s(n * qh * d, 1.0);
            let mut tuning = kernels_xla::tuning::Tuning::DEFAULT;
            tuning.chunks.window_reach = reach;
            let mut b = Bench::new().with_tuning(tuning);
            let kv = pool.bind(&mut b);
            let plan = fire.decode(&mut b);
            let qt = b.bf16(n as u32, (qh * d) as u32, &q);
            let o = b.zeros(Dtype::Bf16, n as u32, (qh * d) as u32);
            let ran = b
                .run(|ctx| {
                    low::attend(
                        ctx,
                        attn::decode(qt, &plan, Some(window), d as u32, 0.125, o)?,
                        kv,
                    )
                })
                .unwrap();
            ran.then(|| b.bytes(o).to_vec())
        };
        let (Some(bounded), Some(every)) = (walked(true), walked(false)) else {
            return;
        };
        assert!(
            bounded == every,
            "window {window}, {kvh}x{d} kv heads: the bounded walk answered other bits than every chunk's"
        );
    }
}

/// A batched decode through the page-copy kernel (`kernel.attn.decode_pages`)
/// answers the reference as the walk does.
#[test]
fn a_batched_decode_through_the_page_copy_kernel_answers_the_reference() {
    let mut tuning = kernels_xla::tuning::Tuning::DEFAULT;
    tuning.chunks.decode_pages = 8;
    for (qh, kvh, d) in [
        (16usize, 8usize, 128usize),
        (8, 2, 128),
        (8, 2, 256),
        (8, 1, 256),
    ] {
        long_decode_with(qh, kvh, d, None, tuning);
    }
    // a sliding window's rows read only the pages their window covers
    long_decode_with(8, 2, 128, Some(100), tuning);
    long_decode_with(8, 2, 256, Some(512), tuning);
}

fn long_decode(qh: usize, kvh: usize, d: usize) {
    long_decode_in(qh, kvh, d, None);
}

fn long_decode_in(qh: usize, kvh: usize, d: usize, window: Option<u32>) {
    long_decode_with(qh, kvh, d, window, kernels_xla::tuning::Tuning::DEFAULT);
}

fn long_decode_with(
    qh: usize,
    kvh: usize,
    d: usize,
    window: Option<u32>,
    tuning: kernels_xla::tuning::Tuning,
) {
    let mut rng = Rng(5);
    let ps = 16;
    // 64 lanes over 90 pages; lane l holds 1 + (l * 37) % 80 pages of a
    // shared pool (lanes may share pages, as prefix sharing does).
    let lanes: Vec<Vec<i32>> = (0..64)
        .map(|l| (0..1 + (l * 37) % 80).map(|p| (p * 13 + l) % 90).collect())
        .collect();
    let pool = Pool::new(&mut rng, ps, kvh, d, 90, lanes, 80);
    let rows: Vec<(i32, i32)> = (0..64)
        .map(|l| (l as i32, (pool.capacity(l) as i32) - 1 - (l as i32 % 5)))
        .collect();
    let fire = Fire::plain(&rows);
    let n = rows.len();
    let q = rng.bf16s(n * qh * d, 1.0);
    let scale = 0.125;
    let mut b = Bench::new().with_tuning(tuning);
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(n as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, n as u32, (qh * d) as u32);
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::decode(qt, &plan, window, d as u32, scale, o)?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let (want, _) = sdpa(
        &q,
        n,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| fire.admitted(&pool, r, window.unwrap_or(0), true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o), &want, qh * d, n, 1e-3);
}

#[test]
fn a_custom_mask_gates_causal_wide_and_bidirectional_rows() {
    custom_mask(3);
}

/// The same with a page bound wide enough that the row-block walk's key
/// chunks each sit inside one lane's span (the windowed mask read).
#[test]
fn a_custom_mask_gates_rows_when_chunks_sit_inside_a_lane() {
    custom_mask(64);
}

fn custom_mask(max_pages: u32) {
    let mut rng = Rng(3);
    let (qh, kvh, d) = (4, 4, 40);
    let pool = Pool::new(
        &mut rng,
        8,
        kvh,
        d,
        6,
        vec![vec![3, 0], vec![5, 1, 4], vec![2]],
        max_pages,
    );
    let runs = [(0, 10, 4), (1, 18, 3), (2, 0, 2)];
    let mut fire = Fire::plain(&prefill_rows(&runs, 1));
    let rows = fire.positions.len();
    let stride = 24u32;
    fire.stride = stride;
    fire.mask = (0..rows * stride as usize)
        .map(|i| u8::from((i * 7 + i / 5) % 3 != 0))
        .collect();
    // Lane 0 masked causally, lane 1 masked wide (flag 2), lane 2 unmasked.
    fire.enabled = fire.req.iter().map(|&l| [1, 2, 0][l as usize]).collect();
    fire.enabled[rows - 1] = 1;
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.15;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.prefill(&mut b);
    // The op's own mask: the plan's plane, inverted.
    let inv: Vec<u8> = fire.mask.iter().map(|&m| 1 - m).collect();
    let mask = b.u8(rows as u32, stride, &inv);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let w = (qh * d) as u32;
    let o_causal = b.zeros(Dtype::Bf16, rows as u32, w);
    let o_wide = b.zeros(Dtype::Bf16, rows as u32, w);
    let lse_wide = b.zeros(Dtype::F32, rows as u32, qh as u32);
    let o_rows = b.zeros(Dtype::Bf16, rows as u32, w);
    let o_plain = b.zeros(Dtype::Bf16, rows as u32, w);
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::masked(qr, &plan, mask, None, d as u32, scale, o_causal)?,
                kv,
            )?;
            low::attend(
                ctx,
                arbiter::masked_lse(
                    qr, &plan, mask, None, false, d as u32, scale, o_wide, lse_wide, 1,
                )?,
                kv,
            )?;
            low::attend(
                ctx,
                arbiter::masked(qr, &plan, mask, None, true, d as u32, scale, o_rows, 100)?,
                kv,
            )?;
            low::attend(
                ctx,
                arbiter::prefill(
                    qr,
                    &plan,
                    &kv,
                    Some(5),
                    d as u32,
                    kvh as u32,
                    scale,
                    o_plain,
                    1,
                )?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let inverted = Fire {
        mask: inv,
        ..Fire::plain(&[])
    };
    let with = |fire: &Fire, mask: &Fire| Fire {
        positions: fire.positions.clone(),
        req: fire.req.clone(),
        mask: mask.mask.clone(),
        enabled: fire.enabled.clone(),
        stride,
    };
    let custom = with(&fire, &inverted);
    let live = rows - 1;
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| custom.admitted(&pool, r, 0, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o_causal), &want, qh * d, live, 1e-3);
    check_rows(&b.read_f32(o_rows), &want, qh * d, live, 1e-3);
    let (want, want_lse) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| custom.admitted(&pool, r, 0, false),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o_wide), &want, qh * d, live, 1e-3);
    let got_lse = b.read_f32(lse_wide);
    for i in 0..live * qh {
        if want_lse[i].is_finite() {
            assert!(
                (got_lse[i] - want_lse[i]).abs() < 2e-3,
                "lse {i}: {} vs {}",
                got_lse[i],
                want_lse[i]
            );
        } else {
            assert_eq!(got_lse[i], f32::NEG_INFINITY, "lse {i}");
        }
    }
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| fire.admitted(&pool, r, 5, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o_plain), &want, qh * d, live, 1e-3);
}

/// The key positions a block selection admits (the GPU kernels' walk).
fn selected_kps(fire: &Fire, sel: &[i32], top_k: usize, ratio: i32, r: usize) -> Vec<usize> {
    let qpos = fire.positions[r];
    let nblocks = (qpos + 1) / ratio;
    let blocks = (top_k as i32).min(nblocks).max(0);
    let keeps = |kp: i32| {
        kp >= 0
            && kp <= qpos
            && (fire.enabled[r] == 0
                || ((kp as u32) < fire.stride
                    && fire.mask[r * fire.stride as usize + kp as usize] != 0))
    };
    let mut out = Vec::new();
    for n in 0..blocks * ratio {
        let c = sel[r * top_k + (n / ratio) as usize];
        if c >= 0 && c < nblocks {
            let kp = c * ratio + n % ratio;
            if keeps(kp) {
                out.push(kp as usize);
            }
        }
    }
    for kp in nblocks * ratio..=qpos {
        if keeps(kp) {
            out.push(kp as usize);
        }
    }
    out
}

#[test]
fn a_selection_reads_its_named_blocks_and_the_open_one() {
    let mut rng = Rng(21);
    let (qh, kvh, d) = (4, 1, 64);
    let pool = Pool::new(
        &mut rng,
        8,
        kvh,
        d,
        8,
        vec![vec![7, 2, 5], vec![1, 0, 3, 6]],
        4,
    );
    let (top_k, ratio) = (3usize, 4u32);
    let runs = [(0, 17, 3), (1, 5, 2), (1, 29, 1)];
    let mut fire = Fire::plain(&prefill_rows(&runs, 1));
    let rows = fire.positions.len();
    fire.stride = 32;
    fire.mask = (0..rows * 32).map(|i| u8::from(i % 7 != 3)).collect();
    fire.enabled = (0..rows).map(|r| u8::from(r == 1)).collect();
    // Block ids: named, repeated, -1 padding, and one past the row's blocks.
    let sel: Vec<i32> = vec![
        0, 3, 1, //
        2, 2, -1, //
        4, 1, 0, //
        0, -1, -1, //
        1, 9, 0, //
        6, 2, 5, //
        0, 0, 0,
    ];
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.125;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let dplan = fire.decode(&mut b);
    let pplan = fire.prefill(&mut b);
    let st = b.i32(rows as u32, top_k as u32, &sel);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let o1 = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let o2 = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::decode_selected(qt, &dplan, st, None, d as u32, scale, ratio, false, o1)?,
                kv,
            )?;
            low::attend(
                ctx,
                attn::prefill_selected(
                    qr, &pplan, st, &kv, None, d as u32, kvh as u32, scale, ratio, false, o2,
                )?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| {
            let lane = fire.req[r] as usize;
            selected_kps(&fire, &sel, top_k, ratio as i32, r)
                .into_iter()
                .map(|kp| pool.slot(lane, kp))
                .collect()
        },
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o1), &want, qh * d, rows - 1, 1e-3);
    check_rows(&b.read_f32(o2), &want, qh * d, rows - 1, 1e-3);
}

#[test]
fn a_per_head_selection_reads_each_kv_heads_own_blocks() {
    let mut rng = Rng(22);
    let (qh, kvh, d) = (4, 2, 64);
    let pool = Pool::new(
        &mut rng,
        8,
        kvh,
        d,
        8,
        vec![vec![7, 2, 5], vec![1, 0, 3, 6]],
        4,
    );
    let (top_k, ratio) = (2usize, 4u32);
    let runs = [(0, 17, 3), (1, 5, 2), (1, 29, 1)];
    let fire = Fire::plain(&prefill_rows(&runs, 1));
    let rows = fire.positions.len();
    // Per row, kv head 0's run, then kv head 1's (block_max's layout).
    let sel: Vec<i32> = vec![
        0, 3, 1, 4, //
        2, -1, 0, 1, //
        1, 4, 3, 4, //
        0, -1, 1, -1, //
        1, 7, 0, 2, //
        5, 7, 2, 6, //
        0, 0, 0, 0,
    ];
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.125;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let dplan = fire.decode(&mut b);
    let pplan = fire.prefill(&mut b);
    let st = b.i32(rows as u32, (kvh * top_k) as u32, &sel);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let o1 = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let o2 = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::decode_selected(qt, &dplan, st, None, d as u32, scale, ratio, true, o1)?,
                kv,
            )?;
            low::attend(
                ctx,
                attn::prefill_selected(
                    qr, &pplan, st, &kv, None, d as u32, kvh as u32, scale, ratio, true, o2,
                )?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let g = qh / kvh;
    let mut want = vec![0.0f32; rows * qh * d];
    for j in 0..kvh {
        let run: Vec<i32> = (0..rows)
            .flat_map(|r| sel[(r * kvh + j) * top_k..(r * kvh + j + 1) * top_k].to_vec())
            .collect();
        let (o, _) = sdpa(
            &q,
            rows,
            qh,
            kvh,
            d,
            d,
            &pool.keys,
            &pool.values,
            f64::from(scale),
            &|r| {
                let lane = fire.req[r] as usize;
                selected_kps(&fire, &run, top_k, ratio as i32, r)
                    .into_iter()
                    .map(|kp| pool.slot(lane, kp))
                    .collect()
            },
            &no_bias,
            &unit,
        );
        for r in 0..rows {
            let at = (r * qh + j * g) * d;
            want[at..at + g * d].copy_from_slice(&o[at..at + g * d]);
        }
    }
    check_rows(&b.read_f32(o1), &want, qh * d, rows - 1, 1e-3);
    check_rows(&b.read_f32(o2), &want, qh * d, rows - 1, 1e-3);
}

#[test]
fn a_relative_bias_adds_by_distance_and_scales_by_length() {
    let mut rng = Rng(8);
    let (qh, kvh, d) = (4, 2, 32);
    let pool = Pool::new(
        &mut rng,
        4,
        kvh,
        d,
        10,
        vec![vec![9, 0, 4, 2, 7], vec![1, 3], vec![5]],
        5,
    );
    let runs = [(0, 12, 6), (1, 3, 2), (2, 1, 1)];
    let fire = Fire::plain(&prefill_rows(&runs, 1));
    let rows = fire.positions.len();
    let extent = 6usize;
    let bias = rng.bf16s(rows * qh * extent, 2.0);
    let q = rng.bf16s(rows * qh * d, 1.0);
    let (scale, floor, alpha) = (0.17f32, 8u32, 0.3f32);
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let dplan = fire.decode(&mut b);
    let pplan = fire.prefill(&mut b);
    let bt = b.f32(rows as u32, (qh * extent) as u32, &bias);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let o1 = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let o2 = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let rel = attn::rel::RelBias {
        bias: bt,
        extent: extent as u32,
        log_floor: floor,
        log_alpha: alpha,
    };
    if !b
        .run(|ctx| {
            low::attend(
                ctx,
                attn::rel::decode_rel(qt, &dplan, rel, Some(9), d as u32, scale, o1)?,
                kv,
            )?;
            low::attend(
                ctx,
                attn::rel::prefill_rel(
                    qr, &pplan, &kv, rel, None, d as u32, kvh as u32, scale, o2,
                )?,
                kv,
            )
        })
        .unwrap()
    {
        return;
    }
    let kps = |r: usize, window: i32| -> Vec<i32> {
        let qpos = fire.positions[r];
        let lo = if window > 0 && qpos >= window {
            qpos - window + 1
        } else {
            0
        };
        (lo..=qpos).collect()
    };
    let mult = |r: usize| {
        let ratio = f64::from(fire.positions[r] + 1) / f64::from(floor);
        if ratio > 1.0 {
            1.0 + f64::from(alpha) * ratio.ln()
        } else {
            1.0
        }
    };
    for (window, got) in [(9, o1), (0, o2)] {
        let (want, _) = sdpa(
            &q,
            rows,
            qh,
            kvh,
            d,
            d,
            &pool.keys,
            &pool.values,
            f64::from(scale),
            &|r| {
                kps(r, window)
                    .into_iter()
                    .map(|kp| pool.slot(fire.req[r] as usize, kp as usize))
                    .collect()
            },
            &|r, h, j| {
                let dist = (fire.positions[r] - kps(r, window)[j]) as usize;
                if dist < extent {
                    f64::from(bias[(r * qh + h) * extent + dist])
                } else {
                    0.0
                }
            },
            &mult,
        );
        check_rows(&b.read_f32(got), &want, qh * d, rows - 1, 1e-3);
    }
}

#[test]
fn sink_merge_and_softcap_fold_rows() {
    let mut rng = Rng(4);
    let (rows, heads, d) = (5usize, 3usize, 16usize);
    let o1 = rng.bf16s(rows * heads * d, 1.0);
    let o2 = rng.bf16s(rows * heads * d, 1.0);
    let mut l1: Vec<f32> = rng.bf16s(rows * heads, 4.0);
    let mut l2: Vec<f32> = rng.bf16s(rows * heads, 4.0);
    l1[2] = f32::NEG_INFINITY;
    l2[4] = f32::NEG_INFINITY;
    let sinks = rng.bf16s(heads, 2.0);
    let x = rng.bf16s(rows * 7, 30.0);
    let mut b = Bench::new();
    let (r, w) = (rows as u32, (heads * d) as u32);
    let t1 = b.bf16(r, w, &o1);
    let t2 = b.bf16(r, w, &o2);
    let s1 = b.f32(r, heads as u32, &l1);
    let s2 = b.f32(r, heads as u32, &l2);
    let o = b.zeros(Dtype::Bf16, r, w);
    let lse = b.zeros(Dtype::F32, r, heads as u32);
    let sunk = b.bf16(r, w, &o1);
    let sk = b.bf16(1, heads as u32, &sinks);
    let xt = b.bf16(r, 7, &x);
    if !b
        .run(|ctx| {
            low::merge_lse(ctx, t1, s1, t2, s2, heads as u32, d as u32, o, lse)?;
            low::sink(ctx, sunk, s1, sk, d as u32)?;
            low::logit_softcap(ctx, xt, 12.5)
        })
        .unwrap()
    {
        return;
    }
    let mut want_o = vec![0.0; rows * heads * d];
    let mut want_l = vec![0.0; rows * heads];
    let mut want_s = vec![0.0; rows * heads * d];
    for c in 0..rows * heads {
        let (a, bb) = (l1[c], l2[c]);
        let (w1, w2, l) = if !bb.is_finite() {
            (1.0, 0.0, a)
        } else if !a.is_finite() {
            (0.0, 1.0, bb)
        } else {
            let m = a.max(bb);
            let (e1, e2) = ((a - m).exp2(), (bb - m).exp2());
            (e1 / (e1 + e2), e2 / (e1 + e2), m + (e1 + e2).log2())
        };
        want_l[c] = l;
        let rs = if a.is_finite() {
            1.0 / (1.0 + (-(a * std::f32::consts::LN_2 - sinks[c % heads])).exp())
        } else {
            1.0
        };
        for i in 0..d {
            want_o[c * d + i] = round_bf16(o1[c * d + i] * w1 + o2[c * d + i] * w2);
            want_s[c * d + i] = round_bf16(o1[c * d + i] * rs);
        }
    }
    assert_close(&b.read_f32(o), &want_o, 1e-2, 1e-2);
    assert_close(&b.read_f32(lse), &want_l, 1e-4, 1e-5);
    assert_close(&b.read_f32(sunk), &want_s, 1e-2, 1e-2);
    let want_x: Vec<f32> = x
        .iter()
        .map(|v| round_bf16(12.5 * (v / 12.5).tanh()))
        .collect();
    assert_close(&b.read_f32(xt), &want_x, 1e-2, 1e-2);
}

#[test]
fn kv_append_lands_each_row_in_its_slot_and_drops_the_padded_ones() {
    let mut rng = Rng(9);
    let (ps, kvh, d, pages) = (4usize, 2usize, 8usize, 5usize);
    let pool = Pool::new(&mut rng, ps, kvh, d, pages, vec![vec![0]], 1);
    let n = 6usize;
    let w = kvh * d;
    let k = rng.bf16s(n * w, 1.0);
    let v = rng.bf16s(n * w, 1.0);
    let page = vec![3, 0, 4, i32::MAX, 1, 5];
    let off = vec![1, 3, 0, 0, 2, 0];
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let shared_keys = b.bf16((pages * ps) as u32, w as u32, &pool.keys);
    let shared = KvPool {
        keys: shared_keys,
        values: shared_keys,
        ..kv
    };
    let kt = b.bf16(n as u32, w as u32, &k);
    let vt = b.bf16(n as u32, w as u32, &v);
    let pt = b.i32(n as u32, 1, &page);
    let ot = b.i32(n as u32, 1, &off);
    if !b
        .run(|ctx| {
            low::kv_append(ctx, kt, vt, &kv, pt, ot)?;
            low::kv_append_shared(ctx, vt, &shared, pt, ot)
        })
        .unwrap()
    {
        return;
    }
    let mut want_k = pool.keys.clone();
    let mut want_v = pool.values.clone();
    let mut want_s = pool.keys.clone();
    for i in 0..n {
        if page[i] < 0 || page[i] as usize >= pages {
            continue;
        }
        let slot = page[i] as usize * ps + off[i] as usize;
        want_k[slot * w..(slot + 1) * w].copy_from_slice(&k[i * w..(i + 1) * w]);
        want_v[slot * w..(slot + 1) * w].copy_from_slice(&v[i * w..(i + 1) * w]);
        want_s[slot * w..(slot + 1) * w].copy_from_slice(&v[i * w..(i + 1) * w]);
    }
    assert_close(&b.read_f32(kv.keys), &want_k, 0.0, 0.0);
    assert_close(&b.read_f32(kv.values), &want_v, 0.0, 0.0);
    assert_close(&b.read_f32(shared_keys), &want_s, 0.0, 0.0);
}

/// A prefill-tall fire's rows land a page at a time (`scatter_slots`): runs
/// that start and end inside a page keep its other rows, pages in any order,
/// a padded row dropped; and a page two runs write falls back to row writes.
#[test]
fn kv_append_writes_a_prefill_by_pages() {
    for twice in [false, true] {
        let mut rng = Rng(21);
        let (ps, kvh, d, pages) = (16usize, 2usize, 64usize, 12usize);
        let pool = Pool::new(&mut rng, ps, kvh, d, pages, vec![vec![0]], 1);
        let w = kvh * d;
        // lane a: 75 rows from offset 5 of page 3; lane b: 53 rows from page 1,
        // its last page (twice) the one lane a starts in, below lane a's rows
        let a = [3, 7, 2, 9, 0];
        let b = if twice { [1, 8, 4, 3] } else { [1, 8, 4, 6] };
        let (mut page, mut off) = (Vec::new(), Vec::new());
        for kp in 5..80 {
            page.push(a[kp / ps]);
            off.push((kp % ps) as i32);
        }
        for kp in 0..53 {
            page.push(b[kp / ps]);
            off.push((kp % ps) as i32);
        }
        // padded rows, to a whole page of rows (144)
        page.push(i32::MAX);
        off.push(0);
        while page.len() % ps != 0 {
            page.push(i32::MAX);
            off.push(0);
        }
        let n = page.len();
        let k = rng.bf16s(n * w, 1.0);
        let v = rng.bf16s(n * w, 1.0);
        let mut bench = Bench::new();
        let kv = pool.bind(&mut bench);
        let kt = bench.bf16(n as u32, w as u32, &k);
        let vt = bench.bf16(n as u32, w as u32, &v);
        let pt = bench.i32(n as u32, 1, &page);
        let ot = bench.i32(n as u32, 1, &off);
        if !bench
            .run(|ctx| low::kv_append(ctx, kt, vt, &kv, pt, ot).map(|_| ()))
            .unwrap()
        {
            return;
        }
        let mut want_k = pool.keys.clone();
        let mut want_v = pool.values.clone();
        for i in 0..n {
            if page[i] < 0 || page[i] as usize >= pages {
                continue;
            }
            let slot = page[i] as usize * ps + off[i] as usize;
            want_k[slot * w..(slot + 1) * w].copy_from_slice(&k[i * w..(i + 1) * w]);
            want_v[slot * w..(slot + 1) * w].copy_from_slice(&v[i * w..(i + 1) * w]);
        }
        assert_close(&bench.read_f32(kv.keys), &want_k, 0.0, 0.0);
        assert_close(&bench.read_f32(kv.values), &want_v, 0.0, 0.0);
    }
}

/// A prefill that appended its rows first reads them through the flash
/// kernel when every row's keys are fresh, and walks the pool otherwise: a
/// fire of three fresh lanes (one spanning several key blocks), a lone row
/// and pads, plain and under a window, against the host reference; a fire
/// with a lane that has positions before it falls back, and still matches.
/// The fresh rows handed over are another append's (to a twin pool), so a
/// fresh read is told from a walk by what it answers.
#[test]
fn a_tall_wide_prefill_lays_its_kv_in_parts() {
    // 1792 rows of 16 kv heads of 256: past one kernel's VMEM, laid in two
    // parts; lane a's page 56 holds rows of both.
    let mut rng = Rng(23);
    let (ps, kvh, d, pages) = (16usize, 16usize, 256usize, 130usize);
    let pool = Pool::new(&mut rng, ps, kvh, d, pages, vec![vec![0]], 1);
    let w = kvh * d;
    let (mut page, mut off) = (Vec::new(), Vec::new());
    for kp in 5..905 {
        page.push((kp / ps * 2) as i32);
        off.push((kp % ps) as i32);
    }
    for kp in 0..880 {
        page.push((kp / ps * 2 + 1) as i32);
        off.push((kp % ps) as i32);
    }
    while page.len() % ps != 0 {
        page.push(i32::MAX);
        off.push(0);
    }
    let n = page.len();
    assert_eq!(n, 1792);
    let k = rng.bf16s(n * w, 1.0);
    let v = rng.bf16s(n * w, 1.0);
    let mut bench = Bench::new();
    let kv = pool.bind(&mut bench);
    let kt = bench.bf16(n as u32, w as u32, &k);
    let vt = bench.bf16(n as u32, w as u32, &v);
    let pt = bench.i32(n as u32, 1, &page);
    let ot = bench.i32(n as u32, 1, &off);
    if !bench
        .run(|ctx| low::kv_append(ctx, kt, vt, &kv, pt, ot).map(|_| ()))
        .unwrap()
    {
        return;
    }
    let mut want_k = pool.keys.clone();
    let mut want_v = pool.values.clone();
    for i in 0..n {
        if page[i] < 0 || page[i] as usize >= pages {
            continue;
        }
        let slot = page[i] as usize * ps + off[i] as usize;
        want_k[slot * w..(slot + 1) * w].copy_from_slice(&k[i * w..(i + 1) * w]);
        want_v[slot * w..(slot + 1) * w].copy_from_slice(&v[i * w..(i + 1) * w]);
    }
    assert_close(&bench.read_f32(kv.keys), &want_k, 0.0, 0.0);
    assert_close(&bench.read_f32(kv.values), &want_v, 0.0, 0.0);
}

#[test]
fn a_prefill_reads_the_rows_it_appended_fresh() {
    let mut rng = Rng(31);
    let (qh, kvh, d, ps) = (4usize, 2usize, 128usize, 16usize);
    let lanes: Vec<Vec<i32>> = vec![
        (0..40).collect(),
        (40..48).collect(),
        (48..52).collect(),
        (52..56).collect(),
    ];
    let mut pool = Pool::new(&mut rng, ps, kvh, d, 60, lanes, 40);
    let runs = [(0, 0, 600), (1, 0, 100), (2, 0, 1), (3, 0, 51)];
    let fire = Fire::plain(&prefill_rows(&runs, 16));
    let rows = fire.positions.len();
    assert_eq!(rows % 16, 0);
    let w = kvh * d;
    let q = rng.bf16s(rows * qh * d, 1.0);
    let k = rng.bf16s(rows * w, 1.0);
    let v = rng.bf16s(rows * w, 1.0);
    let k2 = rng.bf16s(rows * w, 1.0);
    let v2 = rng.bf16s(rows * w, 1.0);
    let sink = (pool.slots / ps - 1) as i32;
    let (mut page, mut off) = (Vec::new(), Vec::new());
    for (&lane, &kp) in fire.req.iter().zip(&fire.positions) {
        if kp < 0 {
            page.push(sink);
            off.push(0);
        } else {
            let slot = pool.slot(lane as usize, kp as usize);
            page.push((slot / ps) as i32);
            off.push((slot % ps) as i32);
        }
    }
    let scale = 0.1;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let twin = pool.bind(&mut b);
    let plan = fire.prefill(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let kt = b.bf16(rows as u32, w as u32, &k);
    let vt = b.bf16(rows as u32, w as u32, &v);
    let kt2 = b.bf16(rows as u32, w as u32, &k2);
    let vt2 = b.bf16(rows as u32, w as u32, &v2);
    let pt = b.i32(rows as u32, 1, &page);
    let ot = b.i32(rows as u32, 1, &off);
    // The same fire with lane 1 holding 20 positions before it: a walk.
    let mut shifted = fire.positions.clone();
    for (p, &l) in shifted.iter_mut().zip(&fire.req) {
        if l == 1 && *p >= 0 {
            *p += 20;
        }
    }
    let old = Fire {
        positions: shifted,
        ..Fire::plain(&prefill_rows(&runs, 16))
    };
    let plan_old = old.prefill(&mut b);
    let (mut page_old, mut off_old) = (page.clone(), off.clone());
    for (i, (&lane, &kp)) in old.req.iter().zip(&old.positions).enumerate() {
        if kp >= 0 {
            let slot = pool.slot(lane as usize, kp as usize);
            page_old[i] = (slot / ps) as i32;
            off_old[i] = (slot % ps) as i32;
        }
    }
    let pt_old = b.i32(rows as u32, 1, &page_old);
    let ot_old = b.i32(rows as u32, 1, &off_old);
    let wq = (qh * d) as u32;
    let o_fresh = b.zeros(Dtype::Bf16, rows as u32, wq);
    let o_win = b.zeros(Dtype::Bf16, rows as u32, wq);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    let o_old = b.zeros(Dtype::Bf16, rows as u32, wq);
    let o_old_win = b.zeros(Dtype::Bf16, rows as u32, wq);
    if !b
        .run(|ctx| {
            low::kv_append(ctx, kt, vt, &kv, pt, ot)?;
            let read = low::kv_append(ctx, kt2, vt2, &twin, pt, ot)?;
            let call = arbiter::prefill_lse(
                qr, &plan, &kv, None, d as u32, kvh as u32, scale, o_fresh, lse, 1000,
            )?;
            low::attend_fresh(ctx, call, kv, read)?;
            let call = arbiter::prefill(
                qr,
                &plan,
                &kv,
                Some(100),
                d as u32,
                kvh as u32,
                scale,
                o_win,
                1000,
            )?;
            low::attend_fresh(ctx, call, kv, read)?;
            // The shifted fire appends its rows to the pool itself: its
            // lane 1 reads 20 older keys there, beside its own rows.
            let read = low::kv_append(ctx, kt2, vt2, &kv, pt_old, ot_old)?;
            let call = arbiter::prefill(
                qr, &plan_old, &kv, None, d as u32, kvh as u32, scale, o_old, 1000,
            )?;
            low::attend_fresh(ctx, call, kv, read)?;
            let call = arbiter::prefill(
                qr,
                &plan_old,
                &kv,
                Some(10),
                d as u32,
                kvh as u32,
                scale,
                o_old_win,
                1000,
            )?;
            low::attend_fresh(ctx, call, kv, read)
        })
        .unwrap()
    {
        return;
    }
    let live = rows - 16;
    let first = Pool {
        keys: pool.keys.clone(),
        values: pool.values.clone(),
        lanes: pool.lanes.clone(),
        ..pool
    };
    // The device pool holds the first append, then the shifted fire's.
    for (rows_k, rows_v, pg, of) in [(&k, &v, &page, &off), (&k2, &v2, &page_old, &off_old)] {
        for (i, &p) in pg.iter().enumerate() {
            if p != sink {
                let slot = p as usize * ps + of[i] as usize;
                pool.keys[slot * w..(slot + 1) * w].copy_from_slice(&rows_k[i * w..(i + 1) * w]);
                pool.values[slot * w..(slot + 1) * w].copy_from_slice(&rows_v[i * w..(i + 1) * w]);
            }
        }
    }
    let (want_old, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| old.admitted(&pool, r, 0, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o_old), &want_old, qh * d, live, 2e-3);
    let (want_old_win, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &|r| old.admitted(&pool, r, 10, true),
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o_old_win), &want_old_win, qh * d, live, 2e-3);
    // The first fire read its fresh rows as the twin's append left them.
    let mut pool = first;
    for (i, &p) in page.iter().enumerate() {
        if p != sink {
            let slot = p as usize * ps + off[i] as usize;
            pool.keys[slot * w..(slot + 1) * w].copy_from_slice(&k2[i * w..(i + 1) * w]);
            pool.values[slot * w..(slot + 1) * w].copy_from_slice(&v2[i * w..(i + 1) * w]);
        }
    }
    for (window, o) in [(0, o_fresh), (100, o_win)] {
        let (want, want_lse) = sdpa(
            &q,
            rows,
            qh,
            kvh,
            d,
            d,
            &pool.keys,
            &pool.values,
            f64::from(scale),
            &|r| fire.admitted(&pool, r, window, true),
            &no_bias,
            &unit,
        );
        check_rows(&b.read_f32(o), &want, qh * d, live, 2e-3);
        if window == 0 {
            assert_close(
                &b.read_f32(lse)[..live * qh],
                &want_lse[..live * qh],
                2e-3,
                1e-4,
            );
        }
    }
}

#[test]
fn a_readout_reads_its_lanes_fresh_rows_or_walks_their_pages() {
    let mut rng = Rng(37);
    let (qh, kvh, d, ps) = (4usize, 2usize, 128usize, 16usize);
    let lanes: Vec<Vec<i32>> = vec![
        (0..40).collect(),
        (40..48).collect(),
        (48..52).collect(),
        (52..56).collect(),
    ];
    let mut pool = Pool::new(&mut rng, ps, kvh, d, 60, lanes, 40);
    let runs = [(0, 0, 600), (1, 0, 100), (2, 0, 1), (3, 0, 51)];
    let fire = Fire::plain(&prefill_rows(&runs, 16));
    let rows = fire.positions.len();
    // Each lane's last row, read out (lane 1's twice, as a padded lane's).
    let readouts: Vec<i32> = vec![599, 699, 700, 751, 699, 699, 699, 699];
    let r = readouts.len();
    let w = kvh * d;
    let q = rng.bf16s(rows * qh * d, 1.0);
    let k = rng.bf16s(rows * w, 1.0);
    let v = rng.bf16s(rows * w, 1.0);
    let k2 = rng.bf16s(rows * w, 1.0);
    let v2 = rng.bf16s(rows * w, 1.0);
    let sink = (pool.slots / ps - 1) as i32;
    let slots_of = |fire: &Fire, pool: &Pool| {
        let (mut page, mut off) = (Vec::new(), Vec::new());
        for (&lane, &kp) in fire.req.iter().zip(&fire.positions) {
            if kp < 0 {
                page.push(sink);
                off.push(0);
            } else {
                let slot = pool.slot(lane as usize, kp as usize);
                page.push((slot / ps) as i32);
                off.push((slot % ps) as i32);
            }
        }
        (page, off)
    };
    // The same fire with lane 1 holding 20 positions before it, two pages
    // the older rectangle holds; and one whose lane 0 holds 300, past it: a
    // walk.
    let runs_far = [(0, 300, 300), (1, 0, 100), (2, 0, 1), (3, 0, 51)];
    let far = Fire::plain(&prefill_rows(&runs_far, rows - 452));
    let readouts_far: Vec<i32> = vec![299, 399, 400, 451, 399, 399, 399, 399];
    let mut shifted = fire.positions.clone();
    for (p, &l) in shifted.iter_mut().zip(&fire.req) {
        if l == 1 && *p >= 0 {
            *p += 20;
        }
    }
    let old = Fire {
        positions: shifted,
        ..Fire::plain(&prefill_rows(&runs, 16))
    };
    let (page, off) = slots_of(&fire, &pool);
    let (page_old, off_old) = slots_of(&old, &pool);
    let (page_far, off_far) = slots_of(&far, &pool);
    let pick = |readouts: &[i32]| -> Vec<f32> {
        readouts
            .iter()
            .flat_map(|&t| q[t as usize * qh * d..(t as usize + 1) * qh * d].to_vec())
            .collect()
    };
    let qr = pick(&readouts);
    let qr_far = pick(&readouts_far);
    let scale = 0.1;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let twin = pool.bind(&mut b);
    let far_kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let plan_old = old.decode(&mut b);
    let plan_far = far.decode(&mut b);
    let qt = b.bf16(r as u32, (qh * d) as u32, &qr);
    let qt_far = b.bf16(r as u32, (qh * d) as u32, &qr_far);
    let kt = b.bf16(rows as u32, w as u32, &k);
    let vt = b.bf16(rows as u32, w as u32, &v);
    let kt2 = b.bf16(rows as u32, w as u32, &k2);
    let vt2 = b.bf16(rows as u32, w as u32, &v2);
    let pt = b.i32(rows as u32, 1, &page);
    let ot = b.i32(rows as u32, 1, &off);
    let pt_old = b.i32(rows as u32, 1, &page_old);
    let ot_old = b.i32(rows as u32, 1, &off_old);
    let pt_far = b.i32(rows as u32, 1, &page_far);
    let ot_far = b.i32(rows as u32, 1, &off_far);
    let at = b.i32(r as u32, 1, &readouts);
    let at_far = b.i32(r as u32, 1, &readouts_far);
    let wq = (qh * d) as u32;
    let outs: Vec<Tensor> = (0..6).map(|_| b.zeros(Dtype::Bf16, r as u32, wq)).collect();
    if !b
        .run(|ctx| {
            // The fresh rows read as appended, not as the twin pool holds
            // them: a walk there would read other keys.
            let read = low::kv_append(ctx, kt, vt, &kv, pt, ot)?;
            assert!(read.is_some(), "a 768-row append is read fresh");
            low::kv_append(ctx, kt2, vt2, &twin, pt, ot)?;
            for (window, o) in [(None, outs[0]), (Some(100), outs[1])] {
                let call = attn::decode(qt, &plan, window, d as u32, scale, o)?;
                low::attend_readouts(
                    ctx,
                    call,
                    twin,
                    read,
                    at,
                    plan.positions,
                    plan.request_of_token,
                )?;
            }
            // The shifted fire's lane 1 reads its 20 older keys from the pool.
            let read = low::kv_append(ctx, kt, vt, &kv, pt_old, ot_old)?;
            for (window, o) in [(None, outs[2]), (Some(10), outs[3])] {
                let call = attn::decode(qt, &plan_old, window, d as u32, scale, o)?;
                low::attend_readouts(
                    ctx,
                    call,
                    kv,
                    read,
                    at,
                    plan_old.positions,
                    plan_old.request_of_token,
                )?;
            }
            let read = low::kv_append(ctx, kt, vt, &far_kv, pt_far, ot_far)?;
            for (window, o) in [(None, outs[4]), (Some(100), outs[5])] {
                let call = attn::decode(qt_far, &plan_far, window, d as u32, scale, o)?;
                low::attend_readouts(
                    ctx,
                    call,
                    far_kv,
                    read,
                    at_far,
                    plan_far.positions,
                    plan_far.request_of_token,
                )?;
            }
            Ok(())
        })
        .unwrap()
    {
        return;
    }
    let first = Pool {
        keys: pool.keys.clone(),
        values: pool.values.clone(),
        lanes: pool.lanes.clone(),
        ..pool
    };
    // What each fire left in its pool: the fresh fire's rows, then the
    // shifted fire's over them.
    let lay = |pool: &mut Pool, pg: &[i32], of: &[i32]| {
        for (i, &p) in pg.iter().enumerate() {
            if p != sink {
                let slot = p as usize * ps + of[i] as usize;
                pool.keys[slot * w..(slot + 1) * w].copy_from_slice(&k[i * w..(i + 1) * w]);
                pool.values[slot * w..(slot + 1) * w].copy_from_slice(&v[i * w..(i + 1) * w]);
            }
        }
    };
    let mut far_pool = Pool {
        keys: first.keys.clone(),
        values: first.values.clone(),
        lanes: first.lanes.clone(),
        ..first
    };
    lay(&mut far_pool, &page_far, &off_far);
    let mut fresh = first;
    lay(&mut fresh, &page, &off);
    lay(&mut pool, &page, &off);
    lay(&mut pool, &page_old, &off_old);
    for (fire, pool, readouts, window, o) in [
        (&fire, &fresh, &readouts, 0, outs[0]),
        (&fire, &fresh, &readouts, 100, outs[1]),
        (&old, &pool, &readouts, 0, outs[2]),
        (&old, &pool, &readouts, 10, outs[3]),
        (&far, &far_pool, &readouts_far, 0, outs[4]),
        (&far, &far_pool, &readouts_far, 100, outs[5]),
    ] {
        let (want, _) = sdpa(
            &q,
            rows,
            qh,
            kvh,
            d,
            d,
            &pool.keys,
            &pool.values,
            f64::from(scale),
            &|r| fire.admitted(pool, r, window, true),
            &no_bias,
            &unit,
        );
        let want: Vec<f32> = readouts
            .iter()
            .flat_map(|&t| want[t as usize * qh * d..(t as usize + 1) * qh * d].to_vec())
            .collect();
        check_rows(&b.read_f32(o), &want, qh * d, r, 2e-3);
    }
}

#[test]
fn dense_attends_within_each_image_segment() {
    let mut rng = Rng(12);
    let (qh, kvh, d) = (4, 2, 24);
    let rows = 13usize;
    let seg = [2i32, 6, 6, 11];
    let q = rng.bf16s(rows * qh * d, 1.0);
    let k = rng.bf16s(rows * kvh * d, 1.0);
    let v = rng.bf16s(rows * kvh * d, 1.0);
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let kt = b.bf16(rows as u32, (kvh * d) as u32, &k);
    let vt = b.bf16(rows as u32, (kvh * d) as u32, &v);
    let st = b.i32(seg.len() as u32, 1, &seg);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    if !b
        .run(|ctx| low::dense::bidirectional(ctx, qt, kt, vt, st, d as u32, 0.2, o))
        .unwrap()
    {
        return;
    }
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        d,
        &k,
        &v,
        f64::from(0.2f32),
        &|r| {
            let r = r as i32;
            match seg.windows(2).find(|w| w[0] <= r && r < w[1]) {
                Some(w) => (w[0] as usize..w[1] as usize).collect(),
                None => Vec::new(),
            }
        },
        &no_bias,
        &unit,
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 2e-3);
}

#[test]
fn ragged_answers_every_mask_form() {
    use attn::ragged::RaggedMask;
    use low::ragged::forward;
    let mut rng = Rng(13);
    let (qh, kvh, d) = (4, 2, 32);
    let qi = [0i32, 3, 3, 8];
    let ki = [0i32, 5, 7, 12];
    let (rows, keys) = (10usize, 12usize);
    let q = rng.bf16s(rows * qh * d, 1.0);
    let k = rng.bf16s(keys * kvh * d, 1.0);
    let v = rng.bf16s(keys * kvh * d, 1.0);
    let q_cls = [0i32, 1, -1, 1, 0, 1, 1, 0, 0, 0];
    let kv_cls = [1i32, 0, 1, -1, 1, 0, 0, 1, 0, 1, -1, 0];
    let table = [1u8, 0, 0, 1];
    let max_len = 4usize;
    let span = 2 * max_len - 1;
    let bias = rng.bf16s(qh * span, 2.0);
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let kt = b.bf16(keys as u32, (kvh * d) as u32, &k);
    let vt = b.bf16(keys as u32, (kvh * d) as u32, &v);
    let qit = b.i32(4, 1, &qi);
    let kit = b.i32(4, 1, &ki);
    let class_keeps = |r: usize, j: usize| {
        q_cls[r] < 0 || kv_cls[j] < 0 || table[(q_cls[r] * 2 + kv_cls[j]) as usize] != 0
    };
    // The classes' rule as a custom mask, each segment's bits from a byte of its own.
    let (mut bits, mut starts) = (Vec::<u8>::new(), Vec::<i32>::new());
    for s in 0..3 {
        starts.push(bits.len() as i32);
        let (q0, q1, k0, k1) = (
            qi[s] as usize,
            qi[s + 1] as usize,
            ki[s] as usize,
            ki[s + 1] as usize,
        );
        let base = bits.len();
        bits.resize(base + ((q1 - q0) * (k1 - k0)).div_ceil(8), 0);
        for (qo, r) in (q0..q1).enumerate() {
            for (kv, j) in (k0..k1).enumerate() {
                if class_keeps(r, j) {
                    let cell = qo * (k1 - k0) + kv;
                    bits[base + cell / 8] |= 1 << (cell % 8);
                }
            }
        }
    }
    starts.push(bits.len() as i32);
    bits.push(0);
    let bitt = b.u8(bits.len() as u32, 1, &bits);
    let startt = b.i32(starts.len() as u32, 1, &starts);
    let bt = b.f32(qh as u32, span as u32, &bias);
    let masks = [
        RaggedMask::Segments,
        RaggedMask::Custom {
            bits: bitt,
            indptr: startt,
        },
        RaggedMask::RelativeBias {
            table: bt,
            max_len: max_len as u32,
        },
    ];
    let outs: Vec<Tensor> = masks
        .iter()
        .map(|_| b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32))
        .collect();
    if !b
        .run(|ctx| {
            for (m, &o) in masks.iter().zip(&outs) {
                forward(ctx, qt, kt, vt, qit, kit, d as u32, kvh as u32, 0.18, m, o)?;
            }
            Ok(())
        })
        .unwrap()
    {
        return;
    }
    let seg_of = |r: usize| (0..3).find(|&s| qi[s] as usize <= r && r < qi[s + 1] as usize);
    let span_of = |r: usize| seg_of(r).map_or(0..0, |s| ki[s] as usize..ki[s + 1] as usize);
    let keep = |which: usize, r: usize, j: usize| match which {
        1 => class_keeps(r, j),
        _ => true,
    };
    for (which, &o) in outs.iter().enumerate() {
        let (want, _) = sdpa(
            &q,
            rows,
            qh,
            kvh,
            d,
            d,
            &k,
            &v,
            f64::from(0.18f32),
            &|r| span_of(r).filter(|&j| keep(which, r, j)).collect(),
            &|r, h, j| {
                if which != 2 {
                    return 0.0;
                }
                let s = seg_of(r).unwrap();
                let (kq, qq) = (j as i64, (r - qi[s] as usize) as i64);
                let at = (kq - qq + max_len as i64 - 1).clamp(0, span as i64 - 1) as usize;
                f64::from(bias[h * span + at])
            },
            &unit,
        );
        check_rows(&b.read_f32(o), &want, qh * d, rows, 2e-3);
    }
}

#[test]
fn score_capture_lands_the_mean_softmax_rows() {
    let mut rng = Rng(14);
    let (qh, kvh, d, ps) = (4usize, 2usize, 32usize, 8usize);
    let pool = Pool::new(
        &mut rng,
        ps,
        kvh,
        d,
        8,
        vec![vec![4, 1, 6], vec![0, 7], vec![2]],
        3,
    );
    let runs = [(0, 18, 5), (1, 3, 2), (2, 2, 1)];
    let fire = Fire::plain(&prefill_rows(&runs, 1));
    let rows = fire.positions.len();
    let (observe, lane_offset, plane_stride, plane, kv_max, requests) =
        (3u32, 1u32, 6u32, 1u32, 20u32, 2u32);
    let slab_rows = (lane_offset + requests + 1) * plane_stride;
    let slab = rng.bf16s((slab_rows * kv_max) as usize, 1.0);
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.2f32;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.prefill(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let st = b.f32(slab_rows, kv_max, &slab);
    if !b
        .run(|ctx| {
            capture(
                ctx,
                qr,
                &plan,
                &kv,
                None,
                d as u32,
                kvh as u32,
                scale,
                observe,
                lane_offset,
                plane_stride,
                plane,
                kv_max,
                requests,
                st,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want = slab.clone();
    let qo = [0usize, 5, 7, 8];
    for r in 0..requests as usize {
        let n = (observe as usize).min(qo[r + 1] - qo[r]);
        for h in 0..qh {
            let row = ((lane_offset as usize + r) * plane_stride as usize + plane as usize + h)
                * kv_max as usize;
            let mut acc = vec![0.0f64; kv_max as usize];
            for w in 0..n {
                let qix = qo[r + 1] - n + w;
                let limit = ((fire.positions[qix] + 1) as usize).min(pool.capacity(r));
                let s: Vec<f64> = (0..limit)
                    .map(|j| {
                        let slot = pool.slot(r, j);
                        (0..d)
                            .map(|i| {
                                f64::from(q[(qix * qh + h) * d + i])
                                    * f64::from(pool.keys[(slot * kvh + h / 2) * d + i])
                            })
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
                for (j, x) in s.iter().enumerate().take(kv_max as usize) {
                    acc[j] += (x - m).exp() / z / n as f64;
                }
            }
            for j in 0..kv_max as usize {
                want[row + j] = acc[j] as f32;
            }
        }
    }
    assert_close(&b.read_f32(st), &want, 1e-5, 1e-3);
}
