//! The paged attention family against host references.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::attn;
use kernels_cerebras::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Tensor};

/// What each test does first: nothing here; `kernels_attn_windows` includes
/// this suite and caps the rows an attention phase takes instead.
#[allow(dead_code)]
fn before() {}

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
        (self.lanes[lane].len() * self.ps).min(self.max_pages as usize * self.ps)
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

/// Softmax attention of every row over the keys `keys(r)` lists, in f64.
fn sdpa(
    q: &[f32],
    rows: usize,
    qh: usize,
    kvh: usize,
    d: usize,
    k: &[f32],
    v: &[f32],
    scale: f64,
    keys: &dyn Fn(usize) -> Vec<usize>,
) -> (Vec<f32>, Vec<f32>) {
    let g = qh / kvh;
    let mut o = vec![0.0f32; rows * qh * d];
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
                .map(|&kr| {
                    let mut dot = 0.0f64;
                    for i in 0..d {
                        dot += f64::from(q[(r * qh + h) * d + i])
                            * f64::from(k[(kr * kvh + kh) * d + i]);
                    }
                    dot * scale
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = e.iter().sum();
            for i in 0..d {
                let mut acc = 0.0f64;
                for (j, &kr) in list.iter().enumerate() {
                    acc += e[j] * f64::from(v[(kr * kvh + kh) * d + i]);
                }
                o[(r * qh + h) * d + i] = round_bf16((acc / l) as f32);
            }
            lse[r * qh + h] = (m * std::f64::consts::LOG2_E + l.log2()) as f32;
        }
    }
    (o, lse)
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

fn check_rows(got: &[f32], want: &[f32], width: usize, rows: usize, atol: f32) {
    assert_close(&got[..rows * width], &want[..rows * width], atol, 1e-2);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "a padded row answered a non-finite value"
    );
}

#[test]
fn decode_reads_each_rows_lane_through_its_pages() {
    crate::before();
    let mut rng = Rng(7);
    let (qh, kvh, d) = (6, 2, 16);
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
            attn::decode_lse(ctx, qt, &plan, &kv, None, d as u32, scale as f32, o, lse)?;
            attn::decode(ctx, qt, &plan, &kv, Some(3), d as u32, scale as f32, ow)
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    assert_close(&b.read_f32(lse)[..3 * qh], &want_lse[..3 * qh], 1e-3, 1e-3);
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 3, true),
    );
    check_rows(&b.read_f32(ow), &want, qh * d, rows, 1e-2);
}

#[test]
fn prefill_walks_ragged_lanes_causally() {
    crate::before();
    let mut rng = Rng(11);
    let (qh, kvh, d, ps) = (4, 2, 8, 4);
    let pool = Pool::new(
        &mut rng,
        ps,
        kvh,
        d,
        12,
        vec![vec![3, 9, 0, 4], vec![7, 1], vec![10, 2, 5]],
        4,
    );
    let runs = [(0, 10, 5), (1, 0, 6), (2, 9, 1)];
    let rows_desc = prefill_rows(&runs, 2);
    let fire = Fire::plain(&rows_desc);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.35;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.prefill(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let ow = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| {
            attn::prefill_lse(
                ctx, qr, &plan, &kv, None, d as u32, kvh as u32, scale, o, lse,
            )?;
            attn::prefill(
                ctx,
                qr,
                &plan,
                &kv,
                Some(4),
                d as u32,
                kvh as u32,
                scale,
                ow,
            )
        })
        .unwrap()
    {
        return;
    }
    let live = rows - 2;
    let (want, want_lse) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        &pool.keys,
        &pool.values,
        scale as f64,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    assert_close(
        &b.read_f32(lse)[..live * qh],
        &want_lse[..live * qh],
        1e-3,
        1e-3,
    );
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        &pool.keys,
        &pool.values,
        scale as f64,
        &|r| fire.admitted(&pool, r, 4, true),
    );
    check_rows(&b.read_f32(ow), &want, qh * d, rows, 1e-2);
}

#[test]
fn a_custom_mask_gates_keys_and_flag_two_sees_the_whole_lane() {
    crate::before();
    let mut rng = Rng(5);
    let (qh, kvh, d, ps) = (2, 2, 8, 4);
    let pool = Pool::new(&mut rng, ps, kvh, d, 6, vec![vec![1, 4], vec![0, 3, 5]], 3);
    let runs = [(0, 3, 3), (1, 2, 2)];
    let rows_desc = prefill_rows(&runs, 1);
    let mut fire = Fire::plain(&rows_desc);
    let rows = fire.positions.len();
    fire.stride = 12;
    fire.enabled = vec![1, 1, 2, 0, 1, 1];
    fire.mask = (0..rows * 12)
        .map(|i| ((i * 7 + 3) % 5 != 0) as u8)
        .collect();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 0.5;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.prefill(&mut b);
    let mask = b.u8(rows as u32, 12, &fire.mask);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let qr = ragged(&mut b, &runs, qt);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let on = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| {
            attn::masked_lse(
                ctx, qr, &plan, mask, &kv, None, true, d as u32, scale, o, lse,
            )?;
            attn::masked(ctx, qr, &plan, mask, &kv, None, false, d as u32, scale, on)
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
        &pool.keys,
        &pool.values,
        scale as f64,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    let live = rows - 1;
    assert_close(
        &b.read_f32(lse)[..live * qh],
        &want_lse[..live * qh],
        1e-3,
        1e-3,
    );
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        &pool.keys,
        &pool.values,
        scale as f64,
        &|r| fire.admitted(&pool, r, 0, false),
    );
    check_rows(&b.read_f32(on), &want, qh * d, rows, 1e-2);
}

#[test]
fn kv_append_lands_each_row_in_its_slot_and_drops_the_padded_ones() {
    crate::before();
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
            attn::kv_append(ctx, kt, vt, &kv, pt, ot)?;
            attn::kv_append_shared(ctx, vt, &shared, pt, ot)
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

/// A pool past one PE's share spreads its lanes (and their pages) over
/// PEs; every row still reads its own lane's pages.
#[test]
fn decode_over_a_wide_pool_spreads_lanes_over_pes() {
    crate::before();
    let mut rng = Rng(21);
    let (qh, kvh, d) = (4, 2, 64);
    // 9 pages x 4 rows x 128 wide: 4608 words a plane, past shard_words().
    let pool = Pool::new(
        &mut rng,
        4,
        kvh,
        d,
        9,
        vec![vec![5, 1, 7], vec![2], vec![0, 6, 8]],
        4,
    );
    // A lane's rows are contiguous: lane 0's live row and its padded row sit together.
    let fire = Fire::plain(&[(0, 9), (0, -1), (1, 2), (2, 11)]);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| attn::decode_lse(ctx, qt, &plan, &kv, None, d as u32, scale as f32, o, lse))
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    let got_lse = b.read_f32(lse);
    for r in [0usize, 2, 3] {
        assert_close(
            &got_lse[r * qh..(r + 1) * qh],
            &want_lse[r * qh..(r + 1) * qh],
            1e-3,
            1e-3,
        );
    }
    assert!(got_lse[qh..2 * qh].iter().all(|x| *x == f32::NEG_INFINITY));
}

/// A request whose pages exceed a PE's share splits them over PEs, each
/// attending its own pages; the host merges the partial outputs by their
/// per-head log-sum-exp.
#[test]
fn decode_over_a_long_request_splits_its_pages_over_pes() {
    crate::before();
    let mut rng = Rng(25);
    let (qh, kvh, d) = (4, 2, 64);
    // 12 pages x 4 rows x 128 wide; a lane may hold 8 pages (8 x 1024 words
    // of keys and values), twice a PE's share: two page groups.
    let pool = Pool::new(
        &mut rng,
        4,
        kvh,
        d,
        12,
        vec![vec![5, 1, 7, 2, 0, 6, 8, 9], vec![3, 4, 10]],
        8,
    );
    // Lane 0's rows reach into both page groups; lane 1's second group is empty.
    let fire = Fire::plain(&[(0, 30), (0, 27), (1, 2)]);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| attn::decode_lse(ctx, qt, &plan, &kv, None, d as u32, scale as f32, o, lse))
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    assert_close(&b.read_f32(lse), &want_lse, 1e-3, 1e-3);
}

/// A page too wide for a PE splits its kv heads over PEs (with its pages
/// over page groups): each PE attends its heads over its pages.
#[test]
fn decode_over_wide_pages_splits_kv_heads_over_pes() {
    crate::before();
    let mut rng = Rng(27);
    let (qh, kvh, d) = (8, 4, 64);
    // 16-row pages of 4 x 64: 8192 words of keys and values a page.
    let pool = Pool::new(&mut rng, 16, kvh, d, 6, vec![vec![3, 0, 5], vec![1, 4]], 3);
    let fire = Fire::plain(&[(0, 40), (0, 7), (1, 20)]);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| attn::decode_lse(ctx, qt, &plan, &kv, None, d as u32, scale as f32, o, lse))
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    assert_close(&b.read_f32(lse), &want_lse, 1e-3, 1e-3);
}

/// A page taller than a PE (32 rows x 256 of one kv head) splits its rows
/// over PEs, each attending a strided subset of the keys, merged by lse.
#[test]
fn decode_over_tall_pages_splits_page_rows_over_pes() {
    crate::before();
    let mut rng = Rng(28);
    let (qh, kvh, d) = (2, 1, 256);
    // 32-row pages of 256: 16384 words of keys and values a page.
    let pool = Pool::new(&mut rng, 32, kvh, d, 3, vec![vec![2, 0], vec![1]], 2);
    let fire = Fire::plain(&[(0, 50), (0, 33), (1, 20)]);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, qh as u32);
    if !b
        .run(|ctx| attn::decode_lse(ctx, qt, &plan, &kv, None, d as u32, scale as f32, o, lse))
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
    assert_close(&b.read_f32(lse), &want_lse, 1e-3, 1e-3);
}

/// The same split without an lse output: the kernel writes one for the
/// merge anyway.
#[test]
fn decode_over_a_long_request_merges_without_an_lse_output() {
    crate::before();
    let mut rng = Rng(26);
    let (qh, kvh, d) = (2, 1, 64);
    let pool = Pool::new(
        &mut rng,
        4,
        kvh,
        d,
        12,
        vec![vec![5, 1, 7, 2, 0, 6, 8, 9, 3, 4, 10, 11]],
        12,
    );
    let fire = Fire::plain(&[(0, 46), (0, 9)]);
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let scale = 1.0 / (d as f64).sqrt();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    if !b
        .run(|ctx| attn::decode(ctx, qt, &plan, &kv, None, d as u32, scale as f32, o))
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
        &pool.keys,
        &pool.values,
        scale,
        &|r| fire.admitted(&pool, r, 0, true),
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows, 1e-2);
}

/// A kv_append whose page pair exceeds a PE splits the page columns over
/// PEs, each writing its columns of its rows into its column block of the
/// pages.
#[test]
fn kv_append_over_wide_pages_splits_columns_over_pes() {
    crate::before();
    kv_append_case(16, 4, 64, 3, 30);
}

/// A kv_append into a wide pool spreads its rows over PEs, each holding the
/// pages its rows write.
#[test]
fn kv_append_over_a_wide_pool_spreads_rows_over_pes() {
    crate::before();
    kv_append_case(4, 2, 64, 5, 23); // 5 x 4 x 128 = 2560 a plane
}

/// `n` rows appended into a `pages` x `ps` pool of `kvh` x `d` columns, one
/// row to a page slot, one row dropped (its page out of range).
fn kv_append_case(ps: usize, kvh: usize, d: usize, pages: usize, seed: u64) {
    let mut rng = Rng(seed);
    let pool = Pool::new(&mut rng, ps, kvh, d, pages, vec![vec![0]], 1);
    let n = 6usize;
    let w = kvh * d;
    let k = rng.bf16s(n * w, 1.0);
    let v = rng.bf16s(n * w, 1.0);
    let page: Vec<i32> = (0..n)
        .map(|i| {
            if i == 3 {
                i32::MAX
            } else {
                ((i * 3) % pages) as i32
            }
        })
        .collect();
    let off: Vec<i32> = (0..n).map(|i| ((i * 7 + 1) % ps) as i32).collect();
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let kt = b.bf16(n as u32, w as u32, &k);
    let vt = b.bf16(n as u32, w as u32, &v);
    let pt = b.i32(n as u32, 1, &page);
    let ot = b.i32(n as u32, 1, &off);
    if !b
        .run(|ctx| attn::kv_append(ctx, kt, vt, &kv, pt, ot))
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
    assert_close(&b.read_f32(kv.keys), &want_k, 0.0, 0.0);
    assert_close(&b.read_f32(kv.values), &want_v, 0.0, 0.0);
}

/// A block selection: each row reads the `ratio`-wide blocks its selection
/// names among the closed blocks before it, and the open block after them;
/// ids outside the closed range are skipped.
#[test]
fn selected_decode_reads_the_named_blocks_and_the_open_one() {
    crate::before();
    let mut rng = Rng(31);
    let (qh, kvh, d, ratio) = (4usize, 2usize, 16usize, 4usize);
    let pool = Pool::new(
        &mut rng,
        4,
        kvh,
        d,
        9,
        vec![vec![5, 1, 7, 2], vec![0, 6, 8]],
        4,
    );
    // Lane 0 at position 14 (3 closed blocks, open block 12..14), lane 1 at
    // position 9 (2 closed, open 8..9), lane 0 at position 2 (none closed).
    let fire = Fire::plain(&[(0, 14), (1, 9), (0, 2), (0, -1)]);
    let sel = [2i32, 0, -1, 1, 7, -1, 0, 5, 0, 0, 0, 0];
    let top_k = 3usize;
    let rows = fire.positions.len();
    let q = rng.bf16s(rows * qh * d, 1.0);
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let plan = fire.decode(&mut b);
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let st = b.i32(rows as u32, top_k as u32, &sel);
    let o = b.zeros(Dtype::Bf16, rows as u32, (qh * d) as u32);
    let scale = 0.25f32;
    if !b
        .run(|ctx| {
            attn::decode_selected(
                ctx,
                qt,
                &plan,
                st,
                &kv,
                None,
                d as u32,
                scale,
                ratio as u32,
                o,
            )
        })
        .unwrap()
    {
        return;
    }
    let keys = |r: usize| -> Vec<usize> {
        let qpos = fire.positions[r];
        if qpos < 0 {
            return Vec::new();
        }
        let lane = fire.req[r] as usize;
        let nblocks = (qpos as usize + 1) / ratio;
        let mut kps: Vec<usize> = Vec::new();
        for s in &sel[r * top_k..(r + 1) * top_k] {
            if *s >= 0 && (*s as usize) < nblocks {
                kps.extend((*s as usize * ratio)..(*s as usize + 1) * ratio);
            }
        }
        kps.extend(nblocks * ratio..=qpos as usize);
        kps.sort_unstable();
        kps.dedup();
        kps.into_iter().map(|kp| pool.slot(lane, kp)).collect()
    };
    let (want, _) = sdpa(
        &q,
        rows,
        qh,
        kvh,
        d,
        &pool.keys,
        &pool.values,
        f64::from(scale),
        &keys,
    );
    check_rows(&b.read_f32(o), &want, qh * d, rows - 1, 2e-2);
}
