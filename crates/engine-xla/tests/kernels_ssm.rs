//! The recurrent (SSM) family against sequential host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::attn::ssm::{self, Committed};
use kernels_xla::{RaggedTensor, RecurrentPool, Tensor};

const DROP: i32 = i32::MAX;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| round_bf16(self.next() * scale)).collect()
    }
    fn f32s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| lo + (self.next() * 0.5 + 0.5) * (hi - lo)).collect()
    }
}

fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

fn pool(state: Tensor, slots: Tensor) -> RecurrentPool {
    RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    }
}

// ------------------------------------------------------------------- conv

/// One lane's conv over `xs` (`n` rows of `c`), from the kept rows `past`
/// (`hist * c`); returns y and the kept rows after `keep` rows.
fn conv_ref(
    xs: &[f32],
    c: usize,
    w: &[f32],
    k: usize,
    dil: usize,
    past: &[f32],
    keep: usize,
    residual: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n = xs.len() / c;
    let hist = (k - 1) * dil + 1;
    let at = |src: isize, ch: usize| -> f32 {
        if src < 0 {
            past[(hist as isize + src) as usize * c + ch]
        } else {
            xs[src as usize * c + ch]
        }
    };
    let mut y = vec![0.0; n * c];
    for t in 0..n {
        for ch in 0..c {
            let mut acc = 0.0;
            for tap in 0..k {
                let src = t as isize - ((k - 1 - tap) * dil) as isize;
                acc += at(src, ch) * w[ch * k + tap];
            }
            let x = xs[t * c + ch];
            y[t * c + ch] = if residual { acc + x } else { silu(acc) };
        }
    }
    let mut next = vec![0.0; hist * c];
    for s in 0..hist {
        let src = keep as isize - hist as isize + s as isize;
        for ch in 0..c {
            next[s * c + ch] = at(src, ch);
        }
    }
    (y, next)
}

#[test]
fn the_decode_conv_and_short_conv_shift_their_windows() {
    let (rows, c, k, dil, slots) = (4u32, 40usize, 4usize, 1usize, 5usize);
    let hist = (k - 1) * dil + 1;
    let mut rng = Rng(7);
    let xs = rng.bf16s(rows as usize * c, 1.0);
    let ws = rng.bf16s(c * k, 0.5);
    let bank0 = rng.f32s(slots * hist * c, -1.0, 1.0);
    let slot_of = [3, 0, DROP, 1];

    for residual in [false, true] {
        let mut b = Bench::new();
        let x = b.bf16(rows, c as u32, &xs);
        let w = b.bf16(c as u32, k as u32, &ws);
        let bank = b.f32(slots as u32, (hist * c) as u32, &bank0);
        let sl = b.i32(rows, 1, &slot_of);
        let y = b.zeros(Dtype::Bf16, rows, c as u32);
        let p = pool(bank, sl);
        let ran = b
            .run(|ctx| {
                if residual {
                    ssm::short_conv(ctx, x, w, &p, k as u32, y)
                } else {
                    ssm::causal_conv1d(ctx, x, w, &p, k as u32, dil as u32, y)
                }
            })
            .unwrap();
        if !ran {
            return;
        }
        let mut want_bank = bank0.clone();
        let mut want_y = vec![0.0; rows as usize * c];
        for r in 0..rows as usize {
            let s = slot_of[r];
            let slot = if s == DROP { 0 } else { s as usize };
            let past = &bank0[slot * hist * c..(slot + 1) * hist * c];
            let (yr, next) = conv_ref(&xs[r * c..(r + 1) * c], c, &ws, k, dil, past, 1, residual);
            if s != DROP {
                want_bank[slot * hist * c..(slot + 1) * hist * c].copy_from_slice(&next);
                want_y[r * c..(r + 1) * c].copy_from_slice(&yr);
            }
        }
        let got_y = b.read_f32(y);
        for r in [0usize, 1, 3] {
            assert_close(&got_y[r * c..(r + 1) * c], &want_y[r * c..(r + 1) * c], 2e-2, 1e-2);
        }
        assert_close(&b.read_f32(bank), &want_bank, 0.0, 0.0);
    }
}

#[test]
fn the_chunked_conv_walks_each_lane_from_its_window() {
    // Lanes of 5, 0, 7 and 2 rows, then two padded rows.
    let (c, k, dil, slots) = (24usize, 3usize, 2usize, 6usize);
    let hist = (k - 1) * dil + 1;
    let indptr = [0i32, 5, 5, 12, 14, 14];
    let rows = 16usize;
    let lane_slot = [4i32, 0, 2, 5, 1];
    let mut slot_of_row = vec![DROP; rows];
    for l in 0..5 {
        for t in indptr[l]..indptr[l + 1] {
            slot_of_row[t as usize] = lane_slot[l];
        }
    }
    let mut rng = Rng(11);
    let xs = rng.bf16s(rows * c, 1.0);
    let ws = rng.bf16s(c * k, 0.5);
    let bank0 = rng.f32s(slots * hist * c, -1.0, 1.0);

    for residual in [false, true] {
        let dil = if residual { 1 } else { dil };
        let hist = (k - 1) * dil + 1;
        let bank0 = &bank0[..slots * hist * c];
        let mut b = Bench::new();
        let x = b.bf16(rows as u32, c as u32, &xs);
        let ip = b.i32(indptr.len() as u32, 1, &indptr);
        let w = b.bf16(c as u32, k as u32, &ws);
        let bank = b.f32(slots as u32, (hist * c) as u32, bank0);
        let sl = b.i32(rows as u32, 1, &slot_of_row);
        let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
        let p = pool(bank, sl);
        let rt = RaggedTensor { data: x, indptr: ip };
        let ran = b
            .run(|ctx| {
                if residual {
                    ssm::short_conv_chunked(ctx, rt, w, &p, k as u32, y)
                } else {
                    ssm::causal_conv1d_chunked(ctx, rt, w, &p, k as u32, dil as u32, y)
                }
            })
            .unwrap();
        if !ran {
            return;
        }
        let mut want_bank = bank0.to_vec();
        let got_y = b.read_f32(y);
        for l in 0..5 {
            let (lo, hi) = (indptr[l] as usize, indptr[l + 1] as usize);
            if hi == lo {
                continue;
            }
            let s = lane_slot[l] as usize;
            let past = &bank0[s * hist * c..(s + 1) * hist * c];
            let (yr, next) =
                conv_ref(&xs[lo * c..hi * c], c, &ws, k, dil, past, hi - lo, residual);
            want_bank[s * hist * c..(s + 1) * hist * c].copy_from_slice(&next);
            assert_close(&got_y[lo * c..hi * c], &yr, 2e-2, 1e-2);
        }
        assert_close(&b.read_f32(bank), &want_bank, 0.0, 0.0);
    }
}

/// At Qwen3.5's conv width (rows of 6144 channels) a lane walked one decode
/// fire at a time reads what its prefill read. (A TPU scatter into a
/// one-row operand of rows this wide lands out-of-range updates on the row
/// instead of dropping them, so the window's patch cannot route its unused
/// rows out of range.)
#[test]
fn a_wide_conv_decodes_what_its_prefill_reads() {
    let (c, k, dil, n, rows) = (6144usize, 4usize, 1usize, 5usize, 8usize);
    let hist = (k - 1) * dil + 1;
    let mut rng = Rng(7);
    let xs = rng.bf16s(rows * c, 2.0);
    let ws = rng.bf16s(c * k, 0.5);
    let bank0 = vec![0.0f32; 2 * hist * c];
    let (want, next) = conv_ref(&xs[..n * c], c, &ws, k, dil, &bank0[..hist * c], n, false);

    let mut b = Bench::new();
    let x = b.bf16(rows as u32, c as u32, &xs);
    let ip = b.i32(2, 1, &[0, n as i32]);
    let w = b.bf16(c as u32, k as u32, &ws);
    let bank = b.f32(2, (hist * c) as u32, &bank0);
    let mut slot_of_row = vec![DROP; rows];
    slot_of_row[..n].fill(1);
    let sl = b.i32(rows as u32, 1, &slot_of_row);
    let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let p = pool(bank, sl);
    let rt = RaggedTensor { data: x, indptr: ip };
    if !b
        .run(|ctx| ssm::causal_conv1d_chunked(ctx, rt, w, &p, k as u32, dil as u32, y))
        .unwrap()
    {
        return;
    }
    let prefill = b.read_f32(y);
    assert_close(&prefill[..n * c], &want, 2e-2, 1e-2);

    let mut b = Bench::new();
    let w = b.bf16(c as u32, k as u32, &ws);
    let bank = b.f32(2, (hist * c) as u32, &bank0);
    let mut decoded = Vec::with_capacity(n * c);
    for r in 0..n {
        let x = b.bf16(1, c as u32, &xs[r * c..(r + 1) * c]);
        let sl = b.i32(1, 1, &[1]);
        let y = b.zeros(Dtype::Bf16, 1, c as u32);
        let p = pool(bank, sl);
        b.run(|ctx| ssm::causal_conv1d(ctx, x, w, &p, k as u32, dil as u32, y))
            .unwrap();
        decoded.extend(b.read_f32(y));
    }
    assert_close(&decoded, &prefill[..n * c], 0.0, 0.0);
    assert_close(&b.read_f32(bank)[hist * c..], &next, 0.0, 0.0);
}

/// A seat window: lanes `lane0..lane0+2` of the seat tables, with replayed
/// rows ahead of their own.
#[test]
fn the_committed_conv_lands_the_window_after_the_commit() {
    let (c, k, dil, slots) = (16usize, 4usize, 1usize, 4usize);
    let hist = (k - 1) * dil + 1;
    let indptr = [0i32, 3, 4, 4];
    let lane0 = 1u32;
    let replay = [9i32, 2, 1, 3];
    let commit = [9i32, 4, 0, 2];
    let seat_slots = [9i32, 3, 1, -1];
    // Extended spans: 5, 2, 3 → 10 rows, plus 2 padded.
    let rows = 12usize;
    let mut rng = Rng(5);
    let xs = rng.bf16s(rows * c, 1.0);
    let ws = rng.bf16s(c * k, 0.5);
    let bank0 = rng.f32s(slots * hist * c, -1.0, 1.0);
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, c as u32, &xs);
    let ip = b.i32(4, 1, &indptr);
    let rp = b.i32(4, 1, &replay);
    let cm = b.i32(4, 1, &commit);
    let ss = b.i32(4, 1, &seat_slots);
    let w = b.bf16(c as u32, k as u32, &ws);
    let bank = b.f32(slots as u32, (hist * c) as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &vec![0; rows]);
    let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let p = pool(bank, sl);
    let seat = Committed {
        replay: rp,
        commit: cm,
        slots: ss,
        lane0,
    };
    if !b
        .run(|ctx| ssm::causal_conv1d_committed(ctx, x, ip, &seat, w, &p, k as u32, dil as u32, y))
        .unwrap()
    {
        return;
    }
    let mut want_bank = bank0.clone();
    let got_y = b.read_f32(y);
    let mut begin = 0usize;
    for r in 0..3 {
        let g = lane0 as usize + r;
        let span = (indptr[r + 1] - indptr[r] + replay[g]) as usize;
        let slot = seat_slots[g];
        if slot >= 0 {
            let s = slot as usize;
            let past = &bank0[s * hist * c..(s + 1) * hist * c];
            let keep = (commit[g] as usize).min(span);
            let (yr, next) =
                conv_ref(&xs[begin * c..(begin + span) * c], c, &ws, k, dil, past, keep, false);
            assert_close(&got_y[begin * c..(begin + span) * c], &yr, 2e-2, 1e-2);
            if keep > 0 {
                want_bank[s * hist * c..(s + 1) * hist * c].copy_from_slice(&next);
            }
        }
        begin += span;
    }
    assert_close(&b.read_f32(bank), &want_bank, 0.0, 0.0);
}

#[test]
fn the_gdn_prep_derives_decay_and_beta() {
    let (rows, vh) = (5usize, 12usize);
    let mut rng = Rng(3);
    let ba: Vec<f32> = rng.bf16s(rows * 2 * vh, 4.0);
    let mut dtb = rng.bf16s(vh, 1.0);
    dtb[0] = 24.0; // past the softplus cut
    let alog = rng.f32s(vh, -1.0, 1.0);
    let mut b = Bench::new();
    let bat = b.bf16(rows as u32, (2 * vh) as u32, &ba);
    let dt = b.bf16(1, vh as u32, &dtb);
    let al = b.f32(1, vh as u32, &alog);
    let g = b.zeros(Dtype::F32, rows as u32, (2 * vh) as u32);
    if !b.run(|ctx| ssm::gdn_prep(ctx, bat, dt, al, g)).unwrap() {
        return;
    }
    let mut want = vec![0.0; rows * 2 * vh];
    for t in 0..rows {
        for h in 0..vh {
            let bv = ba[t * 2 * vh + h];
            let av = ba[t * 2 * vh + vh + h];
            let z = av + dtb[h];
            let sp = if z <= 20.0 { (1.0 + z.exp()).ln() } else { z };
            want[t * 2 * vh + h] = -alog[h].exp() * sp;
            want[t * 2 * vh + vh + h] = 1.0 / (1.0 + (-bv).exp());
        }
    }
    assert_close(&b.read_f32(g), &want, 1e-5, 1e-5);
}

// ------------------------------------------------------------ delta rules

/// One token of the delta rule over a head's state `s` (`dv × dk`): decay
/// by `a` (per key channel), then the delta update; returns `S q`.
fn delta_token(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], a: &[f32], beta: f32) -> Vec<f32> {
    let (dk, dv) = (k.len(), v.len());
    let mut y = vec![0.0; dv];
    for c in 0..dv {
        let row = &mut s[c * dk..(c + 1) * dk];
        let mut mem = 0.0;
        for i in 0..dk {
            row[i] *= a[i];
            mem += row[i] * k[i];
        }
        let delta = (v[c] - mem) * beta;
        let mut acc = 0.0;
        for i in 0..dk {
            row[i] += k[i] * delta;
            acc += row[i] * q[i];
        }
        y[c] = acc;
    }
    y
}

fn l2(x: &[f32], eps: f32, scale: f32) -> Vec<f32> {
    let s: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (s + eps).sqrt() * scale;
    x.iter().map(|v| v * inv).collect()
}

struct Gdn {
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
}

impl Gdn {
    fn width(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
    fn stride(&self) -> usize {
        self.hv * self.dv * self.dk
    }
    /// Row `t`'s tokens through the state of its slot.
    fn token(&self, qkv: &[f32], gates: &[f32], t: usize, state: &mut [f32]) -> Vec<f32> {
        let (hk, hv, dk, dv) = (self.hk, self.hv, self.dk, self.dv);
        let row = &qkv[t * self.width()..(t + 1) * self.width()];
        let mut y = Vec::with_capacity(hv * dv);
        for h in 0..hv {
            let kh = h / (hv / hk);
            let q = l2(&row[kh * dk..(kh + 1) * dk], 1e-6, 1.0 / (dk as f32).sqrt());
            let k = l2(&row[hk * dk + kh * dk..hk * dk + (kh + 1) * dk], 1e-6, 1.0);
            let v = &row[2 * hk * dk + h * dv..2 * hk * dk + (h + 1) * dv];
            let a = vec![gates[t * 2 * hv + h].exp(); dk];
            let beta = gates[t * 2 * hv + hv + h];
            let s = &mut state[h * dv * dk..(h + 1) * dv * dk];
            y.extend(delta_token(s, &q, &k, v, &a, beta));
        }
        y
    }
}

fn gdn_inputs(rng: &mut Rng, g: &Gdn, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let qkv = rng.bf16s(rows * g.width(), 1.0);
    let mut gates = Vec::with_capacity(rows * 2 * g.hv);
    for _ in 0..rows {
        gates.extend(rng.f32s(g.hv, -1.5, -0.01));
        gates.extend(rng.f32s(g.hv, 0.05, 0.95));
    }
    (qkv, gates)
}

#[test]
fn the_gated_delta_step_folds_one_token_per_lane() {
    let g = Gdn {
        hk: 2,
        hv: 4,
        dk: 16,
        dv: 12,
    };
    let (rows, slots) = (4usize, 5usize);
    let mut rng = Rng(21);
    let (qkv, gates) = gdn_inputs(&mut rng, &g, rows);
    let bank0 = rng.f32s(slots * g.stride(), -0.5, 0.5);
    let slot_of = [2i32, DROP, 0, 4];
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, g.width() as u32, &qkv);
    let z = b.zeros(Dtype::Bf16, rows as u32, (g.hv * g.dv) as u32);
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let bank = b.f32(slots as u32, g.stride() as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &slot_of);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let p = pool(bank, sl);
    if !b
        .run(|ctx| {
            ssm::gated_delta(
                ctx, x, z, gt, &p, g.hk as u32, g.hv as u32, g.dk as u32, g.dv as u32, y,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want = bank0.clone();
    let got = b.read_f32(y);
    let w = g.hv * g.dv;
    for t in 0..rows {
        if slot_of[t] == DROP {
            continue;
        }
        let s = slot_of[t] as usize;
        let yr = g.token(&qkv, &gates, t, &mut want[s * g.stride()..(s + 1) * g.stride()]);
        assert_close(&got[t * w..(t + 1) * w], &yr, 1e-4, 1e-3);
    }
    assert_close(&b.read_f32(bank), &want, 1e-5, 1e-4);
}

#[test]
fn the_chunked_gated_delta_matches_the_token_walk() {
    let g = Gdn {
        hk: 1,
        hv: 2,
        dk: 32,
        dv: 24,
    };
    // Lanes of 70 (two chunks), 0, 5, 1 and 130 rows, then 3 padded rows
    // (64-row chunks); then lanes of 300 and 141 rows (128-row chunks).
    for (indptr, rows, lane_slot) in [
        (vec![0i32, 70, 70, 75, 76, 206], 209usize, vec![3i32, 0, 1, 4, 2]),
        (vec![0i32, 300, 441], 443, vec![5, 1]),
    ] {
    let lanes = indptr.len() - 1;
    let slots = 6usize;
    let mut slot_of_row = vec![DROP; rows];
    for l in 0..lanes {
        for t in indptr[l]..indptr[l + 1] {
            slot_of_row[t as usize] = lane_slot[l];
        }
    }
    let mut rng = Rng(33);
    let (qkv, gates) = gdn_inputs(&mut rng, &g, rows);
    let bank0 = rng.f32s(slots * g.stride(), -0.5, 0.5);
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, g.width() as u32, &qkv);
    let ip = b.i32(indptr.len() as u32, 1, &indptr);
    let z = b.zeros(Dtype::Bf16, rows as u32, (g.hv * g.dv) as u32);
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let bank = b.f32(slots as u32, g.stride() as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &slot_of_row);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let p = pool(bank, sl);
    let rt = RaggedTensor { data: x, indptr: ip };
    if !b
        .run(|ctx| {
            ssm::gated_delta_chunked(
                ctx, rt, z, gt, &p, g.hk as u32, g.hv as u32, g.dk as u32, g.dv as u32, y,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want = bank0.clone();
    let got = b.read_f32(y);
    let w = g.hv * g.dv;
    for l in 0..lanes {
        let s = lane_slot[l] as usize;
        for t in indptr[l] as usize..indptr[l + 1] as usize {
            let yr = g.token(&qkv, &gates, t, &mut want[s * g.stride()..(s + 1) * g.stride()]);
            assert_close(&got[t * w..(t + 1) * w], &yr, 2e-4, 2e-3);
        }
    }
    assert_close(&b.read_f32(bank), &want, 1e-4, 1e-3);
    }
}

#[test]
fn the_committed_gated_delta_lands_the_state_after_the_commit() {
    let g = Gdn {
        hk: 2,
        hv: 2,
        dk: 16,
        dv: 16,
    };
    let indptr = [0i32, 3, 4, 4, 70];
    let lane0 = 1u32;
    let replay = [0i32, 2, 1, 3, 0];
    let commit = [0i32, 4, 0, 2, 67];
    let seat_slots = [0i32, 3, 1, -1, 0];
    // Extended spans: 5, 2, 3, 66 → 76 rows, plus 4 padded.
    let rows = 80usize;
    let slots = 4usize;
    let mut rng = Rng(9);
    let (qkv, gates) = gdn_inputs(&mut rng, &g, rows);
    let bank0 = rng.f32s(slots * g.stride(), -0.5, 0.5);
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, g.width() as u32, &qkv);
    let ip = b.i32(5, 1, &indptr);
    let rp = b.i32(5, 1, &replay);
    let cm = b.i32(5, 1, &commit);
    let ss = b.i32(5, 1, &seat_slots);
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let bank = b.f32(slots as u32, g.stride() as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &vec![0; rows]);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let p = pool(bank, sl);
    let seat = Committed {
        replay: rp,
        commit: cm,
        slots: ss,
        lane0,
    };
    if !b
        .run(|ctx| {
            ssm::gated_delta_committed(
                ctx, x, ip, &seat, gt, &p, g.hk as u32, g.hv as u32, g.dk as u32, g.dv as u32, y,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want_bank = bank0.clone();
    let got = b.read_f32(y);
    let w = g.hv * g.dv;
    let mut begin = 0usize;
    for r in 0..4 {
        let gl = lane0 as usize + r;
        let span = (indptr[r + 1] - indptr[r] + replay[gl]) as usize;
        let slot = seat_slots[gl];
        if slot >= 0 {
            let s = slot as usize;
            let mut st = bank0[s * g.stride()..(s + 1) * g.stride()].to_vec();
            let keep = (commit[gl].max(0) as usize).min(span);
            for t in 0..span {
                let yr = g.token(&qkv, &gates, begin + t, &mut st);
                let at = (begin + t) * w;
                assert_close(&got[at..at + w], &yr, 2e-4, 2e-3);
                if t + 1 == keep {
                    want_bank[s * g.stride()..(s + 1) * g.stride()].copy_from_slice(&st);
                }
            }
        }
        begin += span;
    }
    assert_close(&b.read_f32(bank), &want_bank, 1e-4, 1e-3);
}

// -------------------------------------------------------------------- KDA

struct Kda {
    heads: usize,
    d: usize,
    eps: f32,
    floor: f32,
}

impl Kda {
    fn stride(&self) -> usize {
        self.heads * self.d * self.d
    }
    fn token(
        &self,
        mixed: &[f32],
        f: &[f32],
        bproj: &[f32],
        dt: &[f32],
        alog: &[f32],
        t: usize,
        state: &mut [f32],
    ) -> Vec<f32> {
        let (h, d) = (self.heads, self.d);
        let wide = h * d;
        let row = &mixed[t * 3 * wide..(t + 1) * 3 * wide];
        let mut y = Vec::with_capacity(wide);
        for hh in 0..h {
            let q = l2(&row[hh * d..(hh + 1) * d], self.eps, 1.0 / (d as f32).sqrt());
            let k = l2(&row[wide + hh * d..wide + (hh + 1) * d], self.eps, 1.0);
            let v = &row[2 * wide + hh * d..2 * wide + (hh + 1) * d];
            let alpha = alog[hh].exp();
            let a: Vec<f32> = (0..d)
                .map(|i| {
                    let z = f[t * wide + hh * d + i] + dt[hh * d + i];
                    if self.floor != 0.0 {
                        (self.floor / (1.0 + (-alpha * z).exp())).exp()
                    } else {
                        let sp = if z > 20.0 { z } else { (1.0 + z.exp()).ln() };
                        (-alpha * sp).exp()
                    }
                })
                .collect();
            let beta = 1.0 / (1.0 + (-bproj[t * h + hh]).exp());
            let s = &mut state[hh * d * d..(hh + 1) * d * d];
            y.extend(delta_token(s, &q, &k, v, &a, beta));
        }
        y
    }
}

#[test]
fn kda_walks_per_channel_decay_in_step_chunked_and_committed_forms() {
    for floor in [-4.0f32, 0.0] {
        let kd = Kda {
            heads: 2,
            d: 16,
            eps: 1e-6,
            floor,
        };
        let wide = kd.heads * kd.d;
        // Lanes of 21 (two chunks), 0 and 3 rows, then one padded row.
        let indptr = [0i32, 21, 21, 24];
        let rows = 25usize;
        let lane_slot = [2i32, 0, 1];
        let slots = 3usize;
        let mut slot_of_row = vec![DROP; rows];
        for l in 0..3 {
            for t in indptr[l]..indptr[l + 1] {
                slot_of_row[t as usize] = lane_slot[l];
            }
        }
        let mut rng = Rng(77);
        let mixed = rng.bf16s(rows * 3 * wide, 1.0);
        let fp = rng.bf16s(rows * wide, 2.0);
        let bp = rng.bf16s(rows * kd.heads, 2.0);
        let dt = rng.f32s(wide, -0.5, 0.5);
        let alog = rng.f32s(kd.heads, -1.0, 0.5);
        let bank0 = rng.f32s(slots * kd.stride(), -0.5, 0.5);

        // Chunked.
        let mut b = Bench::new();
        let m = b.bf16(rows as u32, (3 * wide) as u32, &mixed);
        let ip = b.i32(4, 1, &indptr);
        let ft = b.bf16(rows as u32, wide as u32, &fp);
        let bt = b.bf16(rows as u32, kd.heads as u32, &bp);
        let dtt = b.f32(1, wide as u32, &dt);
        let alt = b.f32(1, kd.heads as u32, &alog);
        let bank = b.f32(slots as u32, kd.stride() as u32, &bank0);
        let sl = b.i32(rows as u32, 1, &slot_of_row);
        let y = b.zeros(Dtype::F32, rows as u32, wide as u32);
        let p = pool(bank, sl);
        let rt = RaggedTensor { data: m, indptr: ip };
        if !b
            .run(|ctx| {
                ssm::kda_chunked(
                    ctx, rt, ft, bt, dtt, alt, &p, kd.heads as u32, kd.d as u32, kd.eps, kd.floor, y,
                )
            })
            .unwrap()
        {
            return;
        }
        let mut want = bank0.clone();
        let got = b.read_f32(y);
        for l in 0..3 {
            let s = lane_slot[l] as usize;
            for t in indptr[l] as usize..indptr[l + 1] as usize {
                let yr = kd.token(
                    &mixed, &fp, &bp, &dt, &alog, t,
                    &mut want[s * kd.stride()..(s + 1) * kd.stride()],
                );
                assert_close(&got[t * wide..(t + 1) * wide], &yr, 2e-4, 2e-3);
            }
        }
        assert_close(&b.read_f32(bank), &want, 1e-4, 1e-3);

        // Step: rows 0..4 as four lanes, one dropped.
        let step_slots = [1i32, DROP, 0, 2];
        let mut b = Bench::new();
        let m = b.bf16(4, (3 * wide) as u32, &mixed[..4 * 3 * wide]);
        let ft = b.bf16(4, wide as u32, &fp[..4 * wide]);
        let bt = b.bf16(4, kd.heads as u32, &bp[..4 * kd.heads]);
        let dtt = b.f32(1, wide as u32, &dt);
        let alt = b.f32(1, kd.heads as u32, &alog);
        let bank = b.f32(slots as u32, kd.stride() as u32, &bank0);
        let sl = b.i32(4, 1, &step_slots);
        let y = b.zeros(Dtype::F32, 4, wide as u32);
        let p = pool(bank, sl);
        b.run(|ctx| {
            ssm::kda_step(
                ctx, m, ft, bt, dtt, alt, &p, kd.heads as u32, kd.d as u32, kd.eps, kd.floor, y,
            )
        })
        .unwrap();
        let mut want = bank0.clone();
        let got = b.read_f32(y);
        for t in 0..4 {
            if step_slots[t] == DROP {
                continue;
            }
            let s = step_slots[t] as usize;
            let yr = kd.token(
                &mixed, &fp, &bp, &dt, &alog, t,
                &mut want[s * kd.stride()..(s + 1) * kd.stride()],
            );
            assert_close(&got[t * wide..(t + 1) * wide], &yr, 1e-4, 1e-3);
        }
        assert_close(&b.read_f32(bank), &want, 1e-5, 1e-4);

        // Committed: lane 0 replays 2 and commits 9 of its 23; lane 1 commits none.
        let cindptr = [0i32, 21, 22];
        let replay = [2i32, 0];
        let commit = [9i32, 0];
        let seat_slots = [2i32, 1];
        let mut b = Bench::new();
        let m = b.bf16(rows as u32, (3 * wide) as u32, &mixed);
        let ip = b.i32(3, 1, &cindptr);
        let rp = b.i32(2, 1, &replay);
        let cm = b.i32(2, 1, &commit);
        let ss = b.i32(2, 1, &seat_slots);
        let ft = b.bf16(rows as u32, wide as u32, &fp);
        let bt = b.bf16(rows as u32, kd.heads as u32, &bp);
        let dtt = b.f32(1, wide as u32, &dt);
        let alt = b.f32(1, kd.heads as u32, &alog);
        let bank = b.f32(slots as u32, kd.stride() as u32, &bank0);
        let sl = b.i32(rows as u32, 1, &vec![0; rows]);
        let y = b.zeros(Dtype::F32, rows as u32, wide as u32);
        let p = pool(bank, sl);
        let seat = Committed {
            replay: rp,
            commit: cm,
            slots: ss,
            lane0: 0,
        };
        b.run(|ctx| {
            ssm::kda_committed(
                ctx, m, ip, &seat, ft, bt, dtt, alt, &p, kd.heads as u32, kd.d as u32, kd.eps,
                kd.floor, y,
            )
        })
        .unwrap();
        let mut want_bank = bank0.clone();
        let got = b.read_f32(y);
        let mut begin = 0usize;
        for r in 0..2 {
            let span = (cindptr[r + 1] - cindptr[r] + replay[r]) as usize;
            let s = seat_slots[r] as usize;
            let mut st = bank0[s * kd.stride()..(s + 1) * kd.stride()].to_vec();
            let keep = (commit[r] as usize).min(span);
            for t in 0..span {
                let yr = kd.token(&mixed, &fp, &bp, &dt, &alog, begin + t, &mut st);
                let at = (begin + t) * wide;
                assert_close(&got[at..at + wide], &yr, 2e-4, 2e-3);
                if t + 1 == keep {
                    want_bank[s * kd.stride()..(s + 1) * kd.stride()].copy_from_slice(&st);
                }
            }
            begin += span;
        }
        assert_close(&b.read_f32(bank), &want_bank, 1e-4, 1e-3);
    }
}

#[test]
fn the_block_dyn_conv_moves_its_taps_per_row() {
    // Lanes of 5, 0 and 4 rows, one padded row; 12 channels in groups of 4.
    let indptr = [0i32, 5, 5, 9];
    let (rows, c, taps, group) = (10usize, 12usize, 3usize, 4usize);
    let groups = c / group;
    let mut rng = Rng(41);
    let xs = rng.bf16s(rows * c, 1.0);
    let co = rng.bf16s(rows * 2 * taps * groups, 0.5);
    let ba = rng.bf16s(2 * taps * c, 0.5);
    for side in [0usize, 1] {
        let mut b = Bench::new();
        let x = b.bf16(rows as u32, c as u32, &xs);
        let ip = b.i32(4, 1, &indptr);
        let ct = b.bf16(rows as u32, (2 * taps * groups) as u32, &co);
        let bt = b.bf16((2 * taps) as u32, c as u32, &ba);
        let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
        if !b
            .run(|ctx| {
                ssm::block_dyn_conv(
                    ctx,
                    RaggedTensor { data: x, indptr: ip },
                    ct,
                    bt,
                    side as u32,
                    taps as u32,
                    group as u32,
                    y,
                )
            })
            .unwrap()
        {
            return;
        }
        let got = b.read_f32(y);
        for l in 0..3 {
            let (lo, hi) = (indptr[l] as usize, indptr[l + 1] as usize);
            for t in 0..hi - lo {
                let row = lo + t;
                for ch in 0..c {
                    let mut acc = 0.0f32;
                    for k in 0..taps.min(t + 1) {
                        let at = side * taps + k;
                        let coef = ba[at * c + ch] + co[row * 2 * taps * groups + at * groups + ch / group];
                        acc += coef * xs[(lo + t - k) * c + ch];
                    }
                    let g = got[row * c + ch];
                    assert!((g - acc).abs() <= 2e-2 + 1e-2 * acc.abs(), "row {row} ch {ch}: {g} vs {acc}");
                }
            }
        }
    }
}

// ------------------------------------------------------- slot-blocked banks
//
// A state bank may hold each slot as a block of `stride / width` rows (whole
// tiles on a TPU) instead of one `stride`-wide row; the element order inside
// a slot is the same.

#[test]
fn a_slot_blocked_bank_steps_a_wide_decode_fire_in_rounds() {
    // 1 MiB of state per slot: the decode step moves 32 lanes per round, so
    // 40 lanes take two. Padded lanes: one routed out of range, two on the
    // sink slot (one per round).
    let g = Gdn {
        hk: 4,
        hv: 8,
        dk: 128,
        dv: 256,
    };
    let (rows, slots) = (40usize, 45usize);
    let sink = (slots - 1) as i32;
    let mut slot_of: Vec<i32> = (0..rows as i32).map(|l| (l * 7 + 3) % sink).collect();
    slot_of[5] = sink;
    slot_of[33] = DROP;
    slot_of[38] = sink;
    let real = |t: usize| slot_of[t] != DROP && slot_of[t] != sink;
    let mut rng = Rng(91);
    let (qkv, gates) = gdn_inputs(&mut rng, &g, rows);
    let bank0 = rng.f32s(slots * g.stride(), -0.5, 0.5);
    let per = g.stride() / g.dk;
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, g.width() as u32, &qkv);
    let z = b.zeros(Dtype::Bf16, rows as u32, (g.hv * g.dv) as u32);
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let bank = b.f32((slots * per) as u32, g.dk as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &slot_of);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let p = pool(bank, sl);
    if !b
        .run(|ctx| {
            ssm::gated_delta(
                ctx, x, z, gt, &p, g.hk as u32, g.hv as u32, g.dk as u32, g.dv as u32, y,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want = bank0.clone();
    let got = b.read_f32(y);
    let w = g.hv * g.dv;
    for t in (0..rows).filter(|&t| real(t)) {
        let s = slot_of[t] as usize;
        let yr = g.token(&qkv, &gates, t, &mut want[s * g.stride()..(s + 1) * g.stride()]);
        assert_close(&got[t * w..(t + 1) * w], &yr, 1e-4, 1e-3);
    }
    // Every slot but the sink is exact; the sink holds garbage.
    let n = (slots - 1) * g.stride();
    assert_close(&b.read_f32(bank)[..n], &want[..n], 1e-5, 1e-4);
}

#[test]
fn slot_blocked_banks_serve_the_chunked_rule_kda_and_the_conv() {
    // The chunked gated delta: lanes of 70, 0 and 5 rows, one padded row.
    let g = Gdn {
        hk: 1,
        hv: 2,
        dk: 32,
        dv: 24,
    };
    let indptr = [0i32, 70, 70, 75];
    let lane_slot = [3i32, 0, 1];
    let (rows, slots) = (76usize, 5usize);
    let mut slot_of_row = vec![DROP; rows];
    for l in 0..3 {
        for t in indptr[l]..indptr[l + 1] {
            slot_of_row[t as usize] = lane_slot[l];
        }
    }
    let mut rng = Rng(45);
    let (qkv, gates) = gdn_inputs(&mut rng, &g, rows);
    let bank0 = rng.f32s(slots * g.stride(), -0.5, 0.5);
    let per = g.stride() / g.dk;
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, g.width() as u32, &qkv);
    let ip = b.i32(4, 1, &indptr);
    let z = b.zeros(Dtype::Bf16, rows as u32, (g.hv * g.dv) as u32);
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let bank = b.f32((slots * per) as u32, g.dk as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &slot_of_row);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let p = pool(bank, sl);
    let rt = RaggedTensor { data: x, indptr: ip };
    if !b
        .run(|ctx| {
            ssm::gated_delta_chunked(
                ctx, rt, z, gt, &p, g.hk as u32, g.hv as u32, g.dk as u32, g.dv as u32, y,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want = bank0.clone();
    let got = b.read_f32(y);
    let w = g.hv * g.dv;
    for l in 0..3 {
        let s = lane_slot[l] as usize;
        for t in indptr[l] as usize..indptr[l + 1] as usize {
            let yr = g.token(&qkv, &gates, t, &mut want[s * g.stride()..(s + 1) * g.stride()]);
            assert_close(&got[t * w..(t + 1) * w], &yr, 2e-4, 2e-3);
        }
    }
    assert_close(&b.read_f32(bank), &want, 1e-4, 1e-3);

    // One KDA step per lane, the last lane dropped.
    let kd = Kda {
        heads: 3,
        d: 16,
        eps: 1e-6,
        floor: -4.0,
    };
    let wide = kd.heads * kd.d;
    let (rows, slots) = (4usize, 5usize);
    let slot_of = [4i32, 0, 2, DROP];
    let mixed = rng.bf16s(rows * 3 * wide, 1.0);
    let fp = rng.bf16s(rows * wide, 2.0);
    let bp = rng.bf16s(rows * kd.heads, 2.0);
    let dt = rng.f32s(wide, -0.5, 0.5);
    let alog = rng.f32s(kd.heads, -1.0, 0.5);
    let bank0 = rng.f32s(slots * kd.stride(), -0.5, 0.5);
    let per = kd.stride() / kd.d;
    let mut b = Bench::new();
    let m = b.bf16(rows as u32, (3 * wide) as u32, &mixed);
    let ft = b.bf16(rows as u32, wide as u32, &fp);
    let bt = b.bf16(rows as u32, kd.heads as u32, &bp);
    let dtt = b.f32(1, wide as u32, &dt);
    let alt = b.f32(1, kd.heads as u32, &alog);
    let bank = b.f32((slots * per) as u32, kd.d as u32, &bank0);
    let sl = b.i32(rows as u32, 1, &slot_of);
    let y = b.zeros(Dtype::F32, rows as u32, wide as u32);
    let p = pool(bank, sl);
    b.run(|ctx| {
        ssm::kda_step(
            ctx, m, ft, bt, dtt, alt, &p, kd.heads as u32, kd.d as u32, kd.eps, kd.floor, y,
        )
    })
    .unwrap();
    let mut want = bank0.clone();
    let got = b.read_f32(y);
    for t in 0..rows {
        if slot_of[t] == DROP {
            continue;
        }
        let s = slot_of[t] as usize;
        let st = &mut want[s * kd.stride()..(s + 1) * kd.stride()];
        let yr = kd.token(&mixed, &fp, &bp, &dt, &alog, t, st);
        assert_close(&got[t * wide..(t + 1) * wide], &yr, 1e-4, 1e-3);
    }
    assert_close(&b.read_f32(bank), &want, 1e-5, 1e-4);

    // The decode conv over a bank of `hist` rows of channels per slot.
    let (rows, c, k, slots) = (4u32, 40usize, 4usize, 5usize);
    let xs = rng.bf16s(rows as usize * c, 1.0);
    let ws = rng.bf16s(c * k, 0.5);
    let bank0 = rng.f32s(slots * k * c, -1.0, 1.0);
    let slot_of = [3, 0, DROP, 4];
    let mut b = Bench::new();
    let x = b.bf16(rows, c as u32, &xs);
    let w = b.bf16(c as u32, k as u32, &ws);
    let bank = b.f32((slots * k) as u32, c as u32, &bank0);
    let sl = b.i32(rows, 1, &slot_of);
    let y = b.zeros(Dtype::Bf16, rows, c as u32);
    let p = pool(bank, sl);
    b.run(|ctx| ssm::causal_conv1d(ctx, x, w, &p, k as u32, 1, y))
        .unwrap();
    let mut want_bank = bank0.clone();
    let got_y = b.read_f32(y);
    for r in 0..rows as usize {
        if slot_of[r] == DROP {
            continue;
        }
        let s = slot_of[r] as usize;
        let past = &bank0[s * k * c..(s + 1) * k * c];
        let (yr, next) = conv_ref(&xs[r * c..(r + 1) * c], c, &ws, k, 1, past, 1, false);
        want_bank[s * k * c..(s + 1) * k * c].copy_from_slice(&next);
        assert_close(&got_y[r * c..(r + 1) * c], &yr, 2e-2, 1e-2);
    }
    assert_close(&b.read_f32(bank), &want_bank, 0.0, 0.0);
}
