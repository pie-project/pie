//! The gated delta net family against host references.

mod common;

use common::data;
use dtype::Dtype;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::attn::ssm;
use kernels_cerebras::{RaggedTensor, RecurrentPool, Tensor};

const DROP: i32 = i32::MAX;

fn pool(state: Tensor, slots: Tensor) -> RecurrentPool {
    RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    }
}

fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

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
            y[t * c + ch] = round_bf16(silu(acc));
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
fn a_decode_conv_answers_the_host_and_rolls_its_window() {
    let (c, k, dil) = (6usize, 4usize, 1usize);
    let hist = (k - 1) * dil + 1;
    let rows = 4usize;
    let slots_n = 3usize;
    let xs = data(rows * c, 21);
    let ws = data(c * k, 22);
    let bank = data(slots_n * hist * c, 23);
    let slots = [2i32, 0, DROP, 1];
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, c as u32, &xs);
    let w = b.bf16(c as u32, k as u32, &ws);
    let state = b.f32(slots_n as u32, (hist * c) as u32, &bank);
    let st = b.i32(rows as u32, 1, &slots);
    let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    if !b
        .run(|ctx| ssm::causal_conv1d(ctx, x, w, &pool(state, st), k as u32, dil as u32, y))
        .unwrap()
    {
        return;
    }
    let mut want_bank = bank.clone();
    let got_y = b.read_f32(y);
    for r in 0..rows {
        if slots[r] == DROP {
            continue;
        }
        let s = slots[r] as usize;
        let (yr, next) = conv_ref(
            &xs[r * c..(r + 1) * c],
            c,
            &ws,
            k,
            dil,
            &bank[s * hist * c..(s + 1) * hist * c],
            1,
        );
        assert_close(&got_y[r * c..(r + 1) * c], &yr, 1e-3, 1e-2);
        want_bank[s * hist * c..(s + 1) * hist * c].copy_from_slice(&next);
    }
    assert_close(&b.read_f32(state), &want_bank, 0.0, 0.0);
}

#[test]
fn a_chunked_conv_agrees_with_stepping_decodes() {
    chunked_conv(5, 3, 31);
}

/// The chunked conv over rows too wide for a PE (8 rows x 1024 channels)
/// splits its channels over PEs.
#[test]
fn a_wide_chunked_conv_shards_its_channels_over_pes() {
    chunked_conv(1024, 3, 41);
}

fn chunked_conv(c: usize, k: usize, seed: u32) {
    let dil = 1usize;
    let hist = (k - 1) * dil + 1;
    let runs = [(0usize, 4usize), (2, 1), (1, 3)]; // (slot, rows); a lane with zero rows is skipped
    let indptr: Vec<i32> = std::iter::once(0)
        .chain(runs.iter().scan(0, |a, r| {
            *a += r.1 as i32;
            Some(*a)
        }))
        .collect();
    let rows: usize = runs.iter().map(|r| r.1).sum();
    let slots_n = 3usize;
    let xs = data(rows * c, seed);
    let ws = data(c * k, seed + 1);
    let bank = data(slots_n * hist * c, seed + 2);
    let slot_of_row: Vec<i32> = runs
        .iter()
        .flat_map(|r| std::iter::repeat_n(r.0 as i32, r.1))
        .collect();
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, c as u32, &xs);
    let ip = b.i32(indptr.len() as u32, 1, &indptr);
    let w = b.bf16(c as u32, k as u32, &ws);
    let state = b.f32(slots_n as u32, (hist * c) as u32, &bank);
    let st = b.i32(rows as u32, 1, &slot_of_row);
    let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let ran = b
        .run(|ctx| {
            ssm::causal_conv1d_chunked(
                ctx,
                RaggedTensor {
                    data: x,
                    indptr: ip,
                },
                w,
                &pool(state, st),
                k as u32,
                dil as u32,
                y,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let mut want_bank = bank.clone();
    let mut want_y = vec![0.0; rows * c];
    for (l, (slot, n)) in runs.iter().enumerate() {
        let lo = indptr[l] as usize;
        let (yr, next) = conv_ref(
            &xs[lo * c..(lo + n) * c],
            c,
            &ws,
            k,
            dil,
            &bank[slot * hist * c..(slot + 1) * hist * c],
            *n,
        );
        want_y[lo * c..(lo + n) * c].copy_from_slice(&yr);
        want_bank[slot * hist * c..(slot + 1) * hist * c].copy_from_slice(&next);
    }
    assert_close(&b.read_f32(y), &want_y, 1e-3, 1e-2);
    assert_close(&b.read_f32(state), &want_bank, 0.0, 0.0);
}

#[test]
fn gdn_prep_turns_b_and_a_into_gates() {
    let (rows, vh) = (3usize, 4usize);
    let ba: Vec<f32> = data(rows * 2 * vh, 41)
        .iter()
        .map(|v| round_bf16(v * 3.0))
        .collect();
    let dtb = data(vh, 42);
    let alog: Vec<f32> = data(vh, 43).iter().map(|v| v * 0.5).collect();
    let mut b = Bench::new();
    let bat = b.bf16(rows as u32, 2 * vh as u32, &ba);
    let dt = b.bf16(1, vh as u32, &dtb);
    let al = b.f32(1, vh as u32, &alog);
    let gates = b.zeros(Dtype::F32, rows as u32, 2 * vh as u32);
    if !b.run(|ctx| ssm::gdn_prep(ctx, bat, dt, al, gates)).unwrap() {
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
    assert_close(&b.read_f32(gates), &want, 1e-4, 1e-3);
}

/// One token of the delta rule over a head's state `s` (`dv × dk`).
fn delta_token(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], a: f32, beta: f32) -> Vec<f32> {
    let (dk, dv) = (k.len(), v.len());
    let mut y = vec![0.0; dv];
    for c in 0..dv {
        let row = &mut s[c * dk..(c + 1) * dk];
        let mut mem = 0.0;
        for i in 0..dk {
            row[i] *= a;
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
    fn token(&self, qkv: &[f32], gates: &[f32], t: usize, state: &mut [f32]) -> Vec<f32> {
        let (hk, hv, dk, dv) = (self.hk, self.hv, self.dk, self.dv);
        let row = &qkv[t * self.width()..(t + 1) * self.width()];
        let mut y = Vec::with_capacity(hv * dv);
        for h in 0..hv {
            let kh = h / (hv / hk);
            let q = l2(&row[kh * dk..(kh + 1) * dk], 1e-6, 1.0 / (dk as f32).sqrt());
            let k = l2(&row[hk * dk + kh * dk..hk * dk + (kh + 1) * dk], 1e-6, 1.0);
            let v = &row[2 * hk * dk + h * dv..2 * hk * dk + (h + 1) * dv];
            let a = gates[t * 2 * hv + h].exp();
            let beta = gates[t * 2 * hv + hv + h];
            let s = &mut state[h * dv * dk..(h + 1) * dv * dk];
            y.extend(delta_token(s, &q, &k, v, a, beta));
        }
        y
    }
}

fn gates_for(rows: usize, hv: usize, seed: u32) -> Vec<f32> {
    data(rows * 2 * hv, seed)
        .chunks(2 * hv)
        .flat_map(|row| {
            let (g, bt) = row.split_at(hv);
            g.iter()
                .map(|v| -0.75 - 0.74 * v)
                .chain(bt.iter().map(|v| 0.5 + 0.45 * v))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn the_delta_rule_steps_each_lanes_slot() {
    let g = Gdn {
        hk: 2,
        hv: 4,
        dk: 8,
        dv: 6,
    };
    let rows = 3usize;
    let slots_n = 3usize;
    let qkv = data(rows * g.width(), 51);
    let gates = gates_for(rows, g.hv, 52);
    let bank: Vec<f32> = data(slots_n * g.stride(), 53)
        .iter()
        .map(|v| v * 0.3)
        .collect();
    let slots = [1i32, DROP, 0];
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, g.width() as u32, &qkv);
    let z = b.bf16(
        rows as u32,
        (g.hv * g.dv) as u32,
        &vec![0.0; rows * g.hv * g.dv],
    );
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let state = b.f32(slots_n as u32, g.stride() as u32, &bank);
    let st = b.i32(rows as u32, 1, &slots);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let ran = b
        .run(|ctx| {
            ssm::gated_delta(
                ctx,
                qt,
                z,
                gt,
                &pool(state, st),
                g.hk as u32,
                g.hv as u32,
                g.dk as u32,
                g.dv as u32,
                y,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let mut want_bank = bank.clone();
    let got_y = b.read_f32(y);
    for t in 0..rows {
        if slots[t] == DROP {
            continue;
        }
        let s = slots[t] as usize;
        let yt = g.token(
            &qkv,
            &gates,
            t,
            &mut want_bank[s * g.stride()..(s + 1) * g.stride()],
        );
        assert_close(
            &got_y[t * g.hv * g.dv..(t + 1) * g.hv * g.dv],
            &yt,
            1e-4,
            1e-3,
        );
    }
    assert_close(&b.read_f32(state), &want_bank, 1e-5, 1e-4);
}

#[test]
fn the_chunked_delta_rule_walks_each_lane_in_order() {
    let g = Gdn {
        hk: 1,
        hv: 2,
        dk: 8,
        dv: 4,
    };
    let runs = [(1usize, 3usize), (0, 2), (2, 0), (2, 4)];
    let indptr: Vec<i32> = std::iter::once(0)
        .chain(runs.iter().scan(0, |a, r| {
            *a += r.1 as i32;
            Some(*a)
        }))
        .collect();
    let rows: usize = runs.iter().map(|r| r.1).sum();
    let slots_n = 3usize;
    let qkv = data(rows * g.width(), 61);
    let gates = gates_for(rows, g.hv, 62);
    let bank: Vec<f32> = data(slots_n * g.stride(), 63)
        .iter()
        .map(|v| v * 0.3)
        .collect();
    let slot_of_row: Vec<i32> = runs
        .iter()
        .flat_map(|r| std::iter::repeat_n(r.0 as i32, r.1))
        .collect();
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, g.width() as u32, &qkv);
    let ip = b.i32(indptr.len() as u32, 1, &indptr);
    let z = b.bf16(
        rows as u32,
        (g.hv * g.dv) as u32,
        &vec![0.0; rows * g.hv * g.dv],
    );
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let state = b.f32(slots_n as u32, g.stride() as u32, &bank);
    let st = b.i32(rows as u32, 1, &slot_of_row);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let ran = b
        .run(|ctx| {
            ssm::gated_delta_chunked(
                ctx,
                RaggedTensor {
                    data: qt,
                    indptr: ip,
                },
                z,
                gt,
                &pool(state, st),
                g.hk as u32,
                g.hv as u32,
                g.dk as u32,
                g.dv as u32,
                y,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let mut want_bank = bank.clone();
    let mut want_y = vec![0.0; rows * g.hv * g.dv];
    for (l, (slot, n)) in runs.iter().enumerate() {
        let lo = indptr[l] as usize;
        for t in lo..lo + n {
            let yt = g.token(
                &qkv,
                &gates,
                t,
                &mut want_bank[slot * g.stride()..(slot + 1) * g.stride()],
            );
            want_y[t * g.hv * g.dv..(t + 1) * g.hv * g.dv].copy_from_slice(&yt);
        }
    }
    assert_close(&b.read_f32(y), &want_y, 2e-4, 2e-3);
    assert_close(&b.read_f32(state), &want_bank, 1e-4, 1e-3);
}

/// A bank past one PE's share splits the lanes over PEs: each PE holds only
/// its lanes' slots and the host reassembles the bank and the rows.
#[test]
fn a_wide_delta_bank_splits_its_lanes_over_pes() {
    let g = Gdn {
        hk: 1,
        hv: 4,
        dk: 16,
        dv: 16,
    }; // stride 1024 per slot
    let runs = [(3usize, 2usize), (0, 1), (5, 3), (1, 2), (4, 1)]; // 5 lanes, 9 rows
    let indptr: Vec<i32> = std::iter::once(0)
        .chain(runs.iter().scan(0, |a, r| {
            *a += r.1 as i32;
            Some(*a)
        }))
        .collect();
    let rows: usize = runs.iter().map(|r| r.1).sum();
    let slots_n = 6usize; // 6 x 1024 = 6144 words > shard_words()
    let qkv = data(rows * g.width(), 81);
    let gates = gates_for(rows, g.hv, 82);
    let bank: Vec<f32> = data(slots_n * g.stride(), 83)
        .iter()
        .map(|v| v * 0.3)
        .collect();
    let slot_of_row: Vec<i32> = runs
        .iter()
        .flat_map(|r| std::iter::repeat_n(r.0 as i32, r.1))
        .collect();
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, g.width() as u32, &qkv);
    let ip = b.i32(indptr.len() as u32, 1, &indptr);
    let z = b.bf16(
        rows as u32,
        (g.hv * g.dv) as u32,
        &vec![0.0; rows * g.hv * g.dv],
    );
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let state = b.f32(slots_n as u32, g.stride() as u32, &bank);
    let st = b.i32(rows as u32, 1, &slot_of_row);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let ran = b
        .run(|ctx| {
            ssm::gated_delta_chunked(
                ctx,
                RaggedTensor {
                    data: qt,
                    indptr: ip,
                },
                z,
                gt,
                &pool(state, st),
                g.hk as u32,
                g.hv as u32,
                g.dk as u32,
                g.dv as u32,
                y,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    assert!(b.last.as_ref().is_some_and(|r| r.pe.contains("k_delta")));
    let mut want_bank = bank.clone();
    let mut want_y = vec![0.0; rows * g.hv * g.dv];
    for (l, (slot, n)) in runs.iter().enumerate() {
        let lo = indptr[l] as usize;
        for t in lo..lo + n {
            let yt = g.token(
                &qkv,
                &gates,
                t,
                &mut want_bank[slot * g.stride()..(slot + 1) * g.stride()],
            );
            want_y[t * g.hv * g.dv..(t + 1) * g.hv * g.dv].copy_from_slice(&yt);
        }
    }
    assert_close(&b.read_f32(y), &want_y, 2e-4, 2e-3);
    assert_close(&b.read_f32(state), &want_bank, 1e-4, 1e-3);
}

/// One slot's state past a PE's share splits by state row and head: each
/// PE holds one block of its lanes' slots and the host reassembles it.
#[test]
fn a_big_slot_state_splits_into_blocks_over_pes() {
    let g = Gdn {
        hk: 1,
        hv: 2,
        dk: 64,
        dv: 64,
    }; // 8192 words per slot > shard_words()
    let rows = 3usize;
    let slots_n = 3usize;
    let qkv = data(rows * g.width(), 91);
    let gates = gates_for(rows, g.hv, 92);
    let bank: Vec<f32> = data(slots_n * g.stride(), 93)
        .iter()
        .map(|v| v * 0.3)
        .collect();
    // Lanes split over PEs must not share a slot: each lane its own.
    let slots = [1i32, 0, 2];
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, g.width() as u32, &qkv);
    let z = b.bf16(
        rows as u32,
        (g.hv * g.dv) as u32,
        &vec![0.0; rows * g.hv * g.dv],
    );
    let gt = b.f32(rows as u32, (2 * g.hv) as u32, &gates);
    let state = b.f32(slots_n as u32, g.stride() as u32, &bank);
    let st = b.i32(rows as u32, 1, &slots);
    let y = b.zeros(Dtype::F32, rows as u32, (g.hv * g.dv) as u32);
    let ran = b
        .run(|ctx| {
            ssm::gated_delta(
                ctx,
                qt,
                z,
                gt,
                &pool(state, st),
                g.hk as u32,
                g.hv as u32,
                g.dk as u32,
                g.dv as u32,
                y,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let mut want_bank = bank.clone();
    let got_y = b.read_f32(y);
    for t in 0..rows {
        let s = slots[t] as usize;
        let yt = g.token(
            &qkv,
            &gates,
            t,
            &mut want_bank[s * g.stride()..(s + 1) * g.stride()],
        );
        assert_close(
            &got_y[t * g.hv * g.dv..(t + 1) * g.hv * g.dv],
            &yt,
            1e-4,
            1e-3,
        );
    }
    assert_close(&b.read_f32(state), &want_bank, 1e-5, 1e-4);
}

/// A wide conv state could split by channel, but its rows and weights are
/// whole on every PE: the entry refuses rather than overflow a PE.
/// A conv whose slot state (and rows) exceed a PE's share splits its
/// channels over PEs; x and y travel by column block.
#[test]
fn a_wide_conv_shards_its_channels_over_pes() {
    let (c, k, dil) = (2048usize, 4usize, 1usize);
    let hist = (k - 1) * dil + 1;
    let rows = 3usize;
    let slots_n = 3usize;
    let xs = data(rows * c, 94);
    let ws = data(c * k, 95);
    let bank = data(slots_n * hist * c, 96);
    let slots = [1i32, DROP, 0];
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, c as u32, &xs);
    let w = b.bf16(c as u32, k as u32, &ws);
    let state = b.f32(slots_n as u32, (hist * c) as u32, &bank);
    let st = b.i32(rows as u32, 1, &slots);
    let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    if !b
        .run(|ctx| ssm::causal_conv1d(ctx, x, w, &pool(state, st), k as u32, dil as u32, y))
        .unwrap()
    {
        return;
    }
    // The rendered call's last argument says x, y and w are sharded.
    let rendered = b.last.as_ref().expect("rendered");
    let call = rendered
        .pe
        .lines()
        .find(|l| l.contains("k_conv1d(@ptrcast"))
        .expect("the conv call");
    assert!(
        call.trim_end().ends_with(", true);"),
        "the conv should carry x, y and w by channel block: {call}"
    );
    let mut want_bank = bank.clone();
    let got_y = b.read_f32(y);
    for r in 0..rows {
        if slots[r] == DROP {
            continue;
        }
        let s = slots[r] as usize;
        let (yr, next) = conv_ref(
            &xs[r * c..(r + 1) * c],
            c,
            &ws,
            k,
            dil,
            &bank[s * hist * c..(s + 1) * hist * c],
            1,
        );
        assert_close(&got_y[r * c..(r + 1) * c], &yr, 1e-3, 1e-2);
        want_bank[s * hist * c..(s + 1) * hist * c].copy_from_slice(&next);
    }
    assert_close(&b.read_f32(state), &want_bank, 0.0, 0.0);
}
