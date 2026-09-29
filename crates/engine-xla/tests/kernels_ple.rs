//! PLE n-gram hashing against a sequential host reference.

use dtype::Dtype;
use engine_xla::bench::Bench;
use kernels_xla::attn::ple;
use kernels_xla::attn::ssm::Committed;
use kernels_xla::{RaggedTensor, RecurrentPool, Tensor};

const DROP: i32 = i32::MAX;
const MULTS: [u64; 3] = [0x9E37_79B9_7F4A_7C15, 0xC2B2_AE3D_27D4_EB4F, 0x1656_67B1_9E37_79F9];
const PRIMES: [u64; 4] = [1_000_003, 999_983, 65_537, 3_000_000_019];
const OFFSETS: [u64; 4] = [0, 1_000_003, 2_000_000, 4_000_000_000];
const EOS: i32 = 1;
const HPN: usize = 2;

fn pool(state: Tensor, slots: Tensor) -> RecurrentPool {
    RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    }
}

fn cell(c: i32) -> i32 {
    if c == 0 { EOS } else { c - 1 }
}

/// The heads of one window `[id, prev1, prev2]`.
fn hash_row(window: &mut [i32], map: Option<&[i64]>) -> Vec<i32> {
    let mut crossed = false;
    for p in 1..window.len() {
        if crossed {
            window[p] = EOS;
        }
        if window[p] == EOS {
            crossed = true;
        }
    }
    if let Some(m) = map {
        for w in window.iter_mut() {
            *w = m[*w as usize] as i32;
        }
    }
    let mut out = vec![0; PRIMES.len()];
    for order in 2..=MULTS.len() {
        let mut mixed = (window[0] as i64 as u64).wrapping_mul(MULTS[0]);
        for p in 1..order {
            mixed ^= (window[p] as i64 as u64).wrapping_mul(MULTS[p]);
        }
        for k in 0..HPN {
            let h = (order - 2) * HPN + k;
            out[h] = (mixed % PRIMES[h]).wrapping_add(OFFSETS[h]) as i32;
        }
    }
    out
}

/// A lane's rows from its kept cells; returns the heads and the cells after
/// `keep` rows.
fn walk(ids: &[i32], state: &[i32], keep: usize, map: Option<&[i64]>) -> (Vec<i32>, Vec<i32>) {
    let span = MULTS.len() - 1;
    let mut out = Vec::new();
    for t in 0..ids.len() {
        let mut w = vec![ids[t]];
        for p in 1..=span {
            w.push(if t >= p {
                ids[t - p]
            } else {
                cell(state[span - (p - t)])
            });
        }
        out.extend(hash_row(&mut w, map));
    }
    let mut next = vec![0; span];
    for p in 0..span {
        let src = keep as isize - span as isize + p as isize;
        next[p] = if src >= 0 {
            ids[src as usize] + 1
        } else {
            state[p + keep]
        };
    }
    (out, next)
}

fn ids_of(n: usize, seed: u32) -> Vec<i32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40503));
            // Every 7th id is the eos barrier.
            if (h >> 5) % 7 == 0 { EOS } else { ((h >> 8) % 50) as i32 }
        })
        .collect()
}

#[test]
fn ngram_ids_hash_windows_in_decode_chunked_and_committed_forms() {
    let heads = PRIMES.len() as u32;
    let span = (MULTS.len() - 1) as u32;
    let slots = 4usize;
    let bank0: Vec<i32> = vec![0, 0, 17, 5, 1 + EOS, 30, 9, 0];
    let map: Vec<i64> = (0..64).map(|i| (i * 7 + 3) as i64 - 100).collect();

    for with_map in [false, true] {
        let m = if with_map { Some(map.as_slice()) } else { None };
        // Decode: four rows, one dropped.
        let ids = ids_of(4, 1);
        let slot_of = [2i32, DROP, 0, 3];
        let mut b = Bench::new();
        let idt = b.i32(4, 1, &ids);
        let st = b.i32(slots as u32, span, &bank0);
        let sl = b.i32(4, 1, &slot_of);
        let mt = b.i64(64, 1, &map);
        let out = b.zeros(Dtype::I32, 4, heads);
        let p = pool(st, sl);
        let ran = b
            .run(|ctx| {
                ple::ngram_ids(
                    ctx, idt, &p, EOS as u32, &MULTS, &PRIMES, &OFFSETS, HPN as u32,
                    with_map.then_some(mt), out,
                )
            })
            .unwrap();
        if !ran {
            return;
        }
        let got = b.read_i32(out);
        let mut want_bank = bank0.clone();
        for r in 0..4 {
            if slot_of[r] == DROP {
                continue;
            }
            let s = slot_of[r] as usize;
            let (o, next) = walk(&ids[r..r + 1], &bank0[s * 2..s * 2 + 2], 1, m);
            assert_eq!(&got[r * 4..r * 4 + 4], &o[..], "decode row {r}");
            want_bank[s * 2..s * 2 + 2].copy_from_slice(&next);
        }
        assert_eq!(b.read_i32(st), want_bank);

        // Chunked: lanes of 6, 0, 1 and 9 rows, then two padded rows.
        let indptr = [0i32, 6, 6, 7, 16];
        let rows = 18usize;
        let lane_slot = [1i32, 0, 3, 2];
        let mut slot_of_row = vec![DROP; rows];
        for l in 0..4 {
            for t in indptr[l]..indptr[l + 1] {
                slot_of_row[t as usize] = lane_slot[l];
            }
        }
        let ids = ids_of(rows, 2);
        let mut b = Bench::new();
        let idt = b.i32(rows as u32, 1, &ids);
        let ip = b.i32(5, 1, &indptr);
        let st = b.i32(slots as u32, span, &bank0);
        let sl = b.i32(rows as u32, 1, &slot_of_row);
        let mt = b.i64(64, 1, &map);
        let out = b.zeros(Dtype::I32, rows as u32, heads);
        let p = pool(st, sl);
        b.run(|ctx| {
            ple::ngram_ids_chunked(
                ctx,
                RaggedTensor { data: idt, indptr: ip },
                &p,
                EOS as u32,
                &MULTS,
                &PRIMES,
                &OFFSETS,
                HPN as u32,
                with_map.then_some(mt),
                out,
            )
        })
        .unwrap();
        let got = b.read_i32(out);
        let mut want_bank = bank0.clone();
        for l in 0..4 {
            let (lo, hi) = (indptr[l] as usize, indptr[l + 1] as usize);
            if lo == hi {
                continue;
            }
            let s = lane_slot[l] as usize;
            let (o, next) = walk(&ids[lo..hi], &bank0[s * 2..s * 2 + 2], hi - lo, m);
            assert_eq!(&got[lo * 4..hi * 4], &o[..], "chunked lane {l}");
            want_bank[s * 2..s * 2 + 2].copy_from_slice(&next);
        }
        assert_eq!(b.read_i32(st), want_bank);

        // Committed: seat lanes 1..3; own rows 3 and 2, replays 2 and 1.
        let indptr = [0i32, 3, 5];
        let replay = [0i32, 2, 1];
        let commit = [0i32, 1, 3];
        let seat_slots = [0i32, 3, 1];
        let ext_rows = 8usize; // 5 + 3, and two padded
        let own_rows = 6usize;
        let ids = ids_of(ext_rows + 2, 3);
        let mut b = Bench::new();
        let idt = b.i32((ext_rows + 2) as u32, 1, &ids);
        let ip = b.i32(3, 1, &indptr);
        let rp = b.i32(3, 1, &replay);
        let cm = b.i32(3, 1, &commit);
        let ss = b.i32(3, 1, &seat_slots);
        let st = b.i32(slots as u32, span, &bank0);
        let sl = b.i32(1, 1, &[0]);
        let mt = b.i64(64, 1, &map);
        let out = b.zeros(Dtype::I32, own_rows as u32, heads);
        let p = pool(st, sl);
        let seat = Committed {
            replay: rp,
            commit: cm,
            slots: ss,
            lane0: 1,
        };
        b.run(|ctx| {
            ple::ngram_ids_committed(
                ctx, idt, ip, &seat, &p, EOS as u32, &MULTS, &PRIMES, &OFFSETS, HPN as u32,
                with_map.then_some(mt), out,
            )
        })
        .unwrap();
        let got = b.read_i32(out);
        let mut want_bank = bank0.clone();
        let mut begin = 0usize;
        for r in 0..2 {
            let g = 1 + r;
            let own = (indptr[r + 1] - indptr[r]) as usize;
            let span_r = own + replay[g] as usize;
            let s = seat_slots[g] as usize;
            let keep = (commit[g] as usize).min(span_r);
            let (o, next) = walk(&ids[begin..begin + span_r], &bank0[s * 2..s * 2 + 2], keep, m);
            let rep = replay[g] as usize;
            let o0 = indptr[r] as usize;
            assert_eq!(&got[o0 * 4..(o0 + own) * 4], &o[rep * 4..], "committed lane {r}");
            if keep > 0 {
                want_bank[s * 2..s * 2 + 2].copy_from_slice(&next);
            }
            begin += span_r;
        }
        assert_eq!(b.read_i32(st), want_bank);
    }
}

#[test]
fn the_selector_walk_feeds_each_pick_to_the_next_row() {
    // Lanes of 5, 0 and 4 rows; one padded row keeps its pick.
    let indptr = [0i32, 5, 5, 9];
    let (rows, k, rank, vocab) = (10usize, 4usize, 12usize, 40usize);
    let mut s = 0x2545_F491u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let cand: Vec<i32> = (0..rows * k)
        .map(|i| if i == 7 { -3 } else { (next() % vocab as u32) as i32 })
        .collect();
    let unary: Vec<f32> = (0..rows * k).map(|_| (next() % 1000) as f32 / 500.0 - 1.0).collect();
    let bf = |x: f32| engine_xla::bench::round_bf16(x);
    let hp: Vec<f32> = (0..rows * rank).map(|_| bf((next() % 1000) as f32 / 500.0 - 1.0)).collect();
    let pred: Vec<f32> = (0..vocab * rank).map(|_| bf((next() % 1000) as f32 / 500.0 - 1.0)).collect();
    let succ: Vec<f32> = (0..vocab * rank).map(|_| bf((next() % 1000) as f32 / 500.0 - 1.0)).collect();
    let tokens: Vec<i32> = (0..rows).map(|_| (next() % vocab as u32) as i32).collect();
    let before: Vec<i32> = (0..rows as i32).map(|i| 1000 + i).collect();
    for (first, with_hp) in [(0u32, true), (1, false)] {
        let mut b = Bench::new();
        let ct = b.i32(rows as u32, k as u32, &cand);
        let ip = b.i32(4, 1, &indptr);
        let ut = b.f32(rows as u32, k as u32, &unary);
        let ht = b.bf16(rows as u32, rank as u32, &hp);
        let tt = b.i32(rows as u32, 1, &tokens);
        let pt = b.bf16(vocab as u32, rank as u32, &pred);
        let st = b.bf16(vocab as u32, rank as u32, &succ);
        let out = b.i32(rows as u32, 1, &before);
        if !b
            .run(|ctx| {
                ple::selector_walk(
                    ctx,
                    RaggedTensor { data: ct, indptr: ip },
                    ut,
                    with_hp.then_some(ht),
                    tt,
                    pt,
                    st,
                    first,
                    out,
                )
            })
            .unwrap()
        {
            return;
        }
        let mut want = before.clone();
        for l in 0..3 {
            let (lo, hi) = (indptr[l] as usize, indptr[l + 1] as usize);
            if lo == hi {
                continue;
            }
            if first > 0 {
                want[lo] = cand[lo * k];
            }
            let mut prev = tokens[lo];
            for row in lo + first as usize..hi {
                let mut best = 0;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..k {
                    let cid = cand[row * k + c];
                    let mut part = 0.0f32;
                    if prev >= 0 && (prev as usize) < vocab && cid >= 0 && (cid as usize) < vocab {
                        for d in 0..rank {
                            let h = if with_hp { hp[row * rank + d] } else { 1.0 };
                            part += pred[prev as usize * rank + d] * h * succ[cid as usize * rank + d];
                        }
                    }
                    let v = unary[row * k + c] + part;
                    if c == 0 || v > best_v {
                        best_v = v;
                        best = c;
                    }
                }
                want[row] = cand[row * k + best];
                prev = want[row];
            }
        }
        assert_eq!(b.read_i32(out), want, "first {first}");
    }
}
