//! The beam group's step on the device (`kernels_xla::beam::step`) is the
//! host replay's (`engine::beam::step`, `BEAM-PAGE-LISTS.md` §9) fire by
//! fire: widths 2, 4, 8, and 3 and 5 on the width rungs 4 and 8 (the beams
//! past a group's own dead); a prompt ending mid-page and one on a page; at least
//! `3S` fires, the pool growing a column at a time; parents drawn with forks
//! and retirements, and one parent past the beams. After every fire the
//! state (fires, pool, each list's live columns, free, pending), the copies,
//! the lists and the faults equal the replay's. Skipped without a PJRT plugin.

use engine::beam;
use engine_xla::bench::Bench;
use kernels_xla::beam::{At, Layout};
use kernels_xla::hlo::{Elem, Func};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

const PAGE: u32 = 16;
const FRAMES: u32 = 2;

/// One search: the replay's results a fire, and the device's.
fn search(lanes: u32, rung: u32, folded: u32, steps: u32, seed: u64) -> Option<()> {
    let f0 = folded / PAGE;
    let columns = (folded + steps - 1) / PAGE - f0 + 1;
    let total = lanes * (columns + 2) + lanes * FRAMES + 1;
    let cap = total + 8;
    let layout = Layout {
        lanes: i64::from(rung),
        cap: i64::from(cap),
        frames: i64::from(FRAMES),
        page: i64::from(PAGE),
    };
    let words = layout.words() as u32;
    let mut rng = Lcg(seed);
    let mut parents_all: Vec<i32> = Vec::new();
    let mut at_all: Vec<i32> = Vec::new();
    let mut want = Vec::new();
    let mut state = beam::State::new(lanes, cap, FRAMES);
    for t in 0..steps {
        let mut parents: Vec<u32> = if rng.next() % 3 == 0 {
            (0..lanes).collect()
        } else {
            let mut p: Vec<u32> = (0..lanes).map(|_| rng.next() % lanes).collect();
            p.sort_unstable();
            p
        };
        if t == 5 && lanes > 1 {
            parents[lanes as usize - 1] = lanes + 3;
        }
        let slot = folded + t;
        let c = slot / PAGE - f0;
        // the pool grows a column at a time (§3)
        let reserve = (lanes * (c + 2) + lanes * FRAMES + 1).min(total);
        let step = beam::step(
            &mut state,
            &parents,
            beam::Fire {
                page: PAGE,
                folded,
                reserve,
                slot,
            },
        );
        parents_all.extend(parents.iter().map(|&p| p as i32));
        // the rung's dead beams continue themselves
        parents_all.extend((lanes..rung).map(|b| b as i32));
        at_all.extend([folded as i32, reserve as i32, slot as i32, lanes as i32]);
        want.push((step, state.clone(), c));
    }

    let mut b = Bench::new();
    let n = steps;
    let parents_t = b.i32(n, rung, &parents_all);
    let at_t = b.i32(n, 4, &at_all);
    let state0 = b.i32(1, words, &vec![0; words as usize]);
    let states = b.zeros(dtype::Dtype::I32, n, words);
    let lists = b.zeros(dtype::Dtype::I32, n, rung * cap);
    let copies = b.zeros(dtype::Dtype::I32, n, rung * 3);
    let faults = b.zeros(dtype::Dtype::I32, n, rung);
    let ran = b
        .run(|ctx| {
            engine_xla::lower::lower(ctx, |l| {
                let parents_v = l.read(parents_t)?.val;
                let at_v = l.read(at_t)?.val;
                let s0 = l.read(state0)?.val;
                let f: &mut Func = l.f();
                let s0 = f.reshape(s0, &[i64::from(words)])?;
                let z =
                    |f: &mut Func, w: u32| f.const_i(Elem::I32, 0, &[i64::from(n), i64::from(w)]);
                let inits = [
                    s0,
                    z(f, words),
                    z(f, rung * cap),
                    z(f, rung * 3),
                    z(f, rung),
                ];
                let out = f.for_loop(i64::from(n), &inits, |f, ci, st| {
                    let zero = f.const_i(Elem::I32, 0, &[]);
                    let p = f.dynamic_slice(parents_v, &[ci, zero], &[1, i64::from(rung)])?;
                    let p = f.reshape(p, &[i64::from(rung)])?;
                    let a = f.dynamic_slice(at_v, &[ci, zero], &[1, 4])?;
                    let scalar = |f: &mut Func, k: i64| -> kernels_xla::hlo::Built<_> {
                        let v = f.slice_axis(a, 1, k, k + 1)?;
                        f.reshape(v, &[])
                    };
                    let at = At {
                        folded: scalar(f, 0)?,
                        reserve: scalar(f, 1)?,
                        slot: scalar(f, 2)?,
                        live: scalar(f, 3)?,
                    };
                    let step = kernels_xla::beam::step(f, layout, st[0], p, at)?;
                    let row = |f: &mut Func, v, w: u32| -> kernels_xla::hlo::Built<_> {
                        f.reshape(v, &[1, i64::from(w)])
                    };
                    let s_row = row(f, step.state, words)?;
                    let l_row = row(f, step.lists, rung * cap)?;
                    let c_row = row(f, step.copies, rung * 3)?;
                    let f_row = row(f, step.faults, rung)?;
                    Ok(vec![
                        step.state,
                        f.dynamic_update_slice(st[1], s_row, &[ci, zero])?,
                        f.dynamic_update_slice(st[2], l_row, &[ci, zero])?,
                        f.dynamic_update_slice(st[3], c_row, &[ci, zero])?,
                        f.dynamic_update_slice(st[4], f_row, &[ci, zero])?,
                    ])
                })?;
                l.land(states, out[1])?;
                l.land(lists, out[2])?;
                l.land(copies, out[3])?;
                l.land(faults, out[4])
            })
        })
        .expect("the step runs");
    if !ran {
        return None;
    }
    let got_states = b.read_i32(states);
    let got_lists = b.read_i32(lists);
    let got_copies = b.read_i32(copies);
    let got_faults = b.read_i32(faults);
    let (cap_u, lanes_u, rung_u) = (cap as usize, lanes as usize, rung as usize);
    for (t, (step, after, c)) in want.iter().enumerate() {
        let tag = format!("lanes {lanes} rung {rung} folded {folded} seed {seed} fire {t}");
        let row = |v: &[i32], w: usize| v[t * w..(t + 1) * w].to_vec();
        let st = row(&got_states, words as usize);
        assert_eq!(st[0] as u32, after.fired, "{tag}: fires");
        assert_eq!(st[1] as u32, after.pool, "{tag}: pool");
        let live = (*c as usize + 1).min(cap_u);
        for bm in 0..lanes_u {
            let at = 2 + bm * cap_u;
            let got: Vec<u32> = st[at..at + live].iter().map(|&x| x as u32).collect();
            assert_eq!(
                got,
                after.map[bm * cap_u..bm * cap_u + live],
                "{tag}: beam {bm}'s map"
            );
        }
        let at_free = 2 + rung_u * cap_u;
        let free: Vec<bool> = st[at_free..at_free + cap_u]
            .iter()
            .map(|&x| x != 0)
            .collect();
        assert_eq!(free, after.free, "{tag}: free");
        let pending: Vec<bool> = st[at_free + cap_u..].iter().map(|&x| x != 0).collect();
        assert_eq!(pending, after.pending, "{tag}: pending");
        let lists_t = row(&got_lists, rung_u * cap_u);
        for (bm, list) in step.lists.iter().enumerate() {
            let got: Vec<u32> = lists_t[bm * cap_u..bm * cap_u + list.len()]
                .iter()
                .map(|&x| x as u32)
                .collect();
            assert_eq!(&got, list, "{tag}: beam {bm}'s list");
        }
        let copies_t = row(&got_copies, rung_u * 3);
        for (bm, copy) in step.copies.iter().enumerate() {
            let got = [
                copies_t[bm * 3] as u32,
                copies_t[bm * 3 + 1] as u32,
                copies_t[bm * 3 + 2] as u32,
            ];
            if copy[2] == 0 {
                assert_eq!(got[2], 0, "{tag}: beam {bm} copies nothing");
            } else {
                assert_eq!(&got, copy, "{tag}: beam {bm}'s copy");
            }
        }
        let faults_t: Vec<u32> = row(&got_faults, rung_u).iter().map(|&x| x as u32).collect();
        assert_eq!(faults_t[..lanes_u], step.faults[..], "{tag}: faults");
        assert!(
            faults_t[lanes_u..].iter().all(|&f| f == 0),
            "{tag}: a dead beam faults"
        );
    }
    Some(())
}

#[test]
fn a_beam_step_matches_the_host_replay() {
    for (lanes, rung) in [(2u32, 2u32), (4, 4), (8, 8), (3, 4), (5, 8)] {
        for folded in [37u32, 32] {
            for seed in 0..2 {
                if search(lanes, rung, folded, 3 * PAGE + 9, seed).is_none() {
                    return;
                }
            }
        }
    }
}
