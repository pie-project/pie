//! A beam group's head step (`channel::beam_step`) against the host replay
//! of `new-memory/designs/BEAM-PAGE-LISTS.md` (`engine::beam::step`), fire
//! by fire: widths 2, 4 and 8, a prompt ending mid-page and one ending on a
//! page, parents with forks and retirements; and the state moved to another
//! allocation mid-search (a suspend and restore of the instance's bytes),
//! after which the search goes on identically. On a hybrid each beam's rs
//! row lands, through its group's list as the step left it
//! (`channel::RsRows`), on the set's row of the slot §11.1 names.
#![allow(unsafe_code)]
#![cfg(feature = "rocm")]

mod common;

use common::Gpu;
use engine::beam::{self, Fire, State};
use kernels_rocm::runtime::channel::{self, BeamListRun, RsRows, BeamStepRun, RsLane, RsPlane, RsTables};

const PAGE: u32 = 16;
const FRAMES: u32 = 2;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Each beam's parent: none moved, or sorted draws (forks where one
    /// repeats, retirements where one is skipped).
    fn parents(&mut self, lanes: u32) -> Vec<u32> {
        if self.next().is_multiple_of(4) {
            return (0..lanes).collect();
        }
        let mut parents: Vec<u32> = (0..lanes).map(|_| self.next() % lanes).collect();
        // Most fires take their parents in order, as a top-k by score need
        // not: every third keeps the draw's order.
        if !self.next().is_multiple_of(3) {
            parents.sort_unstable();
        }
        parents
    }
}

/// The bytes of a beam's rs row in the synthetic set.
const RS_ROW: u32 = 4;

/// The device id the synthetic translation gives pool page `q`.
fn device_id(q: u32) -> i32 {
    1000 + q as i32
}

fn search(gpu: &Gpu, lanes: u32, folded: u32, steps: u32, seed: u64) {
    // The rs buffer pages `rs_tokens` rows from the frozen slot rounded to
    // them: off a kv page where the prompt ends mid-page (the live frozen
    // slot 38 at 2 rows a page).
    for rs_tokens in [1, 2, 4] {
        search_rs(gpu, lanes, folded, steps, seed, rs_tokens);
    }
}

fn search_rs(gpu: &Gpu, lanes: u32, folded: u32, steps: u32, seed: u64, rs_tokens: u32) {
    let rs_from_slot = folded - folded % rs_tokens;
    let ctx = gpu.ctx();
    let f0 = folded / PAGE;
    let columns = (folded + steps - 1) / PAGE - f0 + 1;
    let reserve =
        lanes * columns + lanes + lanes * FRAMES + u32::from(!folded.is_multiple_of(PAGE));
    let cap = reserve + 8;
    let words = channel::beam_state_words(lanes, cap, FRAMES) as usize;
    let mut state_at = gpu.up(&vec![0u32; words]);
    let trans: Vec<i32> = (0..reserve).map(device_id).collect();
    let trans_at = gpu.up(&trans);
    // A table each beam: its entries from `f0`, one row.
    let indices_at = gpu.up(&vec![-1i32; (lanes * cap) as usize]);
    let write_page_at = gpu.up(&vec![-1i32; lanes as usize]);
    // The set's rs entries, every slot of its reserve, `rs_tokens` rows of
    // `RS_ROW` bytes each, and their slot addresses.
    let rs_entries_set = ((f0 + reserve) * PAGE - rs_from_slot).div_ceil(rs_tokens);
    let rs_entry_bytes = u64::from(rs_tokens * RS_ROW);
    let rs_rows_at = gpu.up(&vec![0u32; (rs_entries_set * rs_tokens) as usize]);
    let rs_bases: Vec<u64> = (0..rs_entries_set)
        .map(|e| rs_rows_at + u64::from(e) * rs_entry_bytes)
        .collect();
    let rs_bases_at = gpu.up(&rs_bases);
    let indptr_at = gpu.up(&(0..=lanes).collect::<Vec<u32>>());
    let mut replay = State::new(lanes, cap, FRAMES);
    let mut rng = Lcg(seed);
    // Each live beam's history, a token a fire unique to (fire, beam): a
    // child's is its parent's and its own, so siblings part at their row.
    let mut histories: Vec<Vec<(u32, u32)>> = Vec::new();
    for t in 0..steps {
        if t == steps / 2 {
            // Suspend and restore: the state's bytes at another address.
            let held: Vec<u32> = gpu.down(state_at, words);
            state_at = gpu.up(&held);
        }
        let slot = folded + t;
        let c = slot / PAGE - f0;
        let o = slot % PAGE;
        // The pool grows a column at a time: what this fire's column needs.
        let pool = (lanes * (c + 2) + lanes * FRAMES + u32::from(!folded.is_multiple_of(PAGE)))
            .min(reserve);
        let parents = rng.parents(lanes);
        let parents_at = gpu.up(&parents);
        let offset_at = gpu.up(&vec![o as i32; lanes as usize]);
        let faults_at = gpu.up(&vec![0u32; lanes as usize]);
        let lists: Vec<BeamListRun> = (0..lanes)
            .map(|b| BeamListRun {
                indices: indices_at + u64::from(b * cap) * 4,
                write_page: write_page_at + u64::from(b) * 4,
                write_offset: offset_at + u64::from(b) * 4,
                lane: b,
                entries: c + 1,
                rows: 1,
                _pad: 0,
            })
            .collect();
        let lists_at = gpu.up(&lists);
        // A hybrid's beams' rs rows (§11.1): the step points each at its list.
        let beams: Vec<RsRows> = (0..lanes)
            .map(|b| RsRows {
                list: state_at + u64::from(channel::BEAM_HEADER + b * cap) * 4,
                bases: rs_bases_at,
                cap,
                f0,
                page: PAGE,
                base: rs_from_slot,
                tokens: rs_tokens,
                bases_len: rs_entries_set,
            })
            .collect();
        let beams_at = gpu.up(&beams);
        let run = BeamStepRun {
            state: state_at,
            parents: parents_at,
            trans: trans_at,
            lists: lists_at,
            faults: faults_at,
            lanes,
            cap,
            reserve: pool,
            page: PAGE,
            folded,
            slot,
            list_count: lanes,
            frames: FRAMES,
            // The pages the engine stages: the lowest-free rule keeps every
            // page the step names below them, or a lane faults.
            backed: pool,
            rows: beams_at,
        };
        let run_at = gpu.up(&[run]);
        channel::beam_step(&ctx, run_at, 1, lanes).expect("beam_step launches");
        gpu.sync();
        let want = beam::step(
            &mut replay,
            &parents,
            Fire {
                page: PAGE,
                folded,
                reserve: pool,
                slot,
            },
        );
        let held: Vec<u32> = gpu.down(state_at, words);
        let at = format!("lanes {lanes} folded {folded} seed {seed} rs {rs_tokens} fire {t}");
        assert_eq!(held[0], 1, "{at}: primed");
        assert_eq!(held[1], replay.fired, "{at}: fires run");
        assert_eq!(held[2], replay.pool, "{at}: the pool's pages");
        let map = 4 + (replay.fired as usize & 1) * (lanes * cap) as usize;
        for b in 0..lanes as usize {
            let row = &held[map + b * cap as usize..map + b * cap as usize + c as usize + 1];
            assert_eq!(row, &want.lists[b][..], "{at}: beam {b}'s list");
        }
        // The common slot (§11.3, §11.4): the live beams' longest common
        // history past the prompt, by brute force over their histories,
        // capped at the slot past the fire's row.
        histories = (0..lanes)
            .map(|b| {
                let mut history = if t == 0 {
                    Vec::new()
                } else {
                    histories[parents[b as usize].min(lanes - 1) as usize].clone()
                };
                history.push((t, b));
                history
            })
            .collect();
        let common_len = (0..=t as usize)
            .take_while(|&k| histories.iter().all(|h| h.get(k) == histories[0].get(k)))
            .count() as u32;
        let common = (folded + common_len).min(slot + 1);
        assert_eq!(held[3], common, "{at}: the common slot");
        let free = 4 + 2 * (lanes * cap) as usize;
        let device_free: Vec<bool> = held[free..free + cap as usize].iter().map(|&w| w != 0).collect();
        assert_eq!(device_free, replay.free, "{at}: free pages");
        let pending = free + cap as usize;
        let device_pending: Vec<bool> = held[pending..pending + (FRAMES * cap) as usize]
            .iter()
            .map(|&w| w != 0)
            .collect();
        assert_eq!(device_pending, replay.pending, "{at}: pending pages");
        let copies = pending + (FRAMES * cap) as usize;
        for b in 0..lanes as usize {
            let copy = [held[copies + 3 * b], held[copies + 3 * b + 1], held[copies + 3 * b + 2]];
            let wanted = if want.copies[b][2] == 0 { [0, 0, 0] } else { want.copies[b] };
            assert_eq!(copy, wanted, "{at}: beam {b}'s copy");
        }
        let faults: Vec<u32> = gpu.down(faults_at, lanes as usize);
        assert_eq!(faults, want.faults, "{at}: faults");
        let resolved: Vec<i32> = gpu.down(indices_at, (lanes * cap) as usize);
        let writes: Vec<i32> = gpu.down(write_page_at, lanes as usize);
        for b in 0..lanes as usize {
            let list: Vec<i32> = want.lists[b].iter().map(|&q| device_id(q)).collect();
            let entries = &resolved[b * cap as usize..b * cap as usize + c as usize + 1];
            assert_eq!(entries, &list[..], "{at}: beam {b}'s resolved entries");
            assert_eq!(writes[b], device_id(want.lists[b][c as usize]), "{at}: beam {b}'s write page");
        }
        // A hybrid's rows (§11.1): each beam lands its row at this fire's
        // slot through the list the step pointed it at, on the set's row of
        // slot (f0 + list[j]) S + slot mod S.
        let rs_lanes: Vec<RsLane> = (0..lanes)
            .map(|b| RsLane {
                at: slot - rs_from_slot,
                replay: 0,
                replay_off: 0,
                page_off: RsLane::ROWS | b,
                ext: 0,
            })
            .collect();
        let rows: Vec<u32> = (0..lanes).map(|b| (t << 8) | b | 0x8000_0000).collect();
        let tables = RsTables {
            lanes: gpu.up(&rs_lanes),
            pages: 0,
            lane_rows: beams_at,
            replay_spans: 0,
            page_tokens: rs_tokens,
        };
        let plane = RsPlane { off: 0, stride: u64::from(RS_ROW) };
        channel::rs_scatter(&ctx, gpu.up(&rows), RS_ROW, indptr_at, lanes, tables, plane)
            .expect("rs_scatter launches");
        gpu.sync();
        let landed: Vec<u32> = gpu.down(rs_rows_at, (rs_entries_set * rs_tokens) as usize);
        for b in 0..lanes as usize {
            let q = want.lists[b][c as usize];
            let sigma = (f0 + q) * PAGE + slot % PAGE - rs_from_slot;
            assert_eq!(landed[sigma as usize], rows[b], "{at}: beam {b}'s rs row");
        }
    }
}

#[test]
fn the_beam_step_follows_the_spec() {
    let Some(gpu) = Gpu::open() else { return };
    for lanes in [1, 2, 4, 8, 96] {
        for folded in [37, 32] {
            for seed in 0..3 {
                search(&gpu, lanes, folded, 3 * PAGE + 9, seed);
            }
        }
    }
}

/// A parent past the group faults that lane alone, which continues itself:
/// its pages stay named.
#[test]
fn a_parent_past_the_group_faults_its_lane() {
    let Some(gpu) = Gpu::open() else { return };
    let (lanes, folded) = (4u32, 37u32);
    let ctx = gpu.ctx();
    let reserve = 4 * 4 + 4 + 4 * FRAMES + 1;
    let cap = reserve;
    let words = channel::beam_state_words(lanes, cap, FRAMES) as usize;
    let state_at = gpu.up(&vec![0u32; words]);
    let trans_at = gpu.up(&(0..reserve).map(device_id).collect::<Vec<_>>());
    let mut replay = State::new(lanes, cap, FRAMES);
    for (t, parents) in [vec![0, 1, 2, 3], vec![0, 0, 9, 1]].into_iter().enumerate() {
        let slot = folded + t as u32;
        let faults_at = gpu.up(&vec![0u32; lanes as usize]);
        let run = BeamStepRun {
            state: state_at,
            parents: gpu.up(&parents),
            trans: trans_at,
            lists: gpu.up(&[BeamListRun::default()]),
            faults: faults_at,
            lanes,
            cap,
            reserve,
            page: PAGE,
            folded,
            slot,
            list_count: 0,
            frames: FRAMES,
            backed: reserve,
            rows: 0,
        };
        channel::beam_step(&ctx, gpu.up(&[run]), 1, lanes).expect("beam_step launches");
        gpu.sync();
        let want = beam::step(
            &mut replay,
            &parents,
            Fire {
                page: PAGE,
                folded,
                reserve,
                slot,
            },
        );
        let faults: Vec<u32> = gpu.down(faults_at, lanes as usize);
        assert_eq!(faults, want.faults, "fire {t}");
    }
    assert_eq!(replay.fired, 2);
}

/// A fork's rs copy on a hybrid (`BEAM-PAGE-LISTS.md` §6.4, §11): rows of
/// 2 a page from slot 38, kv pages of 16 (the live Qwen3.5 frozen slot),
/// two planes of other row bytes. A copy of 5 slots from pool page 1 to 3
/// ends mid rs page; one from the prompt's own page 0 skips its tail's
/// slots below the buffer (they are in the state); each destination row
/// equals its source row, every other byte stays.
#[test]
fn a_fork_copies_its_rs_rows_slot_by_slot() {
    use kernels_rocm::runtime::channel::{PagePlane, RsCopyRun};
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    const T: u32 = 2;
    const BASE: u32 = 38;
    let f0 = BASE / PAGE;
    let planes = [PagePlane { offset: 0, cell: 8 }, PagePlane { offset: 16, cell: 12 }];
    let entry_bytes = 16 + 2 * 12;
    let entries: u32 = 64;
    let pattern: Vec<u8> = (0..entries as usize * entry_bytes)
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    for (src, dst, tokens) in [(1u32, 3u32, 5u32), (0, 2, 9)] {
        let mem = gpu.up(&pattern);
        let bases: Vec<u64> = (0..u64::from(entries)).map(|e| mem + e * entry_bytes as u64).collect();
        let bases_at = gpu.up(&bases);
        let planes_at = gpu.up(&planes);
        let copy_at = gpu.up(&[src, dst, tokens]);
        let fault_at = gpu.up(&[0u32]);
        let run = RsCopyRun {
            copy: copy_at,
            bases: bases_at,
            fault: fault_at,
            reserve: 8,
            page: PAGE,
            tokens: T,
            base: BASE,
            f0,
            floor_rel: BASE - f0 * PAGE,
            plane_at: 0,
            planes: 2,
            bases_len: entries,
            _pad: 0,
        };
        let run_at = gpu.up(&[run]);
        channel::copy_rs_rows(&ctx, run_at, planes_at, 1, 2).expect("copy_rs_rows launches");
        gpu.sync();
        let after: Vec<u8> = gpu.down(mem, pattern.len());
        let fault: Vec<u32> = gpu.down(fault_at, 1);
        assert_eq!(fault, [0], "copy {src} -> {dst} x {tokens}: no fault");
        let row = |slot: u32, plane: &PagePlane| -> std::ops::Range<usize> {
            let (e, r) = ((slot - BASE) / T, (slot - BASE) % T);
            let at = e as usize * entry_bytes + plane.offset as usize + r as usize * plane.cell as usize;
            at..at + plane.cell as usize
        };
        let mut want = pattern.clone();
        for k in 0..tokens {
            let (from, to) = ((f0 + src) * PAGE + k, (f0 + dst) * PAGE + k);
            if from < BASE || to < BASE {
                continue;
            }
            for plane in &planes {
                let bytes = pattern[row(from, plane)].to_vec();
                want[row(to, plane)].copy_from_slice(&bytes);
            }
        }
        assert_eq!(after, want, "copy {src} -> {dst} x {tokens}: rows moved, nothing else");
    }
    // A copy past the pool faults and moves nothing.
    let mem = gpu.up(&pattern);
    let bases: Vec<u64> = (0..u64::from(entries)).map(|e| mem + e * entry_bytes as u64).collect();
    let run = RsCopyRun {
        copy: gpu.up(&[1u32, 9, 4]),
        bases: gpu.up(&bases),
        fault: gpu.up(&[0u32]),
        reserve: 8,
        page: PAGE,
        tokens: T,
        base: BASE,
        f0,
        floor_rel: BASE - f0 * PAGE,
        plane_at: 0,
        planes: 2,
        bases_len: entries,
        _pad: 0,
    };
    let fault_at = run.fault;
    let run_at = gpu.up(&[run]);
    channel::copy_rs_rows(&ctx, run_at, gpu.up(&planes), 1, 2).expect("copy_rs_rows launches");
    gpu.sync();
    assert_eq!(gpu.down::<u32>(fault_at, 1), [channel::PAGES_COPY_BOUNDS]);
    assert_eq!(gpu.down::<u8>(mem, pattern.len()), pattern, "a refused copy moves nothing");
}

/// A group wider than its step was laid for steps nothing and faults every
/// one of its lanes by name ([`channel::PAGES_TOO_WIDE`]), never returning
/// silently; the device steps groups past 64 beams
/// ([`channel::beam_max_lanes`]).
#[test]
fn a_group_wider_than_its_step_faults_every_lane_by_name() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    assert!(channel::beam_max_lanes(&ctx) >= 96, "this device steps 96 beams a group");
    let (lanes, folded) = (8u32, 37u32);
    let reserve = 4 * lanes + lanes + lanes * FRAMES + 1;
    let cap = reserve;
    let words = channel::beam_state_words(lanes, cap, FRAMES) as usize;
    let state_at = gpu.up(&vec![0u32; words]);
    let trans_at = gpu.up(&(0..reserve).map(device_id).collect::<Vec<_>>());
    let faults_at = gpu.up(&vec![0u32; lanes as usize]);
    let run = BeamStepRun {
        state: state_at,
        parents: gpu.up(&(0..lanes).collect::<Vec<u32>>()),
        trans: trans_at,
        lists: gpu.up(&[BeamListRun::default()]),
        faults: faults_at,
        lanes,
        cap,
        reserve,
        page: PAGE,
        folded,
        slot: folded,
        list_count: 0,
        frames: FRAMES,
        backed: reserve,
        rows: 0,
    };
    let run_at = gpu.up(&[run]);
    // Laid for 4 beams: the group of 8 is wider.
    channel::beam_step(&ctx, run_at, 1, 4).expect("beam_step launches");
    gpu.sync();
    let faults: Vec<u32> = gpu.down(faults_at, lanes as usize);
    assert_eq!(faults, vec![channel::PAGES_TOO_WIDE; lanes as usize], "every lane faults wide");
    let held: Vec<u32> = gpu.down(state_at, words);
    assert_eq!(held[0], 0, "the group was not primed: nothing stepped");
}
