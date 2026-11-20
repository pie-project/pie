//! A beam group's head step (`channel::beam_step`) against the host replay
//! of `new-memory/designs/BEAM-PAGE-LISTS.md` (`engine::beam::step`), fire
//! by fire: widths 2, 4 and 8, a prompt ending mid-page and one ending on a
//! page, parents with forks and retirements; and the state moved to another
//! allocation mid-search (a suspend and restore of the instance's bytes),
//! after which the search goes on identically.
#![allow(unsafe_code)]
#![cfg(feature = "rocm")]

mod common;

use common::Gpu;
use engine::beam::{self, Fire, State};
use kernels_rocm::runtime::channel::{self, BeamListRun, BeamRsRun, BeamStepRun};

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
        if self.next() % 4 == 0 {
            return (0..lanes).collect();
        }
        let mut parents: Vec<u32> = (0..lanes).map(|_| self.next() % lanes).collect();
        parents.sort_unstable();
        parents
    }
}

/// The slot address the synthetic rs translation gives the set's rs entry
/// `e` (a hybrid's, `BEAM-PAGE-LISTS.md` §11).
fn rs_base(e: u32) -> u64 {
    0x7000_0000 + u64::from(e) * 0x100
}

/// Where a beam's rs page table starts in the fire's.
const RS_FROM: u32 = 3;

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
    let reserve = lanes * columns + lanes + lanes * FRAMES + u32::from(folded % PAGE > 0);
    let cap = reserve + 8;
    let words = channel::beam_state_words(lanes, cap, FRAMES) as usize;
    let mut state_at = gpu.up(&vec![0u32; words]);
    let trans: Vec<i32> = (0..reserve).map(device_id).collect();
    let trans_at = gpu.up(&trans);
    // A table each beam: its entries from `f0`, one row.
    let indices_at = gpu.up(&vec![-1i32; (lanes * cap) as usize]);
    let write_page_at = gpu.up(&vec![-1i32; lanes as usize]);
    // An rs page table each beam (its run `RS_FROM` past the last's), and
    // the set's rs entries' slot addresses, every slot of its reserve.
    let rs_entries_set = ((f0 + reserve) * PAGE - rs_from_slot).div_ceil(rs_tokens);
    let rs_span = RS_FROM + rs_entries_set;
    let rs_pages_at = gpu.up(&vec![0u64; (lanes * rs_span) as usize]);
    let rs_bases: Vec<u64> = (0..rs_entries_set).map(rs_base).collect();
    let rs_bases_at = gpu.up(&rs_bases);
    let mut replay = State::new(lanes, cap, FRAMES);
    let mut rng = Lcg(seed);
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
        let pool = (lanes * (c + 2) + lanes * FRAMES + u32::from(folded % PAGE > 0)).min(reserve);
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
        let rs_entries = (slot - rs_from_slot) / rs_tokens + 1;
        let rs_runs: Vec<BeamRsRun> = (0..lanes)
            .map(|b| BeamRsRun {
                lanes_tab: 0,
                replay_tok: 0,
                resolved: 0,
                pages: rs_pages_at,
                bases: rs_bases_at,
                lane: b,
                from: b * rs_span + RS_FROM,
                entries: rs_entries,
                base: rs_from_slot,
                tokens: rs_tokens,
                bases_len: rs_entries_set,
            })
            .collect();
        let rs_at = gpu.up(&rs_runs);
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
            rs_count: lanes,
            rs: rs_at,
        };
        let run_at = gpu.up(&[run]);
        channel::beam_step(&ctx, run_at, 1).expect("beam_step launches");
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
        // A hybrid's rs page tables (§11), slot by slot: entry e (slot
        // `base + e T`) holds the set's rows of slot (f0 + list[j]) S +
        // slot mod S, its write page among them; entries on the frozen
        // pages are left as staged.
        let rs_resolved: Vec<u64> = gpu.down(rs_pages_at, (lanes * rs_span) as usize);
        for b in 0..lanes as usize {
            let at_b = b * rs_span as usize + RS_FROM as usize;
            for e in 0..rs_entries {
                let s = rs_from_slot + e * rs_tokens;
                let want_rs = if s < f0 * PAGE {
                    0
                } else {
                    let q = want.lists[b][(s / PAGE - f0) as usize];
                    rs_base(((f0 + q) * PAGE + s % PAGE - rs_from_slot) / rs_tokens)
                };
                assert_eq!(rs_resolved[at_b + e as usize], want_rs, "{at}: beam {b}'s rs entry {e}");
            }
        }
    }
}

#[test]
fn the_beam_step_follows_the_spec() {
    let Some(gpu) = Gpu::open() else { return };
    for lanes in [2, 4, 8] {
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
            rs_count: 0,
            rs: 0,
        };
        channel::beam_step(&ctx, gpu.up(&[run]), 1).expect("beam_step launches");
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
