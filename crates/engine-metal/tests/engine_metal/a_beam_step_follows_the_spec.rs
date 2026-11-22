#![cfg(target_vendor = "apple")]

//! BEAM-PAGE-LISTS §9 on Metal: the device's `beam_step` against the host
//! replay (`engine::beam::step`), fire by fire, at widths 2, 4 and 8, from a
//! `folded` with a tail and one without, over `3S + 9` fires with forks and
//! retirements: every beam's list (its live columns), the free and pending
//! pages, the copies it leaves, its faults, the table entries it resolves
//! and its write pages equal the replay's; the state copied out and back
//! mid-search (a suspend and restore) continues identically. Then
//! `beam_copy_pages` copies a fork's tail slots of every plane, and a copy
//! past the pool copies nothing and faults its lane.
//!
//! Every record and array lies in one buffer bound as the kernel's argument
//! (resident), the run first; the records' addresses point into it.

use engine::beam::{self, Fire, State};
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::runtime::beam::{
    self as kbeam, BeamListRun, BeamRsRun, BeamStepRun, PageCopyRun, PagePlane, Record, RsCopyRun,
};
use poem_ir::Dtype;

const PAGE: u32 = 16;
const FRAMES: u32 = 2;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Parents with forks and retirements; sometimes every beam its own.
    fn parents(&mut self, lanes: u32) -> Vec<u32> {
        if self.next() % 4 == 0 {
            return (0..lanes).collect();
        }
        let mut parents: Vec<u32> = (0..lanes).map(|_| self.next() % lanes).collect();
        parents.sort_unstable();
        parents
    }
}

fn device_id(q: u32) -> i32 {
    1000 + q as i32
}

fn bytes_of<R: Record>(v: &[R]) -> Vec<u8> {
    kbeam::bytes(v)
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One buffer laid by hand: `take` hands out aligned offsets.
struct Arena {
    at: u64,
}

impl Arena {
    fn take(&mut self, bytes: u64) -> u64 {
        let at = self.at.next_multiple_of(16);
        self.at = at + bytes.max(4);
        at
    }
}

struct Dev {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
}

impl Dev {
    fn open() -> Option<Dev> {
        let device = Context::bind().ok()?;
        Some(Dev {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
        })
    }

    fn run(&self, fire: &dyn Fn(&Sink)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit().expect("the commit");
    }

    fn read(&self, buffer: &Buffer, at: u64, bytes: u64) -> Vec<u8> {
        let h = self.handles.bind(buffer, at, bytes).expect("a handle");
        self.handles.read(h, bytes).expect("read")
    }
}

/// The set's rs entry `a` as the tests stage it: a slab slot id.
fn set_entry(a: u32) -> i32 {
    5000 + a as i32
}

/// A search over `steps` fires; with `rs_t > 0` a hybrid's too: each beam's
/// rs page table of `rs_t` rows a page from a buffer base off the kv page
/// boundary, resolved by the step (§11.1) and checked fire by fire.
fn search(dev: &Dev, lanes: u32, folded: u32, steps: u32, seed: u64, rs_t: u32) {
    let f0 = folded / PAGE;
    let columns = (folded + steps - 1) / PAGE - f0 + 1;
    let reserve = lanes * columns + lanes + lanes * FRAMES + u32::from(folded % PAGE > 0);
    let cap = reserve + 8;
    let state_words = kbeam::beam_state_words(lanes, cap, FRAMES);
    let mut arena = Arena { at: 0 };
    let run_at = arena.take(std::mem::size_of::<BeamStepRun>() as u64);
    let lists_at = arena.take(u64::from(lanes) * std::mem::size_of::<BeamListRun>() as u64);
    let state_at = arena.take(state_words * 4);
    let trans_at = arena.take(u64::from(reserve) * 4);
    let parents_at = arena.take(u64::from(lanes) * 4);
    let offset_at = arena.take(u64::from(lanes) * 4);
    let faults_at = arena.take(u64::from(lanes) * 4);
    let indices_at = arena.take(u64::from(lanes * cap) * 4);
    let writes_at = arena.take(u64::from(lanes) * 4);
    // A hybrid's rs tables: the buffer based at a multiple of `rs_t` below
    // `folded` (two rs pages below its kv page's start where it can be, so
    // frozen entries stand below `f0 S`), entries through the last fire's.
    let rt = rs_t.max(1);
    let rs_base = ((f0 * PAGE).saturating_sub(2 * rt) / rt) * rt;
    let rs_cap = (folded + steps - rs_base) / rt + 1;
    let rs_len = ((f0 + reserve) * PAGE - rs_base) / rt;
    let rs_runs_at = arena.take(u64::from(lanes) * std::mem::size_of::<BeamRsRun>() as u64);
    let rs_pages_at = arena.take(u64::from(lanes * rs_cap) * 4);
    let rs_bases_at = arena.take(u64::from(rs_len) * 4);
    let mut buffer = Buffer::zeroed(&dev.device, arena.at).expect("a buffer");
    let base = buffer.address_at(0).expect("an address");
    let address = move |at: u64| base + at;
    let (a_state, a_trans, a_parents, a_offset, a_faults, a_indices, a_writes, a_lists) = (
        address(state_at),
        address(trans_at),
        address(parents_at),
        address(offset_at),
        address(faults_at),
        address(indices_at),
        address(writes_at),
        address(lists_at),
    );
    let trans: Vec<i32> = (0..reserve).map(device_id).collect();
    buffer.write(trans_at, &bytes_of(&trans)).expect("trans");
    let set: Vec<i32> = (0..rs_len).map(set_entry).collect();
    buffer.write(rs_bases_at, &bytes_of(&set)).expect("the set's rs entries");
    let mut replay = State::new(lanes, cap, FRAMES);
    let mut rng = Lcg(seed);
    let runs = dev.handles.bind(&buffer, 0, buffer.bytes()).expect("the runs");
    // Each pair of lanes' common-ancestor depth (§11.4), as the head step
    // keeps it.
    let mut lca: Vec<u32> = vec![0; (lanes * lanes) as usize];
    for t in 0..steps {
        if t == steps / 2 {
            // A suspend and restore: the state out to the host and back.
            let held = dev.read(&buffer, state_at, state_words * 4);
            buffer.write(state_at, &vec![0u8; held.len()]).expect("cleared");
            buffer.write(state_at, &held).expect("restored");
        }
        let slot = folded + t;
        let c = slot / PAGE - f0;
        let o = slot % PAGE;
        let pool = (lanes * (c + 2) + lanes * FRAMES + u32::from(folded % PAGE > 0)).min(reserve);
        let parents = rng.parents(lanes);
        buffer.write(parents_at, &bytes_of(&parents)).expect("parents");
        buffer
            .write(offset_at, &bytes_of(&vec![o as i32; lanes as usize]))
            .expect("offsets");
        buffer
            .write(faults_at, &vec![0u8; lanes as usize * 4])
            .expect("faults");
        buffer
            .write(indices_at, &bytes_of(&vec![-1i32; (lanes * cap) as usize]))
            .expect("indices");
        buffer
            .write(writes_at, &bytes_of(&vec![-1i32; lanes as usize]))
            .expect("writes");
        let lists: Vec<BeamListRun> = (0..lanes)
            .map(|b| BeamListRun {
                indices: a_indices + u64::from(b * cap) * 4,
                write_page: a_writes + u64::from(b) * 4,
                write_offset: a_offset + u64::from(b) * 4,
                lane: b,
                entries: c + 1,
                rows: 1,
                pad: 0,
            })
            .collect();
        buffer.write(lists_at, &bytes_of(&lists)).expect("lists");
        // Each beam's rs row, staged as the set's (-1 here, so an entry the
        // step leaves reads as left), entries through this fire's row.
        let rs_entries = (slot - rs_base) / rt + 1;
        if rs_t > 0 {
            buffer
                .write(rs_pages_at, &bytes_of(&vec![-1i32; (lanes * rs_cap) as usize]))
                .expect("rs pages");
            let rs_runs: Vec<BeamRsRun> = (0..lanes)
                .map(|b| BeamRsRun {
                    pages: address(rs_pages_at) + u64::from(b * rs_cap) * 4,
                    bases: address(rs_bases_at),
                    lane: b,
                    first: 0,
                    entries: rs_entries,
                    base: rs_base,
                    tokens: rs_t,
                    bases_len: rs_len,
                })
                .collect();
            buffer.write(rs_runs_at, &bytes_of(&rs_runs)).expect("rs runs");
        }
        let run = BeamStepRun {
            state: a_state,
            parents: a_parents,
            trans: a_trans,
            lists: a_lists,
            faults: a_faults,
            lanes,
            cap,
            reserve: pool,
            page: PAGE,
            folded,
            slot,
            list_count: lanes,
            frames: FRAMES,
            backed: pool,
            rs_count: if rs_t > 0 { lanes } else { 0 },
            rs: if rs_t > 0 { address(rs_runs_at) } else { 0 },
        };
        buffer.write(run_at, &bytes_of(&[run])).expect("the run");
        dev.run(&|sink| {
            let width = std::mem::size_of::<BeamStepRun>() as u32;
            kbeam::beam_step(sink, Tensor::new(runs, 1, width, Dtype::U8), 1).expect("beam_step encodes");
        });
        // §11.4 on the host: siblings part at this fire's row, other pairs
        // agree as far as their parents did; priming, every pair shares the
        // prompt to here.
        let primed = replay.fired > 0;
        let next: Vec<u32> = (0..lanes * lanes)
            .map(|x| {
                let (i, j) = ((x / lanes) as usize, (x % lanes) as usize);
                if i == j {
                    u32::MAX
                } else if !primed {
                    slot
                } else {
                    // A parent past the lanes continues itself, as the device's.
                    let of = |b: usize| if parents[b] < lanes { parents[b] } else { b as u32 };
                    let (pi, pj) = (of(i), of(j));
                    if pi == pj { slot } else { lca[(pi * lanes + pj) as usize] }
                }
            })
            .collect();
        lca = next;
        let common = lca.iter().copied().min().filter(|&d| d != u32::MAX).unwrap_or(slot + 1).min(slot + 1);
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
        let held = words(&dev.read(&buffer, state_at, state_words * 4));
        let at = format!("lanes {lanes} folded {folded} seed {seed} fire {t}");
        assert_eq!(held[0], 1, "{at}: primed");
        assert_eq!(held[1], replay.fired, "{at}: fires run");
        assert_eq!(held[2], replay.pool, "{at}: the pool's pages");
        // §11.4: the common slot, the least pair's common-ancestor depth,
        // capped at the slot past this fire's row, never below the folded.
        assert_eq!(held[3], common.max(folded), "{at}: the group's common slot");
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
        let faults = words(&dev.read(&buffer, faults_at, u64::from(lanes) * 4));
        assert_eq!(faults, want.faults, "{at}: faults");
        let resolved = words(&dev.read(&buffer, indices_at, u64::from(lanes * cap) * 4));
        let writes = words(&dev.read(&buffer, writes_at, u64::from(lanes) * 4));
        for b in 0..lanes as usize {
            let list: Vec<u32> = want.lists[b].iter().map(|&q| device_id(q) as u32).collect();
            let entries = &resolved[b * cap as usize..b * cap as usize + c as usize + 1];
            assert_eq!(entries, &list[..], "{at}: beam {b}'s resolved entries");
            assert_eq!(
                writes[b],
                device_id(want.lists[b][c as usize]) as u32,
                "{at}: beam {b}'s write page"
            );
        }
        if rs_t > 0 {
            // §11.1 over the replay's map': a frozen entry left, every other
            // the set's entry holding the same rows of the beam's list's page.
            let held = words(&dev.read(&buffer, rs_pages_at, u64::from(lanes * rs_cap) * 4));
            for b in 0..lanes as usize {
                for e in 0..rs_entries {
                    let rs_slot = rs_base + e * rt;
                    let want_entry = if rs_slot < f0 * PAGE {
                        -1i32
                    } else {
                        let q = want.lists[b][(rs_slot / PAGE - f0) as usize];
                        let sigma = (f0 + q) * PAGE + rs_slot % PAGE;
                        set_entry((sigma - rs_base) / rt)
                    };
                    assert_eq!(
                        held[b * rs_cap as usize + e as usize] as i32,
                        want_entry,
                        "{at}: T {rs_t} base {rs_base}: beam {b}'s rs entry {e} (slot {rs_slot})"
                    );
                }
            }
        }
    }
}

/// §11.2 over the four planes of a gated-delta row (conv x, [b|a], keys,
/// u: four cells of different widths), rs pages of `T` = 2 rows from a
/// buffer base at slot 38 (off the kv page of 16): a fork's copy that ends
/// mid rs page, one from the prompt's own page (its slots below the base
/// skipped), and one out of bounds that faults and moves nothing. Every
/// destination row is its source row; every other byte stays.
fn rs_copies_move_every_plane(dev: &Dev) {
    const S: u32 = 16;
    const T: u32 = 2;
    const BASE: u32 = 38;
    let f0 = BASE / S;
    let reserve = 6u32;
    let planes = [
        PagePlane { offset: 0, cell: 24 },
        PagePlane { offset: 2 * 24, cell: 8 },
        PagePlane { offset: 2 * 24 + 2 * 8, cell: 64 },
        PagePlane { offset: 2 * 24 + 2 * 8 + 2 * 64, cell: 128 },
    ];
    let row_bytes: u64 = planes.iter().map(|p| 2 * p.cell).sum();
    let page_bytes = row_bytes.next_multiple_of(256);
    let entries = ((f0 + reserve) * S - BASE) / T;
    // The set's entries on slab slots in reverse, so an entry is not its
    // slot: a copy that confused them would land elsewhere.
    let set: Vec<i32> = (0..entries).map(|a| (entries - 1 - a) as i32).collect();
    let mut arena = Arena { at: 0 };
    let runs_at = arena.take(3 * std::mem::size_of::<RsCopyRun>() as u64);
    let planes_at = arena.take(std::mem::size_of_val(&planes) as u64);
    let copy_at = arena.take(9 * 4);
    let fault_at = arena.take(3 * 4);
    let set_at = arena.take(u64::from(entries) * 4);
    let slab_at = arena.take(u64::from(entries) * page_bytes).next_multiple_of(256);
    let mut buffer = Buffer::zeroed(&dev.device, slab_at + u64::from(entries) * page_bytes + 256)
        .expect("a buffer");
    let base = buffer.address_at(0).expect("an address");
    let address = move |at: u64| base + at;
    let content: Vec<u8> = (0..u64::from(entries) * page_bytes)
        .map(|i| (i * 13 + 5) as u8)
        .collect();
    buffer.write(slab_at, &content).expect("the slab");
    buffer.write(set_at, &bytes_of(&set)).expect("the set's entries");
    buffer.write(planes_at, &bytes_of(&planes)).expect("planes");
    // Beam 0: 5 slots of pool page 1 onto page 3 (ends mid rs page). Beam 1:
    // 9 slots of page 0 (the prompt's own, from slot 32: 32..37 below the
    // base) onto page 4. Beam 2: a copy onto page 9, past the pool.
    let copies = [[1u32, 3, 5], [0, 4, 9], [0, 9, 5]];
    buffer
        .write(copy_at, &bytes_of(&copies.concat()))
        .expect("copies");
    let run = |beam: u64| RsCopyRun {
        copy: address(copy_at + beam * 12),
        bases: address(set_at),
        fault: address(fault_at + beam * 4),
        slab: address(slab_at),
        reserve,
        page: S,
        tokens: T,
        base: BASE,
        f0,
        floor_rel: 0,
        plane_at: 0,
        planes: 4,
        bases_len: entries,
        page_bytes: u32::try_from(page_bytes).expect("a page's bytes"),
    };
    buffer.write(runs_at, &bytes_of(&[run(0), run(1), run(2)])).expect("runs");
    let runs = dev.handles.bind(&buffer, runs_at, buffer.bytes() - runs_at).expect("runs");
    let planes_h = dev
        .handles
        .bind(&buffer, planes_at, std::mem::size_of_val(&planes) as u64)
        .expect("planes");
    let width = std::mem::size_of::<RsCopyRun>() as u32;
    dev.run(&|sink| {
        kbeam::copy_rs_rows(
            sink,
            Tensor::new(runs, 3, width, Dtype::U8),
            Tensor::new(planes_h, 4, 16, Dtype::U8),
            3,
            4,
        )
        .expect("copy_rs_rows encodes");
    });
    let after = dev.read(&buffer, slab_at, u64::from(entries) * page_bytes);
    let mut want = content.clone();
    let row_at = |slot: u32, plane: &PagePlane| -> usize {
        let a = (slot - BASE) / T;
        (u64::from(set[a as usize] as u32) * page_bytes
            + plane.offset
            + u64::from((slot - BASE) % T) * plane.cell) as usize
    };
    for [src, dst, tokens] in copies {
        if dst >= reserve || src >= reserve {
            continue;
        }
        for k in 0..tokens {
            let (from, to) = ((f0 + src) * S + k, (f0 + dst) * S + k);
            if from < BASE || to < BASE {
                continue;
            }
            for plane in &planes {
                let n = plane.cell as usize;
                let (f, d) = (row_at(from, plane), row_at(to, plane));
                let src_row = content[f..f + n].to_vec();
                want[d..d + n].copy_from_slice(&src_row);
            }
        }
    }
    assert_eq!(after, want, "every copied row of all four planes, nothing else");
    let faults = words(&dev.read(&buffer, fault_at, 12));
    assert_eq!(
        faults,
        vec![0, 0, kbeam::PAGES_COPY_BOUNDS],
        "only the copy past the pool faults its lane"
    );
}

/// A fork's tail copy over two planes of a page, and a copy past the pool
/// that copies nothing and faults its lane.
fn copies_move_every_plane(dev: &Dev) {
    let (page, pages) = (16u32, 4u32);
    let planes = [
        PagePlane { offset: 0, cell: 64 },
        PagePlane { offset: 16 * 64, cell: 48 },
    ];
    let page_bytes: u64 = 16 * 64 + 16 * 48;
    let mut arena = Arena { at: 0 };
    let runs_at = arena.take(2 * std::mem::size_of::<PageCopyRun>() as u64);
    let planes_at = arena.take(std::mem::size_of_val(&planes) as u64);
    let copy_at = arena.take(6 * 4);
    let fault_at = arena.take(2 * 4);
    let bases_at = arena.take(u64::from(pages) * 8);
    let pages_at = arena.take(u64::from(pages) * page_bytes);
    let mut buffer = Buffer::zeroed(&dev.device, arena.at).expect("a buffer");
    let base = buffer.address_at(0).expect("an address");
    let address = move |at: u64| base + at;
    let bases: Vec<u64> = (0..u64::from(pages))
        .map(|p| address(pages_at + p * page_bytes))
        .collect();
    let content: Vec<u8> = (0..u64::from(pages) * page_bytes)
        .map(|i| (i * 7 + 3) as u8)
        .collect();
    buffer.write(pages_at, &content).expect("pages");
    buffer.write(bases_at, &bytes_of(&bases)).expect("bases");
    buffer.write(planes_at, &bytes_of(&planes)).expect("planes");
    // Beam 0: 5 slots of page 1 onto page 3. Beam 1: a copy onto page 9,
    // past the pool.
    buffer
        .write(copy_at, &bytes_of(&[1u32, 3, 5, 0, 9, 5]))
        .expect("copies");
    let run = |beam: u64| PageCopyRun {
        copy: address(copy_at + beam * 12),
        base: address(bases_at),
        fault: address(fault_at + beam * 4),
        reserve: pages,
        page,
        floor_rel: 0,
        plane_at: 0,
        planes: 2,
        pad: 0,
    };
    buffer.write(runs_at, &bytes_of(&[run(0), run(1)])).expect("runs");
    let runs = dev.handles.bind(&buffer, runs_at, buffer.bytes() - runs_at).expect("runs");
    let planes_h = dev
        .handles
        .bind(&buffer, planes_at, std::mem::size_of_val(&planes) as u64)
        .expect("planes");
    dev.run(&|sink| {
        kbeam::beam_copy_pages(
            sink,
            Tensor::new(runs, 2, 48, Dtype::U8),
            Tensor::new(planes_h, 2, 16, Dtype::U8),
            2,
            2,
        )
        .expect("beam_copy_pages encodes");
    });
    let after = dev.read(&buffer, pages_at, u64::from(pages) * page_bytes);
    let mut want = content.clone();
    for plane in &planes {
        let n = (5 * plane.cell) as usize;
        let from = (page_bytes + plane.offset) as usize;
        let to = (3 * page_bytes + plane.offset) as usize;
        let src = content[from..from + n].to_vec();
        want[to..to + n].copy_from_slice(&src);
    }
    assert_eq!(after, want, "the fork's 5 tail slots of both planes, nothing else");
    let faults = words(&dev.read(&buffer, fault_at, 8));
    assert_eq!(faults, vec![0, kbeam::PAGES_COPY_BOUNDS], "the copy past the pool faults its lane");
}

/// A group wider than the head step's threadgroup arrays hold
/// (`MAX_LANES`, refused at bind and before a fire on the host) faults every
/// one of its lanes by name on the device, never a step that returns and
/// leaves its lists unresolved (the L40S's audit of CUDA's `beam_step`).
fn a_too_wide_group_faults_every_lane(dev: &Dev) {
    let lanes = kbeam::MAX_LANES + 1;
    let mut arena = Arena { at: 0 };
    let run_at = arena.take(std::mem::size_of::<BeamStepRun>() as u64);
    let faults_at = arena.take(u64::from(lanes) * 4);
    let buffer = Buffer::zeroed(&dev.device, arena.at).expect("a buffer");
    let base = buffer.address_at(0).expect("an address");
    let run = BeamStepRun {
        faults: base + faults_at,
        lanes,
        cap: 1,
        reserve: 1,
        page: PAGE,
        list_count: lanes,
        frames: FRAMES,
        backed: 1,
        ..BeamStepRun::default()
    };
    let mut buffer = buffer;
    buffer.write(run_at, &bytes_of(&[run])).expect("the run");
    let runs = dev.handles.bind(&buffer, 0, buffer.bytes()).expect("the runs");
    dev.run(&|sink| {
        let width = std::mem::size_of::<BeamStepRun>() as u32;
        kbeam::beam_step(sink, Tensor::new(runs, 1, width, Dtype::U8), 1).expect("beam_step encodes");
    });
    let faults = words(&dev.read(&buffer, faults_at, u64::from(lanes) * 4));
    assert_eq!(
        faults,
        vec![kbeam::PAGES_TOO_WIDE; lanes as usize],
        "every lane of a {lanes}-beam group faults as too wide"
    );
}

#[test]
fn a_beam_step_follows_the_spec() {
    let Some(dev) = Dev::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    for lanes in [1, 2, 4, 8] {
        for folded in [37, 32] {
            for seed in 0..3 {
                search(&dev, lanes, folded, 3 * PAGE + 9, seed, 0);
            }
            // A hybrid's rs tables beside, at T = 1, 2 and 4 rows a page.
            for rs_t in [1, 2, 4] {
                search(&dev, lanes, folded, 3 * PAGE + 9, 7, rs_t);
            }
        }
    }
    copies_move_every_plane(&dev);
    rs_copies_move_every_plane(&dev);
    a_too_wide_group_faults_every_lane(&dev);
}
