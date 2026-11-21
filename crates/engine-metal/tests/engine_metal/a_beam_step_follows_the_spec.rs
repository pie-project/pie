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
    self as kbeam, BeamListRun, BeamStepRun, PageCopyRun, PagePlane, Record,
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

fn search(dev: &Dev, lanes: u32, folded: u32, steps: u32, seed: u64) {
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
    let mut replay = State::new(lanes, cap, FRAMES);
    let mut rng = Lcg(seed);
    let runs = dev.handles.bind(&buffer, 0, buffer.bytes()).expect("the runs");
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
            pad: 0,
        };
        buffer.write(run_at, &bytes_of(&[run])).expect("the run");
        dev.run(&|sink| {
            kbeam::beam_step(sink, Tensor::new(runs, 1, 80, Dtype::U8), 1).expect("beam_step encodes");
        });
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
    }
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

#[test]
fn a_beam_step_follows_the_spec() {
    let Some(dev) = Dev::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    for lanes in [2, 4, 8] {
        for folded in [37, 32] {
            for seed in 0..3 {
                search(&dev, lanes, folded, 3 * PAGE + 9, seed);
            }
        }
    }
    copies_move_every_plane(&dev);
}
