//! A host simulation of parking and restoring slots (CONTRACT-BC 22): what
//! a slot held in the pool comes back word for word from the host chunks
//! and the slot file, into fresh slots, the source left as it was.

use engine::chunk::{SlotCopy, SlotRun};
use engine::transfer::MemoryDomain;

use super::super::pool::{Draft, Drafts, FIXED, HostPool, LIST, Layout, RS, Section};
use crate::error::Result;

const DEVICE: MemoryDomain = MemoryDomain::CerebrasDevice(0);
const HOST: MemoryDomain = MemoryDomain::HostPinned;
const DISK: MemoryDomain = MemoryDomain::LocalDisk;

fn draft(tokens: u32, widths: &[u32], slots: u64) -> Draft {
    Draft {
        tokens,
        sections: widths
            .iter()
            .enumerate()
            .map(|(at, &width)| Section {
                row: at as u32,
                plane: 0,
                width,
            })
            .collect(),
        rows: Vec::new(),
        slots,
    }
}

/// A kv list, a record cut into pieces (chunks of at most 64 KiB) and an
/// rs list.
fn pool() -> HostPool {
    let layout = Layout::plan(
        &Drafts {
            list: Some(draft(16, &[64, 32], 64)),
            window: None,
            fixed: Some(draft(1, &[20_000, 30_000], 4)),
            rs: Some(draft(16, &[8], 8)),
        },
        64 << 10,
    )
    .expect("the pool lays out");
    let mut pool = HostPool::new(layout, false);
    assert!(pool.shape(FIXED).unwrap().pieces() > 1, "a record is cut");
    pool.seat_host(32 * pool.layout().chunk_words * 4);
    pool
}

fn fill(pool: &mut HostPool, id: engine::chunk::ShapeId, slot: u32, seed: u32) -> Vec<u32> {
    let words = pool.slot_mut(id, slot).unwrap().unwrap();
    for (at, w) in words.iter_mut().enumerate() {
        *w = seed.wrapping_mul(2_654_435_761) ^ at as u32;
    }
    words.to_vec()
}

fn copy(
    pool: &mut HostPool,
    id: engine::chunk::ShapeId,
    (src, dst): (MemoryDomain, MemoryDomain),
    slots: Vec<(u32, u32)>,
) -> Result<()> {
    pool.copy_tiers(
        &SlotCopy {
            shape: id,
            src,
            dst,
            slots,
            runs: Vec::new(),
        },
        DEVICE,
    )
}

/// `n` refs of shape `id` from chunk `chunk` on.
fn refs(pool: &HostPool, id: engine::chunk::ShapeId, chunk: u32, n: u32) -> Vec<u32> {
    let per = pool.shape(id).unwrap().per_chunk;
    (0..n).map(|i| chunk * per + i).collect()
}

#[test]
fn every_shape_parks_on_host_chunks_and_comes_back() {
    let mut pool = pool();
    let facts = pool.facts();
    facts.validate().expect("the facts hang together");
    assert_eq!(facts.host_chunks, 32);
    for (id, chunk) in [(LIST, 1u32), (RS, 3), (FIXED, 5)] {
        let n = if id == FIXED {
            pool.shape(FIXED).unwrap().pieces()
        } else {
            2
        };
        let device = refs(&pool, id, chunk, n);
        let held: Vec<Vec<u32>> = device
            .iter()
            .enumerate()
            .map(|(at, &slot)| fill(&mut pool, id, slot, 10 * chunk + at as u32))
            .collect();
        // Host refs of the shape, in a host chunk of their own; a record
        // moves as one copy, pieces paired in order.
        let host = refs(&pool, id, chunk, n);
        copy(
            &mut pool,
            id,
            (DEVICE, HOST),
            device.iter().copied().zip(host.iter().copied()).collect(),
        )
        .expect("the slots park");
        let fresh = refs(&pool, id, chunk + 10, n);
        copy(
            &mut pool,
            id,
            (HOST, DEVICE),
            host.iter().copied().zip(fresh.iter().copied()).collect(),
        )
        .expect("the slots come back");
        for ((&slot, &old), want) in fresh.iter().zip(&device).zip(&held) {
            assert_eq!(pool.slot(id, slot).unwrap().unwrap(), want.as_slice());
            assert_eq!(pool.slot(id, old).unwrap().unwrap(), want.as_slice());
        }
    }
    // A host ref past the seated chunks is refused.
    let (past, held) = (refs(&pool, LIST, 32, 1)[0], refs(&pool, LIST, 1, 1)[0]);
    assert!(copy(&mut pool, LIST, (DEVICE, HOST), vec![(held, past)]).is_err());
}

#[test]
fn kv_list_slots_go_to_disk_and_back_and_a_kept_file_keeps_them() {
    let dir = tempfile::tempdir().expect("a scratch dir");
    let mut pool = pool();
    let stride = (pool.shape(LIST).unwrap().slot_words * 4).next_multiple_of(4096);
    pool.seat_disk(dir.path(), 64 * stride, Some("test"))
        .expect("the slot file opens");
    let per = pool.shape(LIST).unwrap().per_chunk;
    assert_eq!(pool.disk_chunks(), 64 / per);
    assert!(pool.disk_persistent() && !pool.disk_kept());
    let device = refs(&pool, LIST, 1, 3);
    let held: Vec<Vec<u32>> = device
        .iter()
        .enumerate()
        .map(|(at, &slot)| fill(&mut pool, LIST, slot, 40 + at as u32))
        .collect();
    // Pool -> disk, disk -> host, host -> disk, disk -> pool.
    copy(
        &mut pool,
        LIST,
        (DEVICE, DISK),
        vec![(device[0], 5), (device[1], 6)],
    )
    .unwrap();
    copy(&mut pool, LIST, (DISK, HOST), vec![(5, 9)]).unwrap();
    copy(&mut pool, LIST, (HOST, DISK), vec![(9, 7)]).unwrap();
    let fresh = refs(&pool, LIST, 2, 2);
    copy(
        &mut pool,
        LIST,
        (DISK, DEVICE),
        vec![(7, fresh[0]), (6, fresh[1])],
    )
    .unwrap();
    assert_eq!(
        pool.slot(LIST, fresh[0]).unwrap().unwrap(),
        held[0].as_slice()
    );
    assert_eq!(
        pool.slot(LIST, fresh[1]).unwrap().unwrap(),
        held[1].as_slice()
    );
    // Only the kv list goes to disk; token runs move only within the pool.
    let rs = refs(&pool, RS, 3, 1);
    assert!(copy(&mut pool, RS, (DEVICE, DISK), vec![(rs[0], 0)]).is_err());
    let runs = SlotCopy {
        shape: LIST,
        src: DEVICE,
        dst: HOST,
        slots: Vec::new(),
        runs: vec![SlotRun {
            src: device[2],
            src_token: 0,
            dst: 0,
            dst_token: 0,
            tokens: 1,
        }],
    };
    assert!(pool.copy_tiers(&runs, DEVICE).is_err());
    // The next run of the same load finds the file kept, slot 5 as left.
    drop(pool);
    let mut again = self::pool();
    again
        .seat_disk(dir.path(), 64 * stride, Some("test"))
        .expect("the kept file opens");
    assert!(again.disk_kept());
    let slot = refs(&again, LIST, 1, 1)[0];
    copy(&mut again, LIST, (DISK, DEVICE), vec![(5, slot)]).unwrap();
    assert_eq!(again.slot(LIST, slot).unwrap().unwrap(), held[0].as_slice());
    // Another load's layout takes the file afresh.
    drop(again);
    let mut other = self::pool();
    other
        .seat_disk(dir.path(), 64 * stride, Some("another"))
        .unwrap();
    assert!(!other.disk_kept());
}

#[test]
fn copies_within_the_pool_keep_their_runs() {
    let mut pool = pool();
    let slots = refs(&pool, LIST, 1, 2);
    let src = fill(&mut pool, LIST, slots[0], 1);
    fill(&mut pool, LIST, slots[1], 2);
    let copy = SlotCopy {
        shape: LIST,
        src: DEVICE,
        dst: DEVICE,
        slots: Vec::new(),
        runs: vec![SlotRun {
            src: slots[0],
            src_token: 0,
            dst: slots[1],
            dst_token: 0,
            tokens: 16,
        }],
    };
    pool.copy_tiers(&copy, DEVICE).unwrap();
    assert_eq!(pool.slot(LIST, slots[1]).unwrap().unwrap(), src.as_slice());
}
