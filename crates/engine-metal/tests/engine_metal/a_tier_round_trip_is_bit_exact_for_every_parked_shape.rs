#![cfg(target_vendor = "apple")]

//! CONTRACT-BC 22 on Metal: every state shape a load parks comes back bit
//! for bit through the disk tier, device -> disk -> device and host -> disk
//! -> host, the disk budget divided into disk chunks of a pool chunk each
//! (on the slot file's direct I/O boundary), any shape's slots in any
//! chunk, laid out as on the device (engine-gpu's
//! a_tier_round_trip_is_bit_exact_for_every_parked_shape). Over the kv list
//! (a dense model), the window lists (gemma 4) and the hybrid's records and
//! rs-list rows (Qwen3.5), and any guest-state shape a load states:
//! - words no bf16 keeps (every one's low half set, NaN payloads, denormals);
//! - three slots parked in one copy, to disk refs out of order, two of them
//!   adjacent in one disk chunk (sharing a 4 KiB block where a slot is
//!   smaller than one), restored into other slots after their own were
//!   overwritten;
//! - a slot written to disk alone after its neighbour, the neighbour read
//!   back intact (a slot off the boundary keeps the bytes beside it);
//! - a negative control: another disk ref restores other bytes.

use engine::chunk::{SlotCopy, SlotShape};
use engine::transfer::MemoryDomain;
use engine_metal::Shell;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with};

const CONTEXT: u32 = 128;
const BUDGET: u64 = 64 << 20;

/// A slot's words, every one's low half set: no bf16 (nor a narrowing of
/// f32 to one) keeps them. Some are NaNs with payloads, some denormals.
fn pattern(seed: u32, bytes: u64) -> Vec<u8> {
    let words = bytes / 4;
    let mut out: Vec<u8> = (0..words)
        .flat_map(|at| {
            let x = (at as u32 ^ seed.wrapping_mul(0x9e37_79b9)).wrapping_mul(0x85eb_ca6b);
            let word = match at % 7 {
                0 => 0x7fc0_0000 | (x & 0x003f_ffff) | 1,
                1 => x & 0x007f_ffff | 1,
                _ => x ^ (x >> 15) | 0x0001_0001,
            };
            word.to_le_bytes()
        })
        .collect();
    out.resize(bytes as usize, (seed as u8) | 1);
    out
}

fn check(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: byte {at} of {} differs ({:#04x} for {:#04x}); {} bytes differ",
            want.len(),
            got[at],
            want[at],
            got.iter().zip(want).filter(|(a, b)| a != b).count()
        );
    }
}

fn copy(shape: &SlotShape, src: MemoryDomain, dst: MemoryDomain, slots: Vec<(u32, u32)>) -> SlotCopy {
    SlotCopy {
        shape: shape.id,
        src,
        dst,
        slots,
        runs: Vec::new(),
    }
}

/// One shape's round trips, on device chunks from `base` (the layout's
/// first free chunk: below it lie the engine's tables and the seats; the
/// test fires nothing, so every chunk past it is its own).
fn round_trip(shell: &mut Shell, shape: &SlotShape, model: &str, host_tier: bool, base: u32) {
    let (device, host, disk) = (
        MemoryDomain::MetalShared,
        MemoryDomain::HostPinned,
        MemoryDomain::LocalDisk,
    );
    let per = shape.slots_per_chunk;
    let bytes = shape.slot_bytes;
    let at = |chunk: u32, index: u32| (base + chunk) * per + index.min(per - 1);
    let name = format!("{model}: shape {} ({bytes} B a slot, {per} a chunk)", shape.id);
    // Three device slots: two adjacent in chunk 1, one at the end of chunk 2.
    let parked = [at(1, 0), at(1, 1.min(per - 1)), at(2, per - 1)];
    let held: Vec<Vec<u8>> = (0..3).map(|k| pattern(100 + 7 * shape.id.0 + k, bytes)).collect();
    // Disk refs: chunk 1's slots 1 and 0 (adjacent, out of order), and the
    // last slot of disk chunk 3.
    let tier = |chunk: u32, index: u32| chunk * per + index.min(per - 1);
    let refs = [tier(1, 1.min(per - 1)), tier(1, 0), tier(3, per - 1)];
    let distinct = per > 1;
    for (k, &slot) in parked.iter().enumerate() {
        if !distinct && k == 1 {
            continue;
        }
        shell
            .write_slot_bytes(shape.id, slot, &held[k])
            .unwrap_or_else(|why| panic!("{name}: slot {slot} takes its bytes: {why}"));
    }
    let pairs: Vec<(u32, u32)> = parked
        .iter()
        .zip(&refs)
        .enumerate()
        .filter(|(k, _)| distinct || *k != 1)
        .map(|(_, (&d, &r))| (d, r))
        .collect();
    shell
        .copy_slots(&copy(shape, device, disk, pairs))
        .unwrap_or_else(|why| panic!("{name}: the slots park to disk: {why}"));
    // Their slots go to another owner, which writes them.
    for &slot in &parked {
        shell
            .write_slot_bytes(shape.id, slot, &vec![0xa5; bytes as usize])
            .expect("the slot is written over");
    }
    // Restored into other slots, in another order.
    let into = [at(4, 0), at(5, per - 1), at(4, 1.min(per - 1))];
    for (k, (&r, &slot)) in refs.iter().zip(&into).enumerate() {
        if !distinct && k == 1 {
            continue;
        }
        shell
            .copy_slots(&copy(shape, disk, device, vec![(r, slot)]))
            .unwrap_or_else(|why| panic!("{name}: disk ref {r} restores: {why}"));
        let got = shell.read_slot_bytes(shape.id, slot).expect("the slot reads");
        check(&format!("{name}: slot {k} device -> disk -> device"), &got, &held[k]);
    }
    if distinct {
        // The neighbour of disk ref `refs[1]` (slot 0 of disk chunk 1) is
        // written alone; slot 0 reads back as it was.
        let fresh = pattern(900 + shape.id.0, bytes);
        shell
            .write_slot_bytes(shape.id, at(6, 0), &fresh)
            .expect("the slot takes its bytes");
        let neighbour = tier(1, 2.min(per - 1));
        let neighbour = if neighbour == refs[0] || neighbour == refs[1] {
            tier(2, 0)
        } else {
            neighbour
        };
        shell
            .copy_slots(&copy(shape, device, disk, vec![(at(6, 0), neighbour)]))
            .expect("the neighbour parks alone");
        shell
            .copy_slots(&copy(shape, disk, device, vec![(refs[1], at(6, 1.min(per - 1)))]))
            .expect("slot 0 of disk chunk 1 restores again");
        let got = shell
            .read_slot_bytes(shape.id, at(6, 1.min(per - 1)))
            .expect("the slot reads");
        check(&format!("{name}: a slot beside a later write"), &got, &held[1]);
        // The control: another disk ref restores other bytes.
        let got = shell.read_slot_bytes(shape.id, into[0]).expect("the slot reads");
        assert_ne!(got, held[1], "{name}: disk refs 0 and 1 restored the same bytes");
    }
    if host_tier {
        // The host tier's way down and back: host -> disk -> host.
        let host_in = vec![tier(1, 0), tier(2, per - 1)];
        let seeded: Vec<u8> = (0..2).flat_map(|k| pattern(500 + 3 * shape.id.0 + k, bytes)).collect();
        shell
            .write_tier_slots(&[(shape.id, host_in.clone())], &seeded)
            .expect("the host slots take the pattern");
        let down = vec![tier(5, per - 1), tier(4, 0)];
        shell
            .copy_slots(&copy(
                shape,
                host,
                disk,
                host_in.iter().copied().zip(down.iter().copied()).collect(),
            ))
            .unwrap_or_else(|why| panic!("{name}: the host slots go down to disk: {why}"));
        shell
            .write_tier_slots(&[(shape.id, host_in.clone())], &vec![0x3c; seeded.len()])
            .expect("the host slots are written over");
        let back = vec![tier(6, 0), tier(6, per - 1)];
        shell
            .copy_slots(&copy(
                shape,
                disk,
                host,
                down.iter().copied().zip(back.iter().copied()).collect(),
            ))
            .unwrap_or_else(|why| panic!("{name}: the disk slots come up to the host: {why}"));
        let got = shell.read_tier_slots(&[(shape.id, back.clone())]).expect("the host slots read");
        if per > 1 || back[0] != back[1] {
            check(&format!("{name}: host -> disk -> host"), &got, &seeded);
        }
    }
    eprintln!(
        "{name}: round trips bit for bit{}",
        if bytes % 4096 == 0 { "" } else { " (off the 4 KiB boundary)" }
    );
}

fn every_shape(model: &str) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of(model, None);
    let mut shell = load_with(&fixture, CONTEXT);
    shell.seat_tier(64 << 20);
    let dir = std::env::temp_dir().join(format!(
        "pie-metal-tier-exact-{model}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("the slot file's dir");
    let chunks = shell
        .seat_disk_kv(Some(BUDGET), Some(&dir), 8, CONTEXT)
        .expect("the slot file opens");
    // The budget divides into disk chunks of a pool chunk each, on the
    // direct I/O boundary, and the facts state them.
    let facts = shell.pool_chunk_facts();
    let chunk_bytes = shell.disk_chunk_bytes();
    assert_eq!(chunk_bytes, facts.chunk_bytes.next_multiple_of(4096));
    assert_eq!(u64::from(chunks), BUDGET / chunk_bytes, "{model}: disk chunks");
    assert_eq!(facts.disk_chunks, chunks, "{model}: the facts state the disk chunks");
    assert!(chunks >= 8, "{model}: the slot file holds {chunks} chunks");
    let host_tier = facts.host_chunks >= 8;
    let base = shell.pool_layout().free_chunk();
    // Only the segments the test's chunks lie in (`base + 7` at the most):
    // the whole pool would starve the tests beside it.
    let segment = facts.segment_chunks.max(1);
    shell
        .commit_segments((base + 8).div_ceil(segment))
        .expect("the segments commit");
    let shapes: Vec<SlotShape> = facts
        .shapes
        .iter()
        .filter(|shape| shape.pinned.is_none() && shape.slots_per_chunk > 0 && shape.slot_bytes > 0)
        .cloned()
        .collect();
    assert!(!shapes.is_empty(), "{model}: a shape shares the pool");
    for shape in &shapes {
        round_trip(&mut shell, shape, model, host_tier, base);
    }
    shell.settle_copies().expect("every copy settles");
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tier_round_trip_is_bit_exact_for_every_parked_shape() {
    every_shape("qwen3-micro-text");
    every_shape("qwen35-tiny");
    every_shape("gemma4-e4b-mini-l1");
}
