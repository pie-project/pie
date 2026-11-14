#![cfg(target_vendor = "apple")]

//! An adapter an instance lands lies in `Adapter` slots, one plane a bank,
//! read through each bank's address table in the tables run. An adapter of
//! zero planes serves the unadapted logits bit for bit, other planes move
//! them, and the same planes landed as another item serve the same logits;
//! a second holder of one item joins its landing (one id, nothing written
//! again); after its slots park to other refs and it is rebound there, its
//! lanes read the same logits; a forgotten id reads the zero plane; and its
//! content key is the digest CUDA states for the same sink and seeds.

use engine::chunk::SlotCopy;
use engine::program::InstanceId;
use engine::transfer::MemoryDomain;
use engine_metal::serve::Seated;
use engine_metal::store::layout::ADAPTER;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};

const CONTEXT: u32 = 256;
const PROMPT: u32 = 24;

/// Bf16 values of a small spread, a seed apart: an adapter's plane.
fn plane(bytes: u64, seed: u64) -> Vec<u8> {
    (0..bytes / 2)
        .flat_map(|at| {
            let x = ((at.wrapping_mul(2_654_435_761).wrapping_add(seed * 97)) % 2001) as f32;
            let value = (x - 1000.0) / 20_000.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// Every bank's plane of one adapter, in bank-name order.
fn planes(shell: &Shell, seed: u64) -> Vec<(String, Vec<u8>)> {
    let mut planes: Vec<(String, Vec<u8>)> = shell
        .banks()
        .into_iter()
        .map(|(name, _, slot)| (name.to_string(), plane(slot, seed + name.len() as u64)))
        .collect();
    planes.sort_by(|a, b| a.0.cmp(&b.0));
    planes
}

/// The prompt's logits on seat `slot`, read through adapter id `adapter`
/// (none: the unadapted arm).
fn logits(shell: &mut Shell, fixture: &Fixture, slot: u32, adapter: Option<u32>) -> Vec<f32> {
    let tokens: Vec<u32> = (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect();
    let mut lane_word = word(&fixture.trace, PROMPT);
    if adapter.is_some() {
        lane_word = shell
            .adapted_word(lane_word)
            .expect("the model has an adapted arm");
    }
    let seated = Seated {
        held: Some(0),
        adapter,
        ..Seated::of(Lane {
            slot,
            word: lane_word,
            tokens: &tokens,
        })
    };
    let mut rows = shell.fire_seated(&[seated]).expect("the fire lands");
    rows.remove(0)
}

#[test]
fn the_adapter_key_is_cudas_digest_of_the_sink() {
    // A fixed sink (site word 5) and seeds: the vector CUDA pins the same.
    let a: Vec<u8> = (0u8..16).collect();
    let b: Vec<u8> = (100u8..124).collect();
    let key =
        engine_metal::api::adapter_digest(5, [("lora_a", &a[..]), ("lora_b", &b[..])].into_iter());
    eprintln!("adapter key {key:#034x}");
    assert_eq!(key, PINNED, "the digest CUDA's adapter_key_of states");
    assert_ne!(
        engine_metal::api::adapter_digest(5, [("lora_a", &a[..]), ("lora_b", &a[..])].into_iter()),
        key,
        "other planes key otherwise"
    );
}

/// The key `the_adapter_key_is_cudas_digest_of_the_sink` pins.
const PINNED: u128 = 0x48d3_ea2e_650c_f1e9_8eff_b3e0_9bc8_eb41;

#[test]
fn an_adapter_in_slots_is_one_landing_through_the_table() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    let layout = shell.pool_layout();
    let adapter = layout
        .adapter
        .expect("a model with banks states `Adapter` slots");
    let facts = shell.pool_chunk_facts();
    let segment = facts.segment_chunks;
    let banks = shell.banks().len() as u32;
    // The item's slots, a bank's plane each, in a chunk past the seats; the
    // refs it parks to, a chunk further.
    let first = layout.free_chunk().div_ceil(segment) * segment;
    shell
        .commit_segments(first.div_ceil(segment) + 1)
        .expect("the segments commit");
    let refs: Vec<u32> = (0..banks)
        .map(|at| first * adapter.per_chunk + at)
        .collect();
    let moved: Vec<u32> = (0..banks)
        .map(|at| (first + 1) * adapter.per_chunk + at)
        .collect();

    let elsewhere: Vec<u32> = (0..banks)
        .map(|at| (first + 2) * adapter.per_chunk + at)
        .collect();
    shell
        .commit_segments((first + 2).div_ceil(segment) + 1)
        .expect("the segments commit");
    let plain = logits(&mut shell, &fixture, 0, None);

    // Zero planes correct nothing; an adapter's planes move the logits.
    let zero_planes: Vec<(String, Vec<u8>)> = planes(&shell, 0)
        .into_iter()
        .map(|(bank, bytes)| (bank, vec![0u8; bytes.len()]))
        .collect();
    let zero = shell
        .land_adapter(InstanceId(9), &elsewhere, zero_planes)
        .expect("the zero adapter lands");
    assert_eq!(
        logits(&mut shell, &fixture, 0, Some(zero)),
        plain,
        "an adapter of zero planes corrects nothing"
    );
    shell.detach_adapter(InstanceId(9));

    let seed = 7;
    let id = shell
        .land_adapter(InstanceId(1), &refs, planes(&shell, seed))
        .expect("the adapter lands in its slots");
    let landed = logits(&mut shell, &fixture, 1, Some(id));
    assert_ne!(landed, plain, "an adapter's planes move its lanes' logits");

    // The same planes as another item (other slots): the same logits.
    let twin = shell
        .land_adapter(InstanceId(5), &elsewhere, planes(&shell, seed))
        .expect("a second item lands");
    assert_ne!(twin, id, "another item is another landing");
    assert_eq!(
        logits(&mut shell, &fixture, 2, Some(twin)),
        landed,
        "the same planes read through another item's table entries"
    );
    shell.detach_adapter(InstanceId(5));
    assert_eq!(
        logits(&mut shell, &fixture, 3, Some(twin)),
        plain,
        "a forgotten id reads the zero plane"
    );

    // A second holder of the same item joins: one landing, one id.
    let joined = shell
        .land_adapter(InstanceId(2), &refs, planes(&shell, seed + 1))
        .expect("the second holder joins");
    assert_eq!(joined, id, "one item is one landing");
    assert_eq!(
        logits(&mut shell, &fixture, 2, Some(joined)),
        landed,
        "joining wrote nothing: the first landing's planes stand"
    );

    // The item parks to other refs and every holder is rebound there.
    shell
        .copy_slots(&SlotCopy {
            shape: ADAPTER,
            src: MemoryDomain::MetalShared,
            dst: MemoryDomain::MetalShared,
            slots: refs.iter().copied().zip(moved.iter().copied()).collect(),
            runs: Vec::new(),
        })
        .expect("the item's slots move");
    shell
        .copy_slots(&SlotCopy {
            shape: ADAPTER,
            src: MemoryDomain::MetalShared,
            dst: MemoryDomain::MetalShared,
            slots: refs.iter().map(|&r| (moved[0] + banks, r)).collect(),
            runs: Vec::new(),
        })
        .expect("the old slots are overwritten");
    for holder in [InstanceId(1), InstanceId(2)] {
        shell
            .rebind_adapter(holder, &moved)
            .expect("the holder is rebound");
    }
    assert_eq!(
        logits(&mut shell, &fixture, 3, Some(id)),
        landed,
        "after the move its lanes read the planes where they lie now"
    );

    // Closing every holder forgets the landing: its refs land anew.
    shell.detach_adapter(InstanceId(1));
    shell.detach_adapter(InstanceId(2));
    let again = shell
        .land_adapter(InstanceId(3), &refs, planes(&shell, seed))
        .expect("a fresh landing");
    assert_eq!(
        logits(&mut shell, &fixture, 0, Some(again)),
        landed,
        "a fresh landing writes its planes"
    );
    eprintln!("an adapter in slots is one landing through the table (id {id})");
}
