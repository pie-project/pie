//! Slots read off the device to the host tier come back whole and in order
//! at every count: a tier read gathers a batch of slots on the device and
//! copies it out at once, and reads a short tail a slot at a time, so the
//! counts here cross a batch both ways (one slot, a short tail, a long
//! tail, whole batches and a remainder). Each slot's bytes land on the
//! host, move to a device slot, and back to other host slots. Skipped
//! without a PJRT plugin.

use engine::chunk::{SlotCopy, SlotKind, SlotRole};
use engine::transfer::MemoryDomain;

use crate::common_llm::{Model, Refs};

#[test]
fn a_tier_read_lands_every_slot_whole() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let device = model.loaded.caps.device.domain;
    let facts = model.loaded.caps.chunks.clone();
    let mut refs = Refs::of_facts(facts.clone());
    let kv = refs.shapes(|s| {
        matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State
    })[0]
        .clone();
    let len = kv.slot_bytes as usize;
    // The slots a batch holds (64 MiB of them, at most 256), as the engine
    // sizes it.
    let batch = ((64u64 << 20) / kv.slot_bytes).clamp(1, 256) as u32;
    let host_slots = (u64::from(facts.host_chunks) * facts.chunk_bytes / kv.slot_bytes) as u32;
    let mut counts = vec![1, 3, batch / 4 + 1, batch, batch + 2];
    counts.retain(|&n| 2 * n <= host_slots);
    assert!(
        counts.len() >= 3,
        "the host tier holds {host_slots} kv slots"
    );
    let mut seed = 0usize;
    for n in counts {
        let on_device: Vec<u32> = (0..n).map(|_| refs.take(kv.id)).collect();
        let bytes: Vec<u8> = (0..n as usize * len)
            .map(|i| ((i + seed).wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect();
        seed += 7919;
        let there: Vec<u32> = (0..n).collect();
        let back: Vec<u32> = (n..2 * n).collect();
        engine::Tiers::write(&mut model.engine, &[(kv.id, there.clone())], &bytes)
            .expect("the host slots write");
        let copy = |model: &mut Model, src, dst, pairs: Vec<(u32, u32)>| {
            engine::Tiers::copy(
                &mut model.engine,
                &SlotCopy {
                    shape: kv.id,
                    src,
                    dst,
                    slots: pairs,
                    runs: Vec::new(),
                },
            )
            .expect("the slots move");
        };
        copy(
            &mut model,
            MemoryDomain::HostPinned,
            device,
            there
                .iter()
                .copied()
                .zip(on_device.iter().copied())
                .collect(),
        );
        // Read back in another order than written: the read keeps the
        // order it is asked in.
        let pairs: Vec<(u32, u32)> = (on_device.iter().rev().copied())
            .zip(back.iter().copied())
            .collect();
        copy(&mut model, device, MemoryDomain::HostPinned, pairs);
        let got =
            engine::Tiers::read(&mut model.engine, &[(kv.id, back)]).expect("the host slots read");
        let want: Vec<u8> = bytes.chunks(len).rev().flatten().copied().collect();
        assert!(
            got == want,
            "{n} slots of {len} bytes came back whole and in order"
        );
    }
}
