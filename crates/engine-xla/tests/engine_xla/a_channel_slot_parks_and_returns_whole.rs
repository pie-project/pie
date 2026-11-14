//! A guest's `Channel` slots move as the runtime parks them: a slot's bytes
//! landed on the host tier come to a device slot, move to another device
//! slot, and back to another host slot whole (CONTRACT-BC 22). Skipped
//! without a PJRT plugin.

use engine::Engine;
use engine::chunk::SlotCopy;
use engine::transfer::MemoryDomain;
use engine_xla::store::layout::CHANNEL;

use crate::common_llm::{Model, Refs};

#[test]
fn a_channel_slot_parks_and_returns_whole() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let device = model.loaded.caps.device.domain;
    let facts = model.loaded.caps.chunks.clone();
    let shape = facts.shape(CHANNEL).expect("the pool states Channel slots").clone();
    assert_eq!(shape.slot_bytes, 1024, "a Channel slot is 1 KiB");
    assert!(facts.host_chunks > 0, "a host tier");
    let mut refs = Refs::of(&model);
    let (d1, d2) = (refs.take(CHANNEL), refs.take(CHANNEL));
    let (h1, h2) = (0u32, 1u32);
    let bytes: Vec<u8> = (0..1024u32).map(|i| (i * 7 + 3) as u8).collect();
    model
        .engine
        .write_tier_slots(&[(CHANNEL, vec![h1])], &bytes)
        .expect("the host slot writes");
    let copy = |model: &mut Model, src, dst, slots: Vec<(u32, u32)>| {
        model
            .engine
            .copy_slots(&SlotCopy {
                shape: CHANNEL,
                src,
                dst,
                slots,
                runs: Vec::new(),
            })
            .expect("the Channel slots move");
    };
    copy(&mut model, MemoryDomain::HostPinned, device, vec![(h1, d1)]);
    copy(&mut model, device, device, vec![(d1, d2)]);
    copy(&mut model, device, MemoryDomain::HostPinned, vec![(d2, h2)]);
    let back = model
        .engine
        .read_tier_slots(&[(CHANNEL, vec![h2])])
        .expect("the host slot reads");
    assert_eq!(back, bytes, "the cell came back whole");
}
