//! A slot copy answers when it has landed (CONTRACT-BC 23): the completion
//! `copy_slots` returns resolves only once the device has run the copy's
//! programs (the pool's newest buffer ready), and the destination then
//! holds the source's bytes. Slots move host -> device, device -> device and
//! device -> host, each completion waited on before the next copy reads
//! what it wrote, and the bytes come back whole; a kv token-run copy (the
//! `copy_kv` path) resolves too. None resolves failed. Skipped without a
//! PJRT plugin.

use std::time::{Duration, Instant};

use engine::Engine;
use engine::chunk::{SlotCopy, SlotKind, SlotRole, SlotRun};
use engine::transfer::MemoryDomain;

use crate::common_llm::{Model, Refs};

/// Waits for `completion` (it resolves on a plugin thread), at most a
/// minute: its outcome.
fn landed(completion: &engine::CopyCompletion, what: &str) {
    let start = Instant::now();
    loop {
        if let Some(outcome) = completion.check() {
            if let Err(why) = outcome {
                panic!("{what}: the copy resolved failed: {why}");
            }
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "{what}: the copy never resolved"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_copy_answers_when_its_slots_hold_the_bytes() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let device = model.loaded.caps.device.domain;
    let facts = model.loaded.caps.chunks.clone();
    let mut refs = Refs::of_facts(facts);
    let kv = refs.shapes(|s| {
        matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State
    })[0]
        .clone();
    let len = kv.slot_bytes as usize;
    let n = 4u32;
    let a: Vec<u32> = (0..n).map(|_| refs.take(kv.id)).collect();
    let b: Vec<u32> = (0..n).map(|_| refs.take(kv.id)).collect();
    let bytes: Vec<u8> = (0..n as usize * len)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
        .collect();
    let there: Vec<u32> = (0..n).collect();
    let back: Vec<u32> = (n..2 * n).collect();
    model
        .engine
        .write_tier_slots(&[(kv.id, there.clone())], &bytes)
        .expect("the host slots write");
    let copy = |model: &mut Model, src, dst, slots: Vec<(u32, u32)>, runs: Vec<SlotRun>| {
        model
            .engine
            .copy_slots(&SlotCopy {
                shape: kv.id,
                src,
                dst,
                slots,
                runs,
            })
            .expect("the slots move")
    };
    let zip = |x: &[u32], y: &[u32]| x.iter().copied().zip(y.iter().copied()).collect();
    let up = copy(
        &mut model,
        MemoryDomain::HostPinned,
        device,
        zip(&there, &a),
        Vec::new(),
    );
    landed(&up, "host -> device");
    let across = copy(&mut model, device, device, zip(&a, &b), Vec::new());
    landed(&across, "device -> device");
    let down = copy(
        &mut model,
        device,
        MemoryDomain::HostPinned,
        zip(&b, &back),
        Vec::new(),
    );
    landed(&down, "device -> host");
    let got = model
        .engine
        .read_tier_slots(&[(kv.id, back)])
        .expect("the host slots read");
    assert!(
        got == bytes,
        "the slots came back whole through three landed copies"
    );

    // A token run within the kv list: the `copy_kv` path answers so too.
    let run = copy(
        &mut model,
        device,
        device,
        Vec::new(),
        vec![SlotRun {
            src: a[0],
            src_token: 1,
            dst: b[1],
            dst_token: 3,
            tokens: 5,
        }],
    );
    landed(&run, "a kv token run");
}
