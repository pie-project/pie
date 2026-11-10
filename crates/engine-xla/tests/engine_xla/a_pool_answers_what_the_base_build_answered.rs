//! Catalog models with random weights fired through the engine: every
//! readout of a fixed scenario, bit for bit. With `PIE_XLA_GOLDEN_DUMP=<dir>`
//! the readouts are written there (one file a model); with
//! `PIE_XLA_GOLDEN=<dir>` they are compared against what a build wrote
//! there. Skipped without a PJRT plugin.

use crate::common_llm::{Model, scenario};

fn file(dir: &str, name: &str) -> std::path::PathBuf {
    std::path::Path::new(dir).join(format!("{name}.bits"))
}

fn encode(readouts: &[Vec<u32>]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in readouts {
        out.extend_from_slice(&(r.len() as u32).to_le_bytes());
        for v in r {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn run(name: &str, layers: Option<u32>, context: u32, long: Option<u32>) {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load(name, layers, context, 7);
    let readouts = scenario(&mut model, long);
    assert!(
        readouts.iter().all(|r| !r.is_empty()),
        "{name}: every fire reads out"
    );
    let bytes = encode(&readouts);
    if let Ok(dir) = std::env::var("PIE_XLA_GOLDEN_DUMP") {
        std::fs::create_dir_all(&dir).expect("the dump directory");
        std::fs::write(file(&dir, name), &bytes).expect("the dump writes");
    }
    if let Ok(dir) = std::env::var("PIE_XLA_GOLDEN") {
        let want = std::fs::read(file(&dir, name)).expect("the golden file reads");
        assert!(
            want == bytes,
            "{name}: the readouts differ from the golden build's ({} vs {} bytes)",
            bytes.len(),
            want.len()
        );
        eprintln!("{name}: {} readouts bit-identical", readouts.len());
    }
}

#[test]
fn a_dense_model_answers_bit_for_bit() {
    run("qwen3-micro-text", None, 256, None);
}

#[test]
fn a_gated_delta_hybrid_answers_bit_for_bit() {
    run("qwen35-tiny", None, 256, None);
}

#[test]
fn a_routed_model_answers_bit_for_bit() {
    run("qwen3-30b-a3b-mini", None, 256, None);
}

#[test]
fn a_windowed_model_answers_bit_for_bit_past_its_window() {
    run("gemma4-e4b-mini-l1", None, 1024, Some(700));
}

/// Prompts a sequence, decodes two steps, then decodes on: with `park`, its
/// kv list slots and its record go to the host tier (the kv slots on to the
/// slot file and back, where the load seats one) between the steps and
/// return to fresh device slots. Every readout.
fn parked(name: &str, park: bool, disk: Option<&std::path::Path>) -> Vec<Vec<u32>> {
    use crate::common_llm::{Refs, Seq, bits, tokens};
    use engine::Engine;
    use engine::chunk::SlotCopy;
    use engine::transfer::MemoryDomain;
    let mut model = Model::load_with(name, None, 256, 7, disk);
    let device = model.loaded.caps.device.domain;
    let mut refs = Refs::of(&model);
    let vocab = model.vocab;
    let mut out = Vec::new();
    let mut a = Seq::new(0);
    let lane = a.lane(&model, &mut refs, tokens(1, 37, vocab));
    out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
    a.held = 37;
    for step in 0..2u64 {
        let lane = a.lane(&model, &mut refs, tokens(10 + step, 1, vocab));
        out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
        a.held += 1;
    }
    if park {
        let facts = model.loaded.caps.chunks.clone().expect("chunk facts");
        assert!(facts.host_chunks > 0, "{name}: a host tier");
        let kv = facts.shapes[0].id;
        let copy = |model: &mut Model, shape, src, dst, slots: Vec<(u32, u32)>| {
            model
                .engine
                .copy_slots(&SlotCopy {
                    shape,
                    src,
                    dst,
                    slots,
                    runs: Vec::new(),
                })
                .expect("the slots move");
        };
        // Out: kv slots to host refs 0.., each record shape's pieces in
        // host chunks after them.
        let host: Vec<(u32, u32)> =
            a.kv.iter()
                .enumerate()
                .map(|(i, &r)| (r, i as u32))
                .collect();
        copy(&mut model, kv, device, MemoryDomain::HostPinned, host);
        let on_disk = facts.disk_chunks > 0;
        if on_disk {
            let to: Vec<(u32, u32)> = (0..a.kv.len() as u32).map(|i| (i, i + 3)).collect();
            copy(
                &mut model,
                kv,
                MemoryDomain::HostPinned,
                MemoryDomain::LocalDisk,
                to,
            );
        }
        let spc = facts.shapes[0].slots_per_chunk;
        let mut chunk = (a.kv.len() as u32).div_ceil(spc);
        let mut records = Vec::new();
        for (shape, pieces) in &a.record {
            let s = facts.shape(*shape).expect("a fixed shape");
            let base = chunk * s.slots_per_chunk;
            chunk += (pieces.len() as u32).div_ceil(s.slots_per_chunk);
            let pairs: Vec<(u32, u32)> = pieces
                .iter()
                .enumerate()
                .map(|(i, p)| (p[0], base + i as u32))
                .collect();
            copy(&mut model, *shape, device, MemoryDomain::HostPinned, pairs);
            records.push((*shape, base, pieces.len()));
        }
        // In: fresh device slots.
        let fresh: Vec<u32> = (0..a.kv.len()).map(|_| refs.take(kv)).collect();
        let (from, back): (MemoryDomain, Vec<(u32, u32)>) = if on_disk {
            (
                MemoryDomain::LocalDisk,
                fresh
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| (i as u32 + 3, r))
                    .collect(),
            )
        } else {
            (
                MemoryDomain::HostPinned,
                fresh
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| (i as u32, r))
                    .collect(),
            )
        };
        copy(&mut model, kv, from, device, back);
        a.kv = fresh;
        for (at, (shape, base, pieces)) in records.into_iter().enumerate() {
            let fresh: Vec<u32> = (0..pieces).map(|_| refs.take(shape)).collect();
            let back: Vec<(u32, u32)> = fresh
                .iter()
                .enumerate()
                .map(|(i, &r)| (base + i as u32, r))
                .collect();
            copy(&mut model, shape, MemoryDomain::HostPinned, device, back);
            a.record[at].1 = fresh.iter().map(|&r| [r, r]).collect();
        }
    }
    for step in 0..3u64 {
        let lane = a.lane(&model, &mut refs, tokens(20 + step, 1, vocab));
        out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
        a.held += 1;
    }
    out
}

/// A working set parked on the host tier (its kv slots through the slot
/// file too) and restored to fresh device slots decodes on bit for bit as
/// it would have unparked (CONTRACT-BC 22).
#[test]
fn a_parked_working_set_decodes_on_as_it_would_have() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("pie-xla-park-{}", std::process::id()));
    for name in ["qwen3-micro-text", "qwen35-tiny"] {
        let stayed = parked(name, false, None);
        let moved = parked(name, true, Some(&dir));
        assert!(
            stayed == moved,
            "{name}: the parked working set answers otherwise"
        );
        eprintln!(
            "{name}: {} readouts bit-identical after park and restore",
            moved.len()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
