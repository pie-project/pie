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
fn a_wide_model_answers_bit_for_bit_past_a_tall_prompt() {
    // kv rows of whole pool rows (8 heads of 128): the tall prompt's pages
    // land by block and by the Mosaic kernel straight into the pool
    run("qwen3-0.6b", Some(2), 1024, Some(700));
}

#[test]
fn a_wide_hybrid_answers_bit_for_bit() {
    // conv and delta state of real widths (6144 conv channels): the decode
    // conv by the pool's Mosaic kernel, the delta step by its own
    run("qwen35-d0.8b", Some(4), 512, None);
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
        let facts = model.loaded.caps.chunks.clone();
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

/// 64 prompts, then three decodes of all 64 lanes (a decode wide enough for
/// the page-copy kernel): every readout.
fn many_lanes(name: &str, layers: Option<u32>) -> Vec<Vec<u32>> {
    use crate::common_llm::set_knob;
    use crate::common_llm::{Refs, Seq, bits, tokens};
    set_knob("SLOTS", 64);
    set_knob("LANES", 64);
    set_knob("TOKENS", 128);
    let mut model = Model::load(name, layers, 256, 7);
    set_knob("SLOTS", 4);
    set_knob("LANES", 8);
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let mut seqs: Vec<Seq> = (0..64).map(Seq::new).collect();
    let mut out = Vec::new();
    for (i, seq) in seqs.iter_mut().enumerate() {
        let n = 17 + (i % 23);
        let lane = seq.lane(&model, &mut refs, tokens(i as u64, n, vocab));
        let r = model.fire(vec![lane]);
        seq.held += n as u32;
        out.push(bits(&r[0]));
    }
    for step in 0..3u32 {
        let lanes: Vec<_> = seqs
            .iter_mut()
            .enumerate()
            .map(|(i, seq)| seq.lane(&model, &mut refs, vec![(i as u32 * 31 + step) % vocab]))
            .collect();
        for r in model.fire(lanes) {
            out.push(bits(&r));
        }
        for seq in &mut seqs {
            seq.held += 1;
        }
    }
    out
}

#[test]
fn a_wide_decode_of_many_lanes_answers_bit_for_bit() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let name = "qwen3-0.6b";
    let bytes = encode(&many_lanes(name, Some(2)));
    let tag = "qwen3-0.6b-many";
    if let Ok(dir) = std::env::var("PIE_XLA_GOLDEN_DUMP") {
        std::fs::create_dir_all(&dir).expect("the dump directory");
        std::fs::write(file(&dir, tag), &bytes).expect("the dump writes");
    }
    if let Ok(dir) = std::env::var("PIE_XLA_GOLDEN") {
        let want = std::fs::read(file(&dir, tag)).expect("the golden file reads");
        assert!(
            want == bytes,
            "{tag}: the readouts differ from the golden build's"
        );
        eprintln!("{tag}: bit-identical");
    }
}
