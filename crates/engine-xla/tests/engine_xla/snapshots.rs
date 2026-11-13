//! Snapshots inside a pass (CONTRACT-BC 16) on XLA: a prompt that lands
//! its recurrent state as of interior rows in records of their own answers
//! what the pass cut at those rows answers, bit for bit: each snapshot's
//! record, the lane's own record, its readout, a lane decoding beside it
//! and a decode from each snapshot. Skipped without a PJRT plugin.

use engine::chunk::{ShapeId, SlotKind};
use engine::fire::{Lane, RsSnapshot};

use crate::common_llm::{Model, Refs, Seq, bits, tokens};

const MODEL: &str = "qwen35-tiny";
const PROMPT: usize = 150;
const ROWS: [u32; 3] = [37, 64, 100];

/// A record: every fixed shape's slot refs, a piece each.
type Record = Vec<(ShapeId, Vec<u32>)>;

/// The fixed shapes a record lies in, and their pieces.
fn fixed_shapes(model: &Model) -> Vec<(ShapeId, u32)> {
    let facts = model.loaded.caps.chunks.clone();
    facts
        .shapes
        .iter()
        .filter_map(|s| match s.kind {
            SlotKind::Fixed { pieces } => Some((s.id, pieces)),
            _ => None,
        })
        .collect()
}

fn record_bytes(model: &Model, record: &Record) -> Vec<u8> {
    let shell = model.engine.shell().expect("loaded");
    let mut out = Vec::new();
    for (shape, pieces) in record {
        let slots: Vec<(ShapeId, u32)> = pieces.iter().map(|&p| (*shape, p)).collect();
        out.extend(shell.slot_bytes(&slots).expect("the record reads"));
    }
    out
}

fn write_record(model: &mut Model, record: &Record, bytes: &[u8]) {
    let shell = model.engine.shell_mut().expect("loaded");
    let mut at = 0usize;
    for (shape, pieces) in record {
        let slots: Vec<(ShapeId, u32)> = pieces.iter().map(|&p| (*shape, p)).collect();
        let n = shell.slot_bytes(&slots).expect("the record reads").len();
        shell
            .write_slot_bytes(&slots, &bytes[at..at + n])
            .expect("the record lands");
        at += n;
    }
}

/// The record a sequence writes.
fn own(seq: &Seq) -> Record {
    seq.record
        .iter()
        .map(|(id, pairs)| (*id, pairs.iter().map(|&[_, cow]| cow).collect()))
        .collect()
}

fn fresh(refs: &mut Refs, shapes: &[(ShapeId, u32)]) -> Record {
    shapes
        .iter()
        .map(|&(shape, pieces)| (shape, (0..pieces).map(|_| refs.take(shape)).collect()))
        .collect()
}

/// A decode from a record: a new sequence on the prompt's kv, folding into
/// `record` in place.
fn decode_from(
    model: &mut Model,
    refs: &mut Refs,
    seq: &Seq,
    row: u32,
    record: &Record,
) -> Vec<u32> {
    let mut from = Seq::new(3);
    from.held = row;
    from.kv = seq.kv.clone();
    from.record = record
        .iter()
        .map(|(shape, pieces)| (*shape, pieces.iter().map(|&r| [r, r]).collect()))
        .collect();
    let lane = from.lane(model, refs, tokens(99, 1, model.vocab));
    bits(&model.fire(vec![lane])[0])
}

struct Landed {
    snaps: Vec<Vec<u8>>,
    own: Vec<u8>,
    readouts: Vec<Vec<u32>>,
    decodes: Vec<Vec<u32>>,
}

/// The prompt, its snapshots (in the pass, or the pass cut at each row and
/// the record read after each piece), a lane `b` decoding beside its last
/// rows, then a decode from each snapshot.
fn fire(snapshots: bool) -> Option<Landed> {
    let _device = crate::common_dit::device()?;
    let mut model = Model::load(MODEL, None, 256, 7);
    assert!(
        model.loaded.caps.rs_snapshots >= ROWS.len() as u32,
        "{MODEL}: the engine lands {} snapshots a lane",
        model.loaded.caps.rs_snapshots
    );
    let shapes = fixed_shapes(&model);
    eprintln!("{MODEL}: a record over fixed shapes {shapes:?}");
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let mut a = Seq::new(0);
    let mut b = Seq::new(1);
    let mut readouts = Vec::new();
    let lane = b.lane(&model, &mut refs, tokens(2, 20, vocab));
    readouts.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
    b.held += 20;
    let prompt = tokens(1, PROMPT, vocab);
    let records: Vec<Record> = ROWS.iter().map(|_| fresh(&mut refs, &shapes)).collect();
    let mut snaps = Vec::new();
    if snapshots {
        let mut lane: Lane = a.lane(&model, &mut refs, prompt.clone());
        lane.rs_snapshots = ROWS
            .iter()
            .zip(&records)
            .map(|(&row, record)| RsSnapshot {
                row,
                slot: record[0].1[0],
                pieces: if record[0].1.len() > 1 {
                    record[0].1.clone()
                } else {
                    Vec::new()
                },
                fixed: record.clone(),
            })
            .collect();
        let lb = b.lane(&model, &mut refs, tokens(21, 1, vocab));
        readouts.extend(model.fire(vec![lane, lb]).iter().map(|v| bits(v)));
        for record in &records {
            snaps.push(record_bytes(&model, record));
        }
    } else {
        let mut from = 0usize;
        for &row in &ROWS {
            let lane = a.lane(&model, &mut refs, prompt[from..row as usize].to_vec());
            model.fire(vec![lane]);
            a.held = row;
            snaps.push(record_bytes(&model, &own(&a)));
            from = row as usize;
        }
        let lane = a.lane(&model, &mut refs, prompt[from..].to_vec());
        let lb = b.lane(&model, &mut refs, tokens(21, 1, vocab));
        readouts.extend(model.fire(vec![lane, lb]).iter().map(|v| bits(v)));
    }
    a.held = PROMPT as u32;
    let own_bytes = record_bytes(&model, &own(&a));
    let mut decodes = Vec::new();
    for (k, &row) in ROWS.iter().enumerate() {
        if !snapshots {
            // The bytes the cut pass read, landed in a record of their own.
            write_record(&mut model, &records[k], &snaps[k]);
        }
        decodes.push(decode_from(&mut model, &mut refs, &a, row, &records[k]));
    }
    Some(Landed {
        snaps,
        own: own_bytes,
        readouts,
        decodes,
    })
}

#[test]
fn snapshots_inside_a_pass_land_what_the_cut_pass_lands() {
    let Some(snapshotted) = fire(true) else {
        return;
    };
    let cut = fire(false).expect("the device is there");
    for (k, (a, b)) in snapshotted.snaps.iter().zip(&cut.snaps).enumerate() {
        assert!(!a.is_empty(), "snapshot {k} reads back");
        assert!(
            a == b,
            "snapshot at row {} = the cut pass's record",
            ROWS[k]
        );
    }
    assert!(snapshotted.own == cut.own, "the lane's own record");
    assert!(
        snapshotted.readouts == cut.readouts,
        "the readouts (the prompt's and the lane decoding beside it)"
    );
    assert!(
        snapshotted.decodes == cut.decodes,
        "a decode from each snapshot"
    );
    assert!(
        snapshotted.snaps.iter().all(|s| *s != snapshotted.own),
        "a snapshot is not the state after the whole prompt"
    );
    eprintln!(
        "{MODEL}: {} snapshots, the record, {} readouts and {} decodes = the cut pass, bit for bit",
        ROWS.len(),
        snapshotted.readouts.len(),
        snapshotted.decodes.len()
    );
}
