//! Snapshots inside a pass (CONTRACT-BC 16, `Lane::rs_snapshots`): a lane
//! prefilling its prompt in one fire lands the recurrent state as of rows
//! inside it in records of their own, a record in pieces where the pool's
//! chunk is smaller than one. Each snapshot's record holds, bit for bit,
//! the state a pass cut at its row lands (the prompt's first rows alone),
//! and the lane's own record and logits are the pass without snapshots'.
//! A snapshot's record also decodes a token to the cut pass's logits.
//! Beside a lane that buffers its row unfolded (the fire's scans then run
//! their committed arms) the snapshots land the same.

use engine::fire::{FoldLen, RsSnapshot, RsVerb};
use engine_wgpu::serve::Seated;
use engine_wgpu::store::layout::{FIXED, RS};

use crate::model_logits::{PAGE, argmax, fixture, load_with, same};
use crate::tiers::{Held, Refs, committed, pieces, rs_tokens};

/// The rows inside the prompt snapshotted.
const ROWS: [u32; 3] = [37, 64, 101];

fn snapshot_lands_the_cut_pass(model: &str, chunk: Option<u64>, beside_buffering: bool) {
    let fixture = fixture(model, None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        256,
        chunk,
        engine::load::Residency::default(),
    ) else {
        return;
    };
    let facts = shell.pool_chunk_facts();
    let pieces_of = pieces(&facts);
    eprintln!("{model}: a record in {pieces_of} piece(s)");
    let (first, end) = committed(&mut shell, &facts);
    let mut refs = Refs::of(&facts, first, end);
    let trace = &fixture.trace;
    let prompt: Vec<u32> = (0..150u32).map(|at| (at * 37 + 11) % 256).collect();

    // A lane that buffers one row unfolded, fired beside the snapshotting
    // one where asked.
    let mut other = Held::new(&facts, &mut refs);
    let short: Vec<u32> = (0..20u32).map(|at| (at * 53 + 5) % 256).collect();
    other.grow(&facts, &mut refs, short.len() as u32);
    let next = argmax(
        &shell
            .fire_seated(&[other.seated(trace, &short, 1)])
            .expect("the other lane prefills")
            .swap_remove(0),
    );
    other.have = short.len() as u32;
    other.grow(&facts, &mut refs, other.have + 1);
    let t_rs = rs_tokens(&facts);
    let verb = RsVerb::Buffer {
        pages: vec![refs.take(RS)],
        base: other.have - other.have % t_rs,
        folded: other.have,
        fold: FoldLen::Host(0),
    };
    let next = [next];

    // The pass with snapshots, each into a record of its own.
    let mut whole = Held::new(&facts, &mut refs);
    whole.grow(&facts, &mut refs, prompt.len() as u32);
    let records: Vec<Vec<u32>> = ROWS
        .iter()
        .map(|_| (0..pieces_of).map(|_| refs.take(FIXED)).collect())
        .collect();
    let snaps: Vec<RsSnapshot> = ROWS
        .iter()
        .zip(&records)
        .map(|(&row, record)| RsSnapshot {
            row,
            slot: record[0],
            pieces: if record.len() > 1 {
                record.clone()
            } else {
                Vec::new()
            },
            fixed: Vec::new(),
        })
        .collect();
    let mut lanes = vec![Seated {
        rs_snapshots: &snaps,
        ..whole.seated(trace, &prompt, 0)
    }];
    if beside_buffering {
        lanes.push(Seated {
            rs: &verb,
            ..other.seated(trace, &next, 1)
        });
    }
    let snapped = shell
        .fire_seated(&lanes)
        .expect("the pass with snapshots lands")
        .swap_remove(0);

    // The same pass with none.
    let mut plain = Held::new(&facts, &mut refs);
    plain.grow(&facts, &mut refs, prompt.len() as u32);
    let unsnapped = shell
        .fire_seated(&[plain.seated(trace, &prompt, 0)])
        .expect("the pass lands")
        .swap_remove(0);
    same(
        &format!("{model}: the pass with snapshots against none"),
        &[snapped],
        &[unsnapped],
    );
    assert_eq!(
        shell.state_bytes(&whole.record).expect("reads"),
        shell.state_bytes(&plain.record).expect("reads"),
        "{model}: the lane's own record, snapshots or none"
    );

    let mut states: Vec<Vec<u8>> = Vec::new();
    for (&row, record) in ROWS.iter().zip(&records) {
        // The pass cut at the snapshot's row.
        let mut cut = Held::new(&facts, &mut refs);
        cut.grow(&facts, &mut refs, row);
        let last = shell
            .fire_seated(&[cut.seated(trace, &prompt[..row as usize], 0)])
            .expect("the cut pass lands")
            .swap_remove(0);
        cut.have = row;
        let want = shell.state_bytes(&cut.record).expect("reads");
        let got = shell.state_bytes(record).expect("reads");
        assert!(
            got.iter().any(|&b| b != 0) && !states.contains(&got),
            "{model}: the snapshot at row {row} landed, apart from the others"
        );
        states.push(got.clone());
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            differ,
            0,
            "{model}: the snapshot at row {row} against the pass cut there ({differ} of {} \
             bytes differ)",
            want.len()
        );

        // Decoding the next token from the snapshot (on the cut pass's kv
        // pages, copied) lands the cut pass's logits.
        let token = [argmax(&last)];
        let mut from_snap = Held {
            record: record.clone(),
            ..cut.clone()
        };
        let mut from_cut = cut.clone();
        from_cut.grow(&facts, &mut refs, row + 1);
        from_snap.pages = from_cut.pages.clone();
        let want = shell
            .fire_seated(&[from_cut.seated(trace, &token, 0)])
            .expect("the cut lane decodes")
            .swap_remove(0);
        // The cut lane's decode wrote its kv row; the snapshot lane writes
        // the same row on the same page, then reads its own state.
        let got = shell
            .fire_seated(&[from_snap.seated(trace, &token, 0)])
            .expect("the snapshot lane decodes")
            .swap_remove(0);
        same(
            &format!("{model}: a decode from the snapshot at row {row}"),
            &[got],
            &[want],
        );
    }
    eprintln!(
        "{model}: {} snapshots = the passes cut at their rows, bit for bit",
        ROWS.len()
    );
}

/// Gated delta and conv states (Qwen3.5's hybrid), a record in one slot.
#[test]
fn a_snapshot_holds_the_state_of_the_pass_cut_at_its_row() {
    snapshot_lands_the_cut_pass("qwen35-tiny", None, false);
}

/// The same with the record cut into pieces (a chunk smaller than it).
#[test]
fn a_snapshot_in_pieces_holds_the_state_of_the_pass_cut_at_its_row() {
    snapshot_lands_the_cut_pass("qwen35-tiny", Some(64 << 10), false);
}

/// The same beside a lane buffering its row: the committed arms.
#[test]
fn a_snapshot_beside_a_buffering_lane_holds_the_state_of_the_pass_cut_at_its_row() {
    snapshot_lands_the_cut_pass("qwen35-tiny", None, true);
    snapshot_lands_the_cut_pass("qwen35-tiny", Some(64 << 10), true);
}
