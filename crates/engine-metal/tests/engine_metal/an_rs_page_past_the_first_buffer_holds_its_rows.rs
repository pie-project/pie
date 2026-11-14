#![cfg(target_vendor = "apple")]

//! A lane's buffered recurrent rows in rs-list pages past the first buffer,
//! in a chunk that holds no whole number of rs slots (a page id then names
//! its chunk in its buffer and its index in that chunk): buffering rows and
//! carrying them into the next fire serves the logits of the same walk on
//! the seats' runs in one buffer, bit for bit.

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::store::layout::{FIXED, LIST, RS};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_near, same, word};

const CONTEXT: u32 = 1024;

fn buffer(pages: &[u32], folded: u32, fold: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: pages.to_vec(),
        base: 0,
        folded,
        fold: FoldLen::Host(fold),
    }
}

/// Where a refs-addressed lane keeps its state: kv pages, record pieces
/// and rs pages, slot refs each.
struct Refs {
    kv: Vec<u32>,
    record: Vec<u32>,
    rs: Vec<u32>,
}

/// Buffers `UNFOLDED_ROWS + 1` rows, folding one, then carries the rest
/// into a one-token fire: both fires' logits.
fn walk(shell: &mut Shell, fixture: &Fixture, rs_pages: u32, refs: Option<&Refs>) -> Vec<Vec<f32>> {
    let bound = engine::fire::UNFOLDED_ROWS;
    let rows: Vec<u32> = (0..=bound).map(|at| (7 + at * 37) % 256).collect();
    let next = [42u32];
    let seat_pages: Vec<u32> = (0..rs_pages).collect();
    let rs_pages = refs.map_or(seat_pages.as_slice(), |refs| refs.rs.as_slice());
    let mut out = Vec::new();
    for (tokens, verb, reset, held) in [
        (&rows[..], buffer(rs_pages, 0, 1), RsReset::Fresh, 0),
        (&next[..], buffer(rs_pages, 1, 0), RsReset::Held, bound + 1),
    ] {
        let lane = Seated {
            held: Some(held),
            rs: &verb,
            rs_reset: reset,
            refs: refs.is_some(),
            pages: refs.map_or(&[][..], |refs| refs.kv.as_slice()),
            record: refs.map(|refs| refs.record.as_slice()),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens,
            })
        };
        out.extend(shell.fire_seated(&[lane]).expect("the fire lands"));
    }
    out
}

#[test]
fn an_rs_page_past_the_first_buffer_holds_its_rows() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    engine_metal::store::pool::cap_parts(1);
    let chunk = [
        96u64 << 10,
        160 << 10,
        224 << 10,
        320 << 10,
        448 << 10,
        3 << 19,
    ]
    .into_iter()
    .find(|&chunk| {
        let shell = load_near(&fixture, CONTEXT, Some(chunk));
        let layout = shell.pool_layout();
        let (ids, parts) = shell.page_ids();
        ids.shift.is_some()
            && parts > 1
            && layout
                .rs
                .is_some_and(|rs| !layout.chunk_bytes.is_multiple_of(rs.slot_bytes))
    })
    .expect("a chunk the rs slots do not tile");
    let mut shell = load_near(&fixture, CONTEXT, Some(chunk));
    let layout = shell.pool_layout();
    let facts = shell.pool_chunk_facts();
    let segment = facts.segment_chunks;
    let per = |id| facts.shape(id).map_or(1, |s| s.slots_per_chunk);
    // The walk's rows (the bound and one more) over the rs slot's tokens.
    let rs_pages = (engine::fire::UNFOLDED_ROWS + 2).div_ceil(layout.rs_tokens.max(1)) + 1;
    // A chunk a shape, from the first segment past the seats: under the
    // buffer cap, a buffer past the first.
    let first = layout.free_chunk().div_ceil(segment).max(1) * segment;
    let pieces = layout.pieces.max(1);
    let rs_chunk = first + 1 + pieces;
    shell
        .commit_segments(rs_chunk / segment + 1)
        .expect("the segments commit");
    let rs_per = per(RS);
    let list_per = per(LIST);
    assert!(
        rs_per >= rs_pages && list_per >= 8,
        "the chunks hold the lane's {rs_pages} rs pages ({rs_per} a chunk) and 4 kv \
         pages, one slot apart ({list_per} a chunk)"
    );
    let refs = Refs {
        kv: (0..4).map(|at| first * list_per + 2 * at + 1).collect(),
        record: (0..pieces).map(|p| (first + 1 + p) * per(FIXED)).collect(),
        // Scattered over the chunk's slots, its last first.
        rs: (0..rs_pages)
            .map(|at| rs_chunk * rs_per + (rs_per - 1 - at))
            .collect(),
    };
    let (ids, parts) = shell.page_ids();
    eprintln!(
        "qwen35-tiny: chunk {} B over {parts} buffers, rs slot {} B of {} tokens ({rs_per} \
         a chunk), a page id's index {} bits; the lane's state from chunk {first}",
        ids.chunk_bytes,
        layout.rs.map_or(0, |rs| rs.slot_bytes),
        layout.rs_tokens,
        ids.shift.unwrap_or(0)
    );
    let got = walk(&mut shell, &fixture, rs_pages, Some(&refs));
    drop(shell);
    engine_metal::store::pool::cap_parts(0);

    // The same walk on the seats' runs at the profile's chunk.
    let mut seats = load_near(&fixture, CONTEXT, None);
    seats.open(0).expect("slot 0 opens");
    let want = walk(&mut seats, &fixture, rs_pages, None);
    drop(seats);
    same(
        "rs pages past the first buffer, split ids, against the seats' runs",
        &got,
        &want,
    );
}
