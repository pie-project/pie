use engine::chunk::{Evacuation, Growth, Reserved, SlotKind, SlotRole};

use super::*;

const PAGE: u32 = 16;

fn list(widths: &[u32], slots: u64) -> Draft {
    Draft {
        tokens: PAGE,
        sections: widths
            .iter()
            .enumerate()
            .map(|(at, &width)| Section {
                row: at as u32,
                plane: 0,
                width,
            })
            .collect(),
        rows: (0..widths.len() as u32).collect(),
        slots,
    }
}

fn record(widths: &[u32], records: u64) -> Draft {
    Draft {
        tokens: 1,
        slots: records,
        ..list(widths, 0)
    }
}

/// A kv list of 64-word rows, a window list, an rs list and a record of
/// 9M words: larger than a chunk, so it is cut.
fn hybrid() -> Layout {
    Layout::plan(
        &Drafts {
            list: Some(list(&[64, 64], 512)),
            window: Some((list(&[32], 256), 128)),
            fixed: Some(record(&[4 << 20, 5 << 20], 6)),
            rs: Some(list(&[8], 12)),
        },
        TARGET_CHUNK_BYTES,
    )
    .expect("the pool lays out")
}

#[test]
fn a_record_larger_than_a_chunk_is_cut_into_equal_pieces() {
    let layout = hybrid();
    let fixed = layout.shape(FIXED).expect("a record");
    let SlotKind::Fixed { pieces } = fixed.kind else {
        panic!("a record is fixed");
    };
    assert!(pieces > 1, "{pieces} pieces");
    assert!(fixed.slot_words * u64::from(pieces) >= fixed.record_words);
    assert!(fixed.slot_words <= layout.chunk_words);
    // A chunk is a whole number of kv slots.
    let kv = layout.shape(LIST).expect("a kv list");
    assert_eq!(layout.chunk_words % kv.slot_words, 0);
    // Every shape's slots fit, past the null chunk.
    let chunks: u64 = layout
        .shapes
        .iter()
        .map(|shape| {
            let slots = match shape.id {
                LIST => 512,
                WINDOW => 256,
                RS => 12,
                _ => 6 * u64::from(pieces),
            };
            slots.div_ceil(u64::from(shape.per_chunk))
        })
        .sum();
    assert_eq!(u64::from(layout.total_chunks), 1 + chunks);
}

#[test]
fn the_facts_state_one_shared_pool() {
    let pool = HostPool::new(hybrid(), true);
    let facts = pool.facts();
    facts.validate().expect("the facts hang together");
    assert!(facts.shapes.iter().all(|shape| shape.pinned.is_none()));
    assert_eq!(facts.reserved.len(), 1);
    assert_eq!(facts.reserved[0].what, Reserved::Null);
    assert_eq!(facts.segments.growth, Growth::Bottom);
    assert_eq!(facts.segments.committed, facts.segments.max);
    let rs = facts.shape(RS).expect("an rs list");
    assert_eq!(rs.role, SlotRole::RsBuffer);
    assert_eq!(
        facts.shape(WINDOW).map(|s| s.kind.window()),
        Some(Some(128))
    );
    assert!(facts.act_chunks.is_empty() && facts.weights.is_empty());
}

#[test]
fn a_ref_in_the_null_chunk_or_past_the_pool_is_refused() {
    let pool = HostPool::new(hybrid(), false);
    let per = pool.shape(LIST).expect("a kv list").per_chunk;
    assert!(pool.slot(LIST, 0).is_err());
    assert!(pool.slot(LIST, per - 1).is_err());
    assert!(pool.slot(LIST, per).is_ok());
    assert!(pool.slot(LIST, per * pool.layout().total_chunks).is_err());
}

fn fill(pool: &mut HostPool, id: ShapeId, slot: u32, seed: u32) {
    let words = pool
        .slot_mut(id, slot)
        .expect("a slot")
        .expect("a host pool");
    for (at, word) in words.iter_mut().enumerate() {
        *word = seed.wrapping_mul(2_654_435_761).wrapping_add(at as u32);
    }
}

#[test]
fn rows_copy_within_every_section() {
    let mut pool = HostPool::new(hybrid(), false);
    let per = pool.shape(LIST).expect("a kv list").per_chunk;
    let (src, dst) = (per, per + 1);
    fill(&mut pool, LIST, src, 7);
    fill(&mut pool, LIST, dst, 9);
    let before = pool.slot(LIST, dst).unwrap().unwrap().to_vec();
    pool.copy_rows(LIST, (src, 2), (dst, 5), 3)
        .expect("the rows copy");
    let from = pool.slot(LIST, src).unwrap().unwrap().to_vec();
    let to = pool.slot(LIST, dst).unwrap().unwrap().to_vec();
    let shape = pool.shape(LIST).unwrap().clone();
    let mut want = before;
    for (section, first, _) in shape.sections() {
        let w = section.width as usize;
        let (s, d) = (first as usize + 2 * w, first as usize + 5 * w);
        want[d..d + 3 * w].copy_from_slice(&from[s..s + 3 * w]);
    }
    assert_eq!(to, want);
    assert!(pool.copy_rows(LIST, (src, 14), (dst, 0), 3).is_err());
}

#[test]
fn an_evacuation_moves_a_chunk_and_its_refs_still_read_it() {
    let layout = Layout {
        segment_chunks: 4,
        ..hybrid()
    };
    let mut pool = HostPool::new(layout, false);
    let total = pool.layout().total_chunks;
    let per = pool.shape(LIST).unwrap().per_chunk;
    // A slot in the last segment's first chunk.
    let last = (total - 1) / 4 * 4;
    let slot = last * per + 1;
    fill(&mut pool, LIST, slot, 3);
    let held = pool.slot(LIST, slot).unwrap().unwrap().to_vec();
    // Chunk 1 is free: move the held chunk there, and release the segment.
    pool.evacuate(&Evacuation {
        moves: vec![(last, 1)],
    })
    .expect("the evacuation applies");
    pool.release(last / 4);
    assert_eq!(pool.committed(), last / 4);
    assert_eq!(pool.slot(LIST, slot).unwrap().unwrap(), held.as_slice());
    // The free logical chunk that sat on physical 1 now lies past the
    // committed segments.
    assert!(pool.slot(LIST, per + 1).is_err());
    pool.commit(last / 4 + 1)
        .expect("the segment commits again");
    assert!(pool.slot(LIST, per + 1).is_ok());
    // A move onto a released segment is refused, whole.
    pool.release(last / 4);
    assert!(
        pool.evacuate(&Evacuation {
            moves: vec![(2, last)],
        })
        .is_err()
    );
    assert_eq!(pool.slot(LIST, slot).unwrap().unwrap(), held.as_slice());
}

#[test]
fn a_dry_pool_checks_refs_and_holds_no_words() {
    let mut pool = HostPool::new(hybrid(), true);
    let per = pool.shape(LIST).unwrap().per_chunk;
    assert!(pool.slot(LIST, per).expect("a ref in the pool").is_none());
    assert!(pool.copy_slots(LIST, &[(per, 0)]).is_err());
    assert!(pool.committed_bytes() > 0);
}
