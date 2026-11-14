#![cfg(target_vendor = "apple")]

//! A lane forked across buffers (CONTRACT-BC 15, 20): over a range of a
//! segment a buffer, a prompted lane forks: the fork shares its full kv
//! pages, takes a copy of its tail page and of its record (`copy_state`,
//! the run's pieces to the cow's) in chunks of another buffer, and both
//! decode the same tokens. Each lane's logits are the never-forked lane's,
//! bit for bit; a hybrid fork given a blank record instead (the control)
//! moves them.

use engine::chunk::{ShapeId, SlotCopy};
use engine::transfer::MemoryDomain;
use engine_metal::serve::Seated;
use engine_metal::store::layout::{FIXED, LIST};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{
    Fixture, argmax, fixture_of, load_with, same, word,
};

const CONTEXT: u32 = 1024;
const PROMPT: u32 = 150;
const STEPS: usize = 8;

/// Slot refs of one shape a chunk at a time from chunk `next`.
struct Refs {
    next: u32,
    open: Vec<(ShapeId, u32, u32, u32)>,
}

impl Refs {
    fn take(&mut self, shape: ShapeId, per: u32) -> u32 {
        if let Some(open) = self.open.iter_mut().find(|o| o.0 == shape)
            && open.3 < open.2
        {
            open.3 += 1;
            return open.1 * open.2 + open.3 - 1;
        }
        let chunk = self.next;
        self.next += 1;
        self.open.retain(|o| o.0 != shape);
        self.open.push((shape, chunk, per, 1));
        chunk * per
    }
}

/// How the second lane comes to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fork {
    /// No second lane: the reference.
    Never,
    /// Forked: full pages shared, the tail page and the record copied.
    Copied,
    /// As `Copied`, with a record nothing wrote (the control).
    Blank,
}

/// A lane's slot refs: its kv pages and its record's pieces.
#[derive(Clone)]
struct Slots {
    pages: Vec<u32>,
    record: Vec<u32>,
}

fn fire(
    shell: &mut Shell,
    fixture: &Fixture,
    slot: u32,
    held: u32,
    fed: &[u32],
    slots: &Slots,
) -> Vec<f32> {
    let seated = Seated {
        held: Some(held),
        refs: true,
        pages: &slots.pages,
        record: (!slots.record.is_empty()).then_some(slots.record.as_slice()),
        ..Seated::of(Lane {
            slot,
            word: word(&fixture.trace, fed.len() as u32),
            tokens: fed,
        })
    };
    let mut rows = shell.fire_seated(&[seated]).expect("the fire lands");
    rows.remove(0)
}

/// The buffer slot ref `slot` of a shape of `per` slots a chunk lies in,
/// over a range of `segment` chunks a buffer.
fn buffer_of(slot: u32, per: u32, segment: u32) -> u32 {
    slot / per / segment
}

/// A prompt, then greedy decodes; with a fork, the second lane decodes the
/// same tokens beside the first. Every logit row of each lane.
fn serve(fixture: &Fixture, fork: Fork) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut shell = load_with(fixture, CONTEXT);
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    let segment = facts.segment_chunks;
    let per = |id: ShapeId| facts.shape(id).map(|s| s.slots_per_chunk);
    let list = per(LIST).expect("a kv list");
    let fixed = per(FIXED);
    let layout = shell.pool_layout();
    let first = layout.free_chunk().div_ceil(segment) * segment;
    // The fork's slots start a segment (a buffer) past the first lane's.
    let second = first + segment;
    shell
        .commit_segments((second + segment).div_ceil(segment))
        .expect("the segments commit");
    let mut here = Refs {
        next: first,
        open: Vec::new(),
    };
    let mut there = Refs {
        next: second,
        open: Vec::new(),
    };
    let page = shell.paging().page_size;
    let mut a = Slots {
        pages: Vec::new(),
        record: fixed.map_or_else(Vec::new, |per| {
            (0..layout.pieces).map(|_| here.take(FIXED, per)).collect()
        }),
    };
    let mut fed: Vec<u32> = (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect();
    let mut held = 0u32;
    while (a.pages.len() as u32) < (held + fed.len() as u32).div_ceil(page) {
        a.pages.push(here.take(LIST, list));
    }
    let prompt = fire(&mut shell, fixture, 0, held, &fed, &a);
    held += fed.len() as u32;
    fed = vec![argmax(&prompt)];
    let mut rows_a = vec![prompt];
    let mut rows_b = Vec::new();

    let mut b = None;
    if fork != Fork::Never {
        // The full pages are shared; the tail page (the one the next row
        // lands in) and the record are the fork's own, in another buffer.
        let full = (held / page) as usize;
        let mut pages: Vec<u32> = a.pages[..full].to_vec();
        let mut copies = Vec::new();
        for &tail in &a.pages[full..] {
            let own = there.take(LIST, list);
            assert_ne!(
                buffer_of(tail, list, segment),
                buffer_of(own, list, segment),
                "the fork's tail page lies in another buffer"
            );
            copies.push((tail, own));
            pages.push(own);
        }
        shell
            .copy_slots(&SlotCopy {
                shape: LIST,
                src: MemoryDomain::MetalShared,
                dst: MemoryDomain::MetalShared,
                slots: copies,
                runs: Vec::new(),
            })
            .expect("the tail page copies");
        let record: Vec<u32> = fixed.map_or_else(Vec::new, |per| {
            a.record.iter().map(|_| there.take(FIXED, per)).collect()
        });
        if let Some(per) = fixed {
            for (&run, &cow) in a.record.iter().zip(&record) {
                assert_ne!(
                    buffer_of(run, per, segment),
                    buffer_of(cow, per, segment),
                    "the fork's record lies in another buffer"
                );
            }
        }
        if fork == Fork::Copied {
            let moves: Vec<(u32, u32)> = a
                .record
                .iter()
                .copied()
                .zip(record.iter().copied())
                .collect();
            shell.copy_state(&moves).expect("the record forks");
        }
        b = Some(Slots { pages, record });
    }

    for _ in 0..STEPS {
        let after = held + fed.len() as u32;
        while (a.pages.len() as u32) < after.div_ceil(page) {
            a.pages.push(here.take(LIST, list));
        }
        let row_a = fire(&mut shell, fixture, 0, held, &fed, &a);
        if let Some(b) = b.as_mut() {
            while (b.pages.len() as u32) < after.div_ceil(page) {
                b.pages.push(there.take(LIST, list));
            }
            rows_b.push(fire(&mut shell, fixture, 1, held, &fed, b));
        }
        held = after;
        fed = vec![argmax(&row_a)];
        rows_a.push(row_a);
    }
    (rows_a, rows_b)
}

fn check(model: &str) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of(model, None);
    let (kept, _) = serve(&fixture, Fork::Never);
    let (first, forked) = serve(&fixture, Fork::Copied);
    same(
        &format!("{model}: the forked-from lane against kept"),
        &first,
        &kept,
    );
    same(
        &format!("{model}: the fork against kept"),
        &forked,
        &kept[1..],
    );
    eprintln!(
        "{model}: both sides of a fork across buffers = kept over {} rows",
        forked.len()
    );
    if fixture
        .trace
        .caches
        .iter()
        .any(|row| matches!(row, poem_ir::CacheRow::State { .. }))
    {
        let (_, blank) = serve(&fixture, Fork::Blank);
        assert_ne!(
            blank,
            kept[1..].to_vec(),
            "{model}: a fork with a blank record moves the logits"
        );
        eprintln!("{model}: a blank-record fork moves the logits (the control)");
    }
}

/// The dense and hybrid lanes forked across buffers, a segment a buffer.
#[test]
fn a_forked_lane_over_buffers_serves_the_unforked_logits() {
    engine_metal::store::pool::cap_parts(1);
    let lifted = std::panic::catch_unwind(|| {
        check("qwen3-micro-text");
        check("qwen35-tiny");
    });
    engine_metal::store::pool::cap_parts(0);
    if let Err(why) = lifted {
        std::panic::resume_unwind(why);
    }
}
