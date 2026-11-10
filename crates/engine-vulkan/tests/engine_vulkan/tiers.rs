//! Slots moved the ways the runtime moves them, on the device: parked on
//! the host tier and in the slot file and restored into fresh slots
//! (CONTRACT-BC 22), a record copied on write for a group whose one lane
//! folds and whose other reads the run side (CONTRACT-BC 15), and a fork's
//! partially filled tail slot of the rs list or of a window list copied
//! before the fork writes it (CONTRACT-BC 20, 21). Every case lands the
//! logits, bit for bit, of the slots never moved.

use std::collections::BTreeMap;

use engine::chunk::{ChunkFacts, ShapeId, SlotCopy, SlotKind, SlotRole};
use engine::fire::{FoldLen, RsVerb};
use engine::transfer::MemoryDomain;
use engine_vulkan::serve::Seated;
use engine_vulkan::store::layout::{FIXED, LIST, RS};
use engine_vulkan::{Lane, Shell};

use crate::model_logits::{Fixture, PAGE, argmax, fixture, load_with, same, word};

const DEVICE: MemoryDomain = MemoryDomain::VulkanDevice(0);

/// The rows one fire carries at most (the tests' budget).
const ROWS: usize = 256;

/// Slot refs handed out as the runtime would: a chunk opened a shape from
/// `next` up, its slots taken from the top down.
pub struct Refs {
    next: u32,
    end: u32,
    per: BTreeMap<ShapeId, u32>,
    open: BTreeMap<ShapeId, (u32, u32)>,
}

impl Refs {
    pub fn of(facts: &ChunkFacts, first: u32, end: u32) -> Refs {
        Refs {
            next: first,
            end,
            per: facts
                .shapes
                .iter()
                .map(|s| (s.id, s.slots_per_chunk))
                .collect(),
            open: BTreeMap::new(),
        }
    }

    pub fn take(&mut self, shape: ShapeId) -> u32 {
        let per = self.per[&shape];
        let (chunk, used) = match self.open.get(&shape) {
            Some(&(chunk, used)) if used < per => (chunk, used),
            _ => {
                let chunk = self.next;
                self.next += 1;
                assert!(chunk < self.end, "the refs stay in their chunks");
                (chunk, 0)
            }
        };
        self.open.insert(shape, (chunk, used + 1));
        chunk * per + (per - 1 - used)
    }
}

/// What one lane holds: its kv list pages, each window list's table (0 a
/// slot trimmed behind its window) and its record's pieces.
#[derive(Clone, Debug, Default)]
pub struct Held {
    pub pages: Vec<u32>,
    pub windows: Vec<(ShapeId, Vec<u32>)>,
    pub record: Vec<u32>,
    pub have: u32,
}

/// The lists a load's lanes page: shape, tokens a slot, window.
fn lists(facts: &ChunkFacts) -> Vec<(ShapeId, u32, Option<u32>)> {
    facts
        .shapes
        .iter()
        .filter(|s| s.role == SlotRole::State)
        .filter_map(|s| match s.kind {
            SlotKind::List { tokens, window, .. } => Some((s.id, tokens, window)),
            _ => None,
        })
        .collect()
}

pub fn pieces(facts: &ChunkFacts) -> u32 {
    match facts.shape(FIXED).map(|s| s.kind) {
        Some(SlotKind::Fixed { pieces }) => pieces,
        _ => 0,
    }
}

impl Held {
    /// A fresh lane: its record's pieces taken.
    pub fn new(facts: &ChunkFacts, refs: &mut Refs) -> Held {
        Held {
            record: (0..pieces(facts)).map(|_| refs.take(FIXED)).collect(),
            windows: lists(facts)
                .into_iter()
                .filter(|l| l.2.is_some())
                .map(|l| (l.0, Vec::new()))
                .collect(),
            ..Held::default()
        }
    }

    /// Every list's slots grown to `after` tokens, and each window list
    /// trimmed as the runtime trims at `have` committed rows.
    pub fn grow(&mut self, facts: &ChunkFacts, refs: &mut Refs, after: u32) {
        let have = self.have;
        for (id, tokens, window) in lists(facts) {
            let table = if id == LIST {
                &mut self.pages
            } else {
                &mut self
                    .windows
                    .iter_mut()
                    .find(|(held, _)| *held == id)
                    .expect("a window list's table")
                    .1
            };
            while (table.len() as u32) < after.div_ceil(tokens) {
                table.push(refs.take(id));
            }
            if let Some(w) = window {
                for (k, slot) in table.iter_mut().enumerate() {
                    if (k as u32 + 1) * tokens - 1 + w < have {
                        *slot = 0;
                    }
                }
            }
        }
    }

    pub fn seated<'a>(
        &'a self,
        trace: &poem_ir::Trace,
        tokens: &'a [u32],
        slot: u32,
    ) -> Seated<'a> {
        Seated {
            pages: &self.pages,
            windows: &self.windows,
            held: Some(self.have),
            refs: true,
            record: (!self.record.is_empty()).then_some(&self.record[..]),
            ..Seated::of(Lane {
                slot,
                word: word(trace, tokens.len() as u32),
                tokens,
            })
        }
    }
}

/// Fires `tokens` on `held` alone, in pieces of a fire's rows; the last
/// row's logits.
fn feed(
    shell: &mut Shell,
    trace: &poem_ir::Trace,
    facts: &ChunkFacts,
    refs: &mut Refs,
    held: &mut Held,
    tokens: &[u32],
) -> Vec<f32> {
    let mut last = Vec::new();
    for piece in tokens.chunks(ROWS) {
        let after = held.have + piece.len() as u32;
        held.grow(facts, refs, after);
        last = shell
            .fire_seated(&[held.seated(trace, piece, 0)])
            .expect("the fire lands")
            .swap_remove(0);
        held.have = after;
    }
    last
}

/// The tiers a slot parks in: the host chunks, or the slot file (kv list
/// slots only).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Host,
    Disk,
}

/// Host and disk refs handed out in order.
struct Parked {
    host: Refs,
    disk: u32,
}

/// Moves `slots` (0: trimmed, kept) of `shape` to `tier` and back into
/// fresh device slots; the fresh refs.
fn park(
    shell: &mut Shell,
    refs: &mut Refs,
    parked: &mut Parked,
    tier: Tier,
    shape: ShapeId,
    slots: &[u32],
) -> Vec<u32> {
    let held: Vec<u32> = slots.iter().copied().filter(|&s| s != 0).collect();
    let (domain, away): (MemoryDomain, Vec<u32>) = match tier {
        Tier::Host => (
            MemoryDomain::HostPinned,
            held.iter().map(|_| parked.host.take(shape)).collect(),
        ),
        Tier::Disk => (
            MemoryDomain::LocalDisk,
            held.iter()
                .map(|_| {
                    parked.disk += 1;
                    parked.disk - 1
                })
                .collect(),
        ),
    };
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: DEVICE,
            dst: domain,
            slots: held.iter().copied().zip(away.iter().copied()).collect(),
            runs: Vec::new(),
        })
        .expect("the slots park");
    let fresh: Vec<u32> = held.iter().map(|_| refs.take(shape)).collect();
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: domain,
            dst: DEVICE,
            slots: away.iter().copied().zip(fresh.iter().copied()).collect(),
            runs: Vec::new(),
        })
        .expect("the slots come back");
    let mut fresh = fresh.into_iter();
    slots
        .iter()
        .map(|&s| {
            if s == 0 {
                0
            } else {
                fresh.next().expect("one each")
            }
        })
        .collect()
}

/// The chunk the lanes' refs start at: past the seats' runs, a segment of
/// their own; and the segments the pool commits for them.
pub fn committed(shell: &mut Shell, facts: &ChunkFacts) -> (u32, u32) {
    let low = 1 + shell.pool_layout().seat_chunks();
    let first = low.div_ceil(facts.segment_chunks) * facts.segment_chunks;
    let segments = facts.segments.max.min(first / facts.segment_chunks + 2);
    shell
        .commit_segments(segments)
        .expect("the segments commit");
    (first, segments * facts.segment_chunks)
}

const STEPS: usize = 10;

/// One lane prefills `prompt` tokens and decodes [`STEPS`]; where `offload`,
/// every slot it holds parks on the host and comes back at decode 2, and
/// its kv list through the slot file (the rest through the host again) at
/// decode 6. Every step's logits.
fn served_offloaded(
    fixture: &Fixture,
    page: u32,
    context: u32,
    prompt: u32,
    offload: bool,
) -> Option<Vec<Vec<f32>>> {
    let residency = engine::load::Residency {
        host_kv_budget: Some(64 << 20),
        ..engine::load::Residency::default()
    };
    let mut shell = load_with(fixture, page, context, None, residency)?;
    let dir = std::env::temp_dir().join(format!(
        "pie-vk-tier-{}-{}",
        std::process::id(),
        u32::from(offload)
    ));
    std::fs::create_dir_all(&dir).expect("the disk tier's dir");
    let disk = shell
        .seat_disk_kv(Some(64 << 20), Some(&dir), 1, context)
        .expect("the disk tier seats");
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    assert!(facts.host_chunks > 0, "a host tier is stated");
    assert!(
        disk > 0 && facts.disk_chunks == disk,
        "a disk tier is stated"
    );
    let (first, end) = committed(&mut shell, &facts);
    let mut refs = Refs::of(&facts, first, end);
    let mut parked = Parked {
        host: Refs::of(&facts, 0, facts.host_chunks),
        disk: 0,
    };
    let trace = &fixture.trace;
    let mut held = Held::new(&facts, &mut refs);
    let prompt: Vec<u32> = (0..prompt).map(|at| (at * 37 + 11) % 256).collect();
    let mut rows = vec![feed(
        &mut shell, trace, &facts, &mut refs, &mut held, &prompt,
    )];
    for step in 0..STEPS {
        if offload && (step == 2 || step == 6) {
            let kv = if step == 2 { Tier::Host } else { Tier::Disk };
            held.pages = park(&mut shell, &mut refs, &mut parked, kv, LIST, &held.pages);
            let windows = std::mem::take(&mut held.windows);
            held.windows = windows
                .into_iter()
                .map(|(id, table)| {
                    let moved = park(&mut shell, &mut refs, &mut parked, Tier::Host, id, &table);
                    (id, moved)
                })
                .collect();
            if !held.record.is_empty() {
                held.record = park(
                    &mut shell,
                    &mut refs,
                    &mut parked,
                    Tier::Host,
                    FIXED,
                    &held.record,
                );
            }
        }
        let token = [argmax(rows.last().expect("a row"))];
        rows.push(feed(
            &mut shell, trace, &facts, &mut refs, &mut held, &token,
        ));
    }
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
    Some(rows)
}

fn offloaded(model: &str, layers: Option<u32>, page: u32, context: u32, prompt: u32) {
    let fixture = fixture(model, layers);
    let Some(kept) = served_offloaded(&fixture, page, context, prompt, false) else {
        return;
    };
    let moved = served_offloaded(&fixture, page, context, prompt, true).expect("loaded before");
    same(
        &format!("{model}: parked on the host and the disk against kept"),
        &moved,
        &kept,
    );
}

#[test]
fn a_dense_lane_parked_on_the_host_and_the_disk_serves_the_same_logits() {
    offloaded("qwen3-micro-text", None, PAGE, 256, 150);
}

#[test]
fn a_hybrid_lane_parked_on_the_host_and_the_disk_serves_the_same_logits() {
    offloaded("qwen35-tiny", None, PAGE, 256, 150);
}

/// The window list's trimmed slots stay trimmed; its held ones park.
#[test]
fn a_windowed_lane_parked_on_the_host_and_the_disk_serves_the_same_logits() {
    offloaded("gemma4-e4b-mini-l6", None, 32, 1024, 700);
}

/// The rs list's tokens a slot.
pub fn rs_tokens(facts: &ChunkFacts) -> u32 {
    match facts.shape(RS).map(|s| s.kind) {
        Some(SlotKind::List { tokens, .. }) => tokens,
        _ => panic!("the load buffers rs rows in an rs list"),
    }
}

/// Copies whole slots `(src, dst)` of `shape` on the device.
fn copy(shell: &mut Shell, shape: ShapeId, pairs: &[(u32, u32)]) {
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: DEVICE,
            dst: DEVICE,
            slots: pairs.to_vec(),
            runs: Vec::new(),
        })
        .expect("the slots copy");
}

/// `held`'s kv pages and record copied into fresh slots: a lane of its
/// own at the same rows.
fn cloned(shell: &mut Shell, facts: &ChunkFacts, refs: &mut Refs, held: &Held) -> Held {
    let mut twin = Held::new(facts, refs);
    twin.have = held.have;
    twin.pages = held.pages.iter().map(|_| refs.take(LIST)).collect();
    let pairs: Vec<(u32, u32)> = held
        .pages
        .iter()
        .copied()
        .zip(twin.pages.iter().copied())
        .collect();
    copy(shell, LIST, &pairs);
    shell
        .copy_state(
            &held
                .record
                .iter()
                .copied()
                .zip(twin.record.iter().copied())
                .collect::<Vec<_>>(),
        )
        .expect("the record copies");
    twin
}

/// CONTRACT-BC 15: a group of two lanes on one working set, one folding
/// its row and one buffering its row unfolded, names `[run, cow]` on both.
/// The folding lane writes the copy, the buffering lane reads the run side,
/// and the run side is left as it was: the logits and the states of two
/// ungrouped lanes on records of their own.
#[test]
fn a_group_writes_its_copy_and_its_reading_lane_reads_the_run() {
    const MODEL: &str = "qwen35-tiny";
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        256,
        None,
        engine::load::Residency::default(),
    ) else {
        return;
    };
    let facts = shell.pool_chunk_facts();
    let (first, end) = committed(&mut shell, &facts);
    let mut refs = Refs::of(&facts, first, end);
    let trace = &fixture.trace;
    let mut base = Held::new(&facts, &mut refs);
    let prompt: Vec<u32> = (0..150u32).map(|at| (at * 37 + 11) % 256).collect();
    let logits = feed(&mut shell, trace, &facts, &mut refs, &mut base, &prompt);
    let token = [argmax(&logits)];
    let other = [(token[0] + 1) % 256];
    let before = shell.state_bytes(&base.record).expect("the state reads");
    let t_rs = rs_tokens(&facts);
    let have = base.have;
    let rs_base = have - have % t_rs;

    // Fires the folding lane `fold` and the buffering lane `buffer`, the
    // group's records where stated.
    let fire = |shell: &mut Shell,
                refs: &mut Refs,
                fold: &mut Held,
                buffer: &mut Held,
                rs_page: u32,
                group: Option<(&[u32], &[u32])>| {
        for lane in [&mut *fold, &mut *buffer] {
            lane.grow(&facts, refs, have + 1);
        }
        let verb = RsVerb::Buffer {
            pages: vec![rs_page],
            base: rs_base,
            folded: have,
            fold: FoldLen::Host(0),
        };
        let lanes = [
            Seated {
                group: group.map(|_| 0),
                record: group.map(|(_, cow)| cow).or(Some(&fold.record[..])),
                record_run: group.map(|(run, _)| run),
                ..fold.seated(trace, &token, 0)
            },
            Seated {
                group: group.map(|_| 0),
                record: group.map(|(_, cow)| cow).or(Some(&buffer.record[..])),
                record_run: group.map(|(run, _)| run),
                rs: &verb,
                ..buffer.seated(trace, &other, 1)
            },
        ];
        shell.fire_seated(&lanes).expect("the fire lands")
    };

    // Ungrouped: each lane on a copy of the record of its own.
    let mut fold_alone = cloned(&mut shell, &facts, &mut refs, &base);
    let mut buffer_alone = cloned(&mut shell, &facts, &mut refs, &base);
    let rs_alone = refs.take(RS);
    let alone = fire(
        &mut shell,
        &mut refs,
        &mut fold_alone,
        &mut buffer_alone,
        rs_alone,
        None,
    );

    // Grouped: both name the base record as the run and one fresh copy.
    let mut fold_grouped = cloned(&mut shell, &facts, &mut refs, &base);
    let mut buffer_grouped = cloned(&mut shell, &facts, &mut refs, &base);
    let run = base.record.clone();
    let cow: Vec<u32> = run.iter().map(|_| refs.take(FIXED)).collect();
    shell
        .copy_state(
            &run.iter()
                .copied()
                .zip(cow.iter().copied())
                .collect::<Vec<_>>(),
        )
        .expect("the record copies on write");
    let rs_grouped = refs.take(RS);
    let grouped = fire(
        &mut shell,
        &mut refs,
        &mut fold_grouped,
        &mut buffer_grouped,
        rs_grouped,
        Some((&run, &cow)),
    );

    same(
        &format!("{MODEL}: a group's lanes against two lanes alone"),
        &grouped,
        &alone,
    );
    let folded = shell.state_bytes(&fold_alone.record).expect("reads");
    assert_eq!(
        shell.state_bytes(&cow).expect("reads"),
        folded,
        "the copy holds the folding lane's state"
    );
    assert_ne!(folded, before, "the fold moved the state");
    assert_eq!(
        shell.state_bytes(&run).expect("reads"),
        before,
        "the run side is as it was"
    );
    assert_eq!(
        shell.state_bytes(&buffer_alone.record).expect("reads"),
        before,
        "a buffering lane folds nothing"
    );
}

/// CONTRACT-BC 20: a fork shares its parent's rs-list slots; the tail
/// slot, partially filled, is copied before the fork writes its row there.
/// The fork lands its parent's logits for the same row, bit for bit, the
/// parent writing its own tail after.
#[test]
fn a_fork_writes_a_copy_of_the_rs_tail_slot() {
    const MODEL: &str = "qwen35-tiny";
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        256,
        None,
        engine::load::Residency::default(),
    ) else {
        return;
    };
    let facts = shell.pool_chunk_facts();
    let (first, end) = committed(&mut shell, &facts);
    let mut refs = Refs::of(&facts, first, end);
    let trace = &fixture.trace;
    let t_rs = rs_tokens(&facts);
    assert!(t_rs > 1, "{MODEL}: an rs slot holds more than one row");
    let mut parent = Held::new(&facts, &mut refs);
    let prompt: Vec<u32> = (0..150u32).map(|at| (at * 37 + 11) % 256).collect();
    let logits = feed(&mut shell, trace, &facts, &mut refs, &mut parent, &prompt);
    let folded = parent.have;
    let rs_base = folded - folded % t_rs;
    let mut rs_pages = vec![refs.take(RS)];
    // Buffers `token` on `lane` past `folded`, through `pages`.
    let buffer =
        |shell: &mut Shell, refs: &mut Refs, lane: &mut Held, pages: &[u32], token: u32| {
            let tokens = [token];
            lane.grow(&facts, refs, lane.have + 1);
            let verb = RsVerb::Buffer {
                pages: pages.to_vec(),
                base: rs_base,
                folded,
                fold: FoldLen::Host(0),
            };
            let seated = Seated {
                rs: &verb,
                ..lane.seated(trace, &tokens, 0)
            };
            let got = shell
                .fire_seated(&[seated])
                .expect("the fire lands")
                .swap_remove(0);
            lane.have += 1;
            got
        };
    // The parent buffers a row or two, the tail slot partly full.
    let mut token = argmax(&logits);
    let rows = (t_rs - 1 - folded % t_rs).clamp(1, 2);
    for _ in 0..rows {
        while (rs_base + (rs_pages.len() as u32) * t_rs) <= parent.have {
            rs_pages.push(refs.take(RS));
        }
        token = argmax(&buffer(
            &mut shell,
            &mut refs,
            &mut parent,
            &rs_pages,
            token,
        ));
    }
    assert!(
        !(parent.have - rs_base).is_multiple_of(t_rs),
        "the tail slot is partly full"
    );
    // The fork: its own record and kv pages, the rs list shared but for a
    // copy of the tail slot.
    let mut fork = cloned(&mut shell, &facts, &mut refs, &parent);
    let tail = *rs_pages.last().expect("a tail slot");
    let copied = refs.take(RS);
    shell
        .copy_slot_pairs(&[(RS, vec![[tail, copied]])])
        .expect("the tail slot copies");
    let mut fork_pages = rs_pages.clone();
    *fork_pages.last_mut().expect("a tail slot") = copied;
    let forked = buffer(&mut shell, &mut refs, &mut fork, &fork_pages, token);
    // The parent then writes its own tail, the same row.
    let parent_row = buffer(&mut shell, &mut refs, &mut parent, &rs_pages, token);
    same(
        &format!("{MODEL}: the fork's row against its parent's"),
        &[forked],
        &[parent_row],
    );
}

/// CONTRACT-BC 21: a fork shares its parent's window slots; the tail slot,
/// partially filled, is copied before the fork writes there. The fork
/// lands its parent's logits for the same token, bit for bit.
#[test]
fn a_fork_writes_a_copy_of_the_window_tail_slot() {
    const MODEL: &str = "gemma4-e4b-mini-l1";
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        1024,
        None,
        engine::load::Residency::default(),
    ) else {
        return;
    };
    let facts = shell.pool_chunk_facts();
    let (first, end) = committed(&mut shell, &facts);
    let mut refs = Refs::of(&facts, first, end);
    let trace = &fixture.trace;
    let mut parent = Held::new(&facts, &mut refs);
    let prompt: Vec<u32> = (0..700u32).map(|at| (at * 37 + 11) % 256).collect();
    let logits = feed(&mut shell, trace, &facts, &mut refs, &mut parent, &prompt);
    let token = [argmax(&logits)];
    // Trimmed as the runtime trims at the rows the prompt committed.
    let have = parent.have;
    parent.grow(&facts, &mut refs, have);
    let (id, table) = parent.windows.first().cloned().expect("a window list");
    assert!(table.contains(&0), "{MODEL}: the window list is trimmed");
    let Some(SlotKind::List { tokens, .. }) = facts.shape(id).map(|s| s.kind) else {
        panic!("the window list is a list");
    };
    assert!(
        !parent.have.is_multiple_of(tokens),
        "the tail slot is partly full"
    );
    // The fork: the parent's window slots by ref but a copy of the tail.
    let tail = *table.last().expect("a tail slot");
    let copied = refs.take(id);
    shell
        .copy_slot_pairs(&[(id, vec![[tail, copied]])])
        .expect("the tail slot copies");
    let mut fork = Held {
        windows: vec![(id, table.clone())],
        have: parent.have,
        ..Held::default()
    };
    *fork.windows[0].1.last_mut().expect("a tail slot") = copied;
    let forked = feed(&mut shell, trace, &facts, &mut refs, &mut fork, &token);
    let parent_row = feed(&mut shell, trace, &facts, &mut refs, &mut parent, &token);
    same(
        &format!("{MODEL}: the fork's row against its parent's"),
        &[forked],
        &[parent_row],
    );
}
