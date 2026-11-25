#![cfg(target_vendor = "apple")]

//! CONTRACT-BC 22 on device: a lane addressed by slot refs whose every slot
//! (kv list pages, window buffer pages, its ring's and record's pieces) parks in the slot
//! file mid-decode, twice, and comes back to fresh slots in fresh chunks
//! lands the never-parked lane's logits bit for bit. A negative control
//! a kind brings back another lane's ring or record, or the kv or window
//! pages in another order.

use std::collections::BTreeMap;

use engine::chunk::{ShapeId, SlotCopy};
use engine::fire::{FoldLen, Lineage, WindowFold};
use engine::transfer::{DeviceReach, MemoryDomain};
use engine_metal::serve::Seated;
use engine_metal::store::layout::{FIXED, LIST, RING, WINDOWS};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{
    Fixture, argmax, fixture_of, load_with, same, word,
};

const CONTEXT: u32 = 1024;
const PROMPT: u32 = 156;
const STEPS: usize = 10;
/// The steps after which every slot parks and comes back.
const PARKS: [usize; 2] = [2, 6];
/// Rows a windowed lane's fold trails its decodes by: its buffer slots are
/// live where it parks.
const LAG: u32 = 7;

/// Slot refs a chunk at a time from `next`, each shape its own chunks.
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

/// Parks `slots` (0: a trimmed slot, skipped) of `shape` to the slot file
/// and back to fresh slot refs: the fresh table. `crossed` brings each slot
/// back as the next one's.
fn park(
    shell: &mut Shell,
    device: &mut Refs,
    disk: &mut Refs,
    shape: ShapeId,
    per: u32,
    slots: &[u32],
    crossed: bool,
) -> Vec<u32> {
    let here = MemoryDomain::Device {
        ordinal: 0,
        reach: DeviceReach::HostShared,
    };
    let out: Vec<(u32, u32)> = slots
        .iter()
        .filter(|&&slot| slot != 0)
        .map(|&slot| (slot, disk.take(shape, per)))
        .collect();
    let mut away: Vec<u32> = out.iter().map(|&(_, parked)| parked).collect();
    if crossed && !away.is_empty() {
        away.rotate_left(1);
    }
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: here,
            dst: MemoryDomain::LocalDisk,
            slots: out,
            runs: Vec::new(),
        })
        .expect("the slots park");
    let fresh: Vec<u32> = slots
        .iter()
        .map(|&slot| if slot == 0 { 0 } else { device.take(shape, per) })
        .collect();
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: MemoryDomain::LocalDisk,
            dst: here,
            slots: away
                .into_iter()
                .zip(fresh.iter().copied().filter(|&slot| slot != 0))
                .collect(),
            runs: Vec::new(),
        })
        .expect("the slots come back");
    fresh
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    List,
    Window,
    Ring,
    Record,
}

/// What the script does at its parking steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Offload {
    Never,
    Parked,
    /// One kind of slot comes back wrong: the kv or the window pages in
    /// another order, or another lane's ring or record.
    Wrong(Kind),
}

/// How a windowed lane's fire folds: to row `after` of `end`, from `folded`.
fn fold(folded: u32, have: u32, after: u32, page: u32) -> WindowFold {
    WindowFold {
        base: if folded < have { folded } else { after } / page * page,
        fold: FoldLen::Host(after - folded),
        lineage: Lineage::Set,
    }
}

/// A prompt, then greedy decodes on one lane by slot refs, a windowed
/// lane's fold trailing its decodes by [`LAG`] rows; every logit row, and
/// the kinds of slot the lane holds.
fn serve(fixture: &Fixture, offload: Offload) -> (Vec<Vec<f32>>, Vec<Kind>) {
    let mut shell = load_with(fixture, CONTEXT);
    let dir = std::env::temp_dir().join(format!(
        "pie-bm-pool-{}-disk-{offload:?}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("a disk tier's dir");
    shell
        .seat_disk_kv(Some(1 << 30), Some(&dir), 1, CONTEXT)
        .expect("the disk tier seats");
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    assert!(facts.disk_chunks > 0, "a disk tier is stated");
    let per = |id: ShapeId| facts.shape(id).map(|s| s.slots_per_chunk.max(1));
    let (list, fixed) = (per(LIST), per(FIXED));
    let layout = shell.pool_layout();
    let ring_pieces = shell.pool_ring().map_or(0, |ring| ring.pieces);
    let buffered = (ring_pieces > 0).then(|| {
        let window = layout.window(0).expect("a ring load's window buffer");
        (window.tokens, per(WINDOWS[0]).expect("its shape"))
    });
    let first = layout.free_chunk().div_ceil(facts.segment_chunks) * facts.segment_chunks;
    shell
        .commit_segments((first + 64).div_ceil(facts.segment_chunks))
        .expect("the segments commit");
    let mut device = Refs {
        next: first,
        open: Vec::new(),
    };
    let mut disk = Refs {
        next: 0,
        open: Vec::new(),
    };
    let own = |device: &mut Refs| -> (Vec<u32>, Vec<u32>) {
        (
            fixed.map_or_else(Vec::new, |per| {
                (0..layout.pieces).map(|_| device.take(FIXED, per)).collect()
            }),
            (0..ring_pieces).map(|_| device.take(RING, 1)).collect(),
        )
    };
    let (mut record, mut ring) = own(&mut device);
    let page = shell.paging().page_size;
    let cow = |ring: &[u32]| -> Vec<[u32; 2]> { ring.iter().map(|&p| [p, p]).collect() };

    // Another lane's record and ring, for the control.
    let (other_record, other_ring) = own(&mut device);
    if matches!(offload, Offload::Wrong(Kind::Record | Kind::Ring)) {
        let tokens: Vec<u32> = (0..40u32).map(|at| (at * 11 + 3) % 256).collect();
        let pages: Vec<u32> = list.map_or_else(Vec::new, |per| {
            (0..40u32.div_ceil(page)).map(|_| device.take(LIST, per)).collect()
        });
        let rings = cow(&other_ring);
        let seated = Seated {
            held: Some(0),
            refs: true,
            pages: &pages,
            record: (!other_record.is_empty()).then_some(other_record.as_slice()),
            ring: buffered.map(|_| rings.as_slice()),
            window: buffered.map(|(page, _)| fold(0, 0, 40, page)),
            ..Seated::of(Lane {
                slot: 1,
                word: word(&fixture.trace, 40),
                tokens: &tokens,
            })
        };
        shell.fire_seated(&[seated]).expect("the other lane fires");
    }

    let mut pages: Vec<u32> = Vec::new();
    // The buffer's slots by window page.
    let mut buffer: BTreeMap<u32, u32> = BTreeMap::new();
    let (mut held, mut folded) = (0u32, 0u32);
    let mut fed: Vec<u32> = (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect();
    let mut rows = Vec::new();
    let mut kinds = Vec::new();
    for step in 0..STEPS {
        let n = fed.len() as u32;
        let end = held + n;
        if let Some(per) = list {
            while (pages.len() as u32) < end.div_ceil(page) {
                pages.push(device.take(LIST, per));
            }
        }
        let after = if step == 0 { end } else { end.saturating_sub(LAG).max(folded) };
        let rings = cow(&ring);
        let mut tables = Vec::new();
        let mut window = None;
        if let Some((page, per)) = buffered {
            let plan = fold(folded, held, after, page);
            buffer.retain(|&at, _| at >= plan.base / page);
            let table: Vec<u32> = if folded < held || after < end {
                (plan.base / page..end.div_ceil(page))
                    .map(|at| *buffer.entry(at).or_insert_with(|| device.take(WINDOWS[0], per)))
                    .collect()
            } else {
                Vec::new()
            };
            tables.push(table);
            window = Some(plan);
        }
        let seated = Seated {
            held: Some(held),
            folded: if buffered.is_some() { folded } else { 0 },
            refs: true,
            pages: &pages,
            windows: &tables,
            record: (!record.is_empty()).then_some(record.as_slice()),
            ring: buffered.map(|_| rings.as_slice()),
            window,
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, n),
                tokens: &fed,
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the fire lands");
        (held, folded) = (end, after);
        fed = vec![argmax(&got[0])];
        rows.extend(got);
        if offload == Offload::Never || !PARKS.contains(&step) {
            continue;
        }
        let wrong = |kind| step == PARKS[1] && offload == Offload::Wrong(kind);
        kinds.clear();
        if let Some(per) = list {
            let crossed = wrong(Kind::List);
            pages = park(&mut shell, &mut device, &mut disk, LIST, per, &pages, crossed);
            kinds.push(Kind::List);
        }
        if let Some((_, per)) = buffered {
            let slots: Vec<u32> = buffer.values().copied().collect();
            assert!(step != PARKS[1] || slots.len() > 1, "the buffer holds pages to cross");
            let crossed = wrong(Kind::Window);
            let fresh = park(&mut shell, &mut device, &mut disk, WINDOWS[0], per, &slots, crossed);
            for (slot, at) in buffer.values_mut().zip(fresh) {
                *slot = at;
            }
            let from = if wrong(Kind::Ring) { &other_ring } else { &ring };
            ring = park(&mut shell, &mut device, &mut disk, RING, 1, from, false);
            kinds.extend([Kind::Window, Kind::Ring]);
        }
        if let Some(per) = fixed {
            let from = if wrong(Kind::Record) { &other_record } else { &record };
            record = park(&mut shell, &mut device, &mut disk, FIXED, per, from, false);
            // A record with no recurrent state holds nothing a fire reads.
            if fixture.trace.caches.iter().any(|row| matches!(row, poem_ir::CacheRow::State { .. })) {
                kinds.push(Kind::Record);
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    (rows, kinds)
}

fn check(model: &str) -> Vec<Kind> {
    let fixture = fixture_of(model, None);
    let (kept, _) = serve(&fixture, Offload::Never);
    let (parked, kinds) = serve(&fixture, Offload::Parked);
    same(&format!("{model}: parked against kept"), &parked, &kept);
    eprintln!("{model}: parked = kept over {} rows", kept.len());
    for kind in &kinds {
        let (wrong, _) = serve(&fixture, Offload::Wrong(*kind));
        assert_ne!(wrong, kept, "{model}: {kind:?} slots brought back wrong move the logits");
    }
    eprintln!("{model}: each of {kinds:?} brought back wrong moves the logits");
    kinds
}

/// A kv list, a window's ring and buffer, and a record beside a kv list: every kind parks.
#[test]
fn a_lane_parked_to_disk_serves_the_never_parked_logits() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let parked: Vec<Kind> = ["qwen3-micro-text", "gemma4-e4b-mini-l1", "qwen35-tiny"]
        .into_iter()
        .flat_map(check)
        .collect();
    for kind in [Kind::List, Kind::Window, Kind::Ring, Kind::Record] {
        assert!(parked.contains(&kind), "no model here parks {kind:?} slots");
    }
}

/// On a load that lays rings, a lane addressed by slot refs that states
/// neither its ring nor a window fold is refused by name, outside the
/// load's own pipeline walk.
#[test]
fn a_ref_lane_stating_no_ring_is_refused() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("gemma4-e4b-mini-l1", None);
    let mut shell = load_with(&fixture, CONTEXT);
    assert!(shell.pool_ring().is_some(), "a windowed load lays rings");
    let layout = shell.pool_layout();
    let per = shell.pool_chunk_facts().shape(FIXED).map(|s| s.slots_per_chunk.max(1));
    let low = layout.free_chunk();
    let record: Vec<u32> = per.map_or_else(Vec::new, |per| {
        (0..layout.pieces).map(|piece| (low + piece) * per).collect()
    });
    let segment = shell.pool_chunk_facts().segment_chunks;
    shell
        .commit_segments((low + layout.pieces + 1).div_ceil(segment))
        .expect("the segments commit");
    let why = shell
        .fire_seated(&[Seated {
            held: Some(0),
            refs: true,
            record: (!record.is_empty()).then_some(record.as_slice()),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, 4),
                tokens: &[1, 2, 3, 4],
            })
        }])
        .expect_err("no page it would read is its own")
        .to_string();
    assert!(why.contains("states no ring pieces and no window fold"), "{why}");
    shell
        .warm_pipelines()
        .expect("the pipeline walk, whose fires are dropped, states neither");
}
