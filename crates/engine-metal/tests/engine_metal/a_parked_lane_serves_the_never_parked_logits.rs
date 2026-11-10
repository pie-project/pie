#![cfg(target_vendor = "apple")]

//! CONTRACT-BC 22 on device: a lane addressed by slot refs whose every slot
//! (kv list pages, window list pages, its record's pieces) parks below the
//! pool mid-decode and comes back to fresh slots in fresh chunks (the host
//! tier after step 2; the kv list pages through the slot file, the rest
//! through the host tier, after step 6) lands the never-parked lane's logits
//! bit for bit, in one process (as `engine-cuda`'s `golden_offload`). A
//! negative control parks the record and brings back another lane's.

use engine::chunk::{ShapeId, SlotCopy, SlotKind};
use engine::transfer::MemoryDomain;
use engine_metal::serve::Seated;
use engine_metal::store::layout::{FIXED, LIST, WINDOWS};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{
    Fixture, argmax, fixture_of, load_with, same, word,
};

const CONTEXT: u32 = 1024;
const PROMPT: u32 = 150;
const STEPS: usize = 10;

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

/// Parks `slots` (0: a trimmed slot, skipped) of `shape` to `tier` and
/// back to fresh slot refs: the fresh table.
#[allow(clippy::too_many_arguments)]
fn park(
    shell: &mut Shell,
    device: &mut Refs,
    host: &mut Refs,
    disk: &mut u32,
    tier: MemoryDomain,
    shape: ShapeId,
    per: u32,
    slots: &[u32],
) -> Vec<u32> {
    let out: Vec<(u32, u32)> = slots
        .iter()
        .filter(|&&slot| slot != 0)
        .map(|&slot| {
            let parked = if tier == MemoryDomain::LocalDisk {
                *disk += 1;
                *disk - 1
            } else {
                host.take(shape, per)
            };
            (slot, parked)
        })
        .collect();
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: MemoryDomain::MetalShared,
            dst: tier,
            slots: out.clone(),
            runs: Vec::new(),
        })
        .expect("the slots park");
    let fresh: Vec<u32> = slots
        .iter()
        .map(|&slot| {
            if slot == 0 {
                0
            } else {
                device.take(shape, per)
            }
        })
        .collect();
    let back: Vec<(u32, u32)> = out
        .iter()
        .zip(fresh.iter().filter(|&&slot| slot != 0))
        .map(|(&(_, parked), &slot)| (parked, slot))
        .collect();
    shell
        .copy_slots(&SlotCopy {
            shape,
            src: tier,
            dst: MemoryDomain::MetalShared,
            slots: back,
            runs: Vec::new(),
        })
        .expect("the slots come back");
    fresh
}

/// What the script does at its parking steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Offload {
    Never,
    Parked,
    /// The record comes back from another record's parked copy.
    Swapped,
}

/// A prompt, then greedy decodes on one lane by slot refs; every logit row.
fn serve(fixture: &Fixture, offload: Offload) -> Vec<Vec<f32>> {
    let mut shell = load_with(fixture, CONTEXT);
    shell.seat_tier(64 << 20);
    let dir = std::env::temp_dir().join(format!(
        "pie-bm-pool-{}-tiers-{offload:?}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("a disk tier's dir");
    shell
        .seat_disk_kv(Some(64 << 20), Some(&dir), 1, CONTEXT)
        .expect("the disk tier seats");
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    assert!(facts.host_chunks > 0, "a host tier is stated");
    let per = |id: ShapeId| facts.shape(id).map(|s| s.slots_per_chunk);
    let list = per(LIST);
    let fixed = per(FIXED);
    let windows: Vec<(ShapeId, u32, u32, u32)> = WINDOWS
        .iter()
        .filter_map(|&id| {
            let shape = facts.shape(id)?;
            let SlotKind::List {
                tokens,
                window: Some(rows),
                ..
            } = shape.kind
            else {
                return None;
            };
            Some((id, shape.slots_per_chunk, tokens, rows))
        })
        .collect();
    assert!(
        list.is_none() || facts.disk_chunks > 0,
        "a disk tier is stated for the kv list"
    );
    let layout = shell.pool_layout();
    let first = (1 + layout.seat_chunks()).div_ceil(facts.segment_chunks) * facts.segment_chunks;
    shell
        .commit_segments((first + 64).div_ceil(facts.segment_chunks))
        .expect("the segments commit");
    let mut device = Refs {
        next: first,
        open: Vec::new(),
    };
    let mut host = Refs {
        next: 0,
        open: Vec::new(),
    };
    let mut disk = 0u32;
    let mut record: Vec<u32> = fixed.map_or_else(Vec::new, |per| {
        (0..layout.pieces)
            .map(|_| device.take(FIXED, per))
            .collect()
    });
    // A second record, filled by a lane of its own, for the negative control.
    let other: Vec<u32> = fixed.map_or_else(Vec::new, |per| {
        (0..layout.pieces)
            .map(|_| device.take(FIXED, per))
            .collect()
    });
    let page = shell.paging().page_size;
    let mut pages: Vec<u32> = Vec::new();
    let mut tables: Vec<Vec<u32>> = vec![Vec::new(); windows.len()];
    let mut held = 0u32;
    let mut fed: Vec<u32> = (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect();
    let mut rows = Vec::new();
    if offload == Offload::Swapped && !other.is_empty() {
        let tokens: Vec<u32> = (0..40u32).map(|at| (at * 11 + 3) % 256).collect();
        let own_pages: Vec<u32> = match list {
            Some(per) => (0..40u32.div_ceil(page))
                .map(|_| device.take(LIST, per))
                .collect(),
            None => Vec::new(),
        };
        let own_windows: Vec<Vec<u32>> = windows
            .iter()
            .map(|&(id, per, tokens_w, _)| {
                (0..40u32.div_ceil(tokens_w))
                    .map(|_| device.take(id, per))
                    .collect()
            })
            .collect();
        let seated = Seated {
            held: Some(0),
            refs: true,
            pages: &own_pages,
            windows: &own_windows,
            record: Some(&other),
            ..Seated::of(Lane {
                slot: 1,
                word: word(&fixture.trace, 40),
                tokens: &tokens,
            })
        };
        shell.fire_seated(&[seated]).expect("the other lane fires");
    }
    for step in 0..STEPS {
        let n = fed.len() as u32;
        let after = held + n;
        if let Some(per) = list {
            while (pages.len() as u32) < after.div_ceil(page) {
                pages.push(device.take(LIST, per));
            }
        }
        for (table, &(id, per, tokens, _)) in tables.iter_mut().zip(&windows) {
            while (table.len() as u32) < after.div_ceil(tokens) {
                table.push(device.take(id, per));
            }
        }
        let seated = Seated {
            held: Some(held),
            refs: true,
            pages: &pages,
            windows: &tables,
            record: (!record.is_empty()).then_some(record.as_slice()),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, n),
                tokens: &fed,
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the fire lands");
        held = after;
        fed = vec![argmax(&got[0])];
        rows.extend(got);
        if offload != Offload::Never && (step == 2 || step == 6) {
            let kv_tier = if step == 2 {
                MemoryDomain::HostPinned
            } else {
                MemoryDomain::LocalDisk
            };
            if let Some(per) = list {
                pages = park(
                    &mut shell,
                    &mut device,
                    &mut host,
                    &mut disk,
                    kv_tier,
                    LIST,
                    per,
                    &pages,
                );
            }
            for (table, &(id, per, _, _)) in tables.iter_mut().zip(&windows) {
                *table = park(
                    &mut shell,
                    &mut device,
                    &mut host,
                    &mut disk,
                    MemoryDomain::HostPinned,
                    id,
                    per,
                    table,
                );
            }
            if let Some(per) = fixed {
                let parked = if offload == Offload::Swapped && step == 2 {
                    &other
                } else {
                    &record
                };
                record = park(
                    &mut shell,
                    &mut device,
                    &mut host,
                    &mut disk,
                    MemoryDomain::HostPinned,
                    FIXED,
                    per,
                    parked,
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    rows
}

fn check(model: &str) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of(model, None);
    let kept = serve(&fixture, Offload::Never);
    let parked = serve(&fixture, Offload::Parked);
    same(&format!("{model}: parked against kept"), &parked, &kept);
    eprintln!("{model}: parked = kept over {} rows", kept.len());
    if fixture
        .trace
        .caches
        .iter()
        .any(|row| matches!(row, poem_ir::CacheRow::State { .. }))
    {
        let swapped = serve(&fixture, Offload::Swapped);
        assert_ne!(
            swapped, kept,
            "{model}: another lane's record brought back moves the logits"
        );
        eprintln!("{model}: a swapped record moves the logits (the control)");
    }
}

#[test]
fn a_parked_dense_lane_serves_the_never_parked_logits() {
    check("qwen3-micro-text");
}

#[test]
fn a_parked_windowed_lane_serves_the_never_parked_logits() {
    check("gemma4-e4b-mini-l1");
}

#[test]
fn a_parked_hybrid_lane_serves_the_never_parked_logits() {
    check("qwen35-tiny");
}
