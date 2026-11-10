//! A host simulation of what a fire does to the pool: the slots a fire's
//! lanes name are staged into its planes, the "fire" reads them through
//! the local tables and writes its rows there, and what it wrote lands back
//! in the slots. The reference is the pool updated in place, slot by slot,
//! as the planes were before the pool switch (one plane a pool, pages
//! addressed by id): every read and every slot after the fire must match it
//! word for word.

use std::collections::HashMap;

use engine::chunk::ShapeId;
use poem_ir::Trace;

use super::super::pool::{FIXED, LIST, RS, WINDOW};
use super::*;
use crate::device::{Buffer, Device};
use crate::store::Pools;
use crate::store::kv::Paging;
use crate::trace::Handles;

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;
const LANES: u32 = 4;

fn trace(name: &str) -> Trace {
    poem_compiler::catalog::deployment(name)
        .expect("the deployment is listed")
        .trace(poem::Platform::Cerebras)
}

fn pools(trace: &Trace) -> Pools {
    pools_at(trace, crate::store::pool::TARGET_CHUNK_BYTES)
}

/// The pools of `trace` with chunks of up to `target` bytes: 1 cuts every
/// record larger than a list slot into pieces.
fn pools_at(trace: &Trace, target: u64) -> Pools {
    let window = crate::store::window_of(trace).expect("the windows agree");
    let paging = Paging::of(PAGE, CONTEXT, LANES, u64::from(LANES * CONTEXT / PAGE))
        .expect("the paging lays out")
        .windowed(window, 64);
    let facts = crate::store::kv::probe(trace).expect("the rows probe");
    let rs = crate::rs::Layout::read(trace).expect("the rs layout reads");
    Pools::reserve_at(
        &Device::host_only(),
        &Handles::new(),
        (trace, paging, LANES),
        &facts,
        rs.as_ref(),
        target,
    )
    .expect("the pools reserve")
}

/// The slots as they would stand in place: what the pool holds of every
/// slot a test names, kept apart from it.
#[derive(Default)]
struct InPlace(HashMap<(ShapeId, u32), Vec<u32>>);

fn word(seed: u32, at: usize) -> u32 {
    seed.wrapping_mul(2_654_435_761) ^ (at as u32).wrapping_mul(40_503)
}

impl InPlace {
    /// Fills slot `slot` of `id` with words of `seed`, in both.
    fn fill(&mut self, pools: &mut Pools, id: ShapeId, slot: u32, seed: u32) {
        let words = pools
            .pool_mut()
            .slot_mut(id, slot)
            .expect("the slot is in the pool")
            .expect("a host pool");
        for (at, w) in words.iter_mut().enumerate() {
            *w = word(seed, at);
        }
        self.0.insert((id, slot), words.to_vec());
    }

    /// Every slot named matches the pool's.
    fn agrees(&self, pools: &Pools) {
        for (&(id, slot), want) in &self.0 {
            let have = pools.pool().slot(id, slot).unwrap().unwrap();
            assert!(have == want.as_slice(), "slot {slot} of shape {id}");
        }
    }

    /// A record's words, whole, from its pieces.
    fn record(&self, pools: &Pools, pieces: &[u32]) -> Vec<u32> {
        let shape = pools.pool().shape(FIXED).unwrap();
        let mut out: Vec<u32> = pieces
            .iter()
            .flat_map(|&p| self.0[&(FIXED, p)].clone())
            .collect();
        out.truncate(shape.record_words as usize);
        out
    }

    fn set_record(&mut self, pools: &Pools, pieces: &[u32], words: &[u32]) {
        let piece = pools.pool().shape(FIXED).unwrap().slot_words as usize;
        for (at, &p) in pieces.iter().enumerate() {
            let from = (at * piece).min(words.len());
            let take = piece.min(words.len() - from);
            self.0.get_mut(&(FIXED, p)).unwrap()[..take].copy_from_slice(&words[from..from + take]);
        }
    }
}

/// Rewrites plane `plane` of pool row `row` through `f`.
fn write_plane(pools: &mut Pools, row: u32, plane: u8, f: impl FnOnce(&mut [u32])) {
    let buffer = pools.take(row, plane).expect("the plane is home");
    let (rows, width) = buffer.shape();
    let mut words = buffer.words().to_vec();
    f(&mut words);
    pools.put(
        row,
        plane,
        Buffer::new(buffer.dtype(), rows, width, words).expect("a plain dtype"),
    );
}

/// One lane of list shape `id`: its slot refs in token order, the rows it
/// holds and the rows the fire adds.
struct ListLane {
    refs: Vec<u32>,
    have: u32,
    rows: u32,
}

/// Reads every row a lane holds after the fire through its local table
/// (what the attention reads), checks it against the slot in place, then
/// writes the fire's rows into the staged planes and into the slot in
/// place alike.
fn read_and_write(
    pools: &mut Pools,
    place: &mut InPlace,
    stage: &mut Stage,
    id: ShapeId,
    lanes: &[ListLane],
    seed: u32,
) {
    let shape = pools.pool().shape(id).unwrap().clone();
    let t = shape.tokens as usize;
    for (at, lane) in lanes.iter().enumerate() {
        let after = (lane.have + lane.rows) as usize;
        let local: Vec<u32> = lane
            .refs
            .iter()
            .map(|&slot| match id {
                WINDOW if slot == 0 => 0,
                WINDOW => stage.window.of(slot).unwrap(),
                RS => stage.rs.of(slot).unwrap(),
                _ => stage.kv.of(slot).unwrap(),
            })
            .collect();
        for (section, first, _) in shape.sections() {
            let w = section.width as usize;
            let plane = pools
                .get(section.row, section.plane)
                .unwrap()
                .words()
                .to_vec();
            for pos in 0..after {
                let (page, row) = (pos / t, pos % t);
                let staged = &plane[(local[page] as usize * t + row) * w..][..w];
                if lane.refs[page] == 0 {
                    assert!(staged.iter().all(|&w| w == 0), "the null page reads zeros");
                    continue;
                }
                let want = &place.0[&(id, lane.refs[page])][first as usize + row * w..][..w];
                assert_eq!(staged, want, "lane {at} row {pos} of shape {id}");
            }
            write_plane(pools, section.row, section.plane, |words| {
                for pos in lane.have as usize..after {
                    let (page, row) = (pos / t, pos % t);
                    for c in 0..w {
                        words[(local[page] as usize * t + row) * w + c] = word(seed + at as u32, c);
                    }
                }
            });
            for pos in lane.have as usize..after {
                let (page, row) = (pos / t, pos % t);
                let slot = place.0.get_mut(&(id, lane.refs[page])).unwrap();
                for c in 0..w {
                    slot[first as usize + row * w + c] = word(seed + at as u32, c);
                }
            }
        }
    }
}

/// `n` distinct refs of shape `id`, from chunk `from` on.
fn refs(pools: &Pools, id: ShapeId, from: u32, n: u32) -> Vec<u32> {
    let per = pools.pool().shape(id).unwrap().per_chunk;
    (0..n).map(|i| from * per + i).collect()
}

#[test]
fn lanes_sharing_a_prefix_read_and_write_as_in_place() {
    let trace = trace("qwen3-30b-a3b-mini-bf16-kv-bf16");
    let mut pools = pools(&trace);
    let mut place = InPlace::default();
    let kv = refs(&pools, LIST, 2, 5);
    for (at, &slot) in kv.iter().enumerate() {
        place.fill(&mut pools, LIST, slot, 100 + at as u32);
    }
    // Lane 0 holds 35 rows and adds 5; lane 1 shares its first two pages,
    // holds 33 and adds 1 into its own third page.
    let lanes = [
        ListLane {
            refs: vec![kv[0], kv[1], kv[2]],
            have: 35,
            rows: 5,
        },
        ListLane {
            refs: vec![kv[0], kv[1], kv[3]],
            have: 33,
            rows: 1,
        },
    ];
    let mut stage = pools.begin();
    for lane in &lanes {
        for &slot in &lane.refs {
            stage.kv.of(slot).unwrap();
        }
    }
    // A kv-less lane's scratch page reads zeros and is kept nowhere.
    let scratch = stage.kv.scratch().unwrap();
    pools.stage(&stage).unwrap();
    let list = pools.pool().shape(LIST).unwrap().clone();
    for (section, ..) in list.sections() {
        let words = pools.get(section.row, section.plane).unwrap().words();
        let w = (PAGE * section.width) as usize;
        assert!(words[scratch as usize * w..][..w].iter().all(|&x| x == 0));
    }
    read_and_write(&mut pools, &mut place, &mut stage, LIST, &lanes, 7);
    pools.unstage(&stage).unwrap();
    place.agrees(&pools);
    assert_eq!(stage.kv.staged().count(), 5, "a shared page is staged once");
}

#[test]
fn records_pieces_and_rs_pages_land_as_in_place() {
    let trace = trace("qwen38-flash-next-mini-bf16-kv-bf16");
    for target in [crate::store::pool::TARGET_CHUNK_BYTES, 1] {
        records_land_as_in_place(&trace, target);
    }
}

fn records_land_as_in_place(trace: &Trace, target: u64) {
    let mut pools = pools_at(trace, target);
    let mut place = InPlace::default();
    let pieces = pools.pool().shape(FIXED).unwrap().pieces();
    eprintln!("chunks of up to {target} bytes: {pieces} piece(s) a record");
    if target == 1 {
        assert!(pieces > 1, "a record larger than a list slot is cut");
    }
    let fixed = refs(&pools, FIXED, 3, 3 * pieces);
    for (at, &slot) in fixed.iter().enumerate() {
        place.fill(&mut pools, FIXED, slot, 300 + at as u32);
    }
    let (cow, run, other) = (
        fixed[..pieces as usize].to_vec(),
        fixed[pieces as usize..2 * pieces as usize].to_vec(),
        fixed[2 * pieces as usize..].to_vec(),
    );
    let mut stage = pools.begin();
    // A group's folding lane writes `cow`; its lane that folds nothing
    // reads `run` (CONTRACT-BC 15); another lane writes its own record.
    let at_cow = stage.records.of(&cow, true).unwrap();
    let at_run = stage.records.of(&run, false).unwrap();
    let at_other = stage.records.of(&other, true).unwrap();
    assert_eq!(stage.records.of(&cow, true).unwrap(), at_cow);
    let has_rs = pools.pool().shape(RS).is_ok();
    assert!(has_rs, "the model buffers rs rows");
    let rs = if has_rs {
        // The rs list's chunk lies past every chunk the records take.
        let per = pools.pool().shape(FIXED).unwrap().per_chunk;
        let rs = refs(&pools, RS, 4 + (3 * pieces).div_ceil(per), 3);
        for (at, &slot) in rs.iter().enumerate() {
            place.fill(&mut pools, RS, slot, 500 + at as u32);
        }
        rs
    } else {
        Vec::new()
    };
    let rs_lanes = if has_rs {
        vec![
            ListLane {
                refs: vec![rs[0], rs[1]],
                have: 3,
                rows: 18,
            },
            ListLane {
                refs: vec![rs[2]],
                have: 0,
                rows: 2,
            },
        ]
    } else {
        Vec::new()
    };
    for lane in &rs_lanes {
        for &slot in &lane.refs {
            stage.rs.of(slot).unwrap();
        }
    }
    pools.stage(&stage).unwrap();
    let shape = pools.pool().shape(FIXED).unwrap().clone();
    for (local, record) in [(at_cow, &cow), (at_run, &run), (at_other, &other)] {
        let whole = place.record(&pools, record);
        for (section, first, words) in shape.sections() {
            let slab = pools.get(section.row, section.plane).unwrap().words();
            let w = words as usize;
            assert_eq!(
                &slab[local as usize * w..][..w],
                &whole[first as usize..][..w],
                "record {local}, row {}",
                section.row
            );
        }
    }
    // The fire folds new state into `cow` and `other`, and writes over the
    // `run` it only read.
    for (local, record, seed) in [(at_cow, &cow, 11u32), (at_other, &other, 13)] {
        let mut whole = place.record(&pools, record);
        for (section, first, words) in shape.sections() {
            let w = words as usize;
            write_plane(&mut pools, section.row, section.plane, |slab| {
                for c in 0..w {
                    slab[local as usize * w + c] = word(seed + section.row, c);
                }
            });
            for c in 0..w {
                whole[first as usize + c] = word(seed + section.row, c);
            }
        }
        place.set_record(&pools, record, &whole);
    }
    for (section, _, words) in shape.sections() {
        let w = words as usize;
        write_plane(&mut pools, section.row, section.plane, |slab| {
            slab[at_run as usize * w..][..w].fill(0xdead);
        });
    }
    if has_rs {
        read_and_write(&mut pools, &mut place, &mut stage, RS, &rs_lanes, 21);
    }
    pools.unstage(&stage).unwrap();
    place.agrees(&pools);
}

#[test]
fn a_window_list_reads_its_trimmed_slots_as_the_null_page() {
    let trace = trace("gemma4-e4b-mini-l6-bf16-kv-bf16");
    let mut pools = pools(&trace);
    assert!(
        pools.pool().shape(WINDOW).is_ok(),
        "the window is shorter than the context"
    );
    let mut place = InPlace::default();
    let window = refs(&pools, WINDOW, 2, 3);
    for (at, &slot) in window.iter().enumerate() {
        place.fill(&mut pools, WINDOW, slot, 700 + at as u32);
    }
    // The first two pages are trimmed (ref 0); the window reads the rest.
    let lanes = [ListLane {
        refs: vec![0, 0, window[0], window[1], window[2]],
        have: 70,
        rows: 4,
    }];
    let mut stage = pools.begin();
    // A page dirtied by an earlier fire's null page still reads zeros.
    let shape = pools.pool().shape(WINDOW).unwrap().clone();
    for (section, ..) in shape.sections() {
        write_plane(&mut pools, section.row, section.plane, |words| {
            words.fill(5)
        });
    }
    for &slot in lanes[0].refs.iter().filter(|&&slot| slot != 0) {
        stage.window.of(slot).unwrap();
    }
    pools.stage(&stage).unwrap();
    read_and_write(&mut pools, &mut place, &mut stage, WINDOW, &lanes, 31);
    pools.unstage(&stage).unwrap();
    place.agrees(&pools);
}

#[test]
fn a_fire_naming_more_than_it_stages_is_refused() {
    let trace = trace("qwen3-30b-a3b-mini-bf16-kv-bf16");
    let pools = pools(&trace);
    let capacity = pools.capacity();
    let mut stage = pools.begin();
    let kv = refs(&pools, LIST, 1, capacity.kv_pages + 1);
    for &slot in &kv[..capacity.kv_pages as usize] {
        stage.kv.of(slot).unwrap();
    }
    assert!(stage.kv.of(kv[capacity.kv_pages as usize]).is_err());
    assert!(stage.kv.of(kv[0]).is_ok(), "a staged ref stages again");
}
