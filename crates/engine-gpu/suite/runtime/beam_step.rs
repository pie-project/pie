//! A beam group's head step and page copies on the first device against the
//! host replay (`engine::beam::step`, `BEAM-PAGE-LISTS.md` §9): widths 2, 4
//! and 8, from a `folded` with a tail and one without, `3S` fires and more
//! with forks (repeated parents) and retirements (beams no child
//! continues), the pool growing a column at a time. After every fire the
//! device's lists (columns `<= c`), free and pending pages, the faults, the
//! resolved table entries and write pages and the pool pages its copies
//! moved equal the replay's. Halfway, the state is moved to other words of
//! the pool (a park and restore into fresh slots) and the search goes on as
//! if it had not moved. A parent past the beams faults that lane, and only
//! it.

use engine::beam::{Fire, State, fault, step};
use engine_gpu::guest::state::BeamLayout;
use kernels_slang::runtime::beam::{self as kb, Records};
use kernels_slang::{Segments, Tensor};
use poem_ir::Dtype;

use crate::gpu::Gpu;

const PAGE: u32 = 16;
const FRAMES: u32 = 2;
/// The pool words a page id counts, and its two planes: 16 slots of one
/// word from word 0, 16 of one word from word 16.
const UNIT: u32 = 32;
const PLANES: [[u32; 2]; 2] = [[0, 1], [16, 1]];

/// A tiny deterministic generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// Where the test lays its store's parts, in words.
struct Lay {
    steps: u32,
    lists: u32,
    copies: u32,
    planes: u32,
    trans: u32,
    parents: u32,
    faults: u32,
    tables: u32,
    writes: u32,
    offsets: u32,
    words: u32,
}

impl Lay {
    fn of(lanes: u32, cap: u32, columns: u32) -> Lay {
        let mut at = 0u32;
        let mut take = |words: u32| {
            let here = at;
            at += words;
            here
        };
        let steps = take(kb::STEP_WORDS);
        let lists = take(lanes * kb::LIST_WORDS);
        let copies = take(lanes * kb::COPY_WORDS);
        let planes = take(2 * PLANES.len() as u32);
        let trans = take(cap);
        let parents = take(lanes);
        let faults = take(lanes);
        let tables = take(lanes * columns);
        let writes = take(lanes);
        let offsets = take(lanes);
        Lay {
            steps,
            lists,
            copies,
            planes,
            trans,
            parents,
            faults,
            tables,
            writes,
            offsets,
            words: at,
        }
    }
}

/// A word of a pool page's content before any copy.
fn content(id: u32, word: u32) -> u32 {
    id * 1000 + word
}

fn run(gpu: &mut Gpu, lanes: u32, folded: u32, seed: u64) {
    let fires = 3 * PAGE + 9;
    let end = folded + fires;
    let layout = BeamLayout::of(lanes, end, PAGE, FRAMES).expect("the layout");
    let cap = layout.cap;
    let state_words = kb::state_words(lanes, cap, FRAMES) as u32;
    let f0 = folded / PAGE;
    let columns = f0 + end.div_ceil(PAGE) + 1;
    let lay = Lay::of(lanes, cap, columns);
    // The pool: two places a state may lie, then the pages.
    let states = [0u32, state_words.next_multiple_of(UNIT)];
    let base = (states[1] + state_words).next_multiple_of(UNIT);
    let id_of = |q: u32| base / UNIT + q;
    let mut pool: Vec<u32> = vec![0; (base + cap * UNIT) as usize];
    for q in 0..cap {
        for w in 0..UNIT {
            pool[(id_of(q) * UNIT + w) as usize] = content(id_of(q), w);
        }
    }
    let pool_h = gpu.up_u32(&pool);
    let store_h = gpu.up_u32(&vec![0u32; lay.words as usize]);
    let mut host = State::new(lanes, cap, FRAMES);
    let mut rng = Rng(seed);
    let mut parents: Vec<u32> = (0..lanes).collect();
    let mut state = states[0];
    let mut store = vec![0u32; lay.words as usize];
    for t in 0..fires {
        if t == fires / 2 {
            // A park and restore into fresh words: the state's bytes move,
            // nothing else.
            let now = gpu.down_u32(pool_h, pool.len());
            pool.copy_from_slice(&now);
            let (from, to) = (states[0] as usize, states[1] as usize);
            pool.copy_within(from..from + state_words as usize, to);
            pool[from..from + state_words as usize].fill(0xDEAD);
            gpu.write_u32(pool_h, &pool);
            state = states[1];
        }
        let slot = folded + t;
        let c = slot / PAGE - f0;
        let reserve = (lanes * (c + 2) + lanes * FRAMES + 1).min(cap);
        let backed = engine_gpu::beam::backed(lanes, slot, folded, PAGE, FRAMES, reserve);
        let trans: Vec<u32> = (0..backed).map(id_of).collect();
        // The step's record, each lane's table (its `c + 1` entries past
        // `f0` staged as sentinels) and write page, the copies, the planes.
        store.fill(0);
        let entries = f0 + c + 1;
        store[lay.steps as usize..(lay.steps + kb::STEP_WORDS) as usize].copy_from_slice(&[
            state,
            lay.parents,
            lay.trans,
            lay.lists,
            lay.faults,
            lanes,
            cap,
            reserve,
            PAGE,
            folded,
            slot,
            lanes,
            FRAMES,
            backed,
            0,
            0,
        ]);
        for b in 0..lanes {
            let table = lay.tables + b * columns;
            for j in 0..entries {
                store[(table + j) as usize] = 0xA000 + b;
            }
            store[(lay.writes + b) as usize] = 0xB000 + b;
            store[(lay.offsets + b) as usize] = slot % PAGE;
            let list = (lay.lists + b * kb::LIST_WORDS) as usize;
            store[list..list + 8].copy_from_slice(&[
                table + f0,
                lay.writes + b,
                lay.offsets + b,
                b,
                entries - f0,
                1,
                0,
                0,
            ]);
            let copy = (lay.copies + b * kb::COPY_WORDS) as usize;
            store[copy..copy + 8].copy_from_slice(&[
                state + kb::copies_word(lanes, cap, FRAMES) as u32 + 3 * b,
                lay.trans,
                lay.faults + b,
                backed,
                PAGE,
                folded - f0 * PAGE,
                UNIT,
                PLANES.len() as u32,
            ]);
            store[(lay.parents + b) as usize] = parents[b as usize];
        }
        for (k, plane) in PLANES.iter().enumerate() {
            store[lay.planes as usize + 2 * k] = plane[0];
            store[lay.planes as usize + 2 * k + 1] = plane[1];
        }
        store[lay.trans as usize..(lay.trans + backed) as usize].copy_from_slice(&trans);
        gpu.write_u32(store_h, &store);
        let records = |runs: u32, count: u32| Records {
            store: Tensor::new(store_h, 1, lay.words, Dtype::U32),
            pool: Segments::one(pool_h, pool.len() as u32),
            runs,
            count,
            planes: lay.planes,
        };
        gpu.run(|c| {
            kb::step(c, records(lay.steps, 1))?;
            kb::copy_pages(c, records(lay.copies, lanes), PLANES.len() as u32)
        });

        let want = step(
            &mut host,
            &parents,
            Fire {
                page: PAGE,
                folded,
                reserve,
                slot,
            },
        );
        // The pool pages: the replay's copies applied to the host's mirror.
        for &[src, dst, tokens] in &want.copies {
            if tokens == 0 || src == dst || want.faults.iter().any(|&f| f == fault::COPY_BOUNDS) {
                continue;
            }
            for plane in PLANES {
                for w in 0..tokens * plane[1] {
                    let from = (id_of(src) * UNIT + plane[0] + w) as usize;
                    let to = (id_of(dst) * UNIT + plane[0] + w) as usize;
                    pool[to] = pool[from];
                }
            }
        }
        let got_store = gpu.down_u32(store_h, lay.words as usize);
        let got_pool = gpu.down_u32(pool_h, pool.len());
        let what = format!("width {lanes}, folded {folded}, fire {t} (slot {slot}, column {c})");
        assert_eq!(
            &got_pool[base as usize..],
            &pool[base as usize..],
            "{what}: the pool pages its copies moved"
        );
        let faults = &got_store[lay.faults as usize..(lay.faults + lanes) as usize];
        assert_eq!(faults, want.faults.as_slice(), "{what}: the faults");
        let s = state as usize;
        assert_eq!(got_pool[s], 1, "{what}: primed");
        assert_eq!(got_pool[s + 1], host.fired, "{what}: the fires run");
        let map = s + kb::map_word(lanes, cap, host.fired) as usize;
        for b in 0..lanes as usize {
            let list = &got_pool[map + b * cap as usize..map + b * cap as usize + (c + 1) as usize];
            assert_eq!(list, want.lists[b].as_slice(), "{what}: beam {b}'s list");
            if want.faults[b] != 0 {
                continue;
            }
            let table = (lay.tables + b as u32 * columns) as usize;
            let resolved: Vec<u32> = want.lists[b].iter().map(|&q| id_of(q)).collect();
            assert_eq!(
                &got_store[table + f0 as usize..table + entries as usize],
                resolved.as_slice(),
                "{what}: beam {b}'s resolved entries"
            );
            assert!(
                got_store[table..table + f0 as usize]
                    .iter()
                    .all(|&w| w == 0xA000 + b as u32),
                "{what}: beam {b}'s frozen entries stay as staged"
            );
            assert_eq!(
                got_store[lay.writes as usize + b],
                id_of(want.lists[b][c as usize]),
                "{what}: beam {b}'s write page"
            );
        }
        let free_at = s + kb::HEADER as usize + 2 * (lanes * cap) as usize;
        let free: Vec<bool> = got_pool[free_at..free_at + cap as usize]
            .iter()
            .map(|&w| w != 0)
            .collect();
        assert_eq!(free, host.free, "{what}: the free pages");
        let pending: Vec<bool> = got_pool
            [free_at + cap as usize..free_at + cap as usize + (FRAMES * cap) as usize]
            .iter()
            .map(|&w| w != 0)
            .collect();
        assert_eq!(pending, host.pending, "{what}: the pending pages");

        // The next fire's parents: forks and retirements.
        parents = (0..lanes)
            .map(|b| match rng.next() % 4 {
                0 => 0,
                1 => b,
                _ => rng.next() % lanes,
            })
            .collect();
    }
    eprintln!("width {lanes}, folded {folded}: {fires} fires equal the replay's");
}

#[test]
fn a_beam_step_follows_the_spec() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    for lanes in [2u32, 4, 8] {
        for (at, folded) in [37u32, 48].into_iter().enumerate() {
            run(&mut gpu, lanes, folded, 0x9E37 + u64::from(lanes) * 31 + at as u64);
        }
    }
}

#[test]
fn a_parent_past_the_beams_faults_its_lane_alone() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (lanes, folded) = (4u32, 21u32);
    let layout = BeamLayout::of(lanes, folded + 8, PAGE, FRAMES).expect("the layout");
    let cap = layout.cap;
    let state_words = kb::state_words(lanes, cap, FRAMES) as u32;
    let lay = Lay::of(lanes, cap, 4);
    let pool_h = gpu.up_u32(&vec![0u32; (state_words + (cap + 64) * UNIT) as usize]);
    let store_h = gpu.up_u32(&vec![0u32; lay.words as usize]);
    let mut host = State::new(lanes, cap, FRAMES);
    for (t, parents) in [[0u32, 1, 2, 3], [0, 9, 1, 1]].iter().enumerate() {
        let slot = folded + t as u32;
        let c = slot / PAGE - folded / PAGE;
        let reserve = (lanes * (c + 2) + lanes * FRAMES + 1).min(cap);
        let backed = engine_gpu::beam::backed(lanes, slot, folded, PAGE, FRAMES, reserve);
        let mut store = vec![0u32; lay.words as usize];
        store[..kb::STEP_WORDS as usize].copy_from_slice(&[
            0,
            lay.parents,
            lay.trans,
            lay.lists,
            lay.faults,
            lanes,
            cap,
            reserve,
            PAGE,
            folded,
            slot,
            0,
            FRAMES,
            backed,
            0,
            0,
        ]);
        for q in 0..backed {
            store[(lay.trans + q) as usize] = 64 + q;
        }
        store[lay.parents as usize..(lay.parents + lanes) as usize].copy_from_slice(parents);
        gpu.write_u32(store_h, &store);
        gpu.run(|c| {
            kb::step(
                c,
                Records {
                    store: Tensor::new(store_h, 1, lay.words, Dtype::U32),
                    pool: Segments::one(pool_h, state_words + (cap + 64) * UNIT),
                    runs: lay.steps,
                    count: 1,
                    planes: lay.planes,
                },
            )
        });
        let want = step(
            &mut host,
            parents,
            Fire {
                page: PAGE,
                folded,
                reserve,
                slot,
            },
        );
        let got = gpu.down_u32(store_h, lay.words as usize);
        let faults = &got[lay.faults as usize..(lay.faults + lanes) as usize];
        assert_eq!(faults, want.faults.as_slice(), "fire {t}'s faults");
        if t == 1 {
            assert_eq!(faults, [0, fault::PARENT, 0, 0], "lane 1 alone faults PARENT");
        }
    }
}
