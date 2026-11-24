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
    common: u32,
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
        let common = take(1);
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
            common,
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
    let mut histories: Vec<Vec<u32>> = vec![Vec::new(); lanes as usize];
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
            lay.common,
            0,
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
            kb::copy_pages(c, records(lay.copies, lanes), PLANES.len() as u32, PAGE)
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
        // §11.4: the common slot, the longest common prefix of the live
        // beams' histories (a token a fire unique to its beam), capped at
        // the slot past this fire's rows and never below the folded slot.
        histories = (0..lanes as usize)
            .map(|b| {
                let mut kept = if t == 0 {
                    Vec::new()
                } else {
                    histories[parents[b] as usize].clone()
                };
                kept.push(t * 64 + b as u32);
                kept
            })
            .collect();
        let mut lcp = slot + 1 - folded;
        for i in 0..lanes as usize {
            for j in i + 1..lanes as usize {
                let common = histories[i]
                    .iter()
                    .zip(&histories[j])
                    .take_while(|(a, b)| a == b)
                    .count() as u32;
                lcp = lcp.min(common);
            }
        }
        let common = (folded + lcp).min(slot + 1).max(folded);
        assert_eq!(got_pool[s + 3], common, "{what}: the common slot in the header");
        assert_eq!(got_store[lay.common as usize], common, "{what}: the common slot");
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

/// Groups wider than the old fixed 64 and than the step's workgroup (256
/// threads), up to the most its tables hold in the workgroup storage every
/// device grants, follow the spec; one past it is refused by name.
#[test]
fn a_beam_group_past_64_and_past_the_workgroup_follows_the_spec() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let most = kernels_slang::runtime::beam::MAX_LANES;
    for lanes in [65u32, 300, most] {
        run(&mut gpu, lanes, 37, 0x7A11 + u64::from(lanes));
    }
    let why = kernels_slang::runtime::beam::admit_lanes(most + 1)
        .expect_err("a group past the step's tables is refused")
        .to_string();
    assert!(why.contains("workgroup storage"), "{why}");
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
            lay.common,
            0,
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

/// A step record past the step's bounds (its pool past the state's columns)
/// steps nothing and faults every lane by the group's code: never a silent
/// skip that leaves the lists unresolved.
#[test]
fn a_record_past_the_steps_bounds_faults_every_lane() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (lanes, folded) = (4u32, 21u32);
    let layout = BeamLayout::of(lanes, folded + 8, PAGE, FRAMES).expect("the layout");
    let cap = layout.cap;
    let state_words = kb::state_words(lanes, cap, FRAMES) as u32;
    let lay = Lay::of(lanes, cap, 4);
    let pool_words = state_words + (cap + 64) * UNIT;
    let pool_h = gpu.up_u32(&vec![0u32; pool_words as usize]);
    let store_h = gpu.up_u32(&vec![0u32; lay.words as usize]);
    let mut store = vec![0u32; lay.words as usize];
    store[..kb::STEP_WORDS as usize].copy_from_slice(&[
        0, lay.parents, lay.trans, lay.lists, lay.faults, lanes, cap,
        // A pool of more pages than the state's columns hold.
        cap + 1,
        PAGE, folded, folded, 0, FRAMES, 0, 0, 0, lay.common, 0, 0, 0,
    ]);
    gpu.write_u32(store_h, &store);
    gpu.run(|c| {
        kb::step(
            c,
            Records {
                store: Tensor::new(store_h, 1, lay.words, Dtype::U32),
                pool: Segments::one(pool_h, pool_words),
                runs: lay.steps,
                count: 1,
                planes: lay.planes,
            },
        )
    });
    let got = gpu.down_u32(store_h, lay.words as usize);
    let faults = &got[lay.faults as usize..(lay.faults + lanes) as usize];
    assert_eq!(faults, [kb::PAGES_TOO_WIDE; 4], "every lane faults the group's code");
    let pool = gpu.down_u32(pool_h, state_words as usize);
    assert!(pool.iter().all(|&w| w == 0), "nothing was stepped");
}

/// The rs planes of a hybrid's page (§11.2's case): one of a word a row
/// from word 0 and one of two words a row after the page's `T` rows of it,
/// in a page of `4 T` words.
fn rs_planes(tokens: u32) -> [[u32; 2]; 2] {
    [[0, 1], [tokens, 2]]
}

/// A hybrid's beams (§9's hybrid case, §11): the set's rs entries appended
/// into the group's state, each beam's resolve record naming its list
/// there, its entries for slots `base..=slot` resolving slot by slot over
/// that list (as `prim/carry.slang` resolves them), and its fork's rows
/// copied plane by plane, against §11.1 and §11.2 over the replay's lists
/// and copies. `tokens` rows an rs page from buffer base `base`.
fn run_rs(gpu: &mut Gpu, lanes: u32, folded: u32, tokens: u32, base: u32, seed: u64) {
    let fires = 3 * PAGE + 9;
    let end = folded + fires;
    let f0 = folded / PAGE;
    let layout = BeamLayout::of(lanes, end, PAGE, FRAMES).expect("the layout");
    let cap = layout.cap;
    // The set's rs entries: every slot from the base through the pool's
    // last page, held past the rest of the state.
    let set_len = ((f0 + cap) * PAGE - base).div_ceil(tokens);
    let layout = layout.buffering(PAGE, tokens);
    assert!(layout.bases >= set_len);
    let bases_pool = layout.bases_word() as u32;
    let state_words = (layout.bytes() / 4) as u32;
    let entries_most = (end - base) / tokens + 1;
    let unit = 4 * tokens;
    let planes = rs_planes(tokens);
    let mut at = 0u32;
    let mut take = |words: u32| {
        let here = at;
        at += words;
        here
    };
    let steps = take(kb::STEP_WORDS);
    let rs_lists = take(lanes * kb::RS_LIST_WORDS);
    let rs_copies = take(lanes * kb::RS_COPY_WORDS);
    let planes_at = take(2 * planes.len() as u32);
    let trans = take(cap);
    let parents_at = take(lanes);
    let faults = take(lanes);
    let bases = take(set_len);
    let resolves = take(lanes * 6);
    let _ = entries_most;
    let common = take(1);
    let words = at;
    // The pool: the state, then the rs pages, unit `u` at words `u * unit`.
    let first_unit = state_words.div_ceil(unit) + 1;
    let set: Vec<u32> = (0..set_len).map(|a| first_unit + a).collect();
    let mut pool: Vec<u32> = vec![0; ((first_unit + set_len) * unit) as usize];
    for (a, &u) in set.iter().enumerate() {
        for w in 0..unit {
            pool[(u * unit + w) as usize] = content(a as u32 + 1, w);
        }
    }
    let pool_h = gpu.up_u32(&pool);
    let store_h = gpu.up_u32(&vec![0u32; words as usize]);
    let mut host = State::new(lanes, cap, FRAMES);
    let mut rng = Rng(seed);
    let mut parents: Vec<u32> = (0..lanes).collect();
    let mut store = vec![0u32; words as usize];
    for t in 0..fires {
        let slot = folded + t;
        let c = slot / PAGE - f0;
        let reserve = (lanes * (c + 2) + lanes * FRAMES + 1).min(cap);
        let backed = engine_gpu::beam::backed(lanes, slot, folded, PAGE, FRAMES, reserve);
        let entries = (slot - base) / tokens + 1;
        store.fill(0);
        store[steps as usize..(steps + kb::STEP_WORDS) as usize].copy_from_slice(&[
            0, parents_at, trans, 0, faults, lanes, cap, reserve, PAGE, folded, slot, 0, FRAMES,
            backed, lanes, rs_lists, common, bases, bases_pool, set_len,
        ]);
        for q in 0..backed {
            store[(trans + q) as usize] = 1 + q;
        }
        store[bases as usize..(bases + set_len) as usize].copy_from_slice(&set);
        for (k, plane) in planes.iter().enumerate() {
            store[planes_at as usize + 2 * k] = plane[0];
            store[planes_at as usize + 2 * k + 1] = plane[1];
        }
        for b in 0..lanes {
            let l = (rs_lists + b * kb::RS_LIST_WORDS) as usize;
            store[l..l + 8].copy_from_slice(&[resolves + 6 * b, bases_pool, b, entries, base, tokens, set_len, 0]);
            let r = (rs_copies + b * kb::RS_COPY_WORDS) as usize;
            store[r..r + 16].copy_from_slice(&[
                kb::copies_word(lanes, cap, FRAMES) as u32 + 3 * b,
                bases_pool,
                faults + b,
                backed,
                PAGE,
                tokens,
                base,
                f0,
                folded - f0 * PAGE,
                unit,
                planes.len() as u32,
                set_len,
                0,
                0,
                0,
                0,
            ]);
            store[(parents_at + b) as usize] = parents[b as usize];
        }
        gpu.write_u32(store_h, &store);
        let records = |runs: u32, count: u32| Records {
            store: Tensor::new(store_h, 1, words, Dtype::U32),
            pool: Segments::one(pool_h, pool.len() as u32),
            runs,
            count,
            planes: planes_at,
        };
        gpu.run(|c| {
            kb::step(c, records(steps, 1))?;
            kb::copy_rs_rows(c, records(rs_copies, lanes), planes.len() as u32, PAGE)
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
        // §11.2 on the host's mirror.
        for (b, &[src, dst, n]) in want.copies.iter().enumerate() {
            if n == 0 || src == dst || want.faults[b] == fault::COPY_BOUNDS {
                continue;
            }
            for k in 0..n {
                let from = (f0 + src) * PAGE + k;
                let to = (f0 + dst) * PAGE + k;
                if from < base || to < base {
                    continue;
                }
                let (fe, te) = ((from - base) / tokens, (to - base) / tokens);
                for plane in planes {
                    for w in 0..plane[1] {
                        let f = set[fe as usize] * unit + plane[0] + ((from - base) % tokens) * plane[1] + w;
                        let d = set[te as usize] * unit + plane[0] + ((to - base) % tokens) * plane[1] + w;
                        pool[d as usize] = pool[f as usize];
                    }
                }
            }
        }
        let got_store = gpu.down_u32(store_h, words as usize);
        let got_pool = gpu.down_u32(pool_h, pool.len());
        let what = format!(
            "width {lanes}, folded {folded}, {tokens} rows an rs page from {base}, fire {t} \
             (slot {slot})"
        );
        assert_eq!(
            &got_store[faults as usize..(faults + lanes) as usize],
            want.faults.as_slice(),
            "{what}: the faults"
        );
        assert_eq!(
            &got_pool[(first_unit * unit) as usize..],
            &pool[(first_unit * unit) as usize..],
            "{what}: the rs rows its copies moved"
        );
        assert_eq!(
            &got_pool[bases_pool as usize..(bases_pool + set_len) as usize],
            set.as_slice(),
            "{what}: the set's entries appended into the state"
        );
        for b in 0..lanes as usize {
            let r = (resolves + 6 * b as u32) as usize;
            let record = &got_store[r..r + 6];
            assert_eq!(
                &record[1..],
                &[f0, PAGE, base, bases_pool, set_len],
                "{what}: beam {b}'s resolve record"
            );
            // §11.1 over the replay's list, and as `prim/carry.slang`
            // resolves it over the list its record names.
            let resolve = |list: &dyn Fn(u32) -> u32, e: u32| {
                let at = base + e * tokens;
                if at < f0 * PAGE {
                    return set[e as usize];
                }
                let sigma = (f0 + list(at / PAGE - f0)) * PAGE + at % PAGE;
                set[((sigma - base) / tokens) as usize]
            };
            for e in 0..entries {
                let want_page = resolve(&|j| want.lists[b][j as usize], e);
                let got_page = resolve(&|j| got_pool[(record[0] + j) as usize], e);
                assert_eq!(got_page, want_page, "{what}: beam {b}'s rs entry {e}");
            }
        }
        parents = (0..lanes)
            .map(|b| match rng.next() % 4 {
                0 => 0,
                1 => b,
                _ => rng.next() % lanes,
            })
            .collect();
    }
    eprintln!("width {lanes}, folded {folded}, rs pages of {tokens} from {base}: {fires} fires equal §11");
}

#[test]
fn a_hybrid_beams_rs_rows_follow_their_lists() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    for lanes in [2u32, 4, 8] {
        for tokens in [1u32, 2, 4] {
            // A frozen slot off a kv page (its buffer from the frozen slot,
            // on a row page), and one on a page boundary.
            for folded in [38u32, 48] {
                let base = folded / tokens * tokens;
                run_rs(&mut gpu, lanes, folded, tokens, base, 0x51 + u64::from(lanes * tokens + folded));
            }
        }
    }
}

/// A fork's rs copy on a hybrid (§6.4, §11.2), engine-cuda's and
/// kernels-rocm's `a_fork_copies_its_rs_rows_slot_by_slot` on this
/// engine's kernel: rows of 2 a page from slot 38, kv pages of 16 (the live
/// Qwen3.5 frozen slot), two planes of other row words. A copy of 5 slots
/// from pool page 1 to 3 ends mid rs page; one from the prompt's own page 0
/// skips its tail's slots below the buffer (they are in the state); each
/// destination row equals its source row, every other word stays. A copy
/// past the pool faults and moves nothing; one reaching a slot past the
/// set's entries copies the rows before it, faults, and moves none from it
/// on (§11.2's stop).
#[test]
fn a_fork_copies_its_rs_rows_slot_by_slot() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    const T: u32 = 2;
    const BASE: u32 = 38;
    let f0 = BASE / PAGE;
    // Planes: 2 words a row from word 0, 3 words a row after the page's two
    // rows of it; an entry (an rs unit) of 10 words.
    let planes: [[u32; 2]; 2] = [[0, 2], [4, 3]];
    let unit = 10u32;
    // The pool: the copy word, the set's entries (as its group's state
    // holds them) from word 8, then the rs units.
    let first_unit = 8u32;
    let bases_pool = 8u32;
    let run_case = |gpu: &mut Gpu, src: u32, dst: u32, tokens: u32, entries: u32| {
        let pool: Vec<u32> = (0..(first_unit + 64) * unit).map(|w| w.wrapping_mul(2_654_435_761)).collect();
        let pool_h = gpu.up_u32(&pool);
        // The store: the record, the plane table, the fault word.
        let mut store = vec![0u32; 128];
        let (record, planes_at, fault_at) = (0usize, 16usize, 120usize);
        store[planes_at..planes_at + 4].copy_from_slice(&[planes[0][0], planes[0][1], planes[1][0], planes[1][1]]);
        store[record..record + 16].copy_from_slice(&[
            0,
            bases_pool,
            fault_at as u32,
            8,
            PAGE,
            T,
            BASE,
            f0,
            BASE - f0 * PAGE,
            unit,
            2,
            entries,
            0,
            0,
            0,
            0,
        ]);
        // The copy word lies in the pool's first words (the state's place),
        // the set's entries after it.
        let mut pool = pool;
        pool[..3].copy_from_slice(&[src, dst, tokens]);
        for e in 0..entries.min(64) {
            pool[(bases_pool + e) as usize] = first_unit + e;
        }
        gpu.write_u32(pool_h, &pool);
        let store_h = gpu.up_u32(&store);
        gpu.run(|c| {
            kb::copy_rs_rows(
                c,
                Records {
                    store: Tensor::new(store_h, 1, 128, Dtype::U32),
                    pool: Segments::one(pool_h, pool.len() as u32),
                    runs: record as u32,
                    count: 1,
                    planes: planes_at as u32,
                },
                2,
                PAGE,
            )
        });
        let after = gpu.down_u32(pool_h, pool.len());
        let fault = gpu.down_u32(store_h, 128)[fault_at];
        (pool, after, fault)
    };
    let row = |slot: u32, plane: [u32; 2]| -> std::ops::Range<usize> {
        let (e, r) = ((slot - BASE) / T, (slot - BASE) % T);
        let at = ((first_unit + e) * unit + plane[0] + r * plane[1]) as usize;
        at..at + plane[1] as usize
    };
    // The copy word lies at pool words 0..3, where the record names it.
    for (src, dst, tokens, entries) in [(1u32, 3u32, 5u32, 64u32), (0, 2, 9, 64), (1, 3, 9, 23)] {
        let (before, after, fault) = run_case(&mut gpu, src, dst, tokens, entries);
        let mut want = before.clone();
        let end = BASE + entries * T;
        let mut stopped = false;
        for k in 0..tokens {
            let (from, to) = ((f0 + src) * PAGE + k, (f0 + dst) * PAGE + k);
            if from >= end || to >= end {
                stopped = true;
                break;
            }
            if from < BASE || to < BASE {
                continue;
            }
            for plane in planes {
                let words = before[row(from, plane)].to_vec();
                want[row(to, plane)].copy_from_slice(&words);
            }
        }
        let what = format!("copy {src} -> {dst} x {tokens} over {entries} entries");
        assert_eq!(fault, if stopped { kb::PAGES_COPY_BOUNDS } else { 0 }, "{what}: its fault");
        assert_eq!(after, want, "{what}: rows moved, nothing else");
    }
    // A copy past the pool faults and moves nothing.
    let (before, after, fault) = run_case(&mut gpu, 1, 9, 4, 64);
    assert_eq!(fault, kb::PAGES_COPY_BOUNDS);
    assert_eq!(after, before, "a refused copy moves nothing");
}
