//! A windowed layer as state + buffer (CONTRACT-BC 19, 21) on the first
//! device against an f64 host reference: each lane's keys below its buffer
//! base read from the ring in its record (row `r` at place `r % R`, R not a
//! page multiple), keys from the base to its first row of the fire from its
//! window-buffer slots, and the fire's own rows from the fire's slots, under
//! one mask (a lane masked past its folded index) and one softmax: decode
//! (whole and split, every walk and tier) and prefill (tiled and walked, a
//! lane's rows past the window). Then the fold: the rows a lane folds land
//! in its ring at `r % R` from the buffer and the fire's slots, the rows of
//! the fire it does not fold in its buffer slots, every other ring row left
//! as it was; and a lane that folds nothing leaves its ring untouched.

use kernels_slang::attn::{ring, sdpa};
use kernels_slang::{KvPool, Ring, Segments, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

const PAGE: usize = 16;
const QH: usize = 8;
const KVH: usize = 2;
/// The rows each ring holds, and the window the layer reads through.
const R: usize = 40;
const W: usize = 32;
/// A record's pieces; the ring lies in the second.
const PIECES: usize = 2;
const PIECE: usize = 1;
/// Words of a record piece before its ring's keys.
const RING_AT: usize = 8;
const MASK_STRIDE: usize = 64;

/// One lane of the fire: its folded index, its buffer base (a page's first
/// row, at or below folded), its first row of the fire and its rows, and
/// its mask past folded where it states one.
#[derive(Clone)]
struct Lane {
    folded: usize,
    base: usize,
    first: usize,
    rows: usize,
    masked: bool,
}

/// Where the pool lays a fire's lanes: slot units of their buffer and fire
/// slots, the record units, and the element values the pool holds.
struct Laid {
    /// A lane's table: its buffer slots from its base, then its fire slots.
    tables: Vec<Vec<u32>>,
    /// Each lane's record units, `PIECES` a lane.
    records: Vec<i32>,
    /// Each lane's keys and values by row, `KVH * hd` elements a row.
    keys: Vec<Vec<Vec<f64>>>,
    values: Vec<Vec<Vec<f64>>>,
    /// The pool's elements as the device holds them.
    pool: Vec<f32>,
    /// The first unit past every slot a table names.
    free: u32,
}

struct Geometry {
    hd: usize,
    /// The ring's rows and the window.
    r: usize,
    w: usize,
    /// Elements of one row (every kv head).
    row: usize,
    /// Words of one slot: `PAGE` rows of keys, then of values.
    slot_words: usize,
    /// Words of one record piece.
    piece_words: usize,
    /// Record units start here (units of `piece_words`).
    record0: usize,
    /// The pool's elements.
    elems: usize,
}

impl Geometry {
    fn of(hd: usize, slots: usize, lanes: usize) -> Geometry {
        Geometry::ringed(hd, slots, lanes, R, W)
    }

    fn ringed(hd: usize, slots: usize, lanes: usize, r: usize, w: usize) -> Geometry {
        let row = KVH * hd;
        let slot_words = PAGE * row;
        let piece_words = RING_AT + r * row + 8;
        let record0 = (slots * slot_words).div_ceil(piece_words);
        let words = (record0 + lanes * PIECES) * piece_words;
        Geometry {
            hd,
            r,
            w,
            row,
            slot_words,
            piece_words,
            record0,
            elems: 2 * words,
        }
    }

    /// Element of row `r` of slot `unit`, its keys or its values.
    fn slot_at(&self, unit: u32, r: usize, values: bool) -> usize {
        2 * unit as usize * self.slot_words + if values { PAGE * self.row } else { 0 } + r * self.row
    }

    /// Element of ring place `place` of lane `lane`, its keys or values.
    fn ring_at(&self, lane: usize, place: usize, values: bool) -> usize {
        let unit = self.record0 + lane * PIECES + PIECE;
        2 * (unit * self.piece_words + RING_AT) + if values { self.r * self.row } else { 0 } + place * self.row
    }
}

/// Lays `lanes` in a pool: every element noise, then each lane's rows where
/// its ring, buffer and fire slots hold them; slot units handed out
/// scattered.
fn lay(g: &Geometry, lanes: &[Lane], seed: u64) -> Laid {
    let mut pool = noise(seed, g.elems);
    // Rounded as the device holds them.
    for x in &mut pool {
        *x = crate::gpu::widen(crate::gpu::bf16(*x));
    }
    let mut next = 0u32;
    let mut take = || {
        // Every other slot, so no two of a lane are neighbours.
        let unit = next * 2 + 1;
        next += 1;
        unit
    };
    let mut tables = Vec::new();
    let mut records = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (l, lane) in lanes.iter().enumerate() {
        let end = lane.first + lane.rows;
        let kv: Vec<Vec<f64>> = (0..end)
            .map(|r| {
                noise(seed + 100 + (l * 1000 + r) as u64, g.row)
                    .into_iter()
                    .map(|x| f64::from(crate::gpu::widen(crate::gpu::bf16(x))))
                    .collect()
            })
            .collect();
        let vv: Vec<Vec<f64>> = (0..end)
            .map(|r| {
                noise(seed + 7 + (l * 1000 + r) as u64 * 3, g.row)
                    .into_iter()
                    .map(|x| f64::from(crate::gpu::widen(crate::gpu::bf16(x))))
                    .collect()
            })
            .collect();
        let buffered = lane.first.saturating_sub(lane.base).div_ceil(PAGE);
        let fired = lane.rows.div_ceil(PAGE);
        let table: Vec<u32> = (0..buffered + fired).map(|_| take()).collect();
        let mut put = |at: usize, row: &[f64]| {
            for (d, x) in row.iter().enumerate() {
                pool[at + d] = *x as f32;
            }
        };
        for r in lane.folded.saturating_sub(g.r)..lane.folded {
            put(g.ring_at(l, r % g.r, false), &kv[r]);
            put(g.ring_at(l, r % g.r, true), &vv[r]);
        }
        // Rows of a buffer slot below folded are the ring's to answer:
        // noise here, so a read of them from the buffer shows.
        for r in lane.folded.max(lane.base)..lane.first {
            let unit = table[(r - lane.base) / PAGE];
            put(g.slot_at(unit, (r - lane.base) % PAGE, false), &kv[r]);
            put(g.slot_at(unit, (r - lane.base) % PAGE, true), &vv[r]);
        }
        for r in lane.first..end {
            let unit = table[buffered + (r - lane.first) / PAGE];
            put(g.slot_at(unit, (r - lane.first) % PAGE, false), &kv[r]);
            put(g.slot_at(unit, (r - lane.first) % PAGE, true), &vv[r]);
        }
        tables.push(table);
        for piece in 0..PIECES {
            records.push((g.record0 + l * PIECES + piece) as i32);
        }
        keys.push(kv);
        values.push(vv);
    }
    Laid {
        tables,
        records,
        keys,
        values,
        pool,
        free: next * 2 + 1,
    }
}

/// Row `p` of lane `l`'s mask column `c` (key `folded + c`).
fn admits(l: usize, p: usize, c: usize) -> bool {
    (c * 5 + l * 3 + p) % 4 != 0
}

/// The keys row `p` of `lane` reads: its window, and past its folded index
/// its mask where it states one (its own key always).
fn keys_of(w: usize, l: usize, lane: &Lane, p: usize) -> Vec<usize> {
    ((p + 1).saturating_sub(w)..=p)
        .filter(|&kp| {
            !lane.masked || kp < lane.folded || kp == p || admits(l, p, kp - lane.folded)
        })
        .collect()
}

fn reference(g: &Geometry, lanes: &[Lane], laid: &Laid, q: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let scale = 1.0 / (g.hd as f64).sqrt();
    let mut o = Vec::new();
    let mut lse = Vec::new();
    let mut row = 0;
    for (l, lane) in lanes.iter().enumerate() {
        for p in lane.first..lane.first + lane.rows {
            let keys = keys_of(g.w, l, lane, p);
            for h in 0..QH {
                let kh = h / (QH / KVH);
                let qv = &q[(row * QH + h) * g.hd..][..g.hd];
                let s: Vec<f64> = keys
                    .iter()
                    .map(|&kp| {
                        let k = &laid.keys[l][kp][kh * g.hd..][..g.hd];
                        qv.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() * scale
                    })
                    .collect();
                let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = s.iter().map(|x| (x - top).exp()).sum();
                lse.push((z.ln() + top) * std::f64::consts::LOG2_E);
                for d in 0..g.hd {
                    o.push(
                        keys.iter()
                            .zip(&s)
                            .map(|(&kp, x)| (x - top).exp() * laid.values[l][kp][kh * g.hd + d])
                            .sum::<f64>()
                            / z,
                    );
                }
            }
            row += 1;
        }
    }
    (o, lse)
}

fn pack(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            u32::from_le_bytes([
                c[0],
                *c.get(1).unwrap_or(&0),
                *c.get(2).unwrap_or(&0),
                *c.get(3).unwrap_or(&0),
            ])
        })
        .collect()
}

/// The fire's tables and the pool over `laid`, uploaded.
struct Staged {
    pool: KvPool,
    tables: sdpa::Tables,
    pool_buf: u32,
}

fn stage(gpu: &mut Gpu, g: &Geometry, lanes: &[Lane], laid: &Laid) -> Staged {
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let (pool_buf, _) = gpu.up(&laid.pool);
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    for table in &laid.tables {
        indices.extend_from_slice(table);
        indptr.push(indices.len() as u32);
    }
    let ring_lanes: Vec<i32> = lanes
        .iter()
        .flat_map(|lane| [lane.folded as i32, lane.base as i32, lane.first as i32])
        .collect();
    let rows: usize = lanes.iter().map(|lane| lane.rows).sum();
    let mut positions = Vec::new();
    let mut owner = Vec::new();
    let mut mask = vec![0u8; rows * MASK_STRIDE];
    let mut words = [sdpa::MASK_NONE, 0].repeat(rows);
    let mut row = 0;
    for (l, lane) in lanes.iter().enumerate() {
        for p in lane.first..lane.first + lane.rows {
            positions.push(p as i32);
            owner.push(l as i32);
            if lane.masked {
                words[2 * row] = lane.folded as u32;
                words[2 * row + 1] = (row * MASK_STRIDE) as u32;
                for c in 0..MASK_STRIDE {
                    let kp = lane.folded + c;
                    mask[row * MASK_STRIDE + c] = u8::from(kp >= p || admits(l, p, c));
                }
            }
            row += 1;
        }
    }
    let (pi, pp) = (gpu.up_u32(&indices), gpu.up_u32(&indptr));
    let (rl, rs) = (gpu.up_i32(&ring_lanes), gpu.up_i32(&laid.records));
    let (pos, own) = (gpu.up_i32(&positions), gpu.up_i32(&owner));
    let (mb, eb) = (gpu.up_u32(&pack(&mask)), gpu.up_u32(&words));
    let words = u32::try_from(g.elems / 2).expect("the pool's words");
    let cells = |w: usize| t(pool_buf, g.elems / w, w, Dtype::Bf16);
    let pool = KvPool {
        keys: cells(g.row),
        values: cells(g.row),
        page_indices: t(pi, indices.len().max(1), 1, Dtype::U32),
        page_indptr: t(pp, indptr.len(), 1, Dtype::U32),
        page_size: PAGE as i32,
        seq_stride: g.row as u64,
        head_stride: g.hd as u64,
        pool: Segments::one(pool_buf, words),
        page_stride: [g.slot_words as u32, g.slot_words as u32],
        origin: [0, (PAGE * g.row / 2) as u32],
        ring: Some(Ring {
            lanes: t(rl, ring_lanes.len(), 1, Dtype::I32),
            slots: t(rs, laid.records.len(), 1, Dtype::I32),
            rows: g.r as u32,
            piece: PIECE as u32,
            pieces: PIECES as u32,
            keys: RING_AT as u32,
            values: (RING_AT + g.r * g.row / 2) as u32,
            slot_stride: g.piece_words as u32,
        }),
    };
    let tables = sdpa::Tables {
        positions: t(pos, rows, 1, Dtype::I32),
        request_of_token: t(own, rows, 1, Dtype::I32),
        mask: t(mb, 1, rows * MASK_STRIDE, Dtype::U8),
        mask_words: t(eb, rows, 2, Dtype::U32),
        row0: 0,
        rows: rows as u32,
    };
    Staged {
        pool,
        tables,
        pool_buf,
    }
}

/// Decode lanes: the ring and a buffer page straddling folded, a lane whose
/// buffer holds only folded rows of a page, a lane with no ring yet, a
/// masked one; prefill lanes: rows past the window from the fire's own,
/// starting inside a page and on one.
fn decodes() -> Vec<Lane> {
    vec![
        Lane { folded: 50, base: 48, first: 55, rows: 1, masked: false },
        Lane { folded: 20, base: 16, first: 20, rows: 1, masked: false },
        Lane { folded: 0, base: 0, first: 3, rows: 1, masked: false },
        Lane { folded: 70, base: 64, first: 81, rows: 1, masked: true },
        Lane { folded: 120, base: 112, first: 120, rows: 1, masked: true },
    ]
}

fn prefills() -> Vec<Lane> {
    vec![
        Lane { folded: 50, base: 48, first: 55, rows: 37, masked: false },
        Lane { folded: 64, base: 64, first: 64, rows: 40, masked: false },
        Lane { folded: 0, base: 0, first: 0, rows: 21, masked: false },
        Lane { folded: 70, base: 64, first: 75, rows: 9, masked: true },
        // Carrying nothing and folding most of its rows: its table starts
        // past its first row, at the folded index after the fire.
        Lane { folded: 30, base: 96, first: 30, rows: 80, masked: false },
    ]
}

#[test]
fn a_windowed_layer_reads_its_ring_buffer_and_fire_rows_under_one_mask() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let narrow = gpu.subgroup() < 32;
    for hd in [64usize, 256] {
        for (what, lanes, prefilling) in [("decode", decodes(), false), ("prefill", prefills(), true)] {
            let g = Geometry::of(hd, 64, lanes.len());
            let laid = lay(&g, &lanes, hd as u64);
            let staged = stage(&mut gpu, &g, &lanes, &laid);
            let rows = staged.tables.rows as usize;
            let (q, qr) = gpu.up(&noise(hd as u64 + 3, rows * QH * hd));
            let qr: Vec<f64> = qr.iter().map(|x| f64::from(*x)).collect();
            let (want_o, want_l) = reference(&g, &lanes, &laid, &qr);
            let qt = t(q, rows, QH * hd, Dtype::Bf16);
            let n = rows * QH * hd;
            let (part_out, part_lse) = (
                gpu.up_f32(&vec![0.0; 8 * n]),
                gpu.up_f32(&vec![0.0; 8 * rows * QH]),
            );
            let split = sdpa::Split {
                part_out: t(part_out, 8 * rows * QH, hd, Dtype::F32),
                part_lse: t(part_lse, 8 * rows, QH, Dtype::F32),
            };
            let tactics: &[(bool, u32, u32, u32, bool)] = if prefilling {
                &[
                    (false, 256, 32, 32, true),
                    (false, 256, 16, 32, true),
                    (false, 256, 8, 32, true),
                    (false, 256, 16, 16, true),
                    (true, 256, 16, 16, true),
                    (true, 256, 32, 32, false),
                    (false, 256, 32, 32, false),
                ]
            } else {
                &[
                    (true, 256, 32, 32, false),
                    (true, 128, 32, 32, false),
                    (false, 256, 32, 32, false),
                    (false, 128, 32, 32, false),
                ]
            };
            for &(subgroup, threads, tile, lanes_a_row, tiled) in tactics {
                let floor = if !crate::backend::SPIRV {
                    32
                } else if lanes_a_row == 16 && tiled {
                    16
                } else {
                    0
                };
                if subgroup && gpu.subgroup() < floor {
                    continue;
                }
                let tactic = sdpa::Tactic {
                    tier: if subgroup {
                        kernels_slang::Tier::Subgroup
                    } else {
                        kernels_slang::Tier::Baseline
                    },
                    narrow,
                    decode_threads: threads,
                    tile_rows: tile,
                    tile_lanes: lanes_a_row,
                    split_groups: 1024,
                    split_min_keys: 8,
                    prefill_work: 0,
                };
                let splits: &[bool] = if prefilling { &[false] } else { &[false, true] };
                for &splitting in splits {
                    for with_lse in [false, true] {
                        let (o, l) = (gpu.zeros(n), gpu.up_f32(&vec![0.0; rows * QH]));
                        let lse = with_lse.then(|| t(l, rows, QH, Dtype::F32));
                        let ot = t(o, rows, QH * hd, Dtype::Bf16);
                        gpu.run(|c| {
                            if tiled {
                                sdpa::prefill(
                                    c,
                                    tactic,
                                    qt,
                                    &staged.tables,
                                    &staged.pool,
                                    Some(g.w as u32),
                                    hd as u32,
                                    (1.0 / (hd as f64).sqrt()) as f32,
                                    ot,
                                    lse,
                                )
                            } else {
                                sdpa::decode(
                                    c,
                                    tactic,
                                    qt,
                                    &staged.tables,
                                    &staged.pool,
                                    Some(g.w as u32),
                                    hd as u32,
                                    (1.0 / (hd as f64).sqrt()) as f32,
                                    ot,
                                    lse,
                                    splitting.then_some(split),
                                )
                            }
                        });
                        let label = format!(
                            "ring {what} d {hd} subgroup {subgroup} threads {threads} tile {tile}x{lanes_a_row} tiled {tiled} split {splitting} lse {with_lse}"
                        );
                        close(&gpu.down(o, n), &want_o, &label);
                        if with_lse {
                            close(&gpu.down_f32(l, rows * QH), &want_l, &format!("{label}: lse"));
                        }
                    }
                }
            }
        }
    }
}

/// The fold of `lanes`, each to `to` (its folded index after the fire): the
/// rows `[folded, to)` into its ring (only the last R land), from its buffer
/// below its first row and from the fire's slots past it; the fire's rows
/// from `to` on into fresh buffer slots from `free`. Returns the jobs and
/// where each unfolded row landed.
fn fold_jobs(
    lanes: &[Lane],
    to: &[usize],
    laid: &Laid,
) -> (Vec<i32>, Vec<(usize, usize, u32, usize)>) {
    let mut jobs = Vec::new();
    let mut moved = Vec::new();
    let mut free = laid.free;
    for (l, lane) in lanes.iter().enumerate() {
        let buffered = lane.first.saturating_sub(lane.base).div_ceil(PAGE);
        let table = &laid.tables[l];
        let source = |r: usize| -> (u32, usize) {
            if r < lane.first {
                (table[(r - lane.base) / PAGE], (r - lane.base) % PAGE)
            } else {
                (table[buffered + (r - lane.first) / PAGE], (r - lane.first) % PAGE)
            }
        };
        for r in lane.folded.max(to[l].saturating_sub(R))..to[l] {
            let (unit, row) = source(r);
            jobs.extend([unit as i32, row as i32, -1 - l as i32, (r % R) as i32]);
        }
        // The rows of the fire it leaves unfolded, into buffer slots of
        // their own from the slot holding row `to` on.
        let end = lane.first + lane.rows;
        let from = to[l].max(lane.first);
        let mut dst = None;
        for r in from..end {
            if dst.is_none() || (r - from) % PAGE == 0 {
                dst = Some(free);
                free += 2;
            }
            let (unit, row) = source(r);
            let at = dst.unwrap_or(0);
            jobs.extend([unit as i32, row as i32, at as i32, ((r - from) % PAGE) as i32]);
            moved.push((l, r, at, (r - from) % PAGE));
        }
    }
    let count = (jobs.len() / 4) as i32;
    let mut table = vec![count];
    table.extend(jobs);
    (table, moved)
}

#[test]
fn a_fold_lands_folded_rows_in_the_ring_and_unfolded_ones_in_the_buffer() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let hd = 64;
    let lanes = prefills();
    // A lane folding its buffer and part of its fire, one past the ring's
    // rows (the oldest drop), one folding all of it, and one folding
    // nothing (its ring untouched: a stop-truncate's lane).
    let to = [70, 64 + 40, 21, 70, 100];
    let g = Geometry::of(hd, 64, lanes.len());
    let laid = lay(&g, &lanes, 9);
    let staged = stage(&mut gpu, &g, &lanes, &laid);
    let (jobs, moved) = fold_jobs(&lanes, &to, &laid);
    let most = (jobs.len() as u32 - 1) / ring::JOB_INTS + 3;
    let mut padded = jobs.clone();
    padded.resize(1 + most as usize * 4, 0);
    let jt = gpu.up_i32(&padded);
    let job_table = Tensor::new(jt, padded.len() as u32, 1, Dtype::I32);
    gpu.run(|c| ring::fold(c, &staged.pool, job_table, most));
    let after = gpu.down(staged.pool_buf, g.elems);
    let mut want: Vec<f64> = laid.pool.iter().map(|x| f64::from(*x)).collect();
    for (l, lane) in lanes.iter().enumerate() {
        for r in lane.folded.max(to[l].saturating_sub(R))..to[l] {
            for d in 0..g.row {
                want[g.ring_at(l, r % R, false) + d] = laid.keys[l][r][d];
                want[g.ring_at(l, r % R, true) + d] = laid.values[l][r][d];
            }
        }
    }
    for &(l, r, unit, row) in &moved {
        for d in 0..g.row {
            want[g.slot_at(unit, row, false) + d] = laid.keys[l][r][d];
            want[g.slot_at(unit, row, true) + d] = laid.values[l][r][d];
        }
    }
    let got: Vec<f64> = after.iter().map(|x| f64::from(*x)).collect();
    let wrong = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(wrong, 0, "{wrong} pool elements differ from the fold's rows");
    // The lane that folded nothing reads its ring as it was.
    let untouched = 3;
    for place in 0..R {
        let at = g.ring_at(untouched, place, false);
        assert_eq!(&after[at..at + g.row], &laid.pool[at..at + g.row], "ring place {place}");
    }
    // Read through after the fold, every lane's ring holds its last R folded rows.
    for (l, lane) in lanes.iter().enumerate() {
        let _ = lane;
        for r in to[l].saturating_sub(R)..to[l] {
            let at = g.ring_at(l, r % R, false);
            let row: Vec<f64> = after[at..at + g.row].iter().map(|x| f64::from(*x)).collect();
            if r >= lanes[l].folded.saturating_sub(R) {
                assert_eq!(row, laid.keys[l][r], "lane {l} row {r} in its ring");
            }
        }
    }
}

/// The ring's cost on decode against the window list's: one row of each of
/// four lanes, 1024 keys of a 1024-row window at 2048 rows, read through a
/// page table (no ring: the old window list), from the ring, and through
/// the ring kernel's table alone. Prints each arm's best of a few rounds
/// and the ring's ratio to the table's; the gate is the serving decode.
#[test]
fn a_ring_decode_costs_what_a_paged_window_does() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let (hd, ctx, window, reps) = (256usize, 2048usize, 1024usize, 64);
    let lanes = |ringed: bool| -> Vec<Lane> {
        (0..4)
            .map(|_| Lane {
                folded: if ringed { ctx } else { 0 },
                base: if ringed { ctx } else { 0 },
                first: ctx,
                rows: 1,
                masked: false,
            })
            .collect()
    };
    let g = Geometry::ringed(hd, 4 * 2 * (ctx / PAGE + 2), 4, window, window);
    let subgroup = crate::backend::SPIRV && gpu.subgroup() >= 32;
    let tactic = sdpa::Tactic {
        tier: if subgroup {
            kernels_slang::Tier::Subgroup
        } else {
            kernels_slang::Tier::Baseline
        },
        narrow: gpu.subgroup() < 32,
        decode_threads: 256,
        tile_rows: 16,
        tile_lanes: 16,
        split_groups: 1,
        split_min_keys: 64,
        prefill_work: 0,
    };
    let mut arms = Vec::new();
    for (what, ringed, ring) in [("table", false, false), ("ring kernel, table", false, true), ("ring", true, true)] {
        let lanes = lanes(ringed);
        let laid = lay(&g, &lanes, 5);
        let mut staged = stage(&mut gpu, &g, &lanes, &laid);
        if !ring {
            staged.pool.ring = None;
        }
        let rows = 4;
        let (q, _) = gpu.up(&noise(77, rows * QH * hd));
        let qt = t(q, rows, QH * hd, Dtype::Bf16);
        let o = gpu.zeros(rows * QH * hd);
        let ot = t(o, rows, QH * hd, Dtype::Bf16);
        arms.push((what, staged, qt, ot));
    }
    let mut best = vec![f64::INFINITY; arms.len()];
    for _ in 0..5 {
        for (at, (_, staged, qt, ot)) in arms.iter().enumerate() {
            let ns = gpu.timed(|c| {
                for _ in 0..reps {
                    sdpa::decode(
                        c,
                        tactic,
                        *qt,
                        &staged.tables,
                        &staged.pool,
                        Some(window as u32),
                        hd as u32,
                        0.0625,
                        *ot,
                        None,
                        None,
                    )?;
                }
                Ok(())
            });
            best[at] = best[at].min(ns / reps as f64);
        }
    }
    for ((what, ..), ns) in arms.iter().zip(&best) {
        eprintln!("ring cost: {what}: {:.1} us a decode", ns / 1000.0);
    }
    eprintln!(
        "ring cost: ring / table {:.3}, ring kernel's table / table {:.3}",
        best[2] / best[0],
        best[1] / best[0]
    );
}
