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
        let (r, w) = (R, W);
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
        2 * unit as usize * self.slot_words
            + if values { PAGE * self.row } else { 0 }
            + r * self.row
    }

    /// Element of ring place `place` of lane `lane`, its keys or values.
    fn ring_at(&self, lane: usize, place: usize, values: bool) -> usize {
        let unit = self.record0 + lane * PIECES + PIECE;
        2 * (unit * self.piece_words + RING_AT)
            + if values { self.r * self.row } else { 0 }
            + place * self.row
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
    !(c * 5 + l * 3 + p).is_multiple_of(4)
}

/// The keys row `p` of `lane` reads: its window, and past its folded index
/// its mask where it states one (its own key always).
fn keys_of(w: usize, l: usize, lane: &Lane, p: usize) -> Vec<usize> {
    ((p + 1).saturating_sub(w)..=p)
        .filter(|&kp| !lane.masked || kp < lane.folded || kp == p || admits(l, p, kp - lane.folded))
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

/// `to`: each lane's folded index after the fire, which the fold reads.
fn stage(gpu: &mut Gpu, g: &Geometry, lanes: &[Lane], laid: &Laid, to: &[usize]) -> Staged {
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let (pool_buf, _) = gpu.up(&laid.pool);
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    for table in &laid.tables {
        indices.extend_from_slice(table);
        indptr.push(indices.len() as u32);
    }
    // A lane's fold, then its record's units.
    let ring_lanes: Vec<i32> = lanes
        .iter()
        .zip(to)
        .enumerate()
        .flat_map(|(l, (lane, &to))| {
            // Alone on its set: its lineage is the set's slots.
            [
                lane.folded as i32,
                lane.base as i32,
                lane.first as i32,
                to as i32,
                0,
                lane.folded as i32,
            ]
            .into_iter()
            .chain(laid.records[l * PIECES..][..PIECES].iter().copied())
        })
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
    let rl = gpu.up_i32(&ring_lanes);
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
            rows: g.r as u32,
            piece: PIECE as u32,
            values_piece: PIECE as u32,
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
        table0: 0,
        rows: rows as u32,
    };
    Staged {
        pool,
        tables,
        pool_buf,
    }
}

/// `pool` moved past 8 GiB of the pool's words: its one buffer the fourth
/// segment of segments of 4 GiB, every place it names (pages' origins, the
/// rings') three segments up. A bf16 element index there passes a u32.
fn high(pool: &KvPool) -> KvPool {
    const SEG: u32 = 1 << 30;
    let mut moved = *pool;
    moved.pool = Segments {
        bufs: [
            kernels_slang::ABSENT,
            kernels_slang::ABSENT,
            kernels_slang::ABSENT,
            pool.pool.bufs[0],
        ],
        words: SEG,
    };
    moved.origin = [pool.origin[0] + 3 * SEG, pool.origin[1] + 3 * SEG];
    if let Some(ring) = &mut moved.ring {
        ring.keys += 3 * SEG;
        ring.values += 3 * SEG;
    }
    moved
}

/// Decode lanes: the ring and a buffer page straddling folded, a lane whose
/// buffer holds only folded rows of a page, a lane with no ring yet, a
/// masked one; prefill lanes: rows past the window from the fire's own,
/// starting inside a page and on one.
fn decodes() -> Vec<Lane> {
    vec![
        Lane {
            folded: 50,
            base: 48,
            first: 55,
            rows: 1,
            masked: false,
        },
        Lane {
            folded: 20,
            base: 16,
            first: 20,
            rows: 1,
            masked: false,
        },
        Lane {
            folded: 0,
            base: 0,
            first: 3,
            rows: 1,
            masked: false,
        },
        Lane {
            folded: 70,
            base: 64,
            first: 81,
            rows: 1,
            masked: true,
        },
        Lane {
            folded: 120,
            base: 112,
            first: 120,
            rows: 1,
            masked: true,
        },
    ]
}

fn prefills() -> Vec<Lane> {
    vec![
        Lane {
            folded: 50,
            base: 48,
            first: 55,
            rows: 37,
            masked: false,
        },
        Lane {
            folded: 64,
            base: 64,
            first: 64,
            rows: 40,
            masked: false,
        },
        Lane {
            folded: 0,
            base: 0,
            first: 0,
            rows: 21,
            masked: false,
        },
        Lane {
            folded: 70,
            base: 64,
            first: 75,
            rows: 9,
            masked: true,
        },
        // Carrying nothing and folding most of its rows: its table starts
        // past its first row, at the folded index after the fire.
        Lane {
            folded: 30,
            base: 96,
            first: 30,
            rows: 80,
            masked: false,
        },
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
        for (what, lanes, tiled) in [("decode", decodes(), false), ("prefill", prefills(), true)] {
            let g = Geometry::of(hd, 64, lanes.len());
            let laid = lay(&g, &lanes, hd as u64);
            let folded: Vec<usize> = lanes.iter().map(|lane| lane.folded).collect();
            let staged = stage(&mut gpu, &g, &lanes, &laid, &folded);
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
            // (subgroup, tile rows, tile lanes): the walked decode, and the
            // tiled prefill at 16 lanes a row and 32.
            let tactics: &[(bool, u32, u32)] = if tiled {
                &[(false, 16, 16), (true, 16, 16), (false, 32, 32)]
            } else {
                &[(true, 32, 32), (false, 32, 32)]
            };
            for &(subgroup, tile, lanes_a_row) in tactics {
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
                    decode_threads: 256,
                    tile_rows: tile,
                    tile_lanes: lanes_a_row,
                    split_groups: 1024,
                    split_min_keys: 8,
                    prefill_work: 0,
                };
                // Split (decode), its LSE, and the pool past 8 GiB of words.
                for (splitting, with_lse, past) in [(false, false, false), (!tiled, true, true)] {
                    let kv = if past {
                        high(&staged.pool)
                    } else {
                        staged.pool
                    };
                    let (o, l) = (gpu.zeros(n), gpu.up_f32(&vec![0.0; rows * QH]));
                    let lse = with_lse.then(|| t(l, rows, QH, Dtype::F32));
                    let ot = t(o, rows, QH * hd, Dtype::Bf16);
                    let (win, scale) = (Some(g.w as u32), (1.0 / (hd as f64).sqrt()) as f32);
                    gpu.run(|c| {
                        if tiled {
                            sdpa::prefill(
                                c,
                                tactic,
                                qt,
                                &staged.tables,
                                &kv,
                                win,
                                hd as u32,
                                scale,
                                ot,
                                lse,
                            )
                        } else {
                            let split = splitting.then_some(split);
                            sdpa::decode(
                                c,
                                tactic,
                                qt,
                                &staged.tables,
                                &kv,
                                win,
                                hd as u32,
                                scale,
                                ot,
                                lse,
                                split,
                            )
                        }
                    });
                    let label = format!(
                        "ring {what} d {hd} subgroup {subgroup} tile {tile}x{lanes_a_row} split {splitting} past 8 GiB {past}"
                    );
                    close(&gpu.down(o, n), &want_o, &label);
                    if with_lse {
                        close(
                            &gpu.down_f32(l, rows * QH),
                            &want_l,
                            &format!("{label}: lse"),
                        );
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
#[allow(clippy::type_complexity)]
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
                (
                    table[buffered + (r - lane.first) / PAGE],
                    (r - lane.first) % PAGE,
                )
            }
        };
        // Rows a wider ring would take too: this ring drops them.
        for r in lane.folded.max(to[l].saturating_sub(R + 5))..to[l] {
            let (unit, row) = source(r);
            jobs.extend([unit as i32, row as i32, -1 - l as i32, r as i32]);
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
            jobs.extend([
                unit as i32,
                row as i32,
                at as i32,
                ((r - from) % PAGE) as i32,
            ]);
            moved.push((l, r, at, (r - from) % PAGE));
        }
    }
    let count = (jobs.len() / 4) as i32;
    let mut table = vec![count];
    table.extend(jobs);
    (table, moved)
}

/// The fold (CONTRACT-BC 19): a lane folding its buffer and part of its
/// fire, one past the ring's rows (the oldest drop), one folding all of it,
/// and one folding nothing (its ring untouched: a stop-truncate's lane). The
/// rows each folds land in its ring at their index modulo the ring's rows,
/// the rows it leaves unfolded in their buffer slots, and nothing else
/// moves; on a ring of `R` rows and, the same jobs, on a narrower one of 24
/// (two windows on one fold's jobs), whose lanes keep their last 24 alone.
#[test]
fn a_fold_lands_folded_rows_in_the_ring_and_unfolded_ones_in_the_buffer() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let hd = 64;
    let lanes = prefills();
    let to = [70, 64 + 40, 21, 70, 100];
    let g = Geometry::of(hd, 64, lanes.len());
    let laid = lay(&g, &lanes, 9);
    let (jobs, moved) = fold_jobs(&lanes, &to, &laid);
    let most = (jobs.len() as u32 - 1) / ring::JOB_INTS + 3;
    for rows in [R, 24] {
        let staged = stage(&mut gpu, &g, &lanes, &laid, &to);
        // Past 8 GiB of the pool's words, as the fire's own slots lie at a
        // big pool's top.
        let mut pool = high(&staged.pool);
        let at = pool.ring.as_mut().expect("the pool holds a ring");
        at.rows = rows as u32;
        let at = *at;
        // The table opens with the layer's geometry (`Pools::ring_layers`).
        let mut padded = vec![
            pool.origin[0] as i32,
            pool.origin[1] as i32,
            pool.page_stride[0] as i32,
            (pool.seq_stride / 2) as i32,
            at.piece as i32,
            at.values_piece as i32,
            at.keys as i32,
            at.values as i32,
            at.rows as i32,
        ];
        assert_eq!(padded.len(), ring::LAYER_INTS as usize);
        padded.extend(&jobs);
        padded.resize(ring::LAYER_INTS as usize + 1 + most as usize * 4, 0);
        let jt = gpu.up_i32(&padded);
        let job_table = Tensor::new(jt, padded.len() as u32, 1, Dtype::I32);
        gpu.run(|c| ring::fold(c, &[&pool], job_table, most));
        let after = gpu.down(staged.pool_buf, g.elems);
        let mut want: Vec<f64> = laid.pool.iter().map(|x| f64::from(*x)).collect();
        for (l, lane) in lanes.iter().enumerate() {
            for r in lane.folded.max(to[l].saturating_sub(rows))..to[l] {
                for d in 0..g.row {
                    want[g.ring_at(l, r % rows, false) + d] = laid.keys[l][r][d];
                    want[g.ring_at(l, r % rows, true) + d] = laid.values[l][r][d];
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
        assert_eq!(
            wrong, 0,
            "ring of {rows}: {wrong} pool elements differ from the fold's rows"
        );
    }
}

/// Beam siblings sharing one set read along their own lineages
/// (CONTRACT-BC 19, `Lineage::Admitted`): two lanes over one folded ring
/// (rows below 48) and one set's buffer whose rows 48..64 interleave theirs
/// (lane 0 the even slots, lane 1 the odd), each decoding a row past them
/// through its mask. A row's window of 32 counts along its lineage (its
/// eight buffered rows, then its own), so it starts at ring row 25, where
/// counting the set's slots would start it at 33 or 34; each lane answers
/// the reference over exactly its lineage's keys.
#[test]
fn beam_siblings_read_their_window_along_their_own_lineage() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let hd = 64usize;
    let g = Geometry::of(hd, 16, 2);
    let (folded, firsts) = (48usize, [64usize, 65]);
    let mut pool: Vec<f32> = noise(41, g.elems)
        .into_iter()
        .map(|x| crate::gpu::widen(crate::gpu::bf16(x)))
        .collect();
    let row_of = |seed: u64| -> Vec<f64> {
        noise(seed, g.row)
            .into_iter()
            .map(|x| f64::from(crate::gpu::widen(crate::gpu::bf16(x))))
            .collect()
    };
    let put = |pool: &mut Vec<f32>, at: usize, row: &[f64]| {
        for (d, x) in row.iter().enumerate() {
            pool[at + d] = *x as f32;
        }
    };
    // The common ancestor's rows in the one record's ring (lane 0's).
    let anc: Vec<(Vec<f64>, Vec<f64>)> = (0..folded)
        .map(|r| (row_of(1000 + r as u64), row_of(5000 + r as u64)))
        .collect();
    for r in folded - R..folded {
        put(&mut pool, g.ring_at(0, r % R, false), &anc[r].0);
        put(&mut pool, g.ring_at(0, r % R, true), &anc[r].1);
    }
    // The set's buffer: slot unit 1 rows 48..64 (lane l's at slot 48 + 2i + l),
    // unit 3 past them; each lane's own row in a fire slot of its own.
    let buffered: Vec<(Vec<f64>, Vec<f64>)> = (0..16)
        .map(|i| (row_of(9000 + i), row_of(9500 + i)))
        .collect();
    for (i, (k, v)) in buffered.iter().enumerate() {
        put(&mut pool, g.slot_at(1, i, false), k);
        put(&mut pool, g.slot_at(1, i, true), v);
    }
    let own: Vec<(Vec<f64>, Vec<f64>)> =
        (0..2).map(|l| (row_of(700 + l), row_of(800 + l))).collect();
    for (l, unit) in [(0usize, 5u32), (1, 7)] {
        put(&mut pool, g.slot_at(unit, 0, false), &own[l].0);
        put(&mut pool, g.slot_at(unit, 0, true), &own[l].1);
    }
    let (pool_buf, _) = gpu.up(&pool);
    let record = [g.record0 as i32, (g.record0 + 1) as i32];
    let mut lanes_table = Vec::new();
    for l in 0..2usize {
        let shift = firsts[l] - folded - 8;
        lanes_table.extend([
            folded as i32,
            48,
            firsts[l] as i32,
            folded as i32,
            shift as i32,
            48,
        ]);
        lanes_table.extend(record);
    }
    let indices = [1u32, 5, 1, 3, 7];
    let indptr = [0u32, 2, 5];
    // Masks from the folded index: a lane's own buffered slots and its row.
    let mut mask = vec![0u8; 2 * MASK_STRIDE];
    let mut words = Vec::new();
    for l in 0..2usize {
        for c in 0..MASK_STRIDE {
            let slot = folded + c;
            let mine = (slot < 64 && (slot - folded) % 2 == l) || slot == firsts[l];
            mask[l * MASK_STRIDE + c] = u8::from(mine);
        }
        words.extend([folded as u32, (l * MASK_STRIDE) as u32]);
    }
    let (pi, pp) = (gpu.up_u32(&indices), gpu.up_u32(&indptr));
    let rl = gpu.up_i32(&lanes_table);
    let pos = gpu.up_i32(&[64, 65]);
    let own_req = gpu.up_i32(&[0, 1]);
    let (mb, eb) = (gpu.up_u32(&pack(&mask)), gpu.up_u32(&words));
    let cells = |w: usize| t(pool_buf, g.elems / w, w, Dtype::Bf16);
    let kv = KvPool {
        keys: cells(g.row),
        values: cells(g.row),
        page_indices: t(pi, indices.len(), 1, Dtype::U32),
        page_indptr: t(pp, indptr.len(), 1, Dtype::U32),
        page_size: PAGE as i32,
        seq_stride: g.row as u64,
        head_stride: hd as u64,
        pool: Segments::one(pool_buf, (g.elems / 2) as u32),
        page_stride: [g.slot_words as u32, g.slot_words as u32],
        origin: [0, (PAGE * g.row / 2) as u32],
        ring: Some(Ring {
            lanes: t(rl, lanes_table.len(), 1, Dtype::I32),
            rows: R as u32,
            piece: PIECE as u32,
            values_piece: PIECE as u32,
            pieces: PIECES as u32,
            keys: RING_AT as u32,
            values: (RING_AT + R * g.row / 2) as u32,
            slot_stride: g.piece_words as u32,
        }),
    };
    let tables = sdpa::Tables {
        positions: t(pos, 2, 1, Dtype::I32),
        request_of_token: t(own_req, 2, 1, Dtype::I32),
        mask: t(mb, 1, 2 * MASK_STRIDE, Dtype::U8),
        mask_words: t(eb, 2, 2, Dtype::U32),
        row0: 0,
        table0: 0,
        rows: 2,
    };
    // The reference: lineage start 25 for both rows.
    let (q, qr) = gpu.up(&noise(77, 2 * QH * hd));
    let scale = 1.0 / (hd as f64).sqrt();
    let mut want = Vec::new();
    for l in 0..2usize {
        let lin = folded + 8;
        let start = lin + 1 - W;
        assert!(
            start < folded && firsts[l] + 1 - W > start,
            "the lineage starts earlier"
        );
        let mut keys: Vec<(&[f64], &[f64])> = (start..folded)
            .map(|r| (anc[r].0.as_slice(), anc[r].1.as_slice()))
            .collect();
        for i in (l..16).step_by(2) {
            keys.push((&buffered[i].0, &buffered[i].1));
        }
        keys.push((&own[l].0, &own[l].1));
        for h in 0..QH {
            let kh = h / (QH / KVH);
            let qv = &qr[(l * QH + h) * hd..][..hd];
            let s: Vec<f64> = keys
                .iter()
                .map(|(k, _)| {
                    qv.iter()
                        .zip(&k[kh * hd..][..hd])
                        .map(|(a, b)| f64::from(*a) * b)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - top).exp()).sum();
            for d in 0..hd {
                want.push(
                    keys.iter()
                        .zip(&s)
                        .map(|((_, v), x)| (x - top).exp() * v[kh * hd + d])
                        .sum::<f64>()
                        / z,
                );
            }
        }
    }
    let qt = t(q, 2, QH * hd, Dtype::Bf16);
    let n = 2 * QH * hd;
    for (subgroup, threads) in [(true, 256u32), (false, 256), (false, 128)] {
        if subgroup && (!crate::backend::SPIRV || gpu.subgroup() < 32) {
            continue;
        }
        let tactic = sdpa::Tactic {
            tier: if subgroup {
                kernels_slang::Tier::Subgroup
            } else {
                kernels_slang::Tier::Baseline
            },
            narrow: gpu.subgroup() < 32,
            decode_threads: threads,
            tile_rows: 16,
            tile_lanes: 16,
            split_groups: 1,
            split_min_keys: 8,
            prefill_work: 0,
        };
        let o = gpu.zeros(n);
        let ot = t(o, 2, QH * hd, Dtype::Bf16);
        gpu.run(|c| {
            sdpa::decode(
                c,
                tactic,
                qt,
                &tables,
                &kv,
                Some(W as u32),
                hd as u32,
                scale as f32,
                ot,
                None,
                None,
            )
        });
        close(
            &gpu.down(o, n),
            &want,
            &format!("siblings' lineage windows, subgroup {subgroup} threads {threads}"),
        );
        let o = gpu.zeros(n);
        let ot = t(o, 2, QH * hd, Dtype::Bf16);
        gpu.run(|c| {
            sdpa::prefill(
                c,
                tactic,
                qt,
                &tables,
                &kv,
                Some(W as u32),
                hd as u32,
                scale as f32,
                ot,
                None,
            )
        });
        close(
            &gpu.down(o, n),
            &want,
            &format!("siblings' lineage windows, tiled, subgroup {subgroup}"),
        );
    }
}
